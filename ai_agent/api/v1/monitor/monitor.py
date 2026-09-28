"""Monitor CRUD endpoints.

Full endpoint paths (prefix /api/v1 added by main.py):
  POST   /api/v1/monitor
  GET    /api/v1/monitor                         (paginated: ?page=1&limit=20&search=)
  GET    /api/v1/monitor/{monitor_id}
  PATCH  /api/v1/monitor/{monitor_id}
  DELETE /api/v1/monitor/{monitor_id}            Soft delete — sets is_active = False
  POST   /api/v1/monitor/{monitor_id}/run        Trigger execution now
  GET    /api/v1/monitor/{monitor_id}/results    Paginated execution history
  GET    /api/v1/monitor/{monitor_id}/export     Latest table-shaped result as .xlsx
"""
import io
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import (
    APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response, status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlmodel import Session, func, or_, select
from starlette.concurrency import run_in_threadpool

from ai_agent.core import db
from ai_agent.core.deps import get_current_user
from ai_agent.core.token_validation import security
from ai_agent.models import Monitor, MonitorResult, MonitorType, User
from ai_agent.services.monitor_export import (
    NotTableShapedError, build_excel_export, safe_filename,
)
from ai_agent.services.monitor_service import get_latest_result, get_result_history

log = logging.getLogger(__name__)

router = APIRouter(
    tags=["Monitor"],
    dependencies=[Depends(security)],
)

PREFIX = "/monitor"


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class MonitorCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=1000)
    monitor_type: int = Field(..., description="ID of the monitor type (FK → monitor_type.id)")
    natural_query: str = Field(..., min_length=1, description="Natural-language question")
    refresh_interval: int = Field(default=3600, ge=60, description="Seconds between runs (min 60)")

    @field_validator("name", "natural_query", mode="before")
    @classmethod
    def strip_strings(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


class MonitorPatch(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=1000)
    monitor_type: Optional[int] = Field(default=None, description="ID of the monitor type")
    natural_query: Optional[str] = Field(default=None, min_length=1)
    refresh_interval: Optional[int] = Field(default=None, ge=60)

    @field_validator("name", "natural_query", mode="before")
    @classmethod
    def strip_strings(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if isinstance(v, str) else v


class PaginatedMonitorsOut(BaseModel):
    items: list[Monitor]
    page: int
    limit: int
    total: int
    has_more: bool


class MonitorRunOut(BaseModel):
    """Response returned after triggering a monitor run.

    With ?sync=true the run is awaited and the outcome fields below are
    populated, so the caller can show a real success/error/follow-up state
    instead of guessing by polling for results.
    """
    monitor_id: int = Field(..., description="ID of the monitor that was triggered")
    message: str = Field(..., description="Human-readable status message")
    status: Optional[str] = Field(
        default=None, description="sync only — 'success' | 'followup' | 'error'",
    )
    error: Optional[str] = Field(
        default=None, description="sync only — why the run failed, if it did",
    )
    followup_question: Optional[str] = Field(
        default=None, description="sync only — clarification the AI needs",
    )
    result_id: Optional[int] = Field(
        default=None, description="sync only — id of the MonitorResult row written",
    )
    result: Optional[MonitorResult] = Field(
        default=None,
        description="sync only — the MonitorResult row itself, so the caller can "
                    "preview the outcome without a second round-trip",
    )


class MonitorRunIn(BaseModel):
    """Optional payload when running a monitor to answer a follow-up question."""
    followup_answer: Optional[str] = Field(default=None, description="Answer to a previous follow-up question")


class PaginatedResultsOut(BaseModel):
    """Paginated list of past execution results for a monitor."""
    items: list[MonitorResult]
    page: int
    limit: int
    total: int
    has_more: bool


class SeedMonitorsOut(BaseModel):
    """Response for the auto-seed-from-DB endpoint."""
    items: list[Monitor]
    message: str


# ── Private helpers ───────────────────────────────────────────────────────────
def _get_owned_monitor(monitor_id: int, user: User, db_session: Session) -> Monitor:
    """Fetch an active monitor by ID and verify ownership. Raises 404 if not found or not owned."""
    monitor = db_session.get(Monitor, monitor_id)
    if not monitor or not monitor.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Monitor not found.")
    if monitor.created_by != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Monitor not found.")
    return monitor


def _build_seed_monitors(
    tables: list[str], user: User, db_session: Session,
) -> list[Monitor]:
    """Create starter monitors for tables that don't already have one.

    Blocking — call it through ``run_in_threadpool`` from an async endpoint.
    Returns the persisted rows, or [] when every table was already covered.
    """
    count_type = db_session.exec(
        select(MonitorType).where(MonitorType.name == "COUNT")
    ).first()
    list_type = db_session.exec(
        select(MonitorType).where(MonitorType.name == "LIST")
    ).first()
    if not count_type:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="COUNT monitor type is not seeded.",
        )

    existing_names = set(db_session.exec(
        select(Monitor.name)
        .where(Monitor.created_by == user.id)
        .where(Monitor.is_active == True)  # noqa: E712
    ).all())

    created: list[Monitor] = []

    for table in tables[:6]:
        name = f"{table} — Row Count"
        if name in existing_names:
            continue
        created.append(Monitor(
            name=name,
            monitor_type_id=count_type.id,
            natural_query=f"How many rows are in the {table} table?",
            is_active=True,
            created_by=user.id,
            organization_id=user.active_organization_id,
        ))

    if list_type:
        for table in tables[:3]:
            name = f"{table} — Recent Rows"
            if name in existing_names:
                continue
            created.append(Monitor(
                name=name,
                monitor_type_id=list_type.id,
                natural_query=f"Show the 10 most recent rows from the {table} table.",
                is_active=True,
                created_by=user.id,
                organization_id=user.active_organization_id,
            ))

    if not created:
        return []

    for m in created:
        db_session.add(m)
    db_session.commit()
    for m in created:
        db_session.refresh(m)

    return created


def owned_monitor(
    monitor_id: int,
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
) -> Monitor:
    """Path-param dependency resolving ``{monitor_id}`` to a monitor you own.

    Declared ``def`` rather than ``async def`` deliberately: FastAPI runs sync
    dependencies in a threadpool, so this blocking query stays off the event
    loop even for the ``async`` endpoints below. Calling ``_get_owned_monitor``
    inline from an ``async def`` would block it.
    """
    return _get_owned_monitor(monitor_id, user, db_session)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    PREFIX,
    response_model=Monitor,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new monitor",
)
def create_monitor(
    body: MonitorCreate,
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    """
    Returns 404 if monitor_type does not exist.
    """
    if not db_session.get(MonitorType, body.monitor_type):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Monitor type not found.")

    monitor = Monitor(
        name=body.name,
        description=body.description,
        monitor_type_id=body.monitor_type,
        natural_query=body.natural_query,
        scheduler_frequency=body.refresh_interval,
        is_active=True,
        created_by=user.id,
        organization_id=user.active_organization_id,
    )
    db_session.add(monitor)
    db_session.commit()
    db_session.refresh(monitor)
    return monitor


@router.get(
    PREFIX,
    response_model=PaginatedMonitorsOut,
    summary="List monitors for the current user",
)
def list_monitors(
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    search: Optional[str] = Query(None, description="Filter by name (partial match)"),
):
    """
    Returns active monitors created by the current user, newest first.
    Paginated — use ?page=1&limit=10. Filter by name with ?search=keyword.
    TODO: confirm cursor vs page-based pagination with Bhuvnesh.

    Only *finished* monitors are listed — this endpoint feeds the Intelligence
    Hub, which must show working cards and nothing else. A monitor is finished
    when it has no clarification outstanding and has produced at least one
    SUCCESS result. Half-built ones (waiting on a follow-up, or only ever
    failed) stay in the Query Assistant, where they can be fixed or discarded.
    """
    proven_monitor_ids = (
        select(MonitorResult.monitor_id)
        .where(MonitorResult.status == "SUCCESS")
        .where(MonitorResult.is_active == True)  # noqa: E712
    )

    base_query = (
        select(Monitor)
        .where(Monitor.created_by == user.id)
        .where(Monitor.is_active == True)  # noqa: E712
        .where(or_(
            Monitor.followup_question.is_(None),
            Monitor.followup_question == "",
        ))
        .where(Monitor.id.in_(proven_monitor_ids))
    )
    if search:
        base_query = base_query.where(Monitor.name.ilike(f"%{search}%"))

    total = db_session.exec(
        select(func.count()).select_from(base_query.subquery())
    ).one()

    offset = (page - 1) * limit
    items = db_session.exec(
        base_query.order_by(Monitor.created_at.desc()).offset(offset).limit(limit)
    ).all()

    return PaginatedMonitorsOut(
        items=items,
        page=page,
        limit=limit,
        total=total,
        has_more=(offset + len(items)) < total,
    )


@router.get(
    f"{PREFIX}/{{monitor_id}}",
    response_model=Monitor,
    summary="Get a single monitor",
)
def get_monitor(monitor: Monitor = Depends(owned_monitor)):
    """
    Returns 404 if the monitor does not exist or belongs to another user.

    Unlike ``GET /monitor`` this does not filter out half-built monitors — the
    Query Assistant loads one by id to resume its follow-up.
    """
    return monitor


@router.patch(
    f"{PREFIX}/{{monitor_id}}",
    response_model=Monitor,
    summary="Update a monitor",
)
def update_monitor(
    body: MonitorPatch,
    monitor: Monitor = Depends(owned_monitor),
    db_session: Session = Depends(db.get_session),
):
    """
    Partial update — only supplied fields are changed.
    When natural_query changes, scheduler_frequency is cleared (re-generated by Monitor Agent in P1).
    Returns 404 if monitor_type does not exist.
    """
    data = body.model_dump(exclude_unset=True)

    if not data:
        return monitor

    if "monitor_type" in data:
        if not db_session.get(MonitorType, data["monitor_type"]):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Monitor type not found.")
        data["monitor_type_id"] = data.pop("monitor_type")

    if "refresh_interval" in data:
        data["scheduler_frequency"] = data.pop("refresh_interval")

    if "natural_query" in data and data["natural_query"] != monitor.natural_query:
        monitor.scheduler_frequency = None
        # llm_query caches the SQL/tool-call built for the OLD wording. Left in
        # place, the next run replays it and the user sees their edit silently
        # ignored. Any follow-up pending on the old wording dies with it.
        monitor.llm_query = None
        monitor.followup_question = None

    if "monitor_type_id" in data and data["monitor_type_id"] != monitor.monitor_type_id:
        # A new type wants a different result shape — a single count reads
        # nothing like the rows a pie chart needs — so the cached query for the
        # old shape can't be reused either.
        monitor.llm_query = None

    for field, value in data.items():
        setattr(monitor, field, value)

    db_session.add(monitor)
    db_session.commit()
    db_session.refresh(monitor)
    return monitor


@router.delete(
    f"{PREFIX}/{{monitor_id}}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Soft-delete a monitor",
)
def delete_monitor(
    monitor: Monitor = Depends(owned_monitor),
    db_session: Session = Depends(db.get_session),
):
    """
    Sets is_active = False — monitor is hidden from all list/get endpoints.
    Returns 404 if the monitor does not exist or belongs to another user.
    """
    monitor.is_active = False
    monitor.deleted_at = datetime.now(timezone.utc)
    db_session.add(monitor)
    db_session.commit()


@router.post(
    f"{PREFIX}/{{monitor_id}}/run",
    response_model=MonitorRunOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger a monitor run now",
)
async def run_monitor(
    background_tasks: BackgroundTasks,
    response: Response,
    body: Optional[MonitorRunIn] = None,
    sync: bool = Query(
        False,
        description="Await the run and return its outcome (200) instead of "
                    "dispatching it in the background (202).",
    ),
    # Ownership / existence check — raises 404 on failure. Resolved as a sync
    # dependency so its query runs in a threadpool, not on the event loop.
    monitor: Monitor = Depends(owned_monitor),
    db_session: Session = Depends(db.get_session),
):
    """
    Triggers execution of the monitor's natural query through its agent.

    By default the run is dispatched as a background task and the response
    returns immediately with 202 Accepted.

    With ?sync=true the run is awaited and its outcome is returned with 200,
    so callers (the Query Assistant creation flow) can surface a failure to
    the user instead of writing a broken monitor to the dashboard.

    Returns 404 if the monitor does not exist or belongs to another user.
    """
    monitor_id = monitor.id
    followup_answer = body.followup_answer if body else None

    if sync:
        from ai_agent.workflows.monitor.monitor_agent import (
            _humanize_error, execute_monitor,
        )

        try:
            outcome = await execute_monitor(
                monitor_id, db_session, followup_answer=followup_answer,
            )
        except Exception as exc:  # noqa: BLE001 — never leak a 500 to the UI
            log.error("Monitor %s sync run failed: %s", monitor_id, exc, exc_info=True)
            response.status_code = status.HTTP_200_OK
            return MonitorRunOut(
                monitor_id=monitor_id,
                message="The monitor could not be run.",
                status="error",
                error=_humanize_error(str(exc)),
            )

        response.status_code = status.HTTP_200_OK
        run_status = outcome.get("status", "error")
        result_row = outcome.get("result")

        messages = {
            "success": "Monitor ran successfully.",
            "followup": "The AI needs more information before it can run this monitor.",
            "error": "The monitor ran but could not produce a valid result.",
        }
        return MonitorRunOut(
            monitor_id=monitor_id,
            message=messages.get(run_status, "Monitor run finished."),
            status=run_status,
            error=outcome.get("error"),
            followup_question=outcome.get("followup_question"),
            result_id=getattr(result_row, "id", None),
            result=result_row if isinstance(result_row, MonitorResult) else None,
        )

    async def _run():
        from ai_agent.workflows.monitor.monitor_agent import execute_monitor
        # A fresh session is required inside the background task because the
        # request-scoped session will be closed before the task finishes.
        from ai_agent.core.db import get_session as _get_session
        with next(_get_session()) as task_session:
            await execute_monitor(monitor_id, task_session, followup_answer=followup_answer)

    background_tasks.add_task(_run)

    return MonitorRunOut(
        monitor_id=monitor_id,
        message="Monitor run triggered. Results will be available shortly.",
    )


@router.post(
    f"{PREFIX}/seed-from-db",
    response_model=SeedMonitorsOut,
    status_code=status.HTTP_201_CREATED,
    summary="Auto-create starter monitors from the organization's connected DB schema",
)
async def seed_monitors_from_db(
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    """
    Inspects the tables exposed by the organization's connected DB MCP
    server(s) and creates a handful of starter monitors from real table
    names (row-count + recent-rows) instead of relying on static mock data.

    Skips tables that already have a matching monitor for this user.
    Newly created monitors are run in the background so results populate
    shortly after this call returns.
    """
    if not user.active_organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active organization — connect a database first.",
        )

    from ai_agent.services.monitor_service import list_org_db_tables

    tables = await list_org_db_tables(user.active_organization_id, db_session)
    if not tables:
        return SeedMonitorsOut(
            items=[], message="No connected database schema found to seed from.",
        )

    # Everything below this point is blocking DB work, and this endpoint is
    # async (it has to await the MCP schema call above). Running it inline
    # would stall the event loop for every other request, so it goes to a
    # threadpool — the same place FastAPI puts a plain `def` endpoint.
    created = await run_in_threadpool(_build_seed_monitors, tables, user, db_session)

    if not created:
        return SeedMonitorsOut(items=[], message="Starter monitors already exist for this schema.")

    created_ids = [m.id for m in created]

    async def _run_all():
        from ai_agent.workflows.monitor.monitor_agent import execute_monitor
        from ai_agent.core.db import get_session as _get_session
        for mid in created_ids:
            with next(_get_session()) as task_session:
                await execute_monitor(mid, task_session)

    background_tasks.add_task(_run_all)

    return SeedMonitorsOut(
        items=created,
        message=f"Created {len(created)} starter monitor(s) from your database schema.",
    )


@router.get(
    f"{PREFIX}/{{monitor_id}}/results",
    response_model=PaginatedResultsOut,
    summary="List execution results for a monitor",
)
def list_monitor_results(
    monitor: Monitor = Depends(owned_monitor),
    db_session: Session = Depends(db.get_session),
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
):
    """
    Returns paginated execution history for a monitor, newest first.
    Returns 404 if the monitor does not exist or belongs to another user.
    """
    items, total = get_result_history(monitor.id, db_session, page=page, limit=limit)
    offset = (page - 1) * limit

    return PaginatedResultsOut(
        items=items,
        page=page,
        limit=limit,
        total=total,
        has_more=(offset + len(items)) < total,
    )


@router.get(
    f"{PREFIX}/{{monitor_id}}/export",
    summary="Export a monitor's latest result to Excel",
)
def export_monitor(
    monitor: Monitor = Depends(owned_monitor),
    db_session: Session = Depends(db.get_session),
):
    """
    Streams an .xlsx of every row in the monitor's latest successful result —
    the same full dataset the "view all data" modal shows, not a page of it.
    """
    result = get_latest_result(monitor.id, db_session)
    if not result or result.status != "SUCCESS":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This monitor has no successful result to export yet.",
        )

    try:
        content = build_excel_export(monitor.name, result)
    except NotTableShapedError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    filename = safe_filename(monitor.name)
    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}.xlsx"'},
    )
