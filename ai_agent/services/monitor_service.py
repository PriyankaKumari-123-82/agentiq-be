"""Monitor service — data access layer for monitor execution.

Responsibilities:
- Load monitor + related objects (organization, MCP tools, monitor type)
- Persist MonitorResult rows
- Fetch latest result / paginated history
- Organization-scoped MCP tool gathering

Rules (project conventions):
- All DB queries live here — never in workflows or routers
- Never raise HTTPException — return None / [] on error
- Never call the LLM — that lives in workflows/monitor/monitor_agent.py
- db_session.rollback() in every except block that wraps a write
"""
import json as _json
import logging
from datetime import datetime, timezone

from sqlmodel import Session, desc, func, select

from ai_agent.core.mcp_manager import get_mcp_manager
from ai_agent.custom_model.monitor import Monitor, MonitorResult, MonitorType
from ai_agent.models import (
    AgentMCPConfigLink,
    AIAgent,
    AIAgentSettings,
    MCPServerConfig,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Organization MCP tools helper
# ---------------------------------------------------------------------------


async def gather_org_mcp_tools(
    organization_id: int,
    db_session: Session,
) -> list[dict]:
    """Connect to every MCP server linked to agents in this organization.

    Returns a flat tool list identical to ``mcp_service.gather_mcp_tools``
    but scoped to the organization rather than a single agent.

    Each returned dict has keys::
        server_name, server_description, tool_name, description,
        input_schema, requires_query_building
    """
    mcp_tools: list[dict] = []

    try:
        # Find all agent_settings whose agent belongs to this organization
        # and that have MCP enabled.
        rows = db_session.exec(
            select(MCPServerConfig)
            .join(
                AgentMCPConfigLink,
                AgentMCPConfigLink.mcp_server_config_id == MCPServerConfig.id,
            )
            .join(
                AIAgentSettings,
                AIAgentSettings.id == AgentMCPConfigLink.agent_settings_id,
            )
            .join(
                AIAgent,
                AIAgent.id == AIAgentSettings.agent_id,
            )
            .where(
                (AIAgent.organization_id == organization_id) | (AIAgent.organization_id.is_(None)),
                AIAgent.deleted == False,  # noqa: E712
                AIAgentSettings.is_mcp_enabled == True,  # noqa: E712
            )
            .distinct()
        ).all()

        if not rows:
            return mcp_tools

        mcp_mgr = get_mcp_manager()
        seen_servers: set[str] = set()

        for srv in rows:
            server_name = srv.name
            if server_name in seen_servers:
                continue
            seen_servers.add(server_name)

            try:
                conn_result = await mcp_mgr.connect(
                    server_name,
                    {"name": server_name, "url": srv.url},
                )
                if conn_result.get("status") == "error":
                    log.warning(
                        "MCP server '%s' unreachable: %s; skipping.",
                        server_name, conn_result.get("error"),
                    )
                    continue

                tools = await mcp_mgr.list_tools(server_name)
                log.info("Loaded MCP tools from '%s': %s", server_name, tools)
            except Exception as e:
                log.warning("Failed to load MCP tools from '%s': %s", server_name, e)
                continue

            for t in tools:
                mcp_tools.append({
                    "server_name": server_name,
                    "server_description": (srv.description or "").strip(),
                    "tool_name": t["name"],
                    "description": t.get("description"),
                    "input_schema": t.get("input_schema") or {},
                    "requires_query_building": bool(srv.requires_query_building),
                })

    except Exception as exc:
        log.error("Failed to gather org MCP tools for org %s: %s", organization_id, exc)

    return mcp_tools


async def list_org_db_tables(
    organization_id: int,
    db_session: Session,
) -> list[str]:
    """Return distinct table names exposed by the organization's DB MCP server(s).

    Used to auto-seed starter monitors from the real schema instead of static
    mock data. Returns [] if no DB (query-building) MCP tool is connected.
    """
    table_names: list[str] = []

    try:
        mcp_tools = await gather_org_mcp_tools(organization_id, db_session)
        db_servers = {
            t["server_name"] for t in mcp_tools if t.get("requires_query_building")
        }
        if not db_servers:
            return []

        mcp_mgr = get_mcp_manager()
        seen: set[str] = set()

        for server_name in db_servers:
            try:
                tools = await mcp_mgr.list_tools(server_name)
                schema_tool = next(
                    (t for t in tools if "schema" in t["name"].lower()), None,
                )
                if not schema_tool:
                    continue

                result = await mcp_mgr.call_tool(server_name, schema_tool["name"], {})
                for item in (result.get("content") or []):
                    text = item.get("text")
                    if not text:
                        continue
                    data = _json.loads(text)
                    if not isinstance(data, list):
                        continue
                    for table in data:
                        if isinstance(table, dict) and table.get("table_name"):
                            name = table["table_name"]
                            if name not in seen:
                                seen.add(name)
                                table_names.append(name)
            except Exception as exc:
                log.warning(
                    "Failed to list tables from MCP server '%s': %s", server_name, exc,
                )
                continue

    except Exception as exc:
        log.error("Failed to list DB tables for org %s: %s", organization_id, exc)

    return table_names


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


def load_monitor_with_context(
    monitor_id: int,
    db_session: Session,
) -> tuple[Monitor | None, MonitorType | None]:
    """Load a monitor and the monitor_type needed for execution.

    Returns:
        (monitor, monitor_type)
        Any element may be None if not found or inactive.
    """
    try:
        monitor = db_session.get(Monitor, monitor_id)
        if not monitor or not monitor.is_active:
            return None, None

        monitor_type = (
            db_session.get(MonitorType, monitor.monitor_type_id)
            if monitor.monitor_type_id
            else None
        )

        return monitor, monitor_type

    except Exception as exc:
        log.error("Failed to load context for monitor %s: %s", monitor_id, exc)
        return None, None


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------


def persist_monitor_result(
    monitor: Monitor,
    monitor_type: MonitorType,
    result: dict,
    status: str,
    execution_time_ms: int,
    db_session: Session,
    error_message: str | None = None,
) -> MonitorResult | None:
    """Write a MonitorResult row and return it.

    Column layout matches the actual DB (monitor_results):
      monitor_id, executed_at, status, execution_time_ms, result_type,
      result (JSON), error_message, is_active.
    """
    try:
        row = MonitorResult(
            monitor_id=monitor.id,
            executed_at=datetime.now(timezone.utc),
            status=status,
            execution_time_ms=execution_time_ms,
            result_type=monitor_type.result_type or "TEXT",
            result=_json.dumps(result),
            error_message=error_message,
            is_active=True,
        )
        db_session.add(row)
        db_session.commit()
        db_session.refresh(row)
        log.info(
            "Persisted MonitorResult id=%s for monitor %s [%s] in %dms",
            row.id, monitor.id, status, execution_time_ms,
        )
        return row

    except Exception as exc:
        log.error("Failed to persist MonitorResult for monitor %s: %s", monitor.id, exc)
        db_session.rollback()
        return None


def update_monitor_llm_query(
    monitor: Monitor,
    llm_query: str,
    db_session: Session,
    followup_question: str | None = None,
) -> None:
    """Update the LLM-generated query (and optional follow-up) on the monitor row."""
    try:
        monitor.llm_query = llm_query
        if followup_question is not None:
            monitor.followup_question = followup_question
        db_session.add(monitor)
        db_session.commit()
        db_session.refresh(monitor)
        log.info("Updated llm_query for monitor %s", monitor.id)
    except Exception as exc:
        log.error("Failed to update llm_query for monitor %s: %s", monitor.id, exc)
        db_session.rollback()


# ---------------------------------------------------------------------------
# Result queries
# ---------------------------------------------------------------------------


def get_latest_result(
    monitor_id: int,
    db_session: Session,
) -> MonitorResult | None:
    """Return the single most recent result row for a monitor."""
    try:
        return db_session.exec(
            select(MonitorResult)
            .where(MonitorResult.monitor_id == monitor_id)
            .where(MonitorResult.is_active == True)  # noqa: E712
            .order_by(desc(MonitorResult.executed_at))
            .limit(1)
        ).first()
    except Exception as exc:
        log.error("Failed to load latest result for monitor %s: %s", monitor_id, exc)
        return None


def get_result_history(
    monitor_id: int,
    db_session: Session,
    page: int = 1,
    limit: int = 20,
) -> tuple[list[MonitorResult], int]:
    """Return paginated execution history for a monitor, newest first.

    Returns:
        (items, total_count)
    """
    try:
        base = (
            select(MonitorResult)
            .where(MonitorResult.monitor_id == monitor_id)
            .where(MonitorResult.is_active == True)  # noqa: E712
        )

        total = db_session.exec(
            select(func.count()).select_from(base.subquery())
        ).one() or 0

        offset = (page - 1) * limit
        items = db_session.exec(
            base.order_by(desc(MonitorResult.executed_at))
            .offset(offset)
            .limit(limit)
        ).all()

        return list(items), total

    except Exception as exc:
        log.error(
            "Failed to load result history for monitor %s: %s", monitor_id, exc
        )
        return [], 0
