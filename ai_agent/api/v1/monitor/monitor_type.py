"""Monitor type catalog — read-only endpoint.

Routes
------
GET  /monitor/type   list all monitor types
"""
import logging

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from ai_agent.core import db
from ai_agent.core.deps import get_current_user
from ai_agent.core.token_validation import security
from ai_agent.models import MonitorType, User

log = logging.getLogger(__name__)

router = APIRouter(
    tags=["Monitor"],
    dependencies=[Depends(security)],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/monitor/type",
    response_model=list[MonitorType],
    summary="List all monitor types",
)
def list_monitor_types(
    _user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    """Return all monitor types from the catalog."""
    return db_session.exec(select(MonitorType).order_by(MonitorType.id)).all()
