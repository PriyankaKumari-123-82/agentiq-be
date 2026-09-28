"""Query Assistant history endpoints.

Full endpoint paths (prefix /api/v1 added by main.py):
  POST /api/v1/query-history                Record a submitted question
  GET  /api/v1/query-history?limit=10       Recent distinct questions, newest first
"""
import logging

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, field_validator
from sqlmodel import Session, desc, select

from ai_agent.core import db
from ai_agent.core.deps import get_current_user
from ai_agent.models import QueryHistoryEntry, User

log = logging.getLogger(__name__)

router = APIRouter()
PREFIX = "/query-history"


class QueryHistoryCreate(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


@router.post(PREFIX, response_model=QueryHistoryEntry)
def record_query(
    body: QueryHistoryCreate,
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    """
    Log a submitted question. A no-op that returns the existing row rather
    than inserting a duplicate when it exactly repeats this user's very last
    entry — retrying or reloading the same question shouldn't spam the list.
    """
    last = db_session.exec(
        select(QueryHistoryEntry)
        .where(QueryHistoryEntry.created_by == user.id)
        .order_by(desc(QueryHistoryEntry.created_at))
        .limit(1)
    ).first()
    if last and last.query.strip().lower() == body.query.lower():
        return last

    entry = QueryHistoryEntry(
        created_by=user.id,
        organization_id=user.active_organization_id,
        query=body.query,
    )
    db_session.add(entry)
    db_session.commit()
    db_session.refresh(entry)
    return entry


@router.get(PREFIX, response_model=list[QueryHistoryEntry])
def list_recent_queries(
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
    limit: int = Query(10, ge=1, le=50),
):
    """
    Recent questions for suggestion chips, newest first, deduplicated by text
    so re-asking the same thing keeps one entry at its latest position rather
    than crowding the list with repeats.
    """
    # Pull extra rows before deduping — worst case every recent row repeats
    # the same handful of questions, and a straight `limit` would starve the
    # result down to just those instead of a real spread of distinct ones.
    candidates = db_session.exec(
        select(QueryHistoryEntry)
        .where(QueryHistoryEntry.created_by == user.id)
        .order_by(desc(QueryHistoryEntry.created_at))
        .limit(limit * 4)
    ).all()

    seen: set[str] = set()
    unique: list[QueryHistoryEntry] = []
    for entry in candidates:
        key = entry.query.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
        if len(unique) >= limit:
            break

    return unique
