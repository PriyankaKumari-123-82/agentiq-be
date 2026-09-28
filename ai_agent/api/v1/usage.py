"""Usage tracking API endpoints.

This module provides endpoints for:
- Getting usage summaries
- Setting usage quotas
- Tracking token/cost consumption
- Usage analytics
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from ai_agent.core import db
from ai_agent.core.usage_tracker import UsageTracker, get_usage_tracker
from ai_agent.models import UsageQuota, User, AIAgent

log = logging.getLogger(__name__)

router = APIRouter(prefix="/usage", tags=["Usage"])


class QuotaCreateRequest(BaseModel):
    """Request to create a usage quota."""
    quota_type: str = Field(description="Type: tokens/cost/requests/audio_seconds")
    limit: float = Field(description="Quota limit value")
    period: str = Field(default="monthly", description="Period: daily/weekly/monthly")
    agent_id: Optional[str] = Field(default=None, description="Agent-specific quota (optional)")
    action_on_exceed: str = Field(default="block", description="Action: block/warn/allow")


class QuotaResponse(BaseModel):
    """Quota response."""
    id: int
    quota_type: str
    period: str
    limit_value: float
    used_value: float
    remaining: float
    period_start: datetime
    period_end: datetime
    is_enforced: bool
    action_on_exceed: str


class UsageSummaryResponse(BaseModel):
    """Usage summary response."""
    period: dict
    totals: dict
    by_provider: dict
    by_operation: dict


class UsageEventResponse(BaseModel):
    """Individual usage event."""
    id: str
    run_id: Optional[str]
    organization_id: Optional[int]
    provider: str
    model: str
    operation: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: Optional[int]
    status: str
    created_at: datetime


@router.get("/user/{user_id}/summary")
async def get_user_usage_summary(
    user_id: int = Path(..., description="User ID"),
    days: int = Query(30, ge=1, le=365, description="Number of days to summarize"),
    agent_id: Optional[str] = Query(None, description="Filter by agent"),
    db_session: Session = Depends(db.get_session),
):
    """Get usage summary for a user."""
    # Verify user exists
    user = db_session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    tracker = get_usage_tracker(db_session)
    
    start_date = datetime.utcnow() - timedelta(days=days)
    end_date = datetime.utcnow()
    
    summary = tracker.get_usage_summary(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        agent_id=agent_id,
    )
    
    return summary


@router.get("/user/{user_id}/events")
async def get_user_usage_events(
    user_id: int = Path(..., description="User ID"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    operation: Optional[str] = Query(None, description="Filter by operation type"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db_session: Session = Depends(db.get_session),
):
    """Get detailed usage events for a user."""
    from ai_agent.models import LLMUsageEvent
    
    user = db_session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    query = select(LLMUsageEvent).where(
        LLMUsageEvent.user_id == user_id
    ).order_by(
        LLMUsageEvent.created_at.desc()
    ).offset(offset).limit(limit)
    
    if operation:
        query = query.where(LLMUsageEvent.operation == operation)
    if provider:
        query = query.where(LLMUsageEvent.provider == provider)
    if start_date:
        query = query.where(LLMUsageEvent.created_at >= start_date)
    if end_date:
        query = query.where(LLMUsageEvent.created_at <= end_date)
    
    events = db_session.exec(query).all()
    
    return {
        "events": [
            {
                "id": e.id,
                "run_id": e.run_id,
                "organization_id": e.organization_id,
                "provider": e.provider,
                "model": e.model,
                "operation": e.operation,
                "input_tokens": e.input_tokens,
                "output_tokens": e.output_tokens,
                "total_tokens": e.total_tokens,
                "cost_usd": e.cost_usd,
                "latency_ms": e.latency_ms,
                "status": e.status,
                "created_at": e.created_at.isoformat(),
            }
            for e in events
        ],
        "count": len(events),
    }


@router.get("/agent/{agent_id}/summary")
async def get_agent_usage_summary(
    agent_id: str = Path(..., description="Agent ID"),
    user_id: int = Query(..., description="User ID (must own agent)"),
    days: int = Query(30, ge=1, le=365),
    db_session: Session = Depends(db.get_session),
):
    """Get usage summary for a specific agent."""
    agent = db_session.get(AIAgent, agent_id)
    if not agent or agent.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Agent not found or access denied")
    
    tracker = get_usage_tracker(db_session)
    
    start_date = datetime.utcnow() - timedelta(days=days)
    end_date = datetime.utcnow()
    
    summary = tracker.get_usage_summary(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        agent_id=agent_id,
    )
    
    return summary


@router.get("/run/{run_id}")
async def get_run_usage(
    run_id: str = Path(..., description="Run ID"),
    user_id: int = Query(...),
    db_session: Session = Depends(db.get_session),
):
    """Get usage details for a specific run."""
    from ai_agent.models import AgentRun
    
    run = db_session.get(AgentRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.user_id != user_id:
        raise HTTPException(status_code=403, detail="Run does not belong to user")
    
    tracker = get_usage_tracker(db_session)
    usage = tracker.get_run_usage(run_id)
    
    if not usage:
        raise HTTPException(status_code=404, detail="Usage data not found")
    
    return usage


@router.get("/quotas/{user_id}")
async def list_quotas(
    user_id: int = Path(...),
    active_only: bool = Query(True, description="Only return active quotas"),
    db_session: Session = Depends(db.get_session),
):
    """List all quotas for a user."""
    user = db_session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    query = select(UsageQuota).where(UsageQuota.user_id == user_id)
    
    if active_only:
        now = datetime.utcnow()
        query = query.where(
            UsageQuota.period_start <= now
        ).where(
            UsageQuota.period_end >= now
        )
    
    quotas = db_session.exec(query).all()
    
    return {
        "quotas": [
            QuotaResponse(
                id=q.id,
                quota_type=q.quota_type,
                period=q.period,
                limit_value=q.limit_value,
                used_value=q.used_value,
                remaining=q.limit_value - q.used_value,
                period_start=q.period_start,
                period_end=q.period_end,
                is_enforced=q.is_enforced,
                action_on_exceed=q.action_on_exceed,
            )
            for q in quotas
        ]
    }


@router.post("/quotas/{user_id}")
async def create_quota(
    user_id: int = Path(...),
    request: QuotaCreateRequest = None,
    db_session: Session = Depends(db.get_session),
):
    """Create a new usage quota for a user."""
    user = db_session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # If agent_id specified, verify it exists and belongs to user
    if request.agent_id:
        agent = db_session.get(AIAgent, request.agent_id)
        if not agent or agent.owner_id != user_id:
            raise HTTPException(status_code=403, detail="Agent not found or access denied")
    
    tracker = get_usage_tracker(db_session)
    
    quota = tracker.create_quota(
        user_id=user_id,
        quota_type=request.quota_type,
        limit=request.limit,
        period=request.period,
        agent_id=request.agent_id,
        action_on_exceed=request.action_on_exceed,
    )
    
    return QuotaResponse(
        id=quota.id,
        quota_type=quota.quota_type,
        period=quota.period,
        limit_value=quota.limit_value,
        used_value=quota.used_value,
        remaining=quota.limit_value - quota.used_value,
        period_start=quota.period_start,
        period_end=quota.period_end,
        is_enforced=quota.is_enforced,
        action_on_exceed=quota.action_on_exceed,
    )


@router.delete("/quotas/{quota_id}")
async def delete_quota(
    quota_id: int = Path(..., description="Quota ID"),
    user_id: int = Query(...),
    db_session: Session = Depends(db.get_session),
):
    """Delete a quota."""
    quota = db_session.get(UsageQuota, quota_id)
    if not quota:
        raise HTTPException(status_code=404, detail="Quota not found")
    if quota.user_id != user_id:
        raise HTTPException(status_code=403, detail="Quota does not belong to user")
    
    db_session.delete(quota)
    db_session.commit()
    
    return {"status": "deleted", "quota_id": quota_id}


@router.get("/cost-estimate")
async def estimate_cost(
    provider: str = Query(..., description="LLM provider"),
    model: str = Query(..., description="Model name"),
    input_tokens: int = Query(..., ge=0, description="Estimated input tokens"),
    output_tokens: int = Query(..., ge=0, description="Estimated output tokens"),
    db_session: Session = Depends(db.get_session),
):
    """Estimate cost for a potential LLM call."""
    tracker = get_usage_tracker(db_session)
    
    cost = tracker._calculate_cost(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    
    return {
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": cost,
        "currency": "USD",
    }


@router.get("/analytics/top-models/{user_id}")
async def get_top_models(
    user_id: int = Path(...),
    days: int = Query(30, ge=1, le=90),
    limit: int = Query(10, ge=1, le=50),
    db_session: Session = Depends(db.get_session),
):
    """Get top models used by a user."""
    from ai_agent.models import LLMUsageEvent
    from sqlalchemy import func
    
    user = db_session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    start_date = datetime.utcnow() - timedelta(days=days)
    
    results = db_session.exec(
        select(
            LLMUsageEvent.provider,
            LLMUsageEvent.model,
            func.count(LLMUsageEvent.id).label("calls"),
            func.sum(LLMUsageEvent.total_tokens).label("tokens"),
            func.sum(LLMUsageEvent.cost_usd).label("cost"),
        )
        .where(LLMUsageEvent.user_id == user_id)
        .where(LLMUsageEvent.created_at >= start_date)
        .group_by(LLMUsageEvent.provider, LLMUsageEvent.model)
        .order_by(func.sum(LLMUsageEvent.total_tokens).desc())
        .limit(limit)
    ).all()
    
    return {
        "models": [
            {
                "provider": r.provider,
                "model": r.model,
                "calls": r.calls,
                "total_tokens": r.tokens or 0,
                "cost_usd": round(r.cost or 0, 4),
            }
            for r in results
        ]
    }
