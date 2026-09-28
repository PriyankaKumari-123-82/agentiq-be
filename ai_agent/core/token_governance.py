"""Token governance and policy enforcement."""
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException
from sqlmodel import Session, select
from sqlalchemy import func

from ai_agent.models import (
    LLMUsageEvent,
    OrganizationTokenPolicy,
    AgentTokenPolicy,
    UserTokenPolicy
)

log = logging.getLogger(__name__)


def _start_of_today() -> datetime:
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def _check_daily_limit(
    db: Session,
    policy,
    scope_label: str,
    consumed_filter,
) -> None:
    """Block the request if this policy's daily cap is enabled and exhausted.

    Weekly and monthly caps are configurable allocations used for display/analytics,
    but only the daily cap actually blocks requests — org, agent and user policies
    are each checked independently, so any one of the three being over its daily
    limit is enough to raise.

    `consumed_filter` scopes the LLMUsageEvent select to the relevant org/agent/user,
    in addition to the "since start of today" filter.
    """
    if not policy.daily_limit_enabled:
        return

    total_consumed = db.exec(
        consumed_filter(select(func.sum(LLMUsageEvent.total_tokens)))
        .where(LLMUsageEvent.created_at >= _start_of_today())
    ).first() or 0

    if total_consumed >= policy.daily_max_tokens:
        raise HTTPException(
            status_code=429,
            detail=f"{scope_label} daily token limit exceeded."
        )


def check_token_policy(
    db: Session,
    user_id: int,
    agent_id: str,
    organization_id: Optional[int] = None
) -> None:
    """Check if the user/agent/organization has exhausted their daily token limit.

    Org, agent and user policies also carry weekly/monthly allocations (used for
    display and analytics), but only the daily cap is enforced here — the request
    is blocked as soon as any one of the three (org, agent, user) is over its own
    daily limit.

    Raises HTTPException(429) if a daily limit is exceeded.
    """

    # 1. Organization Level Policy Check
    if organization_id:
        org_policy = db.exec(
            select(OrganizationTokenPolicy).where(OrganizationTokenPolicy.organization_id == organization_id)
        ).first()

        if org_policy:
            _check_daily_limit(
                db, org_policy, "Organization",
                lambda q: q.where(LLMUsageEvent.organization_id == organization_id),
            )

    # 2. Agent Level Policy Check
    agent_policy = db.exec(
        select(AgentTokenPolicy).where(AgentTokenPolicy.agent_id == agent_id)
    ).first()

    if agent_policy:
        _check_daily_limit(
            db, agent_policy, "Agent",
            lambda q: q.where(LLMUsageEvent.agent_id == agent_id),
        )

    # 3. User Level Policy Check
    user_policy = db.exec(
        select(UserTokenPolicy).where(UserTokenPolicy.user_id == user_id)
    ).first()

    if user_policy:
        _check_daily_limit(
            db, user_policy, "User",
            lambda q: q.where(LLMUsageEvent.user_id == user_id),
        )
