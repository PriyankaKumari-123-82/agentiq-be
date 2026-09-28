from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session, select
from sqlalchemy import func, desc, case

from ai_agent.core import db
from ai_agent.core.deps import get_org_membership
from ai_agent.core.token_validation import security
from ai_agent.models import Organization, OrganizationMembership, LLMUsageEvent, OrganizationTokenPolicy, AgentTokenPolicy, UserTokenPolicy, AIAgent, User
from pydantic import BaseModel

class OrgUsage(BaseModel):
    organization: str
    allocated: str
    daily_allocated: str
    weekly_allocated: str
    used: str
    remaining: str
    usage_percentage: float
    cost: str
    status: str

class AgentUsage(BaseModel):
    agent: str
    organization: str
    used_allocated: str
    daily_allocated: str
    weekly_allocated: str
    usage_percentage: float
    cost: str
    status: str

class UserUsage(BaseModel):
    user: str
    organization: str
    agent: str
    today_usage: str
    monthly_usage: str
    quota: str
    daily_quota: str
    weekly_quota: str
    status: str

class RecentRequest(BaseModel):
    timestamp: str
    user: str
    agent: str
    model: str
    input_tokens: str
    output_tokens: str
    embedding_tokens: str
    tool_tokens: str
    total_tokens: str
    cost: str
    latency: str
    status: str

router = APIRouter(tags=["Token Usage"])

def _verify_org_access(org_id: int, db_session: Session) -> Organization:
    org = db_session.get(Organization, org_id)
    if org is None or not org.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")
    return org

def _format_compact(number: float) -> str:
    if number >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    elif number >= 1_000:
        return f"{number / 1_000:.2f}k"
    return f"{int(number)}"

def _get_status(percentage: float, alert_threshold: float = 80) -> str:
    if percentage >= 100:
        return "Critical"
    elif percentage >= alert_threshold:
        return "Warning"
    return "Healthy"

def _period_allocated(policy, period: str) -> str:
    """Formatted daily/weekly/monthly allocation for a policy, or '∞' if that period isn't capped."""
    enabled = bool(getattr(policy, f"{period}_limit_enabled", False)) if policy else False
    max_tokens = getattr(policy, f"{period}_max_tokens", 0) if policy else 0
    return _format_compact(max_tokens) if enabled and max_tokens else "∞"

@router.get(
    "/token/{org_id}/usage/organizations",
    response_model=List[OrgUsage],
    dependencies=[Depends(security)],
)
async def get_usage_organizations(
    org_id: int,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    membership: OrganizationMembership = Depends(get_org_membership),
    db_session: Session = Depends(db.get_session),
):
    org = _verify_org_access(org_id, db_session)
    now = datetime.now(timezone.utc)
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Query totals
    totals = db_session.exec(
        select(
            func.sum(LLMUsageEvent.total_tokens).label("total_tokens"),
            func.sum(LLMUsageEvent.cost_usd).label("total_cost"),
        ).where(
            LLMUsageEvent.organization_id == org_id,
            LLMUsageEvent.created_at >= start_of_month
        )
    ).first()
    
    tokens = totals.total_tokens or 0
    cost = totals.total_cost or 0
    
    policy = db_session.exec(select(OrganizationTokenPolicy).where(OrganizationTokenPolicy.organization_id == org_id)).first()
    # This view aggregates usage over the current month, so surface the monthly cap here.
    max_tokens = policy.monthly_max_tokens if policy and policy.monthly_limit_enabled else 0
    percentage = (tokens / max_tokens * 100) if max_tokens > 0 else 0
    remaining = max_tokens - tokens if max_tokens > 0 else 0
    
    alert_threshold = policy.alert_threshold_percentage if policy else 80
    status_label = _get_status(percentage, alert_threshold) if max_tokens > 0 else "Healthy"
    
    return [
        OrgUsage(
            organization=org.name,
            allocated=_format_compact(max_tokens) if max_tokens else "∞",
            daily_allocated=_period_allocated(policy, "daily"),
            weekly_allocated=_period_allocated(policy, "weekly"),
            used=_format_compact(tokens),
            remaining=_format_compact(remaining) if max_tokens else "∞",
            usage_percentage=round(percentage, 1) if max_tokens else 0,
            cost=f"${cost:.2f}",
            status=status_label
        )
    ]

@router.get(
    "/token/{org_id}/usage/agents",
    response_model=List[AgentUsage],
    dependencies=[Depends(security)],
)
async def get_usage_agents(
    org_id: int,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    membership: OrganizationMembership = Depends(get_org_membership),
    db_session: Session = Depends(db.get_session),
):
    org = _verify_org_access(org_id, db_session)
    now = datetime.now(timezone.utc)
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Query usage per agent
    query = select(
        AIAgent.id,
        AIAgent.name,
        func.sum(LLMUsageEvent.total_tokens).label("tokens"),
        func.sum(LLMUsageEvent.cost_usd).label("cost")
    ).join(AIAgent, LLMUsageEvent.agent_id == AIAgent.id).where(
        LLMUsageEvent.organization_id == org_id,
        LLMUsageEvent.created_at >= start_of_month
    ).group_by(AIAgent.id, AIAgent.name).order_by(desc("tokens")).offset((page - 1) * limit).limit(limit)
    
    results = db_session.exec(query).all()
    
    agents = []
    for row in results:
        policy = db_session.exec(select(AgentTokenPolicy).where(AgentTokenPolicy.agent_id == row.id)).first()
        # This view aggregates usage over the current month, so surface the monthly cap here.
        max_tokens = policy.monthly_max_tokens if policy and policy.monthly_limit_enabled else 0
        percentage = (row.tokens / max_tokens * 100) if max_tokens > 0 else 0
        alert_threshold = policy.alert_threshold_percentage if policy else 80
        
        agents.append(AgentUsage(
            agent=row.name,
            organization=org.name,
            used_allocated=f"{_format_compact(row.tokens)} / {_format_compact(max_tokens) if max_tokens else '∞'}",
            daily_allocated=_period_allocated(policy, "daily"),
            weekly_allocated=_period_allocated(policy, "weekly"),
            usage_percentage=round(percentage, 1) if max_tokens else 0,
            cost=f"${row.cost:.2f}",
            status=_get_status(percentage, alert_threshold) if max_tokens else "Healthy"
        ))
        
    return agents

@router.get(
    "/token/{org_id}/usage/users",
    response_model=List[UserUsage],
    dependencies=[Depends(security)],
)
async def get_usage_users(
    org_id: int,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    membership: OrganizationMembership = Depends(get_org_membership),
    db_session: Session = Depends(db.get_session),
):
    org = _verify_org_access(org_id, db_session)
    now = datetime.now(timezone.utc)
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Using CTE or subqueries to get daily vs monthly might be complex, so we'll do a simple monthly group by user
    query = select(
        User.id,
        User.email,
        AIAgent.name.label("agent_name"),
        func.sum(LLMUsageEvent.total_tokens).label("monthly_tokens"),
        func.sum(case((LLMUsageEvent.created_at >= start_of_today, LLMUsageEvent.total_tokens), else_=0)).label("today_tokens")
    ).join(User, LLMUsageEvent.user_id == User.id).join(AIAgent, LLMUsageEvent.agent_id == AIAgent.id).where(
        LLMUsageEvent.organization_id == org_id,
        LLMUsageEvent.created_at >= start_of_month
    ).group_by(User.id, User.email, AIAgent.name).order_by(desc("monthly_tokens")).offset((page - 1) * limit).limit(limit)
    
    results = db_session.exec(query).all()
    
    users = []
    for row in results:
        policy = db_session.exec(select(UserTokenPolicy).where(UserTokenPolicy.user_id == row.id)).first()
        # This view aggregates usage over the current month, so surface the monthly cap here.
        max_tokens = policy.monthly_max_tokens if policy and policy.monthly_limit_enabled else 0
        percentage = (row.monthly_tokens / max_tokens * 100) if max_tokens > 0 else 0
        alert_threshold = policy.alert_threshold_percentage if policy else 80
        
        users.append(UserUsage(
            user=row.email,
            organization=org.name,
            agent=row.agent_name,
            today_usage=_format_compact(row.today_tokens),
            monthly_usage=_format_compact(row.monthly_tokens),
            quota=_format_compact(max_tokens) if max_tokens else "∞",
            daily_quota=_period_allocated(policy, "daily"),
            weekly_quota=_period_allocated(policy, "weekly"),
            status=_get_status(percentage, alert_threshold) if max_tokens else "Healthy"
        ))
        
    return users

@router.get(
    "/token/{org_id}/usage/requests",
    response_model=List[RecentRequest],
    dependencies=[Depends(security)],
)
async def get_usage_requests(
    org_id: int,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    membership: OrganizationMembership = Depends(get_org_membership),
    db_session: Session = Depends(db.get_session),
):
    _verify_org_access(org_id, db_session)
    
    query = select(
        LLMUsageEvent,
        User.email,
        AIAgent.name.label("agent_name")
    ).join(User, LLMUsageEvent.user_id == User.id).join(AIAgent, LLMUsageEvent.agent_id == AIAgent.id).where(
        LLMUsageEvent.organization_id == org_id
    ).order_by(desc(LLMUsageEvent.created_at)).offset((page - 1) * limit).limit(limit)
    
    results = db_session.exec(query).all()
    
    requests = []
    for event, email, agent_name in results:
        requests.append(RecentRequest(
            timestamp=event.created_at.strftime("%b %d, %H:%M:%S"),
            user=email,
            agent=agent_name,
            model=event.model,
            input_tokens=f"{event.input_tokens:,}",
            output_tokens=f"{event.output_tokens:,}",
            embedding_tokens=f"{event.embedding_tokens:,}",
            tool_tokens=f"{event.tool_tokens:,}",
            total_tokens=f"{event.total_tokens:,}",
            cost=f"${event.cost_usd:.4f}",
            latency=f"{(event.latency_ms or 0) / 1000:.1f}s",
            status=event.status.capitalize()
        ))
        
    return requests
