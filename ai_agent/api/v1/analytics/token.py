from datetime import datetime, timedelta, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from sqlalchemy import func, desc, case
from pydantic import BaseModel

from ai_agent.core import db
from ai_agent.core.deps import get_org_membership
from ai_agent.core.token_validation import security
from ai_agent.models import Organization, OrganizationMembership, LLMUsageEvent, OrganizationTokenPolicy, AIAgent


class TokenSummaryValue(BaseModel):
    value: str
    total: str | None = None
    percentage: float | None = None
    subtitle: str | None = None
    trend: str | None = None
    trend_direction: str | None = None
    trend_subtitle: str | None = None


class TokenAnalyticsSummary(BaseModel):
    total_tokens_used: TokenSummaryValue
    remaining_tokens: TokenSummaryValue
    estimated_cost: TokenSummaryValue
    active_agents: TokenSummaryValue
    active_users: TokenSummaryValue
    requests_today: TokenSummaryValue
    avg_response_time: TokenSummaryValue
    error_rate: TokenSummaryValue


class TokenConsumptionSeries(BaseModel):
    name: str
    data: list[float]


class TokenConsumptionTrend(BaseModel):
    labels: list[str]
    series: list[TokenConsumptionSeries]


class ModelUsage(BaseModel):
    model: str
    percentage: float
    tokens: str
    color: str


class CostBreakdownItem(BaseModel):
    agent: str
    cost: str
    percentage: float
    color: str


router = APIRouter(tags=["Analytics Token"])


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


@router.get(
    "/analytics/{org_id}/summary",
    response_model=TokenAnalyticsSummary,
    dependencies=[Depends(security)],
)
async def get_analytics_summary(
    org_id: int,
    membership: OrganizationMembership = Depends(get_org_membership),
    db_session: Session = Depends(db.get_session),
):
    _verify_org_access(org_id, db_session)
    now = datetime.now(timezone.utc)
    # Current month period
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Query totals
    totals = db_session.exec(
        select(
            func.sum(LLMUsageEvent.total_tokens).label("total_tokens"),
            func.sum(LLMUsageEvent.cost_usd).label("total_cost"),
            func.count(func.distinct(LLMUsageEvent.agent_id)).label("active_agents"),
            func.count(func.distinct(LLMUsageEvent.user_id)).label("active_users"),
            func.avg(LLMUsageEvent.latency_ms).label("avg_latency"),
            func.count(LLMUsageEvent.id).label("total_requests"),
            func.sum(case((LLMUsageEvent.status != 'success', 1), else_=0)).label("errors")
        ).where(
            LLMUsageEvent.organization_id == org_id,
            LLMUsageEvent.created_at >= start_of_month
        )
    ).first()
    
    tokens = totals.total_tokens or 0
    cost = totals.total_cost or 0
    agents = totals.active_agents or 0
    users = totals.active_users or 0
    latency = totals.avg_latency or 0
    requests = totals.total_requests or 0
    errors = totals.errors or 0
    
    error_rate = (errors / requests * 100) if requests > 0 else 0
    
    # Today requests
    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    requests_today = db_session.exec(
        select(func.count(LLMUsageEvent.id))
        .where(
            LLMUsageEvent.organization_id == org_id,
            LLMUsageEvent.created_at >= start_of_today
        )
    ).first() or 0

    # Policy
    policy = db_session.exec(
        select(OrganizationTokenPolicy).where(OrganizationTokenPolicy.organization_id == org_id)
    ).first()
    # This view aggregates usage over the current month, so surface the monthly cap here.
    max_tokens = policy.monthly_max_tokens if policy and policy.monthly_limit_enabled else 0
    percentage = (tokens / max_tokens * 100) if max_tokens > 0 else 0
    
    remaining = max_tokens - tokens
    if remaining < 0: remaining = 0
    
    return TokenAnalyticsSummary(
        total_tokens_used=TokenSummaryValue(
            value=_format_compact(tokens), 
            total=_format_compact(max_tokens) if max_tokens else None, 
            percentage=round(percentage, 1) if max_tokens else None, 
            subtitle=f"{round(percentage, 1)}% of allocation" if max_tokens else "No limit set"
        ),
        remaining_tokens=TokenSummaryValue(
            value=_format_compact(remaining) if max_tokens else "∞", 
            percentage=round(100 - percentage, 1) if max_tokens else None, 
            subtitle=f"{round(100 - percentage, 1)}% remaining" if max_tokens else ""
        ),
        estimated_cost=TokenSummaryValue(value=f"${cost:.2f}", trend="+0%", trend_direction="up", trend_subtitle="vs last month"),
        active_agents=TokenSummaryValue(value=str(agents), subtitle="In this organization"),
        active_users=TokenSummaryValue(value=str(users), trend="+0", trend_direction="up", trend_subtitle="new users"),
        requests_today=TokenSummaryValue(value=str(requests_today), trend="+0%", trend_direction="up", trend_subtitle="vs yesterday"),
        avg_response_time=TokenSummaryValue(value=f"{(latency / 1000):.2f}s", trend="+0%", trend_direction="down", trend_subtitle="vs last month"),
        error_rate=TokenSummaryValue(value=f"{error_rate:.2f}%", trend="-0%", trend_direction="down", trend_subtitle="vs last month"),
    )


@router.get(
    "/analytics/{org_id}/trend",
    response_model=TokenConsumptionTrend,
    dependencies=[Depends(security)],
)
async def get_analytics_trend(
    org_id: int,
    membership: OrganizationMembership = Depends(get_org_membership),
    db_session: Session = Depends(db.get_session),
):
    _verify_org_access(org_id, db_session)
    
    # Last 7 days
    now = datetime.now(timezone.utc)
    start_date = now - timedelta(days=6)
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Generate last 7 days labels
    labels = [(start_date + timedelta(days=i)).strftime("%b %d") for i in range(7)]
    dates = [(start_date + timedelta(days=i)).date() for i in range(7)]
    
    query = select(
        func.date(LLMUsageEvent.created_at).label("day"),
        func.sum(LLMUsageEvent.input_tokens).label("input"),
        func.sum(LLMUsageEvent.output_tokens).label("output"),
        func.sum(LLMUsageEvent.embedding_tokens).label("embedding"),
        func.sum(LLMUsageEvent.tool_tokens).label("tool")
    ).where(
        LLMUsageEvent.organization_id == org_id,
        LLMUsageEvent.created_at >= start_date
    ).group_by(func.date(LLMUsageEvent.created_at))
    
    results = db_session.exec(query).all()
    
    data_map = {row.day: row for row in results}
    
    input_data = []
    output_data = []
    embedding_data = []
    tool_data = []
    
    for d in dates:
        row = data_map.get(d)
        input_data.append(int(row.input) if row and row.input else 0)
        output_data.append(int(row.output) if row and row.output else 0)
        embedding_data.append(int(row.embedding) if row and row.embedding else 0)
        tool_data.append(int(row.tool) if row and row.tool else 0)
            
    return TokenConsumptionTrend(
        labels=labels,
        series=[
            TokenConsumptionSeries(name="Input Tokens", data=input_data),
            TokenConsumptionSeries(name="Output Tokens", data=output_data),
            TokenConsumptionSeries(name="Embedding Tokens", data=embedding_data),
            TokenConsumptionSeries(name="Tool Tokens", data=tool_data)
        ]
    )


@router.get(
    "/analytics/{org_id}/models",
    response_model=List[ModelUsage],
    dependencies=[Depends(security)],
)
async def get_analytics_models(
    org_id: int,
    membership: OrganizationMembership = Depends(get_org_membership),
    db_session: Session = Depends(db.get_session),
):
    _verify_org_access(org_id, db_session)
    
    now = datetime.now(timezone.utc)
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    query = select(
        LLMUsageEvent.model,
        func.sum(LLMUsageEvent.total_tokens).label("tokens")
    ).where(
        LLMUsageEvent.organization_id == org_id,
        LLMUsageEvent.created_at >= start_of_month
    ).group_by(LLMUsageEvent.model).order_by(desc("tokens")).limit(5)
    
    results = db_session.exec(query).all()
    
    total_tokens = sum(row.tokens for row in results) if results else 1 # avoid div zero
    
    colors = ["#6C5CE7", "#10b981", "#f59e0b", "#ef4444", "#94a3b8"]
    
    response = []
    for i, row in enumerate(results):
        response.append(ModelUsage(
            model=row.model,
            percentage=round((row.tokens / total_tokens) * 100, 1),
            tokens=_format_compact(row.tokens),
            color=colors[i % len(colors)]
        ))
        
    return response


@router.get(
    "/analytics/{org_id}/cost-breakdown",
    response_model=List[CostBreakdownItem],
    dependencies=[Depends(security)],
)
async def get_analytics_cost_breakdown(
    org_id: int,
    membership: OrganizationMembership = Depends(get_org_membership),
    db_session: Session = Depends(db.get_session),
):
    _verify_org_access(org_id, db_session)
    
    now = datetime.now(timezone.utc)
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    query = select(
        AIAgent.name.label("agent_name"),
        func.sum(LLMUsageEvent.cost_usd).label("cost")
    ).join(AIAgent, LLMUsageEvent.agent_id == AIAgent.id).where(
        LLMUsageEvent.organization_id == org_id,
        LLMUsageEvent.created_at >= start_of_month
    ).group_by(AIAgent.name).order_by(desc("cost")).limit(5)
    
    results = db_session.exec(query).all()
    
    total_cost = sum(row.cost for row in results) if results else 1 # avoid div zero
    
    colors = ["#7E22CE", "#A855F7", "#22C55E", "#EAB308", "#94A3B8"]
    
    response = []
    for i, row in enumerate(results):
        response.append(CostBreakdownItem(
            agent=row.agent_name,
            cost=f"${row.cost:.2f}",
            percentage=round((row.cost / total_cost) * 100, 1),
            color=colors[i % len(colors)]
        ))
        
    return response
