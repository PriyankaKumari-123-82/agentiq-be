from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlmodel import Session, select
from typing import List, Optional

from ai_agent.core import db
from ai_agent.core.deps import get_org_membership
from ai_agent.core.token_validation import security
from ai_agent.models import AIAgent, Organization, OrganizationMembership, OrganizationTokenPolicy, AgentTokenPolicy, UserTokenPolicy, User

router = APIRouter(tags=["Token Governance"])


class TokenPolicyUpdate(BaseModel):
    limit_enabled: bool
    max_tokens: int
    reset_period: str
    alerts_enabled: bool
    alert_threshold_percentage: int
    notification_email: Optional[str] = None


class MultiPeriodTokenPolicy(BaseModel):
    """Daily, weekly and monthly caps are independent and can be enabled simultaneously."""
    daily_limit_enabled: bool
    daily_max_tokens: int
    weekly_limit_enabled: bool
    weekly_max_tokens: int
    monthly_limit_enabled: bool
    monthly_max_tokens: int
    alerts_enabled: bool
    alert_threshold_percentage: int
    notification_email: Optional[str] = None


class AgentTokenAllocationCreate(MultiPeriodTokenPolicy):
    agent_ids: List[str]


class AgentTokenAllocationResponse(MultiPeriodTokenPolicy):
    agent_id: str


class AgentPolicyListItem(MultiPeriodTokenPolicy):
    agent_id: str
    agent_name: str
    organization: str
    status: str


class UserTokenAllocationCreate(MultiPeriodTokenPolicy):
    user_ids: List[int]


class UserTokenAllocationResponse(MultiPeriodTokenPolicy):
    user_id: int


class UserPolicyListItem(MultiPeriodTokenPolicy):
    user_id: int
    user_name: str
    organization: str
    status: str


def _policy_to_multi_period(policy) -> dict:
    return dict(
        daily_limit_enabled=policy.daily_limit_enabled,
        daily_max_tokens=policy.daily_max_tokens,
        weekly_limit_enabled=policy.weekly_limit_enabled,
        weekly_max_tokens=policy.weekly_max_tokens,
        monthly_limit_enabled=policy.monthly_limit_enabled,
        monthly_max_tokens=policy.monthly_max_tokens,
        alerts_enabled=policy.alerts_enabled,
        alert_threshold_percentage=policy.alert_threshold_percentage,
        notification_email=policy.notification_email,
    )


def _apply_multi_period(policy, data: MultiPeriodTokenPolicy) -> None:
    policy.daily_limit_enabled = data.daily_limit_enabled
    policy.daily_max_tokens = data.daily_max_tokens
    policy.weekly_limit_enabled = data.weekly_limit_enabled
    policy.weekly_max_tokens = data.weekly_max_tokens
    policy.monthly_limit_enabled = data.monthly_limit_enabled
    policy.monthly_max_tokens = data.monthly_max_tokens
    policy.alerts_enabled = data.alerts_enabled
    policy.alert_threshold_percentage = data.alert_threshold_percentage
    policy.notification_email = data.notification_email


_DEFAULT_ORG_POLICY = dict(
    daily_limit_enabled=False, daily_max_tokens=0,
    weekly_limit_enabled=False, weekly_max_tokens=0,
    monthly_limit_enabled=False, monthly_max_tokens=0,
    alerts_enabled=False, alert_threshold_percentage=80, notification_email=None,
)

_DEFAULT_AGENT_POLICY = dict(
    daily_limit_enabled=False, daily_max_tokens=0,
    weekly_limit_enabled=False, weekly_max_tokens=0,
    monthly_limit_enabled=False, monthly_max_tokens=0,
    alerts_enabled=False, alert_threshold_percentage=80, notification_email=None,
)

_DEFAULT_USER_POLICY = dict(
    daily_limit_enabled=False, daily_max_tokens=0,
    weekly_limit_enabled=False, weekly_max_tokens=0,
    monthly_limit_enabled=False, monthly_max_tokens=0,
    alerts_enabled=False, alert_threshold_percentage=80, notification_email=None,
)


@router.get("/governance/{org_id}/policy", response_model=MultiPeriodTokenPolicy, dependencies=[Depends(security)])
async def get_org_policy(
    org_id: int,
    membership: OrganizationMembership = Depends(get_org_membership),
    db_session: Session = Depends(db.get_session),
):
    policy = db_session.exec(select(OrganizationTokenPolicy).where(OrganizationTokenPolicy.organization_id == org_id)).first()
    if not policy:
        return MultiPeriodTokenPolicy(**_DEFAULT_ORG_POLICY)

    return MultiPeriodTokenPolicy(**_policy_to_multi_period(policy))


@router.post("/governance/{org_id}/policy", response_model=MultiPeriodTokenPolicy, dependencies=[Depends(security)])
async def update_org_policy(
    org_id: int,
    data: MultiPeriodTokenPolicy,
    membership: OrganizationMembership = Depends(get_org_membership),
    db_session: Session = Depends(db.get_session),
):
    policy = db_session.exec(select(OrganizationTokenPolicy).where(OrganizationTokenPolicy.organization_id == org_id)).first()
    if not policy:
        policy = OrganizationTokenPolicy(organization_id=org_id)
        db_session.add(policy)

    _apply_multi_period(policy, data)

    db_session.commit()
    db_session.refresh(policy)

    return data


@router.get("/governance/{agent_id}/policy", response_model=MultiPeriodTokenPolicy, dependencies=[Depends(security)])
async def get_agent_policy(
    agent_id: str,
    db_session: Session = Depends(db.get_session),
):
    policy = db_session.exec(select(AgentTokenPolicy).where(AgentTokenPolicy.agent_id == agent_id)).first()
    if not policy:
        return MultiPeriodTokenPolicy(**_DEFAULT_AGENT_POLICY)
    return MultiPeriodTokenPolicy(**_policy_to_multi_period(policy))


@router.post("/governance/{agent_id}/policy", response_model=MultiPeriodTokenPolicy, dependencies=[Depends(security)])
async def update_agent_policy(
    agent_id: str,
    data: MultiPeriodTokenPolicy,
    db_session: Session = Depends(db.get_session),
):
    policy = db_session.exec(select(AgentTokenPolicy).where(AgentTokenPolicy.agent_id == agent_id)).first()
    if not policy:
        policy = AgentTokenPolicy(agent_id=agent_id)
        db_session.add(policy)

    _apply_multi_period(policy, data)

    db_session.commit()
    db_session.refresh(policy)

    return data


@router.post(
    "/governance/agents/allocation",
    response_model=List[AgentTokenAllocationResponse],
    dependencies=[Depends(security)],
)
async def allocate_agent_tokens(
    data: AgentTokenAllocationCreate,
    db_session: Session = Depends(db.get_session),
):
    """Create/update the same token allocation policy for a list of agents at once."""
    if not data.agent_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="agent_ids must not be empty")

    results: List[AgentTokenAllocationResponse] = []
    for agent_id in data.agent_ids:
        policy = db_session.exec(select(AgentTokenPolicy).where(AgentTokenPolicy.agent_id == agent_id)).first()
        if not policy:
            policy = AgentTokenPolicy(agent_id=agent_id)
            db_session.add(policy)

        _apply_multi_period(policy, data)

        results.append(AgentTokenAllocationResponse(agent_id=agent_id, **_policy_to_multi_period(policy)))

    db_session.commit()

    return results


@router.get(
    "/governance/agents/allocation",
    response_model=List[AgentTokenAllocationResponse],
    dependencies=[Depends(security)],
)
async def get_agent_token_allocations(
    agent_ids: List[str] = Query(..., description="Agent IDs to fetch the token allocation for"),
    db_session: Session = Depends(db.get_session),
):
    """Fetch the token allocation policy for one or more agents by id."""
    policies = db_session.exec(select(AgentTokenPolicy).where(AgentTokenPolicy.agent_id.in_(agent_ids))).all()
    policy_by_agent_id = {policy.agent_id: policy for policy in policies}

    results: List[AgentTokenAllocationResponse] = []
    for agent_id in agent_ids:
        policy = policy_by_agent_id.get(agent_id)
        if not policy:
            results.append(AgentTokenAllocationResponse(agent_id=agent_id, **_DEFAULT_AGENT_POLICY))
            continue

        results.append(AgentTokenAllocationResponse(agent_id=agent_id, **_policy_to_multi_period(policy)))

    return results


@router.get(
    "/governance/{org_id}/agents/policies",
    response_model=List[AgentPolicyListItem],
    dependencies=[Depends(security)],
)
async def list_agent_policies(
    org_id: int,
    membership: OrganizationMembership = Depends(get_org_membership),
    db_session: Session = Depends(db.get_session),
):
    """List every agent in the organization with its token allocation policy, for the Agent Policies table."""
    org = db_session.get(Organization, org_id)
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    agents = db_session.exec(
        select(AIAgent).where(
            AIAgent.organization_id == org_id,
            AIAgent.deleted == False,  # noqa: E712
        )
    ).all()

    policies = db_session.exec(
        select(AgentTokenPolicy).where(AgentTokenPolicy.agent_id.in_([agent.id for agent in agents]))
    ).all()
    policy_by_agent_id = {policy.agent_id: policy for policy in policies}

    results: List[AgentPolicyListItem] = []
    for agent in agents:
        policy = policy_by_agent_id.get(agent.id)
        if not policy:
            results.append(
                AgentPolicyListItem(
                    agent_id=agent.id,
                    agent_name=agent.name,
                    organization=org.name,
                    status="Unlimited",
                    **_DEFAULT_AGENT_POLICY,
                )
            )
            continue

        any_limit_enabled = policy.daily_limit_enabled or policy.weekly_limit_enabled or policy.monthly_limit_enabled
        results.append(
            AgentPolicyListItem(
                agent_id=agent.id,
                agent_name=agent.name,
                organization=org.name,
                status="Active" if any_limit_enabled else "Unlimited",
                **_policy_to_multi_period(policy),
            )
        )

    return results


@router.get("/governance/{user_id}/policy", response_model=MultiPeriodTokenPolicy, dependencies=[Depends(security)])
async def get_user_policy(
    user_id: int,
    db_session: Session = Depends(db.get_session),
):
    policy = db_session.exec(select(UserTokenPolicy).where(UserTokenPolicy.user_id == user_id)).first()
    if not policy:
        return MultiPeriodTokenPolicy(**_DEFAULT_USER_POLICY)
    return MultiPeriodTokenPolicy(**_policy_to_multi_period(policy))


@router.post("/governance/{user_id}/policy", response_model=MultiPeriodTokenPolicy, dependencies=[Depends(security)])
async def update_user_policy(
    user_id: int,
    data: MultiPeriodTokenPolicy,
    db_session: Session = Depends(db.get_session),
):
    policy = db_session.exec(select(UserTokenPolicy).where(UserTokenPolicy.user_id == user_id)).first()
    if not policy:
        policy = UserTokenPolicy(user_id=user_id)
        db_session.add(policy)

    _apply_multi_period(policy, data)

    db_session.commit()
    db_session.refresh(policy)

    return data


@router.post(
    "/governance/users/allocation",
    response_model=List[UserTokenAllocationResponse],
    dependencies=[Depends(security)],
)
async def allocate_user_tokens(
    data: UserTokenAllocationCreate,
    db_session: Session = Depends(db.get_session),
):
    """Create/update the same token allocation policy for a list of users at once."""
    if not data.user_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="user_ids must not be empty")

    results: List[UserTokenAllocationResponse] = []
    for user_id in data.user_ids:
        policy = db_session.exec(select(UserTokenPolicy).where(UserTokenPolicy.user_id == user_id)).first()
        if not policy:
            policy = UserTokenPolicy(user_id=user_id)
            db_session.add(policy)

        _apply_multi_period(policy, data)

        results.append(UserTokenAllocationResponse(user_id=user_id, **_policy_to_multi_period(policy)))

    db_session.commit()

    return results


@router.get(
    "/governance/users/allocation",
    response_model=List[UserTokenAllocationResponse],
    dependencies=[Depends(security)],
)
async def get_user_token_allocations(
    user_ids: List[int] = Query(..., description="User IDs to fetch the token allocation for"),
    db_session: Session = Depends(db.get_session),
):
    """Fetch the token allocation policy for one or more users by id."""
    policies = db_session.exec(select(UserTokenPolicy).where(UserTokenPolicy.user_id.in_(user_ids))).all()
    policy_by_user_id = {policy.user_id: policy for policy in policies}

    results: List[UserTokenAllocationResponse] = []
    for user_id in user_ids:
        policy = policy_by_user_id.get(user_id)
        if not policy:
            results.append(UserTokenAllocationResponse(user_id=user_id, **_DEFAULT_USER_POLICY))
            continue

        results.append(UserTokenAllocationResponse(user_id=user_id, **_policy_to_multi_period(policy)))

    return results


@router.get(
    "/governance/{org_id}/users/policies",
    response_model=List[UserPolicyListItem],
    dependencies=[Depends(security)],
)
async def list_user_policies(
    org_id: int,
    membership: OrganizationMembership = Depends(get_org_membership),
    db_session: Session = Depends(db.get_session),
):
    """List every member of the organization with their token allocation policy, for the User Policies table."""
    org = db_session.get(Organization, org_id)
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    members = db_session.exec(
        select(User)
        .join(OrganizationMembership, OrganizationMembership.user_id == User.id)
        .where(
            OrganizationMembership.organization_id == org_id,
            OrganizationMembership.is_active == True,  # noqa: E712
        )
    ).all()

    policies = db_session.exec(
        select(UserTokenPolicy).where(UserTokenPolicy.user_id.in_([member.id for member in members]))
    ).all()
    policy_by_user_id = {policy.user_id: policy for policy in policies}

    results: List[UserPolicyListItem] = []
    for member in members:
        policy = policy_by_user_id.get(member.id)
        user_name = f"{member.first_name} {member.last_name or ''}".strip()
        if not policy:
            results.append(
                UserPolicyListItem(
                    user_id=member.id,
                    user_name=user_name,
                    organization=org.name,
                    status="Unlimited",
                    **_DEFAULT_USER_POLICY,
                )
            )
            continue

        any_limit_enabled = policy.daily_limit_enabled or policy.weekly_limit_enabled or policy.monthly_limit_enabled
        results.append(
            UserPolicyListItem(
                user_id=member.id,
                user_name=user_name,
                organization=org.name,
                status="Active" if any_limit_enabled else "Unlimited",
                **_policy_to_multi_period(policy),
            )
        )

    return results
