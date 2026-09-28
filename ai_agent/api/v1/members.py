"""Organization member management endpoints.

GET    /organizations/{org_id}/members            — list active members of an org.
PATCH  /organizations/{org_id}/members/{user_id}   — change a member's role.
DELETE /organizations/{org_id}/members/{user_id}   — remove a member, or leave (self-removal).

Viewing is open to any member of the org. Changing a role or removing someone else requires
`manage_members`; removing yourself ("leave") is always allowed except for the owner.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel
from sqlalchemy import func, or_
from sqlmodel import Session, select

from ai_agent.core import db
from ai_agent.core.deps import get_current_user, get_org_membership, require_org_permission, user_has_permission
from ai_agent.core.token_validation import security
from ai_agent.models import OrganizationMembership, Role, User

router = APIRouter(tags=["Members"])


class MemberOut(BaseModel):
    user_id: int
    email: str
    first_name: str
    last_name: str | None
    role_id: int
    role: str | None
    is_owner: bool
    joined_at: datetime


class MemberRoleUpdate(BaseModel):
    role_id: int


class PaginatedMembersOut(BaseModel):
    items: list[MemberOut]
    page: int
    limit: int
    total: int
    has_more: bool


def _to_member_out(membership: OrganizationMembership, user: User, role: Role | None) -> MemberOut:
    return MemberOut(
        user_id=user.id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        role_id=membership.role_id,
        role=role.name if role else None,
        is_owner=membership.is_owner,
        joined_at=membership.joined_at,
    )


def _get_target_membership(org_id: int, user_id: int, db_session: Session) -> OrganizationMembership:
    membership = db_session.exec(
        select(OrganizationMembership).where(
            OrganizationMembership.organization_id == org_id,
            OrganizationMembership.user_id == user_id,
            OrganizationMembership.is_active == True,  # noqa: E712
        )
    ).first()
    if not membership:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found.")
    return membership


@router.get(
    "/organizations/{org_id}/members",
    response_model=PaginatedMembersOut,
    summary="List active members of an organization",
    dependencies=[Depends(security)],
)
def list_members(
    org_id: int,
    search: str | None = Query(None, description="Search members by first name, last name, or email"),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    role_id: int | None = Query(None, alias="role"),
    _membership: OrganizationMembership = Depends(get_org_membership),
    db_session: Session = Depends(db.get_session),
) -> PaginatedMembersOut:
    filters = [
        OrganizationMembership.organization_id == org_id,
        OrganizationMembership.is_active == True,  
    ]
    if role_id is not None:
        filters.append(OrganizationMembership.role_id == role_id)

    search_term = search.strip() if search else None
    if search_term:
        pattern = f"%{search_term}%"
        filters.append(
            or_(
                User.first_name.ilike(pattern),
                User.last_name.ilike(pattern),
                User.email.ilike(pattern),
            )
        )

    offset = (page - 1) * limit
    total = db_session.exec(
        select(func.count())
        .select_from(OrganizationMembership)
        .join(User, User.id == OrganizationMembership.user_id)
        .where(*filters)
    ).one()

    rows = db_session.exec(
        select(OrganizationMembership, User, Role)
        .join(User, User.id == OrganizationMembership.user_id)
        .join(Role, Role.id == OrganizationMembership.role_id, isouter=True)
        .where(*filters)
        .offset(offset)
        .limit(limit)
    ).all()
    items = [_to_member_out(membership, user, role) for membership, user, role in rows]
    return PaginatedMembersOut(
        items=items,
        page=page,
        limit=limit,
        total=total,
        has_more=(offset + len(items)) < total,
    )


@router.patch(
    "/organizations/{org_id}/members/{user_id}",
    response_model=MemberOut,
    summary="Change a member's role",
    dependencies=[Depends(security)],
)
def update_member_role(
    body: MemberRoleUpdate,
    org_id: int,
    user_id: int = Path(..., description="ID of the member to update"),
    _caller_membership: OrganizationMembership = Depends(require_org_permission("manage_members")),
    db_session: Session = Depends(db.get_session),
) -> MemberOut:
    """
    Returns 403 if the caller lacks `manage_members`, or if `user_id` is the org's owner
    (the owner's role can't be changed by anyone — they keep full access via `is_owner`
    regardless of role).
    Returns 404 if the member or the target role doesn't exist in this org.
    """
    target = _get_target_membership(org_id, user_id, db_session)
    if target.is_owner:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot change the owner's role.")

    role = db_session.get(Role, body.role_id)
    if role is None or role.organization_id != org_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found.")

    target.role_id = body.role_id
    target.updated_at = datetime.now(timezone.utc)
    db_session.add(target)
    db_session.commit()
    db_session.refresh(target)

    member_user = db_session.get(User, user_id)
    return _to_member_out(target, member_user, role)


@router.delete(
    "/organizations/{org_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a member, or leave the organization (self-removal)",
    dependencies=[Depends(security)],
)
def remove_member(
    org_id: int,
    user_id: int = Path(..., description="ID of the member to remove"),
    caller_membership: OrganizationMembership = Depends(get_org_membership),
    caller: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
) -> None:
    """
    Removing someone else requires `manage_members`; removing yourself ("leave") is always
    allowed. The organization's owner can never be removed, including by themselves — they
    must transfer ownership or delete the org first (not yet supported).

    If the removed user's active organization was this one, it's reassigned to another of
    their remaining memberships, or cleared if this was their only org.
    """
    is_self = user_id == caller.id
    if not is_self and not user_has_permission(caller_membership, caller, "manage_members", db_session):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions.")

    target = _get_target_membership(org_id, user_id, db_session)
    if target.is_owner:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The organization owner cannot be removed.",
        )

    target.is_active = False
    target.removed_at = datetime.now(timezone.utc)
    db_session.add(target)

    member_user = db_session.get(User, user_id)
    if member_user and member_user.active_organization_id == org_id:
        next_membership = db_session.exec(
            select(OrganizationMembership).where(
                OrganizationMembership.user_id == user_id,
                OrganizationMembership.organization_id != org_id,
                OrganizationMembership.is_active == True,  # noqa: E712
            )
        ).first()
        member_user.active_organization_id = next_membership.organization_id if next_membership else None
        db_session.add(member_user)

    db_session.commit()
