"""Shared FastAPI dependencies for resolving the caller and their organization membership.

Generalizes the permission-checking logic that used to live as private helpers in
`api/v1/roles.py` (`_get_caller` / `_require_admin`), and extends it to check permissions
(`Role.permissions`) instead of a hardcoded role name.
"""

from fastapi import Depends, HTTPException, Path, status
from sqlmodel import Session, select

from ai_agent.core import db
from ai_agent.core.activity_tracking import record_activity
from ai_agent.core.permissions import has_permission
from ai_agent.core.token_validation import decode_token_payload
from ai_agent.models import OrganizationMembership, Role, User


def get_current_user(
    token_payload: dict = Depends(decode_token_payload),
    db_session: Session = Depends(db.get_session),
) -> User:
    """Resolve the authenticated caller from the JWT email claim."""
    email = token_payload.get("email", "").strip().lower()
    user = db_session.exec(select(User).where(User.email == email)).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    if user.active_organization_id is not None:
        record_activity(user.active_organization_id, user.id)

    return user


def _load_membership(user: User, organization_id: int | None, db_session: Session) -> OrganizationMembership:
    if organization_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not belong to any organization.",
        )
    membership = db_session.exec(
        select(OrganizationMembership).where(
            OrganizationMembership.user_id == user.id,
            OrganizationMembership.organization_id == organization_id,
            OrganizationMembership.is_active == True,  # noqa: E712
        )
    ).first()
    if not membership:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not a member of this organization.",
        )
    return membership


def get_active_membership(
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
) -> OrganizationMembership:
    """Resolve the caller's membership in their currently active organization."""
    return _load_membership(user, user.active_organization_id, db_session)


def get_org_membership(
    org_id: int = Path(..., description="Organization ID"),
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
) -> OrganizationMembership:
    """Resolve the caller's membership in the `{org_id}` path-scoped organization.

    Unlike `get_active_membership`, this does not require `org_id` to be the caller's
    currently active organization — a caller can view/manage any org they belong to.
    """
    return _load_membership(user, org_id, db_session)


def get_effective_permissions(
    membership: OrganizationMembership,
    user: User,
    db_session: Session,
) -> list[str]:
    """The caller's effective permission list for `membership`'s organization.

    `is_superuser`/`is_owner` bypass role permissions entirely (they're a safety net
    independent of role state — see ORGANIZATION_COLLABORATION.md §3d), represented here as
    the wildcard so callers can treat this list uniformly.
    """
    if user.is_superuser or membership.is_owner:
        return ["*"]
    role = db_session.get(Role, membership.role_id)
    return role.permissions or [] if role else []


def user_has_permission(
    membership: OrganizationMembership,
    user: User,
    permission: str,
    db_session: Session,
) -> bool:
    return has_permission(get_effective_permissions(membership, user, db_session), permission)


def _require(
    membership: OrganizationMembership,
    user: User,
    permission: str,
    db_session: Session,
) -> None:
    if not user_has_permission(membership, user, permission, db_session):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions.")


def require_permission(permission: str):
    """Dependency factory: caller must hold `permission` in their ACTIVE organization."""

    def dep(
        membership: OrganizationMembership = Depends(get_active_membership),
        user: User = Depends(get_current_user),
        db_session: Session = Depends(db.get_session),
    ) -> OrganizationMembership:
        _require(membership, user, permission, db_session)
        return membership

    return dep


def require_org_permission(permission: str):
    """Dependency factory: caller must hold `permission` in the `{org_id}` path-scoped org."""

    def dep(
        membership: OrganizationMembership = Depends(get_org_membership),
        user: User = Depends(get_current_user),
        db_session: Session = Depends(db.get_session),
    ) -> OrganizationMembership:
        _require(membership, user, permission, db_session)
        return membership

    return dep
