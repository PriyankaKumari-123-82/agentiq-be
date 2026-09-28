"""Organization invitation endpoints.

Admin-facing (require `manage_members` in the org):
    POST   /organizations/{org_id}/invitations                    — invite by email + role.
    GET    /organizations/{org_id}/invitations                    — list invites (filter by status).
    DELETE /organizations/{org_id}/invitations/{invitation_id}    — cancel a pending invite.
    POST   /organizations/{org_id}/invitations/{invitation_id}/resend — resend, refresh token+expiry.

Public (invitee-facing, token-based):
    GET  /invitations/{token}          — resolve an invite for the accept-invite landing page.
    POST /invitations/{token}/accept   — accept: signs up the user if new, else attaches
                                          membership to their existing account.

Business logic lives directly in this router rather than a separate `services/` module,
matching the pattern already established by organization.py/roles.py/members.py in this
codebase — `services/` here is reserved for external-integration wrappers (see identity.py).
"""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func
from sqlmodel import Session, select

from ai_agent.core import db
from ai_agent.core.config import settings
from ai_agent.core.deps import require_org_permission
from ai_agent.core.email_service import email_service
from ai_agent.core.token_validation import security, validate_token
from ai_agent.models import (
    InvitationStatus,
    Organization,
    OrganizationInvitation,
    OrganizationMembership,
    Role,
    User,
    UserSettings,
)
from ai_agent.services.identity import authenticate_user, create_identity_user

log = logging.getLogger(__name__)

router = APIRouter(tags=["Invitations"])

INVITATION_EXPIRY_DAYS = 7


def _generate_token() -> tuple[str, str]:
    """Returns (raw_token, token_hash). Only the hash is ever persisted."""
    raw = secrets.token_urlsafe(32)
    return raw, hashlib.sha256(raw.encode()).hexdigest()


def _hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode()).hexdigest()


def _display_name(user: User | None) -> str:
    """Full name if available, else email, else a generic fallback — for the invite email's
    "invited by" line, which otherwise reads as a raw email address."""
    if not user:
        return "An admin"
    name = " ".join(part for part in (user.first_name, user.last_name) if part)
    return name or user.email


def _now_naive() -> datetime:
    """UTC now with tzinfo stripped.

    Every datetime column in this codebase is a naive TIMESTAMP (see models.py), so a value
    written as tz-aware comes back from the DB naive on the next read. Comparing that against
    a fresh tz-aware `datetime.now(timezone.utc)` raises TypeError — use this instead whenever
    comparing "now" against a value read back from the DB (e.g. `invitation.expires_at`).
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _send_invitation_email(to_email: str, org_name: str, inviter_name: str, role_name: str, raw_token: str) -> None:
    accept_url = f"{settings.FRONTEND_URL}/invitations/accept?token={raw_token}"
    html_body = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;padding:40px 0;">
    <tr><td align="center">
      <table width="560" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.06);">
        <tr><td style="background:linear-gradient(135deg,#1d4ed8,#3b82f6);padding:32px 40px;text-align:center;">
          <h1 style="margin:0;color:#ffffff;font-size:22px;font-weight:700;">AgentIQ</h1>
        </td></tr>
        <tr><td style="padding:40px;">
          <h2 style="margin:0 0 16px;color:#1e293b;font-size:20px;">You've been invited to join {org_name}</h2>
          <p style="margin:0 0 12px;color:#475569;font-size:15px;"><strong>{inviter_name}</strong> has invited you to join <strong>{org_name}</strong> on AgentIQ as <strong>{role_name}</strong>.</p>
          <p style="margin:0 0 28px;color:#475569;font-size:15px;">Click below to accept the invitation and get started.</p>
          <table cellpadding="0" cellspacing="0"><tr><td style="border-radius:8px;background:linear-gradient(135deg,#1d4ed8,#3b82f6);">
            <a href="{accept_url}" target="_blank" rel="noopener noreferrer" style="display:inline-block;padding:12px 28px;color:#ffffff;text-decoration:none;font-size:15px;font-weight:600;">Accept Invitation</a>
          </td></tr></table>
          <p style="margin:24px 0 0;color:#94a3b8;font-size:13px;">This invitation expires in {INVITATION_EXPIRY_DAYS} days. If you weren't expecting this, you can safely ignore this email.</p>
        </td></tr>
        <tr><td style="padding:24px 40px;border-top:1px solid #e2e8f0;text-align:center;">
          <p style="margin:0;color:#94a3b8;font-size:13px;">AgentIQ &mdash; All rights reserved.</p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""
    text_body = (
        f"You've been invited to join {org_name}\n\n"
        f"{inviter_name} has invited you to join {org_name} on AgentIQ as {role_name}.\n\n"
        f"Accept your invitation: {accept_url}\n\n"
        f"This invitation expires in {INVITATION_EXPIRY_DAYS} days.\n\nAgentIQ"
    )
    email_service.send_html_email(
        to_email=to_email,
        subject=f"You've been invited to join {org_name} on AgentIQ",
        html_body=html_body,
        text_body=text_body,
    )


# ---------------------------------------------------------------------------
# Pydantic shapes
# ---------------------------------------------------------------------------

class InvitationOut(BaseModel):
    id: int
    organization_id: int
    email: str
    role_id: int
    role: str | None
    invited_by_email: str | None
    status: str
    expires_at: datetime
    accepted_at: datetime | None
    created_at: datetime


class InvitationCreate(BaseModel):
    email: EmailStr
    role_id: int


class PaginatedInvitationsOut(BaseModel):
    items: list[InvitationOut]
    page: int
    limit: int
    total: int
    has_more: bool


class InvitationTokenInfo(BaseModel):
    organization_name: str
    inviter_name: str
    role_name: str | None
    email: str
    status: str
    is_expired: bool
    user_exists: bool = Field(description="True if the invited email already has an AgentIQ account")


class InvitationAcceptRequest(BaseModel):
    # Required only when the invited email has no existing account (new-user signup branch).
    password: str | None = Field(default=None, min_length=2)
    first_name: str | None = None
    last_name: str | None = None


class InvitationAcceptResponse(BaseModel):
    organization_id: int
    organization_name: str
    role: str | None
    # Populated only for brand-new users, so the frontend can log them straight in.
    access_token: str | None = None
    refresh_token: str | None = None
    token_type: str | None = None
    expires_in: int | None = None


def _to_invitation_out(invitation: OrganizationInvitation, role: Role | None, inviter: User | None) -> InvitationOut:
    return InvitationOut(
        id=invitation.id,
        organization_id=invitation.organization_id,
        email=invitation.email,
        role_id=invitation.role_id,
        role=role.name if role else None,
        invited_by_email=inviter.email if inviter else None,
        status=invitation.status,
        expires_at=invitation.expires_at,
        accepted_at=invitation.accepted_at,
        created_at=invitation.created_at,
    )


# ---------------------------------------------------------------------------
# Admin-facing endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/organizations/{org_id}/invitations",
    response_model=InvitationOut,
    status_code=status.HTTP_201_CREATED,
    summary="Invite a user by email",
    dependencies=[Depends(security)],
)
def create_invitation(
    body: InvitationCreate,
    org_id: int,
    caller: OrganizationMembership = Depends(require_org_permission("manage_members")),
    db_session: Session = Depends(db.get_session),
) -> InvitationOut:
    """
    Returns 400 if inviting yourself.
    Returns 404 if `role_id` doesn't belong to this org.
    Returns 409 if the email is already an active member, or already has a pending invite.
    Returns 502 if the invite email fails to send (the invitation is not created in that case).
    """
    email = body.email.strip().lower()

    caller_user = db_session.get(User, caller.user_id)
    if caller_user and caller_user.email == email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You can't invite yourself.")

    role = db_session.get(Role, body.role_id)
    if role is None or role.organization_id != org_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found.")

    existing_user = db_session.exec(select(User).where(User.email == email)).first()
    if existing_user:
        active_membership = db_session.exec(
            select(OrganizationMembership).where(
                OrganizationMembership.organization_id == org_id,
                OrganizationMembership.user_id == existing_user.id,
                OrganizationMembership.is_active == True,  # noqa: E712
            )
        ).first()
        if active_membership:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This user is already a member of this organization.",
            )

    pending = db_session.exec(
        select(OrganizationInvitation).where(
            OrganizationInvitation.organization_id == org_id,
            OrganizationInvitation.email == email,
            OrganizationInvitation.status == InvitationStatus.PENDING,
        )
    ).first()
    if pending:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An invitation is already pending for this email. Cancel or resend it instead.",
        )

    raw_token, token_hash = _generate_token()
    now = datetime.now(timezone.utc)
    invitation = OrganizationInvitation(
        organization_id=org_id,
        email=email,
        role_id=body.role_id,
        invited_by_user_id=caller.user_id,
        token_hash=token_hash,
        status=InvitationStatus.PENDING,
        expires_at=now + timedelta(days=INVITATION_EXPIRY_DAYS),
    )
    db_session.add(invitation)
    db_session.flush()

    org = db_session.get(Organization, org_id)
    try:
        _send_invitation_email(email, org.name, _display_name(caller_user), role.name, raw_token)
    except Exception as exc:
        db_session.rollback()
        log.error("Failed to send invitation email to %s: %s", email, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to send the invitation email. Please try again.",
        )

    db_session.commit()
    db_session.refresh(invitation)
    return _to_invitation_out(invitation, role, caller_user)


@router.get(
    "/organizations/{org_id}/invitations",
    response_model=PaginatedInvitationsOut,
    summary="List invitations for an organization",
    dependencies=[Depends(security)],
)
def list_invitations(
    org_id: int,
    search: str | None = Query(None, description="Search invitations by email"),
    invite_status: str | None = Query(default=None, alias="status", description="Filter by status"),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    _caller: OrganizationMembership = Depends(require_org_permission("manage_members")),
    db_session: Session = Depends(db.get_session),
) -> PaginatedInvitationsOut:
    filters = [
        OrganizationInvitation.organization_id == org_id,
    ]
    if invite_status and invite_status != "all":
        filters.append(OrganizationInvitation.status == invite_status)

    search_term = search.strip() if search else None
    if search_term:
        pattern = f"%{search_term}%"
        filters.append(OrganizationInvitation.email.ilike(pattern))

    offset = (page - 1) * limit
    total = db_session.exec(
        select(func.count())
        .select_from(OrganizationInvitation)
        .where(*filters)
    ).one()

    invitations = db_session.exec(
        select(OrganizationInvitation)
        .where(*filters)
        .order_by(OrganizationInvitation.created_at.desc())
        .offset(offset)
        .limit(limit)
    ).all()

    items = []
    for invitation in invitations:
        role = db_session.get(Role, invitation.role_id)
        inviter = db_session.get(User, invitation.invited_by_user_id)
        items.append(_to_invitation_out(invitation, role, inviter))
    return PaginatedInvitationsOut(
        items=items,
        page=page,
        limit=limit,
        total=total,
        has_more=(offset + len(items)) < total,
    )


@router.delete(
    "/organizations/{org_id}/invitations/{invitation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel a pending invitation",
    dependencies=[Depends(security)],
)
def cancel_invitation(
    org_id: int,
    invitation_id: int = Path(...),
    _caller: OrganizationMembership = Depends(require_org_permission("manage_members")),
    db_session: Session = Depends(db.get_session),
) -> None:
    invitation = db_session.get(OrganizationInvitation, invitation_id)
    if invitation is None or invitation.organization_id != org_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invitation not found.")
    if invitation.status != InvitationStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only pending invitations can be cancelled.",
        )

    invitation.status = InvitationStatus.CANCELLED
    invitation.cancelled_at = datetime.now(timezone.utc)
    invitation.updated_at = datetime.now(timezone.utc)
    db_session.add(invitation)
    db_session.commit()


@router.post(
    "/organizations/{org_id}/invitations/{invitation_id}/resend",
    response_model=InvitationOut,
    summary="Resend an invitation with a fresh token and expiry",
    dependencies=[Depends(security)],
)
def resend_invitation(
    org_id: int,
    invitation_id: int = Path(...),
    caller: OrganizationMembership = Depends(require_org_permission("manage_members")),
    db_session: Session = Depends(db.get_session),
) -> InvitationOut:
    """
    Allowed for PENDING or EXPIRED invitations (revives an expired one); returns 400 for
    CANCELLED or already-ACCEPTED invitations. Issuing a new token immediately invalidates
    the previous invite link, since only the latest token's hash is stored.
    Returns 502 if the email fails to send — the previous token/expiry are left in place.
    """
    invitation = db_session.get(OrganizationInvitation, invitation_id)
    if invitation is None or invitation.organization_id != org_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invitation not found.")
    if invitation.status not in (InvitationStatus.PENDING, InvitationStatus.EXPIRED):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only pending or expired invitations can be resent.",
        )

    role = db_session.get(Role, invitation.role_id)
    org = db_session.get(Organization, org_id)
    caller_user = db_session.get(User, caller.user_id)

    raw_token, token_hash = _generate_token()
    try:
        _send_invitation_email(
            invitation.email, org.name, _display_name(caller_user),
            role.name if role else "Member", raw_token,
        )
    except Exception as exc:
        log.error("Failed to resend invitation email to %s: %s", invitation.email, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to send the invitation email. Please try again.",
        )

    invitation.token_hash = token_hash
    invitation.status = InvitationStatus.PENDING
    invitation.expires_at = datetime.now(timezone.utc) + timedelta(days=INVITATION_EXPIRY_DAYS)
    invitation.updated_at = datetime.now(timezone.utc)
    db_session.add(invitation)
    db_session.commit()
    db_session.refresh(invitation)
    return _to_invitation_out(invitation, role, caller_user)


# ---------------------------------------------------------------------------
# Public, token-based endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/invitations/{token}",
    response_model=InvitationTokenInfo,
    summary="Resolve an invitation for the accept-invite landing page",
)
def get_invitation(
    token: str,
    db_session: Session = Depends(db.get_session),
) -> InvitationTokenInfo:
    invitation = db_session.exec(
        select(OrganizationInvitation).where(OrganizationInvitation.token_hash == _hash_token(token))
    ).first()
    if invitation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invitation not found.")

    org = db_session.get(Organization, invitation.organization_id)
    role = db_session.get(Role, invitation.role_id)
    inviter = db_session.get(User, invitation.invited_by_user_id)
    user_exists = db_session.exec(select(User).where(User.email == invitation.email)).first() is not None

    return InvitationTokenInfo(
        organization_name=org.name if org else "",
        inviter_name=_display_name(inviter),
        role_name=role.name if role else None,
        email=invitation.email,
        status=invitation.status,
        is_expired=invitation.expires_at < _now_naive(),
        user_exists=user_exists,
    )


@router.post(
    "/invitations/{token}/accept",
    response_model=InvitationAcceptResponse,
    summary="Accept an invitation",
)
async def accept_invitation(
    token: str,
    body: InvitationAcceptRequest,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db_session: Session = Depends(db.get_session),
) -> InvitationAcceptResponse:
    """
    If the invited email has no local account yet, `password` and `first_name` are required
    and a new account is created (email pre-verified — see identity.create_identity_user).
    If the invited email already has an account, the caller must be authenticated as that
    same email (log in first, then call this endpoint).

    Returns 400 if a new-user signup is missing `password`/`first_name`.
    Returns 403 if the caller is logged in as a different email than the invite.
    Returns 404 if the token doesn't match any invitation.
    Returns 409 if the invitation was already accepted, or the user is already a member.
    Returns 410 if the invitation was cancelled or has expired.
    """
    invitation = db_session.exec(
        select(OrganizationInvitation).where(OrganizationInvitation.token_hash == _hash_token(token))
    ).first()
    if invitation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invitation not found.")

    if invitation.status == InvitationStatus.CANCELLED:
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="This invitation was cancelled.")
    if invitation.status == InvitationStatus.ACCEPTED:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This invitation has already been accepted.")
    if invitation.status == InvitationStatus.EXPIRED or invitation.expires_at < _now_naive():
        invitation.status = InvitationStatus.EXPIRED
        db_session.add(invitation)
        db_session.commit()
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="This invitation has expired. Ask the admin to resend it.",
        )

    existing_user = db_session.exec(select(User).where(User.email == invitation.email)).first()
    tokens: dict | None = None

    if existing_user:
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Please log in as {invitation.email} to accept this invitation.",
            )
        payload = await validate_token(credentials.credentials)
        caller_email = payload.get("email", "").strip().lower()
        if caller_email != invitation.email:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Please log in as {invitation.email} to accept this invitation.",
            )
        target_user = existing_user
    else:
        if not body.password or not body.first_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="password and first_name are required to accept this invitation.",
            )
        try:
            await create_identity_user(
                email=invitation.email,
                password=body.password,
                first_name=body.first_name,
                last_name=body.last_name,
                email_verified=True,
            )
        except HTTPException as exc:
            if exc.status_code == status.HTTP_409_CONFLICT:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="An account with this email already exists. Please log in to accept this invitation.",
                )
            raise

        target_user = User(
            email=invitation.email,
            first_name=body.first_name,
            last_name=body.last_name,
        )
        db_session.add(target_user)
        db_session.flush()
        db_session.add(UserSettings(
            user_id=target_user.id,
            theme="light",
            language="en",
        ))

        auth_result = await authenticate_user(email=invitation.email, password=body.password)
        tokens = auth_result

    already_member = db_session.exec(
        select(OrganizationMembership).where(
            OrganizationMembership.organization_id == invitation.organization_id,
            OrganizationMembership.user_id == target_user.id,
            OrganizationMembership.is_active == True,  # noqa: E712
        )
    ).first()
    if already_member:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="You're already a member of this organization.",
        )

    db_session.add(OrganizationMembership(
        user_id=target_user.id,
        organization_id=invitation.organization_id,
        role_id=invitation.role_id,
        is_owner=False,
    ))

    if target_user.active_organization_id is None:
        target_user.active_organization_id = invitation.organization_id
        db_session.add(target_user)

    invitation.status = InvitationStatus.ACCEPTED
    invitation.accepted_at = datetime.now(timezone.utc)
    invitation.accepted_user_id = target_user.id
    invitation.updated_at = datetime.now(timezone.utc)
    db_session.add(invitation)

    db_session.commit()

    org = db_session.get(Organization, invitation.organization_id)
    role = db_session.get(Role, invitation.role_id)

    return InvitationAcceptResponse(
        organization_id=org.id,
        organization_name=org.name,
        role=role.name if role else None,
        access_token=tokens.get("access_token") if tokens else None,
        refresh_token=tokens.get("refresh_token") if tokens else None,
        token_type=tokens.get("token_type") if tokens else None,
        expires_in=tokens.get("expires_in") if tokens else None,
    )
