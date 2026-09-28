from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from fastapi.responses import RedirectResponse
from sqlmodel import Session, select
from sqlalchemy import and_
from pydantic import BaseModel, EmailStr, Field, field_validator

from ai_agent.core import db
from ai_agent.core.config import settings
from ai_agent.core.deps import get_effective_permissions
from ai_agent.core.permissions import WILDCARD
from ai_agent.core.token_validation import decode_token_payload, security
from ai_agent.models import Organization, OrganizationMembership, Role, User, UserSettings
from ai_agent.core.enums import ActionType, EntityType
from ai_agent.services.activity_log import log_activity
from ai_agent.services import pluggy_service
from ai_agent.services.identity import (
    create_identity_user,
    authenticate_user,
    refresh_access_token,
    send_reset_email,
    reset_user_password,
    reset_password_by_token,
    verify_email_by_token,
    update_identity_user,
    exchange_authorization_code,
    get_user_profile
)

ADMIN_ROLE = "ADMIN"

router = APIRouter(tags=["Users"])


class UserCreate(BaseModel):
    email: EmailStr = Field(..., description="User's email address (must be unique)")
    password: str = Field(..., min_length=2, description="User's password (min 2 chars)")
    first_name: str = Field(..., description="User's first name")
    last_name: Optional[str] = Field(None, description="User's last name")
    org_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Organization name. A new organization is created and linked to this user.",
        alias="name",
    )

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()



class LoginRequest(BaseModel):
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., min_length=2, description="User's password")


class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(..., description="Refresh token issued at login")


class LoginResponse(BaseModel):
    access_token: str
    expires_in: int
    refresh_expires_in: int
    refresh_token: str
    token_type: str
    session_state: str
    scope: str


class RegisterResponse(BaseModel):
    message: str
    organization_id: int = Field(..., description="ID of the organization created and linked to the new user.")
    organization_name: str = Field(..., description="Name of the organization created and linked to the new user.")


class UserInfoResponse(BaseModel):
    id: int = Field(..., description="Numeric user ID from local DB")
    email: str = Field(..., description="User email from Keycloak")
    first_name: str | None = Field(default=None, description="User's first name")
    last_name: str | None = Field(default=None, description="User's last name")
    phone: str | None = Field(default=None, description="User's phone number")
    address: str | None = Field(default=None, description="User's address")
    organization_id: int | None = Field(default=None, description="ID of the user's active organization, if any")
    organization_name: str | None = Field(default=None, description="Name of the user's active organization, if any")
    role: str | None = Field(default=None, description="Role name held by the user, e.g. 'ADMIN'")
    permissions: list[str] = Field(
        default_factory=list,
        description="Caller's effective permissions in this org (['*'] for owner/superuser). "
                    "Frontend uses this to decide which admin controls to render.",
    )


class UserProfileUpdate(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, description="User's first name")
    last_name: Optional[str] = Field(None, description="User's last name")
    phone: Optional[str] = Field(None, description="User's phone number")
    address: Optional[str] = Field(None, description="User's address")


class UserProfileResponse(BaseModel):
    id: Optional[int] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None


class ForgotPasswordRequest(BaseModel):
    email: EmailStr = Field(..., description="User's registered email address")


class ConfirmResetPasswordRequest(BaseModel):
    token: str = Field(..., description="Keycloak action token from the reset email")
    new_password: str = Field(..., min_length=6, description="New password (min 6 chars)")
    confirm_password: str = Field(..., min_length=6, description="Confirm new password")

    @field_validator("confirm_password", mode="after")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        if v != info.data.get("new_password"):
            raise ValueError("Passwords do not match")
        return v


class ResetPasswordRequest(BaseModel):
    old_password: str = Field(..., min_length=2, description="Current password")
    new_password: str = Field(..., min_length=2, description="New password")
    confirm_password: str = Field(..., min_length=2, description="Confirm new password")

    @field_validator("confirm_password", mode="after")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        if v != info.data.get("new_password"):
            raise ValueError("Passwords do not match")
        return v

class SSOCallbackRequest(BaseModel):
    code: str = Field(...,
        description="Authorization code Keycloak appended to the frontend's "
        "redirect_uri (?code=...) after a successful SSO sign-in.",
    )
                        
class SSOCallbackResponse(BaseModel):
    access_token: str
    expires_in: int
    refresh_expires_in: int
    refresh_token: str
    token_type: str
    session_state: str
    scope: str


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(
    user: UserCreate,
    db_session: Session = Depends(db.get_session)
):
    """
    Registers a new user and sends an email verification link.
    The user must verify their email before they can log in.

    Returns 409 if email is already registered.
    Returns 503 if the identity provider is unreachable.
    """
    email = user.email  # already lowercased by validator

    # --- 1. Local DB check ---
    existing_user = db_session.exec(
        select(User).where(User.email == email)
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered."
        )

    # --- 2. Register with identity provider (sends verification email) ---
    await create_identity_user(
        email=email,
        password=user.password,
        first_name=user.first_name,
        last_name=user.last_name,
    )

    # --- 3. Create new organization ---
    org = Organization(name=user.org_name)
    db_session.add(org)
    db_session.flush()

    # --- 4. Persist user locally, active org = the one just created ---
    db_user = User(
        email=email,
        first_name=user.first_name,
        last_name=user.last_name,
        active_organization_id=org.id,
    )
    db_session.add(db_user)
    db_session.flush()

    # --- 5. Create the system Admin role for this organization ---
    # is_system=True + wildcard permissions: can't be renamed/deleted (see roles.py),
    # and always has every permission, including ones added to the vocabulary later.
    role = Role(organization_id=org.id, name=ADMIN_ROLE, is_system=True, permissions=[WILDCARD])
    db_session.add(role)
    db_session.flush()

    # --- 6. Membership: new user is the owner/Admin of this organization ---
    db_session.add(OrganizationMembership(
        user_id=db_user.id,
        organization_id=org.id,
        role_id=role.id,
        is_owner=True,
    ))
    db_session.commit()
    db_session.refresh(db_user)
    db_session.refresh(org)

    # --- 7. Create Pluggy tenant for this organization ---
    tenant_id = await pluggy_service.create_tenant(org.name, org.id)
    if tenant_id:
        org.pluggy_tenant_id = tenant_id
        db_session.add(org)
        db_session.commit()
        db_session.refresh(org)

    # --- 8. Default UserSettings (idempotent) ---
    if not db_session.exec(
        select(UserSettings).where(UserSettings.user_id == db_user.id)
    ).first():
        db_session.add(UserSettings(
            user_id=db_user.id,
            theme="light",
            language="en",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ))
        db_session.commit()

    return RegisterResponse(
        message="Registration successful. Please check your email to verify your account before logging in.",
        organization_id=org.id,
        organization_name=org.name,
    )


@router.post("/login", response_model=LoginResponse)
async def login_user(
    credentials: LoginRequest,
):
    """
    Returns 401 if credentials are invalid.
    Returns 503 if the identity provider is unreachable.
    """
    email = credentials.email.strip().lower()
    tokens = await authenticate_user(
        email=email,
        password=credentials.password,
    )
    return LoginResponse(**tokens)


@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(body: RefreshTokenRequest):
    """
    Exchanges a refresh token for a new access/refresh token pair.
    Returns 401 if the refresh token is invalid or expired.
    """
    tokens = await refresh_access_token(body.refresh_token)
    return LoginResponse(**tokens)


@router.get("/user", response_model=UserInfoResponse, dependencies=[Depends(security)])
async def get_user(
    payload: dict = Depends(decode_token_payload),
    db_session: Session = Depends(db.get_session),
):
    """
    Get current user info. Returns name, email, and numeric DB user id.
    Requires Bearer token in Authorization header.
    """
    email = payload.get("email", "").strip().lower()
    db_user = db_session.exec(select(User).where(User.email == email)).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found in local database.",
        )
    # A user can legitimately have no active organization (e.g. just left their only one) —
    # that's a normal state, not an error, so this doesn't 404.
    org, user_role, membership = None, None, None
    if db_user.active_organization_id is not None:
        row = db_session.exec(
                select(Organization, Role, OrganizationMembership)
                .select_from(User)
                .join(Organization, Organization.id == User.active_organization_id)
                .join(
                    OrganizationMembership,
                    and_(
                        OrganizationMembership.user_id == User.id,
                        OrganizationMembership.organization_id == User.active_organization_id,
                        OrganizationMembership.is_active == True,  # noqa: E712
                    ),
                    isouter=True,
                )
                .join(Role, Role.id == OrganizationMembership.role_id, isouter=True)
                .where(User.id == db_user.id)
            ).first()
        org, user_role, membership = row if row else (None, None, None)

    permissions = get_effective_permissions(membership, db_user, db_session) if membership else []

    return UserInfoResponse(
        id=db_user.id,
        email=email,
        first_name=db_user.first_name,
        last_name=db_user.last_name,
        phone=db_user.phone,
        address=db_user.address,
        organization_id=org.id if org else None,
        organization_name=org.name if org else None,
        role=user_role.name if user_role else None,
        permissions=permissions,
    )


@router.put("/user/{user_id}/profile", response_model=UserProfileResponse, response_model_exclude_unset=True, dependencies=[Depends(security)])
async def update_profile(
    payload: UserProfileUpdate,
    user_id: int = Path(..., description="Numeric user ID from the local DB"),
    token_payload: dict = Depends(decode_token_payload),
    db_session: Session = Depends(db.get_session),
):
    """
    Update a user's profile (first_name, last_name, phone, address) by user ID.

    The user is looked up by their numeric DB id. The authenticated caller may only
    update their own profile. Name fields are synced to the identity provider; all
    fields are persisted in the local database.

    Returns 403 if the caller is not the owner of the profile.
    Returns 404 if the user is not found in the local database.
    """
    db_user = db_session.get(User, user_id)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found in local database.",
        )

    # Ownership guard: callers may only update their own profile.
    token_email = token_payload.get("email", "").strip().lower()
    if db_user.email.lower() != token_email:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update your own profile.",
        )

    data = payload.model_dump(exclude_unset=True)

    # Sync name fields to the identity provider so the JWT name claim stays correct.
    if "first_name" in data or "last_name" in data:
        await update_identity_user(
            email=db_user.email,
            first_name=data.get("first_name"),
            last_name=data.get("last_name"),
        )

    # Persist all provided fields locally.
    for field, value in data.items():
        setattr(db_user, field, value)
    db_session.add(db_user)

    # A profile update is one user action, even when several fields changed.
    if data:
        log_activity(
            db_session=db_session,
            user_id=user_id,
            action=ActionType.UPDATE,
            entity_type=EntityType.USER,
            entity_id=str(user_id),
            metadata={"fields": sorted(data.keys())},
        )

    db_session.commit()
    db_session.refresh(db_user)

    # Return only the fields that were updated (plus the id).
    return UserProfileResponse(
        id=db_user.id,
        **{field: getattr(db_user, field) for field in data},
    )


@router.post("/forgot-password", status_code=status.HTTP_200_OK)
async def forgot_password(
    payload: ForgotPasswordRequest,
):
    """
    Initiate a password reset.

    Sends an email to the user with a link to update their password.
    The identity provider handles the new-password / confirm-password UI.

    Returns 404 if the email is not registered in the identity provider.
    Returns 503 if the identity provider is unreachable.
    """
    email = payload.email.strip().lower()
    await send_reset_email(email)
    return {"message": "Password reset email sent."}


@router.post("/confirm-reset-password", status_code=status.HTTP_200_OK)
async def confirm_reset_password(payload: ConfirmResetPasswordRequest):
    """
    Complete a password reset using the token from the reset email.

    The token is the Keycloak action token embedded in the reset link.
    Decodes the JWT to identify the user and updates the password via the Admin API.

    Returns 400 if the token is invalid or expired.
    Returns 503 if the identity provider is unreachable.
    """
    success = await reset_password_by_token(payload.token, payload.new_password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset link. Please request a new password reset.",
        )
    return {"message": "Password reset successfully."}


@router.put("/reset-password", status_code=status.HTTP_200_OK, dependencies=[Depends(security)])
async def reset_password(
    payload: ResetPasswordRequest,
    token_payload: dict = Depends(decode_token_payload),
):
    """
    Reset password for the authenticated user.

    Requires a valid Bearer token. The user must provide their current password
    to set a new one. The new_password and confirm_password must match.

    Returns 400 if new_password and confirm_password do not match.
    Returns 401 if the old_password is incorrect.
    Returns 404 if the user is not found in the identity provider.
    Returns 503 if the identity provider is unreachable.
    """
    email = token_payload.get("email", "").strip().lower()
    await reset_user_password(
        email=email,
        old_password=payload.old_password,
        new_password=payload.new_password,
    )
    return {"message": "Password updated successfully."}


@router.get("/verify-email", status_code=status.HTTP_302_FOUND)
async def verify_email(token: str = Query(..., description="Email verification token")):
    """
    Verify a user's email address using the token from the verification email.

    Looks up the user by their emailVerificationToken attribute, sets emailVerified=True
    via the Keycloak Admin API, and redirects to the frontend email-verified page.
    """
    verified = await verify_email_by_token(token)
    if not verified:
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/email-verified?error=invalid_token",
            status_code=302,
        )
    return RedirectResponse(
        url=f"{settings.FRONTEND_URL}/email-verified",
        status_code=302,
    )

@router.post("/auth/callback", response_model=SSOCallbackResponse)
async def sso_callback(
    payload: SSOCallbackRequest,
    db_session: Session = Depends(db.get_session),
):
    """
    Exchange an SSO authorization code for tokens.

    Generic callback for any identity provider brokered through Keycloak
    (Google, Outlook, Facebook, etc.). Keycloak redirects to the frontend
    with a code, which the frontend sends here for token exchange. The
    endpoint then fetches the provider profile and creates or matches the
    local User by email.

    Returns 400 if no email is provided, 401 if the code is invalid or expired,
    and 503 if the identity provider is unreachable.
    """
    tokens = await exchange_authorization_code(payload.code)
    profile = await get_user_profile(tokens["access_token"])
    email = profile.get("email", "").strip().lower()

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SSO provider did not return an email address for this account.",
        )

    db_user = db_session.exec(
        select(User).where(User.email == email)
    ).first()

    if not db_user:
        db_user = User(
            email=email,
            first_name=profile.get("given_name") or "",
            last_name=profile.get("family_name"),
        )
        db_session.add(db_user)
        db_session.commit()
        db_session.refresh(db_user)

    if not db_session.exec(
        select(UserSettings).where(UserSettings.user_id == db_user.id)
    ).first():
        db_session.add(
            UserSettings(
                user_id=db_user.id,
                theme="light",
                language="en",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        )
        db_session.commit()

    return SSOCallbackResponse(**tokens)