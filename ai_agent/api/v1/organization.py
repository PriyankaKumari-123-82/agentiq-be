"""Organization endpoints.

GET  /organization              — fetch the authenticated user's active organization (legacy, kept for compatibility)
POST /organization               — create a new organization; caller becomes its owner/Admin (legacy path, now allows multiple orgs)
GET  /organizations              — list every organization the authenticated user belongs to
GET  /organizations/{org_id}     — fetch a specific organization the caller belongs to
POST /organizations/{org_id}/switch — set the caller's active organization
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator, model_validator

from sqlmodel import Session, select, func

from ai_agent.core import db
from ai_agent.core.deps import get_current_user, get_org_membership, require_org_permission
from ai_agent.core.enums import ActionType, EntityType
from ai_agent.core.permissions import WILDCARD
from ai_agent.core.token_validation import decode_token_payload, security
from ai_agent.models import AIAgentSettings, ApiKey, Organization, OrganizationMembership, OrganizationSettings, Role, User, OrganizationInvitation, InvitationStatus, AIAgent, OrganizationDocument, UserSession, SessionStatus
from ai_agent.services.activity_log import log_activity
from ai_agent.services.api_keys import normalize_provider
from ai_agent.services.embedding import EmbeddingService, get_org_embedding_model
from ai_agent.services import pluggy_service
from ai_agent.core.vector_db import get_org_document_store
from fastapi import BackgroundTasks
import logging

logger = logging.getLogger(__name__)

ADMIN_ROLE = "ADMIN"

router = APIRouter(tags=["Organizations"])


# ---------------------------------------------------------------------------
# Pydantic shapes
# ---------------------------------------------------------------------------

class OrganizationOut(BaseModel):
    id: int
    name: str

class OrganizationSettingsOut(BaseModel):
    id: int
    name: str
    embedding_model: Optional[str] = Field(
        default=None,
        description="Embedding model used for RAG indexing/retrieval across this organization's agents.",
    )


class OnboardingStatusResponse(BaseModel):
    organization_created: bool = Field(..., description="Whether the user has created or joined an organization")
    api_key_added: bool = Field(..., description="Whether an API key is added for the active organization")
    agent_created: bool = Field(..., description="Whether an agent is created for the active organization")
    embedding_model_selected: bool = Field(..., description="Whether an embedding model is selected")
    is_completed: bool = Field(..., description="Whether the organization setup is fully complete")


class OrganizationRequest(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name of the organization.",
    )

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, v: str) -> str:
        stripped = v.strip() if isinstance(v, str) else v
        if not stripped:
            raise ValueError("name must not be blank.")
        return stripped


class OrganizationResponse(BaseModel):
    user_id: int
    organization: OrganizationOut


class OrganizationSummary(BaseModel):
    id: int
    name: str
    role: str | None = Field(default=None, description="Caller's role name in this organization")
    is_owner: bool
    is_current: bool = Field(description="True if this is the caller's currently active organization")


class PaginatedOrganizationsOut(BaseModel):
    items: list[OrganizationSummary]
    page: int
    limit: int
    total: int
    has_more: bool


class SwitchOrganizationResponse(BaseModel):
    organization_id: int
    organization_name: str
    role: str | None


class OrganizationAPIKeyInfo(BaseModel):
    id: int
    provider: str
    is_default: bool
    is_active: bool
    last_used: datetime | None = None
    created_at: datetime
    updated_at: datetime


class OrganizationAPIKeyRequest(BaseModel):
    provider: str = Field(..., min_length=1, description="Provider name, e.g. openai")
    api_key: str = Field(..., min_length=1, description="API key to encrypt and store")
    is_default: bool = Field(default=False, description="Whether this is the org default provider key")

    @field_validator("provider", mode="before")
    @classmethod
    def strip_provider(cls, v: str) -> str:
        normalized = normalize_provider(v)
        if not normalized:
            raise ValueError("provider must not be blank.")
        return normalized

    @field_validator("api_key", mode="before")
    @classmethod
    def strip_api_key(cls, v: str) -> str:
        stripped = v.strip() if isinstance(v, str) else v
        if not stripped:
            raise ValueError("api_key must not be blank.")
        return stripped


class OrganizationAPIKeyUpdateRequest(BaseModel):
    api_key: str | None = Field(default=None, description="Replacement API key to encrypt and store")
    is_default: bool | None = Field(default=None, description="Whether this is the org default provider key")

    @field_validator("api_key", mode="before")
    @classmethod
    def strip_api_key(cls, v: str | None) -> str | None:
        if v is None:
            return None
        stripped = v.strip() if isinstance(v, str) else v
        if not stripped:
            raise ValueError("api_key must not be blank.")
        return stripped

    @model_validator(mode="after")
    def require_update_value(self) -> "OrganizationAPIKeyUpdateRequest":
        if self.api_key is None and self.is_default is None:
            raise ValueError("Provide api_key or is_default.")
        return self


class OrganizationAPIKeyResponse(BaseModel):
    success: bool
    id: int
    provider: str
    is_default: bool
    message: str


async def _create_organization_for_user(
    user: User,
    org_name: str,
    db_session: Session,
    make_active: bool,
) -> Organization:
    """Create a new organization, make `user` its owner/Admin, optionally switch them into it.

    The embedding model is NOT set here — it's configured separately (via
    POST /organizations/{org_id}/embedding-model) so the caller can prompt
    for it after creation instead of bundling it into org creation.
    """
    org = Organization(name=org_name)
    db_session.add(org)
    db_session.flush()

    role = Role(organization_id=org.id, name=ADMIN_ROLE, is_system=True, permissions=[WILDCARD])
    db_session.add(role)
    db_session.flush()

    db_session.add(OrganizationMembership(
        user_id=user.id,
        organization_id=org.id,
        role_id=role.id,
        is_owner=True,
    ))

    if make_active:
        user.active_organization_id = org.id
        db_session.add(user)

    db_session.commit()
    db_session.refresh(org)

    # Call Pluggy to create the tenant
    tenant_id = await pluggy_service.create_tenant(org.name, org.id)
    if tenant_id:
        org.pluggy_tenant_id = tenant_id
        db_session.add(org)
        db_session.commit()
        db_session.refresh(org)

    return org


# ---------------------------------------------------------------------------
# GET /organization  — fetch caller's active org (legacy path, kept for compatibility)
# ---------------------------------------------------------------------------

@router.get(
    "/organization",
    response_model=OrganizationOut,
    summary="Get the authenticated user's active organization",
    dependencies=[Depends(security)],
)
async def get_organization(
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    """
    Returns the organization the authenticated user is currently active in.
    Returns 404 if the user has no active organization.
    """
    if user.active_organization_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User does not belong to any organization.",
        )

    org = db_session.get(Organization, user.active_organization_id)
    if org is None or not org.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")

    return OrganizationOut(id=org.id, name=org.name)


@router.get("/onboarding-status", response_model=OnboardingStatusResponse, dependencies=[Depends(security)])
async def get_setup_status(
    token_payload: dict = Depends(decode_token_payload),
    db_session: Session = Depends(db.get_session),
):
    email = token_payload.get("email", "").strip().lower()
    db_user = db_session.exec(select(User).where(User.email == email)).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    organization_created = bool(db_user.active_organization_id)
    api_key_added = False
    agent_created = False
    embedding_model_selected = False

    if organization_created:
        api_key_added = db_session.exec(
            select(ApiKey).where(ApiKey.organization_id == db_user.active_organization_id)
        ).first() is not None

        agent = db_session.exec(
            select(AIAgent).where(AIAgent.organization_id == db_user.active_organization_id)
        ).first()
        if agent:
            agent_created = True

        org_settings = db_session.exec(
            select(OrganizationSettings).where(OrganizationSettings.organization_id == db_user.active_organization_id)
        ).first()
        if org_settings and org_settings.embedding_model:
            embedding_model_selected = True

    is_completed = organization_created and api_key_added and agent_created and embedding_model_selected

    return OnboardingStatusResponse(
        organization_created=organization_created,
        api_key_added=api_key_added,
        agent_created=agent_created,
        embedding_model_selected=embedding_model_selected,
        is_completed=is_completed,
    )


# ---------------------------------------------------------------------------
# POST /organization  — create an additional organization
# ---------------------------------------------------------------------------

@router.post(
    "/organization",
    response_model=OrganizationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new organization; caller becomes its owner/Admin",
    dependencies=[Depends(security)],
)
async def create_organization(
    body: OrganizationRequest,
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    """
    Create a new organization and make the authenticated user its owner/Admin.
    A user can belong to (and create) more than one organization; the new
    organization becomes the caller's active organization immediately.
    """
    org = await _create_organization_for_user(user, body.name, db_session, make_active=True)

    return OrganizationResponse(
        user_id=user.id,
        organization=OrganizationOut(id=org.id, name=org.name),
    )


# ---------------------------------------------------------------------------
# GET /organizations  — list every org the caller belongs to
# ---------------------------------------------------------------------------

@router.get(
    "/organizations",
    response_model=PaginatedOrganizationsOut,
    summary="List every organization the authenticated user belongs to",
    dependencies=[Depends(security)],
)
async def list_organizations(
    search: str | None = Query(None, description="Search organizations by name"),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
) -> PaginatedOrganizationsOut:
    filters = [
        OrganizationMembership.user_id == user.id,
        OrganizationMembership.is_active == True,  # noqa: E712
        Organization.is_active == True,  # noqa: E712
    ]

    search_term = search.strip() if search else None
    if search_term:
        filters.append(Organization.name.ilike(f"%{search_term}%"))

    offset = (page - 1) * limit
    rows = db_session.exec(
        select(
            OrganizationMembership,
            Organization,
            Role,
            func.count().over().label("total_count"),
        )
        .join(Organization, Organization.id == OrganizationMembership.organization_id)
        .join(Role, Role.id == OrganizationMembership.role_id, isouter=True)
        .where(*filters)
        .order_by(func.lower(Organization.name), Organization.id)
        .offset(offset)
        .limit(limit)
    ).all()

    total = rows[0].total_count if rows else 0
    items = [
        OrganizationSummary(
            id=org.id,
            name=org.name,
            role=role.name if role else None,
            is_owner=membership.is_owner,
            is_current=(org.id == user.active_organization_id),
        )
        for membership, org, role, _total_count in rows
    ]
    return PaginatedOrganizationsOut(
        items=items,
        page=page,
        limit=limit,
        total=total,
        has_more=(offset + len(items)) < total,
    )


# ---------------------------------------------------------------------------
# GET /organizations/{org_id}  — fetch a specific org the caller belongs to
# ---------------------------------------------------------------------------

@router.get(
    "/organizations/{org_id}",
    response_model=OrganizationOut,
    summary="Get an organization the caller belongs to",
    dependencies=[Depends(security)],
)
async def get_organization_by_id(
    org_id: int,
    membership: OrganizationMembership = Depends(get_org_membership),
    db_session: Session = Depends(db.get_session),
):
    """Returns 403 if the caller is not a member of `org_id`, 404 if it doesn't exist."""
    org = db_session.get(Organization, org_id)
    if org is None or not org.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")
    return OrganizationOut(id=org.id, name=org.name)


# ---------------------------------------------------------------------------
# PATCH /organizations/{org_id}  — rename an organization
# ---------------------------------------------------------------------------

@router.patch(
    "/organizations/{org_id}",
    response_model=OrganizationOut,
    summary="Rename an organization",
    dependencies=[Depends(security)],
)
async def rename_organization(
    body: OrganizationRequest,
    org_id: int,
    background_tasks: BackgroundTasks,
    _membership: OrganizationMembership = Depends(require_org_permission("manage_organization")),
    db_session: Session = Depends(db.get_session),
):
    """Returns 403 if the caller lacks `manage_organization`, 404 if the org doesn't exist."""
    org = db_session.get(Organization, org_id)
    if org is None or not org.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")

    org.name = body.name
    org.updated_at = datetime.now(timezone.utc)
    db_session.add(org)
    db_session.commit()
    db_session.refresh(org)
    
    if org.pluggy_tenant_id:
        background_tasks.add_task(pluggy_service.update_tenant, org.pluggy_tenant_id, org.name)

    return OrganizationOut(id=org.id, name=org.name)


# ---------------------------------------------------------------------------
# GET/POST /organizations/{org_id}/settings  — generic org-level settings
# (currently just embedding_model; future org settings extend this same resource
# instead of getting their own dedicated endpoint)
# ---------------------------------------------------------------------------

class OrganizationSettingsRequest(BaseModel):
    embedding_model: str = Field(
        ...,
        min_length=1,
        description="Embedding model to use for RAG indexing/retrieval across every agent in this organization.",
    )


@router.get(
    "/organizations/{org_id}/settings",
    response_model=OrganizationSettingsOut,
    summary="Get the organization's settings",
    dependencies=[Depends(security)],
)
async def get_organization_settings(
    org_id: int,
    membership: OrganizationMembership = Depends(get_org_membership),
    db_session: Session = Depends(db.get_session),
):
    """Returns `embedding_model: null` if the organization hasn't configured one yet."""
    org = db_session.get(Organization, org_id)
    if org is None or not org.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")
    return OrganizationSettingsOut(id=org.id, name=org.name, embedding_model=get_org_embedding_model(org_id, db_session))


@router.post(
    "/organizations/{org_id}/settings",
    response_model=OrganizationSettingsOut,
    summary="Set the organization's embedding model (one-time; cannot be changed afterward)",
    dependencies=[Depends(security)],
)
async def set_organization_settings(
    org_id: int,
    body: OrganizationSettingsRequest,
    _membership: OrganizationMembership = Depends(require_org_permission("manage_organization")),
    db_session: Session = Depends(db.get_session),
):
    """Returns 409 if the organization already has an embedding model configured —
    it cannot be changed once agents may have indexed documents against it."""
    org = db_session.get(Organization, org_id)
    if org is None or not org.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")

    if get_org_embedding_model(org_id, db_session) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This organization already has an embedding model configured and it cannot be changed.",
        )

    embedding_model = EmbeddingService().get_model_spec(body.embedding_model, db_session=db_session)
    if embedding_model is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Embedding model was not found in the model catalog: {body.embedding_model}",
        )

    db_session.add(OrganizationSettings(organization_id=org_id, embedding_model=body.embedding_model))
    db_session.commit()

    return OrganizationSettingsOut(id=org.id, name=org.name, embedding_model=body.embedding_model)


# ---------------------------------------------------------------------------
# POST /organizations/{org_id}/switch  — set caller's active organization
# ---------------------------------------------------------------------------

@router.post(
    "/organizations/{org_id}/switch",
    response_model=SwitchOrganizationResponse,
    summary="Switch the caller's active organization",
    dependencies=[Depends(security)],
)
async def switch_organization(
    org_id: int,
    user: User = Depends(get_current_user),
    membership: OrganizationMembership = Depends(get_org_membership),
    db_session: Session = Depends(db.get_session),
):
    """Returns 403 if the caller is not a member of `org_id`."""
    org = db_session.get(Organization, org_id)
    if org is None or not org.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")

    user.active_organization_id = org_id
    db_session.add(user)
    db_session.commit()

    role = db_session.get(Role, membership.role_id)
    return SwitchOrganizationResponse(
        organization_id=org.id,
        organization_name=org.name,
        role=role.name if role else None,
    )


class OrganizationAPIKeyInfo(BaseModel):
    id: int
    provider: str
    name: str = ""
    masked_key: str = ""
    is_default: bool
    is_active: bool
    last_used: datetime | None = None
    created_at: datetime
    updated_at: datetime


class OrganizationAPIKeyRequest(BaseModel):
    provider: str = Field(..., min_length=1, description="Provider name, e.g. openai")
    name: str = Field(..., min_length=1, description="Label/name for the API key")
    api_key: str = Field(..., min_length=1, description="API key to encrypt and store")
    is_default: bool = Field(default=False, description="Whether this is the org default key for this provider")

    @field_validator("provider", mode="before")
    @classmethod
    def strip_provider(cls, v: str) -> str:
        normalized = normalize_provider(v)
        if not normalized:
            raise ValueError("provider must not be blank.")
        return normalized

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, v: str) -> str:
        stripped = v.strip() if isinstance(v, str) else v
        if not stripped:
            raise ValueError("name must not be blank.")
        return stripped

    @field_validator("api_key", mode="before")
    @classmethod
    def strip_api_key(cls, v: str) -> str:
        stripped = v.strip() if isinstance(v, str) else v
        if not stripped:
            raise ValueError("api_key must not be blank.")
        return stripped


class OrganizationAPIKeyUpdateRequest(BaseModel):
    name: str | None = Field(default=None, description="Updated name or label for the API key")
    api_key: str | None = Field(default=None, description="Replacement API key to encrypt and store")
    is_default: bool | None = Field(default=None, description="Whether this is the org default key for this provider")

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, v: str | None) -> str | None:
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v

    @field_validator("api_key", mode="before")
    @classmethod
    def strip_api_key(cls, v: str | None) -> str | None:
        if v is None:
            return None
        stripped = v.strip() if isinstance(v, str) else v
        if not stripped:
            raise ValueError("api_key must not be blank.")
        return stripped

    @model_validator(mode="after")
    def require_update_value(self) -> "OrganizationAPIKeyUpdateRequest":
        if self.name is None and self.api_key is None and self.is_default is None:
            raise ValueError("Provide at least one of name, api_key, or is_default.")
        return self


class OrganizationAPIKeyResponse(BaseModel):
    success: bool
    id: int
    provider: str
    name: str
    masked_key: str
    is_default: bool
    message: str


@router.get(
    "/organizations/{org_id}/api-keys",
    response_model=list[OrganizationAPIKeyInfo],
    summary="List organization-level provider API keys",
    dependencies=[Depends(security)],
)
async def list_organization_api_keys(
    org_id: int,
    provider: str | None = Query(default=None, description="Filter active keys by provider"),
    _membership: OrganizationMembership = Depends(require_org_permission("manage_organization")),
    db_session: Session = Depends(db.get_session),
):
    query = select(ApiKey).where(
        ApiKey.organization_id == org_id,
        ApiKey.is_active == True,  # noqa: E712
    )
    if provider:
        query = query.where(ApiKey.provider == normalize_provider(provider))

    api_keys = db_session.exec(query.order_by(ApiKey.provider, ApiKey.created_at.desc())).all()
    return [
        OrganizationAPIKeyInfo(
            id=key.id,
            provider=key.provider,
            name=key.name or f"{key.provider.capitalize()} Key",
            masked_key=key.masked_key,
            is_default=key.is_default,
            is_active=key.is_active,
            last_used=key.last_used,
            created_at=key.created_at,
            updated_at=key.updated_at,
        )
        for key in api_keys
    ]


@router.post(
    "/organizations/{org_id}/api-key",
    response_model=OrganizationAPIKeyResponse,
    summary="Create an organization-level provider API key",
    dependencies=[Depends(security)],
)
async def set_organization_api_key(
    org_id: int,
    body: OrganizationAPIKeyRequest,
    user: User = Depends(get_current_user),
    _membership: OrganizationMembership = Depends(require_org_permission("manage_organization")),
    db_session: Session = Depends(db.get_session),
):
    provider = normalize_provider(body.provider)

    is_default = body.is_default

    if is_default:
        current_default_key = db_session.exec(
            select(ApiKey).where(
                ApiKey.organization_id == org_id,
                ApiKey.provider == provider,
                ApiKey.is_active == True,  # noqa: E712
                ApiKey.is_default == True,  # noqa: E712
            )
        ).first()

        if current_default_key is not None:
            current_default_key.is_default = False
            current_default_key.updated_at = datetime.now(timezone.utc)
            db_session.add(current_default_key)
            db_session.flush()

    api_key = ApiKey(
        organization_id=org_id,
        provider=provider,
        name=body.name,
        is_default=is_default,
        is_active=True,
    )
    api_key.encrypt(body.api_key)
    db_session.add(api_key)
    action = ActionType.CREATE
    message = f"Added API key '{body.name}' for {provider}"

    db_session.flush()
    log_activity(
        db_session=db_session,
        user_id=user.id,
        action=action,
        entity_type=EntityType.API_KEY,
        entity_id=str(api_key.id),
        metadata={"provider": provider, "organization_id": org_id, "name": body.name, "is_default": api_key.is_default},
    )
    db_session.commit()
    db_session.refresh(api_key)
    return OrganizationAPIKeyResponse(
        success=True,
        id=api_key.id,
        provider=api_key.provider,
        name=api_key.name,
        masked_key=api_key.masked_key,
        is_default=api_key.is_default,
        message=message,
    )


@router.put(
    "/organizations/{org_id}/api-keys/{api_key_id}",
    response_model=OrganizationAPIKeyResponse,
    summary="Update an organization-level API key",
    dependencies=[Depends(security)],
)
async def update_organization_api_key(
    org_id: int,
    api_key_id: int,
    body: OrganizationAPIKeyUpdateRequest,
    user: User = Depends(get_current_user),
    _membership: OrganizationMembership = Depends(require_org_permission("manage_organization")),
    db_session: Session = Depends(db.get_session),
):
    api_key = db_session.get(ApiKey, api_key_id)
    if api_key is None or api_key.organization_id != org_id or not api_key.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found.")

    now = datetime.now(timezone.utc)

    # If setting as default, unset previous default for this org + provider
    if body.is_default is True:
        current_default_key = db_session.exec(
            select(ApiKey).where(
                ApiKey.organization_id == org_id,
                ApiKey.provider == api_key.provider,
                ApiKey.is_active == True,  # noqa: E712
                ApiKey.is_default == True,  # noqa: E712
            )
        ).first()
        
        if current_default_key and current_default_key.id != api_key.id:
            current_default_key.is_default = False
            current_default_key.updated_at = now
            db_session.add(current_default_key)
            db_session.flush()

    if body.name is not None:
        api_key.name = body.name
    if body.api_key is not None:
        api_key.encrypt(body.api_key)
    if body.is_default is not None:
        api_key.is_default = body.is_default

    api_key.updated_at = now
    db_session.add(api_key)
    log_activity(
        db_session=db_session,
        user_id=user.id,
        action=ActionType.UPDATE,
        entity_type=EntityType.API_KEY,
        entity_id=str(api_key.id),
        metadata={"provider": api_key.provider, "organization_id": org_id, "name": api_key.name, "is_default": api_key.is_default},
    )
    db_session.commit()
    db_session.refresh(api_key)
    return OrganizationAPIKeyResponse(
        success=True,
        id=api_key.id,
        provider=api_key.provider,
        name=api_key.name,
        masked_key=api_key.masked_key,
        is_default=api_key.is_default,
        message=f"Updated API key '{api_key.name}' for {api_key.provider}",
    )


@router.delete(
    "/organizations/{org_id}/api-keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Deactivate an organization-level provider API key",
    dependencies=[Depends(security)],
)
async def delete_organization_api_key(
    org_id: int,
    key_id: int,
    user: User = Depends(get_current_user),
    _membership: OrganizationMembership = Depends(require_org_permission("manage_organization")),
    db_session: Session = Depends(db.get_session),
):
    api_key = db_session.get(ApiKey, key_id)
    if api_key is None or api_key.organization_id != org_id or not api_key.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found.")

    provider = api_key.provider
    was_default = api_key.is_default

    # Unset agent settings pointing to this deleted key
    for settings_obj in db_session.exec(
        select(AIAgentSettings).where(AIAgentSettings.api_key == key_id)
    ).all():
        settings_obj.api_key = None
        settings_obj.updated_at = datetime.now(timezone.utc)
        db_session.add(settings_obj)

    api_key.is_active = False
    api_key.is_default = False
    api_key.deleted_at = datetime.now(timezone.utc)
    api_key.updated_at = api_key.deleted_at
    db_session.add(api_key)

    # If the deleted key was default, promote another active key for this provider to default
    if was_default:
        remaining_key = db_session.exec(
            select(ApiKey).where(
                ApiKey.organization_id == org_id,
                ApiKey.provider == provider,
                ApiKey.is_active == True,  # noqa: E712
            ).order_by(ApiKey.updated_at.desc())
        ).first()
        if remaining_key:
            remaining_key.is_default = True
            remaining_key.updated_at = datetime.now(timezone.utc)
            db_session.add(remaining_key)

    log_activity(
        db_session=db_session,
        user_id=user.id,
        action=ActionType.DELETE,
        entity_type=EntityType.API_KEY,
        entity_id=str(key_id),
        metadata={"provider": provider, "organization_id": org_id, "name": api_key.name},
    )
    db_session.commit()
    return None

# ---------------------------------------------------------------------------
# DELETE /organizations/{org_id}  — delete an organization
# ---------------------------------------------------------------------------

@router.delete(
    "/organizations/{org_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an organization",
    dependencies=[Depends(security)],
)
async def delete_organization(
    org_id: int,
    membership: OrganizationMembership = Depends(require_org_permission("manage_organization")),
    db_session: Session = Depends(db.get_session),
):
    """
    Deletes an organization and drops its Milvus collection.
    Requires the caller to be the owner of the organization.
    """
    if not membership.is_owner:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the organization owner can delete the organization."
        )

    org = db_session.get(Organization, org_id)
    if org is None or not org.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")

    # 1. Cleanup related DB data
    # Delete invitations
    invitations = db_session.exec(
        select(OrganizationInvitation).where(OrganizationInvitation.organization_id == org_id)
    ).all()
    for inv in invitations:
        inv.status = InvitationStatus.CANCELLED
        inv.cancelled_at = datetime.now(timezone.utc)
        db_session.add(inv)

    # Soft-delete agents and their chat data
    agents = db_session.exec(
        select(AIAgent).where(AIAgent.organization_id == org_id)
    ).all()
    for agent in agents:
        agent.deleted = True
        agent.is_active = False
        db_session.add(agent)
        
        # Soft-delete agent's chat sessions
        sessions = db_session.exec(
            select(UserSession).where(UserSession.agent_id == agent.id)
        ).all()
        for session in sessions:
            session.is_active = False
            session.status = SessionStatus.DELETED
            db_session.add(session)

    # Soft delete memberships
    memberships = db_session.exec(
        select(OrganizationMembership).where(OrganizationMembership.organization_id == org_id)
    ).all()
    for m in memberships:
        m.is_active = False
        m.removed_at = datetime.now(timezone.utc)
        db_session.add(m)

    # Soft delete roles
    roles = db_session.exec(
        select(Role).where(Role.organization_id == org_id)
    ).all()
    for role in roles:
        role.is_active = False
        db_session.add(role)

    # Soft delete org document links
    org_docs = db_session.exec(
        select(OrganizationDocument).where(OrganizationDocument.organization_id == org_id)
    ).all()
    for doc in org_docs:
        doc.is_active = False
        db_session.add(doc)

    # Reset active_organization_id for users currently using it
    active_users = db_session.exec(
        select(User).where(User.active_organization_id == org_id)
    ).all()
    for u in active_users:
        u.active_organization_id = None
        db_session.add(u)

    # 2. Soft-delete the organization & commit transaction
    org.is_active = False
    db_session.add(org)
    db_session.commit()

    # 3. External cleanup (Milvus) - best effort, non-blocking
    try:
        store = get_org_document_store()
        store.drop_collection(org_id)
    except Exception as e:
        logger.error(f"Failed to drop Milvus collection for org {org_id}: {e}")
