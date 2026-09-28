"""Role management endpoints.

GET    /roles          — list roles for the caller's active organization.
POST   /roles          — create a new custom role scoped to the caller's active organization.
PATCH  /roles/{id}     — update a role's name / description / permissions.
DELETE /roles/{id}     — delete a custom role (system roles are protected).

All endpoints require the caller to hold the `manage_roles` permission in their active
organization (see core/permissions.py and core/deps.py) — never a hardcoded role name, since
role names are fully admin-defined.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, or_
from sqlmodel import Session, select

from ai_agent.core import db
from ai_agent.core.deps import get_current_user, require_permission
from ai_agent.core.permissions import PERMISSIONS, VALID_PERMISSION_KEYS
from ai_agent.core.token_validation import security
from ai_agent.models import InvitationStatus, OrganizationInvitation, OrganizationMembership, Role, User

router = APIRouter(tags=["Roles"])


def _validate_permissions(permissions: list[str]) -> list[str]:
    unknown = sorted(set(permissions) - VALID_PERMISSION_KEYS - {"*"})
    if unknown:
        raise ValueError(f"Unknown permission(s): {', '.join(unknown)}")
    return permissions


class RoleOut(BaseModel):
    id: int
    name: str
    description: str | None
    is_system: bool
    permissions: list[str]
    created_at: datetime
    updated_at: datetime


class PaginatedRolesOut(BaseModel):
    items: list[RoleOut]
    page: int
    limit: int
    total: int
    has_more: bool


class RoleCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Role name (unique per org).")
    description: str | None = Field(default=None, max_length=500)
    permissions: list[str] = Field(default_factory=list, description="See GET /permissions for valid keys.")

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        stripped = v.strip().upper() if isinstance(v, str) else v
        if not stripped:
            raise ValueError("name must not be blank.")
        return stripped

    @field_validator("permissions")
    @classmethod
    def validate_permissions(cls, v: list[str]) -> list[str]:
        return _validate_permissions(v)


class RoleUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    permissions: list[str] | None = Field(default=None, description="See GET /permissions for valid keys.")

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, v: str | None) -> str | None:
        if v is None:
            return v
        stripped = v.strip().upper()
        if not stripped:
            raise ValueError("name must not be blank.")
        return stripped

    @field_validator("permissions")
    @classmethod
    def validate_permissions(cls, v: list[str] | None) -> list[str] | None:
        return v if v is None else _validate_permissions(v)


class PermissionOut(BaseModel):
    key: str
    label: str
    description: str
    category: str


@router.get(
    "/permissions",
    response_model=list[PermissionOut],
    summary="List all permission keys that can be granted to a role",
    dependencies=[Depends(security)],
)
def list_permissions(
    user: User = Depends(get_current_user),
) -> list[PermissionOut]:
    """Any authenticated user can read the catalog — it's used to render the role-editor UI."""
    return [
        PermissionOut(key=p.key, label=p.label, description=p.description, category=p.category)
        for p in PERMISSIONS
    ]


def _to_role_out(role: Role) -> RoleOut:
    return RoleOut(
        id=role.id,
        name=role.name,
        description=role.description,
        is_system=role.is_system,
        permissions=role.permissions or [],
        created_at=role.created_at,
        updated_at=role.updated_at,
    )


@router.get(
    "/roles",
    response_model=PaginatedRolesOut,
    summary="List roles for the caller's active organization",
    dependencies=[Depends(security)],
)
def list_roles(
    search: str | None = Query(None, description="Search roles by name or description"),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    membership: OrganizationMembership = Depends(require_permission("manage_roles")),
    db_session: Session = Depends(db.get_session),
) -> PaginatedRolesOut:
    """Return all roles that belong to the authenticated user's active organization."""
    filters = [
        Role.organization_id == membership.organization_id,
    ]

    search_term = search.strip() if search else None
    if search_term:
        pattern = f"%{search_term}%"
        filters.append(
            or_(
                Role.name.ilike(pattern),
                Role.description.ilike(pattern),
            )
        )

    offset = (page - 1) * limit
    total = db_session.exec(
        select(func.count())
        .select_from(Role)
        .where(*filters)
    ).one()

    roles = db_session.exec(
        select(Role)
        .where(*filters)
        .order_by(Role.created_at.desc())
        .offset(offset)
        .limit(limit)
    ).all()
    items = [_to_role_out(r) for r in roles]
    return PaginatedRolesOut(
        items=items,
        page=page,
        limit=limit,
        total=total,
        has_more=(offset + len(items)) < total,
    )


@router.post(
    "/roles",
    response_model=RoleOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new role for the caller's active organization",
    dependencies=[Depends(security)],
)
def create_role(
    body: RoleCreate,
    membership: OrganizationMembership = Depends(require_permission("manage_roles")),
    db_session: Session = Depends(db.get_session),
) -> RoleOut:
    """
    Create a custom role scoped to the caller's active organization.
    Returns 409 if a role with the same name already exists in the org.
    """
    existing = db_session.exec(
        select(Role).where(
            Role.organization_id == membership.organization_id,
            Role.name == body.name,
        )
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A role named '{body.name}' already exists in this organization.",
        )

    now = datetime.now(timezone.utc)
    role = Role(
        organization_id=membership.organization_id,
        name=body.name,
        description=body.description,
        is_system=False,
        permissions=body.permissions,
        created_at=now,
        updated_at=now,
    )
    db_session.add(role)
    db_session.commit()
    db_session.refresh(role)

    return _to_role_out(role)


@router.patch(
    "/roles/{role_id}",
    response_model=RoleOut,
    summary="Update a role's name, description, or permissions",
    dependencies=[Depends(security)],
)
def update_role(
    body: RoleUpdate,
    role_id: int = Path(..., description="ID of the role to update"),
    membership: OrganizationMembership = Depends(require_permission("manage_roles")),
    db_session: Session = Depends(db.get_session),
) -> RoleOut:
    """
    Update `name` / `description` / `permissions` of a role.
    Returns 403 if the role is a system role.
    Returns 404 if the role does not exist or belongs to a different org.
    Returns 409 if renaming to a name already taken in the org.
    """
    role = db_session.get(Role, role_id)
    if role is None or role.organization_id != membership.organization_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found.")

    if role.is_system:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="System roles cannot be modified.",
        )

    data = body.model_dump(exclude_unset=True)

    if "name" in data and data["name"] != role.name:
        conflict = db_session.exec(
            select(Role).where(
                Role.organization_id == membership.organization_id,
                Role.name == data["name"],
            )
        ).first()
        if conflict:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A role named '{data['name']}' already exists in this organization.",
            )

    for field, value in data.items():
        setattr(role, field, value)
    role.updated_at = datetime.now(timezone.utc)

    db_session.add(role)
    db_session.commit()
    db_session.refresh(role)

    return _to_role_out(role)


@router.delete(
    "/roles/{role_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a custom role",
    dependencies=[Depends(security)],
)
def delete_role(
    role_id: int = Path(..., description="ID of the role to delete"),
    membership: OrganizationMembership = Depends(require_permission("manage_roles")),
    db_session: Session = Depends(db.get_session),
) -> None:
    """
    Delete a role from the organization.
    Returns 400 if any active members, or any pending invitations, still reference this role
    (reassign/cancel them first — a pending invitation holds a role_id that would otherwise
    dangle once the invitee accepts).
    Returns 403 if the role is a system role.
    Returns 404 if the role does not exist or belongs to a different org.
    """
    role = db_session.get(Role, role_id)
    if role is None or role.organization_id != membership.organization_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found.")

    if role.is_system:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="System roles cannot be deleted.",
        )

    active_members = db_session.exec(
        select(OrganizationMembership).where(
            OrganizationMembership.role_id == role_id,
            OrganizationMembership.is_active == True,  # noqa: E712
        )
    ).all()
    if active_members:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{len(active_members)} member(s) are assigned this role. Reassign them before deleting.",
        )

    pending_invites = db_session.exec(
        select(OrganizationInvitation).where(
            OrganizationInvitation.role_id == role_id,
            OrganizationInvitation.status == InvitationStatus.PENDING,
        )
    ).all()
    if pending_invites:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{len(pending_invites)} pending invitation(s) use this role. Cancel or reassign them before deleting.",
        )

    db_session.delete(role)
    db_session.commit()
