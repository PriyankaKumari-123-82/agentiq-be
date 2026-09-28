"""Registry of permission keys that can be granted to a Role.

Role *names* are fully admin-defined (see api/v1/roles.py) — this module only defines what a
permission *means* and which endpoints check for it. v1 ships with organization-administration
permissions only; product-feature permissions (e.g. document_upload, chat) can be appended here
later without any schema or endpoint-shape change.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PermissionDef:
    key: str
    label: str
    description: str
    category: str


PERMISSIONS: list[PermissionDef] = [
    PermissionDef(
        "manage_members",
        "Manage Members",
        "Invite, remove, and change roles of members",
        "Organization",
    ),
    PermissionDef(
        "manage_roles",
        "Manage Roles",
        "Create, edit, and delete roles",
        "Organization",
    ),
    PermissionDef(
        "manage_organization",
        "Manage Organization",
        "Rename org, edit org settings",
        "Organization",
    ),
]

VALID_PERMISSION_KEYS: set[str] = {p.key for p in PERMISSIONS}

WILDCARD = "*"


def has_permission(granted: list[str], permission: str) -> bool:
    """True if `granted` (a role's permissions list) covers `permission`."""
    return WILDCARD in granted or permission in granted
