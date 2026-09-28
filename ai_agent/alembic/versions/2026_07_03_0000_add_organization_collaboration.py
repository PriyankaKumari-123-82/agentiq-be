"""Add organization membership/invitation tables, role permissions, multi-org support.

Replaces the single `users.organization_id` FK (one org per user) with an
`organization_memberships` table (many orgs per user, one role per membership), adds
`organization_invitations` for email-based invites, and adds `roles.permissions` so
authorization checks can be permission-based instead of matching on role name.

Revision ID: b1c2d3e4f5a6
Revises: 9b3c1d5e7f8a
Create Date: 2026-07-03 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "b1c2d3e4f5a6"
down_revision = "9b3c1d5e7f8a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- 1. roles.permissions ---
    op.add_column(
        "roles",
        sa.Column("permissions", sa.JSON(), nullable=False, server_default="[]"),
    )
    # Existing bootstrapped "ADMIN" roles were created with is_system=False (a bug — an
    # admin could rename/delete their own only admin role). Fix retroactively: they become
    # the wildcard-permission system role.
    op.execute("""
        UPDATE roles
        SET is_system = true, permissions = '["*"]'::json
        WHERE name = 'ADMIN';
    """)

    # --- 2. organization_memberships ---
    op.create_table(
        "organization_memberships",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("role_id", sa.Integer(), nullable=False),
        sa.Column("is_owner", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("removed_at", sa.DateTime(), nullable=True),
        sa.Column("joined_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "organization_id", name="uq_membership_user_org"),
    )
    op.create_index(op.f("ix_organization_memberships_id"), "organization_memberships", ["id"], unique=False)
    op.create_index(op.f("ix_organization_memberships_user_id"), "organization_memberships", ["user_id"], unique=False)
    op.create_index(op.f("ix_organization_memberships_organization_id"), "organization_memberships", ["organization_id"], unique=False)
    op.create_index(op.f("ix_organization_memberships_role_id"), "organization_memberships", ["role_id"], unique=False)
    op.create_index(op.f("ix_organization_memberships_is_active"), "organization_memberships", ["is_active"], unique=False)

    # Backfill: one membership per existing user, owner of their current (sole) org, with
    # whichever role their existing user_role row points to (today that's always ADMIN).
    op.execute("""
        INSERT INTO organization_memberships
            (user_id, organization_id, role_id, is_owner, is_active, joined_at, created_at, updated_at)
        SELECT u.id, u.organization_id, r.id, true, true, u.created_at, NOW(), NOW()
        FROM users u
        JOIN user_role ur ON ur.user_id = u.id
        JOIN roles r ON r.id = ur.role_id AND r.organization_id = u.organization_id
        WHERE u.organization_id IS NOT NULL;
    """)

    # --- 3. organization_invitations ---
    op.create_table(
        "organization_invitations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("role_id", sa.Integer(), nullable=False),
        sa.Column("invited_by_user_id", sa.Integer(), nullable=False),
        sa.Column("token_hash", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("accepted_at", sa.DateTime(), nullable=True),
        sa.Column("accepted_user_id", sa.Integer(), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"]),
        sa.ForeignKeyConstraint(["invited_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["accepted_user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash", name="uq_invitation_token_hash"),
    )
    op.create_index(op.f("ix_organization_invitations_id"), "organization_invitations", ["id"], unique=False)
    op.create_index(op.f("ix_organization_invitations_organization_id"), "organization_invitations", ["organization_id"], unique=False)
    op.create_index(op.f("ix_organization_invitations_email"), "organization_invitations", ["email"], unique=False)
    op.create_index(op.f("ix_organization_invitations_token_hash"), "organization_invitations", ["token_hash"], unique=False)
    op.create_index(op.f("ix_organization_invitations_status"), "organization_invitations", ["status"], unique=False)

    # Only one PENDING invitation per (org, email) — re-inviting after cancel/expiry is fine.
    op.execute("""
        CREATE UNIQUE INDEX uq_invitation_pending_org_email
        ON organization_invitations (organization_id, lower(email))
        WHERE status = 'pending';
    """)

    # --- 4. users.organization_id -> users.active_organization_id ---
    op.drop_constraint("fk_users_organization_id_organizations", "users", type_="foreignkey")
    op.drop_index(op.f("ix_users_organization_id"), table_name="users")
    op.alter_column("users", "organization_id", new_column_name="active_organization_id")
    op.create_index(op.f("ix_users_active_organization_id"), "users", ["active_organization_id"], unique=False)
    op.create_foreign_key(
        "fk_users_active_organization_id_organizations",
        "users",
        "organizations",
        ["active_organization_id"],
        ["id"],
    )

    # --- 5. Drop user_role (superseded by organization_memberships.role_id) ---
    op.drop_table("user_role")


def downgrade() -> None:
    # --- 5. Recreate user_role, backfilled from organization_memberships ---
    op.create_table(
        "user_role",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("role_id", sa.Integer(), nullable=False),
        sa.Column("assigned_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("user_id", "role_id"),
    )
    op.execute("""
        INSERT INTO user_role (user_id, role_id, assigned_at)
        SELECT user_id, role_id, joined_at FROM organization_memberships WHERE is_active = true;
    """)

    # --- 4. users.active_organization_id -> users.organization_id ---
    op.drop_constraint("fk_users_active_organization_id_organizations", "users", type_="foreignkey")
    op.drop_index(op.f("ix_users_active_organization_id"), table_name="users")
    op.alter_column("users", "active_organization_id", new_column_name="organization_id")
    op.create_index(op.f("ix_users_organization_id"), "users", ["organization_id"], unique=False)
    op.create_foreign_key(
        "fk_users_organization_id_organizations",
        "users",
        "organizations",
        ["organization_id"],
        ["id"],
    )

    # --- 3. organization_invitations ---
    op.execute("DROP INDEX IF EXISTS uq_invitation_pending_org_email;")
    op.drop_index(op.f("ix_organization_invitations_status"), table_name="organization_invitations")
    op.drop_index(op.f("ix_organization_invitations_token_hash"), table_name="organization_invitations")
    op.drop_index(op.f("ix_organization_invitations_email"), table_name="organization_invitations")
    op.drop_index(op.f("ix_organization_invitations_organization_id"), table_name="organization_invitations")
    op.drop_index(op.f("ix_organization_invitations_id"), table_name="organization_invitations")
    op.drop_table("organization_invitations")

    # --- 2. organization_memberships ---
    op.drop_index(op.f("ix_organization_memberships_is_active"), table_name="organization_memberships")
    op.drop_index(op.f("ix_organization_memberships_role_id"), table_name="organization_memberships")
    op.drop_index(op.f("ix_organization_memberships_organization_id"), table_name="organization_memberships")
    op.drop_index(op.f("ix_organization_memberships_user_id"), table_name="organization_memberships")
    op.drop_index(op.f("ix_organization_memberships_id"), table_name="organization_memberships")
    op.drop_table("organization_memberships")

    # --- 1. roles.permissions ---
    op.drop_column("roles", "permissions")
