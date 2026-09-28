"""Add roles and user_roles tables with backfill.

Revision ID: 8e3f1a2b9c4d
Revises: fcccbfcbce0d
Create Date: 2026-06-29 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "8e3f1a2b9c4d"
down_revision = "fcccbfcbce0d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- DDL ---
    op.create_table(
        "roles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("is_system", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "name", name="uq_role_org_name"),
    )
    op.create_index(op.f("ix_roles_id"), "roles", ["id"], unique=False)
    op.create_index(op.f("ix_roles_organization_id"), "roles", ["organization_id"], unique=False)

    op.create_table(
        "user_role",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("role_id", sa.Integer(), nullable=False),
        sa.Column("assigned_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("user_id", "role_id"),
    )

    # --- Backfill ---
    # One "Admin" role per existing organization (unique constraint safe: 1 org per user)
    op.execute("""
        INSERT INTO roles (organization_id, name, description, is_system, created_at, updated_at)
        SELECT id, 'ADMIN', NULL, false, NOW(), NOW()
        FROM organizations;
    """)

    # Link every existing user to the Admin role of their organization
    op.execute("""
        INSERT INTO user_role (user_id, role_id, assigned_at)
        SELECT u.id, r.id, NOW()
        FROM users u
        JOIN roles r ON r.organization_id = u.organization_id AND r.name = 'ADMIN';
    """)


def downgrade() -> None:
    op.drop_table("user_role")
    op.drop_index(op.f("ix_roles_organization_id"), table_name="roles")
    op.drop_index(op.f("ix_roles_id"), table_name="roles")
    op.drop_table("roles")
