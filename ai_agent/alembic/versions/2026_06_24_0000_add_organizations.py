"""Add organizations table and users.org_id FK.

Revision ID: fcccbfcbce0d
Revises: 1636297f7af1
Create Date: 2026-06-24 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "fcccbfcbce0d"
down_revision = "1636297f7af1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "organizations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_organizations_id"), "organizations", ["id"], unique=False)

    op.add_column("users", sa.Column("organization_id", sa.Integer(), nullable=True))
    op.create_index(op.f("ix_users_organization_id"), "users", ["organization_id"], unique=False)
    op.create_foreign_key(
        "fk_users_organization_id_organizations",
        "users",
        "organizations",
        ["organization_id"],
        ["id"],
    )

    # Backfill: create one org per existing user (named after first_name) and link it.
    op.execute("""
        INSERT INTO organizations (name, created_at, updated_at)
        SELECT first_name, NOW(), NOW()
        FROM users;

        UPDATE users
        SET organization_id = (
            SELECT id FROM organizations
            WHERE organizations.name = users.first_name
            ORDER BY id
            LIMIT 1
        )
        WHERE organization_id IS NULL;
    """)


def downgrade() -> None:
    op.drop_constraint("fk_users_organization_id_organizations", "users", type_="foreignkey")
    op.drop_index(op.f("ix_users_organization_id"), table_name="users")
    op.drop_column("users", "organization_id")

    op.drop_index(op.f("ix_organizations_id"), table_name="organizations")
    op.drop_table("organizations")
