"""Add is_active to organizations, roles, org_documents

Revision ID: e4827ce43e2e
Revises: fef8a3fe77d0
Create Date: 2026-08-10 06:04:39.274826

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes

# revision identifiers, used by Alembic.
revision = "e4827ce43e2e"
down_revision = "a6c9e2f4b8d1"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column("organizations", sa.Column("is_active", sa.Boolean(), server_default="true", nullable=True))
    op.create_index(op.f("ix_organizations_is_active"), "organizations", ["is_active"], unique=False)

    op.add_column("roles", sa.Column("is_active", sa.Boolean(), server_default="true", nullable=True))
    op.create_index(op.f("ix_roles_is_active"), "roles", ["is_active"], unique=False)

    op.add_column("organization_documents", sa.Column("is_active", sa.Boolean(), server_default="true", nullable=True))
    op.create_index(op.f("ix_organization_documents_is_active"), "organization_documents", ["is_active"], unique=False)

def downgrade():
    op.drop_index(op.f("ix_organization_documents_is_active"), table_name="organization_documents")
    op.drop_column("organization_documents", "is_active")

    op.drop_index(op.f("ix_roles_is_active"), table_name="roles")
    op.drop_column("roles", "is_active")

    op.drop_index(op.f("ix_organizations_is_active"), table_name="organizations")
    op.drop_column("organizations", "is_active")
