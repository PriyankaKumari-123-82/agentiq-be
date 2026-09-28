"""Add pluggy_tenant_id to organizations

Revision ID: f9d7c4a1b2e8
Revises: b7e2f3a4c5d6
Create Date: 2026-08-03 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "f9d7c4a1b2e8"
down_revision = "b7e2f3a4c5d6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("organizations", sa.Column("pluggy_tenant_id", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("organizations", "pluggy_tenant_id")
