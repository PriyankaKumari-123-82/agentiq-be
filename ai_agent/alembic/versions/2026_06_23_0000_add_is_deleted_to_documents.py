"""Add is_active column to documents table

Revision ID: 2026_06_23_0000
Revises: 2026_06_22_0000
Create Date: 2026-06-23 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "2026_06_23_0000"
down_revision = "2026_06_22_0000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
    )
    op.create_index("ix_documents_is_active", "documents", ["is_active"])


def downgrade() -> None:
    op.drop_index("ix_documents_is_active", table_name="documents")
    op.drop_column("documents", "is_active")
