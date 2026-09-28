"""Add is_cold_build column to ai_agents

Revision ID: 9b3c1d5e7f8a
Revises: 8a2b3c4d5e6f
Create Date: 2026-07-06 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "9b3c1d5e7f8a"
down_revision = "8a2b3c4d5e6f"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "ai_agents",
        sa.Column("is_cold_build", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )


def downgrade():
    op.drop_column("ai_agents", "is_cold_build")
