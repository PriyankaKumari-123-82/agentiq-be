"""Add is_archived to user_sessions.

Revision ID: 8a2b3c4d5e6f
Revises: 8e3f1a2b9c4d
Create Date: 2026-06-30 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "8a2b3c4d5e6f"
down_revision = "8e3f1a2b9c4d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'user_sessions',
        sa.Column('is_archived', sa.Boolean(), nullable=False, server_default='false')
    )
    op.create_index(
        'ix_user_sessions_is_archived',
        'user_sessions',
        ['is_archived']
    )


def downgrade() -> None:
    op.drop_index('ix_user_sessions_is_archived')
    op.drop_column('user_sessions', 'is_archived')
