"""Add soft delete to user_sessions

Revision ID: 2026_05_26_0000
Revises: 61c5d4a06ffc
Create Date: 2026-05-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '2026_05_26_0000'
down_revision: Union[str, Sequence[str], None] = '61c5d4a06ffc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'user_sessions',
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true')
    )
    op.create_index(
        'ix_user_sessions_is_active',
        'user_sessions',
        ['is_active']
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_user_sessions_is_active")
    op.drop_column('user_sessions', 'is_active')
