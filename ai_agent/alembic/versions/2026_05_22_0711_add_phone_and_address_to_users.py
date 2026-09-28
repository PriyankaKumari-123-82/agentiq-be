"""add phone and address to users

Revision ID: 61c5d4a06ffc
Revises: 6be086b9bbf3
Create Date: 2026-05-22 07:11:43.102499

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '61c5d4a06ffc'
down_revision = '6be086b9bbf3'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'users',
        sa.Column('phone', sa.String(), nullable=True)
    )

    op.add_column(
        'users',
        sa.Column('address', sa.String(), nullable=True)
    )


def downgrade():
    op.drop_column('users', 'address')
    op.drop_column('users', 'phone')
