"""add is_subscribed to users

Revision ID: 8f2c4a91b6d3
Revises: 1938f126963c
Create Date: 2026-09-22 00:00:00.000000

Free-trial token quota: a user is either subscribed (brings their own LLM
API key, never quota-checked) or on the trial (served on the platform's env
key, capped at FREE_TRIAL_TOKEN_LIMIT tokens). The trial balance itself is
computed live from llm_usage_events, so no balance column is needed here.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '8f2c4a91b6d3'
down_revision = '1938f126963c'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'users',
        sa.Column('is_subscribed', sa.Boolean(), nullable=False, server_default='false'),
    )
    op.create_index(op.f('ix_users_is_subscribed'), 'users', ['is_subscribed'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_users_is_subscribed'), table_name='users')
    op.drop_column('users', 'is_subscribed')
