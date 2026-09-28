"""add whatsapp to agent settings

Revision ID: d6f0a4b2c3e6
Revises: c5e9f3b1a2c5
Create Date: 2026-08-18 17:05:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd6f0a4b2c3e6'
down_revision = 'c5e9f3b1a2c5'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('ai_agent_settings', sa.Column('whatsapp_enabled', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('ai_agent_settings', sa.Column('whatsapp_number', sa.String(), nullable=True))


def downgrade():
    op.drop_column('ai_agent_settings', 'whatsapp_number')
    op.drop_column('ai_agent_settings', 'whatsapp_enabled')
