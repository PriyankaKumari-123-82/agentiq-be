"""add_voice_columns

Revision ID: a1b2c3d4e5f6
Revises: 985d86032534
Create Date: 2026-02-12 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = '985d86032534'
branch_labels = None
depends_on = None


def upgrade():
    # Add conversation_type to messages table (default 'text' for backward compat)
    op.add_column('messages', sa.Column('conversation_type', sa.String(), nullable=False, server_default='text'))

    # Add llm_voice_prompt to ai_agent_settings table (nullable)
    op.add_column('ai_agent_settings', sa.Column('llm_voice_prompt', sa.Text(), nullable=True))


def downgrade():
    op.drop_column('ai_agent_settings', 'llm_voice_prompt')
    op.drop_column('messages', 'conversation_type')
