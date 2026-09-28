"""Add welcome_message_sent to agent_whatsapp_numbers

Revision ID: d4b2e8f1c9a7
Revises: c3a1e9f7b2d0
Create Date: 2026-09-07 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd4b2e8f1c9a7'
down_revision = 'c3a1e9f7b2d0'
branch_labels = None
depends_on = None


def upgrade():
    # Add welcome_message_sent flag to track whether the Gallabox welcome_template
    # has already been dispatched for a registered number.  Defaults to False so
    # existing rows are treated as not yet sent (safe — they received the old text
    # welcome via send_whatsapp_text_message and are already onboarded).
    op.add_column(
        'agent_whatsapp_numbers',
        sa.Column(
            'welcome_message_sent',
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        )
    )


def downgrade():
    op.drop_column('agent_whatsapp_numbers', 'welcome_message_sent')
