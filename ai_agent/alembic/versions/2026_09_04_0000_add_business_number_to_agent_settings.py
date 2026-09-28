"""Add business_number to ai_agent_settings; relax whatsapp number uniqueness to per-agent

Revision ID: c3a1e9f7b2d0
Revises: 4ea94ac7e147
Create Date: 2026-09-04 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c3a1e9f7b2d0'
down_revision = '4ea94ac7e147'
branch_labels = None
depends_on = None


def upgrade():
    # 1. Add business_number column to ai_agent_settings
    #    Nullable so existing agents are unaffected.
    op.add_column(
        'ai_agent_settings',
        sa.Column('business_number', sa.String(), nullable=True)
    )
    op.create_index(
        'ix_ai_agent_settings_business_number',
        'ai_agent_settings',
        ['business_number'],
    )
    op.create_unique_constraint(
        'uq_ai_agent_settings_business_number',
        'ai_agent_settings',
        ['business_number'],
    )

    # 2. Relax the global uniqueness constraint on agent_whatsapp_numbers.phone_number
    #    to a per-agent constraint: (phone_number, agent_id).
    #    This allows the same user phone to be registered to multiple agents
    #    (each accessed via a different business number), while still preventing
    #    the same number being added twice to the same agent.
    op.drop_constraint('uq_wa_number_phone', 'agent_whatsapp_numbers', type_='unique')
    op.create_unique_constraint(
        'uq_wa_number_phone_agent',
        'agent_whatsapp_numbers',
        ['phone_number', 'agent_id'],
    )


def downgrade():
    # Reverse: restore global uniqueness, remove per-agent constraint
    op.drop_constraint('uq_wa_number_phone_agent', 'agent_whatsapp_numbers', type_='unique')
    op.create_unique_constraint(
        'uq_wa_number_phone',
        'agent_whatsapp_numbers',
        ['phone_number'],
    )

    op.drop_index('ix_ai_agent_settings_business_number', table_name='ai_agent_settings')
    op.drop_constraint('uq_ai_agent_settings_business_number', 'ai_agent_settings', type_='unique')
    op.drop_column('ai_agent_settings', 'business_number')
