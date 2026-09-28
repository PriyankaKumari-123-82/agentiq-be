"""add agent whatsapp numbers table and drop legacy whatsapp_number column

Revision ID: b9e7f8c2a1d4
Revises: 7436154890c0
Create Date: 2026-08-20 12:05:00.000000

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision = 'b9e7f8c2a1d4'
down_revision = 'e2b7c9a4d1f6'
branch_labels = None
depends_on = None


def upgrade():
    # 1. Create agent_whatsapp_numbers table (phone + label only)
    op.create_table(
        'agent_whatsapp_numbers',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('agent_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('phone_number', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('display_label', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['agent_id'], ['ai_agents.id'], name='fk_wa_number_agent_id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name='fk_wa_number_user_id'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('phone_number', name='uq_wa_number_phone'),
    )
    op.create_index('ix_agent_whatsapp_numbers_agent_id', 'agent_whatsapp_numbers', ['agent_id'])
    op.create_index('ix_agent_whatsapp_numbers_user_id', 'agent_whatsapp_numbers', ['user_id'])
    op.create_index('ix_agent_whatsapp_numbers_is_active', 'agent_whatsapp_numbers', ['is_active'])

    # 2. Add whatsapp_phone column to user_sessions for sender-based session lookup
    op.add_column(
        'user_sessions',
        sa.Column('whatsapp_phone', sqlmodel.sql.sqltypes.AutoString(), nullable=True)
    )
    op.create_index('ix_user_sessions_whatsapp_phone', 'user_sessions', ['whatsapp_phone'])

    # 3. Drop single-number column from ai_agent_settings (whatsapp_enabled toggle is preserved)
    op.drop_column('ai_agent_settings', 'whatsapp_number')


def downgrade():
    op.add_column('ai_agent_settings', sa.Column('whatsapp_number', sa.String(), nullable=True))

    op.drop_index('ix_user_sessions_whatsapp_phone', table_name='user_sessions')
    op.drop_column('user_sessions', 'whatsapp_phone')

    op.drop_index('ix_agent_whatsapp_numbers_is_active', table_name='agent_whatsapp_numbers')
    op.drop_index('ix_agent_whatsapp_numbers_user_id', table_name='agent_whatsapp_numbers')
    op.drop_index('ix_agent_whatsapp_numbers_agent_id', table_name='agent_whatsapp_numbers')
    op.drop_table('agent_whatsapp_numbers')
