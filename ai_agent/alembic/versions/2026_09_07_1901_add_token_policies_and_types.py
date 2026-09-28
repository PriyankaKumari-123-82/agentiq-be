"""add token policies and types

Revision ID: 4ea94ac7e147
Revises: c5d6e7f8g9h0
Create Date: 2026-09-07 19:01:43.402281

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision = '4ea94ac7e147'
down_revision = 'c5d6e7f8g9h0'
branch_labels = None
depends_on = None


def upgrade():
    # 1. Add tool_tokens and embedding_tokens to llm_usage_events
    op.add_column('llm_usage_events', sa.Column('tool_tokens', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('llm_usage_events', sa.Column('embedding_tokens', sa.Integer(), nullable=False, server_default='0'))

    # 2. Create organization_token_policies
    op.create_table('organization_token_policies',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('organization_id', sa.Integer(), nullable=False),
        sa.Column('limit_enabled', sa.Boolean(), nullable=False),
        sa.Column('max_tokens', sa.Integer(), nullable=False),
        sa.Column('reset_period', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('alerts_enabled', sa.Boolean(), nullable=False),
        sa.Column('alert_threshold_percentage', sa.Integer(), nullable=False),
        sa.Column('notification_email', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_organization_token_policies_organization_id'), 'organization_token_policies', ['organization_id'], unique=True)

    # 3. Create user_token_policies
    op.create_table('user_token_policies',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('limit_enabled', sa.Boolean(), nullable=False),
        sa.Column('max_tokens', sa.Integer(), nullable=False),
        sa.Column('reset_period', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('alerts_enabled', sa.Boolean(), nullable=False),
        sa.Column('alert_threshold_percentage', sa.Integer(), nullable=False),
        sa.Column('notification_email', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_token_policies_user_id'), 'user_token_policies', ['user_id'], unique=True)

    # 4. Create agent_token_policies
    op.create_table('agent_token_policies',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('agent_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('limit_enabled', sa.Boolean(), nullable=False),
        sa.Column('max_tokens', sa.Integer(), nullable=False),
        sa.Column('reset_period', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('alerts_enabled', sa.Boolean(), nullable=False),
        sa.Column('alert_threshold_percentage', sa.Integer(), nullable=False),
        sa.Column('notification_email', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['agent_id'], ['ai_agents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_agent_token_policies_agent_id'), 'agent_token_policies', ['agent_id'], unique=True)


def downgrade():
    op.drop_index(op.f('ix_agent_token_policies_agent_id'), table_name='agent_token_policies')
    op.drop_table('agent_token_policies')
    
    op.drop_index(op.f('ix_user_token_policies_user_id'), table_name='user_token_policies')
    op.drop_table('user_token_policies')
    
    op.drop_index(op.f('ix_organization_token_policies_organization_id'), table_name='organization_token_policies')
    op.drop_table('organization_token_policies')
    
    op.drop_column('llm_usage_events', 'embedding_tokens')
    op.drop_column('llm_usage_events', 'tool_tokens')