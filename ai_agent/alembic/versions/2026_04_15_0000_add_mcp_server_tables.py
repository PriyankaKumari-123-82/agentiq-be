"""add mcp_server_configs and agent_mcp_config_links tables

Revision ID: d4e5f6a7b8c9
Revises: b2c3d4e5f6a7
Create Date: 2026-04-15 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd4e5f6a7b8c9'
down_revision = 'b2c3d4e5f6a7'
branch_labels = None
depends_on = None


def upgrade():
    # Create mcp_server_configs table
    op.create_table(
        'mcp_server_configs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('server_name', sa.String(), nullable=False),
        sa.Column('command', sa.String(), nullable=True),
        sa.Column('args', sa.JSON(), nullable=True),
        sa.Column('env', sa.JSON(), nullable=True),
        sa.Column('server_url', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_mcp_server_configs_user_id'), 'mcp_server_configs', ['user_id'])
    op.create_index(op.f('ix_mcp_server_configs_server_name'), 'mcp_server_configs', ['server_name'])

    # Create agent_mcp_config_links table
    op.create_table(
        'agent_mcp_config_links',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('agent_settings_id', sa.Integer(), nullable=False),
        sa.Column('mcp_server_config_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['agent_settings_id'], ['ai_agent_settings.id']),
        sa.ForeignKeyConstraint(['mcp_server_config_id'], ['mcp_server_configs.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_agent_mcp_config_links_agent_settings_id'), 'agent_mcp_config_links', ['agent_settings_id'])
    op.create_index(op.f('ix_agent_mcp_config_links_mcp_server_config_id'), 'agent_mcp_config_links', ['mcp_server_config_id'])

    # Add is_mcp_enabled flag to ai_agent_settings
    op.add_column(
        'ai_agent_settings',
        sa.Column('is_mcp_enabled', sa.Boolean(), nullable=False, server_default=sa.text('false'))
    )


def downgrade():
    # Drop is_mcp_enabled column
    op.drop_column('ai_agent_settings', 'is_mcp_enabled')

    # Drop link table
    op.drop_index(op.f('ix_agent_mcp_config_links_mcp_server_config_id'), table_name='agent_mcp_config_links')
    op.drop_index(op.f('ix_agent_mcp_config_links_agent_settings_id'), table_name='agent_mcp_config_links')
    op.drop_table('agent_mcp_config_links')

    # Drop mcp_server_configs table
    op.drop_index(op.f('ix_mcp_server_configs_server_name'), table_name='mcp_server_configs')
    op.drop_index(op.f('ix_mcp_server_configs_user_id'), table_name='mcp_server_configs')
    op.drop_table('mcp_server_configs')
