"""add_user_db_and_is_database_enabled

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-13 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b2c3d4e5f6a7'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'ai_agent_settings',
        sa.Column('is_database_enabled', sa.Boolean(), nullable=False, server_default='false')
    )
    op.alter_column('ai_agent_settings', 'is_database_enabled', server_default=None)

    op.create_table(
        'user_db',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('agent_settings_id', sa.Integer(), nullable=False),
        sa.Column('db_type', sa.String(), nullable=False),
        sa.Column('host', sa.String(), nullable=False),
        sa.Column('port', sa.Integer(), nullable=False),
        sa.Column('db_name', sa.String(), nullable=False),
        sa.Column('username', sa.Text(), nullable=False),
        sa.Column('password', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['agent_settings_id'], ['ai_agent_settings.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('agent_settings_id'),
    )
    op.create_index('ix_user_db_agent_settings_id', 'user_db', ['agent_settings_id'])


def downgrade():
    op.drop_index('ix_user_db_agent_settings_id', table_name='user_db')
    op.drop_table('user_db')
    op.drop_column('ai_agent_settings', 'is_database_enabled')
