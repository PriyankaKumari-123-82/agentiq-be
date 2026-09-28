"""drop stdio columns from mcp_server_configs (SSE-only)

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-04-21 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e5f6a7b8c9d0'
down_revision = 'd4e5f6a7b8c9'
branch_labels = None
depends_on = None


def upgrade():
    # Remove any stdio-only rows (no server_url) and their dependent links
    # so that the NOT NULL constraint on server_url can be applied safely.
    op.execute(
        """
        DELETE FROM agent_mcp_config_links
        WHERE mcp_server_config_id IN (
            SELECT id FROM mcp_server_configs WHERE server_url IS NULL
        )
        """
    )
    op.execute("DELETE FROM mcp_server_configs WHERE server_url IS NULL")

    # Drop stdio transport columns
    op.drop_column('mcp_server_configs', 'command')
    op.drop_column('mcp_server_configs', 'args')
    op.drop_column('mcp_server_configs', 'env')

    # Make server_url NOT NULL
    op.alter_column(
        'mcp_server_configs',
        'server_url',
        existing_type=sa.String(),
        nullable=False,
    )


def downgrade():
    # Restore server_url as nullable
    op.alter_column(
        'mcp_server_configs',
        'server_url',
        existing_type=sa.String(),
        nullable=True,
    )

    # Re-add stdio transport columns
    op.add_column('mcp_server_configs', sa.Column('command', sa.String(), nullable=True))
    op.add_column('mcp_server_configs', sa.Column('args', sa.JSON(), nullable=True))
    op.add_column('mcp_server_configs', sa.Column('env', sa.JSON(), nullable=True))
