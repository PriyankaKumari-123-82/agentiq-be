"""rename mcp_server_configs.server_name/server_url to name/url;
add description + requires_query_building

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-04-24 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f6a7b8c9d0e1'
down_revision = 'e5f6a7b8c9d0'
branch_labels = None
depends_on = None


def upgrade():
    # 1. Rename server_name -> name (keep indexed, nullability unchanged).
    op.alter_column(
        'mcp_server_configs',
        'server_name',
        new_column_name='name',
        existing_type=sa.String(),
    )

    # 2. Rename server_url -> url.
    op.alter_column(
        'mcp_server_configs',
        'server_url',
        new_column_name='url',
        existing_type=sa.String(),
        existing_nullable=False,
    )

    # 3. Add `description` (NOT NULL, server_default '' so existing rows are valid).
    op.add_column(
        'mcp_server_configs',
        sa.Column(
            'description',
            sa.String(),
            nullable=False,
            server_default='',
        ),
    )

    # 4. Add `requires_query_building` (NOT NULL, server_default false).
    op.add_column(
        'mcp_server_configs',
        sa.Column(
            'requires_query_building',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('false'),
        ),
    )


def downgrade():
    # Remove new columns.
    op.drop_column('mcp_server_configs', 'requires_query_building')
    op.drop_column('mcp_server_configs', 'description')

    # Rename url -> server_url, name -> server_name.
    op.alter_column(
        'mcp_server_configs',
        'url',
        new_column_name='server_url',
        existing_type=sa.String(),
        existing_nullable=False,
    )
    op.alter_column(
        'mcp_server_configs',
        'name',
        new_column_name='server_name',
        existing_type=sa.String(),
    )
