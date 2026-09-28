"""Add embed_config and logo_url to public_api_keys

Embed widget appearance (colours, logo, dimensions, branding) is stored per-SDK-key
rather than per-agent, allowing each deployment (website, mobile app, etc.) to have
its own independent branding.

Revision ID: b7e2f3a4c5d6
Revises: d7e8f9a0b1c2
Create Date: 2026-07-27 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "b7e2f3a4c5d6"
down_revision = "d7e8f9a0b1c2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("public_api_keys", sa.Column("embed_config", sa.JSON(), nullable=True))
    op.add_column("public_api_keys", sa.Column("logo_url", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("public_api_keys", "logo_url")
    op.drop_column("public_api_keys", "embed_config")
