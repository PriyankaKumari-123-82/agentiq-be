"""add_zai_model_provider

Revision ID: b7f2c9a4d6e8
Revises: d4b8e21a8f9c
Create Date: 2026-07-24 11:05:00.000000
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b7f2c9a4d6e8'
down_revision = 'd4b8e21a8f9c'
branch_labels = None
depends_on = None


ZAI_MODEL = {
    "provider": "zai",
    "model_name": "glm-4.5-flash",
    "type": "llm",
    "dimension": None,
    "litellm_prefix": "zai",
    "is_default": True,
    "is_active": True,
}


def upgrade():
    bind = op.get_bind()
    bind.execute(
        sa.text(
            """
            INSERT INTO model_provider (
                provider,
                model_name,
                type,
                dimension,
                litellm_prefix,
                is_default,
                is_active
            )
            SELECT
                :provider,
                :model_name,
                :type,
                :dimension,
                :litellm_prefix,
                :is_default,
                :is_active
            WHERE NOT EXISTS (
                SELECT 1
                FROM model_provider
                WHERE provider = :provider
                  AND model_name = :model_name
                  AND type = :type
            )
            """
        ),
        ZAI_MODEL,
    )


def downgrade():
    bind = op.get_bind()
    bind.execute(
        sa.text(
            """
            DELETE FROM model_provider
            WHERE provider = :provider
              AND model_name = :model_name
              AND type = :type
            """
        ),
        {
            "provider": ZAI_MODEL["provider"],
            "model_name": ZAI_MODEL["model_name"],
            "type": ZAI_MODEL["type"],
        },
    )