"""Add organization_settings table for org-level embedding model selection

Revision ID: 12bd6c43c0b4
Revises: b3d7e1a9f2c4
Create Date: 2026-08-14 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "12bd6c43c0b4"
down_revision = "b3d7e1a9f2c4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "organization_settings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("organization_id", sa.Integer(), sa.ForeignKey("organizations.id"), nullable=False, unique=True),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_organization_settings_organization_id",
        "organization_settings",
        ["organization_id"],
    )

    connection = op.get_bind()

    # Backfill: for each organization that already has agents, pick the most
    # common vector_embedding_model among its agents as the org-wide default.
    # (ai_agent_settings.vector_embedding_model is the legacy per-agent column —
    # unrelated to and not renamed by this migration.)
    connection.execute(sa.text(
        """
        INSERT INTO organization_settings (organization_id, embedding_model, created_at, updated_at)
        SELECT organization_id, embedding_model, now(), now()
        FROM (
            SELECT
                a.organization_id AS organization_id,
                s.vector_embedding_model AS embedding_model,
                COUNT(*) AS model_count,
                ROW_NUMBER() OVER (
                    PARTITION BY a.organization_id
                    ORDER BY COUNT(*) DESC, s.vector_embedding_model
                ) AS rn
            FROM ai_agents a
            JOIN ai_agent_settings s ON s.agent_id = a.id
            WHERE a.organization_id IS NOT NULL
              AND s.vector_embedding_model IS NOT NULL
            GROUP BY a.organization_id, s.vector_embedding_model
        ) ranked
        WHERE rn = 1
        """
    ))


def downgrade() -> None:
    op.drop_index("ix_organization_settings_organization_id", table_name="organization_settings")
    op.drop_table("organization_settings")
