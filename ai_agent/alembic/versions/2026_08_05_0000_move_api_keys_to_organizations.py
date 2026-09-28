"""Move provider API keys from agents to organizations.

Revision ID: a6c9e2f4b8d1
Revises: fef8a3fe77d0
Create Date: 2026-08-05 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "a6c9e2f4b8d1"
down_revision = "fef8a3fe77d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("api_keys", sa.Column("organization_id", sa.Integer(), nullable=True))
    op.add_column(
        "api_keys",
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column(
        "api_keys",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.add_column("api_keys", sa.Column("deleted_at", sa.DateTime(), nullable=True))
    op.add_column("ai_agent_settings", sa.Column("api_key", sa.Integer(), nullable=True))

    op.execute("""
        UPDATE api_keys AS k
        SET organization_id = COALESCE(a.organization_id, u.active_organization_id)
        FROM ai_agent_settings AS s
        JOIN ai_agents AS a ON a.id = s.agent_id
        JOIN users AS u ON u.id = a.owner_id
        WHERE k.agent_settings_id = s.id
          AND k.organization_id IS NULL;
    """)

    op.execute("""
        DELETE FROM api_keys AS k
        USING (
            SELECT id
            FROM (
                SELECT
                    id,
                    ROW_NUMBER() OVER (
                        PARTITION BY organization_id, LOWER(provider)
                        ORDER BY updated_at DESC NULLS LAST, id DESC
                    ) AS rn
                FROM api_keys
                WHERE organization_id IS NOT NULL
            ) AS ranked
            WHERE rn > 1
        ) AS duplicates
        WHERE k.id = duplicates.id;
    """)

    op.execute("UPDATE api_keys SET provider = LOWER(TRIM(provider));")
    op.alter_column("api_keys", "organization_id", nullable=False)

    op.drop_constraint("api_keys_agent_settings_id_fkey", "api_keys", type_="foreignkey")
    op.drop_index(op.f("ix_api_keys_agent_settings_id"), table_name="api_keys")
    op.drop_column("api_keys", "agent_settings_id")

    op.create_foreign_key(
        "fk_api_keys_organization_id_organizations",
        "api_keys",
        "organizations",
        ["organization_id"],
        ["id"],
    )
    op.create_index(op.f("ix_api_keys_organization_id"), "api_keys", ["organization_id"])
    op.create_index(op.f("ix_ai_agent_settings_api_key"), "ai_agent_settings", ["api_key"])
    op.create_foreign_key(
        "fk_ai_agent_settings_api_key_api_keys",
        "ai_agent_settings",
        "api_keys",
        ["api_key"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "uq_api_keys_org_provider_active",
        "api_keys",
        ["organization_id", "provider"],
        unique=True,
        postgresql_where=sa.text("is_active = true"),
    )
    op.create_index(
        "uq_api_keys_org_default",
        "api_keys",
        ["organization_id"],
        unique=True,
        postgresql_where=sa.text("is_default = true AND is_active = true"),
    )
    op.alter_column("api_keys", "is_default", server_default=None)
    op.alter_column("api_keys", "is_active", server_default=None)


def downgrade() -> None:
    op.add_column("api_keys", sa.Column("agent_settings_id", sa.Integer(), nullable=True))

    op.execute("""
        UPDATE api_keys AS k
        SET agent_settings_id = s.id
        FROM ai_agent_settings AS s
        JOIN ai_agents AS a ON a.id = s.agent_id
        WHERE a.organization_id = k.organization_id
          AND a.deleted = false
          AND k.agent_settings_id IS NULL;
    """)

    op.drop_index("uq_api_keys_org_default", table_name="api_keys")
    op.drop_constraint("fk_ai_agent_settings_api_key_api_keys", "ai_agent_settings", type_="foreignkey")
    op.drop_index(op.f("ix_ai_agent_settings_api_key"), table_name="ai_agent_settings")
    op.drop_index("uq_api_keys_org_provider_active", table_name="api_keys")
    op.drop_index(op.f("ix_api_keys_organization_id"), table_name="api_keys")
    op.drop_constraint("fk_api_keys_organization_id_organizations", "api_keys", type_="foreignkey")

    op.alter_column("api_keys", "agent_settings_id", nullable=False)
    op.create_index(op.f("ix_api_keys_agent_settings_id"), "api_keys", ["agent_settings_id"])
    op.create_foreign_key(
        "api_keys_agent_settings_id_fkey",
        "api_keys",
        "ai_agent_settings",
        ["agent_settings_id"],
        ["id"],
    )

    op.drop_column("api_keys", "is_default")
    op.drop_column("api_keys", "deleted_at")
    op.drop_column("api_keys", "is_active")
    op.drop_column("api_keys", "organization_id")
    op.drop_column("ai_agent_settings", "api_key")
