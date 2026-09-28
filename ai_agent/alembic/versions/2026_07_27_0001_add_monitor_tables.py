"""add monitor_type, monitor, monitor_result tables

Creates the three Monitor domain tables. monitor_type is seeded with
initial rows by the application at startup or via management command.

Models live in ai_agent/custom_model/monitor.py (imported by models.py).

Revision ID: d7e8f9a0b1c2
Revises: c5e7a9f1d3b2
Create Date: 2026-07-27 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = "d7e8f9a0b1c2"
down_revision = "c5e7a9f1d3b2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── monitor_type ──────────────────────────────────────────────────────────
    op.create_table(
        "monitor_type",
        sa.Column("id",                   sa.Integer(),               nullable=False, autoincrement=True),
        sa.Column("name",                 sa.String(100),             nullable=False),
        sa.Column("title",                sa.String(100),             nullable=False),
        sa.Column("description",          sa.Text(),                  nullable=False, server_default=""),
        sa.Column("configuration_schema", sa.Text(),                  nullable=False, server_default="{}"),
        sa.Column("is_active",            sa.Boolean(),               nullable=False, server_default=sa.true()),
        sa.Column("created_by",           sa.Integer(),               nullable=False),
        sa.Column("created_at",           sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at",           sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_monitor_type"),
        sa.UniqueConstraint("name", name="uq_monitor_type_name"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], name="fk_monitor_type_created_by"),
    )
    op.create_index("ix_monitor_type_name", "monitor_type", ["name"], unique=True)

    # ── monitor ───────────────────────────────────────────────────────────────
    op.create_table(
        "monitor",
        sa.Column("id",                  sa.Integer(),               nullable=False, autoincrement=True),
        sa.Column("organization_id",     sa.Integer(),               nullable=True),
        sa.Column("agent_id",            sa.String(),                nullable=False),
        sa.Column("monitor_type_id",     sa.Integer(),               nullable=False),
        sa.Column("name",                sa.String(200),             nullable=False),
        sa.Column("description",         sa.Text(),                  nullable=True),
        sa.Column("natural_query",       sa.Text(),                  nullable=False),
        sa.Column("scheduler_frequency", sa.Text(),                  nullable=True),
        sa.Column("refresh_interval",    sa.Integer(),               nullable=False, server_default="3600"),
        sa.Column("is_active",           sa.Boolean(),               nullable=False, server_default=sa.true()),
        sa.Column("created_by",          sa.Integer(),               nullable=False),
        sa.Column("deleted_at",          sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at",          sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at",          sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_monitor"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], name="fk_monitor_organization_id"),
        sa.ForeignKeyConstraint(["agent_id"],        ["ai_agents.id"],     name="fk_monitor_agent_id"),
        sa.ForeignKeyConstraint(["monitor_type_id"], ["monitor_type.id"],  name="fk_monitor_monitor_type_id"),
        sa.ForeignKeyConstraint(["created_by"],      ["users.id"],         name="fk_monitor_created_by"),
    )
    op.create_index("ix_monitor_organization_id", "monitor", ["organization_id"], unique=False)
    op.create_index("ix_monitor_agent_id",        "monitor", ["agent_id"],        unique=False)
    op.create_index("ix_monitor_monitor_type_id", "monitor", ["monitor_type_id"], unique=False)
    op.create_index("ix_monitor_is_active",       "monitor", ["is_active"],       unique=False)
    op.create_index("ix_monitor_created_by",      "monitor", ["created_by"],      unique=False)

    # ── monitor_result ────────────────────────────────────────────────────────
    op.create_table(
        "monitor_result",
        sa.Column("id",            sa.Integer(),               nullable=False, autoincrement=True),
        sa.Column("monitor_id",    sa.Integer(),               nullable=False),
        sa.Column("result_type",   sa.String(30),              nullable=False),
        sa.Column("result",        sa.Text(),                  nullable=True),
        sa.Column("error_message", sa.Text(),                  nullable=True),
        sa.Column("is_active",     sa.Boolean(),               nullable=False, server_default=sa.true()),
        sa.Column("created_at",    sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at",    sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_monitor_result"),
        sa.ForeignKeyConstraint(
            ["monitor_id"], ["monitor.id"],
            name="fk_monitor_result_monitor_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_monitor_result_monitor_id", "monitor_result", ["monitor_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_monitor_result_monitor_id", table_name="monitor_result")
    op.drop_table("monitor_result")

    op.drop_index("ix_monitor_created_by",      table_name="monitor")
    op.drop_index("ix_monitor_is_active",       table_name="monitor")
    op.drop_index("ix_monitor_monitor_type_id", table_name="monitor")
    op.drop_index("ix_monitor_agent_id",        table_name="monitor")
    op.drop_index("ix_monitor_organization_id", table_name="monitor")
    op.drop_table("monitor")

    op.drop_index("ix_monitor_type_name", table_name="monitor_type")
    op.drop_table("monitor_type")
