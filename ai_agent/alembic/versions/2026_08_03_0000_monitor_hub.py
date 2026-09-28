"""monitor hub — schema finalisation + monitor type catalog

Revision ID: 9a1b2c3d4e5f
Revises: fef8a3fe77d0
Create Date: 2026-08-03 00:00:00.000000
"""
from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa

revision = "9a1b2c3d4e5f"
down_revision = "e4827ce43e2e"
branch_labels = None
depends_on = None

NOW = datetime.now(timezone.utc)

# Singular names created by d7e8f9a0b1c2 → plural names the SQLModel models use.
_TABLE_RENAMES = (
    ("monitor_type", "monitor_types"),
    ("monitor", "monitors"),
    ("monitor_result", "monitor_results"),
)

MONITOR_TYPES = [
    {
        "name": "COUNT",
        "title": "Count",
        "description": "Returns a single integer count — e.g. number of open POs, pending PRs.",
        "result_type": "NUMBER",
        "configuration_schema": '{"result_type": "NUMBER", "format": "integer"}',
    },
    {
        "name": "KPI",
        "title": "KPI / Metric",
        "description": "Returns a single numeric KPI value — e.g. total spend, average delivery days.",
        "result_type": "NUMBER",
        "configuration_schema": '{"result_type": "NUMBER", "format": "decimal"}',
    },
    {
        "name": "LIST",
        "title": "List / Table",
        "description": "Returns a tabular result — rows and columns. Ideal for top-N lists and detail views.",
        "result_type": "JSON",
        "configuration_schema": '{"result_type": "JSON", "format": "table", "fields": ["rows", "columns"]}',
    },
    {
        "name": "BOOLEAN",
        "title": "Yes / No Alert",
        "description": "Returns true/false — e.g. are there any overdue deliveries?",
        "result_type": "BOOLEAN",
        "configuration_schema": '{"result_type": "BOOLEAN", "format": "boolean"}',
    },
    {
        "name": "TREND",
        "title": "Trend",
        "description": "Returns a series of values over time — e.g. daily order counts for the last 7 days.",
        "result_type": "JSON",
        "configuration_schema": '{"result_type": "JSON", "format": "time_series", "fields": ["date", "value"]}',
    },
    {
        "name": "AI_QUERY",
        "title": "AI Summary",
        "description": "Free-form text answer from the AI — e.g. summarise vendor performance this quarter.",
        "result_type": "TEXT",
        "configuration_schema": '{"result_type": "TEXT", "format": "markdown"}',
    },
    {
        "name": "PIE_CHART",
        "title": "Pie Chart",
        "description": "Returns a breakdown by category — e.g. spend by vendor, POs by status.",
        "result_type": "JSON",
        "configuration_schema": (
            '{"result_type": "JSON", "format": "chart", "chart_type": "pie", '
            '"fields": ["label", "value"]}'
        ),
    },
    {
        "name": "BAR_CHART",
        "title": "Bar Chart",
        "description": "Returns values across categories — e.g. POs per vendor, spend per department.",
        "result_type": "JSON",
        "configuration_schema": (
            '{"result_type": "JSON", "format": "chart", "chart_type": "bar", '
            '"fields": ["label", "value"]}'
        ),
    },
    {
        "name": "LINE_CHART",
        "title": "Line Chart",
        "description": "Returns a trend over time — e.g. daily order counts, spend over the last 30 days.",
        "result_type": "JSON",
        "configuration_schema": (
            '{"result_type": "JSON", "format": "chart", "chart_type": "line", '
            '"fields": ["label", "value"]}'
        ),
    },
]


def _columns(inspector, table: str) -> set[str]:
    return {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    # ── Align table names with the models ────────────────────────────────
    for old, new in _TABLE_RENAMES:
        if old in tables and new not in tables:
            op.rename_table(old, new)
            tables.discard(old)
            tables.add(new)

    if "monitors" not in tables or "monitor_types" not in tables:
        # Base tables absent — d7e8f9a0b1c2 owns creating them.
        return

    inspector = sa.inspect(conn)

    # ── monitor_types.result_type ────────────────────────────────────────
    if "result_type" not in _columns(inspector, "monitor_types"):
        op.add_column(
            "monitor_types",
            sa.Column("result_type", sa.String(), nullable=False, server_default="TEXT"),
        )

    # ── monitors: new columns ────────────────────────────────────────────
    monitor_cols = _columns(inspector, "monitors")
    if "deleted_at" not in monitor_cols:
        op.add_column(
            "monitors",
            sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        )
    if "llm_query" not in monitor_cols:
        op.add_column("monitors", sa.Column("llm_query", sa.String(), nullable=True))
    if "followup_question" not in monitor_cols:
        op.add_column("monitors", sa.Column("followup_question", sa.String(), nullable=True))

    # ── monitors: scheduler_frequency VARCHAR → INTEGER ──────────────────
    # SQLite has dynamic typing and no ALTER COLUMN TYPE, so the conversion is
    # both impossible and unnecessary there.
    if "scheduler_frequency" in monitor_cols and conn.dialect.name != "sqlite":
        current = next(
            c for c in inspector.get_columns("monitors")
            if c["name"] == "scheduler_frequency"
        )
        if isinstance(current["type"], sa.String):
            op.alter_column(
                "monitors", "scheduler_frequency",
                existing_type=sa.VARCHAR(),
                type_=sa.Integer(),
                existing_nullable=True,
                postgresql_using="scheduler_frequency::integer",
            )

    # ── monitors: drop agent scoping (monitors are organization-scoped) ──
    if "agent_id" in monitor_cols:
        fks = {fk["name"] for fk in inspector.get_foreign_keys("monitors")}
        if "fk_monitors_agent_id" in fks:
            op.drop_constraint("fk_monitors_agent_id", "monitors", type_="foreignkey")
        indexes = {ix["name"] for ix in inspector.get_indexes("monitors")}
        if "ix_monitors_agent_id" in indexes:
            op.drop_index("ix_monitors_agent_id", table_name="monitors")
        op.drop_column("monitors", "agent_id")

    # refresh_interval was renamed to scheduler_frequency
    if "refresh_interval" in monitor_cols:
        op.drop_column("monitors", "refresh_interval")

    # ── Seed the monitor type catalog ────────────────────────────────────
    row = conn.execute(sa.text("SELECT id FROM users ORDER BY id LIMIT 1")).fetchone()
    if row is None:
        # No users yet — nothing to attribute the seed rows to.
        return
    created_by = row[0]

    for mt in MONITOR_TYPES:
        exists = conn.execute(
            sa.text("SELECT 1 FROM monitor_types WHERE name = :name"),
            {"name": mt["name"]},
        ).fetchone()
        if exists:
            continue

        conn.execute(
            sa.text("""
                INSERT INTO monitor_types
                    (name, title, description, result_type, configuration_schema,
                     is_active, created_by, created_at, updated_at)
                VALUES
                    (:name, :title, :description, :result_type, :configuration_schema,
                     true, :created_by, :now, :now)
            """),
            {
                "name": mt["name"],
                "title": mt["title"],
                "description": mt["description"],
                "result_type": mt["result_type"],
                "configuration_schema": mt["configuration_schema"],
                "created_by": created_by,
                "now": NOW,
            },
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "monitor_types" in tables:
        conn.execute(
            sa.text("DELETE FROM monitor_types WHERE name = ANY(:names)"),
            {"names": [mt["name"] for mt in MONITOR_TYPES]},
        )

    if "monitors" in tables:
        monitor_cols = _columns(inspector, "monitors")
        op.add_column("monitors", sa.Column("refresh_interval", sa.Integer(), nullable=True))
        op.add_column("monitors", sa.Column("agent_id", sa.String(), nullable=True))
        op.create_foreign_key(
            "fk_monitors_agent_id", "monitors", "ai_agents", ["agent_id"], ["id"],
        )
        op.create_index("ix_monitors_agent_id", "monitors", ["agent_id"], unique=False)
        op.alter_column(
            "monitors", "scheduler_frequency",
            existing_type=sa.Integer(), type_=sa.VARCHAR(), existing_nullable=True,
        )
        for col in ("followup_question", "llm_query", "deleted_at"):
            if col in monitor_cols:
                op.drop_column("monitors", col)

    # Restore the singular table names created by d7e8f9a0b1c2.
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())
    for old, new in _TABLE_RENAMES:
        if new in tables and old not in tables:
            op.rename_table(new, old)
