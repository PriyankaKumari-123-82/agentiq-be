"""align monitor_results with the MonitorResult model

Revision ID: e2a9d4c7b1f3
Revises: 9a1b2c3d4e5f
Create Date: 2026-08-11 00:00:00.000000
"""
from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa

revision = "e2a9d4c7b1f3"
down_revision = "9a1b2c3d4e5f"
branch_labels = None
depends_on = None

TABLE = "monitor_results"

# Withdrawn from the type catalog seeded by 9a1b2c3d4e5f.
RETIRED_MONITOR_TYPES = ("KPI", "BOOLEAN", "TREND", "AI_QUERY")

# Enough to put the catalog back on downgrade; matches 9a1b2c3d4e5f's seed.
RETIRED_TYPE_ROWS = [
    ("KPI", "KPI / Metric",
     "Returns a single numeric KPI value — e.g. total spend, average delivery days.",
     "NUMBER", '{"result_type": "NUMBER", "format": "decimal"}'),
    ("BOOLEAN", "Yes / No Alert",
     "Returns true/false — e.g. are there any overdue deliveries?",
     "BOOLEAN", '{"result_type": "BOOLEAN", "format": "boolean"}'),
    ("TREND", "Trend",
     "Returns a series of values over time — e.g. daily order counts for the last 7 days.",
     "JSON", '{"result_type": "JSON", "format": "time_series", "fields": ["date", "value"]}'),
    ("AI_QUERY", "AI Summary",
     "Free-form text answer from the AI — e.g. summarise vendor performance this quarter.",
     "TEXT", '{"result_type": "TEXT", "format": "markdown"}'),
]


def _columns(inspector) -> dict:
    return {c["name"]: c for c in inspector.get_columns(TABLE)}


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if TABLE not in set(inspector.get_table_names()):
        return

    is_sqlite = conn.dialect.name == "sqlite"
    cols = _columns(inspector)

    # ── executed_at ──────────────────────────────────────────────────────
    if "executed_at" not in cols:
        op.add_column(
            TABLE,
            sa.Column(
                "executed_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )
        # Preserve the run timestamp the old layout recorded.
        if "created_at" in cols:
            conn.execute(sa.text(f"UPDATE {TABLE} SET executed_at = created_at"))

    # ── status ───────────────────────────────────────────────────────────
    if "status" not in cols:
        op.add_column(
            TABLE,
            sa.Column(
                "status",
                sa.String(),
                nullable=False,
                server_default="SUCCESS",
            ),
        )

    # ── execution_time_ms ────────────────────────────────────────────────
    if "execution_time_ms" not in cols:
        op.add_column(TABLE, sa.Column("execution_time_ms", sa.Integer(), nullable=True))

    # ── result: TEXT → JSON ──────────────────────────────────────────────

    result_col = cols.get("result")
    if result_col is not None and not is_sqlite:
        if isinstance(result_col["type"], (sa.Text, sa.String)):
            op.alter_column(
                TABLE,
                "result",
                existing_type=sa.Text(),
                type_=sa.JSON(),
                existing_nullable=True,
                # Rows written before this migration hold JSON-encoded text;
                # anything else is wrapped rather than failing the migration.
                postgresql_using=(
                    "CASE WHEN result IS NULL OR btrim(result) = '' THEN NULL "
                    "WHEN btrim(result) ~ '^[\\[{\"]' "
                    "OR btrim(result) ~ '^-?[0-9]' "
                    "OR btrim(result) IN ('true','false','null') "
                    "THEN result::json "
                    "ELSE to_json(result) END"
                ),
            )

    # ── drop the audit columns the model does not carry ──────────────────
    for col in ("created_at", "updated_at"):
        if col in cols:
            op.drop_column(TABLE, col)

    _retire_monitor_types(conn, inspector)


def _retire_monitor_types(conn, inspector) -> None:
    """Delete the four withdrawn types from the catalog.

    GET /monitor/type ignores is_active, so they have to be deleted rather
    than deactivated. monitors.monitor_type_id references the catalog, so any
    monitor still on a retired type moves to COUNT first; its cached llm_query
    was written for the old result shape, so it is cleared and regenerated on
    the next run.
    """
    tables = set(inspector.get_table_names())
    if "monitor_types" not in tables:
        return

    retired = sa.bindparam("names", value=RETIRED_MONITOR_TYPES, expanding=True)

    if "monitors" in tables:
        count_id = conn.execute(
            sa.text("SELECT id FROM monitor_types WHERE name = 'COUNT'")
        ).scalar()
        # No COUNT row means an unseeded catalog — nothing to reassign onto,
        # and nothing to retire either.
        if count_id is None:
            return

        conn.execute(
            sa.text(
                "UPDATE monitors SET monitor_type_id = :count_id, llm_query = NULL "
                "WHERE monitor_type_id IN ("
                "    SELECT id FROM monitor_types WHERE name IN :names"
                ")"
            ).bindparams(retired, count_id=count_id)
        )

    conn.execute(
        sa.text("DELETE FROM monitor_types WHERE name IN :names").bindparams(retired)
    )


def _restore_monitor_types(conn, inspector) -> None:
    """Put the four withdrawn types back.

    Monitors reassigned to COUNT on the way up stay there — the original type
    is not recorded anywhere, so the move is one-way.
    """
    if "monitor_types" not in set(inspector.get_table_names()):
        return

    created_by = conn.execute(
        sa.text("SELECT id FROM users ORDER BY id LIMIT 1")
    ).scalar()
    if created_by is None:
        return

    for name, title, description, result_type, schema in RETIRED_TYPE_ROWS:
        exists = conn.execute(
            sa.text("SELECT 1 FROM monitor_types WHERE name = :name"), {"name": name}
        ).fetchone()
        if exists:
            continue
        conn.execute(
            sa.text("""
                INSERT INTO monitor_types
                    (name, title, description, result_type, configuration_schema,
                     is_active, created_by, created_at, updated_at)
                VALUES
                    (:name, :title, :description, :result_type, :schema,
                     true, :created_by, :now, :now)
            """),
            {
                "name": name,
                "title": title,
                "description": description,
                "result_type": result_type,
                "schema": schema,
                "created_by": created_by,
                "now": datetime.now(timezone.utc),
            },
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    _restore_monitor_types(conn, inspector)

    if TABLE not in set(inspector.get_table_names()):
        return

    is_sqlite = conn.dialect.name == "sqlite"
    cols = _columns(inspector)

    for col in ("created_at", "updated_at"):
        if col not in cols:
            op.add_column(
                TABLE,
                sa.Column(
                    col,
                    sa.DateTime(timezone=True),
                    nullable=False,
                    server_default=sa.func.now(),
                ),
            )

    # executed_at was the source of created_at on the way up.
    if "executed_at" in cols:
        conn.execute(sa.text(f"UPDATE {TABLE} SET created_at = executed_at"))

    if "result" in cols and not is_sqlite:
        op.alter_column(
            TABLE,
            "result",
            existing_type=sa.JSON(),
            type_=sa.Text(),
            existing_nullable=True,
            postgresql_using="result::text",
        )

    for col in ("execution_time_ms", "status", "executed_at"):
        if col in cols:
            op.drop_column(TABLE, col)
