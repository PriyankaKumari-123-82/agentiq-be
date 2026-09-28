"""temperature UI enforcement — consolidated migration

Revision ID: e9b8c7d6e5f4
Revises: e2a9d4c7b1f3
Create Date: 2026-08-13 00:00:00.000000

Consolidates all DB changes for temperature UI enforcement (move temperature constraints
and model capability flags to the UI layer) into a single idempotent migration:

  1. ai_agent_settings — add CHECK constraint ensuring llm_temperature is in [0.0, 2.0].
     The UI now enforces provider-ceiling and per-model limits before submission,
     but the DB constraint remains as a hard safety net.

  2. llm_usage_events — add nullable `temperature` column so each LLM call can
     record the temperature value actually sent to the provider.

  3. model_provider — add boolean `supports_temperature` (default TRUE).
     Set to FALSE for known OpenAI reasoning models (o1-preview, o1-mini, o3-mini, o1)
     that reject the parameter entirely. The UI reads this flag to disable the
     temperature slider instead of relying on silent backend retries.

All three operations are guarded against re-execution so the migration is safe
to run on a DB that already has some or all of these objects in place.
"""
import sqlalchemy as sa
from alembic import op

revision = "e9b8c7d6e5f4"
down_revision = "12bd6c43c0b4"
branch_labels = None
depends_on = None

# ── Constants ─────────────────────────────────────────────────────────────────

CONSTRAINT_NAME = "chk_agent_llm_temperature_range"
AGENT_SETTINGS_TABLE = "ai_agent_settings"
USAGE_EVENTS_TABLE = "llm_usage_events"
MODEL_PROVIDER_TABLE = "model_provider"

# OpenAI reasoning models that reject the `temperature` parameter outright.
# Add new model IDs here when they are released — no code changes needed.
NO_TEMPERATURE_MODELS = (
    "o1",
    "o1-preview",
    "o1-mini",
    "o3-mini",
    "o1-2024-12-17",
    "o1-mini-2024-09-12",
    "o1-preview-2024-09-12",
    "o3-mini-2025-01-31",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _table_exists(inspector, table: str) -> bool:
    return table in inspector.get_table_names()


def _col_exists(inspector, table: str, col: str) -> bool:
    return col in {c["name"] for c in inspector.get_columns(table)}


def _constraint_exists(inspector, table: str, name: str) -> bool:
    return name in {c["name"] for c in inspector.get_check_constraints(table)}


# ── Upgrade ───────────────────────────────────────────────────────────────────

def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    # 1. CHECK constraint on ai_agent_settings.llm_temperature
    if _table_exists(inspector, AGENT_SETTINGS_TABLE):
        if not _constraint_exists(inspector, AGENT_SETTINGS_TABLE, CONSTRAINT_NAME):
            op.create_check_constraint(
                CONSTRAINT_NAME,
                AGENT_SETTINGS_TABLE,
                "llm_temperature >= 0.0 AND llm_temperature <= 2.0",
            )

    # 2. temperature column on llm_usage_events
    if _table_exists(inspector, USAGE_EVENTS_TABLE):
        if not _col_exists(inspector, USAGE_EVENTS_TABLE, "temperature"):
            op.add_column(
                USAGE_EVENTS_TABLE,
                sa.Column("temperature", sa.Float(), nullable=True),
            )

    # 3. supports_temperature column on model_provider
    if _table_exists(inspector, MODEL_PROVIDER_TABLE):
        if not _col_exists(inspector, MODEL_PROVIDER_TABLE, "supports_temperature"):
            op.add_column(
                MODEL_PROVIDER_TABLE,
                sa.Column(
                    "supports_temperature",
                    sa.Boolean(),
                    nullable=False,
                    server_default=sa.true(),
                ),
            )

        # Backfill: mark known no-temperature models regardless of whether the
        # column was just created or already existed from a partial run.
        placeholders = ", ".join(f"'{m}'" for m in NO_TEMPERATURE_MODELS)
        conn.execute(
            sa.text(
                f"UPDATE {MODEL_PROVIDER_TABLE} "
                f"SET supports_temperature = FALSE "
                f"WHERE model_name IN ({placeholders})"
            )
        )


# ── Downgrade ─────────────────────────────────────────────────────────────────

def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if _table_exists(inspector, MODEL_PROVIDER_TABLE):
        if _col_exists(inspector, MODEL_PROVIDER_TABLE, "supports_temperature"):
            op.drop_column(MODEL_PROVIDER_TABLE, "supports_temperature")

    if _table_exists(inspector, USAGE_EVENTS_TABLE):
        if _col_exists(inspector, USAGE_EVENTS_TABLE, "temperature"):
            op.drop_column(USAGE_EVENTS_TABLE, "temperature")

    if _table_exists(inspector, AGENT_SETTINGS_TABLE):
        if _constraint_exists(inspector, AGENT_SETTINGS_TABLE, CONSTRAINT_NAME):
            op.drop_constraint(CONSTRAINT_NAME, AGENT_SETTINGS_TABLE, type_="check")
