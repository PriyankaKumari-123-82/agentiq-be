"""Monitor domain models.

Imported by ai_agent/models.py so Alembic autogenerate picks them up.

Tables (plural names match the real PostgreSQL tables)
------
monitor_types   - Read-only catalog seeded by migration
monitors        - User-defined monitor cards
monitor_results - One row per execution run
"""
from datetime import datetime, timezone
from enum import Enum

import sqlalchemy as sa
from sqlmodel import Field, SQLModel


# ── Enums ─────────────────────────────────────────────────────────────────────

class MonitorResultType(str, Enum):
    NUMBER  = "NUMBER"
    BOOLEAN = "BOOLEAN"
    TEXT    = "TEXT"
    JSON    = "JSON"


class MonitorResultStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED  = "FAILED"


# ── monitor_types ─────────────────────────────────────────────────────────────

class MonitorType(SQLModel, table=True):
    """Read-only catalog of monitor types. Seeded by migration."""
    __tablename__ = "monitor_types"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(
        sa_column=sa.Column(sa.String(), nullable=False),
        description="Unique name, e.g. 'COUNT', 'LIST', 'AI_QUERY'",
    )
    title: str = Field(
        sa_column=sa.Column(sa.String(), nullable=False),
        description="Human-readable display title shown in the UI",
    )
    description: str | None = Field(
        default=None, sa_column=sa.Column(sa.String(), nullable=True),
    )
    result_type: str = Field(
        sa_column=sa.Column(sa.String(), nullable=False),
        description="NUMBER | BOOLEAN | TEXT | JSON — expected shape of results",
    )
    configuration_schema: str | None = Field(
        default=None, sa_column=sa.Column(sa.String(), nullable=True),
        description="JSON string the LLM uses to format its response",
    )
    is_active: bool = Field(default=True)
    created_by: int = Field(foreign_key="users.id", nullable=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )


# ── monitors ──────────────────────────────────────────────────────────────────

class Monitor(SQLModel, table=True):
    """A user-defined monitor card: natural query + schedule.

    is_active = False  →  monitor is paused (scheduler skips it).
    deleted_at IS NOT NULL  →  monitor is soft-deleted (hidden from all queries).
    Both flags are set together on delete so you know WHEN it was removed.

    NOTE: ``agent_id`` was removed — monitors are scoped to organizations, not
    individual agents.  MCP tools are fetched organization-wide at runtime.
    """
    __tablename__ = "monitors"

    id: int | None = Field(default=None, primary_key=True)
    organization_id: int | None = Field(
        default=None, foreign_key="organizations.id", index=True,
    )
    monitor_type_id: int = Field(
        foreign_key="monitor_types.id", index=True, nullable=False,
    )
    name: str = Field(
        sa_column=sa.Column(sa.String(200), nullable=False),
    )
    description: str | None = Field(
        default=None, sa_column=sa.Column(sa.String(), nullable=True),
    )
    natural_query: str = Field(
        sa_column=sa.Column(sa.String(), nullable=False),
        description="User's original natural-language question — always preserved",
    )
    llm_query: str | None = Field(
        default=None, sa_column=sa.Column(sa.String(), nullable=True),
        description="LLM-generated query (SQL / tool-call JSON). NULL until the LLM builds it.",
    )
    followup_question: str | None = Field(
        default=None, sa_column=sa.Column(sa.String(), nullable=True),
        description="Follow-up question the LLM asks the user before generating llm_query.",
    )
    scheduler_frequency: int | None = Field(
        default=3600,
        description="Seconds between scheduled runs. Default 3600 (1 hour).",
    )
    is_active: bool = Field(
        default=True, index=True,
        description="False = paused or deleted. Scheduler skips inactive monitors.",
    )
    deleted_at: datetime | None = Field(
        default=None,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True),
        description="Set when soft-deleted. NULL = not deleted.",
    )
    created_by: int = Field(
        foreign_key="users.id", index=True, nullable=False,
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )


# ── monitor_results ───────────────────────────────────────────────────────────

class MonitorResult(SQLModel, table=True):
    """One row per monitor execution run.

    result is a JSON column whose content shape is determined by the
    monitor type's result_type:
      NUMBER  → {"value": 42}
      BOOLEAN → {"value": true}
      TEXT    → {"value": "free text answer"}
      JSON    → {"rows": [...], "columns": [...]}

    Column layout matches the actual DB table exactly.
    """
    __tablename__ = "monitor_results"

    id: int | None = Field(default=None, primary_key=True)
    monitor_id: int = Field(
        foreign_key="monitors.id", index=True, nullable=False,
    )
    executed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this run was executed",
    )
    status: str = Field(default="SUCCESS", description="SUCCESS | FAILED")
    execution_time_ms: int | None = Field(default=None)
    result_type: str = Field(description="NUMBER | BOOLEAN | TEXT | JSON")
    result: str | None = Field(
        default=None,
        sa_column=sa.Column(sa.JSON(), nullable=True),
        description="Result payload — stored as JSON in DB",
    )
    error_message: str | None = Field(
        default=None, sa_column=sa.Column(sa.String(), nullable=True),
    )
    is_active: bool = Field(default=True)
