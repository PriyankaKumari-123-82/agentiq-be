"""Query Assistant history — the raw natural-language questions a user has run.

Imported by ai_agent/models.py so Alembic autogenerate picks it up.

Deliberately independent of Monitor: a monitor's natural_query only ever holds
its current text (refining overwrites it, discarding deletes it), so it can't
answer "what did I ask before" on its own. This table logs the question itself,
at the moment it's submitted, regardless of what happens to the monitor after.
"""
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlmodel import Field, SQLModel


class QueryHistoryEntry(SQLModel, table=True):
    """One natural-language question a user submitted in the Query Assistant."""
    __tablename__ = "query_history_entries"

    id: int | None = Field(default=None, primary_key=True)
    created_by: int = Field(foreign_key="users.id", nullable=False, index=True)
    organization_id: int | None = Field(
        default=None, foreign_key="organizations.id", index=True,
    )
    query: str = Field(sa_column=sa.Column(sa.String(), nullable=False))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), index=True,
    )
