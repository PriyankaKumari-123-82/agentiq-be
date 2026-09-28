"""Durable query-log-based ranking improvements.

Records which tables/columns/joins are actually used in executed queries and
feeds that signal back into retrieval scoring.

Storage: PostgreSQL (authoritative) + in-memory cache for hot-path lookups.
All data survives process restarts.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from ai_agent.schema_engine.config import get_config

log = logging.getLogger(__name__)


@dataclass
class QueryLogEntry:
    """One recorded query execution."""
    query_text: str
    tables: List[str]
    columns: List[str] = field(default_factory=list)
    joins: List[str] = field(default_factory=list)
    server_name: str = ""
    generated_sql: str = ""
    retrieval_confidence: float = 0.0
    latency_ms: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = True


class QueryLog:
    """Durable query log: PostgreSQL authority + memory cache for boost lookups."""

    def __init__(self):
        self._cfg = get_config()
        # In-memory cache for hot-path boost lookups (loaded from PG on first access)
        self._table_freq_cache: Dict[str, Counter] = {}
        self._join_freq_cache: Dict[str, Counter] = {}
        self._cache_loaded: Dict[str, bool] = {}
        self._pg_available = False
        self._init_pg()

    def _init_pg(self):
        try:
            from ai_agent.core.db import engine
            self._pg_available = engine is not None
        except Exception:
            self._pg_available = False

    @property
    def enabled(self) -> bool:
        return self._cfg.query_log_enabled

    def record(self, entry: QueryLogEntry) -> None:
        """Record a query execution to PostgreSQL and update caches."""
        if not self.enabled:
            return

        srv = entry.server_name

        # Write to PostgreSQL (authoritative)
        if self._pg_available:
            try:
                self._record_to_pg(entry)
            except Exception as e:
                log.warning("QueryLog: PG write failed: %s", e)

        # Update in-memory cache
        if srv not in self._table_freq_cache:
            self._table_freq_cache[srv] = Counter()
            self._join_freq_cache[srv] = Counter()

        for t in entry.tables:
            self._table_freq_cache[srv][t.lower()] += 1
        for j in entry.joins:
            self._join_freq_cache[srv][j.lower()] += 1

    def table_boost(self, server_name: str, table_name: str) -> float:
        """Return a [0,1] boost factor based on how often this table is queried."""
        self._ensure_cache(server_name)
        freq = self._table_freq_cache.get(server_name, Counter())
        count = freq.get(table_name.lower(), 0)
        if count == 0:
            # Try direct PG lookup as fallback
            if self._pg_available:
                try:
                    return self._pg_table_boost(server_name, table_name)
                except Exception:
                    pass
            return 0.0
        max_count = freq.most_common(1)[0][1] if freq else 1
        return min(count / max(max_count, 1), 1.0)

    def frequently_joined(self, server_name: str, top_n: int = 10) -> List[str]:
        """Return the most common join patterns for a server."""
        self._ensure_cache(server_name)
        freq = self._join_freq_cache.get(server_name, Counter())
        return [j for j, _ in freq.most_common(top_n)]

    def stats(self, server_name: str) -> Dict:
        self._ensure_cache(server_name)
        freq = self._table_freq_cache.get(server_name, Counter())
        return {
            "total_queries": sum(freq.values()),
            "unique_tables": len(freq),
            "top_tables": freq.most_common(5),
            "pg_available": self._pg_available,
        }

    # ═════════════════════════════════════════════════════════════════════
    # PostgreSQL helpers
    # ═════════════════════════════════════════════════════════════════════

    def _get_pg_session(self):
        from ai_agent.core.db import engine
        from sqlmodel import Session
        return Session(engine)

    def _get_database_id(self, session, server_name: str) -> Optional[str]:
        from ai_agent.schema_engine.store.pg_repository import SchemaEngineRepository
        repo = SchemaEngineRepository(session)
        db_row = repo.get_database(server_name)
        return db_row.id if db_row else None

    def _record_to_pg(self, entry: QueryLogEntry) -> None:
        from ai_agent.schema_engine.store.pg_repository import SchemaEngineRepository
        session = self._get_pg_session()
        try:
            repo = SchemaEngineRepository(session)
            db_id = self._get_database_id(session, entry.server_name)
            if not db_id:
                log.debug("QueryLog: no database_id for '%s'", entry.server_name)
                return
            repo.record_query(
                database_id=db_id,
                server_name=entry.server_name,
                user_query=entry.query_text,
                generated_sql=entry.generated_sql or None,
                tables_used=entry.tables or None,
                joins_used=entry.joins or None,
                retrieval_confidence=entry.retrieval_confidence,
                execution_success=entry.success,
                latency_ms=entry.latency_ms,
            )
            repo.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _pg_table_boost(self, server_name: str, table_name: str) -> float:
        from ai_agent.schema_engine.store.pg_repository import SchemaEngineRepository
        session = self._get_pg_session()
        try:
            repo = SchemaEngineRepository(session)
            db_id = self._get_database_id(session, server_name)
            if not db_id:
                return 0.0
            return repo.get_table_usage_boost(db_id, table_name)
        finally:
            session.close()

    def _ensure_cache(self, server_name: str) -> None:
        """Load usage stats from PG into memory cache on first access per server."""
        if server_name in self._cache_loaded:
            return
        self._cache_loaded[server_name] = True

        if not self._pg_available:
            return

        try:
            self._load_cache_from_pg(server_name)
        except Exception as e:
            log.warning("QueryLog: cache warm from PG failed: %s", e)

    def _load_cache_from_pg(self, server_name: str) -> None:
        from ai_agent.schema_engine.store.pg_repository import SchemaEngineRepository
        from ai_agent.schema_engine.store.pg_models import (
            SEJoinUsageStats,
            SETableUsageStats,
        )
        from sqlmodel import select

        session = self._get_pg_session()
        try:
            repo = SchemaEngineRepository(session)
            db_id = self._get_database_id(session, server_name)
            if not db_id:
                return

            # Table usage
            table_counter = Counter()
            table_rows = session.exec(
                select(SETableUsageStats)
                .where(SETableUsageStats.database_id == db_id)
            ).all()
            for row in table_rows:
                table_counter[row.table_name.lower()] = row.query_count
            self._table_freq_cache[server_name] = table_counter

            # Join usage
            join_counter = Counter()
            join_rows = session.exec(
                select(SEJoinUsageStats)
                .where(SEJoinUsageStats.database_id == db_id)
            ).all()
            for row in join_rows:
                join_counter[row.join_pattern.lower()] = row.query_count
            self._join_freq_cache[server_name] = join_counter

            log.info(
                "QueryLog: loaded %d table stats, %d join stats from PG for '%s'",
                len(table_counter), len(join_counter), server_name,
            )
        finally:
            session.close()


# ── Singleton ────────────────────────────────────────────────────────────

_query_log: Optional[QueryLog] = None


def get_query_log() -> QueryLog:
    global _query_log
    if _query_log is None:
        _query_log = QueryLog()
    return _query_log
