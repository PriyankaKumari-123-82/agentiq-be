"""Durable PostgreSQL repository for the schema engine.

This is the authoritative persistence layer. All critical data flows through
here. Redis and in-memory caches are optional acceleration layers only.

Uses SQLModel sessions obtained from the existing db.py engine.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlmodel import Session, select, col

from ai_agent.schema_engine.store.pg_models import (
    SEChunk,
    SEColumn,
    SEDatabase,
    SEJoinUsageStats,
    SEQueryLog,
    SERelationship,
    SERetrievalAuditLog,
    SESchemaVersion,
    SETable,
    SETableCategory,
    SETableUsageStats,
)

log = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SchemaEngineRepository:
    """Synchronous repository wrapping SQLModel sessions.

    The caller is responsible for providing a session (via get_session dependency
    or direct creation). All methods are synchronous since the existing project
    uses sync sessions with psycopg2.
    """

    def __init__(self, session: Session):
        self._s = session

    # ═════════════════════════════════════════════════════════════════════
    # Database CRUD
    # ═════════════════════════════════════════════════════════════════════

    def get_or_create_database(self, server_name: str, database_name: str = "default") -> SEDatabase:
        stmt = select(SEDatabase).where(SEDatabase.server_name == server_name)
        db_row = self._s.exec(stmt).first()
        if db_row:
            return db_row
        db_row = SEDatabase(
            id=str(uuid.uuid4()),
            server_name=server_name,
            database_name=database_name,
        )
        self._s.add(db_row)
        self._s.flush()
        return db_row

    def get_database(self, server_name: str) -> Optional[SEDatabase]:
        stmt = select(SEDatabase).where(SEDatabase.server_name == server_name)
        return self._s.exec(stmt).first()

    def update_database_timestamps(
        self,
        db_id: str,
        *,
        extracted: bool = False,
        summarized: bool = False,
        indexed: bool = False,
        version_hash: Optional[str] = None,
        table_count: Optional[int] = None,
    ) -> None:
        db_row = self._s.get(SEDatabase, db_id)
        if not db_row:
            return
        now = _now()
        if extracted:
            db_row.last_extracted_at = now
        if summarized:
            db_row.last_summarized_at = now
        if indexed:
            db_row.last_indexed_at = now
        if version_hash is not None:
            db_row.version_hash = version_hash
        if table_count is not None:
            db_row.table_count = table_count
        db_row.updated_at = now
        self._s.add(db_row)
        self._s.flush()

    # ═════════════════════════════════════════════════════════════════════
    # Table CRUD
    # ═════════════════════════════════════════════════════════════════════

    def upsert_table(self, database_id: str, table_data: Dict[str, Any]) -> SETable:
        """Upsert a table by (database_id, schema_name, table_name)."""
        stmt = (
            select(SETable)
            .where(SETable.database_id == database_id)
            .where(SETable.schema_name == table_data.get("schema_name", "public"))
            .where(SETable.table_name == table_data["table_name"])
        )
        existing = self._s.exec(stmt).first()

        if existing:
            # Update mutable fields
            for key in ("row_count", "description", "table_comment", "summary",
                        "business_purpose", "domain", "semantic_keywords",
                        "join_hints", "content_hash", "search_text"):
                if key in table_data:
                    setattr(existing, key, table_data[key])
            existing.updated_at = _now()
            self._s.add(existing)
            self._s.flush()
            return existing

        row = SETable(
            id=str(uuid.uuid4()),
            database_id=database_id,
            schema_name=table_data.get("schema_name", "public"),
            table_name=table_data["table_name"],
            row_count=table_data.get("row_count"),
            description=table_data.get("description"),
            table_comment=table_data.get("table_comment"),
            summary=table_data.get("summary"),
            business_purpose=table_data.get("business_purpose"),
            domain=table_data.get("domain"),
            semantic_keywords=table_data.get("semantic_keywords"),
            join_hints=table_data.get("join_hints"),
            content_hash=table_data.get("content_hash", ""),
            search_text=table_data.get("search_text"),
        )
        self._s.add(row)
        self._s.flush()
        return row

    def get_tables_for_database(self, database_id: str) -> List[SETable]:
        stmt = select(SETable).where(SETable.database_id == database_id)
        return list(self._s.exec(stmt).all())

    def delete_tables_not_in(self, database_id: str, keep_table_ids: List[str]) -> int:
        """Delete tables that no longer exist in source. Returns count deleted."""
        stmt = (
            select(SETable)
            .where(SETable.database_id == database_id)
            .where(~col(SETable.id).in_(keep_table_ids))
        )
        removed = list(self._s.exec(stmt).all())
        for t in removed:
            self._s.delete(t)
        self._s.flush()
        return len(removed)

    # ═════════════════════════════════════════════════════════════════════
    # Columns
    # ═════════════════════════════════════════════════════════════════════

    def upsert_columns(self, table_id: str, columns: List[Dict[str, Any]]) -> None:
        """Replace all columns for a table (full refresh)."""
        # Delete existing
        stmt = select(SEColumn).where(SEColumn.table_id == table_id)
        existing = list(self._s.exec(stmt).all())
        for c in existing:
            self._s.delete(c)
        self._s.flush()

        # Insert new
        for i, col_data in enumerate(columns):
            row = SEColumn(
                id=str(uuid.uuid4()),
                table_id=table_id,
                name=col_data["name"],
                data_type=col_data.get("data_type", "unknown"),
                nullable=col_data.get("nullable", True),
                description=col_data.get("description"),
                is_primary_key=col_data.get("is_primary_key", False),
                is_foreign_key=col_data.get("is_foreign_key", False),
                ordinal_position=i,
            )
            self._s.add(row)
        self._s.flush()

    # ═════════════════════════════════════════════════════════════════════
    # Relationships (FK graph)
    # ═════════════════════════════════════════════════════════════════════

    def replace_relationships(
        self,
        database_id: str,
        relationships: List[Dict[str, Any]],
    ) -> None:
        """Full-replace all FK relationships for a database."""
        # Delete existing
        stmt = select(SERelationship).where(SERelationship.database_id == database_id)
        existing = list(self._s.exec(stmt).all())
        for r in existing:
            self._s.delete(r)
        self._s.flush()

        for rel in relationships:
            row = SERelationship(
                id=str(uuid.uuid4()),
                database_id=database_id,
                source_table_id=rel["source_table_id"],
                target_table_id=rel["target_table_id"],
                source_column=rel["source_column"],
                target_column=rel["target_column"],
                constraint_name=rel.get("constraint_name"),
            )
            self._s.add(row)
        self._s.flush()

    def get_relationships(self, database_id: str) -> List[SERelationship]:
        stmt = select(SERelationship).where(SERelationship.database_id == database_id)
        return list(self._s.exec(stmt).all())

    # ═════════════════════════════════════════════════════════════════════
    # Chunks
    # ═════════════════════════════════════════════════════════════════════

    def upsert_chunk(self, chunk_data: Dict[str, Any]) -> SEChunk:
        """Upsert a chunk by table_id."""
        stmt = select(SEChunk).where(SEChunk.table_id == chunk_data["table_id"])
        existing = self._s.exec(stmt).first()

        if existing:
            for key in ("search_text", "summary_text", "compact_ddl", "domain",
                        "column_names_json", "primary_keys_json",
                        "related_tables_json", "row_count", "foreign_key_count",
                        "embedding_id", "embedding_version"):
                if key in chunk_data:
                    setattr(existing, key, chunk_data[key])
            existing.updated_at = _now()
            self._s.add(existing)
            self._s.flush()
            return existing

        row = SEChunk(
            id=chunk_data.get("id", str(uuid.uuid4())),
            database_id=chunk_data["database_id"],
            table_id=chunk_data["table_id"],
            server_name=chunk_data["server_name"],
            table_fqn=chunk_data["table_fqn"],
            table_name=chunk_data["table_name"],
            schema_name=chunk_data.get("schema_name", "public"),
            search_text=chunk_data.get("search_text", ""),
            summary_text=chunk_data.get("summary_text", ""),
            compact_ddl=chunk_data.get("compact_ddl", ""),
            domain=chunk_data.get("domain"),
            column_names_json=chunk_data.get("column_names_json"),
            primary_keys_json=chunk_data.get("primary_keys_json"),
            related_tables_json=chunk_data.get("related_tables_json"),
            row_count=chunk_data.get("row_count"),
            foreign_key_count=chunk_data.get("foreign_key_count", 0),
            embedding_id=chunk_data.get("embedding_id"),
            embedding_version=chunk_data.get("embedding_version"),
        )
        self._s.add(row)
        self._s.flush()
        return row

    def get_chunks_for_database(self, database_id: str) -> List[SEChunk]:
        stmt = select(SEChunk).where(SEChunk.database_id == database_id)
        return list(self._s.exec(stmt).all())

    def get_chunks_by_server(self, server_name: str) -> List[SEChunk]:
        stmt = select(SEChunk).where(SEChunk.server_name == server_name)
        return list(self._s.exec(stmt).all())

    def fulltext_search_chunks(
        self, server_name: str, query: str, limit: int = 20,
    ) -> List[Tuple[SEChunk, float]]:
        """PostgreSQL full-text search on chunk search_text. Returns (chunk, rank)."""
        from sqlalchemy import text as sa_text
        sql = sa_text("""
            SELECT se_chunks.*,
                   ts_rank(
                       to_tsvector('english', coalesce(se_chunks.search_text, '')),
                       plainto_tsquery('english', :query)
                   ) AS rank
            FROM se_chunks
            WHERE se_chunks.server_name = :server_name
              AND to_tsvector('english', coalesce(se_chunks.search_text, ''))
                  @@ plainto_tsquery('english', :query)
            ORDER BY rank DESC
            LIMIT :limit
        """)
        result = self._s.execute(sql, {
            "server_name": server_name,
            "query": query,
            "limit": limit,
        })
        rows = []
        for row in result:
            chunk = self._s.get(SEChunk, row.id)
            if chunk:
                rows.append((chunk, row.rank))
        return rows

    # ═════════════════════════════════════════════════════════════════════
    # Schema Classifier Categories + Sample Questions
    # ═════════════════════════════════════════════════════════════════════

    def replace_classifier_categories(
        self,
        database_id: str,
        categories: List[Dict[str, Any]],
    ) -> List[SETableCategory]:
        """Full-replace all table categories (with their questions) for a database."""
        existing = list(self._s.exec(
            select(SETableCategory).where(SETableCategory.database_id == database_id)
        ).all())
        for cat_row in existing:
            self._s.delete(cat_row)
        self._s.flush()

        rows = []
        for i, cat_data in enumerate(categories):
            row = SETableCategory(
                id=str(uuid.uuid4()),
                database_id=database_id,
                category_name=cat_data["category_name"],
                description=cat_data.get("description"),
                priority=cat_data.get("priority", i),
                table_names_json=json.dumps(cat_data.get("table_names", [])),
                questions=json.dumps(cat_data.get("questions", [])),
            )
            self._s.add(row)
            rows.append(row)
        self._s.flush()
        return rows

    def get_classifier_categories_for_database(self, database_id: str) -> List[SETableCategory]:
        """Categories ordered by priority (most important first)."""
        stmt = (
            select(SETableCategory)
            .where(SETableCategory.database_id == database_id)
            .order_by(col(SETableCategory.priority))
        )
        return list(self._s.exec(stmt).all())

    # ═════════════════════════════════════════════════════════════════════
    # Query Log
    # ═════════════════════════════════════════════════════════════════════

    def record_query(
        self,
        database_id: str,
        server_name: str,
        user_query: str,
        generated_sql: Optional[str] = None,
        tables_used: Optional[List[str]] = None,
        joins_used: Optional[List[str]] = None,
        retrieval_confidence: float = 0.0,
        execution_success: bool = True,
        latency_ms: int = 0,
    ) -> SEQueryLog:
        row = SEQueryLog(
            id=str(uuid.uuid4()),
            database_id=database_id,
            server_name=server_name,
            user_query=user_query,
            generated_sql=generated_sql,
            tables_used_json=json.dumps(tables_used) if tables_used else None,
            joins_used_json=json.dumps(joins_used) if joins_used else None,
            retrieval_confidence=retrieval_confidence,
            execution_success=execution_success,
            latency_ms=latency_ms,
        )
        self._s.add(row)
        self._s.flush()

        # Update usage stats
        if tables_used:
            self._update_table_usage(database_id, tables_used, execution_success)
        if joins_used:
            self._update_join_usage(database_id, joins_used)

        return row

    def _update_table_usage(
        self, database_id: str, tables: List[str], success: bool,
    ) -> None:
        now = _now()
        for table_name in tables:
            stmt = (
                select(SETableUsageStats)
                .where(SETableUsageStats.database_id == database_id)
                .where(SETableUsageStats.table_name == table_name)
            )
            existing = self._s.exec(stmt).first()
            if existing:
                existing.query_count += 1
                if success:
                    existing.success_count += 1
                existing.last_used_at = now
                existing.updated_at = now
                self._s.add(existing)
            else:
                self._s.add(SETableUsageStats(
                    id=str(uuid.uuid4()),
                    database_id=database_id,
                    table_name=table_name,
                    query_count=1,
                    success_count=1 if success else 0,
                    last_used_at=now,
                ))
        self._s.flush()

    def _update_join_usage(self, database_id: str, joins: List[str]) -> None:
        now = _now()
        for pattern in joins:
            stmt = (
                select(SEJoinUsageStats)
                .where(SEJoinUsageStats.database_id == database_id)
                .where(SEJoinUsageStats.join_pattern == pattern)
            )
            existing = self._s.exec(stmt).first()
            if existing:
                existing.query_count += 1
                existing.last_used_at = now
                existing.updated_at = now
                self._s.add(existing)
            else:
                self._s.add(SEJoinUsageStats(
                    id=str(uuid.uuid4()),
                    database_id=database_id,
                    join_pattern=pattern,
                    query_count=1,
                    last_used_at=now,
                ))
        self._s.flush()

    def get_table_usage_boost(self, database_id: str, table_name: str) -> float:
        """Return a 0-1 boost factor based on query frequency."""
        stmt = (
            select(SETableUsageStats)
            .where(SETableUsageStats.database_id == database_id)
            .where(SETableUsageStats.table_name == table_name)
        )
        row = self._s.exec(stmt).first()
        if not row or row.query_count == 0:
            return 0.0

        # Get max to normalise
        from sqlalchemy import func
        max_stmt = (
            select(func.max(SETableUsageStats.query_count))
            .where(SETableUsageStats.database_id == database_id)
        )
        max_count = self._s.exec(max_stmt).first() or 1
        return min(row.query_count / max(max_count, 1), 1.0)

    def get_top_tables(self, database_id: str, limit: int = 10) -> List[Tuple[str, int]]:
        stmt = (
            select(SETableUsageStats.table_name, SETableUsageStats.query_count)
            .where(SETableUsageStats.database_id == database_id)
            .order_by(col(SETableUsageStats.query_count).desc())
            .limit(limit)
        )
        return [(r[0], r[1]) for r in self._s.exec(stmt).all()]

    # ═════════════════════════════════════════════════════════════════════
    # Schema Versions
    # ═════════════════════════════════════════════════════════════════════

    def record_version(
        self,
        database_id: str,
        version_hash: str,
        table_count: int = 0,
        tables_added: int = 0,
        tables_removed: int = 0,
        tables_modified: int = 0,
        embeddings_regenerated: int = 0,
    ) -> SESchemaVersion:
        row = SESchemaVersion(
            id=str(uuid.uuid4()),
            database_id=database_id,
            version_hash=version_hash,
            table_count=table_count,
            tables_added=tables_added,
            tables_removed=tables_removed,
            tables_modified=tables_modified,
            embeddings_regenerated=embeddings_regenerated,
        )
        self._s.add(row)
        self._s.flush()
        return row

    def get_latest_version(self, database_id: str) -> Optional[SESchemaVersion]:
        stmt = (
            select(SESchemaVersion)
            .where(SESchemaVersion.database_id == database_id)
            .order_by(col(SESchemaVersion.created_at).desc())
            .limit(1)
        )
        return self._s.exec(stmt).first()

    # ═════════════════════════════════════════════════════════════════════
    # Retrieval Audit
    # ═════════════════════════════════════════════════════════════════════

    def record_retrieval(self, audit_data: Dict[str, Any]) -> SERetrievalAuditLog:
        row = SERetrievalAuditLog(
            id=str(uuid.uuid4()),
            database_id=audit_data["database_id"],
            user_query=audit_data["user_query"],
            tables_retrieved_json=audit_data.get("tables_retrieved_json"),
            retrieval_method=audit_data.get("retrieval_method", "hybrid"),
            bm25_score=audit_data.get("bm25_score", 0.0),
            vector_score=audit_data.get("vector_score", 0.0),
            graph_score=audit_data.get("graph_score", 0.0),
            total_chunks=audit_data.get("total_chunks", 0),
            token_budget_used=audit_data.get("token_budget_used", 0),
            latency_ms=audit_data.get("latency_ms", 0),
        )
        self._s.add(row)
        self._s.flush()
        return row

    # ═════════════════════════════════════════════════════════════════════
    # Commit
    # ═════════════════════════════════════════════════════════════════════

    def commit(self) -> None:
        self._s.commit()

    def rollback(self) -> None:
        self._s.rollback()
