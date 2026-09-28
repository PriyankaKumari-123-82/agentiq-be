"""Durable metadata store with tiered caching.

Storage hierarchy (authority → cache):
  1. PostgreSQL  — authoritative, durable
  2. Redis       — distributed cache, persistence-validated
  3. In-memory   — local hot cache, non-authoritative

The system functions correctly with Redis or in-memory caches unavailable.
PostgreSQL is the sole source of truth.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ai_agent.schema_engine.config import get_config
from ai_agent.schema_engine.models import DatabaseMetadata, SchemaChunk, TableMetadata

log = logging.getLogger(__name__)


class MetadataStore:
    """Tiered metadata store: PostgreSQL → Redis → memory."""

    def __init__(self):
        self._cfg = get_config()
        self._mem: Dict[str, _CacheEntry] = {}
        self._redis_cache = None
        self._pg_available = False
        self._init_layers()

    def _init_layers(self):
        # L2: Redis cache
        try:
            from ai_agent.schema_engine.store.redis_cache import get_redis_cache
            self._redis_cache = get_redis_cache()
        except Exception as e:
            log.info("MetadataStore: Redis cache unavailable: %s", e)

        # L0: PostgreSQL (verify engine is accessible)
        try:
            from ai_agent.core.db import engine
            self._pg_available = engine is not None
            log.info("MetadataStore: PostgreSQL available")
        except Exception as e:
            log.warning("MetadataStore: PostgreSQL unavailable: %s", e)

    # ── Key helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(server_name: str, db_name: str, kind: str) -> str:
        return f"meta:{kind}:{server_name}:{db_name}"

    # ═════════════════════════════════════════════════════════════════════
    # DatabaseMetadata — read path: L1 → L2 → L0
    # ═════════════════════════════════════════════════════════════════════

    async def get_metadata(
        self, server_name: str, db_name: str = "default",
    ) -> Optional[DatabaseMetadata]:
        key = self._cache_key(server_name, db_name, "meta")

        # L1: in-memory
        entry = self._mem.get(key)
        if entry and not entry.expired(self._cfg.metadata_cache_ttl_hours):
            return entry.value

        # L2: Redis
        if self._redis_cache and self._redis_cache.is_connected:
            raw = self._redis_cache.get(key)
            if raw:
                try:
                    db_meta = self._deserialise_metadata(json.loads(raw))
                    self._mem[key] = _CacheEntry(db_meta)
                    return db_meta
                except Exception as e:
                    log.warning("MetadataStore: Redis deserialise error: %s", e)

        # L0: PostgreSQL
        if self._pg_available:
            try:
                db_meta = self._load_metadata_from_pg(server_name)
                if db_meta:
                    self._mem[key] = _CacheEntry(db_meta)
                    self._cache_to_redis(key, db_meta)
                    return db_meta
            except Exception as e:
                log.warning("MetadataStore: PG load error: %s", e)

        return None

    # ═════════════════════════════════════════════════════════════════════
    # DatabaseMetadata — write path: L0 → L2 → L1
    # ═════════════════════════════════════════════════════════════════════

    async def put_metadata(
        self, meta: DatabaseMetadata, db_name: str = "default",
    ) -> None:
        key = self._cache_key(meta.server_name, db_name, "meta")

        # L0: PostgreSQL (authoritative write)
        if self._pg_available:
            try:
                self._save_metadata_to_pg(meta)
            except Exception as e:
                log.error("MetadataStore: PG write FAILED: %s", e, exc_info=True)

        # L2: Redis cache
        self._cache_to_redis(key, meta)

        # L1: memory cache
        self._mem[key] = _CacheEntry(meta)

    # ═════════════════════════════════════════════════════════════════════
    # Schema Chunks — read path: L1 → L0
    # ═════════════════════════════════════════════════════════════════════

    async def get_chunks(
        self, server_name: str, db_name: str = "default",
    ) -> Optional[List[SchemaChunk]]:
        key = self._cache_key(server_name, db_name, "chunks")

        # L1: memory
        entry = self._mem.get(key)
        if entry and not entry.expired(self._cfg.metadata_cache_ttl_hours):
            return entry.value

        # L0: PostgreSQL
        if self._pg_available:
            try:
                chunks = self._load_chunks_from_pg(server_name)
                if chunks:
                    self._mem[key] = _CacheEntry(chunks)
                    return chunks
            except Exception as e:
                log.warning("MetadataStore: PG chunk load error: %s", e)

        return None

    # ═════════════════════════════════════════════════════════════════════
    # Schema Chunks — write path: L0 → L1
    # ═════════════════════════════════════════════════════════════════════

    async def put_chunks(
        self, server_name: str, chunks: List[SchemaChunk], db_name: str = "default",
    ) -> None:
        key = self._cache_key(server_name, db_name, "chunks")

        # L0: PostgreSQL
        if self._pg_available:
            try:
                self._save_chunks_to_pg(server_name, chunks)
            except Exception as e:
                log.error("MetadataStore: PG chunk write FAILED: %s", e, exc_info=True)

        # L1: memory
        self._mem[key] = _CacheEntry(chunks)

    # ═════════════════════════════════════════════════════════════════════
    # Invalidation — all layers
    # ═════════════════════════════════════════════════════════════════════

    def invalidate(self, server_name: str, db_name: str = "default") -> None:
        for kind in ("meta", "chunks"):
            key = self._cache_key(server_name, db_name, kind)
            self._mem.pop(key, None)
        if self._redis_cache and self._redis_cache.is_connected:
            self._redis_cache.invalidate_server_cache(server_name)

    # ═════════════════════════════════════════════════════════════════════
    # PostgreSQL read/write helpers
    # ═════════════════════════════════════════════════════════════════════

    def _get_pg_session(self):
        from ai_agent.core.db import engine
        from sqlmodel import Session
        return Session(engine)

    def _load_metadata_from_pg(self, server_name: str) -> Optional[DatabaseMetadata]:
        """Reconstruct DatabaseMetadata from PostgreSQL tables."""
        from ai_agent.schema_engine.store.pg_repository import SchemaEngineRepository

        session = self._get_pg_session()
        try:
            repo = SchemaEngineRepository(session)
            db_row = repo.get_database(server_name)
            if not db_row:
                return None

            table_rows = repo.get_tables_for_database(db_row.id)
            tables: Dict[str, TableMetadata] = {}

            for t in table_rows:
                # Load columns
                from sqlmodel import select
                from ai_agent.schema_engine.store.pg_models import SEColumn
                col_rows = list(session.exec(
                    select(SEColumn)
                    .where(SEColumn.table_id == t.id)
                    .order_by(SEColumn.ordinal_position)
                ).all())

                from ai_agent.schema_engine.models import ColumnMetadata, ForeignKeyMetadata
                columns = [
                    ColumnMetadata(
                        name=c.name,
                        data_type=c.data_type,
                        nullable=c.nullable,
                        description=c.description,
                        is_primary_key=c.is_primary_key,
                    )
                    for c in col_rows
                ]

                # Load FKs where this table is the source
                from ai_agent.schema_engine.store.pg_models import SERelationship
                fk_rows = list(session.exec(
                    select(SERelationship)
                    .where(SERelationship.source_table_id == t.id)
                ).all())

                foreign_keys = []
                for fk in fk_rows:
                    # Look up the target table name
                    from ai_agent.schema_engine.store.pg_models import SETable
                    target = session.get(SETable, fk.target_table_id)
                    target_table_name = target.table_name if target else "unknown"
                    foreign_keys.append(ForeignKeyMetadata(
                        source_column=fk.source_column,
                        target_table=target_table_name,
                        target_column=fk.target_column,
                    ))

                fqn = f"{t.schema_name}.{t.table_name}"
                tm = TableMetadata(
                    schema_name=t.schema_name,
                    table_name=t.table_name,
                    columns=columns,
                    primary_keys=[c.name for c in col_rows if c.is_primary_key],
                    foreign_keys=foreign_keys,
                    row_count=t.row_count,
                    table_comment=t.table_comment,
                    description=t.description,
                    summary=t.summary,
                    business_purpose=t.business_purpose,
                    domain=t.domain or "general",
                    semantic_keywords=json.loads(t.semantic_keywords) if t.semantic_keywords else [],
                    join_hints=json.loads(t.join_hints) if t.join_hints else [],
                )
                tables[fqn] = tm

            return DatabaseMetadata(
                server_name=server_name,
                database_name=db_row.database_name,
                tables=tables,
                version_hash=db_row.version_hash,
                extracted_at=db_row.last_extracted_at or datetime.now(timezone.utc),
            )
        finally:
            session.close()

    def _save_metadata_to_pg(self, meta: DatabaseMetadata) -> None:
        """Persist DatabaseMetadata to PostgreSQL tables."""
        from ai_agent.schema_engine.store.pg_repository import SchemaEngineRepository

        session = self._get_pg_session()
        try:
            repo = SchemaEngineRepository(session)
            db_row = repo.get_or_create_database(meta.server_name, meta.database_name)

            # Build a lookup of table_name → SETable.id for FK resolution
            table_id_map: Dict[str, str] = {}

            for fqn, table in meta.tables.items():
                search_text = self._build_search_text(table)
                se_table = repo.upsert_table(db_row.id, {
                    "schema_name": table.schema_name,
                    "table_name": table.table_name,
                    "row_count": table.row_count,
                    "description": table.description,
                    "table_comment": table.table_comment,
                    "summary": table.summary,
                    "business_purpose": table.business_purpose,
                    "domain": table.domain,
                    "semantic_keywords": json.dumps(table.semantic_keywords) if table.semantic_keywords else None,
                    "join_hints": json.dumps(table.join_hints) if table.join_hints else None,
                    "content_hash": self._content_hash(table),
                    "search_text": search_text,
                })
                table_id_map[table.table_name] = se_table.id

                # Upsert columns
                col_data = [
                    {
                        "name": c.name,
                        "data_type": c.data_type,
                        "nullable": c.nullable,
                        "description": c.description,
                        "is_primary_key": c.is_primary_key,
                        "is_foreign_key": c.name in {
                            fk.source_column for fk in table.foreign_keys
                        },
                    }
                    for c in table.columns
                ]
                repo.upsert_columns(se_table.id, col_data)

            # Upsert FK relationships
            relationships = []
            for fqn, table in meta.tables.items():
                source_id = table_id_map.get(table.table_name)
                if not source_id:
                    continue
                for fk in table.foreign_keys:
                    target_id = table_id_map.get(fk.target_table)
                    if not target_id:
                        continue
                    relationships.append({
                        "source_table_id": source_id,
                        "target_table_id": target_id,
                        "source_column": fk.source_column,
                        "target_column": fk.target_column,
                        "constraint_name": None,
                    })
            repo.replace_relationships(db_row.id, relationships)

            # Update database metadata
            repo.update_database_timestamps(
                db_row.id,
                extracted=True,
                summarized=meta.tables and any(t.summary for t in meta.tables.values()),
                version_hash=meta.version_hash,
                table_count=meta.table_count,
            )

            repo.commit()
            log.info(
                "MetadataStore: persisted %d tables to PostgreSQL for '%s'",
                meta.table_count, meta.server_name,
            )
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _load_chunks_from_pg(self, server_name: str) -> Optional[List[SchemaChunk]]:
        """Load SchemaChunks from PostgreSQL."""
        from ai_agent.schema_engine.store.pg_repository import SchemaEngineRepository

        session = self._get_pg_session()
        try:
            repo = SchemaEngineRepository(session)
            se_chunks = repo.get_chunks_by_server(server_name)
            if not se_chunks:
                return None

            chunks = []
            for sc in se_chunks:
                chunk = SchemaChunk(
                    chunk_id=sc.id,
                    server_name=sc.server_name,
                    table_fqn=sc.table_fqn,
                    table_name=sc.table_name,
                    schema_name=sc.schema_name,
                    search_text=sc.search_text,
                    summary_text=sc.summary_text,
                    compact_ddl=sc.compact_ddl,
                    domain=sc.domain,
                    column_names=json.loads(sc.column_names_json) if sc.column_names_json else [],
                    primary_keys=json.loads(sc.primary_keys_json) if sc.primary_keys_json else [],
                    related_tables=json.loads(sc.related_tables_json) if sc.related_tables_json else [],
                    row_count=sc.row_count,
                    foreign_key_count=sc.foreign_key_count,
                    embedding=None,  # Embeddings live in Milvus, not memory
                )
                chunks.append(chunk)
            return chunks
        finally:
            session.close()

    def _save_chunks_to_pg(self, server_name: str, chunks: List[SchemaChunk]) -> None:
        """Persist SchemaChunks to PostgreSQL."""
        from ai_agent.schema_engine.store.pg_repository import SchemaEngineRepository

        session = self._get_pg_session()
        try:
            repo = SchemaEngineRepository(session)
            db_row = repo.get_database(server_name)
            if not db_row:
                log.warning("MetadataStore: no database row for '%s', skipping chunk save", server_name)
                return

            # Map table_name → SETable.id
            table_rows = repo.get_tables_for_database(db_row.id)
            table_id_map = {t.table_name: t.id for t in table_rows}

            for chunk in chunks:
                table_id = table_id_map.get(chunk.table_name)
                if not table_id:
                    log.debug("MetadataStore: no table_id for chunk '%s'", chunk.table_name)
                    continue
                repo.upsert_chunk({
                    "database_id": db_row.id,
                    "table_id": table_id,
                    "server_name": server_name,
                    "table_fqn": chunk.table_fqn,
                    "table_name": chunk.table_name,
                    "schema_name": chunk.schema_name,
                    "search_text": chunk.search_text,
                    "summary_text": chunk.summary_text,
                    "compact_ddl": chunk.compact_ddl,
                    "domain": chunk.domain,
                    "column_names_json": json.dumps(chunk.column_names) if chunk.column_names else None,
                    "primary_keys_json": json.dumps(chunk.primary_keys) if chunk.primary_keys else None,
                    "related_tables_json": json.dumps(chunk.related_tables) if chunk.related_tables else None,
                    "row_count": chunk.row_count,
                    "foreign_key_count": chunk.foreign_key_count,
                })

            repo.commit()
            log.info("MetadataStore: persisted %d chunks to PostgreSQL for '%s'",
                     len(chunks), server_name)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ═════════════════════════════════════════════════════════════════════
    # Redis cache helpers
    # ═════════════════════════════════════════════════════════════════════

    def _cache_to_redis(self, key: str, meta: DatabaseMetadata) -> None:
        if self._redis_cache and self._redis_cache.is_connected:
            try:
                serialised = json.dumps(self._serialise_metadata(meta))
                self._redis_cache.set(
                    key, serialised,
                    ttl_seconds=self._cfg.metadata_cache_ttl_hours * 3600,
                )
            except Exception as e:
                log.warning("MetadataStore: Redis cache write error: %s", e)

    # ═════════════════════════════════════════════════════════════════════
    # Serialisation (for Redis cache layer)
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _serialise_metadata(meta: DatabaseMetadata) -> Dict:
        return {
            "server_name": meta.server_name,
            "database_name": meta.database_name,
            "version_hash": meta.version_hash,
            "extracted_at": meta.extracted_at.isoformat(),
            "tables": {fqn: t.to_dict() for fqn, t in meta.tables.items()},
        }

    @staticmethod
    def _deserialise_metadata(data: Dict) -> DatabaseMetadata:
        tables = {}
        for fqn, td in data.get("tables", {}).items():
            tables[fqn] = TableMetadata.from_dict(td)
        return DatabaseMetadata(
            server_name=data["server_name"],
            database_name=data.get("database_name", "default"),
            tables=tables,
            version_hash=data.get("version_hash", ""),
            extracted_at=datetime.fromisoformat(data["extracted_at"])
            if "extracted_at" in data else datetime.now(timezone.utc),
        )

    # ═════════════════════════════════════════════════════════════════════
    # Helpers
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_search_text(table: TableMetadata) -> str:
        """Build a concatenated text blob for PostgreSQL full-text search."""
        parts = [table.table_name]
        if table.summary:
            parts.append(table.summary)
        if table.business_purpose:
            parts.append(table.business_purpose)
        if table.description:
            parts.append(table.description)
        if table.semantic_keywords:
            parts.extend(table.semantic_keywords)
        parts.extend(c.name for c in table.columns)
        return " ".join(parts)

    @staticmethod
    def _content_hash(table: TableMetadata) -> str:
        """Compute a hash of table content for change detection."""
        content = json.dumps({
            "name": table.table_name,
            "columns": [(c.name, c.data_type) for c in table.columns],
            "pks": table.primary_keys,
            "fks": [(fk.source_column, fk.target_table, fk.target_column)
                    for fk in table.foreign_keys],
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    # ═════════════════════════════════════════════════════════════════════
    # Health
    # ═════════════════════════════════════════════════════════════════════

    def health(self) -> Dict[str, Any]:
        return {
            "pg_available": self._pg_available,
            "redis_connected": (
                self._redis_cache.is_connected if self._redis_cache else False
            ),
            "memory_entries": len(self._mem),
        }


class _CacheEntry:
    """Simple timestamped wrapper."""
    __slots__ = ("value", "created_at")

    def __init__(self, value):
        self.value = value
        self.created_at = datetime.now(timezone.utc)

    def expired(self, ttl_hours: int) -> bool:
        return datetime.now(timezone.utc) - self.created_at > timedelta(hours=ttl_hours)


# ── Singleton ────────────────────────────────────────────────────────────
_store: Optional[MetadataStore] = None


def get_metadata_store() -> MetadataStore:
    global _store
    if _store is None:
        _store = MetadataStore()
    return _store
