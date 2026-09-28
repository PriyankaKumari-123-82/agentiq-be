"""Schema engine health check and diagnostics.

Provides a unified health report across all storage layers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

log = logging.getLogger(__name__)


def schema_engine_health() -> Dict[str, Any]:
    """Return health status of all schema engine storage layers."""
    health: Dict[str, Any] = {"status": "ok"}

    # PostgreSQL
    try:
        from ai_agent.core.db import engine
        from sqlmodel import Session, text
        session = Session(engine)
        try:
            session.exec(text("SELECT 1"))
            health["postgresql"] = {"status": "connected"}
        finally:
            session.close()
    except Exception as e:
        health["postgresql"] = {"status": "error", "error": str(e)}
        health["status"] = "degraded"

    # Milvus
    try:
        from ai_agent.schema_engine.store.vector_store import get_vector_store
        vs = get_vector_store()
        if vs.connect():
            health["milvus"] = {
                "status": "connected",
                "entity_count": vs.count(),
            }
        else:
            health["milvus"] = {"status": "disconnected"}
            health["status"] = "degraded"
    except Exception as e:
        health["milvus"] = {"status": "error", "error": str(e)}
        health["status"] = "degraded"

    # Redis
    try:
        from ai_agent.schema_engine.store.redis_cache import get_redis_cache
        rc = get_redis_cache()
        health["redis"] = rc.health_check()
        if health["redis"].get("status") != "connected":
            health["status"] = "degraded"
    except Exception as e:
        health["redis"] = {"status": "error", "error": str(e)}
        health["status"] = "degraded"

    # MetadataStore
    try:
        from ai_agent.schema_engine.store.metadata_store import get_metadata_store
        store = get_metadata_store()
        health["metadata_store"] = store.health()
    except Exception as e:
        health["metadata_store"] = {"status": "error", "error": str(e)}

    # Schema Engine Config
    try:
        from ai_agent.schema_engine.config import get_config
        cfg = get_config()
        health["config"] = {
            "pg_persist_enabled": cfg.pg_persist_enabled,
            "milvus_persist_enabled": cfg.milvus_persist_enabled,
            "redis_cache_enabled": cfg.redis_cache_enabled,
            "incremental_embedding": cfg.incremental_embedding,
            "distributed_locking": cfg.distributed_locking,
        }
    except Exception:
        pass

    return health


def schema_engine_stats(server_name: str) -> Dict[str, Any]:
    """Return operational stats for a specific server."""
    stats: Dict[str, Any] = {"server_name": server_name}

    # Database row
    try:
        from ai_agent.core.db import engine
        from sqlmodel import Session
        from ai_agent.schema_engine.store.pg_repository import SchemaEngineRepository
        session = Session(engine)
        try:
            repo = SchemaEngineRepository(session)
            db_row = repo.get_database(server_name)
            if db_row:
                stats["database"] = {
                    "id": db_row.id,
                    "table_count": db_row.table_count,
                    "version_hash": db_row.version_hash,
                    "last_extracted_at": str(db_row.last_extracted_at) if db_row.last_extracted_at else None,
                    "last_summarized_at": str(db_row.last_summarized_at) if db_row.last_summarized_at else None,
                    "last_indexed_at": str(db_row.last_indexed_at) if db_row.last_indexed_at else None,
                }

                # Chunk count
                chunks = repo.get_chunks_for_database(db_row.id)
                stats["chunk_count"] = len(chunks)

                # Latest version
                latest = repo.get_latest_version(db_row.id)
                if latest:
                    stats["latest_version"] = {
                        "hash": latest.version_hash,
                        "table_count": latest.table_count,
                        "tables_added": latest.tables_added,
                        "tables_removed": latest.tables_removed,
                        "created_at": str(latest.created_at),
                    }

                # Top tables
                stats["top_tables"] = repo.get_top_tables(db_row.id, limit=5)
        finally:
            session.close()
    except Exception as e:
        stats["error"] = str(e)

    # Milvus embedding count
    try:
        from ai_agent.schema_engine.store.vector_store import get_vector_store
        vs = get_vector_store()
        if vs.is_connected:
            stats["milvus_embeddings"] = vs.count(server_name)
    except Exception:
        pass

    # Query log stats
    try:
        from ai_agent.schema_engine.store.query_log import get_query_log
        ql = get_query_log()
        stats["query_log"] = ql.stats(server_name)
    except Exception:
        pass

    return stats
