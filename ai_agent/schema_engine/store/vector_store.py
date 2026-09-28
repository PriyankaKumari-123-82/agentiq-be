"""Milvus vector store for schema engine embeddings.

This is the authoritative storage for all embedding vectors. Vectors are
NEVER kept only in process memory.

Uses the existing Milvus deployment (pymilvus) configured via settings.
"""

from __future__ import annotations

import hashlib
import logging
import json
from typing import Any, Dict, List, Optional, Tuple

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from ai_agent.core.config import settings
from ai_agent.schema_engine.config import get_config

log = logging.getLogger(__name__)

# Collection name prefix for schema engine
_COLLECTION_PREFIX = "se_embeddings"


def _collection_name() -> str:
    cfg = get_config()
    prefix = getattr(cfg, "milvus_collection_prefix", _COLLECTION_PREFIX)
    return prefix


class SchemaVectorStore:
    """Milvus-backed vector store for schema engine embeddings."""

    def __init__(self):
        self._cfg = get_config()
        self._connected = False
        self._collection: Optional[Collection] = None

    # ═════════════════════════════════════════════════════════════════════
    # Connection
    # ═════════════════════════════════════════════════════════════════════

    def connect(self) -> bool:
        """Connect to Milvus. Returns True on success."""
        if self._connected:
            return True
        try:
            log.info(f"Initializing CollectionPerAgent with Milvus at {settings.MILVUS_HOST}")
            connections.connect(
                alias="schema_engine",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
                user=settings.MILVUS_USER or "",
                password=settings.MILVUS_PASSWORD or "",
            )
            self._connected = True
            log.info(
                "SchemaVectorStore: connected to Milvus at %s:%s",
                settings.MILVUS_HOST, settings.MILVUS_PORT,
            )
            self._ensure_collection()
            return True
        except Exception as e:
            log.error("SchemaVectorStore: Milvus connection failed: %s", e, exc_info=True)
            self._connected = False
            return False

    def _ensure_collection(self) -> None:
        """Create the schema engine collection if it doesn't exist."""
        name = _collection_name()
        if utility.has_collection(name, using="schema_engine"):
            self._collection = Collection(name, using="schema_engine")
            # Check dimension compatibility
            emb_field = next((f for f in self._collection.schema.fields if f.name == "embedding"), None)
            existing_dim = getattr(emb_field, "params", {}).get("dim") if emb_field else None
            if existing_dim and existing_dim != self._cfg.embedding_dimensions:
                log.warning(
                    "SchemaVectorStore: collection '%s' dimension mismatch (%s != %s). Recreating...",
                    name, existing_dim, self._cfg.embedding_dimensions,
                )
                utility.drop_collection(name, using="schema_engine")
                self._collection = None
            else:
                self._collection.load()
                log.info("SchemaVectorStore: loaded existing collection '%s'", name)
                return

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._cfg.embedding_dimensions),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="server_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="embedding_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="version_hash", dtype=DataType.VARCHAR, max_length=64),
        ]
        schema = CollectionSchema(fields, description="Schema engine embeddings")
        self._collection = Collection(name=name, schema=schema, using="schema_engine")

        # Create index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 256},
        }
        self._collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )
        self._collection.load()
        log.info("SchemaVectorStore: created collection '%s' with IVF_FLAT index", name)

    # ═════════════════════════════════════════════════════════════════════
    # Upsert
    # ═════════════════════════════════════════════════════════════════════

    def upsert_embeddings(
        self,
        entries: List[Dict[str, Any]],
    ) -> List[str]:
        """Upsert embedding vectors into Milvus.

        Each entry dict must have:
            - chunk_id: str
            - embedding: List[float]
            - server_name: str
            - table_name: str
            - domain: str (optional, defaults to "general")
            - embedding_type: str (e.g. "table_summary")
            - version_hash: str (content hash when embedding was generated)

        Returns list of Milvus entity IDs.
        """
        if not self._connected:
            if not self.connect():
                log.warning("SchemaVectorStore: cannot upsert, not connected")
                return []

        if not entries:
            return []

        # Generate deterministic IDs based on (server_name, chunk_id, embedding_type)
        ids = []
        embeddings = []
        chunk_ids = []
        server_names = []
        table_names = []
        domains = []
        embedding_types = []
        version_hashes = []

        for entry in entries:
            # Deterministic ID for upsert dedup
            raw_id = f"{entry['server_name']}:{entry['chunk_id']}:{entry.get('embedding_type', 'table_summary')}"
            entity_id = hashlib.md5(raw_id.encode()).hexdigest()[:32]

            ids.append(entity_id)
            embeddings.append(entry["embedding"])
            chunk_ids.append(entry["chunk_id"])
            server_names.append(entry["server_name"])
            table_names.append(entry["table_name"])
            domains.append(entry.get("domain", "general"))
            embedding_types.append(entry.get("embedding_type", "table_summary"))
            version_hashes.append(entry.get("version_hash", ""))

        # Delete existing by IDs (upsert pattern)
        try:
            expr = f'id in {ids}'
            self._collection.delete(expr)
        except Exception:
            pass  # May not exist yet

        # Insert
        try:
            self._collection.insert([
                ids,
                embeddings,
                chunk_ids,
                server_names,
                table_names,
                domains,
                embedding_types,
                version_hashes,
            ])
            self._collection.flush()
            log.info("SchemaVectorStore: upserted %d embeddings", len(ids))
            return ids
        except Exception as e:
            log.error("SchemaVectorStore: upsert failed: %s", e)
            return []

    # ═════════════════════════════════════════════════════════════════════
    # Search
    # ═════════════════════════════════════════════════════════════════════

    def search(
        self,
        query_embedding: List[float],
        server_name: str,
        top_k: int = 20,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings. Returns list of {chunk_id, score, ...}."""
        if not self._connected:
            if not self.connect():
                return []

        # Build filter expression
        expr_parts = [f'server_name == "{server_name}"']
        if domain:
            expr_parts.append(f'domain == "{domain}"')
        filter_expr = " and ".join(expr_parts)

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}

        try:
            results = self._collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["chunk_id", "table_name", "domain", "version_hash"],
            )

            hits = []
            for hit in results[0]:
                hits.append({
                    "id": hit.id,
                    "chunk_id": hit.entity.get("chunk_id"),
                    "table_name": hit.entity.get("table_name"),
                    "domain": hit.entity.get("domain"),
                    "version_hash": hit.entity.get("version_hash"),
                    "score": hit.score,
                })
            return hits

        except Exception as e:
            log.error("SchemaVectorStore: search failed: %s", e)
            return []

    # ═════════════════════════════════════════════════════════════════════
    # Delete
    # ═════════════════════════════════════════════════════════════════════

    def delete_by_server(self, server_name: str) -> int:
        """Delete all embeddings for a server. Returns count deleted."""
        if not self._connected:
            if not self.connect():
                return 0
        try:
            expr = f'server_name == "{server_name}"'
            result = self._collection.delete(expr)
            self._collection.flush()
            deleted = getattr(result, "delete_count", 0)
            log.info("SchemaVectorStore: deleted %d embeddings for '%s'", deleted, server_name)
            return deleted
        except Exception as e:
            log.error("SchemaVectorStore: delete failed: %s", e)
            return 0

    def delete_by_chunk_ids(self, chunk_ids: List[str]) -> int:
        """Delete embeddings by chunk IDs."""
        if not self._connected or not chunk_ids:
            return 0
        try:
            expr = f'chunk_id in {chunk_ids}'
            result = self._collection.delete(expr)
            self._collection.flush()
            return getattr(result, "delete_count", 0)
        except Exception as e:
            log.error("SchemaVectorStore: delete_by_chunk_ids failed: %s", e)
            return 0

    # ═════════════════════════════════════════════════════════════════════
    # Utilities
    # ═════════════════════════════════════════════════════════════════════

    def count(self, server_name: Optional[str] = None) -> int:
        """Count entities in the collection."""
        if not self._connected:
            return 0
        try:
            if server_name:
                expr = f'server_name == "{server_name}"'
                return self._collection.query(expr=expr, output_fields=["count(*)"])[0]["count(*)"]
            return self._collection.num_entities
        except Exception:
            return 0

    def get_existing_versions(self, server_name: str) -> Dict[str, str]:
        """Return {chunk_id: version_hash} for all embeddings of a server.

        Used to skip re-embedding unchanged chunks.
        """
        if not self._connected:
            if not self.connect():
                return {}
        try:
            expr = f'server_name == "{server_name}"'
            results = self._collection.query(
                expr=expr,
                output_fields=["chunk_id", "version_hash"],
                limit=16384,
            )
            return {r["chunk_id"]: r["version_hash"] for r in results}
        except Exception as e:
            log.error("SchemaVectorStore: get_existing_versions failed: %s", e)
            return {}

    @property
    def is_connected(self) -> bool:
        return self._connected


# ── Singleton ─────────────────────────────────────────────────────────────

_vector_store: Optional[SchemaVectorStore] = None


def get_vector_store() -> SchemaVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = SchemaVectorStore()
    return _vector_store
