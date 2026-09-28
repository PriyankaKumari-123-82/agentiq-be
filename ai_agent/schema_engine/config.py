"""Centralised configuration for the schema engine subsystem.

Every component reads its knobs from ``SchemaEngineConfig``.  The defaults are
tuned for a 128 k-token model context window and a database with up to ~10 000
tables.  Override via environment variables prefixed ``SE_`` or by constructing
the config object directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class SchemaEngineConfig:
    """Immutable configuration bag consumed by every schema-engine component."""

    # ── Retrieval ────────────────────────────────────────────────────────
    retrieval_top_k: int = 15
    """Maximum tables returned by the hybrid retriever before pruning."""

    bm25_weight: float = 0.35
    """Weight of BM25 (lexical) score in final ranking (0-1)."""

    vector_weight: float = 0.45
    """Weight of vector (semantic) score in final ranking (0-1)."""

    graph_weight: float = 0.20
    """Weight of graph-expansion score in final ranking (0-1)."""

    # ── Graph Expansion ──────────────────────────────────────────────────
    graph_max_hops: int = 2
    """Maximum FK hops for relationship expansion."""

    graph_auto_expand_hops: int = 1
    """Automatically include tables within this many FK hops of a hit."""

    # ── Token Budget ─────────────────────────────────────────────────────
    schema_context_max_tokens: int = 3000
    """Hard token budget for the schema context injected into LLM prompts."""

    column_budget_per_table: int = 60
    """Max columns to include per table (rest are summarised)."""

    # ── Embeddings ───────────────────────────────────────────────────────
    # Served by the independent embeddings service (settings.EMBEDDINGS_BASE_URL).
    # Override via SE_EMBEDDING_MODEL / SE_EMBEDDING_DIMENSIONS.
    embedding_model: str = "bge-large-en-v1.5"
    """Model used to embed table/column descriptions."""

    embedding_dimensions: int = 1024
    """Embedding vector dimensionality — must match embedding_model's output size."""

    embedding_batch_size: int = 32
    """Batch size when embedding many chunks at once."""

    # ── Summarization ────────────────────────────────────────────────────
    summarization_model: str = "gpt-4o-mini"
    """Lightweight model used for schema summarization."""

    summarization_max_tokens: int = 200
    """Max tokens per table summary."""

    # ── Schema Classifier ────────────────────────────────────────────────
    schema_classifier_max_tokens: int = 4000
    """Max tokens for the single LLM call that clusters all tables into categories."""

    schema_classifier_question_max_tokens: int = 800
    """Max tokens for the per-category sample-question generation call."""

    # ── SQL Validation ───────────────────────────────────────────────────
    sql_max_rows: int = 10_000
    """Default LIMIT injected when the query has none."""

    sql_timeout_seconds: int = 30
    """Execution timeout for generated SQL."""

    sql_allow_write: bool = False
    """If False (default), only SELECT / WITH … SELECT are allowed."""

    # ── Exclusions ───────────────────────────────────────────────────────
    excluded_schemas: List[str] = field(
        default_factory=lambda: ["information_schema", "pg_catalog", "pg_toast", "sys", "mysql", "performance_schema"],
    )
    excluded_tables: List[str] = field(default_factory=list)
    excluded_columns: List[str] = field(
        default_factory=lambda: ["created_by", "updated_by", "deleted_at", "etl_batch_id", "etl_loaded_at"],
    )

    # ── Audit Column Patterns (compressed in context) ────────────────────
    audit_column_patterns: List[str] = field(
        default_factory=lambda: ["created_at", "updated_at", "modified_at", "inserted_at"],
    )

    # ── Domain Classification ────────────────────────────────────────────
    domain_keywords: dict = field(default_factory=lambda: {
        "finance": ["invoice", "payment", "ledger", "account", "transaction", "billing", "credit", "debit", "revenue", "expense"],
        "sales": ["order", "sale", "customer", "quote", "deal", "opportunity", "discount", "cart", "checkout"],
        "hr": ["employee", "salary", "department", "leave", "attendance", "payroll", "benefits", "hiring", "position"],
        "inventory": ["product", "stock", "warehouse", "sku", "item", "inventory", "shipment", "supplier"],
        "support": ["ticket", "issue", "case", "resolution", "sla", "escalation", "feedback", "complaint"],
    })

    # ── Metadata Store ───────────────────────────────────────────────────
    metadata_cache_ttl_hours: int = 24
    """TTL for cached metadata entries (memory + Redis)."""

    # ── Query Log ────────────────────────────────────────────────────────
    query_log_enabled: bool = True
    """Enable learning from executed queries."""

    query_log_max_entries: int = 5_000
    """Max recent entries kept in the query log per server."""

    # ── Durability / Persistence ────────────────────────────────────────
    pg_persist_enabled: bool = True
    """Enable PostgreSQL as the authoritative store for metadata/chunks."""

    milvus_persist_enabled: bool = True
    """Enable Milvus as the authoritative store for embeddings."""

    milvus_collection_prefix: str = "se_embeddings"
    """Milvus collection name for schema engine embeddings."""

    redis_cache_enabled: bool = True
    """Enable Redis as a distributed cache layer."""

    redis_cache_ttl_seconds: int = 3600
    """Default TTL for Redis cache entries (seconds)."""

    redis_persistence_required: bool = False
    """If True, startup will FAIL if Redis AOF is not enabled."""

    # ── Incremental Indexing ──────────────────────────────────────────
    incremental_embedding: bool = True
    """Skip re-embedding chunks whose content hash hasn't changed."""

    schema_version_tracking: bool = True
    """Record schema version snapshots for change auditing."""

    # ── Multi-Instance Safety ─────────────────────────────────────────
    distributed_locking: bool = True
    """Acquire Redis distributed locks during init to prevent races."""

    lock_timeout_seconds: int = 120
    """Maximum time to hold a distributed lock."""

    # ── Security ─────────────────────────────────────────────────────────
    tenant_isolation: bool = True
    """Scope all metadata to the current tenant / user."""

    allowed_schemas: Optional[List[str]] = None
    """If set, ONLY these schemas are visible (allow-list takes precedence)."""

    allowed_tables: Optional[List[str]] = None
    """If set, ONLY these tables are visible."""

    masked_columns: List[str] = field(
        default_factory=lambda: ["ssn", "password", "secret", "token", "credit_card"],
    )
    """Column name patterns whose values are never shown / sampled."""

    @classmethod
    def from_env(cls) -> "SchemaEngineConfig":
        """Build config from ``SE_*`` environment variables."""
        overrides = {}
        prefix = "SE_"
        for fld in cls.__dataclass_fields__:
            env_key = prefix + fld.upper()
            val = os.environ.get(env_key)
            if val is not None:
                target_type = cls.__dataclass_fields__[fld].type
                if target_type in ("int", int):
                    val = int(val)
                elif target_type in ("float", float):
                    val = float(val)
                elif target_type in ("bool", bool):
                    val = val.lower() in ("1", "true", "yes")
                overrides[fld] = val
        return cls(**overrides)


# Module-level singleton (overridable in tests).
_config: Optional[SchemaEngineConfig] = None


def get_config() -> SchemaEngineConfig:
    global _config
    if _config is None:
        _config = SchemaEngineConfig.from_env()
    return _config


def set_config(cfg: SchemaEngineConfig) -> None:
    global _config
    _config = cfg
