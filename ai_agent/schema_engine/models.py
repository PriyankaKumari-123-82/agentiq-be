"""Normalized internal data-models for the schema engine.

These are **not** ORM models — they are plain dataclasses that flow through
every stage of the pipeline.  Persistent storage (see ``store/``) serialises
them to JSON columns.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set


# ═══════════════════════════════════════════════════════════════════════════
# Core metadata models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ColumnMetadata:
    """Single column within a table."""
    name: str
    data_type: str
    nullable: bool = True
    description: Optional[str] = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    default_value: Optional[str] = None
    sample_values: Optional[List[str]] = None

    # ── Derived / enriched after summarization ───────────────────────────
    semantic_keywords: List[str] = field(default_factory=list)

    def to_compact(self) -> str:
        """name:type  (+ PK/FK annotation)."""
        parts = [f"{self.name}:{self.data_type}"]
        if self.is_primary_key:
            parts.append("PK")
        if self.is_foreign_key:
            parts.append("FK")
        return " ".join(parts)


@dataclass
class ForeignKeyMetadata:
    """A single FK relationship."""
    source_column: str
    target_table: str
    target_column: str
    target_schema: Optional[str] = None


@dataclass
class IndexMetadata:
    """Index on a table."""
    name: str
    columns: List[str]
    is_unique: bool = False
    index_type: Optional[str] = None


@dataclass
class TableMetadata:
    """Full metadata for one table (normalised internal model)."""
    schema_name: str
    table_name: str
    columns: List[ColumnMetadata] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[ForeignKeyMetadata] = field(default_factory=list)
    indexes: List[IndexMetadata] = field(default_factory=list)
    row_count: Optional[int] = None
    description: Optional[str] = None
    table_comment: Optional[str] = None
    sample_values: Optional[Dict[str, List[str]]] = None

    # ── Enriched after summarization ─────────────────────────────────────
    summary: Optional[str] = None
    business_purpose: Optional[str] = None
    domain: Optional[str] = None
    semantic_keywords: List[str] = field(default_factory=list)
    join_hints: List[str] = field(default_factory=list)

    @property
    def fqn(self) -> str:
        """Fully-qualified table name (schema.table)."""
        return f"{self.schema_name}.{self.table_name}"

    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def content_hash(self) -> str:
        """Deterministic hash for change-detection / versioning."""
        raw = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ── Serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "table_name": self.table_name,
            "columns": [
                {"name": c.name, "data_type": c.data_type, "nullable": c.nullable,
                 "description": c.description, "is_primary_key": c.is_primary_key,
                 "is_foreign_key": c.is_foreign_key, "sample_values": c.sample_values}
                for c in self.columns
            ],
            "primary_keys": self.primary_keys,
            "foreign_keys": [
                {"source_column": fk.source_column, "target_table": fk.target_table,
                 "target_column": fk.target_column, "target_schema": fk.target_schema}
                for fk in self.foreign_keys
            ],
            "row_count": self.row_count,
            "description": self.description,
            "table_comment": self.table_comment,
            "summary": self.summary,
            "business_purpose": self.business_purpose,
            "domain": self.domain,
            "semantic_keywords": self.semantic_keywords,
            "join_hints": self.join_hints,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TableMetadata":
        cols = [
            ColumnMetadata(
                name=c["name"], data_type=c.get("data_type", c.get("type", "unknown")),
                nullable=c.get("nullable", True), description=c.get("description"),
                is_primary_key=c.get("is_primary_key", False),
                is_foreign_key=c.get("is_foreign_key", False),
                sample_values=c.get("sample_values"),
            )
            for c in d.get("columns", [])
        ]
        fks = [
            ForeignKeyMetadata(
                source_column=fk["source_column"],
                target_table=fk["target_table"],
                target_column=fk["target_column"],
                target_schema=fk.get("target_schema"),
            )
            for fk in d.get("foreign_keys", [])
        ]
        return cls(
            schema_name=d.get("schema_name", "public"),
            table_name=d["table_name"],
            columns=cols,
            primary_keys=d.get("primary_keys", []),
            foreign_keys=fks,
            row_count=d.get("row_count"),
            description=d.get("description"),
            table_comment=d.get("table_comment"),
            summary=d.get("summary"),
            business_purpose=d.get("business_purpose"),
            domain=d.get("domain"),
            semantic_keywords=d.get("semantic_keywords", []),
            join_hints=d.get("join_hints", []),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Database-level container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DatabaseMetadata:
    """All metadata for a single database / MCP server."""
    server_name: str
    database_name: str
    tables: Dict[str, TableMetadata] = field(default_factory=dict)
    version_hash: str = ""
    extracted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def table_count(self) -> int:
        return len(self.tables)

    def get_table(self, name: str) -> Optional[TableMetadata]:
        """Look up by table_name or schema.table_name."""
        if name in self.tables:
            return self.tables[name]
        for t in self.tables.values():
            if t.table_name == name:
                return t
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Schema chunk (unit of retrieval)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SchemaChunk:
    """One searchable unit in the retrieval index.

    Each chunk represents **one table** (table-level chunking) and contains
    all the signals needed for hybrid search: text for BM25, an embedding
    vector slot, and graph-connectivity metadata.
    """
    chunk_id: str
    server_name: str
    table_fqn: str  # schema.table
    table_name: str
    schema_name: str

    # Text representations (built by indexer)
    search_text: str = ""        # concatenated keywords for BM25
    summary_text: str = ""       # human-readable summary for embedding
    compact_ddl: str = ""        # compact DDL for LLM context

    # Graph connectivity
    related_tables: List[str] = field(default_factory=list)  # directly FK-connected
    domain: Optional[str] = None

    # Column names for quick lookup
    column_names: List[str] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)

    # Embedding (populated during indexing)
    embedding: Optional[List[float]] = None

    # Metadata for ranking
    row_count: Optional[int] = None
    foreign_key_count: int = 0

    def to_context_string(self, max_columns: int = 30) -> str:
        """Render this chunk as a compact context string for the LLM."""
        if self.compact_ddl:
            return self.compact_ddl
        return f"{self.table_name}({', '.join(self.column_names[:max_columns])})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "server_name": self.server_name,
            "table_fqn": self.table_fqn,
            "table_name": self.table_name,
            "schema_name": self.schema_name,
            "search_text": self.search_text,
            "summary_text": self.summary_text,
            "compact_ddl": self.compact_ddl,
            "related_tables": self.related_tables,
            "domain": self.domain,
            "column_names": self.column_names,
            "primary_keys": self.primary_keys,
            "row_count": self.row_count,
            "foreign_key_count": self.foreign_key_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SchemaChunk":
        return cls(
            chunk_id=d["chunk_id"],
            server_name=d["server_name"],
            table_fqn=d["table_fqn"],
            table_name=d["table_name"],
            schema_name=d["schema_name"],
            search_text=d.get("search_text", ""),
            summary_text=d.get("summary_text", ""),
            compact_ddl=d.get("compact_ddl", ""),
            related_tables=d.get("related_tables", []),
            domain=d.get("domain"),
            column_names=d.get("column_names", []),
            primary_keys=d.get("primary_keys", []),
            row_count=d.get("row_count"),
            foreign_key_count=d.get("foreign_key_count", 0),
            embedding=None,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Retrieval result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalResult:
    """Returned by the hybrid retriever."""
    chunks: List[SchemaChunk]
    domains: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    confidence: float = 0.0
    token_estimate: int = 0

    @property
    def table_names(self) -> List[str]:
        return [c.table_name for c in self.chunks]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "domains": self.domains,
            "relationships": self.relationships,
            "confidence": self.confidence,
            "token_estimate": self.token_estimate,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RetrievalResult":
        return cls(
            chunks=[SchemaChunk.from_dict(c) for c in d.get("chunks", [])],
            domains=d.get("domains", []),
            relationships=d.get("relationships", []),
            confidence=d.get("confidence", 0.0),
            token_estimate=d.get("token_estimate", 0),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Schema classifier categories + sample questions
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TableCategory:
    """A business-meaningful grouping of related tables (e.g. "complaints")."""
    category_name: str
    description: str
    table_names: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    priority: int = 0
    """Lower is more important. Set from the classifier's ranked order."""


# ═══════════════════════════════════════════════════════════════════════════
# Two-stage planning models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SchemaIdentificationResult:
    """Output of Stage-1 (schema identification — no SQL yet)."""
    tables: List[str]
    columns: List[str] = field(default_factory=list)
    joins: List[str] = field(default_factory=list)
    ambiguities: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class SQLPlanResult:
    """Output of Stage-2 (SQL generation)."""
    sql: str
    explanation: str = ""
    tables_used: List[str] = field(default_factory=list)
    columns_used: List[str] = field(default_factory=list)
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class SQLValidationResult:
    """Result of pre-execution SQL validation."""
    is_valid: bool
    cleaned_sql: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    tables_referenced: List[str] = field(default_factory=list)
    columns_referenced: List[str] = field(default_factory=list)
    has_limit: bool = False
    estimated_complexity: str = "low"  # low / medium / high
