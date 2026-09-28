"""SQLModel-based PostgreSQL models for durable schema engine storage.

These tables store ALL authoritative schema engine data:
- Extracted database/table/column metadata
- FK relationships and graph edges
- Schema chunks with search text
- Embedding references (vectors live in Milvus)
- Query logs and usage statistics
- Schema versions for incremental sync
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, List

import sqlalchemy as sa
from sqlmodel import Field, SQLModel, Relationship, Column


# ═══════════════════════════════════════════════════════════════════════════
# Schema Databases (top-level container)
# ═══════════════════════════════════════════════════════════════════════════

class SEDatabase(SQLModel, table=True):
    """A registered database/MCP server in the schema engine."""
    __tablename__ = "se_databases"

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )
    server_name: str = Field(index=True, unique=True)
    database_name: str = Field(default="default")
    version_hash: str = Field(default="")
    table_count: int = Field(default=0)
    is_active: bool = Field(default=True)
    last_extracted_at: Optional[datetime] = None
    last_summarized_at: Optional[datetime] = None
    last_indexed_at: Optional[datetime] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )

    # Relationships
    tables: List["SETable"] = Relationship(back_populates="database")
    versions: List["SESchemaVersion"] = Relationship(back_populates="database")


# ═══════════════════════════════════════════════════════════════════════════
# Schema Tables
# ═══════════════════════════════════════════════════════════════════════════

class SETable(SQLModel, table=True):
    """One table within a database."""
    __tablename__ = "se_tables"
    __table_args__ = (
        sa.UniqueConstraint("database_id", "schema_name", "table_name"),
        sa.Index("ix_se_tables_fqn", "schema_name", "table_name"),
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )
    database_id: str = Field(foreign_key="se_databases.id", index=True)
    schema_name: str = Field(default="public")
    table_name: str = Field(index=True)

    # Metadata
    row_count: Optional[int] = None
    description: Optional[str] = None
    table_comment: Optional[str] = None

    # Summarization results
    summary: Optional[str] = None
    business_purpose: Optional[str] = None
    domain: Optional[str] = Field(default=None, index=True)
    semantic_keywords: Optional[str] = Field(
        default=None,
        description="JSON array of keywords",
    )
    join_hints: Optional[str] = Field(
        default=None,
        description="JSON array of join hints",
    )

    # Content hash for change detection
    content_hash: str = Field(default="")

    # Full-text search column (auto-populated by trigger/application)
    search_text: Optional[str] = Field(
        default=None,
        sa_column=Column(sa.Text),
        description="Concatenated searchable text for PostgreSQL FTS",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )

    # Relationships
    database: "SEDatabase" = Relationship(back_populates="tables")
    columns: List["SEColumn"] = Relationship(back_populates="table")
    outgoing_fks: List["SERelationship"] = Relationship(
        back_populates="source_table",
        sa_relationship_kwargs={"foreign_keys": "[SERelationship.source_table_id]"},
    )
    incoming_fks: List["SERelationship"] = Relationship(
        back_populates="target_table",
        sa_relationship_kwargs={"foreign_keys": "[SERelationship.target_table_id]"},
    )
    chunk: Optional["SEChunk"] = Relationship(back_populates="table")


# ═══════════════════════════════════════════════════════════════════════════
# Schema Columns
# ═══════════════════════════════════════════════════════════════════════════

class SEColumn(SQLModel, table=True):
    """One column within a table."""
    __tablename__ = "se_columns"
    __table_args__ = (
        sa.UniqueConstraint("table_id", "name"),
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )
    table_id: str = Field(foreign_key="se_tables.id", index=True)
    name: str = Field(index=True)
    data_type: str = Field(default="unknown")
    nullable: bool = Field(default=True)
    description: Optional[str] = None
    is_primary_key: bool = Field(default=False)
    is_foreign_key: bool = Field(default=False)
    ordinal_position: int = Field(default=0)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    table: "SETable" = Relationship(back_populates="columns")


# ═══════════════════════════════════════════════════════════════════════════
# Schema Relationships (FK graph edges)
# ═══════════════════════════════════════════════════════════════════════════

class SERelationship(SQLModel, table=True):
    """Foreign key relationship between two tables (graph edge)."""
    __tablename__ = "se_relationships"
    __table_args__ = (
        sa.Index(
            "ix_se_relationships_pair",
            "source_table_id", "target_table_id",
        ),
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )
    database_id: str = Field(foreign_key="se_databases.id", index=True)
    source_table_id: str = Field(foreign_key="se_tables.id", index=True)
    target_table_id: str = Field(foreign_key="se_tables.id", index=True)
    source_column: str
    target_column: str
    constraint_name: Optional[str] = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    source_table: "SETable" = Relationship(
        back_populates="outgoing_fks",
        sa_relationship_kwargs={"foreign_keys": "[SERelationship.source_table_id]"},
    )
    target_table: "SETable" = Relationship(
        back_populates="incoming_fks",
        sa_relationship_kwargs={"foreign_keys": "[SERelationship.target_table_id]"},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Schema Chunks (unit of retrieval — one per table)
# ═══════════════════════════════════════════════════════════════════════════

class SEChunk(SQLModel, table=True):
    """Indexed schema chunk (one per table). Authoritative retrieval unit."""
    __tablename__ = "se_chunks"

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )
    database_id: str = Field(foreign_key="se_databases.id", index=True)
    table_id: str = Field(foreign_key="se_tables.id", unique=True, index=True)
    server_name: str = Field(index=True)
    table_fqn: str = Field(index=True)
    table_name: str = Field(index=True)
    schema_name: str = Field(default="public")

    # Text representations
    search_text: str = Field(default="")
    summary_text: str = Field(default="")
    compact_ddl: str = Field(default="")

    # Metadata for ranking
    domain: Optional[str] = Field(default=None, index=True)
    column_names_json: Optional[str] = Field(
        default=None, description="JSON array of column names"
    )
    primary_keys_json: Optional[str] = Field(
        default=None, description="JSON array of PK column names"
    )
    related_tables_json: Optional[str] = Field(
        default=None, description="JSON array of FK-connected table names"
    )
    row_count: Optional[int] = None
    foreign_key_count: int = Field(default=0)

    # Embedding reference (actual vector in Milvus)
    embedding_id: Optional[str] = Field(
        default=None,
        description="Milvus entity ID for the embedding vector",
    )
    embedding_version: Optional[str] = Field(
        default=None,
        description="Hash of summary_text when embedding was computed",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )

    # Relationships
    table: "SETable" = Relationship(back_populates="chunk")


# ═══════════════════════════════════════════════════════════════════════════
# Table Categories + Sample Questions
# ═══════════════════════════════════════════════════════════════════════════

class SETableCategory(SQLModel, table=True):
    """A business-meaningful grouping of related tables (e.g. "complaints")."""
    __tablename__ = "se_table_categories"
    __table_args__ = (
        sa.UniqueConstraint("database_id", "category_name"),
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )
    database_id: str = Field(foreign_key="se_databases.id", index=True)
    category_name: str = Field(index=True)
    description: Optional[str] = None
    priority: int = Field(default=0, index=True, description="Lower is more important")
    table_names_json: str = Field(
        default="[]",
        description="JSON array of table names in this category",
    )
    questions: str = Field(
        default="[]",
        description="JSON array of sample question strings, in display order",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Query Logs (durable query-learning)
# ═══════════════════════════════════════════════════════════════════════════

class SEQueryLog(SQLModel, table=True):
    """Persisted query execution record for learning."""
    __tablename__ = "se_query_logs"

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )
    database_id: str = Field(foreign_key="se_databases.id", index=True)
    server_name: str = Field(index=True)
    user_query: str
    generated_sql: Optional[str] = None
    tables_used_json: Optional[str] = Field(
        default=None, description="JSON array of table names"
    )
    joins_used_json: Optional[str] = Field(
        default=None, description="JSON array of join expressions"
    )
    retrieval_confidence: float = Field(default=0.0)
    execution_success: bool = Field(default=True)
    latency_ms: int = Field(default=0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(sa.DateTime(timezone=True), index=True),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Usage Statistics (aggregated from query logs)
# ═══════════════════════════════════════════════════════════════════════════

class SETableUsageStats(SQLModel, table=True):
    """Aggregated table usage frequency for retrieval boost."""
    __tablename__ = "se_table_usage_stats"
    __table_args__ = (
        sa.UniqueConstraint("database_id", "table_name"),
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )
    database_id: str = Field(foreign_key="se_databases.id", index=True)
    table_name: str = Field(index=True)
    query_count: int = Field(default=0)
    success_count: int = Field(default=0)
    last_used_at: Optional[datetime] = None
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )


class SEJoinUsageStats(SQLModel, table=True):
    """Aggregated join pattern frequency."""
    __tablename__ = "se_join_usage_stats"
    __table_args__ = (
        sa.UniqueConstraint("database_id", "join_pattern"),
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )
    database_id: str = Field(foreign_key="se_databases.id", index=True)
    join_pattern: str = Field(
        index=True,
        description="e.g. 'orders JOIN customers ON orders.customer_id=customers.id'",
    )
    query_count: int = Field(default=0)
    last_used_at: Optional[datetime] = None
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Schema Versions (for incremental sync / change detection)
# ═══════════════════════════════════════════════════════════════════════════

class SESchemaVersion(SQLModel, table=True):
    """Tracks schema versions for incremental sync."""
    __tablename__ = "se_schema_versions"

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )
    database_id: str = Field(foreign_key="se_databases.id", index=True)
    version_hash: str = Field(index=True)
    table_count: int = Field(default=0)
    tables_added: int = Field(default=0)
    tables_removed: int = Field(default=0)
    tables_modified: int = Field(default=0)
    embeddings_regenerated: int = Field(default=0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    database: "SEDatabase" = Relationship(back_populates="versions")


# ═══════════════════════════════════════════════════════════════════════════
# Retrieval Audit Log
# ═══════════════════════════════════════════════════════════════════════════

class SERetrievalAuditLog(SQLModel, table=True):
    """Audit log of retrieval operations for observability."""
    __tablename__ = "se_retrieval_audit_logs"

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )
    database_id: str = Field(foreign_key="se_databases.id", index=True)
    user_query: str
    tables_retrieved_json: Optional[str] = None
    retrieval_method: str = Field(default="hybrid")
    bm25_score: float = Field(default=0.0)
    vector_score: float = Field(default=0.0)
    graph_score: float = Field(default=0.0)
    total_chunks: int = Field(default=0)
    token_budget_used: int = Field(default=0)
    latency_ms: int = Field(default=0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(sa.DateTime(timezone=True), index=True),
    )
