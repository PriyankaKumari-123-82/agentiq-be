"""Add schema engine persistent storage tables.

Revision ID: se_001
Revises: 5983b6791e44
Create Date: 2026-05-16 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "se_001"
down_revision = "5983b6791e44"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── se_databases ─────────────────────────────────────────────────────
    op.create_table(
        "se_databases",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("server_name", sa.String(), nullable=False, unique=True, index=True),
        sa.Column("database_name", sa.String(), nullable=False, server_default="default"),
        sa.Column("version_hash", sa.String(), server_default=""),
        sa.Column("table_count", sa.Integer(), server_default="0"),
        sa.Column("is_active", sa.Boolean(), server_default="true"),
        sa.Column("last_extracted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_summarized_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )

    # ── se_tables ────────────────────────────────────────────────────────
    op.create_table(
        "se_tables",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("database_id", sa.String(), sa.ForeignKey("se_databases.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("schema_name", sa.String(), nullable=False, server_default="public"),
        sa.Column("table_name", sa.String(), nullable=False, index=True),
        sa.Column("row_count", sa.BigInteger(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("table_comment", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("business_purpose", sa.Text(), nullable=True),
        sa.Column("domain", sa.String(), nullable=True, index=True),
        sa.Column("semantic_keywords", sa.Text(), nullable=True),
        sa.Column("join_hints", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(), server_default=""),
        sa.Column("search_text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.UniqueConstraint("database_id", "schema_name", "table_name",
                           name="uq_se_tables_fqn"),
    )
    op.create_index("ix_se_tables_fqn", "se_tables", ["schema_name", "table_name"])

    # ── Full-text search index on se_tables.search_text ──────────────────
    op.execute("""
        CREATE INDEX ix_se_tables_fts
        ON se_tables
        USING gin(to_tsvector('english', coalesce(search_text, '')))
    """)

    # ── se_columns ───────────────────────────────────────────────────────
    op.create_table(
        "se_columns",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("table_id", sa.String(), sa.ForeignKey("se_tables.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("name", sa.String(), nullable=False, index=True),
        sa.Column("data_type", sa.String(), server_default="unknown"),
        sa.Column("nullable", sa.Boolean(), server_default="true"),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_primary_key", sa.Boolean(), server_default="false"),
        sa.Column("is_foreign_key", sa.Boolean(), server_default="false"),
        sa.Column("ordinal_position", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.UniqueConstraint("table_id", "name", name="uq_se_columns_table_name"),
    )

    # ── se_relationships (FK graph edges) ────────────────────────────────
    op.create_table(
        "se_relationships",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("database_id", sa.String(), sa.ForeignKey("se_databases.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("source_table_id", sa.String(),
                  sa.ForeignKey("se_tables.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("target_table_id", sa.String(),
                  sa.ForeignKey("se_tables.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("source_column", sa.String(), nullable=False),
        sa.Column("target_column", sa.String(), nullable=False),
        sa.Column("constraint_name", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    op.create_index("ix_se_relationships_pair", "se_relationships",
                    ["source_table_id", "target_table_id"])

    # ── se_chunks (one per table, stores retrieval text) ─────────────────
    op.create_table(
        "se_chunks",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("database_id", sa.String(), sa.ForeignKey("se_databases.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("table_id", sa.String(), sa.ForeignKey("se_tables.id", ondelete="CASCADE"),
                  nullable=False, unique=True, index=True),
        sa.Column("server_name", sa.String(), nullable=False, index=True),
        sa.Column("table_fqn", sa.String(), nullable=False, index=True),
        sa.Column("table_name", sa.String(), nullable=False, index=True),
        sa.Column("schema_name", sa.String(), server_default="public"),
        sa.Column("search_text", sa.Text(), server_default=""),
        sa.Column("summary_text", sa.Text(), server_default=""),
        sa.Column("compact_ddl", sa.Text(), server_default=""),
        sa.Column("domain", sa.String(), nullable=True, index=True),
        sa.Column("column_names_json", sa.Text(), nullable=True),
        sa.Column("primary_keys_json", sa.Text(), nullable=True),
        sa.Column("related_tables_json", sa.Text(), nullable=True),
        sa.Column("row_count", sa.BigInteger(), nullable=True),
        sa.Column("foreign_key_count", sa.Integer(), server_default="0"),
        sa.Column("embedding_id", sa.String(), nullable=True),
        sa.Column("embedding_version", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )

    # ── Full-text search on chunks ───────────────────────────────────────
    op.execute("""
        CREATE INDEX ix_se_chunks_fts
        ON se_chunks
        USING gin(to_tsvector('english', coalesce(search_text, '')))
    """)

    # ── se_query_logs ────────────────────────────────────────────────────
    op.create_table(
        "se_query_logs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("database_id", sa.String(), sa.ForeignKey("se_databases.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("server_name", sa.String(), nullable=False, index=True),
        sa.Column("user_query", sa.Text(), nullable=False),
        sa.Column("generated_sql", sa.Text(), nullable=True),
        sa.Column("tables_used_json", sa.Text(), nullable=True),
        sa.Column("joins_used_json", sa.Text(), nullable=True),
        sa.Column("retrieval_confidence", sa.Float(), server_default="0.0"),
        sa.Column("execution_success", sa.Boolean(), server_default="true"),
        sa.Column("latency_ms", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()"), index=True),
    )

    # ── se_table_usage_stats ─────────────────────────────────────────────
    op.create_table(
        "se_table_usage_stats",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("database_id", sa.String(), sa.ForeignKey("se_databases.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("table_name", sa.String(), nullable=False, index=True),
        sa.Column("query_count", sa.Integer(), server_default="0"),
        sa.Column("success_count", sa.Integer(), server_default="0"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.UniqueConstraint("database_id", "table_name",
                           name="uq_se_table_usage_stats"),
    )

    # ── se_join_usage_stats ──────────────────────────────────────────────
    op.create_table(
        "se_join_usage_stats",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("database_id", sa.String(), sa.ForeignKey("se_databases.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("join_pattern", sa.Text(), nullable=False, index=True),
        sa.Column("query_count", sa.Integer(), server_default="0"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.UniqueConstraint("database_id", "join_pattern",
                           name="uq_se_join_usage_stats"),
    )

    # ── se_schema_versions ───────────────────────────────────────────────
    op.create_table(
        "se_schema_versions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("database_id", sa.String(), sa.ForeignKey("se_databases.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("version_hash", sa.String(), nullable=False, index=True),
        sa.Column("table_count", sa.Integer(), server_default="0"),
        sa.Column("tables_added", sa.Integer(), server_default="0"),
        sa.Column("tables_removed", sa.Integer(), server_default="0"),
        sa.Column("tables_modified", sa.Integer(), server_default="0"),
        sa.Column("embeddings_regenerated", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )

    # ── se_retrieval_audit_logs ──────────────────────────────────────────
    op.create_table(
        "se_retrieval_audit_logs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("database_id", sa.String(), sa.ForeignKey("se_databases.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("user_query", sa.Text(), nullable=False),
        sa.Column("tables_retrieved_json", sa.Text(), nullable=True),
        sa.Column("retrieval_method", sa.String(), server_default="hybrid"),
        sa.Column("bm25_score", sa.Float(), server_default="0.0"),
        sa.Column("vector_score", sa.Float(), server_default="0.0"),
        sa.Column("graph_score", sa.Float(), server_default="0.0"),
        sa.Column("total_chunks", sa.Integer(), server_default="0"),
        sa.Column("token_budget_used", sa.Integer(), server_default="0"),
        sa.Column("latency_ms", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()"), index=True),
    )


def downgrade() -> None:
    op.drop_table("se_retrieval_audit_logs")
    op.drop_table("se_schema_versions")
    op.drop_table("se_join_usage_stats")
    op.drop_table("se_table_usage_stats")
    op.drop_table("se_query_logs")
    op.drop_table("se_chunks")
    op.drop_table("se_relationships")
    op.drop_table("se_columns")
    op.drop_table("se_tables")
    op.drop_table("se_databases")
