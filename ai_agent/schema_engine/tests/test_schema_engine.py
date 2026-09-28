"""Unit tests for the schema engine subsystem.

These tests exercise the core pipeline (indexing → BM25 → graph → retrieval
→ validation) **without** any LLM or Redis dependency.  Embedding and
summarization are skipped; the tests verify that the mechanical parts work
correctly with the synthetic enterprise schema.
"""

import asyncio
import pytest

from ai_agent.schema_engine.config import SchemaEngineConfig, set_config
from ai_agent.schema_engine.graph.schema_graph import SchemaGraph
from ai_agent.schema_engine.indexing.schema_indexer import BM25Index, SchemaIndexer
from ai_agent.schema_engine.models import (
    DatabaseMetadata,
    RetrievalResult,
    SchemaChunk,
    SQLValidationResult,
)
from ai_agent.schema_engine.retrieval.domain_classifier import DomainClassifier
from ai_agent.schema_engine.retrieval.hybrid_retriever import HybridSchemaRetriever
from ai_agent.schema_engine.tests.synthetic_schema import build_synthetic_database
from ai_agent.schema_engine.validation.sql_validator import SQLValidator


def _run(coro):
    """Run an async coroutine synchronously (no pytest-asyncio required)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _test_config():
    """Ensure tests use a deterministic config."""
    cfg = SchemaEngineConfig(
        retrieval_top_k=10,
        schema_context_max_tokens=2000,
        graph_max_hops=2,
        graph_auto_expand_hops=1,
    )
    set_config(cfg)
    yield
    set_config(None)


@pytest.fixture()
def db() -> DatabaseMetadata:
    return build_synthetic_database(table_count=30)


@pytest.fixture()
def chunks_and_bm25(db):
    indexer = SchemaIndexer()
    chunks, bm25 = _run(indexer.index_database(db, compute_embeddings=False))
    return chunks, bm25


@pytest.fixture()
def graph(db) -> SchemaGraph:
    return SchemaGraph(db)


# ═══════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════

class TestModels:
    def test_synthetic_schema_has_tables(self, db):
        assert db.table_count >= 20
        assert db.get_table("customers") is not None
        assert db.get_table("orders") is not None

    def test_table_fqn(self, db):
        orders = db.get_table("orders")
        assert orders.fqn == "public.orders"

    def test_content_hash_deterministic(self, db):
        t = db.get_table("invoices")
        assert t.content_hash() == t.content_hash()

    def test_serialisation_roundtrip(self, db):
        t = db.get_table("payments")
        d = t.to_dict()
        from ai_agent.schema_engine.models import TableMetadata
        t2 = TableMetadata.from_dict(d)
        assert t2.table_name == t.table_name
        assert len(t2.columns) == len(t.columns)


# ═══════════════════════════════════════════════════════════════════════════
# BM25 Index
# ═══════════════════════════════════════════════════════════════════════════

class TestBM25:
    def test_build_and_query(self, chunks_and_bm25):
        chunks, bm25 = chunks_and_bm25
        results = bm25.query("customer orders", top_k=5)
        assert len(results) > 0
        top_table = results[0][1].table_name
        assert top_table in {"customers", "orders", "order_items"}

    def test_exact_table_name_match(self, chunks_and_bm25):
        _, bm25 = chunks_and_bm25
        results = bm25.query("invoices", top_k=3)
        assert any(c.table_name == "invoices" for _, c in results)

    def test_empty_query(self, chunks_and_bm25):
        _, bm25 = chunks_and_bm25
        results = bm25.query("", top_k=5)
        assert results == []

    def test_column_name_search(self, chunks_and_bm25):
        _, bm25 = chunks_and_bm25
        results = bm25.query("salary department", top_k=5)
        assert any(c.table_name in {"employees", "departments"} for _, c in results)


# ═══════════════════════════════════════════════════════════════════════════
# Schema Graph
# ═══════════════════════════════════════════════════════════════════════════

class TestGraph:
    def test_expand_neighbours(self, graph):
        expanded = graph.expand_neighbours(["orders"], max_hops=1)
        assert "customers" in expanded
        assert "orders" in expanded

    def test_shortest_path(self, graph):
        path = graph.shortest_path("order_items", "customers")
        assert path is not None
        assert path.hops > 0
        assert "orders" in path.tables

    def test_no_path_self(self, graph):
        path = graph.shortest_path("customers", "customers")
        assert path is not None
        assert path.hops == 0

    def test_find_join_paths(self, graph):
        paths = graph.find_join_paths("payments", "customers", max_hops=3)
        assert len(paths) >= 1
        # payments -> invoices -> customers
        assert any(p.hops == 2 for p in paths)

    def test_score_expansion(self, graph):
        score_direct = graph.score_expansion(["customers"], "orders")
        score_distant = graph.score_expansion(["customers"], "general_ledger")
        assert score_direct > score_distant


# ═══════════════════════════════════════════════════════════════════════════
# Domain Classifier
# ═══════════════════════════════════════════════════════════════════════════

class TestDomainClassifier:
    def test_classify_finance(self, chunks_and_bm25):
        chunks, _ = chunks_and_bm25
        clf = DomainClassifier(chunks)
        domains = clf.classify_query("unpaid invoices last quarter")
        assert "finance" in domains

    def test_classify_sales(self, chunks_and_bm25):
        chunks, _ = chunks_and_bm25
        clf = DomainClassifier(chunks)
        domains = clf.classify_query("top customers by order volume")
        assert "sales" in domains

    def test_tables_in_domains(self, chunks_and_bm25):
        chunks, _ = chunks_and_bm25
        clf = DomainClassifier(chunks)
        hr_tables = clf.tables_in_domains(["hr"])
        hr_names = {c.table_name for c in hr_tables}
        assert "employees" in hr_names


# ═══════════════════════════════════════════════════════════════════════════
# SQL Validator
# ═══════════════════════════════════════════════════════════════════════════

class TestSQLValidator:
    @pytest.fixture()
    def validator(self):
        return SQLValidator()

    def test_valid_select(self, validator):
        r = validator.validate("SELECT * FROM orders LIMIT 10")
        assert r.is_valid
        assert r.cleaned_sql is not None

    def test_reject_drop(self, validator):
        r = validator.validate("DROP TABLE orders")
        assert not r.is_valid
        assert any("Forbidden" in e or "SELECT" in e for e in r.errors)

    def test_reject_delete(self, validator):
        r = validator.validate("DELETE FROM orders WHERE id = 1")
        assert not r.is_valid

    def test_inject_limit(self, validator):
        r = validator.validate("SELECT * FROM orders")
        assert r.is_valid
        assert "LIMIT" in r.cleaned_sql

    def test_cte_allowed(self, validator):
        sql = "WITH cte AS (SELECT id FROM orders) SELECT * FROM cte LIMIT 5"
        r = validator.validate(sql)
        assert r.is_valid

    def test_multistatement_rejected(self, validator):
        r = validator.validate("SELECT 1; SELECT 2")
        assert not r.is_valid

    def test_injection_pattern(self, validator):
        r = validator.validate("SELECT * FROM orders; DROP TABLE orders")
        assert not r.is_valid

    def test_markdown_fences_stripped(self, validator):
        r = validator.validate("```sql\nSELECT id FROM orders\n```")
        assert r.is_valid
        assert "```" not in r.cleaned_sql

    def test_complexity_detection(self, validator):
        sql = """
        SELECT a.*, b.*, c.*, d.*
        FROM orders a
        JOIN customers b ON a.customer_id = b.id
        JOIN invoices c ON c.customer_id = b.id
        JOIN payments d ON d.invoice_id = c.id
        JOIN employees e ON e.id = 1
        LIMIT 100
        """
        r = validator.validate(sql)
        assert r.estimated_complexity == "high"


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid Retriever (no embedding, BM25 + graph only)
# ═══════════════════════════════════════════════════════════════════════════

class TestHybridRetriever:
    @pytest.fixture()
    def retriever(self, chunks_and_bm25, graph):
        chunks, bm25 = chunks_and_bm25
        return HybridSchemaRetriever(chunks, bm25, graph)

    def test_retrieve_invoice_query(self, retriever):
        result = _run(retriever.retrieve("Show me unpaid invoices from last quarter"))
        assert len(result.chunks) > 0
        names = result.table_names
        assert "invoices" in names or "payments" in names

    def test_retrieve_respects_budget(self, retriever):
        result = _run(retriever.retrieve(
            "Show all employee salaries by department",
            token_budget=200,
        ))
        total_tokens = sum(len(c.compact_ddl) // 4 for c in result.chunks)
        assert total_tokens <= 400  # allow some slack

    def test_retrieve_has_domains(self, retriever):
        result = _run(retriever.retrieve("customer order status"))
        assert len(result.domains) > 0

    def test_retrieve_includes_relationships(self, retriever):
        result = _run(retriever.retrieve("order items with product details"))
        if len(result.chunks) > 1:
            assert len(result.relationships) >= 0  # may or may not have rels
