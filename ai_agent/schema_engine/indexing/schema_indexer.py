"""Build searchable ``SchemaChunk`` objects from ``DatabaseMetadata``.

This module is the **single entry-point** for turning raw metadata into
indexed, search-ready chunks.  It:

1. Creates one ``SchemaChunk`` per table (table-level chunking).
2. Builds a BM25-friendly ``search_text`` field.
3. Builds a ``summary_text`` field for embedding.
4. Generates embeddings in batches via ``LLMGateway.embed``.
5. Stores everything in the ``MetadataStore``.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Set

from ai_agent.schema_engine.config import get_config
from ai_agent.schema_engine.models import (
    DatabaseMetadata,
    SchemaChunk,
    TableMetadata,
)

log = logging.getLogger(__name__)


# Columns whose name suggests they identify a business entity rather than
# describe one. Used only to hint COUNT(DISTINCT ...) candidates when a table
# declares no primary key — a hint, never a correctness guarantee.
_BUSINESS_KEY_RE = re.compile(
    r"(^|_)(id|no|num|number|code|key|ref|reference|doc|document|"
    r"order|requisition|invoice|ticket|job|batch|serial)s?($|_)",
    re.IGNORECASE,
)


def _looks_like_business_key(column_name: str) -> bool:
    """True when a column name reads like an entity identifier."""
    return bool(_BUSINESS_KEY_RE.search(column_name))


# ═══════════════════════════════════════════════════════════════════════════
# BM25 (in-process, no external dependency)
# ═══════════════════════════════════════════════════════════════════════════

class BM25Index:
    """Lightweight Okapi-BM25 implementation operating over ``SchemaChunk``."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: List[SchemaChunk] = []
        self._doc_freqs: Counter = Counter()
        self._doc_lens: List[int] = []
        self._avg_dl: float = 0.0
        self._token_lists: List[List[str]] = []
        self._n: int = 0

    def build(self, chunks: List[SchemaChunk]) -> None:
        self._chunks = chunks
        self._n = len(chunks)
        self._token_lists = [self._tokenize(c.search_text) for c in chunks]
        self._doc_lens = [len(tl) for tl in self._token_lists]
        self._avg_dl = sum(self._doc_lens) / max(self._n, 1)

        self._doc_freqs = Counter()
        for tl in self._token_lists:
            for token in set(tl):
                self._doc_freqs[token] += 1

    def query(self, text: str, top_k: int = 10) -> List[tuple[float, SchemaChunk]]:
        tokens = self._tokenize(text)
        scores: List[float] = []
        for idx in range(self._n):
            score = 0.0
            tf_map = Counter(self._token_lists[idx])
            dl = self._doc_lens[idx]
            for t in tokens:
                if t not in self._doc_freqs:
                    continue
                df = self._doc_freqs[t]
                idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)
                tf = tf_map.get(t, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1))
                score += idf * numerator / denominator
            scores.append(score)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(s, self._chunks[i]) for i, s in ranked[:top_k] if s > 0]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())


# ═══════════════════════════════════════════════════════════════════════════
# Schema Indexer (orchestrator)
# ═══════════════════════════════════════════════════════════════════════════

class SchemaIndexer:
    """Build ``SchemaChunk`` objects + BM25 index + embeddings from metadata."""

    def __init__(self):
        self._cfg = get_config()

    async def index_database(
        self, db: DatabaseMetadata, *, compute_embeddings: bool = True,
    ) -> tuple[List[SchemaChunk], BM25Index]:
        """Create chunks, build BM25, optionally embed.

        Returns (chunks, bm25_index).
        """
        chunks = self._build_chunks(db)
        log.info("SchemaIndexer: built %d chunks for '%s'", len(chunks), db.server_name)

        # BM25 index
        bm25 = BM25Index()
        bm25.build(chunks)

        # Embeddings (persisted to Milvus with incremental skip)
        if compute_embeddings:
            await self._embed_chunks(
                chunks, server_name=db.server_name,
            )

        return chunks, bm25

    # ── Chunk building ───────────────────────────────────────────────────

    def _build_chunks(self, db: DatabaseMetadata) -> List[SchemaChunk]:
        chunks: List[SchemaChunk] = []
        cfg = self._cfg

        for fqn, table in db.tables.items():
            # search_text: aggressive keyword blob for BM25
            search_tokens = self._build_search_tokens(table)
            search_text = " ".join(search_tokens)

            # summary_text: human-readable for embedding
            summary_text = self._build_summary_text(table)

            # compact DDL for LLM context
            compact_ddl = self._build_compact_ddl(table, cfg.column_budget_per_table)

            # related tables from FKs
            related = list({fk.target_table for fk in table.foreign_keys})

            chunk = SchemaChunk(
                chunk_id=f"{db.server_name}:{fqn}",
                server_name=db.server_name,
                table_fqn=fqn,
                table_name=table.table_name,
                schema_name=table.schema_name,
                search_text=search_text,
                summary_text=summary_text,
                compact_ddl=compact_ddl,
                related_tables=related,
                domain=table.domain,
                column_names=table.column_names,
                primary_keys=table.primary_keys,
                row_count=table.row_count,
                foreign_key_count=len(table.foreign_keys),
            )
            chunks.append(chunk)

        return chunks

    @staticmethod
    def _build_search_tokens(table: TableMetadata) -> List[str]:
        """Produce a flat list of searchable tokens for BM25."""
        tokens: List[str] = []
        # Table name (with underscores split)
        tokens.extend(table.table_name.lower().split("_"))
        tokens.append(table.table_name.lower())
        # Schema
        tokens.append(table.schema_name.lower())
        # Column names
        for c in table.columns:
            tokens.extend(c.name.lower().split("_"))
            tokens.append(c.name.lower())
        # Summary / description
        if table.summary:
            tokens.extend(re.findall(r"[a-z0-9]+", table.summary.lower()))
        if table.description:
            tokens.extend(re.findall(r"[a-z0-9]+", table.description.lower()))
        # Semantic keywords from summarization
        tokens.extend(kw.lower() for kw in table.semantic_keywords)
        # FK target table names
        for fk in table.foreign_keys:
            tokens.extend(fk.target_table.lower().split("_"))
        return tokens

    @staticmethod
    def _build_summary_text(table: TableMetadata) -> str:
        """Human-readable description for embedding."""
        parts = [f"Table {table.fqn}"]
        if table.summary:
            parts.append(table.summary)
        elif table.table_comment:
            parts.append(table.table_comment)
        else:
            parts.append(f"Contains columns: {', '.join(table.column_names[:15])}")
        if table.domain:
            parts.append(f"Domain: {table.domain}")
        if table.semantic_keywords:
            parts.append(f"Keywords: {', '.join(table.semantic_keywords[:10])}")
        return ". ".join(parts)

    @staticmethod
    def _build_compact_ddl(table: TableMetadata, col_budget: int) -> str:
        """Compact one-block DDL for injecting into LLM context."""
        cols: List[str] = []
        audit_cols: List[str] = []
        cfg = get_config()
        audit_patterns = set(cfg.audit_column_patterns)

        for c in table.columns[:col_budget]:
            if c.name.lower() in audit_patterns:
                audit_cols.append(c.name)
                continue
            annotation = ""
            if c.is_primary_key:
                annotation = " PK"
            elif c.is_foreign_key:
                annotation = " FK"
            col_str = f"{c.name}:{c.data_type}{annotation}"
            if c.sample_values:
                shown = c.sample_values[:15]
                # Quote each value (SQL-escaping embedded quotes) so the LLM
                # sees the exact literal — including empty strings ('') which
                # would otherwise be invisible.
                vals = ", ".join(
                    "'" + str(v).replace("'", "''") + "'" for v in shown
                )
                if len(c.sample_values) > 15:
                    vals += ", ..."
                col_str += f" [possible_values: {vals}]"
            cols.append(col_str)

        remainder = len(table.columns) - col_budget
        if remainder > 0:
            cols.append(f"...+{remainder} cols")
        if audit_cols:
            cols.append(f"[audit: {','.join(audit_cols)}]")

        fk_hints = []
        for fk in table.foreign_keys:
            fk_hints.append(f"{fk.source_column}->{fk.target_table}.{fk.target_column}")

        header = table.table_name
        if table.summary:
            header += f" -- {table.summary[:80]}"

        lines = [header, f"  ({', '.join(cols)})"]
        if fk_hints:
            lines.append(f"  FK: {'; '.join(fk_hints)}")

        # Grain. Without this the model assumes one row == one business entity
        # and writes COUNT(*) where COUNT(DISTINCT <key>) is meant. Flat
        # warehouse/ERP extracts routinely carry no PK at all, so the absence
        # of one is itself the signal worth stating out loud.
        if table.row_count is not None:
            lines.append(f"  ROWS: ~{table.row_count}")
        if table.primary_keys:
            lines.append(
                f"  GRAIN: one row per ({', '.join(table.primary_keys)})"
            )
        else:
            id_like = [
                c.name for c in table.columns[:col_budget]
                if _looks_like_business_key(c.name)
            ]
            hint = (
                "  GRAIN: no primary key declared — one row is NOT one business "
                "entity. This is typically a line-item / event level extract, so "
                "an identifier column repeats across rows."
            )
            if id_like:
                hint += (
                    " Count entities with COUNT(DISTINCT <col>) using: "
                    + ", ".join(id_like[:6])
                )
            lines.append(hint)
        return "\n".join(lines)

    # ── Embedding (Milvus-backed, incremental) ──────────────────────────

    async def _embed_chunks(
        self,
        chunks: List[SchemaChunk],
        *,
        server_name: str = "",
    ) -> None:
        """Generate embeddings and persist to Milvus.

        Skips chunks whose summary_text hash matches the version already
        stored in Milvus (incremental embedding).
        """
        import hashlib
        from ai_agent.schema_engine.schema_embedding_client import embed_texts

        cfg = self._cfg
        batch_size = cfg.embedding_batch_size

        # ── Load existing embedding versions from Milvus ─────────────
        existing_versions: Dict[str, str] = {}
        vector_store = None
        try:
            from ai_agent.schema_engine.store.vector_store import get_vector_store
            vector_store = get_vector_store()
            if vector_store.connect():
                existing_versions = vector_store.get_existing_versions(server_name)
                log.info(
                    "SchemaIndexer: %d existing embeddings in Milvus for '%s'",
                    len(existing_versions), server_name,
                )
        except Exception as e:
            log.error("SchemaIndexer: Milvus not available, embedding to memory only: %s", e, exc_info=True)

        # ── Determine which chunks need (re-)embedding ───────────────
        def _text_hash(text: str) -> str:
            return hashlib.md5(text.encode()).hexdigest()[:16]

        chunks_to_embed: List[SchemaChunk] = []
        for chunk in chunks:
            version = _text_hash(chunk.summary_text)
            chunk_id = chunk.table_fqn
            if existing_versions.get(chunk_id) == version:
                # Embedding is current — skip
                continue
            chunks_to_embed.append(chunk)

        if not chunks_to_embed:
            log.info("SchemaIndexer: all %d chunks have current embeddings, skipping", len(chunks))
            return

        log.info(
            "SchemaIndexer: embedding %d / %d chunks (incremental)",
            len(chunks_to_embed), len(chunks),
        )

        # ── Generate embeddings via the configured schema-engine model ──
        texts = [c.summary_text for c in chunks_to_embed]
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                all_embeddings.extend(await embed_texts(cfg, batch))
            except Exception as e:
                log.warning("SchemaIndexer: embedding batch %d failed: %s", i, e)
                all_embeddings.extend([[] for _ in batch])

        # ── Persist to Milvus ────────────────────────────────────────
        milvus_entries = []
        for idx, chunk in enumerate(chunks_to_embed):
            if idx >= len(all_embeddings) or not all_embeddings[idx]:
                continue
            embedding = all_embeddings[idx]
            chunk.embedding = embedding  # Also set in memory for current session
            version = _text_hash(chunk.summary_text)

            milvus_entries.append({
                "chunk_id": chunk.table_fqn,
                "embedding": embedding,
                "server_name": server_name,
                "table_name": chunk.table_name,
                "domain": chunk.domain or "general",
                "embedding_type": "table_summary",
                "version_hash": version,
            })

        if vector_store and vector_store.is_connected and milvus_entries:
            try:
                ids = vector_store.upsert_embeddings(milvus_entries)
                log.info("SchemaIndexer: persisted %d embeddings to Milvus", len(ids))
            except Exception as e:
                log.error("SchemaIndexer: Milvus upsert failed: %s", e)

        embedded_count = sum(1 for c in chunks if c.embedding)
        log.info(
            "SchemaIndexer: embedded %d / %d chunks (%d new)",
            embedded_count, len(chunks), len(chunks_to_embed),
        )
