"""Hybrid schema retriever: BM25 + vector + graph.

This is the **main entry-point** for schema retrieval.  Given a user query it:

1. Runs BM25 (lexical) search over ``SchemaChunk.search_text``.
2. Runs vector (semantic) search over ``SchemaChunk.embedding``.
3. Expands results via the FK graph (graph signal).
4. Fuses scores with configurable weights.
5. Prunes to a token budget.
6. Returns a ``RetrievalResult`` ready for the planning stage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Dict, List, Optional, Set

from ai_agent.schema_engine.config import get_config
from ai_agent.schema_engine.graph.schema_graph import SchemaGraph
from ai_agent.schema_engine.indexing.schema_indexer import BM25Index
from ai_agent.schema_engine.models import RetrievalResult, SchemaChunk

log = logging.getLogger(__name__)


class HybridSchemaRetriever:
    """Combine lexical, semantic, and graph signals for schema retrieval."""

    def __init__(
        self,
        chunks: List[SchemaChunk],
        bm25_index: BM25Index,
        graph: SchemaGraph,
    ):
        self._cfg = get_config()
        self._chunks = chunks
        self._chunk_map: Dict[str, SchemaChunk] = {c.chunk_id: c for c in chunks}
        self._table_to_chunk: Dict[str, SchemaChunk] = {c.table_name: c for c in chunks}
        self._bm25 = bm25_index
        self._graph = graph

    @staticmethod
    def _query_cache_key(server_name: str, user_query: str) -> str:
        """Deterministic cache key for a (server, query) pair."""
        raw = f"{server_name}\x00{user_query.strip().lower()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    async def retrieve(
        self,
        user_query: str,
        *,
        top_k: Optional[int] = None,
        token_budget: Optional[int] = None,
    ) -> RetrievalResult:
        """Run the full hybrid pipeline and return pruned results."""
        cfg = self._cfg
        top_k = top_k or cfg.retrieval_top_k
        token_budget = token_budget or cfg.schema_context_max_tokens

        server_name = self._chunks[0].server_name if self._chunks else ""
        cache_key = self._query_cache_key(server_name, user_query)

        # ── Cache read (Redis) ────────────────────────────────────────
        cached = self._read_retrieval_cache(server_name, cache_key)
        if cached is not None:
            log.debug(
                "HybridRetriever: cache HIT for server='%s' key=%s (%d tables)",
                server_name, cache_key[:8], len(cached.chunks),
            )
            return cached

        # Stage A: BM25
        bm25_hits = self._bm25.query(user_query, top_k=top_k * 2)
        bm25_scores: Dict[str, float] = {}
        if bm25_hits:
            max_bm25 = max(s for s, _ in bm25_hits) or 1.0
            bm25_scores = {c.chunk_id: s / max_bm25 for s, c in bm25_hits}

        # Stage B: Vector (cosine similarity)
        vector_scores = await self._vector_search(user_query, top_k * 2)

        # Collect candidate chunk IDs from both signals
        candidate_ids: Set[str] = set(bm25_scores.keys()) | set(vector_scores.keys())

        # Stage C: Graph expansion on top-scoring seeds
        seed_tables = self._top_seed_tables(bm25_scores, vector_scores, n=5)
        expanded = self._graph.expand_neighbours(seed_tables, max_hops=cfg.graph_auto_expand_hops)
        for tname in expanded:
            chunk = self._table_to_chunk.get(tname)
            if chunk:
                candidate_ids.add(chunk.chunk_id)

        # Fuse scores
        fused: List[tuple[float, SchemaChunk]] = []
        for cid in candidate_ids:
            chunk = self._chunk_map.get(cid)
            if not chunk:
                continue
            bm = bm25_scores.get(cid, 0.0) * cfg.bm25_weight
            vs = vector_scores.get(cid, 0.0) * cfg.vector_weight
            gs = self._graph.score_expansion(seed_tables, chunk.table_name) * cfg.graph_weight
            fused.append((bm + vs + gs, chunk))

        fused.sort(key=lambda x: x[0], reverse=True)

        # Prune to token budget
        selected = self._prune_to_budget(fused, token_budget)

        # Collect domains and relationships
        domains = list({c.domain for c in selected if c.domain})
        relationships: List[str] = []
        selected_names = [c.table_name for c in selected]
        for c in selected:
            for rt in c.related_tables:
                if rt in selected_names:
                    relationships.append(f"{c.table_name} <-> {rt}")

        confidence = fused[0][0] if fused else 0.0
        token_est = sum(len(c.compact_ddl) // 4 for c in selected)

        result = RetrievalResult(
            chunks=selected,
            domains=domains,
            relationships=list(set(relationships)),
            confidence=min(confidence, 1.0),
            token_estimate=token_est,
        )
        log.info(
            "HybridRetriever: %d tables, %d domains, ~%d tokens, confidence=%.2f",
            len(selected), len(domains), token_est, result.confidence,
        )

        # ── Persist retrieval audit (best-effort) ────────────────────
        self._record_retrieval_audit(
            user_query, selected, bm25_scores, vector_scores, token_est,
        )

        # ── Cache write (Redis, TTL = 1 h) ───────────────────────────
        self._write_retrieval_cache(server_name, cache_key, result)

        return result

    # ── Retrieval cache helpers ────────────────────────────────────────

    def _read_retrieval_cache(
        self, server_name: str, cache_key: str,
    ) -> Optional[RetrievalResult]:
        """Return a cached RetrievalResult or None on miss/error."""
        try:
            from ai_agent.schema_engine.store.redis_cache import get_redis_cache
            rc = get_redis_cache()
            if not rc.is_connected:
                return None
            raw = rc.get_cached_retrieval(server_name, cache_key)
            if raw is None:
                return None
            return RetrievalResult.from_dict(json.loads(raw))
        except Exception as e:
            log.debug("HybridRetriever: cache read error (non-critical): %s", e)
            return None

    def _write_retrieval_cache(
        self, server_name: str, cache_key: str, result: RetrievalResult,
    ) -> None:
        """Persist a RetrievalResult to Redis. Best-effort, never raises."""
        try:
            from ai_agent.schema_engine.store.redis_cache import get_redis_cache
            rc = get_redis_cache()
            if not rc.is_connected:
                return
            rc.cache_retrieval_result(
                server_name, cache_key, json.dumps(result.to_dict()), ttl_seconds=3600,
            )
        except Exception as e:
            log.debug("HybridRetriever: cache write error (non-critical): %s", e)

    # ── Vector search (Milvus-first, in-memory fallback) ────────────────

    async def _vector_search(
        self, query: str, top_k: int,
    ) -> Dict[str, float]:
        """Semantic search: Milvus → in-memory fallback."""
        # Embed the user query
        try:
            from ai_agent.schema_engine.schema_embedding_client import embed_texts
            cfg = self._cfg
            vectors = await embed_texts(cfg, [query])
            q_emb = vectors[0]
        except Exception as e:
            log.warning("HybridRetriever: query embedding failed: %s", e)
            return {}

        # ── Primary: Milvus vector search ────────────────────────────
        try:
            from ai_agent.schema_engine.store.vector_store import get_vector_store
            vs = get_vector_store()
            if vs.is_connected or vs.connect():
                server_name = self._chunks[0].server_name if self._chunks else ""
                hits = vs.search(q_emb, server_name=server_name, top_k=top_k)
                if hits:
                    scores: Dict[str, float] = {}
                    for hit in hits:
                        # Map chunk_id (table_fqn) back to our chunk_map
                        table_fqn = hit.get("chunk_id", "")
                        for chunk in self._chunks:
                            if chunk.table_fqn == table_fqn:
                                scores[chunk.chunk_id] = hit.get("score", 0.0)
                                break
                    if scores:
                        log.debug("HybridRetriever: Milvus returned %d hits", len(scores))
                        return scores
        except Exception as e:
            log.debug("HybridRetriever: Milvus search unavailable, falling back: %s", e)

        # ── Fallback: in-memory cosine similarity ────────────────────
        embedded = [c for c in self._chunks if c.embedding]
        if not embedded:
            return {}

        scores = {}
        for chunk in embedded:
            sim = self._cosine_sim(q_emb, chunk.embedding)
            if sim > 0.0:
                scores[chunk.chunk_id] = sim
        return scores

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _top_seed_tables(
        self,
        bm25_scores: Dict[str, float],
        vector_scores: Dict[str, float],
        n: int = 5,
    ) -> List[str]:
        """Pick the best tables from BM25+vector to seed graph expansion."""
        combined: Dict[str, float] = {}
        for cid, s in bm25_scores.items():
            combined[cid] = combined.get(cid, 0) + s
        for cid, s in vector_scores.items():
            combined[cid] = combined.get(cid, 0) + s
        top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:n]
        result = []
        for cid, _ in top:
            chunk = self._chunk_map.get(cid)
            if chunk:
                result.append(chunk.table_name)
        return result

    @staticmethod
    def _prune_to_budget(
        ranked: List[tuple[float, SchemaChunk]],
        token_budget: int,
    ) -> List[SchemaChunk]:
        """Greedily add chunks until the token budget is exhausted."""
        selected: List[SchemaChunk] = []
        used_tokens = 0
        for _, chunk in ranked:
            chunk_tokens = len(chunk.compact_ddl) // 4  # ~4 chars/token
            if used_tokens + chunk_tokens > token_budget and selected:
                break
            selected.append(chunk)
            used_tokens += chunk_tokens
        return selected

    def _record_retrieval_audit(
        self,
        user_query: str,
        selected: List[SchemaChunk],
        bm25_scores: Dict[str, float],
        vector_scores: Dict[str, float],
        token_est: int,
    ) -> None:
        """Best-effort persist retrieval metrics to PostgreSQL."""
        try:
            import json
            from ai_agent.core.db import engine
            from sqlmodel import Session
            from ai_agent.schema_engine.store.pg_repository import SchemaEngineRepository

            server_name = self._chunks[0].server_name if self._chunks else ""
            session = Session(engine)
            try:
                repo = SchemaEngineRepository(session)
                db_row = repo.get_database(server_name)
                if not db_row:
                    return
                repo.record_retrieval({
                    "database_id": db_row.id,
                    "user_query": user_query,
                    "tables_retrieved_json": json.dumps(
                        [c.table_name for c in selected]
                    ),
                    "retrieval_method": "hybrid",
                    "bm25_score": max(bm25_scores.values()) if bm25_scores else 0.0,
                    "vector_score": max(vector_scores.values()) if vector_scores else 0.0,
                    "total_chunks": len(selected),
                    "token_budget_used": token_est,
                })
                repo.commit()
            except Exception:
                session.rollback()
            finally:
                session.close()
        except Exception as e:
            log.debug("HybridRetriever: audit log failed (non-critical): %s", e)
