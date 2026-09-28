"""Schema Engine Orchestrator — single entry-point for the runtime.

The orchestrator owns the full lifecycle:

    extract → summarize → index → retrieve → plan → validate

Storage hierarchy:
    - PostgreSQL: authoritative metadata, chunks, query logs, versions
    - Milvus: authoritative embeddings
    - Redis: distributed cache + locks
    - Memory: ephemeral hot cache

On restart the orchestrator loads all data from durable stores.
Cold rebuild (extract + summarize + embed) only happens when data is missing.

Usage from the runtime::

    orch = await SchemaEngineOrchestrator.for_server(
        server_name="my-db",
        mcp_manager=mcp_manager,
        server_tools=state.mcp_tools,
    )
    context_str = await orch.build_context(user_query)
    plan = await orch.plan_sql(user_query)
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

from ai_agent.schema_engine.config import get_config, SchemaEngineConfig
from ai_agent.schema_engine.graph.schema_graph import SchemaGraph
from ai_agent.schema_engine.indexing.schema_indexer import BM25Index, SchemaIndexer
from ai_agent.schema_engine.models import (
    DatabaseMetadata,
    RetrievalResult,
    SchemaChunk,
    SQLPlanResult,
    SQLValidationResult,
)
from ai_agent.schema_engine.planning.sql_planner import SQLPlanner
from ai_agent.schema_engine.retrieval.hybrid_retriever import HybridSchemaRetriever
from ai_agent.schema_engine.store.metadata_store import MetadataStore, get_metadata_store
from ai_agent.schema_engine.validation.sql_validator import SQLValidator, get_sql_validator

log = logging.getLogger(__name__)


class SchemaEngineError(Exception):
    """Structured application error from the schema engine.

    Attributes:
        error_code: Machine-readable error code (e.g. ``SCHEMA_METADATA_MISSING``).
        message: User-friendly error message.
    """

    def __init__(self, error_code: str, message: str):
        self.error_code = error_code
        self.message = message
        super().__init__(f"[{error_code}] {message}")


class SchemaEngineOrchestrator:
    """Facade that wires extract → index → retrieve → plan → validate."""

    def __init__(
        self,
        server_name: str,
        db_meta: DatabaseMetadata,
        chunks: List[SchemaChunk],
        bm25: BM25Index,
        graph: SchemaGraph,
        api_key: Optional[str] = None,
    ):
        self._cfg = get_config()
        self.server_name = server_name
        self._meta = db_meta
        self._chunks = chunks
        self._bm25 = bm25
        self._graph = graph
        self._api_key = api_key
        self._retriever = HybridSchemaRetriever(chunks, bm25, graph)
        self._planner = SQLPlanner(graph=graph, api_key=api_key)
        self._validator = get_sql_validator()

    # ── Factory (lazy init) ──────────────────────────────────────────────

    @classmethod
    async def for_server(
        cls,
        server_name: str,
        mcp_manager: Any,
        server_tools: List[Dict[str, Any]],
        *,
        agent_id: Optional[str] = None,
        force_refresh: bool = False,
        compute_embeddings: bool = True,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> "SchemaEngineOrchestrator":
        """Build (or load from durable stores) a fully-initialised orchestrator.

        ``agent_id`` is optional bookkeeping only — it records the
        ``is_cold_build`` flag on the agent row and triggers FAQ regeneration.
        All schema data is keyed by ``server_name``, so callers without an
        agent (e.g. org-scoped monitors) may pass None and still share the
        same index.

        Startup path (fast restart):
        1. Load metadata + chunks from PostgreSQL (via MetadataStore).
        2. Rebuild BM25 index and graph in-memory (deterministic, fast).
        3. Embeddings remain in Milvus — used at retrieval time.

        Cold build path (only when data is missing or force_refresh):
        1. Extract metadata via the MCP server's schema tool.
        2. Summarize tables (LLM batch call).
        3. Build chunks + BM25 index + embeddings.
        4. Build FK graph.
        5. Persist everything to PostgreSQL + Milvus.
        """
        start = time.monotonic()
        store = get_metadata_store()

        # ── Check if cold build is already done ───────────────────────
        cold_build_done = _check_cold_build_done(agent_id) if agent_id else False

        # ── Acquire distributed lock for this server ─────────────────
        lock = None
        try:
            from ai_agent.schema_engine.store.redis_cache import get_redis_cache
            redis_cache = get_redis_cache()
            lock = redis_cache.acquire_lock(
                f"schema_engine:init:{server_name}",
                timeout_seconds=120,
            )
        except Exception:
            pass  # Proceed without lock (single-instance mode)

        try:
            # ── Fast restart: load from durable storage ──────────────
            # If cold build is done, we ALWAYS load from durable.
            # If cold build is NOT done, we still try durable first
            if not force_refresh:
                cached = await _load_from_durable(store, server_name, api_key=api_key)
                if cached:
                    elapsed = (time.monotonic() - start) * 1000
                    log.info(
                        "SchemaEngine: fast restart for '%s' in %.0fms "
                        "(%d tables, %d chunks from durable storage)",
                        server_name, elapsed,
                        cached.table_count, cached.chunk_count,
                    )
                    return cached

                # Cold build was done before but data is missing from PG —
                # return a structured error instead of triggering another cold build.
                if cold_build_done:
                    log.error(
                        "SchemaEngine: cold build already completed for agent '%s' "
                        "but no data found in durable storage (store health: %s)",
                        agent_id, store.health(),
                    )
                    raise SchemaEngineError(
                        error_code="SCHEMA_METADATA_MISSING",
                        message=(
                            "Schema metadata is unavailable. "
                            "Please rebuild the schema from Settings."
                        ),
                    )

            # ── Cold build (only happens once per server) ────────────
            log.warning(
                "SchemaEngine: cold build triggered for '%s' (store health: %s)",
                server_name, store.health(),
            )

            # 1. Extract
            from ai_agent.schema_engine.extractors.mcp_extractor import MCPMetadataExtractor
            extractor = MCPMetadataExtractor(mcp_manager, server_tools)
            db_meta = await extractor.extract(server_name)

            # 2. Enrich enum/flag columns with distinct values (best-effort)
            try:
                from ai_agent.schema_engine.enrichment.enum_enricher import EnumEnricher
                enricher = EnumEnricher(mcp_manager, server_tools)
                db_meta = await enricher.enrich(db_meta, server_name)
            except Exception as e:
                log.warning("SchemaEngine: enum enrichment failed, continuing: %s", e)

            # 3. Summarize (best-effort; failures fall back to heuristic)
            try:
                from ai_agent.schema_engine.summarization.schema_summarizer import get_summarizer
                summarizer = get_summarizer()
                db_meta = await summarizer.summarize_database(
                    db_meta, api_key=api_key, model=model, provider=provider,
                )
            except Exception as e:
                log.warning("SchemaEngine: summarization failed, continuing: %s", e)

            # 4. Index (embeddings persisted to Milvus with incremental skip)
            indexer = SchemaIndexer()
            chunks, bm25 = await indexer.index_database(
                db_meta, compute_embeddings=compute_embeddings,
            )

            # 5. Graph
            graph = SchemaGraph(db_meta)

            # 6. Persist to PostgreSQL (authoritative)
            await store.put_metadata(db_meta)
            await store.put_chunks(server_name, chunks)

            # 7. Record schema version
            _record_version(server_name, db_meta)

            # 8. Mark cold build as done
            if agent_id:
                _mark_cold_build_done(agent_id)

            # 9. Run the schema classifier: categorize tables + generate sample
            # questions (best-effort — failures are logged and swallowed so
            # they never break cold build itself). Can also be re-triggered
            # manually via POST /agents/{agent_id}/classified_schema.
            try:
                from ai_agent.core.db import engine as db_engine
                from ai_agent.schema_engine.store.pg_repository import SchemaEngineRepository
                from ai_agent.services.schema_classifier_service import SchemaClassifierService
                from sqlmodel import Session as _Session

                classify_session = _Session(db_engine)
                try:
                    repo = SchemaEngineRepository(classify_session)
                    service = SchemaClassifierService()
                    categories = await service.save_classified_schema(
                        repo, server_name, db_meta, chunks, api_key=api_key,
                    )
                    repo.commit()
                    log.info(
                        "SchemaEngine: classified '%s' tables into %d categories",
                        server_name, len(categories),
                    )
                except Exception:
                    classify_session.rollback()
                    raise
                finally:
                    classify_session.close()
            except Exception as e:
                log.warning("SchemaEngine: schema classification failed, continuing: %s", e)

            elapsed = (time.monotonic() - start) * 1000
            log.info(
                "SchemaEngine: cold build for '%s' completed in %.0fms — "
                "%d tables, %d chunks",
                server_name, elapsed, db_meta.table_count, len(chunks),
            )
            return cls(server_name, db_meta, chunks, bm25, graph, api_key=api_key)

        finally:
            if lock:
                lock.release()

    # ── Public API ───────────────────────────────────────────────────────

    async def retrieve(
        self,
        user_query: str,
        *,
        top_k: Optional[int] = None,
        token_budget: Optional[int] = None,
    ) -> RetrievalResult:
        """Retrieve relevant schema chunks for a user query."""
        return await self._retriever.retrieve(
            user_query, top_k=top_k, token_budget=token_budget,
        )

    async def build_context(
        self,
        user_query: str,
        *,
        token_budget: Optional[int] = None,
    ) -> str:
        """Build a compact schema context string for LLM prompts.

        This is the main method the agent runtime should call to replace
        the old full-schema approach.
        """
        budget = token_budget or self._cfg.schema_context_max_tokens
        result = await self.retrieve(user_query, token_budget=budget)
        context_parts = [c.compact_ddl for c in result.chunks]
        context = "\n\n".join(context_parts)

        # Hard safety cap (chars)
        max_chars = budget * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "\n... (truncated)"

        return context

    async def plan_sql(
        self,
        user_query: str,
    ) -> SQLPlanResult:
        """Full two-stage pipeline: retrieve → identify → generate → validate."""
        retrieval = await self.retrieve(user_query)
        plan = await self._planner.plan(user_query, retrieval)

        # Validate the generated SQL
        validation = self._validator.validate(plan.sql, retrieval=retrieval)
        plan.is_valid = validation.is_valid
        plan.validation_errors = validation.errors
        plan.tables_used = validation.tables_referenced
        plan.columns_used = validation.columns_referenced

        if validation.cleaned_sql:
            plan.sql = validation.cleaned_sql

        log.info(
            "SchemaEngine plan_sql: valid=%s, tables=%s, errors=%s",
            plan.is_valid, plan.tables_used, plan.validation_errors,
        )
        return plan

    def validate(self, sql: str) -> SQLValidationResult:
        """Validate SQL without planning."""
        return self._validator.validate(sql)

    @property
    def table_count(self) -> int:
        return self._meta.table_count

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)


# ── Durable loader (fast restart) ────────────────────────────────────────

async def _load_from_durable(
    store: MetadataStore,
    server_name: str,
    *,
    api_key: Optional[str] = None,
) -> Optional[SchemaEngineOrchestrator]:
    """Reconstruct orchestrator from PostgreSQL + Milvus (no LLM calls).

    This is the fast restart path:
    1. Load metadata from PG (via MetadataStore L0).
    2. Load chunks from PG (via MetadataStore L0).
    3. Rebuild BM25 index in-memory (deterministic, fast).
    4. Rebuild FK graph in-memory (deterministic, fast).
    5. Embeddings stay in Milvus — used at retrieval time.
    """
    meta = await store.get_metadata(server_name)
    if meta is None:
        log.warning(
            "SchemaEngine: _load_from_durable — no metadata in PG/Redis/memory for '%s', "
            "store health=%s",
            server_name, store.health(),
        )
        return None
    chunks = await store.get_chunks(server_name)
    if chunks is None:
        log.warning(
            "SchemaEngine: _load_from_durable — metadata found (%d tables) but no chunks "
            "in PG/memory for '%s'",
            len(meta.tables), server_name,
        )
        return None
    log.debug(
        "SchemaEngine: _load_from_durable — loaded %d tables, %d chunks from durable store for '%s'",
        len(meta.tables), len(chunks), server_name,
    )

    # Rebuild ephemeral indexes (fast, deterministic)
    bm25 = BM25Index()
    bm25.build(chunks)
    graph = SchemaGraph(meta)

    return SchemaEngineOrchestrator(
        server_name=server_name,
        db_meta=meta,
        chunks=chunks,
        bm25=bm25,
        graph=graph,
        api_key=api_key,
    )


# ── Cold build flag helpers ─────────────────────────────────────────────

def _check_cold_build_done(agent_id: str) -> bool:
    """Check if cold build has been completed for this agent."""
    try:
        from ai_agent.core.db import engine
        from sqlmodel import Session
        from ai_agent.models import AIAgent

        session = Session(engine)
        try:
            agent = session.get(AIAgent, agent_id)
            return agent.is_cold_build if agent else False
        finally:
            session.close()
    except Exception as e:
        log.warning("SchemaEngine: failed to check is_cold_build flag: %s", e)
        return False


def _mark_cold_build_done(agent_id: str) -> None:
    """Mark cold build as done on ai_agents after successful completion."""
    try:
        from ai_agent.core.db import engine
        from sqlmodel import Session
        from ai_agent.models import AIAgent

        session = Session(engine)
        try:
            agent = session.get(AIAgent, agent_id)
            if agent:
                agent.is_cold_build = True
                session.add(agent)
                session.commit()
                log.info("SchemaEngine: marked is_cold_build=True for agent '%s'", agent_id)
                
                # Trigger FAQ regeneration after cold build completes
                _trigger_faq_regeneration(agent_id, session)
            else:
                log.warning("SchemaEngine: agent '%s' not found, cannot mark cold build done", agent_id)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    except Exception as e:
        log.warning("SchemaEngine: failed to mark is_cold_build flag: %s", e)


def _trigger_faq_regeneration(agent_id: str, session: "Session") -> None:
    """Trigger FAQ regeneration after schema changes (cold build completion or MCP updates).
    
    This is called when:
    1. DB MCP cold build completes (schema now available)
    2. MCP configuration changes
    
    FAQs are regenerated to reflect actual DB schema + API tools instead of just system prompt.
    """
    try:
        import asyncio
        from ai_agent.workflows.faq.faq import generate_faqs
        
        # Use asyncio.run to execute async FAQ generation in sync context
        log.info("Triggering FAQ regeneration for agent '%s' after schema/MCP changes", agent_id)
        asyncio.run(generate_faqs(agent_id, session, force_refresh=True))
        log.info("FAQ regeneration completed for agent '%s'", agent_id)
    except Exception as e:
        log.warning("FAQ regeneration failed for agent '%s': %s", agent_id, e)


# ── Schema versioning ───────────────────────────────────────────────────

def _record_version(server_name: str, db_meta: DatabaseMetadata) -> None:
    """Record a schema version snapshot to PostgreSQL."""
    try:
        from ai_agent.core.db import engine
        from sqlmodel import Session
        from ai_agent.schema_engine.store.pg_repository import SchemaEngineRepository

        session = Session(engine)
        try:
            repo = SchemaEngineRepository(session)
            db_row = repo.get_database(server_name)
            if not db_row:
                return

            version_hash = db_meta.version_hash or hashlib.md5(
                str(db_meta.table_count).encode()
            ).hexdigest()[:16]

            prev = repo.get_latest_version(db_row.id)
            prev_count = prev.table_count if prev else 0

            repo.record_version(
                database_id=db_row.id,
                version_hash=version_hash,
                table_count=db_meta.table_count,
                tables_added=max(0, db_meta.table_count - prev_count),
                tables_removed=max(0, prev_count - db_meta.table_count),
            )
            repo.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()
    except Exception as e:
        log.warning("SchemaEngine: version recording failed: %s", e)


# ── Per-server singleton cache (non-authoritative) ───────────────────────

_orchestrators: Dict[str, SchemaEngineOrchestrator] = {}


async def get_orchestrator(
    server_name: str,
    mcp_manager: Any,
    server_tools: List[Dict[str, Any]],
    *,
    agent_id: Optional[str] = None,
    force_refresh: bool = False,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> SchemaEngineOrchestrator:
    """Get or create the orchestrator for a given MCP server.

    The in-memory _orchestrators dict is a non-authoritative cache.
    All data is backed by PostgreSQL + Milvus.

    ``agent_id`` is optional — see ``for_server``. The cache and every durable
    store are keyed by ``server_name``, so an agent-less caller reuses whatever
    index has already been built for that server.
    """
    if server_name in _orchestrators and not force_refresh:
        # Update api_key on cached orchestrator (may change between runs)
        _orchestrators[server_name]._api_key = api_key
        _orchestrators[server_name]._planner._api_key = api_key
        return _orchestrators[server_name]

    orch = await SchemaEngineOrchestrator.for_server(
        server_name=server_name,
        mcp_manager=mcp_manager,
        server_tools=server_tools,
        agent_id=agent_id,
        force_refresh=force_refresh,
        api_key=api_key,
        model=model,
        provider=provider,
    )
    _orchestrators[server_name] = orch
    return orch


async def prewarm_all_db_servers(mcp_manager: Any) -> None:
    """Pre-warm the schema engine for every DB-aware MCP server already
    linked to an agent, at app startup.

    Without this, the first DB-backed chat request after every restart/
    reload pays the schema engine's "load from durable storage" cost
    (Postgres read + in-memory BM25/graph rebuild) instead of it happening
    here, invisibly, at boot — same rationale as pre-loading the guardrails
    service. Servers that have never been cold-built yet still get built
    lazily on first use, same as today; this only front-loads work for
    servers that already have durable data (the common case).

    Fully best-effort: a failure for one server, or for this whole step,
    must never block app startup — callers should not wrap this in their
    own try/except for that reason.
    """
    try:
        from ai_agent.core.db import engine as db_engine
        from sqlmodel import Session, select
        from ai_agent.models import MCPServerConfig, AgentMCPConfigLink

        session = Session(db_engine)
        try:
            rows = session.exec(
                select(MCPServerConfig)
                .join(
                    AgentMCPConfigLink,
                    AgentMCPConfigLink.mcp_server_config_id == MCPServerConfig.id,
                )
                .where(MCPServerConfig.requires_query_building == True)  # noqa: E712
            ).all()
            server_names = {row.name for row in rows}
        finally:
            session.close()

        for server_name in server_names:
            try:
                await get_orchestrator(
                    server_name=server_name,
                    mcp_manager=mcp_manager,
                    server_tools=[],
                )
                log.info("Schema engine pre-warmed for server '%s'", server_name)
            except Exception as e:
                log.warning(
                    "Schema engine pre-warm failed for '%s' (non-fatal, will "
                    "load lazily on first request instead): %s",
                    server_name, e,
                )
    except Exception as e:
        log.warning("Schema engine pre-warm step failed entirely (non-fatal): %s", e)
