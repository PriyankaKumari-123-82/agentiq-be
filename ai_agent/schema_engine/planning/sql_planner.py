"""Two-stage SQL planning pipeline.

Stage 1 – Schema Identification
    Given the user query and retrieved schema chunks, the LLM identifies
    relevant tables, columns, joins, and ambiguities.  **No SQL is generated.**

Stage 2 – SQL Generation
    Using only the pruned schema context from Stage 1 (validated tables
    and columns), the LLM generates the actual SQL query.

This separation prevents hallucinated table/column references and keeps
token usage minimal.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from ai_agent.schema_engine.config import get_config
from ai_agent.schema_engine.graph.schema_graph import SchemaGraph
from ai_agent.schema_engine.models import (
    RetrievalResult,
    SchemaChunk,
    SchemaIdentificationResult,
    SQLPlanResult,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Prompt templates
# ═══════════════════════════════════════════════════════════════════════════

_STAGE1_PROMPT = """\
You are a database schema analyst. Given the user's question and the available
database schema, identify ONLY the relevant tables, columns, and joins needed
to answer the question.

**DO NOT generate SQL.** Only identify the schema elements.

Available Schema:
{schema_context}

User Question: {user_query}

Respond with a JSON object (no markdown fences):
{{
  "tables": ["table1", "table2"],
  "columns": ["table1.col1", "table2.col2"],
  "joins": ["table1.fk_col = table2.pk_col"],
  "ambiguities": ["any unclear references or assumptions"],
  "reasoning": "brief explanation of your choices"
}}"""


_STAGE2_PROMPT = """\
You are a SQL expert. Generate a read-only SQL query to answer the user's
question using ONLY the tables and columns listed below.

Validated Schema (use ONLY these):
{pruned_schema}

Join paths:
{join_paths}

User Question: {user_query}

Rules:
- ONLY use SELECT or WITH … SELECT.
- NEVER use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, or any DDL/DML.
- Include a LIMIT clause (max {max_rows} rows) unless counting/aggregating.
- Use explicit table aliases for clarity.
- Reference only the columns listed above.

Respond with ONLY the SQL query, no markdown fences, no commentary."""


# ═══════════════════════════════════════════════════════════════════════════
# SQL Planner
# ═══════════════════════════════════════════════════════════════════════════

class SQLPlanner:
    """Orchestrates the two-stage planning pipeline."""

    def __init__(self, graph: Optional[SchemaGraph] = None, api_key: Optional[str] = None):
        self._cfg = get_config()
        self._graph = graph
        self._api_key = api_key

    async def plan(
        self,
        user_query: str,
        retrieval: RetrievalResult,
    ) -> SQLPlanResult:
        """Run Stage 1 + Stage 2 and return a SQL plan."""
        # Stage 1: Schema Identification
        identification = await self._stage1_identify(user_query, retrieval)

        # Validate identified tables exist in retrieved chunks
        valid_tables = self._validate_tables(identification, retrieval)

        # Build pruned schema context from validated tables only
        pruned_schema = self._build_pruned_schema(valid_tables, identification, retrieval)

        # Discover join paths
        join_paths = self._discover_joins(valid_tables, identification)

        # Stage 2: SQL Generation
        sql_result = await self._stage2_generate(user_query, pruned_schema, join_paths)

        return sql_result

    # ── Stage 1 ──────────────────────────────────────────────────────────

    async def _stage1_identify(
        self, user_query: str, retrieval: RetrievalResult,
    ) -> SchemaIdentificationResult:
        from ai_agent.core.llm_gateway import get_llm_gateway
        llm = get_llm_gateway()

        schema_context = "\n".join(c.compact_ddl for c in retrieval.chunks)

        prompt = _STAGE1_PROMPT.format(
            schema_context=schema_context,
            user_query=user_query,
        )

        resp = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
            api_key=self._api_key,
        )

        try:
            data = json.loads(resp.content)
            result = SchemaIdentificationResult(
                tables=data.get("tables", []),
                columns=data.get("columns", []),
                joins=data.get("joins", []),
                ambiguities=data.get("ambiguities", []),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, AttributeError):
            log.warning("Stage1: failed to parse JSON, using all retrieved tables")
            result = SchemaIdentificationResult(
                tables=retrieval.table_names,
                reasoning="Parse failure – using all retrieved tables.",
            )

        log.info(
            "Stage1: identified %d tables, %d columns, %d joins",
            len(result.tables), len(result.columns), len(result.joins),
        )
        return result

    # ── Validation ───────────────────────────────────────────────────────

    @staticmethod
    def _validate_tables(
        identification: SchemaIdentificationResult,
        retrieval: RetrievalResult,
    ) -> List[SchemaChunk]:
        """Keep only tables that actually exist in the retrieved set."""
        retrieved_map = {c.table_name.lower(): c for c in retrieval.chunks}
        # Also map by fqn
        for c in retrieval.chunks:
            retrieved_map[c.table_fqn.lower()] = c

        valid: List[SchemaChunk] = []
        for tname in identification.tables:
            chunk = retrieved_map.get(tname.lower())
            if chunk:
                valid.append(chunk)
            else:
                log.warning("Stage1 referenced non-retrieved table: %s (ignored)", tname)
        return valid

    # ── Pruned schema ────────────────────────────────────────────────────

    @staticmethod
    def _build_pruned_schema(
        valid_chunks: List[SchemaChunk],
        identification: SchemaIdentificationResult,
        retrieval: RetrievalResult,
    ) -> str:
        """Build compact schema containing only validated tables + columns."""
        parts: List[str] = []
        id_cols = {c.lower() for c in identification.columns}

        for chunk in valid_chunks:
            # If columns were specified, filter
            if id_cols:
                relevant_cols = [
                    cn for cn in chunk.column_names
                    if f"{chunk.table_name.lower()}.{cn.lower()}" in id_cols
                    or cn.lower() in id_cols
                ]
                # Always include PKs
                for pk in chunk.primary_keys:
                    if pk not in relevant_cols:
                        relevant_cols.append(pk)
                # If none matched, show all (safety fallback)
                if not relevant_cols:
                    relevant_cols = chunk.column_names

                cols_str = ", ".join(relevant_cols[:30])
                parts.append(f"{chunk.table_name}({cols_str})")
            else:
                parts.append(chunk.compact_ddl)

        return "\n".join(parts)

    # ── Join discovery ───────────────────────────────────────────────────

    def _discover_joins(
        self,
        valid_chunks: List[SchemaChunk],
        identification: SchemaIdentificationResult,
    ) -> str:
        """Produce join hints from the graph or Stage-1 output."""
        lines: List[str] = []
        # Use Stage-1 joins first
        if identification.joins:
            lines.extend(identification.joins)

        # Augment from graph
        if self._graph and len(valid_chunks) > 1:
            names = [c.table_name for c in valid_chunks]
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    path = self._graph.shortest_path(names[i], names[j])
                    if path and path.hops > 0:
                        for sql_join in path.to_sql_joins():
                            if sql_join not in lines:
                                lines.append(sql_join)

        return "\n".join(lines) if lines else "No explicit joins needed."

    # ── Stage 2 ──────────────────────────────────────────────────────────

    async def _stage2_generate(
        self, user_query: str, pruned_schema: str, join_paths: str,
    ) -> SQLPlanResult:
        from ai_agent.core.llm_gateway import get_llm_gateway
        llm = get_llm_gateway()

        prompt = _STAGE2_PROMPT.format(
            pruned_schema=pruned_schema,
            join_paths=join_paths,
            user_query=user_query,
            max_rows=self._cfg.sql_max_rows,
        )

        resp = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000,
            api_key=self._api_key,
        )

        sql = resp.content.strip()
        # Strip markdown fences if present
        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            sql = sql.strip()

        return SQLPlanResult(
            sql=sql,
            tables_used=[],
            columns_used=[],
        )
