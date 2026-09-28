"""Generate human-readable table summaries, business purposes, and semantic
keywords using a lightweight LLM call.

Summaries are generated **once** per table (after extraction) and cached
alongside the metadata.  Re-summarization happens only when the table's
content hash changes.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from ai_agent.schema_engine.config import get_config
from ai_agent.schema_engine.models import DatabaseMetadata, TableMetadata

log = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = """\
You are a database documentation expert. For the table below, produce a JSON
object with EXACTLY these keys:

{{"summary": "<1-sentence human-readable description>",
  "business_purpose": "<what business function this table serves>",
  "domain": "<one of: finance, sales, hr, inventory, support, general>",
  "semantic_keywords": ["<keyword1>", "<keyword2>", ...],
  "join_hints": ["<hint about how this table is commonly joined>", ...]}}

Table: {fqn}
Comment: {comment}
Columns (name : type): {columns}
Primary keys: {pks}
Foreign keys: {fks}
Row count: {row_count}

Respond ONLY with the JSON object, no markdown fences."""


class SchemaSummarizer:
    """Batch-summarize tables using a lightweight LLM."""

    def __init__(self):
        self._cfg = get_config()

    async def summarize_database(
        self, db: DatabaseMetadata, *, force: bool = False,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> DatabaseMetadata:
        """Enrich every table in *db* with a summary, domain, keywords.

        Skips tables that already have a summary unless *force* is True.
        """
        from ai_agent.core.llm_gateway import get_llm_gateway
        llm = get_llm_gateway()

        tables_needing_summary = [
            t for t in db.tables.values()
            if force or not t.summary
        ]
        if not tables_needing_summary:
            log.info("SchemaSummarizer: all %d tables already summarised", db.table_count)
            return db

        log.info("SchemaSummarizer: summarising %d / %d tables", len(tables_needing_summary), db.table_count)

        # Process in small batches to avoid rate limits
        batch_size = 5
        for i in range(0, len(tables_needing_summary), batch_size):
            batch = tables_needing_summary[i:i + batch_size]
            for table in batch:
                try:
                    await self._summarize_table(table, llm, api_key=api_key, model=model, provider=provider)
                except Exception as e:
                    log.warning("SchemaSummarizer: failed on %s: %s", table.fqn, e)
                    # Fall back to heuristic summary
                    self._heuristic_summary(table)

        return db

    async def _summarize_table(
        self, table: TableMetadata, llm, *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        cols_str = ", ".join(
            f"{c.name}:{c.data_type}" + (
                " [possible_values: "
                + ", ".join("'" + str(v).replace("'", "''") + "'"
                            for v in c.sample_values[:10])
                + "]"
                if c.sample_values else ""
            )
            for c in table.columns[:40]
        )
        if len(table.columns) > 40:
            cols_str += f", ... (+{len(table.columns) - 40} more)"

        fks_str = ", ".join(
            f"{fk.source_column}->{fk.target_table}.{fk.target_column}"
            for fk in table.foreign_keys
        ) or "none"

        prompt = _SUMMARIZE_PROMPT.format(
            fqn=table.fqn,
            comment=table.table_comment or table.description or "none",
            columns=cols_str,
            pks=", ".join(table.primary_keys) or "none",
            fks=fks_str,
            row_count=table.row_count or "unknown",
        )

        effective_model = model or self._cfg.summarization_model

        resp = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            provider=provider,
            model=effective_model,
            temperature=0.0,
            max_tokens=self._cfg.summarization_max_tokens,
            api_key=api_key,
        )

        try:
            data = json.loads(resp.content)
            table.summary = data.get("summary", "")
            table.business_purpose = data.get("business_purpose", "")
            table.domain = data.get("domain", "general")
            table.semantic_keywords = data.get("semantic_keywords", [])
            table.join_hints = data.get("join_hints", [])
        except (json.JSONDecodeError, AttributeError):
            log.warning("SchemaSummarizer: bad JSON for %s, using heuristic", table.fqn)
            self._heuristic_summary(table)

    @staticmethod
    def _heuristic_summary(table: TableMetadata) -> None:
        """Cheap fallback that doesn't call the LLM."""
        name = table.table_name.replace("_", " ").title()
        table.summary = table.table_comment or table.description or f"Table: {name}"
        table.business_purpose = table.summary
        table.domain = "general"
        table.semantic_keywords = [
            w for w in table.table_name.lower().split("_") if len(w) > 2
        ]


# ── Singleton ────────────────────────────────────────────────────────────

_summarizer = None

def get_summarizer() -> SchemaSummarizer:
    global _summarizer
    if _summarizer is None:
        _summarizer = SchemaSummarizer()
    return _summarizer
