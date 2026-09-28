"""Schema classifier: clusters a database's tables into business-meaningful
categories and generates a handful of grounded sample questions for each.

Two-stage pipeline, both LLM-backed:
1. ``SchemaClassifier`` — one call that clusters all tables using the compact
   per-table summaries cold build already produced (name, summary,
   business_purpose, keywords) — no raw columns/enums needed here, and no
   extra DB/Milvus round-trip since ``DatabaseMetadata`` is already in memory
   at this point in cold build.
2. ``SchemaClassifierQuestionGenerator`` — one call per category, using that
   category's ``compact_ddl`` chunks (real column names + enriched enum
   values) so questions are concrete and answerable, not generic guesses.

``SchemaClassifierEngine`` wires the two stages together and is the only entry point
callers (``schema_engine/orchestrator.py``) need.

Prompts live in ``ai_agent/prompts/schema_classifier.py`` (same basename as
this module).
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from ai_agent.prompts.schema_classifier import SCHEMA_CLASSIFIER_PROMPT, SCHEMA_CLASSIFIER_QUESTION_PROMPT
from ai_agent.schema_engine.config import get_config
from ai_agent.schema_engine.models import DatabaseMetadata, SchemaChunk, TableCategory

log = logging.getLogger(__name__)

_MAX_QUESTIONS = 5


class SchemaClassifier:
    """Groups all tables in a database into business categories via one LLM call."""

    def __init__(self):
        self._cfg = get_config()

    async def classify(
        self, db: DatabaseMetadata, *, api_key: Optional[str] = None,
    ) -> List[TableCategory]:
        from ai_agent.core.llm_gateway import get_llm_gateway
        llm = get_llm_gateway()

        log.info(
            "SchemaClassifier: classifying %d tables for '%s'",
            len(db.tables), db.server_name,
        )

        tables_str = "\n".join(
            f"- {t.table_name}: {t.summary or 'no summary'} "
            f"(domain hint: {t.domain or 'general'}, keywords: {', '.join(t.semantic_keywords)})"
            for t in db.tables.values()
        )

        resp = await llm.chat(
            messages=[{
                "role": SCHEMA_CLASSIFIER_PROMPT["role"],
                "content": SCHEMA_CLASSIFIER_PROMPT["content"].format(tables=tables_str),
            }],
            model=self._cfg.summarization_model,
            temperature=0.0,
            max_tokens=self._cfg.schema_classifier_max_tokens,
            api_key=api_key,
        )

        try:
            data = json.loads(resp.content)
        except (json.JSONDecodeError, AttributeError):
            log.warning("SchemaClassifier: bad JSON response, falling back to domain-based grouping")
            return self._fallback_by_domain(db)

        categories = [
            TableCategory(
                category_name=c.get("name", "general"),
                description=c.get("description", ""),
                table_names=c.get("tables", []),
                priority=i,
            )
            for i, c in enumerate(data.get("categories", []))
        ]
        if categories:
            log.info(
                "SchemaClassifier: '%s' grouped into %d categories: %s",
                db.server_name, len(categories),
                ", ".join(c.category_name for c in categories),
            )
        return categories or self._fallback_by_domain(db)

    @staticmethod
    def _fallback_by_domain(db: DatabaseMetadata) -> List[TableCategory]:
        """Cheap, LLM-free fallback: group by the existing (coarse) domain field."""
        buckets: Dict[str, List[str]] = {}
        for t in db.tables.values():
            buckets.setdefault(t.domain or "general", []).append(t.table_name)
        return [
            TableCategory(
                category_name=domain,
                description=f"Tables related to {domain}.",
                table_names=names,
                priority=i,
            )
            for i, (domain, names) in enumerate(buckets.items())
        ]


class SchemaClassifierQuestionGenerator:
    """Generates sample questions for one category at a time."""

    def __init__(self):
        self._cfg = get_config()

    async def generate(
        self,
        category: TableCategory,
        chunks_by_table: Dict[str, SchemaChunk],
        *,
        api_key: Optional[str] = None,
    ) -> List[str]:
        from ai_agent.core.llm_gateway import get_llm_gateway
        llm = get_llm_gateway()

        ddl = "\n".join(
            chunks_by_table[name].compact_ddl
            for name in category.table_names
            if name in chunks_by_table
        )
        if not ddl:
            log.warning(
                "SchemaClassifierQuestionGenerator: no chunks found for category '%s' (%d tables), skipping",
                category.category_name, len(category.table_names),
            )
            return []

        resp = await llm.chat(
            messages=[{
                "role": SCHEMA_CLASSIFIER_QUESTION_PROMPT["role"],
                "content": SCHEMA_CLASSIFIER_QUESTION_PROMPT["content"].format(
                    name=category.category_name, description=category.description, ddl=ddl,
                ),
            }],
            model=self._cfg.summarization_model,
            temperature=0.2,
            max_tokens=self._cfg.schema_classifier_question_max_tokens,
            api_key=api_key,
        )

        try:
            data = json.loads(resp.content)
            questions = data.get("questions", [])
        except (json.JSONDecodeError, AttributeError):
            log.warning("SchemaClassifierQuestionGenerator: bad JSON response for category '%s'", category.category_name)
            return []

        return questions[:_MAX_QUESTIONS]


class SchemaClassifierEngine:
    """Builds business categories + sample questions for a database's tables."""

    def __init__(self):
        self._classifier = SchemaClassifier()
        self._question_gen = SchemaClassifierQuestionGenerator()

    async def build(
        self,
        db: DatabaseMetadata,
        chunks: List[SchemaChunk],
        *,
        api_key: Optional[str] = None,
    ) -> List[TableCategory]:
        categories = await self._classifier.classify(db, api_key=api_key)

        chunks_by_table = {c.table_name: c for c in chunks}
        for category in categories:
            try:
                category.questions = await self._question_gen.generate(
                    category, chunks_by_table, api_key=api_key,
                )
            except Exception as e:
                log.warning(
                    "SchemaClassifierEngine: question generation failed for '%s': %s",
                    category.category_name, e,
                )

        return categories


def run() -> SchemaClassifierEngine:
    """Build a fresh ``SchemaClassifierEngine`` on demand (no caching/singleton)."""
    return SchemaClassifierEngine()
