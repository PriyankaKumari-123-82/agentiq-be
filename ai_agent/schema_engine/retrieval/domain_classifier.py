"""Hierarchical domain classification for schema retrieval.

Level 1: Identify relevant business domains (finance, sales, hr, …)
Level 2: Retrieve tables within those domains.
Level 3: Retrieve relevant columns within those tables.

This pre-filter dramatically reduces the search space before BM25/vector
kicks in.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set

from ai_agent.schema_engine.config import get_config
from ai_agent.schema_engine.models import SchemaChunk

log = logging.getLogger(__name__)


class DomainClassifier:
    """Keyword-based hierarchical domain classifier."""

    def __init__(self, chunks: Optional[List[SchemaChunk]] = None):
        self._cfg = get_config()
        self._chunks = chunks or []
        self._domain_index: Dict[str, List[SchemaChunk]] = {}
        if chunks:
            self._build_index()

    def _build_index(self) -> None:
        for chunk in self._chunks:
            domain = chunk.domain or "general"
            self._domain_index.setdefault(domain, []).append(chunk)

    # ── Level 1: Domain identification ───────────────────────────────────

    def classify_query(self, query: str) -> List[str]:
        """Return ranked list of domains relevant to the query."""
        tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        scores: Dict[str, float] = {}

        for domain, keywords in self._cfg.domain_keywords.items():
            overlap = tokens & set(keywords)
            if overlap:
                scores[domain] = len(overlap) / len(keywords)

        # Also check if any domain keywords appear as substrings
        for domain, keywords in self._cfg.domain_keywords.items():
            for kw in keywords:
                if kw in query.lower() and domain not in scores:
                    scores[domain] = 0.3  # partial match

        if not scores:
            return ["general"]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked]

    # ── Level 2: Tables in domains ───────────────────────────────────────

    def tables_in_domains(self, domains: List[str]) -> List[SchemaChunk]:
        """Return all chunks belonging to the given domains."""
        result: List[SchemaChunk] = []
        seen: Set[str] = set()
        for d in domains:
            for chunk in self._domain_index.get(d, []):
                if chunk.chunk_id not in seen:
                    result.append(chunk)
                    seen.add(chunk.chunk_id)
        # Always include 'general' domain as fallback
        if "general" not in domains:
            for chunk in self._domain_index.get("general", []):
                if chunk.chunk_id not in seen:
                    result.append(chunk)
                    seen.add(chunk.chunk_id)
        return result

    # ── Level 3: Column filtering ────────────────────────────────────────

    @staticmethod
    def filter_columns(
        chunks: List[SchemaChunk],
        query: str,
        max_per_table: int = 30,
    ) -> Dict[str, List[str]]:
        """For each table, return the most relevant column names."""
        tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        result: Dict[str, List[str]] = {}

        for chunk in chunks:
            # Score each column by keyword overlap with query
            scored: List[tuple[float, str]] = []
            for col_name in chunk.column_names:
                col_tokens = set(col_name.lower().split("_"))
                overlap = len(tokens & col_tokens)
                is_pk = col_name in chunk.primary_keys
                score = overlap + (2.0 if is_pk else 0.0)
                scored.append((score, col_name))

            scored.sort(key=lambda x: x[0], reverse=True)
            result[chunk.table_name] = [name for _, name in scored[:max_per_table]]

        return result
