"""Token-budget management utilities.

Provides helpers for:
- Estimating token counts without an API call.
- Compressing schema context to fit a budget.
- Pruning audit columns, repetitive metadata, and unused indexes.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

from ai_agent.schema_engine.config import get_config
from ai_agent.schema_engine.models import SchemaChunk


class TokenOptimizer:
    """Compress and prune schema context to meet a token budget."""

    def __init__(self):
        self._cfg = get_config()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Cheap ~4-chars-per-token estimate."""
        return max(len(text) // 4, 1)

    def optimize_context(
        self,
        chunks: List[SchemaChunk],
        token_budget: int,
        query: str = "",
    ) -> str:
        """Build a schema context string that fits within *token_budget*.

        Strategy (from most to least aggressive):
        1. Start with compact DDL per chunk.
        2. Exclude audit/ETL columns.
        3. If still over budget, truncate to top-scored chunks.
        4. Hard-cap as last resort.
        """
        budget_chars = token_budget * 4
        cfg = self._cfg
        audit_set = set(cfg.audit_column_patterns)
        excluded = {c.lower() for c in cfg.excluded_columns}

        parts: List[str] = []
        used = 0

        for chunk in chunks:
            # Build an ultra-compact representation
            block = self._compress_chunk(chunk, audit_set, excluded)
            block_len = len(block) + 2  # +2 for separator

            if used + block_len > budget_chars and parts:
                break
            parts.append(block)
            used += block_len

        result = "\n\n".join(parts)

        # Hard safety cap
        if len(result) > budget_chars:
            result = result[:budget_chars] + "\n... (truncated)"

        return result

    def _compress_chunk(
        self,
        chunk: SchemaChunk,
        audit_cols: Set[str],
        excluded_cols: Set[str],
    ) -> str:
        """Produce a compressed DDL block for a single chunk."""
        # Filter columns
        cols = []
        audit_found: List[str] = []
        for cn in chunk.column_names:
            if cn.lower() in excluded_cols:
                continue
            if cn.lower() in audit_cols:
                audit_found.append(cn)
                continue
            is_pk = cn in chunk.primary_keys
            cols.append(f"{cn}{'(PK)' if is_pk else ''}")

        # Compress audit timestamps
        if audit_found:
            cols.append(f"[audit: {','.join(audit_found)}]")

        # Truncate columns to budget
        col_budget = self._cfg.column_budget_per_table
        if len(cols) > col_budget:
            remainder = len(cols) - col_budget
            cols = cols[:col_budget] + [f"+{remainder} more"]

        header = chunk.table_name
        if chunk.summary_text and len(chunk.summary_text) < 80:
            header += f" -- {chunk.summary_text}"

        return f"{header}\n  ({', '.join(cols)})"

    @staticmethod
    def compress_repetitive(text: str) -> str:
        """Replace patterns like multiple timestamp columns with a summary."""
        # Collapse consecutive audit columns
        text = re.sub(
            r"(created_at:\w+,?\s*updated_at:\w+,?\s*(?:deleted_at:\w+,?\s*)?)",
            "[audit timestamps] ",
            text,
        )
        return text
