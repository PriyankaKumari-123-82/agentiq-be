"""Enrich extracted schema with per-column distinct values (enum detection).

Called once per cold build, after extraction and before summarization.

Value source preference:
  1. If the MCP schema response already declares values for a column
     (``enum_values`` / ``distinct_values`` / ``allowed_values`` /
     ``sample_values``), the extractor stores them and we use them directly —
     NO database query is issued for that column.
  2. Otherwise we fall back to sampling the database.

Detection is **data-driven, not name-based**: rather than guessing enum-ness
from column names (status, type, flag, …), each table is sampled with ONE
cheap query covering all its candidate columns:

    SELECT col1, col2, … FROM "{schema}"."{table}" LIMIT 200

(no DISTINCT / ORDER BY → the DB returns the first N rows without a full-table
scan). Per-column distinct cardinality is then computed locally:
  - <= 10 distinct values  -> treat as enum, store ``column.sample_values``
  -  > 10 distinct values  -> high-cardinality (ID / free-text), discard

This issues ~1 query per table instead of one-per-column, which keeps the
cold build fast and avoids overwhelming a remote MCP server.

Columns that can never be categorical (PK/FK, JSON/blob/binary, large text,
temporal, continuous numerics) are skipped without sampling.

Entirely best-effort — failures are logged and skipped, never blocking
the pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ai_agent.schema_engine.models import ColumnMetadata, DatabaseMetadata, TableMetadata

log = logging.getLogger(__name__)

# Data types that can NEVER be enum-like and are skipped without sampling:
#   - structured / binary blobs
#   - free-text fields
#   - temporal types (continuous)
#   - continuous numeric measures (floats / decimals / money)
#   - opaque identifiers (uuid)
# Everything else (varchar, char, int, smallint, tinyint, bool, enum, …) is
# sampled and classified purely by its actual distinct-value cardinality.
_SKIP_DTYPE_PATTERNS = frozenset({
    "json", "jsonb", "blob", "bytea", "image", "binary", "varbinary",
    "text", "tinytext", "mediumtext", "longtext", "clob", "ntext",
    "date", "time", "datetime", "timestamp", "year", "interval",
    "float", "double", "decimal", "numeric", "real", "money",
    "uuid", "uniqueidentifier",
})

# Keywords to identify the execute-query MCP tool
_QUERY_TOOL_KEYWORDS = frozenset({
    "execute", "query", "run_query", "run-query", "exec_query",
    "execute_query", "execute-query", "run_sql", "run-sql", "execute_sql",
})

# We only need enough distinct values to judge low- vs high-cardinality.
# Fetching beyond this adds query cost with no classification benefit.
_MAX_DISTINCT_VALUES = 15

# A real status/enum field has very few distinct values. More than this is
# usually lookup/master data, an ID, or free text — so it is discarded.
_ENUM_MAX_CARDINALITY = 10

# Rows fetched per table (single query) to estimate each column's cardinality.
# No DISTINCT / ORDER BY → the DB returns the first N rows cheaply without a
# full-table scan, so one fast query replaces one-DISTINCT-query-per-column.
_SAMPLE_ROWS = 200


class EnumEnricher:
    """Fetches distinct values for low-cardinality columns via MCP query tool."""

    def __init__(self, mcp_manager: Any, server_tools: List[Dict[str, Any]]):
        self._mcp = mcp_manager
        self._tools = server_tools

    async def enrich(self, db: DatabaseMetadata, server_name: str) -> DatabaseMetadata:
        """Populate ``column.sample_values`` for likely-enum columns in-place."""
        query_tool = self._find_query_tool()
        if query_tool is None:
            log.info("EnumEnricher: no execute-query tool on '%s', skipping", server_name)
            return db

        sql_key = self._find_sql_arg_key(query_tool)
        if not sql_key:
            log.info(
                "EnumEnricher: cannot detect SQL arg key for tool '%s', skipping",
                query_tool["tool_name"],
            )
            return db

        import asyncio

        # MySQL/MariaDB uses backtick quoting and no schema prefix;
        # PostgreSQL uses double-quote quoting with a schema prefix.
        tool_desc = (query_tool.get("description") or "").lower()
        is_mysql = "mysql" in tool_desc or "mariadb" in tool_desc

        total = 0
        tables_done = 0

        for fqn, table in db.tables.items():
            candidates = self._enum_candidates(table)
            if not candidates:
                continue
            try:
                rows = await asyncio.wait_for(
                    self._sample_table(
                        server_name, query_tool, sql_key, is_mysql,
                        table.schema_name, table.table_name, candidates,
                    ),
                    timeout=15.0,
                )
                tables_done += 1
            except asyncio.TimeoutError:
                tables_done += 1
                log.debug("EnumEnricher: timeout sampling %s, skipping", fqn)
                continue
            except Exception as e:
                log.debug("EnumEnricher: skipped %s — %s", fqn, e)
                continue

            if not rows:
                continue

            # Compute distinct cardinality per column from the sampled rows.
            for col in candidates:
                seen: set = set()
                overflow = False
                for r in rows:
                    v = r.get(col.name)
                    if v is None:
                        continue
                    seen.add(str(v))
                    if len(seen) > _ENUM_MAX_CARDINALITY:
                        overflow = True  # high-cardinality → not an enum
                        break
                if not overflow and 0 < len(seen) <= _ENUM_MAX_CARDINALITY:
                    col.sample_values = sorted(seen)[:_MAX_DISTINCT_VALUES]
                    total += 1

        log.info(
            "EnumEnricher: sampled %d tables, enriched %d columns on '%s'",
            tables_done, total, server_name,
        )
        return db

    # ── Private helpers ──────────────────────────────────────────────────

    def _find_query_tool(self) -> Optional[Dict]:
        best, best_score = None, 0
        for tool in self._tools:
            name = tool.get("tool_name", "").lower()
            desc = (tool.get("description") or "").lower()
            score = sum(1 for kw in _QUERY_TOOL_KEYWORDS if kw in name or kw in desc)
            if score > best_score:
                best_score = score
                best = tool
        return best if best_score > 0 else None

    @staticmethod
    def _find_sql_arg_key(tool: Dict) -> Optional[str]:
        schema = tool.get("input_schema") or tool.get("inputSchema") or {}
        props = schema.get("properties", {})
        for key in ("sql", "query", "statement", "queryString", "query_string"):
            if key in props:
                return key
        # Fallback: first string-typed property
        for key, defn in props.items():
            if defn.get("type") == "string":
                return key
        return None

    @staticmethod
    def _enum_candidates(table: TableMetadata) -> List[ColumnMetadata]:
        """Select columns worth sampling — data-driven, NOT name-based.

        We no longer guess enum-ness from column names. Instead every column
        whose type *could* hold a small set of repeated values is sampled, and
        the cardinality of the actual distinct values (see ``enrich``) decides
        whether it is treated as an enum. Only columns that can never be
        categorical are skipped here:
          - primary / foreign keys (always high-cardinality identifiers)
          - JSON / blob / binary / large-text / temporal / continuous-numeric
            types (see ``_SKIP_DTYPE_PATTERNS``)
        """
        candidates: List[ColumnMetadata] = []
        for col in table.columns:
            if col.is_primary_key or col.is_foreign_key:
                continue
            if col.sample_values:
                continue  # already enriched
            dtype = col.data_type.lower()
            if any(skip in dtype for skip in _SKIP_DTYPE_PATTERNS):
                continue
            candidates.append(col)
        return candidates

    async def _sample_table(
        self,
        server_name: str,
        query_tool: Dict,
        sql_key: str,
        is_mysql: bool,
        schema_name: str,
        table_name: str,
        cols: List[ColumnMetadata],
    ) -> List[Dict[str, Any]]:
        """Fetch a single row sample covering all candidate columns at once.

        One cheap ``SELECT col1, col2, … LIMIT N`` per table (no DISTINCT, no
        ORDER BY) replaces one DISTINCT-query-per-column. Cardinality is then
        computed locally from the sampled rows.
        """
        if is_mysql:
            col_list = ", ".join(f"`{c.name}`" for c in cols)
            table_ref = f"`{table_name}`"
        else:
            col_list = ", ".join(f'"{c.name}"' for c in cols)
            table_ref = f'"{schema_name}"."{table_name}"'

        sql = f"SELECT {col_list} FROM {table_ref} LIMIT {_SAMPLE_ROWS}"
        result = await self._mcp.call_tool(
            server_name, query_tool["tool_name"], {sql_key: sql},
        )
        if result.get("isError"):
            return []
        return self._parse_rows(result)

    @staticmethod
    def _parse_rows(result: Dict) -> List[Dict[str, Any]]:
        """Extract a list of row dicts from an MCP execute-query result."""
        contents = result.get("content", [])
        raw_text = ""
        if isinstance(contents, list):
            for item in contents:
                if isinstance(item, dict) and item.get("type") == "text":
                    raw_text = item.get("text", "")
                    break
        elif isinstance(contents, str):
            raw_text = contents

        if not raw_text:
            return []

        try:
            data = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            return []

        # Unwrap OData envelope {"value": [...]}
        if isinstance(data, dict):
            data = data.get("value", data)
        if isinstance(data, dict):
            data = [data]  # single-row result
        if not isinstance(data, list):
            return []

        return [r for r in data if isinstance(r, dict)]
