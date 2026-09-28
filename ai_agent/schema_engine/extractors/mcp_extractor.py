"""Extract database metadata via an MCP server's schema tool.

This extractor is used when the database is accessed through an MCP server
that exposes a ``get-schema`` (or similarly named) tool.  It calls the tool,
parses the JSON response, and produces a normalised ``DatabaseMetadata``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ai_agent.schema_engine.config import get_config
from ai_agent.schema_engine.extractors.base import BaseMetadataExtractor
from ai_agent.schema_engine.models import (
    ColumnMetadata,
    DatabaseMetadata,
    ForeignKeyMetadata,
    TableMetadata,
)

log = logging.getLogger(__name__)


_TRUE_STRINGS = frozenset({"yes", "true", "y", "1"})
_FALSE_STRINGS = frozenset({"no", "false", "n", "0"})


def _normalize_boolean_metadata(val: Any, *, field_name: str, none_value: bool) -> bool:
    """Parse a boolean-like metadata value into a Python bool.

    Supported input types:
    - ``bool`` — returned as-is.
    - ``int`` — ``1`` → ``True``, ``0`` → ``False``.
    - ``str`` — case-insensitive; ``"YES"``/``"Y"``/``"TRUE"``/``"1"`` → ``True``,
      ``"NO"``/``"N"``/``"FALSE"``/``"0"`` → ``False``.
    - ``None`` — returns ``none_value`` (caller-specified default).

    Raises ``ValueError`` for any other type or unrecognised string value.
    The ``field_name`` is included in the error message for diagnostics.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        if val == 1:
            return True
        if val == 0:
            return False
        raise ValueError(f"Unexpected {field_name} integer value: {val!r}")
    if isinstance(val, str):
        normalised = val.strip().lower()
        if normalised in _TRUE_STRINGS:
            return True
        if normalised in _FALSE_STRINGS:
            return False
        raise ValueError(f"Unexpected {field_name} string value: {val!r}")
    if val is None:
        return none_value
    raise ValueError(f"Unexpected {field_name} type {type(val).__name__}: {val!r}")


class MCPMetadataExtractor(BaseMetadataExtractor):
    """Pull schema metadata through the MCP manager.

    The caller supplies an ``mcp_manager`` and the server's tool list so we
    can auto-discover the schema-returning tool and call it.
    """

    # Heuristic keywords for identifying the schema tool
    _SCHEMA_KEYWORDS = frozenset({
        "schema", "tables", "describe", "list_tables",
        "get_schema", "get-schema", "introspect", "get_tables", "get-tables",
    })

    def __init__(self, mcp_manager, server_tools: List[Dict[str, Any]]):
        self._mcp = mcp_manager
        self._tools = server_tools
        self._cfg = get_config()

    async def extract(
        self,
        server_name: str,
        database_name: str = "default",
        *,
        include_sample_values: bool = True,
        sample_limit: int = 5,
    ) -> DatabaseMetadata:
        schema_tool = self._find_schema_tool()
        if schema_tool is None:
            raise ValueError(f"No schema tool found on MCP server '{server_name}'")

        result = await self._mcp.call_tool(
            server_name, schema_tool["tool_name"], {},
        )
        if result.get("isError"):
            raise RuntimeError(
                f"Schema tool '{schema_tool['tool_name']}' returned error: "
                f"{result}"
            )

        raw = self._flatten_result(result)
        parsed = self._try_parse_json(raw)
        tables = self._normalise(parsed, server_name)

        db = DatabaseMetadata(
            server_name=server_name,
            database_name=database_name,
            tables={t.fqn: t for t in tables},
        )
        log.info(
            "MCPExtractor: extracted %d tables from '%s'",
            db.table_count, server_name,
        )
        return db

    async def health_check(self) -> bool:
        return bool(self._tools)

    # ── Private helpers ──────────────────────────────────────────────────

    def _find_schema_tool(self) -> Optional[Dict]:
        best, best_score = None, 0
        for tool in self._tools:
            name = tool.get("tool_name", "").lower()
            desc = (tool.get("description") or "").lower()
            score = sum(1 for kw in self._SCHEMA_KEYWORDS if kw in name or kw in desc)
            if score > best_score:
                best_score = score
                best = tool
        return best if best_score > 0 else None

    @staticmethod
    def _flatten_result(result: Dict) -> str:
        """Flatten MCP call_tool result to a single text blob."""
        contents = result.get("content", [])
        if isinstance(contents, list):
            parts = [c.get("text", "") for c in contents if isinstance(c, dict)]
            return "\n".join(parts)
        return str(contents)

    # @staticmethod
    # def _try_parse_json(raw: str) -> Any:
    #     """Best-effort JSON parse, handling common wrappers."""
    #     try:
    #         parsed = json.loads(raw)
    #         # Unwrap e.g. {"jsonb_pretty": "…"}
    #         if isinstance(parsed, dict) and set(parsed.keys()) == {"jsonb_pretty"}:
    #             parsed = json.loads(parsed["jsonb_pretty"])
    #         return parsed
    #     except (json.JSONDecodeError, TypeError):
    #         return raw

    @staticmethod
    def _try_parse_json(value: Any, max_depth: int = 10) -> Any:
        """
        Recursively and safely parse nested/double-encoded JSON.

        Handles:
        - Raw JSON strings
        - Double/triple encoded JSON
        - Nested JSON inside dict/list values
        - Common wrappers like {"jsonb_pretty": "..."}
        - Already parsed objects

        Returns original value if parsing is not possible.
        """

        WRAPPER_KEYS = {
            "jsonb_pretty",
            "data",
            "payload",
            "body",
            "value",
            "schema_json"
        }

        def _parse(obj: Any, depth: int) -> Any:
            if depth > max_depth:
                return obj

            # Parse strings
            if isinstance(obj, str):
                s = obj.strip()

                if not s:
                    return obj

                # Attempt JSON parse only for likely JSON
                if s[0] in ('{', '[', '"'):
                    try:
                        parsed = json.loads(s)
                        return _parse(parsed, depth + 1)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        return obj

                return obj

            # Parse dict recursively
            if isinstance(obj, dict):
                # Unwrap known single-key wrappers
                if len(obj) == 1:
                    key = next(iter(obj))
                    if key in WRAPPER_KEYS:
                        return _parse(obj[key], depth + 1)

                return {
                    k: _parse(v, depth + 1)
                    for k, v in obj.items()
                }

            # Parse list recursively
            if isinstance(obj, list):
                return [_parse(item, depth + 1) for item in obj]

            return obj

        return _parse(value, 0)

    def _normalise(
        self,
        parsed: Any,
        server_name: str,
    ) -> List[TableMetadata]:
        """Convert arbitrarily-shaped schema JSON into ``TableMetadata`` list."""
        cfg = self._cfg
        raw_tables = self._extract_table_list(parsed)
        tables: List[TableMetadata] = []

        for raw in raw_tables:
            if not isinstance(raw, dict):
                continue

            tname = raw.get("name") or raw.get("table_name") or raw.get("tableName")
            if not tname:
                continue

            schema_name = raw.get("schema") or raw.get("schema_name") or "public"

            # Skip excluded schemas / tables
            if schema_name.lower() in {s.lower() for s in cfg.excluded_schemas}:
                continue
            if tname.lower() in {t.lower() for t in cfg.excluded_tables}:
                continue

            raw_cols = raw.get("columns", [])
            if isinstance(raw_cols, str):
                try:
                    raw_cols = json.loads(raw_cols)
                except json.JSONDecodeError:
                    log.warning(
                        "MCPExtractor: table '%s' has malformed 'columns' string, skipping columns",
                        tname,
                    )
                    raw_cols = []
            columns = self._parse_columns(raw_cols, cfg)
            fks = self._parse_foreign_keys(raw.get("foreign_keys", []))
            pks = raw.get("primary_keys", raw.get("primaryKeys", []))
            if not pks:
                pks = [c.name for c in columns if c.is_primary_key]

            # Mark FK columns
            fk_col_names = {fk.source_column for fk in fks}
            for c in columns:
                if c.name in fk_col_names:
                    c.is_foreign_key = True

            tables.append(TableMetadata(
                schema_name=schema_name,
                table_name=tname,
                columns=columns,
                primary_keys=pks,
                foreign_keys=fks,
                row_count=raw.get("row_count") or raw.get("rowCount"),
                description=raw.get("description"),
                table_comment=raw.get("comment") or raw.get("table_comment"),
                sample_values=raw.get("sample_values") or raw.get("sampleValues"),
            ))

        return tables

    # ── Parsing helpers ──────────────────────────────────────────────────

    @staticmethod
    def _extract_table_list(parsed: Any) -> List:
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("tables", "schema", "result", "data"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            # Keys are table names
            return [{"name": k, **(v if isinstance(v, dict) else {})}
                    for k, v in parsed.items() if isinstance(v, (dict, list))]
        return []

    @staticmethod
    def _parse_columns(raw_cols: Any, cfg) -> List[ColumnMetadata]:
        if not isinstance(raw_cols, list):
            return []
        cols: List[ColumnMetadata] = []
        excluded = {c.lower() for c in cfg.excluded_columns}
        for c in raw_cols:
            if isinstance(c, str):
                try:
                    c = json.loads(c)
                except json.JSONDecodeError:
                    log.warning(
                        "MCPExtractor: skipping malformed column entry (not valid JSON): %r",
                        c[:80],
                    )
                    continue
            if not isinstance(c, dict):
                continue
            name = c.get("name") or c.get("column") or c.get("column_name")
            if not name:
                continue
            if name.lower() in excluded:
                continue
            # Prefer enum/allowed values already declared in the schema
            # response. When present, the enricher will skip DB sampling for
            # this column (it only samples columns with no values yet).
            schema_values = (
                c.get("sample_values")
                or c.get("enum_values")
                or c.get("distinct_values")
                or c.get("allowed_values")
            )
            raw_nullable = c.get("nullable", c.get("is_nullable"))
            raw_pk = c.get("is_primary_key", c.get("primary_key", c.get("key")))
            cols.append(ColumnMetadata(
                name=name,
                data_type=c.get("type") or c.get("data_type") or "unknown",
                nullable=_normalize_boolean_metadata(raw_nullable, field_name="is_nullable", none_value=True),
                description=c.get("description") or c.get("comment"),
                is_primary_key=_normalize_boolean_metadata(raw_pk, field_name="is_primary_key", none_value=False),
                sample_values=schema_values,
            ))
        return cols

    @staticmethod
    def _parse_foreign_keys(raw_fks: Any) -> List[ForeignKeyMetadata]:
        if not isinstance(raw_fks, list):
            return []
        fks: List[ForeignKeyMetadata] = []
        for fk in raw_fks:
            if not isinstance(fk, dict):
                continue
            src = fk.get("column") or fk.get("source_column") or fk.get("sourceColumn")
            tgt_table = fk.get("references_table") or fk.get("target_table") or fk.get("targetTable")
            tgt_col = fk.get("references_column") or fk.get("target_column") or fk.get("targetColumn")
            if src and tgt_table and tgt_col:
                fks.append(ForeignKeyMetadata(
                    source_column=src,
                    target_table=tgt_table,
                    target_column=tgt_col,
                    target_schema=fk.get("target_schema"),
                ))
        return fks
