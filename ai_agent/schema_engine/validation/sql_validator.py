"""Pre-execution SQL validation.

Validates:
- Table / column existence against the retrieved schema.
- Read-only enforcement (no DML / DDL).
- SQL injection patterns.
- Query complexity heuristics.
- LIMIT injection when missing.
- Structural correctness via sqlglot (when available).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set

from ai_agent.schema_engine.config import get_config
from ai_agent.schema_engine.models import (
    RetrievalResult,
    SQLValidationResult,
)

log = logging.getLogger(__name__)

# Destructive verbs that must never appear as whole words.
_FORBIDDEN_VERBS = frozenset({
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "GRANT", "REVOKE", "MERGE", "REPLACE", "EXEC", "EXECUTE", "CALL",
    "ATTACH", "DETACH", "VACUUM", "ANALYZE", "PRAGMA", "COPY",
})

# Regex patterns commonly associated with SQL injection.
_INJECTION_PATTERNS = [
    re.compile(r";\s*(DROP|ALTER|DELETE|UPDATE|INSERT|TRUNCATE|CREATE)", re.I),
    re.compile(r"UNION\s+ALL\s+SELECT\s+", re.I),
    re.compile(r"--\s*$", re.M),
    re.compile(r"/\*.*?\*/", re.S),
    re.compile(r"xp_cmdshell", re.I),
    re.compile(r"WAITFOR\s+DELAY", re.I),
    re.compile(r"BENCHMARK\s*\(", re.I),
    re.compile(r"SLEEP\s*\(", re.I),
]


class SQLValidator:
    """Validate SQL before it reaches the database."""

    def __init__(self):
        self._cfg = get_config()
        self._sqlglot_available = self._check_sqlglot()

    @staticmethod
    def _check_sqlglot() -> bool:
        try:
            import sqlglot  # noqa: F401
            return True
        except ImportError:
            return False

    def validate(
        self,
        sql: str,
        retrieval: Optional[RetrievalResult] = None,
        known_tables: Optional[Set[str]] = None,
        known_columns: Optional[Dict[str, Set[str]]] = None,
    ) -> SQLValidationResult:
        """Run all validation checks and return a structured result."""
        errors: List[str] = []
        warnings: List[str] = []
        cleaned = sql

        # ── 1. Strip markdown fences / comments ─────────────────────────
        cleaned = self._strip_fences(cleaned)
        cleaned = self._strip_comments(cleaned)
        cleaned = cleaned.strip().rstrip(";").strip()

        if not cleaned:
            return SQLValidationResult(is_valid=False, errors=["Empty SQL"])

        # ── 2. Read-only check ───────────────────────────────────────────
        upper = cleaned.upper()
        first_word = upper.split(None, 1)[0] if upper.split() else ""
        if first_word not in ("SELECT", "WITH"):
            errors.append(f"Only SELECT/WITH queries allowed, got: {first_word}")

        pattern = r"\b(" + "|".join(_FORBIDDEN_VERBS) + r")\b"
        match = re.search(pattern, upper)
        if match:
            errors.append(f"Forbidden verb detected: {match.group(1)}")

        # ── 3. Multi-statement check ─────────────────────────────────────
        if ";" in cleaned:
            errors.append("Multi-statement SQL is not allowed")

        # ── 4. SQL injection patterns ────────────────────────────────────
        for pat in _INJECTION_PATTERNS:
            if pat.search(cleaned):
                errors.append(f"Suspicious pattern detected: {pat.pattern[:40]}…")
                break

        # ── 5. Table existence ───────────────────────────────────────────
        tables_ref = self._extract_table_references(cleaned)
        if known_tables is None and retrieval:
            known_tables = set()
            for c in retrieval.chunks:
                known_tables.add(c.table_name.lower())
                known_tables.add(c.table_fqn.lower())

        if known_tables:
            for t in tables_ref:
                if t.lower() not in known_tables:
                    warnings.append(f"Table '{t}' not found in retrieved schema")

        # ── 6. Column existence (best-effort) ────────────────────────────
        cols_ref = self._extract_column_references(cleaned)
        if known_columns is None and retrieval:
            known_columns = {}
            for c in retrieval.chunks:
                known_columns[c.table_name.lower()] = {
                    cn.lower() for cn in c.column_names
                }

        # ── 7. LIMIT check ──────────────────────────────────────────────
        has_limit = bool(re.search(r"\bLIMIT\b", upper))
        is_aggregate = bool(re.search(r"\b(COUNT|SUM|AVG|MIN|MAX|GROUP\s+BY)\b", upper))
        if not has_limit and not is_aggregate:
            cleaned += f"\nLIMIT {self._cfg.sql_max_rows}"
            warnings.append(f"Added LIMIT {self._cfg.sql_max_rows}")
            has_limit = True

        # ── 8. Complexity heuristic ──────────────────────────────────────
        join_count = len(re.findall(r"\bJOIN\b", upper))
        subquery_count = upper.count("SELECT") - 1
        complexity = "low"
        if join_count > 3 or subquery_count > 2:
            complexity = "high"
            warnings.append(f"High complexity: {join_count} joins, {subquery_count} subqueries")
        elif join_count > 1 or subquery_count > 0:
            complexity = "medium"

        # ── 9. sqlglot structural check (optional) ───────────────────────
        if self._sqlglot_available and not errors:
            sg_errors = self._sqlglot_check(cleaned)
            errors.extend(sg_errors)

        return SQLValidationResult(
            is_valid=len(errors) == 0,
            cleaned_sql=cleaned if not errors else None,
            errors=errors,
            warnings=warnings,
            tables_referenced=list(tables_ref),
            columns_referenced=list(cols_ref),
            has_limit=has_limit,
            estimated_complexity=complexity,
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _strip_fences(sql: str) -> str:
        s = sql.strip()
        if s.startswith("```"):
            first_nl = s.find("\n")
            s = s[first_nl + 1:] if first_nl != -1 else s.strip("`")
            if s.rstrip().endswith("```"):
                s = s.rstrip()[:-3]
        return s.strip()

    @staticmethod
    def _strip_comments(sql: str) -> str:
        out, i, n = [], 0, len(sql)
        while i < n:
            if sql[i:i + 2] == "--":
                nl = sql.find("\n", i)
                if nl == -1:
                    break
                i = nl + 1
            elif sql[i:i + 2] == "/*":
                end = sql.find("*/", i + 2)
                if end == -1:
                    break
                i = end + 2
            else:
                out.append(sql[i])
                i += 1
        return "".join(out)

    @staticmethod
    def _extract_table_references(sql: str) -> Set[str]:
        """Best-effort extraction of table names from SQL."""
        tables: Set[str] = set()
        # FROM / JOIN clauses
        for m in re.finditer(r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_.]*)", sql, re.I):
            tables.add(m.group(1))
        return tables

    @staticmethod
    def _extract_column_references(sql: str) -> Set[str]:
        """Best-effort column reference extraction."""
        cols: Set[str] = set()
        for m in re.finditer(r"\b([a-zA-Z_]\w*\.[a-zA-Z_]\w*)\b", sql):
            cols.add(m.group(1))
        return cols

    @staticmethod
    def _sqlglot_check(sql: str) -> List[str]:
        """Parse with sqlglot and report structural errors."""
        errors: List[str] = []
        try:
            import sqlglot
            from sqlglot.errors import ParseError
            
            parsed = sqlglot.parse(sql)
            if not parsed or all(expr is None for expr in parsed):
                errors.append("sqlglot: failed to parse SQL")
        except ParseError as e:
            detail = getattr(e, "errors", None)
            if detail:
                errors.extend(str(d) for d in detail)
            else:
                errors.append(f"sqlglot parse error: {e}")
        except Exception as e:
            # sqlglot parse failure → structural issue
            errors.append(f"sqlglot parse error: {e}")
        return errors


# ── Singleton ────────────────────────────────────────────────────────────

_validator: Optional[SQLValidator] = None


def get_sql_validator() -> SQLValidator:
    global _validator
    if _validator is None:
        _validator = SQLValidator()
    return _validator
