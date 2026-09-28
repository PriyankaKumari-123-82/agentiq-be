"""Which result columns hold money.

The frontend cannot answer this question. By the time a value reaches a table
cell it is just a number, and the column name alone is a guess: `total_value`
is currency but `order_quantity` is not, and both are DOUBLE PRECISION. The
distinction lives in the schema documentation, which is server-side, so the
decision is made here and shipped to the client as `money_fields` on the
result payload.

Ground truth comes from the curated COLUMN_MAP_*.md references in
ai_agent/prompts/, where the business author marked every currency column's
meaning with "(INR)". A name heuristic remains as a fallback for databases
that have no column map — a guess, but now a single server-side one that can
be corrected by adding a column map rather than by shipping frontend code.
"""
import logging
import re
from functools import lru_cache
from pathlib import Path

import sqlglot
from sqlglot import exp

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

_MAP_TABLE_RE = re.compile(r"^##\s+Table:\s*`([\w.]+)`", re.MULTILINE)

_MAP_ROW_RE = re.compile(r"^\|\s*\d+\s*\|\s*`([\w$]+)`\s*\|(.*)$")

_CURRENCY_MARKER_RE = re.compile(r"\((?:INR|USD|EUR|GBP)\)|\bcurrency\b", re.IGNORECASE)

_NAME_FALLBACK_RE = re.compile(
    r"(?:^|_)(value|price|cost|amount|spend|spent|revenue|budget|salary"
    r"|fee|fees|charge|charges|payment|payments|expense|expenses|invoice"
    r"|gross|net|tax|discount|rebate|freight|duty|tariff|surcharge"
    r"|miro|grn|po|net_price|unit_price|base_price|total_price"
    r"|delivery|committed|actual|released|open|utilized)(?:_|$)",
    re.IGNORECASE,
)

_NEVER_MONEY_RE = re.compile(
    r"(?:^|_)(count|qty|quantity|number|no|id|code|type|status|indicator"
    r"|group|date|time|rate|percent|pct|ratio|flag|name|text|desc"
    r"|description|item|unit|uom)(?:_|$)",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _load_currency_columns() -> tuple[frozenset[tuple[str, str]], frozenset[str]]:
    """Parse every COLUMN_MAP_*.md into currency column sets.

    Returns (qualified, bare):
      qualified — {(table, column)} pairs, the precise form
      bare      — {column} names, used when the owning table can't be resolved
                  from the SQL (an unaliased expression, an unparseable query)

    Both are returned because table resolution is best-effort: preferring the
    qualified match keeps two clients' identically-named columns from tagging
    each other, while the bare set still catches the common case.
    """
    qualified: set[tuple[str, str]] = set()
    bare: set[str] = set()

    for path in sorted(_PROMPTS_DIR.glob("COLUMN_MAP_*.md")):
        try:
            text = path.read_text()
        except OSError as exc:
            log.warning("Could not read column map %s: %s", path, exc)
            continue

        current_table = ""
        for line in text.splitlines():
            table_match = _MAP_TABLE_RE.match(line)
            if table_match:
                current_table = table_match.group(1).lower()
                continue
            row = _MAP_ROW_RE.match(line)
            if not row or not _CURRENCY_MARKER_RE.search(row.group(2)):
                continue
            column = row.group(1).lower()
            bare.add(column)
            if current_table:
                qualified.add((current_table, column))

    if bare:
        log.info("Currency columns loaded from column maps: %d", len(bare))
    return frozenset(qualified), frozenset(bare)


def _looks_like_money_name(column: str) -> bool:
    """Heuristic fallback for columns with no column-map entry."""
    if _NEVER_MONEY_RE.search(column):
        return False
    return bool(_NAME_FALLBACK_RE.search(column))


def _is_currency_column(table: str | None, column: str) -> bool:
    qualified, bare = _load_currency_columns()
    col = column.lower()
    if table and (table.lower(), col) in qualified:
        return True
    if col in bare:
        return True
    return _looks_like_money_name(col)


def _source_columns_by_output(sql: str) -> dict[str, list[tuple[str | None, str]]]:
    """Map each SELECT output name to the source columns feeding it.

    ``SELECT SUM(d.miro_value) AS total_spend`` yields
    ``{"total_spend": [("d", "miro_value")]}`` — without this, an aliased
    aggregate loses every trace of the currency column it came from and the
    output would be judged on the alias text alone.

    Returns {} when the SQL can't be parsed; callers then fall back to judging
    the output name directly.
    """
    try:
        tree = sqlglot.parse_one(sql, error_level=sqlglot.ErrorLevel.IGNORE)
    except Exception:
        return {}
    if tree is None:
        return {}

    alias_to_table: dict[str, str] = {}
    for t in tree.find_all(exp.Table):
        if t.name:
            alias_to_table[t.alias_or_name.lower()] = t.name.lower()

    out: dict[str, list[tuple[str | None, str]]] = {}
    try:
        selects = tree.selects
    except Exception:
        return {}

    for select in selects:
        name = (select.alias_or_name or "").lower()
        if not name or name == "*":
            continue
        cols: list[tuple[str | None, str]] = []
        for c in select.find_all(exp.Column):
            qualifier = (c.table or "").lower()
            cols.append((alias_to_table.get(qualifier, qualifier or None), c.name))
        out[name] = cols
    return out


def detect_money_fields(columns: list[str], sql: str | None = None) -> list[str]:
    """Return which of *columns* hold monetary amounts.

    *columns* are the output column names as they appear in the result rows;
    *sql* is the query that produced them, used to trace an alias back to the
    real column. A COUNT(*) alias resolves to no source columns and is never
    money, however it was named.
    """
    if not columns:
        return []

    by_output = _source_columns_by_output(sql) if sql else {}
    money: list[str] = []

    for column in columns:
        sources = by_output.get(column.lower())
        if sources:
            is_money = any(_is_currency_column(t, c) for t, c in sources)
        elif sources == []:
            is_money = False
        else:
            is_money = _is_currency_column(None, column)
        if is_money:
            money.append(column)

    return money


def detect_scalar_is_money(sql: str | None) -> bool:
    """Whether a single-value (NUMBER) result is a monetary amount.

    A scalar has no column name of its own, so the only evidence is the
    query's lone SELECT output. Anything else — no SQL, several outputs —
    is treated as not money rather than guessed at.
    """
    if not sql:
        return False
    by_output = _source_columns_by_output(sql)
    if len(by_output) != 1:
        return False
    return bool(detect_money_fields([next(iter(by_output))], sql))
