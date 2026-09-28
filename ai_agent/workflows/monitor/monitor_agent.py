"""Monitor execution workflow.

Flow
----
execute_monitor(monitor_id, db_session, followup_answer=None)
    │
    ├─ load monitor + monitor_type                    (via monitor_service)
    ├─ gather organization-wide MCP tools             (via monitor_service)
    │
    └─ MonitorAgent.execute()
          │
          ├─ Phase 1 — QUERY GENERATION
          │   ├─ build schema context from MCP tools
          │   ├─ ask LLM to generate SQL / tool-call
          │   │   ├─ if LLM asks a follow-up → return early with followup_question
          │   │   └─ if LLM returns a query  → persist llm_query, continue
          │   └─ if followup_answer supplied → re-invoke with enriched context
          │
          ├─ Phase 2 — EXECUTION
          │   ├─ if llm_query has tool_name → call MCP tool
          │   └─ if no tools / direct LLM   → answer from LLM knowledge
          │
          └─ Phase 3 — FORMATTING
              └─ format raw result → typed dict per configuration_schema

The result is then formatted by monitor_service.persist_monitor_result()
and written to monitor_results.

OpenAI API key is loaded from settings.OPENAI_API_KEY (env var).
Scheduler integration is NOT part of this file — left for later.
"""
import json
import logging
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlmodel import Session

from ai_agent.core.config import settings
from ai_agent.core.llm_gateway import get_llm_gateway
from ai_agent.core.mcp_manager import get_mcp_manager
from ai_agent.custom_model.monitor import Monitor, MonitorType
from ai_agent.prompts.monitor import (
    MONITOR_CHART_INSTRUCTIONS,
    MONITOR_DIRECT_LLM_PROMPT,
    MONITOR_FOLLOWUP_SYSTEM,
    MONITOR_QUERY_GENERATION_SYSTEM,
    MONITOR_QUERY_GENERATION_USER,
    MONITOR_SCHEMA_FOLLOWUP_CORRECTION,
    MONITOR_SQL_REPAIR_PROMPT,
    MONITOR_SYSTEM_CONTEXT,
    MONITOR_TABLE_SELECTION_SYSTEM,
    MONITOR_TABLE_SELECTION_USER,
)
from ai_agent.services.currency_fields import (
    detect_money_fields,
    detect_scalar_is_money,
)
from ai_agent.services.monitor_service import (
    gather_org_mcp_tools,
    load_monitor_with_context,
    persist_monitor_result,
    update_monitor_llm_query,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class MonitorExecutionError(Exception):
    """Raised when a monitor cannot produce a valid, trustworthy result.

    Anything raised here is persisted as a FAILED MonitorResult with the
    message in error_message — it must NEVER be stored as a SUCCESS value,
    otherwise error text ("Table 'x' doesn't exist") leaks into the UI
    dressed up as real data.

    ``repairable`` marks errors that describe something wrong with the SQL
    itself (rejected by the database, failed validation, wrong result shape).
    Those are fed back to the model and rewritten, bounded by
    ``MonitorAgent.MAX_SQL_REPAIR_ATTEMPTS``. Infrastructure problems — no tool
    connected, empty output — are not repairable because rewriting the SQL
    cannot fix them.
    """

    def __init__(self, message: str, *, repairable: bool = False):
        super().__init__(message)
        self.repairable = repairable


# ---------------------------------------------------------------------------
# User-facing error messages
# ---------------------------------------------------------------------------

# Raw engine errors ("Unknown column 'close_date' in 'field list'") are useless
# to a business user and leak internals into the UI. Each pattern maps a class
# of failure to something actionable. The raw text is still logged and kept for
# the repair loop — only what the user reads is rewritten.
_ERROR_TRANSLATIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"unknown column '([^']+)'", re.I),
     "The AI referenced a field that doesn't exist in your database ({0}). "
     "Try describing what you want in business terms."),
    (re.compile(r"table '([^']+)' does ?n'?t exist", re.I),
     "The AI referenced a table that doesn't exist in your database ({0}). "
     "Try naming the business entity you mean."),
    (re.compile(r"unknown table '([^']+)'", re.I),
     "The AI referenced a table that doesn't exist in your database ({0})."),
    (re.compile(r"does ?n'?t yet support", re.I),
     "Your database version doesn't support the query the AI wrote. "
     "Try simplifying the question."),
    (re.compile(r"you have an error in your sql syntax", re.I),
     "The AI produced a query your database couldn't understand. "
     "Try rephrasing the question."),
    # Both orderings of the engine phrasing — MySQL says "Column 'x' in field
    # list is ambiguous", Postgres "column reference \"x\" is ambiguous". The
    # bare word alone is far too broad: it also appears in messages we author
    # and in LLM refusals, which would then be rewritten as a database fault.
    (re.compile(
        r"\b(?:column|field|reference|identifier)\b[^\n]{0,80}\bambiguous\b"
        r"|\bambiguous\b[^\n]{0,80}\b(?:column|field|reference|identifier)\b",
        re.I,
     ),
     "The query was ambiguous to your database — it matched the same field in "
     "more than one table. Try being more specific."),
    (re.compile(r"access denied|permission denied", re.I),
     "The database connection doesn't have permission to read that data."),
    (re.compile(r"(timed? ?out|lock wait)", re.I),
     "The query took too long to run. Try narrowing it to a smaller range."),
    (re.compile(r"division by zero", re.I),
     "The calculation divided by zero — there may be no matching data."),
    (re.compile(r"unknown database", re.I),
     "The configured database could not be found."),
    (re.compile(r"(can'?t connect|connection refused|server has gone away)", re.I),
     "Could not reach your database. Check the connection and try again."),
]

_RESULT_SHAPE_HINT = re.compile(
    r"expects (a single number|a yes/no answer|structured/tabular data)", re.I,
)


def _humanize_error(message: str) -> str:
    """Rewrite an internal/database error into something a user can act on.

    Messages we author ourselves are already user-facing and pass through
    unchanged; only recognisable engine errors get translated.
    """
    text = (message or "").strip()
    if not text:
        return "This monitor could not be run."

    for pattern, template in _ERROR_TRANSLATIONS:
        match = pattern.search(text)
        if match:
            detail = match.group(1) if match.groups() else ""
            return template.format(detail)

    # Result-shape mismatches: keep the explanation, drop the raw payload dump.
    shape = _RESULT_SHAPE_HINT.search(text)
    if shape:
        return (
            f"The query didn't return {shape.group(1)} as this monitor type "
            "expects. Try a different monitor type, or rephrase the question."
        )

    # Unrecognised: keep it short so a stack-trace-ish blob never fills the UI.
    if len(text) > 200:
        return text[:200].rstrip() + "…"
    return text


# ---------------------------------------------------------------------------
# SQL validation
# ---------------------------------------------------------------------------

# Statements that must never be generated for a read-only monitor.
_FORBIDDEN_SQL_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "REPLACE", "GRANT", "REVOKE", "ATTACH", "MERGE",
)

# FROM/JOIN <table> — captures optionally schema-qualified, quoted identifiers.
_TABLE_REF_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+([`\"\[]?[\w.]+[`\"\]]?)",
    re.IGNORECASE,
)

# "Table: orders (id int, ...)" — the minified form the legacy loader emits.
_SCHEMA_TABLE_RE = re.compile(r"^Table:\s*([\w.]+)", re.IGNORECASE | re.MULTILINE)

# Schema-engine compact_ddl blocks look like:
#     job_card -- Stores repair job records
#       (id:int PK, status:varchar [possible_values: 'open'], ...)
#       FK: cust_id->customer.id
# i.e. the table name is a bare identifier at column 0; detail lines are
# indented. Anything starting with '-' is a comment/separator, not a table.
_SCHEMA_ENGINE_TABLE_RE = re.compile(
    r"^(?!--)([A-Za-z_][\w$]*(?:\.[A-Za-z_][\w$]*)?)\s*(?:--.*)?$",
    re.MULTILINE,
)


def _extract_table_names(schema_context: str) -> set[str]:
    """Return the set of table names present in the schema context.

    Understands all three shapes this context can arrive in:
      1. schema-engine ``compact_ddl`` blocks (the normal path)
      2. the legacy minified "Table: name (cols)" form
      3. raw JSON carrying {"table_name": ...} entries

    Returns an empty set when the schema is unavailable or unrecognised —
    callers must then skip the table-existence check rather than reject
    every query.
    """
    if not schema_context:
        return set()

    names = {m.group(1).lower() for m in _SCHEMA_TABLE_RE.finditer(schema_context)}
    names.update(
        m.lower() for m in re.findall(r'"table_name"\s*:\s*"([\w.]+)"', schema_context)
    )
    names.update(
        m.group(1).lower() for m in _SCHEMA_ENGINE_TABLE_RE.finditer(schema_context)
    )
    # "Table: x" also matches the bare-identifier pattern via its "Table" word;
    # drop that artefact so it can never mask a real table called "table".
    names.discard("table")
    return names


def _normalise_table_ref(ref: str) -> str:
    """Strip quoting/backticks and any schema prefix from a SQL table reference."""
    cleaned = ref.strip().strip('`"[]')
    # db.table / schema.table → table
    if "." in cleaned:
        cleaned = cleaned.rsplit(".", 1)[-1]
    return cleaned.lower()


def _validate_sql(query: str, known_tables: set[str]) -> None:
    """Validate a generated SQL query before it is executed.

    Raises MonitorExecutionError when the query is not a safe, runnable
    SELECT against tables that actually exist. Catching this here means the
    user gets a real error message instead of the database's raw complaint
    surfacing as a monitor "value".
    """
    sql = (query or "").strip()
    if not sql:
        raise MonitorExecutionError(
            "The AI did not generate a query to run.", repairable=True,
        )

    # Strip a leading SQL comment block so the SELECT check sees real syntax.
    body = re.sub(r"^\s*(--[^\n]*\n|/\*.*?\*/)+", "", sql, flags=re.DOTALL).strip()
    upper = body.upper()

    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise MonitorExecutionError(
            "Only read-only SELECT queries can be run by a monitor, but the AI "
            f"generated: {body[:120]}"
        )

    for keyword in _FORBIDDEN_SQL_KEYWORDS:
        # \b…\b so UPDATED_AT / CREATED_BY columns don't trip the guard.
        if re.search(rf"\b{keyword}\b", upper):
            raise MonitorExecutionError(
                f"The generated query contains a forbidden {keyword} statement. "
                "Monitors may only read data."
            )

    if not known_tables:
        # Schema unknown — SELECT-only guard above is all we can enforce.
        return

    referenced = {_normalise_table_ref(m) for m in _TABLE_REF_RE.findall(body)}
    # Ignore CTE names defined in this same query (WITH x AS (...)).
    cte_names = {
        m.lower() for m in re.findall(r"\b(\w+)\s+AS\s*\(", body, re.IGNORECASE)
    }
    unknown = {t for t in referenced if t and t not in known_tables and t not in cte_names}

    if unknown:
        missing = ", ".join(sorted(unknown))
        raise MonitorExecutionError(
            f"The generated query references table(s) that do not exist in your "
            f"database: {missing}. Try rephrasing your question.",
            repairable=True,
        )

_COL_SPEC_RE = re.compile(
    r"(?P<name>[A-Za-z_][\w$]*):(?P<type>[\w()]+)"
    r"(?:\s+(?:PK|FK))*"
    r"(?:\s*\[possible_values:\s*"
    r"(?P<values>(?:'(?:[^']|'')*'(?:,\s*)?)*)"
    r"(?P<truncated>\.\.\.)?\])?"
)

# "Table: x (col1 type1, col2 type2)" — the legacy minified form has no
# possible_values annotations, only names.
_LEGACY_TABLE_COLS_RE = re.compile(
    r"^Table:\s*([\w.]+)\s*\(([^)]*)\)", re.IGNORECASE | re.MULTILINE,
)


def _extract_table_columns(schema_context: str) -> dict[str, dict[str, set[str] | None]]:
    """Best-effort {table_name: {column_name: possible_values_or_None}}.

    A column maps to None when it has no possible_values annotation, or the
    sample was truncated — either way there's nothing to validate a literal
    against. A table absent from the result (or mapping to {}) means its
    columns could not be confidently parsed; callers must skip that table's
    checks entirely rather than validate against a partial picture.
    """
    if not schema_context:
        return {}

    tables: dict[str, dict[str, set[str] | None]] = {}
    lines = schema_context.splitlines()

    # Schema-engine compact_ddl form: table name on its own line, columns on
    # the very next line in parentheses (see _SCHEMA_ENGINE_TABLE_RE above).
    for i, line in enumerate(lines):
        m = _SCHEMA_ENGINE_TABLE_RE.match(line)
        if not m:
            continue
        table_name = m.group(1).lower()
        if table_name == "table" or i + 1 >= len(lines):
            continue
        col_line = lines[i + 1].strip()
        if not col_line.startswith("("):
            continue

        cols: dict[str, set[str] | None] = {}
        for cm in _COL_SPEC_RE.finditer(col_line):
            col_name = cm.group("name").lower()
            if cm.group("values") is None or cm.group("truncated"):
                cols[col_name] = None
            else:
                cols[col_name] = {
                    v.replace("''", "'").lower()
                    for v in re.findall(r"'((?:[^']|'')*)'", cm.group("values"))
                }
        if cols:
            tables[table_name] = cols

    # Legacy minified form, layered on top — names only, no possible_values.
    for m in _LEGACY_TABLE_COLS_RE.finditer(schema_context):
        table_name = m.group(1).lower()
        cols = tables.setdefault(table_name, {})
        for part in m.group(2).split(","):
            name = part.strip().split()[0].lower() if part.strip() else ""
            if name:
                cols.setdefault(name, None)

    return tables


def _sqlglot_tree(query: str):
    """Parse SQL for the column/value guard. None on any failure — the guard
    is best-effort and must never block a query sqlglot can't parse (dialect
    quirks are exactly what MONITOR_SQL_REPAIR_PROMPT already works around)."""
    try:
        return sqlglot.parse_one(query, error_level=sqlglot.ErrorLevel.IGNORE)
    except Exception:
        return None


def _literal_strings(node) -> list[str] | None:
    """The string value(s) if *node* is a bare string literal, else None —
    None means "not a plain literal", used to bail out of a comparison that
    isn't a real column-vs-literal filter (e.g. a join condition)."""
    if isinstance(node, exp.Literal) and node.is_string:
        return [node.this]
    return None


def _validate_columns_and_values(
    query: str, table_columns: dict[str, dict[str, set[str] | None]],
) -> None:
    """Reject a query that references a nonexistent column, or filters a
    low-cardinality column against a literal that isn't one of its real
    values. Raises MonitorExecutionError(repairable=True); never raises when
    the schema or the SQL can't be parsed with confidence.
    """
    if not table_columns:
        return

    tree = _sqlglot_tree(query)
    if tree is None:
        return

    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}

    real_tables: set[str] = set()
    alias_to_table: dict[str, str] = {}
    for t in tree.find_all(exp.Table):
        real_name = t.name.lower()
        if not real_name or real_name in cte_names:
            continue
        real_tables.add(real_name)
        alias_to_table[t.alias_or_name.lower()] = real_name

    has_nested_scope = bool(cte_names) or bool(list(tree.find_all(exp.Subquery)))
    single_table = (
        next(iter(real_tables))
        if not has_nested_scope and len(real_tables) == 1
        else None
    )

    def resolve(col: exp.Column) -> str | None:
        qualifier = (col.table or "").lower()
        return alias_to_table.get(qualifier) if qualifier else single_table

    for col in tree.find_all(exp.Column):
        table = resolve(col)
        known_cols = table_columns.get(table) if table else None
        if not known_cols:
            continue
        if col.name.lower() not in known_cols:
            raise MonitorExecutionError(
                f"The generated query references column '{col.name}' which does "
                f"not exist on table '{table}'. Try rephrasing your question.",
                repairable=True,
            )

    comparisons: list[tuple[exp.Column, list[str]]] = []
    for eq in tree.find_all(exp.EQ, exp.NEQ):
        if isinstance(eq.left, exp.Column):
            vals = _literal_strings(eq.right)
            if vals:
                comparisons.append((eq.left, vals))
        elif isinstance(eq.right, exp.Column):
            vals = _literal_strings(eq.left)
            if vals:
                comparisons.append((eq.right, vals))
    for in_expr in tree.find_all(exp.In):
        if not isinstance(in_expr.this, exp.Column):
            continue
        vals: list[str] = []
        for e in in_expr.expressions:
            v = _literal_strings(e)
            if v is None:
                vals = []
                break
            vals.extend(v)
        if vals:
            comparisons.append((in_expr.this, vals))

    for col, literals in comparisons:
        table = resolve(col)
        known_cols = table_columns.get(table) if table else None
        if not known_cols:
            continue
        possible = known_cols.get(col.name.lower())
        if possible is None:  # no annotation, or the sample was truncated
            continue
        bad = [lit for lit in literals if lit.lower() not in possible]
        if bad:
            shown = ", ".join(f"'{v}'" for v in sorted(possible))
            raise MonitorExecutionError(
                f"The generated query filters {table}.{col.name} = "
                f"{', '.join(repr(b) for b in bad)}, but that is not a real "
                f"value for this column (known values: {shown}). Try "
                "rephrasing the business term you used.",
                repairable=True,
            )


# ---------------------------------------------------------------------------
# Answer quality checks
# ---------------------------------------------------------------------------

# Phrases that mean the model punted instead of answering. Storing these as a
# monitor value is what produced results like "I'm unable to access specific
# data for the last 10 days" sitting in the dashboard as a SUCCESS.
_REFUSAL_MARKERS = (
    "i'm unable", "i am unable", "i cannot", "i can't", "i can not",
    "unable to provide", "unable to access", "unable to retrieve",
    "please hold on", "i don't have access", "i do not have access",
    "as an ai", "i'm sorry", "i am sorry",
    "no data available", "please provide more",
)


# A refusal leads with its excuse — "I'm unable to access that data." Scanning
# the whole response instead meant a legitimate summary that mentioned "no data
# available" halfway through was discarded as a failure.
_REFUSAL_SCAN_CHARS = 160

# An AI_QUERY monitor's whole output is prose, and prose may legitimately open
# with a caveat ("No data available for Q3, but Q4 shows…") before delivering a
# real answer. A genuine refusal is short and says nothing else, so for TEXT
# results the marker only counts when the refusal IS the entire answer.
_REFUSAL_MAX_TEXT_CHARS = 400


def _looks_like_refusal(text: str, *, result_type: str = "") -> bool:
    """True when the model declined to answer rather than answering."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return False

    opening = lowered[:_REFUSAL_SCAN_CHARS]
    if not any(marker in opening for marker in _REFUSAL_MARKERS):
        return False

    if result_type.upper() == "TEXT" and len(lowered) > _REFUSAL_MAX_TEXT_CHARS:
        return False

    return True


# Follow-ups that ask the user about the database layout. Business users don't
# know their schema — resolving table/column names is the agent's job, and the
# full schema is handed to it. Questions like "which table contains the repair
# statuses?" must never reach the UI.
_SCHEMA_QUESTION_MARKERS = (
    "which table", "what table", "which tables", "what tables",
    "which column", "what column", "which columns", "what columns",
    "which field", "what field", "which database", "what database",
    "which schema", "what schema", "table name", "column name",
    "table contains", "tables contain", "table holds", "table stores",
)


def _is_schema_question(text: str) -> bool:
    """True when a follow-up is asking the user about the database layout."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in _SCHEMA_QUESTION_MARKERS)


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _get_openai_key() -> str:
    """Return the OpenAI API key from environment/settings.

    Raises ValueError if not configured — callers should catch and
    handle gracefully.
    """
    key = settings.OPENAI_API_KEY
    if not key:
        raise ValueError("OPENAI_API_KEY is not set in the environment")
    return key


async def _call_llm(
    messages: list[dict],
    api_key: str,
    temperature: float = 0.3,
) -> str:
    """Call the LLM and return the raw content string.

    Uses openai provider with gpt-4o model.
    """
    gateway = get_llm_gateway()
    response = await gateway.chat(
        messages=messages,
        provider="openai",
        model="gpt-4o",
        temperature=temperature,
        api_key=api_key,
    )
    return response.content or ""


def _parse_llm_json(raw: str) -> dict | None:
    """Attempt to parse the LLM's JSON response.

    Handles markdown code fences (```json ... ```) gracefully.
    """
    text = raw.strip()
    if text.startswith("```"):
        # Strip opening fence (e.g., ```json)
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Chart types
# ---------------------------------------------------------------------------

_CHART_TYPES = ("pie", "bar", "line")


def _get_chart_type(monitor_type: MonitorType) -> str | None:
    """Return 'pie' | 'bar' | 'line' if monitor_type renders as a chart, else None.

    Driven by configuration_schema — {"format": "chart", "chart_type": "pie"} —
    set by the PIE_CHART/BAR_CHART/LINE_CHART rows seeded alongside the
    original six monitor types.
    """
    if not monitor_type or not monitor_type.configuration_schema:
        return None
    try:
        cfg = json.loads(monitor_type.configuration_schema)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(cfg, dict) or cfg.get("format") != "chart":
        return None
    chart_type = cfg.get("chart_type")
    return chart_type if chart_type in _CHART_TYPES else None


def _chart_instructions(monitor_type: MonitorType) -> str:
    """Prompt snippet describing the expected shape, or '' for non-chart monitors."""
    chart_type = _get_chart_type(monitor_type)
    if not chart_type:
        return ""
    return MONITOR_CHART_INSTRUCTIONS.format(chart_type=chart_type)


def _format_chart_value(chart_type: str, parsed) -> dict:
    """Normalise parsed JSON into {"chartType": ..., "data": [...]}.

    Accepts either a bare list of rows (typical DB tool output, or a direct
    LLM answer) or an already-wrapped {"chartType": ..., "data": [...]}
    object. chart_type always comes from the monitor's own configuration —
    never trusted from the LLM — so a monitor can't render as the wrong
    chart just because the model echoed a different chartType string.

    Raises:
        MonitorExecutionError if the data isn't a non-empty list of rows
        each carrying at least a label and a value.
    """
    if isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
        data = parsed["data"]
    elif isinstance(parsed, list):
        data = parsed
    else:
        raise MonitorExecutionError(
            f"This monitor renders as a {chart_type} chart, but the query did "
            "not return a list of rows to chart.",
            repairable=True,
        )

    if not data:
        raise MonitorExecutionError("The query returned no rows to chart.")

    if not all(isinstance(row, dict) and len(row) >= 2 for row in data):
        raise MonitorExecutionError(
            "Each chart row must have a label and a numeric value.",
            repairable=True,
        )

    return {"chartType": chart_type, "data": data}


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

# Keys DB/MCP tools commonly wrap result rows in. Unwrapped before looking
# for the actual cell so {"rows": [[20]], "columns": [...]} doesn't read as
# two values.
_ROW_ENVELOPE_KEYS = ("value", "rows", "data", "result", "results", "records")


def _unwrap_envelope(node):
    """Peel result wrappers until we reach the payload itself."""
    for _ in range(4):  # bounded — these never nest deeply
        if not isinstance(node, dict):
            break
        for key in _ROW_ENVELOPE_KEYS:
            if key in node:
                node = node[key]
                break
        else:
            break
    return node


def _collect_scalars(node, out: list, depth: int = 0) -> None:
    """Gather leaf values, stopping early once ambiguity is proven."""
    if depth > 4 or len(out) > 1:
        return
    if isinstance(node, dict):
        for v in node.values():
            _collect_scalars(v, out, depth + 1)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _collect_scalars(v, out, depth + 1)
    elif node is not None:
        out.append(node)


def _single_scalar(parsed):
    """Return the one scalar in a result payload, or None if 0 or 2+.

    A DB tool can express "the answer is 20" in many shapes depending on the
    MCP server — {"COUNT(*)": 20}, [{"count": 20}], [[20]], [20],
    {"rows": [[20]]}. All of them mean the same thing and must be accepted;
    anything genuinely ambiguous (two or more values) must not be.
    """
    unwrapped = _unwrap_envelope(parsed)
    found: list = []
    _collect_scalars(unwrapped, found)
    return found[0] if len(found) == 1 else None


def _parse_number_token(token) -> float | None:
    """Parse one value as a number, tolerating formatting a DB/LLM may add."""
    if isinstance(token, bool):
        return None  # bool is an int subclass — don't silently read True as 1
    if isinstance(token, (int, float)):
        return float(token)
    if not isinstance(token, str):
        return None
    text = token.strip().replace(",", "").replace("_", "")
    text = text.lstrip("$₹€£¥").rstrip("%").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _coerce_number(raw: str) -> float | None:
    """Best-effort extraction of a single number from tool/LLM output.

    Returns None only when there is no unambiguous single number present.
    """
    direct = _parse_number_token(raw)
    if direct is not None:
        return direct

    parsed = _parse_llm_json(raw)
    if parsed is None:
        return None

    scalar = _single_scalar(parsed)
    if scalar is None:
        return None
    return _parse_number_token(scalar)


_TRUE_TOKENS = {"true", "yes", "y", "1", "t"}
_FALSE_TOKENS = {"false", "no", "n", "0", "f"}


def _coerce_boolean(raw: str) -> bool | None:
    """Extract a yes/no answer, accepting the same result shapes as numbers."""
    def _token_to_bool(token):
        if isinstance(token, bool):
            return token
        if isinstance(token, (int, float)):
            if token == 1:
                return True
            if token == 0:
                return False
            return None
        if isinstance(token, str):
            lowered = token.strip().lower()
            if lowered in _TRUE_TOKENS:
                return True
            if lowered in _FALSE_TOKENS:
                return False
        return None

    direct = _token_to_bool(raw)
    if direct is not None:
        return direct

    parsed = _parse_llm_json(raw)
    if parsed is None:
        return None

    scalar = _single_scalar(parsed)
    if scalar is None:
        return None
    return _token_to_bool(scalar)


def _output_columns(value) -> list[str]:
    """Output column names in a formatted JSON result value, in order.

    Covers the three shapes the frontend renders: a chart envelope, a list of
    row dicts, and a single flat record.
    """
    if isinstance(value, dict) and isinstance(value.get("data"), list):
        rows = value["data"]
    elif isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        rows = [value]
    else:
        return []

    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return columns


def _format_result(
    monitor_type: MonitorType, raw_text: str, sql: str | None = None,
) -> dict:
    """Convert raw LLM/tool output into a typed result dict.

    Shape follows monitor_type.result_type:
      NUMBER  → {"value": <float>, "is_money": <bool>}
      BOOLEAN → {"value": <bool>, "message": "<str>"}
      TEXT    → {"value": "<str>"}
      JSON    → {"value": <parsed_json>, "money_fields": [<column>, ...]}

    ``sql`` is the query that produced the output. It is what lets an aliased
    aggregate (``SUM(miro_value) AS total_spend``) be recognised as currency —
    the client sees only the alias and could never work that out.

    Raises:
        MonitorExecutionError when the output cannot honestly be represented
        as the configured result_type. Returning the raw text instead would
        store prose like "I am unable to provide a numeric count" as if it
        were a NUMBER result.
    """
    result_type = (monitor_type.result_type or "TEXT").upper()
    raw = (raw_text or "").strip()

    if not raw:
        raise MonitorExecutionError("The query returned no output.")

    if _looks_like_refusal(raw, result_type=result_type):
        raise MonitorExecutionError(
            f"The AI could not answer this question: {raw[:200]}"
        )

    if result_type == "NUMBER":
        number = _coerce_number(raw)
        if number is None:
            raise MonitorExecutionError(
                "This monitor expects a single number, but the query returned: "
                f"{raw[:200]}",
                repairable=True,
            )
        return {"value": number, "is_money": detect_scalar_is_money(sql)}

    if result_type == "BOOLEAN":
        boolean = _coerce_boolean(raw)
        if boolean is None:
            raise MonitorExecutionError(
                f"This monitor expects a yes/no answer, but the query returned: {raw[:200]}",
                repairable=True,
            )
        return {"value": boolean, "message": raw}

    if result_type == "JSON":
        parsed = _parse_llm_json(raw)
        if parsed is None:
            raise MonitorExecutionError(
                "This monitor expects structured/tabular data, but the query "
                f"returned: {raw[:200]}",
                repairable=True,
            )

        chart_type = _get_chart_type(monitor_type)
        if chart_type:
            value = _format_chart_value(chart_type, parsed)
        else:
            value = parsed

        return {
            "value": value,
            "money_fields": detect_money_fields(_output_columns(value), sql),
        }

    # TEXT or fallback
    return {"value": raw}


# ---------------------------------------------------------------------------
# Schema context builder
# ---------------------------------------------------------------------------

def _build_tool_descriptions(mcp_tools: list[dict]) -> str:
    """Build a compact text block describing available MCP tools."""
    if not mcp_tools:
        return "(no tools available)"

    parts = []
    for t in mcp_tools:
        name = t.get("tool_name", "unknown")
        desc = t.get("description", "")
        server = t.get("server_name", "")
        qb = "DB query tool" if t.get("requires_query_building") else "API tool"
        parts.append(f"- {name} ({qb}, server: {server}): {desc}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# MonitorAgent
# ---------------------------------------------------------------------------

class MonitorAgent:
    """Executes a monitor's natural_query through LLM + MCP tools.

    Three phases:
      1. Query generation — LLM translates natural_query into SQL/tool-call
      2. Execution — runs the query via MCP or answers directly
      3. Formatting — shapes the result per monitor_type.result_type

    Does NOT stream — returns the final result once complete.
    """

    # How many times a rejected query may be rewritten. Each attempt costs one
    # LLM call plus one database round-trip, so this is deliberately bounded —
    # but more than one, because errors often surface in sequence (bad column,
    # then bad dialect construct) and a single retry can't converge.
    MAX_SQL_REPAIR_ATTEMPTS = 3

    def __init__(
        self,
        monitor: Monitor,
        monitor_type: MonitorType,
        mcp_tools: list[dict],
        db_session: Session,
        api_key: str,
        schema_context: str = "",
    ):
        self._monitor = monitor
        self._monitor_type = monitor_type
        self._mcp_tools = mcp_tools
        self._db = db_session
        self._api_key = api_key
        self._schema_context = schema_context

    # ── Phase 1: Query Generation ─────────────────────────────────────────

    async def generate_query(
        self,
        followup_answer: str | None = None,
    ) -> dict:
        """Ask the LLM to generate an executable query.

        Returns:
            {"status": "query", "query": "...", "tool_name": "..." | None}
            {"status": "followup", "followup": "..."}
            {"status": "error", "error": "..."}
        """
        tool_descriptions = _build_tool_descriptions(self._mcp_tools)
        result_type = self._monitor_type.result_type or "TEXT"
        chart_instructions = _chart_instructions(self._monitor_type)

        if followup_answer and self._monitor.followup_question:
            # Re-invoke with the user's answer to the follow-up
            system = MONITOR_FOLLOWUP_SYSTEM.format(
                natural_query=self._monitor.natural_query,
                followup_question=self._monitor.followup_question,
                followup_answer=followup_answer,
                schema_context=self._schema_context or "(not available)",
                chart_instructions=chart_instructions,
                tool_descriptions=tool_descriptions,
            )
            messages = [
                {"role": "system", "content": system},
            ]
        else:
            system = MONITOR_QUERY_GENERATION_SYSTEM.format(
                schema_context=self._schema_context or "(not available)",
                tool_descriptions=tool_descriptions,
            )
            user_prompt = MONITOR_QUERY_GENERATION_USER.format(
                natural_query=self._monitor.natural_query,
                result_type=result_type,
                chart_instructions=chart_instructions,
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ]

        try:
            raw = await _call_llm(messages, self._api_key)
            parsed = _parse_llm_json(raw)

            if not parsed:
                log.warning(
                    "Monitor %s: LLM did not return valid JSON: %r",
                    self._monitor.id, raw[:200],
                )
                return {"status": "error", "error": f"Invalid LLM response: {raw[:200]}"}

            # A follow-up asking the user which table/column to use is the
            # agent failing to do its own job. Push back once with the schema
            # still in context rather than passing the buck to the user.
            if "followup" in parsed and _is_schema_question(parsed["followup"]):
                log.warning(
                    "Monitor %s: LLM asked a schema question (%r) — retrying with correction",
                    self._monitor.id, str(parsed["followup"])[:120],
                )
                correction = MONITOR_SCHEMA_FOLLOWUP_CORRECTION.format(
                    followup_question=parsed["followup"],
                )
                retry_messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": correction},
                ]
                raw = await _call_llm(retry_messages, self._api_key)
                parsed = _parse_llm_json(raw) or {}

                if "followup" in parsed and _is_schema_question(parsed["followup"]):
                    # Still asking. Fail honestly instead of making the user
                    # guess at their own database layout.
                    return {
                        "status": "error",
                        "error": (
                            "Could not identify which tables in your database answer "
                            "this question. Try naming the business entity you mean "
                            "(for example 'repair jobs' or 'purchase orders')."
                        ),
                    }

            if "followup" in parsed:
                # LLM needs clarification
                return {"status": "followup", "followup": parsed["followup"]}

            if "query" in parsed:
                return {
                    "status": "query",
                    "query": parsed["query"],
                    "tool_name": parsed.get("tool_name"),
                }

            return {"status": "error", "error": f"Unexpected LLM response: {parsed}"}

        except Exception as exc:
            log.error(
                "Monitor %s: query generation failed: %s",
                self._monitor.id, exc, exc_info=True,
            )
            return {"status": "error", "error": str(exc)}

    # ── Phase 2: Execution ────────────────────────────────────────────────

    async def execute_query(self, query: str, tool_name: str | None) -> str:
        """Execute the generated query via MCP tool or return the query as-is.

        Returns the raw text result from the tool call.

        Raises:
            MonitorExecutionError if the query is unsafe/invalid, the tool is
            missing, or the tool reports a failure. Errors must surface as
            errors — never as a result value.
        """
        if not self._mcp_tools:
            # No tools configured — return the query itself as the answer
            # (the LLM likely generated a direct answer instead of SQL)
            return query

        if not tool_name:
            # The LLM is allowed to omit tool_name, but a generated SQL query
            # must still be executed — fall back to the available DB
            # query-building tool rather than echoing the SQL back as text.
            # requires_query_building is set per-SERVER (see
            # gather_org_mcp_tools), so every tool on that server carries it —
            # including the schema/introspection tool. Exclude it by name
            # (same "schema" substring check _load_schema_context uses) so we
            # don't run the SQL against the schema lookup tool by mistake.
            db_tool = next(
                (t for t in self._mcp_tools
                 if t.get("requires_query_building")
                 and "schema" not in t.get("tool_name", "").lower()),
                None,
            )
            if db_tool:
                tool_name = db_tool["tool_name"]
            else:
                return query

        # Find the matching tool's server
        tool_info = next(
            (t for t in self._mcp_tools if t["tool_name"] == tool_name),
            None,
        )
        if not tool_info:
            log.warning(
                "Monitor %s: tool '%s' not found in available tools",
                self._monitor.id, tool_name,
            )
            raise MonitorExecutionError(
                f"The AI tried to use a tool ('{tool_name}') that is not connected "
                "to your organization."
            )

        server_name = tool_info["server_name"]
        mcp_mgr = get_mcp_manager()

        # Determine arguments — for DB query tools, use the correct param name
        # from schema, and validate the SQL before it ever reaches the database.
        if tool_info.get("requires_query_building"):
            _validate_sql(query, _extract_table_names(self._schema_context))
            _validate_columns_and_values(query, _extract_table_columns(self._schema_context))
            props = tool_info.get("input_schema", {}).get("properties", {})
            param_name = "query" if "query" in props else "sql"
            args = {param_name: query}
        else:
            # For non-DB tools, try parsing the query as JSON args
            try:
                args = json.loads(query) if isinstance(query, str) else query
            except (json.JSONDecodeError, ValueError):
                args = {"query": query}

        try:
            result = await mcp_mgr.call_tool(server_name, tool_name, args)
        except Exception as exc:
            log.error(
                "Monitor %s: tool execution failed (%s/%s): %s",
                self._monitor.id, server_name, tool_name, exc,
            )
            raise MonitorExecutionError(
                f"Could not run the query: {exc}", repairable=True,
            ) from exc

        # Extract text from MCP result
        parts = []
        for item in (result.get("content") or []):
            text_val = item.get("text")
            if text_val:
                parts.append(str(text_val))
        text = "\n".join(parts).strip()

        # The MCP protocol reports tool-level failures via isError — without
        # this check a message like "Table 'x' doesn't exist" was stored as a
        # perfectly good monitor value.
        if result.get("isError"):
            log.warning(
                "Monitor %s: tool '%s' reported an error: %s",
                self._monitor.id, tool_name, text[:200],
            )
            raise MonitorExecutionError(
                text or f"Tool '{tool_name}' reported an error.", repairable=True,
            )

        if not text:
            raise MonitorExecutionError("The query ran but returned no data.")

        return text

    # ── Phase 2b: Self-repair ─────────────────────────────────────────────

    async def _repair_query(
        self,
        failed_query: str,
        error_message: str,
        tool_name: str | None,
        attempts: list[tuple[str, str]] | None = None,
    ) -> str | None:
        """Ask the LLM to rewrite SQL the database rejected.

        The database names its own limitation in the error text (e.g. "This
        version of MariaDB doesn't yet support 'LIMIT & IN/ALL/ANY/SOME
        subquery'"), so feeding it back lets the model adapt to the dialect
        without us having to detect it up front.

        ``attempts`` carries (query, error) pairs already tried, so the model
        stops cycling between two equally broken queries.

        Returns the corrected SQL, or None if repair wasn't possible.
        """
        attempts = attempts or []
        try:
            history = ""
            if attempts:
                lines = "\n".join(
                    f"  {i}. {q}\n     → failed with: {e}"
                    for i, (q, e) in enumerate(attempts, start=1)
                )
                history = f"\nAlready tried (do NOT repeat these):\n{lines}\n"

            prompt = MONITOR_SQL_REPAIR_PROMPT.format(
                natural_query=self._monitor.natural_query,
                previous_attempts=history,
                failed_query=failed_query,
                error_message=error_message,
                schema_context=self._schema_context or "(not available)",
                tool_name=tool_name or "",
            )
            raw = await _call_llm([{"role": "user", "content": prompt}], self._api_key)
            parsed = _parse_llm_json(raw)
            if not isinstance(parsed, dict):
                return None
            repaired = parsed.get("query")
            if not repaired or not isinstance(repaired, str):
                return None

            # Reject a query we already know fails — otherwise the loop just
            # burns attempts re-running the same broken SQL.
            already_tried = {q.strip() for q, _ in attempts}
            already_tried.add((failed_query or "").strip())
            if repaired.strip() in already_tried:
                return None
            return repaired
        except Exception as exc:
            log.warning("Monitor %s: SQL repair failed: %s", self._monitor.id, exc)
            return None

    async def run_and_format(
        self, query: str, tool_name: str | None,
    ) -> tuple[dict, str]:
        """Execute a query and shape the result, repairing the SQL when the
        database rejects it.

        Each failure is fed back with the full history of what has already been
        tried, so successive attempts converge instead of cycling through a
        different error every time. Bounded by MAX_SQL_REPAIR_ATTEMPTS.

        Returns (formatted_result, query_actually_used) so the caller can
        persist the repaired SQL rather than the broken original.
        """
        attempts: list[tuple[str, str]] = []
        current = query

        for attempt_no in range(self.MAX_SQL_REPAIR_ATTEMPTS + 1):
            try:
                raw = await self.execute_query(current, tool_name)
                result = _format_result(self._monitor_type, raw, sql=current)
                if attempt_no:
                    log.info(
                        "Monitor %s: SQL repair succeeded on attempt %d",
                        self._monitor.id, attempt_no,
                    )
                return result, current
            except MonitorExecutionError as exc:
                # Only SQL problems are worth a rewrite, and only when we are
                # actually running SQL through a tool.
                if not exc.repairable or not self._mcp_tools:
                    raise
                if attempt_no >= self.MAX_SQL_REPAIR_ATTEMPTS:
                    raise

                log.info(
                    "Monitor %s: repair attempt %d after: %s",
                    self._monitor.id, attempt_no + 1, str(exc)[:160],
                )
                repaired = await self._repair_query(
                    current, str(exc), tool_name, attempts,
                )
                if not repaired:
                    raise

                attempts.append((current, str(exc)))
                current = repaired

        # Unreachable — the loop either returns or raises.
        raise MonitorExecutionError("The query could not be run.")

    # ── Phase 3: Direct LLM (no tools) ────────────────────────────────────

    async def direct_llm_answer(self) -> str:
        """Answer the query directly from LLM knowledge (no MCP tools)."""
        user_prompt = MONITOR_DIRECT_LLM_PROMPT.format(
            natural_query=self._monitor.natural_query,
            result_type=self._monitor_type.result_type or "TEXT",
            chart_instructions=_chart_instructions(self._monitor_type),
        )
        system_prompt = MONITOR_SYSTEM_CONTEXT.format(
            natural_query=self._monitor.natural_query,
            monitor_type_name=self._monitor_type.name,
            monitor_type_description=self._monitor_type.description or "",
            result_type=self._monitor_type.result_type or "TEXT",
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            return await _call_llm(messages, self._api_key)
        except Exception as exc:
            log.error(
                "Monitor %s: direct LLM call failed: %s",
                self._monitor.id, exc, exc_info=True,
            )
            return ""

    # ── Full execution pipeline ───────────────────────────────────────────

    async def execute(
        self,
        followup_answer: str | None = None,
    ) -> dict:
        """Run the full monitor pipeline.

        Returns:
            {
                "status": "success" | "followup" | "error",
                "result": {...} | None,
                "followup_question": "..." | None,
                "error": "..." | None,
            }
        """
        # ── No tools → direct LLM answer ─────────────────────────────────
        if not self._mcp_tools:
            log.info(
                "Monitor %s: no MCP tools — using direct LLM answer",
                self._monitor.id,
            )
            raw_answer = await self.direct_llm_answer()
            try:
                result = _format_result(self._monitor_type, raw_answer)
            except MonitorExecutionError as exc:
                return {"status": "error", "error": str(exc), "result": None}
            return {"status": "success", "result": result}

        # ── Has tools → query generation + execution ──────────────────────

        # If we already have an llm_query and no follow-up needed, reuse it
        if self._monitor.llm_query and not followup_answer:
            log.info(
                "Monitor %s: reusing existing llm_query",
                self._monitor.id,
            )
            parsed = _parse_llm_json(self._monitor.llm_query)
            if parsed and "query" in parsed:
                query = parsed["query"]
                tool_name = parsed.get("tool_name")
            else:
                query = self._monitor.llm_query
                tool_name = None

            try:
                result, used_query = await self.run_and_format(query, tool_name)
            except MonitorExecutionError as exc:
                # The cached query is bad — drop it so the next run regenerates
                # instead of failing the same way forever.
                update_monitor_llm_query(self._monitor, "", self._db)
                return {"status": "error", "error": str(exc), "result": None}

            # If a repair was needed, cache the fixed SQL, not the broken one.
            if used_query != query:
                update_monitor_llm_query(
                    self._monitor,
                    json.dumps({"query": used_query, "tool_name": tool_name}),
                    self._db,
                    followup_question=None,
                )
            return {"status": "success", "result": result}

        # Generate the query
        gen_result = await self.generate_query(followup_answer=followup_answer)

        if gen_result["status"] == "followup":
            # LLM needs more info — persist the question and return
            followup_q = gen_result["followup"]
            update_monitor_llm_query(
                self._monitor, "", self._db, followup_question=followup_q,
            )
            return {
                "status": "followup",
                "followup_question": followup_q,
                "result": None,
            }

        if gen_result["status"] == "error":
            return {
                "status": "error",
                "error": gen_result["error"],
                "result": None,
            }

        # We got a query — validate + execute before persisting it, so a bad
        # query is never cached as this monitor's llm_query.
        query = gen_result["query"]
        tool_name = gen_result.get("tool_name")

        try:
            result, query = await self.run_and_format(query, tool_name)
        except MonitorExecutionError as exc:
            return {"status": "error", "error": str(exc), "result": None}

        query_json = json.dumps({"query": query, "tool_name": tool_name})
        update_monitor_llm_query(
            self._monitor, query_json, self._db, followup_question=None,
        )

        return {"status": "success", "result": result}


# ---------------------------------------------------------------------------
# Schema context loader
# ---------------------------------------------------------------------------

# Above this rendered size the full column-level schema is narrowed to the
# tables relevant to the question. Blind truncation is never acceptable: it
# used to cut the table list alphabetically, hiding most of a large database
# and leaving the model no option but to ask the user which table to use.
_MAX_SCHEMA_CHARS = 24000


def _parse_schema_payload(text: str) -> dict[str, str]:
    """Parse a schema tool's JSON output into {table_name: "Table: x (cols)"}.

    Returns {} when the payload isn't the recognised table/columns shape, in
    which case the caller falls back to using the raw text.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}

    if not isinstance(data, list) or not data:
        return {}
    if not isinstance(data[0], dict) or "table_name" not in data[0]:
        return {}

    tables: dict[str, str] = {}
    for table in data:
        if not isinstance(table, dict):
            continue
        t_name = table.get("table_name")
        if not t_name:
            continue
        cols_data = table.get("columns", [])
        try:
            cols = json.loads(cols_data) if isinstance(cols_data, str) else cols_data
            col_defs = [
                f"{c.get('column_name')} {c.get('data_type')}"
                for c in cols if isinstance(c, dict)
            ]
            tables[t_name] = f"Table: {t_name} ({', '.join(col_defs)})"
        except Exception:
            tables[t_name] = f"Table: {t_name}"

    return tables


async def _select_relevant_tables(
    natural_query: str,
    table_names: list[str],
    api_key: str,
) -> list[str]:
    """Ask the LLM which tables are needed, given every table NAME.

    Names alone are small enough to fit even for a several-hundred-table
    database, so the model sees the whole database rather than an
    alphabetically truncated slice of it.

    Returns [] on any failure — the caller then degrades to a name-only schema.
    """
    try:
        messages = [
            {
                "role": "system",
                "content": MONITOR_TABLE_SELECTION_SYSTEM.format(
                    table_names="\n".join(f"- {n}" for n in table_names),
                ),
            },
            {
                "role": "user",
                "content": MONITOR_TABLE_SELECTION_USER.format(
                    natural_query=natural_query,
                ),
            },
        ]
        raw = await _call_llm(messages, api_key)
        parsed = _parse_llm_json(raw)
        if not isinstance(parsed, dict):
            return []
        picked = parsed.get("tables")
        if not isinstance(picked, list):
            return []
        # Only keep names that really exist, case-insensitively.
        by_lower = {n.lower(): n for n in table_names}
        return [
            by_lower[str(p).lower()]
            for p in picked
            if str(p).lower() in by_lower
        ]
    except Exception as exc:
        log.warning("Table selection failed: %s", exc)
        return []

_BAJAJ_TABLES = frozenset({
    "pending_pr", "pr_to_po_conversion", "ordering_trend", "delivery_analysis",
})
_BAJAJ_COLUMN_MAP_PATH = Path(__file__).resolve().parents[2] / "prompts" / "COLUMN_MAP_bajaj.md"


@lru_cache(maxsize=1)
def _load_bajaj_column_map() -> str:
    try:
        return _BAJAJ_COLUMN_MAP_PATH.read_text()
    except OSError:
        log.warning("Bajaj column map not found at %s", _BAJAJ_COLUMN_MAP_PATH)
        return ""


def _augment_schema_with_column_map(schema_context: str) -> str:
    """Append the bajaj column map when the retrieved schema touches its tables."""
    if not schema_context or not (_extract_table_names(schema_context) & _BAJAJ_TABLES):
        return schema_context
    column_map = _load_bajaj_column_map()
    if not column_map:
        return schema_context
    return (
        f"{schema_context}\n\n"
        "--- Business column reference (authoritative — use these exact "
        "column names, enum codes, and joins) ---\n"
        f"{column_map}"
    )


async def _load_schema_context_via_engine(
    db_servers: set[str],
    mcp_tools: list[dict],
    natural_query: str,
    api_key: str | None,
) -> str:
    """Build schema context with the shared schema_engine (the /chat path).

    The engine does BM25 + vector + FK-graph hybrid retrieval against a
    pre-summarised, pre-embedded index, returning only the tables relevant to
    ``natural_query`` within a token budget. Because every store is keyed by
    server_name, monitors reuse whatever index chat has already built.

    ``agent_id`` is None: monitors are organization-scoped and have no agent.
    That only skips the is_cold_build bookkeeping — retrieval is unaffected.

    Returns "" if the engine yields nothing, so the caller can fall back.
    """
    from ai_agent.schema_engine.orchestrator import get_orchestrator

    mcp_mgr = get_mcp_manager()
    parts: list[str] = []

    for server_name in db_servers:
        server_tools = [
            t for t in mcp_tools if t.get("server_name") == server_name
        ]
        orch = await get_orchestrator(
            server_name=server_name,
            mcp_manager=mcp_mgr,
            server_tools=server_tools,
            agent_id=None,
            api_key=api_key,
        )
        ctx = await orch.build_context(natural_query)
        if ctx:
            log.info(
                "Monitor schema engine: '%s' → %d chars from %d tables",
                server_name, len(ctx), orch.table_count,
            )
            parts.append(ctx)

    return "\n\n".join(parts)


async def _load_schema_context(
    mcp_tools: list[dict],
    natural_query: str | None = None,
    api_key: str | None = None,
) -> str:
    """Load schema context for the DB MCP servers available to this monitor.

    Primary path is the shared ``schema_engine`` — the same retrieval pipeline
    /chat uses, which is why chat can answer questions about a 400-table
    database without asking the user to name tables.

    If the engine is unavailable (not provisioned, Milvus/Redis down, cold
    build fails), falls back to reading the schema tool directly and narrowing
    it, so monitors keep working rather than hard-failing.
    """
    db_servers = {
        t["server_name"]
        for t in mcp_tools
        if t.get("requires_query_building")
    }

    if not db_servers:
        return ""

    if natural_query:
        try:
            ctx = await _load_schema_context_via_engine(
                db_servers, mcp_tools, natural_query, api_key,
            )
            if ctx:
                return ctx
            log.warning("Schema engine returned empty context — using fallback loader")
        except Exception as exc:
            log.warning(
                "Schema engine unavailable (%s) — using fallback loader", exc,
            )

    mcp_mgr = get_mcp_manager()
    all_tables: dict[str, str] = {}
    raw_parts: list[str] = []

    for server_name in db_servers:
        try:
            # Look for a schema tool (e.g., get_schema, describe_tables)
            tools = await mcp_mgr.list_tools(server_name)
            schema_tool = next(
                (t for t in tools if "schema" in t["name"].lower()),
                None,
            )
            if not schema_tool:
                continue

            result = await mcp_mgr.call_tool(server_name, schema_tool["name"], {})
            for item in (result.get("content") or []):
                text = item.get("text")
                if not text:
                    continue
                parsed_tables = _parse_schema_payload(text)
                if parsed_tables:
                    all_tables.update(parsed_tables)
                else:
                    raw_parts.append(f"-- Server: {server_name}\n{text}")
        except Exception as exc:
            log.warning(
                "Failed to load schema from MCP server '%s': %s",
                server_name, exc,
            )

    if not all_tables:
        # Unrecognised payload shape — fall back to the raw text.
        return "\n\n".join(raw_parts)

    rendered = "\n".join(all_tables.values())
    if len(rendered) <= _MAX_SCHEMA_CHARS:
        return rendered

    log.info(
        "Schema too large (%d chars, %d tables) — narrowing to relevant tables",
        len(rendered), len(all_tables),
    )

    if natural_query and api_key:
        picked = await _select_relevant_tables(
            natural_query, list(all_tables), api_key,
        )
        if picked:
            narrowed = "\n".join(all_tables[t] for t in picked)
            # Always include the full table-name index too, so the model can
            # still spot a better table than the ones pre-selected.
            index = ", ".join(all_tables)
            return (
                f"{narrowed}\n\n"
                f"-- All tables in this database:\n-- {index}"
            )

    # Degrade to names only rather than truncating the list — an incomplete
    # picture of every table beats a complete picture of the first few.
    return "\n".join(f"Table: {name}" for name in all_tables)


# ---------------------------------------------------------------------------
# Module-level entry point
# ---------------------------------------------------------------------------

async def execute_monitor(
    monitor_id: int,
    db_session: Session,
    followup_answer: str | None = None,
) -> dict:
    """Execute a monitor and persist the result.

    Called by:
    - POST /api/v1/monitor/{id}/run   (manual trigger)
    - Celery task run_monitor()        (scheduled — later)

    Returns:
        {
            "status": "success" | "followup" | "error",
            "result": MonitorResult | None,
            "followup_question": str | None,
            "error": str | None,
        }

    Raises:
        ValueError  if monitor not found
    """
    # ── Load context ──────────────────────────────────────────────────────
    monitor, monitor_type = load_monitor_with_context(monitor_id, db_session)

    if not monitor:
        raise ValueError(f"Monitor {monitor_id} not found or inactive")
    if not monitor_type:
        raise ValueError(f"MonitorType {monitor.monitor_type_id} for monitor {monitor_id} not found")

    # ── Get API key from environment ──────────────────────────────────────
    try:
        api_key = _get_openai_key()
    except ValueError as exc:
        return {
            "status": "error",
            "error": str(exc),
            "result": None,
        }

    # ── Gather organization MCP tools ─────────────────────────────────────
    mcp_tools: list[dict] = []
    if monitor.organization_id:
        try:
            mcp_tools = await gather_org_mcp_tools(
                monitor.organization_id, db_session,
            )
            log.info(
                "Monitor %s: %d MCP tool(s) loaded for org %s",
                monitor_id,
                len(mcp_tools),
                monitor.organization_id,
            )
        except Exception as exc:
            log.warning(
                "Monitor %s: failed to load MCP tools — will run LLM-only: %s",
                monitor_id,
                exc,
            )

    # ── Load schema context for DB tools ──────────────────────────────────
    schema_context = ""
    if mcp_tools:
        try:
            schema_context = await _load_schema_context(
                mcp_tools,
                natural_query=monitor.natural_query,
                api_key=api_key,
            )
            schema_context = _augment_schema_with_column_map(schema_context)
        except Exception as exc:
            log.warning("Monitor %s: schema context load failed: %s", monitor_id, exc)

    # ── Execute ───────────────────────────────────────────────────────────
    start = datetime.now(timezone.utc)
    try:
        agent = MonitorAgent(
            monitor=monitor,
            monitor_type=monitor_type,
            mcp_tools=mcp_tools,
            db_session=db_session,
            api_key=api_key,
            schema_context=schema_context,
        )
        outcome = await agent.execute(followup_answer=followup_answer)
        execution_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)

        # ── Follow-up → return early, no result persisted yet ─────────────
        if outcome["status"] == "followup":
            return {
                "status": "followup",
                "followup_question": outcome.get("followup_question"),
                "result": None,
            }

        # ── Error ─────────────────────────────────────────────────────────
        if outcome["status"] == "error":
            raw_error = outcome.get("error") or "Unknown error"
            # The raw engine error is kept in the log for debugging; the user
            # only ever sees a message written for them.
            log.warning("Monitor %s failed: %s", monitor_id, raw_error)
            user_error = _humanize_error(raw_error)

            result_row = persist_monitor_result(
                monitor=monitor,
                monitor_type=monitor_type,
                result={},
                status="FAILED",
                execution_time_ms=execution_ms,
                db_session=db_session,
                error_message=user_error[:1000],
            )
            return {
                "status": "error",
                "error": user_error,
                "result": result_row,
            }

        # ── Success ───────────────────────────────────────────────────────
        result_payload = outcome.get("result", {"value": ""})
        log.info(
            "Monitor %s: result_payload=%r",
            monitor_id,
            str(result_payload)[:120],
        )

        result_row = persist_monitor_result(
            monitor=monitor,
            monitor_type=monitor_type,
            result=result_payload,
            status="SUCCESS",
            execution_time_ms=execution_ms,
            db_session=db_session,
        )
        return {
            "status": "success",
            "result": result_row,
        }

    except Exception as exc:
        log.error("Monitor %s execution failed: %s", monitor_id, exc, exc_info=True)
        execution_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        user_error = _humanize_error(str(exc))
        result_row = persist_monitor_result(
            monitor=monitor,
            monitor_type=monitor_type,
            result={},
            status="FAILED",
            execution_time_ms=execution_ms,
            db_session=db_session,
            error_message=user_error[:1000],
        )
        return {
            "status": "error",
            "error": user_error,
            "result": result_row,
        }
