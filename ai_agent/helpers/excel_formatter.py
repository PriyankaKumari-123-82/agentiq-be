"""LLM-powered column filtering for Excel / CSV exports.

When data comes from an external source (MCP tool with requires_query_building=False),
this helper makes a lightweight LLM call to decide which columns are actually
relevant to the user's query and strips the rest before generating the file.

For database query results (requires_query_building=True) the caller should
skip this step entirely — every DB column the query selected is intentional.

Key design: the LLM is shown **pre-flattened + IST-converted** sample values
so it can inspect actual content (e.g. see that 'Online Meeting URL' is
'Not Available' vs 'Online Meeting' is a real URL, or that 'Body Preview'
contains a wall of HTML). The column list sent back is always validated against
the original raw keys so nothing gets lost.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)# Maximum number of sample rows to send to the LLM for context.
_SAMPLE_ROWS = 5

# Minimum columns to keep even if LLM returns a very short list.
_MIN_KEEP = 1


async def filter_columns_via_llm(
    rows: List[Dict[str, Any]],
    user_query: str,
    llm: Any,
    litellm_model_string: Optional[str],
    api_key: Optional[str],
) -> List[str]:
    """Make an LLM call to pick only the relevant columns for the user's query.

    Parameters
    ----------
    rows:
        The **raw** data rows from the MCP tool (before any transformation).
    user_query:
        The original user message / request that triggered this data fetch.
    llm:
        An ``LLMGateway`` instance (or any object with a ``chat()`` coroutine).
    litellm_model_string:
        LiteLLM model string to use (e.g. ``"openai/gpt-4o-mini"``).
    api_key:
        API key for the LLM provider.

    Returns
    -------
    list[str]
        The column keys to **keep**. Falls back to all columns on any error.

    Design note
    -----------
    We extract the column list from the **raw** rows (so every top-level key is
    captured), but we show the LLM **pre-flattened + IST-converted** sample
    values. This means the LLM sees:
      - ``"Online Meeting": "https://teams.microsoft.com/..."``  → keep
      - ``"Online Meeting URL": "Not Available"``               → drop
      - ``"Body Preview": "_______ Microsoft Teams meeting ..."`` → drop
    instead of opaque nested dicts, making its column-relevance judgment
    much more accurate.
    """
    # Lazy import to avoid circular dependency (excel_formatter ← date_formatter
    # is fine; the reverse would not be).
    from ai_agent.helpers.date_formatter import transform_dates_in_obj

    if not rows:
        return []

    # ── Step 1: extract all top-level column names from the raw data ──────────
    all_columns = list(rows[0].keys())

    if not all_columns:
        return []

    # If there are very few columns, no filtering is needed.
    if len(all_columns) <= 5:
        return all_columns

    # ── Step 2: build a flattened sample so the LLM sees real values ──────────
    # Apply IST date conversion, then flatten nested dicts/lists to scalars.
    # We do this only on the first _SAMPLE_ROWS rows — just for the prompt.
    raw_sample = rows[:_SAMPLE_ROWS]
    try:
        converted_sample = transform_dates_in_obj(raw_sample)
        # apply_cell_flattening is defined later in this file; call it directly
        # rather than importing ourselves.
        flattened_sample = apply_cell_flattening(converted_sample)
    except Exception:
        # If anything goes wrong, fall back to the raw sample
        flattened_sample = raw_sample

    # Truncate very long string values in the sample so the prompt stays compact.
    def _truncate(v: Any, max_len: int = 120) -> Any:
        if isinstance(v, str) and len(v) > max_len:
            return v[:max_len] + " ...[truncated]"
        return v

    display_sample = [
        {k: _truncate(v) for k, v in row.items()}
        for row in flattened_sample
    ]

    sample_json = json.dumps(display_sample, ensure_ascii=False, default=str)

    prompt = f"""You are a data analyst helping to prepare an Excel export for a business user.

The user asked: "{user_query}"

Below are ALL column names available in the dataset, plus sample rows.

Column names:
{json.dumps(all_columns, ensure_ascii=False)}

Sample rows (first {_SAMPLE_ROWS} rows of data):
{sample_json}

Task: Select ONLY the columns that are directly useful and readable in a spreadsheet for the user's question.

ALWAYS EXCLUDE columns that are:
- Internal system identifiers: UUIDs, ETags, change keys, sync tokens, transaction IDs, occurrence IDs, series master IDs
- OData metadata: any column starting with @odata, calendar navigation links (Calendar@odata.*)
- API/Technical URLs: raw API links (e.g., 'url', 'comments_url', 'events_url'), OData navigation links
- Boolean/flag fields with limited value: "Is Online Meeting", "Is All Day", "Is Cancelled", "Is Reminder On", "Is Organizer", "Is Draft", "Hide Attendees", "Response Requested", "Allow New Time Proposals"
- Reminder settings: "Reminder Minutes Before Start"
- Raw body/preview content: "Body", "Body Preview" (raw HTML or long unformatted text)
- Timezone and internal scheduling fields: "Original Start Time Zone", "Original End Time Zone", "I Cal Uid", "Uid", "Change Key"
- Fields that duplicate other kept columns

ALWAYS KEEP columns that are:
- Event/item title or subject
- Start and end date/time
- Location or venue
- Attendees or participants
- Organizer or creator
- ANY column that can contain a web link or URL (e.g., "Online Meeting", "Join URL", "HTML URL", "Web Link"). ALWAYS include these even if the sample values are empty or "Not Available".
- Status or response status
- Any field directly answering the user's question

Return ONLY a JSON array of the column name strings to keep, in the same order they appear in the input list. No explanation, no markdown, just the raw JSON array.
Example output: ["Subject", "Start", "End", "Location", "Attendees", "Organizer", "Online Meeting"]"""

    try:
        response = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            model=litellm_model_string,
            temperature=0.0,
            max_tokens=500,
            api_key=api_key,
        )
        raw = (response.content or "").strip()

        # Strip optional markdown fences
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip().rstrip("```").strip()

        selected: Any = json.loads(raw)
        if not isinstance(selected, list):
            raise ValueError(f"LLM returned non-list: {type(selected)}")

        # Keep only keys that actually exist in the data (LLM might hallucinate)
        valid = [col for col in selected if col in all_columns]
        if len(valid) < _MIN_KEEP:
            log.warning(
                "filter_columns_via_llm: LLM returned too few valid columns "
                "(%d); falling back to all %d columns",
                len(valid),
                len(all_columns),
            )
            return all_columns

        log.info(
            "filter_columns_via_llm: kept %d/%d columns via LLM",
            len(valid),
            len(all_columns),
        )
        return valid

    except Exception as exc:
        log.warning(
            "filter_columns_via_llm: LLM call failed (%s); keeping all columns",
            exc,
        )
        return all_columns


def filter_rows_to_columns(
    rows: List[Dict[str, Any]],
    columns_to_keep: List[str],
) -> List[Dict[str, Any]]:
    """Return rows restricted to only the selected columns, in that order."""
    if not columns_to_keep:
        return rows
    keep_set = set(columns_to_keep)
    return [
        {k: row[k] for k in columns_to_keep if k in row}
        for row in rows
        if any(k in keep_set for k in row)
    ]


# ---------------------------------------------------------------------------
# Generic cell flattening
# ---------------------------------------------------------------------------

def flatten_cell_value(val: Any) -> Any:
    """Recursively flatten a nested dict/list into a human-readable scalar
    suitable for an Excel cell.

    The logic is intentionally **generic** — it recognizes common API
    response shapes (OData, Graph, GitHub, GitLab, Jira, Slack, etc.) without
    hard-coding any specific service:

    * ``{"dateTime": "...", ...}``           → the dateTime string
    * ``{"joinUrl": "https://..."}``         → the joinUrl string
    * ``{"login": "pk-pb", ...}``            → "pk-pb" (or "Name (login)")
    * ``{"html_url": "https://..."}``        → human web link
    * ``{"displayName": "..."}``              → displayName
    * ``{"full_name": "owner/repo"}``        → full_name
    * ``{"title": "..."}``                    → title
    * ``{"name": "...", "address": "..."}``   → "Name <address>"
    * ``{"emailAddress": {...}}``             → recurse into emailAddress
    * ``[item, item, ...]``                   → CSV of flattened items
    * Primitive (str/int/float/bool/None)     → as-is

    The function is safe to call on already-flat values; it returns them
    unchanged.
    """
    # ── Primitives ──────────────────────────────────────────────────────────
    if val is None or isinstance(val, (int, float, bool)):
        return val
    if isinstance(val, str):
        return val

    # ── List ────────────────────────────────────────────────────────────────
    if isinstance(val, list):
        flat_items = [flatten_cell_value(item) for item in val]
        # If every item flattened to a string/scalar, join them.
        readable = [str(item) for item in flat_items if item not in (None, "")]
        return ", ".join(readable) if readable else ""

    # ── Dict ────────────────────────────────────────────────────────────────
    if isinstance(val, dict):
        # 1. OData / Graph datetime objects: {"dateTime": "...", "timeZone": "..."}
        # After transform_dates_in_obj() the dateTime value is already IST.
        if "dateTime" in val:
            return val["dateTime"]

        # 2. Join / Meeting URLs: {"joinUrl": "https://...", ...}
        if "joinUrl" in val:
            return val["joinUrl"]

        # 3. User / Account objects with login or username (GitHub, GitLab, Jira, etc.)
        # e.g. {"login": "pk-pb", "id": 217000641, "avatar_url": "...", ...}
        if "login" in val:
            login = str(val["login"]).strip()
            name = str(val.get("name", "")).strip() if val.get("name") else ""
            if name and name != login:
                return f"{name} ({login})"
            return login

        if "username" in val:
            return val["username"]
        if "user_name" in val:
            return val["user_name"]

        # 4. Email address objects: {"name": "...", "address": "..."}
        if "name" in val and "address" in val:
            name = val.get("name", "").strip()
            addr = val.get("address", "").strip()
            if name and addr:
                return f"{name} <{addr}>"
            return name or addr

        # 5. Repository / Object with full_name (e.g. GitHub repos)
        if "full_name" in val:
            return val["full_name"]

        # 6. Display Name / Title / Subject / Label / Summary / Value
        if "displayName" in val:
            return val["displayName"]
        if "title" in val:
            return val["title"]
        if "summary" in val:
            return val["summary"]
        if "label" in val:
            return val["label"]
        if "name" in val:
            return val["name"]

        # 7. HTML / Web URLs (preferred human web links for PRs, issues, commits)
        # e.g. Pull Request dict: {"html_url": "https://github.com/...", "diff_url": "...", ...}
        if "html_url" in val:
            return val["html_url"]
        if "web_url" in val:
            return val["web_url"]

        # 8. Wrappers with nested user / author / emailAddress
        if "emailAddress" in val:
            return flatten_cell_value(val["emailAddress"])
        if "user" in val and isinstance(val["user"], dict):
            return flatten_cell_value(val["user"])
        if "author" in val and isinstance(val["author"], dict):
            return flatten_cell_value(val["author"])

        # 9. Attendee / response status objects:
        if "emailAddress" in val or ("status" in val and "emailAddress" in val):
            email_part = flatten_cell_value(val.get("emailAddress", {}))
            response = val.get("status", {})
            if isinstance(response, dict):
                reply = response.get("response", "")
            else:
                reply = str(response)
            if reply and reply not in ("none", "notResponded", ""):
                return f"{email_part} ({reply})"
            return email_part

        # 10. Generic URL / link fields
        if "url" in val and isinstance(val["url"], str) and val["url"].startswith("http"):
            return val["url"]
        if "href" in val and isinstance(val["href"], str) and val["href"].startswith("http"):
            return val["href"]

        # 11. Generic fallback: filter out technical metadata keys (URLs, IDs, system flags)
        # before joining remaining values to avoid dumping 20+ joined system fields
        filtered_parts = []
        for k, v in val.items():
            k_lower = str(k).lower()
            if any(k_lower.endswith(suffix) for suffix in ("_id", "id", "_url", "url", "_links", "etag", "sha", "key")):
                continue
            if k_lower in ("node_id", "gravatar_id", "avatar_url", "site_admin", "type", "type_name", "type_id", "state"):
                continue
            flat_v = flatten_cell_value(v)
            if flat_v not in (None, "", [], {}):
                filtered_parts.append(str(flat_v))

        if filtered_parts:
            return ", ".join(filtered_parts)

        # Fallback if filtering dropped everything: pick primary link or ID
        for preferred_key in ("html_url", "web_url", "url", "href", "id"):
            if preferred_key in val and val[preferred_key]:
                return str(val[preferred_key])

        parts = [str(flatten_cell_value(v)) for v in val.values() if v not in (None, "", [], {})]
        return ", ".join(parts) if parts else ""

    # Catch-all
    return str(val)


def apply_cell_flattening(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply ``flatten_cell_value`` to every value in every row.

    Call this **after** ``transform_dates_in_obj`` and **after**
    ``expand_list_columns`` so all remaining nested dicts are extracted
    as plain strings before the DataFrame is created.
    """
    return [
        {k: flatten_cell_value(v) for k, v in row.items()}
        for row in rows
    ]


# ---------------------------------------------------------------------------
# List-column expansion
# ---------------------------------------------------------------------------

def _singularize(col_name: str) -> str:
    """Best-effort singularization of a plural column name.

    Handles common patterns found in API responses:
      Attendees  → Attendee
      Locations  → Location
      Categories → Category
      Addresses  → Address
      Responses  → Response
      Items      → Item

    Falls back to the original name if no rule matches, so
    "Status" → "Status", "Type" → "Type" are unaffected.
    """
    if col_name.endswith("ies"):
        return col_name[:-3] + "y"   # Categories → Category
    if col_name.endswith("ses"):
        return col_name[:-2]          # Addresses → Address
    if col_name.endswith("s") and not col_name.endswith("ss"):
        return col_name[:-1]          # Attendees → Attendee, Locations → Location
    return col_name


def expand_list_columns(
    rows: List[Dict[str, Any]],
    large_list_threshold: int = 3,
) -> List[Dict[str, Any]]:
    """Expand list-valued columns into numbered sub-columns.

    For each column whose value is a Python ``list``, the list is replaced by
    separate numbered columns so every item gets its own Excel cell:

    Before::

        {"Attendees": [alice, bob, ...x10]}

    After (with large_list_threshold=3)::

        # In-place (small lists ≤ threshold sub-columns):
        {"Tags 1": "urgent", "Tags 2": "finance"}

        # Deferred to end (large lists > threshold sub-columns):
        {...other cols..., "Attendee 1": "alice@...", ..., "Attendee 10": "..."}

    Key behaviours
    --------------
    - Column name is singularized (``Attendees`` → ``Attendee``) and suffixed
      with a 1-based integer (``Attendee 1``, ``Attendee 2``, …).
    - The max width is determined globally across **all** rows so every row has
      the same set of columns (empty string for absent slots).
    - Each individual item is passed through ``flatten_cell_value`` so nested
      dicts (e.g. attendee objects with emailAddress) become readable strings.
    - **Small lists** (≤ ``large_list_threshold`` items) stay in their original
      column position — they don't clutter the view much.
    - **Large lists** (> ``large_list_threshold`` items) are moved to the **end**
      of the row so that important neighbouring columns (e.g. Organizer, Online
      Meeting, Response Status) remain visible without scrolling past a wall of
      attendee columns.

    Call this **after** ``transform_dates_in_obj`` (so datetimes are already
    IST strings) and **before** ``apply_cell_flattening`` (which handles the
    remaining non-list nested dicts).
    """
    if not rows:
        return rows

    # ── pass 1: find list columns and their maximum length ───────────────────
    list_col_max: Dict[str, int] = {}
    for row in rows:
        for col, val in row.items():
            if isinstance(val, list):
                list_col_max[col] = max(list_col_max.get(col, 0), len(val))

    if not list_col_max:
        return rows  # nothing to expand

    # Classify: small (stays in place) vs large (deferred to end)
    small_list_cols = {col for col, n in list_col_max.items() if n <= large_list_threshold}
    large_list_cols = {col for col, n in list_col_max.items() if n > large_list_threshold}

    # Pre-compute column names for each list column:
    # - max_len > 1  → ["Attendee 1", "Attendee 2", ...]
    # - max_len == 1 → ["Attendee"] (no number suffix needed)
    expanded_names: Dict[str, List[str]] = {}
    for col, max_len in list_col_max.items():
        if max_len > 1:
            expanded_names[col] = [f"{_singularize(col)} {i + 1}" for i in range(max_len)]
        elif max_len == 1:
            expanded_names[col] = [_singularize(col)]
        else:
            expanded_names[col] = []

    def _expand_items(col: str, val: Any) -> Dict[str, Any]:
        """Return {sub_col: flattened_item} for one list column value."""
        items = val if isinstance(val, list) else [val]
        return {
            sub_col: (flatten_cell_value(items[idx]) if idx < len(items) else "")
            for idx, sub_col in enumerate(expanded_names[col])
        }

    # ── pass 2: rebuild each row ─────────────────────────────────────────────
    # Order: non-list cols + in-place small-list cols → then large-list cols at end
    result: List[Dict[str, Any]] = []
    for row in rows:
        inline: Dict[str, Any] = {}    # non-list + small lists (preserve order)
        deferred: Dict[str, Any] = {}  # large lists → pushed to end

        for col, val in row.items():
            if col in small_list_cols:
                inline.update(_expand_items(col, val))
            elif col in large_list_cols:
                deferred.update(_expand_items(col, val))
            else:
                inline[col] = val

        result.append({**inline, **deferred})

    if large_list_cols:
        log.info(
            "expand_list_columns: %d large list column(s) moved to end: %s",
            len(large_list_cols),
            sorted(large_list_cols),
        )
    if small_list_cols:
        log.info(
            "expand_list_columns: %d small list column(s) kept in place: %s",
            len(small_list_cols),
            sorted(small_list_cols),
        )
    return result

