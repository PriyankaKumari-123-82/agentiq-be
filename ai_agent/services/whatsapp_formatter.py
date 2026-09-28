"""Utility for formatting table data into generic WhatsApp single-column itemized cards without emojis.

Uses lightweight LLM column classification (sending only column headers to save tokens)
and a deterministic Python renderer to build WhatsApp-formatted bulleted messages.
"""

from typing import Any, Dict, List, Optional, Tuple
import re
import json
import logging

from ai_agent.helpers.date_formatter import convert_utc_to_ist

log = logging.getLogger(__name__)


def _extract_url_from_markdown(text: str) -> Tuple[str, Optional[str]]:
    """Extract label and URL from markdown link like '[Event Link](https://...)'."""
    match = re.match(r"^\[(.*?)\]\((.*?)\)$", text.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return text.strip(), None


def _clean_header_name(header: str) -> str:
    """Normalize header name for classification."""
    return re.sub(r"[^\w\s]", "", str(header)).strip().lower()


async def classify_headers_with_llm(
    headers: List[str],
    agent_settings: Optional[Any] = None,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Use a lightweight LLM call to classify column headers into semantic roles.

    Sends ONLY column header names (saving ~95%+ tokens) and returns a JSON mapping.
    """
    if not headers or not api_key:
        return None

    try:
        from ai_agent.core.llm_gateway import get_llm_gateway

        system_prompt = (
            "Classify tabular column headers into semantic roles.\n"
            "Return JSON object with keys:\n"
            "- \"id\": single header string for item ID/key/code (or null)\n"
            "- \"title\": single header string for item title/subject/summary/name (or null)\n"
            "- \"start_time\": header for start date/time (or null)\n"
            "- \"end_time\": header for end date/time (or null)\n"
            "- \"dates\": array of date/time headers\n"
            "- \"people\": array of attendee/user/author/assignee headers\n"
            "- \"links\": array of URL/link headers\n"
            "- \"status\": array of status/state/priority headers\n"
            "- \"location\": array of location/venue headers\n\n"
            "Use verbatim header names from the input list."
        )

        user_prompt = f"Column headers: {json.dumps(headers)}"

        gateway = get_llm_gateway()
        response = await gateway.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            provider=agent_settings.llm_provider if agent_settings else None,
            model=agent_settings.llm_model if agent_settings else None,
            temperature=0.0,
            response_format={"type": "json_object"},
            api_key=api_key,
        )

        content = response.content.strip()
        if content:
            return json.loads(content)

    except Exception as e:
        log.warning(f"LLM header classification failed, using regex fallback: {e}")

    return None


def format_table_to_whatsapp_cards(
    title: Optional[str],
    headers: List[str],
    rows: List[List[Any]],
    max_rows: int = 20,
    column_mapping: Optional[Dict[str, Any]] = None,
) -> str:
    """Convert tabular headers and rows into generic single-column bulleted row cards for WhatsApp.

    No emojis. Uses bullets (`- `) and WhatsApp bold formatting (`*text*`).
    Supports optional LLM-provided column_mapping or regex fallback classification.
    """
    if not headers or not rows:
        return ""

    num_total_rows = len(rows)
    display_rows = rows[:max_rows]

    # Normalize header info
    header_info = []
    header_by_orig = {}
    for idx, h in enumerate(headers):
        orig_str = str(h).strip()
        clean_h = _clean_header_name(h)
        if clean_h.replace(" ", "").replace("_", "") == "showas":
            continue  # Exclude 'Show As' column
        item = {
            "index": idx,
            "original": orig_str,
            "clean": clean_h,
        }
        header_info.append(item)
        header_by_orig[orig_str.lower()] = item

    # Classification using column_mapping if present
    id_col = None
    title_col = None
    start_time_col = None
    end_time_col = None
    general_date_cols = []
    person_cols = []
    link_cols = []
    status_cols = []
    location_cols = []

    if column_mapping and isinstance(column_mapping, dict):
        def find_col(name_val: Any) -> Optional[Dict[str, Any]]:
            if isinstance(name_val, str) and name_val.strip().lower() in header_by_orig:
                return header_by_orig[name_val.strip().lower()]
            return None

        def find_cols(name_list: Any) -> List[Dict[str, Any]]:
            res = []
            if isinstance(name_list, list):
                for name in name_list:
                    c = find_col(name)
                    if c and c not in res:
                        res.append(c)
            return res

        id_col = find_col(column_mapping.get("id"))
        title_col = find_col(column_mapping.get("title"))
        start_time_col = find_col(column_mapping.get("start_time"))
        end_time_col = find_col(column_mapping.get("end_time"))
        general_date_cols = find_cols(column_mapping.get("dates"))
        person_cols = find_cols(column_mapping.get("people"))
        link_cols = find_cols(column_mapping.get("links"))
        status_cols = find_cols(column_mapping.get("status"))
        location_cols = find_cols(column_mapping.get("location"))

    # Regex fallback if classification is missing
    def match_word(clean_h: str, patterns: Tuple[str, ...]) -> bool:
        for p in patterns:
            if re.search(r"\b" + re.escape(p) + r"\b", clean_h):
                return True
        return False

    if not id_col:
        id_patterns = ("id", "key", "number", "code", "ticket", "pr", "job id", "issue key")
        id_col = next((h for h in header_info if match_word(h["clean"], id_patterns)), None)

    if not title_col:
        title_patterns = ("subject", "summary", "title", "name", "topic", "task", "item")
        title_col = next(
            (h for h in header_info if h != id_col and match_word(h["clean"], title_patterns)),
            None
        )

    if not start_time_col:
        start_time_col = next(
            (h for h in header_info if match_word(h["clean"], ("start date", "start time", "start", "from"))),
            None
        )

    if not end_time_col:
        end_time_col = next(
            (h for h in header_info if match_word(h["clean"], ("end date", "end time", "end", "to"))),
            None
        )

    if not general_date_cols:
        general_date_cols = [
            h for h in header_info
            if match_word(h["clean"], ("date", "time", "created", "updated", "due"))
            and h not in (start_time_col, end_time_col)
        ]

    if not person_cols:
        person_cols = [
            h for h in header_info
            if match_word(h["clean"], ("attendee", "assignee", "author", "user", "person", "member", "owner", "reporter"))
        ]

    if not link_cols:
        link_cols = [
            h for h in header_info
            if match_word(h["clean"], ("link", "url", "join", "href"))
        ]

    if not status_cols:
        status_cols = [
            h for h in header_info
            if match_word(h["clean"], ("status", "state", "priority", "severity", "stage"))
        ]

    if not location_cols:
        location_cols = [
            h for h in header_info
            if match_word(h["clean"], ("location", "venue", "room", "address"))
        ]

    # Track categorized indices to avoid duplicate output in fallback generic section
    categorized_indices = set()
    for category in ([id_col, title_col, start_time_col, end_time_col] + general_date_cols + person_cols + link_cols + status_cols + location_cols):
        if category:
            categorized_indices.add(category["index"])

    cards = []
    
    if title:
        clean_title = title.strip()
        cards.append(f"*{clean_title}*")

    # Collect all date/time column indices for conversion
    _date_col_indices: set = set()
    for _c in ([start_time_col, end_time_col] + general_date_cols):
        if _c:
            _date_col_indices.add(_c["index"])

    for row_idx, row in enumerate(display_rows, start=1):
        lines = []

        def get_val(col_def: Optional[Dict[str, Any]], _date_idxs: set = _date_col_indices) -> str:
            if not col_def:
                return ""
            idx = col_def["index"]
            if idx < len(row) and row[idx] is not None:
                raw = str(row[idx]).strip()
                if idx in _date_idxs:
                    return convert_utc_to_ist(raw)
                return raw
            return ""

        id_val = get_val(id_col)
        title_val = get_val(title_col)

        # Main bullet item header (Bold subject/title or ID)
        if title_val and id_val and title_val.lower() != id_val.lower():
            lines.append(f"- *[{id_val}] {title_val}*")
        elif title_val:
            lines.append(f"- *{title_val}*")
        elif id_val:
            lines.append(f"- *{id_val}*")
        else:
            lines.append(f"- *Item {row_idx}*")

        # Date & Time Range Sub-bullet
        start_val = get_val(start_time_col)
        end_val = get_val(end_time_col)
        if start_val and end_val:
            common_date = ""
            if "," in start_val and "," in end_val:
                s_parts = start_val.split(",", 1)
                e_parts = end_val.split(",", 1)
                if s_parts[0].strip() == e_parts[0].strip():
                    common_date = s_parts[0].strip() + ", "
                    time_range = f"{s_parts[1].strip()} – {e_parts[1].strip()}"
                    lines.append(f"  - Time: {common_date}{time_range}")
            if len(lines) == 1 or not lines[-1].startswith("  - Time:"):
                lines.append(f"  - Time: {start_val} – {end_val}")
        elif start_val:
            lines.append(f"  - Time: {start_val}")
        elif end_val:
            lines.append(f"  - Time: {end_val}")
        elif general_date_cols:
            date_vals = [get_val(c) for c in general_date_cols if get_val(c)]
            if date_vals:
                lines.append(f"  - Date: {', '.join(date_vals)}")

        # People / Attendees Sub-bullet
        people_vals = []
        for p_col in person_cols:
            val = get_val(p_col)
            if val and val not in people_vals and val.lower() not in ("none", "n/a", "null"):
                if "@" in val and any("@" not in existing for existing in people_vals):
                    continue
                people_vals.append(val)
        if people_vals:
            lines.append(f"  - Attendees: {', '.join(people_vals)}")

        # Status / Priority Sub-bullets
        for st_col in status_cols:
            val = get_val(st_col)
            if val:
                lines.append(f"  - {st_col['original']}: {val}")

        # Location Sub-bullet
        for loc_col in location_cols:
            val = get_val(loc_col)
            if val:
                lines.append(f"  - Location: {val}")

        # Links / URLs Sub-bullet
        for l_col in link_cols:
            val = get_val(l_col)
            if val:
                label, url = _extract_url_from_markdown(val)
                if url:
                    lines.append(f"  - Link: [{label}]({url})")
                elif val.startswith("http://") or val.startswith("https://"):
                    lines.append(f"  - Link: {val}")
                elif val.lower() in ("event link", "join meeting", "meeting link", "link"):
                    lines.append(f"  - Link: Join Meeting")
                else:
                    lines.append(f"  - Link: {val}")

        # Generic / Uncategorized Sub-bullets
        for col in header_info:
            if col["index"] not in categorized_indices:
                val = get_val(col)
                if val and val.lower() not in ("yes", "no", "true", "false", "none", "null", "n/a", "teamsforbusiness"):
                    lines.append(f"  - {col['original']}: {val}")

        cards.append("\n".join(lines))

    if num_total_rows > max_rows:
        cards.append(f"_...showing preview of first {max_rows} rows of {num_total_rows} total rows._")

    return "\n\n".join(cards)


async def format_table_with_llm(
    title: Optional[str],
    headers: List[str],
    rows: List[List[Any]],
    agent_settings: Optional[Any] = None,
    api_key: Optional[str] = None,
    max_rows: int = 15,
) -> str:
    """Format tabular data into a WhatsApp bulleted message.

    Uses a lightweight LLM call to classify headers (~30-50 tokens only)
    and then renders the full message deterministically in Python.
    """
    if not headers or not rows:
        return ""

    column_mapping = None
    if api_key:
        column_mapping = await classify_headers_with_llm(
            headers=headers,
            agent_settings=agent_settings,
            api_key=api_key,
        )

    return format_table_to_whatsapp_cards(
        title=title,
        headers=headers,
        rows=rows,
        max_rows=max_rows,
        column_mapping=column_mapping,
    )


def parse_markdown_tables(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Parse markdown pipe tables out of raw text.

    Returns:
        Tuple of (clean_text_without_pipe_tables, list_of_extracted_tables)
    """
    if "|" not in text:
        return text.strip(), []

    lines = text.splitlines()
    cleaned_lines = []
    tables = []

    current_table_lines = []

    def process_table_lines(tbl_lines: List[str]):
        if len(tbl_lines) < 2:
            return
        
        sep_line = tbl_lines[1].strip()
        if not re.match(r"^\|?\s*:?-+:?\s*(\|?\s*:?-+:?\s*)*\|?$", sep_line):
            return

        def parse_row(line_str: str) -> List[str]:
            cells = line_str.strip().strip("|").split("|")
            return [c.strip() for c in cells]

        headers = parse_row(tbl_lines[0])
        rows = [parse_row(l) for l in tbl_lines[2:] if l.strip().startswith("|")]

        if headers and rows:
            tables.append({
                "headers": headers,
                "rows": rows,
            })

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            current_table_lines.append(line)
        else:
            if current_table_lines:
                process_table_lines(current_table_lines)
                current_table_lines = []
            
            lower = stripped.lower()
            if "preview of the first" in lower or "download the file below" in lower:
                continue
            if "[download the full dataset](sandbox:" in lower:
                continue
            cleaned_lines.append(line)

    if current_table_lines:
        process_table_lines(current_table_lines)

    clean_text = "\n".join(cleaned_lines).strip()
    return clean_text, tables
