"""Excel export for table-shaped monitor results.

Mirrors the frontend's table classification closely enough that a monitor which renders as a
table card renders the same rows/columns here — this is the API's judgment call
about "is this exportable", so it has to agree with the UI's judgment about
"is this a table" or the Export button and this endpoint would disagree about
which monitors even support it.
"""
import io
import json
import re

from openpyxl import Workbook

from ai_agent.custom_model.monitor import MonitorResult

_INTERVAL_TEXT_RE = re.compile(
    r"(-?\d+)\s*years?\s+(-?\d+)\s*mons?\s+(-?\d+)\s*days?\s+"
    r"(-?\d+)\s*hours?\s+(-?\d+)\s*mins?\s+(-?[\d.]+)\s*secs?",
    re.IGNORECASE,
)

_ACRONYM_RE = re.compile(r"\b(id|url|api|sql|uuid|ip|po|pr|rfq)\b", re.IGNORECASE)
_CAMEL_BOUNDARY_RE = re.compile(r"([a-z\d])([A-Z])")
_WORD_START_RE = re.compile(r"\b\w")


class NotTableShapedError(Exception):
    """Raised when a monitor's latest result isn't table-shaped — nothing to export."""


def humanize_key(key: str) -> str:
    """snake_case / camelCase column name -> a header a person would write."""
    text = re.sub(r"[_-]+", " ", key)
    text = _CAMEL_BOUNDARY_RE.sub(r"\1 \2", text)
    text = _ACRONYM_RE.sub(lambda m: m.group(0).upper(), text)
    text = _WORD_START_RE.sub(lambda m: m.group(0).upper(), text)
    return text.strip()


def format_interval_value(value: object) -> str | None:
    """Turn {"type":"interval","value":"0 years 0 mons 5 days ..."} into "5 days".

    Returns None for anything that isn't this specific shape.
    """
    if not isinstance(value, dict):
        return None
    if value.get("type") != "interval" or not isinstance(value.get("value"), str):
        return None

    text = value["value"]
    match = _INTERVAL_TEXT_RE.search(text)
    if not match:
        return text  # unrecognised text shape — still better than a JSON dump

    years, months, days, hours, mins = (int(match.group(i)) for i in range(1, 6))
    secs = round(float(match.group(6)))

    def unit(n: int, label: str) -> str:
        return f"{n} {label}{'' if abs(n) == 1 else 's'}"

    parts = [
        unit(years, "year") if years else None,
        unit(months, "month") if months else None,
        unit(days, "day") if days else None,
        unit(hours, "hour") if hours else None,
        unit(mins, "minute") if mins else None,
        unit(secs, "second") if secs else None,
    ]
    parts = [p for p in parts if p is not None]
    return " ".join(parts) if parts else "0 minutes"


def _export_cell_value(value: object):
    """Same normalisation the frontend's displayCell applies, minus the parts a
    spreadsheet cell shouldn't get: numbers/booleans stay real numbers/booleans
    so SUM/sort work in Excel, rather than becoming formatted text.
    """
    if value is None or value == "":
        return ""
    if isinstance(value, bool):  # bool is an int subclass — check first
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        interval = format_interval_value(value)
        return interval if interval is not None else json.dumps(value)
    if isinstance(value, list):
        return json.dumps(value)
    return str(value)


def _parse_payload(raw: str | None) -> object:
    """The result column is double-JSON-encoded (a known backend gap) — the
    stored value decodes to a JSON string that itself needs parsing. Tolerant
    of an already-parsed object too, in case that gap gets fixed later.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw
    return raw


def _is_scalar_or_interval(value: object) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, dict):
        return format_interval_value(value) is not None
    return False


def _classify_table(parsed: object) -> tuple[list[str], list[dict]] | None:
    """Extract (columns, rows) if this result renders as a table card on the
    frontend, matching describeMonitorResult's table + flat-record branches.
    Returns None for anything that renders as a chart, metric, or text there.
    """
    value = parsed.get("value") if isinstance(parsed, dict) and "value" in parsed else parsed

    # Chart payload — {"chartType": "pie", "data": [...]} — not a table.
    if isinstance(value, dict) and value.get("chartType") and isinstance(value.get("data"), list):
        return None

    if isinstance(value, list):
        if not value:
            return None
        if all(isinstance(row, dict) for row in value):
            rows = value
        else:
            # A list of scalars still deserves a table rather than a JSON dump.
            rows = [{"#": i + 1, "value": v} for i, v in enumerate(value)]
        columns: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
        return columns, rows

    if isinstance(value, dict) and len(value) > 1:
        if all(_is_scalar_or_interval(v) for v in value.values()):
            return list(value.keys()), [value]

    return None


def _safe_sheet_name(name: str) -> str:
    """Excel sheet names: max 31 chars, and : \\ / ? * [ ] are rejected outright."""
    cleaned = re.sub(r'[:\\/?*\[\]]', " ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return (cleaned or "Data")[:31]


def safe_filename(name: str) -> str:
    """Strip characters that break a filename or could smuggle a header injection
    into Content-Disposition (callers put this straight into that header)."""
    cleaned = re.sub(r'[\r\n"]', " ", name)
    cleaned = re.sub(r'[/\\:*?<>|]', " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "monitor-export"


def build_excel_export(monitor_name: str, result: MonitorResult) -> bytes:
    """Build an .xlsx of a monitor's latest result — every row, not a page of it.

    Raises:
        NotTableShapedError if the result isn't table-shaped (chart, metric,
        text, boolean, or empty) — there is nothing row-shaped to export.
    """
    parsed = _parse_payload(result.result)
    if parsed is None:
        raise NotTableShapedError("This monitor's latest result has no data to export.")

    classified = _classify_table(parsed)
    if classified is None:
        raise NotTableShapedError(
            "This monitor's latest result isn't a table — only table-shaped "
            "monitor results can be exported to Excel."
        )
    columns, rows = classified

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = _safe_sheet_name(monitor_name)

    sheet.append([humanize_key(c) for c in columns])
    for row in rows:
        sheet.append([_export_cell_value(row.get(c)) for c in columns])

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()
