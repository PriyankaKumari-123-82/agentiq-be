import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

# ── IST constant ──────────────────────────────────────────────────────────────
_IST_OFFSET = timedelta(hours=5, minutes=30)
_IST_TZ = timezone(_IST_OFFSET, name="IST")


def _is_unwanted_key(key: Any) -> bool:
    """Check if a dictionary key represents an unwanted field (e.g. 'Show As')."""
    if not isinstance(key, str):
        return False
    k_clean = key.strip().lower().replace("_", "").replace(" ", "")
    return k_clean in ("showas",)


def convert_utc_to_ist(value: str) -> str:
    """Convert a UTC datetime string to IST format.

    Handles patterns like:
      - "2026-09-14 09:30 UTC"
      - "2026-09-14T09:30:00Z"
      - "2026-09-14T09:30:00.000Z"
      - "2026-09-14T09:30:00+00:00"
      - "Weekly Mon-Fri, 06:30 UTC"   (recurring – prefix preserved)

    Returns the original string unchanged if it does not match any known
    UTC datetime pattern (safe to call on any arbitrary string).
    """
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return s

    # ── Pattern 1: full datetime ─────────────────────────────────────────────
    # Matches: "YYYY-MM-DD HH:MM"          (no tz suffix – assume UTC)
    #          "YYYY-MM-DDTHH:MM:SS.mmmZ"  (ISO 8601 with fractional seconds)
    #          "YYYY-MM-DD HH:MM UTC"
    m = re.match(
        r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})(?::\d{2})?(?:\.\d+)?\s*(?:UTC|Z|\+00:00)?$",
        s,
        re.IGNORECASE,
    )
    if m:
        try:
            dt_utc = datetime.strptime(
                f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M"
            ).replace(tzinfo=timezone.utc)
            dt_ist = dt_utc + _IST_OFFSET
            return dt_ist.strftime("%d %b %Y, %I:%M %p IST")
        except ValueError:
            pass

    # ── Pattern 2: recurring "Weekly Mon-Fri, 06:30 UTC" ────────────────────
    m2 = re.match(
        r"^(.+?),\s*(\d{2}:\d{2})\s*(?:UTC|Z|\+00:00)?$",
        s,
        re.IGNORECASE,
    )
    if m2:
        prefix = m2.group(1).strip()
        time_str = m2.group(2).strip()
        try:
            dt_utc = datetime.strptime(time_str, "%H:%M").replace(
                tzinfo=timezone.utc
            )
            dt_ist = dt_utc + _IST_OFFSET
            return f"{prefix}, {dt_ist.strftime('%I:%M %p')} IST"
        except ValueError:
            pass

    return s


def transform_dates_in_obj(obj: Any) -> Any:
    """Recursively convert all UTC datetime strings to IST and filter out unwanted
    internal/redundant fields (such as 'Show As') in a Python object.

    Works on any combination of dict, list, and str (including JSON strings).
    Non-string leaf values (int, float, bool, None, …) are returned as-is.

    Usage:
        result_data = transform_dates_in_obj(tool_result)
    """
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith(("{", "[")):
            try:
                parsed = json.loads(s)
                return transform_dates_in_obj(parsed)
            except Exception:
                pass
        return convert_utc_to_ist(obj)
    if isinstance(obj, dict):
        return {
            k: transform_dates_in_obj(v)
            for k, v in obj.items()
            if not _is_unwanted_key(k)
        }
    if isinstance(obj, list):
        return [transform_dates_in_obj(item) for item in obj]
    return obj

