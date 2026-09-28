"""Tracks per-organization user activity in Redis, for the Milvus collection unload job.

Each authenticated request records the caller in a per-org sorted set
(`org:active:{organization_id}`, member=user_id, score=timestamp). The unload job reads the
highest score to determine whether anyone in the org has been active recently.
"""

from __future__ import annotations

import logging
import time

from ai_agent.core.redis_client import get_redis

log = logging.getLogger(__name__)

_ACTIVE_KEY_PREFIX = "org:active:"
_DEFAULT_INACTIVITY_SECONDS = 30 * 60


def record_activity(organization_id: int, user_id: int) -> None:
    """Record that `user_id` was active in `organization_id`, right now."""
    client = get_redis()
    if client is None:
        return
    try:
        client.zadd(f"{_ACTIVE_KEY_PREFIX}{organization_id}", {str(user_id): time.time()})
    except Exception as e:
        log.warning("activity_tracking: failed to record activity: %s", e)


def is_organization_inactive(
    organization_id: int,
    *,
    now: float | None = None,
    inactivity_seconds: int = _DEFAULT_INACTIVITY_SECONDS,
) -> bool:
    """Return whether an org has had no Redis-tracked activity in the window."""
    client = get_redis()
    if client is None:
        return False

    try:
        latest = client.zrevrange(
            f"{_ACTIVE_KEY_PREFIX}{organization_id}",
            0,
            0,
            withscores=True,
        )
    except Exception as e:
        log.warning("activity_tracking: failed to read activity: %s", e)
        return False

    if not latest:
        return True

    try:
        latest_timestamp = float(latest[0][1])
    except (IndexError, TypeError, ValueError) as e:
        log.warning("activity_tracking: invalid activity data: %s", e)
        return False

    reference_time = time.time() if now is None else now
    return latest_timestamp < reference_time - inactivity_seconds
