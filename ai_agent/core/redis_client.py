"""Shared Redis client singleton.

Best-effort: Redis being unavailable must never break the caller. `get_redis()` returns
`None` in that case, and callers are expected to degrade gracefully.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis

log = logging.getLogger(__name__)

_redis: "redis.Redis | None" = None
_connect_failed = False


def get_redis() -> "redis.Redis | None":
    global _redis, _connect_failed
    if _redis is not None:
        return _redis
    if _connect_failed:
        return None
    try:
        import redis

        from ai_agent.core.config import settings

        client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        client.ping()
        _redis = client
        return _redis
    except Exception as e:
        log.warning("redis_client: Redis unavailable: %s", e)
        _connect_failed = True
        return None
