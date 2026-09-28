"""Redis distributed cache + persistence validation for schema engine.

Redis is used ONLY as:
- Distributed hot cache (not authoritative)
- Distributed locks for multi-instance safety
- Short-term retrieval acceleration

The system MUST function correctly without Redis (degraded but operational).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional

from ai_agent.core.config import settings
from ai_agent.schema_engine.config import get_config

log = logging.getLogger(__name__)

_KEY_PREFIX = "se:"


class RedisCache:
    """Distributed Redis cache with persistence validation."""

    def __init__(self):
        self._cfg = get_config()
        self._redis = None
        self._persistence_validated = False
        self._connected = False

    # ═════════════════════════════════════════════════════════════════════
    # Connection + Persistence Validation
    # ═════════════════════════════════════════════════════════════════════

    def connect(self) -> bool:
        """Connect to Redis and validate persistence settings."""
        if self._connected:
            return True
        try:
            import redis
            url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
            self._redis = redis.from_url(url, decode_responses=True)
            self._redis.ping()
            self._connected = True
            log.info("RedisCache: connected to %s", url)

            # Validate persistence
            self._validate_persistence()
            return True
        except Exception as e:
            log.warning("RedisCache: connection failed, running without cache: %s", e)
            self._redis = None
            self._connected = False
            return False

    def _validate_persistence(self) -> None:
        """Validate Redis persistence configuration.

        Checks:
        - appendonly (AOF) is enabled
        - RDB save points are configured
        """
        if not self._redis:
            return

        try:
            info = self._redis.info("persistence")
            config = self._redis.config_get("appendonly")

            aof_enabled = config.get("appendonly", "no") == "yes"
            rdb_last_save = info.get("rdb_last_save_time", 0)
            rdb_changes = info.get("rdb_changes_since_last_save", -1)

            # Check AOF
            if not aof_enabled:
                msg = (
                    "Redis AOF (appendonly) is DISABLED. Schema engine cache data "
                    "may be lost on Redis restart. Set 'appendonly yes' in redis.conf."
                )
                if getattr(self._cfg, "redis_persistence_required", False):
                    log.critical(msg)
                    raise RuntimeError(msg)
                else:
                    log.warning(msg)

            # Check RDB
            rdb_config = self._redis.config_get("save")
            save_value = rdb_config.get("save", "")
            if not save_value or save_value == '""':
                log.warning(
                    "Redis RDB snapshots are DISABLED. Consider enabling for "
                    "data safety."
                )
            else:
                log.info("RedisCache: RDB save policy: %s", save_value)

            # Check appendfsync
            fsync_config = self._redis.config_get("appendfsync")
            fsync = fsync_config.get("appendfsync", "unknown")
            log.info(
                "RedisCache: persistence validated — AOF=%s, appendfsync=%s, "
                "rdb_last_save=%s",
                aof_enabled, fsync, rdb_last_save,
            )
            self._persistence_validated = True

        except RuntimeError:
            raise
        except Exception as e:
            log.warning("RedisCache: persistence validation failed: %s", e)

    # ═════════════════════════════════════════════════════════════════════
    # Cache Operations
    # ═════════════════════════════════════════════════════════════════════

    def get(self, key: str) -> Optional[str]:
        """Get a cached value. Returns None on miss or error."""
        if not self._connected:
            return None
        try:
            return self._redis.get(f"{_KEY_PREFIX}{key}")
        except Exception:
            return None

    def get_json(self, key: str) -> Optional[Any]:
        """Get and JSON-decode a cached value."""
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    def set(self, key: str, value: str, ttl_seconds: Optional[int] = None) -> bool:
        """Set a cache value with optional TTL."""
        if not self._connected:
            return False
        try:
            ttl = ttl_seconds or getattr(self._cfg, "redis_cache_ttl_seconds", 3600)
            self._redis.setex(f"{_KEY_PREFIX}{key}", timedelta(seconds=ttl), value)
            return True
        except Exception:
            return False

    def set_json(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """JSON-encode and set a cache value."""
        try:
            return self.set(key, json.dumps(value), ttl_seconds)
        except (TypeError, ValueError):
            return False

    def delete(self, key: str) -> bool:
        if not self._connected:
            return False
        try:
            self._redis.delete(f"{_KEY_PREFIX}{key}")
            return True
        except Exception:
            return False

    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern."""
        if not self._connected:
            return 0
        try:
            keys = list(self._redis.scan_iter(f"{_KEY_PREFIX}{pattern}"))
            if keys:
                return self._redis.delete(*keys)
            return 0
        except Exception:
            return 0

    # ═════════════════════════════════════════════════════════════════════
    # Distributed Locking
    # ═════════════════════════════════════════════════════════════════════

    def acquire_lock(
        self, name: str, timeout_seconds: int = 60, blocking: bool = True,
    ) -> Optional["_RedisLock"]:
        """Acquire a distributed lock for multi-instance safety.

        Usage:
            lock = cache.acquire_lock("schema_engine:index:my_server")
            if lock:
                try:
                    # do work
                finally:
                    lock.release()
        """
        if not self._connected:
            # Return a no-op lock when Redis unavailable (single-instance mode)
            return _NoOpLock()
        try:
            lock = self._redis.lock(
                f"{_KEY_PREFIX}lock:{name}",
                timeout=timeout_seconds,
                blocking=blocking,
                blocking_timeout=10,
            )
            acquired = lock.acquire()
            if acquired:
                return _RedisLock(lock)
            return None
        except Exception as e:
            log.warning("RedisCache: lock acquisition failed: %s", e)
            return _NoOpLock()

    # ═════════════════════════════════════════════════════════════════════
    # Schema Engine specific cache keys
    # ═════════════════════════════════════════════════════════════════════

    def cache_retrieval_result(
        self, server_name: str, query_hash: str, result_json: str,
        ttl_seconds: int = 300,
    ) -> bool:
        """Cache a retrieval result for fast repeated queries."""
        key = f"retrieval:{server_name}:{query_hash}"
        return self.set(key, result_json, ttl_seconds)

    def get_cached_retrieval(self, server_name: str, query_hash: str) -> Optional[str]:
        key = f"retrieval:{server_name}:{query_hash}"
        return self.get(key)

    def invalidate_server_cache(self, server_name: str) -> int:
        """Invalidate all cache entries for a server."""
        return self.delete_pattern(f"*:{server_name}:*")

    # ═════════════════════════════════════════════════════════════════════
    # Health / Diagnostics
    # ═════════════════════════════════════════════════════════════════════

    def health_check(self) -> Dict[str, Any]:
        """Return health/diagnostic info."""
        if not self._connected:
            return {"status": "disconnected", "persistence_validated": False}
        try:
            info = self._redis.info("memory")
            return {
                "status": "connected",
                "persistence_validated": self._persistence_validated,
                "used_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
                "connected_clients": self._redis.info("clients").get("connected_clients", 0),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @property
    def is_connected(self) -> bool:
        return self._connected


class _RedisLock:
    """Wrapper around redis lock for context management."""
    def __init__(self, lock):
        self._lock = lock

    def release(self) -> None:
        try:
            self._lock.release()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


class _NoOpLock:
    """No-op lock when Redis is unavailable."""
    def release(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ── Singleton ─────────────────────────────────────────────────────────────

_redis_cache: Optional[RedisCache] = None


def get_redis_cache() -> RedisCache:
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCache()
        _redis_cache.connect()
    return _redis_cache
