"""Lightweight async URL-shortening helper using the TinyURL public API.

Usage:
    from ai_agent.core.url_shortener import shorten_url, shorten_urls_in_obj

    short = await shorten_url("https://very-long-url...")
    # Returns the shortened URL, or the original URL on failure.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Optional

import httpx

log = logging.getLogger(__name__)

_TINYURL_API = "https://tinyurl.com/api-create.php"
_TIMEOUT = 5.0  # seconds

_CACHE: dict[str, str] = {}
URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+', re.IGNORECASE)


async def shorten_url(long_url: str, *, timeout: float = _TIMEOUT) -> str:
    """Return a shortened TinyURL for *long_url*.

    Falls back to returning *long_url* unchanged if the request fails for any
    reason (network error, rate-limit, etc.) so callers never have to worry
    about error handling. Results are cached in memory.
    """
    if not long_url:
        return long_url
    if long_url in _CACHE:
        return _CACHE[long_url]

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(_TINYURL_API, params={"url": long_url})
            resp.raise_for_status()
            short = resp.text.strip()
            if short.startswith("http"):
                log.debug("Shortened URL: %s -> %s", long_url[:80], short)
                _CACHE[long_url] = short
                return short
            log.warning("TinyURL returned unexpected body: %s", short[:200])
    except Exception as exc:
        log.warning("URL shortening failed (%s). Using original URL.", exc)
    return long_url


async def shorten_urls_in_obj(obj: Any, *, timeout: float = _TIMEOUT) -> Any:
    """Recursively find and shorten all URLs in any JSON-compatible structure.

    Concurrent and cached.
    """
    urls_found = set()

    def extract_urls(item: Any) -> None:
        if isinstance(item, str):
            for match in URL_REGEX.finditer(item):
                urls_found.add(match.group(0))
        elif isinstance(item, dict):
            for k, v in item.items():
                extract_urls(k)
                extract_urls(v)
        elif isinstance(item, (list, tuple, set)):
            for elem in item:
                extract_urls(elem)

    extract_urls(obj)

    if not urls_found:
        return obj

    url_mapping: dict[str, str] = {}

    async def shorten_and_map(url: str) -> None:
        shortened = await shorten_url(url, timeout=timeout)
        url_mapping[url] = shortened

    await asyncio.gather(*(shorten_and_map(u) for u in urls_found))

    def replace_urls(item: Any) -> Any:
        if isinstance(item, str):
            def repl(match: re.Match) -> str:
                u = match.group(0)
                return url_mapping.get(u, u)
            return URL_REGEX.sub(repl, item)
        elif isinstance(item, dict):
            return {replace_urls(k): replace_urls(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [replace_urls(elem) for elem in item]
        elif isinstance(item, tuple):
            return tuple(replace_urls(elem) for elem in item)
        elif isinstance(item, set):
            return {replace_urls(elem) for elem in item}
        return item

    return replace_urls(obj)
