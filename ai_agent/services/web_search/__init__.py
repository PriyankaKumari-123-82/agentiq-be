"""Web Search module for AgentIQ.

This module provides web search functionality with support for multiple providers.
The WebSearchService is the single entry point used by the agent runtime.
"""
from ai_agent.services.web_search.service import WebSearchService, get_web_search_service
from ai_agent.services.web_search.schemas import (
    WebSearchResult,
    WebSearchItem,
    WebSearchContext
)
from ai_agent.services.web_search.base import WebSearchProvider

__all__ = [
    "WebSearchService",
    "get_web_search_service",
    "WebSearchResult",
    "WebSearchItem",
    "WebSearchContext",
    "WebSearchProvider",
]
