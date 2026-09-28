"""Tavily web search provider."""

import logging
from typing import Optional

from ai_agent.services.web_search.base import WebSearchProvider, WebSearchResult, WebSearchItem

log = logging.getLogger(__name__)


class TavilyWebSearchProvider(WebSearchProvider):
    """Web search using Tavily API."""

    async def search(
        self,
        query: str,
        api_key: str,
        max_results: int = 5,
        model: Optional[str] = None
    ) -> WebSearchResult:
        """Perform a web search using Tavily.
        
        Args:
            query: Search query
            api_key: Tavily API key
            max_results: Maximum number of results to return
            model: Not used for Tavily (kept for interface compatibility)
            
        Raises:
            ValueError: If API key is missing
            RuntimeError: If search fails
        """
        if not api_key:
            raise ValueError("Tavily API key is required")

        try:
            # Import tavily-python package
            from tavily import AsyncTavilyClient
        except ImportError:
            log.error("tavily-python package not installed")
            raise RuntimeError(
                "tavily-python package is required. Install with: pip install tavily-python"
            )
        
        client = AsyncTavilyClient(api_key=api_key)
        
        try:
            # Execute search
            response = await client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",  # Can be "basic" or "advanced"
            )
            
            # Extract results
            raw_results = response.get("results", [])
            log.info(f"Tavily API returned {len(raw_results)} results")
            
            items = []
            for result in raw_results:
                items.append(
                    WebSearchItem(
                        title=result.get("title", ""),
                        url=result.get("url"),  # Tavily provides actual URLs!
                        content=result.get("content", ""),
                        score=result.get("score", 0.0),
                        published_date=result.get("published_date"),
                    )
                )
            
            log.info(f"Tavily search returned {len(items)} results to agent")
            
            return WebSearchResult(
                query=query,
                provider="tavily",
                results=items,
            )
            
        except Exception as e:
            log.error(f"Tavily web search failed: {e}", exc_info=True)
            raise RuntimeError(f"Tavily web search failed: {e}")
