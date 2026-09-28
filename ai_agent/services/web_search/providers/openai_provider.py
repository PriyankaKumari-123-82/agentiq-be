"""OpenAI web search provider."""

import logging

from openai import AsyncOpenAI

from ai_agent.services.web_search.base import WebSearchProvider, WebSearchResult, WebSearchItem

log = logging.getLogger(__name__)


class OpenAIWebSearchProvider(WebSearchProvider):
    """Web search using OpenAI's models (e.g. gpt-4o with search)."""

    async def search(self, query: str, api_key: str, max_results: int = 5, model: str | None = None) -> WebSearchResult:
        """Perform a web search using OpenAI.
        
        Args:
            query: Search query
            api_key: OpenAI API key
            max_results: Maximum number of results (not used, kept for interface compatibility)
            model: OpenAI model to use for web search (required)
            
        Raises:
            ValueError: If model is not provided
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        client = AsyncOpenAI(api_key=api_key)
        
        try:
            # Use the responses API with native web search tool
            response = await client.responses.create(
                model=model,
                tools=[
                    {"type": "web_search"}
                ],
                input=query
            )
            
            # Extract content from OpenAI Responses API
            content = getattr(response, 'output_text', '') or str(response)
            
            # OpenAI doesn't natively return structured search results (snippets, urls) in the standard chat API
            # without specific tools or plugins. We wrap the final answer as a single "result" item.
            items = [
                WebSearchItem(
                    title=f"Web Search: {query}",
                    url=None,
                    content=content,
                    score=1.0,
                )
            ]
                
            return WebSearchResult(
                query=query,
                provider="openai",
                results=items,
            )
        except Exception as e:
            log.error(f"OpenAI web search failed: {e}")
            raise RuntimeError(f"OpenAI web search failed: {e}")
