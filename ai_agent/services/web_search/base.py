"""Abstract base class for web search providers."""
from abc import ABC, abstractmethod

from ai_agent.services.web_search.schemas import WebSearchResult, WebSearchItem


class WebSearchProvider(ABC):
    """Abstract base class for all web search providers.
    
    Each provider implementation must implement the search() method
    and return results in the standardized WebSearchResult format.
    """
    
    @abstractmethod
    async def search(
        self,
        query: str,
        api_key: str,
        max_results: int = 5,
        model: str | None = None
    ) -> WebSearchResult:
        """Execute a web search and return formatted results.
        
        Args:
            query: The search query string
            api_key: API key for the provider
            max_results: Maximum number of results to return
            model: Optional model name for providers that support multiple models
            
        Returns:
            WebSearchResult: Standardized search results
            
        Raises:
            Exception: If the search fails
        """
        pass
