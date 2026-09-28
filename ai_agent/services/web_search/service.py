"""Main Web Search Service - single entry point for all web search operations."""
import logging
from typing import Optional
from sqlmodel import Session

from ai_agent.models import AIAgentSettings
from ai_agent.services.api_keys import get_api_key
from ai_agent.services.web_search.base import WebSearchProvider
from ai_agent.services.web_search.providers import OpenAIWebSearchProvider, TavilyWebSearchProvider
from ai_agent.services.web_search.schemas import WebSearchResult

log = logging.getLogger(__name__)


class WebSearchService:
    """Main service for web search operations.
    
    This is the single entry point used by the agent runtime.
    It handles provider resolution, API key fetching, and search execution.
    """
    
    # Provider registry mapping provider names to their classes
    PROVIDERS = {
        "openai": OpenAIWebSearchProvider,
        "tavily": TavilyWebSearchProvider,
    }
    
    def __init__(self, db_session: Session):
        """Initialize the web search service.
        
        Args:
            db_session: Database session for fetching API keys
        """
        self.db = db_session
    
    async def search(
        self,
        query: str,
        agent_settings: AIAgentSettings,
        max_results: Optional[int] = 5,
        **kwargs
    ) -> WebSearchResult:
        """Execute a web search using the configured provider.
        
        This is the main entry point called by the agent runtime.
        
        Args:
            query: The search query string
            agent_settings: Agent settings containing provider and API key configuration
            max_results: Maximum number of results to return (default: 5)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            WebSearchResult: Standardized search results
            
        Raises:
            ValueError: If provider is not configured or not supported
            Exception: If API key is missing or invalid
        """
        # Get the provider name
        provider_name = agent_settings.websearch_provider
        
        # Get the model name 
        model_name = agent_settings.websearch_model 
        
        # Get the API key
        api_key = self._get_provider_api_key(agent_settings, provider_name)
        if not api_key:
            log.error(f"No API key found for provider: {provider_name}")
            raise ValueError(f"Missing API key for {provider_name}")
        
        # Get the provider
        provider = self._get_provider(provider_name)
        
        # Execute the search
        result = await provider.search(query=query, api_key=api_key, max_results=max_results, model=model_name)
        log.info(f"Web search completed: {len(result.results)} results found")
        return result
    
    def _get_provider(self, provider_name: str) -> WebSearchProvider:
        """Get the provider instance for the given provider name.
        
        Args:
            provider_name: The provider name (e.g., 'openai', 'tavily')
            
        Returns:
            WebSearchProvider instance
        """
        provider = self.PROVIDERS[provider_name]()
        log.info(f"Initialized {provider_name} web search provider")
        return provider
    
    def _get_provider_api_key(
        self,
        agent_settings: AIAgentSettings,
        provider_name: str
    ) -> Optional[str]:
        """Fetch and decrypt the API key for the given provider.
        
        Args:
            agent_settings: Agent settings with API key references
            provider_name: The provider name
            
        Returns:
            Decrypted API key or None if not found
        """
        # Get the dedicated web search API key
        if agent_settings.websearch_api_key_id:
            api_key = get_api_key(self.db, agent_settings.websearch_api_key_id)
            if api_key:
                log.info(f"Using web search API key for {provider_name}")
                return api_key
        
        return None


def get_web_search_service(db_session: Session) -> WebSearchService:
    """Factory function to get a WebSearchService instance.
    
    Args:
        db_session: Database session
        
    Returns:
        WebSearchService instance
    """
    return WebSearchService(db_session)
