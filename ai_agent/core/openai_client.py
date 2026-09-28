import logging
from openai import OpenAI
from typing import List, Dict, Any, Optional
from ai_agent.core.config import settings

log = logging.getLogger(__file__)

def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Create OpenAI client.
    
    Args:
        api_key: Optional API key. If not provided, uses settings.OPENAI_API_KEY
        
    Returns:
        OpenAI client instance
    """
    openai_api_key = api_key or settings.OPENAI_API_KEY
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not available")
    client = OpenAI(api_key=openai_api_key)
    log.info(f"OpenAI client created (using {'provided' if api_key else 'env'} key)")
    return client


def create_chat_completion(
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    web_search_enabled: bool = False,
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    Create a chat completion using the LLM.
    
    Args:
        messages: List of message dictionaries for the conversation
        model: Model to use (defaults to settings.LLM_MODEL)
        temperature: Temperature for response generation (0.0-1.0)
        max_tokens: Maximum tokens to generate
        api_key: Optional API key for per-agent keys
        **kwargs: Additional parameters to pass to the API
        
    Returns:
        str: The generated response content
    """
    client = get_openai_client(api_key)
    tools = []
    if web_search_enabled:
        tools.append({"type": "web_search"})
    response = client.responses.create(
        model=model or settings.LLM_MODEL,
        input=messages,
        tools=tools if tools else None,
        temperature=temperature,
        max_output_tokens=max_tokens,
        **kwargs
    )
    log.info(f"[create_chat_completion] Response: {response}")
    return response.output_text
