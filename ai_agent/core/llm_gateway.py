"""LLM Gateway using LiteLLM for multi-provider support.

This module provides a unified interface for multiple LLM providers,
with support for streaming, token tracking, fallbacks, and cost optimization.
"""
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from datetime import datetime

import litellm
from litellm import completion, acompletion, embedding, aembedding, token_counter
from litellm.utils import CustomStreamWrapper

from ai_agent.core.config import settings

log = logging.getLogger(__name__)


def is_llm_quota_error(exc: Exception) -> bool:
    """True if `exc` is an LLM provider rate-limit/quota error (HTTP 429)."""
    if isinstance(exc, litellm.exceptions.RateLimitError):
        return True
    return getattr(exc, "status_code", None) == 429


_QUOTA_EXHAUSTED_KEYWORDS = (
    "insufficient_quota",
    "exceeded your current quota",
    "no credits remaining",
    "add credits",
    "billing",
)


def llm_quota_error_message(exc: Exception) -> str:
    """User-facing message for a provider quota/rate-limit error."""
    text = str(exc).lower()
    if any(keyword in text for keyword in _QUOTA_EXHAUSTED_KEYWORDS):
        return "Your API quota has been exhausted. Please check your provider plan and billing details."
    return "The LLM provider is rate-limiting requests right now. Please try again in a moment."


@dataclass
class LLMResponse:
    """Structured LLM response with usage metadata."""
    content: str
    raw_response: Any
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None


@dataclass
class StreamingChunk:
    """A chunk from a streaming LLM response."""
    content: str
    is_finished: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None


class LLMGateway:
    """Unified gateway for multiple LLM providers via LiteLLM."""

    def __init__(self):
        """Initialize the LLM Gateway."""
        # Configure LiteLLM
        litellm.set_verbose = settings.DEBUG
        litellm.drop_params = True  # Drop unsupported params instead of failing
        log.info("LLMGateway initialized with LiteLLM")

    def _get_model_string(self, provider: str, model: str) -> str:
        """Build the LiteLLM model string"""
        if "/" in model:
            return model
        return f"{provider.strip().lower()}/{model}"

    def _estimate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Estimate cost in USD for the API call."""
        try:
            import litellm

            prompt_cost, completion_cost = litellm.cost_per_token(
                model=self._get_model_string(provider, model),
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
            return (prompt_cost or 0.0) + (completion_cost or 0.0)
        except Exception as e:
            log.warning(f"Failed to estimate cost: {e}")
            return 0.0

    async def chat(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict] = None,
        timeout: int = 60,
        fallback_models: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> LLMResponse:
        """Execute a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            provider: LLM provider (openai, anthropic, google, etc.)
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: List of tool definitions for function calling
            tool_choice: Tool selection strategy
            response_format: JSON mode configuration
            timeout: Request timeout in seconds
            fallback_models: Fallback models if primary fails
            user_id: User ID for tracking
            trace_id: Trace ID for observability
            
        Returns:
            LLMResponse with content and usage metadata
        """
        provider = provider or settings.LLM_PROVIDER
        model = model or settings.LLM_MODEL
        if not api_key:
            raise ValueError(
                f"API key is not configured for provider '{provider}'. "
                "Please add the API key before using this agent."
            )
        
        model_string = self._get_model_string(provider, model)
        
        # Build request parameters
        kwargs = {
            "model": model_string,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout,
            "metadata": {
                "user_id": user_id,
                "trace_id": trace_id,
            }
        }
        
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
            
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
                
        if response_format:
            kwargs["response_format"] = response_format
            
        if fallback_models:
            kwargs["fallbacks"] = [
                self._get_model_string(provider, m) for m in fallback_models
            ]
        
        kwargs["api_key"] = api_key

        start_time = datetime.utcnow()
        
        try:
            response = await acompletion(**kwargs)
            
            # Calculate latency
            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Extract usage - handle both dict and LiteLLM wrapper objects
            usage = getattr(response, "usage", {}) or {}
            if hasattr(usage, 'get'):
                input_tokens = usage.get("prompt_tokens", 0) or 0
                output_tokens = usage.get("completion_tokens", 0) or 0
                prompt_details = usage.get("prompt_tokens_details", {})
            else:
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0
                prompt_details = getattr(usage, "prompt_tokens_details", {})
            
            # Handle cached_tokens from prompt_details (could be dict or object)
            if hasattr(prompt_details, 'get'):
                cached_tokens = prompt_details.get("cached_tokens", 0) or 0
            else:
                cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
            
            # Get content and tool calls
            message = response.choices[0].message
            content = message.get("content", "") or ""
            tool_calls = message.get("tool_calls")
            finish_reason = response.choices[0].get("finish_reason")
            
            # Estimate cost
            cost_usd = self._estimate_cost(
                provider, model, input_tokens, output_tokens, cached_tokens
            )
            
            log.info(
                f"LLM call: {provider}/{model}, tokens={input_tokens}+{output_tokens}, "
                f"cost=${cost_usd:.4f}, latency={latency_ms}ms"
            )
            
            return LLMResponse(
                content=content,
                raw_response=response,
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
            )
            
        except Exception as e:
            log.error(f"LLM call failed: {e}")
            raise

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        timeout: int = 60,
        user_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> AsyncIterator[StreamingChunk]:
        """Execute a streaming chat completion.
        
        Yields StreamingChunk objects as content is generated.
        """
        provider = provider or settings.LLM_PROVIDER
        model = model or settings.LLM_MODEL
        
        model_string = self._get_model_string(provider, model)
        
        kwargs = {
            "model": model_string,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "timeout": timeout,
            "metadata": {
                "user_id": user_id,
                "trace_id": trace_id,
            }
        }
        
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools

        try:
            response = await acompletion(**kwargs)
            
            full_content = ""
            tool_calls_buffer = []
            
            start_time = datetime.utcnow()
            
            async for chunk in response:
                delta = chunk.choices[0].delta
                content = delta.get("content", "") or ""
                full_content += content
                
                # Check for tool calls in streaming
                if delta.get("tool_calls"):
                    tool_calls_buffer.extend(delta["tool_calls"])
                
                # Check if finished
                is_finished = chunk.choices[0].get("finish_reason") is not None
                
                # Extract usage from final chunk
                if is_finished and hasattr(chunk, 'usage'):
                    usage = chunk.usage or {}
                    input_tokens = usage.get("prompt_tokens", 0) or 0
                    output_tokens = usage.get("completion_tokens", 0) or 0
                    prompt_details = usage.get("prompt_tokens_details", {})
                    if hasattr(prompt_details, 'get'):
                        cached_tokens = prompt_details.get("cached_tokens", 0) or 0
                    else:
                        cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
                
                yield StreamingChunk(
                    content=content,
                    is_finished=is_finished,
                    finish_reason=chunk.choices[0].get("finish_reason"),
                    tool_calls=tool_calls_buffer if is_finished else None,
                )
            
        except Exception as e:
            log.error(f"Streaming LLM call failed: {e}")
            raise

    async def embed(
        self,
        texts: List[str],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            provider: Embedding provider
            model: Embedding model
            user_id: User ID for tracking
            api_key: API key override for the LLM provider
            
        Returns:
            Dict with embeddings and usage info
        """
        provider = provider or "openai"
        model = model or settings.VECTOR_EMBEDDING_MODEL
        
        model_string = self._get_model_string(provider, model)
        
        kwargs = {
            "model": model_string,
            "input": texts,
            "metadata": {"user_id": user_id},
        }
        if api_key:
            kwargs["api_key"] = api_key
        
        try:
            start_time = datetime.utcnow()
            response = await aembedding(**kwargs)
            
            embeddings = [item["embedding"] for item in response["data"]]
            usage = response.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return {
                "embeddings": embeddings,
                "input_tokens": input_tokens,
                "total_tokens": total_tokens,
            }
            
        except Exception as e:
            log.error(f"Embedding failed: {e}")
            raise

    def count_tokens(
        self,
        messages: List[Dict[str, str]],
        provider: str,
        model: str,
    ) -> int:
        """Count tokens in messages without making API call."""
        model_string = self._get_model_string(provider, model)
        
        try:
            return token_counter(
                model=model_string,
                messages=messages,
            )
        except Exception as e:
            log.warning(f"Token counting failed: {e}")
            # Fallback: rough estimate (4 chars per token)
            total_chars = sum(len(m.get("content", "")) for m in messages)
            return total_chars // 4

    def supports_tools(self, provider: str, model: str) -> bool:
        """Check if provider/model supports function calling."""
        # Most modern models support tools
        supported = [
            "gpt-4", "gpt-4o", "gpt-3.5-turbo",
            "claude-3", "claude-3-5",
            "gemini-1.5", "gemini-pro",
            "mistral-large", "mistral-medium",
            "command-r", "command-r-plus",
        ]
        
        model_lower = model.lower()
        return any(s in model_lower for s in supported)

    def supports_json_mode(self, provider: str, model: str) -> bool:
        """Check if provider/model supports structured JSON output."""
        return self.supports_tools(provider, model)

    def supports_vision(self, provider: str, model: str) -> bool:
        """Check if provider/model supports vision/image input."""
        vision_models = [
            "gpt-4o", "gpt-4-turbo", "gpt-4-vision",
            "claude-3", "claude-3-5",
            "gemini-1.5", "gemini-pro-vision",
        ]
        
        model_lower = model.lower()
        return any(v in model_lower for v in vision_models)


# Singleton instance
_llm_gateway: Optional[LLMGateway] = None


def get_llm_gateway() -> LLMGateway:
    """Get or create the LLM Gateway singleton."""
    global _llm_gateway
    if _llm_gateway is None:
        _llm_gateway = LLMGateway()
    return _llm_gateway
