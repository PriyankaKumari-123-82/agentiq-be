"""Langfuse integration for observability and tracing.

This module provides:
- Automatic tracing of LLM calls
- Agent run tracking
- Prompt management
- Usage analytics
"""
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

from ai_agent.core.config import settings

log = logging.getLogger(__name__)


class LangfuseIntegration:
    """Langfuse integration for Agent IQ observability."""

    def __init__(self):
        """Initialize Langfuse client."""
        self.enabled = False
        self.client: Optional[Langfuse] = None
        
        # Check for Langfuse credentials
        public_key = getattr(settings, 'LANGFUSE_PUBLIC_KEY', None)
        secret_key = getattr(settings, 'LANGFUSE_SECRET_KEY', None)
        host = getattr(settings, 'LANGFUSE_HOST', 'https://cloud.langfuse.com')
        
        if public_key and secret_key:
            try:
                self.client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                self.enabled = True
                log.info(f"Langfuse initialized: {host}")
            except Exception as e:
                log.warning(f"Failed to initialize Langfuse: {e}")
        else:
            log.info("Langfuse not configured - observability disabled")

    def is_enabled(self) -> bool:
        """Check if Langfuse is enabled."""
        return self.enabled and self.client is not None

    def create_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[Any]:
        """Create a new trace.
        
        Args:
            name: Trace name
            user_id: User ID
            session_id: Session ID
            metadata: Additional metadata
            tags: Tags for the trace
            
        Returns:
            Trace object or None if disabled
        """
        if not self.is_enabled():
            return None
        
        try:
            trace = self.client.trace(
                name=name,
                user_id=str(user_id) if user_id else None,
                session_id=session_id,
                metadata=metadata,
                tags=tags or [],
            )
            return trace
        except Exception as e:
            log.warning(f"Failed to create Langfuse trace: {e}")
            return None

    def create_span(
        self,
        trace_id: str,
        name: str,
        input_data: Optional[Any] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[Any]:
        """Create a span within a trace.
        
        Args:
            trace_id: Parent trace ID
            name: Span name
            input_data: Input data
            metadata: Span metadata
            
        Returns:
            Span object or None
        """
        if not self.is_enabled():
            return None
        
        try:
            span = self.client.span(
                trace_id=trace_id,
                name=name,
                input=input_data,
                metadata=metadata,
            )
            return span
        except Exception as e:
            log.warning(f"Failed to create Langfuse span: {e}")
            return None

    def create_generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        provider: str,
        input_data: Any,
        output_data: Optional[Any] = None,
        usage: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[Any]:
        """Create a generation event (LLM call).
        
        Args:
            trace_id: Parent trace ID
            name: Generation name
            model: Model used
            provider: Provider name
            input_data: Input (prompt/messages)
            output_data: Output (completion)
            usage: Token usage dict
            metadata: Additional metadata
            
        Returns:
            Generation object or None
        """
        if not self.is_enabled():
            return None
        
        try:
            generation = self.client.generation(
                trace_id=trace_id,
                name=name,
                model=model,
                model_parameters={
                    "provider": provider,
                },
                input=input_data,
                output=output_data,
                usage=usage,
                metadata=metadata,
            )
            return generation
        except Exception as e:
            log.warning(f"Failed to create Langfuse generation: {e}")
            return None

    def score_trace(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
    ):
        """Add a score to a trace.
        
        Args:
            trace_id: Trace ID
            name: Score name (e.g., 'user_feedback', 'correctness')
            value: Score value (0-1 typically)
            comment: Optional comment
        """
        if not self.is_enabled():
            return
        
        try:
            self.client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
            )
        except Exception as e:
            log.warning(f"Failed to score trace: {e}")

    def update_trace(
        self,
        trace_id: str,
        output: Optional[Any] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        """Update a trace with final output.
        
        Args:
            trace_id: Trace ID
            output: Final output
            metadata: Updated metadata
            tags: Updated tags
        """
        if not self.is_enabled():
            return
        
        try:
            # Get the trace and update it
            trace = self.client.get_trace(trace_id)
            if trace:
                trace.update(
                    output=output,
                    metadata=metadata,
                    tags=tags,
                )
        except Exception as e:
            log.warning(f"Failed to update trace: {e}")

    def get_prompt(self, prompt_name: str, version: Optional[int] = None) -> Optional[str]:
        """Get a managed prompt from Langfuse.
        
        Args:
            prompt_name: Name of the prompt
            version: Specific version (optional)
            
        Returns:
            Prompt text or None
        """
        if not self.is_enabled():
            return None
        
        try:
            prompt = self.client.get_prompt(prompt_name, version=version)
            return prompt.prompt if prompt else None
        except Exception as e:
            log.warning(f"Failed to get prompt {prompt_name}: {e}")
            return None


def get_langfuse() -> LangfuseIntegration:
    """Get or create the Langfuse integration singleton."""
    global _langfuse_integration
    try:
        return _langfuse_integration
    except NameError:
        _langfuse_integration = LangfuseIntegration()
        return _langfuse_integration


# Decorator for automatic tracing
def trace_agent_run(name: Optional[str] = None):
    """Decorator to trace agent runs with Langfuse.
    
    Usage:
        @trace_agent_run(name="chat_completion")
        async def my_agent_function(...):
            ...
    """
    def decorator(func):
        if not get_langfuse().is_enabled():
            return func
        
        return observe(name=name or func.__name__)(func)
    return decorator
