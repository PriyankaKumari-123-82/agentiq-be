"""Web Search data models and schemas."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class WebSearchItem:
    """A single search result item."""
    title: str
    url: Optional[str] = None
    content: str = ""
    score: float = 1.0
    published_date: Optional[str] = None


@dataclass
class WebSearchResult:
    """Standardized web search result from any provider."""
    query: str
    provider: str
    results: List[WebSearchItem] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class WebSearchContext:
    """Formatted context ready for injection into LLM prompts."""
    context_text: str
    results: List[WebSearchItem]
    source_count: int
    query: str
