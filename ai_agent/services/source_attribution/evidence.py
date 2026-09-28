"""Run-scoped evidence objects used by source attribution.

These objects represent evidence available during an agent run. They are not
database tables and are not directly returned as user-facing citations.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ai_agent.core.enums import SourceType


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    """A normalized, safe-to-prompt piece of evidence.

    ``source_ref`` is an opaque ID valid for one agent run only. It is what
    the LLM will reference in a later citation marker. ``source_id`` is the
    stable identifier used internally for authorization and persistence.
    """

    source_ref: str
    source_type: SourceType
    source_id: str
    title: str
    excerpt: str
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SourceRegistry:
    """Evidence made available to the final-answer LLM during one run."""

    _sources: Dict[str, SourceEvidence] = field(default_factory=dict)

    def register(self, evidence: SourceEvidence) -> None:
        """Register evidence once; conflicting run-local IDs are invalid."""
        if not evidence.source_ref:
            raise ValueError("source_ref is required")
        if evidence.source_ref in self._sources:
            raise ValueError(f"Duplicate source_ref: {evidence.source_ref}")
        self._sources[evidence.source_ref] = evidence

    def get(self, source_ref: str) -> Optional[SourceEvidence]:
        """Return registered evidence, or None for an unknown LLM reference."""
        return self._sources.get(source_ref)

    def values(self) -> list[SourceEvidence]:
        """Return evidence in registration order for deterministic prompting."""
        return list(self._sources.values())
