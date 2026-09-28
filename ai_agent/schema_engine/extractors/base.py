"""Abstract base for metadata extractors."""

from __future__ import annotations

import abc
from typing import Optional

from ai_agent.schema_engine.models import DatabaseMetadata


class BaseMetadataExtractor(abc.ABC):
    """Contract every extractor must satisfy."""

    @abc.abstractmethod
    async def extract(
        self,
        server_name: str,
        database_name: str = "default",
        *,
        include_sample_values: bool = True,
        sample_limit: int = 5,
    ) -> DatabaseMetadata:
        """Return normalised ``DatabaseMetadata`` for the target database."""
        ...

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is reachable."""
        ...
