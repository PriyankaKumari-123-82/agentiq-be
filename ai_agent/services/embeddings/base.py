"""Shared contract and validation for embedding providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModelSpec:
    id: str
    provider: str
    dimension: int


class EmbeddingProvider(ABC):
    """Provider-independent embedding interface."""

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        model: EmbeddingModelSpec,
    ) -> list[list[float]]:
        """Return one vector for every input text, preserving input order."""


def iter_batches(texts: list[str], batch_size: int):
    if batch_size <= 0:
        raise RuntimeError("EMBEDDINGS_BATCH_SIZE must be greater than zero")
    for start in range(0, len(texts), batch_size):
        yield texts[start : start + batch_size]
