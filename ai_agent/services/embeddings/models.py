"""
Supported embedding-model catalog.
"""

from dataclasses import asdict, dataclass
from enum import Enum


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    BAAI = "baai"


@dataclass(frozen=True)
class EmbeddingModelSpec:
    id: str
    display_name: str
    provider: EmbeddingProvider
    dimension: int

    def to_dict(self) -> dict[str, str | int]:
        data = asdict(self)
        data["provider"] = self.provider.value
        return data


EMBEDDING_MODELS: dict[str, EmbeddingModelSpec] = {
    "text-embedding-3-small": EmbeddingModelSpec(
        id="text-embedding-3-small",
        display_name="text-embedding-3-small",
        provider=EmbeddingProvider.OPENAI,
        dimension=1536,
    ),
    "bge-large-en-v1.5": EmbeddingModelSpec(
        id="bge-large-en-v1.5",
        display_name="bge-large-en-v1.5",
        provider=EmbeddingProvider.BAAI,
        dimension=1024,
    ),
}
