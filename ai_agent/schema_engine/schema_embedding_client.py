"""Embedding client for the schema engine.

Routes through the independent embeddings service (settings.EMBEDDINGS_BASE_URL /
EMBEDDINGS_API_KEY) using the model/dimension configured on ``SchemaEngineConfig``
(env-driven via ``SE_EMBEDDING_MODEL`` / ``SE_EMBEDDING_DIMENSIONS``).
"""

from __future__ import annotations

from typing import List

from ai_agent.schema_engine.config import SchemaEngineConfig
from ai_agent.services.embeddings.base import EmbeddingModelSpec
from ai_agent.services.embeddings.baai_provider import BAAIEmbeddingProvider

_provider = BAAIEmbeddingProvider()


async def embed_texts(cfg: SchemaEngineConfig, texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts using the schema engine's configured model."""
    model = EmbeddingModelSpec(
        id=cfg.embedding_model,
        provider="baai",
        dimension=cfg.embedding_dimensions,
    )
    return await _provider.embed(texts, model)
