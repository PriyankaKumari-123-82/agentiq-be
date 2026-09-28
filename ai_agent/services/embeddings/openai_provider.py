import logging

from openai import AsyncOpenAI

from ai_agent.core.config import settings
from ai_agent.services.embeddings.base import EmbeddingModelSpec, EmbeddingProvider

log = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            if not settings.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    async def embed(
        self,
        texts: list[str],
        model: EmbeddingModelSpec,
    ) -> list[list[float]]:
        client = self._get_client()

        response = await client.embeddings.create(model=model.id, input=texts)
        vectors = [item.embedding for item in response.data]
        log.info(
            f"openai_embed: model={model.id} texts={len(texts)} "
            f"dim={len(vectors[0]) if vectors else 0}"
        )
        return vectors
