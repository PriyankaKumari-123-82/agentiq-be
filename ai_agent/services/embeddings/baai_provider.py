"""BAAI embedding provider using an OpenAI-compatible endpoint."""

from openai import APIStatusError, APITimeoutError, AsyncOpenAI

from ai_agent.core.config import settings
from ai_agent.services.embeddings.base import EmbeddingModelSpec, EmbeddingProvider


class BAAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            if not settings.EMBEDDINGS_BASE_URL:
                raise RuntimeError("EMBEDDINGS_BASE_URL is not set")
            self._client = AsyncOpenAI(
                api_key=settings.EMBEDDINGS_API_KEY,
                base_url=f"{settings.EMBEDDINGS_BASE_URL}/v1",
                timeout=settings.EMBEDDINGS_TIMEOUT,
            )
        return self._client

    async def embed(
        self,
        texts: list[str],
        model: EmbeddingModelSpec,
    ) -> list[list[float]]:
        client = self._get_client()
        try:
            response = await client.embeddings.create(
                model=model.id,
                input=texts,
            )
        except APITimeoutError as exc:
            raise RuntimeError("BAAI embedding request timed out") from exc
        except APIStatusError as exc:
            raise RuntimeError(
                f"BAAI embedding request failed with HTTP {exc.status_code}"
            ) from exc

        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
