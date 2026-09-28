"""Embedding service for model catalog lookup and provider routing."""

import logging
from typing import Union

from ai_agent.core.config import settings
from ai_agent.core.db import engine
from ai_agent.models import ModelProvider, OrganizationSettings
from ai_agent.services.embeddings.base import EmbeddingModelSpec, iter_batches
from ai_agent.services.embeddings.openai_provider import OpenAIEmbeddingProvider
from ai_agent.services.embeddings.baai_provider import BAAIEmbeddingProvider
from sqlmodel import Session, select


log = logging.getLogger(__name__)


def get_org_embedding_model(organization_id: int | None, db_session: Session) -> str | None:
    """Look up the embedding model configured for an organization.

    This is the single source of truth for which model is used to index and
    retrieve RAG documents for every agent in the organization.
    """
    if organization_id is None:
        return None
    org_settings = db_session.exec(
        select(OrganizationSettings).where(OrganizationSettings.organization_id == organization_id)
    ).first()
    return org_settings.embedding_model if org_settings else None


class EmbeddingService:

    def __init__(self):
        self._providers = {
            "openai": OpenAIEmbeddingProvider(),
            "baai": BAAIEmbeddingProvider(),
        }

    def get_model_spec(
        self,
        model_id: str,
        db_session: Session | None = None,
    ) -> EmbeddingModelSpec | None:
        def _query(session: Session) -> EmbeddingModelSpec | None:
            row = session.exec(
                select(ModelProvider).where(
                    ModelProvider.model_name == model_id,
                    ModelProvider.type == "embeddings",
                )
            ).first()
            if row is None or row.dimension is None:
                return None
            return EmbeddingModelSpec(
                id=row.model_name,
                provider=row.provider,
                dimension=row.dimension,
            )

        if db_session is not None:
            return _query(db_session)

        with Session(engine) as session:
            return _query(session)

    async def create_embedding(
        self,
        texts: Union[str, list[str]],
        *,
        model: EmbeddingModelSpec,
    ) -> Union[list[float], list[list[float]]]:
        single = isinstance(texts, str)
        input_texts = [texts] if single else list(texts)
        if not input_texts:
            return []

        provider = self._providers.get(model.provider)
        vectors: list[list[float]] = []
        for batch in iter_batches(input_texts, settings.EMBEDDINGS_BATCH_SIZE):
            vectors.extend(await provider.embed(batch, model))

        log.info(
            "Generated %d embeddings with provider=%s model=%s dimension=%d",
            len(vectors),
            model.provider,
            model.id,
            model.dimension,
        )
        return vectors[0] if single else vectors
