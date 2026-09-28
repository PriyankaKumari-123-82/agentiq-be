"""Reranking service for RAG retrieval optimization.

Calls an externally-deployed TEI (Text Embeddings Inference) reranker service
over HTTP to reorder candidate documents by relevance.
"""
import logging
from typing import Optional

import httpx

from ai_agent.core.config import settings

log = logging.getLogger(__name__)


class RerankingService:
    """Client for TEI reranker service."""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        """create and return the httpx client.
        
        Raises:
            RuntimeError: If RERANK_BASE_URL is not configured.
        """
        if self._client is None:
            if not settings.RERANK_BASE_URL:
                raise RuntimeError("RERANK_BASE_URL is not set")
            
            headers = {"Content-Type": "application/json"}
            if settings.RERANK_API_KEY:
                headers["Authorization"] = f"Bearer {settings.RERANK_API_KEY}"
            
            self._client = httpx.AsyncClient(
                base_url=settings.RERANK_BASE_URL,
                timeout=settings.RERANK_TIMEOUT,
                headers=headers,
            )
        return self._client

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Rerank a list of text candidates by relevance to the query.
        
        Args:
            query: The search query or question.
            texts: List of candidate text chunks to rerank.
            
        Returns:
            List of relevance scores in the same order as the input texts.
            
        Raises:
            RuntimeError: If the reranker service is not configured or the request fails.
        """
        if not texts:
            return []

        client = self._get_client()
        
        try:
            payload = {
                "query": query,
                "texts": texts,
            }
            
            response = await client.post("/rerank", json=payload)
            response.raise_for_status()
            
            # Response format: [{"index": int, "score": float}, ...]
            results = response.json()
            
            # Create a score array in the original input order
            scores = [0.0] * len(texts)
            for item in results:
                idx = item["index"]
                score = item["score"]
                scores[idx] = score
            
            log.info("Reranking completed: %d candidates processed", len(texts))
            
            return scores
            
        except Exception as e:
            log.error("Reranking failed: %s", e, exc_info=True)
            raise

reranking_service = RerankingService()
