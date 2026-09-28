"""RAG Service for retrieval-augmented generation.

Retrieves documents from the org-level Milvus collection scoped to the
documents the requesting agent is permitted to access.

Flow:
  1. Resolve the agent's organization_id.
  2. Fetch permitted document_ids from AgentDocument (filtered by agent).
  3. Search the org's Milvus collection (org_{org_id}) filtered by those doc IDs.
  4. Rerank results using external TEI service.
"""
import logging
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from ai_agent.core.config import settings
from ai_agent.core.response_schemas import CitationBlock
from ai_agent.core.vector_db import get_permitted_document_ids
from ai_agent.models import AIAgent, AIAgentSettings, AgentDocument
from ai_agent.services.embedding import EmbeddingService, get_org_embedding_model
from ai_agent.services.reranking import reranking_service

log = logging.getLogger(__name__)


class RAGService:

    async def retrieve(
        self,
        query: str,
        agent_id: str,
        agent_settings: Optional[AIAgentSettings] = None,
        top_k: int = 10,
        db_session: Optional[Session] = None,
        session_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Retrieve relevant document chunks for a query.

        Args:
            query: The user query string.
            agent_id: The ID of the requesting agent.
            agent_settings: Agent settings (unused; kept for call-site compatibility).
            top_k: Maximum number of chunks to retrieve.
            db_session: SQLModel session for access-control lookups.
            session_id: Optional session ID to include chat-scoped documents.

        Returns:
            Dict with keys: context (str), citations (list), tokens_used (int).
        """
        if db_session is None:
            log.warning("RAGService.retrieve: no db_session provided, skipping")
            return {"context": "", "citations": [], "tokens_used": 0}

        from ai_agent.core.vector_db import DocumentStore
        from ai_agent.models import SessionDocument
        from sqlmodel import select

        # ------------------------------------------------------------------
        # 1. Resolve the agent's organization
        # ------------------------------------------------------------------
        agent = db_session.get(AIAgent, agent_id)
        if not agent or agent.deleted:
            log.warning("RAGService.retrieve: agent %s not found", agent_id)
            return {"context": "", "citations": [], "tokens_used": 0}

        org_id: int = agent.organization_id or 0

        # ------------------------------------------------------------------
        # 2. Access-control: resolve permitted document IDs for this agent
        # ------------------------------------------------------------------
        # Every agent has knowledge-base access via the org's shared collection —
        # there's no per-agent opt-out anymore.
        permitted_document_ids: List[int] = list(get_permitted_document_ids(agent_id, db_session))

        # Add session documents if session_id is provided
        if session_id:
            session_docs = db_session.exec(
                select(SessionDocument.document_id).where(SessionDocument.session_id == session_id)
            ).all()
            if session_docs:
                permitted_document_ids.extend(session_docs)

        if not permitted_document_ids:
            log.info(
                "RAGService.retrieve: agent %s has no linked documents or session documents in org %d",
                agent_id, org_id,
            )
            return {"context": "", "citations": [], "tokens_used": 0}

        log.info(
            "RAGService.retrieve: agent %s, org %d, %d permitted document(s)",
            agent_id, org_id, len(permitted_document_ids),
        )

        # ------------------------------------------------------------------
        # 3. Embed the query
        # ------------------------------------------------------------------
        try:
            embedding_service = EmbeddingService()
            model_id = get_org_embedding_model(org_id, db_session)
            embedding_model = embedding_service.get_model_spec(model_id) if model_id else None
            if embedding_model is None:
                raise ValueError(f"Embedding model not found for org {org_id} (model_id={model_id})")
            query_embedding = await embedding_service.create_embedding(
                query,
                model=embedding_model,
            )
            log.info(
                "Query embedded model=%s dim=%d",
                embedding_model.id,
                len(query_embedding),
            )

            # ------------------------------------------------------------------
            # 4. Search the org's collection filtered by permitted doc IDs
            # ------------------------------------------------------------------
            org_store = DocumentStore()
            collection_name = org_store.get_collection_name(org_id, db_session)
            results = org_store.search(
                collection_name=collection_name,
                query_embedding=query_embedding,
                document_ids=list(permitted_document_ids),
                top_k=settings.RERANK_INITIAL_K,
            )

            log.info("Retrieved %d initial chunks for query in org_%d", len(results), org_id)

            # ------------------------------------------------------------------
            # 5. Rerank results
            # ------------------------------------------------------------------
            if results:
                # Extract texts in order
                texts = [r["text"] for r in results]
                
                # Call reranker
                scores = await reranking_service.rerank(query=query, texts=texts)
                
                # Assign scores back to results
                for i, r in enumerate(results):
                    r["rerank_score"] = scores[i]
                    r["original_rank"] = i  # Track original position
                
                # Sort by rerank score descending
                results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
                
                # Log all reranked chunks with scores
                log.info("Reranked chunks (sorted by score):")
                for new_rank, r in enumerate(results):
                    original_pos = r.get("original_rank", -1)
                    score = r.get("rerank_score", 0)
                    log.info(f"  index={new_rank}: original_index={original_pos}, score={score:.4f}")
                
                log.info("Keeping top %d chunks for context", settings.RERANK_TOP_K)
            
            # Always truncate to RERANK_TOP_K
            results = results[:settings.RERANK_TOP_K]
            
            log.info("Returning %d chunks for query in org_%d", len(results), org_id)

            context_parts = []
            citations = []

            for i, r in enumerate(results):
                content = r["text"]
                context_parts.append(f"[Source {i+1}]\n{content}\n")
                metadata = r.get("metadata") or {}
                citation = CitationBlock(
                    citation_id=f"citation_{i+1}",
                    source_type="document",
                    source_id=str(r.get("document_id", metadata.get("document_id", ""))),
                    source_name=metadata.get("file_name", "Unknown Document"),
                    content=content[:500],
                    page=metadata.get("page"),
                    section=metadata.get("section"),
                    relevance_score=r.get("rerank_score", r["score"]),
                )
                citations.append(citation)

            context = "\n".join(context_parts)
            tokens_used = len(context) // 4

            log.info(
                "RAG: %d citations, ~%d tokens context for agent %s",
                len(citations), tokens_used, agent_id,
            )
            return {
                "context": context,
                "citations": citations,
                "tokens_used": tokens_used,
                "nodes_count": len(results),
            }

        except Exception as e:
            log.error("RAG retrieval failed: %s", e, exc_info=True)
            return {"context": "", "citations": [], "tokens_used": 0, "error": str(e)}

    async def add_document(
        self,
        document_id: int,
        organization_id: int,
        file_content: bytes,
        file_name: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        metadata: Optional[Dict] = None,
        embedding_model_id: str | None = None,
        db_session: Any = None,
    ) -> Dict[str, Any]:
        """Embed and index a document into the org's collection.

        Args:
            document_id: The Postgres documents.id.
            organization_id: The org to embed into.
            file_content: Raw bytes of the file.
            file_name: Original filename (used for chunking and metadata).
            chunk_size: Text chunk size in characters.
            chunk_overlap: Chunk overlap in characters.
            metadata: Extra key/value pairs stored alongside each chunk.
            embedding_model_id: Override the default embedding model.
        """
        from ai_agent.core.vector_db import DocumentStore
        from ai_agent.core.config import settings as app_settings

        try:
            embedding_service = EmbeddingService()
            model_id = embedding_model_id or app_settings.VECTOR_EMBEDDING_MODEL
            embedding_model = embedding_service.get_model_spec(model_id)
            if embedding_model is None:
                return {"success": False, "error": f"Embedding model not found: {model_id}"}

            org_store = DocumentStore()
            chunks = org_store.create_chunks(
                file_path=file_name,
                chunk_size=chunk_size,
                overlap_size=chunk_overlap,
                file_content=file_content,
            )

            if not chunks:
                return {"success": False, "error": "No chunks extracted from document"}

            for chunk in chunks:
                m = chunk.setdefault("metadata", {})
                m["document_id"] = document_id
                m["organization_id"] = organization_id
                if metadata:
                    m.update(metadata)

            texts = [c["text"] for c in chunks]
            embeddings = await embedding_service.create_embedding(texts, model=embedding_model)

            collection_name = org_store.create_collection(organization_id, db_session, embedding_model.dimension)
            org_store.insert(
                collection_name=collection_name,
                org_id=organization_id,
                document_id=document_id,
                embeddings=embeddings,
                texts=texts,
                metadatas=[c.get("metadata", {}) for c in chunks],
                expected_dimension=embedding_model.dimension,
            )

            log.info("Added document %s with %d chunks (doc_id=%d, org_id=%d)",
                     file_name, len(chunks), document_id, organization_id)
            return {
                "success": True,
                "chunks_created": len(chunks),
                "document_id": document_id,
            }

        except Exception as e:
            log.error("Failed to add document: %s", e, exc_info=True)
            return {"success": False, "error": str(e)}


# Singleton
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def search_vector_store_for_context(
    session: Any,
    agent_settings: Any,
    user_message: str,
    db_session: Optional[Session] = None,
) -> str:
    """Search the vector store for context relevant to ``user_message``.

    Synchronous wrapper around ``RAGService.retrieve`` to support legacy pipelines.
    """
    if not (session and getattr(session, "agent_id", None) and agent_settings):
        return ""

    if db_session is None:
        log.warning("search_vector_store_for_context: no db_session, skipping vector search")
        return ""

    try:
        import asyncio
        import concurrent.futures
        rag_service = get_rag_service()
        top_k = settings.VECTOR_MAX_CHUNKS

        def _run_coro():
            return asyncio.run(
                rag_service.retrieve(
                    query=user_message,
                    agent_id=session.agent_id,
                    agent_settings=agent_settings,
                    top_k=top_k,
                    db_session=db_session,
                    session_id=getattr(session, "id", None),
                )
            )

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_run_coro)
                    result = future.result()
            else:
                result = loop.run_until_complete(
                    rag_service.retrieve(
                        query=user_message,
                        agent_id=session.agent_id,
                        agent_settings=agent_settings,
                        top_k=top_k,
                        db_session=db_session,
                        session_id=getattr(session, "id", None),
                    )
                )
        except RuntimeError:
            result = _run_coro()

        return result.get("context", "") if isinstance(result, dict) else ""
    except Exception as e:
        log.error("Error searching vector store: %s", e, exc_info=True)
        return ""

