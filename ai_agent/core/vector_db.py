from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import json
import logging
import uuid as uuid_mod
from pathlib import Path

from ai_agent.core.config import settings

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logger
logger = logging.getLogger(__name__)


def get_permitted_document_ids(
    agent_id: str,
    db_session: Any,
) -> List[int]:
    """Return the document IDs accessible to the given agent using AgentDocument.

    Org-level isolation is enforced by the Milvus collection (``org_{org_id}``)
    rather than by filtering the agent_documents table on organization_id.
    """
    from sqlmodel import select
    from ai_agent.models import AgentDocument
    link_ids = db_session.exec(
        select(AgentDocument.document_id).where(
            AgentDocument.agent_id == agent_id,
        )
    ).all()
    return list(link_ids)


class VectorStoreService(ABC):
    """
    Abstract base class for managing vector storage for AI agents.
    Keeps your business logic independent from Milvus details.
    """

    @abstractmethod
    def create_agent_store(self, agent_id: str, dimension: int | None = None):
        """Initialize storage for an agent (collection or partition)."""
        pass

    @abstractmethod
    def insert_vectors(
        self,
        collection_name: str,
        embeddings: List[List[float]],
        payloads: List[Dict[str, Any]],
        agent_id: str,
        expected_dimension: int | None = None,
    ):
        """Insert vectors + metadata for an agent."""
        pass

    @abstractmethod
    def search_vectors(
        self,
        collection_name: str,
        query_embedding: List[float],
        agent_id: str,
        top_k: int = settings.VECTOR_MAX_CHUNKS,
        expected_dimension: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Search vectors for a given agent."""
        pass

    @abstractmethod
    def delete_agent_store(self, collection_name: str):
        """Delete the agent's storage (collection or partition)."""
        pass


_collection_per_agent_instance: "CollectionPerAgent | None" = None


class CollectionPerAgent(VectorStoreService):
    _loaded_collections: set = set()

    def __new__(cls):
        global _collection_per_agent_instance
        if _collection_per_agent_instance is None:
            instance = super().__new__(cls)
            host = settings.MILVUS_HOST
            if not host.startswith(("http://", "https://", "tcp://", "unix://")):
                host = f"http://{host}:{settings.MILVUS_PORT}"
            logger.info(f"Initializing CollectionPerAgent with Milvus at {host}")
            connections.connect(
                alias="default",
                uri=host,
                user=settings.MILVUS_USER,
                password=settings.MILVUS_PASSWORD,
            )
            logger.info("Successfully connected to Milvus")
            _collection_per_agent_instance = instance
        return _collection_per_agent_instance

    @staticmethod
    def _resolve_dimension(dimension: int | None) -> int:
        return dimension if dimension is not None else settings.MILVUS_DIMENSION

    def _ensure_collection(
        self,
        collection_name: str,
        expected_dimension: int | None = None,
    ) -> None:
        expected_dimension = self._resolve_dimension(expected_dimension)
        logger.info(f"Ensuring collection exists: {collection_name}")
        if not utility.has_collection(collection_name):
            logger.info(f"Collection {collection_name} does not exist, creating new collection")
            self.create_agent_store(collection_name, dimension=expected_dimension)
            return

        logger.info(f"Collection {collection_name} exists, validating schema")
        collection = Collection(collection_name)
        
        # Check for required fields
        required_fields = ["embedding", "agent_id"]
        for field_name in required_fields:
            field = next((f for f in collection.schema.fields if f.name == field_name), None)
            if field is None:
                logger.error(f"Collection '{collection_name}' missing required field '{field_name}'")
                raise ValueError(f"Collection '{collection_name}' missing required field '{field_name}'")
        
        schema_embedding_field = next(
            (f for f in collection.schema.fields if f.name == "embedding"),
            None,
        )
        embedding_dim = getattr(schema_embedding_field, "params", {}).get("dim")
        if embedding_dim is not None:
            embedding_dim = int(embedding_dim)
        if embedding_dim != expected_dimension:
            logger.error(f"Collection '{collection_name}' embedding dim mismatch: expected {expected_dimension}, got {embedding_dim}")
            raise ValueError(
                f"Collection '{collection_name}' embedding dim mismatch: "
                f"expected {expected_dimension}, got {embedding_dim}"
            )

        if collection_name not in CollectionPerAgent._loaded_collections:
            collection.load()
            CollectionPerAgent._loaded_collections.add(collection_name)
            logger.info(f"Collection {collection_name} loaded successfully")

    def create_agent_store(self, collection_name: str, dimension: int | None = None):
        """
        Create a new collection for storing agent vectors and create an index on the embedding field.
        
        Args:
            collection_name: Name of the collection to create
        """
        dimension = self._resolve_dimension(dimension)
        logger.info(f"Creating new collection: {collection_name} with dimension {dimension}")
        
        # Define the collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="agent_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        
        # Create the collection
        schema = CollectionSchema(fields, description=f"Collection: {collection_name}")
        collection = Collection(name=collection_name, schema=schema)
        logger.info(f"Collection schema created for {collection_name}")
        
        # Create an index on the embedding field for efficient similarity search
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"Index created on embedding field for collection {collection_name}")
        
        # Load collection to memory to make it ready for search
        collection.load()
        CollectionPerAgent._loaded_collections.add(collection_name)
        logger.info(f"Collection {collection_name} created and loaded successfully")

    def insert_vectors(
        self,
        collection_name: str,
        embeddings: List[List[float]],
        payloads: List[Dict[str, Any]],
        agent_id: str,
        expected_dimension: int | None = None,
    ):
        expected_dimension = self._resolve_dimension(expected_dimension)
        logger.info(f"Inserting {len(payloads)} vectors into collection: {collection_name} for agent: {agent_id}")
        self._ensure_collection(collection_name, expected_dimension=expected_dimension)

        texts = [p["text"] for p in payloads]
        metadatas = [p.get("metadata", {}) for p in payloads]
        agent_ids = [agent_id] * len(payloads)

        if len(embeddings) != len(texts):
            logger.error(f"Embeddings count mismatch: {len(embeddings)} embeddings for {len(texts)} texts")
            raise ValueError(
                f"Embeddings count mismatch: {len(embeddings)} embeddings for {len(texts)} texts"
            )
        for index, embedding in enumerate(embeddings):
            if len(embedding) != expected_dimension:
                logger.error(
                    "Embedding dimension mismatch at index %d: expected %d, got %d",
                    index,
                    expected_dimension,
                    len(embedding),
                )
                raise ValueError(
                    f"Embedding dimension mismatch at index {index}: "
                    f"expected {expected_dimension}, got {len(embedding)}"
                )

        col = Collection(collection_name)
        col.insert([agent_ids, embeddings, texts, metadatas])
        col.flush()
        entity_count = col.num_entities
        logger.info(
            f"insert_vectors: collection={collection_name} agent={agent_id} "
            f"inserted={len(embeddings)} total_entities={entity_count}"
        )

    def search_vectors(
        self,
        collection_name: str,
        query_embedding,
        agent_id: str,
        top_k: int = settings.VECTOR_MAX_CHUNKS,
        expected_dimension: int | None = None,
    ):
        """
        Search for similar vectors in the specified collection.
        
        Args:
            collection_name: Name of the collection to search in
            query_embedding: The embedding vector to search with
            agent_id: Optional agent ID to filter results by
            top_k: Number of results to return
            
        Returns:
            List of matching documents with their scores
        """
        expected_dimension = self._resolve_dimension(expected_dimension)
        logger.info(f"Searching in collection {collection_name} with top_k={top_k}, query_embedding_dim={len(query_embedding)}, agent_id={agent_id}")

        if len(query_embedding) != expected_dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {expected_dimension}, "
                f"got {len(query_embedding)}"
            )

        # Check if collection exists
        if not utility.has_collection(collection_name):
            logger.error(f"Collection {collection_name} does not exist!")
            return []

        self._ensure_collection(
            collection_name,
            expected_dimension=expected_dimension,
        )

        col = Collection(collection_name)

        if collection_name not in CollectionPerAgent._loaded_collections:
            col.load()
            CollectionPerAgent._loaded_collections.add(collection_name)
            logger.info(f"Collection {collection_name} loaded into memory")

        expr = f"agent_id == '{agent_id}'"
        
        results = col.search(
            [query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["text", "metadata", "agent_id"],
            expr=expr
        )
        
        logger.info(f"Raw search results count: {len(results[0]) if results else 0}")
        
        search_results = [
            {
                "text": hit.entity.get("text"),
                "metadata": hit.entity.get("metadata"),
                "agent_id": hit.entity.get("agent_id"),
                "score": hit.distance,
            }
            for hit in results[0]
        ]
        
        logger.info(f"Returning {len(search_results)} results from {len(results[0]) if results else 0} total results")
        return search_results

    def delete_by_document_id(self, collection_name: str, document_id: int, agent_id: str | None = None) -> int:
        if not utility.has_collection(collection_name):
            logger.warning(f"delete_by_document_id: collection {collection_name} does not exist, skipping")
            return 0

        col = Collection(collection_name)
        if collection_name not in CollectionPerAgent._loaded_collections:
            col.load()
            CollectionPerAgent._loaded_collections.add(collection_name)

        if agent_id:
            expr = f"metadata['document_id'] == {document_id} and agent_id == '{agent_id}'"
        else:
            expr = f"metadata['document_id'] == {document_id}"
        result = col.delete(expr)
        col.flush()
        deleted = result.delete_count
        logger.info(f"delete_by_document_id: collection={collection_name} document_id={document_id} agent_id={agent_id} deleted={deleted}")
        return deleted

    def delete_agent_store(self, collection_name: str):
        """
        Delete the specified collection.

        Args:
            collection_name: Name of the collection to delete
        """
        logger.info(f"Deleting collection: {collection_name}")
        Collection(collection_name).drop()
        CollectionPerAgent._loaded_collections.discard(collection_name)
        logger.info(f"Successfully deleted collection: {collection_name}")

    def create_chunks_from_file(self, file_path: str, chunk_size: int, overlap_size: int, file_content: bytes = None) -> List[Dict[str, Any]]:
        """
        Create chunks from a file based on specified chunk size and overlap size.
        Uses LangChain for efficient document processing.

        Args:
            file_path: Path to file to be chunked (or original filename if using file_content)
            chunk_size: Size of each chunk in characters
            overlap_size: Number of characters to overlap between chunks
            file_content: Optional bytes content to process instead of reading from file_path

        Returns:
            List of dictionaries containing chunked data with schema fields
        """
        logger.info(f"Creating chunks from file: {file_path}, chunk_size={chunk_size}, overlap_size={overlap_size}")
        
        # Load document using appropriate LangChain loader
        if file_content is not None:
            logger.info("Processing from bytes content")
            # Process from bytes content
            import tempfile
            import os
            
            temp_file_path = None
            try:
                # Create temporary file only for LangChain processing
                with tempfile.NamedTemporaryFile(mode='wb', suffix=os.path.splitext(file_path)[1], delete=False) as temp_file:
                    temp_file.write(file_content)
                    temp_file_path = temp_file.name
                
                if file_path.lower().endswith('.pdf'):
                    logger.info("Loading PDF from temporary file")
                    loader = PyPDFLoader(temp_file_path)
                elif file_path.lower().endswith(('.docx', '.doc')):
                    logger.info("Loading DOCX from temporary file")
                    from docx import Document as DocxDocument
                    from langchain.schema import Document as LCDocument
                    docx_doc = DocxDocument(temp_file_path)
                    full_text = "\n".join(p.text for p in docx_doc.paragraphs if p.text.strip())
                    documents = [LCDocument(page_content=full_text, metadata={"source": file_path})]
                    loader = None
                else:
                    logger.info("Loading text file from temporary file")
                    loader = TextLoader(temp_file_path, encoding='utf-8')
                
                if loader is not None:
                    documents = loader.load()
                logger.info(f"Loaded {len(documents)} document pages from temporary file")
            finally:
                # Always clean up temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    logger.info("Cleaned up temporary file")
        else:
            # Process from file path
            logger.info(f"Processing from file path: {file_path}")
            file_obj = Path(file_path)
            if not file_obj.exists():
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if file_path.lower().endswith('.pdf'):
                logger.info("Loading PDF file")
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif file_path.lower().endswith(('.docx', '.doc')):
                logger.info("Loading DOCX file")
                from docx import Document as DocxDocument
                from langchain.schema import Document as LCDocument
                docx_doc = DocxDocument(file_path)
                full_text = "\n".join(p.text for p in docx_doc.paragraphs if p.text.strip())
                documents = [LCDocument(page_content=full_text, metadata={"source": file_path})]
            else:
                logger.info("Loading text file")
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
            logger.info(f"Loaded {len(documents)} document pages from file")
        
        # Create text splitter with agent settings
        logger.info("Creating text splitter")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split documents into chunks
        logger.info("Splitting documents into chunks")
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Convert to our schema format
        result = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "text": chunk.page_content,
                "metadata": {
                    "file_path": file_path,
                    "file_name": Path(file_path).name,
                    "chunk_start": chunk.metadata.get('start', i * chunk_size),
                    "chunk_end": chunk.metadata.get('end', (i + 1) * chunk_size),
                    "chunk_size": len(chunk.page_content),
                    "overlap_size": overlap_size,
                    "source": chunk.metadata.get('source', file_path),
                    "page": chunk.metadata.get('page', None)
                }
            }
            result.append(chunk_data)
        
        logger.info(f"Processed {len(result)} chunks with metadata")
        return result




# =============================================================================
# Document Store — one Milvus collection per organisation
# =============================================================================
# Collection naming:  org_{organization_id}   (e.g. org_1, org_42)
# Each collection holds all document chunks that belong to that org.
# Access control is enforced at query time by filtering on the permitted
# document_ids resolved from AgentDocument (Postgres).
# =============================================================================

_document_store_instance: "DocumentStore | None" = None


class DocumentStore:
    """Manages per-organisation Milvus collections for document chunk storage.

    One collection per org: ``org_{organization_id}``.

    Schema (each collection)
    ------------------------
    id           INT64  primary key (auto)
    document_id  INT64  scalar index — used in WHERE clause for access control
    embedding    FLOAT_VECTOR
    text         VARCHAR(5000)
    metadata     JSON
    """

    # Class-level collection cache:  {collection_name: Collection}
    _collections: dict = {}
    _connected: bool = False

    def __new__(cls):
        global _document_store_instance
        if _document_store_instance is None:
            instance = super().__new__(cls)
            _document_store_instance = instance
        return _document_store_instance

    def _ensure_connected(self) -> None:
        """Establish Milvus connection once (idempotent)."""
        if DocumentStore._connected:
            return
        host = settings.MILVUS_HOST
        if not host.startswith(("http://", "https://", "tcp://", "unix://")):
            host = f"http://{host}:{settings.MILVUS_PORT}"
        logger.info("DocumentStore: connecting to Milvus at %s", host)
        connections.connect(
            alias="default",
            uri=host,
            user=settings.MILVUS_USER,
            password=settings.MILVUS_PASSWORD,
        )
        DocumentStore._connected = True
        logger.info("DocumentStore: connected to Milvus")

    def get_collection_name(self, org_id: int, db_session: Any) -> str:
        """Fetch the Milvus collection name for an org from the DB.
        """
        from sqlmodel import select
        from ai_agent.models import VectorCollection

        record = db_session.exec(
            select(VectorCollection).where(VectorCollection.organization_id == org_id)
        ).first()
        if record is None:
            raise ValueError(
                f"No VectorCollection found for org_id={org_id}. "
            )
        return record.collection_name

    @staticmethod
    def _resolve_dimension(dimension: int | None) -> int:
        return dimension if dimension is not None else settings.MILVUS_DIMENSION

    def _load_collection(self, collection_name: str) -> Collection:
        """Load and return an existing Milvus collection by name.

        Raises ``ValueError`` if the collection does not exist in Milvus.
        Always use this for read operations (search, delete).
        """
        self._ensure_connected()

        if collection_name in DocumentStore._collections:
            return DocumentStore._collections[collection_name]

        if not utility.has_collection(collection_name):
            raise ValueError(
                f"DocumentStore: collection '{collection_name}' does not exist. "
                "Embed a document first to create it."
            )

        collection = Collection(collection_name)
        collection.load()
        DocumentStore._collections[collection_name] = collection
        logger.info("DocumentStore: collection '%s' loaded from Milvus", collection_name)
        return collection

    def create_collection(
        self,
        org_id: int,
        db_session: Any,
        expected_dimension: int,
    ) -> str:
        """Ensure both the Milvus collection and the db ``VectorCollection`` row exist.

        This creates the Milvus collection FIRST. Only if Milvus succeeds
        then it save the row to Postgres, preventing stale DB records on Milvus failure.
        """
        from sqlmodel import select
        from ai_agent.models import VectorCollection
        import uuid

        record = db_session.exec(
            select(VectorCollection).where(VectorCollection.organization_id == org_id)
        ).first()

        self._ensure_connected()

        if record is not None:
            collection_name = record.collection_name
            if collection_name in DocumentStore._collections:
                return collection_name
            if utility.has_collection(collection_name):
                self._load_collection(collection_name)
                return collection_name
        else:
            collection_name = f"org_{uuid.uuid4().hex}"

        logger.info("DocumentStore: creating collection '%s' (dim=%d)", collection_name, expected_dimension)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_id", dtype=DataType.INT64),
            FieldSchema(name="organization_id", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=expected_dimension),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields, description=f"Document embeddings — {collection_name}")
        collection = Collection(name=collection_name, schema=schema)
        # Scalar index for fast WHERE filtering on document_id
        collection.create_index(
            field_name="document_id",
            index_params={"index_type": "STL_SORT"},
        )
        # Scalar index on organization_id for future cross-org queries
        collection.create_index(
            field_name="organization_id",
            index_params={"index_type": "STL_SORT"},
        )
        # ANN index on embedding vectors
        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
            },
        )
        collection.load()
        DocumentStore._collections[collection_name] = collection
        logger.info("DocumentStore: collection '%s' created and loaded", collection_name)

        if record is None:
            record = VectorCollection(organization_id=org_id, collection_name=collection_name)
            db_session.add(record)
            db_session.commit()
            db_session.refresh(record)
            logger.info("DocumentStore: saved PG VectorCollection for org_id=%d", org_id)

        return collection_name

    def insert(
        self,
        collection_name: str,
        org_id: int,
        document_id: int,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        expected_dimension: int | None = None,
    ) -> None:
        """Insert document chunk embeddings into a collection.

        Args:
            collection_name: Pre-resolved Milvus collection name.
            org_id: Organization ID — stored alongside each vector.
            document_id: The Postgres ``documents.id`` for the source document.
            embeddings: List of embedding vectors.
            texts: Parallel list of chunk text strings.
            metadatas: Parallel list of metadata dicts per chunk.
            expected_dimension: Embedding vector dimension.
        """
        if not embeddings:
            return
        if len(embeddings) != len(texts) or len(embeddings) != len(metadatas):
            raise ValueError(
                f"DocumentStore.insert: mismatch — "
                f"{len(embeddings)} embeddings, {len(texts)} texts, {len(metadatas)} metadatas"
            )
        col = self._load_collection(collection_name)

        doc_ids = [document_id] * len(texts)
        org_ids = [org_id] * len(texts)

        col.insert([doc_ids, org_ids, embeddings, texts, metadatas])
        col.flush()
        logger.info(
            "DocumentStore: inserted %d chunks for document_id=%d in collection '%s'",
            len(embeddings), document_id, collection_name,
        )

    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        document_ids: List[int],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search a collection, restricted to the given document IDs.

        Args:
            collection_name: Pre-resolved Milvus collection name.
            query_embedding: The query vector.
            document_ids: List of permitted ``document_id`` values.
            top_k: Maximum number of results.

        Returns:
            List of dicts with keys ``text``, ``metadata``, ``document_id``, ``score``.
        """
        if not document_ids:
            logger.info("DocumentStore.search: no permitted document_ids, returning empty")
            return []

        if not utility.has_collection(collection_name):
            logger.warning("DocumentStore.search: collection '%s' does not exist", collection_name)
            return []

        col = self._load_collection(collection_name)

        ids_str = ", ".join(str(i) for i in document_ids)
        expr = f"document_id in [{ids_str}]"

        results = col.search(
            [query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["text", "metadata", "document_id", "organization_id"],
            expr=expr,
        )

        search_results = [
            {
                "text": hit.entity.get("text"),
                "metadata": hit.entity.get("metadata"),
                "document_id": hit.entity.get("document_id"),
                "organization_id": hit.entity.get("organization_id"),
                "score": hit.distance,
            }
            for hit in results[0]
        ]
        logger.info(
            "DocumentStore.search: %d results in collection '%s' for %d permitted docs",
            len(search_results), collection_name, len(document_ids),
        )
        return search_results

    def delete_by_document_id(self, collection_name: str, document_id: int) -> int:
        """Delete all chunks belonging to a document from a collection.

        Returns the number of deleted entities.
        """
        if not utility.has_collection(collection_name):
            logger.warning(
                "DocumentStore.delete_by_document_id: collection '%s' does not exist, skipping",
                collection_name,
            )
            return 0

        col = self._load_collection(collection_name)
        expr = f"document_id == {document_id}"
        result = col.delete(expr)
        col.flush()
        deleted = result.delete_count
        logger.info(
            "DocumentStore: deleted %d chunks for document_id=%d from collection '%s'",
            deleted, document_id, collection_name,
        )
        return deleted

    def drop_collection(self, collection_name: str) -> None:
        """Drop the entire Milvus collection.

        Call this when an organisation is permanently deleted.
        """
        self._ensure_connected()
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            DocumentStore._collections.pop(collection_name, None)
            logger.info("DocumentStore: dropped collection '%s'", collection_name)
        else:
            logger.info("DocumentStore.drop_collection: '%s' not found, skipping", collection_name)

    def create_chunks(
        self,
        file_path: str,
        chunk_size: int,
        overlap_size: int,
        file_content: bytes,
    ) -> List[Dict[str, Any]]:
        """Split a document into text chunks (delegates to CollectionPerAgent chunker)."""
        store = CollectionPerAgent()
        return store.create_chunks_from_file(
            file_path=file_path,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            file_content=file_content,
        )


# ---------------------------------------------------------------------------
# Backward-compat alias — existing code that imports GlobalDocumentStore or
# OrgDocumentStore still works. Migrate callers to DocumentStore over time.
# ---------------------------------------------------------------------------
OrgDocumentStore = DocumentStore
GlobalDocumentStore = DocumentStore


def get_org_document_store() -> DocumentStore:
    """Return the singleton DocumentStore instance."""
    return DocumentStore()


def get_global_document_store() -> DocumentStore:
    """Backward-compatible alias."""
    return DocumentStore()
