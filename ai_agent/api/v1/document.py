from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Iterable, Set
import io
import logging
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile, HTTPException, status, Query
from sqlmodel import Session, select
from pydantic import BaseModel, Field, ConfigDict
import os

from ai_agent.core import db
from sqlalchemy.sql import func
from ai_agent.core.aws_s3 import s3_client
from ai_agent.models import (
    Document, AIAgent, AIAgentSettings, Organization, User,
    OrganizationMembership, AgentDocument, OrganizationDocument,
)
from ai_agent.core.enums import ActionType, EntityType
from ai_agent.core.config import settings

from ai_agent.services.embedding import EmbeddingService, get_org_embedding_model
from ai_agent.core.vector_db import DocumentStore
from ai_agent.services.activity_log import log_activity

log = logging.getLogger(__name__)

router = APIRouter(tags=["Documents"])


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class DocumentOut(BaseModel):
    """Response model for a Document."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    file_name: str
    description: Optional[str] = None
    file_path: str
    file_size: int
    mime_type: str
    created_at: datetime
    updated_at: datetime
    is_active: bool
    # Populated by _enrich_doc_out
    linked_organizations: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Organizations this document belongs to: [{id, name}]",
    )
    linked_agents: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Agents linked to this document (org-scoped): [{id, name, organization_id}]",
    )


class DownloadUrlResponse(BaseModel):
    document_id: int
    download_url: str
    file_name: str


class DocumentUpdateRequest(BaseModel):
    description: Optional[str] = Field(
        None, description="Updated description. Omit to keep the current value."
    )
    chunk_size: Optional[int] = Field(
        None, description="chunk size when embedding for new organizations."
    )
    chunk_overlap: Optional[int] = Field(
        None, description="chunk overlap when embedding for new organizations."
    )
    agent_ids: Optional[List[str]] = Field(
        None,
        description=(
            "Full desired list of agent IDs to link (within each org this doc belongs to). "
            "Agents not in this list are unlinked. Each agent is linked only in the org it belongs to."
        ),
    )
    organization_ids: Optional[List[int]] = Field(
        None,
        description=(
            "Full desired list of organization IDs. "
            "New orgs get their own embedding; removed orgs lose their embedding. "
            "If removed from ALL orgs the document is soft-deleted."
        ),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def check_user_org_memberships(user_id: int, org_ids: Iterable[int], db_session: Session) -> None:
    """Verify that user_id is a member of all specified org_ids.

    Raises HTTP 403 if not.
    """
    org_set = set(o for o in org_ids if o is not None)
    if not org_set:
        return
    valid_memberships = db_session.exec(
        select(OrganizationMembership.organization_id).where(
            OrganizationMembership.user_id == user_id,
            OrganizationMembership.organization_id.in_(list(org_set)),
        )
    ).all()
    missing_orgs = org_set - set(valid_memberships)
    if missing_orgs:
        missing_id = sorted(missing_orgs)[0]
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User {user_id} is not a member of organization {missing_id}",
        )


def check_document_access(document: Document, user_id: int, db_session: Session) -> None:
    """Verify that user_id owns the document or belongs to at least one of its organizations.

    Raises HTTP 403 if not.
    """
    if document.user_id == user_id:
        return
    org_links = db_session.exec(
        select(OrganizationDocument.organization_id).where(OrganizationDocument.document_id == document.id)
    ).all()
    if not org_links:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No permission to access this document")
    membership = db_session.exec(
        select(OrganizationMembership.id).where(
            OrganizationMembership.user_id == user_id,
            OrganizationMembership.organization_id.in_(list(set(org_links))),
        )
    ).first()
    if not membership:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No permission to access this document")


def _enrich_doc_out(doc: Document, db_session: Session) -> DocumentOut:
    """Build a DocumentOut with linked_organizations and linked_agents."""
    out = DocumentOut.model_validate(doc)

    # --- Linked organizations ---
    org_links = db_session.exec(
        select(OrganizationDocument, Organization.name)
        .join(Organization, OrganizationDocument.organization_id == Organization.id)
        .where(OrganizationDocument.document_id == doc.id)
    ).all()
    out.linked_organizations = [
        {"id": link[0].organization_id, "name": link[1]} for link in org_links
    ]

    # --- Linked agents ---
    agent_links = db_session.exec(
        select(AgentDocument, AIAgent.name)
        .join(AIAgent, AgentDocument.agent_id == AIAgent.id)
        .where(
            AgentDocument.document_id == doc.id,
            AIAgent.deleted == False,
        )
    ).all()
    out.linked_agents = [
        {
            "id": link[0].agent_id,
            "name": link[1],
        }
        for link in agent_links
    ]

    return out


async def _embed_into_org(
    org_id: int,
    document_id: int,
    file_name: str,
    file_content: bytes,
    full_s3_path: str,
    user_id: int,
    embedding_model_id: str | None = None,
    session_id: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> None:
    try:
        embedding_service = EmbeddingService()
        model_id = embedding_model_id or settings.VECTOR_EMBEDDING_MODEL
        log.info(
            "Embedding document_id=%d (org_id=%d) using model=%s",
            document_id, org_id, model_id,
        )
        embedding_model = embedding_service.get_model_spec(model_id)
        if embedding_model is None:
            raise ValueError(f"Embedding model not found in catalog: {model_id}")

        doc_store = DocumentStore()

        with Session(db.engine) as bg_session:
            collection_name = doc_store.create_collection(
                org_id=org_id,
                db_session=bg_session,
                expected_dimension=embedding_model.dimension,
            )

        actual_chunk_size = chunk_size if chunk_size is not None else settings.VECTOR_CHUNK_SIZE
        actual_chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.VECTOR_CHUNK_OVERLAP

        chunks = doc_store.create_chunks(
            file_path=file_name,
            chunk_size=actual_chunk_size,
            overlap_size=actual_chunk_overlap,
            file_content=file_content,
        )

        for chunk in chunks:
            m = chunk.setdefault("metadata", {})
            m["user_id"] = user_id
            m["document_id"] = document_id
            m["organization_id"] = org_id
            m["s3_path"] = full_s3_path
            if session_id:
                m["session_id"] = session_id

        chunks = [c for c in chunks if c.get("text", "").strip()]
        texts = [c["text"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]
        if texts:
            embeddings = await embedding_service.create_embedding(texts, model=embedding_model)
            doc_store.insert(
                collection_name=collection_name,
                org_id=org_id,
                document_id=document_id,
                embeddings=embeddings,
                texts=texts,
                metadatas=metadatas,
                expected_dimension=embedding_model.dimension,
            )
            log.info(
                "Embedded document_id=%d into %s (org_id=%d) (%d chunks)",
                document_id, collection_name, org_id, len(texts),
            )

    except Exception as e:
        log.error(
            "Embedding failed for document_id=%d org_id=%d: %s",
            document_id, org_id, e,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/document/download", response_model=DownloadUrlResponse, status_code=status.HTTP_200_OK)
async def generate_download_url(
    document_id: int = Query(...),
    expiry_seconds: int = Query(3600, ge=60, le=604800),
    user_id: int = Query(...),
    db_session: Session = Depends(db.get_session),
):

    """Generate a presigned S3 download URL for a document."""
    document = db_session.get(Document, document_id)
    if not document or not document.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found or deleted",
        )

    # Permission check: user must be a member of an org that owns this document (or owner)
    check_document_access(document, user_id, db_session)

    download_url = s3_client.generate_presigned_url(
        object_name=document.file_path,
        expiration=expiry_seconds,
    )
    return DownloadUrlResponse(
        document_id=document.id,
        download_url=download_url,
        file_name=document.file_name,
    )


@router.post("/document/upload", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    user_id: int = Query(...),
    agent_ids: Optional[List[str]] = Query(None, description="Agent IDs to link (filtered by org)"),
    organization_ids: Optional[List[int]] = Query(None, description="At least one org ID is required"),
    description: Optional[str] = Form(None, description="Optional description of the document"),
    chunk_size: Optional[int] = Form(None, description="Optional custom chunk size"),
    chunk_overlap: Optional[int] = Form(None, description="Optional custom chunk overlap"),
    file: UploadFile = File(..., description="Only txt, pdf, doc, docx files accepted"),
    db_session: Session = Depends(db.get_session),
):

    """Upload a document.

    - One Document row is created regardless of how many orgs are specified.
    - An OrganizationDocument row is created per org.
    - The file is embedded once per org into that org's Milvus collection (org_{id}).
    - Agent links (AgentDocument) are created across specified agent_ids.
    - At least one organization_id is required.
    """
    # --- Require at least one org ---
    if organization_ids is None or len(organization_ids) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one organization_id is required when uploading a document.",
        )
    target_org_ids = list(set(o for o in organization_ids if o is not None))

    target_agent_ids = list(set(a for a in (agent_ids or []) if a))

    # --- Verify org memberships ---
    check_user_org_memberships(user_id, target_org_ids, db_session)

    # --- Validate file extension ---
    file_extension = os.path.splitext(file.filename)[1].lower()
    if not file_extension or file_extension[1:] not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}",
        )

    original_filename = os.path.basename(file.filename)
    file_content = await file.read()
    file_size = len(file_content)

    timestamp = int(datetime.now(timezone.utc).timestamp())
    folder_prefix = f"organizations/{target_org_ids[0]}"
    s3_key = f"{folder_prefix}/{timestamp}_{original_filename}"
    full_s3_path = f"{s3_client.bucket_name}/{s3_key}"

    await s3_client.upload_fileobj(
        file_obj=io.BytesIO(file_content),
        object_name=s3_key,
        content_type=file.content_type,
    )

    # --- Create single Document row ---
    db_doc = Document(
        user_id=user_id,
        file_name=original_filename,
        file_path=full_s3_path,
        file_size=file_size,
        mime_type=file.content_type,
        description=description,
    )
    db_session.add(db_doc)
    db_session.flush()  # get db_doc.id

    # --- Create per-org links ---
    for org_id in target_org_ids:
        db_session.add(OrganizationDocument(
            document_id=db_doc.id,
            organization_id=org_id,
            user_id=user_id,
        ))

    # --- Agent links (scoped to agents that belong to any of the target orgs) ---
    for agent_id in target_agent_ids:
        agent = db_session.get(AIAgent, agent_id)
        if agent and not agent.deleted and agent.organization_id in target_org_ids:
            existing = db_session.exec(
                select(AgentDocument).where(
                    AgentDocument.document_id == db_doc.id,
                    AgentDocument.agent_id == agent_id,
                )
            ).first()
            if not existing:
                db_session.add(AgentDocument(
                    document_id=db_doc.id,
                    agent_id=agent_id,
                    user_id=user_id,
                ))


    log_activity(
        db_session=db_session,
        user_id=user_id,
        action=ActionType.CREATE,
        entity_type=EntityType.DOCUMENT,
        entity_id=str(db_doc.id),
    )
    db_session.commit()
    db_session.refresh(db_doc)

    # --- Embed per org in the background (async, non-blocking) ---
    for org_id in target_org_ids:
        resolved_model_id = get_org_embedding_model(org_id, db_session)

        background_tasks.add_task(
            _embed_into_org,
            org_id=org_id,
            document_id=db_doc.id,
            file_name=original_filename,
            file_content=file_content,
            full_s3_path=full_s3_path,
            user_id=user_id,
            embedding_model_id=resolved_model_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    return _enrich_doc_out(db_doc, db_session)


@router.get("/document/documents", response_model=List[DocumentOut])
def list_documents(
    organization_ids: Optional[List[int]] = Query(
        None,
        description="Filter by organization IDs. Documents linked to any of these orgs are included.",
    ),
    agent_ids: Optional[List[str]] = Query(
        None,
        description="Filter by agent IDs. Documents linked to any of these agents are included.",
    ),
    db_session: Session = Depends(db.get_session),
):
    """List active documents filtered by organization and/or agent associations.

    - Results are the **union** of documents matching any provided ``organization_ids``
      or ``agent_ids``.
    - At least one of ``organization_ids`` or ``agent_ids`` must be supplied.
    - Each document in the response is enriched with ``linked_organizations`` and
      ``linked_agents`` association data.
    """
    target_org_ids = list(set(o for o in (organization_ids or []) if o is not None))
    target_agent_ids = list(set(a for a in (agent_ids or []) if a))

    if not target_org_ids and not target_agent_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one of 'organization_ids' or 'agent_ids' must be provided.",
        )

    doc_id_set: Set[int] = set()

    # --- Collect doc IDs from org filter ---
    if target_org_ids:
        org_doc_ids = db_session.exec(
            select(OrganizationDocument.document_id).where(
                OrganizationDocument.organization_id.in_(target_org_ids)
            )
        ).all()
        doc_id_set.update(org_doc_ids)

    # --- Collect doc IDs from agent filter ---
    if target_agent_ids:
        agent_doc_ids = db_session.exec(
            select(AgentDocument.document_id).where(
                AgentDocument.agent_id.in_(target_agent_ids)
            )
        ).all()
        doc_id_set.update(agent_doc_ids)

    if not doc_id_set:
        return []

    docs = db_session.exec(
        select(Document).where(
            Document.id.in_(list(doc_id_set)),
            Document.is_active == True,
        ).order_by(Document.created_at.desc())
    ).all()

    return [_enrich_doc_out(d, db_session) for d in docs]


@router.delete("/document/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    db_session: Session = Depends(db.get_session),
):
    """Soft-delete a document and purge its vectors from ALL org collections.

    Also deletes the S3 file if no other active document shares the same file path.
    """
    document = db_session.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document {document_id} not found")
    if not document.is_active:
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Document already deleted")

    # Collect org IDs before deleting links
    org_links = db_session.exec(
        select(OrganizationDocument).where(OrganizationDocument.document_id == document_id)
    ).all()
    org_ids = [ol.organization_id for ol in org_links]

    # Purge vectors from every org collection FIRST (before DB soft-delete)
    doc_store = DocumentStore()
    for org_id in org_ids:
        try:
            collection_name = doc_store.get_collection_name(org_id, db_session)
            doc_store.delete_by_document_id(collection_name=collection_name, document_id=document_id)
        except Exception as e:
            log.error("Failed to delete vectors for doc %d from org_%d: %s", document_id, org_id, e)

    # Soft-delete
    document.deleted_at = datetime.now(timezone.utc)
    document.is_active = False
    db_session.add(document)

    # Delete all agent and org links
    for link in db_session.exec(select(AgentDocument).where(AgentDocument.document_id == document_id)).all():
        db_session.delete(link)
    for ol in org_links:
        db_session.delete(ol)

    log_activity(
        db_session=db_session,
        user_id=document.user_id,
        action=ActionType.DELETE,
        entity_type=EntityType.DOCUMENT,
        entity_id=str(document.id),
    )
    db_session.commit()

    # Delete S3 file if no other active doc shares the same path
    other_docs = db_session.exec(
        select(Document).where(
            Document.file_path == document.file_path,
            Document.id != document_id,
            Document.is_active == True,
        )
    ).all()
    if not other_docs:
        try:
            s3_client.delete_file(document.file_path)
        except Exception as e:
            log.error("Failed to delete S3 file for doc %d: %s", document_id, e)

    return None


@router.patch("/document/{document_id}", response_model=DocumentOut)
async def update_document(
    document_id: int,
    request: DocumentUpdateRequest,
    background_tasks: BackgroundTasks,
    user_id: int = Query(...),
    db_session: Session = Depends(db.get_session),
):

    """Update a document's description, org memberships, and/or agent links.

    Organization changes:
    - New org → creates OrganizationDocument + embeds into that org's collection.
    - Removed org → deletes OrganizationDocument + purges vectors from that org.
    - If removed from ALL orgs → document is soft-deleted.

    Agent changes:
    - Diff applied via set operations across (document_id, agent_id) using AgentDocument.
    """
    document = db_session.get(Document, document_id)
    if not document or not document.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found or inactive")

    # Permission: user must own the doc or be a member of at least one of the doc's orgs
    check_document_access(document, user_id, db_session)
    existing_org_links = db_session.exec(
        select(OrganizationDocument).where(OrganizationDocument.document_id == document_id)
    ).all()
    existing_org_ids = {ol.organization_id for ol in existing_org_links}

    # --- Description ---
    if request.description:
        new_desc = request.description.strip()
        if new_desc:
            document.description = new_desc
            db_session.add(document)

    # --- Organization diff ---
    orgs_to_embed: List[int] = []          # new orgs that need embedding
    orgs_to_purge: List[int] = []          # removed orgs whose vectors must be purged

    if request.organization_ids is not None:
        new_org_ids = set(o for o in request.organization_ids if o is not None)
        orgs_to_add = new_org_ids - existing_org_ids
        orgs_to_remove = existing_org_ids - new_org_ids

        # Verify membership in all new orgs cleanly
        check_user_org_memberships(user_id, orgs_to_add, db_session)

        # Create new org links
        for org_id in orgs_to_add:
            db_session.add(OrganizationDocument(
                document_id=document_id,
                organization_id=org_id,
                user_id=user_id,
            ))
            orgs_to_embed.append(org_id)

        # Remove org links
        for org_link in existing_org_links:
            if org_link.organization_id in orgs_to_remove:
                db_session.delete(org_link)
                orgs_to_purge.append(org_link.organization_id)

        # If no orgs left → soft-delete the document
        remaining_orgs = new_org_ids  # = existing - removed + added
        if not remaining_orgs:
            agent_links_to_delete = db_session.exec(
                select(AgentDocument).where(AgentDocument.document_id == document_id)
            ).all()
            for al in agent_links_to_delete:
                db_session.delete(al)

            document.deleted_at = datetime.now(timezone.utc)
            document.is_active = False
            db_session.add(document)
            log_activity(
                db_session=db_session,
                user_id=user_id,
                action=ActionType.DELETE,
                entity_type=EntityType.DOCUMENT,
                entity_id=str(document.id),
            )
            db_session.commit()
            # Purge all remaining org embeddings
            doc_store = DocumentStore()
            for org_id in orgs_to_purge:
                try:
                    collection_name = doc_store.get_collection_name(org_id, db_session)
                    doc_store.delete_by_document_id(collection_name=collection_name, document_id=document_id)
                except Exception as e:
                    log.error("Vector purge failed for doc %d org %d: %s", document_id, org_id, e)
            raise HTTPException(
                status_code=status.HTTP_200_OK,
                detail="Document soft-deleted: removed from all organizations",
            )

    # --- Agent diff (decoupled set operations across document-agent pairs) ---
    current_org_ids = (existing_org_ids | set(orgs_to_embed)) - set(orgs_to_purge)

    if request.agent_ids is not None:
        existing_agent_links = db_session.exec(
            select(AgentDocument).where(AgentDocument.document_id == document_id)
        ).all()
        current_linked_agents = {al.agent_id for al in existing_agent_links}

        new_agent_id_set = set(request.agent_ids)
        valid_agent_set = set()
        if new_agent_id_set and current_org_ids:
            valid_agents = db_session.exec(
                select(AIAgent.id).where(
                    AIAgent.id.in_(list(new_agent_id_set)),
                    AIAgent.deleted == False,
                    AIAgent.organization_id.in_(list(current_org_ids)),
                )
            ).all()
            valid_agent_set = set(valid_agents)

        agents_to_remove = current_linked_agents - valid_agent_set
        agents_to_add = valid_agent_set - current_linked_agents

        for al in existing_agent_links:
            if al.agent_id in agents_to_remove:
                db_session.delete(al)

        for agent_id in agents_to_add:
            db_session.add(AgentDocument(
                document_id=document_id,
                agent_id=agent_id,
                user_id=user_id,
            ))
    elif orgs_to_purge:
        # If orgs were removed and agent_ids was not explicitly updated, remove any agent links for agents in purged orgs
        purged_agents = db_session.exec(
            select(AgentDocument)
            .join(AIAgent, AgentDocument.agent_id == AIAgent.id)
            .where(
                AgentDocument.document_id == document_id,
                AIAgent.organization_id.in_(orgs_to_purge),
            )
        ).all()
        for al in purged_agents:
            db_session.delete(al)

    document.updated_at = datetime.now(timezone.utc)
    db_session.add(document)
    log_activity(
        db_session=db_session,
        user_id=user_id,
        action=ActionType.UPDATE,
        entity_type=EntityType.DOCUMENT,
        entity_id=str(document.id),
    )
    db_session.commit()
    db_session.refresh(document)

    # --- Embed into newly added orgs ---
    if orgs_to_embed:
        s3_path = document.file_path
        file_content: Optional[bytes] = None
        try:
            file_content = await s3_client.read_object(s3_path)
        except Exception as e:
            log.error("Could not download file from S3 for re-embedding: %s", e)

        if file_content:
            for org_id in orgs_to_embed:
                resolved_model_id = get_org_embedding_model(org_id, db_session)

                background_tasks.add_task(
                    _embed_into_org,
                    org_id=org_id,
                    document_id=document_id,
                    file_name=document.file_name,
                    file_content=file_content,
                    full_s3_path=s3_path,
                    user_id=user_id,
                    embedding_model_id=resolved_model_id,
                    chunk_size=request.chunk_size,
                    chunk_overlap=request.chunk_overlap,
                )

    # --- Purge embeddings from removed orgs ---
    doc_store = DocumentStore()
    for org_id in orgs_to_purge:
        try:
            collection_name = doc_store.get_collection_name(org_id, db_session)
            doc_store.delete_by_document_id(collection_name=collection_name, document_id=document_id)
        except Exception as e:
            log.error("Vector purge failed for doc %d org %d: %s", document_id, org_id, e)

    return _enrich_doc_out(document, db_session)
