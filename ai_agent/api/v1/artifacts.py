"""Artifacts API for generated file outputs.

This module provides endpoints for:
- Generating artifacts (PDF, CSV, XLSX, JSON, etc.)
- Listing artifacts for a run/conversation
- Downloading artifacts with presigned URLs
- Managing artifact lifecycle
"""
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from ai_agent.core import db
from ai_agent.core.aws_s3 import s3_client
from ai_agent.core.response_schemas import ArtifactSummary
from ai_agent.models import Artifact, AgentRun, User, AIAgent, UserSession
from ai_agent.services.artifact_service import (
    SUPPORTED_ARTIFACT_TYPES,
    create_artifact,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/artifacts", tags=["Artifacts"])


class GenerateArtifactRequest(BaseModel):
    """Request to generate an artifact."""
    artifact_type: str = Field(description="Type: pdf/csv/xlsx/json/image/audio/etc")
    data: dict = Field(description="Data to convert to artifact")
    filename: Optional[str] = Field(default=None, description="Desired filename")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata")


class ArtifactResponse(ArtifactSummary):
    """Artifact response with download URL."""
    download_url: Optional[str] = Field(default=None, description="Presigned download URL")
    created_at: datetime = Field(description="Creation timestamp")


class ArtifactListResponse(BaseModel):
    """List of artifacts."""
    artifacts: List[ArtifactResponse]
    total: int = Field(description="Total count")
    has_more: bool = Field(default=False)


@router.post("/generate/{run_id}")
async def generate_artifact(
    run_id: str,
    request: GenerateArtifactRequest,
    user_id: int = Query(..., description="User ID"),
    db_session: Session = Depends(db.get_session),
):
    """Generate an artifact from data and store it.

    This endpoint is typically called by the UI / external clients. The agent
    runtime uses the same underlying ``artifact_service.create_artifact`` so
    behavior stays in sync between programmatic and agentic generation.
    """
    # Verify run exists and belongs to user before any work.
    run = db_session.get(AgentRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.user_id != user_id:
        raise HTTPException(status_code=403, detail="Run does not belong to user")

    try:
        summary = await create_artifact(
            db_session,
            run_id=run_id,
            user_id=user_id,
            agent_id=run.agent_id,
            session_id=run.session_id,
            conversation_id=run.conversation_id,
            artifact_type=request.artifact_type,
            data=request.data,
            filename=request.filename,
            metadata=request.metadata,
            description=(request.metadata or {}).get("description") if request.metadata else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Failed to generate artifact")
        raise HTTPException(status_code=500, detail=f"Failed to generate artifact: {e}")

    artifact = db_session.get(Artifact, summary.id)
    return ArtifactResponse(
        id=summary.id,
        artifact_type=summary.artifact_type,
        mime_type=summary.mime_type,
        filename=summary.filename,
        size_bytes=summary.size_bytes,
        description=summary.description,
        download_url=summary.download_url,
        created_at=artifact.created_at if artifact else datetime.utcnow(),
    )


@router.get("/run/{run_id}")
async def list_run_artifacts(
    run_id: str,
    user_id: int = Query(...),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db_session: Session = Depends(db.get_session),
):
    """List all artifacts for a run."""
    # Verify run belongs to user
    run = db_session.get(AgentRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.user_id != user_id:
        raise HTTPException(status_code=403, detail="Run does not belong to user")
    
    # Query artifacts
    query = select(Artifact).where(
        Artifact.run_id == run_id
    ).where(
        Artifact.user_id == user_id
    ).offset(offset).limit(limit + 1)
    
    artifacts = db_session.exec(query).all()
    has_more = len(artifacts) > limit
    artifacts = artifacts[:limit]
    
    # Build responses with presigned URLs
    artifact_responses = []
    for artifact in artifacts:
        download_url = None
        if artifact.storage_key:
            download_url = s3_client.generate_presigned_url(
                object_name=artifact.storage_key,
                expiration=3600,
            )
        artifact_responses.append(ArtifactResponse(
            id=artifact.id,
            artifact_type=artifact.artifact_type,
            mime_type=artifact.mime_type,
            filename=artifact.filename,
            size_bytes=artifact.size_bytes,
            description=artifact.extra_metadata.get("description") if artifact.extra_metadata else None,
            download_url=download_url,
            created_at=artifact.created_at,
        ))
    
    return ArtifactListResponse(
        artifacts=artifact_responses,
        total=len(artifact_responses),
        has_more=has_more,
    )


@router.get("/conversation/{conversation_id}")
async def list_conversation_artifacts(
    conversation_id: str,
    user_id: int = Query(...),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db_session: Session = Depends(db.get_session),
):
    """List all artifacts for a conversation."""
    # Query artifacts
    query = select(Artifact).where(
        Artifact.conversation_id == conversation_id
    ).where(
        Artifact.user_id == user_id
    ).offset(offset).limit(limit + 1)
    
    artifacts = db_session.exec(query).all()
    has_more = len(artifacts) > limit
    artifacts = artifacts[:limit]
    
    # Build responses
    artifact_responses = []
    for artifact in artifacts:
        download_url = None
        if artifact.storage_key:
            download_url = s3_client.generate_presigned_url(
                object_name=artifact.storage_key,
                expiration=3600,
            )
        artifact_responses.append(ArtifactResponse(
            id=artifact.id,
            artifact_type=artifact.artifact_type,
            mime_type=artifact.mime_type,
            filename=artifact.filename,
            size_bytes=artifact.size_bytes,
            description=artifact.extra_metadata.get("description") if artifact.extra_metadata else None,
            download_url=download_url,
            created_at=artifact.created_at,
        ))
    
    return ArtifactListResponse(
        artifacts=artifact_responses,
        total=len(artifact_responses),
        has_more=has_more,
    )


@router.get("/{artifact_id}")
async def get_artifact(
    artifact_id: str = Path(..., description="Artifact ID"),
    user_id: int = Query(...),
    expiry_seconds: int = Query(3600, ge=60, le=604800),
    db_session: Session = Depends(db.get_session),
):
    """Get artifact details and download URL."""
    artifact = db_session.get(Artifact, artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if artifact.user_id != user_id:
        raise HTTPException(status_code=403, detail="Artifact does not belong to user")
    
    # Check if expired
    if artifact.expires_at and artifact.expires_at < datetime.utcnow():
        raise HTTPException(status_code=410, detail="Artifact has expired")
    
    download_url = None
    if artifact.storage_key:
        download_url = s3_client.generate_presigned_url(
            object_name=artifact.storage_key,
            expiration=expiry_seconds,
        )
    
    return ArtifactResponse(
        id=artifact.id,
        artifact_type=artifact.artifact_type,
        mime_type=artifact.mime_type,
        filename=artifact.filename,
        size_bytes=artifact.size_bytes,
        description=artifact.extra_metadata.get("description") if artifact.extra_metadata else None,
        download_url=download_url,
        created_at=artifact.created_at,
    )


@router.get("/{artifact_id}/download")
async def download_artifact(
    artifact_id: str = Path(..., description="Artifact ID"),
    user_id: int = Query(...),
    db_session: Session = Depends(db.get_session),
):
    """Generate a presigned S3 URL and redirect the browser to download the file."""
    from fastapi.responses import RedirectResponse

    artifact = db_session.get(Artifact, artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if artifact.user_id != user_id:
        raise HTTPException(status_code=403, detail="Artifact does not belong to user")
    if artifact.expires_at and artifact.expires_at < datetime.utcnow():
        raise HTTPException(status_code=410, detail="Artifact has expired")

    storage_key = artifact.storage_key or ""

    # Generate presigned URL and redirect browser to S3
    try:
        presigned = s3_client.generate_presigned_url(
            object_name=storage_key,
            expiration=300,
        )
    except Exception as e:
        log.exception("Failed to generate presigned URL for artifact %s", artifact_id)
        raise HTTPException(status_code=500, detail="Could not generate download URL")

    if not presigned:
        raise HTTPException(status_code=404, detail="File not accessible")

    return RedirectResponse(url=presigned)


@router.delete("/{artifact_id}")
async def delete_artifact(
    artifact_id: str = Path(..., description="Artifact ID"),
    user_id: int = Query(...),
    db_session: Session = Depends(db.get_session),
):
    """Delete an artifact (soft delete)."""
    artifact = db_session.get(Artifact, artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if artifact.user_id != user_id:
        raise HTTPException(status_code=403, detail="Artifact does not belong to user")
    
    # Set expiration to now (soft delete)
    artifact.expires_at = datetime.utcnow()
    db_session.commit()
    
    return {"status": "deleted", "artifact_id": artifact_id}


@router.get("/agent/{agent_id}/stats")
async def get_agent_artifact_stats(
    agent_id: str = Path(...),
    user_id: int = Query(...),
    db_session: Session = Depends(db.get_session),
):
    """Get artifact statistics for an agent."""
    # Verify agent belongs to user
    agent = db_session.get(AIAgent, agent_id)
    if not agent or agent.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Agent not found or access denied")
    
    # Count artifacts by type
    from sqlalchemy import func
    
    type_counts = db_session.exec(
        select(Artifact.artifact_type, func.count(Artifact.id))
        .where(Artifact.agent_id == agent_id)
        .where(Artifact.user_id == user_id)
        .group_by(Artifact.artifact_type)
    ).all()
    
    # Total size
    total_size = db_session.exec(
        select(func.sum(Artifact.size_bytes))
        .where(Artifact.agent_id == agent_id)
        .where(Artifact.user_id == user_id)
    ).first() or 0
    
    return {
        "agent_id": agent_id,
        "total_artifacts": sum(c[1] for c in type_counts),
        "by_type": {t: c for t, c in type_counts},
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }
