"""Document processing background tasks.

This module handles:
- Chunking and embedding documents
- Vector store insertion
- Progress tracking
"""
import asyncio
import logging
from typing import Optional
from datetime import datetime

from celery import shared_task
from sqlmodel import Session, select

from ai_agent.core.celery_config import celery_app
from ai_agent.core.db import engine
from ai_agent.core.rag_service import get_rag_service
from ai_agent.models import (
    Document, DocumentIngestionJob, DocumentIngestionStatus,
    AIAgent
)
from ai_agent.services.embedding import get_org_embedding_model
from ai_agent.core.aws_s3 import s3_client

log = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_document_ingestion(self, job_id: str):
    """Process a document ingestion job in the background.
    
    Args:
        job_id: The DocumentIngestionJob ID
    """
    log.info(f"[Job {job_id}] Starting document ingestion")
    
    with Session(engine) as db_session:
        # Get the job
        job = db_session.get(DocumentIngestionJob, job_id)
        if not job:
            log.error(f"[Job {job_id}] Job not found")
            return {"status": "error", "error": "Job not found"}
        
        # Update job status
        job.status = DocumentIngestionStatus.PROCESSING
        job.started_at = datetime.utcnow()
        job.celery_task_id = self.request.id
        db_session.commit()
        
        try:
            # Get document
            document = db_session.get(Document, job.document_id)
            if not document:
                raise ValueError(f"Document {job.document_id} not found")
            
            # Get the agent to resolve its organization
            agent = db_session.get(AIAgent, job.agent_id)
            if not agent:
                raise ValueError(f"Agent {job.agent_id} not found")

            org_embedding_model = get_org_embedding_model(agent.organization_id, db_session)
            if not org_embedding_model:
                raise ValueError(
                    f"No embedding model configured for organization {agent.organization_id}"
                )
            
            # Download file from S3
            log.info(f"[Job {job_id}] Downloading file from S3")
            
            # Parse S3 path
            s3_path = document.file_path
            if "/" in s3_path:
                # Format: bucket_name/key
                parts = s3_path.split("/", 1)
                if len(parts) == 2:
                    bucket, key = parts
                else:
                    key = s3_path
            else:
                key = s3_path
            
            # Download
            file_content = s3_client.download_file(key)
            
            if not file_content:
                raise ValueError("Failed to download file from storage")
            
            log.info(f"[Job {job_id}] Downloaded {len(file_content)} bytes")
            
            # Update progress
            job.progress_percent = 20
            db_session.commit()
            
            # Get RAG service
            rag_service = get_rag_service()
            
            # Process document
            log.info(f"[Job {job_id}] Processing document with RAG service")
            
            result = asyncio.run(
                rag_service.add_document(
                    document_id=job.document_id,
                    organization_id=agent.organization_id,
                    file_content=file_content,
                    file_name=document.file_name,
                    chunk_size=job.chunk_size,
                    chunk_overlap=job.chunk_overlap,
                    embedding_model_id=org_embedding_model,
                    metadata={
                        "job_id": job_id,
                        "user_id": job.user_id,
                        "mime_type": document.mime_type,
                    },
                )
            )
            
            if not result.get("success"):
                raise ValueError(result.get("error", "Unknown processing error"))
            
            # Update progress
            job.progress_percent = 100
            job.total_chunks = result.get("chunks_created", 0)
            job.processed_chunks = result.get("chunks_created", 0)
            job.status = DocumentIngestionStatus.INDEXED
            job.completed_at = datetime.utcnow()
            db_session.commit()
            
            log.info(f"[Job {job_id}] Completed with {job.total_chunks} chunks")
            
            return {
                "status": "success",
                "job_id": job_id,
                "chunks_created": job.total_chunks,
            }
            
        except Exception as exc:
            log.error(f"[Job {job_id}] Failed: {exc}")
            
            # Update job with error
            job.status = DocumentIngestionStatus.FAILED
            job.error_message = str(exc)
            job.error_details = {
                "exception_type": type(exc).__name__,
                "task_id": self.request.id,
            }
            db_session.commit()
            
            # Retry if appropriate
            if self.request.retries < self.max_retries:
                log.info(f"[Job {job_id}] Retrying ({self.request.retries + 1}/{self.max_retries})")
                raise self.retry(exc=exc)
            
            return {
                "status": "error",
                "job_id": job_id,
                "error": str(exc),
            }


@celery_app.task
def retry_failed_ingestion_jobs():
    """Retry failed ingestion jobs (scheduled task)."""
    with Session(engine) as db_session:
        failed_jobs = db_session.exec(
            select(DocumentIngestionJob).where(
                DocumentIngestionJob.status == DocumentIngestionStatus.FAILED
            )
        ).all()
        
        retried = 0
        for job in failed_jobs:
            # Only retry jobs less than 24 hours old
            if job.created_at > datetime.utcnow() - __import__('datetime').timedelta(hours=24):
                process_document_ingestion.delay(job.id)
                retried += 1
        
        log.info(f"Retried {retried} failed ingestion jobs")
        return {"retried": retried}


@celery_app.task
def cleanup_stale_processing_jobs():
    """Mark processing jobs as failed if they've been stuck too long."""
    with Session(engine) as db_session:
        stale_jobs = db_session.exec(
            select(DocumentIngestionJob).where(
                DocumentIngestionJob.status == DocumentIngestionStatus.PROCESSING
            ).where(
                DocumentIngestionJob.started_at < datetime.utcnow() - __import__('datetime').timedelta(hours=2)
            )
        ).all()
        
        for job in stale_jobs:
            job.status = DocumentIngestionStatus.FAILED
            job.error_message = "Job timed out after 2 hours"
            db_session.add(job)
        
        db_session.commit()
        
        log.info(f"Marked {len(stale_jobs)} stale jobs as failed")
        return {"cleaned": len(stale_jobs)}
