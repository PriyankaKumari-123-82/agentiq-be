"""Cleanup and maintenance tasks.

This module handles:
- Expired artifact cleanup
- Old run data archival
- Vector store maintenance
"""
import logging
from datetime import datetime, timedelta

from celery import shared_task
from sqlmodel import Session, select, delete

from ai_agent.core.celery_config import celery_app
from ai_agent.core.db import engine
from ai_agent.models import Artifact, AgentRun, LLMUsageEvent
from ai_agent.core.aws_s3 import s3_client

log = logging.getLogger(__name__)


@celery_app.task
def cleanup_expired_artifacts():
    """Delete expired artifacts from storage and database."""
    with Session(engine) as db_session:
        # Find expired artifacts
        expired = db_session.exec(
            select(Artifact).where(
                Artifact.expires_at < datetime.utcnow()
            )
        ).all()
        
        deleted_count = 0
        failed_count = 0
        
        for artifact in expired:
            try:
                # Delete from S3
                if artifact.storage_key:
                    s3_client.delete_object(artifact.storage_key)
                
                # Delete from database
                db_session.delete(artifact)
                deleted_count += 1
                
            except Exception as e:
                log.error(f"Failed to delete artifact {artifact.id}: {e}")
                failed_count += 1
        
        db_session.commit()
        
        log.info(f"Cleaned up {deleted_count} expired artifacts, {failed_count} failed")
        return {
            "deleted": deleted_count,
            "failed": failed_count,
        }


@celery_app.task
def cleanup_old_runs(older_than_days: int = 90):
    """Archive or delete old agent runs."""
    with Session(engine) as db_session:
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        
        # Find old completed runs
        old_runs = db_session.exec(
            select(AgentRun).where(
                AgentRun.completed_at < cutoff
            ).where(
                AgentRun.status.in_(["completed", "failed", "cancelled"])
            ).limit(1000)  # Process in batches
        ).all()
        
        # For now, just log - in production, you might archive these
        log.info(f"Found {len(old_runs)} runs older than {older_than_days} days")
        
        # Optionally: archive to cold storage or delete
        # For now, we keep them for audit purposes
        
        return {"found": len(old_runs)}


@celery_app.task
def vacuum_vector_store():
    """Perform maintenance on vector stores."""
    # This would trigger Milvus compaction or similar maintenance
    log.info("Vector store vacuum task - placeholder")
    return {"status": "placeholder"}
