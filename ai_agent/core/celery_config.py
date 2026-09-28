"""Celery configuration for background task processing.

This module configures Celery for:
- Document ingestion jobs
- Batch processing
- Cleanup tasks
"""
import os
from celery import Celery
from ai_agent.core.config import settings

# Create Celery app
celery_app = Celery(
    "ai_agent",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    include=["ai_agent.tasks.document_tasks"],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    worker_prefetch_multiplier=1,  # Process one task at a time per worker
    broker_connection_retry_on_startup=True,
)

# Queue configuration
celery_app.conf.task_routes = {
    "ai_agent.tasks.document_tasks.*": {"queue": "documents"},
    "ai_agent.tasks.cleanup.*": {"queue": "cleanup"},
}

# Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    "cleanup-expired-artifacts": {
        "task": "ai_agent.tasks.cleanup.cleanup_expired_artifacts",
        "schedule": 86400.0,  # Daily
    },
    "cleanup-old-runs": {
        "task": "ai_agent.tasks.cleanup.cleanup_old_runs",
        "schedule": 86400.0,  # Daily
    },
}


def get_celery_app():
    """Get the Celery app instance."""
    return celery_app
