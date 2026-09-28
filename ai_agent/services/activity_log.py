"""Activity logging service for tracking user actions."""

from datetime import datetime, timezone
from sqlmodel import Session
from typing import Any, Dict, Optional
from uuid import uuid4

from ai_agent.models import AuditLog
from ai_agent.core.enums import ActionType, EntityType


def log_activity(
    db_session: Session,
    user_id: int,
    action: ActionType,
    entity_type: EntityType,
    entity_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AuditLog:
    """
    Log a user activity to the activity log table.

    Args:
        db_session: Database session
        user_id: ID of the user performing the action
        action: Action performed (create | update | delete)
        entity_type: Type of entity (doc | agent | api_key | user | settings | llm_settings | vector_db_settings)
        entity_id: Row ID of the affected entity (optional)
        metadata: Additional display or change context (optional)

    Returns:
        The created audit-log record
    """
    activity = AuditLog(
        id=str(uuid4()),
        actor_user_id=user_id,
        action=action,
        resource_type=entity_type,
        resource_id=entity_id,
        extra_metadata=metadata,
        created_at=datetime.now(timezone.utc),
    )
    # The caller owns the transaction. This keeps the business mutation and
    # its audit event atomic instead of committing them independently.
    db_session.add(activity)
    return activity
