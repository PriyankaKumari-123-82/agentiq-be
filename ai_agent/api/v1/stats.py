from fastapi import APIRouter, Depends, HTTPException, status, Path, Query
from sqlmodel import Session, select, func, and_
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta, date
from pydantic import BaseModel, Field, field_validator

from ai_agent.core import db
from ai_agent.core.token_validation import decode_token_payload, security
from ai_agent.models import User, Document, AIAgent, LLMUsageEvent, AuditLog, UserSession, SessionStatus


class StatusResponse(BaseModel):
    """Response model for status information."""
    code: int = Field(..., description="Status code (200 for success, 4xx/5xx for errors)")
    message: str = Field(..., description="Human-readable status message")


class StatsData(BaseModel):
    """Response model for statistics data."""
    total_conversations: int = Field(..., description="Total number of conversations")
    total_documents: int = Field(..., description="Total number of documents indexed")
    active_sessions: int = Field(..., description="Number of active user sessions")
    total_agents: int = Field(..., description="Total number of AI agents")
    timestamp: str = Field(..., description="ISO format timestamp of the response")


class ActivityItem(BaseModel):
    """Model for a single activity item."""
    id: str = Field(..., description="Unique identifier for the activity")
    type: str = Field(..., description="Type of activity")
    timestamp: datetime = Field(..., description="When the activity occurred")
    entity_id: Optional[str] = Field(None, description="Row ID of the affected entity")
    title: Optional[str] = Field(None, description="Activity title")
    subtitle: Optional[str] = Field(None, description="Entity or change summary")
    user_id: int = Field(..., description="ID of the user who performed the activity")
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    deleted_at: Optional[datetime] = None
    message_content: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    settings_updated: Optional[Dict[str, Any]] = None

    @field_validator("timestamp", "deleted_at", mode="before")
    @classmethod
    def normalize_activity_timestamp(cls, value: datetime | None) -> datetime | None:
        """Treat naive database timestamps as UTC for unambiguous API output."""
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class RecentActivitiesResponse(BaseModel):
    """Response model for recent activities."""
    status: StatusResponse = Field(..., description="Status information")
    activities: List[ActivityItem] = Field(..., description="List of recent activities")

class StatsResponse(BaseModel):
    """Response model for the statistics endpoint."""
    status: StatusResponse = Field(..., description="Status information")
    data: StatsData = Field(..., description="Statistics data")

router = APIRouter(tags=["Stats"])

@router.get("/stats/token-usage")
async def get_token_usage(
    user_id: int = Query(..., description="ID of the user to get token usage for"),
    start_date: date = Query(..., description="Start date (YYYY-MM-DD), inclusive"),
    end_date: date = Query(..., description="End date (YYYY-MM-DD), inclusive"),
    top_n: int = Query(6, ge=1, le=20, description="Max agents to include"),
    db_session: Session = Depends(db.get_session),
) -> Dict[str, Any]:
    user = db_session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User with ID {user_id} not found")

    days = (end_date - start_date).days + 1
    labels_dates = [start_date + timedelta(days=i) for i in range(days)]
    labels = [d.strftime('%d %b') for d in labels_dates]
    date_keys = [d.strftime('%Y-%m-%d') for d in labels_dates]  # compute once

    start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)

    # Step 1: rank agents by total tokens IN SQL, only pull top_n
    totals_query = (
        select(
            LLMUsageEvent.agent_id,
            AIAgent.name,
            func.sum(LLMUsageEvent.total_tokens).label('total'),
        )
        .join(AIAgent, AIAgent.id == LLMUsageEvent.agent_id)
        .where(
            LLMUsageEvent.user_id == user_id,
            LLMUsageEvent.created_at >= start_dt,
            LLMUsageEvent.created_at <= end_dt,
            AIAgent.owner_id == user_id,
            AIAgent.deleted == False,
        )
        .group_by(LLMUsageEvent.agent_id, AIAgent.name)
        .order_by(func.sum(LLMUsageEvent.total_tokens).desc())
        .limit(top_n)
    )
    top_rows = db_session.exec(totals_query).all()
    if not top_rows:
        return {"labels": labels, "series": [], "top_agents": []}

    top_agent_ids = [r[0] for r in top_rows]
    agent_name_map = {r[0]: (r[1] or r[0]) for r in top_rows}
    top_agents = [
        {"agent_id": aid, "display_name": agent_name_map[aid], "total_tokens": int(total or 0)}
        for aid, _, total in top_rows
    ]

    # Step 2: daily breakdown ONLY for the top agents
    day_expr = func.date(LLMUsageEvent.created_at)
    daily_query = (
        select(LLMUsageEvent.agent_id, day_expr.label('day'), func.sum(LLMUsageEvent.total_tokens).label('tokens'))
        .where(
            LLMUsageEvent.user_id == user_id,
            LLMUsageEvent.agent_id.in_(top_agent_ids),
            LLMUsageEvent.created_at >= start_dt,
            LLMUsageEvent.created_at <= end_dt,
        )
        .group_by(LLMUsageEvent.agent_id, day_expr)
    )
    daily_rows = db_session.exec(daily_query).all()

    data_by_agent: Dict[str, Dict[str, int]] = {}
    for agent_id, day, tokens in daily_rows:
        day_key = day if isinstance(day, str) else day.strftime('%Y-%m-%d')
        data_by_agent.setdefault(agent_id, {})[day_key] = int(tokens or 0)

    series = [
        {
            "agent_id": aid,
            "display_name": agent_name_map[aid],
            "data": [data_by_agent.get(aid, {}).get(k, 0) for k in date_keys],
        }
        for aid in top_agent_ids
    ]

    return {"labels": labels, "series": series, "top_agents": top_agents}

@router.get("/stats/{user_id}", response_model=StatsResponse)
async def get_application_stats(
    user_id: int = Path(..., description="ID of the user to get statistics for"),
    db_session: Session = Depends(db.get_session)
) -> StatsResponse:
    """
    Get application status and statistics.
    
    Returns:
        Dict containing:
        - total_conversations: Total number of conversations
        - total_documents: Total number of documents indexed
        - active_sessions: Number of active user sessions
        - total_agents: Total number of AI agents
    """
    user = db_session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get total conversations for the user with non-deleted agents
    total_conversations = db_session.exec(
        select(func.count())
        .select_from(UserSession)
        .join(AIAgent, UserSession.agent_id == AIAgent.id)
        .where(and_(
            UserSession.user_id == user_id,
            AIAgent.deleted == False
        ))
    ).one_or_none() or 0
    
    # Get total non-deleted documents for the user
    total_documents = db_session.exec(
        select(func.count())
        .where(and_(
            Document.user_id == user_id,
            Document.deleted_at.is_(None)  # Only count non-deleted documents
        ))
    ).one_or_none() or 0
    
    # Get active sessions for the user (excluding deleted agents and soft-deleted sessions)
    active_sessions = db_session.exec(
        select(func.count())
        .select_from(UserSession)
        .join(AIAgent, UserSession.agent_id == AIAgent.id)
        .where(and_(
            UserSession.user_id == user_id,
            UserSession.status == SessionStatus.ACTIVE,
            UserSession.is_active == True,
            AIAgent.deleted == False
        ))
    ).one_or_none() or 0
    
    # Get total non-deleted agents owned by the user
    total_agents = db_session.exec(
        select(func.count())
        .where(and_(
            AIAgent.owner_id == user_id,
            AIAgent.deleted == False
        ))
    ).one_or_none() or 0
    
    stats_data = StatsData(
        total_conversations=total_conversations,
        total_documents=total_documents,
        active_sessions=active_sessions,
        total_agents=total_agents,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    return StatsResponse(
        status=StatusResponse(code=200, message="Statistics retrieved successfully"),
        data=stats_data
    )

@router.get(
    "/recent-activities/{user_id}",
    response_model=RecentActivitiesResponse,
    dependencies=[Depends(security)],
)
async def get_recent_activities(
    user_id: int = Path(..., description="ID of the user to get activities for"),
    limit: int = 10,
    token_payload: dict = Depends(decode_token_payload),
    db_session: Session = Depends(db.get_session)
) -> RecentActivitiesResponse:
    """
    Get recent activities for a user from the activity log table.
    
    Returns:
        List of recent activities with details including timestamps and relevant metadata.
    """
    # Resolve the validated token to a local user before looking up the
    # requested user ID. This prevents user-ID enumeration and cross-user
    # access to activity history.
    token_email = token_payload.get("email", "").strip().lower()
    authenticated_user = db_session.exec(
        select(User).where(func.lower(User.email) == token_email)
    ).first()
    if not authenticated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Authenticated user not found in local database.",
        )
    if authenticated_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own recent activities.",
        )

    # Mutations come from the explicit audit log. Chat messages continue to
    # come from their source table to preserve the existing API contract.
    activities = db_session.exec(
        select(AuditLog, User)
        .join(User, AuditLog.actor_user_id == User.id)
        .where(AuditLog.actor_user_id == user_id)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
    ).all()

    activity_items: List[ActivityItem] = []
    for activity, user in activities:
        # Determine activity type based on action and entity type
        entity_type = activity.resource_type
        action = {
            "create": "created",
            "update": "updated",
            "delete": "deleted",
        }.get(activity.action, activity.action)
        activity_type = f"{entity_type}_{action}"
        if entity_type == "document" and action == "created":
            activity_type = "document_upload"

        title = {
            "agent_created": "Agent created",
            "agent_updated": "Agent updated",
            "agent_deleted": "Agent deleted",
            "document_upload": "Document uploaded",
            "document_deleted": "Document deleted",
            "settings_updated": "Settings updated",
            "api_key_created": "API key added",
            "api_key_updated": "API key updated",
            "api_key_deleted": "API key deleted",
            "user_updated": "Profile updated",
        }.get(activity_type, f"{entity_type.replace('_', ' ').title()} {action}")
        metadata = activity.extra_metadata or {}
        subtitle = metadata.get("subtitle")
        fields = metadata.get("fields")
        if fields:
            subtitle = ", ".join(str(field).replace("_", " ").title() for field in fields)
        if entity_type == "api_key" and metadata.get("provider"):
            subtitle = str(metadata["provider"]).title()

        item = ActivityItem(
            id=str(activity.id),
            type=activity_type,
            timestamp=activity.created_at,
            entity_id=activity.resource_id,
            title=title,
            subtitle=subtitle,
            user_id=user_id,
            agent_id=activity.resource_id if entity_type == "agent" else None,
            deleted_at=activity.created_at if action == "deleted" else None,
        )

        if entity_type == "agent" and activity.resource_id:
            agent = db_session.get(AIAgent, activity.resource_id)
            if agent:
                item.agent_name = agent.name
                item.subtitle = agent.name
        elif entity_type == "document" and activity.resource_id:
            document = db_session.get(Document, int(activity.resource_id))
            if document:
                item.file_name = document.file_name
                item.file_size = document.file_size
                item.mime_type = document.mime_type
                item.subtitle = document.file_name


        activity_items.append(item)

    # PostgreSQL TIMESTAMP WITHOUT TIME ZONE columns return naive datetimes —
    # strip tzinfo upfront so the same value works for both SQL filters and
    # Python-level comparisons without a TypeError.
    cutoff_time = (datetime.now(timezone.utc) - timedelta(days=7)).replace(tzinfo=None)

    # Keep pre-audit history visible. Explicit audit rows win when an entity
    # already has an event of the same kind.
    logged_keys = {
        (item.entity_id, item.type)
        for item in activity_items
        if item.entity_id is not None
    }
    documents = db_session.exec(
        select(Document).where(
            Document.user_id == user_id,
            (Document.created_at >= cutoff_time) | (Document.deleted_at >= cutoff_time),
        )
    ).all()
    for document in documents:
        document_id = str(document.id)
        created_key = (document_id, "document_upload")
        if document.created_at >= cutoff_time and created_key not in logged_keys:
            activity_items.append(ActivityItem(
                id=f"doc_{document.id}",
                type="document_upload",
                title="Document uploaded",
                subtitle=document.file_name,
                timestamp=document.created_at,
                entity_id=document_id,
                user_id=user_id,
                file_name=document.file_name,
                file_size=document.file_size,
                mime_type=document.mime_type,
            ))
        deleted_key = (document_id, "document_deleted")
        if document.deleted_at and document.deleted_at >= cutoff_time and deleted_key not in logged_keys:
            activity_items.append(ActivityItem(
                id=f"doc_del_{document.id}",
                type="document_deleted",
                title="Document deleted",
                subtitle=document.file_name,
                timestamp=document.deleted_at,
                entity_id=document_id,
                user_id=user_id,
                file_name=document.file_name,
                deleted_at=document.deleted_at,
            ))

    agents = db_session.exec(
        select(AIAgent).where(
            AIAgent.owner_id == user_id,
            (AIAgent.created_at >= cutoff_time) | (AIAgent.updated_at >= cutoff_time),
        )
    ).all()
    for agent in agents:
        agent_id = str(agent.id)
        if (agent_id, "agent_created") not in logged_keys:
            activity_items.append(ActivityItem(
                id=f"agent_{agent.id}", type="agent_created",
                title="Agent created", subtitle=agent.name, timestamp=agent.created_at,
                entity_id=agent_id, user_id=user_id,
                agent_id=agent_id, agent_name=agent.name,
            ))
        if agent.created_at >= cutoff_time and agent.updated_at > agent.created_at + timedelta(seconds=1):
            event_type = "agent_deleted" if agent.deleted else "agent_updated"
            if (agent_id, event_type) not in logged_keys:
                activity_items.append(ActivityItem(
                    id=f"agent_{event_type}_{agent.id}", type=event_type,
                    title="Agent deleted" if agent.deleted else "Agent updated",
                    subtitle=agent.name,
                    timestamp=agent.updated_at, entity_id=agent_id, user_id=user_id,
                    agent_id=agent_id, agent_name=agent.name,
                    deleted_at=agent.updated_at if agent.deleted else None,
                ))


    activity_items.sort(key=lambda item: item.timestamp, reverse=True)

    return RecentActivitiesResponse(
        status=StatusResponse(code=200, message="Recent activities retrieved successfully"),
        activities=activity_items[:limit]
    )


@router.get(
    "/recent-activities",
    response_model=RecentActivitiesResponse,
    dependencies=[Depends(security)],
    responses={
        401: {"description": "Missing or invalid bearer token"},
        404: {"description": "Authenticated user not found locally"},
    },
)
async def get_my_recent_activities(
    limit: int = Query(10, ge=1, le=100),
    token_payload: dict = Depends(decode_token_payload),
    db_session: Session = Depends(db.get_session),
) -> RecentActivitiesResponse:
    """Return recent activities for the authenticated local user."""
    token_email = token_payload.get("email", "").strip().lower()
    authenticated_user = db_session.exec(
        select(User).where(func.lower(User.email) == token_email)
    ).first()
    if not authenticated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Authenticated user not found in local database.",
        )

    result = await get_recent_activities(
        user_id=authenticated_user.id,
        limit=limit,
        token_payload=token_payload,
        db_session=db_session,
    )
    return result