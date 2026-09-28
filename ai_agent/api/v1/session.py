from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, File, UploadFile, Query
from sqlmodel import Session, select
from sqlalchemy.sql import func
from sqlalchemy import or_
from sqlalchemy.orm import aliased
import os
import io
from ai_agent.core.aws_s3 import s3_client
from ai_agent.services.embedding import get_org_embedding_model
from ai_agent.api.v1.document import _embed_into_org

from ai_agent.core import db
from ai_agent.core.config import settings
from ai_agent.core.token_validation import decode_token_payload, security
from ai_agent.core.enums import ActionType, EntityType
from ai_agent.models import User, UserSession, SessionStatus, Message, ConversationRole, MessageFeedback, AIAgent, Artifact, Document, SessionDocument
from ai_agent.services.activity_log import log_activity
from ai_agent.services.artifact_service import artifact_payload
from pydantic import BaseModel, Field
from uuid import uuid4
import base64

router = APIRouter(tags=["Session"])

class SessionCreate(BaseModel):
    user_id: int = Field(..., gt=0, description="Unique identifier for the user")
    agent_id: str = Field(..., description="ID of the AI agent to associate with this session")

class SessionOut(BaseModel):
    id: int
    user_id: int
    agent_id: str = Field(..., description="ID of the AI agent associated with this session")
    session_id: str
    title: Optional[str] = None
    status: str
    is_archived: bool = False
    created_at: datetime            
    updated_at: datetime
    conversation_id: str
    
class SessionListResponse(BaseModel):
    sessions: List[SessionOut]

class SessionResponse(BaseModel):
    session: Dict[str, Any]
    conversations: List[Dict[str, Any]]
    hasMore: bool
    nextCursor: str | None

class SessionDocumentUploadResponse(BaseModel):
    id: int
    file_name: str
    file_size: int
    session_id: str
    extracted: bool

@router.post(
    "/session",
    response_model=SessionOut,
    status_code=status.HTTP_201_CREATED,
    tags=["Session"],
    dependencies=[Depends(security)],
)
def create_session(
    session_data: SessionCreate,
    payload: dict = Depends(decode_token_payload),
    db_session: Session = Depends(db.get_session)
):
    """Create a new session for a user with a required AI agent"""
    email = (payload.get("email") or "").strip().lower()
    user = db_session.exec(
        select(User).where(func.lower(User.email) == email)
    ).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )
    if user.id != session_data.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only create your own sessions",
        )
        
    # Verify agent exists and is valid
    if not session_data.agent_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent ID is required"
        )
        
    agent = db_session.get(AIAgent, session_data.agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with ID '{session_data.agent_id}' not found"
        )
    
    session_id = str(uuid4())
    conversation_id = str(uuid4())
    
    session = UserSession(
        user_id=session_data.user_id,
        session_id=session_id,
        title=None,
        status=SessionStatus.ACTIVE,
        agent_id=session_data.agent_id
    )
    
    db_session.add(session)
    db_session.flush()
    
    message = Message(
        conversation_id=conversation_id,
        session_id=session.id,
        agent_id=session_data.agent_id,
        role=ConversationRole.USER,
        content="",
        counter=0,
        prompt=None
    )
    
    db_session.add(message)
    log_activity(
        db_session=db_session,
        user_id=session.user_id,
        action=ActionType.START,
        entity_type=EntityType.CONVERSATION,
        entity_id=session.session_id,
        metadata={"subtitle": agent.name, "agent_id": agent.id},
    )
    db_session.commit()
    db_session.refresh(session)
    
    response_data = session.model_dump()
    response_data["conversation_id"] = conversation_id
    response_data["agent_id"] = session_data.agent_id
    return response_data

@router.get("/user/{user_id}/sessions", response_model=SessionListResponse, tags=["Session"], dependencies=[Depends(security)])
def list_user_sessions(
    user_id: str,
    agent_id: Optional[str] = None,
    user_status: Optional[SessionStatus] = None,
    keyword: Optional[str] = None,
    include_archived: bool = False,
    payload: dict = Depends(decode_token_payload),
    db_session: Session = Depends(db.get_session),
):
    """List all sessions for a user, optionally filtered by agent, excluding deleted agents.

    Args:
        user_id: ID of the user whose sessions to retrieve (must match authenticated user)
        agent_id: Optional ID of the agent to filter sessions by
        user_status: Optional status to filter sessions by
        keyword: Optional keyword to search in session titles
        include_archived: If true, return only archived sessions; otherwise return only active sessions (default: false)
    """
    
    # Get authenticated user's email and ID
    email = (payload.get("email") or "").strip().lower()
    authenticated_user = db_session.exec(select(User).where(User.email == email)).first()
    if not authenticated_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Ensure user can only access their own sessions
    if str(authenticated_user.id) != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own sessions"
        )
    
    # Query to get sessions with agent join to filter out deleted agents and soft-deleted sessions
    query = (
        select(UserSession)
        .join(UserSession.agent)
        .where(
            (UserSession.user_id == user_id) &
            (AIAgent.deleted == False) &  # Exclude deleted agents
            (UserSession.is_active == True)  # Exclude soft-deleted sessions
        )
    )
    
    # Filter archived sessions based on include_archived flag
    if include_archived:
        query = query.where(or_(UserSession.status == SessionStatus.ARCHIVED, UserSession.is_archived == True))
    else:
        query = query.where((UserSession.status != SessionStatus.ARCHIVED) & (UserSession.is_archived == False))
    
    # Add agent_id filter if provided
    if agent_id:
        query = query.where(UserSession.agent_id == agent_id)

    if user_status:
        query = query.where(UserSession.status == user_status)

    if keyword:
        msg_alias = aliased(Message)
        query = (
            query.outerjoin(msg_alias, UserSession.id == msg_alias.session_id)
            .where(
                or_(
                    UserSession.title.ilike(f'%{keyword}%'),
                    msg_alias.content.ilike(f'%{keyword}%')
                )
            )
            .distinct()
        )
    
    sessions = db_session.exec(query.order_by(UserSession.updated_at.desc())).all()
    session_ids = [session.id for session in sessions]
    
    if session_ids:
        latest_conversations_subq = (
            select(
                Message.session_id,
                func.max(Message.id).label('max_id')
            )
            .where(Message.session_id.in_(session_ids))
            .group_by(Message.session_id)
            .subquery()
        )
        
        latest_messages = db_session.exec(
            select(
                latest_conversations_subq.c.session_id,
                Message.conversation_id
            )
            .join(
                Message,
                latest_conversations_subq.c.max_id == Message.id,
                isouter=True
            )
        ).all()
        
        session_to_conversation = {sid: cid for sid, cid in latest_messages}
    else:
        session_to_conversation = {}
    
    response_sessions = []
    for session in sessions:
        # Skip sessions without an agent_id
        if not session.agent_id:
            continue
            
        session_dict = session.model_dump()
        session_dict["conversation_id"] = session_to_conversation.get(session.id, "")
        # Include agent_id as None if not available
        session_dict["agent_id"] = getattr(session, 'agent_id', None)
        response_sessions.append(session_dict)
    
    return {"sessions": response_sessions}

@router.get("/session/{session_id}", response_model=SessionResponse, tags=["Session"])
def get_session(
    session_id: str,
    cursor: str | None = None,
    limit: int = settings.DEFAULT_PAGE_SIZE,
    db_session: Session = Depends(db.get_session)
):
    """Get session details by session_id with paginated conversations and their feedback"""
    session = db_session.exec(
        select(UserSession)
        .where(
            (UserSession.session_id == session_id) &
            (UserSession.is_active == True)  # Treat soft-deleted sessions as not found
        )
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    query = select(Message).where(Message.session_id == session.id)
    
    if cursor:
        try:
            cursor_str = base64.urlsafe_b64decode(cursor).decode()
            counter_str, _ = cursor_str.split('|', 1)
            counter = int(counter_str)
            query = query.where(Message.counter < counter)  
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid cursor"
            )
    
    query = query.order_by(Message.counter.desc()).limit(limit + 1)
    messages = db_session.exec(query).all()
    
    has_more = len(messages) > limit
    if has_more:
        messages = messages[:-1]
    
    next_cursor = None
    if has_more and messages:
        last_message = messages[-1]
        cursor_str = f"{last_message.counter}|{last_message.id}"
        next_cursor = base64.urlsafe_b64encode(cursor_str.encode()).decode()
    
    message_ids = [msg.id for msg in messages]
    all_feedback = {}
    artifacts_by_message: Dict[int, List[Dict[str, Any]]] = {}
    if message_ids:
        feedback_list = db_session.exec(
            select(MessageFeedback)
            .where(MessageFeedback.message_id.in_(message_ids))
        ).all()
        
        for feedback in feedback_list:
            if feedback.message_id not in all_feedback:
                all_feedback[feedback.message_id] = {
                    "id": feedback.id,
                    "message_id": feedback.message_id,
                    "is_positive": feedback.is_positive,
                    "created_at": feedback.created_at.isoformat() if feedback.created_at else None,
                    "updated_at": feedback.updated_at.isoformat() if feedback.updated_at else None
                }

        artifact_rows = db_session.exec(
            select(Artifact).where(Artifact.message_id.in_(message_ids))
        ).all()
        for artifact in artifact_rows:
            if artifact.message_id is None:
                continue
            artifacts_by_message.setdefault(artifact.message_id, []).append(
                artifact_payload(artifact)
            )
    
    response = {
        "session": session.model_dump(),
        "conversations": [
            {
                **msg.model_dump(),
                "feedback": all_feedback.get(msg.id),
                "artifacts": artifacts_by_message.get(msg.id),
            }
            for msg in messages
        ],
        "hasMore": has_more,
        "nextCursor": next_cursor
    }
    
    return response

@router.delete("/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Session"], dependencies=[Depends(security)])
def delete_session(
    session_id: str,
    payload: dict = Depends(decode_token_payload),
    db_session: Session = Depends(db.get_session)
):
    """Soft-delete a session by setting is_active to False"""
    email = (payload.get("email") or "").strip().lower()
    user = db_session.exec(select(User).where(User.email == email)).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    session = db_session.exec(
        select(UserSession)
        .where(
            (UserSession.session_id == session_id) &
            (UserSession.is_active == True)  # Already deleted sessions return 404
        )
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    if session.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own sessions"
        )

    session.is_active = False
    session.updated_at = datetime.now(timezone.utc)
    db_session.add(session)
    db_session.commit()



@router.patch(
    "/session/{session_id}/status",
    response_model=SessionOut,
    tags=["Session"],
    dependencies=[Depends(security)],
)
def update_session_status(
    session_id: str,
    new_status: SessionStatus,
    payload: dict = Depends(decode_token_payload),
    db_session: Session = Depends(db.get_session)
):
    """Update session status (active/archived/deleted)"""
    query = (
        select(UserSession, Message.conversation_id)
        .join(Message, UserSession.id == Message.session_id)
        .where(
            (UserSession.session_id == session_id) &
            (UserSession.is_active == True)  # deleted sessions are 404
        )
    )

    result = db_session.exec(query).first()
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    session, conversation_id = result
    email = (payload.get("email") or "").strip().lower()
    user = db_session.exec(
        select(User).where(func.lower(User.email) == email)
    ).first()
    if not user or session.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update your own sessions",
        )

    if new_status == SessionStatus.DELETED:
        session.is_active = False
        session.status = SessionStatus.DELETED
        log_activity(
            db_session=db_session,
            user_id=session.user_id,
            action=ActionType.DELETE,
            entity_type=EntityType.CONVERSATION,
            entity_id=session.session_id,
            metadata={"subtitle": session.title or "Conversation"},
        )
    elif new_status == SessionStatus.ARCHIVED:
        session.status = SessionStatus.ARCHIVED
        session.is_archived = True
        log_activity(
            db_session=db_session,
            user_id=session.user_id,
            action=ActionType.UPDATE,
            entity_type=EntityType.CONVERSATION,
            entity_id=session.session_id,
            metadata={"subtitle": session.title or "Conversation", "status": "archived"},
        )
    else:
        # ACTIVE / unarchive transitions
        session.status = new_status
        session.is_archived = False
        log_activity(
            db_session=db_session,
            user_id=session.user_id,
            action=ActionType.UPDATE,
            entity_type=EntityType.CONVERSATION,
            entity_id=session.session_id,
            metadata={"subtitle": session.title or "Conversation", "status": "active"},
        )

    session.updated_at = datetime.now(timezone.utc)
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)

    response_data = session.model_dump()
    response_data["conversation_id"] = conversation_id
    response_data["agent_id"] = session.agent_id
    return response_data


@router.post(
    "/session/{session_id}/upload/document",
    response_model=SessionDocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Session"],
    dependencies=[Depends(security)],
)
async def upload_session_document(
    session_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    payload: dict = Depends(decode_token_payload),
    db_session: Session = Depends(db.get_session),
):
    email = (payload.get("email") or "").strip().lower()
    user = db_session.exec(select(User).where(func.lower(User.email) == email)).first()
    if not user:
        raise HTTPException(status_code=403, detail="Access denied")
    user_id = user.id

    # 1. Validate session
    user_session = db_session.exec(select(UserSession).where(UserSession.session_id == session_id)).first()
    if not user_session or user_session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Session not found or access denied")
        
    # 2. Get agent for org_id
    agent = db_session.get(AIAgent, user_session.agent_id)
    if not agent or not agent.organization_id:
        raise HTTPException(status_code=400, detail="Session agent has no organization")

    # 3. Save to S3
    file_extension = os.path.splitext(file.filename)[1].lower()
    original_filename = os.path.basename(file.filename)
    file_content = await file.read()
    file_size = len(file_content)

    timestamp = int(datetime.now(timezone.utc).timestamp())
    folder_prefix = f"organizations/{agent.organization_id}"
    s3_key = f"{folder_prefix}/{timestamp}_session_{session_id}_{original_filename}"
    full_s3_path = f"{s3_client.bucket_name}/{s3_key}"

    # We must use put_object directly because the IAM policy does NOT allow
    # CreateMultipartUpload (which upload_fileobj implicitly tries for >8MB files).
    s3_client.s3_client.put_object(
        Bucket=s3_client.bucket_name,
        Key=s3_key,
        Body=file_content,
        ContentType=file.content_type or 'application/octet-stream'
    )
   

    # 4. Create Document
    db_doc = Document(
        user_id=user_id,
        file_name=original_filename,
        file_path=full_s3_path,
        file_size=file_size,
        mime_type=file.content_type,
    )
    db_session.add(db_doc)
    db_session.flush()

    # 5. Create SessionDocument
    db_session.add(SessionDocument(
        document_id=db_doc.id,
        session_id=user_session.id,
        user_id=user_id,
    ))
    db_session.commit()
    db_session.refresh(db_doc)

    resolved_model_id = get_org_embedding_model(agent.organization_id, db_session)
    # 6. Embed (Synchronous to prevent race conditions during retrieval)
    await _embed_into_org(
        org_id=agent.organization_id,
        document_id=db_doc.id,
        file_name=original_filename,
        file_content=file_content,
        full_s3_path=full_s3_path,
        user_id=user_id,
        embedding_model_id=resolved_model_id,
        session_id=session_id,
    )

    return SessionDocumentUploadResponse(
        id=db_doc.id,
        file_name=db_doc.file_name,
        file_size=db_doc.file_size,
        session_id=session_id,
        extracted=True
    )
