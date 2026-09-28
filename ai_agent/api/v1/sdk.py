"""SDK Management API for Embeddable Agent Widget.

This module provides:
- Public API key management (scoped, domain-restricted keys)
- SDK session management (anonymous users)
- Widget configuration endpoints
- CORS validation for embeds
"""
import logging
import secrets
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Header, Request
from pydantic import BaseModel, Field, validator
from sqlmodel import Session, select

from ai_agent.core import db
from ai_agent.core.config import settings
from ai_agent.core.agent_runtime import get_agent_runtime
from ai_agent.core.enums import ActionType, EntityType
from ai_agent.models import (
    User,
    AIAgent,
    RunMode,
    UserSession,
    Message,
    AgentRun,
    Artifact,
    ConversationRole,
    ConversationType,
)
from ai_agent.core.response_schemas import content_block_to_markdown
from ai_agent.services.artifact_service import (
    artifact_payload,
    bind_artifacts_to_message,
    persist_chart_artifacts_from_blocks,
)
from ai_agent.services.activity_log import log_activity
from ai_agent.services.api_keys import get_api_key
from ai_agent.models_sdk import (
    PublicAPIKey, SDKSession,
    SDKConfigPreset, RateLimitLog
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/sdk", tags=["SDK"])


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateAPIKeyRequest(BaseModel):
    """Request to create a public API key."""
    agent_id: str = Field(description="Agent to expose via this key")
    name: str = Field(description="Name: 'Website Widget', 'Mobile App'")
    description: Optional[str] = Field(default=None)
    
    # Restrictions
    allowed_domains: List[str] = Field(
        default=[],
        description="Allowed domains: ['example.com', '*.example.com']"
    )
    allowed_ips: List[str] = Field(default=[])
    
    # Limits
    rate_limit_requests_per_minute: int = Field(default=60, ge=1, le=1000)
    rate_limit_tokens_per_day: Optional[int] = Field(default=None)
    expires_days: Optional[int] = Field(default=365, description="Days until expiration")
    
    # Features
    enable_voice: bool = Field(default=True)
    enable_file_upload: bool = Field(default=False)
    enable_streaming: bool = Field(default=True)
    allowed_modes: List[str] = Field(default=["chat", "tool_use"])
    
    @validator('allowed_domains')
    def validate_domains(cls, v):
        for domain in v:
            if '*' in domain and not domain.startswith('*.'):
                raise ValueError(f"Wildcard must be prefix: {domain}")
        return v


class APIKeyResponse(BaseModel):
    """Response with API key (only shown once on creation)."""
    id: str
    key: str = Field(description="The actual API key (pk_live_... or pk_test_...)")
    name: str
    agent_id: str
    agent_name: str
    allowed_domains: List[str]
    rate_limit_requests_per_minute: int
    expires_at: Optional[datetime]
    created_at: datetime
    
    # Usage stats
    total_requests: int
    last_used_at: Optional[datetime]


class APIKeyListItem(BaseModel):
    """List item for API keys (key not included)."""
    id: str
    name: str
    agent_id: str
    agent_name: str
    key_type: str
    is_active: bool
    allowed_domains: List[str]
    expires_at: Optional[datetime]
    created_at: datetime
    total_requests: int
    last_used_at: Optional[datetime]


class ValidateEmbedRequest(BaseModel):
    """Request to validate an embed key (CORS preflight)."""
    api_key: str
    domain: str = Field(description="Origin domain")
    

class ValidateEmbedResponse(BaseModel):
    """Response for embed validation."""
    valid: bool
    agent_id: Optional[str]
    agent_name: Optional[str]
    features: Optional[Dict[str, Any]]
    error: Optional[str]


class CreateSDKSessionRequest(BaseModel):
    """Create an anonymous SDK session."""
    api_key: str
    anonymous_user_id: Optional[str] = Field(default=None, description="Client-generated ID")
    user_agent: Optional[str] = Field(default=None)
    referrer: Optional[str] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default={})


class SDKSessionResponse(BaseModel):
    """SDK session response."""
    session_id: str
    agent_id: str
    agent_name: str
    agent_description: Optional[str]
    features: Dict[str, Any]
    expires_at: datetime


class SDKMessageRequest(BaseModel):
    """Send message via SDK."""
    session_id: str
    message: str
    mode: str = Field(default="chat")
    stream: bool = Field(default=False)


class SDKConfigResponse(BaseModel):
    """SDK widget configuration."""
    agent_id: str
    agent_name: str
    theme: str
    primary_color: Optional[str]
    logo_url: Optional[str]
    position: str
    initial_message: Optional[str]
    placeholder_text: str
    show_file_upload: bool
    show_voice_button: bool
    suggested_prompts: List[str]


# ============================================================================
# API Key Management
# ============================================================================

@router.post("/keys", response_model=APIKeyResponse)
async def create_api_key(
    request: CreateAPIKeyRequest,
    user_id: int = Query(..., description="User ID (owner)"),
    db_session: Session = Depends(db.get_session),
):
    """Create a new public API key for SDK embedding.
    
    The key is shown ONLY on creation. Store it securely - it cannot be
    retrieved later.
    
    Example:
        {
          "agent_id": "agent_123",
          "name": "Website Widget",
          "allowed_domains": ["example.com", "*.example.com"],
          "rate_limit_requests_per_minute": 60
        }
    
    Returns:
        {
          "id": "pk_live_abc123",
          "key": "pk_live_abc123",
          "name": "Website Widget",
          ...
        }
    """
    # Verify user and agent
    user = db_session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    agent = db_session.get(AIAgent, request.agent_id)
    if not agent or agent.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Agent not found or access denied")
    
    # Generate key
    prefix = "pk_live" if not any(d.startswith("localhost") for d in request.allowed_domains) else "pk_test"
    key_id = f"{prefix}_{secrets.token_urlsafe(16)}"
    
    # Calculate expiration
    expires_at = None
    if request.expires_days:
        expires_at = datetime.utcnow() + timedelta(days=request.expires_days)
    
    # Create key record
    api_key = PublicAPIKey(
        id=key_id,
        user_id=user_id,
        agent_id=request.agent_id,
        key_type="live" if prefix == "pk_live" else "test",
        name=request.name,
        description=request.description,
        allowed_domains=request.allowed_domains,
        allowed_ips=request.allowed_ips,
        rate_limit_requests_per_minute=request.rate_limit_requests_per_minute,
        rate_limit_tokens_per_day=request.rate_limit_tokens_per_day,
        enable_voice=request.enable_voice,
        enable_file_upload=request.enable_file_upload,
        enable_streaming=request.enable_streaming,
        allowed_modes=request.allowed_modes,
        expires_at=expires_at,
    )
    
    db_session.add(api_key)
    db_session.commit()
    db_session.refresh(api_key)
    
    log.info(f"Created public API key {key_id} for agent {request.agent_id}")
    
    return APIKeyResponse(
        id=key_id,
        key=key_id,  # This is the only time the key is shown
        name=api_key.name,
        agent_id=agent.id,
        agent_name=agent.name,
        allowed_domains=api_key.allowed_domains,
        rate_limit_requests_per_minute=api_key.rate_limit_requests_per_minute,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
        total_requests=0,
        last_used_at=None,
    )


@router.get("/keys", response_model=List[APIKeyListItem])
async def list_api_keys(
    user_id: int = Query(...),
    agent_id: Optional[str] = Query(None),
    db_session: Session = Depends(db.get_session),
):
    """List all public API keys (without the actual key values)."""
    query = select(PublicAPIKey).where(PublicAPIKey.user_id == user_id)
    
    if agent_id:
        query = query.where(PublicAPIKey.agent_id == agent_id)
    
    keys = db_session.exec(query.order_by(PublicAPIKey.created_at.desc())).all()
    
    result = []
    for key in keys:
        agent = db_session.get(AIAgent, key.agent_id)
        result.append(APIKeyListItem(
            id=key.id,
            name=key.name,
            agent_id=key.agent_id,
            agent_name=agent.name if agent else "Unknown",
            key_type=key.key_type,
            is_active=key.is_active,
            allowed_domains=key.allowed_domains,
            expires_at=key.expires_at,
            created_at=key.created_at,
            total_requests=key.total_requests,
            last_used_at=key.last_used_at,
        ))
    
    return result


@router.delete("/keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    user_id: int = Query(...),
    db_session: Session = Depends(db.get_session),
):
    """Revoke a public API key."""
    key = db_session.get(PublicAPIKey, key_id)
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    if key.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    key.is_active = False
    db_session.commit()
    
    return {"status": "revoked", "key_id": key_id}


@router.get("/keys/{key_id}/stats")
async def get_key_stats(
    key_id: str,
    user_id: int = Query(...),
    days: int = Query(30, ge=1, le=90),
    db_session: Session = Depends(db.get_session),
):
    """Get usage statistics for an API key."""
    key = db_session.get(PublicAPIKey, key_id)
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    if key.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get session stats
    from sqlalchemy import func
    
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    sessions_count = db_session.exec(
        select(func.count(SDKSession.id))
        .where(SDKSession.api_key_id == key_id)
        .where(SDKSession.created_at >= cutoff)
    ).one()
    
    messages_count = db_session.exec(
        select(func.count(Message.id))
        .join(UserSession, Message.session_id == UserSession.id)
        .join(SDKSession, UserSession.session_id == SDKSession.id)
        .where(SDKSession.api_key_id == key_id)
        .where(Message.created_at >= cutoff)
    ).one()

    tokens_sum = db_session.exec(
        select(func.sum(AgentRun.total_tokens))
        .select_from(AgentRun)
        .join(UserSession, AgentRun.session_id == UserSession.id)
        .join(SDKSession, UserSession.session_id == SDKSession.id)
        .where(SDKSession.api_key_id == key_id)
        .where(AgentRun.created_at >= cutoff)
    ).one() or 0
    
    return {
        "key_id": key_id,
        "period_days": days,
        "total_sessions": sessions_count,
        "total_messages": messages_count,
        "total_tokens": tokens_sum,
        "total_requests": key.total_requests,
        "last_used_at": key.last_used_at.isoformat() if key.last_used_at else None,
    }


# ============================================================================
# Embed Validation (CORS Preflight)
# ============================================================================

@router.post("/validate-embed", response_model=ValidateEmbedResponse)
async def validate_embed(
    request: ValidateEmbedRequest,
    db_session: Session = Depends(db.get_session),
):
    """Validate if a domain is allowed to use an API key.
    
    This should be called client-side before initializing the widget.
    It performs domain matching and returns feature flags.
    """
    # Get the key
    key = db_session.get(PublicAPIKey, request.api_key)
    if not key:
        return ValidateEmbedResponse(valid=False, error="Invalid API key")
    
    if not key.is_active:
        return ValidateEmbedResponse(valid=False, error="API key is revoked")
    
    if key.expires_at and key.expires_at < datetime.utcnow():
        return ValidateEmbedResponse(valid=False, error="API key has expired")
    
    # Check domain
    domain_allowed = False
    request_domain = request.domain.lower().replace("http://", "").replace("https://", "").split("/")[0]
    
    if not key.allowed_domains:
        # No domain restrictions
        domain_allowed = True
    else:
        for allowed in key.allowed_domains:
            allowed = allowed.lower()
            if allowed.startswith("*."):
                # Wildcard match
                suffix = allowed[2:]  # Remove *.
                if request_domain.endswith(suffix):
                    domain_allowed = True
                    break
            elif allowed == request_domain:
                domain_allowed = True
                break
    
    if not domain_allowed:
        log.warning(f"Domain {request_domain} not allowed for key {key.id}")
        return ValidateEmbedResponse(
            valid=False,
            error=f"Domain '{request_domain}' is not authorized"
        )
    
    # Get agent info
    agent = db_session.get(AIAgent, key.agent_id)
    
    return ValidateEmbedResponse(
        valid=True,
        agent_id=key.agent_id,
        agent_name=agent.name if agent else "Agent",
        features={
            "voice": key.enable_voice,
            "file_upload": key.enable_file_upload,
            "streaming": key.enable_streaming,
            "artifacts": key.enable_artifacts,
            "modes": key.allowed_modes,
        },
    )


# ============================================================================
# SDK Session Management
# ============================================================================

@router.post("/sessions", response_model=SDKSessionResponse)
async def create_sdk_session(
    request: CreateSDKSessionRequest,
    client_ip: Optional[str] = Header(None, alias="X-Forwarded-For"),
    db_session: Session = Depends(db.get_session),
):
    """Create an anonymous session for SDK usage.
    
    This is called by the embeddable widget to start a conversation.
    No authentication required - only valid API key + domain check.
    """
    # Validate API key
    key = db_session.get(PublicAPIKey, request.api_key)
    if not key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not key.is_active:
        raise HTTPException(status_code=401, detail="API key revoked")
    
    if key.expires_at and key.expires_at < datetime.utcnow():
        raise HTTPException(status_code=401, detail="API key expired")
    
    # IP check if configured
    if key.allowed_ips and client_ip:
        if client_ip not in key.allowed_ips:
            log.warning(f"IP {client_ip} not allowed for key {key.id}")
            raise HTTPException(status_code=403, detail="IP not authorized")
    
    # Rate limit check
    recent_requests = db_session.exec(
        select(RateLimitLog)
        .where(RateLimitLog.api_key_id == key.id)
        .where(RateLimitLog.created_at >= datetime.utcnow() - timedelta(minutes=1))
    ).all()
    
    if len(recent_requests) >= key.rate_limit_requests_per_minute:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Create session
    import uuid
    session_id = f"sess_{uuid.uuid4().hex[:16]}"
    
    session = SDKSession(
        id=session_id,
        api_key_id=key.id,
        anonymous_user_id=request.anonymous_user_id,
        user_agent=request.user_agent,
        ip_address=client_ip,
        referrer=request.referrer,
        extra_metadata=request.metadata or {},
    )
    
    db_session.add(session)
    
    # Update key stats
    key.last_used_at = datetime.utcnow()
    key.total_requests += 1
    
    db_session.commit()
    db_session.refresh(session)
    
    # Get agent details
    agent = db_session.get(AIAgent, key.agent_id)
    
    log.info(f"Created SDK session {session_id} for agent {key.agent_id}")
    
    return SDKSessionResponse(
        session_id=session_id,
        agent_id=key.agent_id,
        agent_name=agent.name if agent else "Agent",
        agent_description=agent.description if agent else None,
        features={
            "voice": key.enable_voice,
            "file_upload": key.enable_file_upload,
            "streaming": key.enable_streaming,
            "artifacts": key.enable_artifacts,
            "modes": key.allowed_modes,
        },
        expires_at=session.expires_at,
    )


# ============================================================================
# Widget Configuration
# ============================================================================

@router.get("/config/{api_key}")
async def get_widget_config(
    api_key: str,
    origin: Optional[str] = Header(None),
    db_session: Session = Depends(db.get_session),
):
    """Get widget configuration for an embed key.
    
    This endpoint is called by the embeddable widget to get:
    - Agent info
    - Appearance settings
    - Feature flags
    - Suggested prompts
    """
    key = db_session.get(PublicAPIKey, api_key)
    if not key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not key.is_active:
        raise HTTPException(status_code=401, detail="API key revoked")
    
    agent = db_session.get(AIAgent, key.agent_id)
    
    from ai_agent.models import AIAgentSettings
    agent_settings = db_session.exec(
        select(AIAgentSettings).where(AIAgentSettings.agent_id == key.agent_id)
    ).first()
    
    from ai_agent.core.aws_s3 import s3_client
    
    return {
        "agent_id": key.agent_id,
        "agent_name": agent.name if agent else "Agent",
        "agent_description": agent.description if agent else None,
        # embed_config and logo_url now live on the SDK key, not the agent
        "embed_config": key.embed_config,
        "logo_url": s3_client.get_presigned_url_from_uri(key.logo_url),
        "features": {
            "voice": key.enable_voice,
            "file_upload": key.enable_file_upload,
            "streaming": key.enable_streaming,
            "artifacts": key.enable_artifacts,
        },
        "theme": "light",
        "position": "bottom-right",
        "placeholder_text": "Type your message...",
        "suggested_prompts": [],
    }

# Embed Config & Logo Endpoints (scoped to SDK key)

class EmbedConfigUpdate(BaseModel):
    """Request body for updating embed widget configuration."""
    embed_config: Dict[str, Any] = Field(..., description="Embed widget customization settings")


@router.get("/keys/{api_key}/embed-config")
def get_embed_config(
    api_key: str,
    db_session: Session = Depends(db.get_session),
):
    """Retrieve the embed widget customization settings for an SDK key."""
    key = db_session.get(PublicAPIKey, api_key)
    if not key or not key.is_active:
        raise HTTPException(status_code=404, detail="SDK key not found")

    from ai_agent.core.aws_s3 import s3_client
    return {
        "embed_config": key.embed_config,
        "logo_url": s3_client.get_presigned_url_from_uri(key.logo_url),
    }


@router.put("/keys/{api_key}/embed-config")
def update_embed_config(
    api_key: str,
    payload: EmbedConfigUpdate,
    db_session: Session = Depends(db.get_session),
):
    """Save embed widget customization settings for an SDK key."""
    key = db_session.get(PublicAPIKey, api_key)
    if not key or not key.is_active:
        raise HTTPException(status_code=404, detail="SDK key not found")

    key.embed_config = payload.embed_config
    key.updated_at = datetime.now()
    db_session.add(key)
    db_session.commit()
    db_session.refresh(key)
    return {"success": True, "embed_config": key.embed_config}


from fastapi import UploadFile, File as FastAPIFile


@router.post("/keys/{api_key}/upload-logo")
async def upload_sdk_logo(
    api_key: str,
    file: UploadFile = FastAPIFile(..., description="Logo image file (PNG, JPG, SVG, WEBP)"),
    db_session: Session = Depends(db.get_session),
):
    """Upload a custom logo for a specific SDK key. Stored in S3, URI persisted on the key."""
    key = db_session.get(PublicAPIKey, api_key)
    if not key or not key.is_active:
        raise HTTPException(status_code=404, detail="SDK key not found")

    allowed_types = {"image/png", "image/jpeg", "image/svg+xml", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type: {file.content_type}. Allowed: {', '.join(allowed_types)}",
        )

    from ai_agent.core.aws_s3 import s3_client
    import uuid as uuid_mod

    ext = file.filename.rsplit(".", 1)[-1] if file.filename and "." in file.filename else "png"
    s3_key = f"sdk-logos/{key.agent_id}/{api_key}/{uuid_mod.uuid4().hex}.{ext}"

    await s3_client.upload_fileobj(
        file_obj=file.file,
        object_name=s3_key,
        content_type=file.content_type,
    )

    key.logo_url = f"s3://{s3_key}"
    key.updated_at = datetime.now()
    db_session.add(key)
    db_session.commit()
    db_session.refresh(key)

    return {"success": True, "logo_url": s3_client.get_presigned_url_from_uri(key.logo_url)}


# ============================================================================
# SDK Session Messaging
# ============================================================================

@router.post("/sessions/{session_id}/message")
async def send_sdk_message(
    session_id: str,
    request: SDKMessageRequest,
    db_session: Session = Depends(db.get_session),
):
    """Send a message in an SDK session.
    
    This uses the AgentRuntime internally and persists messages in the
    regular messages table so SDK conversations share the same storage model.
    """
    # Validate session
    session = db_session.get(SDKSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session expired")
    
    if session.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Session expired")
    
    # Get API key for feature checks
    key = db_session.get(PublicAPIKey, session.api_key_id)
    
    # Check mode is allowed
    if request.mode not in key.allowed_modes:
        raise HTTPException(status_code=400, detail=f"Mode '{request.mode}' not allowed")

    # Integration  with AgentRuntime
    runtime_session = db_session.exec(
        select(UserSession).where(UserSession.session_id == session_id)
    ).first()
    if not runtime_session:
        runtime_session = UserSession(
            user_id=key.user_id,
            agent_id=key.agent_id,
            session_id=session_id,
            title=f"Embed Session :{request.message.strip()[:60]}" if request.message.strip() else f"Embed-{session_id[:8]}"
        )
        db_session.add(runtime_session)
        db_session.flush()
        agent = db_session.get(AIAgent, key.agent_id)
        log_activity(
            db_session=db_session,
            user_id=key.user_id,
            action=ActionType.START,
            entity_type=EntityType.CONVERSATION,
            entity_id=runtime_session.session_id,
            metadata={
                "subtitle": runtime_session.title or (agent.name if agent else "Conversation"),
                "agent_id": key.agent_id,
            },
        )
        db_session.commit()
        db_session.refresh(runtime_session)

    previous_messages = db_session.exec(
        select(Message)
        .where(Message.session_id == runtime_session.id)
        .where(Message.conversation_id == session_id)
        .order_by(Message.counter)
    ).all()

    next_counter = (
        max((msg.counter for msg in previous_messages), default=-1) + 1
    )

    user_msg = Message(
        session_id=runtime_session.id,
        agent_id=key.agent_id,
        conversation_id=session_id,
        role=ConversationRole.USER,
        content=request.message,
        counter=next_counter,
        conversation_type=ConversationType.TEXT,
    )
    db_session.add(user_msg)
    db_session.commit()
    db_session.refresh(user_msg)

    messages = [
        {"role": msg.role, "content": msg.content}
        for msg in previous_messages
    ]
    messages.append({"role": "user", "content": request.message})

    mode_map = {
        "chat": RunMode.CHAT,
        "tool_use": RunMode.TOOL_USE,
        "planner": RunMode.PLANNER,
        "autonomous": RunMode.AUTONOMOUS,
    }

    agent = db_session.get(AIAgent, key.agent_id)
    api_key = None
    if agent and agent.settings:
        api_key = get_api_key(db_session, agent.settings.api_key)

    try:
        runtime = await get_agent_runtime(db_session)
        response = await runtime.run(
            agent_id=key.agent_id,
            user_id=key.user_id,
            session_id=runtime_session.id,
            conversation_id=session_id,
            messages=messages,
            mode=mode_map.get(request.mode, RunMode.CHAT),
            input_message_id=user_msg.id,
            stream=False,
            api_key=api_key,
        )
    except Exception as e:
        from ai_agent.schema_engine.orchestrator import SchemaEngineError

        if isinstance(e, SchemaEngineError):
            log.warning(f"Schema engine error for SDK session {session_id}: [{e.error_code}] {e.message}")
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": e.error_code,
                    "message": e.message,
                },
            )
        log.error(f"AgentRuntime error for SDK session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

    assistant_content = "\n\n".join(
        md for md in (content_block_to_markdown(b) for b in response.blocks) if md
    )
    usage = response.usage
    assistant_msg = Message(
        session_id=runtime_session.id,
        agent_id=key.agent_id,
        conversation_id=session_id,
        role=ConversationRole.ASSISTANT,
        content=assistant_content,
        counter=next_counter + 1,
        conversation_type=ConversationType.TEXT,
    )
    db_session.add(assistant_msg)

    session.message_count += 2
    session.token_count += usage.total_tokens
    session.last_activity_at = datetime.utcnow()
    key.total_requests += 1
    key.total_tokens += usage.total_tokens
    key.last_used_at = datetime.utcnow()

    db_session.commit()
    db_session.refresh(assistant_msg)
    bind_artifacts_to_message(
        db_session,
        artifacts=response.artifacts,
        message_id=assistant_msg.id,
    )
    try:
        persist_chart_artifacts_from_blocks(
            db_session,
            blocks=response.blocks,
            run_id=response.run_id,
            user_id=key.user_id,
            agent_id=key.agent_id,
            session_id=assistant_msg.session_id,
            conversation_id=assistant_msg.conversation_id,
            message_id=assistant_msg.id,
        )
    except Exception:
        log.exception(
            "Failed to persist chart artifacts run_id=%s message_id=%s",
            response.run_id,
            assistant_msg.id,
        )
    response.message_id = assistant_msg.id
    
    return {
        "message_id": assistant_msg.id,
        "content": assistant_msg.content,
        "role": "assistant",
        "created_at": assistant_msg.created_at.isoformat(),
        "run_id": response.run_id,
        "blocks": [block.dict() for block in response.blocks],
        "artifacts": response.artifacts,
        "citations": response.citations,
        "usage": usage.dict(),
    }


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db_session: Session = Depends(db.get_session),
):
    """Get message history for an SDK session."""
    session = db_session.get(SDKSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    runtime_session = db_session.exec(
        select(UserSession).where(UserSession.session_id == session_id)
    ).first()
    if not runtime_session:
        return {
            "messages": [],
            "session_id": session_id,
        }
    
    messages = db_session.exec(
        select(Message)
        .where(Message.session_id == runtime_session.id)
        .where(Message.conversation_id == session_id)
        .order_by(Message.counter.desc())
        .offset(offset)
        .limit(limit)
    ).all()

    message_ids = [msg.id for msg in messages]
    artifacts_by_message: Dict[int, List[Dict[str, Any]]] = {}
    if message_ids:
        artifact_rows = db_session.exec(
            select(Artifact).where(Artifact.message_id.in_(message_ids))
        ).all()
        for artifact in artifact_rows:
            if artifact.message_id is None:
                continue
            artifacts_by_message.setdefault(artifact.message_id, []).append(
                artifact_payload(artifact)
            )
    
    return {
        "messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "artifacts": artifacts_by_message.get(msg.id),
                "created_at": msg.created_at.isoformat(),
            }
            for msg in reversed(messages)  # Chronological order
        ],
        "session_id": session_id,
    }


# ============================================================================
# Config Presets
# ============================================================================

@router.post("/config-presets")
async def create_config_preset(
    user_id: int = Query(...),
    name: str = Query(...),
    theme: str = Query("light"),
    position: str = Query("bottom-right"),
    db_session: Session = Depends(db.get_session),
):
    """Create a configuration preset for widgets."""
    import uuid
    
    preset = SDKConfigPreset(
        id=f"preset_{uuid.uuid4().hex[:8]}",
        user_id=user_id,
        name=name,
        theme=theme,
        position=position,
    )
    
    db_session.add(preset)
    db_session.commit()
    
    return {"preset_id": preset.id, "name": name}


@router.get("/config-presets")
async def list_config_presets(
    user_id: int = Query(...),
    db_session: Session = Depends(db.get_session),
):
    """List configuration presets."""
    presets = db_session.exec(
        select(SDKConfigPreset).where(SDKConfigPreset.user_id == user_id)
    ).all()
    
    return {
        "presets": [
            {
                "id": p.id,
                "name": p.name,
                "theme": p.theme,
                "position": p.position,
            }
            for p in presets
        ]
    }
