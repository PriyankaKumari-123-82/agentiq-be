import json
import logging
import re
from datetime import datetime, timezone
from typing import List, Literal, Optional, Dict, Any, Union
from uuid import uuid4
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, root_validator
from typing import Dict, List as ListType
from sqlmodel import Session, select, func

from ai_agent.models import AIAgent, AIAgentSettings, User, ApiKey, Document, UserSession, MCPServerConfig, AgentMCPConfigLink, AgentDocument, ModelProvider, AgentWhatsappNumber
from ai_agent.core.enums import ActionType, EntityType
from ai_agent.core.db import get_session
from ai_agent.core.config import settings

from ai_agent.core.token_validation import security
from ai_agent.core.vector_db import CollectionPerAgent
from ai_agent.services.activity_log import log_activity
from ai_agent.services.embedding import EmbeddingService, get_org_embedding_model

from ai_agent.core.security import encryption_manager
from ai_agent.services.whatsapp_service import send_whatsapp_text_message, send_whatsapp_template_message

log = logging.getLogger(__name__)
router = APIRouter(tags=["Agents"])


class MCPTestResponse(BaseModel):
    success: bool
    message: str = ""


class ModelProviderItem(BaseModel):
    model_name: str
    type: str
    dimension: Optional[int] = None
    # False when the model rejects the temperature parameter (e.g. o1-preview, o3-mini).
    # The UI reads this to disable the temperature slider for unsupported models.
    supports_temperature: bool = True
    max_llm_temperature: float 
    is_llm: bool = True
    is_websearch: bool = False


class ModelProviderGroup(BaseModel):
    provider: str
    models: ListType[ModelProviderItem]


def _get_embedding_dimension_from_catalog(db: Session, model_name: str) -> int:
    row = db.exec(
        select(ModelProvider).where(
            ModelProvider.model_name == model_name,
            ModelProvider.type == "embeddings",
        )
    ).first()
    if row is None or row.dimension is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Embedding model was not found in the model catalog or has no dimension set: "
                f"{model_name}"
            ),
        )
    return row.dimension


@router.get("/model-providers", response_model=ListType[ModelProviderGroup])
async def get_model_providers(
    provider: Optional[str] = Query(default=None, description="Filter by provider"),
    type: Optional[str] = Query(default=None, description="Filter by model type"),
    is_llm: Optional[bool] = Query(default=None, description="Filter by whether it supports LLM"),
    is_websearch: Optional[bool] = Query(default=None, description="Filter by whether it supports web search"),
    db: Session = Depends(get_session),
):
    """Return the DB-backed model catalog grouped by provider."""
    query = select(ModelProvider).where(ModelProvider.is_active == True)
    if provider:
        query = query.where(ModelProvider.provider == provider.strip().lower())
    if type:
        query = query.where(ModelProvider.type == type)
    if is_llm is not None:
        query = query.where(ModelProvider.is_llm == is_llm)
    if is_websearch is not None:
        query = query.where(ModelProvider.is_websearch == is_websearch)

    rows = db.exec(query.order_by(ModelProvider.provider, ModelProvider.id)).all()
    grouped: dict[str, list[ModelProviderItem]] = {}
    for row in rows:
        grouped.setdefault(row.provider, []).append(
            ModelProviderItem(
                model_name=row.model_name,
                type=row.type,
                dimension=row.dimension,
                supports_temperature=row.supports_temperature,
                max_llm_temperature=row.max_llm_temperature,
                is_llm=row.is_llm,
                is_websearch=row.is_websearch,
            )
        )

    return [
        ModelProviderGroup(provider=provider_name, models=models)
        for provider_name, models in grouped.items()
    ]


class MCPServerInput(BaseModel):
    """Inline MCP server config provided during agent creation/update (SSE transport)."""
    name: str = Field(..., description="Display name for this MCP server")
    url: str = Field(..., description="URL for SSE/HTTP transport")
    description: str = Field(..., description="Human-readable description of what this MCP server exposes.")
    requires_query_building: bool = Field(
        default=False,
        description=(
            "If True, this MCP server exposes a generic database and the agent should "
            "fetch schema + generate SQL. If False, treat the server as purpose-built tools."
        ),
    )


class AgentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    is_active: bool = True
    user_id: int = Field(..., description="ID of the user who owns this agent")
    organization_id: Optional[int] = Field(default=None, description="Organization ID this agent belongs to")
    llm_provider: Optional[str] = Field(
        default=None,
        description="The LLM provider to use for this agent"
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="The LLM model to use for this agent"
    )
    api_key: Optional[int] = Field(
        default=None,
        description="Organization API key row ID selected for this agent's LLM provider",
    )
    llm_temperature: Optional[float] = Field(
        default=None,
        ge=0.0, le=2.0,
        description="Temperature for model responses"
    )
    llm_system_prompt: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="Custom system prompt for the agent. Can be a string or a JSON object. If not provided, the default from settings will be used."
    )
    llm_voice_prompt: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="Custom voice prompt for the agent. Can be a string or a JSON object. If not provided, the default from settings will be used."
    )
    llm_json_prompt: bool = Field(
        default=False,
        description="Whether to use JSON format for model prompts (default: False)"
    )
    preserve_memory: bool = Field(
        default=True,
        description="Whether to preserve conversation history for this agent (default: True)"
    )
    allow_websearch: bool = Field(
        default=True,
        description="Whether the agent is allowed to search the web for answers (default: True)"
    )
    websearch_provider: Optional[str] = Field(
        default=None,
        description="Selected web search provider"
    )
    websearch_api_key_id: Optional[int] = Field(
        default=None,
        description="API key row for the web search provider"
    )
    websearch_model: Optional[str] = Field(
        default=None,
        description="Selected model for web search (e.g., gpt-5.4-mini)"
    )
    is_mcp_enabled: bool = Field(
        default=False,
        description="Whether MCP server integration is enabled for this agent"
    )
    whatsapp_enabled: bool = Field(
        default=False,
        description="Whether WhatsApp integration is enabled for this agent"
    )
    mcp_server_configs: Optional[List[MCPServerInput]] = Field(
        default=None,
        description="List of MCP server configurations for this agent"
    )

class AgentResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    owner_id: int
    organization_id: Optional[int] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime
    number_of_conversations: int = 0


class PaginatedAgentsOut(BaseModel):
    items: List[AgentResponse]
    page: int
    limit: int
    total: int
    has_more: bool


class AgentSettingsUpdate(BaseModel):
    """Model for updating agent settings.
    
    Attributes:
        description: Optional description of the agent
        llm_provider: The LLM provider to use
        llm_model: The LLM model to use
    """
    name: Optional[str] = Field(
        default=None,
        description="The name of the agent"
    )
    description: Optional[str] = Field(
        default=None,
        description="The description of the agent"
    )
    is_active: Optional[bool] = Field(
        default=None,
        description="Whether the agent is active"
    )
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    api_key: Optional[int] = None
    llm_temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    llm_max_tokens: Optional[int] = None
    llm_system_prompt: Optional[Union[str, Dict[str, Any]]] = None
    llm_voice_prompt: Optional[Union[str, Dict[str, Any]]] = None
    llm_json_prompt: Optional[bool] = None
    preserve_memory: Optional[bool] = None
    allow_websearch: Optional[bool] = None
    websearch_provider: Optional[str] = None
    websearch_api_key_id: Optional[int] = None
    websearch_model: Optional[str] = None
    is_mcp_enabled: Optional[bool] = Field(
        default=None,
        description="Whether MCP server integration is enabled for this agent"
    )
    whatsapp_enabled: Optional[bool] = Field(
        default=None,
        description="Whether WhatsApp integration is enabled for this agent"
    )
    mcp_server_configs: Optional[List[MCPServerInput]] = Field(
        default=None,
        description="List of MCP server configurations. Replaces existing MCP servers for this agent."
    )

    model_config = ConfigDict(extra='ignore')


def get_safe_collection_name(agent_id: str) -> str:
    """
    Generate a safe collection name from an agent ID.
    
    Args:
        agent_id: The agent ID to generate a collection name for
        
    Returns:
        A safe collection name that can be used with Milvus
    """
    # Convert to lowercase and replace non-word characters with underscore
    safe_id = re.sub(r'\W+', '_', agent_id.lower())
    
    # Ensure it starts with a letter or underscore
    if not safe_id[0].isalpha() and safe_id[0] != '_':
        safe_id = 'a' + safe_id
        
    # Ensure it's not too long (Milvus has a 255 character limit)
    safe_id = safe_id[:240]  # Leave some room for the prefix
    
    return f"agent_{safe_id}"


@router.post("/agents", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
def create_agent(
    agent_data: AgentCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_session)
):
    """Create a new AI agent."""
    # Check if user exists
    user = db.get(User, agent_data.user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    # Create agent
    agent = AIAgent(
        id=str(uuid4()),
        name=agent_data.name,
        description=agent_data.description,
        owner_id=agent_data.user_id,
        organization_id=agent_data.organization_id or getattr(user, "active_organization_id", None),
        is_active=agent_data.is_active,
    )
    
    # Handle llm_system_prompt JSON conversion if json_prompt is True
    system_prompt = agent_data.llm_system_prompt
    if agent_data.llm_json_prompt and system_prompt is not None:
        try:
            if not isinstance(system_prompt, str):
                # If it's a dict, convert to JSON string
                system_prompt = json.dumps(system_prompt, ensure_ascii=False)
            else:
                # If it's a string, validate and normalize the JSON
                parsed = json.loads(system_prompt)
                system_prompt = json.dumps(parsed, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError) as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"detail": [{"type": "json_invalid", "loc": ["body", "llm_system_prompt"], "msg": str(e)}]})

    requested_provider = agent_data.llm_provider.strip().lower()
    requested_model = agent_data.llm_model

    temperature = agent_data.llm_temperature if agent_data.llm_temperature is not None else settings.DEFAULT_LLM_TEMPERATURE

    # Embedding model is chosen once at the organization level; every agent
    # in the org inherits it for both indexing and retrieval.
    org_embedding_model = get_org_embedding_model(agent.organization_id, db)
    if org_embedding_model is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Organization has no embedding model configured.",
        )

    agent_settings = AIAgentSettings(
        agent_id=agent.id,
        llm_provider=requested_provider,
        llm_model=requested_model,
        llm_temperature=temperature,
        llm_max_tokens=settings.LLM_MAX_TOKENS,
        llm_system_prompt=system_prompt or settings.LLM_SYSTEM_PROMPT,
        api_key=agent_data.api_key,
        allow_websearch=agent_data.allow_websearch,
        websearch_provider=agent_data.websearch_provider,
        websearch_api_key_id=agent_data.websearch_api_key_id,
        websearch_model=agent_data.websearch_model,
    )

    # Document-level embeddings: agents no longer own individual Milvus collections.
    # Organization collections (org_{org_id}) are shared across all agents in that org,
    # so every agent is eligible to retrieve during RAG once documents are linked to it
    # via AgentDocument — there's no reason to opt out per agent.
    embedding_model = EmbeddingService().get_model_spec(org_embedding_model)
    if embedding_model is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Embedding model was not found in the model catalog: "
                f"{org_embedding_model}"
            ),
        )
    agent_settings.has_data_store = True

    agent_settings.is_mcp_enabled = agent_data.is_mcp_enabled
    agent_settings.whatsapp_enabled = agent_data.whatsapp_enabled

    db.add(agent)
    db.add(agent_settings)
    db.flush()  # flush to get agent_settings.id before using it

    # Create and link MCP server configs if enabled and provided
    if agent_data.is_mcp_enabled and agent_data.mcp_server_configs:
        for cfg in agent_data.mcp_server_configs:
            mcp_srv = MCPServerConfig(
                user_id=agent_data.user_id,
                name=cfg.name,
                url=cfg.url,
                description=cfg.description,
                requires_query_building=cfg.requires_query_building,
            )
            db.add(mcp_srv)
            db.flush()
            db.add(AgentMCPConfigLink(
                agent_settings_id=agent_settings.id,
                mcp_server_config_id=mcp_srv.id,
            ))

    # Log activity
    log_activity(
        db_session=db,
        user_id=agent_data.user_id,
        action=ActionType.CREATE,
        entity_type=EntityType.AGENT,
        entity_id=agent.id,
    )
    db.commit()
    db.refresh(agent)

    # Kick off initial FAQ generation in the background (non-blocking)
    if agent_data.is_mcp_enabled:
        log.info("Scheduling initial FAQ generation for agent '%s'", agent.id)
        from ai_agent.workflows.faq.faq import generate_faqs
        background_tasks.add_task(generate_faqs, agent.id, db, False)
    return _compute_agent_response(agent, db)


def _compute_agent_response(agent: AIAgent, db: Session) -> AgentResponse:
    conversation_count = db.exec(
        select(func.count())
        .select_from(UserSession)
        .where(UserSession.agent_id == agent.id)
    ).one()

    return AgentResponse(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        owner_id=agent.owner_id,
        organization_id=agent.organization_id,
        is_active=agent.is_active,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
        number_of_conversations=conversation_count
    )


@router.get("/agents", response_model=PaginatedAgentsOut)
def list_agents(
    user_id: int = Query(..., description="ID of the user whose agents to list"),
    organization_ids: Optional[List[int]] = Query(None, description="Filter agents by one or more organization IDs"),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search agents by name"),
    db: Session = Depends(get_session)
) -> PaginatedAgentsOut:
    """List all agents for a specific user or organization(s)."""
    # Verify user exists
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    filters = [
        AIAgent.deleted == False,  # noqa: E712
    ]
    if organization_ids:
        filters.append(AIAgent.organization_id.in_(organization_ids))
    else:
        filters.append(AIAgent.owner_id == user_id)

    search_term = search.strip() if search else None
    if search_term:
        filters.append(AIAgent.name.ilike(f"%{search_term}%"))

    offset = (page - 1) * limit
    agents = db.exec(
        select(AIAgent, func.count().over().label("total_count"))
        .where(*filters)
        .order_by(func.lower(AIAgent.name), AIAgent.id)
        .offset(offset)
        .limit(limit)
    ).all()

    total = agents[0].total_count if agents else 0
    items = [_compute_agent_response(agent, db) for agent, _total_count in agents]
    return PaginatedAgentsOut(
        items=items,
        page=page,
        limit=limit,
        total=total,
        has_more=(offset + len(items)) < total,
    )


@router.get("/agents/{agent_id}", response_model=AgentResponse)
def get_agent(
    agent_id: str,
    db: Session = Depends(get_session)
):
    """Get a specific agent by ID."""
    agent = db.get(AIAgent, agent_id)
    if not agent or agent.deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    return _compute_agent_response(agent, db)


@router.put("/agents/{agent_id}/settings", response_model=AIAgentSettings)
def update_agent_settings(
    agent_id: str,
    settings_update: AgentSettingsUpdate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_session)
):
    """
    Update agent settings.
    
    You can update any of the following settings:
    - llm_provider: The LLM provider to use
    - llm_model: The LLM model to use
    - llm_temperature: Temperature for model responses
    - llm_max_tokens: Maximum number of tokens to generate
    - llm_system_prompt: System prompt (can be a string or JSON object if llm_json_prompt is True)
    - llm_voice_prompt: Voice prompt (can be a string or JSON object if llm_json_prompt is True)
    - llm_json_prompt: Whether to use JSON format for prompts
    - preserve_memory: Whether to preserve conversation history
    - allow_websearch: Whether to allow web search
    
    Only include the fields you want to update. Omitted fields will remain unchanged.
    """
    # Verify agent exists and is not deleted
    agent = db.get(AIAgent, agent_id)
    if not agent or agent.deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    
    # Get or create settings
    settings = db.exec(
        select(AIAgentSettings).where(AIAgentSettings.agent_id == agent_id)
    ).first()
    
    if not settings:
        settings = AIAgentSettings(agent_id=agent_id)
        db.add(settings)
    # Update agent fields if provided
    agent_field_changed = False
    if settings_update.name is not None:
        agent.name = settings_update.name
        agent_field_changed = True
    if settings_update.description is not None:
        agent.description = settings_update.description
        agent_field_changed = True
    if settings_update.is_active is not None:
        agent.is_active = settings_update.is_active
        agent_field_changed = True
    if agent_field_changed:
        agent.updated_at = datetime.now(timezone.utc)

    # Update settings fields that were provided
    update_data = settings_update.dict(exclude_unset=True, exclude={'name', 'description', 'is_active'})
    if 'llm_provider' in update_data and update_data['llm_provider'] is not None:
        update_data['llm_provider'] = update_data['llm_provider'].strip().lower()

    # Handle llm_system_prompt JSON conversion if provided
    if 'llm_system_prompt' in update_data:
        system_prompt = update_data['llm_system_prompt']
        if system_prompt is not None:
            if settings_update.llm_json_prompt:
                if isinstance(system_prompt, dict):
                    # Convert dict to JSON string
                    update_data['llm_system_prompt'] = json.dumps(system_prompt, ensure_ascii=False)
                elif isinstance(system_prompt, str):
                    # Validate string is valid JSON if json_prompt is True
                    try:
                        parsed = json.loads(system_prompt)
                        update_data['llm_system_prompt'] = json.dumps(parsed, ensure_ascii=False)
                    except json.JSONDecodeError:
                        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"detail": [{"type": "json_invalid", "loc": ["body", "llm_system_prompt"], "msg": "Invalid JSON format"}]})
    
    # Handle mcp_server_configs: replace existing MCP server configs and links
    mcp_configs_changed = False
    if 'mcp_server_configs' in update_data:
        update_data.pop('mcp_server_configs')
        mcp_configs = settings_update.mcp_server_configs
        if mcp_configs is not None:
            mcp_configs_changed = True
            # Remove existing links and their MCP server config rows
            existing_links = db.exec(
                select(AgentMCPConfigLink).where(AgentMCPConfigLink.agent_settings_id == settings.id)
            ).all()
            old_mcp_ids = [link.mcp_server_config_id for link in existing_links]
            for link in existing_links:
                db.delete(link)
            for old_id in old_mcp_ids:
                old_srv = db.get(MCPServerConfig, old_id)
                if old_srv:
                    db.delete(old_srv)
            # Create new MCP server configs and links
            for cfg in mcp_configs:
                mcp_srv = MCPServerConfig(
                    user_id=agent.owner_id,
                    name=cfg.name,
                    url=cfg.url,
                    description=cfg.description,
                    requires_query_building=cfg.requires_query_building,
                )
                db.add(mcp_srv)
                db.flush()
                db.add(AgentMCPConfigLink(
                    agent_settings_id=settings.id,
                    mcp_server_config_id=mcp_srv.id,
                ))
    # Apply all updates
    for field, value in update_data.items():
        setattr(settings, field, value)
    
    # Update the updated_at timestamp
    settings.updated_at = datetime.now(timezone.utc)
    
    # Commit all changes
    # Log settings update
    changed_fields = set(update_data.keys())
    if changed_fields:
        log_activity(
            db_session=db,
            user_id=agent.owner_id,
            action=ActionType.UPDATE,
            entity_type=EntityType.SETTINGS,
            entity_id=str(settings.id),
            metadata={"fields": sorted(changed_fields)},
        )

    db.commit()
    db.refresh(settings)
    if settings_update.description is not None:
        db.refresh(agent)

    # Schedule FAQ regeneration in the background if MCP configs changed
    if mcp_configs_changed:
        log.info("MCP configs changed for agent '%s', scheduling FAQ regeneration", agent_id)
        from ai_agent.workflows.faq.faq import generate_faqs
        background_tasks.add_task(generate_faqs, agent_id, db, True)

    return settings


class BusinessNumberUpdate(BaseModel):
    """Request body for updating an agent's WhatsApp business number."""
    business_number: Optional[str] = Field(
        default=None,
        description=(
            "Business WhatsApp number that receives incoming messages for this agent "
            "(E.164 digits, e.g. '919818894299'). Pass null or empty string to unbind."
        ),
    )


@router.put("/agents/{agent_id}/business-number", response_model=AIAgentSettings)
def update_agent_business_number(
    agent_id: str,
    payload: BusinessNumberUpdate,
    db: Session = Depends(get_session),
):
    """
    Assign, update, or remove the WhatsApp business number for an agent.

    - Pass a valid E.164 digit string (e.g. ``"919818894299"``) to bind a number.
    - Pass ``null`` or an empty string to unbind the current number.
    - Returns ``409 Conflict`` if the number is already in use by another agent.
    """
    agent = db.get(AIAgent, agent_id)
    if not agent or agent.deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    settings = db.exec(
        select(AIAgentSettings).where(AIAgentSettings.agent_id == agent_id)
    ).first()
    if not settings:
        settings = AIAgentSettings(agent_id=agent_id)

    raw_bn = payload.business_number.strip() if payload.business_number else None
    if not raw_bn:
        settings.business_number = None
    else:
        digits = raw_bn.lstrip("+").strip()
        if not digits.isdigit():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="business_number must contain only digits",
            )
        # Enforce uniqueness across agents
        existing = db.exec(
            select(AIAgentSettings).where(
                AIAgentSettings.business_number == digits,
                AIAgentSettings.agent_id != agent_id,
            )
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This business number is already in use by another agent.",
            )
        settings.business_number = digits

    settings.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(settings)
    return settings


@router.get("/agents/{agent_id}/settings")
def get_agent_settings(
    agent_id: str,
    db: Session = Depends(get_session)
):
    """Get settings for a specific agent including agent info."""
    # Verify agent exists and is not deleted
    agent = db.get(AIAgent, agent_id)
    if not agent or agent.deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
        
    settings = db.exec(
        select(AIAgentSettings).where(AIAgentSettings.agent_id == agent_id)
    ).first()
    
    if not settings:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Settings not found for this agent")

    # Fetch MCP server configs linked to this agent's settings
    mcp_links = db.exec(
        select(AgentMCPConfigLink).where(AgentMCPConfigLink.agent_settings_id == settings.id)
    ).all()
    mcp_configs_data = []
    for link in mcp_links:
        mcp_srv = db.get(MCPServerConfig, link.mcp_server_config_id)
        if mcp_srv:
            mcp_configs_data.append({
                "id": mcp_srv.id,
                "name": mcp_srv.name,
                "url": mcp_srv.url,
                "description": mcp_srv.description,
                "requires_query_building": mcp_srv.requires_query_building,
            })

    # Return combined agent and settings data
    return {
        # Agent info
        "id": agent.id,
        "name": agent.name,
        "description": agent.description,
        "organization_id": agent.organization_id,
        "is_active": agent.is_active,
        "created_at": agent.created_at,
        "updated_at": agent.updated_at,
        "has_data_store": agent.has_data_store if hasattr(agent, 'has_data_store') else None,
        # Settings
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_temperature": settings.llm_temperature,
        "llm_max_tokens": settings.llm_max_tokens,
        "llm_system_prompt": settings.llm_system_prompt,
        "llm_voice_prompt": settings.llm_voice_prompt,
        "llm_json_prompt": settings.llm_json_prompt,
        "preserve_memory": settings.preserve_memory,
        "allow_websearch": settings.allow_websearch,
        "websearch_provider": settings.websearch_provider,
        "websearch_api_key_id": settings.websearch_api_key_id,
        "websearch_model": settings.websearch_model,
        "api_key": settings.api_key,
        "is_mcp_enabled": settings.is_mcp_enabled,
        "whatsapp_enabled": settings.whatsapp_enabled,
        "business_number": settings.business_number,
        # Nested configs
        "mcp_server_configs": mcp_configs_data,
    }


@router.delete("/agents/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_agent(
    agent_id: str,
    db: Session = Depends(get_session)
):
    """
    Soft delete an agent by marking it as deleted.
    The agent will not be returned in list endpoints unless explicitly requested.
    """
    agent = db.get(AIAgent, agent_id)
    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    
    if agent.deleted:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Agent is already deleted")
    
    agent.deleted = True
    agent.updated_at = datetime.now(timezone.utc)
    db.add(agent)

    # Remove all document-agent links for this agent.
    # Documents themselves are not affected — they are managed independently
    # via OrganizationDocument and will remain accessible to other agents.
    links = db.exec(select(AgentDocument).where(AgentDocument.agent_id == agent_id)).all()
    for link in links:
        db.delete(link)

    # Log activity
    log_activity(
        db_session=db,
        user_id=agent.owner_id,
        action=ActionType.DELETE,
        entity_type=EntityType.AGENT,
        entity_id=agent_id,
    )
    db.commit()

    return None



# Agent FAQs

class FAQItem(BaseModel):
    """A single FAQ question for an agent."""
    question: str = Field(..., description="The FAQ question text")
    source: str = Field(..., description="How this FAQ was generated: 'agent_config' or 'chat_history'")


@router.get(
    "/agents/{agent_id}/faqs",
    response_model=List[FAQItem],
    summary="Get dynamically generated FAQs for an agent",
    dependencies=[Depends(security)],
)
async def get_agent_faqs(
    agent_id: str,
    refresh: bool = Query(False, description="Force regeneration of FAQs"),
    db: Session = Depends(get_session),
):
    """Get dynamically generated FAQs for an agent.

    FAQs are generated based on the agent's configuration (name, description,
    system prompt, MCP tools/DB schemas) and past conversation data. Results
    are cached and reused until ``?refresh=true`` is passed.
    """
    # Verify agent exists
    agent = db.get(AIAgent, agent_id)
    if not agent or agent.deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    from ai_agent.workflows.faq.faq import generate_faqs
    from ai_agent.services.faq_service import get_cached_faqs

    try:
        if not refresh:
            cached = get_cached_faqs(agent_id, db)
            if cached:
                return [FAQItem(question=f["question"], source=f["source"]) for f in cached]
                
        faqs = await generate_faqs(agent_id, db, force_refresh=refresh)
        return [FAQItem(question=f["question"], source=f["source"]) for f in faqs]
    except Exception as e:
        log.error("FAQ generation failed for agent %s: %s", agent_id, e)
        # Return empty list rather than failing the whole request
        return []


class FAQRefreshRequest(BaseModel):
    """Request payload for FAQ refresh endpoint."""
    agent_ids: List[str] = Field(..., min_length=1, description="List of agent IDs to refresh")

class FAQRefreshResponse(BaseModel):
    """Response from FAQ refresh endpoint."""
    success: bool
    message: str
    agents_processed: int
    agents_failed: List[str] = Field(default_factory=list)


@router.post(
    "/agents/faqs/refresh",
    response_model=FAQRefreshResponse,
    summary="Refresh FAQs for multiple agents",
    dependencies=[Depends(security)],
)
async def refresh_faqs(
    request: FAQRefreshRequest,
    db: Session = Depends(get_session),
):
    """Unified FAQ refresh endpoint supporting bulk refresh.
    
    Usage examples:
    - Manual refresh after significant conversation history accumulation
    - Bulk migration of existing agents to schema-aware FAQs
    
    Request body:
        agent_ids: List of agent IDs to refresh
    
    Returns:
        200: FAQs successfully regenerated
        404: If any agent is not found, will process valid ones and report missing in failed
    """
    from ai_agent.workflows.faq.faq import generate_faqs

    processed = 0
    failed = []

    for agent_id in request.agent_ids:
        agent = db.get(AIAgent, agent_id)
        if not agent or agent.deleted:
            log.warning("Agent '%s' not found for FAQ refresh", agent_id)
            failed.append(agent_id)
            continue

        try:
            log.info("Bulk FAQ refresh: processing agent '%s' (%s)", agent.name, agent.id)
            await generate_faqs(agent.id, db, force_refresh=True)
            processed += 1
        except Exception as e:
            log.error("Bulk FAQ refresh failed for agent %s: %s", agent.id, e)
            failed.append(agent.id)

    success = len(failed) == 0
    
    if success:
        message = f"Successfully refreshed FAQs for {processed} agent(s)"
    else:
        message = f"Refreshed {processed} agent(s), {len(failed)} failed"

    return FAQRefreshResponse(
        success=success,
        message=message,
        agents_processed=processed,
        agents_failed=failed
    )



class APIKeyInfo(BaseModel):
    """API key info without the actual key (masked)."""
    provider: str
    has_key: bool
    last_used: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class APIKeySetRequest(BaseModel):
    """Request to set an API key."""
    provider: str = Field(..., description="Provider name (e.g., 'openai', 'anthropic', 'google')")
    api_key: str = Field(..., description="The API key to store (will be encrypted)")


class APIKeySetResponse(BaseModel):
    """Response after setting an API key."""
    success: bool
    provider: str
    message: str


@router.get("/agents/{agent_id}/api-keys", response_model=ListType[APIKeyInfo])
def get_agent_api_keys(
    agent_id: str,
    db: Session = Depends(get_session)
):
    """
    Get all API keys configured for an agent (keys are masked).
    
    Returns a list of providers and whether each has a key configured.
    """
    # Verify agent exists
    agent = db.get(AIAgent, agent_id)
    if not agent or agent.deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    
    # Get agent settings
    settings = db.exec(
        select(AIAgentSettings).where(AIAgentSettings.agent_id == agent_id)
    ).first()
    
    if not settings:
        return []
    
    # Load API keys
    api_keys = db.exec(
        select(ApiKey).where(ApiKey.agent_settings_id == settings.id)
    ).all()
    
    return [
        APIKeyInfo(
            provider=key.provider,
            has_key=True,
            last_used=key.last_used,
            created_at=key.created_at,
            updated_at=key.updated_at,
        )
    ]


@router.post("/agents/{agent_id}/api-keys", response_model=APIKeySetResponse)
def set_agent_api_key(
    agent_id: str,
    request: APIKeySetRequest,
    db: Session = Depends(get_session)
):
    """
    Set or update an API key for a specific provider on an agent.
    
    The API key will be encrypted before storage.
    """
    # Verify agent exists
    agent = db.get(AIAgent, agent_id)
    if not agent or agent.deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    
    # Get or create agent settings
    settings = db.exec(
        select(AIAgentSettings).where(AIAgentSettings.agent_id == agent_id)
    ).first()
    
    if not settings:
        # Create default settings
        settings = AIAgentSettings(
            agent_id=agent_id,
            llm_provider=settings.LLM_PROVIDER if 'settings' in locals() else "openai",
            llm_model=settings.LLM_MODEL if 'settings' in locals() else "gpt-4o",
            llm_temperature=0.7,
            llm_max_tokens=1000,
        )
        db.add(settings)
        db.flush()
    
    # Check if key already exists
    existing_key = db.exec(
        select(ApiKey).where(
            ApiKey.agent_settings_id == settings.id,
            ApiKey.provider == request.provider
        )
    ).first()
    
    if existing_key:
        # Update existing key
        existing_key.encrypt(request.api_key)
        existing_key.updated_at = datetime.now(timezone.utc)
        db.add(existing_key)
        message = f"Updated API key for {request.provider}"
    else:
        # Create new key
        new_key = ApiKey(
            agent_settings_id=settings.id,
            provider=request.provider,
        )
        new_key.encrypt(request.api_key)
        db.add(new_key)
        message = f"Added API key for {request.provider}"
    
    # Log activity
    db.flush()
    api_key = existing_key if existing_key else new_key
    action = ActionType.UPDATE if existing_key else ActionType.CREATE
    log_activity(
        db_session=db,
        user_id=agent.owner_id,
        action=action,
        entity_type=EntityType.API_KEY,
        entity_id=str(api_key.id),
        metadata={"provider": request.provider},
    )
    db.commit()

    return APIKeySetResponse(
        success=True,
        provider=request.provider,
        message=message
    )


@router.delete("/agents/{agent_id}/api-keys/{provider}", status_code=status.HTTP_204_NO_CONTENT)
def delete_agent_api_key(
    agent_id: str,
    provider: str,
    db: Session = Depends(get_session)
):
    """
    Delete an API key for a specific provider on an agent.
    """
    # Verify agent exists
    agent = db.get(AIAgent, agent_id)
    if not agent or agent.deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    
    # Get agent settings
    settings = db.exec(
        select(AIAgentSettings).where(AIAgentSettings.agent_id == agent_id)
    ).first()
    
    if not settings:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Settings not found for this agent")
    
    # Find the API key
    api_key = db.exec(
        select(ApiKey).where(
            ApiKey.agent_settings_id == settings.id,
            ApiKey.provider == provider
        )
    ).first()
    
    if not api_key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No API key found for provider '{provider}'")
    
    api_key_id = api_key.id
    db.delete(api_key)
    # Log activity
    log_activity(
        db_session=db,
        user_id=agent.owner_id,
        action=ActionType.DELETE,
        entity_type=EntityType.API_KEY,
        entity_id=str(api_key_id),
        metadata={"provider": provider},
    )
    db.commit()

    return None


@router.post("/test-mcp-connection", response_model=MCPTestResponse)
async def test_mcp_connection(body: MCPServerInput):
    """
    Test an MCP server connection without saving anything.
    Spawns the server, performs the MCP handshake, lists available tools,
    then disconnects. Returns success status and discovered tools.
    """
    from ai_agent.core.mcp_manager import get_mcp_manager

    mcp_mgr = get_mcp_manager()
    result = await mcp_mgr.test_connection(
        {"name": body.name, "url": body.url},
        requires_query_building=body.requires_query_building,
    )

    if result["success"]:
        tools = result.get("tools", [])
        return MCPTestResponse(
            success=True,
            message=f"Connection successful. {len(tools)} tool(s) discovered.",
        )
    else:
        return MCPTestResponse(
            success=False,
            message=result.get("message", "Unknown error"),
        )


# ---------------------------------------------------------------------------
# WhatsApp Number Management
# ---------------------------------------------------------------------------

class WhatsappNumberCreate(BaseModel):
    """Payload for adding a new WhatsApp number to an agent."""
    phone_number: str = Field(..., description="Phone in E.164 digits, e.g. '919876543210'")
    display_label: Optional[str] = Field(default=None, description="Optional label, e.g. 'Support Line'")
    user_id: int = Field(..., description="ID of the logged-in user adding this number")


class WhatsappNumberUpdate(BaseModel):
    """Payload for updating a WhatsApp number record (all fields optional)."""
    display_label: Optional[str] = None
    is_active: Optional[bool] = None


class WhatsappNumberResponse(BaseModel):
    """Public representation of an AgentWhatsappNumber row."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    agent_id: str
    user_id: int
    phone_number: str
    display_label: Optional[str]
    is_active: bool
    welcome_message_sent: bool
    created_at: datetime
    updated_at: datetime


@router.get(
    "/agents/{agent_id}/whatsapp-numbers",
    response_model=List[WhatsappNumberResponse],
    summary="List WhatsApp numbers linked to an agent",
)
def list_whatsapp_numbers(
    agent_id: str,
    db: Session = Depends(get_session),
):
    """Return all WhatsApp numbers registered for the given agent."""
    agent = db.get(AIAgent, agent_id)
    if not agent or agent.deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    rows = db.exec(
        select(AgentWhatsappNumber)
        .where(AgentWhatsappNumber.agent_id == agent_id)
        .order_by(AgentWhatsappNumber.created_at)
    ).all()
    return rows


@router.post(
    "/agents/{agent_id}/whatsapp-numbers",
    response_model=WhatsappNumberResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a WhatsApp number to an agent",
)
def add_whatsapp_number(
    agent_id: str,
    body: WhatsappNumberCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_session),
):
    """
    Link a new WhatsApp number (Gallabox channel) to an agent.
    Both phone_number and gallabox_channel_id must be globally unique.
    """
    agent = db.get(AIAgent, agent_id)
    if not agent or agent.deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    # Validate the user exists
    user = db.get(User, body.user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Ensure agent has a business number configured before allowing user numbers
    agent_settings = db.exec(select(AIAgentSettings).where(AIAgentSettings.agent_id == agent_id)).first()
    if not agent_settings or not agent_settings.business_number:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You must configure a Business Number for this agent before adding user numbers.",
        )

    # Normalize phone to digits only
    normalized_phone = re.sub(r"[^\d]", "", body.phone_number)
    if not normalized_phone:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="phone_number must contain at least one digit",
        )

    if normalized_phone == agent_settings.business_number:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The user number cannot be the same as the agent's business number.",
        )

    # Check uniqueness: same phone cannot be added to the SAME agent twice.
    # (The same phone CAN be linked to different agents via different business numbers.)
    existing_phone = db.exec(
        select(AgentWhatsappNumber).where(
            AgentWhatsappNumber.phone_number == normalized_phone,
            AgentWhatsappNumber.agent_id == agent_id,
        )
    ).first()
    if existing_phone:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Phone number '{normalized_phone}' is already linked to this agent.",
        )

    new_number = AgentWhatsappNumber(
        agent_id=agent_id,
        user_id=body.user_id,
        phone_number=normalized_phone,
        display_label=body.display_label,
        is_active=True,
        welcome_message_sent=False,
    )
    db.add(new_number)
    db.commit()
    db.refresh(new_number)

    log.info(
        "Added WhatsApp number '%s' to agent '%s' by user %d",
        normalized_phone, agent_id, body.user_id,
    )

    # Capture plain Python values BEFORE the session expires them so that the
    # background task closure does not trigger a lazy-load on a dead session
    # (which causes SQLAlchemy's _load_expired error).
    number_id = new_number.id
    already_sent = new_number.welcome_message_sent
    recipient_name = user.first_name or "WhatsApp User"
    agent_name = agent.name

    # Send the approved Gallabox welcome_template — best-effort, does not block
    # the response.  Guarded by welcome_message_sent to avoid duplicate sends.
    if not already_sent:
        session_bind = db.get_bind()

        async def _send_and_mark_sent() -> None:
            """Fire the template then flip the flag on success."""
            sent = await send_whatsapp_template_message(
                phone_number=normalized_phone,
                recipient_name=recipient_name,
                agent_name=agent_name,
            )
            if sent:
                # Re-open a fresh DB connection inside the background task
                from sqlmodel import Session as _Session
                with _Session(session_bind) as bg_db:
                    bg_number = bg_db.get(AgentWhatsappNumber, number_id)
                    if bg_number:
                        bg_number.welcome_message_sent = True
                        bg_db.add(bg_number)
                        bg_db.commit()
                        log.info(
                            "Marked welcome_message_sent=True for WhatsApp number id=%d ('%s')",
                            number_id,
                            normalized_phone,
                        )

        background_tasks.add_task(_send_and_mark_sent)

    return new_number


@router.patch(
    "/agents/{agent_id}/whatsapp-numbers/{number_id}",
    response_model=WhatsappNumberResponse,
    summary="Update a WhatsApp number record",
)
def update_whatsapp_number(
    agent_id: str,
    number_id: int,
    body: WhatsappNumberUpdate,
    db: Session = Depends(get_session),
):
    """Update label, credentials, or active status of a WhatsApp number."""
    agent = db.get(AIAgent, agent_id)
    if not agent or agent.deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    number = db.exec(
        select(AgentWhatsappNumber)
        .where(AgentWhatsappNumber.id == number_id, AgentWhatsappNumber.agent_id == agent_id)
    ).first()
    if not number:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="WhatsApp number not found")



    update_data = body.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(number, field, value)
    number.updated_at = datetime.now(timezone.utc)

    db.add(number)
    db.commit()
    db.refresh(number)
    return number


@router.delete(
    "/agents/{agent_id}/whatsapp-numbers/{number_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a WhatsApp number from an agent",
)
def delete_whatsapp_number(
    agent_id: str,
    number_id: int,
    db: Session = Depends(get_session),
):
    """Permanently remove a WhatsApp number from an agent."""
    agent = db.get(AIAgent, agent_id)
    if not agent or agent.deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    number = db.exec(
        select(AgentWhatsappNumber)
        .where(AgentWhatsappNumber.id == number_id, AgentWhatsappNumber.agent_id == agent_id)
    ).first()
    if not number:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="WhatsApp number not found")

    db.delete(number)
    db.commit()
    log.info("Deleted WhatsApp number id=%d from agent '%s'", number_id, agent_id)
    return None
