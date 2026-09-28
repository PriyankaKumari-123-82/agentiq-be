"""Agent-scoped MCP server configuration endpoints."""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlmodel import Session, select

from ai_agent.core import db
from ai_agent.core.deps import get_current_user
from ai_agent.core.token_validation import security
from ai_agent.models import AIAgent, AIAgentSettings, AgentMCPConfigLink, MCPServerConfig, User
from ai_agent.schema_engine.orchestrator import get_orchestrator
from ai_agent.schema_engine.store.pg_repository import SchemaEngineRepository
from ai_agent.services.api_keys import get_api_key
from ai_agent.services.schema_classifier_service import SchemaClassifierService

log = logging.getLogger(__name__)

router = APIRouter(
    tags=[" MCP "],
    dependencies=[Depends(security)],
)


class MCPServerConfigCreate(BaseModel):
    name: str = Field(..., min_length=1, description="Display name for this MCP server")
    url: str = Field(..., min_length=1, description="MCP-URL")
    description: str = Field(..., description="Description of what this MCP server exposes.")
    requires_query_building: bool = Field(
        default=False
    )

    @field_validator("name", "url", "description", mode="before")
    @classmethod
    def strip_string_fields(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value


class MCPServerConfigPatch(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1)
    url: Optional[str] = Field(default=None, min_length=1)
    description: Optional[str] = None
    requires_query_building: Optional[bool] = None

    @field_validator("name", "url", "description", mode="before")
    @classmethod
    def strip_string_fields(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value


class MCPServerConfigResponse(BaseModel):
    id: int
    name: str
    url: str
    description: str
    requires_query_building: bool
    created_at: datetime
    updated_at: datetime


def _to_response(config: MCPServerConfig) -> MCPServerConfigResponse:
    return MCPServerConfigResponse(
        id=config.id,
        name=config.name,
        url=config.url,
        description=config.description,
        requires_query_building=config.requires_query_building,
        created_at=config.created_at,
        updated_at=config.updated_at,
    )


def _get_owned_agent_settings(
    agent_id: str,
    user: User,
    db_session: Session,
) -> AIAgentSettings:
    agent = db_session.get(AIAgent, agent_id)
    if not agent or agent.deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    if agent.owner_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    settings = db_session.exec(
        select(AIAgentSettings).where(AIAgentSettings.agent_id == agent_id)
    ).first()
    if not settings:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Settings not found for this agent")
    return settings


def _get_linked_config(
    settings: AIAgentSettings,
    config_id: int,
    db_session: Session,
) -> tuple[MCPServerConfig, AgentMCPConfigLink]:
    link = db_session.exec(
        select(AgentMCPConfigLink).where(
            AgentMCPConfigLink.agent_settings_id == settings.id,
            AgentMCPConfigLink.mcp_server_config_id == config_id,
        )
    ).first()
    if not link:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MCP server config not found for this agent")

    config = db_session.get(MCPServerConfig, config_id)
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MCP server config not found")
    return config, link


def _agent_has_config_with_url(
    settings: AIAgentSettings,
    url: str,
    db_session: Session,
    exclude_config_id: int | None = None,
) -> bool:
    query = (
        select(MCPServerConfig)
        .join(
            AgentMCPConfigLink,
            AgentMCPConfigLink.mcp_server_config_id == MCPServerConfig.id,
        )
        .where(
            AgentMCPConfigLink.agent_settings_id == settings.id,
            MCPServerConfig.url == url,
        )
    )
    if exclude_config_id is not None:
        query = query.where(MCPServerConfig.id != exclude_config_id)
    return db_session.exec(query).first() is not None


@router.get("/{agent_id}/mcp-servers", response_model=list[MCPServerConfigResponse])
def list_agent_mcp_servers(
    agent_id: str,
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    settings = _get_owned_agent_settings(agent_id, user, db_session)
    configs = db_session.exec(
        select(MCPServerConfig)
        .join(
            AgentMCPConfigLink,
            AgentMCPConfigLink.mcp_server_config_id == MCPServerConfig.id,
        )
        .where(AgentMCPConfigLink.agent_settings_id == settings.id)
        .order_by(MCPServerConfig.created_at.asc())
    ).all()
    return [_to_response(config) for config in configs]


@router.post("/{agent_id}/mcp-servers", response_model=MCPServerConfigResponse, status_code=status.HTTP_201_CREATED)
def add_agent_mcp_server(
    agent_id: str,
    body: MCPServerConfigCreate,
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    settings = _get_owned_agent_settings(agent_id, user, db_session)
    if _agent_has_config_with_url(settings, body.url, db_session):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This agent already has an MCP server config with this URL",
        )

    config = MCPServerConfig(
        user_id=user.id,
        name=body.name,
        url=body.url,
        description=body.description,
        requires_query_building=body.requires_query_building,
    )
    db_session.add(config)
    db_session.flush()
    db_session.add(AgentMCPConfigLink(
        agent_settings_id=settings.id,
        mcp_server_config_id=config.id,
    ))
    db_session.commit()
    db_session.refresh(config)
    return _to_response(config)


@router.patch("/{agent_id}/mcp-servers/{config_id}", response_model=MCPServerConfigResponse)
def update_agent_mcp_server(
    agent_id: str,
    config_id: int,
    body: MCPServerConfigPatch,
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    settings = _get_owned_agent_settings(agent_id, user, db_session)
    config, _link = _get_linked_config(settings, config_id, db_session)
    update_data = body.model_dump(exclude_unset=True)

    if not update_data:
        return _to_response(config)

    next_url = update_data.get("url", config.url)
    if _agent_has_config_with_url(settings, next_url, db_session, exclude_config_id=config.id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This agent already has an MCP server config with this URL",
        )

    for field, value in update_data.items():
        setattr(config, field, value)
    config.updated_at = datetime.now(timezone.utc)
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)
    return _to_response(config)


@router.delete("/{agent_id}/mcp-servers/{config_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_agent_mcp_server(
    agent_id: str,
    config_id: int,
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    settings = _get_owned_agent_settings(agent_id, user, db_session)
    config, link = _get_linked_config(settings, config_id, db_session)

    db_session.delete(link)
    db_session.delete(config)
    db_session.commit()
    return None


def _get_mcp_servers(settings: AIAgentSettings) -> list[MCPServerConfig]:
    """MCP server configs linked to this agent that expose a generic DB (query-builder)."""
    return [
        link.mcp_server_config
        for link in settings.mcp_links
        if link.mcp_server_config.requires_query_building
    ]


@router.get("/agents/{agent_id}/classified_schema")
def get_agent_classified_schema(
    agent_id: str,
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    """Table categories + sample questions for this agent's query-builder DB(s).

    Populated once during cold build (or via POST /agents/{agent_id}/classified_schema).
    Ordered by priority — most business-critical category first.
    """
    settings = _get_owned_agent_settings(agent_id, user, db_session)
    servers = _get_mcp_servers(settings)
    repo = SchemaEngineRepository(db_session)

    service = SchemaClassifierService(servers)
    return {"categories": service.get_classified_schema(repo)}


@router.post("/agents/{agent_id}/classified_schema")
async def classify_agent_schema(
    agent_id: str,
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    """Manually (re)build table categories + sample questions for this agent.

    For agents cold-built before the schema classifier existed, or to refresh
    categories after the underlying schema changed. Requires the agent's
    schema to already be cold-built (chat with it at least once first) —
    reuses the already-extracted/summarized schema, so this itself is cheap.
    """
    log.info("POST /agents/%s/classified_schema called", agent_id)

    settings = _get_owned_agent_settings(agent_id, user, db_session)
    servers = _get_mcp_servers(settings)
    if not servers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This agent has no query-builder-enabled MCP servers to categorize.",
        )

    if not settings.agent.is_cold_build:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Schema hasn't been built yet for this agent. Open a chat "
                "with it first to trigger the initial schema build."
            ),
        )

    api_key = get_api_key(db_session, settings.api_key)
    repo = SchemaEngineRepository(db_session)
    service = SchemaClassifierService(servers)

    results = {}
    for srv in servers:
        orch = await get_orchestrator(
            server_name=srv.name, mcp_manager=None, server_tools=[],
            agent_id=agent_id, api_key=api_key,
            model=settings.llm_model, provider=settings.llm_provider,
        )
        categories = await service.save_classified_schema(
            repo, srv.name, orch._meta, orch._chunks, api_key=api_key,
        )
        db_session.commit()
        results[srv.name] = len(categories)

    return {"categorized": results}
