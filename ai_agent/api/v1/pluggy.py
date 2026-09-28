from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import EmailStr
from sqlmodel import Session, select

from ai_agent.core import db
from ai_agent.core.config import settings
from ai_agent.core.deps import get_current_user
from ai_agent.core.token_validation import security
from ai_agent.models import User, Organization, AIAgent, MCPServerConfig, AgentMCPConfigLink, AIAgentSettings
from ai_agent.services import pluggy_service


router = APIRouter(tags=["Pluggy"])


@router.get(
    "/connectors",
    summary="Get all available Pluggy connectors",
    dependencies=[Depends(security)],
)
async def get_connectors():
    """Returns a list of all global available Pluggy connectors."""
    return await pluggy_service.get_connectors()


@router.get(
    "/connectors/personalized",
    summary="Get personalized Pluggy connectors for the active organization",
    dependencies=[Depends(security)],
)
async def get_personalized_connectors(
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    """
    Returns a list of personalized (connected) Pluggy connectors for the user's
    active organization.
    """
    if not user.active_organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User does not have an active organization."
        )

    org = db_session.get(Organization, user.active_organization_id)
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")
        
    if not org.pluggy_tenant_id:
        return []

    connectors = await pluggy_service.get_personalized_connectors(org.pluggy_tenant_id)
    
    # Augment with linked agent count
    for connector in connectors:
        connector_key = connector.get("connector")
        if not connector_key:
            connector["linked_agent_count"] = 0
            continue
            
        count = db_session.exec(
            select(AIAgent)
            .join(AIAgentSettings, AIAgent.id == AIAgentSettings.agent_id)
            .join(AgentMCPConfigLink, AgentMCPConfigLink.agent_settings_id == AIAgentSettings.id)
            .join(MCPServerConfig, MCPServerConfig.id == AgentMCPConfigLink.mcp_server_config_id)
            .where(
                AIAgent.organization_id == org.id,
                AIAgent.deleted == False,
                MCPServerConfig.name == connector_key
            )
        ).all()
        connector["linked_agent_count"] = len(count)
        
    return connectors


@router.get(
    "/connectors/personalized/{connector_key}/agents",
    summary="Get agents linked to a personalized connector",
    dependencies=[Depends(security)],
)
async def get_connector_agents(
    connector_key: str,
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    if not user.active_organization_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User does not have an active organization.")
        
    # Get all active agents in the organization
    agents = db_session.exec(
        select(AIAgent)
        .where(
            AIAgent.organization_id == user.active_organization_id,
            AIAgent.deleted == False
        )
    ).all()
    
    result = []
    for agent in agents:
        # Check if linked
        is_linked = False
        if agent.settings:
            link = db_session.exec(
                select(AgentMCPConfigLink)
                .join(MCPServerConfig, MCPServerConfig.id == AgentMCPConfigLink.mcp_server_config_id)
                .where(
                    AgentMCPConfigLink.agent_settings_id == agent.settings.id,
                    MCPServerConfig.name == connector_key
                )
            ).first()
            if link:
                is_linked = True
                
        result.append({
            "agent_id": agent.id,
            "agent_name": agent.name,
            "is_linked": is_linked
        })
        
    return result


@router.post(
    "/connectors/personalized/{connector_key}/link-agent/{agent_id}",
    summary="Link an agent to a Pluggy connector",
    dependencies=[Depends(security)],
)
async def link_agent_to_connector(
    connector_key: str,
    agent_id: str,
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    if not user.active_organization_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User does not have an active organization.")
        
    org = db_session.get(Organization, user.active_organization_id)
    if not org or not org.pluggy_tenant_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Organization is missing a Pluggy tenant.")
        
    agent = db_session.get(AIAgent, agent_id)
    if not agent or agent.deleted or agent.organization_id != org.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found in active organization.")
        
    if not agent.settings:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Agent settings not found.")

    # Check if already linked
    existing_link = db_session.exec(
        select(AgentMCPConfigLink)
        .join(MCPServerConfig, MCPServerConfig.id == AgentMCPConfigLink.mcp_server_config_id)
        .where(
            AgentMCPConfigLink.agent_settings_id == agent.settings.id,
            MCPServerConfig.name == connector_key
        )
    ).first()
    
    if existing_link:
        return {"status": "success", "message": "Agent already linked"}
        
    # Get the server URL from Pluggy
    mcp_url = await pluggy_service.get_connector_mcp_url(org.pluggy_tenant_id, connector_key)
    if not mcp_url:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve MCP server URL from Pluggy.")

    # Determine if it is a DATABASE connector
    personalized_connectors = await pluggy_service.get_personalized_connectors(org.pluggy_tenant_id)
    connector_data = next((c for c in personalized_connectors if c.get("connector") == connector_key), None)
    
    requires_query_building = False
    if connector_data and connector_data.get("connectorType") == "DATABASE":
        requires_query_building = True

    # Find or create MCPServerConfig
    mcp_config = db_session.exec(
        select(MCPServerConfig)
        .where(
            MCPServerConfig.name == connector_key,
            MCPServerConfig.user_id == user.id,
            MCPServerConfig.url == mcp_url
        )
    ).first()
    
    if not mcp_config:
        mcp_config = MCPServerConfig(
            user_id=user.id,
            name=connector_key,
            url=mcp_url,
            description=f"Pluggy Connector: {connector_key}",
            requires_query_building=requires_query_building
        )
        db_session.add(mcp_config)
        db_session.flush()
    else:
        if mcp_config.requires_query_building != requires_query_building:
            mcp_config.requires_query_building = requires_query_building
            db_session.add(mcp_config)
            db_session.flush()
        
    # Create the link
    new_link = AgentMCPConfigLink(
        agent_settings_id=agent.settings.id,
        mcp_server_config_id=mcp_config.id
    )
    db_session.add(new_link)
    
    # Enable MCP on agent
    agent.settings.is_mcp_enabled = True
    db_session.add(agent.settings)
    
    db_session.commit()
    return {"status": "success", "message": f"Agent linked to {connector_key}"}


@router.delete(
    "/connectors/personalized/{connector_key}/link-agent/{agent_id}",
    summary="Unlink an agent from a Pluggy connector",
    dependencies=[Depends(security)],
)
async def unlink_agent_from_connector(
    connector_key: str,
    agent_id: str,
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    agent = db_session.get(AIAgent, agent_id)
    if not agent or agent.deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found.")
        
    if not agent.settings:
        return {"status": "success", "message": "Agent not linked"}
        
    # Find the link
    link = db_session.exec(
        select(AgentMCPConfigLink)
        .join(MCPServerConfig, MCPServerConfig.id == AgentMCPConfigLink.mcp_server_config_id)
        .where(
            AgentMCPConfigLink.agent_settings_id == agent.settings.id,
            MCPServerConfig.name == connector_key
        )
    ).first()
    
    if link:
        mcp_config_id = link.mcp_server_config_id
        db_session.delete(link)
        db_session.flush()
        
        # If no other agents use this MCP config, we can delete it
        other_links = db_session.exec(
            select(AgentMCPConfigLink).where(AgentMCPConfigLink.mcp_server_config_id == mcp_config_id)
        ).all()
        
        if not other_links:
            mcp_config = db_session.get(MCPServerConfig, mcp_config_id)
            if mcp_config:
                db_session.delete(mcp_config)
                
        # If agent has no other MCP links, disable MCP
        if not agent.settings.mcp_links: # wait, the relationship might not update instantly, lets manually count
            remaining_mcp = db_session.exec(
                select(AgentMCPConfigLink).where(AgentMCPConfigLink.agent_settings_id == agent.settings.id)
            ).all()
            if not remaining_mcp:
                agent.settings.is_mcp_enabled = False
                db_session.add(agent.settings)
                
        db_session.commit()
        
    return {"status": "success", "message": f"Agent unlinked from {connector_key}"}


# ---------------------------------------------------------------------------
# OAuth endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/connectors/{connector_key}/oauth-url",
    summary="Get the Pluggy OAuth URL for a connector",
    dependencies=[Depends(security)],
)
async def get_oauth_url(
    connector_key: str,
    redirect_url: str = Query(
        default=None,
        description="Frontend URL Pluggy will redirect to after OAuth completes",
    ),
    user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    """Return the Pluggy OAuth connect URL for the current user's organisation.

    The frontend opens this URL in a popup. After the user authenticates with
    Microsoft, Pluggy redirects back to ``redirect_url``.
    """
    if not user.active_organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User does not have an active organization.",
        )

    org = db_session.get(Organization, user.active_organization_id)
    if not org or not org.pluggy_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Organization is not linked to a Pluggy tenant.",
        )

    effective_redirect = redirect_url or settings.FRONTEND_URL

    oauth_url = pluggy_service.build_oauth_url(
        pluggy_tenant_id=org.pluggy_tenant_id,
        connector_id=settings.PLUGGY_MS_TEAMS_CONNECTOR_ID,
        redirect_url=effective_redirect,
    )

    import logging
    logging.getLogger(__name__).info(f"OAuth URL requested for connector '{connector_key}' (user_id={user.id}): {oauth_url}")

    return {"oauth_url": oauth_url}
