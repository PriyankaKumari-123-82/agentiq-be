"""MCP (Model Context Protocol) server interactions for the message flow.

Owns:
    - ``gather_mcp_tools`` — connects every MCP server linked to the agent
      and returns a flat ``[{server_name, tool_name, ...}, ...]`` list.
    - ``_extract_mcp_tool_output_text`` — flattens the heterogeneous
      ``content`` list returned by an MCP ``call_tool`` into a single text
      blob suitable for an LLM prompt.

This module is intentionally thin: it does not classify, route, or build
prompts. It is the boundary between the message flow and the MCP runtime.
"""
import logging
from typing import Any, Dict

from sqlmodel import Session, select

from ai_agent.core.mcp_manager import get_mcp_manager
from ai_agent.models import (
    AgentMCPConfigLink,
    AIAgentSettings,
    MCPServerConfig,
)

log = logging.getLogger(__file__)


async def gather_mcp_tools(
    agent_settings: AIAgentSettings,
    db_session: Session,
) -> list[dict]:
    """Connect to every MCP server linked to this agent and return a flat tool list.

    Each returned dict has keys::

        server_name, server_description, tool_name, description,
        input_schema, requires_query_building

    Servers that fail to connect are logged at WARN level and silently
    skipped, so a single bad server cannot take routing offline.
    """
    mcp_tools: list[dict] = []

    if not agent_settings or not agent_settings.is_mcp_enabled:
        return mcp_tools

    rows = db_session.exec(
        select(MCPServerConfig)
        .join(
            AgentMCPConfigLink,
            AgentMCPConfigLink.mcp_server_config_id == MCPServerConfig.id,
        )
        .where(AgentMCPConfigLink.agent_settings_id == agent_settings.id)
    ).all()

    if not rows:
        return mcp_tools

    mcp_mgr = get_mcp_manager()

    for srv in rows:
        server_name = srv.name
        try:
            # connect() is idempotent: returns already_connected if live.
            conn_result = await mcp_mgr.connect(
                server_name,
                {"name": server_name, "url": srv.url},
            )
            if conn_result.get("status") == "error":
                log.warning(
                    "MCP server '%s' unreachable: %s; skipping its tools.",
                    server_name, conn_result.get("error"),
                )
                continue

            tools = await mcp_mgr.list_tools(server_name)
            log.info(f"Loaded MCP tools from '{server_name}': {tools}")
        except Exception as e:
            log.warning("Failed to load MCP tools from '%s': %s", server_name, e)
            continue

        for t in tools:
            mcp_tools.append({
                "server_name": server_name,
                "server_description": (srv.description or "").strip(),
                "tool_name": t["name"],
                "description": t.get("description"),
                "input_schema": t.get("input_schema") or {},
                "requires_query_building": bool(srv.requires_query_building),
            })

    return mcp_tools


def _extract_mcp_tool_output_text(result: Dict[str, Any]) -> str:
    """Flatten an MCP ``call_tool`` result's ``content`` list into a single
    text blob. Binary parts are stubbed with a placeholder so the LLM still
    sees them. Returns ``"(tool returned no content)"`` when empty."""
    parts: list[str] = []
    for item in (result.get("content") or []):
        text_val = item.get("text")
        if text_val:
            parts.append(str(text_val))
            continue
        data_val = item.get("data")
        if data_val is not None:
            parts.append(f"[binary data, mimeType={item.get('mimeType')}]")
    return "\n".join(parts) if parts else "(tool returned no content)"
