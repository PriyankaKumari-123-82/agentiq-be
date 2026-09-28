"""FAQ service layer — data preparation, caching, and DB operations.

Owns all data access for the FAQ feature:
- Cache read/write for agent FAQs
- Conversation history loading
- MCP context building
- FAQ invalidation

The service does NOT call the LLM directly. It prepares data for the
workflow (``workflows.faq.faq.FAQAgent``) and persists its output.
"""
import logging
from typing import Optional

from sqlmodel import Session, select, delete

from ai_agent.models import (
    AgentFAQ,
    AIAgent,
    AIAgentSettings,
    AgentMCPConfigLink,
    FAQSource,
    MCPServerConfig,
    Message,
    UserSession,
)

log = logging.getLogger(__name__)

TARGET_FAQ_COUNT = 6


def get_cached_faqs(agent_id: str, db_session: Session) -> list[dict]:
    """Return cached FAQs for an agent, or an empty list if the cache is stale.

    A cache hit requires at least ``TARGET_FAQ_COUNT`` rows.
    """
    cached = db_session.exec(
        select(AgentFAQ)
        .where(AgentFAQ.agent_id == agent_id)
        .order_by(AgentFAQ.id)
    ).all()

    if cached and len(cached) >= TARGET_FAQ_COUNT:
        return [{"question": f.question, "source": f.source} for f in cached[:TARGET_FAQ_COUNT]]
    return []


def persist_faqs(agent_id: str, faqs: list[dict], db_session: Session) -> None:
    """Replace cached FAQs for an agent with new ones."""
    try:
        db_session.exec(delete(AgentFAQ).where(AgentFAQ.agent_id == agent_id))
        for faq_data in faqs:
            db_session.add(AgentFAQ(
                agent_id=agent_id,
                question=faq_data["question"],
                source=faq_data["source"],
                version=1,
            ))
        db_session.commit()
    except Exception as e:
        log.error("FAQ persistence failed: %s", e)
        db_session.rollback()


def invalidate_faqs(agent_id: str, db_session: Session) -> None:
    """Delete all FAQs for an agent to force regeneration.

    Called when:
    - Cold build completes (schema now available)
    - MCP configuration changes
    """
    try:
        db_session.exec(delete(AgentFAQ).where(AgentFAQ.agent_id == agent_id))
        db_session.commit()
        log.info("Invalidated FAQs for agent '%s'", agent_id)
    except Exception as e:
        log.error("Failed to invalidate FAQs for agent '%s': %s", agent_id, e)
        db_session.rollback()


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------


def load_agent_with_settings(
    agent_id: str, db_session: Session
) -> tuple[Optional[AIAgent], Optional[AIAgentSettings]]:
    """Load agent and its settings. Returns (None, None) if agent is missing or deleted."""
    agent = db_session.get(AIAgent, agent_id)
    if not agent or agent.deleted:
        return None, None

    agent_settings = db_session.exec(
        select(AIAgentSettings).where(AIAgentSettings.agent_id == agent_id)
    ).first()
    return agent, agent_settings


# Conversation history

def load_user_prompts(agent_id: str, db_session: Session, limit: int = 200) -> list[str]:
    """Load recent user-role messages for an agent (across all sessions).

    Returns raw prompt strings for the LLM to analyze topics from.
    """
    return list(db_session.exec(
        select(Message.content)
        .join(UserSession, UserSession.id == Message.session_id)
        .where(UserSession.agent_id == agent_id)
        .where(Message.role == "user")
        .where(Message.content != None)  # noqa: E711
        .where(Message.content != "")
        .order_by(Message.created_at.desc())
        .limit(limit)
    ).all())


def load_conversation_history(conversation_id: str, db_session: Session, limit: int = 20) -> str:
    """Load and format recent messages of a conversation for follow-up generation."""
    recent_messages = db_session.exec(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    ).all()

    recent_messages.reverse()
    lines = [f"{msg.role}: {msg.content}" for msg in recent_messages]
    return "\n".join(lines) or "No previous context."

# MCP / DB context building

async def build_mcp_context(
    agent_settings: Optional[AIAgentSettings], db_session: Session
) -> Optional[str]:
    """Gather MCP tools and build a text summary for prompt context.

    Returns None when MCP is disabled or no tools are found.
    """
    if not (agent_settings and agent_settings.is_mcp_enabled):
        return None

    try:
        from ai_agent.services.mcp_service import gather_mcp_tools
        mcp_tools = await gather_mcp_tools(agent_settings, db_session)
    except Exception as e:
        log.warning("FAQ: failed to gather MCP tools: %s", e)
        return None

    if not mcp_tools:
        return None

    parts = []

    # API-style tools
    api_tools = [t for t in mcp_tools if not t.get("requires_query_building")]
    if api_tools:
        lines = []
        for t in api_tools[:20]:
            desc = (t.get("description") or "").strip()[:100]
            lines.append(f"  - {t['tool_name']}: {desc}")
        parts.append("API Tools:\n" + "\n".join(lines))

    # DB-style tools
    db_tools = [t for t in mcp_tools if t.get("requires_query_building")]
    if db_tools:
        lines = []
        for t in db_tools[:10]:
            srv_desc = (t.get("server_description") or "").strip()[:150]
            lines.append(f"  - Server: {t.get('server_name')} — {srv_desc}")
        parts.append("Database Connections:\n" + "\n".join(lines))

    context = "\n\n".join(parts) if parts else None

    # Enrich with DB schema if applicable
    if db_tools:
        schema_ctx = await _get_db_schema_context(agent_settings, db_session)
        if schema_ctx:
            context = f"{context}\n\n{schema_ctx}" if context else schema_ctx

    return context


async def _get_db_schema_context(
    agent_settings: AIAgentSettings, db_session: Session
) -> Optional[str]:
    """Try to get DB schema info from MCP DB servers (best-effort)."""
    try:
        from ai_agent.core.mcp_manager import get_mcp_manager

        rows = db_session.exec(
            select(MCPServerConfig)
            .join(AgentMCPConfigLink, AgentMCPConfigLink.mcp_server_config_id == MCPServerConfig.id)
            .where(AgentMCPConfigLink.agent_settings_id == agent_settings.id)
            .where(MCPServerConfig.requires_query_building == True)  # noqa: E712
        ).all()

        if not rows:
            return None

        mcp_mgr = get_mcp_manager()
        schema_parts = []

        for srv in rows:
            try:
                conn = await mcp_mgr.connect(srv.name, {"name": srv.name, "url": srv.url})
                if conn.get("status") == "error":
                    continue
                tools = await mcp_mgr.list_tools(srv.name)
                schema_tool = next(
                    (t for t in tools if any(k in t.get("name", "").lower() for k in ("schema", "tables", "describe"))),
                    None,
                )
                if schema_tool:
                    result = await mcp_mgr.call_tool(srv.name, schema_tool["name"], {})
                    text = ""
                    for item in (result.get("content") or []):
                        if item.get("text"):
                            text += item["text"]
                    if text:
                        schema_parts.append(f"Database ({srv.name}):\n{text[:2000]}")
            except Exception as e:
                log.warning("FAQ service: failed to fetch schema from '%s': %s", srv.name, e)

        return "\n\n".join(schema_parts) if schema_parts else None
    except Exception as e:
        log.warning("FAQ service: schema context gathering failed: %s", e)
        return None
