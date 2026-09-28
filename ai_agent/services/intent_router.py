"""Top-level intent classifier and dispatcher for the message API.

Public entry point: ``handle(chatbot_input, db_session)``. The FastAPI
route in ``ai_agent.api.v1.message`` is a thin wrapper around it.

Responsibilities:
    1. Load conversation state (session, history, counter).
    2. Resolve agent runtime config (flags, custom prompt).
    3. Load DB schema + MCP tools (best-effort).
    4. Build the classifier menu and call the classifier LLM.
    5. Route the chosen intent to a pipeline in ``services.pipelines``.

Everything to the right of step 4 lives in ``pipelines``; this module only
*decides* which pipeline runs.
"""
import logging
from typing import Final

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlmodel import Session, select

from ai_agent.core.openai_client import create_chat_completion
from ai_agent.core.prompt import (
    INTENT_DOWNLOAD_TEMPLATE,
    INTENT_NONE,
    build_intent_classifier_messages,
    build_intent_options,
)
from ai_agent.models import AIAgentSettings, Message, UserSession
from ai_agent.services.context import (
    AgentRuntimeConfig,
    ChatbotInput,
    DispatchContext,
    ResponseModel,
)
from ai_agent.services.mcp_service import gather_mcp_tools
from ai_agent.services.api_keys import get_api_key
from ai_agent.services.pipelines import (
    answer_via_knowledge_base,
    answer_via_mcp_tool,
    answer_via_pdf_download,
    build_query_from_mcp_schema,
)

log = logging.getLogger(__file__)

__all__: Final = [
    "handle",
    "route_by_intent",
    "DispatchContext",
    "AgentRuntimeConfig",
]


# ---------------------------------------------------------------------------
# Setup helpers (kept small so ``handle`` reads top-to-bottom)
# ---------------------------------------------------------------------------


def load_conversation_state(
    chatbot_input: ChatbotInput,
    db_session: Session,
) -> tuple[UserSession, list[dict], int]:
    """Load session, previous conversation, and the next user-message counter.

    Side effects: backfills ``session.title`` from the first user message
    when missing. Raises HTTP 404 if the session does not exist.
    """
    session = db_session.exec(
        select(UserSession)
        .where(UserSession.session_id == chatbot_input.session_id)
    ).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # Load messages in chronological order (oldest -> newest) so downstream
    # consumers (agent runtime planner, tool selector, observer) see a
    # well-formed transcript and can correctly identify the CURRENT user turn
    # as the last user-role message. Loading DESC here previously caused the
    # agent to answer the immediately-prior turn instead of the current one.
    db_messages = db_session.exec(
        select(Message)
        .where(Message.conversation_id == chatbot_input.conversation_id)
        .order_by(Message.counter.asc())
    ).all()
    previous_conversation = [
        {"role": msg.role, "content": msg.content} for msg in db_messages
    ]

    if not session.title:
        title = chatbot_input.message[:50] + (
            "..." if len(chatbot_input.message) > 50 else ""
        )
        session.title = title
        db_session.add(session)
        db_session.commit()

    max_counter = db_session.exec(
        select(func.max(Message.counter))
        .where(Message.conversation_id == chatbot_input.conversation_id)
    ).first() or 0
    user_counter = max_counter + 1

    return session, previous_conversation, user_counter


def resolve_agent_runtime(
    session: UserSession,
    db_session: Session,
) -> AgentRuntimeConfig:
    """Look up the session's agent settings (if any) and resolve the runtime
    flags ``handle`` cares about, with safe defaults when the session is not
    bound to an agent."""
    if not session.agent_id:
        return AgentRuntimeConfig(
            agent_settings=None,
            custom_prompt=None,
            web_search_enabled=False,
            json_prompt=False,
            preserve_memory=True,
        )

    agent_settings = db_session.exec(
        select(AIAgentSettings)
        .where(AIAgentSettings.agent_id == session.agent_id)
    ).first()
    if not agent_settings:
        return AgentRuntimeConfig(
            agent_settings=None,
            custom_prompt=None,
            web_search_enabled=False,
            json_prompt=False,
            preserve_memory=True,
        )

    return AgentRuntimeConfig(
        agent_settings=agent_settings,
        custom_prompt=agent_settings.llm_system_prompt or None,
        web_search_enabled=agent_settings.allow_websearch,
        json_prompt=agent_settings.llm_json_prompt,
        preserve_memory=agent_settings.preserve_memory,
    )


async def load_mcp_tools_safely(
    agent_settings: AIAgentSettings | None,
    db_session: Session,
) -> list[dict]:
    """Return the agent's connected MCP tools, or an empty list if MCP is
    disabled / a server is unreachable. Failures are logged, never raised:
    routing must remain available even when MCP is degraded."""
    if not (agent_settings and agent_settings.is_mcp_enabled):
        return []
    try:
        tools = await gather_mcp_tools(agent_settings, db_session)
        log.info("Loaded %d MCP tool(s) for agent", len(tools))
        return tools
    except Exception as e:
        log.error(f"Failed to gather MCP tools: {e}")
        return []


def build_intent_menu(
    mcp_tools: list[dict],
    runtime: AgentRuntimeConfig,
) -> tuple[list[dict], dict[str, bool]]:
    """Build the classifier menu and the MCP-server dispatch table.

    Returns:
        intent_options: ordered options for the classifier prompt
            (built-in intents + one entry per connected MCP server).
        mcp_server_flags: ``{server_name: requires_query_building}`` used
            downstream to branch between the dynamic-SQL flow and the
            single-tool action flow.
    """
    mcp_server_options: list[dict] = []
    mcp_server_flags: dict[str, bool] = {}
    for info in mcp_tools:
        srv_name = info["server_name"]
        if srv_name in mcp_server_flags:
            continue
        mcp_server_options.append({
            "name": srv_name,
            "description": (info.get("server_description") or "").strip(),
        })
        mcp_server_flags[srv_name] = bool(info.get("requires_query_building"))

    agent = runtime.agent_settings
    intent_options = build_intent_options(
        is_pdf_export_enabled=True,
        mcp_servers=mcp_server_options,
    )
    return intent_options, mcp_server_flags


def classify_intent(user_question: str, intent_options: list[dict]) -> str:
    """Single-step intent classifier LLM call.

    Returns the verbatim option ``name`` chosen by the LLM (a built-in
    intent constant or an MCP server name). Any LLM-side failure degrades
    to ``INTENT_NONE`` so the caller can render a clean fallback response.
    Errors in the pure-Python prompt assembly are real bugs and propagate.
    """
    intent_messages = build_intent_classifier_messages(
        user_question=user_question,
        intent_options=intent_options,
    )
    log.info(f"Intent classifier messages: {intent_messages}")

    try:
        selected_intent = create_chat_completion(
            messages=intent_messages,
            temperature=0.0,
        ).strip()
    except Exception as e:
        log.error(f"Intent classifier LLM call failed: {e}")
        return INTENT_NONE

    log.info(f"Intent classifier picked: {selected_intent!r}")
    return selected_intent


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


async def route_by_intent(
    selected_intent: str,
    ctx: DispatchContext,
) -> ResponseModel:
    """Route the classifier's selected intent to the correct pipeline.

    Most pipelines return a plain string that is wrapped into a
    ``ResponseModel`` with ``message_id=0``; only the RAG pipeline
    persists to the DB and returns a ``ResponseModel`` with the real
    message id.
    """
    def _wrap(text: str) -> ResponseModel:
        return ResponseModel(
            message_id=0,
            response=text,
            session_id=ctx.chatbot_input.session_id,
            conversation_id=ctx.chatbot_input.conversation_id,
        )

    # MCP server picked directly by the classifier.
    if selected_intent in ctx.mcp_servers:
        from ai_agent.core.config import settings
        temperature = ctx.agent_settings.llm_temperature
        if ctx.mcp_servers[selected_intent]:
            return _wrap(await build_query_from_mcp_schema(
                server_name=selected_intent,
                mcp_tools=ctx.mcp_tools,
                user_question=ctx.chatbot_input.message,
                temperature=temperature,
            ))
        return _wrap(await answer_via_mcp_tool(
            server_name=selected_intent,
            mcp_tools=ctx.mcp_tools,
            user_question=ctx.chatbot_input.message,
            temperature=temperature,
        ))

    if selected_intent == INTENT_DOWNLOAD_TEMPLATE:
        return _wrap(await answer_via_pdf_download(ctx))

    # INTENT_NONE and any unhandled intent fall through to the
    # knowledge-base / general-chat pipeline. It uses RAG when the agent has
    # a data store, and plain LLM chat otherwise — so greetings and general
    # questions are always answered.
    return answer_via_knowledge_base(ctx)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def handle(
    chatbot_input: ChatbotInput,
    db_session: Session,
) -> ResponseModel:
    """Linear orchestrator: load state → classify intent → route to pipeline.

    All non-trivial logic lives in helpers above; this body should stay
    readable end-to-end with no nested branching.
    """
    session, previous_conversation, user_counter = load_conversation_state(
        chatbot_input, db_session,
    )
    runtime = resolve_agent_runtime(session, db_session)
    api_key = get_api_key(db_session, runtime.agent_settings.api_key)

    mcp_tools = await load_mcp_tools_safely(runtime.agent_settings, db_session)
    intent_options, mcp_server_flags = build_intent_menu(mcp_tools, runtime)
    selected_intent = classify_intent(chatbot_input.message, intent_options)
    log.info(f"Selected intent: {selected_intent}")

    ctx = DispatchContext(
        chatbot_input=chatbot_input,
        session=session,
        db_session=db_session,
        previous_conversation=previous_conversation,
        custom_prompt=runtime.custom_prompt,
        web_search_enabled=runtime.web_search_enabled,
        json_prompt=runtime.json_prompt,
        agent_settings=runtime.agent_settings,
        api_key=api_key,
        messages_to_include=previous_conversation if runtime.preserve_memory else [],
        user_counter=user_counter,
        mcp_tools=mcp_tools,
        mcp_servers=mcp_server_flags,
    )
    res = await route_by_intent(selected_intent, ctx)
    if session.agent_id:
        try:
            from ai_agent.workflows.faq.faq import generate_follow_ups
            res.follow_up_faqs = await generate_follow_ups(
                agent_id=session.agent_id,
                user_question=chatbot_input.message,
                conversation_id=chatbot_input.conversation_id,
                db_session=db_session,
                api_key=api_key,
            )
        except Exception as e:
            log.warning("Failed to fetch follow-ups in intent_router.handle: %s", e)
    return res
