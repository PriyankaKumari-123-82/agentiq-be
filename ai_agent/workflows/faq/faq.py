"""FAQ generation workflow for AI agents.

Generates contextual FAQ questions per agent by analyzing:
1. Past conversation data (agent-wide, across all users)
2. Agent configuration (system prompt, MCP tools, DB schemas)

FAQs are cached in the ``agent_faqs`` table and reused until invalidated.

Architecture
    ``FAQAgent`` — single agent that handles both initial FAQ generation
    and conversation-aware follow-up generation. All data preparation
    (DB queries, MCP context) lives in ``services.faq_service``; this
    module only orchestrates LLM calls.

Module-level convenience functions (``generate_faqs``,
``generate_follow_ups``) wrap the agent class for callers that don't
need the full object lifecycle.
"""
import json
import logging
import re

from ai_agent.core.config import settings
from ai_agent.core.llm_gateway import get_llm_gateway
from ai_agent.models import AIAgent, AIAgentSettings, FAQSource
from ai_agent.prompts.faq import (
    FAQ_FROM_HISTORY_SYSTEM_PROMPT,
    FAQ_FROM_HISTORY_USER_PROMPT,
    FAQ_SYSTEM_PROMPT,
    FAQ_USER_PROMPT,
    FOLLOWUP_SYSTEM_PROMPT,
    FOLLOWUP_USER_PROMPT,
)
from ai_agent.services.faq_service import TARGET_FAQ_COUNT
from ai_agent.services.api_keys import get_api_key

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _clean_llm_response(content: str) -> list[str]:
    """Parse and clean an LLM response into a list of FAQ strings.

    Handles:
    - Markdown code fences (```json ... ```)
    - Deduplication (case-insensitive)
    - Length filtering (5 < len < 150)
    """
    content = content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = re.sub(r"^```\w*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)

    try:
        faqs = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return []

    if not isinstance(faqs, list):
        return []

    seen: set[str] = set()
    unique: list[str] = []
    for q in faqs:
        q = str(q).strip()
        normalized = q.lower()
        if 5 < len(q) < 150 and normalized not in seen:
            seen.add(normalized)
            unique.append(q)
    return unique


async def _call_llm(
    messages: list[dict],
    agent_settings: AIAgentSettings | None,
    api_key: str | None,
) -> list[str]:
    """Call the LLM and return parsed FAQ strings."""
    gateway = get_llm_gateway()
    try:
        temperature = agent_settings.llm_temperature
        response = await gateway.chat(
            messages=messages,
            provider=agent_settings.llm_provider if agent_settings else None,
            model=agent_settings.llm_model if agent_settings else None,
            temperature=temperature,
            api_key=api_key,
        )
        return _clean_llm_response(response.content)
    except Exception as e:
        log.error("FAQ LLM call failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# FAQAgent — single agent for FAQ + follow-up generation
# ---------------------------------------------------------------------------


class FAQAgent:
    """Generates FAQ and follow-up questions for a single AI agent.

    Accepts pre-built context (prepared by ``faq_service``) and produces
    questions via LLM calls. No DB access — all data comes from the caller.

    Usage::

        agent_obj = FAQAgent(agent=agent, agent_settings=settings)
        faqs = await agent_obj.generate_faqs(
            user_prompts=["show me orders", ...],
            mcp_context="API Tools: ...",
        )
        follow_ups = await agent_obj.generate_follow_ups(
            user_question="...",
            conversation_history="...",
        )
    """

    def __init__(self, agent: AIAgent, agent_settings: AIAgentSettings | None, api_key: str | None = None):
        self._agent = agent
        self._agent_settings = agent_settings
        self._api_key = api_key

    async def generate_faqs(
        self,
        user_prompts: list[str] | None = None,
        mcp_context: str | None = None,
    ) -> list[dict]:
        """Generate FAQs by combining history-based and config-based strategies.

        Returns a list of dicts: ``[{"question": "...", "source": "..."}]``
        """
        all_faqs: list[dict] = []

        # Strategy 1: generate from conversation history
        if user_prompts:
            history_faqs = await self._generate_from_history(user_prompts)
            for q in history_faqs:
                all_faqs.append({"question": q, "source": FAQSource.CHAT_HISTORY})

        # Strategy 2: fill remaining slots from agent config
        remaining = TARGET_FAQ_COUNT - len(all_faqs)
        if remaining > 0:
            config_faqs = await self._generate_from_config(
                mcp_context=mcp_context, count=remaining
            )
            for q in config_faqs:
                all_faqs.append({"question": q, "source": FAQSource.AGENT_CONFIG})

        return all_faqs[:TARGET_FAQ_COUNT]

    async def generate_follow_ups(
        self,
        user_question: str,
        conversation_history: str,
        count: int = 3,
    ) -> list[str]:
        """Generate follow-up questions from the current conversation.

        Returns up to ``count`` follow-up question strings.
        """
        messages = self._build_followup_prompt(
            main_question=user_question,
            conversation_history=conversation_history,
            count=count,
        )
        questions = await _call_llm(messages, self._agent_settings, self._api_key)
        return questions[:count]

    # ------------------------------------------------------------------
    # Internal prompt builders
    # ------------------------------------------------------------------

    async def _generate_from_history(self, user_prompts: list[str]) -> list[str]:
        """Generate FAQs by sending raw user prompts to the LLM for topic analysis."""
        # Sample user questions for the prompt (cap for token budget)
        sample = user_prompts[:50]
        questions_text = "\n".join(f"- {q}" for q in sample)

        system_prompt = self._agent_settings.llm_system_prompt if self._agent_settings else None
        prompt_snippet = ""
        if system_prompt:
            truncated = system_prompt[:1000]
            prompt_snippet = f"\n\nAgent system prompt:\n{truncated}"

        count = min(TARGET_FAQ_COUNT, len(sample))
        messages = [
            {"role": "system", "content": FAQ_FROM_HISTORY_SYSTEM_PROMPT.format(count=count)},
            {
                "role": "user",
                "content": FAQ_FROM_HISTORY_USER_PROMPT.format(
                    agent_name=self._agent.name,
                    user_questions=questions_text,
                    prompt_snippet=prompt_snippet,
                    count=count,
                ),
            },
        ]
        faqs = await _call_llm(messages, self._agent_settings, self._api_key)
        return faqs[:count]

    async def _generate_from_config(
        self, mcp_context: str | None = None, count: int = TARGET_FAQ_COUNT
    ) -> list[str]:
        """Generate FAQs from agent configuration (system prompt, MCP tools)."""
        context_parts = [f"Agent name: {self._agent.name}"]
        if self._agent.description:
            context_parts.append(f"Agent description: {self._agent.description}")

        system_prompt = self._agent_settings.llm_system_prompt if self._agent_settings else None
        if system_prompt:
            truncated = system_prompt[:2000]
            context_parts.append(f"System prompt:\n{truncated}")
        if mcp_context:
            context_parts.append(f"Available tools/capabilities:\n{mcp_context}")

        context = "\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": FAQ_SYSTEM_PROMPT.format(count=count)},
            {"role": "user", "content": FAQ_USER_PROMPT.format(count=count, context=context)},
        ]
        return await _call_llm(messages, self._agent_settings, self._api_key)

    def _build_followup_prompt(
        self,
        main_question: str,
        conversation_history: str,
        count: int = 3,
    ) -> list[dict]:
        """Build LLM messages to generate follow-up questions."""
        system_prompt = self._agent_settings.llm_system_prompt if self._agent_settings else None
        prompt_snippet = ""
        if system_prompt:
            truncated = system_prompt[:1000]
            prompt_snippet = f"\n\nAgent system prompt:\n{truncated}"

        return [
            {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT.format(count=count)},
            {
                "role": "user",
                "content": FOLLOWUP_USER_PROMPT.format(
                    agent_name=self._agent.name,
                    conversation_history=conversation_history,
                    main_question=main_question,
                    prompt_snippet=prompt_snippet,
                    count=count,
                ),
            },
        ]


# ---------------------------------------------------------------------------
# Module-level convenience functions (public API for callers)
# ---------------------------------------------------------------------------


async def generate_faqs(
    agent_id: str,
    db_session: "Session",
    force_refresh: bool = False,
) -> list[dict]:
    """Get cached FAQs or generate new ones for the given agent.

    Returns a list of dicts: ``[{"question": "...", "source": "..."}]``
    """
    from ai_agent.services.faq_service import (
        build_mcp_context,
        get_cached_faqs,
        load_agent_with_settings,
        load_user_prompts,
        persist_faqs,
    )

    # Check cache first
    if not force_refresh:
        cached = get_cached_faqs(agent_id, db_session)
        if cached:
            return cached

    # Load agent and settings
    agent, agent_settings = load_agent_with_settings(agent_id, db_session)
    if not agent:
        return []

    # Prepare data (service layer)
    user_prompts = load_user_prompts(agent_id, db_session)
    mcp_context = await build_mcp_context(agent_settings, db_session)

    # Generate via LLM (workflow layer)
    api_key = get_api_key(db_session, agent_settings.api_key)
    faq_agent = FAQAgent(agent=agent, agent_settings=agent_settings, api_key=api_key)
    try:
        all_faqs = await faq_agent.generate_faqs(
            user_prompts=user_prompts or None,
            mcp_context=mcp_context,
        )
    except Exception as e:
        log.error("FAQ generation failed for agent '%s': %s", agent_id, e)
        return []

    if not all_faqs:
        return []

    # Persist (service layer)
    persist_faqs(agent_id, all_faqs, db_session)
    return all_faqs


async def generate_follow_ups(
    agent_id: str,
    user_question: str,
    conversation_id: str,
    db_session: "Session",
    api_key: str | None,
) -> list[str]:
    """Generate follow-up questions based on the current conversation.

    Returns up to 3 follow-up question strings.
    """
    from ai_agent.services.faq_service import (
        load_agent_with_settings,
        load_conversation_history,
    )

    agent, agent_settings = load_agent_with_settings(agent_id, db_session)
    if not agent:
        return []

    conversation_history = load_conversation_history(conversation_id, db_session)

    faq_agent = FAQAgent(agent=agent, agent_settings=agent_settings, api_key=api_key)
    return await faq_agent.generate_follow_ups(
        user_question=user_question,
        conversation_history=conversation_history,
    )
