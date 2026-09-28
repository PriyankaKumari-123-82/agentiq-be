"""Per-request context objects and shared API boundary schemas.

Two responsibilities kept together to minimise module count and avoid an
``intent_router`` ↔ ``pipelines`` import cycle:

    1. ``DispatchContext`` + ``AgentRuntimeConfig`` — dataclasses passed
       between the router and the pipelines.
    2. ``ChatbotInput``, ``MessageResponse``, ``ResponseModel`` — Pydantic
       models for the ``/send-message`` request/response boundary. They live
       here (not in the route file) so the service layer can import them
       without pulling in FastAPI / the HTTP layer.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, Field
from sqlmodel import Session

from ai_agent.models import AIAgentSettings, UserSession


# ---------------------------------------------------------------------------
# Pydantic request/response models for the /send-message endpoint
# ---------------------------------------------------------------------------


class ChatbotInput(BaseModel):
    conversation_id: str
    session_id: str
    message: str
    agent_id: str | None = Field(
        default=None,
        description="ID of the AI agent handling this conversation (if any)",
    )
    ask_again: bool = False


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    artifacts: list[dict[str, Any]] | None = None
    counter: int
    conversation_type: str = "text"
    created_at: datetime
    prompt: str | None = None
    session_id: str
    feedback: Dict[str, Any] | None = None


class ResponseModel(BaseModel):
    message_id: int
    response: str
    session_id: str
    conversation_id: str
    follow_up_faqs: list[str] = []


@dataclass
class AgentRuntimeConfig:
    """Resolved per-request agent configuration. Centralises the flag fan-out
    that ``send_message`` needs from ``AIAgentSettings`` so we don't pass a
    handful of unrelated booleans around."""
    agent_settings: AIAgentSettings | None
    custom_prompt: str | None
    web_search_enabled: bool
    json_prompt: bool
    preserve_memory: bool


@dataclass
class DispatchContext:
    """All state needed by the per-intent pipelines, bundled to avoid long
    parameter lists. Built once in ``intent_router.handle`` after classifier
    input is ready, then passed into ``route_by_intent`` and the individual
    pipeline functions."""
    chatbot_input: ChatbotInput
    session: UserSession
    db_session: Session
    previous_conversation: list
    custom_prompt: str | None
    web_search_enabled: bool
    json_prompt: bool
    agent_settings: AIAgentSettings | None
    api_key: str | None
    messages_to_include: list
    user_counter: int
    mcp_tools: list[dict]
    mcp_servers: dict[str, bool]
