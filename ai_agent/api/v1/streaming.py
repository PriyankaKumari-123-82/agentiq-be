"""Streaming API endpoints for real-time agent responses.

This module provides Server-Sent Events (SSE) endpoints for streaming
agent responses with block-based content delivery.
"""
import json
import logging
from typing import AsyncIterator, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlmodel import Session
import asyncio

from ai_agent.core import db
from ai_agent.core.deps import get_current_user
from ai_agent.core.agent_runtime import AgentRuntime, get_agent_runtime
from ai_agent.core.response_schemas import StreamingEvent, AgentResponse, content_block_to_markdown
from ai_agent.core.usage_tracker import UsageTracker, get_usage_tracker
from ai_agent.services import intent_router
from ai_agent.services.artifact_service import (
    bind_artifacts_to_message,
    persist_chart_artifacts_from_blocks,
)
from ai_agent.services.context import ChatbotInput
from ai_agent.models import (
    AgentRun, RunStatus, UserSession, Message, ConversationRole, ConversationType, User
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/stream", tags=["Streaming"])


class StreamRequest(BaseModel):
    """Request to start a streaming agent run."""
    conversation_id: str = Field(description="Conversation ID")
    session_id: str = Field(description="User session ID")
    message: str = Field(description="User message")
    agent_id: Optional[str] = Field(default=None, description="Agent ID (optional)")
    mode: str = Field(default="chat", description="Execution mode: chat/tool_use/planner/autonomous")
    ask_again: bool = Field(default=False, description="Regenerate previous response")
    stream_citations: bool = Field(default=True, description="Stream citations as they arrive")
    stream_tool_calls: bool = Field(default=True, description="Stream tool call progress")


class StreamCancelRequest(BaseModel):
    """Request to cancel an ongoing stream."""
    run_id: str = Field(description="Run ID to cancel")


def serialize_event(event: StreamingEvent) -> str:
    """Serialize a streaming event to SSE format."""
    data = {
        "event_type": event.event_type,
        "run_id": event.run_id,
        "timestamp": event.timestamp.isoformat(),
        "data": event.data,
    }
    return f"data: {json.dumps(data)}\n\n"


async def stream_agent_response(
    runtime: AgentRuntime,
    request: StreamRequest,
    user_id: int,
    session: UserSession,
    db_session: Session,
    usage_tracker: UsageTracker,
) -> AsyncIterator[str]:
    """Stream agent response as SSE events.
    
    Yields SSE-formatted strings.
    """
    run_id: Optional[str] = None
    
    try:
        # Load conversation history
        from ai_agent.services.intent_router import load_conversation_state
        
        chatbot_input = ChatbotInput(
            conversation_id=request.conversation_id,
            session_id=request.session_id,
            message=request.message,
            agent_id=request.agent_id,
            ask_again=request.ask_again,
        )
        
        user_session, previous_conversation, user_counter = load_conversation_state(
            chatbot_input, db_session
        )
        
        # Build messages
        messages = previous_conversation if previous_conversation else []
        messages.append({"role": "user", "content": request.message})
        
        # Determine agent ID
        agent_id = request.agent_id or user_session.agent_id
        if not agent_id:
            # Fallback: yield error
            yield serialize_event(StreamingEvent(
                event_type="run.error",
                run_id="",
                data={"error": "No agent specified for this session"}
            ))
            return
        
        # Create user message
        user_msg = Message(
            session_id=session.id,
            agent_id=agent_id,
            conversation_id=request.conversation_id,
            role=ConversationRole.USER,
            content=request.message,
            counter=user_counter,
            conversation_type=ConversationType.TEXT,
        )
        db_session.add(user_msg)
        db_session.commit()
        db_session.refresh(user_msg)
        
        # Yield run started
        run_id = str(__import__('uuid').uuid4())
        yield serialize_event(StreamingEvent(
            event_type="run.started",
            run_id=run_id,
            data={
                "agent_id": agent_id,
                "mode": request.mode,
                "user_message_id": user_msg.id,
            }
        ))
        
        # Check quota
        quota_check = usage_tracker.check_quota(
            user_id=user_id,
            agent_id=agent_id,
            quota_type="tokens",
            requested_amount=1000,  # Minimum estimate
        )
        
        if not quota_check["allowed"]:
            yield serialize_event(StreamingEvent(
                event_type="run.quota_exceeded",
                run_id=run_id,
                data={
                    "message": quota_check["message"],
                    "remaining": quota_check.get("remaining"),
                }
            ))
            yield serialize_event(StreamingEvent(
                event_type="run.completed",
                run_id=run_id,
                data={"status": "quota_exceeded"}
            ))
            return
        
        # Yield progress: loading context
        yield serialize_event(StreamingEvent(
            event_type="progress.context_loading",
            run_id=run_id,
            data={"message": "Loading conversation context..."}
        ))
        
        # Execute the run (non-streaming for now, but yielding progress)
        # For full streaming, we'd need to modify AgentRuntime to yield events
        from ai_agent.models import RunMode
        
        mode_map = {
            "chat": RunMode.CHAT,
            "tool_use": RunMode.TOOL_USE,
            "planner": RunMode.PLANNER,
            "autonomous": RunMode.AUTONOMOUS,
        }
        mode = mode_map.get(request.mode, RunMode.CHAT)
        
        response = await runtime.run(
            agent_id=agent_id,
            user_id=user_id,
            session_id=session.id,
            conversation_id=request.conversation_id,
            messages=messages,
            mode=mode,
            input_message_id=user_msg.id,
            stream=False,  # We're handling streaming at this level
        )
        
        # Yield content blocks. Convert structured blocks (tables, code,
        # etc.) into markdown so the chat UI renders them correctly. Also
        # include the raw structured payload for clients that want to render
        # natively (e.g. interactive tables).
        for i, block in enumerate(response.blocks):
            raw_content = block.content
            if hasattr(raw_content, "model_dump"):
                raw_payload = raw_content.model_dump()
            elif isinstance(raw_content, (dict, list, str, int, float, bool)) or raw_content is None:
                raw_payload = raw_content
            else:
                raw_payload = str(raw_content)
            yield serialize_event(StreamingEvent(
                event_type="message.delta",
                run_id=run_id,
                data={
                    "block_index": i,
                    "block_type": block.type,
                    "content": content_block_to_markdown(block),
                    "raw": raw_payload,
                }
            ))
            # Small delay to simulate streaming
            await asyncio.sleep(0.01)
        
        # Yield citations if present
        if response.citations and request.stream_citations:
            for citation in response.citations:
                yield serialize_event(StreamingEvent(
                    event_type="citation.received",
                    run_id=run_id,
                    data={
                        "citation_id": citation.citation_id,
                        "source_name": citation.source_name,
                        "relevance_score": citation.relevance_score,
                    }
                ))
        
        # Yield tool calls if present
        if response.tool_calls and request.stream_tool_calls:
            for tool_call in response.tool_calls:
                yield serialize_event(StreamingEvent(
                    event_type="tool_call.started",
                    run_id=run_id,
                    data={
                        "tool_name": tool_call.tool_name,
                        "tool_id": tool_call.tool_id,
                        "arguments": tool_call.arguments,
                    }
                ))
        
        # Save assistant message. Render structured blocks into markdown so
        # the chat UI formats them correctly when the conversation is
        # reloaded from the database.
        assistant_content = "\n\n".join(
            md for md in (content_block_to_markdown(b) for b in response.blocks) if md
        )
        
        assistant_msg = Message(
            session_id=session.id,
            agent_id=agent_id,
            conversation_id=request.conversation_id,
            role=ConversationRole.ASSISTANT,
            content=assistant_content,
            counter=user_counter + 1,
            conversation_type=ConversationType.TEXT,
        )
        db_session.add(assistant_msg)
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
                user_id=user_id,
                agent_id=agent_id,
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
        
        # Yield completion
        yield serialize_event(StreamingEvent(
            event_type="run.completed",
            run_id=run_id,
            data={
                "status": "success",
                "message_id": assistant_msg.id,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "cost_usd": response.usage.cost_usd,
                },
                "artifacts": [a.dict() for a in response.artifacts],
            }
        ))
        
    except Exception as e:
        from ai_agent.schema_engine.orchestrator import SchemaEngineError

        if isinstance(e, SchemaEngineError):
            log.warning(f"Streaming schema engine error: [{e.error_code}] {e.message}")
            yield serialize_event(StreamingEvent(
                event_type="run.error",
                run_id=run_id or "",
                data={
                    "error_code": e.error_code,
                    "message": e.message,
                }
            ))
        else:
            log.error(f"Streaming error: {e}")
            yield serialize_event(StreamingEvent(
                event_type="run.error",
                run_id=run_id or "",
                data={
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            ))
        yield serialize_event(StreamingEvent(
            event_type="run.completed",
            run_id=run_id or "",
            data={"status": "error"}
        ))


@router.post("/message")
async def stream_message(
    request: StreamRequest,
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    """Stream an agent response as Server-Sent Events.
    
    This endpoint provides real-time streaming of agent responses with:
    - Progress updates
    - Content blocks as they're generated
    - Citation discovery
    - Tool call progress
    - Usage information
    
    Example:
        curl -N -H "Accept: text/event-stream" \
             -X POST http://api/stream/message \
             -d '{"conversation_id": "abc", "session_id": "xyz", "message": "Hello"}'
    """
    # Verify session exists
    from ai_agent.services.intent_router import load_conversation_state
    from ai_agent.services.context import ChatbotInput
    
    chatbot_input = ChatbotInput(
        conversation_id=request.conversation_id,
        session_id=request.session_id,
        message=request.message,
    )

    # Identity is derived from the validated Keycloak token, never from the client.
    user_id = current_user.id

    try:
        session, _, _ = load_conversation_state(chatbot_input, db_session)
    except HTTPException:
        raise HTTPException(status_code=404, detail="Session not found")

    # Verify the authenticated user owns this session
    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Session does not belong to user")
    
    # Get runtime and usage tracker
    runtime = await get_agent_runtime(db_session)
    usage_tracker = get_usage_tracker(db_session)
    
    return StreamingResponse(
        stream_agent_response(runtime, request, user_id, session, db_session, usage_tracker),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.post("/cancel")
async def cancel_stream(
    request: StreamCancelRequest,
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    """Cancel an ongoing streaming run.

    Note: This sets the run status to cancelled, but the actual
    stream may continue until the next checkpoint. The client should
    disconnect to fully stop receiving events.
    """
    run = db_session.get(AgentRun, request.run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    # Only the run's owner may cancel it
    if run.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Run does not belong to user")

    if run.status == RunStatus.COMPLETED:
        return {"status": "already_completed", "run_id": request.run_id}
    
    run.status = RunStatus.CANCELLED
    run.completed_at = datetime.utcnow()
    db_session.commit()
    
    return {
        "status": "cancelled",
        "run_id": request.run_id,
    }


@router.get("/runs/{run_id}")
async def get_run_status(
    run_id: str,
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    """Get the status and results of a run."""
    from ai_agent.core.usage_tracker import UsageTracker

    run = db_session.get(AgentRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Run does not belong to user")
    
    usage_tracker = UsageTracker(db_session)
    usage_summary = usage_tracker.get_run_usage(run_id)
    
    return {
        "run_id": run_id,
        "status": run.status,
        "agent_id": run.agent_id,
        "mode": run.mode,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "latency_ms": run.latency_ms,
        "total_steps": run.total_steps,
        "usage": usage_summary,
        "error": run.error,
    }
