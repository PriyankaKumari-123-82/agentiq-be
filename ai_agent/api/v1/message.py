import base64
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from ai_agent.core import db
from ai_agent.core.config import settings
from ai_agent.core.deps import get_current_user
from ai_agent.core.email_service import email_service
from ai_agent.core.generate_documents import DocumentGenerator
from ai_agent.helpers.pdf_service import generate_conversation_pdf, upload_pdf_to_s3
from ai_agent.models import (
    AIAgent,
    Artifact,
    FeedbackType,
    Message,
    MessageFeedback,
    User,
    DeliveryMethod,
)
from ai_agent.services.artifact_service import (
    artifact_payload,
    bind_artifacts_to_message,
    persist_chart_artifacts_from_blocks,
)
from ai_agent.services.api_keys import get_api_key
from ai_agent.services import intent_router
from ai_agent.services.context import (  # re-exported below
    ChatbotInput,
    MessageResponse,
    ResponseModel,
)
from ai_agent.core.response_schemas import content_block_to_markdown

log = logging.getLogger(__file__)
router = APIRouter(prefix="/message", tags=["Message"])


# Backwards-compatible re-exports for any external importers.
__all__ = ["ChatbotInput", "MessageResponse", "ResponseModel"]


# ---------------------------------------------------------------------------
# Pydantic models owned by the remaining endpoints
# ---------------------------------------------------------------------------


class FeedbackResponse(BaseModel):
    id: int
    message_id: int
    is_positive: bool
    feedback_type: FeedbackType
    created_at: datetime
    updated_at: datetime | None = None


class FeedbackListResponse(BaseModel):
    message_id: str
    feedbacks: List[FeedbackResponse]


class WriteFeedbackInput(BaseModel):
    message_id: str
    feedback: FeedbackType


class ConversationOut(BaseModel):
    conversation_id: str
    messages: List[MessageResponse]
    hasMore: bool
    nextCursor: str | None = None


class ChatHistoryResponse(BaseModel):
    id: int
    session_id: str
    conversation_id: str
    role: str
    prompt: str | None
    content: str
    artifacts: List[Dict[str, Any]] | None = None
    counter: int
    created_at: datetime
    feedback: Dict | None = None


class ChatHistoryGroup(BaseModel):
    date: str
    session_id: str
    title: str | None
    messages: list[ChatHistoryResponse]


class TranscriptRequest(BaseModel):
    conversation_id: str
    delivery_method: DeliveryMethod = Field(description="Delivery method: 'download' or 'email'")
    user_id: int | None = Field(
        default=None,
        description="User ID for email delivery (required if delivery_method is 'email')",
    )


class TranscriptResponse(BaseModel):
    download_link: str | None = Field(
        default=None,
        description="Download link (only for delivery_method='download')",
    )
    filename: str | None = Field(
        default=None,
        description="Filename (only for delivery_method='download')",
    )
    message: str
    delivery_method: DeliveryMethod


class PaginatedChatHistoryResponse(BaseModel):
    groups: list[ChatHistoryGroup]
    hasMore: bool
    nextCursor: str | None


# ---------------------------------------------------------------------------
# /send-message — thin route, delegates to the service layer
# ---------------------------------------------------------------------------


@router.post("/send-message", response_model=ResponseModel)
async def send_message(
    *,
    chatbot_input: ChatbotInput,
    db_session: Session = Depends(db.get_session),
) -> Any:
    """Classify the user's intent and dispatch to the matching pipeline.

    All logic lives in :mod:`ai_agent.services.intent_router`.
    """
    return await intent_router.handle(chatbot_input, db_session)


# ---------------------------------------------------------------------------
# /send-message-v2 — new endpoint using AgentRuntime with block-based responses
# ---------------------------------------------------------------------------

@router.post("/send-message-v2")
async def send_message_v2(
    *,
    chatbot_input: ChatbotInput,
    mode: str = Query("chat", description="Execution mode: chat/tool_use/planner/autonomous"),
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(db.get_session),
):
    """Send a message using the new AgentRuntime with block-based responses.
    
    This endpoint provides:
    - Multi-LLM support via LiteLLM
    - Block-based responses (markdown, tables, citations, artifacts)
    - Usage tracking and explainability
    - MCP tool execution
    - RAG with LlamaIndex
    
    Example response:
        {
            "message_id": 123,
            "run_id": "uuid",
            "blocks": [
                {"type": "markdown", "content": "..."},
                {"type": "table", "content": {...}},
                {"type": "citation", "content": {...}}
            ],
            "citations": [...],
            "usage": {"input_tokens": 100, "output_tokens": 50, "cost_usd": 0.002},
            "artifacts": [...]
        }
    """
    from ai_agent.core.agent_runtime import get_agent_runtime
    from ai_agent.core.usage_tracker import get_usage_tracker
    from ai_agent.core.guardrails import get_guardrails_service
    from ai_agent.core.response_schemas import AgentResponse, ContentBlock, BlockType
    from ai_agent.models import RunMode, ConversationRole, ConversationType
    from ai_agent.services.intent_router import load_conversation_state

    # Identity is derived from the validated Keycloak token (get_current_user),
    # never from client-supplied query parameters. Everything downstream keys off
    # these server-trusted values.
    user_id = current_user.id
    user_email = current_user.email

    # Load session
    try:
        user_session, previous_conversation, user_counter = load_conversation_state(
            chatbot_input, db_session
        )
    except HTTPException as e:
        raise e

    # Verify the authenticated user owns this session
    if user_session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Session does not belong to user")
    
    # Check guardrails
    guardrails = get_guardrails_service()
    violations = guardrails.check_input(
        text=chatbot_input.message,
        user_id=user_id,
        agent_id=chatbot_input.agent_id or user_session.agent_id,
    )
    
    critical_violations = [v for v in violations if v.severity in ("high", "critical")]
    if critical_violations:
        return {
            "error": "guardrail_violation",
            "message": guardrails.get_remediation(critical_violations[0]),
            "violations": [
                {"type": v.violation_type, "message": v.message}
                for v in critical_violations
            ]
        }
    
    # Check quota
    usage_tracker = get_usage_tracker(db_session)
    quota_check = usage_tracker.check_quota(
        user_id=user_id,
        agent_id=chatbot_input.agent_id or user_session.agent_id,
        quota_type="tokens",
        requested_amount=1000,
    )
    
    if not quota_check["allowed"]:
        return {
            "error": "quota_exceeded",
            "message": quota_check["message"],
            "remaining": quota_check.get("remaining"),
        }
    
    # Create user message
    user_msg = Message(
        session_id=user_session.id,
        agent_id=chatbot_input.agent_id or user_session.agent_id,
        conversation_id=chatbot_input.conversation_id,
        role=ConversationRole.USER,
        content=chatbot_input.message,
        counter=user_counter,
        conversation_type=ConversationType.TEXT,
    )
    db_session.add(user_msg)
    db_session.commit()
    db_session.refresh(user_msg)
    
    # Prepare messages
    messages = previous_conversation if previous_conversation else []
    messages.append({"role": "user", "content": chatbot_input.message})
    
    # Map mode string to RunMode
    mode_map = {
        "chat": RunMode.CHAT,
        "tool_use": RunMode.TOOL_USE,
        "planner": RunMode.PLANNER,
        "autonomous": RunMode.AUTONOMOUS,
    }
    run_mode = mode_map.get(mode, RunMode.CHAT)
    
    # Get agent runtime
    runtime = await get_agent_runtime(db_session)
    
    # Fetch user's API key for the agent's LLM provider
    from sqlmodel import select
    agent = db_session.exec(select(AIAgent).where(AIAgent.id == (chatbot_input.agent_id or user_session.agent_id))).first()
    api_key = None
    if agent and agent.settings:
        api_key = get_api_key(db_session, agent.settings.api_key)
    
    try:
        # Execute the run
        response = await runtime.run(
            agent_id=chatbot_input.agent_id or user_session.agent_id,
            user_id=user_id,
            session_id=user_session.id,
            conversation_id=chatbot_input.conversation_id,
            messages=messages,
            mode=run_mode,
            input_message_id=user_msg.id,
            stream=False,
            api_key=api_key,
            user_email=user_email,
        )
        
        # Save assistant message. Render structured blocks (tables, code,
        # etc.) into markdown so the chat UI can format them correctly.
        assistant_content = "\n\n".join(
            md for md in (content_block_to_markdown(b) for b in response.blocks) if md
        )
        
        assistant_msg = Message(
            session_id=user_session.id,
            agent_id=chatbot_input.agent_id or user_session.agent_id,
            conversation_id=chatbot_input.conversation_id,
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
                agent_id=chatbot_input.agent_id or user_session.agent_id,
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
        
        # Update response with message ID
        response.message_id = assistant_msg.id
        
        return response.dict()
        
    except Exception as e:
        from ai_agent.schema_engine.orchestrator import SchemaEngineError
        from ai_agent.core.llm_gateway import is_llm_quota_error, llm_quota_error_message

        if isinstance(e, SchemaEngineError):
            log.warning(f"Schema engine error: [{e.error_code}] {e.message}")
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": e.error_code,
                    "message": e.message,
                },
            )
        if is_llm_quota_error(e):
            log.warning(f"LLM quota/rate-limit error: {e}")
            raise HTTPException(
                status_code=429,
                detail={
                    "error_code": "llm_quota_exceeded",
                    "message": llm_quota_error_message(e),
                },
            )
        log.error(f"AgentRuntime error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


# ---------------------------------------------------------------------------
# /{conversation_id} — paginated message history
# ---------------------------------------------------------------------------


@router.get("/{conversation_id}", response_model=ConversationOut)
async def get_conversation(
    *,
    conversation_id: str,
    cursor: str | None = None,
    limit: int = settings.DEFAULT_PAGE_SIZE,
    session: Session = Depends(db.get_session),
):
    """Retrieve the conversation chain for a given conversation_id with pagination."""
    query = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.counter.desc())
    )

    if cursor:
        decoded_cursor = base64.urlsafe_b64decode(cursor).decode()
        cursor_counter_str, cursor_id_str = decoded_cursor.split("|")
        cursor_counter = int(cursor_counter_str)
        cursor_id = int(cursor_id_str)
        query = query.where(
            (Message.counter > cursor_counter)
            | ((Message.counter == cursor_counter) & (Message.id > cursor_id))
        )

    query = query.limit(limit + 1)
    messages = session.exec(query).all()

    if not messages and not cursor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    has_more = len(messages) > limit
    if has_more:
        messages = messages[:-1]

    next_cursor = None
    if has_more and messages:
        last_message = messages[-1]
        cursor_str = f"{last_message.counter}|{last_message.id}"
        next_cursor = base64.urlsafe_b64encode(cursor_str.encode()).decode()

    message_ids = [msg.id for msg in messages]
    feedback_query = (
        select(MessageFeedback)
        .where(MessageFeedback.message_id.in_(message_ids))
    )
    feedbacks = {fb.message_id: fb for fb in session.exec(feedback_query).all()}

    artifacts_by_message: Dict[int, List[Dict[str, Any]]] = {}
    if message_ids:
        artifact_rows = session.exec(
            select(Artifact).where(Artifact.message_id.in_(message_ids))
        ).all()
        for artifact in artifact_rows:
            if artifact.message_id is None:
                continue
            artifacts_by_message.setdefault(artifact.message_id, []).append(
                artifact_payload(artifact)
            )

    message_responses = []
    for msg in messages:
        feedback = feedbacks.get(msg.id)
        feedback_data = None
        if feedback:
            feedback_data = {
                "feedback": (
                    FeedbackType.THUMBS_UP if feedback.is_positive
                    else FeedbackType.THUMBS_DOWN
                ),
                "id": feedback.id,
                "is_positive": feedback.is_positive,
                "created_at": feedback.created_at.isoformat(),
                "updated_at": (
                    feedback.updated_at.isoformat() if feedback.updated_at else None
                ),
            }

        message_responses.append(MessageResponse(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            artifacts=artifacts_by_message.get(msg.id),
            counter=msg.counter,
            conversation_type=msg.conversation_type,
            created_at=msg.created_at,
            feedback=feedback_data,
            prompt=msg.prompt,
            session_id=str(msg.session_id),
        ))

    return ConversationOut(
        conversation_id=conversation_id,
        messages=message_responses,
        hasMore=has_more,
        nextCursor=next_cursor,
    )


# ---------------------------------------------------------------------------
# /feedback — write + read
# ---------------------------------------------------------------------------


@router.post("/feedback", response_model=FeedbackResponse)
def write_feedback(
    write_feedback_input: WriteFeedbackInput,
    db_session: Session = Depends(db.get_session),
):
    message = db_session.get(Message, write_feedback_input.message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    existing_feedback = db_session.exec(
        select(MessageFeedback)
        .where(MessageFeedback.message_id == write_feedback_input.message_id)
    ).first()

    if existing_feedback:
        existing_feedback.is_positive = (
            write_feedback_input.feedback == FeedbackType.THUMBS_UP
        )
        existing_feedback.updated_at = datetime.now(timezone.utc)
        db_session.add(existing_feedback)
        db_session.commit()
        db_session.refresh(existing_feedback)

        return FeedbackResponse(
            id=existing_feedback.id,
            message_id=existing_feedback.message_id,
            is_positive=existing_feedback.is_positive,
            feedback_type=(
                FeedbackType.THUMBS_UP if existing_feedback.is_positive
                else FeedbackType.THUMBS_DOWN
            ),
            created_at=existing_feedback.created_at,
            updated_at=existing_feedback.updated_at,
        )

    new_feedback = MessageFeedback(
        message_id=write_feedback_input.message_id,
        is_positive=write_feedback_input.feedback == FeedbackType.THUMBS_UP,
    )
    db_session.add(new_feedback)
    db_session.commit()
    db_session.refresh(new_feedback)

    return FeedbackResponse(
        id=new_feedback.id,
        message_id=new_feedback.message_id,
        is_positive=new_feedback.is_positive,
        feedback_type=(
            FeedbackType.THUMBS_UP if new_feedback.is_positive
            else FeedbackType.THUMBS_DOWN
        ),
        created_at=new_feedback.created_at,
        updated_at=new_feedback.updated_at,
    )


@router.get("/feedback/{message_id}", response_model=FeedbackListResponse)
def get_feedback(
    *, message_id: str, db_session: Session = Depends(db.get_session),
):
    message = db_session.get(Message, message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    feedbacks = db_session.exec(
        select(MessageFeedback)
        .where(MessageFeedback.message_id == message_id)
        .order_by(MessageFeedback.created_at)
    ).all()

    if not feedbacks:
        raise HTTPException(status_code=404, detail="No feedback found for this message")

    return FeedbackListResponse(
        message_id=message_id,
        feedbacks=[
            FeedbackResponse(
                id=fb.id,
                message_id=fb.message_id,
                is_positive=fb.is_positive,
                feedback_type=(
                    FeedbackType.THUMBS_UP if fb.is_positive
                    else FeedbackType.THUMBS_DOWN
                ),
                created_at=fb.created_at,
                updated_at=fb.updated_at,
            )
            for fb in feedbacks
        ],
    )


# ---------------------------------------------------------------------------
# /export — conversation transcript (PDF download or email)
# ---------------------------------------------------------------------------


async def _handle_download_delivery(
    pdf_path: str,
    filename: str,
    conversation_id: str,
) -> TranscriptResponse:
    """Upload the generated PDF to S3 and return a presigned download link."""
    presigned_url = await upload_pdf_to_s3(pdf_path=pdf_path, filename=filename)

    if not presigned_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload PDF to S3",
        )

    return TranscriptResponse(
        download_link=presigned_url,
        filename=filename,
        message="Conversation transcript generated successfully",
        delivery_method=DeliveryMethod.DOWNLOAD,
    )


def _handle_email_delivery(
    pdf_path: str,
    filename: str,
    conversation_id: str,
    email_address: str,
) -> TranscriptResponse:
    """Send the generated PDF as an email attachment."""
    subject = f"Conversation Transcript - {conversation_id}"
    body = (
        "Dear User,\n\n"
        f"Please find attached the transcript for conversation ID: {conversation_id}\n\n"
        "This transcript contains all messages from your conversation in PDF format.\n\n"
        "Best regards,\n"
        "AI Agent Team"
    )

    with open(pdf_path, "rb") as f:
        attachment_bytes = f.read()

    email_service.send_email_with_bytes_attachment(
        to_email=email_address,
        subject=subject,
        body=body,
        attachment_bytes=attachment_bytes,
        attachment_filename=filename,
        mime_type="application/pdf",
    )

    log.info(
        f"Successfully emailed transcript for conversation {conversation_id} "
        f"to {email_address}"
    )

    return TranscriptResponse(
        download_link=None,
        filename=None,
        message=f"Conversation transcript has been sent to {email_address}",
        delivery_method=DeliveryMethod.EMAIL,
    )


@router.post("/export", response_model=TranscriptResponse)
async def generate_conversation_transcript(
    *,
    transcript_request: TranscriptRequest,
    db_session: Session = Depends(db.get_session),
) -> Any:
    """Generate a PDF transcript of a conversation and either provide a
    download link or email it to the user."""
    if transcript_request.delivery_method not in [DeliveryMethod.DOWNLOAD, DeliveryMethod.EMAIL]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="delivery_method must be either 'download' or 'email'",
        )

    if transcript_request.delivery_method == DeliveryMethod.EMAIL and not transcript_request.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required when delivery_method is 'email'",
        )

    messages = db_session.exec(
        select(Message)
        .where(Message.conversation_id == transcript_request.conversation_id)
        .order_by(Message.counter.asc())
    ).all()

    if not messages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    previous_conversation = [
        {"role": msg.role, "content": msg.content} for msg in messages
    ]

    pdf_content = generate_conversation_pdf(
        previous_conversation=previous_conversation,
        user_message="Generate complete conversation transcript",
    )
     
    if not pdf_content:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate PDF content",
        )

    doc_generator = DocumentGenerator()
    filename = (
        f"conversation_transcript_{transcript_request.conversation_id}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    pdf_path = doc_generator.generate_pdf_from_content(
        pdf_content=pdf_content,
        filename=filename,
    )

    if not pdf_path:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create PDF document",
        )

    if transcript_request.delivery_method == DeliveryMethod.DOWNLOAD:
        return await _handle_download_delivery(
            pdf_path=pdf_path,
            filename=filename,
            conversation_id=transcript_request.conversation_id,
        )

    user = db_session.get(User, transcript_request.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return _handle_email_delivery(
        pdf_path=pdf_path,
        filename=filename,
        conversation_id=transcript_request.conversation_id,
        email_address=user.email,
    )
