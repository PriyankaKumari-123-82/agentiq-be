from typing import Optional
import re
import uuid
import httpx
import logging

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, Query
from sqlmodel import Session, select

from ai_agent.core.config import settings
from ai_agent.core.db import engine
from ai_agent.models import (
    User, AIAgent, AIAgentSettings, UserSession, Message, ConversationRole, SessionStatus,
    AgentWhatsappNumber
)
from ai_agent.services.context import ChatbotInput
from ai_agent.api.v1.message import send_message_v2
from ai_agent.services.artifact_service import create_artifact
from ai_agent.core.enums import ArtifactType
from ai_agent.core.url_shortener import shorten_url
from ai_agent.services.whatsapp_service import send_whatsapp_text_message
from ai_agent.services.whatsapp_formatter import parse_markdown_tables, format_table_to_whatsapp_cards

log = logging.getLogger(__name__)

router = APIRouter(tags=["Webhook"])


def _normalize_phone(phone_str: Optional[str]) -> str:
    if not phone_str:
        return ""
    return re.sub(r"[^\d]", "", str(phone_str))


_URL_RE = re.compile(r"https?://\S+")


async def _shorten_urls_in_text(text: str) -> str:
    """Replace every raw HTTP/HTTPS URL in *text* with its TinyURL equivalent.

    Preserves surrounding text; only the URL tokens themselves are swapped.
    Falls back to the original URL if TinyURL is unavailable (handled inside
    ``shorten_url``).
    """
    urls = list(dict.fromkeys(_URL_RE.findall(text)))  # deduplicate, preserve order
    for url in urls:
        # Strip trailing punctuation that may have been included in the regex match
        clean_url = url.rstrip(".,;:)\"'")
        short = await shorten_url(clean_url)
        if short != clean_url:
            # Replace full matched token (including any trailing chars we stripped)
            text = text.replace(url, short)
    return text


async def _format_blocks_for_whatsapp(
    blocks: list,
    db_session: Session,
    run_id: str,
    user_id: int,
    agent_id: str,
    session_id: int,
    conversation_id: str,
) -> str:
    """
    Convert AgentResponse blocks to a WhatsApp-safe plain-text string.
    - markdown → text + formatted itemized table cards
    - table    → summary text + Excel download link (generated on the fly)
    - artifact → download link
    - chart/code/reasoning → skipped
    """
    parts = []
    
    for block in blocks:
        if not isinstance(block, dict):
            continue
            
        btype = block.get("type")
        content = block.get("content")
        
        if btype in ("markdown", "text"):
            if isinstance(content, str):
                clean_text, tables = parse_markdown_tables(content)
                if clean_text:
                    clean_text = await _shorten_urls_in_text(clean_text)
                    parts.append(clean_text)
                for tbl in tables:
                    formatted_table = format_table_to_whatsapp_cards(
                        title=None,
                        headers=tbl["headers"],
                        rows=tbl["rows"],
                    )
                    if formatted_table:
                        parts.append(formatted_table)
                
        elif btype == "table":
            if not isinstance(content, dict):
                continue
                
            rows = content.get("rows", [])
            cols = content.get("columns", [])
            num_rows = len(rows)
            col_names = [c.get("name", "") if isinstance(c, dict) else str(c) for c in cols]
            
            title = content.get("title") or "Data Table"

            formatted_cards = format_table_to_whatsapp_cards(
                title=title,
                headers=col_names,
                rows=rows,
            )
            
            # Create a safe filename from the title
            safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_").lower()
            filename = f"{safe_title}.xlsx" if safe_title else "export.xlsx"
            
            # Generate Excel file on the fly
            try:
                artifact = await create_artifact(
                    db_session=db_session,
                    run_id=run_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    artifact_type=ArtifactType.XLSX.value,
                    data=rows,
                    filename=filename
                )
                download_url = await shorten_url(artifact.download_url or "")

                table_part = formatted_cards if formatted_cards else f"*{title}* ({num_rows} rows)"
                table_part += f"\n\nDownload Excel: {download_url}"
                parts.append(table_part)
            except Exception as e:
                log.error(f"Failed to generate Excel for WhatsApp table: {e}")
                if formatted_cards:
                    parts.append(formatted_cards)
                
        elif btype == "artifact":
            if isinstance(content, dict):
                url = content.get("download_url") or content.get("artifact", {}).get("download_url")
                fname = content.get("filename") or content.get("artifact", {}).get("filename") or "file"
                if url:
                    short_url = await shorten_url(url)
                    parts.append(f"📥 {fname}: {short_url}")
                    
    return "\n\n".join(parts)


async def process_gallabox_webhook_task(payload: dict):
    """Background task to process an incoming WhatsApp message and respond."""

    # ── Parse sender phone (who sent the message) ────────────────────────────
    wa = payload.get("whatsapp") if isinstance(payload.get("whatsapp"), dict) else {}
    sender_phone = wa.get("from")
    if not sender_phone:
        sender_phone = payload.get("phone")
    if not sender_phone and isinstance(payload.get("contact"), dict):
        sender_phone = payload["contact"].get("phone")
    if not sender_phone and isinstance(payload.get("message"), dict):
        sender_phone = payload["message"].get("recipient_id")
    if not sender_phone:
        sender_phone = payload.get("recipient_id")

    # ── Early exit for unsupported / non-text message types ──────────────────
    msg_type = wa.get("type")
    if msg_type and msg_type != "text":
        log.info(f"Ignoring non-text message type '{msg_type}' from {sender_phone}. Skipping.")
        return

    # ── Parse text ───────────────────────────────────────────────────────────
    text = wa.get("text", {}).get("body") if wa else None
    if not text:
        text = payload.get("message")
        if isinstance(text, dict):
            text = text.get("text", {}).get("body") or text.get("body")
    if not text:
        text = payload.get("text")
    if not text and "message" in payload and isinstance(payload["message"], str):
        text = payload["message"]
    if not text and isinstance(payload.get("reply"), dict):
        text = payload["reply"].get("text")

    if not sender_phone or not text:
        log.warning(f"Webhook missing sender phone or text. Payload keys: {list(payload.keys())}")
        return

    normalized_sender = _normalize_phone(sender_phone)

    with Session(engine) as db_session:
        # ── Step 1: Route by business number (preferred) ─────────────────────
        # Gallabox sends the recipient (business) number in payload["channelNumber"],
        # payload["whatsapp"]["to"], payload["to"], or payload["channel"]["phone"].
        wa_to = wa.get("to") if wa else None
        if not wa_to:
            wa_to = payload.get("to")
        if not wa_to:
            wa_to = payload.get("channelNumber")
        if not wa_to and isinstance(payload.get("channel"), dict):
            wa_to = payload["channel"].get("phone") or payload["channel"].get("channelNumber")

        normalized_business = _normalize_phone(wa_to) if wa_to else None

        agent_id: str | None = None
        owner_user_id: int | None = None
        agent_settings = None

        if normalized_business:
            # Business-number-first: find the agent whose business_number matches the
            # number that received this message.
            agent_settings = db_session.exec(
                select(AIAgentSettings).where(
                    AIAgentSettings.business_number == normalized_business,
                    AIAgentSettings.whatsapp_enabled == True,
                )
            ).first()

            if agent_settings:
                agent_id = agent_settings.agent_id
                # Derive owner from the agent row
                agent_obj = db_session.get(AIAgent, agent_id)
                owner_user_id = agent_obj.owner_id if agent_obj else None
                log.info(
                    "Business-number routing: business='%s' → agent='%s'",
                    normalized_business, agent_id,
                )

                # Optional sender whitelist: if the agent has any registered user numbers,
                # only allow senders that are in that list.
                registered_numbers = db_session.exec(
                    select(AgentWhatsappNumber).where(
                        AgentWhatsappNumber.agent_id == agent_id,
                        AgentWhatsappNumber.is_active == True,
                    )
                ).all()

                if registered_numbers:
                    sender_in_whitelist = any(
                        r.phone_number == normalized_sender for r in registered_numbers
                    )
                    if not sender_in_whitelist:
                        log.warning(
                            "Sender '%s' is not in the whitelist for agent '%s'. Ignoring.",
                            normalized_sender, agent_id,
                        )
                        return
                # No registered numbers → open agent (anyone can message via the business number)
            else:
                log.warning(
                    "No agent found for business number '%s'. Ignoring message.",
                    normalized_business,
                )
                return
        else:
            log.warning("No recipient business number found in the webhook payload. Ignoring message.")
            return

        # ── Fallback removed: We now strictly require a business number match ────

        # ── Step 2: Fetch owner user (needed for send_message_v2) ────────────
        owner_user = db_session.get(User, owner_user_id)
        if not owner_user:
            log.error(f"Owner user_id={owner_user_id} not found in DB. Aborting.")
            return

        # ── Step 3: Find or create UserSession keyed on (sender_phone, agent_id) ──
        # We use whatsapp_phone on UserSession to identify returning senders so
        # the conversation continues across multiple messages without creating
        # platform User rows for external WhatsApp contacts.
        session = db_session.exec(
            select(UserSession).where(
                UserSession.whatsapp_phone == normalized_sender,
                UserSession.agent_id == agent_id,
                UserSession.is_active == True,
                UserSession.is_archived == False,
            ).order_by(UserSession.updated_at.desc())
        ).first()

        if not session:
            agent_obj = db_session.get(AIAgent, agent_id)
            agent_name = agent_obj.name if agent_obj else agent_id
            session_uuid = str(uuid.uuid4())

            # Resolve sender display name from payload
            sender_name = "WhatsApp User"
            if isinstance(payload.get("contact"), dict):
                sender_name = payload["contact"].get("name") or sender_name
            elif isinstance(payload.get("sender"), dict):
                sender_name = payload["sender"].get("name") or sender_name

            session = UserSession(
                user_id=owner_user_id,          # Platform user (number owner)
                agent_id=agent_id,
                session_id=session_uuid,
                whatsapp_phone=normalized_sender,  # Identifies this external sender
                title=f"{sender_phone} : WhatsApp - {agent_name}",
                status=SessionStatus.ACTIVE,
            )
            db_session.add(session)
            db_session.commit()
            db_session.refresh(session)
            log.info(
                f"Created new WhatsApp session '{session_uuid}' for "
                f"sender='{normalized_sender}' on agent='{agent_name}'"
            )

        # ── Step 4: Resolve / seed conversation_id ────────────────────────────
        last_message = db_session.exec(
            select(Message)
            .where(Message.session_id == session.id)
            .order_by(Message.counter.desc())
        ).first()

        if last_message and last_message.conversation_id:
            conversation_id = last_message.conversation_id
        else:
            conversation_id = str(uuid.uuid4())
            seed_msg = Message(
                conversation_id=conversation_id,
                session_id=session.id,
                agent_id=agent_id,
                role=ConversationRole.USER,
                content="",
                counter=0,
                prompt=None,
            )
            db_session.add(seed_msg)
            db_session.commit()
            log.info(f"Seeded conversation_id='{conversation_id}' for session='{session.session_id}'")

        # ── Step 5: Run the agent ─────────────────────────────────────────────
        chatbot_input = ChatbotInput(
            conversation_id=conversation_id,
            session_id=session.session_id,
            message=text,
            agent_id=agent_id,
        )

        try:
            response = await send_message_v2(
                chatbot_input=chatbot_input,
                mode="chat",
                current_user=owner_user,
                db_session=db_session,
            )
        except Exception as e:
            log.error(f"Error running agent for webhook: {e}")
            return

        # ── Step 6: Format and send reply ─────────────────────────────────────
        blocks = response.get("blocks", []) if isinstance(response, dict) else []
        run_id = response.get("run_id", str(uuid.uuid4()))

        text_response = await _format_blocks_for_whatsapp(
            blocks=blocks,
            db_session=db_session,
            run_id=run_id,
            user_id=owner_user_id,
            agent_id=agent_id,
            session_id=session.id,
            conversation_id=conversation_id,
        )

        if not text_response:
            log.warning("Agent returned no text response.")
            return

        # Resolve sender display name for the reply recipient field
        sender_display_name = "WhatsApp User"
        if isinstance(payload.get("contact"), dict):
            sender_display_name = payload["contact"].get("name") or sender_display_name
        elif isinstance(payload.get("sender"), dict):
            sender_display_name = payload["sender"].get("name") or sender_display_name

        await send_whatsapp_text_message(
            phone_number=sender_phone,
            text_content=text_response,
            channel_id=payload.get("channelId"),
            recipient_name=sender_display_name,
        )




@router.post(
    "/gallabox",
    summary="Webhook for incoming Gallabox messages",
)
async def gallabox_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    try:
        payload = await request.json()
        log.info(f"Received webhook payload: {payload}")    
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
        
    background_tasks.add_task(process_gallabox_webhook_task, payload)
    return {"status": "accepted"}
