"""Voice pipeline: OpenAI STT → AgentRuntime → OpenAI TTS.

The browser streams mic audio (PCM16 mono 24kHz, base64) over our WebSocket.
We forward it to an OpenAI Realtime *transcription-only* session (GA protocol,
``?intent=transcription``) which handles VAD and speech-to-text. Each final
transcript is processed as a normal agent turn through ``AgentRuntime.run()``
— the same brain as ``/send-message-v2`` (tools, SQL, RAG, guardrails, quota,
usage tracking) — and the reply is streamed back as audio via OpenAI TTS
(``/v1/audio/speech``, PCM16 mono 24kHz).

The events we emit to the client are the ones ``useVoiceChat.ts`` already
consumes; the frontend requires no changes:

- ``input_audio_buffer.speech_started``                       (barge-in)
- ``conversation.item.input_audio_transcription.completed``   (user bubble)
- ``response.created`` / ``response.audio_transcript.delta`` /
  ``response.audio_transcript.done``                          (assistant text)
- ``response.audio.delta``                                    (audio chunks)
- ``response.done``
"""
import asyncio
import base64
import json
import logging
import re
from typing import Any, List, Optional
from uuid import uuid4

import websockets
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI
from sqlalchemy import func
from sqlmodel import Session, select

from ai_agent.core.config import settings
from ai_agent.core.db import engine
from ai_agent.core.response_schemas import (
    BlockType,
    ContentBlock,
    content_block_to_markdown,
)
from ai_agent.models import (
    AIAgent,
    AIAgentSettings,
    ConversationRole,
    ConversationType,
    Message,
    RunMode,
    User,
    UserSession,
)
from ai_agent.services.api_keys import get_api_key

log = logging.getLogger(__name__)

router = APIRouter(prefix="/message", tags=["Voice"])

# ── Spoken-text policy ───────────────────────────────────────────────────────
# The voice speaks conversational prose only. Rich blocks stay in the chat UI
# and are announced with a short notice instead of being read aloud.
TABLE_NOTE = "I've added a table with the results to the chat."
CHART_NOTE = "I've added a chart to the chat."
FILE_NOTE = "The file is ready to download in the chat."
CODE_NOTE = "I've put a code snippet in the chat."
TRUNCATION_NOTE = " More details are in the chat."
SPOKEN_CHAR_CAP = 900

FALLBACK_SPOKEN_LINE = "I've posted my answer in the chat."
APOLOGY_LINE = "Sorry, I ran into a problem answering that. Please try again."

# Keep planner prompts bounded on long conversations.
HISTORY_MESSAGE_LIMIT = 30

# PCM16 mono 24kHz ⇒ 48,000 bytes/s. 12,000 bytes = 125ms per audio delta.
TTS_CHUNK_BYTES = 12_000

_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)


def _strip_markdown_for_speech(text: str) -> tuple[str, set]:
    """Reduce markdown to speakable prose.

    Returns the cleaned text plus flags for content that was removed and
    should be announced instead: ``{"table", "code"}``.
    """
    flags: set = set()

    lines: List[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            flags.add("code")
            continue
        if in_fence:
            continue
        if stripped.startswith("|"):
            flags.add("table")
            continue
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = _IMAGE_RE.sub("", cleaned)
    cleaned = _LINK_RE.sub(r"\1", cleaned)
    cleaned = _HEADING_RE.sub("", cleaned)
    # Emphasis / inline-code markers. Keep "_" — it appears inside identifiers.
    cleaned = cleaned.replace("**", "").replace("__", "")
    cleaned = cleaned.replace("*", "").replace("`", "")
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip(), flags


def _derive_spoken_text(blocks: List[ContentBlock]) -> str:
    """Build the TTS input from an AgentResponse's content blocks."""
    parts: List[str] = []
    notes: List[str] = []

    def add_note(note: str) -> None:
        if note not in notes:
            notes.append(note)

    for block in blocks:
        if block.type == BlockType.MARKDOWN and isinstance(block.content, str):
            prose, flags = _strip_markdown_for_speech(block.content)
            if prose:
                parts.append(prose)
            if "table" in flags:
                add_note(TABLE_NOTE)
            if "code" in flags:
                add_note(CODE_NOTE)
        elif block.type == BlockType.TABLE:
            add_note(TABLE_NOTE)
        elif block.type == BlockType.CHART:
            add_note(CHART_NOTE)
        elif block.type == BlockType.ARTIFACT:
            add_note(FILE_NOTE)
        elif block.type == BlockType.CODE:
            add_note(CODE_NOTE)
        # citations / tool blocks are not spoken

    spoken = " ".join(parts + notes).strip()
    if not spoken:
        return FALLBACK_SPOKEN_LINE
    if len(spoken) > SPOKEN_CHAR_CAP:
        spoken = spoken[:SPOKEN_CHAR_CAP].rsplit(" ", 1)[0] + TRUNCATION_NOTE
    return spoken


# ── DB helpers (same semantics as the previous implementation) ──────────────
def _get_next_counter(db_session: Session, conversation_id: str) -> int:
    max_counter = db_session.exec(
        select(func.max(Message.counter))
        .where(Message.conversation_id == conversation_id)
    ).first() or 0
    return max_counter + 1


def _save_message(
    db_session: Session,
    session_id: int,
    agent_id: str,
    conversation_id: str,
    role: str,
    content: str,
) -> Message:
    msg = Message(
        session_id=session_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        role=role,
        content=content,
        counter=_get_next_counter(db_session, conversation_id),
        conversation_type=ConversationType.VOICE,
    )
    db_session.add(msg)
    db_session.commit()
    db_session.refresh(msg)
    return msg


def _load_history(db_session: Session, conversation_id: str) -> List[dict]:
    """Previous conversation as role/content dicts, oldest → newest, capped."""
    db_messages = db_session.exec(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.counter.desc())
        .limit(HISTORY_MESSAGE_LIMIT)
    ).all()
    return [
        {"role": msg.role, "content": msg.content}
        for msg in reversed(db_messages)
    ]


def _build_voice_hint(llm_voice_prompt: Optional[str]) -> dict:
    hint = (
        "The user is talking to you over voice. Keep this reply short, "
        "conversational, and natural to listen to. Prefer a spoken-style "
        "answer over long lists unless the user asked for detail."
    )
    if llm_voice_prompt and llm_voice_prompt.strip():
        hint = f"{hint}\n\n{llm_voice_prompt.strip()}"
    return {"role": "system", "content": hint}


# ── OpenAI STT (GA realtime transcription session) ──────────────────────────
def _stt_session_config() -> dict:
    return {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": settings.OPENAI_STT_MODEL},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": settings.OPENAI_STT_VAD_THRESHOLD,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": settings.OPENAI_STT_SILENCE_MS,
                    },
                }
            },
        },
    }


async def _open_stt_session(api_key: str):
    stt_ws = await websockets.connect(
        f"{settings.OPENAI_REALTIME_URL}?intent=transcription",
        additional_headers={"Authorization": f"Bearer {api_key}"},
        open_timeout=10,
        max_size=2**23,
    )
    await stt_ws.send(json.dumps(_stt_session_config()))
    return stt_ws


# ── Connection state ─────────────────────────────────────────────────────────
class _VoiceSession:
    """Per-connection context shared by the relay/listener/worker tasks."""

    def __init__(self) -> None:
        self.transcripts: asyncio.Queue[str] = asyncio.Queue()
        self.barge_in = asyncio.Event()
        self.send_lock = asyncio.Lock()

    # Populated at connect time (plain values only — no ORM objects, so no
    # lazy-loading against a closed DB session later).
    user_session_db_id: int
    user_id: int
    user_email: Optional[str]
    agent_id: str
    conversation_id: str
    api_key: str
    voice_hint: dict


async def _emit(client_ws: WebSocket, ctx: _VoiceSession, payload: dict) -> None:
    async with ctx.send_lock:
        await client_ws.send_json(payload)


# ── Tasks ────────────────────────────────────────────────────────────────────
async def _relay_client_to_stt(client_ws: WebSocket, stt_ws, ctx: _VoiceSession) -> None:
    """Forward mic audio from the browser to the OpenAI transcription session."""
    while True:
        message = await client_ws.receive()
        if message["type"] == "websocket.disconnect":
            return

        if "text" in message and message["text"]:
            try:
                payload = json.loads(message["text"])
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "input_audio_buffer.append" and payload.get("audio"):
                await stt_ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": payload["audio"],
                }))

        elif "bytes" in message and message["bytes"]:
            await stt_ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(message["bytes"]).decode("ascii"),
            }))


async def _listen_stt(stt_ws, client_ws: WebSocket, ctx: _VoiceSession) -> None:
    """Consume STT events: barge-in signal + final transcripts."""
    async for raw in stt_ws:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type", "")

        if event_type == "input_audio_buffer.speech_started":
            ctx.barge_in.set()
            await _emit(client_ws, ctx, {"type": "input_audio_buffer.speech_started"})

        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = (event.get("transcript") or "").strip()
            if transcript:
                await ctx.transcripts.put(transcript)

        elif event_type == "error":
            log.error(f"Voice STT: error event from OpenAI: {event.get('error')}")

    # OpenAI closed the STT socket — surface it and end the session.
    log.warning("Voice STT: transcription session closed by OpenAI")
    raise ConnectionError("OpenAI transcription session closed")


async def _turn_worker(client_ws: WebSocket, ctx: _VoiceSession) -> None:
    """Process transcripts strictly one at a time through AgentRuntime."""
    while True:
        transcript = await ctx.transcripts.get()
        ctx.barge_in.clear()

        # Show the user's bubble immediately.
        await _emit(client_ws, ctx, {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": transcript,
        })

        try:
            await _process_turn(client_ws, ctx, transcript)
        except (WebSocketDisconnect, RuntimeError):
            raise  # client gone — let the session tear down
        except Exception:
            log.exception("Voice turn failed")
            await _speak_simple(client_ws, ctx, APOLOGY_LINE)


async def _process_turn(client_ws: WebSocket, ctx: _VoiceSession, transcript: str) -> None:
    from ai_agent.core.agent_runtime import get_agent_runtime
    from ai_agent.core.guardrails import get_guardrails_service
    from ai_agent.core.usage_tracker import get_usage_tracker

    db_session = Session(engine)
    try:
        # Guardrails + quota first — same order and semantics as
        # /send-message-v2 (nothing is persisted when the turn is refused).
        guardrails = get_guardrails_service()
        violations = guardrails.check_input(
            text=transcript, user_id=ctx.user_id, agent_id=ctx.agent_id,
        )
        critical = [v for v in violations if v.severity in ("high", "critical")]
        if critical:
            await _speak_simple(
                client_ws, ctx, guardrails.get_remediation(critical[0])
            )
            return

        quota = get_usage_tracker(db_session).check_quota(
            user_id=ctx.user_id,
            agent_id=ctx.agent_id,
            quota_type="tokens",
            requested_amount=1000,
        )
        if not quota["allowed"]:
            await _speak_simple(
                client_ws, ctx,
                quota.get("message") or "You've reached your usage limit.",
            )
            return

        history = _load_history(db_session, ctx.conversation_id)
        user_msg = _save_message(
            db_session,
            session_id=ctx.user_session_db_id,
            agent_id=ctx.agent_id,
            conversation_id=ctx.conversation_id,
            role=ConversationRole.USER,
            content=transcript,
        )

        messages = history + [ctx.voice_hint, {"role": "user", "content": transcript}]

        runtime = await get_agent_runtime(db_session)
        response = await runtime.run(
            agent_id=ctx.agent_id,
            user_id=ctx.user_id,
            session_id=ctx.user_session_db_id,
            conversation_id=ctx.conversation_id,
            messages=messages,
            mode=RunMode.CHAT,
            input_message_id=user_msg.id,
            stream=False,
            api_key=ctx.llm_api_key,
            user_email=ctx.user_email,
        )

        assistant_content = "\n\n".join(
            md for md in (content_block_to_markdown(b) for b in response.blocks) if md
        ) or FALLBACK_SPOKEN_LINE

        _save_message(
            db_session,
            session_id=ctx.user_session_db_id,
            agent_id=ctx.agent_id,
            conversation_id=ctx.conversation_id,
            role=ConversationRole.ASSISTANT,
            content=assistant_content,
        )

        spoken = _derive_spoken_text(response.blocks)
        run_id = response.run_id

        # Full v2 payload for the client — the same fields /send-message-v2
        # returns, so the chat can render thinking steps, tool calls, tables
        # and download cards identically for voice turns.
        try:
            full = json.loads(response.model_dump_json())
        except AttributeError:  # pydantic v1 fallback
            full = json.loads(response.json())
        agent_payload = {
            key: full.get(key)
            for key in (
                "blocks", "citations", "tool_calls", "tool_results",
                "usage", "run_id", "artifacts",
            )
        }
    finally:
        db_session.close()

    # Text reaches the client before any audio: instant visible feedback
    # while TTS spins up.
    await _emit(client_ws, ctx, {"type": "response.created", "response": {"id": run_id}})
    await _emit(client_ws, ctx, {
        "type": "response.agent.completed",
        "response_id": run_id,
        "response": agent_payload,
    })
    await _emit(client_ws, ctx, {"type": "response.audio_transcript.delta", "delta": assistant_content})
    await _emit(client_ws, ctx, {"type": "response.audio_transcript.done", "transcript": assistant_content})

    # Audio is skipped when the user has already moved on (a newer transcript
    # is queued, or they started speaking again) — matching the frontend,
    # which hard-stops playback on speech_started.
    if ctx.transcripts.empty() and not ctx.barge_in.is_set():
        await _stream_tts(client_ws, ctx, spoken, run_id)

    await _emit(client_ws, ctx, {"type": "response.done"})


async def _speak_simple(client_ws: WebSocket, ctx: _VoiceSession, text: str) -> None:
    """Emit a plain text+audio response outside a normal agent run
    (guardrail refusals, quota limits, error apologies)."""
    response_id = str(uuid4())
    await _emit(client_ws, ctx, {"type": "response.created", "response": {"id": response_id}})
    await _emit(client_ws, ctx, {"type": "response.audio_transcript.delta", "delta": text})
    await _emit(client_ws, ctx, {"type": "response.audio_transcript.done", "transcript": text})
    if ctx.transcripts.empty() and not ctx.barge_in.is_set():
        await _stream_tts(client_ws, ctx, text, response_id)
    await _emit(client_ws, ctx, {"type": "response.done"})


async def _stream_tts(
    client_ws: WebSocket, ctx: _VoiceSession, text: str, response_id: str
) -> None:
    """Stream OpenAI TTS audio to the client as response.audio.delta events.

    Output format "pcm" is PCM16 mono 24kHz — exactly what the frontend's
    AudioContext plays. Chunks are kept even-sized so Int16Array decoding
    on the client never sees a half sample.
    """
    client = AsyncOpenAI(api_key=ctx.openai_key)
    leftover = b""
    try:
        async with client.audio.speech.with_streaming_response.create(
            model=settings.OPENAI_TTS_MODEL,
            voice=settings.OPENAI_TTS_VOICE,
            input=text,
            response_format="pcm",
        ) as tts_response:
            async for chunk in tts_response.iter_bytes(chunk_size=TTS_CHUNK_BYTES):
                if ctx.barge_in.is_set() or not ctx.transcripts.empty():
                    log.info("Voice TTS: interrupted — stopping audio stream")
                    return
                chunk = leftover + chunk
                if len(chunk) % 2:
                    chunk, leftover = chunk[:-1], chunk[-1:]
                else:
                    leftover = b""
                if not chunk:
                    continue
                await _emit(client_ws, ctx, {
                    "type": "response.audio.delta",
                    "response_id": response_id,
                    "delta": base64.b64encode(chunk).decode("ascii"),
                })
    except Exception:
        # Text already reached the chat; a TTS failure shouldn't kill the turn.
        log.exception("Voice TTS: streaming failed")


# ── Endpoint ─────────────────────────────────────────────────────────────────
@router.websocket("/voice")
async def voice_agent_pipeline(
    client_ws: WebSocket,
    agent_id: str = Query(...),
    session_id: str = Query(...),
    conversation_id: str = Query(...),
):
    """Voice session routed through AgentRuntime.

    Connect: ws://host/api/v1/message/voice?agent_id=X&session_id=Y&conversation_id=Z
    """
    await client_ws.accept()
    log.info(
        f"Voice WS: client connected (session={session_id}, agent={agent_id}, "
        f"convo={conversation_id})"
    )

    ctx = _VoiceSession()

    # Connect-time validation and key resolution. Everything the turn worker
    # needs later is copied out as plain values; the DB session is closed
    # before the long-lived relay starts.
    db_session = Session(engine)
    try:
        user_session = db_session.exec(
            select(UserSession).where(UserSession.session_id == session_id)
        ).first()
        if not user_session:
            await client_ws.send_json({
                "type": "error",
                "error": {"type": "session_not_found",
                          "message": f"Session '{session_id}' not found"},
            })
            await client_ws.close(code=4004, reason="Session not found")
            return

        agent_settings = db_session.exec(
            select(AIAgentSettings).where(AIAgentSettings.agent_id == agent_id)
        ).first()
        agent = db_session.get(AIAgent, agent_id)

        # The voice transport uses OpenAI Realtime STT and OpenAI TTS.
        # Agent-specific instructions appended to the voice-turn prompt.
        llm_voice_prompt = None
        api_key = get_api_key(db_session, agent_settings.api_key) 

        if agent_settings and agent:
            llm_voice_prompt = agent_settings.llm_voice_prompt
            # Ensure the agent uses OpenAI, as STT/TTS currently require an OpenAI key.

            await client_ws.close(code=1011, reason="Voice not configured")
            return

        user = db_session.get(User, user_session.user_id)

        if not user_session.title:
            user_session.title = "Voice conversation"
            db_session.add(user_session)
            db_session.commit()

        ctx.user_session_db_id = user_session.id
        ctx.user_id = user_session.user_id
        ctx.user_email = user.email if user else None
        ctx.agent_id = agent_id
        ctx.conversation_id = conversation_id
        ctx.api_key = api_key
        ctx.voice_hint = _build_voice_hint(llm_voice_prompt)
    finally:
        db_session.close()

    # Open the OpenAI transcription session.
    try:
        stt_ws = await _open_stt_session(ctx.api_key)
        log.info("Voice WS: OpenAI transcription session connected")
    except Exception as e:
        log.error(f"Voice WS: failed to connect OpenAI STT: {e}", exc_info=True)
        try:
            await client_ws.send_json({
                "type": "error",
                "error": {"type": "stt_unavailable",
                          "message": "Speech-to-text service unavailable"},
            })
            await client_ws.close(code=1011, reason="STT unavailable")
        except Exception:
            pass
        return

    tasks = [
        asyncio.create_task(_relay_client_to_stt(client_ws, stt_ws, ctx)),
        asyncio.create_task(_listen_stt(stt_ws, client_ws, ctx)),
        asyncio.create_task(_turn_worker(client_ws, ctx)),
    ]
    try:
        # First task to finish (client disconnect, STT drop, or a fatal send
        # failure) tears the whole session down.
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            exc = task.exception()
            if exc and not isinstance(exc, (WebSocketDisconnect, ConnectionError)):
                log.error(f"Voice WS: task failed: {exc}", exc_info=exc)
    except Exception as e:
        log.error(f"Voice WS: unexpected error: {e}", exc_info=True)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await stt_ws.close()
        except Exception:
            pass
        log.info("Voice WS: session ended")
