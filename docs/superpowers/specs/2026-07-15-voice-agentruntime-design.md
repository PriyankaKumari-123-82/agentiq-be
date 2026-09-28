# AG-52: Voice pipeline through AgentRuntime (STT → Agent → TTS)

**Date:** 2026-07-15
**Ticket:** AG-52
**Status:** Approved (Approach A)

## Problem

The voice endpoint (`ai_agent/api/v1/voice.py`) is a bidirectional proxy to the
OpenAI Realtime API. OpenAI's own model answers the user, so AgentRuntime — and
with it SQL/MCP tools, RAG, guardrails, usage tracking — is completely bypassed
during voice sessions. Additionally, the proxy uses the retired beta Realtime
protocol (`OpenAI-Beta: realtime=v1`); OpenAI now closes those connections with
`invalid_request_error.beta_api_shape_disabled`, so voice is broken outright
(FE shows "Voice connection lost", WS close code 1006).

## Goals

1. Every voice reply is produced by `AgentRuntime.run()` — the same pipeline as
   `/send-message-v2` (tools, SQL, RAG, guardrails, quota, usage tracking).
2. OpenAI-only speech stack: `gpt-4o-transcribe` (STT, GA Realtime
   transcription session) and `gpt-4o-mini-tts` (TTS via `/v1/audio/speech`).
3. Zero frontend changes. The FE hook (`useVoiceChat.ts`) keeps working
   unmodified.
4. Remove Sarvam entirely (code, config, dependency).

## Non-goals

- Restructuring AgentRuntime internals for latency.
- Streaming partial agent responses (runtime streaming is a TODO stub).
- WebRTC or client-side STT.

## Frontend contract (must be preserved verbatim)

FE → BE: `{"type": "input_audio_buffer.append", "audio": "<b64 PCM16 mono 24kHz>"}`
(text frames; raw binary frames also possible — wrap them the same way).

BE → FE events the hook consumes:

| Event | Fields | FE behavior |
|---|---|---|
| `input_audio_buffer.speech_started` | — | stops assistant playback (barge-in) |
| `conversation.item.input_audio_transcription.completed` | `transcript` | shows user bubble |
| `response.created` | `response.id` | cancels stale audio |
| `response.agent.completed` | `response_id`, `response` (same payload shape as `/send-message-v2`: blocks, tool_calls, tool_results, citations, usage, artifacts) | renders the rich v2 message incl. the "Thinking" panel; added post-MVP for text-chat parity |
| `response.audio_transcript.delta` | `delta` | partial assistant text |
| `response.audio_transcript.done` | `transcript` | final assistant text |
| `response.audio.delta` | `delta` (b64 PCM16@24kHz), `response_id` | plays audio |
| `response.done` | — | schedules audio cleanup |

## Architecture

Three asyncio tasks per WS connection plus a serial turn queue:

1. **client relay** — receives FE frames, forwards `input_audio_buffer.append`
   to the OpenAI STT WebSocket unchanged (wraps binary frames if any).
2. **STT event listener** — consumes OpenAI transcription-session events:
   - `input_audio_buffer.speech_started` → relay to FE + set barge-in flag
     (stops any in-flight TTS streaming).
   - `conversation.item.input_audio_transcription.completed` → push transcript
     onto the turn queue.
3. **turn worker** — processes transcripts strictly one at a time:
   1. Guardrails `check_input`; on critical violation, speak the remediation
      text instead of running the agent.
   2. Quota check (`usage_tracker.check_quota`), same semantics as v2.
   3. Save user `Message` (`conversation_type=VOICE`).
   4. Build history: all prior messages of the conversation (capped at last
      30) + a system "voice style" hint (concise, conversational; includes
      `agent_settings.llm_voice_prompt` when set) + the new user message.
   5. `runtime.run(...)` with a fresh DB session, per-agent LLM API key,
      `mode=CHAT`.
   6. Save assistant `Message` (blocks rendered via
      `content_block_to_markdown`, `conversation_type=VOICE`).
   7. Emit `response.created` (id = run_id), `response.audio_transcript.delta`
      (full text once), `response.audio_transcript.done` — text reaches the FE
      before any audio.
   8. If a newer transcript is already queued or barge-in fired → skip audio.
      Otherwise stream TTS (`/v1/audio/speech`, `response_format="pcm"` =
      PCM16 mono 24kHz) and emit `response.audio.delta` chunks (b64, even-sized).
   9. Emit `response.done`.

### OpenAI STT session (GA protocol)

- URL: `wss://api.openai.com/v1/realtime?intent=transcription`
- Headers: `Authorization: Bearer <key>` only (no `OpenAI-Beta`).
- Configure with:

```json
{
  "type": "session.update",
  "session": {
    "type": "transcription",
    "audio": {
      "input": {
        "format": {"type": "audio/pcm", "rate": 24000},
        "transcription": {"model": "gpt-4o-transcribe"},
        "turn_detection": {
          "type": "server_vad",
          "threshold": 0.5,
          "prefix_padding_ms": 300,
          "silence_duration_ms": 500
        }
      }
    }
  }
}
```

API key: `agent_settings.get_api_key("openai")` falling back to
`settings.OPENAI_API_KEY` (same as today).

### Spoken-text derivation (`_derive_spoken_text`)

- MARKDOWN blocks: strip markdown syntax (headings, emphasis, inline code,
  links → link text, code fences dropped), drop pipe-table lines (consecutive
  runs replaced by one table notice).
- TABLE → "I've added a table with the results to the chat." (once)
- CHART → "I've added a chart to the chat." (once)
- ARTIFACT → "The file is ready to download in the chat." (once)
- CODE → "I've put a code snippet in the chat." (once)
- Cap ≈ 900 chars, append "More details are in the chat." when truncated.

### Turn-taking / barge-in

- Serial worker: a turn always completes its agent run and DB writes (same
  semantics as text chat), but audio is skipped when superseded.
- `speech_started` during TTS streaming stops the stream immediately — this
  mirrors the FE playback manager, which hard-stops playback on the same event.
- No task cancellation of agent runs → no orphaned RUNNING `AgentRun` records.

### Error handling

- STT WS connect failure/drop → `{"type": "error", ...}` to FE, close WS
  (FE toasts and lets the user retry).
- Agent run failure → speak/emit a short apology text; session stays alive.
- TTS failure → log, emit `response.done` (text already delivered).
- FE disconnect → cancel tasks, close STT WS, close DB sessions.

## Config changes (`core/config.py`)

Remove: `SARVAM_API_KEY`, `SARVAM_STT_MODEL`, `SARVAM_STT_LANGUAGE`,
`SARVAM_STT_MODE`, `OPENAI_REALTIME_MODEL`, `OPENAI_REALTIME_VOICE`.

Add:

| Setting | Default |
|---|---|
| `OPENAI_STT_MODEL` | `gpt-4o-transcribe` |
| `OPENAI_STT_VAD_THRESHOLD` | `0.5` |
| `OPENAI_STT_SILENCE_MS` | `500` |
| `OPENAI_TTS_MODEL` | `gpt-4o-mini-tts` |
| `OPENAI_TTS_VOICE` | `alloy` |

`OPENAI_REALTIME_URL` stays (base URL; transcription URL derives from it).

## Other removals

- `sarvamai` from `pyproject.toml`.
- `create_voice_instructions()` from `core/prompt.py` (voice.py was the only
  caller).
- `SentenceTransformer` embedder + `_search_vector_context` in voice.py
  (RAG now runs inside AgentRuntime via RAGService, as for text chat).

## Performance notes

- Response text is emitted before TTS begins (instant perceived feedback).
- TTS audio streamed chunk-by-chunk while OpenAI generates it.
- History capped at 30 messages to bound planner prompt size.
- Removing the module-level `sentence_transformers` import cuts app import
  cost; the lazy ~2s embedder load on first voice turn disappears.
- Known latency floor: AgentRuntime runs 3+ sequential LLM calls
  (plan → select → synthesize), so expect ~3–10s of silence before audio on
  tool-using turns. Acceptable for the demo; runtime streaming is a future
  ticket.

## Testing

- Unit tests (pytest, no network): `_derive_spoken_text` (prose pass-through,
  markdown stripping, table-line removal, canned notices, truncation);
  turn-superseding logic if separable.
- Manual E2E script `scripts/test_voice_ws.py`: connects to the WS, streams a
  WAV file as `input_audio_buffer.append` frames at real-time pace, prints all
  received events, saves received audio to a file. Verifies the full loop
  without a browser.
- Browser E2E via existing FE as final check.

## References

- OpenAI GA realtime transcription: https://developers.openai.com/api/docs/guides/realtime-transcription
- GA WebSocket connection: https://developers.openai.com/api/docs/guides/realtime-websocket
- VAD config: https://developers.openai.com/api/docs/guides/realtime-vad
