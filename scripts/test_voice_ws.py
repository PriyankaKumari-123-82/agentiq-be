"""Manual E2E test for the voice pipeline — no browser or microphone needed.

Streams a WAV file to the voice WebSocket exactly the way the frontend does
(base64 PCM16 mono 24kHz in ``input_audio_buffer.append`` events, real-time
pace), prints every event received, and saves the TTS reply audio.

Usage (host, against the dockerized backend):

    python scripts/test_voice_ws.py \
        --url "ws://localhost:8002/api/v1/message/voice" \
        --agent-id <AGENT_ID> --session-id <SESSION_ID> \
        --conversation-id <CONVO_ID> \
        --wav question.wav

The WAV must be PCM16. Mono 24kHz is ideal; other rates/channels are
converted on the fly (crude linear resample — fine for testing).

Reply audio is written to ``voice_reply.wav`` (PCM16 mono 24kHz).

Requires: pip install websockets
"""
import argparse
import asyncio
import base64
import json
import struct
import sys
import time
import wave

import websockets

CHUNK_MS = 170  # ≈ what the frontend sends per ScriptProcessor buffer
TARGET_RATE = 24000


def load_wav_as_pcm24k(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if width != 2:
        sys.exit(f"Expected 16-bit PCM wav, got sample width {width}")

    samples = list(struct.unpack(f"<{len(frames) // 2}h", frames))
    if n_channels > 1:  # take channel 0
        samples = samples[::n_channels]

    if rate != TARGET_RATE:
        ratio = rate / TARGET_RATE
        out_len = int(len(samples) / ratio)
        resampled = []
        for i in range(out_len):
            src = i * ratio
            idx = int(src)
            frac = src - idx
            if idx + 1 < len(samples):
                val = int(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
            else:
                val = samples[idx]
            resampled.append(max(-32768, min(32767, val)))
        samples = resampled
        print(f"[i] Resampled {rate}Hz → {TARGET_RATE}Hz")

    return struct.pack(f"<{len(samples)}h", *samples)


async def run(args: argparse.Namespace) -> None:
    pcm = load_wav_as_pcm24k(args.wav)
    url = (
        f"{args.url}?agent_id={args.agent_id}"
        f"&session_id={args.session_id}&conversation_id={args.conversation_id}"
    )
    print(f"[i] Connecting: {url}")

    reply_audio = bytearray()
    chunk_bytes = int(TARGET_RATE * 2 * CHUNK_MS / 1000)

    async with websockets.connect(url, max_size=2**23) as ws:
        print("[i] Connected. Streaming audio at real-time pace…")

        async def sender():
            for i in range(0, len(pcm), chunk_bytes):
                chunk = pcm[i:i + chunk_bytes]
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }))
                await asyncio.sleep(CHUNK_MS / 1000)
            print("[i] Audio fully sent. Sending trailing silence…")
            silence = base64.b64encode(b"\x00" * chunk_bytes).decode("ascii")
            for _ in range(12):  # ~2s so server VAD detects end of speech
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append", "audio": silence,
                }))
                await asyncio.sleep(CHUNK_MS / 1000)

        async def receiver():
            start = time.monotonic()
            got_done = False
            while True:
                try:
                    raw = await asyncio.wait_for(
                        ws.recv(), timeout=args.timeout if not got_done else 3
                    )
                except asyncio.TimeoutError:
                    if got_done:
                        return
                    print(f"[!] No event within {args.timeout}s — giving up")
                    return
                event = json.loads(raw)
                etype = event.get("type", "?")
                t = time.monotonic() - start
                if etype == "response.audio.delta":
                    reply_audio.extend(base64.b64decode(event["delta"]))
                    print(f"[{t:6.2f}s] {etype} (+{len(event['delta']) * 3 // 4} bytes)")
                elif etype == "response.audio_transcript.delta":
                    print(f"[{t:6.2f}s] {etype}: {event.get('delta', '')[:120]}…")
                else:
                    print(f"[{t:6.2f}s] {etype}: "
                          f"{json.dumps({k: v for k, v in event.items() if k != 'type'})[:200]}")
                # A turn is "open" from response.created until its
                # response.done — only exit while no turn is in flight.
                if etype == "response.created":
                    got_done = False
                elif etype == "response.done":
                    got_done = True

        await asyncio.gather(sender(), receiver())

    if reply_audio:
        with wave.open(args.out, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_RATE)
            wf.writeframes(bytes(reply_audio))
        print(f"[✓] Reply audio saved to {args.out} "
              f"({len(reply_audio) / (TARGET_RATE * 2):.1f}s)")
    else:
        print("[!] No reply audio received")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://localhost:8002/api/v1/message/voice")
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--wav", required=True, help="16-bit PCM WAV question")
    parser.add_argument("--out", default="voice_reply.wav")
    parser.add_argument("--timeout", type=float, default=90.0,
                        help="Max seconds to wait for agent+TTS events")
    args = parser.parse_args()
    asyncio.run(run(args))
