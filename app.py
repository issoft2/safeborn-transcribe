import asyncio
import io
import logging
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import math


def _detect_cpu_count() -> int:
    """os.cpu_count() reports the HOST's total cores, not what a container is
    actually allotted — in a Docker/K8s deployment with a CPU limit set, this
    silently oversizes every thread/pool calculation below. Prefer the cgroup
    v2 quota (/sys/fs/cgroup/cpu.max: "$MAX $PERIOD" in microseconds, where
    max/period is the real usable core count) when it's available and finite.
    """
    cpu_max_path = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max_path.exists():
        try:
            max_str, period_str = cpu_max_path.read_text().split()
            if max_str != "max":
                quota = int(max_str) / int(period_str)
                if quota > 0:
                    return max(1, math.ceil(quota))  # round down: safer to under- than over-allocate
        except (ValueError, OSError):
            pass
    return os.cpu_count() or 2


_CPU_COUNT = _detect_cpu_count()

os.environ.setdefault("OMP_NUM_THREADS", str(max(1, _CPU_COUNT - 1)))
os.environ.setdefault("MKL_NUM_THREADS", str(max(1, _CPU_COUNT - 1)))

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from kokoro import KPipeline
from faster_whisper import WhisperModel


app = FastAPI(title="SafeBorn Voice Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logger = logging.getLogger("safeborn-voice")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False

logger.info("Loading Whisper Speech-to-Text model...")
# Optimized Whisper initialization to prevent CPU cache thrashing
stt_model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=2, num_workers=1)

logger.info("Loading Kokoro neural voice pipeline into memory...")
tts_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
logger.info("Voice engine services are warm and ready!")


torch.set_num_threads(max(1, _CPU_COUNT - 1))
torch.set_num_interop_threads(1)

tts_executor = ThreadPoolExecutor(max_workers=1)
stt_executor = ThreadPoolExecutor(max_workers=min(2, _CPU_COUNT))

logger.info(
    f"CPU sizing: detected={_CPU_COUNT} (host os.cpu_count()={os.cpu_count()}), "
    f"torch.set_num_threads={torch.get_num_threads()}, "
    f"interop={torch.get_num_interop_threads()}"
)


logging.info(
    f"CPU affinity: {len(os.sched_getaffinity(0))} CPUs "
    f"{os.sched_getaffinity(0)}"
)

logging.info(
    f"""
CPU config:
Detected CPUs: {_CPU_COUNT}
OMP: {os.getenv('OMP_NUM_THREADS')}
MKL: {os.getenv('MKL_NUM_THREADS')}
Torch threads: {torch.get_num_threads()}
Torch interop: {torch.get_num_interop_threads()}
"""
)

def _run_transcription(audio_bytes: bytes) -> str:
    started = time.monotonic()
    audio_file = io.BytesIO(audio_bytes)
    
    segments, _info = stt_model.transcribe(
        audio_file,
        beam_size=1,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    text = " ".join(segment.text for segment in segments).strip()
    logger.info(f"[timing] transcription took {time.monotonic() - started:.2f}s")
    return text


def _run_tts(clean_text: str, speed: float) -> bytes:
    overall_start = time.monotonic()

    logger.info("=" * 70)
    logger.info("TTS Request: %d chars", len(clean_text))
    logger.info("Text: %r", clean_text)

    # -------------------------------------------------------
    # Stage 1 - Initialize Kokoro generator
    # NOTE:
    # Frontend already performs sentence/chunk splitting.
    # Do not split here again. Backend receives one speech unit
    # and focuses only on synthesis.
    # -------------------------------------------------------
    t = time.monotonic()

    generator = tts_pipeline(
        clean_text,
        voice="af_heart",
        speed=speed,
    )

    logger.info(
        "[Stage 1] Generator created in %.3f sec",
        time.monotonic() - t,
    )

    # -------------------------------------------------------
    # Stage 2 - Generate waveform segments
    #
    # Kokoro may internally yield multiple segments even though
    # frontend already sent a single speech chunk.
    # -------------------------------------------------------
    audio_segments = []

    stage2_start = time.monotonic()
    last_segment_time = stage2_start

    for idx, (_gs, _ps, audio) in enumerate(generator, start=1):

        now = time.monotonic()

        logger.info(
            "[Stage 2] Kokoro segment %d generated in %.3f sec",
            idx,
            now - last_segment_time,
        )

        last_segment_time = now

        if audio is not None and len(audio) > 0:
            audio_segments.append(audio)

    logger.info(
        "[Stage 2] Total synthesis time: %.3f sec",
        time.monotonic() - stage2_start,
    )

    if not audio_segments:
        raise ValueError("No audio generated")

    # -------------------------------------------------------
    # Stage 3 - Merge waveform segments
    # -------------------------------------------------------
    t = time.monotonic()

    combined_audio = np.concatenate(audio_segments)

    logger.info(
        "[Stage 3] Audio concatenation: %.3f sec",
        time.monotonic() - t,
    )

    # -------------------------------------------------------
    # Stage 4 - Encode WAV
    # -------------------------------------------------------
    t = time.monotonic()

    wav_io = io.BytesIO()

    sf.write(
        wav_io,
        combined_audio,
        24000,
        format="WAV",
        subtype="PCM_16",
    )

    wav_bytes = wav_io.getvalue()

    logger.info(
        "[Stage 4] WAV encoding: %.3f sec",
        time.monotonic() - t,
    )

    # -------------------------------------------------------
    # Total request time
    # -------------------------------------------------------
    logger.info(
        "[TOTAL] %.3f sec",
        time.monotonic() - overall_start,
    )

    logger.info("=" * 70)

    return wav_bytes


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    loop = asyncio.get_event_loop()
    try:
        transcription = await loop.run_in_executor(stt_executor, _run_transcription, audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"text": transcription}


_PCM_SAMPLE_RATE = 24000  # Kokoro output rate
_PCM_CHANNELS = 1
_PCM_FORMAT_INT16LE = 1  # sample_format code, matches client decoder


def _pcm_header_bytes() -> bytes:
    """
    8-byte binary header the progressive stream prepends before the PCM
    body. Kept in the response body (not HTTP headers) so it survives
    the chat_service proxy and gateway without needing CORS
    Access-Control-Expose-Headers gymnastics.

    Layout (little-endian):
      offset 0, uint32: sample_rate
      offset 4, uint16: channels
      offset 6, uint16: sample_format  (1 = int16le)
    """
    return struct.pack("<IHH", _PCM_SAMPLE_RATE, _PCM_CHANNELS, _PCM_FORMAT_INT16LE)


def _kokoro_segments(clean_text: str, speed: float):
    """
    Wraps the raw Kokoro generator so caller only sees non-empty audio
    segments and gets per-segment timing logs, matching what _run_tts
    logs today. Blocking generator — must be consumed on tts_executor,
    never on the event loop thread.
    """
    overall_start = time.monotonic()
    logger.info("=" * 70)
    logger.info("TTS progressive request: %d chars", len(clean_text))
    logger.info("Text: %r", clean_text)

    generator = tts_pipeline(clean_text, voice="af_heart", speed=speed)

    stage_start = time.monotonic()
    last_segment_time = stage_start
    total_samples = 0

    for idx, (_gs, _ps, audio) in enumerate(generator, start=1):
        now = time.monotonic()
        seg_samples = len(audio) if audio is not None else 0
        logger.info(
            "[Stage 2] Kokoro segment %d generated in %.3f sec (%d samples)",
            idx, now - last_segment_time, seg_samples,
        )
        last_segment_time = now
        if audio is not None and seg_samples > 0:
            total_samples += seg_samples
            yield audio

    logger.info(
        "[TOTAL] progressive stream: %.3f sec, %d samples (~%.2f sec audio)",
        time.monotonic() - overall_start,
        total_samples,
        total_samples / _PCM_SAMPLE_RATE if total_samples else 0.0,
    )
    logger.info("=" * 70)


def _float32_to_int16le_bytes(audio) -> bytes:
    """
    Kokoro yields torch.Tensor segments (float32 waveform in [-1, 1]),
    not numpy arrays — np.concatenate() in the buffered /tts-stream path
    silently auto-converts these via numpy's array protocol, which is
    why that path never surfaced this. Tensors don't have numpy's
    .astype(), so it must be converted explicitly here before encoding
    to signed 16-bit PCM little-endian (the format the client's Web
    Audio API decoder expects — matches _PCM_FORMAT_INT16LE header code).
    """
    arr = audio.detach().cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio)
    clipped = np.clip(arr, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


@app.get("/tts-progressive")
async def text_to_speech_progressive(text: str = Query(...), speed: float = Query(0.90)):
    """
    Streams raw PCM as each Kokoro segment is produced, instead of
    concatenating all segments into a WAV and returning it in one go
    (see /tts-stream, which still does the buffered thing for the
    native app path that needs a complete file to play).

    First bytes reach the client in ~200-400 ms — the time it takes
    Kokoro to produce its first segment — rather than after the entire
    utterance has been synthesized (~1-2 s for a typical reply).
    Combined with the frontend's incremental-per-sentence streaming,
    that's the difference between "delay, then AI speaks" and "AI
    starts speaking almost immediately."

    Body wire format: 8-byte header then raw int16 LE PCM samples.
    See _pcm_header_bytes() for the layout.
    """
    clean_text = " ".join(text.splitlines())
    loop = asyncio.get_event_loop()

    # asyncio.run_in_executor doesn't understand StopIteration — in an
    # async context it becomes RuntimeError. Use a sentinel to make the
    # generator exhaustion explicit and portable.
    _SENTINEL = object()

    def _safe_next(gen):
        try:
            return next(gen)
        except StopIteration:
            return _SENTINEL

    async def stream_body():
        # Prepend the 8-byte format header so the client knows sample
        # rate / channels / format before decoding any samples.
        yield _pcm_header_bytes()

        segments = _kokoro_segments(clean_text, speed)
        try:
            while True:
                segment = await loop.run_in_executor(tts_executor, _safe_next, segments)
                if segment is _SENTINEL:
                    break
                yield _float32_to_int16le_bytes(segment)
        except Exception as exc:  # noqa: BLE001 - anything here is a synth failure worth logging + terminating the stream
            logger.error("progressive_tts_failed", exc_info=exc)

    return StreamingResponse(
        stream_body(),
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/tts-stream")
async def text_to_speech(text: str = Query(...), speed: float = Query(0.90)):
    # Clean up structural text breaks cleanly
    clean_text = " ".join(text.splitlines())

    loop = asyncio.get_event_loop()
    try:
        wav_bytes = await loop.run_in_executor(tts_executor, _run_tts, clean_text, speed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Generation failure: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "inline; filename=\"speech.wav\"",
            "Cache-Control": "no-cache"
        }
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}