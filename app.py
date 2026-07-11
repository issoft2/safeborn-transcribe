import asyncio
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from kokoro import KPipeline
from faster_whisper import WhisperModel


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
                    return max(1, int(quota))  # round down: safer to under- than over-allocate
        except (ValueError, OSError):
            pass
    return os.cpu_count() or 2


_CPU_COUNT = _detect_cpu_count()

# Must be set before importing torch/kokoro/faster_whisper — some backends
# (OpenMP/MKL) read these once at import/initialization time, not per-call.
# Without this, PyTorch defaults to grabbing every visible CPU core for its
# internal math ops on every single inference call, with zero awareness that
# another synthesis job might be running concurrently.
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, _CPU_COUNT - 1)))
os.environ.setdefault("MKL_NUM_THREADS", str(max(1, _CPU_COUNT - 1)))

app = FastAPI(title="SafeBorn Voice Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# print() follows normal stdout buffering — line-buffered on a real terminal,
# but fully block-buffered (waits for several KB) once stdout is piped/redirected,
# which is exactly what happens inside a container or behind a process manager.
# logging.StreamHandler flushes after every record regardless, which is why
# uvicorn's own access logs (which go through `logging`) show up fine while our
# print() calls didn't. Configuring our own handler explicitly here rather than
# relying on however uvicorn's root logger happens to be set up for this process.
logger = logging.getLogger("safeborn-voice")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False

logger.info("Loading Whisper Speech-to-Text model...")
# Optimized Whisper initialization to prevent CPU cache thrashing
stt_model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=2)

logger.info("Loading Kokoro neural voice pipeline into memory...")
tts_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
logger.info("Voice engine services are warm and ready!")

# torch.set_num_threads always takes precedence over the OMP/MKL env vars set
# above, so this is the actual authoritative bound — the env vars are a
# best-effort backstop for anything that reads them directly instead of going
# through torch. Leaving one core of headroom for the event loop / request
# handling rather than letting Kokoro claim the entire quota.
torch.set_num_threads(max(1, _CPU_COUNT - 1))

# stt_model.transcribe(...) and tts_pipeline(...) are blocking, CPU-bound calls.
# Running them directly inside `async def` routes stalls the event loop for
# every other request (including /health) until they finish. Offload them here.
#
# IMPORTANT: match these to the *correct* route below — /transcribe must use
# stt_executor and /tts-stream must use tts_executor. They were swapped once
# already (stt_executor had max_workers=2 but was wired to /tts-stream), which
# silently let two TTS synthesis calls run concurrently and contend for CPU —
# exactly the problem torch.set_num_threads above is trying to prevent.
tts_executor = ThreadPoolExecutor(max_workers=1)
stt_executor = ThreadPoolExecutor(max_workers=2)

logger.info(
    f"CPU sizing: detected={_CPU_COUNT} (host os.cpu_count()={os.cpu_count()}), "
    f"torch.set_num_threads={torch.get_num_threads()}, "
    f"interop={torch.get_num_interop_threads()}"
)

def _run_transcription(audio_bytes: bytes) -> str:
    started = time.monotonic()
    audio_file = io.BytesIO(audio_bytes)
    # vad_filter strips non-speech segments (silence, background noise) before
    # decoding. Without it, Whisper tends to hallucinate plausible-sounding
    # phrases from silence/noise rather than returning nothing — harmless with
    # manual push-to-talk, but the hands-free loop can legitimately end a turn
    # on mostly-silence (a long pause, ambient noise crossing the volume
    # threshold), and a hallucinated transcript there gets sent as if she'd
    # actually said it.
    #
    # beam_size=1 (greedy decoding) instead of 3: cuts decoding work roughly
    # 3x for a small accuracy cost that mostly doesn't matter on short
    # conversational utterances. If transcription quality noticeably degrades
    # in testing, raise this back up — it's a direct latency/accuracy trade.
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
    logger.info(f"TTS Request: {len(clean_text)} chars")
    logger.info(f"Text: {clean_text!r}")

    # -------------------------------------------------------
    # Stage 1 - Create generator
    # -------------------------------------------------------
    t = time.monotonic()

    generator = tts_pipeline(
        clean_text,
        voice="af_heart",
        speed=speed,
        split_pattern=r"[.!?\n]"
    )

    logger.info(
        "[Stage 1] Generator created in %.3f sec",
        time.monotonic() - t
    )

    # -------------------------------------------------------
    # Stage 2 - Generate chunks
    # -------------------------------------------------------
    audio_chunks = []

    stage2_start = time.monotonic()
    last_chunk = stage2_start

    for idx, (_gs, _ps, audio) in enumerate(generator, start=1):

        now = time.monotonic()

        logger.info(
            "[Stage 2] Chunk %d generated in %.3f sec",
            idx,
            now - last_chunk
        )

        last_chunk = now

        if audio is not None and len(audio) > 0:
            audio_chunks.append(audio)

    logger.info(
        "[Stage 2] Total chunk generation: %.3f sec",
        time.monotonic() - stage2_start
    )

    if not audio_chunks:
        raise ValueError("No audio generated")

    # -------------------------------------------------------
    # Stage 3 - Concatenate
    # -------------------------------------------------------
    t = time.monotonic()

    combined_audio = np.concatenate(audio_chunks)

    logger.info(
        "[Stage 3] np.concatenate(): %.3f sec",
        time.monotonic() - t
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
        time.monotonic() - t
    )

    # -------------------------------------------------------
    # Total
    # -------------------------------------------------------
    logger.info(
        "[TOTAL] %.3f sec",
        time.monotonic() - overall_start
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