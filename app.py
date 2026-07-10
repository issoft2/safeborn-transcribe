import asyncio
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor

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

# stt_model.transcribe(...) and tts_pipeline(...) are blocking, CPU-bound calls.
# Running them directly inside `async def` routes stalls the event loop for
# every other request (including /health) until they finish. Offload them here.
_executor = ThreadPoolExecutor(max_workers=2)


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
    started = time.monotonic()
    generator = tts_pipeline(clean_text, voice='af_heart', speed=speed, split_pattern=r'[.!?\n]')

    audio_chunks = []
    for _gs, _ps, audio in generator:
        if audio is not None and len(audio) > 0:
            audio_chunks.append(audio)

    if not audio_chunks:
        raise ValueError("No audio could be generated")

    combined_audio = np.concatenate(audio_chunks)

    wav_io = io.BytesIO()
    sf.write(wav_io, combined_audio, 24000, format='WAV', subtype='PCM_16')
    logger.info(f"[timing] tts for {len(clean_text)} chars took {time.monotonic() - started:.2f}s")
    return wav_io.getvalue()


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    loop = asyncio.get_event_loop()
    try:
        transcription = await loop.run_in_executor(_executor, _run_transcription, audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"text": transcription}


@app.get("/tts-stream")
async def text_to_speech(text: str = Query(...), speed: float = Query(0.90)):
    # Clean up structural text breaks cleanly
    clean_text = " ".join(text.splitlines())

    loop = asyncio.get_event_loop()
    try:
        wav_bytes = await loop.run_in_executor(_executor, _run_tts, clean_text, speed)
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