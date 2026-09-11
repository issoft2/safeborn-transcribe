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
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
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

# WHICH WHISPER. "small" rather than "base".
#
# base is the second-smallest model in the family, and its weakness is
# exactly our users: accents under-represented in its training data, on
# short utterances with no surrounding context to lean on. A Nigerian
# mother saying "start" to the Labour Companion got back "star", "stat",
# "sat" — on one recorded session, a dozen times over fifty seconds
# before one landed. small is roughly three times the parameters and is
# where accented English starts working properly.
#
# WHAT IT COSTS, both of which want watching on the first deploy:
#
#   Latency. Whisper pads every clip to 30 seconds before encoding, so
#   the encoder pass costs the same for a one-second command as for a
#   full sentence, and that pass is what triples. Transcription measured
#   around 3s on base; expect meaningfully more.
#
#   Memory. int8 small is roughly 250MB of weights against base's 90,
#   in a container that is also holding Kokoro. If this OOMs on deploy,
#   that is what happened.
#
# Both are one environment variable away from being put back, without a
# code change or a rebuild.
#
# WHY ".en". The English-only models generally beat their multilingual
# counterparts on English at identical size — they spend their whole
# capacity on one language instead of ninety-nine — and the labour
# command path already pins language="en", so nothing is given up there.
#
# WHAT IT GIVES UP, AND IT IS NOT NOTHING. This model can only produce
# English. Every other language becomes English-shaped noise, silently:
# there is no error, just a wrong transcript. Two consequences worth
# holding on to:
#
#   The AI coach path does not pin a language, so it was at least
#   attempting other languages before. In practice on "base" it was
#   attempting them very badly — Yoruba and Hausa are among the worst
#   served languages in Whisper, and Igbo is not in its language list at
#   all — so little is actually lost. But it is a change, not a no-op.
#
#   Nigerian Pidgin is not a Whisper language either way. An English
#   model is arguably the closer fit for it, but that is a guess and
#   wants a speaker to confirm rather than an assumption from here.
#
# STT_MODEL_SIZE=small puts the multilingual model back.
_STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "small.en")

# Deliberately NOT derived from _CPU_COUNT like OMP, MKL and torch are.
# The 2 here is load-bearing: Whisper and Kokoro share this container,
# and whoever wrote the line below measured cache thrashing when they
# competed. If small proves too slow, this is the first lever to try —
# but it is a trade against TTS, not a free win.
_STT_CPU_THREADS = int(os.getenv("STT_CPU_THREADS", "2"))

logger.info("Loading Whisper Speech-to-Text model (%s)...", _STT_MODEL_SIZE)
# Optimized Whisper initialization to prevent CPU cache thrashing
stt_model = WhisperModel(
    _STT_MODEL_SIZE,
    device="cpu",
    compute_type="int8",
    cpu_threads=_STT_CPU_THREADS,
    num_workers=1,
)

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
STT model: {_STT_MODEL_SIZE} (int8, cpu_threads={_STT_CPU_THREADS})
OMP: {os.getenv('OMP_NUM_THREADS')}
MKL: {os.getenv('MKL_NUM_THREADS')}
Torch threads: {torch.get_num_threads()}
Torch interop: {torch.get_num_interop_threads()}
"""
)

# The vocabulary the Labour Companion listens for. faster-whisper
# prepends these to the decoder prompt (see Tokenizer/get_prompt), so
# words that appear here are far likelier to come back out.
#
# HOTWORDS RATHER THAN initial_prompt, AND SHORT. The two compose —
# get_prompt prepends hotwords and then any previous tokens — so using
# both would only make the prompt longer. Length is the thing to avoid:
# Whisper's best-known failure on a near-silent clip is echoing its own
# prompt back as the transcript, and we have just turned the
# voice-activity filter off, so near-silent clips now reach the decoder.
# A bare word list is the smallest bias that does the job, and the app
# shows the mother what it heard, so an echo is visible rather than
# silent.
#
# WHY IT MATTERS SO MUCH HERE. A labour command is ONE WORD with no
# surrounding context, which is close to the worst input a Whisper-class
# model can be given: it is trained on continuous speech and leans hard
# on context it does not have. A Nigerian mother saying "start" gets back
# "star", "stat", "sat", "tart" — and on a recorded session she said it
# perhaps a dozen times over fifty seconds before one landed. The prompt
# is what puts those words back within reach.
_COMMAND_HOTWORDS = "start begin stop over done finished help hurts"


def _run_transcription(audio_bytes: bytes, mode: str = "speech") -> str:
    """
    `mode` picks the decoder settings, because the two callers of this
    service want opposite things.

      "speech"   the AI coach — whole sentences, conversational, and the
                 settings that have always been here.
      "command"  the Labour Companion — one word, often under a second,
                 spoken by a woman in pain who is not going to repeat
                 herself ten times.

    WHAT "command" CHANGES, AND WHY EACH ONE:

    language="en"        Whisper otherwise detects the language from the
                         audio. On a one-second clip of accented English
                         that detection is a coin toss, and picking the
                         wrong language does not degrade the transcript,
                         it destroys it.

                         Passed unconditionally, including on an
                         English-only model where it is redundant:
                         faster-whisper only warns when the language
                         differs from "en", and keeping it here means
                         STT_MODEL_SIZE can be switched back to a
                         multilingual build without this silently
                         reverting to language detection.

    hotwords             Biases decoding toward the words she is actually
                         going to say. See _COMMAND_HOTWORDS.

    vad_filter=False     THIS IS THE ONE MOST LIKELY TO HAVE BEEN EATING
                         HER COMMANDS. The voice-activity filter, with a
                         500ms silence window, is being handed a clip of
                         roughly a second and a half: a short word and the
                         600ms of quiet that told the client to stop
                         recording. It can trim that to nothing, and
                         faster-whisper then returns zero segments, which
                         this function joins into "". An empty string is
                         indistinguishable from silence downstream, so the
                         app simply listened again and said nothing.

                         The client already does its own voice-activity
                         detection to decide when to close the microphone.
                         Doing it twice, with the second one unaware of
                         how short the clip is, is how the audio went
                         missing.

    condition_on_previous_text=False
                         Each command is its own utterance with no history.
                         Left on, Whisper can carry text between calls and
                         loop on it.

    beam_size=5          Greedy decoding is a reasonable trade over a long
                         sentence where context can rescue a bad step. Over
                         a single short word there is no context to rescue
                         anything, and the clip is small enough that the
                         extra beams cost little.
    """
    started = time.monotonic()
    audio_file = io.BytesIO(audio_bytes)

    if mode == "command":
        segments, _info = stt_model.transcribe(
            audio_file,
            beam_size=5,
            language="en",
            hotwords=_COMMAND_HOTWORDS,
            condition_on_previous_text=False,
            vad_filter=False,
        )
    else:
        segments, _info = stt_model.transcribe(
            audio_file,
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

    text = " ".join(segment.text for segment in segments).strip()
    logger.info(
        "[timing] transcription (%s) took %.2fs -> %r",
        mode,
        time.monotonic() - started,
        text,
    )
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
async def transcribe_audio(
    audio: UploadFile = File(...),
    mode: str = Form("speech"),
):
    """
    `mode` is optional and defaults to "speech", so every existing caller
    keeps exactly the behaviour it has now. The Labour Companion's proxy
    sends "command" — see _run_transcription for what that changes.
    """
    audio_bytes = await audio.read()
    requested = mode if mode in {"speech", "command"} else "speech"
    loop = asyncio.get_event_loop()
    try:
        transcription = await loop.run_in_executor(
            stt_executor, _run_transcription, audio_bytes, requested
        )
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
    Kokoro yields float32 waveforms in [-1, 1]. Convert to signed
    16-bit PCM little-endian, the format the client's Web Audio API
    decoder expects (matches _PCM_FORMAT_INT16LE header code).
    """
    clipped = np.clip(audio, -1.0, 1.0)
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