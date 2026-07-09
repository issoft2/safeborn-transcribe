import os
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel
from kokoro import KPipeline
import io
import soundfile as sf

app = FastAPI(title="SafeBorn Voice Engine")

# Initialize Faster-Whisper (using 'base' optimized for CPU running on low memeoy)
# 'cpu' compute_type="int8" drops RAM usage drastically for Railway containers
stt_model = WhisperModel("base", device="cpu", compute_type="int8")

# Initialize Kokoro Pipeline ('a' for American English, automatically downloads the 82MB weights)
tts_pipeline = KPipeline(lang_code='a')

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Accept binary audi from mobile app and returns fast text transcription."""
    try:
        # Read file tream bytes directly
        audio_bytes = await audio.read()
        audio_file = io.BytesIO(audio_bytes)

        # Transcribe audi using the CPU-optimized engine
        segments, info = stt_model.transcribe(audio_file, beam_size=3)
        transcription = " ".join([segment.text for segment in segments])

        return {"text": transcription.strip()}
    except Exception as e:
        return {"error:": str(e)}, 500  
        

@app.get("/tts-stream")
async def text_to_speech(text: str = Query(...)):
    """Streams neural voice chunks sentence-by-sentence to prevent 504 gateway timeouts."""
    try:
        # 1. Use Kokoro's generator pipeline to break text down cleanly
        generator = tts_pipeline(text, voice='af_heart', speed=0.9, split_pattern=r'\n')

        def audio_stream_generator():
            for gs, ps, audio in generator:
                # Convert this individual sentence slice directly to memory bytes
                wav_io = io.BytesIO()
                sf.write(wav_io, audio, 24000, format='WAV', subtype='PCM_16')
                yield wav_io.getvalue()

        # 2. Return an active streaming pipe back through your proxy network
        headers = {
            "Content-Disposition": "inline; filename=\"speech.wav\"",
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }

        return StreamingResponse(
            audio_stream_generator(), 
            media_type="audio/wav", 
            headers=headers
        )
            
    except Exception as e:
        return {"error": str(e)}, 500
