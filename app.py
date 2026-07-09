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
    """Generates hyper-realistic neural voice audio stream matching the text input:"""
    try:
        # Generate audio frame ('af_heart' is a warm, highly empathetic female voice)
        generator = tts_pipeline(text, voice='af_heart', speed=0.9, split_pattern=r'\n')

        # Grab the first complete sentence chunk array
        for gs, ps, audio in generator:
            # Convert raw float32 array into an optimized WAV container stream
            wav_io = io.BytesIO()
            sf.write(wav_io, audio, 24000, format="WAV", subtypes="PCM_16")
            wav_io.seek(0)

            return StreamingResponse(wav_io, media_type="audio/wav")
        
    except Exception as e:
        return {"error: ": str(e)}, 500
