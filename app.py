import io
import os
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Query, UploadFile, File  # <-- Added UploadFile, File imports
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from kokoro import KPipeline
from faster_whisper import WhisperModel  # <-- Added missing Faster-Whisper import

app = FastAPI(title="SafeBorn Voice Engine")

# Enable CORS so your mobile app can communicate with it smoothly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CRITICAL: LOAD MODELS GLOBALLY AT STARTUP ───────────────────────
print("Loading Whisper Speech-to-Text model...")
stt_model = WhisperModel("base", device="cpu", compute_type="int8")

print("Loading Kokoro neural voice pipeline into memory...")
tts_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
print("Voice engine services are warm and ready!")


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Accepts binary audio from mobile app and returns fast text transcription."""
    try:
        # Read file stream bytes directly
        audio_bytes = await audio.read()
        audio_file = io.BytesIO(audio_bytes)

        # Transcribe audio using the CPU-optimized engine
        segments, info = stt_model.transcribe(audio_file, beam_size=3)
        transcription = " ".join([segment.text for segment in segments])

        return {"text": transcription.strip()}
    except Exception as e:
        return {"error": str(e)}, 500  # <-- Fixed typo in key string "error:"


@app.get("/tts-stream")
async def text_to_speech(text: str = Query(...)):
    """
    Processes full multi-line text into a single unified WAV container 
    on a warm global model, taking ~1-2 seconds total.
    """
    try:
        # Clean up any weird multiple spacing anomalies from the incoming text
        clean_text = " ".join(text.splitlines())
        
        # Generate the audio chunks using the warm global pipeline
        generator = tts_pipeline(clean_text, voice='af_heart', speed=0.95)
        
        audio_chunks = []
        for gs, ps, audio in generator:
            if audio is not None and len(audio) > 0:
                audio_chunks.append(audio)
        
        if not audio_chunks:
            return {"error": "No audio could be generated"}, 400

        # Stitch arrays seamlessly 
        combined_audio = np.concatenate(audio_chunks)
        
        # Build the final single-header WAV payload
        wav_io = io.BytesIO()
        sf.write(wav_io, combined_audio, 24000, format='WAV', subtype='PCM_16')
        wav_io.seek(0)
        
        return StreamingResponse(
            wav_io, 
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=\"speech.wav\"",
                "Cache-Control": "no-cache"
            }
        )
            
    except Exception as e:
        print(f"Generation failure: {str(e)}")
        return {"error": str(e)}, 500

@app.get("/health")
async def health_check():
    return {"status": "healthy"}