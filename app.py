import io
import os
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Query, UploadFile, File
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

print("Loading Whisper Speech-to-Text model...")
# Optimized Whisper initialization to prevent CPU cache thrashing
stt_model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=2)

print("Loading Kokoro neural voice pipeline into memory...")
tts_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
print("Voice engine services are warm and ready!")


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()
        audio_file = io.BytesIO(audio_bytes)
        segments, info = stt_model.transcribe(audio_file, beam_size=3)
        transcription = " ".join([segment.text for segment in segments])
        return {"text": transcription.strip()}
    except Exception as e:
        return {"error": str(e)}, 500  


@app.get("/tts-stream")
async def text_to_speech(text: str = Query(...), speed: float = Query(0.90)):
    try:
        # Clean up structural text breaks cleanly
        clean_text = " ".join(text.splitlines())
        
        # SPEED FIX: Added 'split_pattern' to keep the generator moving fast without heavy NLP logic
        generator = tts_pipeline(clean_text, voice='af_heart', speed=speed, split_pattern=r'[.!?\n]')
        
        audio_chunks = []
        for gs, ps, audio in generator:
            if audio is not None and len(audio) > 0:
                audio_chunks.append(audio)
        
        if not audio_chunks:
            return {"error": "No audio could be generated"}, 400

        combined_audio = np.concatenate(audio_chunks)
        
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