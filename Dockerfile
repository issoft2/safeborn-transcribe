# Dockerfile
FROM python:3.10-slim

# Install system dependencies needed for audio compilation
RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak-ng \
    libsndfile1 \
    && rm -rf /lib/lists/*

WORKDIR /app

# Install optimized packages directly
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    faster-whisper \
    kokoro-onnx \
    soundfile \
    python-multipart

COPY . .

EXPOSE 8081

# Spin up Uvicorn listening on the internal Railway execution port
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]