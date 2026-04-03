FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Set huggingface cache to be inside the container working directory
ENV HF_HOME="/app/model_cache"

# Set work directory
WORKDIR /app

# Install system dependencies (ffmpeg is strictly required for audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download all heavy ML models into the Docker image
# This ensures instantaneous startup when the container boots!
COPY preload.py .
RUN python preload.py

# Copy the rest of the application
COPY . .

# Expose port 7860 (Standard HuggingFace Spaces port, easily overriden)
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
