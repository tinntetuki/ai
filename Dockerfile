# Multi-stage Dockerfile for AI Content Creator
# Optimized for production deployment with minimal image size

# Stage 1: Base image with system dependencies
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgthread-2.0-0 \
    libgomp1 \
    wget \
    curl \
    git \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Stage 2: Python dependencies
FROM base as dependencies

# Upgrade pip and install basic tools
RUN python3.10 -m pip install --user --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY --chown=app:app requirements.txt .

# Install Python dependencies
RUN python3.10 -m pip install --user -r requirements.txt

# Install PyTorch with CUDA support
RUN python3.10 -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p data/input data/output data/temp logs cache

# Install application in development mode
RUN python3.10 -m pip install --user -e .

# Download initial models (optional, comment out for faster builds)
# RUN python3.10 -c "
# import whisper
# import torch
# from ultralytics import YOLO
#
# # Download Whisper models
# whisper.load_model('tiny')
# whisper.load_model('base')
#
# # Download YOLO models
# YOLO('yolov8n.pt')
# YOLO('yolov8s.pt')
#
# print('Models downloaded successfully')
# "

# Stage 4: Production image
FROM base as production

# Copy Python packages from dependencies stage
COPY --from=dependencies /home/app/.local /home/app/.local

# Copy application from application stage
COPY --from=application /home/app /home/app

# Set PATH to include user's local bin
ENV PATH=/home/app/.local/bin:$PATH

# Create volume mount points
VOLUME ["/home/app/data", "/home/app/logs"]

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000

# Default command
CMD ["python3.10", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 5: Development image
FROM application as development

# Install development dependencies
RUN python3.10 -m pip install --user \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    isort \
    flake8 \
    mypy \
    pre-commit \
    ipython \
    jupyter

# Set development environment
ENV DEBUG=True
ENV LOG_LEVEL=DEBUG

# Development command with auto-reload
CMD ["python3.10", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 6: Worker image (for background processing)
FROM production as worker

# Install additional worker dependencies
RUN python3.10 -m pip install --user \
    celery \
    redis

# Worker command
CMD ["python3.10", "-m", "celery", "worker", "-A", "src.worker.celery_app", "--loglevel=info"]