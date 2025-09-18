# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an AI-powered video and image content creation platform built with Python. The project specializes in video upscaling, speech-to-text, text-to-speech, and product detection using various AI models and computer vision techniques.

## Development Commands

### Quick Setup
```bash
# Complete development setup
make setup-dev

# Complete Docker setup
make setup-docker

# Production deployment
make deploy
```

### Core Development
```bash
# Install dependencies and development mode
make install && make dev

# Run API server with hot reload on port 8000
make run-api

# Run background worker for async processing
make run-worker

# Format code using black and isort (line length 88)
make format

# Lint with flake8 and mypy (strict type checking enabled)
make lint

# Run all tests with coverage
make test-cov

# Run specific test file
pytest tests/unit/test_specific.py -v

# Clean build artifacts
make clean
```

### Docker Operations
```bash
# Full production deployment with monitoring
./scripts/deploy.sh deploy

# Development environment with hot reload
./scripts/deploy.sh dev

# Service management
./scripts/deploy.sh start|stop|restart|status|health

# View logs and monitor
./scripts/deploy.sh logs api
./scripts/deploy.sh monitor

# Backup and restore
./scripts/deploy.sh backup
./scripts/deploy.sh restore backup_name

# GPU setup for Docker
./scripts/setup-gpu.sh setup
```

### CLI Tool Usage
The `ai-content` command provides access to all AI processing functions:
```bash
# Video upscaling with Real-ESRGAN
ai-content upscale-video input.mp4 output.mp4 --scale 4 --model RealESRGAN_x4plus

# Speech recognition with Whisper
ai-content transcribe video.mp4 transcript.json --model base --language auto

# Text-to-speech with multiple engines
ai-content synthesize-speech "Text content" output.wav --provider edge --voice zh-CN-XiaoxiaoNeural

# Product detection with YOLO + CLIP
ai-content detect-products video.mp4 analysis.json --annotate --speech-analysis

# Complete workflow example
python examples/complete_workflow.py input_video.mp4 --language zh --voice-style professional
```

## Architecture Overview

### Multi-Modal AI Pipeline
The platform implements a complete AI content creation pipeline with four main processing engines that can work independently or in combination:

1. **Video Enhancement Engine** (`src/ai_engine/video_upscaler.py`)
   - Real-ESRGAN super-resolution with multiple model variants
   - Smart segment detection for selective processing
   - Optimized for Mac M1 Pro and GPU acceleration

2. **Speech Processing Engine** (`src/ai_engine/speech_processor.py`)
   - OpenAI Whisper for multi-language transcription
   - Multiple output formats (JSON, SRT, VTT, plain text)
   - Keyword detection and speech analysis

3. **TTS Synthesis Engine** (`src/ai_engine/tts_processor.py`)
   - Multi-provider support (Edge TTS, Google TTS, Azure, system TTS)
   - Voice profiles with style, speed, pitch, volume control
   - Voiceover replacement with timing preservation

4. **Product Detection Engine** (`src/ai_engine/product_detector.py`)
   - YOLO object detection + CLIP semantic understanding
   - Product tracking across frames with importance scoring
   - Speech-visual correlation analysis

### API Architecture
- **FastAPI application** (`src/api/main.py`) with modular router structure
- **Asynchronous processing** via background workers and task queues
- **Multi-format responses** supporting both synchronous and async operations
- **Frontend integration** serving React-based web interface at root path

### Processing Workflow Architecture
```
Input → AI Engine → Processing Queue → Results Storage → API Response
  ↓         ↓            ↓                ↓               ↓
Video   Upscaler    Redis/Celery      PostgreSQL    JSON/Binary
Audio   Whisper     Background        File System   Download Links
Text    TTS Engines Workers           Metadata DB   Status Updates
```

### Factory Pattern Implementation
All AI engines use factory functions for consistent initialization:
- `create_upscaler()` - Video enhancement
- `create_speech_processor()` - Speech recognition
- `create_tts_processor()` - Text-to-speech
- `create_product_detector()` - Object detection
- `create_subtitle_generator()` - Subtitle generation

### Configuration Management
- **Environment-based config** with `.env` files (copy from `.env.docker` template)
- **Multi-environment support**: development, Docker, production
- **Model configuration**: AI model selection and parameters
- **Processing limits**: file sizes, timeout, concurrency settings
- **External API integration**: Azure Speech, OpenAI keys (optional)

### Containerized Deployment
- **Multi-stage Dockerfile** with production/development/worker targets
- **Complete service stack**: API, workers, Redis, PostgreSQL, Nginx, monitoring
- **GPU acceleration** with NVIDIA Docker support
- **Health checks** and automatic recovery
- **Horizontal scaling** support for workers and API instances

## Important Implementation Details

### AI Model Management
- Models are downloaded automatically on first use
- GPU memory management with configurable tile sizes for large videos
- Batch processing support for multiple files with dependency tracking

### Data Flow Patterns
- **Async processing**: Long-running AI tasks use background workers
- **Progressive results**: Real-time status updates and progress tracking
- **Multi-format output**: Each processor supports multiple output formats
- **Temporary file management**: Automatic cleanup with configurable retention

### Error Handling Strategy
- **Graceful degradation**: Fallback to CPU if GPU unavailable
- **Retry mechanisms**: Automatic retry for transient failures
- **Comprehensive logging**: Structured logging with loguru
- **User feedback**: Clear error messages and processing status

### Testing and Quality
- **Code formatting**: Black (line length 88) + isort with "black" profile
- **Type checking**: mypy with strict settings (`disallow_untyped_defs = true`)
- **Testing structure**: Separate unit and integration test directories
- **Coverage reporting**: HTML and terminal coverage reports

### Performance Considerations
- **Memory efficiency**: Streaming processing for large files
- **GPU optimization**: CUDA memory allocation tuning
- **Caching strategy**: Redis for frequently accessed data
- **File size limits**: Configurable per content type (video: 500MB, audio: 100MB)