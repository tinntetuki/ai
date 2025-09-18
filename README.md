# 🎬 AI Content Creator

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AI-powered video and image content creation platform with advanced video processing, speech recognition, text-to-speech, and intelligent product detection capabilities.

## 🌟 Features

### 🎥 Video Processing
- **Video Super-Resolution**: Upscale videos using Real-ESRGAN (2x, 4x, 8x)
- **Smart Segment Detection**: Automatically identify important video segments
- **Intelligent Video Editing**: AI-powered editing based on speech and visual analysis
- **Video Annotation**: Automatic product detection and annotation

### 🎤 Speech Processing
- **Speech-to-Text**: Multi-language transcription using OpenAI Whisper
- **Multiple Output Formats**: JSON, SRT, VTT, plain text
- **Keyword Detection**: Identify product mentions and key phrases
- **Speaker Analysis**: Audio quality assessment and speech timing

### 🔊 Text-to-Speech
- **Multi-Engine Support**: Edge TTS, Google TTS, Azure Speech, System TTS
- **Voice Profiles**: Customizable voices with style, speed, and pitch control
- **Voiceover Creation**: Replace video audio with generated speech
- **Batch Processing**: Process multiple text segments with timing preservation

### 🛍️ AI Product Detection
- **Object Detection**: YOLO-based product identification
- **Semantic Understanding**: CLIP-powered product description
- **Product Tracking**: Cross-frame tracking with importance scoring
- **Speech-Visual Correlation**: Analyze relationship between speech and visual content

### 📝 Subtitle Generation
- **Automatic Subtitles**: Generate styled subtitles from transcription
- **Custom Styling**: Font, color, position, and animation options
- **Multi-language Support**: Support for various languages and character sets
- **Keyword Highlighting**: Emphasize important product terms

## 🚀 Quick Start

### 🐳 Docker Deployment (Recommended)

**One-command deployment:**
```bash
git clone https://github.com/your-username/ai-content-creator.git
cd ai-content-creator
./scripts/deploy.sh deploy
```

**Available at:**
- **Web Interface**: http://localhost
- **API Documentation**: http://localhost/docs
- **Monitoring Dashboard**: http://localhost:3000 (admin/admin123)

### 📦 Manual Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/ai-content-creator.git
cd ai-content-creator
```

2. **Install dependencies:**
```bash
# Quick setup with Make
make setup-dev

# Or manual installation
pip install -r requirements.txt
pip install -e .
```

3. **Set up environment:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 🎮 Development Setup

```bash
# Docker development environment
make setup-docker

# Or local development
make setup-dev
make run-api
```

### Quick Examples

#### 🎥 Video Upscaling
```bash
# Upscale entire video
ai-content upscale-video input.mp4 output.mp4 --scale 4

# Smart segment processing
ai-content upscale-video input.mp4 output.mp4 --smart-segments --segments 3
```

#### 🎤 Speech Recognition
```bash
# Transcribe video to JSON
ai-content transcribe video.mp4 transcript.json --model base

# Generate SRT subtitles
ai-content transcribe video.mp4 subtitles.srt --format srt --language zh
```

#### 🔊 Text-to-Speech
```bash
# Basic text synthesis
ai-content synthesize-speech "Hello world" output.wav --language zh-CN

# Create voiceover from video
ai-content create-voiceover video.mp4 new_video.mp4 --voice zh-CN-XiaoxiaoNeural
```

#### 🛍️ Product Detection
```bash
# Detect and analyze products
ai-content detect-products video.mp4 analysis.json --annotate --speech-analysis

# Focus on specific keywords
ai-content detect-products video.mp4 results.json --keywords "phone,laptop,camera"
```

## 🖥️ Web Interface

### Access Options

**Docker (Recommended):**
```bash
make deploy  # Production deployment
# or
make start   # Development with hot reload
```

**Manual:**
```bash
make run-api
# Visit http://localhost:8000
```

**Features:**
- Drag & drop file uploads
- Real-time processing status
- Parameter configuration
- Result downloads and preview

### API Endpoints

- **Video Processing**: `/api/v1/upscale/`
- **Speech Recognition**: `/api/v1/speech/`
- **Text-to-Speech**: `/api/v1/tts/`
- **Product Detection**: `/api/v1/products/`
- **API Documentation**: `/docs` (Swagger UI)

## 🛠️ Development

### Development Commands

```bash
# Start API server with auto-reload
make run-api

# Run background worker
make run-worker

# Format code
make format

# Lint code
make lint

# Run tests
make test

# Run tests with coverage
make test-cov

# Clean build artifacts
make clean
```

### Project Structure

```
├── src/
│   ├── ai_engine/          # Core AI processing modules
│   │   ├── video_upscaler.py
│   │   ├── speech_processor.py
│   │   ├── tts_processor.py
│   │   ├── product_detector.py
│   │   ├── video_editor.py
│   │   ├── subtitle_generator.py
│   │   └── batch_processor.py
│   ├── api/                # FastAPI REST API
│   │   ├── main.py
│   │   ├── video_upscale.py
│   │   ├── speech_to_text.py
│   │   ├── text_to_speech.py
│   │   └── product_detection.py
│   ├── frontend/           # Web interface
│   ├── content_manager/    # Content handling
│   ├── utils/              # Shared utilities
│   └── cli.py              # Command line interface
├── tests/                  # Test files
├── data/                   # Data directory
├── docs/                   # Documentation
├── scripts/                # Utility scripts
└── examples/               # Example scripts
```

## 🎯 Supported Models

### Video Upscaling
- **RealESRGAN_x4plus**: General purpose (recommended)
- **RealESRGAN_x2plus**: Faster processing
- **RealESRGAN_x4plus_anime_6B**: Optimized for anime/cartoon

### Speech Recognition
- **tiny**: ~39MB, ~32x realtime (fastest)
- **base**: ~74MB, ~16x realtime (recommended)
- **small**: ~244MB, ~6x realtime
- **medium**: ~769MB, ~2x realtime
- **large**: ~1550MB, ~1x realtime (best quality)

### Text-to-Speech Voices

#### Chinese Voices
- **zh-CN-XiaoxiaoNeural**: Professional female
- **zh-CN-YunxiNeural**: Professional male
- **zh-CN-XiaoyiNeural**: Friendly female

#### English Voices
- **en-US-JennyNeural**: Professional female
- **en-US-GuyNeural**: Professional male
- **en-US-AriaNeural**: Conversational female

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# Processing Paths
DATA_PATH=./data
OUTPUT_PATH=./data/output
TEMP_PATH=./data/temp

# External APIs (optional)
AZURE_SPEECH_KEY=your_azure_key
AZURE_SPEECH_REGION=your_region
OPENAI_API_KEY=your_openai_key

# Processing Settings
MAX_WORKERS=4
MAX_FILE_SIZE=500MB
PROCESSING_TIMEOUT=3600
```

### GPU Support

For GPU acceleration:
```bash
# NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Mac M1/M2
# PyTorch with MPS support is automatically detected
```

## 🎮 Usage Examples

### Video Content Creation Pipeline

```python
from src.ai_engine import (
    create_upscaler,
    create_speech_processor,
    create_tts_processor,
    create_product_detector
)

# 1. Enhance video quality
upscaler = create_upscaler(scale=4)
upscaler.upscale_video("input.mp4", "enhanced.mp4")

# 2. Transcribe speech
speech_processor = create_speech_processor()
transcription = speech_processor.transcribe_video("enhanced.mp4")

# 3. Detect products
detector = create_product_detector()
products = detector.detect_products_in_video("enhanced.mp4")

# 4. Create new voiceover
tts_processor = create_tts_processor()
await tts_processor.create_voiceover_from_transcription(
    transcription, voice_profile, "final_video.mp4"
)
```

### Batch Processing

```python
from src.ai_engine.batch_processor import BatchProcessor, ProcessingTask

processor = BatchProcessor()

tasks = [
    ProcessingTask(
        task_id="task1",
        task_type="transcribe",
        input_path="video1.mp4",
        output_path="transcript1.json",
        parameters={"model": "base", "language": "zh"}
    ),
    ProcessingTask(
        task_id="task2",
        task_type="upscale",
        input_path="video1.mp4",
        output_path="enhanced1.mp4",
        parameters={"scale": 4}
    )
]

results = await processor.process_batch(tasks)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e .[dev]

# Run pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## 📋 Requirements

### System Requirements
- **Python**: 3.9 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 10GB free space for models and processing
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Dependencies
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision operations
- **FFmpeg**: Video processing (auto-installed with moviepy)
- **Real-ESRGAN**: Video super-resolution
- **Whisper**: Speech recognition
- **Ultralytics**: Object detection

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for video super-resolution
- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Ultralytics](https://github.com/ultralytics/ultralytics) for object detection
- [OpenAI CLIP](https://github.com/openai/CLIP) for vision-language understanding
- [Microsoft Edge TTS](https://github.com/rany2/edge-tts) for text-to-speech

## 📞 Support

- 📧 Email: support@ai-content-creator.com
- 💬 Discord: [Join our community](https://discord.gg/ai-content-creator)
- 📖 Documentation: [Full Documentation](https://docs.ai-content-creator.com)
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/ai-content-creator/issues)

---

⭐ **Star this repository if you find it helpful!**