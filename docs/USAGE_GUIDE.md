# Usage Guide - AI Content Creator

This guide provides practical examples and best practices for using the AI Content Creator platform.

## Quick Start Tutorial

### 1. Installation and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/ai-content-creator.git
cd ai-content-creator

# Install dependencies
make install

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Start the API server
make run-api
```

### 2. Basic Video Processing Workflow

#### Step 1: Video Upscaling
```bash
# Upscale a video to 4x resolution
ai-content upscale-video input.mp4 enhanced.mp4 --scale 4

# Use smart segments for large videos
ai-content upscale-video input.mp4 enhanced.mp4 --smart-segments --segments 3
```

#### Step 2: Extract and Enhance Audio
```bash
# Transcribe the enhanced video
ai-content transcribe enhanced.mp4 transcript.json --model base --format json

# Generate SRT subtitles
ai-content transcribe enhanced.mp4 subtitles.srt --format srt --language zh
```

#### Step 3: Create Professional Voiceover
```bash
# Replace audio with professional voiceover
ai-content create-voiceover enhanced.mp4 final.mp4 \
  --voice zh-CN-XiaoxiaoNeural \
  --style professional \
  --preserve-timing
```

## Use Cases and Examples

### 1. E-commerce Product Videos

#### Automatic Product Detection
```bash
# Detect products with annotation
ai-content detect-products product_demo.mp4 analysis.json \
  --annotate \
  --keywords "phone,case,charger,headphones" \
  --speech-analysis

# Generate detailed report
ai-content annotate-video product_demo.mp4 annotated.mp4 \
  --keywords "phone,case,charger" \
  --bbox-color "0,255,0" \
  --font-scale 0.8
```

#### Complete Product Video Enhancement Pipeline
```python
from src.ai_engine import (
    create_upscaler,
    create_speech_processor,
    create_tts_processor,
    create_product_detector
)

async def enhance_product_video(input_path, output_path):
    # Step 1: Enhance video quality
    upscaler = create_upscaler(scale=4)
    enhanced_path = "temp_enhanced.mp4"
    upscaler.upscale_video(input_path, enhanced_path)

    # Step 2: Analyze products
    detector = create_product_detector()
    products = detector.detect_products_in_video(
        enhanced_path,
        keywords=["phone", "laptop", "camera", "watch"]
    )

    # Step 3: Create professional transcription
    speech_processor = create_speech_processor()
    transcription = speech_processor.transcribe_video(enhanced_path)

    # Step 4: Generate professional voiceover
    tts_processor = create_tts_processor()
    voice_profile = tts_processor.create_product_voice_profile(
        language="zh",
        voice_type="professional",
        gender="female"
    )

    await tts_processor.create_voiceover_from_transcription(
        transcription=transcription,
        voice_profile=voice_profile,
        output_path="temp_audio.wav"
    )

    # Step 5: Combine everything
    await tts_processor.replace_video_audio(
        video_path=enhanced_path,
        new_audio_path="temp_audio.wav",
        output_path=output_path
    )

    # Step 6: Create annotated version
    detector.annotate_video_with_products(
        video_path=output_path,
        detections=products,
        output_path=output_path.replace('.mp4', '_annotated.mp4')
    )

    return {
        "enhanced_video": output_path,
        "annotated_video": output_path.replace('.mp4', '_annotated.mp4'),
        "products_detected": len(products),
        "transcription": transcription
    }
```

### 2. Educational Content Creation

#### Multi-language Tutorial Videos
```bash
# Create Chinese tutorial
ai-content transcribe tutorial_en.mp4 transcript_en.json --language en

# Translate and create Chinese voiceover
# (Manual translation step)
ai-content synthesize-speech "中文教程内容" tutorial_zh_audio.wav \
  --language zh-CN \
  --voice zh-CN-XiaoxiaoNeural \
  --style friendly

# Replace audio in video
ai-content create-voiceover tutorial_en.mp4 tutorial_zh.mp4 \
  --provider edge \
  --voice zh-CN-XiaoxiaoNeural \
  --language zh-CN
```

#### Automatic Subtitle Generation
```python
from src.ai_engine.subtitle_generator import create_subtitle_generator
from src.ai_engine.speech_processor import create_speech_processor

def create_educational_subtitles(video_path, output_path):
    # Transcribe video
    speech_processor = create_speech_processor()
    transcription = speech_processor.transcribe_video(video_path)

    # Create styled subtitles
    subtitle_generator = create_subtitle_generator()

    # Educational style with keyword highlighting
    style = subtitle_generator.create_educational_style(
        highlight_keywords=True,
        keywords=["important", "key point", "remember", "note"]
    )

    # Generate subtitle video
    subtitle_generator.create_subtitle_video(
        video_path=video_path,
        transcription=transcription,
        output_path=output_path,
        style=style
    )

    return transcription
```

### 3. Social Media Content

#### Short-form Video Creation
```bash
# Extract highlights from long video
ai-content detect-products long_video.mp4 analysis.json --smart-segments

# Create 30-second highlights
ai-content upscale-video long_video.mp4 highlights.mp4 \
  --duration 30 \
  --smart-segments \
  --segments 1
```

#### TikTok/Instagram Optimization
```python
async def create_social_media_content(video_path):
    from src.ai_engine.video_editor import create_video_editor
    from src.ai_engine.speech_processor import create_speech_processor

    # Analyze video content
    speech_processor = create_speech_processor()
    transcription = speech_processor.transcribe_video(video_path)

    # Find engaging segments
    video_editor = create_video_editor()
    highlights = video_editor.detect_engaging_segments(
        video_path=video_path,
        transcription=transcription,
        min_duration=15,
        max_duration=60
    )

    # Create vertical format clips
    clips = []
    for i, segment in enumerate(highlights[:3]):  # Top 3 segments
        clip_path = f"clip_{i}.mp4"

        # Extract and format segment
        video_editor.create_vertical_clip(
            video_path=video_path,
            start_time=segment.start,
            end_time=segment.end,
            output_path=clip_path,
            aspect_ratio="9:16"  # Instagram/TikTok format
        )

        clips.append(clip_path)

    return clips
```

### 4. Podcast and Audio Content

#### Podcast Enhancement
```bash
# Clean up audio quality and transcribe
ai-content transcribe podcast.mp3 transcript.json \
  --model medium \
  --format json \
  --keywords "guest,interview,topic,question"

# Generate clean voiceover
ai-content synthesize-speech "Welcome to our podcast..." intro.wav \
  --provider edge \
  --voice en-US-JennyNeural \
  --style professional
```

#### Automatic Show Notes Generation
```python
def generate_show_notes(audio_path):
    from src.ai_engine.speech_processor import create_speech_processor

    # Transcribe podcast
    processor = create_speech_processor(model_size="medium")
    transcription = processor.transcribe_video(audio_path)

    # Detect key topics
    key_segments = processor.get_speech_segments(
        transcription,
        min_duration=10.0,
        confidence_threshold=-0.5
    )

    # Generate timestamps for show notes
    show_notes = []
    for segment in key_segments:
        if any(keyword in segment.text.lower() for keyword in
               ["topic", "question", "important", "key", "summary"]):
            show_notes.append({
                "timestamp": f"{int(segment.start//60):02d}:{int(segment.start%60):02d}",
                "content": segment.text[:100] + "..."
            })

    return show_notes
```

## Advanced Workflows

### 1. Batch Processing

#### Process Multiple Videos
```python
from src.ai_engine.batch_processor import BatchProcessor, ProcessingTask

async def batch_process_videos(video_files):
    processor = BatchProcessor()

    tasks = []
    for i, video_file in enumerate(video_files):
        # Upscaling task
        upscale_task = ProcessingTask(
            task_id=f"upscale_{i}",
            task_type="upscale",
            input_path=video_file,
            output_path=f"enhanced_{i}.mp4",
            parameters={"scale": 4, "model": "RealESRGAN_x4plus"}
        )

        # Transcription task (depends on upscaling)
        transcribe_task = ProcessingTask(
            task_id=f"transcribe_{i}",
            task_type="transcribe",
            input_path=f"enhanced_{i}.mp4",
            output_path=f"transcript_{i}.json",
            parameters={"model": "base", "language": "auto"},
            dependencies=[f"upscale_{i}"]
        )

        tasks.extend([upscale_task, transcribe_task])

    # Process all tasks
    results = await processor.process_batch(tasks)
    return results
```

### 2. Custom Voice Profiles

#### Create Brand-specific Voice
```python
def create_brand_voice_profile(brand_style="professional"):
    from src.ai_engine.tts_processor import VoiceProfile

    brand_configs = {
        "professional": {
            "provider": "edge",
            "voice_name": "zh-CN-XiaoxiaoNeural",
            "language": "zh-CN",
            "style": "professional",
            "speed": 0.95,
            "pitch": 1.0,
            "volume": 1.0
        },
        "friendly": {
            "provider": "edge",
            "voice_name": "zh-CN-XiaoyiNeural",
            "language": "zh-CN",
            "style": "friendly",
            "speed": 1.05,
            "pitch": 1.1,
            "volume": 1.1
        },
        "luxury": {
            "provider": "edge",
            "voice_name": "zh-CN-YunxiNeural",
            "language": "zh-CN",
            "style": "calm",
            "speed": 0.9,
            "pitch": 0.95,
            "volume": 0.9
        }
    }

    config = brand_configs.get(brand_style, brand_configs["professional"])
    return VoiceProfile(**config)
```

### 3. Integration with External Services

#### Webhook Integration
```python
import asyncio
import aiohttp

async def process_with_webhook(video_url, webhook_url):
    """Process video and send results to webhook"""

    # Download video
    async with aiohttp.ClientSession() as session:
        async with session.get(video_url) as response:
            video_content = await response.read()

    # Save temporarily
    temp_path = "temp_video.mp4"
    with open(temp_path, 'wb') as f:
        f.write(video_content)

    try:
        # Process video
        result = await enhance_product_video(temp_path, "processed.mp4")

        # Send webhook notification
        async with aiohttp.ClientSession() as session:
            await session.post(webhook_url, json={
                "status": "completed",
                "result": result
            })

    except Exception as e:
        # Send error webhook
        async with aiohttp.ClientSession() as session:
            await session.post(webhook_url, json={
                "status": "failed",
                "error": str(e)
            })
```

## Performance Optimization

### 1. GPU Configuration

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Optimize for specific GPU
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 2. Memory Management

```python
# Optimize processing for large files
def process_large_video(video_path, chunk_duration=300):  # 5-minute chunks
    from src.ai_engine.video_upscaler import create_upscaler

    upscaler = create_upscaler(tile_size=256)  # Smaller tiles for less memory

    # Process in segments
    success = upscaler.upscale_video(
        input_path=video_path,
        output_path="output.mp4",
        segment_duration=chunk_duration
    )

    return success
```

### 3. Caching Strategy

```python
import functools
import hashlib

@functools.lru_cache(maxsize=100)
def get_cached_transcription(file_hash, model_size):
    """Cache transcription results"""
    cache_path = f"cache/transcription_{file_hash}_{model_size}.json"

    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    return None

def cache_transcription_result(file_hash, model_size, result):
    """Save transcription to cache"""
    os.makedirs("cache", exist_ok=True)
    cache_path = f"cache/transcription_{file_hash}_{model_size}.json"

    with open(cache_path, 'w') as f:
        json.dump(result, f)
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors
```bash
# Reduce processing parameters
ai-content upscale-video input.mp4 output.mp4 --scale 2  # Lower scale
ai-content transcribe input.mp4 output.json --model tiny  # Smaller model
```

#### 2. Slow Processing
```python
# Use GPU acceleration
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

# Optimize batch size
from src.ai_engine.batch_processor import BatchConfig

config = BatchConfig(
    max_workers=2,  # Reduce workers
    max_concurrent_tasks=1  # Process one at a time
)
```

#### 3. Audio Quality Issues
```bash
# Use higher quality TTS model
ai-content synthesize-speech "text" output.wav \
  --provider edge \
  --voice zh-CN-XiaoxiaoNeural \
  --speed 0.95 \
  --volume 1.1
```

### Debug Mode

```bash
# Enable debug logging
export DEBUG=True
export LOG_LEVEL=DEBUG

# Run with verbose output
ai-content upscale-video input.mp4 output.mp4 --scale 4 --verbose
```

## Best Practices

### 1. File Organization
```
project/
├── input/          # Original files
├── processing/     # Temporary files
├── output/         # Final results
├── cache/          # Cached results
└── logs/           # Processing logs
```

### 2. Quality Settings by Use Case

| Use Case | Video Scale | Audio Model | TTS Quality |
|----------|-------------|-------------|-------------|
| Social Media | 2x | base | edge |
| Professional | 4x | medium | edge + style |
| Archive | 8x | large | azure premium |
| Preview | 2x | tiny | gtts |

### 3. Monitoring and Logging

```python
import logging
from loguru import logger

# Configure structured logging
logger.add(
    "logs/processing_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time} | {level} | {module} | {message}"
)

# Monitor processing metrics
def log_processing_metrics(task_type, duration, file_size, success):
    logger.info(
        f"Processing completed",
        extra={
            "task_type": task_type,
            "duration": duration,
            "file_size": file_size,
            "success": success
        }
    )
```

## Next Steps

1. **Explore Advanced Features**: Try batch processing and custom voice profiles
2. **Integration**: Connect with your existing workflow using the API
3. **Optimization**: Fine-tune settings for your specific use case
4. **Automation**: Set up automated processing pipelines
5. **Monitoring**: Implement logging and monitoring for production use

For more examples and detailed API documentation, visit:
- [API Documentation](API.md)
- [GitHub Repository](https://github.com/your-username/ai-content-creator)
- [Community Examples](https://github.com/your-username/ai-content-creator/tree/main/examples)