# AI Content Creator API Documentation

## Overview

The AI Content Creator API provides powerful endpoints for video processing, speech recognition, text-to-speech synthesis, and product detection. This RESTful API is built with FastAPI and provides both synchronous and asynchronous processing capabilities.

**Base URL**: `http://localhost:8000`
**API Version**: v1
**Documentation**: `/docs` (Swagger UI)

## Authentication

Currently, the API does not require authentication. For production deployments, implement appropriate security measures.

## Rate Limiting

Default rate limits apply to prevent abuse. Contact support for enterprise-level access.

## Response Format

All API responses follow a consistent JSON format:

```json
{
    "success": true,
    "data": {},
    "message": "Operation completed successfully",
    "task_id": "uuid-string",
    "timestamp": "2024-01-01T00:00:00Z"
}
```

For errors:
```json
{
    "success": false,
    "error": "Error description",
    "code": "ERROR_CODE",
    "timestamp": "2024-01-01T00:00:00Z"
}
```

## Endpoints

### Health Check

#### GET `/health`
Check API health status.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-01T00:00:00Z"
}
```

### Video Upscaling

#### POST `/api/v1/upscale/upload`
Upload a video for super-resolution enhancement.

**Parameters:**
- `file` (form-data): Video file (MP4, AVI, MOV, MKV)
- `scale` (form-data, optional): Upscaling factor (2, 4, 8) - default: 4
- `model` (form-data, optional): Model name - default: "RealESRGAN_x4plus"
- `smart_segments` (form-data, optional): Use smart segment detection - default: false
- `max_segments` (form-data, optional): Maximum segments for smart processing - default: 3

**Request Example:**
```bash
curl -X POST \
  http://localhost:8000/api/v1/upscale/upload \
  -F "file=@video.mp4" \
  -F "scale=4" \
  -F "model=RealESRGAN_x4plus"
```

**Response:**
```json
{
    "task_id": "upscale_abc123",
    "status": "processing",
    "message": "Video upscaling started"
}
```

#### GET `/api/v1/upscale/status/{task_id}`
Check upscaling task status.

**Response:**
```json
{
    "task_id": "upscale_abc123",
    "status": "completed",
    "progress": 100,
    "start_time": "2024-01-01T00:00:00Z",
    "end_time": "2024-01-01T00:05:00Z",
    "result": {
        "input_resolution": "1920x1080",
        "output_resolution": "7680x4320",
        "file_size": 15728640,
        "processing_time": 300
    }
}
```

#### GET `/api/v1/upscale/download/{task_id}`
Download the upscaled video.

**Response:** Binary video file

#### GET `/api/v1/upscale/models`
Get available upscaling models.

**Response:**
```json
{
    "models": [
        {
            "name": "RealESRGAN_x4plus",
            "scale": 4,
            "description": "General purpose model"
        },
        {
            "name": "RealESRGAN_x2plus",
            "scale": 2,
            "description": "Faster processing model"
        }
    ]
}
```

### Speech Recognition

#### POST `/api/v1/speech/transcribe`
Transcribe speech from audio/video files.

**Parameters:**
- `file` (form-data): Audio/video file
- `model` (form-data, optional): Whisper model size - default: "base"
- `language` (form-data, optional): Language code or "auto" - default: "auto"
- `format` (form-data, optional): Output format (json, srt, vtt, txt) - default: "json"
- `keywords` (form-data, optional): Comma-separated keywords to detect

**Request Example:**
```bash
curl -X POST \
  http://localhost:8000/api/v1/speech/transcribe \
  -F "file=@audio.mp3" \
  -F "model=base" \
  -F "language=auto" \
  -F "format=json"
```

**Response:**
```json
{
    "task_id": "speech_def456",
    "status": "processing",
    "message": "Transcription started"
}
```

#### GET `/api/v1/speech/status/{task_id}`
Check transcription task status.

**Response:**
```json
{
    "task_id": "speech_def456",
    "status": "completed",
    "progress": 100,
    "result": {
        "language": "zh",
        "duration": 120.5,
        "segments_count": 25,
        "confidence": 0.92
    }
}
```

#### GET `/api/v1/speech/download/{task_id}`
Download transcription result.

**Response:** JSON/SRT/VTT/TXT file based on requested format

#### GET `/api/v1/speech/models`
Get available speech recognition models.

**Response:**
```json
{
    "models": [
        {
            "name": "tiny",
            "size": "39MB",
            "speed": "~32x realtime"
        },
        {
            "name": "base",
            "size": "74MB",
            "speed": "~16x realtime"
        }
    ]
}
```

### Text-to-Speech

#### POST `/api/v1/tts/synthesize`
Convert text to speech.

**Request Body:**
```json
{
    "text": "Hello, this is a test message.",
    "voice_profile": {
        "provider": "edge",
        "voice_name": "zh-CN-XiaoxiaoNeural",
        "language": "zh-CN",
        "gender": "female",
        "style": "professional",
        "speed": 1.0,
        "pitch": 1.0,
        "volume": 1.0
    }
}
```

**Response:**
```json
{
    "task_id": "tts_ghi789",
    "status": "processing",
    "message": "TTS synthesis started"
}
```

#### POST `/api/v1/tts/voiceover-from-video`
Create voiceover from video transcription.

**Parameters:**
- `file` (form-data): Video file
- `voice_profile` (form-data): JSON string of voice configuration
- `preserve_timing` (form-data, optional): Maintain original timing - default: true
- `add_pauses` (form-data, optional): Add pauses between segments - default: true

**Request Example:**
```bash
curl -X POST \
  http://localhost:8000/api/v1/tts/voiceover-from-video \
  -F "file=@video.mp4" \
  -F 'voice_profile={"provider":"edge","voice_name":"zh-CN-XiaoxiaoNeural","language":"zh-CN"}'
```

#### GET `/api/v1/tts/status/{task_id}`
Check TTS task status.

**Response:**
```json
{
    "task_id": "tts_ghi789",
    "status": "completed",
    "progress": 100,
    "result": {
        "duration": 45.2,
        "voice_used": "zh-CN-XiaoxiaoNeural",
        "text_length": 150
    }
}
```

#### GET `/api/v1/tts/download/{task_id}/{output_type}`
Download TTS result.

**Parameters:**
- `output_type`: "audio" or "video" (for voiceover)

**Response:** Binary audio/video file

#### GET `/api/v1/tts/voices`
Get available TTS voices.

**Parameters:**
- `provider` (query, optional): Filter by provider (edge, gtts, pyttsx3)

**Response:**
```json
{
    "voices": [
        {
            "name": "zh-CN-XiaoxiaoNeural",
            "short_name": "Xiaoxiao",
            "language": "zh-CN",
            "gender": "Female",
            "styles": ["professional", "friendly"],
            "provider": "edge"
        }
    ]
}
```

#### GET `/api/v1/tts/voice-profiles`
Get recommended voice profiles for different use cases.

**Response:**
```json
{
    "profiles": {
        "professional_chinese_female": {
            "provider": "edge",
            "voice_name": "zh-CN-XiaoxiaoNeural",
            "language": "zh-CN",
            "style": "professional"
        }
    }
}
```

### Product Detection

#### POST `/api/v1/products/detect`
Detect products in video content.

**Parameters:**
- `file` (form-data): Video file
- `model` (form-data, optional): YOLO model - default: "yolov8s.pt"
- `confidence` (form-data, optional): Confidence threshold (0.1-0.9) - default: 0.5
- `sample_rate` (form-data, optional): Frames per second to sample - default: 1.0
- `keywords` (form-data, optional): Comma-separated product keywords
- `annotate` (form-data, optional): Create annotated video - default: false
- `speech_analysis` (form-data, optional): Include speech correlation - default: false

**Request Example:**
```bash
curl -X POST \
  http://localhost:8000/api/v1/products/detect \
  -F "file=@product_video.mp4" \
  -F "confidence=0.5" \
  -F "annotate=true" \
  -F "keywords=phone,laptop,camera"
```

**Response:**
```json
{
    "task_id": "products_jkl012",
    "status": "processing",
    "message": "Product detection started"
}
```

#### POST `/api/v1/products/annotate`
Create annotated video with product detections.

**Parameters:**
- `file` (form-data): Video file
- `detections` (form-data): JSON string of detection results
- `style` (form-data, optional): JSON string of annotation style configuration

#### GET `/api/v1/products/status/{task_id}`
Check product detection task status.

**Response:**
```json
{
    "task_id": "products_jkl012",
    "status": "completed",
    "progress": 100,
    "result": {
        "detections_count": 45,
        "unique_products": 8,
        "processing_time": 180,
        "top_products": ["phone", "laptop", "camera"]
    }
}
```

#### GET `/api/v1/products/analysis/{task_id}`
Get detailed product analysis results.

**Response:**
```json
{
    "task_id": "products_jkl012",
    "analysis": {
        "total_detections": 45,
        "unique_products": 8,
        "categories": {
            "electronics": 6,
            "accessories": 2
        },
        "top_products": [
            {
                "name": "phone",
                "confidence": 0.89,
                "appearances": 15,
                "duration": 45.2
            }
        ],
        "speech_correlation": {
            "sync_ratio": 0.75,
            "mentioned_products": 6
        }
    }
}
```

#### GET `/api/v1/products/download/{task_id}/{output_type}`
Download product detection results.

**Parameters:**
- `output_type`: "analysis" (JSON), "video" (annotated), or "report" (detailed report)

#### GET `/api/v1/products/models`
Get available detection models.

**Response:**
```json
{
    "models": [
        {
            "name": "yolov8n.pt",
            "size": "Nano",
            "speed": "Fast",
            "accuracy": "Good"
        },
        {
            "name": "yolov8s.pt",
            "size": "Small",
            "speed": "Balanced",
            "accuracy": "Better"
        }
    ]
}
```

#### GET `/api/v1/products/categories`
Get supported product categories.

**Response:**
```json
{
    "categories": {
        "electronics": ["phone", "laptop", "tablet", "camera"],
        "clothing": ["shirt", "pants", "shoes", "bag"],
        "home": ["chair", "table", "lamp", "decoration"]
    }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| `FILE_TOO_LARGE` | Uploaded file exceeds size limit |
| `UNSUPPORTED_FORMAT` | File format not supported |
| `PROCESSING_FAILED` | Processing task failed |
| `TASK_NOT_FOUND` | Task ID not found |
| `INVALID_PARAMETERS` | Request parameters are invalid |
| `QUOTA_EXCEEDED` | API usage quota exceeded |
| `MODEL_NOT_AVAILABLE` | Requested AI model not available |

## Status Codes

| Status | Description |
|--------|-------------|
| `pending` | Task is queued for processing |
| `processing` | Task is currently being processed |
| `completed` | Task completed successfully |
| `failed` | Task failed with error |

## File Limits

| Type | Max Size | Supported Formats |
|------|----------|-------------------|
| Video | 500 MB | MP4, AVI, MOV, MKV, WMV |
| Audio | 100 MB | MP3, WAV, M4A, AAC, FLAC |
| Image | 50 MB | JPG, PNG, GIF, BMP, WEBP |

## Usage Examples

### Python SDK Example

```python
import requests
import json

# Upload video for upscaling
def upscale_video(file_path):
    url = "http://localhost:8000/api/v1/upscale/upload"

    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'scale': 4, 'model': 'RealESRGAN_x4plus'}

        response = requests.post(url, files=files, data=data)
        return response.json()

# Check task status
def check_status(task_id, endpoint):
    url = f"http://localhost:8000/api/v1/{endpoint}/status/{task_id}"
    response = requests.get(url)
    return response.json()

# Download result
def download_result(task_id, endpoint, output_path):
    url = f"http://localhost:8000/api/v1/{endpoint}/download/{task_id}"

    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True
    return False

# Example usage
result = upscale_video("input_video.mp4")
task_id = result['task_id']

# Poll for completion
import time
while True:
    status = check_status(task_id, "upscale")
    if status['status'] == 'completed':
        download_result(task_id, "upscale", "upscaled_video.mp4")
        break
    elif status['status'] == 'failed':
        print(f"Processing failed: {status.get('error')}")
        break
    time.sleep(5)
```

### JavaScript Example

```javascript
// Upload and process video
async function processVideo(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('scale', '4');

    const response = await fetch('/api/v1/upscale/upload', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    return result.task_id;
}

// Poll task status
async function pollStatus(taskId, endpoint) {
    while (true) {
        const response = await fetch(`/api/v1/${endpoint}/status/${taskId}`);
        const status = await response.json();

        if (status.status === 'completed') {
            return status;
        } else if (status.status === 'failed') {
            throw new Error(status.error);
        }

        await new Promise(resolve => setTimeout(resolve, 2000));
    }
}

// Example usage
const fileInput = document.getElementById('video-file');
const file = fileInput.files[0];

processVideo(file)
    .then(taskId => pollStatus(taskId, 'upscale'))
    .then(result => {
        console.log('Processing completed:', result);
        // Download the result
        window.location.href = `/api/v1/upscale/download/${result.task_id}`;
    })
    .catch(error => {
        console.error('Processing failed:', error);
    });
```

## Webhooks (Future Feature)

Support for webhooks to notify your application when processing is complete will be added in a future version.

```json
{
    "webhook_url": "https://your-app.com/webhook/ai-content",
    "events": ["task.completed", "task.failed"]
}
```

## Support

- **Documentation**: `/docs` (Swagger UI)
- **API Status**: `/health`
- **GitHub Issues**: [Report bugs and feature requests](https://github.com/your-username/ai-content-creator/issues)
- **Email**: support@ai-content-creator.com

## Changelog

### v1.0.0
- Initial release
- Video upscaling with Real-ESRGAN
- Speech-to-text with Whisper
- Text-to-speech with multiple providers
- Product detection with YOLO + CLIP
- Web interface
- Comprehensive API documentation