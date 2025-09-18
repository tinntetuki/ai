"""
FastAPI endpoints for speech-to-text functionality
"""

import os
import uuid
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger

from ..ai_engine.speech_processor import create_speech_processor, TranscriptionResult


router = APIRouter(prefix="/api/v1/speech", tags=["speech-to-text"])


class TranscriptionRequest(BaseModel):
    model_size: Optional[str] = "base"
    language: Optional[str] = None
    output_format: Optional[str] = "json"
    segment_duration: Optional[float] = None


class TranscriptionResponse(BaseModel):
    task_id: str
    status: str
    message: str
    language: Optional[str] = None
    duration: Optional[float] = None
    download_url: Optional[str] = None


class SegmentInfo(BaseModel):
    start: float
    end: float
    text: str
    confidence: Optional[float] = None


class TranscriptionDetails(BaseModel):
    language: str
    duration: float
    full_text: str
    segments: List[SegmentInfo]


# In-memory task tracking
transcription_tasks = {}


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_size: str = Query("base", description="Whisper model size"),
    language: Optional[str] = Query(None, description="Language code (auto-detect if None)"),
    output_format: str = Query("json", description="Output format: json, srt, vtt, txt"),
    segment_duration: Optional[float] = Query(None, description="Process only first N seconds")
):
    """
    Transcribe video or audio file to text

    Args:
        file: Video or audio file to transcribe
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Language code (e.g., 'zh', 'en') or None for auto-detection
        output_format: Output format (json, srt, vtt, txt)
        segment_duration: Process only specified duration in seconds

    Returns:
        Task information with unique ID
    """

    # Validate file type
    allowed_types = ['video/', 'audio/']
    if not any(file.content_type.startswith(t) for t in allowed_types):
        raise HTTPException(status_code=400, detail="File must be video or audio")

    # Validate model size
    valid_models = ['tiny', 'base', 'small', 'medium', 'large']
    if model_size not in valid_models:
        raise HTTPException(status_code=400, detail=f"Model size must be one of: {valid_models}")

    # Validate output format
    valid_formats = ['json', 'srt', 'vtt', 'txt']
    if output_format not in valid_formats:
        raise HTTPException(status_code=400, detail=f"Output format must be one of: {valid_formats}")

    # Generate unique task ID
    task_id = str(uuid.uuid4())

    # Setup paths
    input_dir = Path(os.getenv("INPUT_PATH", "./data/input"))
    output_dir = Path(os.getenv("OUTPUT_PATH", "./data/output"))
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = input_dir / f"{task_id}_{file.filename}"

    # Determine output file extension
    ext_map = {'json': '.json', 'srt': '.srt', 'vtt': '.vtt', 'txt': '.txt'}
    output_path = output_dir / f"{task_id}_transcription{ext_map[output_format]}"

    # Save uploaded file
    try:
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"File uploaded for transcription: {input_path}")

    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")

    # Initialize task
    transcription_tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "message": "Transcription started",
        "language": None,
        "duration": None
    }

    # Start background processing
    background_tasks.add_task(
        process_transcription,
        task_id=task_id,
        input_path=str(input_path),
        output_path=str(output_path),
        model_size=model_size,
        language=language,
        output_format=output_format,
        segment_duration=segment_duration
    )

    return TranscriptionResponse(
        task_id=task_id,
        status="processing",
        message="Transcription started"
    )


@router.get("/status/{task_id}", response_model=TranscriptionResponse)
async def get_transcription_status(task_id: str):
    """Get the status of a transcription task"""

    if task_id not in transcription_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = transcription_tasks[task_id]

    download_url = None
    if task["status"] == "completed" and os.path.exists(task["output_path"]):
        download_url = f"/api/v1/speech/download/{task_id}"

    return TranscriptionResponse(
        task_id=task_id,
        status=task["status"],
        message=task["message"],
        language=task.get("language"),
        duration=task.get("duration"),
        download_url=download_url
    )


@router.get("/details/{task_id}", response_model=TranscriptionDetails)
async def get_transcription_details(task_id: str):
    """Get detailed transcription results"""

    if task_id not in transcription_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = transcription_tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Transcription not completed")

    # Load result from JSON file
    json_path = task["output_path"]
    if not json_path.endswith('.json'):
        # Find corresponding JSON file
        base_path = str(Path(task["output_path"]).with_suffix('.json'))
        json_path = base_path

    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Transcription details not found")

    try:
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = [
            SegmentInfo(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                confidence=seg.get("confidence")
            )
            for seg in data["segments"]
        ]

        return TranscriptionDetails(
            language=data["language"],
            duration=data["duration"],
            full_text=data["full_text"],
            segments=segments
        )

    except Exception as e:
        logger.error(f"Failed to load transcription details: {e}")
        raise HTTPException(status_code=500, detail="Failed to load transcription details")


@router.get("/download/{task_id}")
async def download_transcription(task_id: str):
    """Download the transcription file"""

    if task_id not in transcription_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = transcription_tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Transcription not completed")

    output_path = task["output_path"]
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Transcription file not found")

    filename = os.path.basename(output_path)

    # Determine media type
    if filename.endswith('.json'):
        media_type = 'application/json'
    elif filename.endswith('.srt'):
        media_type = 'text/plain'
    elif filename.endswith('.vtt'):
        media_type = 'text/vtt'
    else:
        media_type = 'text/plain'

    return FileResponse(
        output_path,
        media_type=media_type,
        filename=filename
    )


@router.post("/analyze-keywords")
async def analyze_product_keywords(
    task_id: str,
    keywords: List[str]
):
    """
    Analyze transcription for product keyword mentions

    Args:
        task_id: Transcription task ID
        keywords: List of product keywords to search for

    Returns:
        Segments containing product mentions
    """

    if task_id not in transcription_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = transcription_tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Transcription not completed")

    try:
        # Load transcription result
        json_path = str(Path(task["output_path"]).with_suffix('.json'))

        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Find keyword mentions
        keyword_segments = []
        for segment in data["segments"]:
            text_lower = segment["text"].lower()
            matched_keywords = [
                keyword for keyword in keywords
                if keyword.lower() in text_lower
            ]

            if matched_keywords:
                keyword_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "matched_keywords": matched_keywords
                })

        return {
            "task_id": task_id,
            "total_segments": len(data["segments"]),
            "keyword_segments": keyword_segments,
            "keyword_count": len(keyword_segments)
        }

    except Exception as e:
        logger.error(f"Keyword analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Keyword analysis failed")


async def process_transcription(
    task_id: str,
    input_path: str,
    output_path: str,
    model_size: str,
    language: Optional[str],
    output_format: str,
    segment_duration: Optional[float]
):
    """Background task for transcription processing"""

    try:
        logger.info(f"Starting transcription task {task_id}")

        # Update status
        transcription_tasks[task_id]["message"] = "Loading Whisper model..."

        # Create speech processor
        processor = create_speech_processor(model_size=model_size)

        # Update status
        transcription_tasks[task_id]["message"] = "Processing audio..."

        # Transcribe video
        result = processor.transcribe_video(
            video_path=input_path,
            language=language,
            segment_duration=segment_duration
        )

        # Save in requested format
        processor.save_transcription(result, output_path, format=output_format)

        # Also save JSON version for API access
        if output_format != "json":
            json_path = str(Path(output_path).with_suffix('.json'))
            processor.save_transcription(result, json_path, format="json")

        # Update task status
        transcription_tasks[task_id].update({
            "status": "completed",
            "message": "Transcription completed successfully",
            "language": result.language,
            "duration": result.duration
        })

        logger.info(f"Transcription task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Transcription task {task_id} error: {e}")
        transcription_tasks[task_id].update({
            "status": "failed",
            "message": f"Error: {str(e)}"
        })

    finally:
        # Cleanup input file
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup input file: {e}")


@router.delete("/task/{task_id}")
async def cancel_transcription_task(task_id: str):
    """Cancel and cleanup a transcription task"""

    if task_id not in transcription_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = transcription_tasks[task_id]

    # Cleanup files
    for path_key in ["input_path", "output_path"]:
        if path_key in task and os.path.exists(task[path_key]):
            try:
                os.remove(task[path_key])
            except Exception as e:
                logger.warning(f"Failed to cleanup {path_key}: {e}")

    # Remove task
    del transcription_tasks[task_id]

    return {"message": "Task cancelled and cleaned up"}


@router.get("/models")
async def list_whisper_models():
    """List available Whisper models"""

    models = [
        {
            "name": "tiny",
            "size": "~39 MB",
            "speed": "~32x realtime",
            "description": "Fastest, lowest quality"
        },
        {
            "name": "base",
            "size": "~74 MB",
            "speed": "~16x realtime",
            "description": "Good balance of speed and quality"
        },
        {
            "name": "small",
            "size": "~244 MB",
            "speed": "~6x realtime",
            "description": "Better quality, slower"
        },
        {
            "name": "medium",
            "size": "~769 MB",
            "speed": "~2x realtime",
            "description": "High quality, much slower"
        },
        {
            "name": "large",
            "size": "~1550 MB",
            "speed": "~1x realtime",
            "description": "Best quality, slowest"
        }
    ]

    return {"models": models}