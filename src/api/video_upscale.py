"""
FastAPI endpoints for video upscaling functionality
"""

import os
import uuid
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger

from ..ai_engine.video_upscaler import create_upscaler


router = APIRouter(prefix="/api/v1/upscale", tags=["video-upscale"])


class UpscaleRequest(BaseModel):
    scale: Optional[int] = 4
    model_name: Optional[str] = "RealESRGAN_x4plus"
    segment_duration: Optional[float] = None
    smart_segments: Optional[bool] = True


class UpscaleResponse(BaseModel):
    task_id: str
    status: str
    message: str
    output_url: Optional[str] = None


# In-memory task tracking (replace with Redis in production)
tasks = {}


@router.post("/upload", response_model=UpscaleResponse)
async def upload_and_upscale_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    scale: int = 4,
    model_name: str = "RealESRGAN_x4plus",
    segment_duration: Optional[float] = None,
    smart_segments: bool = True
):
    """
    Upload and upscale a video file

    Args:
        file: Video file to upscale
        scale: Upscaling factor (2, 4, or 8)
        model_name: Real-ESRGAN model name
        segment_duration: Duration to process (None for full video)
        smart_segments: Use intelligent segment detection

    Returns:
        Task information with unique ID
    """

    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Generate unique task ID
    task_id = str(uuid.uuid4())

    # Setup paths
    input_dir = Path(os.getenv("INPUT_PATH", "./data/input"))
    output_dir = Path(os.getenv("OUTPUT_PATH", "./data/output"))
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = input_dir / f"{task_id}_{file.filename}"
    output_path = output_dir / f"{task_id}_upscaled_{file.filename}"

    # Save uploaded file
    try:
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Video uploaded: {input_path}")

    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")

    # Initialize task
    tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "message": "Processing started"
    }

    # Start background processing
    background_tasks.add_task(
        process_video_upscale,
        task_id=task_id,
        input_path=str(input_path),
        output_path=str(output_path),
        scale=scale,
        model_name=model_name,
        segment_duration=segment_duration,
        smart_segments=smart_segments
    )

    return UpscaleResponse(
        task_id=task_id,
        status="processing",
        message="Video upscaling started"
    )


@router.get("/status/{task_id}", response_model=UpscaleResponse)
async def get_upscale_status(task_id: str):
    """Get the status of an upscaling task"""

    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]

    output_url = None
    if task["status"] == "completed" and os.path.exists(task["output_path"]):
        output_url = f"/api/v1/upscale/download/{task_id}"

    return UpscaleResponse(
        task_id=task_id,
        status=task["status"],
        message=task["message"],
        output_url=output_url
    )


@router.get("/download/{task_id}")
async def download_upscaled_video(task_id: str):
    """Download the upscaled video file"""

    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")

    output_path = task["output_path"]
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")

    filename = os.path.basename(output_path)
    return FileResponse(
        output_path,
        media_type='video/mp4',
        filename=filename
    )


async def process_video_upscale(
    task_id: str,
    input_path: str,
    output_path: str,
    scale: int,
    model_name: str,
    segment_duration: Optional[float],
    smart_segments: bool
):
    """Background task for video upscaling"""

    def progress_callback(progress: float):
        tasks[task_id]["progress"] = progress
        tasks[task_id]["message"] = f"Processing... {progress:.1f}%"

    try:
        logger.info(f"Starting upscale task {task_id}")

        # Create upscaler
        upscaler = create_upscaler(scale=scale, model_name=model_name)

        # Determine processing segments
        target_segments = None
        if smart_segments and not segment_duration:
            target_segments = upscaler.get_smart_segments(input_path, max_segments=3)
            logger.info(f"Smart segments: {target_segments}")

        # Process video
        success = upscaler.upscale_video(
            input_path=input_path,
            output_path=output_path,
            segment_duration=segment_duration,
            target_segments=target_segments,
            progress_callback=progress_callback
        )

        if success:
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["progress"] = 100
            tasks[task_id]["message"] = "Video upscaling completed successfully"
            logger.info(f"Task {task_id} completed successfully")
        else:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["message"] = "Video upscaling failed"
            logger.error(f"Task {task_id} failed")

    except Exception as e:
        logger.error(f"Task {task_id} error: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"Error: {str(e)}"

    finally:
        # Cleanup input file
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup input file: {e}")


@router.delete("/task/{task_id}")
async def cancel_task(task_id: str):
    """Cancel and cleanup a task"""

    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]

    # Cleanup files
    for path_key in ["input_path", "output_path"]:
        if path_key in task and os.path.exists(task[path_key]):
            try:
                os.remove(task[path_key])
            except Exception as e:
                logger.warning(f"Failed to cleanup {path_key}: {e}")

    # Remove task
    del tasks[task_id]

    return {"message": "Task cancelled and cleaned up"}


@router.get("/models")
async def list_available_models():
    """List available Real-ESRGAN models"""

    models = [
        {
            "name": "RealESRGAN_x4plus",
            "scale": 4,
            "description": "General purpose 4x upscaling, good for most content"
        },
        {
            "name": "RealESRGAN_x2plus",
            "scale": 2,
            "description": "2x upscaling, faster processing"
        },
        {
            "name": "RealESRGAN_x4plus_anime_6B",
            "scale": 4,
            "description": "Optimized for anime/cartoon content"
        }
    ]

    return {"models": models}