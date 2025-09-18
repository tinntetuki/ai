"""
FastAPI endpoints for product detection and annotation functionality
"""

import os
import uuid
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from loguru import logger

from ..ai_engine.product_detector import create_product_detector, AnnotationStyle
from ..ai_engine.speech_processor import create_speech_processor


router = APIRouter(prefix="/api/v1/products", tags=["product-detection"])


class ProductDetectionRequest(BaseModel):
    sample_rate: float = 1.0
    confidence_threshold: float = 0.5
    product_keywords: Optional[List[str]] = None
    yolo_model: str = "yolov8n.pt"
    clip_model: str = "ViT-B/32"


class AnnotationStyleRequest(BaseModel):
    bbox_color: List[int] = [0, 255, 0]
    bbox_thickness: int = 2
    text_color: List[int] = [255, 255, 255]
    text_bg_color: List[int] = [0, 0, 0]
    font_scale: float = 0.7
    show_confidence: bool = True
    show_description: bool = True


class ProductDetectionResponse(BaseModel):
    task_id: str
    status: str
    message: str
    total_detections: Optional[int] = None
    unique_products: Optional[int] = None
    processing_time: Optional[float] = None
    download_url: Optional[str] = None


class ProductAnalysisResponse(BaseModel):
    task_id: str
    detections_count: int
    tracks_count: int
    categories: Dict[str, int]
    top_products: List[Dict[str, Any]]
    speech_correlation: Optional[Dict[str, Any]] = None


# In-memory task tracking
detection_tasks = {}


@router.post("/detect", response_model=ProductDetectionResponse)
async def detect_products_in_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sample_rate: float = Form(1.0),
    confidence_threshold: float = Form(0.5),
    product_keywords: Optional[str] = Form(None),
    yolo_model: str = Form("yolov8n.pt"),
    clip_model: str = Form("ViT-B/32"),
    include_speech_analysis: bool = Form(False),
    create_annotated_video: bool = Form(False)
):
    """
    Detect products in uploaded video

    Args:
        file: Video file to analyze
        sample_rate: Frames per second to sample
        confidence_threshold: Minimum detection confidence
        product_keywords: Comma-separated product keywords
        yolo_model: YOLO model to use
        clip_model: CLIP model for semantic understanding
        include_speech_analysis: Analyze correlation with speech
        create_annotated_video: Generate annotated video output

    Returns:
        Task information with unique ID
    """

    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Parse keywords
    keywords = None
    if product_keywords:
        keywords = [k.strip() for k in product_keywords.split(',')]

    # Generate unique task ID
    task_id = str(uuid.uuid4())

    # Setup paths
    input_dir = Path(os.getenv("INPUT_PATH", "./data/input"))
    output_dir = Path(os.getenv("OUTPUT_PATH", "./data/output"))
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = input_dir / f"{task_id}_{file.filename}"
    results_path = output_dir / f"{task_id}_product_analysis.json"
    annotated_video_path = output_dir / f"{task_id}_annotated_video.mp4"

    # Save uploaded file
    try:
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Video uploaded for product detection: {input_path}")

    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")

    # Initialize task
    detection_tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "input_path": str(input_path),
        "results_path": str(results_path),
        "annotated_video_path": str(annotated_video_path) if create_annotated_video else None,
        "message": "Product detection started",
        "total_detections": None,
        "unique_products": None,
        "processing_time": None
    }

    # Start background processing
    background_tasks.add_task(
        process_product_detection,
        task_id=task_id,
        input_path=str(input_path),
        results_path=str(results_path),
        annotated_video_path=str(annotated_video_path) if create_annotated_video else None,
        sample_rate=sample_rate,
        confidence_threshold=confidence_threshold,
        product_keywords=keywords,
        yolo_model=yolo_model,
        clip_model=clip_model,
        include_speech_analysis=include_speech_analysis
    )

    return ProductDetectionResponse(
        task_id=task_id,
        status="processing",
        message="Product detection started"
    )


@router.post("/annotate", response_model=ProductDetectionResponse)
async def create_annotated_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    style: Optional[str] = Form(None),  # JSON string
    sample_rate: float = Form(1.0),
    confidence_threshold: float = Form(0.5),
    product_keywords: Optional[str] = Form(None)
):
    """
    Create annotated video with product detections

    Args:
        file: Video file to annotate
        style: JSON string of annotation style settings
        sample_rate: Detection sampling rate
        confidence_threshold: Minimum detection confidence
        product_keywords: Product keywords to focus on

    Returns:
        Task information
    """

    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Parse style settings
    annotation_style = None
    if style:
        try:
            style_dict = json.loads(style)
            annotation_style = AnnotationStyle(
                bbox_color=tuple(style_dict.get("bbox_color", [0, 255, 0])),
                bbox_thickness=style_dict.get("bbox_thickness", 2),
                text_color=tuple(style_dict.get("text_color", [255, 255, 255])),
                text_bg_color=tuple(style_dict.get("text_bg_color", [0, 0, 0])),
                font_scale=style_dict.get("font_scale", 0.7),
                show_confidence=style_dict.get("show_confidence", True),
                show_description=style_dict.get("show_description", True)
            )
        except Exception as e:
            logger.warning(f"Invalid style settings: {e}")

    # Parse keywords
    keywords = None
    if product_keywords:
        keywords = [k.strip() for k in product_keywords.split(',')]

    # Generate unique task ID
    task_id = str(uuid.uuid4())

    # Setup paths
    input_dir = Path(os.getenv("INPUT_PATH", "./data/input"))
    output_dir = Path(os.getenv("OUTPUT_PATH", "./data/output"))
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = input_dir / f"{task_id}_{file.filename}"
    output_path = output_dir / f"{task_id}_annotated.mp4"

    # Save uploaded file
    try:
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Video uploaded for annotation: {input_path}")

    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")

    # Initialize task
    detection_tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "message": "Video annotation started",
        "annotation_style": annotation_style
    }

    # Start background processing
    background_tasks.add_task(
        process_video_annotation,
        task_id=task_id,
        input_path=str(input_path),
        output_path=str(output_path),
        annotation_style=annotation_style,
        sample_rate=sample_rate,
        confidence_threshold=confidence_threshold,
        keywords=keywords
    )

    return ProductDetectionResponse(
        task_id=task_id,
        status="processing",
        message="Video annotation started"
    )


@router.get("/status/{task_id}", response_model=ProductDetectionResponse)
async def get_detection_status(task_id: str):
    """Get the status of a product detection task"""

    if task_id not in detection_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = detection_tasks[task_id]

    download_url = None
    if task["status"] == "completed":
        if "annotated_video_path" in task and task["annotated_video_path"] and os.path.exists(task["annotated_video_path"]):
            download_url = f"/api/v1/products/download/{task_id}/video"
        elif "output_path" in task and os.path.exists(task["output_path"]):
            download_url = f"/api/v1/products/download/{task_id}/video"
        elif "results_path" in task and os.path.exists(task["results_path"]):
            download_url = f"/api/v1/products/download/{task_id}/results"

    return ProductDetectionResponse(
        task_id=task_id,
        status=task["status"],
        message=task["message"],
        total_detections=task.get("total_detections"),
        unique_products=task.get("unique_products"),
        processing_time=task.get("processing_time"),
        download_url=download_url
    )


@router.get("/analysis/{task_id}")
async def get_product_analysis(task_id: str):
    """Get detailed product analysis results"""

    if task_id not in detection_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = detection_tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")

    results_path = task.get("results_path")
    if not results_path or not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="Analysis results not found")

    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)

        return JSONResponse(content=analysis_data)

    except Exception as e:
        logger.error(f"Failed to load analysis results: {e}")
        raise HTTPException(status_code=500, detail="Failed to load analysis results")


@router.get("/download/{task_id}/{output_type}")
async def download_detection_output(task_id: str, output_type: str):
    """Download detection output files"""

    if task_id not in detection_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = detection_tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")

    # Determine output path
    if output_type == "video":
        if "annotated_video_path" in task and task["annotated_video_path"]:
            output_path = task["annotated_video_path"]
            media_type = 'video/mp4'
        elif "output_path" in task:
            output_path = task["output_path"]
            media_type = 'video/mp4'
        else:
            raise HTTPException(status_code=404, detail="Annotated video not available")
    elif output_type == "results":
        output_path = task.get("results_path")
        media_type = 'application/json'
    else:
        raise HTTPException(status_code=400, detail="Invalid output type")

    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")

    filename = os.path.basename(output_path)
    return FileResponse(
        output_path,
        media_type=media_type,
        filename=filename
    )


@router.get("/models")
async def list_available_models():
    """List available detection models"""

    models = {
        "yolo_models": [
            {
                "name": "yolov8n.pt",
                "size": "Small",
                "speed": "Fast",
                "accuracy": "Good",
                "description": "Nano model, best for real-time processing"
            },
            {
                "name": "yolov8s.pt",
                "size": "Small",
                "speed": "Fast",
                "accuracy": "Better",
                "description": "Small model, good balance"
            },
            {
                "name": "yolov8m.pt",
                "size": "Medium",
                "speed": "Medium",
                "accuracy": "High",
                "description": "Medium model, better accuracy"
            },
            {
                "name": "yolov8l.pt",
                "size": "Large",
                "speed": "Slow",
                "accuracy": "Very High",
                "description": "Large model, best accuracy"
            }
        ],
        "clip_models": [
            {
                "name": "ViT-B/32",
                "description": "Balanced vision-text understanding"
            },
            {
                "name": "ViT-B/16",
                "description": "Higher resolution, better quality"
            },
            {
                "name": "ViT-L/14",
                "description": "Large model, best performance"
            }
        ]
    }

    return models


@router.get("/categories")
async def get_product_categories():
    """Get supported product categories"""

    categories = {
        "electronics": {
            "items": ["phone", "laptop", "tablet", "watch", "camera", "headphones"],
            "description": "Electronic devices and gadgets"
        },
        "clothing": {
            "items": ["shirt", "pants", "dress", "shoes", "hat", "jacket"],
            "description": "Clothing and fashion items"
        },
        "home": {
            "items": ["chair", "table", "lamp", "vase", "decoration"],
            "description": "Home and furniture items"
        },
        "beauty": {
            "items": ["cosmetics", "perfume", "skincare", "makeup"],
            "description": "Beauty and personal care products"
        },
        "sports": {
            "items": ["ball", "equipment", "sportswear"],
            "description": "Sports and fitness equipment"
        },
        "food": {
            "items": ["snack", "drink", "fruit", "cake"],
            "description": "Food and beverage items"
        }
    }

    return {"categories": categories}


async def process_product_detection(
    task_id: str,
    input_path: str,
    results_path: str,
    annotated_video_path: Optional[str],
    sample_rate: float,
    confidence_threshold: float,
    product_keywords: Optional[List[str]],
    yolo_model: str,
    clip_model: str,
    include_speech_analysis: bool
):
    """Background task for product detection processing"""

    import time
    start_time = time.time()

    try:
        logger.info(f"Starting product detection task {task_id}")

        # Update status
        detection_tasks[task_id]["message"] = "Initializing detection models..."

        # Create detector
        detector = create_product_detector(
            yolo_model=yolo_model,
            clip_model=clip_model
        )

        # Update status
        detection_tasks[task_id]["message"] = "Detecting products in video..."

        # Detect products
        detections = detector.detect_products_in_video(
            video_path=input_path,
            sample_rate=sample_rate,
            confidence_threshold=confidence_threshold,
            product_keywords=product_keywords
        )

        # Update status
        detection_tasks[task_id]["message"] = "Creating product tracks..."

        # Create product tracks
        tracks = detector.create_product_tracks(detections)

        # Speech analysis if requested
        speech_analysis = None
        if include_speech_analysis:
            detection_tasks[task_id]["message"] = "Analyzing speech correlation..."

            speech_processor = create_speech_processor(model_size="base")
            transcription = speech_processor.transcribe_video(input_path)

            speech_analysis = detector.analyze_products_with_speech(
                detections, transcription
            )

        # Generate summary
        detection_tasks[task_id]["message"] = "Generating analysis summary..."

        summary = detector.generate_product_summary(tracks, speech_analysis)

        # Create annotated video if requested
        if annotated_video_path:
            detection_tasks[task_id]["message"] = "Creating annotated video..."

            success = detector.annotate_video_with_products(
                video_path=input_path,
                detections=detections,
                output_path=annotated_video_path
            )

            if not success:
                logger.warning("Failed to create annotated video")

        # Save results
        results = {
            "task_id": task_id,
            "processing_info": {
                "sample_rate": sample_rate,
                "confidence_threshold": confidence_threshold,
                "product_keywords": product_keywords,
                "models": {
                    "yolo": yolo_model,
                    "clip": clip_model
                }
            },
            "detections": [
                {
                    "bbox": {
                        "x1": d.bbox.x1, "y1": d.bbox.y1,
                        "x2": d.bbox.x2, "y2": d.bbox.y2
                    },
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "timestamp": d.timestamp,
                    "description": d.description,
                    "category": d.category
                }
                for d in detections
            ],
            "tracks": [
                {
                    "track_id": t.track_id,
                    "class_name": t.class_name,
                    "confidence_avg": t.confidence_avg,
                    "first_appearance": t.first_appearance,
                    "last_appearance": t.last_appearance,
                    "importance_score": t.importance_score,
                    "description": t.description
                }
                for t in tracks
            ],
            "summary": summary,
            "speech_analysis": speech_analysis
        }

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Update task status
        detection_tasks[task_id].update({
            "status": "completed",
            "message": "Product detection completed successfully",
            "total_detections": len(detections),
            "unique_products": len(tracks),
            "processing_time": processing_time
        })

        logger.info(f"Product detection task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Product detection task {task_id} error: {e}")
        detection_tasks[task_id].update({
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


async def process_video_annotation(
    task_id: str,
    input_path: str,
    output_path: str,
    annotation_style: Optional[AnnotationStyle],
    sample_rate: float,
    confidence_threshold: float,
    keywords: Optional[List[str]]
):
    """Background task for video annotation"""

    try:
        logger.info(f"Starting video annotation task {task_id}")

        # Create detector
        detector = create_product_detector()

        # Update status
        detection_tasks[task_id]["message"] = "Detecting products..."

        # Detect products
        detections = detector.detect_products_in_video(
            video_path=input_path,
            sample_rate=sample_rate,
            confidence_threshold=confidence_threshold,
            product_keywords=keywords
        )

        # Update status
        detection_tasks[task_id]["message"] = "Creating annotated video..."

        # Create annotated video
        success = detector.annotate_video_with_products(
            video_path=input_path,
            detections=detections,
            output_path=output_path,
            style=annotation_style
        )

        if success:
            detection_tasks[task_id].update({
                "status": "completed",
                "message": "Video annotation completed successfully",
                "total_detections": len(detections)
            })
        else:
            detection_tasks[task_id].update({
                "status": "failed",
                "message": "Video annotation failed"
            })

        logger.info(f"Video annotation task {task_id} completed")

    except Exception as e:
        logger.error(f"Video annotation task {task_id} error: {e}")
        detection_tasks[task_id].update({
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
async def cancel_detection_task(task_id: str):
    """Cancel and cleanup a detection task"""

    if task_id not in detection_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = detection_tasks[task_id]

    # Cleanup files
    cleanup_paths = ["input_path", "results_path", "annotated_video_path", "output_path"]
    for path_key in cleanup_paths:
        if path_key in task and task[path_key] and os.path.exists(task[path_key]):
            try:
                os.remove(task[path_key])
            except Exception as e:
                logger.warning(f"Failed to cleanup {path_key}: {e}")

    # Remove task
    del detection_tasks[task_id]

    return {"message": "Task cancelled and cleaned up"}