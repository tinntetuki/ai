"""
FastAPI endpoints for text-to-speech functionality
"""

import os
import uuid
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger

from ..ai_engine.tts_processor import create_tts_processor, VoiceProfile
from ..ai_engine.speech_processor import create_speech_processor


router = APIRouter(prefix="/api/v1/tts", tags=["text-to-speech"])


class VoiceProfileRequest(BaseModel):
    provider: str = "edge"
    voice_name: str
    language: str
    gender: str = "female"
    style: Optional[str] = None
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0


class TTSRequest(BaseModel):
    text: str
    voice_profile: VoiceProfileRequest


class VoiceoverRequest(BaseModel):
    voice_profile: VoiceProfileRequest
    preserve_timing: bool = True
    add_pauses: bool = True


class TTSResponse(BaseModel):
    task_id: str
    status: str
    message: str
    duration: Optional[float] = None
    download_url: Optional[str] = None


# In-memory task tracking
tts_tasks = {}


@router.post("/synthesize", response_model=TTSResponse)
async def synthesize_text(
    background_tasks: BackgroundTasks,
    request: TTSRequest
):
    """
    Synthesize text to speech

    Args:
        request: TTS synthesis request

    Returns:
        Task information with unique ID
    """

    # Validate text length
    if len(request.text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Generate unique task ID
    task_id = str(uuid.uuid4())

    # Setup paths
    output_dir = Path(os.getenv("OUTPUT_PATH", "./data/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{task_id}_tts_output.wav"

    # Initialize task
    tts_tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "output_path": str(output_path),
        "message": "TTS synthesis started",
        "duration": None
    }

    # Start background processing
    background_tasks.add_task(
        process_tts_synthesis,
        task_id=task_id,
        text=request.text,
        voice_profile=request.voice_profile,
        output_path=str(output_path)
    )

    return TTSResponse(
        task_id=task_id,
        status="processing",
        message="TTS synthesis started"
    )


@router.post("/voiceover-from-video", response_model=TTSResponse)
async def create_voiceover_from_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    voice_provider: str = Form("edge"),
    voice_name: str = Form(...),
    language: str = Form(...),
    gender: str = Form("female"),
    style: Optional[str] = Form(None),
    speed: float = Form(1.0),
    pitch: float = Form(1.0),
    volume: float = Form(1.0),
    preserve_timing: bool = Form(True),
    add_pauses: bool = Form(True)
):
    """
    Create voiceover from video transcription

    Args:
        file: Video file to transcribe and create voiceover for
        voice_*: Voice configuration parameters

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
    audio_output_path = output_dir / f"{task_id}_voiceover.wav"
    video_output_path = output_dir / f"{task_id}_video_with_voiceover.mp4"

    # Save uploaded file
    try:
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Video uploaded for voiceover: {input_path}")

    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")

    # Create voice profile
    voice_profile = VoiceProfileRequest(
        provider=voice_provider,
        voice_name=voice_name,
        language=language,
        gender=gender,
        style=style,
        speed=speed,
        pitch=pitch,
        volume=volume
    )

    # Initialize task
    tts_tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "input_path": str(input_path),
        "audio_output_path": str(audio_output_path),
        "video_output_path": str(video_output_path),
        "message": "Voiceover creation started",
        "duration": None
    }

    # Start background processing
    background_tasks.add_task(
        process_voiceover_creation,
        task_id=task_id,
        input_path=str(input_path),
        voice_profile=voice_profile,
        audio_output_path=str(audio_output_path),
        video_output_path=str(video_output_path),
        preserve_timing=preserve_timing,
        add_pauses=add_pauses
    )

    return TTSResponse(
        task_id=task_id,
        status="processing",
        message="Voiceover creation started"
    )


@router.get("/status/{task_id}", response_model=TTSResponse)
async def get_tts_status(task_id: str):
    """Get the status of a TTS task"""

    if task_id not in tts_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tts_tasks[task_id]

    download_url = None
    if task["status"] == "completed":
        # Determine which output file to serve
        if "video_output_path" in task and os.path.exists(task["video_output_path"]):
            download_url = f"/api/v1/tts/download/{task_id}/video"
        elif "audio_output_path" in task and os.path.exists(task["audio_output_path"]):
            download_url = f"/api/v1/tts/download/{task_id}/audio"
        elif "output_path" in task and os.path.exists(task["output_path"]):
            download_url = f"/api/v1/tts/download/{task_id}/audio"

    return TTSResponse(
        task_id=task_id,
        status=task["status"],
        message=task["message"],
        duration=task.get("duration"),
        download_url=download_url
    )


@router.get("/download/{task_id}/{output_type}")
async def download_tts_output(task_id: str, output_type: str):
    """Download TTS output file"""

    if task_id not in tts_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tts_tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")

    # Determine output path
    if output_type == "video" and "video_output_path" in task:
        output_path = task["video_output_path"]
        media_type = 'video/mp4'
    elif output_type == "audio":
        # Try different audio output paths
        if "audio_output_path" in task:
            output_path = task["audio_output_path"]
        else:
            output_path = task["output_path"]
        media_type = 'audio/wav'
    else:
        raise HTTPException(status_code=400, detail="Invalid output type")

    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")

    filename = os.path.basename(output_path)
    return FileResponse(
        output_path,
        media_type=media_type,
        filename=filename
    )


@router.get("/voices")
async def list_available_voices(provider: str = Query("edge")):
    """List available voices for a TTS provider"""

    try:
        tts_processor = create_tts_processor()
        voices = await tts_processor.get_available_voices(provider)

        return {
            "provider": provider,
            "voices": voices,
            "count": len(voices)
        }

    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        raise HTTPException(status_code=500, detail="Failed to list voices")


@router.get("/voice-profiles")
async def get_preset_voice_profiles():
    """Get preset voice profiles optimized for product videos"""

    profiles = []

    # Chinese profiles
    for voice_type in ["professional", "friendly", "energetic"]:
        for gender in ["female", "male"]:
            tts_processor = create_tts_processor()
            profile = tts_processor.create_product_voice_profile(
                language="zh",
                voice_type=voice_type,
                gender=gender
            )

            profiles.append({
                "id": f"zh_{voice_type}_{gender}",
                "name": f"中文 {voice_type.title()} ({gender.title()})",
                "language": "zh-CN",
                "voice_type": voice_type,
                "gender": gender,
                "provider": profile.provider,
                "voice_name": profile.voice_name,
                "style": profile.style,
                "speed": profile.speed,
                "pitch": profile.pitch,
                "volume": profile.volume
            })

    # English profiles
    for voice_type in ["professional", "friendly", "energetic"]:
        for gender in ["female", "male"]:
            tts_processor = create_tts_processor()
            profile = tts_processor.create_product_voice_profile(
                language="en",
                voice_type=voice_type,
                gender=gender
            )

            profiles.append({
                "id": f"en_{voice_type}_{gender}",
                "name": f"English {voice_type.title()} ({gender.title()})",
                "language": "en-US",
                "voice_type": voice_type,
                "gender": gender,
                "provider": profile.provider,
                "voice_name": profile.voice_name,
                "style": profile.style,
                "speed": profile.speed,
                "pitch": profile.pitch,
                "volume": profile.volume
            })

    return {"profiles": profiles}


async def process_tts_synthesis(
    task_id: str,
    text: str,
    voice_profile: VoiceProfileRequest,
    output_path: str
):
    """Background task for TTS synthesis"""

    try:
        logger.info(f"Starting TTS synthesis task {task_id}")

        # Create TTS processor
        tts_processor = create_tts_processor()

        # Create voice profile
        voice_config = VoiceProfile(
            provider=voice_profile.provider,
            voice_name=voice_profile.voice_name,
            language=voice_profile.language,
            gender=voice_profile.gender,
            style=voice_profile.style,
            speed=voice_profile.speed,
            pitch=voice_profile.pitch,
            volume=voice_profile.volume
        )

        # Update status
        tts_tasks[task_id]["message"] = "Synthesizing speech..."

        # Synthesize
        result_path = await tts_processor.synthesize_text(
            text=text,
            voice_profile=voice_config,
            output_path=output_path
        )

        # Get audio duration
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(result_path)
        duration = len(audio) / 1000.0

        # Update task status
        tts_tasks[task_id].update({
            "status": "completed",
            "message": "TTS synthesis completed successfully",
            "duration": duration
        })

        logger.info(f"TTS synthesis task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"TTS synthesis task {task_id} error: {e}")
        tts_tasks[task_id].update({
            "status": "failed",
            "message": f"Error: {str(e)}"
        })


async def process_voiceover_creation(
    task_id: str,
    input_path: str,
    voice_profile: VoiceProfileRequest,
    audio_output_path: str,
    video_output_path: str,
    preserve_timing: bool,
    add_pauses: bool
):
    """Background task for voiceover creation"""

    try:
        logger.info(f"Starting voiceover creation task {task_id}")

        # Create processors
        speech_processor = create_speech_processor(model_size="base")
        tts_processor = create_tts_processor()

        # Update status
        tts_tasks[task_id]["message"] = "Transcribing original video..."

        # Transcribe original video
        transcription = speech_processor.transcribe_video(input_path)

        # Create voice profile
        voice_config = VoiceProfile(
            provider=voice_profile.provider,
            voice_name=voice_profile.voice_name,
            language=voice_profile.language,
            gender=voice_profile.gender,
            style=voice_profile.style,
            speed=voice_profile.speed,
            pitch=voice_profile.pitch,
            volume=voice_profile.volume
        )

        # Update status
        tts_tasks[task_id]["message"] = "Creating voiceover..."

        # Create voiceover
        await tts_processor.create_voiceover_from_transcription(
            transcription=transcription,
            voice_profile=voice_config,
            output_path=audio_output_path,
            preserve_timing=preserve_timing,
            add_pauses=add_pauses
        )

        # Update status
        tts_tasks[task_id]["message"] = "Replacing video audio..."

        # Replace video audio
        success = await tts_processor.replace_video_audio(
            video_path=input_path,
            new_audio_path=audio_output_path,
            output_path=video_output_path,
            fade_duration=0.5
        )

        if success:
            # Get duration
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(audio_output_path)
            duration = len(audio) / 1000.0

            # Update task status
            tts_tasks[task_id].update({
                "status": "completed",
                "message": "Voiceover creation completed successfully",
                "duration": duration
            })

            logger.info(f"Voiceover creation task {task_id} completed successfully")
        else:
            raise Exception("Failed to replace video audio")

    except Exception as e:
        logger.error(f"Voiceover creation task {task_id} error: {e}")
        tts_tasks[task_id].update({
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
async def cancel_tts_task(task_id: str):
    """Cancel and cleanup a TTS task"""

    if task_id not in tts_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tts_tasks[task_id]

    # Cleanup files
    cleanup_paths = ["input_path", "output_path", "audio_output_path", "video_output_path"]
    for path_key in cleanup_paths:
        if path_key in task and os.path.exists(task[path_key]):
            try:
                os.remove(task[path_key])
            except Exception as e:
                logger.warning(f"Failed to cleanup {path_key}: {e}")

    # Remove task
    del tts_tasks[task_id]

    return {"message": "Task cancelled and cleaned up"}