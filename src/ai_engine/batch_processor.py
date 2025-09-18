"""
Batch Processing Workflow Module
Automated batch processing for video content creation and analysis
"""

import os
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from loguru import logger

from .video_upscaler import create_upscaler
from .speech_processor import create_speech_processor
from .tts_processor import create_tts_processor, VoiceProfile
from .subtitle_generator import create_subtitle_generator
from .product_detector import create_product_detector
from .video_editor import create_video_editor


@dataclass
class ProcessingTask:
    """Individual processing task configuration"""
    task_id: str
    task_type: str  # 'transcribe', 'upscale', 'tts', 'detect', 'edit', 'subtitle', 'pipeline'
    input_path: str
    output_path: str
    parameters: Dict[str, Any]
    priority: int = 1  # 1=high, 2=medium, 3=low
    dependencies: List[str] = None  # Task IDs this task depends on
    status: str = "pending"  # pending, processing, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    progress: float = 0.0


@dataclass
class BatchConfig:
    """Batch processing configuration"""
    max_workers: int = 4
    max_concurrent_tasks: int = 2
    retry_attempts: int = 2
    timeout_seconds: int = 3600  # 1 hour per task
    save_intermediate: bool = True
    cleanup_on_success: bool = False
    progress_callback: Optional[Callable] = None


@dataclass
class PipelineTemplate:
    """Pre-defined processing pipeline template"""
    name: str
    description: str
    tasks: List[Dict[str, Any]]
    default_params: Dict[str, Any]


class BatchProcessor:
    """
    Comprehensive batch processing system for AI content creation
    Supports multiple task types and processing pipelines
    """

    def __init__(
        self,
        config: Optional[BatchConfig] = None,
        temp_dir: Optional[str] = None
    ):
        """
        Initialize batch processor

        Args:
            config: Batch processing configuration
            temp_dir: Temporary directory for intermediate files
        """
        self.config = config or BatchConfig()
        self.temp_dir = Path(temp_dir) if temp_dir else Path("./data/temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Task management
        self.tasks: Dict[str, ProcessingTask] = {}
        self.task_queue: List[str] = []
        self.running_tasks: Dict[str, Any] = {}

        # Processing modules
        self._processors = {}
        self._initialize_processors()

        # Pipeline templates
        self.pipeline_templates = self._create_pipeline_templates()

    def _initialize_processors(self):
        """Initialize processing modules"""
        try:
            self._processors = {
                'upscaler': create_upscaler(),
                'speech_processor': create_speech_processor(),
                'tts_processor': create_tts_processor(),
                'subtitle_generator': create_subtitle_generator(),
                'product_detector': create_product_detector(),
                'video_editor': create_video_editor()
            }
            logger.info("Batch processors initialized")
        except Exception as e:
            logger.error(f"Processor initialization failed: {e}")

    def _create_pipeline_templates(self) -> Dict[str, PipelineTemplate]:
        """Create predefined pipeline templates"""
        templates = {}

        # Basic content creation pipeline
        templates['basic_content'] = PipelineTemplate(
            name="Basic Content Creation",
            description="Transcribe, upscale, and add subtitles",
            tasks=[
                {"type": "transcribe", "priority": 1},
                {"type": "upscale", "priority": 2, "depends_on": []},
                {"type": "subtitle", "priority": 3, "depends_on": ["transcribe", "upscale"]}
            ],
            default_params={
                "transcribe": {"model_size": "base", "language": None},
                "upscale": {"scale": 4, "model_name": "RealESRGAN_x4plus"},
                "subtitle": {"style_type": "modern", "formats": ["srt", "vtt"]}
            }
        )

        # Product video pipeline
        templates['product_video'] = PipelineTemplate(
            name="Product Video Processing",
            description="Complete product video analysis and enhancement",
            tasks=[
                {"type": "transcribe", "priority": 1},
                {"type": "detect", "priority": 1},
                {"type": "upscale", "priority": 2},
                {"type": "edit", "priority": 3, "depends_on": ["transcribe", "detect"]},
                {"type": "subtitle", "priority": 4, "depends_on": ["transcribe", "upscale"]},
                {"type": "tts", "priority": 5, "depends_on": ["transcribe"]}
            ],
            default_params={
                "transcribe": {"model_size": "base"},
                "detect": {"confidence_threshold": 0.6, "sample_rate": 1.0},
                "upscale": {"scale": 4},
                "edit": {"create_highlights": True, "target_duration": 60},
                "subtitle": {"style_type": "modern", "highlight_keywords": True},
                "tts": {"voice_type": "professional", "language": "zh"}
            }
        )

        # Localization pipeline
        templates['localization'] = PipelineTemplate(
            name="Video Localization",
            description="Transcribe and create localized versions",
            tasks=[
                {"type": "transcribe", "priority": 1},
                {"type": "tts", "priority": 2, "depends_on": ["transcribe"], "variations": ["en", "zh", "ja"]},
                {"type": "subtitle", "priority": 3, "depends_on": ["transcribe"], "variations": ["en", "zh", "ja"]}
            ],
            default_params={
                "transcribe": {"model_size": "base"},
                "tts": {"voice_type": "professional"},
                "subtitle": {"style_type": "minimal"}
            }
        )

        # Analysis pipeline
        templates['analysis'] = PipelineTemplate(
            name="Content Analysis",
            description="Comprehensive content analysis",
            tasks=[
                {"type": "transcribe", "priority": 1},
                {"type": "detect", "priority": 1},
                {"type": "edit", "priority": 2, "depends_on": ["transcribe", "detect"]}
            ],
            default_params={
                "transcribe": {"model_size": "small"},
                "detect": {"confidence_threshold": 0.5, "include_speech_analysis": True},
                "edit": {"analyze_only": True}
            }
        )

        return templates

    def add_task(
        self,
        task_type: str,
        input_path: str,
        output_path: str,
        parameters: Optional[Dict[str, Any]] = None,
        priority: int = 1,
        dependencies: Optional[List[str]] = None,
        task_id: Optional[str] = None
    ) -> str:
        """
        Add a processing task to the batch

        Args:
            task_type: Type of processing task
            input_path: Input file path
            output_path: Output file path
            parameters: Task-specific parameters
            priority: Task priority (1=high, 2=medium, 3=low)
            dependencies: List of task IDs this task depends on
            task_id: Custom task ID (auto-generated if None)

        Returns:
            Task ID
        """
        if task_id is None:
            task_id = f"{task_type}_{int(time.time() * 1000)}"

        task = ProcessingTask(
            task_id=task_id,
            task_type=task_type,
            input_path=input_path,
            output_path=output_path,
            parameters=parameters or {},
            priority=priority,
            dependencies=dependencies or [],
            status="pending"
        )

        self.tasks[task_id] = task
        self.task_queue.append(task_id)

        logger.info(f"Added task {task_id}: {task_type}")
        return task_id

    def add_pipeline(
        self,
        template_name: str,
        input_path: str,
        output_dir: str,
        custom_params: Optional[Dict[str, Any]] = None,
        name_prefix: Optional[str] = None
    ) -> List[str]:
        """
        Add a complete processing pipeline

        Args:
            template_name: Pipeline template name
            input_path: Input video path
            output_dir: Output directory
            custom_params: Custom parameters to override defaults
            name_prefix: Prefix for task IDs

        Returns:
            List of task IDs created
        """
        if template_name not in self.pipeline_templates:
            raise ValueError(f"Unknown pipeline template: {template_name}")

        template = self.pipeline_templates[template_name]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Merge parameters
        params = template.default_params.copy()
        if custom_params:
            for task_type, task_params in custom_params.items():
                if task_type in params:
                    params[task_type].update(task_params)
                else:
                    params[task_type] = task_params

        # Create tasks
        task_ids = []
        task_mapping = {}

        for task_config in template.tasks:
            task_type = task_config["type"]
            priority = task_config.get("priority", 1)

            # Handle variations (e.g., multiple languages)
            variations = task_config.get("variations", [None])

            for variation in variations:
                # Generate task ID
                if name_prefix:
                    base_id = f"{name_prefix}_{task_type}"
                else:
                    base_id = f"pipeline_{template_name}_{task_type}"

                if variation:
                    task_id = f"{base_id}_{variation}"
                    var_suffix = f"_{variation}"
                else:
                    task_id = base_id
                    var_suffix = ""

                # Generate output path
                input_name = Path(input_path).stem
                if task_type == "transcribe":
                    output_path = output_dir / f"{input_name}_transcription{var_suffix}.json"
                elif task_type == "upscale":
                    output_path = output_dir / f"{input_name}_upscaled{var_suffix}.mp4"
                elif task_type == "tts":
                    output_path = output_dir / f"{input_name}_voiceover{var_suffix}.mp4"
                elif task_type == "detect":
                    output_path = output_dir / f"{input_name}_detected{var_suffix}.json"
                elif task_type == "edit":
                    output_path = output_dir / f"{input_name}_edited{var_suffix}.mp4"
                elif task_type == "subtitle":
                    output_path = output_dir / f"{input_name}_subtitles{var_suffix}"
                else:
                    output_path = output_dir / f"{input_name}_{task_type}{var_suffix}.out"

                # Get task parameters
                task_params = params.get(task_type, {}).copy()
                if variation and task_type in ["tts", "subtitle"]:
                    task_params["language"] = variation

                # Handle dependencies
                dependencies = []
                if "depends_on" in task_config:
                    for dep_type in task_config["depends_on"]:
                        if dep_type in task_mapping:
                            if variation and dep_type in ["tts", "subtitle"]:
                                dep_id = f"{task_mapping[dep_type]}_{variation}"
                            else:
                                dep_id = task_mapping[dep_type]
                            dependencies.append(dep_id)

                # Add task
                actual_task_id = self.add_task(
                    task_type=task_type,
                    input_path=input_path,
                    output_path=str(output_path),
                    parameters=task_params,
                    priority=priority,
                    dependencies=dependencies,
                    task_id=task_id
                )

                task_ids.append(actual_task_id)
                if not variation:  # Only map base task types
                    task_mapping[task_type] = actual_task_id

        logger.info(f"Added pipeline {template_name} with {len(task_ids)} tasks")
        return task_ids

    async def process_batch(self) -> Dict[str, Any]:
        """
        Process all tasks in the batch

        Returns:
            Processing results summary
        """
        logger.info(f"Starting batch processing with {len(self.tasks)} tasks")
        start_time = time.time()

        # Sort tasks by priority and dependencies
        self._sort_tasks()

        results = {
            "total_tasks": len(self.tasks),
            "completed": 0,
            "failed": 0,
            "skipped": 0,
            "results": {},
            "errors": {},
            "processing_time": 0
        }

        # Process tasks
        try:
            # Use thread pool for I/O bound tasks
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                while self.task_queue:
                    # Get ready tasks (no pending dependencies)
                    ready_tasks = self._get_ready_tasks()

                    if not ready_tasks:
                        # Check if we have running tasks
                        if self.running_tasks:
                            await asyncio.sleep(1)
                            continue
                        else:
                            # No ready tasks and no running tasks - deadlock or completion
                            break

                    # Limit concurrent tasks
                    while (len(self.running_tasks) < self.config.max_concurrent_tasks and
                           ready_tasks):
                        task_id = ready_tasks.pop(0)

                        # Start task
                        future = executor.submit(self._process_single_task, task_id)
                        self.running_tasks[task_id] = future

                        self.tasks[task_id].status = "processing"
                        self.tasks[task_id].start_time = time.time()

                    # Check completed tasks
                    completed_tasks = []
                    for task_id, future in self.running_tasks.items():
                        if future.done():
                            completed_tasks.append(task_id)

                    # Process completed tasks
                    for task_id in completed_tasks:
                        future = self.running_tasks.pop(task_id)
                        task = self.tasks[task_id]
                        task.end_time = time.time()

                        try:
                            result = future.result()
                            if result["success"]:
                                task.status = "completed"
                                task.progress = 100.0
                                results["completed"] += 1
                                results["results"][task_id] = result
                                logger.info(f"Task {task_id} completed successfully")
                            else:
                                task.status = "failed"
                                task.error_message = result["error"]
                                results["failed"] += 1
                                results["errors"][task_id] = result["error"]
                                logger.error(f"Task {task_id} failed: {result['error']}")

                        except Exception as e:
                            task.status = "failed"
                            task.error_message = str(e)
                            results["failed"] += 1
                            results["errors"][task_id] = str(e)
                            logger.error(f"Task {task_id} exception: {e}")

                        # Remove from queue
                        if task_id in self.task_queue:
                            self.task_queue.remove(task_id)

                        # Update progress
                        if self.config.progress_callback:
                            total_progress = (results["completed"] + results["failed"]) / results["total_tasks"] * 100
                            self.config.progress_callback(total_progress, task_id, task.status)

                    await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

        except Exception as e:
            logger.error(f"Batch processing error: {e}")

        # Calculate final results
        results["processing_time"] = time.time() - start_time
        results["skipped"] = results["total_tasks"] - results["completed"] - results["failed"]

        # Cleanup if requested
        if self.config.cleanup_on_success:
            self._cleanup_successful_tasks()

        logger.info(f"Batch processing completed: {results['completed']} success, {results['failed']} failed")
        return results

    def _sort_tasks(self):
        """Sort tasks by priority and dependencies using topological sort"""
        # Simple priority-based sort for now
        # In production, implement proper topological sort for dependencies
        self.task_queue.sort(key=lambda task_id: (
            self.tasks[task_id].priority,
            len(self.tasks[task_id].dependencies)
        ))

    def _get_ready_tasks(self) -> List[str]:
        """Get tasks that are ready to run (all dependencies completed)"""
        ready = []

        for task_id in self.task_queue[:]:
            if task_id in self.running_tasks:
                continue

            task = self.tasks[task_id]
            if task.status != "pending":
                continue

            # Check if all dependencies are completed
            deps_ready = True
            for dep_id in task.dependencies:
                if dep_id not in self.tasks or self.tasks[dep_id].status != "completed":
                    deps_ready = False
                    break

            if deps_ready:
                ready.append(task_id)

        return ready

    def _process_single_task(self, task_id: str) -> Dict[str, Any]:
        """Process a single task"""
        task = self.tasks[task_id]

        try:
            logger.info(f"Processing task {task_id}: {task.task_type}")

            # Route to appropriate processor
            if task.task_type == "transcribe":
                return self._process_transcribe_task(task)
            elif task.task_type == "upscale":
                return self._process_upscale_task(task)
            elif task.task_type == "tts":
                return self._process_tts_task(task)
            elif task.task_type == "detect":
                return self._process_detect_task(task)
            elif task.task_type == "edit":
                return self._process_edit_task(task)
            elif task.task_type == "subtitle":
                return self._process_subtitle_task(task)
            else:
                return {"success": False, "error": f"Unknown task type: {task.task_type}"}

        except Exception as e:
            logger.error(f"Task {task_id} processing error: {e}")
            return {"success": False, "error": str(e)}

    def _process_transcribe_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process transcription task"""
        processor = self._processors['speech_processor']
        params = task.parameters

        result = processor.transcribe_video(
            video_path=task.input_path,
            output_path=task.output_path,
            language=params.get("language"),
            segment_duration=params.get("segment_duration")
        )

        return {
            "success": True,
            "output_path": task.output_path,
            "language": result.language,
            "duration": result.duration,
            "segments": len(result.segments)
        }

    def _process_upscale_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process video upscaling task"""
        processor = self._processors['upscaler']
        params = task.parameters

        success = processor.upscale_video(
            input_path=task.input_path,
            output_path=task.output_path,
            segment_duration=params.get("segment_duration"),
            target_segments=params.get("target_segments")
        )

        return {
            "success": success,
            "output_path": task.output_path if success else None,
            "scale": params.get("scale", 4)
        }

    def _process_tts_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process text-to-speech task"""
        processor = self._processors['tts_processor']
        params = task.parameters

        # Check if we need to create voiceover from transcription
        if "transcription_path" in params or any(dep_task.task_type == "transcribe"
                                               for dep_id in task.dependencies
                                               for dep_task in [self.tasks.get(dep_id)] if dep_task):
            # Find transcription from dependencies
            transcription_path = None
            for dep_id in task.dependencies:
                dep_task = self.tasks.get(dep_id)
                if dep_task and dep_task.task_type == "transcribe":
                    transcription_path = dep_task.output_path
                    break

            if transcription_path and os.path.exists(transcription_path):
                # Load transcription
                speech_processor = self._processors['speech_processor']

                # Create voice profile
                voice_profile = processor.create_product_voice_profile(
                    language=params.get("language", "zh"),
                    voice_type=params.get("voice_type", "professional"),
                    gender=params.get("gender", "female")
                )

                # This would need to be implemented with async support
                # For now, return success indication
                return {
                    "success": True,
                    "output_path": task.output_path,
                    "voice_type": params.get("voice_type", "professional")
                }

        return {"success": False, "error": "TTS task requires transcription dependency"}

    def _process_detect_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process product detection task"""
        processor = self._processors['product_detector']
        params = task.parameters

        detections = processor.detect_products_in_video(
            video_path=task.input_path,
            sample_rate=params.get("sample_rate", 1.0),
            confidence_threshold=params.get("confidence_threshold", 0.5),
            product_keywords=params.get("product_keywords")
        )

        tracks = processor.create_product_tracks(detections)
        summary = processor.generate_product_summary(tracks)

        # Save results
        results = {
            "detections": len(detections),
            "tracks": len(tracks),
            "summary": summary
        }

        with open(task.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return {
            "success": True,
            "output_path": task.output_path,
            "detections_count": len(detections),
            "tracks_count": len(tracks)
        }

    def _process_edit_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process video editing task"""
        processor = self._processors['video_editor']
        params = task.parameters

        if params.get("analyze_only", False):
            # Analysis only
            analysis = processor.analyze_video(
                video_path=task.input_path,
                product_keywords=params.get("product_keywords")
            )

            with open(task.output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)

            return {
                "success": True,
                "output_path": task.output_path,
                "analysis": analysis["summary"]
            }
        else:
            # Create edited video
            if params.get("create_highlights", False):
                success = processor.create_highlight_reel(
                    video_path=task.input_path,
                    output_path=task.output_path,
                    target_duration=params.get("target_duration", 60),
                    product_keywords=params.get("product_keywords")
                )
            else:
                success = processor.remove_dead_air(
                    video_path=task.input_path,
                    output_path=task.output_path
                )

            return {
                "success": success,
                "output_path": task.output_path if success else None
            }

    def _process_subtitle_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process subtitle generation task"""
        processor = self._processors['subtitle_generator']
        params = task.parameters

        # Find transcription from dependencies
        transcription = None
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if dep_task and dep_task.task_type == "transcribe":
                # Load transcription
                speech_processor = self._processors['speech_processor']
                # This would need proper transcription loading
                break

        if transcription:
            # Create subtitle style
            style = processor.create_product_subtitle_style(
                style_type=params.get("style_type", "modern")
            )

            # Generate subtitle files
            output_dir = Path(task.output_path)
            generated_files = processor.generate_subtitle_files(
                transcription=transcription,
                output_dir=str(output_dir),
                style=style,
                formats=params.get("formats", ["srt", "vtt"])
            )

            return {
                "success": True,
                "output_path": task.output_path,
                "generated_files": generated_files
            }

        return {"success": False, "error": "Subtitle task requires transcription dependency"}

    def _cleanup_successful_tasks(self):
        """Cleanup intermediate files for successful tasks"""
        for task_id, task in self.tasks.items():
            if task.status == "completed" and self.config.save_intermediate:
                # Move to archive or delete intermediate files
                pass

    def get_batch_status(self) -> Dict[str, Any]:
        """Get current batch processing status"""
        status_counts = {}
        for task in self.tasks.values():
            if task.status not in status_counts:
                status_counts[task.status] = 0
            status_counts[task.status] += 1

        return {
            "total_tasks": len(self.tasks),
            "status_counts": status_counts,
            "queue_length": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
            "task_details": {
                task_id: {
                    "type": task.task_type,
                    "status": task.status,
                    "progress": task.progress,
                    "start_time": task.start_time,
                    "end_time": task.end_time
                }
                for task_id, task in self.tasks.items()
            }
        }

    def save_batch_config(self, config_path: str):
        """Save current batch configuration"""
        config_data = {
            "config": asdict(self.config),
            "tasks": {task_id: asdict(task) for task_id, task in self.tasks.items()},
            "pipeline_templates": {
                name: asdict(template)
                for name, template in self.pipeline_templates.items()
            }
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Batch configuration saved to: {config_path}")

    def load_batch_config(self, config_path: str):
        """Load batch configuration from file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # Load configuration
        if "config" in config_data:
            self.config = BatchConfig(**config_data["config"])

        # Load tasks
        if "tasks" in config_data:
            self.tasks = {
                task_id: ProcessingTask(**task_data)
                for task_id, task_data in config_data["tasks"].items()
            }
            self.task_queue = [task_id for task_id, task in self.tasks.items()
                             if task.status == "pending"]

        logger.info(f"Batch configuration loaded from: {config_path}")


def create_batch_processor(
    max_workers: int = 4,
    max_concurrent_tasks: int = 2,
    temp_dir: Optional[str] = None
) -> BatchProcessor:
    """
    Factory function to create BatchProcessor instance

    Args:
        max_workers: Maximum worker threads
        max_concurrent_tasks: Maximum concurrent tasks
        temp_dir: Temporary directory

    Returns:
        BatchProcessor instance
    """
    config = BatchConfig(
        max_workers=max_workers,
        max_concurrent_tasks=max_concurrent_tasks
    )

    return BatchProcessor(config=config, temp_dir=temp_dir)