"""
Video Super-Resolution Module using Real-ESRGAN
Optimized for Mac M1 Pro and short-form content videos
"""

import os
import tempfile
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from loguru import logger
import moviepy.editor as mp
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


class VideoUpscaler:
    """
    Video super-resolution processor using Real-ESRGAN
    Optimized for 5-minute or shorter videos on Mac M1 Pro
    """

    def __init__(
        self,
        model_name: str = 'RealESRGAN_x4plus',
        scale: int = 4,
        tile_size: int = 512,
        gpu_id: Optional[int] = None
    ):
        """
        Initialize video upscaler

        Args:
            model_name: Real-ESRGAN model name
            scale: Upscaling factor (2, 4, or 8)
            tile_size: Processing tile size for memory efficiency
            gpu_id: GPU device ID (None for CPU/MPS on Mac)
        """
        self.model_name = model_name
        self.scale = scale
        self.tile_size = tile_size
        self.gpu_id = gpu_id
        self.upsampler = None

        # Initialize model
        self._load_model()

    def _load_model(self):
        """Load Real-ESRGAN model"""
        try:
            # Define model architecture
            if 'x4plus' in self.model_name:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                              num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
            else:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                              num_block=23, num_grow_ch=32, scale=2)
                netscale = 2

            # Initialize upsampler
            self.upsampler = RealESRGANer(
                scale=netscale,
                model_path=None,  # Will download automatically
                model=model,
                tile=self.tile_size,
                tile_pad=10,
                pre_pad=0,
                half=False,  # Set to True if using CUDA
                gpu_id=self.gpu_id
            )

            logger.info(f"Real-ESRGAN model {self.model_name} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Real-ESRGAN model: {e}")
            raise

    def upscale_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Upscale a single frame

        Args:
            frame: Input frame as numpy array (BGR format)

        Returns:
            Upscaled frame
        """
        try:
            # Real-ESRGAN expects RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Upscale
            output, _ = self.upsampler.enhance(frame_rgb, outscale=self.scale)

            # Convert back to BGR
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            return output_bgr

        except Exception as e:
            logger.error(f"Frame upscaling failed: {e}")
            # Return original frame if upscaling fails
            return frame

    def upscale_video(
        self,
        input_path: str,
        output_path: str,
        segment_duration: Optional[float] = None,
        target_segments: Optional[List[Tuple[float, float]]] = None,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Upscale video with optional segmentation

        Args:
            input_path: Path to input video
            output_path: Path to output video
            segment_duration: Duration to process (None for full video)
            target_segments: List of (start, end) time segments to upscale
            progress_callback: Callback function for progress updates

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting video upscaling: {input_path}")

            # Load video
            video = mp.VideoFileClip(input_path)

            # Determine segments to process
            if target_segments:
                segments_to_process = target_segments
            elif segment_duration:
                segments_to_process = [(0, min(segment_duration, video.duration))]
            else:
                segments_to_process = [(0, video.duration)]

            processed_clips = []

            for start_time, end_time in segments_to_process:
                logger.info(f"Processing segment: {start_time:.2f}s - {end_time:.2f}s")

                # Extract segment
                segment = video.subclip(start_time, end_time)

                # Process frames
                def process_frame(get_frame, t):
                    frame = get_frame(t)
                    # Convert to uint8 if needed
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)

                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Upscale
                    upscaled_bgr = self.upscale_frame(frame_bgr)

                    # Convert back to RGB for MoviePy
                    upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)

                    # Normalize to 0-1 range
                    return upscaled_rgb.astype(np.float32) / 255.0

                # Apply processing
                processed_segment = segment.fl(process_frame, apply_to=['mask'])
                processed_clips.append(processed_segment)

                if progress_callback:
                    progress = (end_time - start_time) / video.duration * 100
                    progress_callback(progress)

            # Combine processed segments with original
            if len(processed_clips) == 1 and segments_to_process[0] == (0, video.duration):
                # Full video processed
                final_video = processed_clips[0]
            else:
                # Combine with original segments
                final_clips = []
                last_end = 0

                for i, (start_time, end_time) in enumerate(segments_to_process):
                    # Add original segment before processed one
                    if start_time > last_end:
                        final_clips.append(video.subclip(last_end, start_time))

                    # Add processed segment
                    final_clips.append(processed_clips[i])
                    last_end = end_time

                # Add remaining original content
                if last_end < video.duration:
                    final_clips.append(video.subclip(last_end, video.duration))

                final_video = mp.concatenate_videoclips(final_clips)

            # Export with optimized settings for Mac M1
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                fps=video.fps,
                preset='medium',  # Good balance for M1 Pro
                threads=8  # Optimal for M1 Pro
            )

            # Cleanup
            video.close()
            final_video.close()

            logger.info(f"Video upscaling completed: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Video upscaling failed: {e}")
            return False

    def get_smart_segments(
        self,
        video_path: str,
        max_segments: int = 3
    ) -> List[Tuple[float, float]]:
        """
        Automatically identify segments that would benefit most from upscaling
        (e.g., product close-ups, text overlays, face shots)

        Args:
            video_path: Path to input video
            max_segments: Maximum number of segments to return

        Returns:
            List of (start_time, end_time) tuples
        """
        try:
            video = mp.VideoFileClip(video_path)
            duration = video.duration

            # For now, return evenly spaced segments
            # TODO: Add CV-based scene detection for product/face identification
            segment_length = min(30, duration / max_segments)  # 30 seconds max per segment

            segments = []
            for i in range(max_segments):
                start = i * (duration / max_segments)
                end = min(start + segment_length, duration)
                if end > start:
                    segments.append((start, end))

            video.close()
            return segments

        except Exception as e:
            logger.error(f"Smart segment detection failed: {e}")
            # Fallback: return first 30 seconds
            return [(0, min(30, duration))]


def create_upscaler(
    scale: int = 4,
    model_name: str = 'RealESRGAN_x4plus'
) -> VideoUpscaler:
    """
    Factory function to create VideoUpscaler instance

    Args:
        scale: Upscaling factor
        model_name: Real-ESRGAN model name

    Returns:
        VideoUpscaler instance
    """
    return VideoUpscaler(
        model_name=model_name,
        scale=scale,
        tile_size=512,  # Optimized for M1 Pro memory
        gpu_id=None  # Use MPS on Mac
    )