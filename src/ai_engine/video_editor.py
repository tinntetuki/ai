"""
Intelligent Video Editing Module
Automated video editing based on speech content and visual analysis
"""

import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import numpy as np
from loguru import logger

import moviepy.editor as mp
import cv2
from scipy.signal import find_peaks
import librosa

from .speech_processor import create_speech_processor, TranscriptionResult, TranscriptionSegment


@dataclass
class EditPoint:
    """Single edit point with timing and reasoning"""
    start: float
    end: float
    edit_type: str  # 'cut', 'highlight', 'speed_up', 'slow_down'
    confidence: float
    reason: str
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class VideoSegment:
    """Video segment with metadata"""
    start: float
    end: float
    content_type: str  # 'speech', 'silence', 'product_demo', 'transition'
    importance_score: float
    text_content: Optional[str] = None
    keywords: Optional[List[str]] = None


class IntelligentVideoEditor:
    """
    Intelligent video editor for content creation
    Optimized for product videos and content repurposing
    """

    def __init__(
        self,
        speech_model_size: str = "base",
        silence_threshold: float = -30.0,  # dB
        min_segment_duration: float = 0.5,  # seconds
        max_silence_duration: float = 2.0   # seconds
    ):
        """
        Initialize intelligent video editor

        Args:
            speech_model_size: Whisper model size for transcription
            silence_threshold: Audio level threshold for silence detection
            min_segment_duration: Minimum segment duration to keep
            max_silence_duration: Maximum silence duration to preserve
        """
        self.speech_model_size = speech_model_size
        self.silence_threshold = silence_threshold
        self.min_segment_duration = min_segment_duration
        self.max_silence_duration = max_silence_duration

        # Initialize speech processor
        self.speech_processor = create_speech_processor(model_size=speech_model_size)

    def analyze_video(
        self,
        video_path: str,
        product_keywords: Optional[List[str]] = None
    ) -> Dict:
        """
        Comprehensive video analysis for intelligent editing

        Args:
            video_path: Path to input video
            product_keywords: List of product-related keywords

        Returns:
            Analysis results dictionary
        """
        try:
            logger.info(f"Analyzing video: {video_path}")

            # Load video
            video = mp.VideoFileClip(video_path)
            duration = video.duration

            # Speech analysis
            logger.info("Analyzing speech content...")
            transcription = self.speech_processor.transcribe_video(video_path)

            # Audio analysis
            logger.info("Analyzing audio patterns...")
            audio_analysis = self._analyze_audio(video_path)

            # Visual analysis
            logger.info("Analyzing visual content...")
            visual_analysis = self._analyze_visual_content(video_path, sample_rate=1.0)

            # Content segmentation
            logger.info("Segmenting content...")
            segments = self._segment_content(transcription, audio_analysis, visual_analysis)

            # Product keyword analysis
            product_segments = []
            if product_keywords:
                logger.info("Analyzing product keywords...")
                product_segments = self.speech_processor.detect_product_mentions(
                    transcription, product_keywords
                )

            # Generate edit suggestions
            logger.info("Generating edit suggestions...")
            edit_points = self._generate_edit_suggestions(
                segments, transcription, audio_analysis, product_segments
            )

            video.close()

            analysis_result = {
                "duration": duration,
                "transcription": transcription,
                "audio_analysis": audio_analysis,
                "visual_analysis": visual_analysis,
                "segments": segments,
                "product_segments": product_segments,
                "edit_points": edit_points,
                "summary": {
                    "total_segments": len(segments),
                    "speech_segments": len([s for s in segments if s.content_type == 'speech']),
                    "silence_segments": len([s for s in segments if s.content_type == 'silence']),
                    "product_mentions": len(product_segments),
                    "edit_suggestions": len(edit_points)
                }
            }

            logger.info("Video analysis completed")
            return analysis_result

        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            raise

    def create_highlight_reel(
        self,
        video_path: str,
        output_path: str,
        target_duration: float = 60.0,
        product_keywords: Optional[List[str]] = None,
        include_intro: bool = True,
        fade_duration: float = 0.5
    ) -> bool:
        """
        Create a highlight reel from the original video

        Args:
            video_path: Path to input video
            output_path: Path to output video
            target_duration: Target duration for highlight reel
            product_keywords: Product keywords to prioritize
            include_intro: Include video intro/opening
            fade_duration: Duration of fade transitions

        Returns:
            True if successful
        """
        try:
            logger.info(f"Creating highlight reel: {target_duration}s")

            # Analyze video
            analysis = self.analyze_video(video_path, product_keywords)

            # Select best segments
            selected_segments = self._select_highlight_segments(
                analysis["segments"],
                analysis["product_segments"],
                target_duration,
                include_intro
            )

            # Load original video
            video = mp.VideoFileClip(video_path)

            # Create clips from selected segments
            clips = []
            for segment in selected_segments:
                clip = video.subclip(segment.start, segment.end)

                # Add fade transitions
                if fade_duration > 0:
                    clip = clip.fadein(fade_duration).fadeout(fade_duration)

                clips.append(clip)

            # Concatenate clips
            if clips:
                final_video = mp.concatenate_videoclips(clips, method="compose")

                # Export
                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    fps=video.fps
                )

                # Cleanup
                final_video.close()
                for clip in clips:
                    clip.close()

            video.close()

            logger.info(f"Highlight reel created: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Highlight reel creation failed: {e}")
            return False

    def remove_dead_air(
        self,
        video_path: str,
        output_path: str,
        silence_threshold: Optional[float] = None,
        max_silence_gap: float = 1.0,
        preserve_pauses: bool = True
    ) -> bool:
        """
        Remove excessive silence and dead air from video

        Args:
            video_path: Path to input video
            output_path: Path to output video
            silence_threshold: Audio threshold for silence detection
            max_silence_gap: Maximum silence duration to preserve
            preserve_pauses: Keep natural speaking pauses

        Returns:
            True if successful
        """
        try:
            logger.info("Removing dead air from video")

            # Use instance threshold if not specified
            if silence_threshold is None:
                silence_threshold = self.silence_threshold

            # Analyze audio
            audio_analysis = self._analyze_audio(video_path)
            silence_segments = audio_analysis["silence_segments"]

            # Load video
            video = mp.VideoFileClip(video_path)

            # Determine segments to keep
            keep_segments = []
            last_end = 0

            for silence_start, silence_end in silence_segments:
                # Keep content before silence
                if silence_start > last_end:
                    keep_segments.append((last_end, silence_start))

                # Keep short silences if preserving pauses
                silence_duration = silence_end - silence_start
                if preserve_pauses and silence_duration <= max_silence_gap:
                    keep_segments.append((silence_start, silence_end))
                elif preserve_pauses and silence_duration > max_silence_gap:
                    # Keep only a portion of long silence
                    reduced_end = silence_start + max_silence_gap
                    keep_segments.append((silence_start, reduced_end))

                last_end = silence_end

            # Keep remaining content
            if last_end < video.duration:
                keep_segments.append((last_end, video.duration))

            # Create clips from kept segments
            clips = []
            for start, end in keep_segments:
                if end - start >= self.min_segment_duration:
                    clip = video.subclip(start, end)
                    clips.append(clip)

            # Concatenate clips
            if clips:
                final_video = mp.concatenate_videoclips(clips, method="compose")

                # Export
                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    fps=video.fps
                )

                # Cleanup
                final_video.close()
                for clip in clips:
                    clip.close()

            video.close()

            original_duration = video.duration
            final_duration = sum(end - start for start, end in keep_segments)
            time_saved = original_duration - final_duration

            logger.info(f"Dead air removal completed")
            logger.info(f"Original: {original_duration:.1f}s, Final: {final_duration:.1f}s")
            logger.info(f"Time saved: {time_saved:.1f}s ({time_saved/original_duration*100:.1f}%)")

            return True

        except Exception as e:
            logger.error(f"Dead air removal failed: {e}")
            return False

    def create_product_segments(
        self,
        video_path: str,
        output_dir: str,
        product_keywords: List[str],
        segment_padding: float = 2.0,
        min_segment_duration: float = 5.0
    ) -> List[str]:
        """
        Extract segments containing product mentions

        Args:
            video_path: Path to input video
            output_dir: Directory to save product segments
            product_keywords: Product keywords to search for
            segment_padding: Extra time to include before/after mentions
            min_segment_duration: Minimum duration for extracted segments

        Returns:
            List of paths to created segment files
        """
        try:
            logger.info("Extracting product segments")

            # Analyze video for product mentions
            analysis = self.analyze_video(video_path, product_keywords)
            product_segments = analysis["product_segments"]

            if not product_segments:
                logger.warning("No product mentions found")
                return []

            # Load video
            video = mp.VideoFileClip(video_path)

            # Create output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            created_files = []

            for i, (segment, keywords) in enumerate(product_segments):
                # Calculate segment boundaries with padding
                start_time = max(0, segment.start - segment_padding)
                end_time = min(video.duration, segment.end + segment_padding)

                # Skip if too short
                if end_time - start_time < min_segment_duration:
                    continue

                # Create clip
                clip = video.subclip(start_time, end_time)

                # Generate filename
                keyword_str = "_".join(keywords[:2])  # Use first 2 keywords
                filename = f"product_segment_{i+1}_{keyword_str}_{start_time:.1f}s.mp4"
                output_path = output_dir / filename

                # Export
                clip.write_videofile(
                    str(output_path),
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    fps=video.fps,
                    verbose=False,
                    logger=None
                )

                clip.close()
                created_files.append(str(output_path))

                logger.info(f"Created product segment: {filename}")

            video.close()

            logger.info(f"Created {len(created_files)} product segments")
            return created_files

        except Exception as e:
            logger.error(f"Product segment extraction failed: {e}")
            return []

    def _analyze_audio(self, video_path: str) -> Dict:
        """Analyze audio for silence detection and energy patterns"""
        try:
            # Extract audio
            temp_audio = self.speech_processor.extract_audio_from_video(video_path)

            # Load audio with librosa
            y, sr = librosa.load(temp_audio, sr=None)

            # Calculate RMS energy
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

            # Convert to dB
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)

            # Time axis
            times = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr, hop_length=512)

            # Find silence segments
            silence_mask = rms_db < self.silence_threshold
            silence_segments = self._get_silence_segments(times, silence_mask)

            # Energy statistics
            energy_stats = {
                "mean_energy": float(np.mean(rms_db)),
                "std_energy": float(np.std(rms_db)),
                "max_energy": float(np.max(rms_db)),
                "min_energy": float(np.min(rms_db))
            }

            # Cleanup
            os.unlink(temp_audio)

            return {
                "energy_profile": rms_db.tolist(),
                "times": times.tolist(),
                "silence_segments": silence_segments,
                "energy_stats": energy_stats,
                "silence_ratio": len([s for s in silence_segments]) / len(times)
            }

        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return {}

    def _analyze_visual_content(self, video_path: str, sample_rate: float = 1.0) -> Dict:
        """Analyze visual content for scene changes and activity"""
        try:
            video = mp.VideoFileClip(video_path)
            duration = video.duration

            # Sample frames
            sample_times = np.arange(0, duration, sample_rate)
            frame_differences = []

            prev_frame = None
            for t in sample_times:
                frame = video.get_frame(t)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(frame_gray, prev_frame)
                    diff_score = np.mean(diff)
                    frame_differences.append(diff_score)

                prev_frame = frame_gray

            video.close()

            # Find scene changes (peaks in frame differences)
            if frame_differences:
                peaks, _ = find_peaks(frame_differences, height=np.mean(frame_differences) + np.std(frame_differences))
                scene_changes = [sample_times[p] for p in peaks]
            else:
                scene_changes = []

            return {
                "frame_differences": frame_differences,
                "sample_times": sample_times.tolist(),
                "scene_changes": scene_changes,
                "visual_activity": float(np.mean(frame_differences)) if frame_differences else 0.0
            }

        except Exception as e:
            logger.error(f"Visual analysis failed: {e}")
            return {}

    def _segment_content(
        self,
        transcription: TranscriptionResult,
        audio_analysis: Dict,
        visual_analysis: Dict
    ) -> List[VideoSegment]:
        """Segment video content based on multiple analysis factors"""
        segments = []

        # Create segments from speech transcription
        for trans_segment in transcription.segments:
            importance_score = self._calculate_importance_score(
                trans_segment, audio_analysis, visual_analysis
            )

            segment = VideoSegment(
                start=trans_segment.start,
                end=trans_segment.end,
                content_type='speech',
                importance_score=importance_score,
                text_content=trans_segment.text
            )
            segments.append(segment)

        # Add silence segments
        silence_segments = audio_analysis.get("silence_segments", [])
        for start, end in silence_segments:
            if end - start >= self.min_segment_duration:
                segment = VideoSegment(
                    start=start,
                    end=end,
                    content_type='silence',
                    importance_score=0.1  # Low importance
                )
                segments.append(segment)

        # Sort by start time
        segments.sort(key=lambda x: x.start)

        return segments

    def _calculate_importance_score(
        self,
        trans_segment: TranscriptionSegment,
        audio_analysis: Dict,
        visual_analysis: Dict
    ) -> float:
        """Calculate importance score for a transcription segment"""
        score = 0.5  # Base score

        # Text-based scoring
        text = trans_segment.text.lower()

        # Product/sales keywords increase importance
        sales_keywords = ['产品', '购买', '优惠', '特价', '推荐', '功能', '质量', '效果']
        for keyword in sales_keywords:
            if keyword in text:
                score += 0.1

        # Confidence-based scoring
        if trans_segment.confidence:
            score += trans_segment.confidence * 0.2

        # Duration-based scoring (moderate length preferred)
        duration = trans_segment.end - trans_segment.start
        if 3 <= duration <= 15:  # Sweet spot for product mentions
            score += 0.1
        elif duration > 30:  # Very long segments might be less focused
            score -= 0.1

        # Audio energy scoring
        if "energy_profile" in audio_analysis:
            # Find corresponding audio energy
            start_idx = int(trans_segment.start * 2)  # Approximate frame index
            end_idx = int(trans_segment.end * 2)

            energy_profile = audio_analysis["energy_profile"]
            if start_idx < len(energy_profile) and end_idx <= len(energy_profile):
                segment_energy = np.mean(energy_profile[start_idx:end_idx])
                mean_energy = audio_analysis["energy_stats"]["mean_energy"]

                if segment_energy > mean_energy:
                    score += 0.1  # Higher energy = more engagement

        return min(1.0, max(0.0, score))  # Clamp to [0, 1]

    def _generate_edit_suggestions(
        self,
        segments: List[VideoSegment],
        transcription: TranscriptionResult,
        audio_analysis: Dict,
        product_segments: List
    ) -> List[EditPoint]:
        """Generate intelligent edit suggestions"""
        edit_points = []

        # Suggest removing long silence segments
        for segment in segments:
            if (segment.content_type == 'silence' and
                segment.end - segment.start > self.max_silence_duration):

                edit_points.append(EditPoint(
                    start=segment.start,
                    end=segment.end,
                    edit_type='cut',
                    confidence=0.8,
                    reason=f"Long silence ({segment.end - segment.start:.1f}s)",
                    priority=2
                ))

        # Suggest highlighting product segments
        for segment, keywords in product_segments:
            edit_points.append(EditPoint(
                start=segment.start - 1,  # Include context
                end=segment.end + 1,
                edit_type='highlight',
                confidence=0.9,
                reason=f"Product mention: {', '.join(keywords)}",
                priority=1
            ))

        # Suggest cutting low-importance segments
        for segment in segments:
            if (segment.content_type == 'speech' and
                segment.importance_score < 0.3 and
                segment.end - segment.start > 10):  # Long, low-importance segments

                edit_points.append(EditPoint(
                    start=segment.start,
                    end=segment.end,
                    edit_type='cut',
                    confidence=0.6,
                    reason=f"Low importance content (score: {segment.importance_score:.2f})",
                    priority=3
                ))

        # Sort by priority and confidence
        edit_points.sort(key=lambda x: (x.priority, -x.confidence))

        return edit_points

    def _select_highlight_segments(
        self,
        segments: List[VideoSegment],
        product_segments: List,
        target_duration: float,
        include_intro: bool
    ) -> List[VideoSegment]:
        """Select best segments for highlight reel"""
        selected = []
        total_duration = 0

        # Include intro if requested (first 5 seconds)
        if include_intro:
            intro_segments = [s for s in segments if s.start < 5 and s.content_type == 'speech']
            if intro_segments:
                intro = intro_segments[0]
                selected.append(intro)
                total_duration += intro.end - intro.start

        # Prioritize product segments
        product_segment_objects = [seg for seg, _ in product_segments]
        for segment in product_segment_objects:
            if total_duration >= target_duration:
                break

            # Find corresponding VideoSegment
            matching_segments = [
                s for s in segments
                if (s.start <= segment.start <= s.end or s.start <= segment.end <= s.end)
                and s not in selected
            ]

            for seg in matching_segments:
                if total_duration + (seg.end - seg.start) <= target_duration:
                    selected.append(seg)
                    total_duration += seg.end - seg.start

        # Fill remaining time with high-importance segments
        remaining_segments = [s for s in segments if s not in selected and s.content_type == 'speech']
        remaining_segments.sort(key=lambda x: x.importance_score, reverse=True)

        for segment in remaining_segments:
            if total_duration >= target_duration:
                break

            segment_duration = segment.end - segment.start
            if total_duration + segment_duration <= target_duration:
                selected.append(segment)
                total_duration += segment_duration

        # Sort by start time
        selected.sort(key=lambda x: x.start)

        return selected

    def _get_silence_segments(self, times: np.ndarray, silence_mask: np.ndarray) -> List[Tuple[float, float]]:
        """Extract silence segments from silence mask"""
        segments = []
        in_silence = False
        start_time = None

        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                # Start of silence
                start_time = times[i]
                in_silence = True
            elif not is_silent and in_silence:
                # End of silence
                if start_time is not None:
                    segments.append((start_time, times[i]))
                in_silence = False

        # Handle case where video ends in silence
        if in_silence and start_time is not None:
            segments.append((start_time, times[-1]))

        return segments


def create_video_editor(
    speech_model_size: str = "base",
    silence_threshold: float = -30.0
) -> IntelligentVideoEditor:
    """
    Factory function to create IntelligentVideoEditor instance

    Args:
        speech_model_size: Whisper model size for transcription
        silence_threshold: Audio threshold for silence detection

    Returns:
        IntelligentVideoEditor instance
    """
    return IntelligentVideoEditor(
        speech_model_size=speech_model_size,
        silence_threshold=silence_threshold,
        min_segment_duration=0.5,
        max_silence_duration=2.0
    )