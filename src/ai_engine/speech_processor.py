"""
Speech-to-Text Processing Module using OpenAI Whisper
Optimized for video content analysis and subtitle generation
"""

import os
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from loguru import logger

import whisper
import moviepy.editor as mp
from pydub import AudioSegment
import librosa
import soundfile as sf


@dataclass
class TranscriptionSegment:
    """Single transcription segment with timing and confidence"""
    start: float
    end: float
    text: str
    confidence: Optional[float] = None
    language: Optional[str] = None


@dataclass
class TranscriptionResult:
    """Complete transcription result"""
    segments: List[TranscriptionSegment]
    language: str
    duration: float
    full_text: str


class SpeechProcessor:
    """
    Speech-to-text processor using Whisper
    Optimized for content creation and video analysis
    """

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        compute_type: str = "float32"
    ):
        """
        Initialize speech processor

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Computing device (None for auto-detection)
            compute_type: Computation precision (float32, float16, int8)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None

        # Load model
        self._load_model()

    def _load_model(self):
        """Load Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info("Whisper model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def extract_audio_from_video(
        self,
        video_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        sample_rate: int = 16000
    ) -> str:
        """
        Extract audio from video file

        Args:
            video_path: Path to video file
            start_time: Start time in seconds (None for beginning)
            end_time: End time in seconds (None for full duration)
            sample_rate: Audio sample rate

        Returns:
            Path to extracted audio file
        """
        try:
            logger.info(f"Extracting audio from: {video_path}")

            # Load video
            video = mp.VideoFileClip(video_path)

            # Extract audio segment
            if start_time is not None or end_time is not None:
                start = start_time or 0
                end = end_time or video.duration
                audio = video.subclip(start, end).audio
            else:
                audio = video.audio

            # Create temporary audio file
            temp_audio = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            )
            temp_audio_path = temp_audio.name
            temp_audio.close()

            # Export audio
            audio.write_audiofile(
                temp_audio_path,
                fps=sample_rate,
                verbose=False,
                logger=None
            )

            # Cleanup
            video.close()
            audio.close()

            logger.info(f"Audio extracted to: {temp_audio_path}")
            return temp_audio_path

        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            raise

    def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        word_timestamps: bool = True
    ) -> TranscriptionResult:
        """
        Transcribe audio to text using Whisper

        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detection)
            task: Task type ('transcribe' or 'translate')
            word_timestamps: Include word-level timestamps

        Returns:
            TranscriptionResult with segments and metadata
        """
        try:
            logger.info(f"Transcribing audio: {audio_path}")

            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                word_timestamps=word_timestamps,
                verbose=False
            )

            # Parse segments
            segments = []
            for segment in result["segments"]:
                segments.append(TranscriptionSegment(
                    start=segment["start"],
                    end=segment["end"],
                    text=segment["text"].strip(),
                    confidence=segment.get("avg_logprob"),
                    language=result["language"]
                ))

            # Get audio duration
            duration = librosa.get_duration(filename=audio_path)

            transcription_result = TranscriptionResult(
                segments=segments,
                language=result["language"],
                duration=duration,
                full_text=result["text"]
            )

            logger.info(f"Transcription completed. Language: {result['language']}")
            return transcription_result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def transcribe_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        language: Optional[str] = None,
        segment_duration: Optional[float] = None
    ) -> TranscriptionResult:
        """
        Transcribe video directly with automatic audio extraction

        Args:
            video_path: Path to video file
            output_path: Path to save transcription JSON (optional)
            language: Language code (None for auto-detection)
            segment_duration: Process only specified duration

        Returns:
            TranscriptionResult
        """
        temp_audio_path = None
        try:
            # Extract audio
            temp_audio_path = self.extract_audio_from_video(
                video_path,
                start_time=0 if segment_duration else None,
                end_time=segment_duration
            )

            # Transcribe
            result = self.transcribe_audio(
                temp_audio_path,
                language=language,
                word_timestamps=True
            )

            # Save to file if requested
            if output_path:
                self.save_transcription(result, output_path)

            return result

        finally:
            # Cleanup temporary audio file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp audio: {e}")

    def save_transcription(
        self,
        result: TranscriptionResult,
        output_path: str,
        format: str = "json"
    ):
        """
        Save transcription result to file

        Args:
            result: TranscriptionResult to save
            output_path: Output file path
            format: Output format ('json', 'srt', 'vtt', 'txt')
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == "json":
                self._save_json(result, output_path)
            elif format == "srt":
                self._save_srt(result, output_path)
            elif format == "vtt":
                self._save_vtt(result, output_path)
            elif format == "txt":
                self._save_txt(result, output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Transcription saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save transcription: {e}")
            raise

    def _save_json(self, result: TranscriptionResult, path: Path):
        """Save as JSON format"""
        data = {
            "language": result.language,
            "duration": result.duration,
            "full_text": result.full_text,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "confidence": seg.confidence
                }
                for seg in result.segments
            ]
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_srt(self, result: TranscriptionResult, path: Path):
        """Save as SRT subtitle format"""
        with open(path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result.segments, 1):
                start_time = self._seconds_to_srt_time(segment.start)
                end_time = self._seconds_to_srt_time(segment.end)

                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment.text}\n\n")

    def _save_vtt(self, result: TranscriptionResult, path: Path):
        """Save as WebVTT format"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")

            for segment in result.segments:
                start_time = self._seconds_to_vtt_time(segment.start)
                end_time = self._seconds_to_vtt_time(segment.end)

                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment.text}\n\n")

    def _save_txt(self, result: TranscriptionResult, path: Path):
        """Save as plain text"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(result.full_text)

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def _seconds_to_vtt_time(self, seconds: float) -> str:
        """Convert seconds to WebVTT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60

        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    def get_speech_segments(
        self,
        result: TranscriptionResult,
        min_duration: float = 1.0,
        confidence_threshold: float = -1.0
    ) -> List[TranscriptionSegment]:
        """
        Filter speech segments by duration and confidence

        Args:
            result: TranscriptionResult to filter
            min_duration: Minimum segment duration
            confidence_threshold: Minimum confidence score

        Returns:
            Filtered list of segments
        """
        filtered_segments = []

        for segment in result.segments:
            duration = segment.end - segment.start
            confidence = segment.confidence or 0

            if (duration >= min_duration and
                confidence >= confidence_threshold and
                segment.text.strip()):
                filtered_segments.append(segment)

        return filtered_segments

    def detect_product_mentions(
        self,
        result: TranscriptionResult,
        product_keywords: List[str]
    ) -> List[Tuple[TranscriptionSegment, List[str]]]:
        """
        Detect product mentions in transcription

        Args:
            result: TranscriptionResult to analyze
            product_keywords: List of product-related keywords

        Returns:
            List of (segment, matched_keywords) tuples
        """
        product_segments = []

        for segment in result.segments:
            text_lower = segment.text.lower()
            matched_keywords = [
                keyword for keyword in product_keywords
                if keyword.lower() in text_lower
            ]

            if matched_keywords:
                product_segments.append((segment, matched_keywords))

        return product_segments


def create_speech_processor(
    model_size: str = "base",
    device: Optional[str] = None
) -> SpeechProcessor:
    """
    Factory function to create SpeechProcessor instance

    Args:
        model_size: Whisper model size (tiny, base, small, medium, large)
        device: Computing device (None for auto-detection)

    Returns:
        SpeechProcessor instance
    """
    return SpeechProcessor(
        model_size=model_size,
        device=device,
        compute_type="float32"  # Stable for Mac M1
    )