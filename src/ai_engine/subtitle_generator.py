"""
Subtitle Generation and Styling Module
Automated subtitle creation with customizable styles for video content
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import colorsys
from loguru import logger

import moviepy.editor as mp
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .speech_processor import TranscriptionResult, TranscriptionSegment


@dataclass
class SubtitleStyle:
    """Subtitle styling configuration"""
    # Text properties
    font_family: str = "Arial"
    font_size: int = 24
    font_weight: str = "bold"  # "normal", "bold"
    font_color: str = "#FFFFFF"  # White

    # Background/Border properties
    background_color: Optional[str] = "#000000"  # Black, None for transparent
    background_opacity: float = 0.7
    border_color: Optional[str] = "#000000"  # Black outline
    border_width: int = 2

    # Positioning
    position: str = "bottom"  # "bottom", "top", "center", "custom"
    horizontal_align: str = "center"  # "left", "center", "right"
    vertical_margin: int = 50  # pixels from edge
    horizontal_margin: int = 20  # pixels from edge

    # Animation and effects
    fade_in_duration: float = 0.3
    fade_out_duration: float = 0.3
    typewriter_effect: bool = False
    highlight_keywords: bool = False
    keyword_color: str = "#FFD700"  # Gold

    # Layout
    max_width_percent: float = 0.8  # Max width as percentage of video width
    line_spacing: float = 1.2
    max_lines: int = 2

    # Product video specific
    brand_color: Optional[str] = None
    call_to_action_style: bool = False  # Special styling for CTA text


@dataclass
class SubtitleSegment:
    """Individual subtitle segment"""
    start_time: float
    end_time: float
    text: str
    style: SubtitleStyle
    keywords: Optional[List[str]] = None
    is_cta: bool = False  # Call-to-action segment
    confidence: Optional[float] = None


class SubtitleGenerator:
    """
    Advanced subtitle generator with styling and animation support
    Optimized for product videos and content creation
    """

    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize subtitle generator

        Args:
            temp_dir: Temporary directory for processing files
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "ai_subtitle_generator"
        self.temp_dir.mkdir(exist_ok=True)

        # Font cache
        self._font_cache = {}

    def create_styled_subtitles(
        self,
        video_path: str,
        transcription: TranscriptionResult,
        output_path: str,
        style: Optional[SubtitleStyle] = None,
        product_keywords: Optional[List[str]] = None,
        auto_style_optimization: bool = True
    ) -> bool:
        """
        Create styled subtitles and embed them in video

        Args:
            video_path: Path to input video
            transcription: Transcription result with timing
            output_path: Path to output video with subtitles
            style: Subtitle style configuration
            product_keywords: Keywords to highlight
            auto_style_optimization: Automatically optimize styling

        Returns:
            True if successful
        """
        try:
            logger.info("Creating styled subtitles")

            # Use default style if none provided
            if style is None:
                style = self.create_product_subtitle_style()

            # Load video
            video = mp.VideoFileClip(video_path)

            # Optimize style for video if requested
            if auto_style_optimization:
                style = self._optimize_style_for_video(video, style)

            # Create subtitle segments
            subtitle_segments = self._create_subtitle_segments(
                transcription, style, product_keywords
            )

            # Generate subtitle clips
            subtitle_clips = []
            for segment in subtitle_segments:
                clip = self._create_subtitle_clip(
                    segment, video.size, video.fps
                )
                if clip:
                    subtitle_clips.append(clip)

            # Composite video with subtitles
            if subtitle_clips:
                final_video = mp.CompositeVideoClip([video] + subtitle_clips)
            else:
                final_video = video

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
            video.close()
            for clip in subtitle_clips:
                clip.close()

            logger.info(f"Styled subtitles completed: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Subtitle creation failed: {e}")
            return False

    def generate_subtitle_files(
        self,
        transcription: TranscriptionResult,
        output_dir: str,
        style: Optional[SubtitleStyle] = None,
        product_keywords: Optional[List[str]] = None,
        formats: List[str] = ["srt", "vtt", "ass"]
    ) -> Dict[str, str]:
        """
        Generate subtitle files in multiple formats

        Args:
            transcription: Transcription result
            output_dir: Output directory
            style: Subtitle style
            product_keywords: Keywords to highlight
            formats: List of formats to generate

        Returns:
            Dictionary mapping format to file path
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            if style is None:
                style = self.create_product_subtitle_style()

            # Create subtitle segments
            subtitle_segments = self._create_subtitle_segments(
                transcription, style, product_keywords
            )

            generated_files = {}

            for format_type in formats:
                if format_type == "srt":
                    file_path = output_dir / "subtitles.srt"
                    self._generate_srt(subtitle_segments, str(file_path))
                elif format_type == "vtt":
                    file_path = output_dir / "subtitles.vtt"
                    self._generate_vtt(subtitle_segments, str(file_path))
                elif format_type == "ass":
                    file_path = output_dir / "subtitles.ass"
                    self._generate_ass(subtitle_segments, str(file_path), style)
                elif format_type == "json":
                    file_path = output_dir / "subtitles.json"
                    self._generate_json(subtitle_segments, str(file_path))
                else:
                    logger.warning(f"Unsupported format: {format_type}")
                    continue

                generated_files[format_type] = str(file_path)
                logger.info(f"Generated {format_type.upper()}: {file_path}")

            return generated_files

        except Exception as e:
            logger.error(f"Subtitle file generation failed: {e}")
            return {}

    def create_product_subtitle_style(
        self,
        brand_color: Optional[str] = None,
        style_type: str = "modern"  # "modern", "classic", "minimal", "dynamic"
    ) -> SubtitleStyle:
        """
        Create optimized subtitle style for product videos

        Args:
            brand_color: Brand color hex code
            style_type: Style preset type

        Returns:
            Configured SubtitleStyle
        """
        styles = {
            "modern": {
                "font_family": "Arial Bold",
                "font_size": 28,
                "font_color": "#FFFFFF",
                "background_color": "#000000",
                "background_opacity": 0.8,
                "border_color": "#333333",
                "border_width": 1,
                "position": "bottom",
                "fade_in_duration": 0.2,
                "fade_out_duration": 0.2,
                "highlight_keywords": True,
                "keyword_color": brand_color or "#00D4FF",
                "call_to_action_style": True
            },
            "classic": {
                "font_family": "Times New Roman",
                "font_size": 24,
                "font_color": "#FFFFFF",
                "background_color": None,
                "border_color": "#000000",
                "border_width": 3,
                "position": "bottom",
                "fade_in_duration": 0.5,
                "fade_out_duration": 0.5,
                "highlight_keywords": True,
                "keyword_color": brand_color or "#FFD700"
            },
            "minimal": {
                "font_family": "Helvetica",
                "font_size": 22,
                "font_color": "#FFFFFF",
                "background_color": None,
                "border_color": "#000000",
                "border_width": 1,
                "position": "bottom",
                "fade_in_duration": 0.1,
                "fade_out_duration": 0.1,
                "highlight_keywords": False
            },
            "dynamic": {
                "font_family": "Arial Black",
                "font_size": 32,
                "font_color": "#FFFFFF",
                "background_color": "#FF6B35",
                "background_opacity": 0.9,
                "border_color": "#FFFFFF",
                "border_width": 2,
                "position": "center",
                "fade_in_duration": 0.3,
                "fade_out_duration": 0.3,
                "typewriter_effect": True,
                "highlight_keywords": True,
                "keyword_color": "#FFFF00",
                "call_to_action_style": True
            }
        }

        config = styles.get(style_type, styles["modern"])

        return SubtitleStyle(
            font_family=config.get("font_family", "Arial"),
            font_size=config.get("font_size", 24),
            font_color=config.get("font_color", "#FFFFFF"),
            background_color=config.get("background_color"),
            background_opacity=config.get("background_opacity", 0.7),
            border_color=config.get("border_color"),
            border_width=config.get("border_width", 2),
            position=config.get("position", "bottom"),
            fade_in_duration=config.get("fade_in_duration", 0.3),
            fade_out_duration=config.get("fade_out_duration", 0.3),
            typewriter_effect=config.get("typewriter_effect", False),
            highlight_keywords=config.get("highlight_keywords", False),
            keyword_color=config.get("keyword_color", "#FFD700"),
            brand_color=brand_color,
            call_to_action_style=config.get("call_to_action_style", False)
        )

    def _create_subtitle_segments(
        self,
        transcription: TranscriptionResult,
        style: SubtitleStyle,
        product_keywords: Optional[List[str]] = None
    ) -> List[SubtitleSegment]:
        """Create subtitle segments from transcription"""
        segments = []
        cta_keywords = ["购买", "下单", "优惠", "折扣", "立即", "马上", "抢购", "限时"]

        for trans_segment in transcription.segments:
            # Detect if this is a call-to-action segment
            is_cta = any(keyword in trans_segment.text for keyword in cta_keywords)

            # Find matching product keywords
            matched_keywords = []
            if product_keywords:
                for keyword in product_keywords:
                    if keyword in trans_segment.text:
                        matched_keywords.append(keyword)

            # Create segment with appropriate styling
            segment_style = style
            if is_cta and style.call_to_action_style:
                # Enhance CTA styling
                segment_style = self._create_cta_style(style)

            segment = SubtitleSegment(
                start_time=trans_segment.start,
                end_time=trans_segment.end,
                text=trans_segment.text.strip(),
                style=segment_style,
                keywords=matched_keywords if matched_keywords else None,
                is_cta=is_cta,
                confidence=trans_segment.confidence
            )

            segments.append(segment)

        return segments

    def _create_cta_style(self, base_style: SubtitleStyle) -> SubtitleStyle:
        """Create enhanced style for call-to-action segments"""
        cta_style = SubtitleStyle(**asdict(base_style))

        # Enhance for CTA
        cta_style.font_size = int(base_style.font_size * 1.2)
        cta_style.font_weight = "bold"
        cta_style.background_color = base_style.brand_color or "#FF6B35"
        cta_style.background_opacity = 0.95
        cta_style.border_width = 3
        cta_style.fade_in_duration = 0.4
        cta_style.fade_out_duration = 0.4

        return cta_style

    def _create_subtitle_clip(
        self,
        segment: SubtitleSegment,
        video_size: Tuple[int, int],
        fps: float
    ) -> Optional[mp.VideoClip]:
        """Create MoviePy clip for subtitle segment"""
        try:
            # Calculate positioning
            width, height = video_size
            style = segment.style

            # Determine position
            if style.position == "bottom":
                y_pos = height - style.vertical_margin - style.font_size
            elif style.position == "top":
                y_pos = style.vertical_margin
            elif style.position == "center":
                y_pos = height // 2 - style.font_size // 2
            else:
                y_pos = height - style.vertical_margin - style.font_size

            # Create text clip
            if style.typewriter_effect:
                clip = self._create_typewriter_clip(segment, video_size)
            else:
                clip = self._create_static_text_clip(segment, video_size)

            if clip is None:
                return None

            # Set timing
            clip = clip.set_start(segment.start_time).set_duration(
                segment.end_time - segment.start_time
            )

            # Add fade effects
            if style.fade_in_duration > 0:
                clip = clip.fadein(style.fade_in_duration)
            if style.fade_out_duration > 0:
                clip = clip.fadeout(style.fade_out_duration)

            return clip

        except Exception as e:
            logger.error(f"Failed to create subtitle clip: {e}")
            return None

    def _create_static_text_clip(
        self,
        segment: SubtitleSegment,
        video_size: Tuple[int, int]
    ) -> Optional[mp.VideoClip]:
        """Create static text clip"""
        try:
            style = segment.style
            text = segment.text

            # Highlight keywords if enabled
            if style.highlight_keywords and segment.keywords:
                # This is simplified - in practice, you'd want more sophisticated highlighting
                for keyword in segment.keywords:
                    text = text.replace(keyword, f"⭐{keyword}⭐")

            # Create text clip with MoviePy
            text_clip = mp.TextClip(
                text,
                fontsize=style.font_size,
                color=style.font_color,
                font=style.font_family,
                stroke_color=style.border_color if style.border_color else None,
                stroke_width=style.border_width if style.border_color else 0
            )

            # Set position
            width, height = video_size
            if style.horizontal_align == "center":
                text_clip = text_clip.set_position("center")
            elif style.horizontal_align == "left":
                text_clip = text_clip.set_position((style.horizontal_margin, "center"))
            else:  # right
                text_clip = text_clip.set_position((width - style.horizontal_margin, "center"))

            # Add background if specified
            if style.background_color and style.background_opacity > 0:
                # Create background rectangle
                bg_width = min(int(text_clip.w * 1.2), int(width * style.max_width_percent))
                bg_height = int(text_clip.h * 1.3)

                bg_clip = mp.ColorClip(
                    size=(bg_width, bg_height),
                    color=self._hex_to_rgb(style.background_color)
                ).set_opacity(style.background_opacity)

                # Composite text over background
                text_clip = mp.CompositeVideoClip([
                    bg_clip.set_position("center"),
                    text_clip.set_position("center")
                ])

            return text_clip

        except Exception as e:
            logger.error(f"Failed to create static text clip: {e}")
            return None

    def _create_typewriter_clip(
        self,
        segment: SubtitleSegment,
        video_size: Tuple[int, int]
    ) -> Optional[mp.VideoClip]:
        """Create typewriter effect text clip"""
        try:
            # Simplified typewriter effect
            # In practice, you'd create multiple frames showing progressive text reveal
            text = segment.text
            duration = segment.end_time - segment.start_time

            # For now, just return static text
            # Full implementation would create animated text frames
            return self._create_static_text_clip(segment, video_size)

        except Exception as e:
            logger.error(f"Failed to create typewriter clip: {e}")
            return None

    def _optimize_style_for_video(
        self,
        video: mp.VideoClip,
        style: SubtitleStyle
    ) -> SubtitleStyle:
        """Optimize subtitle style based on video characteristics"""
        # Analyze video for brightness, colors, etc.
        # Adjust font size based on video resolution
        width, height = video.size

        optimized_style = SubtitleStyle(**asdict(style))

        # Scale font size with resolution
        if width >= 1920:  # 1080p or higher
            optimized_style.font_size = max(28, style.font_size)
        elif width >= 1280:  # 720p
            optimized_style.font_size = max(24, style.font_size)
        else:  # Lower resolution
            optimized_style.font_size = max(20, style.font_size)

        # Adjust margins based on resolution
        optimized_style.vertical_margin = max(30, int(height * 0.05))
        optimized_style.horizontal_margin = max(20, int(width * 0.02))

        return optimized_style

    def _generate_srt(self, segments: List[SubtitleSegment], output_path: str):
        """Generate SRT subtitle file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = self._seconds_to_srt_time(segment.start_time)
                end_time = self._seconds_to_srt_time(segment.end_time)

                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment.text}\n\n")

    def _generate_vtt(self, segments: List[SubtitleSegment], output_path: str):
        """Generate WebVTT subtitle file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")

            for segment in segments:
                start_time = self._seconds_to_vtt_time(segment.start_time)
                end_time = self._seconds_to_vtt_time(segment.end_time)

                f.write(f"{start_time} --> {end_time}\n")

                # Add styling if it's a CTA segment
                if segment.is_cta:
                    f.write(f"<c.cta>{segment.text}</c.cta>\n\n")
                else:
                    f.write(f"{segment.text}\n\n")

    def _generate_ass(
        self,
        segments: List[SubtitleSegment],
        output_path: str,
        style: SubtitleStyle
    ):
        """Generate Advanced SubStation Alpha subtitle file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            # ASS header
            f.write("[Script Info]\n")
            f.write("Title: AI Generated Subtitles\n")
            f.write("ScriptType: v4.00+\n\n")

            # Styles section
            f.write("[V4+ Styles]\n")
            f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")

            # Default style
            f.write(f"Style: Default,{style.font_family},{style.font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,0,2,{style.horizontal_margin},{style.horizontal_margin},{style.vertical_margin},1\n")

            # CTA style
            f.write(f"Style: CTA,{style.font_family},{int(style.font_size * 1.2)},&H00FFFFFF,&H000000FF,&H00FF6B35,&H80FF6B35,1,0,0,0,100,100,0,0,1,3,0,2,{style.horizontal_margin},{style.horizontal_margin},{style.vertical_margin},1\n\n")

            # Events section
            f.write("[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

            for segment in segments:
                start_time = self._seconds_to_ass_time(segment.start_time)
                end_time = self._seconds_to_ass_time(segment.end_time)
                style_name = "CTA" if segment.is_cta else "Default"

                f.write(f"Dialogue: 0,{start_time},{end_time},{style_name},,0,0,0,,{segment.text}\n")

    def _generate_json(self, segments: List[SubtitleSegment], output_path: str):
        """Generate JSON subtitle file with full metadata"""
        data = {
            "format": "AI Content Creator Subtitles",
            "version": "1.0",
            "segments": []
        }

        for segment in segments:
            segment_data = {
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.end_time - segment.start_time,
                "text": segment.text,
                "is_cta": segment.is_cta,
                "keywords": segment.keywords,
                "confidence": segment.confidence,
                "style": asdict(segment.style)
            }
            data["segments"].append(segment_data)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

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

    def _seconds_to_ass_time(self, seconds: float) -> str:
        """Convert seconds to ASS time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int((seconds % 1) * 100)

        return f"{hours:01d}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def create_subtitle_generator(temp_dir: Optional[str] = None) -> SubtitleGenerator:
    """
    Factory function to create SubtitleGenerator instance

    Args:
        temp_dir: Temporary directory for processing

    Returns:
        SubtitleGenerator instance
    """
    return SubtitleGenerator(temp_dir=temp_dir)