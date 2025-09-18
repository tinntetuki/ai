"""
Text-to-Speech Processing Module
Support for multiple TTS engines optimized for content creation
"""

import os
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import json
from loguru import logger

import edge_tts
from gtts import gTTS
import pyttsx3
from pydub import AudioSegment
import soundfile as sf
import numpy as np

from .speech_processor import TranscriptionResult, TranscriptionSegment


@dataclass
class VoiceProfile:
    """Voice profile configuration"""
    provider: str  # 'edge', 'gtts', 'pyttsx3', 'azure'
    voice_name: str
    language: str
    gender: str  # 'male', 'female', 'neutral'
    style: Optional[str] = None  # emotional style for edge-tts
    speed: float = 1.0  # speech speed multiplier
    pitch: float = 1.0  # pitch multiplier
    volume: float = 1.0  # volume multiplier


@dataclass
class TTSSegment:
    """TTS segment with timing and voice info"""
    text: str
    start_time: float
    end_time: float
    voice_profile: VoiceProfile
    audio_path: Optional[str] = None
    duration: Optional[float] = None


class TTSProcessor:
    """
    Text-to-Speech processor with multiple engine support
    Optimized for video content creation and voiceovers
    """

    def __init__(self):
        """Initialize TTS processor"""
        self.temp_dir = Path(tempfile.gettempdir()) / "ai_content_creator_tts"
        self.temp_dir.mkdir(exist_ok=True)

        # Available voices cache
        self._edge_voices = None
        self._pyttsx3_engine = None

    async def get_available_voices(self, provider: str = "edge") -> List[Dict]:
        """
        Get available voices for a TTS provider

        Args:
            provider: TTS provider ('edge', 'gtts', 'pyttsx3')

        Returns:
            List of voice information dictionaries
        """
        try:
            if provider == "edge":
                return await self._get_edge_voices()
            elif provider == "gtts":
                return self._get_gtts_voices()
            elif provider == "pyttsx3":
                return self._get_pyttsx3_voices()
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            logger.error(f"Failed to get voices for {provider}: {e}")
            return []

    async def _get_edge_voices(self) -> List[Dict]:
        """Get Microsoft Edge TTS voices"""
        if self._edge_voices is None:
            try:
                voices = await edge_tts.list_voices()
                self._edge_voices = [
                    {
                        "name": voice["Name"],
                        "short_name": voice["ShortName"],
                        "language": voice["Locale"],
                        "gender": voice["Gender"],
                        "styles": voice.get("StyleList", []),
                        "provider": "edge"
                    }
                    for voice in voices
                ]
            except Exception as e:
                logger.error(f"Failed to load Edge voices: {e}")
                self._edge_voices = []

        return self._edge_voices

    def _get_gtts_voices(self) -> List[Dict]:
        """Get Google TTS supported languages"""
        # Common gTTS languages for product videos
        languages = [
            {"name": "Chinese (Simplified)", "code": "zh", "provider": "gtts"},
            {"name": "Chinese (Traditional)", "code": "zh-tw", "provider": "gtts"},
            {"name": "English (US)", "code": "en", "provider": "gtts"},
            {"name": "English (UK)", "code": "en-uk", "provider": "gtts"},
            {"name": "Japanese", "code": "ja", "provider": "gtts"},
            {"name": "Korean", "code": "ko", "provider": "gtts"},
            {"name": "Spanish", "code": "es", "provider": "gtts"},
            {"name": "French", "code": "fr", "provider": "gtts"},
        ]
        return languages

    def _get_pyttsx3_voices(self) -> List[Dict]:
        """Get system TTS voices"""
        try:
            if self._pyttsx3_engine is None:
                self._pyttsx3_engine = pyttsx3.init()

            voices = self._pyttsx3_engine.getProperty('voices')
            return [
                {
                    "name": voice.name,
                    "id": voice.id,
                    "language": getattr(voice, 'languages', ['en']),
                    "gender": "unknown",
                    "provider": "pyttsx3"
                }
                for voice in voices if voice
            ]

        except Exception as e:
            logger.error(f"Failed to get pyttsx3 voices: {e}")
            return []

    async def synthesize_text(
        self,
        text: str,
        voice_profile: VoiceProfile,
        output_path: Optional[str] = None
    ) -> str:
        """
        Synthesize text to speech

        Args:
            text: Text to synthesize
            voice_profile: Voice configuration
            output_path: Output audio file path (auto-generated if None)

        Returns:
            Path to generated audio file
        """
        try:
            if output_path is None:
                output_path = self.temp_dir / f"tts_{hash(text)}_{voice_profile.provider}.wav"

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if voice_profile.provider == "edge":
                await self._synthesize_edge(text, voice_profile, str(output_path))
            elif voice_profile.provider == "gtts":
                self._synthesize_gtts(text, voice_profile, str(output_path))
            elif voice_profile.provider == "pyttsx3":
                self._synthesize_pyttsx3(text, voice_profile, str(output_path))
            else:
                raise ValueError(f"Unsupported provider: {voice_profile.provider}")

            # Apply audio modifications (speed, pitch, volume)
            if (voice_profile.speed != 1.0 or
                voice_profile.pitch != 1.0 or
                voice_profile.volume != 1.0):
                self._modify_audio(str(output_path), voice_profile)

            logger.info(f"TTS synthesis completed: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise

    async def _synthesize_edge(self, text: str, voice_profile: VoiceProfile, output_path: str):
        """Synthesize using Microsoft Edge TTS"""
        # Build SSML if style is specified
        if voice_profile.style:
            ssml_text = f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{voice_profile.language}"><voice name="{voice_profile.voice_name}"><mstts:express-as style="{voice_profile.style}">{text}</mstts:express-as></voice></speak>'
        else:
            ssml_text = text

        # Create communicator
        communicate = edge_tts.Communicate(ssml_text, voice_profile.voice_name)

        # Generate and save audio
        with open(output_path, "wb") as file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    file.write(chunk["data"])

    def _synthesize_gtts(self, text: str, voice_profile: VoiceProfile, output_path: str):
        """Synthesize using Google TTS"""
        tts = gTTS(text=text, lang=voice_profile.language, slow=False)

        # Save to temporary MP3 file first
        temp_mp3 = str(output_path).replace('.wav', '.mp3')
        tts.save(temp_mp3)

        # Convert MP3 to WAV
        audio = AudioSegment.from_mp3(temp_mp3)
        audio.export(output_path, format="wav")

        # Cleanup temporary MP3
        os.unlink(temp_mp3)

    def _synthesize_pyttsx3(self, text: str, voice_profile: VoiceProfile, output_path: str):
        """Synthesize using pyttsx3 (system TTS)"""
        if self._pyttsx3_engine is None:
            self._pyttsx3_engine = pyttsx3.init()

        # Set voice
        self._pyttsx3_engine.setProperty('voice', voice_profile.voice_name)

        # Set rate (speed)
        rate = self._pyttsx3_engine.getProperty('rate')
        self._pyttsx3_engine.setProperty('rate', int(rate * voice_profile.speed))

        # Set volume
        self._pyttsx3_engine.setProperty('volume', voice_profile.volume)

        # Save to file
        self._pyttsx3_engine.save_to_file(text, output_path)
        self._pyttsx3_engine.runAndWait()

    def _modify_audio(self, audio_path: str, voice_profile: VoiceProfile):
        """Apply speed, pitch, and volume modifications to audio"""
        try:
            # Load audio
            audio = AudioSegment.from_wav(audio_path)

            # Apply speed change
            if voice_profile.speed != 1.0:
                # Speed up/slow down playback
                audio = audio.speedup(playback_speed=voice_profile.speed)

            # Apply volume change
            if voice_profile.volume != 1.0:
                # Convert to dB change
                db_change = 20 * np.log10(voice_profile.volume)
                audio = audio + db_change

            # Apply pitch change (approximate using speed + tempo correction)
            if voice_profile.pitch != 1.0:
                # This is a simplified pitch shift
                # For better quality, consider using librosa or other libraries
                sample_rate = audio.frame_rate
                samples = np.array(audio.get_array_of_samples())

                if audio.channels == 2:
                    samples = samples.reshape((-1, 2))

                # Simple pitch shift by resampling
                new_rate = int(sample_rate * voice_profile.pitch)

                # Write with new sample rate, then read back at original rate
                temp_path = audio_path + ".temp"
                sf.write(temp_path, samples, new_rate)

                # Read back and convert
                modified_audio = AudioSegment.from_wav(temp_path)
                os.unlink(temp_path)

                audio = modified_audio

            # Save modified audio
            audio.export(audio_path, format="wav")

        except Exception as e:
            logger.warning(f"Audio modification failed, using original: {e}")

    async def create_voiceover_from_transcription(
        self,
        transcription: TranscriptionResult,
        voice_profile: VoiceProfile,
        output_path: str,
        preserve_timing: bool = True,
        add_pauses: bool = True
    ) -> str:
        """
        Create voiceover from transcription with timing preservation

        Args:
            transcription: Original transcription result
            voice_profile: Voice configuration for new voiceover
            output_path: Output audio file path
            preserve_timing: Maintain original timing
            add_pauses: Add pauses between segments

        Returns:
            Path to generated voiceover file
        """
        try:
            logger.info("Creating voiceover from transcription")

            # Generate TTS for each segment
            tts_segments = []
            temp_files = []

            for i, segment in enumerate(transcription.segments):
                # Generate audio for this segment
                temp_audio_path = self.temp_dir / f"segment_{i}.wav"

                audio_path = await self.synthesize_text(
                    text=segment.text,
                    voice_profile=voice_profile,
                    output_path=str(temp_audio_path)
                )

                temp_files.append(audio_path)

                # Create TTS segment
                tts_segment = TTSSegment(
                    text=segment.text,
                    start_time=segment.start,
                    end_time=segment.end,
                    voice_profile=voice_profile,
                    audio_path=audio_path
                )

                # Get actual audio duration
                audio = AudioSegment.from_wav(audio_path)
                tts_segment.duration = len(audio) / 1000.0  # Convert to seconds

                tts_segments.append(tts_segment)

            # Combine segments with timing
            final_audio = self._combine_tts_segments(
                tts_segments,
                transcription.duration,
                preserve_timing,
                add_pauses
            )

            # Export final audio
            final_audio.export(output_path, format="wav")

            # Cleanup temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

            logger.info(f"Voiceover created: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Voiceover creation failed: {e}")
            raise

    def _combine_tts_segments(
        self,
        tts_segments: List[TTSSegment],
        total_duration: float,
        preserve_timing: bool,
        add_pauses: bool
    ) -> AudioSegment:
        """Combine TTS segments into final audio"""

        # Create base silence audio
        final_audio = AudioSegment.silent(duration=int(total_duration * 1000))

        for i, segment in enumerate(tts_segments):
            # Load segment audio
            segment_audio = AudioSegment.from_wav(segment.audio_path)

            if preserve_timing:
                # Place audio at original timing
                start_ms = int(segment.start_time * 1000)
                end_ms = int(segment.end_time * 1000)
                original_duration = end_ms - start_ms

                # Adjust speed if necessary to fit original timing
                if len(segment_audio) != original_duration:
                    speed_ratio = len(segment_audio) / original_duration
                    if abs(speed_ratio - 1.0) > 0.1:  # Significant difference
                        segment_audio = segment_audio.speedup(playback_speed=speed_ratio)

                # Overlay at correct position
                final_audio = final_audio.overlay(segment_audio, position=start_ms)

            else:
                # Sequential placement with pauses
                if i == 0:
                    final_audio = segment_audio
                else:
                    if add_pauses:
                        # Add pause between segments
                        pause_duration = 500  # 0.5 seconds
                        pause = AudioSegment.silent(duration=pause_duration)
                        final_audio = final_audio + pause + segment_audio
                    else:
                        final_audio = final_audio + segment_audio

        return final_audio

    async def replace_video_audio(
        self,
        video_path: str,
        new_audio_path: str,
        output_path: str,
        fade_duration: float = 0.5
    ) -> bool:
        """
        Replace video audio with new voiceover

        Args:
            video_path: Original video file
            new_audio_path: New audio file
            output_path: Output video file
            fade_duration: Fade in/out duration

        Returns:
            True if successful
        """
        try:
            import moviepy.editor as mp

            logger.info("Replacing video audio")

            # Load video and new audio
            video = mp.VideoFileClip(video_path)
            new_audio = mp.AudioFileClip(new_audio_path)

            # Apply fade if requested
            if fade_duration > 0:
                new_audio = new_audio.fadein(fade_duration).fadeout(fade_duration)

            # Adjust audio duration to match video
            if new_audio.duration > video.duration:
                new_audio = new_audio.subclip(0, video.duration)
            elif new_audio.duration < video.duration:
                # Loop audio if shorter
                loops_needed = int(video.duration / new_audio.duration) + 1
                new_audio = mp.concatenate_audioclips([new_audio] * loops_needed)
                new_audio = new_audio.subclip(0, video.duration)

            # Set new audio
            final_video = video.set_audio(new_audio)

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
            video.close()
            new_audio.close()
            final_video.close()

            logger.info(f"Video with new audio created: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Audio replacement failed: {e}")
            return False

    def create_product_voice_profile(
        self,
        language: str = "zh",
        voice_type: str = "professional",  # 'professional', 'friendly', 'energetic'
        gender: str = "female"
    ) -> VoiceProfile:
        """
        Create optimized voice profile for product videos

        Args:
            language: Target language
            voice_type: Voice style
            gender: Preferred gender

        Returns:
            Configured VoiceProfile
        """
        # Optimized settings for different voice types
        voice_configs = {
            "professional": {
                "speed": 0.95,
                "pitch": 1.0,
                "volume": 1.0,
                "style": "professional"
            },
            "friendly": {
                "speed": 1.0,
                "pitch": 1.05,
                "volume": 1.1,
                "style": "friendly"
            },
            "energetic": {
                "speed": 1.1,
                "pitch": 1.1,
                "volume": 1.2,
                "style": "excited"
            }
        }

        config = voice_configs.get(voice_type, voice_configs["professional"])

        # Default to Edge TTS for best quality
        if language.startswith("zh"):
            # Chinese voices
            voice_name = "zh-CN-XiaoxiaoNeural" if gender == "female" else "zh-CN-YunxiNeural"
            lang_code = "zh-CN"
        else:
            # English voices
            voice_name = "en-US-JennyNeural" if gender == "female" else "en-US-GuyNeural"
            lang_code = "en-US"

        return VoiceProfile(
            provider="edge",
            voice_name=voice_name,
            language=lang_code,
            gender=gender,
            style=config.get("style"),
            speed=config["speed"],
            pitch=config["pitch"],
            volume=config["volume"]
        )


def create_tts_processor() -> TTSProcessor:
    """
    Factory function to create TTSProcessor instance

    Returns:
        TTSProcessor instance
    """
    return TTSProcessor()