"""
Command Line Interface for AI Content Creator
"""

import click
import os
from pathlib import Path
from loguru import logger

from .ai_engine.video_upscaler import create_upscaler
from .ai_engine.speech_processor import create_speech_processor
from .ai_engine.tts_processor import create_tts_processor
from .ai_engine.product_detector import create_product_detector


@click.group()
def main():
    """AI Content Creator CLI"""
    pass


@main.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--scale', default=4, help='Upscaling factor (2, 4, or 8)')
@click.option('--model', default='RealESRGAN_x4plus', help='Real-ESRGAN model name')
@click.option('--duration', type=float, help='Duration to process (seconds)')
@click.option('--smart-segments', is_flag=True, help='Use smart segment detection')
@click.option('--segments', type=int, default=3, help='Number of smart segments')
def upscale_video(input_path, output_path, scale, model, duration, smart_segments, segments):
    """Upscale a video file using Real-ESRGAN"""

    try:
        logger.info(f"Starting video upscaling: {input_path}")

        # Create upscaler
        upscaler = create_upscaler(scale=scale, model_name=model)

        # Determine processing strategy
        target_segments = None
        if smart_segments and not duration:
            target_segments = upscaler.get_smart_segments(input_path, max_segments=segments)
            logger.info(f"Smart segments: {target_segments}")

        # Progress callback
        def progress_callback(progress: float):
            click.echo(f"Progress: {progress:.1f}%")

        # Process video
        success = upscaler.upscale_video(
            input_path=input_path,
            output_path=output_path,
            segment_duration=duration,
            target_segments=target_segments,
            progress_callback=progress_callback
        )

        if success:
            click.echo(f"✅ Video upscaling completed: {output_path}")
        else:
            click.echo("❌ Video upscaling failed")
            exit(1)

    except Exception as e:
        logger.error(f"CLI upscale error: {e}")
        click.echo(f"❌ Error: {e}")
        exit(1)


@main.command()
@click.option('--host', default='0.0.0.0', help='API host')
@click.option('--port', default=8000, help='API port')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host, port, reload):
    """Start the API server"""

    try:
        import uvicorn
        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            reload=reload,
            access_log=True
        )
    except ImportError:
        click.echo("❌ uvicorn not installed. Run: pip install uvicorn")
        exit(1)


@main.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--model', default='base', help='Whisper model size (tiny, base, small, medium, large)')
@click.option('--language', help='Language code (e.g., zh, en) or auto-detect')
@click.option('--format', 'output_format', default='json', help='Output format (json, srt, vtt, txt)')
@click.option('--duration', type=float, help='Process only first N seconds')
@click.option('--keywords', help='Comma-separated product keywords to analyze')
def transcribe(input_path, output_path, model, language, output_format, duration, keywords):
    """Transcribe video/audio to text using Whisper"""

    try:
        logger.info(f"Starting transcription: {input_path}")

        # Create speech processor
        processor = create_speech_processor(model_size=model)

        # Transcribe
        result = processor.transcribe_video(
            video_path=input_path,
            language=language,
            segment_duration=duration
        )

        # Save transcription
        processor.save_transcription(result, output_path, format=output_format)

        click.echo(f"✅ Transcription completed: {output_path}")
        click.echo(f"Language detected: {result.language}")
        click.echo(f"Duration: {result.duration:.2f} seconds")
        click.echo(f"Segments: {len(result.segments)}")

        # Analyze keywords if provided
        if keywords:
            keyword_list = [k.strip() for k in keywords.split(',')]
            product_segments = processor.detect_product_mentions(result, keyword_list)

            if product_segments:
                click.echo(f"\n🎯 Found {len(product_segments)} segments with product keywords:")
                for segment, matched_keywords in product_segments:
                    click.echo(f"  {segment.start:.1f}s-{segment.end:.1f}s: {matched_keywords}")
            else:
                click.echo(f"\n❌ No product keywords found: {keyword_list}")

    except Exception as e:
        logger.error(f"CLI transcription error: {e}")
        click.echo(f"❌ Error: {e}")
        exit(1)


@main.command()
@click.argument('text')
@click.argument('output_path', type=click.Path())
@click.option('--provider', default='edge', help='TTS provider (edge, gtts, pyttsx3)')
@click.option('--voice', help='Voice name')
@click.option('--language', default='zh-CN', help='Language code')
@click.option('--gender', default='female', help='Voice gender')
@click.option('--style', help='Voice style (for edge-tts)')
@click.option('--speed', default=1.0, help='Speech speed multiplier')
@click.option('--pitch', default=1.0, help='Pitch multiplier')
@click.option('--volume', default=1.0, help='Volume multiplier')
def synthesize_speech(text, output_path, provider, voice, language, gender, style, speed, pitch, volume):
    """Synthesize text to speech"""

    try:
        logger.info(f"Starting TTS synthesis: {len(text)} characters")

        # Create TTS processor
        tts_processor = create_tts_processor()

        # Auto-select voice if not specified
        if not voice:
            if language.startswith('zh'):
                voice = "zh-CN-XiaoxiaoNeural" if gender == "female" else "zh-CN-YunxiNeural"
            else:
                voice = "en-US-JennyNeural" if gender == "female" else "en-US-GuyNeural"

        # Create voice profile
        from .ai_engine.tts_processor import VoiceProfile
        voice_profile = VoiceProfile(
            provider=provider,
            voice_name=voice,
            language=language,
            gender=gender,
            style=style,
            speed=speed,
            pitch=pitch,
            volume=volume
        )

        # Synthesize
        import asyncio
        result_path = asyncio.run(tts_processor.synthesize_text(
            text=text,
            voice_profile=voice_profile,
            output_path=output_path
        ))

        # Get duration
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(result_path)
        duration = len(audio) / 1000.0

        click.echo(f"✅ TTS synthesis completed: {output_path}")
        click.echo(f"Duration: {duration:.2f} seconds")
        click.echo(f"Voice: {voice} ({provider})")

    except Exception as e:
        logger.error(f"CLI TTS synthesis error: {e}")
        click.echo(f"❌ Error: {e}")
        exit(1)


@main.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--provider', default='edge', help='TTS provider')
@click.option('--voice', help='Voice name')
@click.option('--language', default='zh-CN', help='Language code')
@click.option('--gender', default='female', help='Voice gender')
@click.option('--style', help='Voice style')
@click.option('--speed', default=1.0, help='Speech speed')
@click.option('--preserve-timing', is_flag=True, help='Preserve original timing')
@click.option('--add-pauses', is_flag=True, help='Add pauses between segments')
def create_voiceover(video_path, output_path, provider, voice, language, gender, style, speed, preserve_timing, add_pauses):
    """Create voiceover from video transcription"""

    try:
        logger.info(f"Creating voiceover for: {video_path}")

        # Create processors
        speech_processor = create_speech_processor(model_size="base")
        tts_processor = create_tts_processor()

        click.echo("📝 Transcribing video...")

        # Transcribe video
        transcription = speech_processor.transcribe_video(video_path)

        click.echo(f"Language detected: {transcription.language}")
        click.echo(f"Segments: {len(transcription.segments)}")

        # Auto-select voice if not specified
        if not voice:
            if language.startswith('zh'):
                voice = "zh-CN-XiaoxiaoNeural" if gender == "female" else "zh-CN-YunxiNeural"
            else:
                voice = "en-US-JennyNeural" if gender == "female" else "en-US-GuyNeural"

        # Create voice profile
        from .ai_engine.tts_processor import VoiceProfile
        voice_profile = VoiceProfile(
            provider=provider,
            voice_name=voice,
            language=language,
            gender=gender,
            style=style,
            speed=speed,
            pitch=1.0,
            volume=1.0
        )

        click.echo("🎤 Creating voiceover...")

        # Create voiceover
        import asyncio
        audio_output = str(Path(output_path).with_suffix('.wav'))

        asyncio.run(tts_processor.create_voiceover_from_transcription(
            transcription=transcription,
            voice_profile=voice_profile,
            output_path=audio_output,
            preserve_timing=preserve_timing,
            add_pauses=add_pauses
        ))

        click.echo("🎬 Replacing video audio...")

        # Replace video audio
        success = asyncio.run(tts_processor.replace_video_audio(
            video_path=video_path,
            new_audio_path=audio_output,
            output_path=output_path,
            fade_duration=0.5
        ))

        if success:
            click.echo(f"✅ Voiceover completed: {output_path}")
            click.echo(f"Voice: {voice} ({provider})")
        else:
            click.echo("❌ Failed to replace video audio")
            exit(1)

    except Exception as e:
        logger.error(f"CLI voiceover creation error: {e}")
        click.echo(f"❌ Error: {e}")
        exit(1)


@main.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--sample-rate', default=1.0, help='Frames per second to sample')
@click.option('--confidence', default=0.5, help='Minimum detection confidence')
@click.option('--keywords', help='Comma-separated product keywords')
@click.option('--yolo-model', default='yolov8n.pt', help='YOLO model to use')
@click.option('--annotate', is_flag=True, help='Create annotated video')
@click.option('--speech-analysis', is_flag=True, help='Include speech correlation analysis')
def detect_products(video_path, output_path, sample_rate, confidence, keywords, yolo_model, annotate, speech_analysis):
    """Detect products in video and generate analysis"""

    try:
        logger.info(f"Starting product detection: {video_path}")

        # Parse keywords
        keyword_list = None
        if keywords:
            keyword_list = [k.strip() for k in keywords.split(',')]

        # Create detector
        detector = create_product_detector(yolo_model=yolo_model)

        click.echo("🔍 Detecting products in video...")

        # Detect products
        detections = detector.detect_products_in_video(
            video_path=video_path,
            sample_rate=sample_rate,
            confidence_threshold=confidence,
            product_keywords=keyword_list
        )

        click.echo(f"Found {len(detections)} product detections")

        # Create product tracks
        click.echo("📊 Creating product tracks...")
        tracks = detector.create_product_tracks(detections)

        click.echo(f"Created {len(tracks)} product tracks")

        # Speech analysis if requested
        speech_analysis_result = None
        if speech_analysis:
            click.echo("🎤 Analyzing speech correlation...")
            from .ai_engine.speech_processor import create_speech_processor

            speech_processor = create_speech_processor(model_size="base")
            transcription = speech_processor.transcribe_video(video_path)

            speech_analysis_result = detector.analyze_products_with_speech(
                detections, transcription
            )

            click.echo(f"Speech sync ratio: {speech_analysis_result.get('sync_ratio', 0):.2f}")

        # Generate summary
        click.echo("📈 Generating analysis summary...")
        summary = detector.generate_product_summary(tracks, speech_analysis_result)

        # Save results
        import json
        results = {
            "detections": [
                {
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "timestamp": d.timestamp,
                    "bbox": {"x1": d.bbox.x1, "y1": d.bbox.y1, "x2": d.bbox.x2, "y2": d.bbox.y2},
                    "description": d.description,
                    "category": d.category
                }
                for d in detections
            ],
            "tracks": [
                {
                    "class_name": t.class_name,
                    "confidence_avg": t.confidence_avg,
                    "duration": t.last_appearance - t.first_appearance,
                    "importance_score": t.importance_score
                }
                for t in tracks
            ],
            "summary": summary,
            "speech_analysis": speech_analysis_result
        }

        # Determine output format
        if output_path.endswith('.json'):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            click.echo(f"✅ Analysis saved to: {output_path}")

        # Create annotated video if requested
        if annotate:
            video_output = str(Path(output_path).with_suffix('.mp4'))
            click.echo("🎬 Creating annotated video...")

            success = detector.annotate_video_with_products(
                video_path=video_path,
                detections=detections,
                output_path=video_output
            )

            if success:
                click.echo(f"✅ Annotated video: {video_output}")
            else:
                click.echo("❌ Failed to create annotated video")

        # Print summary
        click.echo(f"\n📊 Detection Summary:")
        click.echo(f"  Total detections: {len(detections)}")
        click.echo(f"  Unique products: {len(tracks)}")
        if tracks:
            top_product = max(tracks, key=lambda x: x.importance_score)
            click.echo(f"  Top product: {top_product.class_name} (score: {top_product.importance_score:.2f})")

    except Exception as e:
        logger.error(f"CLI product detection error: {e}")
        click.echo(f"❌ Error: {e}")
        exit(1)


@main.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--sample-rate', default=1.0, help='Detection sampling rate')
@click.option('--confidence', default=0.5, help='Minimum confidence threshold')
@click.option('--keywords', help='Product keywords to focus on')
@click.option('--bbox-color', default='0,255,0', help='Bounding box color (R,G,B)')
@click.option('--text-color', default='255,255,255', help='Text color (R,G,B)')
@click.option('--font-scale', default=0.7, help='Font scale for text')
def annotate_video(video_path, output_path, sample_rate, confidence, keywords, bbox_color, text_color, font_scale):
    """Create annotated video with product detections"""

    try:
        logger.info(f"Starting video annotation: {video_path}")

        # Parse colors
        bbox_color_rgb = tuple(map(int, bbox_color.split(',')))
        text_color_rgb = tuple(map(int, text_color.split(',')))

        # Parse keywords
        keyword_list = None
        if keywords:
            keyword_list = [k.strip() for k in keywords.split(',')]

        # Create detector
        detector = create_product_detector()

        click.echo("🔍 Detecting products...")

        # Detect products
        detections = detector.detect_products_in_video(
            video_path=video_path,
            sample_rate=sample_rate,
            confidence_threshold=confidence,
            product_keywords=keyword_list
        )

        click.echo(f"Found {len(detections)} detections")

        # Create annotation style
        from .ai_engine.product_detector import AnnotationStyle
        style = AnnotationStyle(
            bbox_color=bbox_color_rgb,
            text_color=text_color_rgb,
            font_scale=font_scale,
            show_confidence=True,
            show_description=True
        )

        click.echo("🎬 Creating annotated video...")

        # Create annotated video
        success = detector.annotate_video_with_products(
            video_path=video_path,
            detections=detections,
            output_path=output_path,
            style=style
        )

        if success:
            click.echo(f"✅ Annotated video created: {output_path}")
        else:
            click.echo("❌ Video annotation failed")
            exit(1)

    except Exception as e:
        logger.error(f"CLI video annotation error: {e}")
        click.echo(f"❌ Error: {e}")
        exit(1)


@main.command()
def models():
    """List available models"""

    click.echo("🎥 Real-ESRGAN Upscaling Models:")
    click.echo("=" * 40)

    upscale_models = [
        ("RealESRGAN_x4plus", "4x", "General purpose, good for most content"),
        ("RealESRGAN_x2plus", "2x", "Faster processing, lower quality"),
        ("RealESRGAN_x4plus_anime_6B", "4x", "Optimized for anime/cartoon content"),
    ]

    for name, scale, desc in upscale_models:
        click.echo(f"  {name}")
        click.echo(f"    Scale: {scale}")
        click.echo(f"    Description: {desc}")
        click.echo()

    click.echo("🎤 Whisper Speech Recognition Models:")
    click.echo("=" * 40)

    whisper_models = [
        ("tiny", "~39 MB", "~32x realtime", "Fastest, lowest quality"),
        ("base", "~74 MB", "~16x realtime", "Good balance of speed and quality"),
        ("small", "~244 MB", "~6x realtime", "Better quality, slower"),
        ("medium", "~769 MB", "~2x realtime", "High quality, much slower"),
        ("large", "~1550 MB", "~1x realtime", "Best quality, slowest"),
    ]

    for name, size, speed, desc in whisper_models:
        click.echo(f"  {name}")
        click.echo(f"    Size: {size}")
        click.echo(f"    Speed: {speed}")
        click.echo(f"    Description: {desc}")
        click.echo()

    click.echo("🔊 Text-to-Speech Providers:")
    click.echo("=" * 40)

    tts_providers = [
        ("edge", "Microsoft Edge TTS", "High quality, many voices, styles support"),
        ("gtts", "Google TTS", "Good quality, free, limited voices"),
        ("pyttsx3", "System TTS", "Uses system voices, offline"),
        ("azure", "Azure Cognitive Services", "Premium quality, requires API key"),
    ]

    for provider, name, desc in tts_providers:
        click.echo(f"  {provider}")
        click.echo(f"    Name: {name}")
        click.echo(f"    Description: {desc}")
        click.echo()

    click.echo("🎯 Recommended Voice Profiles for Product Videos:")
    click.echo("=" * 50)

    profiles = [
        ("Professional Female (Chinese)", "zh-CN-XiaoxiaoNeural", "Calm, clear, trustworthy"),
        ("Professional Male (Chinese)", "zh-CN-YunxiNeural", "Authoritative, confident"),
        ("Friendly Female (Chinese)", "zh-CN-XiaoyiNeural", "Warm, approachable"),
        ("Professional Female (English)", "en-US-JennyNeural", "Clear, professional"),
        ("Professional Male (English)", "en-US-GuyNeural", "Deep, confident"),
        ("Friendly Female (English)", "en-US-AriaNeural", "Conversational, engaging"),
    ]

    for profile_name, voice_name, desc in profiles:
        click.echo(f"  {profile_name}")
        click.echo(f"    Voice: {voice_name}")
        click.echo(f"    Style: {desc}")
        click.echo()

    click.echo("🔍 Object Detection Models:")
    click.echo("=" * 40)

    detection_models = [
        ("yolov8n.pt", "Nano", "Fast", "Good for real-time"),
        ("yolov8s.pt", "Small", "Balanced", "Good speed/accuracy balance"),
        ("yolov8m.pt", "Medium", "Slower", "Better accuracy"),
        ("yolov8l.pt", "Large", "Slow", "Best accuracy"),
    ]

    for model, size, speed, desc in detection_models:
        click.echo(f"  {model}")
        click.echo(f"    Size: {size}")
        click.echo(f"    Speed: {speed}")
        click.echo(f"    Description: {desc}")
        click.echo()

    click.echo("🧠 CLIP Models for Semantic Understanding:")
    click.echo("=" * 45)

    clip_models = [
        ("ViT-B/32", "Balanced vision-text understanding"),
        ("ViT-B/16", "Higher resolution, better quality"),
        ("ViT-L/14", "Large model, best performance"),
    ]

    for model, desc in clip_models:
        click.echo(f"  {model}")
        click.echo(f"    Description: {desc}")
        click.echo()

    click.echo("📦 Supported Product Categories:")
    click.echo("=" * 40)

    categories = [
        ("electronics", "Phones, laptops, cameras, headphones"),
        ("clothing", "Shirts, shoes, bags, accessories"),
        ("home", "Furniture, decorations, appliances"),
        ("beauty", "Cosmetics, skincare, perfumes"),
        ("sports", "Equipment, sportswear, accessories"),
        ("food", "Snacks, drinks, packaged foods"),
    ]

    for category, items in categories:
        click.echo(f"  {category}")
        click.echo(f"    Items: {items}")
        click.echo()


if __name__ == '__main__':
    main()