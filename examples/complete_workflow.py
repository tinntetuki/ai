#!/usr/bin/env python3
"""
Complete AI Content Creator Workflow Example

This example demonstrates a full content creation pipeline:
1. Video enhancement (upscaling)
2. Speech recognition and transcription
3. Product detection and analysis
4. Professional voiceover generation
5. Subtitle creation and video annotation

Usage:
    python examples/complete_workflow.py input_video.mp4
"""

import asyncio
import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import AI Content Creator modules
from src.ai_engine.video_upscaler import create_upscaler
from src.ai_engine.speech_processor import create_speech_processor
from src.ai_engine.tts_processor import create_tts_processor
from src.ai_engine.product_detector import create_product_detector
from src.ai_engine.subtitle_generator import create_subtitle_generator


class ContentCreationWorkflow:
    """Complete content creation workflow manager"""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir = Path(tempfile.gettempdir()) / "ai_content_workflow"
        self.temp_dir.mkdir(exist_ok=True)

        # Initialize processors
        self.upscaler = create_upscaler(scale=4)
        self.speech_processor = create_speech_processor(model_size="base")
        self.tts_processor = create_tts_processor()
        self.product_detector = create_product_detector()
        self.subtitle_generator = create_subtitle_generator()

        logger.info("Content creation workflow initialized")

    async def process_video(
        self,
        input_path: str,
        enhance_video: bool = True,
        detect_products: bool = True,
        create_voiceover: bool = True,
        add_subtitles: bool = True,
        target_language: str = "zh",
        voice_style: str = "professional"
    ) -> Dict[str, Any]:
        """
        Process video through complete AI workflow

        Args:
            input_path: Path to input video
            enhance_video: Whether to upscale video quality
            detect_products: Whether to detect and analyze products
            create_voiceover: Whether to generate new voiceover
            add_subtitles: Whether to add styled subtitles
            target_language: Target language for TTS
            voice_style: Voice style (professional, friendly, energetic)

        Returns:
            Dictionary with all processing results and output paths
        """
        input_path = Path(input_path)
        base_name = input_path.stem
        results = {}

        try:
            logger.info(f"Starting workflow for: {input_path}")

            # Step 1: Video Enhancement
            enhanced_path = input_path
            if enhance_video:
                logger.info("Step 1: Enhancing video quality...")
                enhanced_path = self.output_dir / f"{base_name}_enhanced.mp4"

                success = self.upscaler.upscale_video(
                    input_path=str(input_path),
                    output_path=str(enhanced_path)
                )

                if success:
                    results['enhanced_video'] = str(enhanced_path)
                    logger.info(f"Video enhanced: {enhanced_path}")
                else:
                    logger.warning("Video enhancement failed, using original")
                    enhanced_path = input_path

            # Step 2: Speech Recognition and Transcription
            logger.info("Step 2: Transcribing speech...")
            transcription = self.speech_processor.transcribe_video(
                video_path=str(enhanced_path),
                language=target_language if target_language != "auto" else None
            )

            transcript_path = self.output_dir / f"{base_name}_transcript.json"
            self.speech_processor.save_transcription(
                transcription, str(transcript_path), format="json"
            )

            # Also save as SRT
            srt_path = self.output_dir / f"{base_name}_subtitles.srt"
            self.speech_processor.save_transcription(
                transcription, str(srt_path), format="srt"
            )

            results['transcription'] = {
                'json': str(transcript_path),
                'srt': str(srt_path),
                'language': transcription.language,
                'duration': transcription.duration,
                'segments': len(transcription.segments)
            }

            logger.info(f"Transcription completed: {transcription.language} "
                       f"({len(transcription.segments)} segments)")

            # Step 3: Product Detection and Analysis
            if detect_products:
                logger.info("Step 3: Detecting products...")

                detections = self.product_detector.detect_products_in_video(
                    video_path=str(enhanced_path),
                    sample_rate=1.0,
                    confidence_threshold=0.5
                )

                if detections:
                    # Create product tracks
                    tracks = self.product_detector.create_product_tracks(detections)

                    # Analyze speech-product correlation
                    speech_analysis = self.product_detector.analyze_products_with_speech(
                        detections, transcription
                    )

                    # Generate summary
                    summary = self.product_detector.generate_product_summary(
                        tracks, speech_analysis
                    )

                    # Save analysis
                    analysis_path = self.output_dir / f"{base_name}_product_analysis.json"
                    with open(analysis_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'detections': len(detections),
                            'tracks': len(tracks),
                            'summary': summary,
                            'speech_analysis': speech_analysis
                        }, f, ensure_ascii=False, indent=2)

                    # Create annotated video
                    annotated_path = self.output_dir / f"{base_name}_annotated.mp4"
                    annotation_success = self.product_detector.annotate_video_with_products(
                        video_path=str(enhanced_path),
                        detections=detections,
                        output_path=str(annotated_path)
                    )

                    results['product_analysis'] = {
                        'analysis': str(analysis_path),
                        'annotated_video': str(annotated_path) if annotation_success else None,
                        'detections_count': len(detections),
                        'unique_products': len(tracks),
                        'speech_sync_ratio': speech_analysis.get('sync_ratio', 0)
                    }

                    logger.info(f"Product analysis completed: {len(detections)} detections, "
                               f"{len(tracks)} unique products")

            # Step 4: Professional Voiceover Generation
            if create_voiceover:
                logger.info("Step 4: Creating professional voiceover...")

                # Create voice profile
                voice_profile = self.tts_processor.create_product_voice_profile(
                    language=target_language,
                    voice_type=voice_style,
                    gender="female"
                )

                # Generate voiceover audio
                voiceover_audio_path = self.temp_dir / f"{base_name}_voiceover.wav"
                await self.tts_processor.create_voiceover_from_transcription(
                    transcription=transcription,
                    voice_profile=voice_profile,
                    output_path=str(voiceover_audio_path),
                    preserve_timing=True,
                    add_pauses=True
                )

                # Replace video audio
                voiceover_video_path = self.output_dir / f"{base_name}_voiceover.mp4"
                success = await self.tts_processor.replace_video_audio(
                    video_path=str(enhanced_path),
                    new_audio_path=str(voiceover_audio_path),
                    output_path=str(voiceover_video_path),
                    fade_duration=0.5
                )

                if success:
                    results['voiceover'] = {
                        'video': str(voiceover_video_path),
                        'audio': str(voiceover_audio_path),
                        'voice_profile': voice_profile.__dict__
                    }
                    logger.info(f"Voiceover created: {voiceover_video_path}")

                    # Update enhanced_path for subtitle generation
                    enhanced_path = voiceover_video_path

            # Step 5: Styled Subtitle Generation
            if add_subtitles:
                logger.info("Step 5: Adding styled subtitles...")

                # Create subtitle style for product videos
                style = self.subtitle_generator.create_product_style(
                    highlight_keywords=True,
                    keywords=self._extract_product_keywords(
                        results.get('product_analysis', {})
                    )
                )

                subtitle_video_path = self.output_dir / f"{base_name}_final.mp4"
                subtitle_success = self.subtitle_generator.create_subtitle_video(
                    video_path=str(enhanced_path),
                    transcription=transcription,
                    output_path=str(subtitle_video_path),
                    style=style
                )

                if subtitle_success:
                    results['final_video'] = str(subtitle_video_path)
                    logger.info(f"Final video with subtitles: {subtitle_video_path}")

            # Step 6: Create Summary Report
            report_path = self.output_dir / f"{base_name}_report.json"
            self._create_summary_report(results, report_path)

            logger.info("Workflow completed successfully!")
            return results

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            raise

    def _extract_product_keywords(self, product_analysis: Dict) -> list:
        """Extract product keywords from analysis for subtitle highlighting"""
        keywords = []

        if 'summary' in product_analysis:
            summary = product_analysis['summary']
            if 'top_products' in summary:
                for product in summary['top_products']:
                    keywords.append(product.get('class_name', ''))

        return keywords[:10]  # Limit to top 10 keywords

    def _create_summary_report(self, results: Dict[str, Any], report_path: Path):
        """Create comprehensive summary report"""
        report = {
            'workflow_version': '1.0.0',
            'processing_summary': {
                'steps_completed': len([k for k in results.keys() if results[k]]),
                'total_outputs': len(results),
                'final_video': results.get('final_video', results.get('voiceover', {}).get('video'))
            },
            'quality_metrics': {
                'video_enhanced': 'enhanced_video' in results,
                'speech_transcribed': 'transcription' in results,
                'products_detected': 'product_analysis' in results,
                'voiceover_generated': 'voiceover' in results,
                'subtitles_added': 'final_video' in results
            },
            'output_files': results,
            'recommendations': self._generate_recommendations(results)
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Summary report created: {report_path}")

    def _generate_recommendations(self, results: Dict[str, Any]) -> list:
        """Generate improvement recommendations based on results"""
        recommendations = []

        # Check transcription quality
        if 'transcription' in results:
            segments = results['transcription'].get('segments', 0)
            duration = results['transcription'].get('duration', 0)

            if duration > 0 and segments / duration < 0.2:  # Low segment density
                recommendations.append(
                    "Consider using a larger speech model for better transcription accuracy"
                )

        # Check product detection
        if 'product_analysis' in results:
            sync_ratio = results['product_analysis'].get('speech_sync_ratio', 0)

            if sync_ratio < 0.5:
                recommendations.append(
                    "Low speech-visual product correlation detected. "
                    "Consider improving product mentions in speech."
                )

            detections = results['product_analysis'].get('detections_count', 0)
            if detections == 0:
                recommendations.append(
                    "No products detected. Check video content or adjust detection parameters."
                )

        return recommendations


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="AI Content Creator - Complete Workflow")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--language", default="zh", help="Target language (zh, en, etc.)")
    parser.add_argument("--voice-style", default="professional",
                       choices=["professional", "friendly", "energetic"],
                       help="Voice style for TTS")
    parser.add_argument("--skip-enhance", action="store_true",
                       help="Skip video enhancement")
    parser.add_argument("--skip-products", action="store_true",
                       help="Skip product detection")
    parser.add_argument("--skip-voiceover", action="store_true",
                       help="Skip voiceover generation")
    parser.add_argument("--skip-subtitles", action="store_true",
                       help="Skip subtitle generation")

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_video)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Create workflow
    workflow = ContentCreationWorkflow(output_dir=args.output_dir)

    # Run workflow
    try:
        results = asyncio.run(workflow.process_video(
            input_path=str(input_path),
            enhance_video=not args.skip_enhance,
            detect_products=not args.skip_products,
            create_voiceover=not args.skip_voiceover,
            add_subtitles=not args.skip_subtitles,
            target_language=args.language,
            voice_style=args.voice_style
        ))

        print("\n" + "="*60)
        print("WORKFLOW COMPLETED SUCCESSFULLY!")
        print("="*60)

        if 'final_video' in results:
            print(f"📹 Final Video: {results['final_video']}")
        elif 'voiceover' in results:
            print(f"📹 Enhanced Video: {results['voiceover']['video']}")
        elif 'enhanced_video' in results:
            print(f"📹 Enhanced Video: {results['enhanced_video']}")

        if 'transcription' in results:
            print(f"📝 Transcription: {results['transcription']['json']}")
            print(f"📑 Subtitles: {results['transcription']['srt']}")

        if 'product_analysis' in results:
            print(f"🛍️  Product Analysis: {results['product_analysis']['analysis']}")
            if results['product_analysis']['annotated_video']:
                print(f"📺 Annotated Video: {results['product_analysis']['annotated_video']}")

        print(f"📊 Summary Report: {args.output_dir}/{input_path.stem}_report.json")
        print("="*60)

        return 0

    except Exception as e:
        print(f"Error: Workflow failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())