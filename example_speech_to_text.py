#!/usr/bin/env python3
"""
Example usage of the speech-to-text functionality
"""

import os
from pathlib import Path
from src.ai_engine.speech_processor import create_speech_processor


def example_basic_transcription():
    """Basic video transcription example"""

    print("🎤 Basic Video Transcription Example")
    print("=" * 40)

    # Setup paths
    input_video = "path/to/your/video.mp4"  # Replace with actual path
    output_json = "path/to/your/transcription.json"

    # Create speech processor (base model for good balance)
    processor = create_speech_processor(model_size='base')

    # Transcribe video
    result = processor.transcribe_video(
        video_path=input_video,
        output_path=output_json,
        language=None  # Auto-detect language
    )

    print(f"✅ Transcription completed!")
    print(f"Language detected: {result.language}")
    print(f"Duration: {result.duration:.2f} seconds")
    print(f"Total segments: {len(result.segments)}")
    print(f"Full text preview: {result.full_text[:200]}...")


def example_subtitle_generation():
    """Generate subtitle files example"""

    print("\n📝 Subtitle Generation Example")
    print("=" * 40)

    input_video = "path/to/your/product_video.mp4"
    processor = create_speech_processor(model_size='base')

    # Transcribe
    result = processor.transcribe_video(input_video)

    # Save in multiple subtitle formats
    output_dir = Path("subtitles")
    output_dir.mkdir(exist_ok=True)

    formats = ['srt', 'vtt', 'json', 'txt']
    for fmt in formats:
        output_path = output_dir / f"video_subtitles.{fmt}"
        processor.save_transcription(result, str(output_path), format=fmt)
        print(f"✅ {fmt.upper()} subtitle saved: {output_path}")


def example_product_keyword_analysis():
    """Analyze video for product keywords"""

    print("\n🎯 Product Keyword Analysis Example")
    print("=" * 40)

    input_video = "path/to/your/product_video.mp4"
    processor = create_speech_processor(model_size='base')

    # Transcribe
    result = processor.transcribe_video(input_video)

    # Define product keywords
    product_keywords = [
        "产品", "商品", "价格", "优惠", "折扣", "质量",
        "功能", "特点", "使用", "效果", "推荐", "购买"
    ]

    # Find product mentions
    product_segments = processor.detect_product_mentions(result, product_keywords)

    print(f"Found {len(product_segments)} segments with product keywords:")
    print()

    for segment, keywords in product_segments:
        print(f"⏰ {segment.start:.1f}s - {segment.end:.1f}s")
        print(f"🔑 Keywords: {', '.join(keywords)}")
        print(f"💬 Text: {segment.text}")
        print("-" * 50)


def example_segment_filtering():
    """Filter transcription segments by quality and duration"""

    print("\n🔍 Segment Filtering Example")
    print("=" * 40)

    input_video = "path/to/your/video.mp4"
    processor = create_speech_processor(model_size='small')  # Better quality

    # Transcribe
    result = processor.transcribe_video(input_video)

    # Filter high-quality segments
    good_segments = processor.get_speech_segments(
        result,
        min_duration=2.0,      # At least 2 seconds
        confidence_threshold=-0.5  # Good confidence
    )

    print(f"Original segments: {len(result.segments)}")
    print(f"High-quality segments: {len(good_segments)}")
    print()

    print("📈 Best segments:")
    for segment in good_segments[:5]:  # Show top 5
        print(f"  {segment.start:.1f}s-{segment.end:.1f}s: {segment.text[:50]}...")


def example_partial_transcription():
    """Transcribe only part of a long video"""

    print("\n⏱️ Partial Video Transcription Example")
    print("=" * 40)

    input_video = "path/to/your/long_video.mp4"
    processor = create_speech_processor(model_size='base')

    # Transcribe only first 2 minutes
    result = processor.transcribe_video(
        video_path=input_video,
        segment_duration=120.0  # 2 minutes
    )

    print(f"✅ Transcribed first 2 minutes")
    print(f"Language: {result.language}")
    print(f"Segments: {len(result.segments)}")

    # Save as SRT for video editing software
    processor.save_transcription(result, "first_2min_subtitles.srt", format="srt")
    print("📁 SRT subtitle file saved for video editing")


def example_batch_transcription():
    """Batch transcribe multiple videos"""

    print("\n📁 Batch Transcription Example")
    print("=" * 40)

    input_dir = Path("path/to/input/videos")
    output_dir = Path("path/to/output/transcriptions")
    output_dir.mkdir(exist_ok=True)

    processor = create_speech_processor(model_size='base')

    # Find all video files
    video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.mov"))
    print(f"Found {len(video_files)} videos to transcribe")

    results = []
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_file.name}")

        try:
            # Transcribe
            result = processor.transcribe_video(str(video_file))

            # Save JSON and SRT
            base_name = video_file.stem
            json_path = output_dir / f"{base_name}.json"
            srt_path = output_dir / f"{base_name}.srt"

            processor.save_transcription(result, str(json_path), format="json")
            processor.save_transcription(result, str(srt_path), format="srt")

            results.append({
                "file": video_file.name,
                "language": result.language,
                "duration": result.duration,
                "segments": len(result.segments)
            })

            print(f"  ✅ Completed: {result.language}, {result.duration:.1f}s")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({
                "file": video_file.name,
                "error": str(e)
            })

    # Summary
    print(f"\n📊 Batch Transcription Summary:")
    successful = len([r for r in results if 'error' not in r])
    failed = len([r for r in results if 'error' in r])
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")


def example_real_time_transcription():
    """Simulate real-time transcription for short segments"""

    print("\n🔴 Real-time Style Transcription Example")
    print("=" * 40)

    input_video = "path/to/your/video.mp4"
    processor = create_speech_processor(model_size='tiny')  # Fast model

    # Process in 30-second chunks
    import moviepy.editor as mp

    video = mp.VideoFileClip(input_video)
    duration = video.duration
    video.close()

    chunk_duration = 30.0  # 30 seconds
    chunks = int(duration // chunk_duration) + (1 if duration % chunk_duration > 0 else 0)

    print(f"Processing {chunks} chunks of {chunk_duration}s each")

    all_segments = []
    for i in range(chunks):
        start_time = i * chunk_duration
        end_time = min((i + 1) * chunk_duration, duration)

        print(f"\n⏰ Chunk {i+1}/{chunks}: {start_time:.1f}s - {end_time:.1f}s")

        # Extract and transcribe chunk
        temp_audio = processor.extract_audio_from_video(
            input_video,
            start_time=start_time,
            end_time=end_time
        )

        result = processor.transcribe_audio(temp_audio)

        # Adjust timestamps to global time
        for segment in result.segments:
            segment.start += start_time
            segment.end += start_time
            all_segments.append(segment)

        print(f"  📝 Transcribed: {len(result.segments)} segments")

        # Cleanup
        os.unlink(temp_audio)

    print(f"\n✅ Total segments transcribed: {len(all_segments)}")


if __name__ == "__main__":
    print("🎤 AI Speech-to-Text Examples")
    print("=" * 50)

    print("\n📝 Instructions:")
    print("1. Update the file paths in each example function")
    print("2. Make sure your input video files exist")
    print("3. Run: python example_speech_to_text.py")
    print("4. Or import and call specific example functions")

    print("\n🚀 Available Examples:")
    print("- example_basic_transcription(): Simple video transcription")
    print("- example_subtitle_generation(): Generate SRT/VTT subtitles")
    print("- example_product_keyword_analysis(): Find product mentions")
    print("- example_segment_filtering(): Filter high-quality segments")
    print("- example_partial_transcription(): Transcribe first N minutes")
    print("- example_batch_transcription(): Process multiple videos")
    print("- example_real_time_transcription(): Chunk-based processing")

    print("\n💡 For Mac M1 Pro Optimization:")
    print("- Use 'base' model for best balance")
    print("- Use 'tiny' model for fastest processing")
    print("- Process in chunks for long videos")
    print("- Specify language if known for better accuracy")

    print("\n🎯 For Product Video Analysis:")
    print("- Use keyword detection to find product segments")
    print("- Filter by confidence for better quality")
    print("- Generate SRT files for video editing")
    print("- Save JSON for programmatic analysis")

    # Uncomment to run examples:
    # example_basic_transcription()
    # example_subtitle_generation()
    # example_product_keyword_analysis()
    # example_segment_filtering()
    # example_partial_transcription()
    # example_batch_transcription()
    # example_real_time_transcription()