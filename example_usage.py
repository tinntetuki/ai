#!/usr/bin/env python3
"""
Example usage of the video upscaling functionality
"""

import os
from pathlib import Path
from src.ai_engine.video_upscaler import create_upscaler


def example_basic_upscale():
    """Basic video upscaling example"""

    print("🎬 Basic Video Upscaling Example")
    print("=" * 40)

    # Setup paths
    input_video = "path/to/your/input_video.mp4"  # Replace with actual path
    output_video = "path/to/your/output_video_4x.mp4"

    # Create upscaler (4x scale)
    upscaler = create_upscaler(scale=4, model_name='RealESRGAN_x4plus')

    # Progress callback
    def show_progress(progress):
        print(f"Progress: {progress:.1f}%")

    # Upscale the entire video
    success = upscaler.upscale_video(
        input_path=input_video,
        output_path=output_video,
        progress_callback=show_progress
    )

    if success:
        print(f"✅ Success! Output saved to: {output_video}")
    else:
        print("❌ Upscaling failed")


def example_smart_segments():
    """Smart segment upscaling example - ideal for product videos"""

    print("\n🎯 Smart Segment Upscaling Example")
    print("=" * 40)

    input_video = "path/to/your/product_video.mp4"
    output_video = "path/to/your/product_video_enhanced.mp4"

    upscaler = create_upscaler(scale=4)

    # Get smart segments (product close-ups, text, faces)
    segments = upscaler.get_smart_segments(input_video, max_segments=3)
    print(f"Smart segments detected: {segments}")

    # Upscale only the important segments
    success = upscaler.upscale_video(
        input_path=input_video,
        output_path=output_video,
        target_segments=segments,
        progress_callback=lambda p: print(f"Processing segments: {p:.1f}%")
    )

    if success:
        print(f"✅ Enhanced video saved to: {output_video}")


def example_partial_upscale():
    """Upscale only first 30 seconds - good for testing"""

    print("\n⏱️ Partial Video Upscaling Example")
    print("=" * 40)

    input_video = "path/to/your/long_video.mp4"
    output_video = "path/to/your/first_30s_upscaled.mp4"

    upscaler = create_upscaler(scale=2)  # 2x for faster processing

    # Upscale only first 30 seconds
    success = upscaler.upscale_video(
        input_path=input_video,
        output_path=output_video,
        segment_duration=30.0,  # 30 seconds
        progress_callback=lambda p: print(f"Processing: {p:.1f}%")
    )

    if success:
        print(f"✅ First 30 seconds upscaled: {output_video}")


def example_batch_processing():
    """Process multiple videos"""

    print("\n📁 Batch Processing Example")
    print("=" * 40)

    input_dir = Path("path/to/input/videos")
    output_dir = Path("path/to/output/videos")
    output_dir.mkdir(exist_ok=True)

    upscaler = create_upscaler(scale=4)

    # Process all MP4 files in directory
    video_files = list(input_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} videos to process")

    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_file.name}")

        output_file = output_dir / f"upscaled_{video_file.name}"

        success = upscaler.upscale_video(
            input_path=str(video_file),
            output_path=str(output_file),
            segment_duration=60.0,  # First minute only
            progress_callback=lambda p: print(f"  Progress: {p:.1f}%")
        )

        if success:
            print(f"  ✅ Completed: {output_file}")
        else:
            print(f"  ❌ Failed: {video_file.name}")


if __name__ == "__main__":
    print("🎥 AI Video Upscaling Examples")
    print("=" * 50)

    # Check if input files exist before running examples
    print("\n📝 Instructions:")
    print("1. Update the file paths in each example function")
    print("2. Make sure your input video files exist")
    print("3. Run: python example_usage.py")
    print("4. Or import and call specific example functions")

    print("\n🚀 Available Examples:")
    print("- example_basic_upscale(): Full video 4x upscaling")
    print("- example_smart_segments(): Intelligent segment processing")
    print("- example_partial_upscale(): Process first 30 seconds only")
    print("- example_batch_processing(): Process multiple videos")

    print("\n💡 For Mac M1 Pro Optimization:")
    print("- Use scale=2 or scale=4 for best performance")
    print("- Process segments for videos >5 minutes")
    print("- Monitor memory usage with Activity Monitor")

    # Uncomment to run examples:
    # example_basic_upscale()
    # example_smart_segments()
    # example_partial_upscale()
    # example_batch_processing()