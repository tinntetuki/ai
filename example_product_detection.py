#!/usr/bin/env python3
"""
Example usage of the product detection and annotation functionality
"""

import os
import json
from pathlib import Path
from src.ai_engine.product_detector import create_product_detector, AnnotationStyle
from src.ai_engine.speech_processor import create_speech_processor


def example_basic_product_detection():
    """Basic product detection in video"""

    print("🔍 Basic Product Detection Example")
    print("=" * 40)

    # Setup paths
    input_video = "path/to/your/product_video.mp4"  # Replace with actual path
    output_json = "path/to/your/detection_results.json"

    if not os.path.exists(input_video):
        print(f"⚠️ Video file not found: {input_video}")
        print("Please update the path to your actual video file")
        return

    # Create detector
    detector = create_product_detector(
        yolo_model="yolov8n.pt",  # Fast model for demo
        clip_model="ViT-B/32"
    )

    print("🔍 Detecting products in video...")

    # Detect products
    detections = detector.detect_products_in_video(
        video_path=input_video,
        sample_rate=1.0,  # 1 frame per second
        confidence_threshold=0.5,
        product_keywords=["phone", "laptop", "camera", "watch"]
    )

    print(f"✅ Found {len(detections)} product detections")

    # Print detection summary
    if detections:
        print("\n📊 Detection Summary:")
        product_counts = {}
        for detection in detections:
            if detection.class_name not in product_counts:
                product_counts[detection.class_name] = 0
            product_counts[detection.class_name] += 1

        for product, count in product_counts.items():
            print(f"  {product}: {count} detections")

        # Show first few detections
        print("\n🎯 Sample Detections:")
        for i, detection in enumerate(detections[:5]):
            print(f"  {i+1}. {detection.class_name} at {detection.timestamp:.1f}s "
                  f"(confidence: {detection.confidence:.2f})")

    # Save results
    results = {
        "total_detections": len(detections),
        "detections": [
            {
                "class_name": d.class_name,
                "confidence": d.confidence,
                "timestamp": d.timestamp,
                "bbox": {
                    "x1": d.bbox.x1, "y1": d.bbox.y1,
                    "x2": d.bbox.x2, "y2": d.bbox.y2
                },
                "description": d.description,
                "category": d.category
            }
            for d in detections
        ]
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Results saved to: {output_json}")


def example_annotated_video_creation():
    """Create annotated video with product highlights"""

    print("\n🎬 Annotated Video Creation Example")
    print("=" * 40)

    input_video = "path/to/your/product_video.mp4"
    output_video = "path/to/your/annotated_video.mp4"

    if not os.path.exists(input_video):
        print(f"⚠️ Video file not found: {input_video}")
        return

    # Create detector
    detector = create_product_detector()

    print("🔍 Detecting products...")

    # Detect products with specific keywords
    detections = detector.detect_products_in_video(
        video_path=input_video,
        sample_rate=2.0,  # 2 frames per second for better coverage
        confidence_threshold=0.6,  # Higher confidence for cleaner results
        product_keywords=["phone", "laptop", "tablet", "camera", "headphones"]
    )

    print(f"Found {len(detections)} detections")

    # Create custom annotation style
    style = AnnotationStyle(
        bbox_color=(0, 255, 255),  # Cyan bounding box
        bbox_thickness=3,
        text_color=(255, 255, 255),  # White text
        text_bg_color=(0, 0, 0),  # Black background
        font_scale=0.8,
        show_confidence=True,
        show_description=True
    )

    print("🎬 Creating annotated video...")

    # Create annotated video
    success = detector.annotate_video_with_products(
        video_path=input_video,
        detections=detections,
        output_path=output_video,
        style=style
    )

    if success:
        print(f"✅ Annotated video created: {output_video}")
    else:
        print("❌ Failed to create annotated video")


def example_product_tracking():
    """Demonstrate product tracking across video"""

    print("\n📊 Product Tracking Example")
    print("=" * 40)

    input_video = "path/to/your/product_demo.mp4"

    if not os.path.exists(input_video):
        print(f"⚠️ Video file not found: {input_video}")
        return

    # Create detector
    detector = create_product_detector()

    print("🔍 Detecting and tracking products...")

    # Detect products
    detections = detector.detect_products_in_video(
        video_path=input_video,
        sample_rate=2.0,
        confidence_threshold=0.5
    )

    print(f"Found {len(detections)} detections")

    # Create product tracks
    tracks = detector.create_product_tracks(
        detections,
        max_gap=2.0,  # Allow 2-second gaps
        min_track_length=1.0  # Minimum 1-second tracks
    )

    print(f"Created {len(tracks)} product tracks")

    # Analyze tracks
    print("\n📈 Track Analysis:")
    for i, track in enumerate(tracks[:10], 1):  # Show top 10 tracks
        duration = track.last_appearance - track.first_appearance
        print(f"  {i}. {track.class_name}")
        print(f"     Duration: {duration:.1f}s ({track.first_appearance:.1f}s - {track.last_appearance:.1f}s)")
        print(f"     Confidence: {track.confidence_avg:.2f}")
        print(f"     Importance: {track.importance_score:.2f}")
        print(f"     Detections: {len(track.detections)}")
        if track.description:
            print(f"     Description: {track.description}")
        print()

    # Find most important product
    if tracks:
        most_important = max(tracks, key=lambda x: x.importance_score)
        print(f"🏆 Most Important Product: {most_important.class_name}")
        print(f"   Importance Score: {most_important.importance_score:.2f}")
        print(f"   Screen Time: {most_important.last_appearance - most_important.first_appearance:.1f}s")


def example_speech_product_correlation():
    """Analyze correlation between speech and visual product detection"""

    print("\n🎤 Speech-Product Correlation Example")
    print("=" * 40)

    input_video = "path/to/your/product_presentation.mp4"

    if not os.path.exists(input_video):
        print(f"⚠️ Video file not found: {input_video}")
        return

    # Create processors
    detector = create_product_detector()
    speech_processor = create_speech_processor(model_size="base")

    print("📝 Transcribing speech...")

    # Transcribe speech
    transcription = speech_processor.transcribe_video(input_video)

    print(f"Language: {transcription.language}")
    print(f"Duration: {transcription.duration:.1f}s")
    print(f"Speech segments: {len(transcription.segments)}")

    print("\n🔍 Detecting products...")

    # Detect products
    detections = detector.detect_products_in_video(
        video_path=input_video,
        sample_rate=1.0,
        confidence_threshold=0.5
    )

    print(f"Visual detections: {len(detections)}")

    print("\n🔗 Analyzing correlation...")

    # Analyze correlation
    analysis = detector.analyze_products_with_speech(
        detections, transcription, sync_threshold=2.0
    )

    print("📊 Correlation Results:")
    print(f"  Sync Ratio: {analysis.get('sync_ratio', 0):.2%}")
    print(f"  Average Sync Score: {analysis.get('avg_sync_score', 0):.2f}")
    print(f"  Synced Segments: {analysis.get('synced_segments', 0)}/{analysis.get('total_speech_segments', 0)}")

    # Show correlations
    correlations = analysis.get('correlations', [])
    if correlations:
        print(f"\n🎯 Top Correlations:")
        high_sync = [c for c in correlations if c['sync_score'] > 0.5]
        for i, correlation in enumerate(high_sync[:5], 1):
            print(f"  {i}. {correlation['segment_start']:.1f}s-{correlation['segment_end']:.1f}s")
            print(f"     Text: \"{correlation['text'][:50]}...\"")
            print(f"     Products: {', '.join(correlation['products'])}")
            print(f"     Sync Score: {correlation['sync_score']:.2f}")
            print()

    # Product mentions analysis
    product_summary = analysis.get('product_summary', {})
    if product_summary:
        print("📦 Product Summary:")
        for product, stats in product_summary.items():
            print(f"  {product}:")
            print(f"    Detections: {stats['count']}")
            print(f"    Avg Confidence: {stats['avg_confidence']:.2f}")
            print(f"    Duration: {stats['duration']:.1f}s")
            print(f"    Category: {stats['category']}")
            print()


def example_custom_product_categories():
    """Detect specific product categories for different industries"""

    print("\n🏷️ Custom Product Categories Example")
    print("=" * 40)

    input_video = "path/to/your/category_video.mp4"

    if not os.path.exists(input_video):
        print(f"⚠️ Video file not found: {input_video}")
        return

    # Define category-specific keywords
    categories = {
        "electronics": ["phone", "laptop", "tablet", "camera", "headphones", "watch"],
        "fashion": ["shirt", "dress", "shoes", "bag", "hat", "jacket"],
        "beauty": ["makeup", "lipstick", "perfume", "skincare", "cosmetics"],
        "home": ["furniture", "lamp", "vase", "decoration", "pillow"],
        "sports": ["ball", "equipment", "shoes", "clothing"],
        "food": ["snack", "drink", "fruit", "bottle", "package"]
    }

    detector = create_product_detector()

    for category, keywords in categories.items():
        print(f"\n🔍 Detecting {category} products...")

        detections = detector.detect_products_in_video(
            video_path=input_video,
            sample_rate=1.0,
            confidence_threshold=0.6,
            product_keywords=keywords
        )

        if detections:
            print(f"  Found {len(detections)} {category} products:")
            product_types = set(d.class_name for d in detections)
            for product_type in product_types:
                count = sum(1 for d in detections if d.class_name == product_type)
                print(f"    - {product_type}: {count} detections")
        else:
            print(f"  No {category} products detected")


def example_batch_video_analysis():
    """Analyze multiple videos for product content"""

    print("\n📁 Batch Video Analysis Example")
    print("=" * 40)

    video_dir = Path("path/to/your/videos")
    output_dir = Path("path/to/your/analysis_results")
    output_dir.mkdir(exist_ok=True)

    if not video_dir.exists():
        print(f"⚠️ Video directory not found: {video_dir}")
        print("Please update the path to your video directory")
        return

    # Find video files
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.mov"))
    print(f"Found {len(video_files)} videos to analyze")

    if not video_files:
        print("No video files found in directory")
        return

    detector = create_product_detector()

    batch_results = {}

    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Analyzing: {video_file.name}")

        try:
            # Detect products
            detections = detector.detect_products_in_video(
                video_path=str(video_file),
                sample_rate=0.5,  # Lower rate for batch processing
                confidence_threshold=0.6
            )

            # Create tracks
            tracks = detector.create_product_tracks(detections)

            # Generate summary
            summary = detector.generate_product_summary(tracks)

            # Store results
            batch_results[video_file.name] = {
                "detections_count": len(detections),
                "tracks_count": len(tracks),
                "summary": summary
            }

            print(f"  ✅ {len(detections)} detections, {len(tracks)} tracks")

            # Save individual results
            result_file = output_dir / f"{video_file.stem}_analysis.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "video": video_file.name,
                    "detections": len(detections),
                    "tracks": len(tracks),
                    "summary": summary
                }, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            batch_results[video_file.name] = {"error": str(e)}

    # Save batch summary
    summary_file = output_dir / "batch_analysis_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, ensure_ascii=False, indent=2)

    print(f"\n📊 Batch Analysis Summary:")
    successful = len([r for r in batch_results.values() if 'error' not in r])
    failed = len([r for r in batch_results.values() if 'error' in r])
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📁 Results saved to: {output_dir}")


def example_real_time_product_monitoring():
    """Simulate real-time product detection monitoring"""

    print("\n🔴 Real-time Product Monitoring Example")
    print("=" * 40)

    input_video = "path/to/your/live_stream.mp4"

    if not os.path.exists(input_video):
        print(f"⚠️ Video file not found: {input_video}")
        return

    detector = create_product_detector(yolo_model="yolov8n.pt")  # Fast model

    print("🔍 Starting real-time monitoring...")

    # Simulate real-time processing by analyzing short segments
    import moviepy.editor as mp

    video = mp.VideoFileClip(input_video)
    duration = video.duration
    segment_length = 10.0  # 10-second segments

    segments = int(duration // segment_length) + (1 if duration % segment_length > 0 else 0)

    all_products = set()
    segment_results = []

    for i in range(segments):
        start_time = i * segment_length
        end_time = min((i + 1) * segment_length, duration)

        print(f"\n⏰ Segment {i+1}/{segments}: {start_time:.1f}s - {end_time:.1f}s")

        # Extract segment
        temp_segment = f"temp_segment_{i}.mp4"
        segment_clip = video.subclip(start_time, end_time)
        segment_clip.write_videofile(temp_segment, verbose=False, logger=None)

        # Detect products in segment
        detections = detector.detect_products_in_video(
            video_path=temp_segment,
            sample_rate=2.0,
            confidence_threshold=0.7
        )

        # Analyze segment
        segment_products = set(d.class_name for d in detections)
        new_products = segment_products - all_products

        if new_products:
            print(f"  🆕 New products detected: {', '.join(new_products)}")
            all_products.update(new_products)

        if detections:
            print(f"  📊 Segment summary: {len(detections)} detections")
            product_counts = {}
            for detection in detections:
                if detection.class_name not in product_counts:
                    product_counts[detection.class_name] = 0
                product_counts[detection.class_name] += 1

            for product, count in product_counts.items():
                print(f"    - {product}: {count}")

        segment_results.append({
            "segment": i + 1,
            "start_time": start_time,
            "end_time": end_time,
            "detections": len(detections),
            "products": list(segment_products),
            "new_products": list(new_products)
        })

        # Cleanup
        segment_clip.close()
        os.unlink(temp_segment)

    video.close()

    print(f"\n📈 Monitoring Complete:")
    print(f"  Total unique products: {len(all_products)}")
    print(f"  Products found: {', '.join(all_products)}")

    # Save monitoring results
    monitoring_results = {
        "total_segments": segments,
        "total_products": len(all_products),
        "unique_products": list(all_products),
        "segment_results": segment_results
    }

    with open("real_time_monitoring_results.json", 'w', encoding='utf-8') as f:
        json.dump(monitoring_results, f, ensure_ascii=False, indent=2)

    print("  📁 Results saved to: real_time_monitoring_results.json")


if __name__ == "__main__":
    print("🔍 AI Product Detection Examples")
    print("=" * 50)

    print("\n📝 Instructions:")
    print("1. Update the file paths in each example function")
    print("2. Make sure your input video files exist")
    print("3. Install required dependencies: ultralytics, clip-by-openai")
    print("4. Run: python example_product_detection.py")
    print("5. Or import and call specific example functions")

    print("\n🚀 Available Examples:")
    print("- example_basic_product_detection(): Simple product detection")
    print("- example_annotated_video_creation(): Create annotated videos")
    print("- example_product_tracking(): Track products across frames")
    print("- example_speech_product_correlation(): Sync with speech")
    print("- example_custom_product_categories(): Category-specific detection")
    print("- example_batch_video_analysis(): Analyze multiple videos")
    print("- example_real_time_product_monitoring(): Real-time monitoring")

    print("\n💡 For Mac M1 Pro Optimization:")
    print("- Use yolov8n.pt for fastest detection")
    print("- Set sample_rate=0.5-1.0 for good balance")
    print("- Use confidence_threshold=0.6+ for cleaner results")
    print("- Process in segments for long videos")

    print("\n🎯 For Product Videos:")
    print("- Focus on specific product keywords")
    print("- Use speech correlation for better insights")
    print("- Create annotated videos for presentations")
    print("- Track products to measure screen time")

    # Uncomment to run examples:
    # example_basic_product_detection()
    # example_annotated_video_creation()
    # example_product_tracking()
    # example_speech_product_correlation()
    # example_custom_product_categories()
    # example_batch_video_analysis()
    # example_real_time_product_monitoring()