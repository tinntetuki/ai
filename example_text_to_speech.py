#!/usr/bin/env python3
"""
Example usage of the text-to-speech functionality
"""

import asyncio
import os
from pathlib import Path
from src.ai_engine.tts_processor import create_tts_processor, VoiceProfile
from src.ai_engine.speech_processor import create_speech_processor


async def example_basic_tts():
    """Basic text-to-speech example"""

    print("🔊 Basic Text-to-Speech Example")
    print("=" * 40)

    # Create TTS processor
    tts_processor = create_tts_processor()

    # Chinese text example
    chinese_text = "欢迎来到我们的产品展示！这款产品具有出色的性能和优质的品质，现在购买还能享受特别优惠。"

    # Create Chinese voice profile
    chinese_voice = VoiceProfile(
        provider="edge",
        voice_name="zh-CN-XiaoxiaoNeural",
        language="zh-CN",
        gender="female",
        style="friendly",
        speed=1.0,
        pitch=1.0,
        volume=1.0
    )

    # Synthesize Chinese
    chinese_output = "output_chinese.wav"
    await tts_processor.synthesize_text(
        text=chinese_text,
        voice_profile=chinese_voice,
        output_path=chinese_output
    )

    print(f"✅ Chinese TTS completed: {chinese_output}")

    # English text example
    english_text = "Welcome to our amazing product showcase! This item features excellent performance and premium quality. Buy now for special discounts!"

    # Create English voice profile
    english_voice = VoiceProfile(
        provider="edge",
        voice_name="en-US-JennyNeural",
        language="en-US",
        gender="female",
        style="professional",
        speed=0.95,
        pitch=1.0,
        volume=1.0
    )

    # Synthesize English
    english_output = "output_english.wav"
    await tts_processor.synthesize_text(
        text=english_text,
        voice_profile=english_voice,
        output_path=english_output
    )

    print(f"✅ English TTS completed: {english_output}")


async def example_voice_styles():
    """Demonstrate different voice styles for product videos"""

    print("\n🎭 Voice Styles Example")
    print("=" * 40)

    tts_processor = create_tts_processor()

    product_text = "这是一款革命性的产品，采用最新技术，为您带来前所未有的体验。立即购买，享受限时优惠！"

    styles = [
        ("professional", "专业风格", "calm"),
        ("friendly", "友好风格", "friendly"),
        ("energetic", "充满活力", "excited"),
    ]

    for style_name, chinese_name, edge_style in styles:
        print(f"\n🎤 {chinese_name} ({style_name})")

        # Create voice profile for each style
        voice_profile = tts_processor.create_product_voice_profile(
            language="zh",
            voice_type=style_name,
            gender="female"
        )

        output_path = f"product_{style_name}.wav"

        await tts_processor.synthesize_text(
            text=product_text,
            voice_profile=voice_profile,
            output_path=output_path
        )

        print(f"  ✅ Generated: {output_path}")


async def example_voiceover_replacement():
    """Replace video audio with new voiceover"""

    print("\n🎬 Voiceover Replacement Example")
    print("=" * 40)

    # Input video path (replace with your actual video)
    input_video = "path/to/your/input_video.mp4"
    output_video = "path/to/your/output_with_voiceover.mp4"

    if not os.path.exists(input_video):
        print(f"⚠️ Video file not found: {input_video}")
        print("Please update the path to your actual video file")
        return

    try:
        # Create processors
        speech_processor = create_speech_processor(model_size="base")
        tts_processor = create_tts_processor()

        print("📝 Step 1: Transcribing original video...")

        # Transcribe original video
        transcription = speech_processor.transcribe_video(input_video)

        print(f"Original language: {transcription.language}")
        print(f"Duration: {transcription.duration:.2f} seconds")
        print(f"Segments: {len(transcription.segments)}")

        print("\n🎤 Step 2: Creating new voiceover...")

        # Create professional Chinese female voice
        voice_profile = tts_processor.create_product_voice_profile(
            language="zh",
            voice_type="professional",
            gender="female"
        )

        # Create voiceover audio
        audio_output = "new_voiceover.wav"
        await tts_processor.create_voiceover_from_transcription(
            transcription=transcription,
            voice_profile=voice_profile,
            output_path=audio_output,
            preserve_timing=True,
            add_pauses=True
        )

        print(f"✅ Voiceover audio created: {audio_output}")

        print("\n🎬 Step 3: Replacing video audio...")

        # Replace video audio
        success = await tts_processor.replace_video_audio(
            video_path=input_video,
            new_audio_path=audio_output,
            output_path=output_video,
            fade_duration=0.5
        )

        if success:
            print(f"✅ Video with new voiceover: {output_video}")
        else:
            print("❌ Failed to replace video audio")

    except Exception as e:
        print(f"❌ Error: {e}")


async def example_multilingual_voiceover():
    """Create voiceovers in multiple languages"""

    print("\n🌍 Multilingual Voiceover Example")
    print("=" * 40)

    tts_processor = create_tts_processor()

    # Product description in multiple languages
    texts = {
        "zh-CN": "这款产品采用先进技术，为您提供卓越的用户体验。现在购买享受特别折扣！",
        "en-US": "This product uses advanced technology to provide you with an exceptional user experience. Buy now for special discounts!",
        "ja-JP": "この製品は先進技術を使用し、優れたユーザーエクスペリエンスを提供します。今すぐ購入して特別割引をお楽しみください！",
        "ko-KR": "이 제품은 첨단 기술을 사용하여 뛰어난 사용자 경험을 제공합니다. 지금 구매하시면 특별 할인 혜택을 받으실 수 있습니다!"
    }

    # Voice configurations for different languages
    voices = {
        "zh-CN": ("zh-CN-XiaoxiaoNeural", "female"),
        "en-US": ("en-US-JennyNeural", "female"),
        "ja-JP": ("ja-JP-NanamiNeural", "female"),
        "ko-KR": ("ko-KR-SunHiNeural", "female")
    }

    for lang_code, text in texts.items():
        print(f"\n🎤 Creating {lang_code} voiceover...")

        voice_name, gender = voices[lang_code]

        voice_profile = VoiceProfile(
            provider="edge",
            voice_name=voice_name,
            language=lang_code,
            gender=gender,
            style="friendly",
            speed=1.0,
            pitch=1.0,
            volume=1.0
        )

        output_path = f"product_description_{lang_code}.wav"

        await tts_processor.synthesize_text(
            text=text,
            voice_profile=voice_profile,
            output_path=output_path
        )

        print(f"  ✅ {lang_code} voiceover: {output_path}")


async def example_batch_voiceover_generation():
    """Generate voiceovers for multiple product descriptions"""

    print("\n📁 Batch Voiceover Generation Example")
    print("=" * 40)

    tts_processor = create_tts_processor()

    # Product descriptions for different items
    products = [
        {
            "name": "智能手机",
            "description": "全新智能手机，配备最新处理器和高清摄像头，为您带来极致体验。限时特惠，立即抢购！",
            "voice_type": "energetic"
        },
        {
            "name": "无线耳机",
            "description": "高品质无线耳机，降噪功能出色，音质清晰纯净。适合运动和日常使用，现在下单享受优惠价格。",
            "voice_type": "professional"
        },
        {
            "name": "智能手表",
            "description": "功能强大的智能手表，健康监测、运动跟踪、智能提醒应有尽有。时尚设计，品质保证。",
            "voice_type": "friendly"
        }
    ]

    output_dir = Path("product_voiceovers")
    output_dir.mkdir(exist_ok=True)

    for i, product in enumerate(products, 1):
        print(f"\n[{i}/{len(products)}] Generating voiceover for: {product['name']}")

        # Create voice profile for this product
        voice_profile = tts_processor.create_product_voice_profile(
            language="zh",
            voice_type=product["voice_type"],
            gender="female"
        )

        # Generate filename
        safe_name = product["name"].replace(" ", "_")
        output_path = output_dir / f"{safe_name}_{product['voice_type']}.wav"

        # Synthesize
        await tts_processor.synthesize_text(
            text=product["description"],
            voice_profile=voice_profile,
            output_path=str(output_path)
        )

        print(f"  ✅ Generated: {output_path}")
        print(f"  🎤 Voice: {product['voice_type']} style")

    print(f"\n📊 Batch generation completed!")
    print(f"Generated {len(products)} voiceovers in: {output_dir}")


async def example_custom_voice_effects():
    """Demonstrate custom voice effects and modifications"""

    print("\n🎛️ Custom Voice Effects Example")
    print("=" * 40)

    tts_processor = create_tts_processor()

    base_text = "欢迎来到我们的特别优惠活动！这里有最好的产品和最优的价格！"

    effects = [
        {"name": "正常语速", "speed": 1.0, "pitch": 1.0, "volume": 1.0},
        {"name": "快速播报", "speed": 1.3, "pitch": 1.1, "volume": 1.0},
        {"name": "慢速详解", "speed": 0.8, "pitch": 0.95, "volume": 1.0},
        {"name": "高音调", "speed": 1.0, "pitch": 1.2, "volume": 1.0},
        {"name": "低音调", "speed": 1.0, "pitch": 0.8, "volume": 1.0},
        {"name": "大音量", "speed": 1.0, "pitch": 1.0, "volume": 1.3},
    ]

    for effect in effects:
        print(f"\n🎵 {effect['name']}")

        voice_profile = VoiceProfile(
            provider="edge",
            voice_name="zh-CN-XiaoxiaoNeural",
            language="zh-CN",
            gender="female",
            style="excited",
            speed=effect["speed"],
            pitch=effect["pitch"],
            volume=effect["volume"]
        )

        output_path = f"voice_effect_{effect['name']}.wav"

        await tts_processor.synthesize_text(
            text=base_text,
            voice_profile=voice_profile,
            output_path=output_path
        )

        print(f"  ✅ Generated: {output_path}")
        print(f"  ⚙️ Speed: {effect['speed']}, Pitch: {effect['pitch']}, Volume: {effect['volume']}")


async def example_tts_with_timing_analysis():
    """Analyze and optimize TTS timing for video content"""

    print("\n⏱️ TTS Timing Analysis Example")
    print("=" * 40)

    # Simulate transcription segments with timing
    from src.ai_engine.speech_processor import TranscriptionSegment, TranscriptionResult

    segments = [
        TranscriptionSegment(start=0.0, end=3.5, text="欢迎来到我们的产品展示", confidence=-0.1),
        TranscriptionSegment(start=4.0, end=8.2, text="这款产品具有革命性的功能", confidence=-0.05),
        TranscriptionSegment(start=9.0, end=12.5, text="现在购买享受特别优惠价格", confidence=-0.08),
        TranscriptionSegment(start=13.0, end=16.0, text="数量有限，抓紧时间下单", confidence=-0.03),
    ]

    transcription = TranscriptionResult(
        segments=segments,
        language="zh",
        duration=16.0,
        full_text=" ".join([seg.text for seg in segments])
    )

    tts_processor = create_tts_processor()

    print("📊 Original timing analysis:")
    total_chars = sum(len(seg.text) for seg in segments)
    total_time = transcription.duration

    print(f"  Total duration: {total_time:.1f}s")
    print(f"  Total characters: {total_chars}")
    print(f"  Average speaking rate: {total_chars/total_time:.1f} chars/second")

    print("\n🎤 Creating optimized voiceover...")

    # Create voice profile optimized for the original timing
    voice_profile = VoiceProfile(
        provider="edge",
        voice_name="zh-CN-XiaoxiaoNeural",
        language="zh-CN",
        gender="female",
        style="professional",
        speed=1.0,  # Will be auto-adjusted
        pitch=1.0,
        volume=1.0
    )

    # Create voiceover with timing preservation
    output_path = "timing_optimized_voiceover.wav"
    await tts_processor.create_voiceover_from_transcription(
        transcription=transcription,
        voice_profile=voice_profile,
        output_path=output_path,
        preserve_timing=True,
        add_pauses=True
    )

    print(f"✅ Timing-optimized voiceover: {output_path}")

    # Analyze new timing
    from pydub import AudioSegment
    audio = AudioSegment.from_wav(output_path)
    new_duration = len(audio) / 1000.0

    print(f"\n📈 Results:")
    print(f"  Original duration: {total_time:.1f}s")
    print(f"  New voiceover duration: {new_duration:.1f}s")
    print(f"  Time difference: {abs(new_duration - total_time):.1f}s")


async def main():
    """Run all examples"""

    print("🔊 AI Text-to-Speech Examples")
    print("=" * 50)

    print("\n📝 Instructions:")
    print("1. Update file paths in examples as needed")
    print("2. Ensure you have internet connection for Edge TTS")
    print("3. Run: python example_text_to_speech.py")
    print("4. Or import and call specific example functions")

    print("\n🚀 Available Examples:")
    print("- example_basic_tts(): Simple text synthesis")
    print("- example_voice_styles(): Different voice styles")
    print("- example_voiceover_replacement(): Replace video audio")
    print("- example_multilingual_voiceover(): Multiple languages")
    print("- example_batch_voiceover_generation(): Batch processing")
    print("- example_custom_voice_effects(): Voice effects")
    print("- example_tts_with_timing_analysis(): Timing optimization")

    print("\n💡 For Mac M1 Pro & Product Videos:")
    print("- Use Edge TTS for best quality")
    print("- Chinese: zh-CN-XiaoxiaoNeural (professional female)")
    print("- English: en-US-JennyNeural (clear female)")
    print("- Adjust speed: 0.9-1.1 for natural speech")
    print("- Use 'friendly' or 'professional' styles")

    print("\n🎯 Optimization Tips:")
    print("- Match voice style to product type")
    print("- Use timing preservation for dubbing")
    print("- Test different speeds for target audience")
    print("- Add fade effects for smooth transitions")

    # Uncomment to run examples:
    # await example_basic_tts()
    # await example_voice_styles()
    # await example_voiceover_replacement()
    # await example_multilingual_voiceover()
    # await example_batch_voiceover_generation()
    # await example_custom_voice_effects()
    # await example_tts_with_timing_analysis()


if __name__ == "__main__":
    asyncio.run(main())