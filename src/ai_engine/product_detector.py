"""
Product Detection and Annotation Module
AI-powered product recognition and automatic annotation for video content
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
import json
import numpy as np
from loguru import logger

import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
import moviepy.editor as mp

# Computer Vision imports
try:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    logger.warning("Ultralytics not available, some features will be limited")

try:
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    logger.warning("CLIP not available, semantic understanding will be limited")

from .speech_processor import TranscriptionResult


@dataclass
class BoundingBox:
    """Bounding box coordinates"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        """Get area of bounding box"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def width(self) -> float:
        """Get width of bounding box"""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Get height of bounding box"""
        return self.y2 - self.y1


@dataclass
class ProductDetection:
    """Single product detection result"""
    bbox: BoundingBox
    class_name: str
    confidence: float
    frame_index: int
    timestamp: float
    features: Optional[np.ndarray] = None
    description: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None


@dataclass
class AnnotationStyle:
    """Annotation styling configuration"""
    bbox_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    bbox_thickness: int = 2
    text_color: Tuple[int, int, int] = (255, 255, 255)  # White
    text_bg_color: Tuple[int, int, int] = (0, 0, 0)  # Black
    font_scale: float = 0.7
    font_thickness: int = 2
    label_padding: int = 5
    show_confidence: bool = True
    show_description: bool = True


@dataclass
class ProductTrack:
    """Product tracking across frames"""
    track_id: int
    detections: List[ProductDetection]
    class_name: str
    confidence_avg: float
    first_appearance: float
    last_appearance: float
    description: Optional[str] = None
    importance_score: float = 0.0


class ProductDetector:
    """
    AI-powered product detection and annotation system
    Optimized for e-commerce and product demonstration videos
    """

    def __init__(
        self,
        yolo_model_path: str = "yolov8n.pt",
        clip_model_name: str = "ViT-B/32",
        device: Optional[str] = None
    ):
        """
        Initialize product detector

        Args:
            yolo_model_path: Path to YOLO model or model name
            clip_model_name: CLIP model name for semantic understanding
            device: Computing device (None for auto-detection)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model = None
        self.clip_model = None
        self.clip_preprocess = None

        # Load models
        self._load_models(yolo_model_path, clip_model_name)

        # Product categories optimized for e-commerce
        self.product_categories = {
            "electronics": ["phone", "laptop", "tablet", "watch", "camera", "headphones", "speaker"],
            "clothing": ["shirt", "pants", "dress", "shoes", "hat", "jacket", "bag"],
            "home": ["chair", "table", "lamp", "vase", "pillow", "decoration"],
            "beauty": ["cosmetics", "perfume", "skincare", "makeup"],
            "sports": ["ball", "equipment", "clothing", "shoes"],
            "food": ["snack", "drink", "fruit", "cake", "bottle"],
            "toys": ["toy", "doll", "game", "puzzle"],
            "books": ["book", "magazine", "notebook"]
        }

    def _load_models(self, yolo_model_path: str, clip_model_name: str):
        """Load detection and understanding models"""
        try:
            # Load YOLO model
            if HAS_ULTRALYTICS:
                logger.info(f"Loading YOLO model: {yolo_model_path}")
                self.yolo_model = YOLO(yolo_model_path)
                if hasattr(self.yolo_model, 'to'):
                    self.yolo_model.to(self.device)
                logger.info("YOLO model loaded successfully")
            else:
                logger.warning("YOLO model not available")

            # Load CLIP model
            if HAS_CLIP:
                logger.info(f"Loading CLIP model: {clip_model_name}")
                self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
                logger.info("CLIP model loaded successfully")
            else:
                logger.warning("CLIP model not available")

        except Exception as e:
            logger.error(f"Model loading failed: {e}")

    def detect_products_in_video(
        self,
        video_path: str,
        sample_rate: float = 1.0,
        confidence_threshold: float = 0.5,
        product_keywords: Optional[List[str]] = None
    ) -> List[ProductDetection]:
        """
        Detect products throughout video

        Args:
            video_path: Path to video file
            sample_rate: Frames per second to sample
            confidence_threshold: Minimum confidence for detections
            product_keywords: Specific product keywords to focus on

        Returns:
            List of product detections
        """
        try:
            logger.info(f"Detecting products in video: {video_path}")

            video = mp.VideoFileClip(video_path)
            fps = video.fps
            duration = video.duration

            # Calculate sampling interval
            frame_interval = max(1, int(fps / sample_rate))
            total_frames = int(duration * fps)

            detections = []
            frame_count = 0

            for frame_idx in range(0, total_frames, frame_interval):
                timestamp = frame_idx / fps

                try:
                    # Extract frame
                    frame = video.get_frame(timestamp)
                    frame_rgb = (frame * 255).astype(np.uint8)

                    # Detect objects in frame
                    frame_detections = self._detect_objects_in_frame(
                        frame_rgb, frame_idx, timestamp, confidence_threshold
                    )

                    # Filter for products if keywords provided
                    if product_keywords:
                        frame_detections = self._filter_by_keywords(
                            frame_detections, product_keywords
                        )

                    detections.extend(frame_detections)
                    frame_count += 1

                    if frame_count % 100 == 0:
                        logger.info(f"Processed {frame_count} frames, found {len(detections)} detections")

                except Exception as e:
                    logger.warning(f"Error processing frame {frame_idx}: {e}")
                    continue

            video.close()

            # Post-process detections
            detections = self._post_process_detections(detections)

            logger.info(f"Product detection completed: {len(detections)} detections found")
            return detections

        except Exception as e:
            logger.error(f"Video product detection failed: {e}")
            return []

    def _detect_objects_in_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        confidence_threshold: float
    ) -> List[ProductDetection]:
        """Detect objects in a single frame"""
        detections = []

        if not self.yolo_model:
            return detections

        try:
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False)

            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.data.cpu().numpy()

                    for box in boxes:
                        x1, y1, x2, y2, conf, class_id = box
                        class_id = int(class_id)

                        if conf < confidence_threshold:
                            continue

                        # Get class name
                        class_name = self.yolo_model.names.get(class_id, f"class_{class_id}")

                        # Check if it's a potential product
                        if self._is_product_class(class_name):
                            bbox = BoundingBox(
                                x1=float(x1),
                                y1=float(y1),
                                x2=float(x2),
                                y2=float(y2),
                                confidence=float(conf)
                            )

                            # Extract features if CLIP is available
                            features = None
                            description = None
                            if self.clip_model:
                                features, description = self._extract_clip_features(
                                    frame, bbox
                                )

                            detection = ProductDetection(
                                bbox=bbox,
                                class_name=class_name,
                                confidence=float(conf),
                                frame_index=frame_idx,
                                timestamp=timestamp,
                                features=features,
                                description=description,
                                category=self._get_product_category(class_name)
                            )

                            detections.append(detection)

        except Exception as e:
            logger.error(f"Frame detection failed: {e}")

        return detections

    def _extract_clip_features(
        self,
        frame: np.ndarray,
        bbox: BoundingBox
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Extract CLIP features and description from detected region"""
        if not self.clip_model or not self.clip_preprocess:
            return None, None

        try:
            # Crop detected region
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            cropped = frame[y1:y2, x1:x2]

            if cropped.size == 0:
                return None, None

            # Convert to PIL Image
            pil_image = Image.fromarray(cropped)

            # Preprocess for CLIP
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                features = image_features.cpu().numpy().flatten()

            # Generate description using predefined text prompts
            text_candidates = [
                "a product for sale",
                "an electronic device",
                "clothing item",
                "home decoration",
                "beauty product",
                "food item",
                "toy or game",
                "sports equipment"
            ]

            text_inputs = clip.tokenize(text_candidates).to(self.device)

            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_inputs)

                # Calculate similarities
                similarities = torch.cosine_similarity(image_features, text_features, dim=1)
                best_match_idx = similarities.argmax().item()

                description = text_candidates[best_match_idx]

            return features, description

        except Exception as e:
            logger.error(f"CLIP feature extraction failed: {e}")
            return None, None

    def _is_product_class(self, class_name: str) -> bool:
        """Check if detected class is likely a product"""
        # Common COCO classes that are products
        product_classes = {
            "bottle", "cup", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush", "wine glass",
            "fork", "knife", "spoon", "backpack", "umbrella", "handbag",
            "tie", "suitcase", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket"
        }

        return class_name.lower() in product_classes

    def _get_product_category(self, class_name: str) -> Optional[str]:
        """Get product category for detected class"""
        class_name_lower = class_name.lower()

        for category, items in self.product_categories.items():
            if any(item in class_name_lower for item in items):
                return category

        return "general"

    def _filter_by_keywords(
        self,
        detections: List[ProductDetection],
        keywords: List[str]
    ) -> List[ProductDetection]:
        """Filter detections based on product keywords"""
        if not keywords:
            return detections

        filtered = []
        keywords_lower = [k.lower() for k in keywords]

        for detection in detections:
            # Check class name
            if any(keyword in detection.class_name.lower() for keyword in keywords_lower):
                filtered.append(detection)
                continue

            # Check description if available
            if detection.description:
                if any(keyword in detection.description.lower() for keyword in keywords_lower):
                    filtered.append(detection)
                    continue

        return filtered

    def _post_process_detections(
        self,
        detections: List[ProductDetection]
    ) -> List[ProductDetection]:
        """Post-process detections to remove duplicates and improve quality"""
        if not detections:
            return detections

        # Sort by timestamp
        detections.sort(key=lambda x: x.timestamp)

        # Remove very similar detections (likely duplicates)
        filtered_detections = []
        for detection in detections:
            is_duplicate = False

            for existing in filtered_detections[-10:]:  # Check last 10
                if (abs(detection.timestamp - existing.timestamp) < 0.5 and
                    detection.class_name == existing.class_name and
                    self._bbox_overlap(detection.bbox, existing.bbox) > 0.7):
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_detections.append(detection)

        return filtered_detections

    def _bbox_overlap(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate IoU (Intersection over Union) between two bounding boxes"""
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = bbox1.area
        area2 = bbox2.area
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def create_product_tracks(
        self,
        detections: List[ProductDetection],
        max_gap: float = 2.0,
        min_track_length: float = 1.0
    ) -> List[ProductTrack]:
        """Create product tracks from individual detections"""
        try:
            tracks = []
            used_detections = set()

            # Group detections by class
            class_groups = {}
            for detection in detections:
                if detection.class_name not in class_groups:
                    class_groups[detection.class_name] = []
                class_groups[detection.class_name].append(detection)

            track_id = 0

            for class_name, class_detections in class_groups.items():
                class_detections.sort(key=lambda x: x.timestamp)

                for detection in class_detections:
                    if id(detection) in used_detections:
                        continue

                    # Start new track
                    track_detections = [detection]
                    used_detections.add(id(detection))

                    last_detection = detection

                    # Find continuous detections
                    for candidate in class_detections:
                        if id(candidate) in used_detections:
                            continue

                        if (candidate.timestamp - last_detection.timestamp <= max_gap and
                            self._bbox_overlap(candidate.bbox, last_detection.bbox) > 0.3):
                            track_detections.append(candidate)
                            used_detections.add(id(candidate))
                            last_detection = candidate

                    # Create track if long enough
                    if track_detections[-1].timestamp - track_detections[0].timestamp >= min_track_length:
                        confidence_avg = np.mean([d.confidence for d in track_detections])

                        track = ProductTrack(
                            track_id=track_id,
                            detections=track_detections,
                            class_name=class_name,
                            confidence_avg=confidence_avg,
                            first_appearance=track_detections[0].timestamp,
                            last_appearance=track_detections[-1].timestamp,
                            description=track_detections[0].description,
                            importance_score=self._calculate_track_importance(track_detections)
                        )

                        tracks.append(track)
                        track_id += 1

            # Sort tracks by importance
            tracks.sort(key=lambda x: x.importance_score, reverse=True)

            logger.info(f"Created {len(tracks)} product tracks")
            return tracks

        except Exception as e:
            logger.error(f"Track creation failed: {e}")
            return []

    def _calculate_track_importance(self, detections: List[ProductDetection]) -> float:
        """Calculate importance score for a product track"""
        if not detections:
            return 0.0

        # Factors: duration, confidence, size, frequency
        duration = detections[-1].timestamp - detections[0].timestamp
        avg_confidence = np.mean([d.confidence for d in detections])
        avg_size = np.mean([d.bbox.area for d in detections])
        frequency = len(detections) / max(duration, 1.0)

        # Normalize and combine scores
        score = (
            min(duration / 10.0, 1.0) * 0.3 +  # Duration (up to 10s)
            avg_confidence * 0.4 +              # Confidence
            min(avg_size / 50000, 1.0) * 0.2 +  # Size (normalized)
            min(frequency / 2.0, 1.0) * 0.1     # Frequency
        )

        return score

    def annotate_video_with_products(
        self,
        video_path: str,
        detections: List[ProductDetection],
        output_path: str,
        style: Optional[AnnotationStyle] = None,
        show_tracks: bool = True
    ) -> bool:
        """
        Create annotated video with product detections

        Args:
            video_path: Input video path
            detections: Product detections
            output_path: Output video path
            style: Annotation styling
            show_tracks: Show product tracking

        Returns:
            True if successful
        """
        try:
            logger.info("Creating annotated video")

            if style is None:
                style = AnnotationStyle()

            # Group detections by frame
            frame_detections = {}
            for detection in detections:
                frame_idx = detection.frame_index
                if frame_idx not in frame_detections:
                    frame_detections[frame_idx] = []
                frame_detections[frame_idx].append(detection)

            # Load video
            video = mp.VideoFileClip(video_path)

            def annotate_frame(get_frame, t):
                frame = get_frame(t)
                frame_idx = int(t * video.fps)

                if frame_idx in frame_detections:
                    # Convert to uint8 if needed
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)

                    # Draw annotations
                    frame = self._draw_annotations(
                        frame, frame_detections[frame_idx], style
                    )

                    # Convert back to float
                    frame = frame.astype(np.float32) / 255.0

                return frame

            # Apply annotations
            annotated_video = video.fl(annotate_frame, apply_to=['mask'])

            # Export
            annotated_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                fps=video.fps
            )

            # Cleanup
            video.close()
            annotated_video.close()

            logger.info(f"Annotated video created: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Video annotation failed: {e}")
            return False

    def _draw_annotations(
        self,
        frame: np.ndarray,
        detections: List[ProductDetection],
        style: AnnotationStyle
    ) -> np.ndarray:
        """Draw annotations on frame"""
        annotated_frame = frame.copy()

        for detection in detections:
            # Draw bounding box
            x1, y1 = int(detection.bbox.x1), int(detection.bbox.y1)
            x2, y2 = int(detection.bbox.x2), int(detection.bbox.y2)

            cv2.rectangle(
                annotated_frame,
                (x1, y1), (x2, y2),
                style.bbox_color,
                style.bbox_thickness
            )

            # Prepare label text
            label_parts = [detection.class_name]

            if style.show_confidence:
                label_parts.append(f"{detection.confidence:.2f}")

            if style.show_description and detection.description:
                label_parts.append(detection.description)

            label_text = " | ".join(label_parts)

            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                style.font_scale,
                style.font_thickness
            )

            # Draw text background
            label_y = y1 - text_height - style.label_padding
            if label_y < 0:
                label_y = y2 + text_height + style.label_padding

            cv2.rectangle(
                annotated_frame,
                (x1, label_y - text_height - style.label_padding),
                (x1 + text_width + style.label_padding * 2, label_y + style.label_padding),
                style.text_bg_color,
                -1
            )

            # Draw text
            cv2.putText(
                annotated_frame,
                label_text,
                (x1 + style.label_padding, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                style.font_scale,
                style.text_color,
                style.font_thickness
            )

        return annotated_frame

    def analyze_products_with_speech(
        self,
        detections: List[ProductDetection],
        transcription: TranscriptionResult,
        sync_threshold: float = 2.0
    ) -> Dict[str, Any]:
        """
        Analyze correlation between visual product detection and speech

        Args:
            detections: Product detections
            transcription: Speech transcription
            sync_threshold: Time threshold for sync analysis

        Returns:
            Analysis results
        """
        try:
            logger.info("Analyzing product-speech correlation")

            # Group detections by product class
            product_detections = {}
            for detection in detections:
                if detection.class_name not in product_detections:
                    product_detections[detection.class_name] = []
                product_detections[detection.class_name].append(detection)

            # Analyze correlation with speech segments
            correlations = []

            for segment in transcription.segments:
                segment_start = segment.start
                segment_end = segment.end
                segment_text = segment.text.lower()

                # Find detections within time window
                nearby_detections = []
                for detection in detections:
                    if (segment_start - sync_threshold <= detection.timestamp <= segment_end + sync_threshold):
                        nearby_detections.append(detection)

                if nearby_detections:
                    # Check for product mentions in text
                    mentioned_products = []
                    for detection in nearby_detections:
                        class_name = detection.class_name.lower()
                        if class_name in segment_text or any(word in segment_text for word in class_name.split()):
                            mentioned_products.append(detection)

                    correlation = {
                        "segment_start": segment_start,
                        "segment_end": segment_end,
                        "text": segment.text,
                        "visual_detections": len(nearby_detections),
                        "mentioned_products": len(mentioned_products),
                        "sync_score": len(mentioned_products) / max(len(nearby_detections), 1),
                        "products": [d.class_name for d in nearby_detections]
                    }

                    correlations.append(correlation)

            # Calculate overall statistics
            total_segments = len(transcription.segments)
            synced_segments = len([c for c in correlations if c["sync_score"] > 0])
            avg_sync_score = np.mean([c["sync_score"] for c in correlations]) if correlations else 0

            analysis_result = {
                "total_detections": len(detections),
                "unique_products": len(product_detections),
                "total_speech_segments": total_segments,
                "synced_segments": synced_segments,
                "sync_ratio": synced_segments / max(total_segments, 1),
                "avg_sync_score": avg_sync_score,
                "correlations": correlations,
                "product_summary": {
                    product: {
                        "count": len(dets),
                        "avg_confidence": np.mean([d.confidence for d in dets]),
                        "duration": max([d.timestamp for d in dets]) - min([d.timestamp for d in dets]),
                        "category": dets[0].category if dets else None
                    }
                    for product, dets in product_detections.items()
                }
            }

            logger.info(f"Product-speech analysis completed: {sync_ratio:.2f} sync ratio")
            return analysis_result

        except Exception as e:
            logger.error(f"Product-speech analysis failed: {e}")
            return {}

    def generate_product_summary(
        self,
        tracks: List[ProductTrack],
        analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive product summary"""
        try:
            if not tracks:
                return {"error": "No product tracks provided"}

            # Basic statistics
            total_products = len(tracks)
            total_duration = sum(track.last_appearance - track.first_appearance for track in tracks)
            avg_confidence = np.mean([track.confidence_avg for track in tracks])

            # Category breakdown
            categories = {}
            for track in tracks:
                category = track.detections[0].category if track.detections else "unknown"
                if category not in categories:
                    categories[category] = 0
                categories[category] += 1

            # Top products by importance
            top_products = sorted(tracks, key=lambda x: x.importance_score, reverse=True)[:5]

            summary = {
                "overview": {
                    "total_products": total_products,
                    "total_duration": total_duration,
                    "avg_confidence": avg_confidence,
                    "categories": categories
                },
                "top_products": [
                    {
                        "class_name": track.class_name,
                        "importance_score": track.importance_score,
                        "duration": track.last_appearance - track.first_appearance,
                        "first_seen": track.first_appearance,
                        "description": track.description,
                        "category": track.detections[0].category if track.detections else None
                    }
                    for track in top_products
                ],
                "timeline": [
                    {
                        "timestamp": track.first_appearance,
                        "event": "product_appearance",
                        "product": track.class_name,
                        "confidence": track.confidence_avg
                    }
                    for track in tracks
                ]
            }

            # Add speech analysis if available
            if analysis:
                summary["speech_correlation"] = {
                    "sync_ratio": analysis.get("sync_ratio", 0),
                    "avg_sync_score": analysis.get("avg_sync_score", 0),
                    "synced_segments": analysis.get("synced_segments", 0)
                }

            return summary

        except Exception as e:
            logger.error(f"Product summary generation failed: {e}")
            return {"error": str(e)}


def create_product_detector(
    yolo_model: str = "yolov8n.pt",
    clip_model: str = "ViT-B/32",
    device: Optional[str] = None
) -> ProductDetector:
    """
    Factory function to create ProductDetector instance

    Args:
        yolo_model: YOLO model path or name
        clip_model: CLIP model name
        device: Computing device

    Returns:
        ProductDetector instance
    """
    return ProductDetector(
        yolo_model_path=yolo_model,
        clip_model_name=clip_model,
        device=device
    )