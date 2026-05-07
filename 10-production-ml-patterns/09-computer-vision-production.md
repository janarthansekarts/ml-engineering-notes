# Computer Vision in Production

## The Problem / Why This Matters

Computer vision (CV) in production presents unique engineering challenges compared to other ML domains: models are large (vision transformers and CNNs have hundreds of millions of parameters), inputs are high-dimensional (images: millions of pixels, video: 30+ frames/second), preprocessing is critical (resizing, normalization, augmentation pipelines), and inference is compute-intensive (GPU required for real-time). In 2026, production CV powers autonomous vehicles, medical imaging, quality inspection in manufacturing, retail analytics, content moderation, document processing, and augmented reality. The engineering patterns — efficient model serving with TensorRT optimization, video processing pipelines, edge deployment for latency-sensitive applications, and multi-model cascades — are distinct from text-based ML. Key challenges include: handling variable image resolutions, GPU memory management for batch processing, video stream handling at 30 FPS (Frames Per Second), deployment to edge devices (mobile, IoT, cameras), and building annotation pipelines for training data. Modern CV increasingly uses vision-language models (CLIP, GPT-4V/5V, Gemini vision) that understand both images and text, enabling zero-shot classification and visual question answering without task-specific training.

---

## The Analogy

Think of production computer vision like a surveillance security team:

- **Image classification** = A guard who looks at a snapshot and says "this is a parking lot" or "this is an office." Simple judgment, one label per image.
- **Object detection** = A guard who circles every person, vehicle, and package in the image and labels each one. More work per image, but much more useful information.
- **Video analysis** = A team of guards watching live feeds 24/7, tracking objects across frames, detecting events (person entering restricted area), and alerting in real-time. Enormous data volume, needs to keep up with the stream.
- **Edge deployment** = Putting a smart camera at the entrance that does all analysis locally (no cloud), because sending video to the cloud is too slow and expensive. Constrained hardware, optimized models.

---

## Deep Dive

### Computer Vision Tasks in Production

```yaml
CV_Tasks:
  image_classification:
    what: "Assign one or more labels to an image"
    examples: "Product categorization, medical diagnosis, content moderation"
    models: "EfficientNet-V2, ViT (Vision Transformer), ConvNeXt"
    latency: "5-20ms per image (GPU)"
    
  object_detection:
    what: "Locate and classify objects within an image (bounding boxes)"
    examples: "Autonomous driving, retail shelf analysis, safety monitoring"
    models: "YOLOv8/v9, DETR, RT-DETR, Co-DETR"
    latency: "10-50ms per image (GPU)"
    
  semantic_segmentation:
    what: "Label every pixel (which class does each pixel belong to?)"
    examples: "Medical imaging (tumor boundaries), autonomous driving (road vs. sidewalk)"
    models: "Segment Anything (SAM 2), Mask2Former, DeepLabV3+"
    latency: "20-100ms per image (GPU)"
    
  video_analysis:
    what: "Process video streams (tracking, action recognition, event detection)"
    examples: "Traffic monitoring, sports analytics, security surveillance"
    components:
      tracking: "ByteTrack, BoT-SORT (track objects across frames)"
      action_recognition: "VideoMAE, TimeSformer (classify activities)"
      anomaly_detection: "Detect unusual events in video stream"
    throughput: "30 FPS real-time processing required"
    
  ocr_document:
    what: "Extract text and structure from documents/images"
    examples: "Invoice processing, receipt scanning, form digitization"
    models: "PaddleOCR, DocTR, Azure Document Intelligence, Google Document AI"
    
  vision_language:
    what: "Understand images with text prompts (visual QA, captioning)"
    examples: "Image search, content description, accessibility alt-text"
    models: "CLIP, GPT-5V, Gemini 2.5 Pro Vision, LLaVA-Next"
    latency: "100ms-2s (depends on model size)"
```

### Production Architecture

```yaml
Architecture:
  image_processing_pipeline:
    ingestion:
      sources: "API upload, camera stream, S3 bucket, message queue"
      validation: "Check format, resolution, file size, corruption"
      
    preprocessing:
      resize: "Normalize to model input size (224x224, 640x640)"
      normalize: "Pixel normalization (ImageNet mean/std or model-specific)"
      augmentation: "Test-time augmentation (TTA) for better accuracy"
      batching: "Group images for GPU batch inference"
      
    inference:
      model_serving: "Triton Inference Server (multi-model, dynamic batching)"
      optimization: "TensorRT (NVIDIA), OpenVINO (Intel), CoreML (Apple)"
      gpu_management: "Model placement, memory management, multi-model sharing"
      
    postprocessing:
      nms: "Non-Maximum Suppression (deduplicate overlapping detections)"
      filtering: "Confidence threshold, class filtering"
      tracking: "Link detections across frames (video)"
      
  video_processing:
    stream_ingestion: "RTSP, WebRTC, or decoded from file"
    frame_sampling: "Process every Nth frame (not all 30 FPS needed)"
    keyframe_detection: "Process only when scene changes significantly"
    tracking: "Track objects between processed frames"
    event_detection: "Trigger alerts on specific events"
```

### Implementation

```python
# Production computer vision pipeline

"""
Production CV system handling image classification, object detection,
and video analysis with GPU-optimized serving.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ImageInput:
    """Image input for processing."""
    image_id: str
    pixels: np.ndarray  # (H, W, C) uint8
    metadata: Dict = None


@dataclass
class Detection:
    """Object detection result."""
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized
    
    
@dataclass
class CVResult:
    """Computer vision processing result."""
    image_id: str
    task: str
    detections: List[Detection] = None
    classification: str = None
    confidence: float = 0.0
    latency_ms: float = 0.0


class ProductionCVPipeline:
    """
    Production computer vision pipeline with:
    - Preprocessing (resize, normalize)
    - Batched GPU inference
    - Post-processing (NMS, filtering)
    - Multi-model support
    """
    
    def __init__(
        self,
        detection_model,
        classification_model,
        preprocessing_config: Dict,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        batch_size: int = 16,
    ):
        self.detector = detection_model
        self.classifier = classification_model
        self.config = preprocessing_config
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.batch_size = batch_size
    
    def detect_objects(self, images: List[ImageInput]) -> List[CVResult]:
        """
        Run object detection on batch of images.
        
        Pipeline: preprocess → batch → inference → NMS → filter
        """
        results = []
        
        # Process in batches for GPU efficiency
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            
            start = time.perf_counter()
            
            # Preprocess batch
            preprocessed = [self._preprocess(img) for img in batch]
            batch_tensor = np.stack(preprocessed)
            
            # GPU inference (batched)
            raw_detections = self.detector.predict_batch(batch_tensor)
            
            # Post-process each image's detections
            for img, detections in zip(batch, raw_detections):
                # Apply NMS (Non-Maximum Suppression)
                filtered = self._apply_nms(detections)
                
                # Apply confidence threshold
                filtered = [d for d in filtered if d.confidence >= self.confidence_threshold]
                
                latency = (time.perf_counter() - start) * 1000 / len(batch)
                
                results.append(CVResult(
                    image_id=img.image_id,
                    task="detection",
                    detections=filtered,
                    confidence=max((d.confidence for d in filtered), default=0),
                    latency_ms=latency,
                ))
        
        return results
    
    def classify(self, images: List[ImageInput]) -> List[CVResult]:
        """
        Classify images (single label per image).
        """
        results = []
        
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            
            start = time.perf_counter()
            
            preprocessed = [self._preprocess(img) for img in batch]
            batch_tensor = np.stack(preprocessed)
            
            # Batch classification
            predictions = self.classifier.predict_batch(batch_tensor)
            
            for img, (label, confidence) in zip(batch, predictions):
                latency = (time.perf_counter() - start) * 1000 / len(batch)
                results.append(CVResult(
                    image_id=img.image_id,
                    task="classification",
                    classification=label,
                    confidence=confidence,
                    latency_ms=latency,
                ))
        
        return results
    
    def _preprocess(self, image: ImageInput) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Steps:
        1. Resize to model input size
        2. Convert color space if needed
        3. Normalize pixel values
        4. Transpose to (C, H, W) format
        """
        target_size = self.config.get("input_size", (640, 640))
        mean = self.config.get("mean", [0.485, 0.456, 0.406])
        std = self.config.get("std", [0.229, 0.224, 0.225])
        
        # Resize (preserve aspect ratio with padding)
        img = self._resize_with_padding(image.pixels, target_size)
        
        # Normalize (uint8 → float32, ImageNet normalization)
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        
        # Transpose: (H, W, C) → (C, H, W)
        img = img.transpose(2, 0, 1)
        
        return img
    
    def _resize_with_padding(
        self, image: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Resize image preserving aspect ratio, pad to target size."""
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Compute scale factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize (in production: use cv2 or PIL)
        # Simplified — in production use cv2.resize
        resized = image  # Placeholder for actual resize
        
        # Pad to target size (letterbox)
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8) + 114
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized[:new_h, :new_w]
        
        return padded
    
    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """
        Non-Maximum Suppression: remove overlapping detections.
        
        When multiple bounding boxes overlap significantly,
        keep only the one with highest confidence.
        """
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        sorted_dets = sorted(detections, key=lambda d: -d.confidence)
        keep = []
        
        while sorted_dets:
            best = sorted_dets.pop(0)
            keep.append(best)
            
            # Remove detections that overlap too much with best
            sorted_dets = [
                d for d in sorted_dets
                if self._iou(best.bbox, d.bbox) < self.nms_threshold
                or d.class_name != best.class_name
            ]
        
        return keep
    
    def _iou(self, box1, box2) -> float:
        """Compute IoU (Intersection over Union) between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-6)


class VideoProcessingPipeline:
    """
    Video stream processing for real-time analysis.
    
    Key patterns:
    - Frame sampling (not every frame needs processing)
    - Object tracking (link detections across frames)
    - Event detection (alert on specific conditions)
    - Keyframe processing (only process when scene changes)
    """
    
    def __init__(
        self,
        detector,
        tracker,
        frame_skip: int = 5,  # Process every 5th frame
        event_rules: Dict = None,
    ):
        self.detector = detector
        self.tracker = tracker
        self.frame_skip = frame_skip
        self.event_rules = event_rules or {}
        self.frame_count = 0
        self.tracked_objects = {}
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Process a single video frame.
        
        Not every frame needs full detection.
        Pattern: detect on keyframes, track between them.
        """
        self.frame_count += 1
        
        if self.frame_count % self.frame_skip == 0:
            # Full detection on this frame
            detections = self.detector.detect(frame)
            
            # Update tracker with new detections
            tracked = self.tracker.update(detections)
            self.tracked_objects = tracked
            
            # Check event rules
            events = self._check_events(tracked)
            
            return {
                "frame_number": self.frame_count,
                "detections": len(detections),
                "tracked_objects": len(tracked),
                "events": events,
                "processing_type": "full_detection",
            }
        else:
            # Lightweight tracking only (predict object positions)
            predicted = self.tracker.predict()
            
            return {
                "frame_number": self.frame_count,
                "tracked_objects": len(predicted),
                "processing_type": "tracking_only",
            }
    
    def _check_events(self, tracked_objects: Dict) -> List[Dict]:
        """
        Check for events (e.g., person in restricted zone, crowd forming).
        """
        events = []
        
        for rule_name, rule_fn in self.event_rules.items():
            triggered = rule_fn(tracked_objects)
            if triggered:
                events.append({
                    "event_type": rule_name,
                    "timestamp": time.time(),
                    "objects_involved": triggered,
                })
        
        return events


class EdgeDeployment:
    """
    Deploy CV models to edge devices (cameras, mobile, IoT).
    
    Constraints:
    - Limited compute (no data center GPU)
    - Limited memory (1-8 GB RAM)
    - Power budget (battery-operated devices)
    - Network latency (can't send every frame to cloud)
    
    Optimization techniques:
    - Model quantization (INT8, INT4)
    - Model pruning (remove unimportant neurons)
    - Knowledge distillation (train small model to mimic large)
    - Hardware-specific compilation (TFLite, CoreML, ONNX)
    """
    
    def __init__(self, model_format: str = "onnx"):
        self.model_format = model_format
    
    @staticmethod
    def optimization_guide() -> Dict:
        """Guide for edge deployment optimization."""
        return {
            "mobile_phone": {
                "frameworks": ["TFLite", "CoreML", "ONNX Runtime Mobile"],
                "model_size": "< 50MB",
                "inference_time": "< 100ms",
                "techniques": [
                    "INT8 quantization (2-4x speedup)",
                    "MobileNet/EfficientNet-Lite architecture",
                    "Input resolution reduction (320x320 instead of 640x640)",
                ],
            },
            "edge_gpu": {
                "devices": ["NVIDIA Jetson Orin", "Intel NCS2", "Google Coral"],
                "frameworks": ["TensorRT", "OpenVINO", "Edge TPU"],
                "model_size": "< 500MB",
                "inference_time": "< 30ms",
                "techniques": [
                    "FP16 inference (TensorRT)",
                    "Model pruning (50% sparsity)",
                    "Multi-stream processing",
                ],
            },
            "browser_webgpu": {
                "frameworks": ["ONNX Runtime Web", "TensorFlow.js", "WebGPU"],
                "model_size": "< 20MB (download over network)",
                "inference_time": "< 200ms",
                "techniques": [
                    "INT8 quantization",
                    "Tiny model architectures",
                    "WebGPU compute shaders",
                ],
            },
        }
```

### Model Optimization for Serving

```yaml
Model_Optimization:
  tensorrt:
    what: "NVIDIA compiler for optimized GPU inference"
    speedup: "2-5x over native PyTorch"
    techniques: "Layer fusion, kernel auto-tuning, precision calibration"
    supported: "All NVIDIA GPUs (T4, A10G, L4, A100, H100)"
    
  quantization:
    int8:
      speedup: "2-4x"
      accuracy_loss: "< 1% for classification, < 2% for detection"
      calibration: "Run 100-1000 representative images to calibrate"
    int4:
      speedup: "4-8x"
      accuracy_loss: "2-5% (acceptable for many applications)"
      
  onnx_runtime:
    what: "Cross-platform inference runtime"
    benefit: "Deploy same model on CPU, GPU, mobile, web"
    optimizations: "Graph optimization, operator fusion, memory planning"
    
  batching:
    dynamic: "Collect requests, batch, process together (Triton)"
    benefit: "10x throughput on GPU (amortize overhead)"
    config: "Max batch size: 32, max delay: 10ms"
```

---

## How It Works in Practice

### Manufacturing Quality Inspection

```yaml
Quality_Inspection:
  use_case: "Detect defects on manufactured parts (assembly line)"
  
  setup:
    cameras: "4 high-resolution cameras per station (12 megapixels)"
    throughput: "1 part per second (4 images per part)"
    latency: "< 500ms total (decision before part moves to next station)"
    
  pipeline:
    capture: "Cameras triggered by sensor (part in position)"
    preprocess: "Crop to ROI, normalize lighting, resize to 1024x1024"
    detect: "Object detection model finds defect regions"
    classify: "Classification model grades defect severity (minor/major/critical)"
    decision: "Critical defect → reject part, minor → flag for review"
    
  model:
    architecture: "YOLOv9 fine-tuned on factory-specific defect data"
    training_data: "50,000 annotated images (5,000 with defects)"
    augmentation: "Rotation, brightness variation, synthetic defects"
    
  edge_deployment:
    hardware: "NVIDIA Jetson Orin (per inspection station)"
    optimization: "TensorRT FP16 (20ms inference per image)"
    benefit: "No cloud latency, works without internet"
```

---

## Interview Tip

> When asked about computer vision in production: "I structure CV systems around three concerns: preprocessing, efficient inference, and deployment target. For preprocessing: images vary wildly in production (different resolutions, lighting, orientations). I standardize with resize-and-pad (letterbox) to preserve aspect ratio, normalize to model-specific statistics (ImageNet mean/std for pretrained models), and validate inputs (corrupt files, wrong formats). For inference efficiency: I use Triton Inference Server with TensorRT optimization (2-5x speedup over native PyTorch). Dynamic batching (collect requests for 10ms, process together) gives 10x throughput on GPU. For object detection: Non-Maximum Suppression (NMS) removes overlapping detections. I tune confidence threshold based on precision/recall needs (higher threshold = fewer false positives, lower recall). For video: I don't process every frame — frame skipping (every 5th frame) with object tracking (ByteTrack) between detections gives real-time performance (30 FPS) with detection-quality results. For edge deployment: TensorRT on Jetson Orin (20ms inference), or ONNX + INT8 quantization for mobile (100ms on-device). Model architecture choice depends on target: YOLO for detection (best speed/accuracy trade-off in 2026), EfficientNet/ViT for classification, SAM 2 for segmentation. For vision-language tasks (zero-shot classification, visual QA): CLIP or GPT-5V/Gemini Vision — no training data needed, but higher latency and cost."

---

## Common Mistakes

1. **Not preserving aspect ratio** — Stretching images to model input size. Distorted proportions confuse the model. Solution: resize with padding (letterbox) — scale to fit, pad remaining area with neutral color.

2. **Processing every video frame** — Running full detection on all 30 FPS. Wastes 5x compute (adjacent frames are nearly identical). Solution: process every Nth frame (N=3-5), track objects between detections. Same result at 20% compute cost.

3. **Ignoring preprocessing consistency** — Training with one normalization (ImageNet mean/std), deploying with different normalization (or none). Model accuracy drops silently. Solution: serialize preprocessing config with model. Validate preprocessing produces same statistics as training.

4. **Not tuning confidence threshold** — Using default 0.5 threshold for all deployments. Medical imaging needs 0.3 (high recall, don't miss disease), spam filtering needs 0.8 (high precision, don't block legitimate). Solution: tune threshold on validation set for your specific precision/recall trade-off.

5. **Sending all images to cloud** — Uploading every camera frame for cloud processing. High bandwidth cost, latency, and privacy concerns. Solution: process at edge (Jetson, Coral), only send alerts/events to cloud. Edge-first architecture for privacy-sensitive CV.

---

## Key Takeaways

- Production CV: preprocessing + optimized inference + post-processing + deployment
- Object detection: YOLO v8/v9 (best speed/accuracy), RT-DETR (transformer-based)
- Video: frame skipping + object tracking (ByteTrack) — don't process every frame
- Optimization: TensorRT (2-5x GPU speedup), INT8 quantization (2-4x), dynamic batching (10x)
- Edge: Jetson Orin (TensorRT), mobile (TFLite/CoreML), browser (WebGPU)
- Preprocessing: resize-with-padding, normalize, validate inputs consistently
- NMS: remove overlapping detections, tune confidence threshold per application
- Vision-language models: CLIP, GPT-5V — zero-shot classification without training
- Video pipeline: detect on keyframes, track between them, alert on events
- Model serving: Triton Inference Server (multi-model, dynamic batching, GPU management)
