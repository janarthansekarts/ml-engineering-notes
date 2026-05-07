# Edge ML Engineering

## The Problem / Why This Matters

Not all ML inference can happen in the cloud. When you need: sub-10ms latency (autonomous vehicles can't wait for a network roundtrip), zero internet dependency (industrial sensors in remote locations, airplane mode, underground), strict privacy (medical devices processing patient data, on-device biometrics), or cost elimination (billions of IoT inferences that would be unaffordable via cloud APIs) — you must run ML models directly on edge devices. Edge ML engineering is the discipline of deploying ML models on resource-constrained hardware: mobile phones, IoT devices, embedded systems, browsers, wearables, and vehicles. In 2026, this is a massive field: phones run 7B parameter language models locally, cars run real-time object detection at 60fps, smart speakers process wake words without sending audio to the cloud, and browsers run image generation via WebGPU. The engineering challenges are unique: limited memory (2-8GB RAM on phones, 512MB on IoT), limited compute (mobile NPUs, not H100s), battery constraints (can't drain phone battery for AI features), model size limits (can't download a 14GB model over cellular), and diverse hardware targets (ARM CPUs, Apple Neural Engine, Qualcomm NPU, Intel NPU, WebGPU). Edge ML requires mastering: model optimization (quantization, pruning, architecture search), runtime frameworks (TFLite, ONNX Runtime, CoreML, MediaPipe, llama.cpp), hardware-specific deployment, and on-device evaluation.

---

## The Analogy

Think of edge ML like cooking:

- **Cloud inference** = Ordering from a gourmet restaurant. Best quality, full kitchen with professional equipment, but takes 30 minutes to deliver, costs $50 per meal, and you can't eat during a power outage (internet down).
- **Edge inference** = Cooking at home. Limited kitchen equipment (small stove vs. industrial kitchen), fewer ingredients available (model must be small), but: instant results (no delivery wait), free after initial grocery purchase (no per-inference cost), works during blackouts (offline), and your recipes stay private (data stays on device).
- **Optimization** = Adapting a 5-star restaurant recipe for a home kitchen. Can't use industrial equipment, so you simplify (quantization), skip non-essential steps (pruning), and find equivalent shortcuts (efficient architectures). Result tastes 90% as good in 1/10th the time with 1/100th the cost.

---

## Deep Dive

### Edge Hardware Landscape

```yaml
Edge_Hardware_2026:
  mobile_phones:
    apple:
      chip: "A18/M4 (Apple Neural Engine — 16-core NPU)"
      performance: "38 TOPS (Tera Operations Per Second)"
      memory: "8-16 GB unified memory"
      framework: "CoreML, MLX"
      llm_capable: "Yes — runs 7B INT4 models at ~20 tokens/sec"
      
    android_flagship:
      chip: "Snapdragon 8 Gen 4 (Hexagon NPU)"
      performance: "75 TOPS"
      memory: "12-24 GB RAM"
      framework: "NNAPI, QNN (Qualcomm Neural Network), MediaPipe"
      llm_capable: "Yes — runs 7-13B INT4 models"
      
    android_mid_range:
      chip: "Snapdragon 7 Gen 3 / MediaTek Dimensity"
      performance: "20-35 TOPS"
      memory: "6-8 GB RAM"
      framework: "TFLite, NNAPI"
      llm_capable: "Limited — 2-3B models only"
      
  laptops_desktops:
    apple_silicon:
      chip: "M4 Pro/Max/Ultra"
      performance: "Up to 76 TOPS (Neural Engine)"
      memory: "Up to 192 GB unified (M4 Ultra)"
      advantage: "Unified memory — no PCIe bottleneck between CPU/GPU/NPU"
      frameworks: ["MLX", "CoreML", "llama.cpp with Metal"]
      
    intel_npu:
      chip: "Intel Core Ultra (Meteor Lake+)"
      npu: "Intel AI Boost (up to 34 TOPS)"
      frameworks: ["OpenVINO", "ONNX Runtime"]
      use_case: "Always-on AI features (low power)"
      
    nvidia_laptop_gpu:
      chip: "RTX 4090 Mobile / RTX 5090 Mobile"
      memory: "16-24 GB VRAM"
      performance: "~300-500 TOPS (FP16)"
      frameworks: ["CUDA", "TensorRT", "vLLM"]
      capability: "Runs 70B INT4 models locally"
      
  embedded_iot:
    nvidia_jetson:
      models: ["Orin Nano (20 TOPS)", "Orin NX (70 TOPS)", "Thor (2000 TOPS)"]
      use: "Robotics, autonomous vehicles, industrial vision"
      frameworks: ["TensorRT", "DeepStream", "Isaac ROS"]
      
    raspberry_pi:
      model: "Pi 5 (ARM Cortex-A76, 8GB RAM)"
      performance: "~2 TOPS (CPU only)"
      use: "Hobby projects, lightweight inference"
      frameworks: ["TFLite", "ONNX Runtime", "llama.cpp"]
      
    microcontrollers:
      examples: ["ESP32-S3", "STM32", "nRF52"]
      memory: "512 KB - 8 MB RAM"
      performance: "< 0.1 TOPS"
      models: "TinyML (< 1MB models, keyword detection, anomaly detection)"
      framework: "TFLite Micro, Edge Impulse"
      
  browser:
    webgpu:
      what: "GPU compute in web browsers (Chrome, Edge, Firefox)"
      performance: "Depends on client hardware (1-50+ TOPS)"
      advantage: "Zero install, zero server cost, runs on user's device"
      frameworks: ["Transformers.js", "ONNX Runtime Web", "MediaPipe Web"]
      capability: "3-7B INT4 models in browser"
      limitation: "Initial model download (1-4 GB), client hardware varies"
      
    webnn:
      what: "Web Neural Network API (NPU access from browser)"
      status: "Emerging (Chrome 124+, limited support)"
      advantage: "Access device NPU from browser (more efficient than WebGPU)"
```

### Edge ML Frameworks

```yaml
Edge_Frameworks:
  tflite:
    name: "TensorFlow Lite"
    target: "Mobile (Android/iOS), embedded, microcontrollers"
    format: "FlatBuffer (.tflite)"
    features:
      - "INT8/FP16 quantization (post-training and QAT)"
      - "GPU delegate (OpenGL/Vulkan/Metal)"
      - "NNAPI delegate (Android NPU)"
      - "DSP/XNNA delegates (Qualcomm Hexagon)"
    model_zoo: "300+ pre-optimized models (MobileNet, EfficientNet, etc.)"
    
  onnx_runtime:
    name: "ONNX Runtime"
    target: "Universal (cloud, edge, mobile, browser)"
    format: "ONNX (.onnx)"
    advantage: "Single format works everywhere"
    execution_providers:
      - "CUDA (NVIDIA GPU)"
      - "CoreML (Apple)"
      - "QNN (Qualcomm)"
      - "OpenVINO (Intel)"
      - "WebGPU (browser)"
      - "CPU (ARM NEON, AVX-512)"
    quantization: "Built-in INT4/INT8 quantization"
    
  coreml:
    name: "CoreML"
    target: "Apple devices (iOS, macOS, watchOS)"
    strength: "Best performance on Apple Silicon (uses Neural Engine)"
    format: ".mlpackage"
    features:
      - "Neural Engine acceleration (maximum efficiency)"
      - "Privacy (all processing on-device)"
      - "Dynamic model loading/unloading"
    conversion: "coremltools (from PyTorch/TensorFlow)"
    
  mediapipe:
    name: "MediaPipe (Google)"
    target: "Mobile and web (cross-platform)"
    focus: "On-device AI tasks (hand tracking, face detection, LLM inference)"
    models:
      - "MediaPipe LLM Inference (Gemma 2B on-device)"
      - "Object detection, pose estimation"
      - "Hand/face/body landmark detection"
    advantage: "Production-ready pipeline (preprocessing + model + postprocessing)"
    
  llama_cpp:
    name: "llama.cpp"
    target: "CPU inference for LLMs (any platform)"
    format: "GGUF (quantized models)"
    strength: "Runs LLMs on CPU (AVX, ARM NEON) and Apple Metal"
    platforms: ["Windows", "macOS", "Linux", "iOS", "Android"]
    performance: "7B Q4 model: 15-30 tokens/sec on modern laptop"
    ecosystem: ["Ollama (user-friendly wrapper)", "LM Studio (GUI)", "Jan.ai"]
    
  mlx:
    name: "MLX (Apple)"
    target: "Apple Silicon (M1/M2/M3/M4)"
    strength: "Optimized for Apple unified memory architecture"
    advantage: "Near-GPU performance without separate VRAM (uses unified RAM)"
    performance: "30-50 tokens/sec for 7B model on M4 Pro"
```

### Implementation Patterns

```python
# Edge ML deployment patterns

"""
Patterns for deploying ML models on edge devices: mobile, embedded,
browser, and IoT. Covers model optimization, deployment, and evaluation.
"""

edge_ml_patterns = {
    "model_optimization_for_edge": {
        "size_budget_analysis": {
            "mobile_app": {
                "total_app_size": "< 200 MB (App Store limits)",
                "model_budget": "50-150 MB (after quantization)",
                "models_that_fit": [
                    "MobileNetV4 (5 MB — image classification)",
                    "Gemma 2B INT4 (1.5 GB — downloaded separately)",
                    "Whisper Tiny (40 MB — speech recognition)",
                    "YOLO v8 nano (6 MB — object detection)",
                ],
            },
            "iot_embedded": {
                "total_memory": "512 KB - 8 MB",
                "model_budget": "< 1 MB",
                "models_that_fit": [
                    "Keyword spotting (100 KB)",
                    "Anomaly detection (50 KB)",
                    "Simple classifier (200 KB)",
                ],
            },
            "browser": {
                "download_budget": "< 5 GB (user patience for initial download)",
                "memory_budget": "< 8 GB (typical device RAM)",
                "models_that_fit": [
                    "Phi-4 Mini INT4 (2 GB)",
                    "Stable Diffusion Turbo (2 GB)",
                    "Whisper Small (300 MB)",
                ],
            },
        },
        
        "optimization_pipeline": {
            "step_1_architecture": {
                "what": "Choose efficient architecture from the start",
                "mobile_architectures": [
                    "MobileNetV4 (image classification/detection)",
                    "EfficientNet-Lite (balanced accuracy/speed)",
                    "YOLO v8 nano/small (real-time detection)",
                    "Gemma 2B / Phi-4 Mini (on-device LLM)",
                ],
                "techniques": [
                    "Depthwise separable convolutions (MobileNet)",
                    "Squeeze-and-excitation (EfficientNet)",
                    "Grouped Query Attention (efficient transformer)",
                ],
            },
            "step_2_quantization": {
                "what": "Reduce precision for edge deployment",
                "mobile_gpu": "FP16 (2× faster than FP32, minimal quality loss)",
                "mobile_npu": "INT8 (maximum NPU acceleration)",
                "cpu_only": "INT4 GGUF (smallest size, good CPU performance)",
                "method": "Post-training quantization with calibration data",
            },
            "step_3_pruning": {
                "what": "Remove unnecessary parameters",
                "mobile": "30-50% structured pruning + fine-tune",
                "benefit": "Directly smaller model (no sparse hardware needed)",
            },
            "step_4_operator_optimization": {
                "what": "Use hardware-optimized operators",
                "actions": [
                    "Convert to platform-specific format (CoreML, TFLite, ONNX)",
                    "Enable hardware delegates (GPU, NPU, DSP)",
                    "Fuse batch normalization with convolution",
                    "Replace attention with sliding window for streaming",
                ],
            },
        },
    },
    
    "on_device_llm": {
        "deployment_options": {
            "llama_cpp_mobile": {
                "platform": "iOS / Android",
                "models": "GGUF format (Gemma 2B, Phi-4 Mini, Llama 3 8B)",
                "performance": "10-20 tokens/sec on flagship phones",
                "memory": "2-4 GB for INT4 7B model",
                "integration": "C++ library with Java/Swift bindings",
            },
            "mediapipe_llm": {
                "platform": "Android / iOS / Web",
                "models": "Gemma 2B, Phi models",
                "advantage": "Google's production-ready framework",
                "performance": "15-25 tokens/sec on flagship Android",
            },
            "mlx_apple": {
                "platform": "macOS / iOS",
                "models": "Any model converted to MLX format",
                "advantage": "Optimized for Apple unified memory",
                "performance": "30-50 tokens/sec on M4 Pro (7B model)",
            },
            "webllm": {
                "platform": "Chrome / Edge (WebGPU required)",
                "models": "Llama 3 8B INT4, Phi-4 Mini",
                "advantage": "Zero server cost, runs in browser",
                "limitation": "Requires modern GPU, large download",
            },
        },
        "use_cases": [
            "Offline text summarization (plane, subway)",
            "Private journaling assistant (no data leaves device)",
            "Smart keyboard suggestions (predictive text on steroids)",
            "On-device document analysis (legal, medical privacy)",
            "Coding assistant without internet (air-gapped environments)",
        ],
    },
    
    "real_time_inference": {
        "computer_vision": {
            "object_detection": {
                "model": "YOLOv8 nano (6 MB)",
                "target": "30-60 fps on mobile phone",
                "optimization": [
                    "INT8 quantization for NPU",
                    "Smaller input resolution (320×320 instead of 640×640)",
                    "Frame skipping (process every 2nd frame)",
                ],
            },
            "pose_estimation": {
                "model": "MediaPipe Pose (BlazePose, 3 MB)",
                "target": "30 fps real-time body tracking",
                "deployment": "MediaPipe framework (handles camera + inference + rendering)",
            },
            "ocr": {
                "model": "PaddleOCR Mobile (10 MB)",
                "target": "< 100ms per frame (text recognition)",
                "use_case": "Point camera at document → instant text extraction",
            },
        },
        "audio_processing": {
            "wake_word": {
                "model": "Custom small model (200 KB)",
                "target": "Always-on, < 1% battery impact",
                "approach": "Tiny neural network on DSP (hardware-level efficiency)",
                "latency": "< 10ms detection",
            },
            "speech_to_text": {
                "model": "Whisper Tiny/Small (40-300 MB)",
                "target": "Real-time transcription",
                "deployment": "On-device (privacy) or streaming to cloud (better quality)",
            },
        },
    },
    
    "deployment_strategies": {
        "model_delivery": {
            "bundled": {
                "what": "Model shipped with app binary",
                "pros": "Works immediately, no download needed",
                "cons": "App size increases, updates require app update",
                "when": "Small models (< 50 MB), critical path features",
            },
            "on_demand_download": {
                "what": "Model downloaded after app install",
                "pros": "Smaller initial app size, can update model independently",
                "cons": "Feature unavailable until download completes",
                "when": "Large models (LLMs), optional features",
            },
            "progressive": {
                "what": "Small model bundled, large model downloaded in background",
                "pros": "Immediate basic functionality, upgraded over time",
                "when": "Quality-tiered features (basic → premium model)",
            },
        },
        "model_updates": {
            "approach": "Over-the-air model updates (like app updates)",
            "mechanism": "Check version → download delta → validate → swap atomically",
            "safety": "Always keep previous working model as fallback",
            "frequency": "Weekly-monthly (depends on drift and new capabilities)",
        },
    },
}


# Edge performance benchmarks
edge_benchmarks = {
    "iphone_16_pro": {
        "chip": "A18 Pro (Apple Neural Engine)",
        "gemma_2b_int4": "25 tokens/sec",
        "mobilenet_v4": "2ms per inference (500 fps theoretical)",
        "whisper_tiny": "10× realtime (10 sec audio in 1 sec)",
        "stable_diffusion": "~3 seconds per image (512×512)",
    },
    "samsung_galaxy_s25": {
        "chip": "Snapdragon 8 Gen 4",
        "gemma_2b_int4": "20 tokens/sec",
        "yolov8_nano": "45 fps (real-time object detection)",
        "mediapipe_pose": "30 fps (full body tracking)",
    },
    "macbook_m4_pro": {
        "chip": "M4 Pro (16GB unified memory)",
        "llama_3_8b_int4": "40 tokens/sec (llama.cpp)",
        "phi_4_14b_int4": "25 tokens/sec",
        "whisper_large": "30× realtime",
        "stable_diffusion_xl": "~5 seconds per image",
    },
    "raspberry_pi_5": {
        "chip": "ARM Cortex-A76 (8GB RAM)",
        "phi_4_mini_int4": "3 tokens/sec (usable for batch, not interactive)",
        "yolov8_nano": "15 fps",
        "keyword_detection": "Real-time (< 5ms)",
    },
}
```

---

## How It Works in Practice

### Edge Deployment Production System

```yaml
Edge_Deployment_Production:
  scenario: "On-device AI assistant for healthcare (patient data cannot leave device)"
  
  requirements:
    privacy: "HIPAA compliant — zero patient data leaves device"
    device: "iPad Pro M4 (medical facility tablets)"
    features:
      - "Transcribe doctor-patient conversations (on-device STT)"
      - "Extract medical entities (conditions, medications, procedures)"
      - "Generate visit summaries (on-device LLM)"
      - "Suggest ICD codes (classification)"
    latency: "< 3 seconds for summary generation"
    battery: "Full shift (8 hours) without charging"
    
  architecture:
    speech_to_text:
      model: "Whisper Small (300 MB, fine-tuned on medical terminology)"
      framework: "CoreML (Apple Neural Engine)"
      performance: "Real-time transcription (streaming, 20× realtime)"
      
    entity_extraction:
      model: "Fine-tuned BERT-base (110M params, 200MB CoreML)"
      task: "NER — extract medications, conditions, procedures"
      performance: "< 50ms per sentence"
      
    summary_generation:
      model: "Gemma 2 9B INT4 (fine-tuned for medical summaries)"
      framework: "MLX (optimized for Apple Silicon)"
      performance: "30 tokens/sec on M4 Pro"
      memory: "5 GB for model + 2 GB for KV cache"
      
    code_suggestion:
      model: "Fine-tuned classifier (MobileNet variant, 10 MB)"
      task: "Suggest top-5 ICD codes from extracted entities"
      performance: "< 20ms"
      
  deployment:
    model_delivery: "MDM (Mobile Device Management) push — models pre-loaded on facility iPads"
    updates: "Weekly model updates via facility WiFi (off-hours)"
    fallback: "Previous model version always retained"
    monitoring: "On-device metrics (latency, accuracy confidence) uploaded to dashboard"
    
  results:
    privacy: "Zero data leaves device ✓ (HIPAA compliant)"
    latency: "2.1 seconds for full summary (< 3 sec target ✓)"
    battery: "12 hours runtime (8 hour shift + margin ✓)"
    accuracy: "91% entity extraction, 88% summary quality (physician-rated)"
    cost: "Zero per-inference cloud cost (one-time model development)"
```

---

## Interview Tip

> When asked about edge ML: "Edge ML is about running inference under tight constraints — limited memory, compute, battery, and connectivity. My framework: (1) Hardware-aware model selection: choose architecture designed for edge (MobileNet, EfficientNet, Phi-4 Mini) rather than shrinking a cloud model. Purpose-built edge models outperform compressed large models. (2) Optimization pipeline: quantize (INT4 for LLMs, INT8 for CNNs), prune (30-50% structured pruning), then convert to hardware-specific format (CoreML for Apple, TFLite for Android, ONNX for universal). (3) Framework selection: CoreML for Apple (uses Neural Engine, maximum efficiency), MediaPipe for cross-platform mobile, llama.cpp/MLX for on-device LLMs, TFLite for embedded/IoT. (4) On-device LLMs are now real — 2-9B parameter models run at 15-40 tokens/sec on flagship phones and laptops (INT4 quantization). Use cases: offline assistants, private document processing, keyboard suggestions. (5) Deployment: bundle small models with app (<50 MB), download large models on-demand (1-4 GB LLMs). Always keep previous model version as fallback. (6) Key constraint I respect: battery. AI features that drain battery in 2 hours will be disabled by users. Design for always-on efficiency (DSP/NPU for continuous tasks) versus burst processing (GPU for short inferences)."

---

## Common Mistakes

1. **Deploying cloud model on edge without optimization** — Taking a 7B FP16 model (14GB) and trying to run it on a phone with 8GB RAM. It won't fit, or will be unusably slow. Solution: quantize to INT4 (3.5GB), use edge-optimized runtime (llama.cpp, CoreML), and verify memory budget includes KV cache.

2. **Ignoring hardware-specific acceleration** — Using CPU inference when the device has an NPU/GPU that's 10× faster. Solution: use hardware delegates (CoreML Neural Engine, NNAPI, TFLite GPU delegate). The difference between CPU and NPU inference on mobile is often 10-20×.

3. **Not testing on real devices** — Model works on development machine (powerful laptop) but fails on target device (budget phone, Raspberry Pi). Solution: test on actual target hardware early. Performance on emulators/simulators doesn't reflect real device constraints.

4. **Downloading huge models on cellular** — App requires 4GB model download on first launch. Users on cellular data or slow connections abandon the app. Solution: progressive loading (small bundled model → large downloaded in background), WiFi-only downloads for large models, clear progress indication.

5. **Single model for all devices** — Same model for flagship phone (16GB RAM) and budget phone (4GB RAM). Either the flagship is underutilized or the budget phone crashes. Solution: model tiering — small/medium/large variants selected based on device capability detection at runtime.

---

## Key Takeaways

- Edge ML: run models on-device (phones, laptops, IoT, browsers) — no cloud dependency
- Motivations: latency (<10ms), privacy (data stays on device), offline, cost (no per-inference API)
- 2026 reality: phones run 2-9B LLMs at 15-40 tokens/sec, browsers run 3-7B via WebGPU
- Hardware: mobile NPUs (38-75 TOPS), Apple Neural Engine, Qualcomm Hexagon, Intel NPU
- Frameworks: CoreML (Apple), TFLite (Android/embedded), ONNX Runtime (universal), llama.cpp (LLMs)
- Optimization: quantize (INT4/INT8) → prune → convert to hardware-specific format
- Model delivery: bundle small models, download large models on-demand, progressive loading
- Battery: critical constraint — use NPU/DSP for always-on, GPU for burst processing
- On-device LLMs: Gemma 2B, Phi-4 Mini (run on phones); Llama 3 8B (laptops); via llama.cpp/MLX
- WebGPU: run ML in browser, zero server cost, zero install (Transformers.js, ONNX Runtime Web)
- Testing: always validate on real target hardware (emulators don't reflect true performance)
