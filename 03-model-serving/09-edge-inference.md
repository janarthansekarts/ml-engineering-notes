# Edge Inference

## The Problem / Why This Matters

Not every ML prediction can or should go through a cloud server. Sending audio from a smart speaker to the cloud adds 200-500ms of network latency — unacceptable for real-time voice commands. A self-driving car can't depend on cloud connectivity for split-second decisions. A mobile health app can't send patient data to external servers due to privacy regulations like HIPAA (Health Insurance Portability and Accountability Act). Edge inference means running ML models directly on the user's device — smartphone, browser, IoT device, vehicle, or edge server — without a round-trip to the cloud. The challenge: edge devices have severely constrained resources compared to cloud GPUs. A smartphone has 4-8GB RAM (shared with OS and apps), a low-power NPU (Neural Processing Unit) with 10-30 TOPS (Tera Operations Per Second), and a battery that can't sustain continuous GPU computation. Getting a model that was trained on 8× H100 GPUs to run efficiently on a phone requires aggressive optimization: quantization, architecture changes, runtime optimization, and hardware-specific compilation. In 2026, edge inference has matured significantly — with frameworks like ONNX Runtime, TensorFlow Lite, CoreML, and WebGPU enabling production-quality on-device ML across all platforms.

---

## The Analogy

Think of edge inference like cooking at home vs ordering from a restaurant:

- **Cloud inference** = Ordering from a fancy restaurant. Professional kitchen (H100 GPUs), expert chefs (optimized serving), any dish possible (any model size). But: delivery takes time (network latency), you pay per meal (API cost), and you need a phone connection (internet dependency).
- **Edge inference** = Cooking at home. Limited kitchen (phone NPU), limited ingredients (4GB RAM), limited skill (constrained compute). But: instant meal (zero network latency), no delivery cost (no API fees), works during power outage (offline capable), and no one sees what you're eating (privacy).
- **Model optimization for edge** = Adapting restaurant recipes for home cooking. Can't use the industrial oven (H100 GPU), so: fewer ingredients (quantization), simpler technique (smaller architecture), pre-prepared components (compiled optimized ops). The meal is 90% as good in 10% of the time.

---

## Deep Dive

### Edge Inference Frameworks

```yaml
Edge_Frameworks:
  tensorflow_lite:
    name: "TFLite (TensorFlow Lite)"
    what: "Google's framework for on-device ML inference"
    platforms: "Android, iOS, Linux (embedded), microcontrollers"
    model_format: ".tflite (FlatBuffer format, optimized for mobile)"
    features:
      - "INT8/INT16 quantization (post-training and QAT)"
      - "GPU delegate (OpenGL ES, Metal)"
      - "NNAPI delegate (Android Neural Networks API)"
      - "XNNPACK (optimized CPU inference for ARM)"
      - "Coral Edge TPU delegate (Google's edge AI chip)"
    typical_models: "Image classification, object detection, NLP (DistilBERT), speech"
    size: "Runtime adds ~1 MB to app size"
    
  onnx_runtime_mobile:
    name: "ONNX Runtime (Mobile/Web)"
    what: "Microsoft's cross-platform inference engine"
    platforms: "Android, iOS, Windows, Linux, WebAssembly"
    model_format: ".onnx (Open Neural Network Exchange format)"
    features:
      - "Multiple execution providers (CPU, GPU, NPU, CoreML, NNAPI)"
      - "Quantization (INT8, INT4 via ONNX quantization tools)"
      - "Graph optimization (constant folding, node fusion)"
      - "ONNX Runtime Web (runs in browser via WebAssembly/WebGPU)"
    advantage: "Train in any framework (PyTorch, TensorFlow) → export to ONNX → run anywhere"
    
  coreml:
    name: "CoreML"
    what: "Apple's ML framework for on-device inference (iOS, macOS, watchOS)"
    platforms: "iOS 11+, macOS 10.13+, watchOS, tvOS, visionOS"
    model_format: ".mlmodel / .mlpackage"
    features:
      - "Automatic hardware selection (CPU, GPU, or Neural Engine)"
      - "INT8/FP16 quantization"
      - "Dynamic shapes (variable input sizes)"
      - "Model compilation (compiled .mlmodelc for faster loading)"
      - "On-device training (personalization)"
    advantage: "Deepest integration with Apple hardware (Neural Engine, Unified Memory)"
    tools: "coremltools (convert from PyTorch/TensorFlow/ONNX)"
    
  mediapipe:
    name: "MediaPipe"
    what: "Google's framework for multimodal on-device ML"
    focus: "Vision + hand/face/pose tracking + audio"
    platforms: "Android, iOS, Web, Python"
    pre_built_tasks:
      - "Face detection and landmark tracking"
      - "Hand tracking and gesture recognition"
      - "Pose estimation"
      - "Object detection and tracking"
      - "Image segmentation"
      - "LLM inference (Gemma on-device)"
    advantage: "Pre-optimized pipelines — plug-and-play for common vision tasks"
    
  mlx:
    name: "MLX"
    what: "Apple's ML framework for Apple Silicon (M-series chips)"
    focus: "LLM inference and training on Mac"
    platforms: "macOS (M1/M2/M3/M4 Apple Silicon)"
    features:
      - "Unified memory (model directly accesses system RAM — no GPU copy needed)"
      - "Lazy evaluation (compute only what's needed)"
      - "NumPy-like API"
      - "4-bit quantization support"
      - "Full LLM inference (Llama, Mistral, Gemma locally)"
    performance: "30+ tokens/sec for 8B model on M4 Max (128 GB unified memory)"
    use_case: "Local LLM development, private inference on Mac"
```

### WebGPU and Browser-Based Inference

```yaml
WebGPU_Inference:
  what: "Run ML models directly in the web browser using GPU acceleration"
  
  webgpu:
    name: "WebGPU"
    what: "Modern web standard for GPU access (successor to WebGL for compute)"
    status: "Supported in Chrome, Edge, Firefox (2024+)"
    performance: "10-100× faster than WebAssembly for ML workloads"
    
  frameworks:
    transformers_js:
      what: "HuggingFace Transformers ported to JavaScript"
      runtime: "ONNX Runtime Web (WebAssembly + WebGPU)"
      models: "200+ pre-converted models (BERT, DistilBERT, Whisper, GPT-2)"
      use_case: "NLP, vision, audio in browser — no server needed"
      
    web_llm:
      what: "Run LLMs entirely in browser via WebGPU"
      models: "Llama-3B, Gemma-2B, Phi-3-mini (quantized)"
      performance: "10-20 tokens/sec for 3B model in browser"
      limitation: "Limited by browser memory (2-4 GB typically available)"
      use_case: "Private, zero-server LLM inference (all computation on user's device)"
      
    mediapipe_web:
      what: "Google MediaPipe tasks running in browser"
      use_case: "Real-time face/hand/pose tracking in browser applications"
      
  advantages:
    - "Zero server cost (user's device does all computation)"
    - "Complete privacy (data never leaves device)"
    - "Works offline (after initial model download)"
    - "No deployment infrastructure needed (just serve static files)"
    - "Cross-platform (any device with a modern browser)"
    
  limitations:
    - "Model size limited by device memory (2-4 GB in browser)"
    - "Slower than native (WebGPU overhead vs CUDA)"
    - "Battery drain on mobile browsers"
    - "Initial model download can be large (100MB-2GB)"
    - "Not all models convert cleanly to WebGPU-compatible formats"
```

### On-Device LLMs

```yaml
On_Device_LLMs:
  landscape_2026:
    apple:
      framework: "MLX + Apple Intelligence"
      hardware: "Neural Engine + Unified Memory (up to 192 GB on M4 Max)"
      models: "Apple Foundation Models (3B on-device), Llama via MLX"
      performance: "40+ tok/s for 8B Q4 on M4 Max"
      
    android:
      framework: "MediaPipe LLM Inference API + Google AI Edge"
      hardware: "Qualcomm Hexagon NPU, Samsung Exynos NPU, Google Tensor"
      models: "Gemma 2B/7B, Phi-3-mini (quantized)"
      performance: "10-20 tok/s for 2B model on flagship phone"
      
    desktop:
      framework: "llama.cpp, Ollama, LM Studio"
      hardware: "CPU (AVX2/AVX512), GPU (NVIDIA/AMD), Apple Silicon"
      models: "Any GGUF model — Llama, Mistral, Phi, Gemma"
      performance: "20-50 tok/s for 8B Q4 on modern hardware"
      
  model_size_constraints:
    smartphone_4gb_ram:
      usable_for_model: "~2 GB (rest for OS + apps)"
      max_model: "2B parameters at INT4 (~1.2 GB)"
      examples: "Gemma-2B, Phi-3-mini (3.8B at Q2_K)"
      
    smartphone_8gb_ram:
      usable_for_model: "~4 GB"
      max_model: "7-8B parameters at INT4 (~4 GB)"
      examples: "Llama-4-8B Q4, Gemma-7B Q4"
      
    laptop_16gb_ram:
      usable_for_model: "~10 GB"
      max_model: "13B at INT4 or 8B at FP16"
      
    mac_m4_max_128gb:
      usable_for_model: "~100 GB (unified memory)"
      max_model: "70B at INT4 (~35 GB) or 70B at FP8 (~70 GB)"
      performance: "Competitive with cloud GPU for single-user inference"
      
  optimization_for_mobile:
    quantization: "INT4 (GGUF Q4_K_M) — essential for fitting models"
    kv_cache: "Limit context length (2K-4K instead of 8K) to save memory"
    speculative_decoding: "Use tiny draft model (100M) to propose, larger model to verify"
    sliding_window: "Only attend to recent N tokens (not full history)"
    layer_offload: "Keep some layers on CPU, hot layers on NPU/GPU"
```

### Model Optimization for Edge

```python
# Converting a PyTorch model for multi-platform edge deployment

import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import coremltools as ct

# 1. Export to ONNX (universal intermediate format)
model = load_my_pytorch_model()
model.eval()

dummy_input = torch.randn(1, 512)  # Example input shape
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}}
)

# 2. Quantize for edge (INT8 — works on all platforms)
quantize_dynamic(
    model_input="model.onnx",
    model_output="model_int8.onnx",
    weight_type=QuantType.QInt8,
)
# Result: ~4× smaller, ~2× faster on CPU

# 3. Convert to CoreML (iOS/macOS)
mlmodel = ct.convert(
    "model.onnx",
    convert_to="mlprogram",  # Latest CoreML format
    compute_precision=ct.precision.FLOAT16,  # FP16 for Neural Engine
    minimum_deployment_target=ct.target.iOS17,
)
mlmodel.save("model.mlpackage")

# 4. Convert to TFLite (Android)
import tensorflow as tf

# Load ONNX → TensorFlow → TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("tf_saved_model/")
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # INT8 quantization
converter.target_spec.supported_types = [tf.float16]  # FP16 fallback
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

```yaml
Edge_Optimization_Techniques:
  architecture_changes:
    knowledge_distillation:
      what: "Train small student model to mimic large teacher model"
      example: "DistilBERT (66M params) distilled from BERT (110M) — 97% accuracy, 60% faster"
      use: "When you need BERT quality but phone-friendly size"
      
    pruning_for_edge:
      what: "Remove redundant weights/neurons, then fine-tune"
      benefit: "30-50% smaller model, faster inference"
      
    efficient_architectures:
      what: "Use architectures designed for mobile from the start"
      examples:
        - "MobileNet (efficient vision — depthwise separable convolutions)"
        - "EfficientNet (compound scaling for mobile)"
        - "TinyLlama (1.1B — trained for efficiency)"
        - "Gemma-2B (Google's efficient small LLM)"
        - "Phi-3 mini (3.8B — high quality per parameter)"
        
  runtime_optimization:
    operator_fusion:
      what: "Combine multiple operations into one (Conv + BatchNorm + ReLU → single fused op)"
      benefit: "Fewer memory reads, less kernel launch overhead"
      
    hardware_delegates:
      what: "Route operations to specialized hardware (NPU, DSP, GPU)"
      examples:
        - "CoreML: auto-selects CPU/GPU/Neural Engine per operation"
        - "NNAPI: routes to available Android hardware accelerators"
        - "TFLite GPU delegate: offloads to mobile GPU"
        
    model_caching:
      what: "Cache compiled/optimized model on device"
      benefit: "Skip optimization on subsequent loads (instant startup)"
```

### Edge Inference Use Cases

```yaml
Use_Cases:
  real_time_requirements:
    voice_assistant:
      latency_need: "< 200ms end-to-end"
      why_edge: "Network round-trip alone is 50-200ms, eating entire budget"
      models: "Wake word detection, VAD (Voice Activity Detection), speech-to-text"
      
    autonomous_driving:
      latency_need: "< 50ms for object detection"
      why_edge: "Life-safety — can't depend on network connectivity"
      models: "YOLO, LiDAR point cloud processing, path planning"
      hardware: "NVIDIA Orin (edge GPU), custom automotive chips"
      
    ar_vr:
      latency_need: "< 20ms (avoid motion sickness)"
      why_edge: "Any perceptible delay between head movement and visual update causes nausea"
      models: "Hand tracking, scene understanding, eye tracking"
      hardware: "Qualcomm XR chips, Apple Vision Pro chip"
      
  privacy_requirements:
    health_monitoring:
      why_edge: "HIPAA compliance — patient data can't leave device"
      models: "ECG analysis, fall detection, activity recognition"
      
    face_id:
      why_edge: "Biometric data never leaves device (Apple/Android policy)"
      models: "Face detection, depth estimation, liveness check"
      
    keyboard_prediction:
      why_edge: "Typing data is extremely sensitive (passwords, messages)"
      models: "Next-word prediction, autocorrect, smart compose"
      
  cost_requirements:
    high_volume_simple_ml:
      why_edge: "1 billion daily predictions × $0.001 per cloud inference = $1M/day"
      alternative: "Run on user's device = $0/day in inference cost"
      models: "Recommendation ranking, content filtering, spam detection"
```

---

## How It Works in Practice

### Platform Selection Guide

```yaml
Selection_Guide:
  ios_only:
    framework: "CoreML"
    why: "Best performance (Neural Engine access), seamless Xcode integration"
    tools: "coremltools for conversion, Xcode for testing"
    
  android_only:
    framework: "TFLite or ONNX Runtime Mobile"
    why: "TFLite for Google ecosystem, ONNX RT for PyTorch models"
    hardware: "NNAPI delegate for Qualcomm/Samsung NPU access"
    
  cross_platform_mobile:
    framework: "ONNX Runtime Mobile"
    why: "Same model format works on iOS (CoreML EP) and Android (NNAPI EP)"
    alternative: "TFLite (both platforms supported)"
    
  browser:
    framework: "Transformers.js (NLP/Vision) or MediaPipe (Vision/Tracking)"
    runtime: "ONNX Runtime Web with WebGPU acceleration"
    
  local_llm_mac:
    framework: "MLX or llama.cpp (via Ollama)"
    why: "MLX: native Apple Silicon optimization, unified memory. Ollama: easy setup."
    
  local_llm_any:
    framework: "llama.cpp (via Ollama or LM Studio)"
    why: "Cross-platform, GGUF quantization, AVX2/CUDA/Metal support"
    
  embedded_iot:
    framework: "TFLite Micro or ONNX Runtime (embedded)"
    hardware: "Coral Edge TPU, ARM Cortex-M, RISC-V"
```

---

## Interview Tip

> When asked about edge/on-device inference: "Edge inference decision depends on three factors: latency requirement (network adds 50-200ms — unacceptable for real-time voice/AR/driving), privacy (health data, biometrics, typing — can't leave device), and cost at scale (1B daily predictions × API cost = massive expense vs free on-device). My optimization pipeline: (1) quantize aggressively — INT4 for LLMs (GGUF Q4_K_M), INT8 for classification models, (2) use efficient architectures designed for mobile (MobileNet, DistilBERT, Gemma-2B, Phi-3 mini), (3) target the right runtime — CoreML on iOS (accesses Neural Engine directly), TFLite on Android (NNAPI delegate for NPU), ONNX Runtime for cross-platform. For on-device LLMs in 2026: 2B models run well on flagship phones (10-20 tok/s), 8B models on laptops (30+ tok/s via llama.cpp), and 70B on M4 Max Macs (unified 128GB memory). The key trade-off: edge gives zero-latency, zero-cost, full-privacy inference but limits model size and quality. Hybrid approach: simple models on-device, complex queries routed to cloud."

---

## Common Mistakes

1. **Deploying full-precision models to edge** — Sending an FP32 model (500 MB) to a mobile app. After INT8 quantization: 125 MB with minimal quality loss. After INT4: 62 MB. Always quantize for edge — there's no excuse for FP32 on mobile in 2026.

2. **Ignoring model load time** — A 500 MB model takes 3-5 seconds to load on a phone. If your app loads the model on every screen transition, the UX is terrible. Load once at app startup, keep in memory, or use lazy loading with a loading indicator.

3. **Not testing on actual target devices** — Model runs fine on your M4 Max MacBook but crashes on a 3-year-old Android phone with 4 GB RAM. Always test on lowest-spec supported device. Edge inference performance varies 10-50× across device generations.

4. **Using cloud-optimized architectures on edge** — Deploying a 7B LLM to a phone that has 4 GB total RAM. Use architectures designed for edge: Gemma-2B, Phi-3-mini, DistilBERT, MobileNet. Or use knowledge distillation to create a small student model.

5. **Not implementing graceful fallback** — Edge model fails (OOM, unsupported op, too slow) with no fallback. Production apps should: try edge inference → if fails/too slow → fall back to cloud API → if no network → return cached/default response. Never let the user see a crash.

---

## Key Takeaways

- Edge inference: run models on-device for zero latency, zero API cost, and full privacy
- When to use edge: real-time requirements (<200ms), privacy regulations, high-volume cost optimization
- CoreML (iOS), TFLite (Android), ONNX Runtime (cross-platform), llama.cpp (local LLMs)
- WebGPU enables browser-based inference — 10-100× faster than WebAssembly for ML
- On-device LLMs in 2026: 2B on phones, 8B on laptops, 70B on high-end Macs
- Always quantize: INT4 for LLMs (GGUF), INT8 for traditional models — no FP32 on edge
- Use efficient architectures: MobileNet, DistilBERT, Gemma-2B, Phi-3-mini
- ONNX as universal format: train anywhere → convert to ONNX → deploy to any platform
- Test on actual target hardware (not just your development machine)
- Hybrid approach: edge for simple/fast, cloud for complex/quality-critical
