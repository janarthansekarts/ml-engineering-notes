# Model Optimization for Inference

## The Problem / Why This Matters

A model trained in FP32 or BF16 is optimized for learning accuracy, not inference efficiency. Serving that same model unmodified wastes compute, memory, and money. A 70B parameter model in FP16 requires 140GB of GPU memory — that's two H100 GPUs just to load the model, before any requests are processed. But with INT4 quantization, the same model fits in 35GB (one GPU), serves 4x more concurrent requests, and generates tokens faster — with minimal quality loss (often <1% on benchmarks). Model optimization for inference encompasses: quantization (reducing numerical precision), pruning (removing unimportant weights), distillation (training smaller models to mimic larger ones), compilation (graph optimization), and architecture changes (smaller models, speculative decoding). These techniques can collectively reduce serving costs by 2-10x while maintaining acceptable quality. In 2026, quantization is the most impactful single optimization — every production LLM deployment uses some form of quantization.

---

## The Analogy

Think of model optimization like compressing a video for streaming:

- **Original model (FP32)** = Uncompressed 4K video. Maximum quality, but massive file size. Requires expensive bandwidth (GPU memory) to stream.
- **Quantization (INT8/INT4)** = Video compression (H.265). Dramatically smaller file, barely noticeable quality loss to viewers. The key insight: most information in the original is redundant for the final viewer experience.
- **Pruning** = Removing frames that are nearly identical to adjacent frames. Less data to process, barely visible difference.
- **Distillation** = Remaking the movie as a shorter version that captures all the key plot points. Smaller, faster to watch, covers the essentials.
- **TensorRT compilation** = Optimizing the streaming protocol. Same video, but delivered with minimal buffering through protocol-level optimization.

---

## Deep Dive

### Quantization

```yaml
Quantization_Overview:
  what: "Reduce numerical precision of model weights (and optionally activations)"
  why: "Lower precision = less memory + faster computation + more throughput"
  
  precision_levels:
    fp32:
      bits: 32
      memory: "4 bytes per parameter"
      use: "Training only (not for inference in 2026)"
      
    fp16_bf16:
      bits: 16
      memory: "2 bytes per parameter"
      quality_loss: "Negligible"
      use: "Default training precision, acceptable for inference"
      
    int8:
      bits: 8
      memory: "1 byte per parameter (2x savings vs FP16)"
      quality_loss: "< 1% on most benchmarks"
      throughput_gain: "1.5-2x vs FP16"
      use: "Production inference (good balance of quality and speed)"
      
    int4:
      bits: 4
      memory: "0.5 bytes per parameter (4x savings vs FP16)"
      quality_loss: "1-3% on most benchmarks"
      throughput_gain: "2-4x vs FP16"
      use: "Memory-constrained deployment (fit larger models on fewer GPUs)"
      
    fp8:
      bits: 8
      formats: "E4M3 (higher precision), E5M2 (higher range)"
      memory: "1 byte per parameter"
      quality_loss: "< 0.5% (better than INT8 for most models)"
      hardware: "H100/B200 tensor cores native support"
      use: "Emerging standard for inference on latest hardware"

  quantization_approaches:
    post_training_quantization:
      name: "PTQ (Post-Training Quantization)"
      what: "Quantize an already-trained model without retraining"
      advantage: "Fast (minutes to hours), no training needed"
      disadvantage: "May lose more quality than QAT (Quantization-Aware Training)"
      types:
        weight_only: "Only quantize weights (not activations)"
        weight_and_activation: "Quantize both (more savings, more complex)"
        
    quantization_aware_training:
      name: "QAT"
      what: "Simulate quantization during training (model learns to handle it)"
      advantage: "Better quality retention"
      disadvantage: "Requires retraining (expensive for large models)"
      use: "When PTQ quality loss is unacceptable"
```

### Quantization Methods (2026)

```yaml
Quantization_Methods:
  gptq:
    full_name: "GPT Quantization"
    type: "Weight-only PTQ (Post-Training Quantization)"
    how: "Layer-by-layer quantization using second-order information (Hessian)"
    precision: "INT4, INT8"
    speed: "Hours on a single GPU for 70B model"
    quality: "Good (state-of-art when released, now matched by AWQ)"
    tools: "AutoGPTQ library"
    
  awq:
    full_name: "Activation-aware Weight Quantization"
    type: "Weight-only PTQ"
    how: "Protect salient weights (those that produce large activations) from quantization error"
    insight: "Not all weights are equally important — weights connected to large activations matter more"
    precision: "INT4"
    quality: "Better than GPTQ (especially at 4-bit)"
    speed: "Fast calibration (minutes with small calibration dataset)"
    tools: "AutoAWQ library"
    recommendation: "Default choice for INT4 quantization in 2026"
    
  gguf:
    full_name: "GPT-Generated Unified Format"
    type: "Weight-only quantization for CPU/hybrid inference"
    how: "Various quantization schemes (Q4_K_M, Q5_K_M, Q8_0, etc.)"
    target: "llama.cpp, Ollama, LM Studio (local inference)"
    precision: "2-8 bit (various schemes)"
    quality: "Varies by scheme (Q4_K_M is popular balance)"
    use: "Local/edge deployment, CPU inference, consumer hardware"
    
  smoothquant:
    type: "Weight + activation quantization (INT8)"
    how: "Transfer quantization difficulty from activations to weights (smoothing)"
    insight: "Activations have outliers that are hard to quantize — smooth them first"
    precision: "INT8 for both weights and activations"
    benefit: "INT8 matmul is 2x faster than FP16 on modern GPUs"
    use: "When both weight and activation quantization is needed"
    
  fp8_quantization:
    type: "FP8 (8-bit floating point)"
    how: "Use FP8 on H100/B200 tensor cores for native acceleration"
    precision: "E4M3 for weights, E5M2 for gradients/activations"
    quality: "Better than INT8 (floating point preserves more information)"
    speed: "2x throughput on H100 tensor cores vs FP16"
    status: "Becoming standard for H100/B200 inference in 2026"
    tools: "NVIDIA Transformer Engine, vLLM FP8 support"
```

### Practical Quantization

```python
# AWQ Quantization - Current best practice for INT4
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-4-8B"
quant_path = "./llama-4-8b-awq"

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Quantize (uses calibration data to determine optimal quantization)
quant_config = {
    "zero_point": True,     # Use zero-point quantization
    "q_group_size": 128,    # Group size for quantization (128 is standard)
    "w_bit": 4,             # 4-bit weights
}

model.quantize(
    tokenizer, 
    quant_config=quant_config,
    calib_data="pileval",   # Calibration dataset (small, representative)
)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

# Serve with vLLM
# vllm serve ./llama-4-8b-awq --quantization awq
```

```python
# Serving quantized model with vLLM
from vllm import LLM, SamplingParams

# Load AWQ-quantized model
llm = LLM(
    model="./llama-4-8b-awq",
    quantization="awq",
    gpu_memory_utilization=0.9,
    max_model_len=8192,
)

# Compare memory usage:
# FP16: ~16 GB for 8B model
# AWQ INT4: ~5 GB for 8B model (3.2x reduction)
# This means: fit model on smaller GPU, or serve more concurrent requests

# Generate
outputs = llm.generate(
    ["Explain quantization in ML:"],
    SamplingParams(temperature=0.7, max_tokens=256)
)
```

### Pruning

```yaml
Pruning:
  what: "Remove parameters (set to zero) that contribute least to model output"
  why: "Many model weights are near-zero or redundant — removing them saves compute and memory"
  
  types:
    unstructured_pruning:
      what: "Remove individual weights (set to zero)"
      sparsity: "Can achieve 50-90% sparsity"
      challenge: "Sparse matrices don't accelerate on standard hardware (irregular memory access)"
      hardware_needed: "Sparse tensor cores (A100/H100 support 2:4 structured sparsity)"
      
    structured_pruning:
      what: "Remove entire neurons, attention heads, or layers"
      advantage: "Resulting model is dense (standard hardware accelerates it)"
      challenge: "Less fine-grained — may remove important capacity"
      methods:
        - "Remove attention heads with lowest importance scores"
        - "Remove entire layers (layer dropping)"
        - "Reduce hidden dimensions"
        
    2_4_sparsity:
      what: "NVIDIA's hardware-supported sparsity: out of every 4 weights, 2 must be zero"
      hardware: "A100/H100 sparse tensor cores"
      speedup: "~2x on supported hardware"
      quality: "Usually <1% accuracy loss with fine-tuning after pruning"
      
  practical_status_2026:
    for_llms: "Pruning is LESS popular than quantization for LLMs"
    reason: "Quantization is simpler, well-supported, and gives better cost/quality trade-off"
    when_pruning_makes_sense: "Combined with quantization for extreme compression (edge deployment)"
```

### Compilation and Graph Optimization

```yaml
Compilation:
  tensorrt:
    what: "NVIDIA's deep learning inference optimizer and runtime"
    how: "Compile model graph into optimized CUDA kernels"
    optimizations:
      - "Layer fusion (combine multiple operations)"
      - "Precision calibration (INT8/FP16 with minimal accuracy loss)"
      - "Kernel auto-tuning (select best kernel for hardware)"
      - "Memory optimization (tensor reuse)"
    speedup: "1.5-5x vs PyTorch eager mode"
    limitation: "NVIDIA GPUs only, compilation time (minutes-hours), static shapes"
    
  onnx_runtime:
    what: "Microsoft's cross-platform inference engine"
    how: "Run ONNX models with hardware-specific optimizations"
    supports: "CPU (Intel, AMD, ARM), GPU (NVIDIA, AMD), NPU (Neural Processing Unit)"
    advantage: "Portable — same model runs on different hardware"
    typical_speedup: "1.3-2x vs PyTorch on CPU"
    
  torch_compile_inference:
    what: "PyTorch 2.0+ compilation for inference (same as training but in eval mode)"
    how: "torch.compile(model, mode='max-autotune') in eval mode"
    speedup: "1.2-2x vs eager mode"
    advantage: "No export step — works directly with PyTorch models"
    
  tensorrt_llm:
    what: "NVIDIA's LLM-specific inference optimization (TensorRT for LLMs)"
    features:
      - "In-flight batching (continuous batching)"
      - "KV cache management"
      - "Multi-GPU tensor parallelism"
      - "INT4/INT8/FP8 quantization"
      - "Speculative decoding"
    integration: "Used as backend for Triton Inference Server"
    performance: "Often fastest raw throughput for LLMs on NVIDIA hardware"
    complexity: "More complex setup than vLLM"
```

---

## How It Works in Practice

### Optimization Decision Framework

```yaml
Decision_Framework:
  step_1_start_with_quantization:
    question: "Does INT4 (AWQ) quality meet your requirements?"
    if_yes: "Deploy with AWQ INT4 — 4x memory savings, 2-4x throughput gain"
    if_no: "Try INT8 or FP8 — less compression but better quality"
    
  step_2_hardware_specific:
    h100_b200: "Use FP8 quantization (native tensor core support, best quality/speed)"
    a100: "Use INT8 (SmoothQuant) or INT4 (AWQ)"
    l4_consumer: "Use INT4 (AWQ/GPTQ) — maximize memory efficiency"
    cpu_edge: "Use GGUF (llama.cpp compatible) with Q4_K_M or Q5_K_M"
    
  step_3_additional_optimization:
    if_more_speed_needed: "Add TensorRT compilation (1.5-2x additional)"
    if_more_memory_needed: "Add pruning (2:4 sparsity for 2x with fine-tuning)"
    if_quality_critical: "Skip INT4, use FP8 or INT8 only"
    
  real_world_examples:
    llama_4_8b:
      fp16: "16 GB, ~100 tok/s on H100"
      awq_int4: "5 GB, ~300 tok/s on H100"
      decision: "Always use AWQ for serving (3x faster, fits on L4)"
      
    llama_4_70b:
      fp16: "140 GB (needs 2× H100)"
      awq_int4: "35 GB (fits on 1× H100)"
      fp8: "70 GB (fits on 1× H100, better quality than INT4)"
      decision: "FP8 on H100 (best quality for one GPU), INT4 if need L4/A100"
```

---

## Interview Tip

> When asked about model optimization for inference: "My optimization stack for production LLM serving: (1) Quantization — AWQ INT4 as default (4x memory reduction, 2-4x throughput, <2% quality loss). FP8 on H100/B200 for better quality retention. GGUF for CPU/edge deployment. (2) Serving framework — vLLM with PagedAttention and continuous batching (2-4x better memory utilization and throughput vs naive serving). (3) Compilation — TensorRT or torch.compile for additional 1.5-2x on top. Together these give 5-10x cost reduction compared to naive FP16 serving. Key trade-off: quality vs cost. INT4 loses 1-3% on benchmarks but saves 4x memory — for most applications this trade-off is easily worthwhile. For quality-critical applications (medical, legal), I use FP8 (0.5% loss) or INT8 (1% loss) instead. Always benchmark on YOUR evaluation set — generic benchmarks don't reflect domain-specific quality impact."

---

## Common Mistakes

1. **Serving in FP32** — Deploying a model in full FP32 precision for inference. There is ZERO reason to serve in FP32 in 2026. BF16 gives identical quality with half the memory. INT4/INT8 give further massive savings with minimal quality loss.

2. **Not evaluating quantized model on YOUR data** — Assuming benchmark numbers apply to your use case. A model that loses 2% on MMLU might lose 10% on your specific domain task. Always evaluate quantized model on your actual evaluation set before deploying.

3. **Using GPTQ when AWQ is available** — GPTQ was state-of-art in 2023 but AWQ generally produces better quality at the same bit-width (especially INT4). AWQ is also faster to calibrate. Use AWQ as default in 2026.

4. **Quantizing then not benchmarking throughput** — Quantizing to save memory but not measuring actual serving performance. Quantization benefits depend on hardware, batch size, and sequence length. Always benchmark: tokens/sec, TTFT (Time To First Token), and throughput under load.

5. **Over-optimizing for latency without considering batch effects** — Optimizing single-request latency (small batch) when production serves many concurrent requests (large batch). Optimization priorities differ: small batch = latency-sensitive (memory bandwidth matters), large batch = throughput-sensitive (compute matters). Profile your actual serving pattern.

---

## Key Takeaways

- Quantization is the #1 optimization for inference: AWQ INT4 (4x memory, 2-4x speed, <2% quality loss)
- FP8: emerging best practice on H100/B200 — better quality than INT8, native hardware acceleration
- AWQ > GPTQ for INT4 in 2026 (better quality, faster calibration)
- GGUF: standard for local/edge deployment (llama.cpp, Ollama) — various bit-widths available
- Pruning: less impactful than quantization for LLMs — useful for extreme compression scenarios
- TensorRT: maximum GPU inference speed but more complex setup (NVIDIA-only)
- ONNX Runtime: portable cross-platform optimization (CPU + GPU)
- torch.compile: easiest optimization (one line of code) — 1.2-2x speedup
- Always evaluate quantized models on YOUR task data — benchmark numbers may not apply
- Optimization stack: quantization (4x) + serving framework (2-4x) + compilation (1.5x) = 10-20x total improvement
