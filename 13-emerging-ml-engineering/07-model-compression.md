# Model Compression

## The Problem / Why This Matters

State-of-the-art models are getting larger: GPT-5 has 1.8 trillion parameters, Llama 3 405B takes 810GB in FP16, and even "small" models like Llama 3 70B need 140GB — requiring multiple expensive GPUs just to load into memory. For production ML engineers, this creates a fundamental tension: larger models produce better outputs, but the cost, latency, and infrastructure requirements make them impractical for many deployments. Model compression bridges this gap by making models smaller, faster, and cheaper while preserving as much quality as possible. In 2026, the compression toolkit includes: quantization (reduce numerical precision — INT4 cuts memory 4×), pruning (remove unnecessary parameters — 30-60% sparsity with minimal quality loss), knowledge distillation (train a smaller model to mimic a larger one), and speculative decoding (use draft model to speed up large model generation). These techniques aren't just academic — they're the difference between deploying a model on a $1/hour L4 GPU versus needing an $8/hour H100 cluster. They're the difference between 20ms latency (on-device) and 500ms latency (cloud API). For edge deployment (mobile, browser, IoT), compression is mandatory since there's no other way to fit powerful models on constrained hardware. Model compression engineering encompasses: selecting the right technique for your constraints, implementing compression without excessive quality loss, serving compressed models efficiently, and evaluating quality/speed trade-offs.

---

## The Analogy

Think of model compression like video compression:

- **Uncompressed model (FP32)** = Raw video footage (ProRes 4K). Massive file sizes, perfect quality, requires expensive equipment to play back. Used in production studios but impractical for streaming to phones.
- **Quantization** = Converting to H.264/H.265. Same resolution but uses efficient encoding. 10× smaller file with barely perceptible quality difference. Most viewers can't tell the difference. This is the standard for streaming.
- **Pruning** = Cropping irrelevant areas from the frame and reducing frame rate for static scenes. Remove parts of the video that don't contribute to understanding. 30-50% smaller with minimal information loss.
- **Distillation** = A professional video editor creates a highlight reel. Takes 2-hour footage and creates a 20-minute version that captures 90% of the important content. Fundamentally smaller, loses some nuance, but serves most viewers perfectly.
- **Speculative decoding** = Predictive buffering. A cheap low-quality preview loads instantly, then high-quality renders in parallel and replaces it — viewer sees seamless high quality but gets fast initial response.

---

## Deep Dive

### Quantization

```yaml
Quantization:
  what: "Reduce numerical precision of model weights (and optionally activations)"
  
  precision_levels:
    fp32:
      bits: 32
      bytes_per_param: 4
      quality_loss: "None (baseline)"
      use: "Training only (not practical for inference)"
      model_7b_size: "28 GB"
      
    fp16_bf16:
      bits: 16
      bytes_per_param: 2
      quality_loss: "Negligible"
      use: "Standard GPU inference"
      model_7b_size: "14 GB"
      note: "BF16 preferred (better training stability, same memory)"
      
    int8:
      bits: 8
      bytes_per_param: 1
      quality_loss: "< 1% on most benchmarks"
      use: "When FP16 doesn't fit on GPU, or CPU inference"
      model_7b_size: "7 GB"
      techniques: ["LLM.int8() (bitsandbytes)", "SmoothQuant"]
      
    int4:
      bits: 4
      bytes_per_param: 0.5
      quality_loss: "1-3% (task dependent)"
      use: "Primary production format (best quality/size trade-off)"
      model_7b_size: "3.5 GB"
      techniques: ["GPTQ", "AWQ", "GGUF (llama.cpp)"]
      
    int3_int2:
      bits: "2-3"
      bytes_per_param: "0.25-0.375"
      quality_loss: "5-15% (significant on complex tasks)"
      use: "Ultra-constrained environments (wearables, tiny IoT)"
      model_7b_size: "1.75-2.6 GB"
      
  quantization_methods:
    post_training_quantization:
      what: "Quantize already-trained model (no additional training)"
      
      gptq:
        name: "GPTQ (GPT Quantization)"
        approach: "Layer-by-layer quantization minimizing reconstruction error"
        calibration: "Requires small calibration dataset (128-1024 samples)"
        quality: "Best INT4 quality for most models"
        speed: "Quantization takes 1-4 hours for 70B model"
        serving: "Optimized kernels (ExLlama, AutoGPTQ)"
        
      awq:
        name: "AWQ (Activation-Aware Weight Quantization)"
        approach: "Protect 1% most important weights (based on activation magnitudes)"
        key_insight: "Not all weights equally important — protect salient ones"
        quality: "Slightly better than GPTQ on average"
        speed: "Faster quantization (minutes for 7B)"
        serving: "vLLM native support"
        
      gguf:
        name: "GGUF (GPT-Generated Unified Format)"
        approach: "llama.cpp quantization (multiple quant levels per layer)"
        variants: ["Q4_K_M (recommended)", "Q5_K_M (higher quality)", "Q3_K_S (smaller)"]
        strength: "CPU inference optimized (AVX, ARM NEON, Metal)"
        serving: "llama.cpp, Ollama, LM Studio"
        
    quantization_aware_training:
      what: "Train model knowing it will be quantized (learns to be robust to low precision)"
      approach: "Simulate quantization during forward pass, full precision gradients"
      quality: "Better than post-training quant (model adapts to reduced precision)"
      cost: "Requires full training run (expensive)"
      when: "Maximum quality at INT4 is critical"
      
  practical_guidelines:
    recommendation_2026:
      default: "AWQ INT4 (best quality + vLLM support)"
      cpu_edge: "GGUF Q4_K_M (llama.cpp optimized)"
      maximum_quality: "INT8 (minimal quality loss)"
      research_training: "FP16/BF16 (no quantization)"
```

### Pruning

```yaml
Pruning:
  what: "Remove parameters (weights) that contribute minimally to model output"
  
  types:
    unstructured_pruning:
      what: "Set individual weights to zero (any position)"
      sparsity: "50-90% weights can be zeroed"
      quality_at_50_percent: "< 1% degradation"
      quality_at_90_percent: "5-10% degradation"
      challenge: "Sparse matrices hard to accelerate without specialized hardware"
      hardware: "NVIDIA Ampere+ (sparse tensor cores, 2:4 structured sparsity)"
      
    structured_pruning:
      what: "Remove entire neurons, attention heads, or layers"
      approaches:
        head_pruning: "Remove attention heads with lowest importance"
        layer_pruning: "Remove entire transformer layers (middle layers often redundant)"
        width_pruning: "Reduce hidden dimension (fewer neurons per layer)"
      advantage: "Results in smaller dense model (no sparse hardware needed)"
      typical_reduction: "20-40% of model removed"
      quality: "2-5% degradation at 30% pruning"
      
    semi_structured:
      name: "2:4 Sparsity (NVIDIA)"
      what: "For every 4 consecutive weights, 2 must be zero"
      benefit: "2× speedup on NVIDIA Ampere+ GPUs (hardware supported)"
      quality: "< 1% degradation (minimal)"
      support: "Native in PyTorch 2.0+, NVIDIA TensorRT"
      
  pruning_criteria:
    magnitude: "Remove smallest absolute-value weights"
    gradient: "Remove weights with smallest gradient (least impact on loss)"
    sensitivity: "Remove weights whose removal changes output least"
    wanda: "Pruning based on weight × activation (state-of-art 2026)"
    
  practical_pruning_pipeline:
    step_1: "Identify importance scores for all weights"
    step_2: "Create pruning mask (which weights to remove)"
    step_3: "Apply mask (zero out / remove selected weights)"
    step_4: "Fine-tune briefly (recover quality lost from pruning)"
    step_5: "Evaluate on benchmark suite"
    step_6: "If quality acceptable → deploy; if not → reduce sparsity"
```

### Knowledge Distillation

```yaml
Knowledge_Distillation:
  what: "Train a smaller (student) model to mimic a larger (teacher) model"
  
  approaches:
    output_distillation:
      teacher_provides: "Final outputs (answers, classifications)"
      student_learns: "To produce same outputs as teacher"
      data: "Unlabeled data → run through teacher → use teacher outputs as labels"
      quality: "Student achieves 85-90% of teacher quality"
      example:
        teacher: "GPT-5 (1.8T parameters)"
        student: "Phi-4 14B (trained on GPT-5 outputs)"
        result: "14B model with quality approaching 1.8T model on specific tasks"
        
    logit_distillation:
      teacher_provides: "Full probability distribution (soft labels)"
      student_learns: "Teacher's confidence over ALL classes (not just top-1)"
      benefit: "Richer signal — student learns what teacher is uncertain about"
      temperature: "Scale logits by temperature T (softer distribution = more knowledge transfer)"
      
    reasoning_distillation:
      teacher_provides: "Step-by-step reasoning chains (chain-of-thought)"
      student_learns: "HOW to reason, not just final answers"
      process:
        - "Give complex problems to o3/o4 (reasoning model)"
        - "Capture full reasoning trace"
        - "Train student to reproduce reasoning then answer"
      quality: "Student achieves 90-95% of teacher (learns reasoning patterns)"
      note: "This is how Phi-4 and Orca models were created"
      
    feature_distillation:
      teacher_provides: "Intermediate layer representations"
      student_learns: "To match teacher's internal representations"
      benefit: "Deeper knowledge transfer (not just input-output mapping)"
      complexity: "Requires architecture alignment (dimension matching)"
      
  practical_distillation:
    data_generation:
      volume: "10K-1M examples (more = better, diminishing returns after 100K)"
      diversity: "Cover full task distribution (easy + hard + edge cases)"
      quality_filter: "Remove teacher errors (LLM-as-judge, score ≥ 4/5)"
      
    training:
      method: "Fine-tune student on teacher outputs (LoRA or full)"
      epochs: "3-5 (watch for overfitting)"
      evaluation: "Compare student vs. teacher on held-out test set"
      
    when_to_use:
      - "Need to reduce serving cost (student is 10-100× cheaper to run)"
      - "Need lower latency (smaller model = faster generation)"
      - "Need on-device deployment (teacher doesn't fit on device)"
      - "Task is well-defined (can generate comprehensive training data)"
```

### Speculative Decoding

```python
# Speculative decoding and other inference acceleration

"""
Techniques for faster inference without changing model quality:
speculative decoding, layer skipping, and early exit.
"""

inference_acceleration = {
    "speculative_decoding": {
        "what": "Draft tokens with cheap model, verify with expensive model in parallel",
        "mechanism": {
            "step_1": "Draft model generates K tokens autoregressively (fast, small model)",
            "step_2": "Target model evaluates ALL K tokens in single forward pass (parallel)",
            "step_3": "Accept tokens where draft matches target (typically 70-90% match)",
            "step_4": "Reject remaining tokens, regenerate from rejection point",
            "step_5": "Net result: target model quality, 2-3× faster generation",
        },
        "why_it_works": {
            "key_insight": "Verifying K tokens in parallel (one forward pass) is almost as fast as generating 1 token",
            "typical_acceptance": "70-90% of draft tokens accepted",
            "speedup": "2-3× for well-matched draft/target pairs",
        },
        "configurations": {
            "same_architecture": {
                "target": "Llama 3 70B",
                "draft": "Llama 3 8B (same tokenizer, similar architecture)",
                "speedup": "2.5×",
                "quality_loss": "ZERO (target model verifies everything)",
            },
            "self_speculative": {
                "target": "Model uses its own early layers as draft",
                "mechanism": "Early exit after layer N → predict → verify with full model",
                "advantage": "No separate draft model needed",
                "speedup": "1.5-2×",
            },
            "medusa": {
                "what": "Add multiple prediction heads to target model",
                "mechanism": "Each head predicts a different future token",
                "advantage": "Trained specifically for this target model (high acceptance)",
                "speedup": "2-3× with trained heads",
            },
        },
        "serving_support": {
            "vllm": "Native speculative decoding support",
            "tgi": "Speculation with configurable draft model",
            "tensorrt_llm": "Optimized speculation kernels",
        },
    },
    
    "kv_cache_compression": {
        "what": "Reduce memory used by KV cache during generation",
        "problem": "KV cache grows linearly with sequence length — dominant memory user for long sequences",
        "techniques": {
            "grouped_query_attention": {
                "what": "Share KV heads across multiple query heads",
                "example": "8 KV heads shared across 32 query heads (4× KV memory reduction)",
                "models": "Standard in Llama 3, Mistral, Gemma 2 (built into architecture)",
            },
            "quantized_kv_cache": {
                "what": "Store KV cache in INT8 instead of FP16",
                "memory_reduction": "2×",
                "quality_impact": "< 0.5% (KV values are less sensitive to precision)",
                "support": "vLLM native support",
            },
            "sliding_window": {
                "what": "Only keep last N tokens in KV cache (discard old)"   ,
                "benefit": "Fixed memory regardless of sequence length",
                "use_case": "Streaming/chat (recent context most relevant)",
                "model": "Mistral uses sliding window (4096 tokens) by default",
            },
            "paged_attention": {
                "what": "Manage KV cache like virtual memory pages (avoid fragmentation)",
                "benefit": "Near-optimal memory utilization (no wasted pre-allocated space)",
                "implementation": "vLLM's core innovation — PagedAttention",
            },
        },
    },
    
    "compilation_and_fusion": {
        "torch_compile": {
            "what": "PyTorch 2.0+ graph compilation for optimized execution",
            "speedup": "1.3-2× for inference (fuses operations, eliminates overhead)",
            "usage": "model = torch.compile(model, mode='reduce-overhead')",
            "modes": {
                "default": "Good balance of compile time and speedup",
                "reduce-overhead": "Best for inference (eliminates Python overhead)",
                "max-autotune": "Best possible speed (long compile time)",
            },
        },
        "tensorrt": {
            "what": "NVIDIA's inference optimizer (graph optimization + kernel fusion)",
            "speedup": "2-5× vs. native PyTorch",
            "process": "Export model → TensorRT optimizes → deploy optimized engine",
            "support": "TensorRT-LLM for language models",
        },
        "onnx_runtime": {
            "what": "Cross-platform inference runtime (CPU + GPU + NPU)",
            "strength": "Works everywhere (Windows, Linux, macOS, mobile, edge)",
            "quantization": "Built-in INT4/INT8 quantization during optimization",
        },
    },
    
    "practical_speedups": {
        "batch_7b_model": {
            "technique": "Continuous batching (vLLM)",
            "hardware": "1× L4 24GB",
            "unoptimized": "5 requests/second",
            "optimized": "80 requests/second",
            "optimizations": ["INT4 quant", "PagedAttention", "continuous batching"],
            "speedup": "16×",
        },
        "speculative_70b": {
            "technique": "Speculative decoding with 8B draft",
            "hardware": "2× H100",
            "without_speculation": "30 tokens/second",
            "with_speculation": "75 tokens/second",
            "speedup": "2.5×",
            "quality": "Identical (target verifies all tokens)",
        },
    },
}
```

---

## How It Works in Practice

### Compression Pipeline for Production

```yaml
Compression_Production:
  scenario: "Deploy Llama 3 70B for real-time customer support (latency < 2 sec)"
  
  baseline:
    model: "Llama 3 70B FP16"
    memory: "140 GB (needs 2× H100 80GB)"
    latency: "4.2 seconds (TTFT + generation)"
    throughput: "15 requests/second"
    cost: "$16/hr (2× H100)"
    
  optimization_steps:
    step_1_quantization:
      technique: "AWQ INT4"
      memory: "35 GB (fits on 1× H100 80GB with KV cache room)"
      latency: "3.1 seconds"
      quality_loss: "1.8% on eval benchmarks"
      cost: "$8/hr (1× H100)"
      savings: "50% cost reduction"
      
    step_2_speculative_decoding:
      draft_model: "Llama 3 8B (INT4, fits alongside 70B on same GPU)"
      acceptance_rate: "78%"
      latency: "1.4 seconds (2.2× faster)"
      quality_loss: "0% additional (target verifies all tokens)"
      throughput: "35 requests/second"
      
    step_3_kv_cache_optimization:
      technique: "Quantized KV cache (INT8) + PagedAttention"
      effect: "Handle 3× more concurrent requests (less memory per request)"
      throughput: "95 requests/second"
      
    step_4_compilation:
      technique: "torch.compile(mode='reduce-overhead')"
      effect: "1.3× speedup on forward pass"
      final_latency: "1.1 seconds (from 4.2 seconds baseline)"
      
  final_result:
    model: "Llama 3 70B AWQ INT4 + speculative decoding + optimizations"
    hardware: "1× H100 80GB"
    latency: "1.1 seconds (target was < 2 sec ✓)"
    throughput: "95 requests/second"
    quality: "98.2% of FP16 quality"
    cost: "$8/hr (vs. $16/hr baseline — 50% savings)"
    improvement: "3.8× faster, 50% cheaper, fits on single GPU"
```

---

## Interview Tip

> When asked about model compression: "I approach compression as an optimization pipeline: quantization first (biggest bang for buck), then architectural optimizations, then speculative decoding for latency. Quantization: AWQ INT4 is my default — 4× memory reduction with 1-3% quality loss. This typically lets me go from 2 GPUs to 1 GPU (50% cost savings immediately). For edge/mobile, GGUF Q4_K_M with llama.cpp (CPU-optimized). Speculative decoding: I pair a small draft model (same architecture family) with the target model. Zero quality loss, 2-3× faster generation. The key insight is that verifying K tokens in parallel (one forward pass) is nearly free, and the draft model is right 70-90% of the time. For maximum compression: knowledge distillation — train a 7B model on 70B model's outputs. Achieves 85-90% of teacher quality at 1/10th the serving cost. Use when task is well-defined and you can generate comprehensive training data. Pruning: 2:4 structured sparsity gives 2× speedup on NVIDIA Ampere+ GPUs with <1% quality loss — it's essentially free performance if your hardware supports it. My evaluation discipline: always measure quality AFTER compression on YOUR task's test set, not just general benchmarks. Some tasks (math, reasoning) degrade more than others (classification, extraction)."

---

## Common Mistakes

1. **Quantizing without per-task evaluation** — Applying INT4 quantization and assuming the 1-3% degradation from benchmarks applies to your specific task. Some tasks (multi-step math, coding) degrade 5-10% while others (classification) degrade 0.5%. Solution: always evaluate on YOUR test set after quantization. If degradation is unacceptable, try INT8 or a different quantization method.

2. **Using wrong quantization format for deployment target** — Quantizing with GPTQ (GPU-optimized) then trying to run on CPU. Or using GGUF (CPU-optimized) on GPU. Solution: match quantization format to serving infrastructure. GPU → AWQ/GPTQ (vLLM/TGI). CPU → GGUF (llama.cpp/Ollama). Mobile → ONNX/TFLite.

3. **Ignoring speculative decoding** — Accepting slow generation speed without trying speculative decoding. It's zero quality loss with 2-3× speedup — essentially free performance. Solution: always try speculative decoding when generation latency matters. Use a smaller model from the same family as draft (Llama 3 8B as draft for Llama 3 70B).

4. **Pruning without fine-tuning afterward** — Pruning 50% of weights and deploying immediately. Quality drops significantly. Solution: always fine-tune briefly (1-3 epochs on task data) after pruning. This recovers most of the quality lost from pruning.

5. **Over-compressing for the task** — Using INT2 quantization or extreme pruning when INT4 would suffice. Chasing smaller size beyond what's needed, sacrificing quality unnecessarily. Solution: determine your actual constraints (memory budget, latency target) and use the LEAST aggressive compression that meets them. Don't compress more than necessary.

---

## Key Takeaways

- Model compression: make models smaller/faster while preserving quality
- Quantization (INT4): 4× memory reduction, 1-3% quality loss — should be default for inference
- AWQ: best general-purpose INT4 method (vLLM support). GGUF: best for CPU/edge (llama.cpp)
- Pruning: remove 30-60% of weights with structured/unstructured approaches
- 2:4 sparsity: 2× NVIDIA GPU speedup with <1% quality loss (hardware-native support)
- Knowledge distillation: train small model on large model outputs → 85-95% quality at 1/10th cost
- Speculative decoding: 2-3× faster generation with ZERO quality loss (draft model + verification)
- KV cache optimization: quantized cache (INT8), PagedAttention, sliding window
- torch.compile: 1.3-2× speedup from graph compilation (one line of code)
- Compression pipeline: quantize → speculative decode → compile → evaluate
- Always evaluate compression on YOUR task (not just general benchmarks)
- Combined optimizations: 3-5× total speedup from stacking techniques
