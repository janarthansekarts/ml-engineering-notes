# GPU Computing Fundamentals

## The Problem / Why This Matters

Modern ML (Machine Learning) is fundamentally a compute problem. Training a neural network means performing trillions of floating-point operations on massive tensors (multi-dimensional arrays). CPUs (Central Processing Units) are designed for serial execution — fast at one thing at a time but terrible at doing thousands of multiplications simultaneously. GPUs (Graphics Processing Units) flip this: they have thousands of cores designed for parallel computation, making them 10-100x faster than CPUs for ML workloads. Understanding GPU architecture isn't optional for ML engineers — it directly determines: which models you can train (memory limits), how fast you can iterate (throughput), how much you spend (hardware costs), and which optimizations apply to your workload (tensor cores, memory hierarchy, kernel fusion). In 2026, with NVIDIA H100/H200/B200 chips dominating ML infrastructure and GPU supply still constrained, understanding GPU computing is essential for making architecture decisions, debugging training issues, and optimizing both training and inference costs.

---

## The Analogy

Think of CPU vs GPU like a race car vs a bus fleet:

- **CPU** = A Formula 1 race car. Incredibly fast for a single passenger (single-threaded task), with sophisticated navigation (branch prediction, out-of-order execution). But it can only carry one person at a time.
- **GPU** = A fleet of 10,000 buses. Each bus is slower than the F1 car, but they all move simultaneously. Need to transport 10,000 people across town? The bus fleet finishes in one trip while the F1 car takes 10,000 trips.
- **ML training** = Moving 10,000 people (matrix multiplications). You don't need the F1 car's sophistication — you need massive parallelism. That's why GPUs dominate ML.
- **Tensor Cores** = Express buses that carry 4x more people per trip (specialized hardware for matrix multiplication).

---

## Deep Dive

### GPU Architecture for ML

```yaml
GPU_Architecture:
  core_concept:
    description: "GPU = thousands of simple cores executing the same operation on different data (SIMD/SIMT)"
    simt: "Single Instruction Multiple Threads — one instruction, thousands of threads execute it simultaneously"
    
  nvidia_gpu_hierarchy:
    gpu_chip:
      contains: "Multiple Streaming Multiprocessors (SMs)"
      example: "H100 has 132 SMs"
    streaming_multiprocessor:
      contains: "CUDA cores + Tensor Cores + shared memory + registers"
      cuda_cores: "General-purpose floating-point units"
      tensor_cores: "Specialized matrix multiplication hardware (4th gen in H100)"
    warp:
      definition: "Group of 32 threads executing in lockstep"
      importance: "All 32 threads in a warp execute the SAME instruction simultaneously"
      implication: "Branch divergence (if/else) within a warp wastes cycles"
      
  memory_hierarchy:
    registers:
      location: "Per-thread (fastest)"
      size: "~256KB per SM"
      latency: "1 cycle"
      
    shared_memory:
      location: "Per-SM (shared between threads in a block)"
      size: "H100: 228KB per SM"
      latency: "~20-30 cycles"
      use: "Thread communication, data reuse within block"
      
    l2_cache:
      location: "Chip-level (shared across all SMs)"
      size: "H100: 50MB"
      latency: "~200 cycles"
      
    hbm:
      full_name: "High Bandwidth Memory"
      location: "Off-chip (GPU main memory)"
      size: "H100: 80GB HBM3, H200: 141GB HBM3e"
      bandwidth: "H100: 3.35 TB/s, H200: 4.8 TB/s"
      latency: "~400-600 cycles"
      
  bottleneck_insight:
    principle: "Most ML operations are MEMORY-BOUND, not compute-bound"
    explanation: "The GPU can compute faster than it can read/write memory"
    implication: "Optimization often means reducing memory transfers, not reducing math"
    metric: "Arithmetic intensity = FLOPs / bytes transferred (higher = better GPU utilization)"
```

### NVIDIA GPU Lineup (2026)

```yaml
NVIDIA_GPUs_2026:
  b200:
    generation: "Blackwell (newest)"
    memory: "192GB HBM3e"
    performance: "~2.5x H100 per watt for training"
    tensor_cores: "5th generation"
    availability: "Limited (ramping production 2026)"
    use_case: "Frontier model training, largest models"
    cloud_cost: "$6-10/hour (early access pricing)"
    
  h200:
    generation: "Hopper (enhanced)"
    memory: "141GB HBM3e (76% more than H100)"
    bandwidth: "4.8 TB/s (43% more than H100)"
    tensor_cores: "4th generation"
    availability: "Growing availability (2025-2026)"
    use_case: "LLM fine-tuning, large model inference, training"
    cloud_cost: "$4-6/hour"
    advantage: "More memory = larger batch sizes, less memory pressure"
    
  h100:
    generation: "Hopper"
    memory: "80GB HBM3"
    bandwidth: "3.35 TB/s"
    tensor_cores: "4th generation"
    fp8_performance: "3958 TFLOPS (peak)"
    availability: "Widely available (most common high-end ML GPU)"
    use_case: "Standard for LLM training and high-throughput inference"
    cloud_cost: "$2-4/hour"
    
  a100:
    generation: "Ampere (previous gen)"
    memory: "40GB or 80GB HBM2e"
    bandwidth: "2 TB/s (80GB variant)"
    tensor_cores: "3rd generation"
    availability: "Widely available, good pricing"
    use_case: "Most production workloads, fine-tuning, medium inference"
    cloud_cost: "$1-2.50/hour"
    value: "Best price/performance for many workloads in 2026"
    
  l4:
    generation: "Ada Lovelace (inference-optimized)"
    memory: "24GB GDDR6"
    tensor_cores: "4th generation"
    availability: "Widely available"
    use_case: "Cost-efficient inference, smaller model training"
    cloud_cost: "$0.50-1/hour"
    
  t4:
    generation: "Turing (older)"
    memory: "16GB GDDR6"
    availability: "Ubiquitous, cheapest GPU option"
    use_case: "Budget inference, development, small models"
    cloud_cost: "$0.35-0.75/hour"
```

### CUDA and ML Computation

```yaml
CUDA_Fundamentals:
  what_is_cuda:
    full_name: "Compute Unified Device Architecture"
    purpose: "Programming model for NVIDIA GPUs"
    reality: "ML engineers rarely write CUDA directly — frameworks (PyTorch, TensorFlow) handle it"
    but_understanding_matters: "Debugging, optimization, and choosing right approaches require CUDA understanding"
    
  execution_model:
    kernel: "Function that runs on GPU (thousands of threads simultaneously)"
    grid: "Collection of thread blocks launched for a kernel"
    block: "Group of threads that can share memory and synchronize"
    thread: "Single execution unit (runs on one CUDA core)"
    
  ml_relevant_cuda_concepts:
    tensor_cores:
      what: "Specialized hardware for matrix multiply-accumulate (D = A×B + C)"
      operation: "Multiplies 4×4 matrices in one cycle (vs many cycles for CUDA cores)"
      precision: "FP16×FP16→FP32, BF16×BF16→FP32, FP8×FP8→FP16, INT8×INT8→INT32"
      impact: "10-20x faster than CUDA cores for matrix operations"
      usage: "Automatic via PyTorch (with correct precision settings)"
      
    memory_coalescing:
      what: "Adjacent threads access adjacent memory locations"
      why: "GPU memory is accessed in chunks — scattered access wastes bandwidth"
      impact: "Coalesced access can be 10x faster than scattered access"
      
    kernel_fusion:
      what: "Combining multiple operations into one GPU kernel"
      why: "Reduces memory round-trips (compute in registers instead of reading/writing HBM)"
      example: "Fusing LayerNorm + ReLU + Linear into one kernel"
      tools: "torch.compile (TorchDynamo + TorchInductor), FlashAttention"
      
  frameworks_over_cuda:
    pytorch: "Most popular for research and production ML (Python API, CUDA backend)"
    jax: "Google's framework (functional, XLA compilation, TPU support)"
    triton: "OpenAI's language for writing custom GPU kernels (simpler than CUDA)"
    purpose: "ML engineers use frameworks — CUDA knowledge helps debug/optimize"
```

### Memory Management for ML

```yaml
GPU_Memory_Management:
  what_consumes_memory:
    model_parameters:
      description: "Weights and biases of the neural network"
      example: "Llama-4-8B: 8 billion params × 2 bytes (FP16) = 16 GB"
      example_large: "Llama-4-70B: 70B × 2 bytes = 140 GB (multiple GPUs required)"
      
    optimizer_states:
      description: "Adam optimizer stores momentum and variance for each parameter"
      overhead: "2x model size for Adam (momentum + variance, both FP32)"
      example: "8B model in FP16 (16GB) + Adam states in FP32 (64GB) = 80GB total"
      
    gradients:
      description: "Computed during backward pass, same size as parameters"
      overhead: "1x model size (FP16 or FP32 depending on precision)"
      
    activations:
      description: "Intermediate computation results kept for backward pass"
      overhead: "Proportional to batch_size × sequence_length × hidden_size × num_layers"
      key_insight: "Often the LARGEST memory consumer during training"
      reduction: "Activation checkpointing — recompute instead of store (time vs memory trade-off)"
      
    kv_cache:
      description: "Cached key-value pairs for autoregressive generation (inference only)"
      overhead: "Grows with sequence length × batch_size"
      importance: "Often the memory bottleneck for LLM inference at long context"
      optimization: "PagedAttention (vLLM), quantized KV cache"
      
  memory_optimization:
    mixed_precision:
      what: "Use FP16/BF16 for forward/backward, FP32 for critical operations"
      benefit: "~50% memory reduction, 2x throughput on tensor cores"
      how: "torch.cuda.amp.autocast() in PyTorch"
      
    gradient_accumulation:
      what: "Accumulate gradients over multiple micro-batches before updating"
      benefit: "Simulates larger batch size without needing memory for full batch"
      trade_off: "Slower (more forward/backward passes per update)"
      
    activation_checkpointing:
      what: "Don't store all activations — recompute during backward pass"
      benefit: "Dramatic memory reduction (store only checkpoint layers)"
      trade_off: "~30% more compute (recomputation cost)"
      
    model_parallelism:
      what: "Split model across multiple GPUs"
      benefit: "Train models larger than single GPU memory"
      types: "Tensor parallelism, pipeline parallelism, ZeRO optimization"
```

---

## How It Works in Practice

### Choosing the Right GPU

```yaml
Example:
  decision_framework:
    model_type: "What are you training/serving?"
    
    scenarios:
      xgboost_tabular:
        requirement: "CPU usually sufficient, GPU optional"
        recommended: "No GPU needed, or T4 if using GPU-accelerated XGBoost"
        cost: "$0/hour (CPU) or $0.35/hour (T4)"
        
      bert_fine_tuning:
        requirement: "~4-8 GB GPU memory"
        recommended: "T4 (16GB) or L4 (24GB)"
        cost: "$0.35-1/hour"
        batch_size: "32 on T4, 64 on L4"
        
      llama_4_8b_fine_tuning:
        requirement: "~24-40 GB for LoRA fine-tuning (QLoRA: ~16 GB)"
        recommended: "A100 40GB (LoRA) or L4 24GB (QLoRA with 4-bit)"
        cost: "$1-2.50/hour"
        
      llama_4_70b_fine_tuning:
        requirement: "~160 GB for QLoRA (4-bit base + LoRA adapters)"
        recommended: "2x H100 80GB or 4x A100 80GB"
        cost: "$4-10/hour"
        
      llm_inference_8b:
        requirement: "~16 GB (FP16) or ~5 GB (INT4)"
        recommended: "L4 (cost-efficient) or T4 (budget, INT4 quantized)"
        throughput: "L4: ~200 tokens/sec, T4: ~100 tokens/sec"
        
      llm_inference_70b:
        requirement: "~140 GB (FP16) or ~35 GB (INT4)"
        recommended: "2x H100 (FP16, max quality) or 1x H100 (INT4, good quality)"
        throughput: "2x H100: ~100 tokens/sec per request"
```

---

## Interview Tip

> When asked about GPU computing for ML: "I think about GPUs across three dimensions: memory (determines max model size — H100 has 80GB, enough for 40B parameter model in FP16), compute (determines training speed — tensor cores provide 10-20x speedup for matrix operations vs CUDA cores), and bandwidth (determines whether you're compute-bound or memory-bound — most ML operations are memory-bound). For GPU selection, I match to workload: T4/L4 for inference and small model training, A100 for standard training and fine-tuning, H100/H200 for LLM training and high-throughput serving. Key optimizations: mixed precision (FP16/BF16 for 2x speedup), activation checkpointing (trade compute for memory), and gradient accumulation (simulate large batches). For LLM inference, the KV cache is usually the memory bottleneck — PagedAttention (vLLM) solves this elegantly."

---

## Common Mistakes

1. **Using H100 for XGBoost** — Over-provisioning GPU for workloads that don't benefit from it. XGBoost on tabular data barely uses GPU. Match hardware to workload — use expensive GPUs only for workloads that saturate them.

2. **Ignoring memory bandwidth** — Choosing GPU based on FLOPS alone. Most ML operations are memory-bound (limited by how fast data moves, not how fast it computes). H200's 4.8 TB/s bandwidth is more impactful than raw TFLOPS for inference.

3. **Not using mixed precision** — Training in FP32 when FP16/BF16 would give 2x speedup with negligible quality loss. BF16 (Brain Float 16) is especially safe — same dynamic range as FP32, just less precision.

4. **Running out of memory without understanding why** — GPU OOM (Out of Memory) with "batch_size=1". Not realizing that optimizer states (Adam: 2x model), gradients (1x model), and activations (variable, often largest) consume far more memory than the model weights alone.

5. **Paying for idle GPUs** — GPU instances running 24/7 for training jobs that run a few hours per week. Use spot instances, auto-scaling to zero, or serverless GPU platforms for sporadic workloads.

---

## Key Takeaways

- GPUs dominate ML because of massive parallelism (thousands of cores) suited for matrix operations
- Memory hierarchy: registers > shared memory > L2 cache > HBM (each level: larger, slower)
- Tensor Cores: specialized matrix multiply hardware — 10-20x faster than CUDA cores for matmul
- Most ML operations are MEMORY-BOUND — optimization means reducing data movement, not just math
- GPU memory consumed by: parameters + optimizer states + gradients + activations (activations often largest)
- NVIDIA lineup 2026: B200 (frontier) > H200 (more memory) > H100 (standard high-end) > A100 (value) > L4 (inference) > T4 (budget)
- Mixed precision (FP16/BF16): 2x throughput, ~50% memory reduction — use always
- Activation checkpointing: trade ~30% more compute for dramatic memory savings
- Match GPU to workload: T4 for small inference, A100 for training, H100+ for LLM work
- Understanding GPU architecture helps debug OOM errors, choose optimizations, and select hardware
