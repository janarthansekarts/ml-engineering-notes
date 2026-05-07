# AI Compiler Optimization

## The Problem / Why This Matters

Neural networks are mathematically defined as computation graphs — sequences of matrix multiplications, activations, normalizations, and attention operations. How efficiently these operations execute depends entirely on how they're mapped to hardware. Naive execution (interpreting Python/PyTorch line-by-line) wastes 60-80% of available GPU compute due to: kernel launch overhead (thousands of small GPU calls), memory bandwidth bottlenecks (moving data between GPU memory and compute units), missed fusion opportunities (operations that could run together but don't), and suboptimal memory allocation. AI compilers solve this by analyzing the computation graph, optimizing it (fusing operations, reordering for memory efficiency, selecting optimal kernel implementations), and generating hardware-specific code that extracts maximum performance. In 2026, the AI compiler landscape includes: `torch.compile` (PyTorch's native compiler using TorchDynamo + TorchInductor), XLA (Google's compiler for TPUs and GPUs), TensorRT (NVIDIA's inference optimizer), TVM (Apache's open-source ML compiler), and Triton (NVIDIA's GPU programming language for custom kernels). For ML engineers, understanding AI compilers is the difference between serving a model at 30 tokens/second versus 100+ tokens/second on identical hardware — a 3× efficiency gain that directly translates to 3× cost reduction or 3× throughput increase.

---

## The Analogy

Think of AI compilers like a factory optimization consultant:

- **Naive execution (Python/PyTorch eager mode)** = A factory where each worker does one operation, then passes the product to the next worker across the factory floor. Constant walking back and forth (memory transfers), workers standing idle waiting for materials (GPU idle cycles), and no consideration for which operations could happen simultaneously.
- **AI compiler** = An industrial engineer who redesigns the factory floor. Groups related operations together (kernel fusion), eliminates unnecessary transport (memory optimization), ensures workers never idle (compute utilization), and customizes workstations for each specific product (hardware-specific kernels). Same workers, same machines, but 2-5× more output.
- **Graph optimization** = Rearranging the assembly line so that sequential operations that use the same materials happen right next to each other (no need to put the material back in the warehouse between steps).
- **Kernel fusion** = Instead of 3 workers each doing a tiny operation on a piece, one worker does all 3 operations in sequence without putting the piece down.

---

## Deep Dive

### AI Compiler Landscape

```yaml
AI_Compiler_Landscape:
  pytorch_ecosystem:
    torch_compile:
      what: "PyTorch's built-in compiler (PyTorch 2.0+)"
      components:
        dynamo: "Python bytecode analysis → captures computation graph"
        aot_autograd: "Ahead-of-time autograd for backward pass"
        inductor: "Generates optimized Triton/C++ kernels"
      usage: "model = torch.compile(model)"
      speedup: "1.3-2× inference, 1.2-1.5× training"
      advantage: "Zero code changes (one line), works with existing PyTorch"
      modes:
        default: "Good balance of compile time and performance"
        reduce_overhead: "Minimal Python overhead (best for inference)"
        max_autotune: "Tries all kernel variants (longest compile, fastest runtime)"
        
    triton:
      what: "NVIDIA's GPU programming language (Python-like syntax for GPU kernels)"
      purpose: "Write custom GPU kernels without CUDA C++ complexity"
      use_case: "Custom operations (flash attention, fused kernels)"
      advantage: "10× easier than CUDA, near-CUDA performance"
      who_uses: "Library authors (flash-attention, xformers, vLLM kernels)"
      
    torch_tensorrt:
      what: "TensorRT integration in PyTorch"
      approach: "Convert PyTorch model → TensorRT optimized engine"
      speedup: "2-5× vs. eager PyTorch (especially for inference)"
      limitation: "Static shapes preferred (dynamic shapes harder to optimize)"
      
  google_ecosystem:
    xla:
      name: "XLA (Accelerated Linear Algebra)"
      what: "Google's compiler for TPUs and GPUs"
      approach: "Trace computation graph → optimize → compile for hardware"
      frameworks: ["JAX (native)", "PyTorch/XLA (bridge)", "TensorFlow"]
      optimizations:
        - "Operation fusion (reduce memory roundtrips)"
        - "Memory layout optimization (for TPU tile structure)"
        - "Automatic parallelization (SPMD)"
      strength: "Best for TPU workloads, excellent for large model training"
      
    jax:
      what: "Google's ML framework (XLA-native, functional programming)"
      features:
        jit: "JIT compilation via XLA (jax.jit decorator)"
        vmap: "Automatic vectorization (batching for free)"
        pmap: "Automatic parallelization across devices"
        grad: "Automatic differentiation"
      advantage: "Composable transformations (jit + vmap + pmap = compiled parallel batched code)"
      
  nvidia_ecosystem:
    tensorrt:
      what: "NVIDIA's inference optimization toolkit"
      approach: "Graph optimization → kernel selection → engine building"
      optimizations:
        - "Layer fusion (merge adjacent operations)"
        - "Precision calibration (mixed FP16/INT8)"
        - "Kernel auto-tuning (select fastest kernel per operation)"
        - "Memory optimization (workspace allocation)"
      speedup: "2-5× vs. native framework (PyTorch/TF)"
      limitation: "Build step required, static graph preferred"
      
    tensorrt_llm:
      what: "TensorRT optimized for LLM inference"
      features:
        - "In-flight batching (continuous batching)"
        - "Paged KV cache"
        - "Tensor parallelism"
        - "Quantization (INT4/INT8/FP8)"
        - "Speculative decoding"
      performance: "Fastest LLM inference on NVIDIA GPUs"
      
  open_source:
    tvm:
      name: "TVM (Tensor Virtual Machine)"
      what: "Open-source ML compiler framework"
      approach: "Hardware-agnostic IR → optimize → generate code for any target"
      targets: ["NVIDIA GPUs", "AMD GPUs", "ARM CPUs", "RISC-V", "FPGAs", "WebGPU"]
      advantage: "Universal — one model, any hardware"
      variant: "Apache TVM (community), OctoML (commercial)"
      
    mlir:
      name: "MLIR (Multi-Level Intermediate Representation)"
      what: "Compiler infrastructure for building domain-specific compilers"
      by: "Google/LLVM project"
      used_by: ["TensorFlow", "JAX", "IREE (inference runtime)", "Torch-MLIR"]
      purpose: "Common infrastructure — don't rebuild compiler from scratch for each framework"
```

### Key Optimization Techniques

```yaml
Compiler_Optimizations:
  kernel_fusion:
    what: "Combine multiple operations into single GPU kernel"
    before: "3 kernels: LayerNorm → Linear → GELU (3 memory roundtrips)"
    after: "1 fused kernel: LayerNorm+Linear+GELU (1 memory roundtrip)"
    benefit: "Eliminate memory bandwidth bottleneck (GPU compute >> memory bandwidth)"
    example:
      unfused: "Read input from memory → LayerNorm → write to memory → read → Linear → write → read → GELU → write"
      fused: "Read input from memory → LayerNorm → Linear → GELU → write final output"
      savings: "4 fewer memory read/write operations (60%+ time reduction for memory-bound ops)"
      
  flash_attention:
    what: "Memory-efficient attention implementation (no materialized attention matrix)"
    problem: "Standard attention: O(N²) memory for attention matrix (128K context = 128GB!)"
    solution: "Compute attention in tiles — never materialize full N×N matrix"
    benefit:
      memory: "O(N) instead of O(N²) — enables million-token contexts"
      speed: "2-4× faster (reduced memory I/O)"
    versions:
      flash_attention_1: "2022 — tile-based exact attention"
      flash_attention_2: "2023 — better work partitioning, 2× faster"
      flash_attention_3: "2024 — H100 optimized, asynchronous, FP8"
    importance: "Without FlashAttention, long-context LLMs wouldn't be practical"
    
  operator_rewriting:
    what: "Replace operations with mathematically equivalent but faster versions"
    examples:
      - "SoftMax → Online SoftMax (streaming, memory-efficient)"
      - "BatchNorm → fused BN (combine normalize + scale + shift)"
      - "Multi-head attention → Grouped Query Attention (fewer KV heads)"
      - "Conv2D small kernels → Winograd transform (fewer multiplications)"
      
  memory_optimization:
    activation_checkpointing:
      what: "Don't store intermediate activations — recompute during backward pass"
      trade_off: "30% more compute, 60-80% less memory"
      when: "Training large models (memory is bottleneck)"
      
    memory_planning:
      what: "Compiler allocates memory optimally (reuse buffers, minimize fragmentation)"
      benefit: "Fit larger models or batches in same memory"
      
    offloading:
      what: "Move unused tensors to CPU, bring back when needed"
      tools: "DeepSpeed ZeRO-Offload, PyTorch FSDP CPU offload"
      
  quantization_and_mixed_precision:
    fp8_training:
      what: "FP8 (8-bit floating point) for forward pass"
      hardware: "H100, H200, B200 (native FP8 tensor cores)"
      benefit: "2× throughput vs. FP16 with minimal quality loss"
      support: "transformer_engine library (NVIDIA)"
      
    mixed_precision:
      what: "Use FP16/BF16 for compute, FP32 for accumulation"
      standard: "Default for all modern training (AMP — Automatic Mixed Precision)"
      benefit: "2× speedup, 2× memory reduction vs. FP32"
```

### Implementation Patterns

```python
# AI compiler optimization patterns

"""
Practical patterns for applying AI compilation and kernel optimization
to ML workloads. Covers torch.compile, custom Triton kernels, and
optimization strategies.
"""

compiler_patterns = {
    "torch_compile_patterns": {
        "basic_usage": {
            "one_liner": "model = torch.compile(model)",
            "with_mode": "model = torch.compile(model, mode='reduce-overhead')",
            "explanation": "Captures computation graph, optimizes, generates Triton kernels",
        },
        "modes": {
            "default": {
                "compile_time": "~30 seconds for first run",
                "speedup": "1.3-1.7× inference",
                "when": "General use, acceptable compile time",
            },
            "reduce_overhead": {
                "compile_time": "~45 seconds",
                "speedup": "1.5-2× inference (minimizes Python overhead)",
                "when": "Inference serving (compile once, run millions of times)",
            },
            "max_autotune": {
                "compile_time": "~5-10 minutes (tries many kernel variants)",
                "speedup": "1.7-2.5× (picks best kernel per operation)",
                "when": "Production models (invest compile time for maximum speed)",
            },
        },
        "gotchas": {
            "dynamic_shapes": {
                "problem": "Different input shapes trigger recompilation",
                "fix": "Use torch.compile(dynamic=True) or pad inputs to fixed sizes",
            },
            "graph_breaks": {
                "problem": "Non-compilable operations force graph breaks (reduces benefit)",
                "common_causes": ["print() statements", "data-dependent control flow", "unsupported ops"],
                "diagnosis": "TORCH_LOGS=graph_breaks python script.py",
                "fix": "Refactor to remove breaks, or accept partial compilation",
            },
            "first_run_latency": {
                "problem": "First call triggers compilation (30s-10min delay)",
                "fix": "Warm up with dummy input before serving (compile during startup)",
            },
        },
    },
    
    "optimization_strategy": {
        "step_1_profile_first": {
            "what": "Understand where time is spent before optimizing",
            "tools": [
                "torch.profiler (PyTorch built-in)",
                "NVIDIA Nsight Systems (GPU timeline)",
                "NVIDIA Nsight Compute (kernel-level analysis)",
            ],
            "key_questions": [
                "Is workload compute-bound or memory-bound?",
                "Which operations take most time?",
                "What's GPU utilization? (< 80% = room for optimization)",
                "Are there unnecessary memory copies?",
            ],
        },
        "step_2_easy_wins": {
            "what": "Low-effort, high-impact optimizations",
            "actions": [
                "Enable torch.compile (1 line of code, 1.3-2× speedup)",
                "Use FlashAttention (usually auto-enabled in recent PyTorch)",
                "Enable BF16/FP16 (torch.autocast, 2× speedup)",
                "Use PagedAttention for LLM serving (vLLM)",
                "Enable continuous batching (vLLM/TGI)",
            ],
        },
        "step_3_quantization": {
            "what": "Reduce precision for inference",
            "actions": [
                "INT4 weights (AWQ/GPTQ — 4× memory reduction)",
                "INT8 KV cache (2× more concurrent requests)",
                "FP8 compute (H100/H200 — 2× compute throughput)",
            ],
        },
        "step_4_advanced": {
            "what": "Custom optimizations for specific bottlenecks",
            "actions": [
                "Custom Triton kernels for unique operations",
                "TensorRT-LLM engine build (maximum NVIDIA performance)",
                "Operation fusion (identify unfused patterns in profiler)",
                "Memory layout optimization (channel-last for Conv, contiguous for attention)",
            ],
        },
    },
    
    "serving_compilation": {
        "vllm_optimizations": {
            "what": "Optimizations built into vLLM serving engine",
            "automatic": [
                "PagedAttention (KV cache memory management)",
                "Continuous batching (maximize GPU utilization)",
                "FlashAttention/FlashInfer (memory-efficient attention)",
                "CUDA graphs (eliminate kernel launch overhead for prefill)",
                "Quantization support (AWQ, GPTQ, FP8)",
                "Speculative decoding",
            ],
            "configuration": {
                "max_model_len": "Set based on expected max sequence length",
                "gpu_memory_utilization": "0.90 (leave 10% for safety)",
                "quantization": "awq or gptq (if using quantized model)",
                "speculative_model": "Path to draft model (for spec decoding)",
            },
        },
        "tensorrt_llm_pipeline": {
            "what": "Maximum performance on NVIDIA hardware",
            "process": [
                "1. Convert model to TensorRT-LLM format",
                "2. Build engine (hardware-specific optimizations)",
                "3. Deploy with Triton Inference Server",
            ],
            "speedup": "1.5-2× faster than vLLM (maximum kernel optimization)",
            "trade_off": "More complex setup, less flexible (rebuild for shape changes)",
        },
    },
    
    "training_compilation": {
        "what": "Compiler optimizations for model training",
        "techniques": {
            "compile_training_loop": {
                "approach": "torch.compile the model (forward + backward compiled)",
                "benefit": "15-30% training speedup",
                "note": "Backward pass compilation is newer, some ops may cause breaks",
            },
            "activation_checkpointing": {
                "what": "Recompute activations in backward pass instead of storing",
                "benefit": "50-80% memory savings (fit larger batch sizes)",
                "cost": "~30% additional compute",
                "when": "GPU memory is the limiting factor for batch size",
            },
            "fsdp_with_compile": {
                "what": "Fully Sharded Data Parallel with torch.compile",
                "benefit": "Distributed training with compilation optimizations",
                "support": "PyTorch 2.4+ (mature integration)",
            },
        },
    },
}


# Performance benchmarks
performance_benchmarks = {
    "llama_3_8b_inference": {
        "hardware": "1× H100 80GB",
        "eager_pytorch": {
            "throughput": "45 tokens/sec (batch=1)",
            "batch_32": "1200 tokens/sec total",
        },
        "torch_compile": {
            "throughput": "72 tokens/sec (batch=1, 1.6×)",
            "batch_32": "1900 tokens/sec total (1.58×)",
        },
        "vllm_int4": {
            "throughput": "120 tokens/sec (batch=1, 2.7×)",
            "batch_32": "3200 tokens/sec total (2.67×)",
        },
        "tensorrt_llm_int4": {
            "throughput": "180 tokens/sec (batch=1, 4×)",
            "batch_32": "4800 tokens/sec total (4×)",
        },
    },
}
```

---

## How It Works in Practice

### Compiler Optimization in Production

```yaml
Compiler_Optimization_Production:
  scenario: "Optimize LLM serving for cost efficiency (reduce GPU bill by 50%)"
  
  baseline:
    model: "Llama 3 70B FP16"
    serving: "Naive PyTorch (eager mode, no optimization)"
    hardware: "4× H100 (tensor parallel)"
    throughput: "800 tokens/sec (total across batch)"
    cost: "$32/hr"
    
  optimization_journey:
    phase_1_torch_compile:
      change: "Add torch.compile(model, mode='max-autotune')"
      effort: "1 line of code + 10 min compile time"
      throughput: "1,200 tokens/sec (1.5×)"
      cost: "$32/hr (same hardware, more throughput)"
      
    phase_2_quantization:
      change: "AWQ INT4 quantization"
      effect: "Model fits on 2× H100 instead of 4× (half the GPUs)"
      throughput: "1,100 tokens/sec (slightly less per GPU, but half the cost)"
      cost: "$16/hr (2× H100 instead of 4×)"
      
    phase_3_vllm:
      change: "Switch to vLLM serving (PagedAttention + continuous batching)"
      effect: "Much better batching efficiency"
      throughput: "2,800 tokens/sec"
      cost: "$16/hr (same hardware)"
      
    phase_4_speculative:
      change: "Add speculative decoding (Llama 3 8B draft)"
      effect: "2.5× generation speedup"
      throughput: "6,500 tokens/sec"
      cost: "$16/hr"
      
  result:
    improvement: "8× throughput improvement, 50% cost reduction"
    original: "800 tokens/sec at $32/hr = $0.04 per 1K tokens"
    final: "6,500 tokens/sec at $16/hr = $0.0025 per 1K tokens"
    cost_per_token: "16× cheaper per token"
```

---

## Interview Tip

> When asked about AI compilers: "AI compilers bridge the gap between model definition and hardware execution. My optimization workflow: (1) Profile first — use torch.profiler or Nsight Systems to identify bottlenecks. Is the workload compute-bound (need faster math) or memory-bound (need less data movement)? Most LLM inference is memory-bandwidth-bound. (2) Easy wins first: torch.compile (1 line, 1.3-2× speedup), FlashAttention (memory-efficient, 2-4× faster attention), BF16 precision (2× compute throughput). (3) For inference serving: vLLM with PagedAttention + continuous batching transforms throughput. Then layer on quantization (INT4 for weights, INT8 for KV cache). (4) For maximum NVIDIA performance: TensorRT-LLM builds hardware-specific optimized engines — 2-5× faster than naive PyTorch. Trade-off is less flexibility and longer build times. Key insight: most performance comes from eliminating memory bandwidth waste (kernel fusion) rather than faster computation. A fused kernel that does LayerNorm+Linear+GELU in one GPU kernel eliminates 4 unnecessary memory roundtrips. On H100, memory bandwidth is 3.35 TB/s but compute is 989 TFLOPS — computation is cheap, moving data is expensive. torch.compile automatically handles most fusion opportunities; for exotic operations, custom Triton kernels give CUDA-level performance with Python-level ergonomics."

---

## Common Mistakes

1. **Optimizing before profiling** — Spending days writing custom Triton kernels for an operation that takes 2% of runtime. The real bottleneck is somewhere else. Solution: ALWAYS profile first (torch.profiler, Nsight Systems). Optimize the operation that takes the most time.

2. **Ignoring torch.compile** — Writing complex optimization code when `torch.compile(model)` would give 1.5× speedup for free. Solution: try torch.compile FIRST. It handles 80% of common optimizations automatically. Only go deeper if you need more.

3. **Not warming up compiled models** — Deploying compiled model and first request takes 30+ seconds (compilation). Users experience terrible first-request latency. Solution: warm up during server startup with dummy inputs (triggers compilation before serving traffic).

4. **Fighting graph breaks instead of accepting them** — Spending days trying to eliminate all graph breaks in torch.compile. Sometimes the gain from eliminating a graph break is minimal. Solution: check TORCH_LOGS to identify graph breaks, fix the important ones (in hot path), accept minor ones (in setup/logging code).

5. **Using TensorRT for rapidly changing models** — Building TensorRT engines (takes minutes-hours) for models that change weekly. Engine rebuild cost exceeds optimization benefit. Solution: use TensorRT for stable production models. Use vLLM + torch.compile for models that change frequently (flexibility > maximum speed).

---

## Key Takeaways

- AI compilers: optimize computation graphs for hardware (fuse operations, eliminate memory waste)
- torch.compile: 1 line of code, 1.3-2× speedup, should be default for all PyTorch inference
- FlashAttention: O(N) memory instead of O(N²), enables long-context LLMs, 2-4× faster
- Kernel fusion: biggest optimization — eliminate unnecessary memory roundtrips between operations
- Most ML workloads are memory-bandwidth-bound (not compute-bound) — reduce data movement
- vLLM: PagedAttention + continuous batching + optimized kernels = production LLM serving
- TensorRT-LLM: maximum NVIDIA performance (2-5× vs. naive PyTorch), less flexibility
- Triton: write custom GPU kernels in Python (10× easier than CUDA, near-CUDA performance)
- XLA/JAX: Google's compiler stack, native for TPUs, excellent for distributed training
- Optimization order: profile → torch.compile → quantize → serve with vLLM → TensorRT if needed
- FP8: next-generation training/inference precision on H100/H200/B200 (2× throughput vs FP16)
- Combined optimizations: 4-16× faster than naive eager mode execution
