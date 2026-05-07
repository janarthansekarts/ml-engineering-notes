# Inference Hardware

## The Problem / Why This Matters

The hardware you choose for model inference determines your latency floor, maximum throughput, and cost per prediction. A model served on the wrong hardware delivers either unacceptable performance or unacceptable cost — often both. The inference hardware landscape in 2026 is fragmented and rapidly evolving: NVIDIA dominates with H100/H200/B200 GPUs but faces competition from custom silicon (Google TPUs, AWS Inferentia/Trainium, Apple Neural Engine), CPU-optimized inference (Intel AMX, AMD EPYC), and specialized inference accelerators (Groq LPU, Cerebras WSE). The key insight: training hardware selection is about maximum throughput (batch everything, saturate compute), but inference hardware selection is about latency-throughput balance (real users are waiting). Different workloads (LLM generation, embedding, classification, vision) have different bottlenecks (memory bandwidth, compute, batch size), which means different optimal hardware. Understanding these trade-offs is what separates a $10K/month serving bill from a $100K/month one for the same workload.

---

## The Analogy

Think of inference hardware like vehicle choices for a delivery service:

- **CPU (Intel/AMD)** = A reliable sedan. Works for any delivery, handles diverse cargo. Not fast, but versatile and cheap per vehicle. Great when you have many small packages (lightweight models) or occasional deliveries (low traffic).
- **NVIDIA H100 GPU** = A high-performance truck. Carries massive loads at high speed. Expensive per vehicle but incredible throughput per trip. Worth it when you're delivering constantly and cargo is heavy (large models, high traffic).
- **Google TPU** = A specialized conveyor belt system. Extremely efficient for a specific type of cargo (matrix operations). Can't do everything a truck can, but for its specialty, nothing is faster or cheaper.
- **AWS Inferentia** = A company-owned delivery drone. Designed by the warehouse (AWS) specifically for their delivery patterns. Cheaper than renting trucks, good performance, but only works within their system.
- **Groq LPU** = A maglev train. Insanely fast (deterministic latency), but requires special track (specific models, specific formats). When it works, nothing matches its speed.
- **Apple Neural Engine** = An e-bike. Perfect for local deliveries (on-device inference). Incredibly efficient for its size, but can't handle heavy cargo (large models).

---

## Deep Dive

### NVIDIA GPUs for Inference

```yaml
NVIDIA_Inference_GPUs:
  h100:
    name: "H100 (Hopper)"
    released: 2023
    memory: "80 GB HBM3"
    memory_bandwidth: "3.35 TB/s"
    fp16_tflops: 989
    fp8_tflops: 1979
    int8_tflops: 1979
    key_feature: "FP8 tensor cores, Transformer Engine"
    use_case: "Large LLM serving, high-throughput inference"
    cost_cloud: "$2-4/hour (spot/on-demand varies by provider)"
    sweet_spot: "Models 7B-70B parameters, high concurrency"
    
  h200:
    name: "H200 (Hopper, more memory)"
    released: 2024
    memory: "141 GB HBM3e"
    memory_bandwidth: "4.8 TB/s (1.4x H100)"
    compute: "Same as H100"
    key_feature: "More memory + bandwidth = more KV cache = more concurrent requests"
    use_case: "Very large models (70B+ at FP8), high concurrency LLM serving"
    advantage: "Serve Llama-70B FP8 on single GPU (70 GB model + 71 GB KV cache)"
    
  b200:
    name: "B200 (Blackwell)"
    released: 2025
    memory: "192 GB HBM3e"
    memory_bandwidth: "8 TB/s"
    fp8_tflops: ~4500
    fp4_tflops: ~9000
    key_feature: "FP4 tensor cores, 2x compute over H100, massive memory"
    use_case: "Largest models (400B+), maximum throughput"
    innovation: "Native FP4 support — enables 4-bit at hardware level"
    
  l4:
    name: "L4 (Ada Lovelace)"
    released: 2023
    memory: "24 GB GDDR6"
    memory_bandwidth: "300 GB/s"
    fp16_tflops: 120
    int8_tops: 242
    tdp: "72W"
    key_feature: "Low power, efficient for smaller models"
    use_case: "Small models (up to 13B quantized), video inference, embedding models"
    cost_cloud: "$0.50-1/hour"
    sweet_spot: "Cost-efficient for 7-8B quantized models, embedding workloads"
    
  a10g:
    name: "A10G"
    memory: "24 GB GDDR6X"
    use_case: "Budget inference for smaller models"
    cost_cloud: "$0.30-0.80/hour"
    sweet_spot: "Traditional ML models, 7B quantized LLMs"
    
  selection_by_model_size:
    up_to_7b_quantized: "L4 (24 GB, ~$0.50/hr) — AWQ INT4 fits easily"
    7b_to_13b: "A10G or L4 — INT4 quantization required"
    13b_to_70b_quantized: "H100 80GB — INT4/FP8 fits most models"
    70b_fp16: "2× H100 or 1× H200 — tensor parallelism"
    70b_quantized: "1× H100 (INT4) or 1× H200 (FP8)"
    400b_plus: "4-8× H100/H200 or 2-4× B200"
```

### Memory Bandwidth vs Compute (Critical for Inference)

```yaml
Memory_vs_Compute:
  key_insight: |
    Inference (especially LLM generation) is often MEMORY BANDWIDTH BOUND, not compute bound.
    During autoregressive generation, each token needs to read ALL model weights from memory
    but only does one matrix-vector multiply per layer. The GPU spends most time waiting for data.
    
  arithmetic_intensity:
    definition: "FLOPS per byte loaded from memory"
    batch_1_generation: "Very low (~1 FLOP/byte) — memory bandwidth bound"
    batch_32_generation: "Medium — starting to utilize compute"
    batch_256_generation: "High — compute bound"
    batch_1_prefill: "High (matrix-matrix multiply) — compute bound"
    
  implications:
    small_batch_inference:
      bottleneck: "Memory bandwidth"
      optimization: "Higher bandwidth memory (HBM3e), model compression (fewer bytes to read)"
      best_hardware: "H200 (4.8 TB/s) > H100 (3.35 TB/s) for single-user latency"
      
    large_batch_inference:
      bottleneck: "Compute (TFLOPS)"
      optimization: "More compute cores, higher clock speed, larger batches"
      best_hardware: "B200 (most TFLOPS) for high-throughput serving"
      
    kv_cache_capacity:
      bottleneck: "Memory size"
      impact: "More memory = more concurrent requests (each request has KV cache)"
      calculation: |
        KV cache per token = 2 × num_layers × hidden_size × 2 bytes (FP16)
        Llama-4-70B: 2 × 80 × 8192 × 2 = 2.6 MB per token
        8K context: 2.6 MB × 8192 = 21 GB per request
        H100 (80 GB) with 35 GB for INT4 model: 45 GB for KV cache ≈ 2 concurrent 8K requests
        H200 (141 GB) with 35 GB for INT4 model: 106 GB for KV cache ≈ 5 concurrent 8K requests
```

### Google TPU (Tensor Processing Unit)

```yaml
Google_TPU:
  what: "Custom ASIC (Application-Specific Integrated Circuit) designed for ML workloads"
  
  tpu_v5e:
    target: "Inference and light training"
    memory: "16 GB HBM2e per chip"
    interconnect: "ICI (Inter-Chip Interconnect) for multi-chip"
    cost: "Significantly cheaper than H100 for inference"
    use_case: "Embedding models, moderate LLM serving"
    
  tpu_v5p:
    target: "Training and large-model inference"
    memory: "95 GB HBM2e per chip"
    pods: "Up to 8,960 chips in a pod"
    use_case: "Large model training and serving"
    
  tpu_v6e_trillium:
    released: 2024
    improvement: "4.7x compute over v5e"
    memory: "32 GB HBM per chip"
    use_case: "Current generation for inference"
    
  advantages:
    - "Lower cost than NVIDIA GPUs for pure inference (at scale)"
    - "Excellent for JAX/TensorFlow workloads"
    - "Pod-scale availability (massive multi-chip deployments)"
    - "GCP integration (Vertex AI endpoints)"
    
  disadvantages:
    - "Only available on GCP (vendor lock-in)"
    - "Limited framework support (JAX best, PyTorch via PyTorch/XLA)"
    - "Less mature ecosystem than CUDA"
    - "vLLM/TGI don't run natively (need JetStream or SAX for LLM serving)"
    
  when_to_use:
    - "Already on GCP"
    - "Large-scale inference (100+ chips)"
    - "Cost-sensitive workloads where vendor lock-in is acceptable"
    - "JAX-based models"
```

### AWS Custom Silicon

```yaml
AWS_Inferentia_Trainium:
  inferentia2:
    what: "AWS custom inference chip (2nd generation)"
    cores: "2 NeuronCores per chip"
    memory: "32 GB HBM per chip"
    use_case: "Inference workloads on AWS"
    instance: "inf2.xlarge (1 chip) to inf2.48xlarge (12 chips)"
    cost: "~40% cheaper than comparable GPU instances for supported models"
    
  trainium2:
    what: "AWS custom training and inference chip (2nd generation)"
    released: 2024
    memory: "96 GB HBM per chip"
    use_case: "Training and large-model inference"
    instance: "trn2.48xlarge (16 chips)"
    
  neuron_sdk:
    what: "AWS SDK for compiling and running models on Inferentia/Trainium"
    frameworks: "PyTorch (via torch_neuronx), TensorFlow, JAX"
    limitation: "Not all operations supported — model must compile successfully"
    llm_serving: "transformers-neuronx library for LLM serving"
    
  advantages:
    - "Significant cost savings (30-50% vs GPU for supported workloads)"
    - "Deep AWS integration (SageMaker, ECS, EKS)"
    - "Good for standard architectures (BERT, GPT, Stable Diffusion)"
    
  disadvantages:
    - "AWS only (maximum vendor lock-in)"
    - "Compilation can fail for non-standard architectures"
    - "Smaller community and fewer tools than CUDA ecosystem"
    - "Lag behind NVIDIA in supporting latest model architectures"
    
  when_to_use:
    - "AWS-committed infrastructure"
    - "Cost-sensitive inference workloads"
    - "Standard model architectures that compile cleanly"
    - "High-volume inference (amortize compilation effort)"
```

### Specialized Inference Accelerators

```yaml
Specialized_Accelerators:
  groq_lpu:
    name: "Groq LPU (Language Processing Unit)"
    architecture: "TSP (Tensor Streaming Processor) — deterministic, no memory hierarchy"
    key_feature: "Deterministic latency — no variability, no memory thrashing"
    performance: "500+ tokens/second for Llama 70B (fastest single-user latency)"
    memory: "SRAM-only (no HBM) — limited by model size"
    limitation: "Model must fit in on-chip SRAM (large models need many chips)"
    access: "Groq Cloud API (not self-hosted yet for most users)"
    use_case: "Ultra-low latency inference where speed matters more than cost"
    
  cerebras_wse:
    name: "Cerebras WSE-3 (Wafer-Scale Engine)"
    what: "Entire silicon wafer as one chip (44x reticle limit)"
    memory: "44 GB on-chip SRAM"
    cores: "900,000+ cores"
    key_feature: "Weight streaming — stream model weights through massive compute array"
    performance: "Extremely high throughput for large models"
    access: "Cerebras Cloud (API access) or on-premises CS-3 system"
    
  apple_neural_engine:
    name: "Apple Neural Engine (ANE)"
    what: "NPU (Neural Processing Unit) in Apple Silicon (M1-M4, A-series)"
    performance: "38 TOPS (M4) — efficient for on-device inference"
    memory: "Unified memory (shared with CPU/GPU) — up to 192 GB on M4 Max"
    frameworks: "CoreML, MLX (Apple's ML framework)"
    use_case: "On-device inference, local LLMs (via MLX/llama.cpp)"
    advantage: "Run 7B models locally with good speed (MLX optimized for unified memory)"
```

### CPU Inference

```yaml
CPU_Inference:
  when_cpu_makes_sense:
    - "Model is small (<1B parameters)"
    - "Latency requirements are moderate (>100ms acceptable)"
    - "Traffic is low/bursty (GPU would be underutilized)"
    - "Budget doesn't justify GPU ($0.01/hour vs $2/hour)"
    - "Embedding models (one forward pass, efficient on CPU)"
    
  optimizations:
    intel_amx:
      what: "Advanced Matrix Extensions (hardware matrix acceleration in Xeon CPUs)"
      speedup: "3-10x vs standard CPU for INT8 inference"
      
    avx512_vnni:
      what: "Vector Neural Network Instructions"
      what_for: "INT8/INT16 dot products in vector registers"
      
    onnx_runtime_cpu:
      what: "Microsoft's inference engine optimized for CPU"
      features: "Graph optimization, quantization, multi-threading"
      
    llama_cpp:
      what: "C++ implementation of LLM inference optimized for CPU"
      features: "GGUF quantization, AVX2/AVX512 optimization, Apple Metal support"
      throughput: "10-30 tokens/sec for 7B Q4 on modern CPU"
      use: "Local development, edge deployment, consumer hardware"
      
  comparison:
    7b_model_int4:
      gpu_h100: "~300 tok/s"
      gpu_l4: "~50 tok/s"
      apple_m4_max: "~30 tok/s (MLX)"
      cpu_epyc_llama_cpp: "~15 tok/s"
      cpu_intel_xeon: "~12 tok/s"
      
    embedding_model_384d:
      gpu_l4: "~5000 embeddings/sec"
      cpu_optimized: "~500-1000 embeddings/sec"
      cost_per_embedding: "CPU often cheaper at low volume"
```

---

## How It Works in Practice

### Hardware Selection Decision Tree

```yaml
Selection_Guide:
  step_1_model_type:
    llm_generation: "Go to step 2a (memory bandwidth matters most)"
    embedding_classification: "Go to step 2b (compute matters most, batching friendly)"
    vision_model: "Go to step 2c (compute heavy, GPU preferred)"
    
  step_2a_llm_serving:
    traffic_high_many_users: "H100 or H200 (PagedAttention + continuous batching)"
    traffic_moderate: "L4 with INT4 quantization (cost-efficient)"
    latency_critical: "Groq LPU (fastest single-user) or H200 (best bandwidth)"
    cost_critical_aws: "Inferentia2 (40% cheaper if model compiles)"
    cost_critical_gcp: "TPU v5e (cheaper than GPU for inference)"
    local_development: "Apple Silicon + MLX or llama.cpp (GGUF)"
    
  step_2b_embedding_classification:
    high_volume: "GPU (L4/A10G) with large batch sizes — maximize throughput"
    low_volume: "CPU (cost-effective, simpler infrastructure)"
    cost_optimized_aws: "Inferentia2 (great for BERT-class models)"
    
  step_2c_vision:
    real_time_video: "GPU required (L4 minimum)"
    batch_processing: "GPU for throughput, CPU possible for low volume"
    edge_mobile: "Apple ANE, Qualcomm Hexagon, dedicated NPU"
```

---

## Interview Tip

> When asked about inference hardware choices: "Hardware selection depends on the inference bottleneck. LLM autoregressive generation is memory-bandwidth bound at small batch sizes — each token requires reading ALL weights but only one matrix-vector multiply per layer. So I prioritize bandwidth: H200 (4.8 TB/s) > H100 (3.35 TB/s) for latency-sensitive single-user serving. At high concurrency (large batches), generation becomes compute-bound and total TFLOPS matter more — B200 (4500 FP8 TFLOPS) dominates. For cost optimization: quantization (INT4/FP8) reduces bytes to read, which directly speeds up bandwidth-bound generation AND reduces GPU memory needed (fewer/smaller GPUs). KV cache capacity limits concurrent requests — H200's 141GB means 5x more concurrent long-context requests vs L4's 24GB. My heuristic: 8B model → L4 ($0.50/hr, INT4), 70B model → H100/H200 (FP8), 400B+ → multi-GPU B200. For embedding/classification (batch-friendly, compute-bound), L4 or even Inferentia2 gives excellent $/throughput."

---

## Common Mistakes

1. **Choosing hardware based on training benchmarks** — Training is compute-bound (large batches, full GPU utilization). Inference (especially LLM generation) is memory-bandwidth-bound at typical batch sizes. A GPU with 2x more TFLOPS but same bandwidth won't give 2x faster single-user generation.

2. **Not accounting for KV cache memory** — Calculating GPU needs based only on model size. A 35GB quantized model on an 80GB H100 seems like it has 45GB free — but at 8K context length with large KV cache, you might only fit 2-3 concurrent requests. Always calculate: model memory + KV cache × concurrent requests.

3. **Using GPU for low-traffic embedding models** — Running a BERT (Bidirectional Encoder Representations from Transformers) embedding model on an H100 that handles 50 requests/minute. The GPU is 95% idle. CPU or Inferentia would serve this at 10-20x lower cost.

4. **Ignoring total cost of ownership** — Comparing only instance $/hour without considering utilization. An H100 at $3/hour with 90% utilization is cheaper per inference than an L4 at $0.50/hour with 10% utilization. Match hardware capacity to actual traffic.

5. **Vendor lock-in without exit strategy** — Building entirely on AWS Inferentia or Google TPU without ONNX export path. If pricing changes or features lag, migration becomes a months-long project. Maintain ONNX or standard model formats as escape hatch.

---

## Key Takeaways

- LLM inference is memory-bandwidth-bound (small batch) — prioritize TB/s over TFLOPS for latency
- At high concurrency (large batch), inference becomes compute-bound — TFLOPS matter
- H100: 80GB HBM3, 3.35 TB/s — workhorse for LLM serving
- H200: 141GB HBM3e, 4.8 TB/s — more concurrent requests, better single-user latency
- B200: 192GB, 8 TB/s, FP4 support — maximum performance (2025+)
- L4: 24GB, 72W — cost-efficient for small quantized models and embeddings
- TPU/Inferentia: 30-50% cost savings but vendor lock-in and limited ecosystem
- Groq LPU: fastest single-user latency (deterministic, SRAM-only architecture)
- CPU: viable for small models, embeddings at low volume, or edge deployment
- KV cache capacity = maximum concurrent requests (calculate before choosing hardware)
- Always quantize before deciding hardware — INT4 model needs 4x less memory and bandwidth
