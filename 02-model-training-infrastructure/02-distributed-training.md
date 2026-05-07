# Distributed Training

## The Problem / Why This Matters

Modern ML (Machine Learning) models have grown beyond what any single GPU can handle. GPT-4 class models have over a trillion parameters, requiring hundreds of GPUs training simultaneously for weeks. Even "smaller" models like Llama-4-70B need distributed training — a single H100 with 80GB memory cannot hold the model (140GB in FP16) plus optimizer states (560GB for Adam in FP32) plus activations. Distributed training splits the workload across multiple GPUs (often hundreds or thousands) connected via high-speed networking. But distributed training isn't just "use more GPUs" — it introduces communication overhead, synchronization challenges, fault tolerance requirements, and fundamentally different parallelism strategies. ML engineers must understand WHEN to use distributed training (single GPU insufficient), WHICH parallelism strategy fits their model and hardware (data, tensor, pipeline, or hybrid), and HOW to configure it efficiently (minimizing communication overhead while maximizing GPU utilization). In 2026, even fine-tuning a 70B model requires multi-GPU setups, making distributed training knowledge essential for all ML engineers.

---

## The Analogy

Think of distributed training like building a house with multiple construction crews:

- **Data Parallelism** = Each crew builds a complete miniature model of the house (full model replica), but each uses different materials (different data batch). They periodically meet to share what they learned and agree on the best design (gradient synchronization).
- **Model Parallelism (Tensor)** = Each crew specializes in one component: one does electrical, one does plumbing, one does framing. They must work in sequence because plumbing depends on framing being done. The house is built one layer at a time, passed between crews.
- **Pipeline Parallelism** = The house is split into floors. Crew 1 builds Floor 1, then passes it up. While Crew 2 works on Floor 2, Crew 1 starts the next house's Floor 1. Assembly line for houses.
- **3D Parallelism** = All three simultaneously — the massive factory that builds skyscrapers. Each floor (pipeline) has specialized crews (tensor) and multiple teams per specialty (data).

---

## Deep Dive

### Data Parallelism

```yaml
Data_Parallelism:
  concept:
    what: "Replicate the entire model on every GPU, split DATA across GPUs"
    how:
      - "Each GPU gets a copy of the full model"
      - "Each GPU processes a different mini-batch of data"
      - "After forward + backward pass, GPUs synchronize gradients"
      - "All GPUs apply the same gradient update (models stay in sync)"
      
  gradient_synchronization:
    allreduce:
      what: "Average gradients across all GPUs"
      how: "Ring AllReduce — each GPU sends/receives to neighbors in a ring"
      communication: "2 × (n-1)/n × model_size per step (n = num GPUs)"
      example: "8 GPUs, 2GB model → ~3.5GB communication per step"
      
    implementations:
      pytorch_ddp:
        name: "DistributedDataParallel (DDP)"
        how: "Overlap gradient computation with communication (backward hook)"
        advantage: "Near-linear scaling for communication-light models"
        limitation: "Each GPU needs full model + optimizer in memory"
        
      pytorch_fsdp:
        name: "Fully Sharded Data Parallel (FSDP)"
        how: "Shard model parameters, gradients, optimizer states across GPUs"
        advantage: "Memory of single GPU: model_size/N (N = num GPUs)"
        trade_off: "More communication (gather params before forward/backward)"
        
  scaling_efficiency:
    ideal: "N GPUs → N× throughput"
    reality: "Sub-linear due to communication overhead"
    typical: "8 GPUs → 7.5× throughput (94% efficiency) on fast interconnect"
    factors:
      - "Interconnect bandwidth (NVLink > PCIe > Ethernet)"
      - "Model size (larger models → better compute-to-communication ratio)"
      - "Batch size (larger batches → less frequent sync relative to compute)"
      
  when_to_use:
    - "Model fits in single GPU memory (most common scenario)"
    - "Want to increase effective batch size / training throughput"
    - "Up to 8-64 GPUs (beyond this, communication overhead grows)"
```

### Model Parallelism (Tensor Parallelism)

```yaml
Tensor_Parallelism:
  concept:
    what: "Split individual layers (tensors) across multiple GPUs"
    why: "When a single layer is too large for one GPU's memory"
    how: "Split weight matrices column-wise or row-wise across GPUs"
    
  example_linear_layer:
    single_gpu:
      operation: "Y = X × W where W is [4096, 4096]"
      memory: "W takes 32MB in FP16"
    tensor_parallel_2_gpus:
      gpu_0: "Y_0 = X × W[:, :2048]  (first half of columns)"
      gpu_1: "Y_1 = X × W[:, 2048:]  (second half of columns)"
      combine: "Y = concat(Y_0, Y_1)  (AllGather across GPUs)"
      memory_per_gpu: "16MB (half the weight matrix)"
      
  attention_parallelism:
    description: "Split attention heads across GPUs (natural split point)"
    example: "32 attention heads on 4 GPUs → 8 heads per GPU"
    communication: "AllReduce after attention output projection"
    
  mlp_parallelism:
    description: "Split feed-forward layers column-wise (first linear) and row-wise (second linear)"
    communication: "One AllReduce per transformer block (after second linear)"
    
  frameworks:
    megatron_lm:
      what: "NVIDIA's framework for tensor + pipeline parallelism"
      use: "Training models with 10B-1T parameters"
      features: "Optimized AllReduce, sequence parallelism, activation checkpointing"
    deepspeed:
      what: "Microsoft's distributed training library"
      tensor_parallel: "Via integration with Megatron-LM"
      
  characteristics:
    communication_frequency: "Every layer (very frequent)"
    communication_volume: "Proportional to hidden_size × batch_size × sequence_length"
    requirement: "HIGH bandwidth interconnect (NVLink mandatory, Ethernet too slow)"
    typical_scale: "2-8 GPUs within a single node (intra-node only)"
    
  when_to_use:
    - "Individual layers too large for one GPU"
    - "Training models >10B parameters"
    - "GPUs connected by NVLink (>600 GB/s)"
    - "Always combined with data parallelism for scale"
```

### Pipeline Parallelism

```yaml
Pipeline_Parallelism:
  concept:
    what: "Split model into sequential stages (groups of layers), place each stage on different GPU"
    how: "Input flows through GPU 0 (layers 0-10) → GPU 1 (layers 11-20) → GPU 2 (layers 21-30) → ..."
    problem: "Naive approach: only one GPU active at a time (bubble problem)"
    solution: "Micro-batching — split batch into smaller pieces, pipeline them through stages"
    
  the_bubble_problem:
    naive_pipeline:
      description: "GPU 1 waits for GPU 0 to finish forward pass, GPU 0 waits for GPU 1 during backward"
      utilization: "1/N (N = num stages) — terrible with many stages"
      
    micro_batching_solution:
      description: "Split batch into M micro-batches, pipeline them through stages"
      utilization: "Approaches 1 as M increases (more micro-batches = smaller bubble)"
      bubble_fraction: "(num_stages - 1) / (num_stages - 1 + num_microbatches)"
      example: "4 stages, 16 micro-batches → bubble = 3/19 ≈ 16% idle time"
      
  schedules:
    gpipe:
      description: "All micro-batches forward, then all backward"
      bubble: "Large (all forward before any backward)"
      memory: "High (must store activations for all micro-batches)"
      
    pipedream_1f1b:
      description: "Alternate forward and backward passes (1 forward, 1 backward)"
      bubble: "Smaller than GPipe"
      memory: "Lower (limited number of in-flight micro-batches)"
      
    interleaved_1f1b:
      description: "Each GPU handles multiple non-contiguous stages"
      bubble: "Even smaller (more granular scheduling)"
      trade_off: "More communication (data moves between GPUs more often)"
      
    zero_bubble:
      description: "Advanced schedule that eliminates pipeline bubble almost entirely"
      how: "Reorder forward/backward passes to fill gaps"
      status: "Research → production adoption (2025-2026)"
      
  characteristics:
    communication_frequency: "Once per stage boundary per micro-batch"
    communication_volume: "Activation size (batch × sequence × hidden) — moderate"
    requirement: "Moderate bandwidth (can work over Ethernet, NVLink preferred)"
    typical_scale: "4-16 stages (across nodes acceptable)"
    
  when_to_use:
    - "Model too large for one GPU but tensor parallelism insufficient"
    - "Inter-node training (Ethernet between nodes, NVLink within)"
    - "Combined with tensor parallelism and data parallelism (3D parallelism)"
```

### 3D Parallelism (Hybrid)

```yaml
3D_Parallelism:
  concept:
    what: "Combine all three parallelism strategies simultaneously"
    why: "Largest models (100B-1T+ parameters) require all techniques"
    how: "Tensor parallel within node, pipeline parallel across nodes, data parallel for throughput"
    
  typical_configuration:
    example_model: "175B parameter model"
    total_gpus: "512 H100 GPUs (64 nodes × 8 GPUs)"
    
    tensor_parallel: 8
    meaning: "8 GPUs within each node split each layer (uses NVLink)"
    
    pipeline_parallel: 8
    meaning: "8 pipeline stages across 8 groups of nodes"
    
    data_parallel: 8
    meaning: "8 replicas of the full pipeline (each processes different data)"
    
    verification: "TP(8) × PP(8) × DP(8) = 512 GPUs ✓"
    
  real_world_examples:
    gpt_3_175b:
      total_gpus: "1024 A100 GPUs"
      configuration: "TP=8, PP=16, DP=8"
      training_time: "~34 days"
      
    llama_3_405b:
      total_gpus: "16,384 H100 GPUs"
      configuration: "TP=8, PP=16, DP=128"
      training_time: "~54 days"
      
  key_principle:
    rule: "Use TP for intra-node (high bandwidth), PP for inter-node (moderate bandwidth), DP for scale"
    reason: "TP requires very high bandwidth (every layer), PP requires moderate bandwidth (once per stage)"
    
  frameworks:
    megatron_deepspeed: "NVIDIA + Microsoft collaboration (most common for frontier models)"
    fairscale: "Meta's distributed training library"
    colossal_ai: "Open-source framework for large model training with automated parallelism"
    maxtext: "Google's framework for TPU-based training"
```

### ZeRO Optimization

```yaml
ZeRO_Optimization:
  full_name: "Zero Redundancy Optimizer"
  creator: "Microsoft (DeepSpeed library)"
  
  problem_solved:
    description: "In standard data parallelism, EVERY GPU stores full copy of: model params + gradients + optimizer states"
    waste: "8 GPUs → 8 copies of everything (7 copies are redundant)"
    example: "10B model with Adam: 120GB per GPU × 8 GPUs = 960GB total (but only 120GB unique)"
    
  zero_stages:
    stage_1:
      name: "Optimizer State Partitioning"
      what: "Partition optimizer states across GPUs (each GPU stores 1/N of states)"
      memory_reduction: "4x for Adam (optimizer states are 2/3 of memory in mixed precision)"
      communication: "Same as DDP (AllReduce gradients)"
      
    stage_2:
      name: "Gradient Partitioning"
      what: "Partition gradients across GPUs (each GPU stores 1/N of gradients)"
      memory_reduction: "8x"
      communication: "Reduce-Scatter (instead of AllReduce) — slightly more efficient"
      
    stage_3:
      name: "Parameter Partitioning"
      what: "Partition model parameters across GPUs (each GPU stores 1/N of params)"
      memory_reduction: "N× (linear with number of GPUs)"
      communication: "AllGather parameters before forward/backward (more communication)"
      trade_off: "Significant communication overhead — mitigated by prefetching"
      
  memory_comparison:
    model: "10B params, 8 GPUs, Adam, mixed precision"
    
    standard_ddp:
      per_gpu: "Params 20GB + Grads 20GB + Optimizer 80GB = 120GB"
      total: "960GB across 8 GPUs"
      
    zero_stage_1:
      per_gpu: "Params 20GB + Grads 20GB + Optimizer 10GB = 50GB"
      total: "400GB"
      
    zero_stage_2:
      per_gpu: "Params 20GB + Grads 2.5GB + Optimizer 10GB = 32.5GB"
      total: "260GB"
      
    zero_stage_3:
      per_gpu: "Params 2.5GB + Grads 2.5GB + Optimizer 10GB = 15GB"
      total: "120GB (no redundancy)"
      
  zero_infinity:
    what: "Offload to CPU memory and NVMe SSD (when GPU memory isn't enough)"
    enables: "Training trillion-parameter models on limited GPU clusters"
    trade_off: "Slower (CPU/NVMe much slower than GPU memory)"
    use_case: "Research with limited GPU budget, very large models"
```

---

## How It Works in Practice

### Choosing a Parallelism Strategy

```yaml
Decision_Framework:
  model_fits_in_one_gpu:
    condition: "Model + optimizer + activations < GPU memory"
    strategy: "Data Parallelism (DDP)"
    implementation: "PyTorch DDP (simplest, most efficient)"
    example: "BERT (0.3B params) on A100 80GB — plenty of room"
    
  model_fits_with_optimization:
    condition: "Model fits with mixed precision + gradient accumulation"
    strategy: "Data Parallelism + memory optimizations"
    implementation: "PyTorch DDP + AMP + gradient checkpointing"
    example: "7B model with LoRA on A100 80GB"
    
  model_needs_multi_gpu_memory:
    condition: "Model parameters + optimizer > single GPU memory"
    strategy: "FSDP (ZeRO Stage 3) or DeepSpeed ZeRO"
    implementation: "PyTorch FSDP or DeepSpeed ZeRO-3"
    example: "70B model fine-tuning on 8x A100 80GB"
    
  model_needs_tensor_parallel:
    condition: "Very large model, single layers are huge, intra-node only"
    strategy: "Tensor Parallel + Data Parallel"
    implementation: "Megatron-LM or DeepSpeed with tensor parallel"
    example: "70B full pre-training on 64 H100s"
    
  frontier_model:
    condition: ">100B parameters, thousands of GPUs"
    strategy: "3D Parallelism (TP + PP + DP)"
    implementation: "Megatron-DeepSpeed"
    example: "Training 175B+ model on 512+ H100 GPUs"
    
  practical_rule_of_thumb:
    most_common: "FSDP handles 80% of distributed training needs in 2026"
    reason: "Simpler than Megatron, handles models up to ~100B on reasonable GPU counts"
    when_megatron: "Only for frontier models (>100B) or when maximum efficiency is critical"
```

---

## Interview Tip

> When asked about distributed training: "I think about distributed training in terms of what resource is limited. (1) If I need more throughput but the model fits on one GPU → Data Parallelism (DDP), replicate model, split data, sync gradients. (2) If the model doesn't fit on one GPU → either FSDP/ZeRO-3 (shard everything across GPUs, simpler to set up) or Tensor Parallelism (split individual layers, requires NVLink). (3) For very large models spanning many nodes → Pipeline Parallelism (split by layer groups) combined with TP within nodes and DP for throughput — that's 3D parallelism. Key insight: TP requires high bandwidth (every layer communicates) so it's intra-node only (NVLink). PP has moderate communication (once per stage) so it can span nodes (Ethernet/InfiniBand). DP communication is per-step (AllReduce) and scales with model size. ZeRO eliminates the memory redundancy of DDP — Stage 3 gives N× memory reduction with N GPUs at the cost of more communication."

---

## Common Mistakes

1. **Using tensor parallelism across nodes** — TP requires extremely high bandwidth (communication every layer). Over Ethernet (even 100Gbps), TP is catastrophically slow. TP must be within a node (NVLink: 900 GB/s). Use pipeline parallelism for cross-node splitting.

2. **Not matching parallelism to hardware topology** — A common mistake is treating all GPUs as equal. GPUs within a node communicate via NVLink (fast), nodes communicate via InfiniBand or Ethernet (slow). Strategy must match topology: TP intra-node, PP/DP inter-node.

3. **Using DeepSpeed ZeRO-3 when DDP suffices** — ZeRO-3 adds communication overhead. If the model fits on one GPU with mixed precision + gradient checkpointing, simple DDP is faster. Only use ZeRO when you actually need the memory savings.

4. **Forgetting the pipeline bubble** — Pipeline parallelism seems efficient on paper but the bubble (idle time) can waste 15-30% of compute with few micro-batches. Always configure enough micro-batches (4-8× pipeline stages) to minimize the bubble fraction.

5. **Not tuning communication overlap** — Modern frameworks overlap computation with communication (computing layer N while communicating layer N-1). This requires correct configuration — wrong bucket sizes in DDP or wrong prefetch settings in FSDP destroy the overlap and tank performance.

---

## Key Takeaways

- Data Parallelism: replicate model, split data, sync gradients — simplest, works when model fits on one GPU
- Tensor Parallelism: split individual layers across GPUs — requires NVLink, intra-node only
- Pipeline Parallelism: split model into sequential stages — moderate bandwidth requirement, cross-node OK
- 3D Parallelism: TP + PP + DP combined — for frontier models (100B+ parameters)
- ZeRO: eliminates memory redundancy in data parallelism (Stage 1-3: optimizer → gradients → parameters)
- FSDP: PyTorch's native ZeRO Stage 3 — handles most distributed training needs in 2026
- Scaling efficiency: ~94% for 8 GPUs with fast interconnect, degrades with more GPUs and slower networks
- Hardware topology matters: NVLink within node (900 GB/s), InfiniBand across nodes (400 Gbps)
- Communication-computation overlap: critical for efficiency (bucket sizes, prefetching, scheduling)
- Practical rule: start with DDP → FSDP → Megatron as model size increases
