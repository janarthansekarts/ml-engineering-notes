# Large Model Training

## The Problem / Why This Matters

Training models with tens to hundreds of billions of parameters is one of the most complex engineering challenges in computing. A 70B parameter model in FP16 is 140GB — already larger than any single GPU's memory. Adding optimizer states (Adam: 2× model size in FP32 = 560GB) and activations, the total memory requirement exceeds 1TB. This cannot fit on one GPU, or even one node of 8 GPUs. Training must be distributed across dozens to thousands of GPUs, running for weeks to months, costing millions of dollars. The challenges are stacked: (1) memory — fit the model, optimizer, gradients, and activations across available GPUs, (2) compute efficiency — maximize MFU (Model FLOPs Utilization), avoiding GPU idle time, (3) communication — minimize the overhead of synchronizing across GPUs, (4) fault tolerance — hardware WILL fail during a multi-week run, (5) convergence — ensure the model actually learns correctly at scale (training instabilities are common). In 2026, as organizations train 70B-400B+ models for competitive AI products, understanding large model training — 3D parallelism, ZeRO optimization, communication strategies, and training recipes — separates ML engineers who can handle frontier workloads from those who can't.

---

## The Analogy

Think of training a large model like building a space station:

- **The model** = The space station itself. Too large and complex for any single factory (GPU) to build.
- **3D Parallelism** = The construction strategy. Different modules built at different factories (pipeline parallelism), each module assembled by multiple teams (tensor parallelism), and multiple identical stations built simultaneously to compare approaches (data parallelism).
- **Communication** = The supply chain between factories. If trucks (InfiniBand) are slow, factories sit idle waiting for parts. The supply chain is often the bottleneck, not the factory capacity.
- **Checkpointing** = Photographing the station's construction state daily. If a factory burns down (GPU fails), you can rebuild from the latest photograph instead of starting over.
- **Training stability** = Keeping the station structurally sound as you add modules. One bad weld (exploding gradient) can compromise the entire structure.

---

## Deep Dive

### Memory Requirements for Large Models

```yaml
Memory_Requirements:
  formula: "Total memory = Parameters + Gradients + Optimizer States + Activations"
  
  examples:
    7b_model:
      parameters: "7B × 2 bytes (BF16) = 14 GB"
      gradients: "7B × 2 bytes (BF16) = 14 GB"
      optimizer_adam: "7B × 4 bytes (FP32 momentum) + 7B × 4 bytes (FP32 variance) = 56 GB"
      master_weights: "7B × 4 bytes (FP32 copy) = 28 GB"
      activations: "~20-60 GB (depends on batch size, sequence length)"
      total: "~130-170 GB (without checkpointing)"
      fits_on: "2× H100 80GB (barely), or 1× H100 with ZeRO-3"
      
    70b_model:
      parameters: "70B × 2 bytes = 140 GB"
      gradients: "70B × 2 bytes = 140 GB"
      optimizer_adam: "70B × 8 bytes = 560 GB"
      master_weights: "70B × 4 bytes = 280 GB"
      activations: "~200-600 GB (batch dependent)"
      total: "~1.3-1.7 TB"
      fits_on: "16-32 × H100 80GB (with ZeRO-3 or 3D parallelism)"
      
    405b_model:
      parameters: "405B × 2 bytes = 810 GB"
      gradients: "405B × 2 bytes = 810 GB"
      optimizer_adam: "405B × 8 bytes = 3.2 TB"
      master_weights: "405B × 4 bytes = 1.6 TB"
      activations: "~1-4 TB"
      total: "~7-10 TB"
      fits_on: "128-512 × H100 80GB (requires 3D parallelism)"
      
  memory_reduction_techniques:
    mixed_precision: "BF16 params + FP32 optimizer: reduces param memory by 50%"
    zero_stage_3: "Shard everything: divide total by N GPUs"
    activation_checkpointing: "Reduce activation memory by 5-10x (recompute trade-off)"
    flash_attention: "O(n) attention memory instead of O(n²)"
```

### 3D Parallelism in Practice

```yaml
3D_Parallelism_Implementation:
  configuration_example:
    model: "175B parameters (GPT-3 scale)"
    target_gpus: 512
    
    tensor_parallel: 8
    reason: "Intra-node only (NVLink: 900 GB/s). Split each layer across 8 GPUs in one node"
    
    pipeline_parallel: 8
    reason: "8 pipeline stages across 8 nodes. Uses InfiniBand (400 Gbps)"
    
    data_parallel: 8
    reason: "8 data-parallel replicas for throughput. 512 / (8 × 8) = 8"
    
  gpu_assignment:
    description: "How 512 GPUs are organized"
    node_0_gpus_0_7: "TP group 0, PP stage 0, DP replica 0"
    node_1_gpus_8_15: "TP group 1, PP stage 1, DP replica 0"
    node_7_gpus_56_63: "TP group 7, PP stage 7, DP replica 0"
    node_8_gpus_64_71: "TP group 0, PP stage 0, DP replica 1"
    pattern: "TP within node → PP across nodes → DP for replicas"
    
  communication_analysis:
    tensor_parallel:
      frequency: "Every transformer layer (2 AllReduce per block)"
      volume: "hidden_size × batch × seq_len ≈ 50-100 MB per operation"
      bandwidth_needed: "900 GB/s (NVLink provides this)"
      
    pipeline_parallel:
      frequency: "Once per micro-batch per pipeline stage boundary"
      volume: "batch × seq_len × hidden_size ≈ 100-500 MB per transfer"
      bandwidth_needed: "50 GB/s sufficient (InfiniBand provides 50 GB/s)"
      
    data_parallel:
      frequency: "Once per optimizer step (after all micro-batches)"
      volume: "Full model gradient: 175B × 2 bytes / 8 DP = 44 GB per replica"
      optimization: "AllReduce overlapped with backward computation"

  megatron_deepspeed_config:
    description: "Typical configuration file structure"
    parameters:
      tensor_model_parallel_size: 8
      pipeline_model_parallel_size: 8
      micro_batch_size: 1
      global_batch_size: 2048
      gradient_accumulation_steps: 32  # 2048 / (8 DP × 1 micro × 8 PP_microbatch)
      sequence_length: 4096
      
      optimizer:
        type: "Adam"
        params:
          lr: 0.00015
          betas: [0.9, 0.95]
          eps: 1e-8
          weight_decay: 0.1
          
      lr_schedule:
        type: "cosine"
        warmup_steps: 2000
        min_lr: 1.5e-5
```

### Training Recipes for Large Models

```yaml
Training_Recipes:
  llama_3_405b_recipe:
    source: "Meta AI (2024)"
    model_size: "405B parameters"
    data: "15 trillion tokens"
    gpus: "16,384 H100"
    training_time: "~54 days"
    parallelism: "TP=8, PP=16, DP=128"
    precision: "BF16 training, FP32 optimizer"
    sequence_length: "8192 (extended to 128K in later stages)"
    batch_size: "Started at 4M tokens, ramped to 16M tokens"
    optimizer: "AdamW (β1=0.9, β2=0.95, ε=1e-8, wd=0.1)"
    lr: "Peak 1.5e-4, cosine decay to 1.5e-5"
    warmup: "2000 steps linear warmup"
    gradient_clipping: "1.0"
    
  deepseek_v3_recipe:
    source: "DeepSeek AI (2024-2025)"
    model_size: "671B total (37B active — Mixture of Experts)"
    data: "14.8 trillion tokens"
    efficiency: "Only $5.5M training cost (extremely efficient)"
    innovations:
      - "Multi-head Latent Attention (MLA) — reduces KV cache"
      - "DeepSeekMoE — fine-grained MoE architecture"
      - "FP8 training — 2x throughput on H100 tensor cores"
      - "Auxiliary-loss-free load balancing"
    training_stability: "No loss spikes or rollbacks reported"
    
  common_recipe_elements:
    initialization:
      method: "GPT-2 style (normal init with std = 0.02)"
      residual: "Scale residual layers by 1/√(2N) where N = num layers"
    
    data_mixing:
      strategy: "Multiple data sources with fixed mixing ratios"
      typical: "Web 80%, Code 10%, Books 5%, Scientific 3%, Instruction 2%"
      curriculum: "Increase quality/instruction data toward end of training"
      
    sequence_length_warmup:
      strategy: "Start with shorter sequences, increase during training"
      benefit: "Faster initial training (shorter = cheaper per step)"
      typical: "2048 → 4096 → 8192 during different phases"
      
    batch_size_warmup:
      strategy: "Start with small batch, increase to target"
      benefit: "Better early-training signal (more updates per token)"
      typical: "Start 256K tokens → ramp to 4M tokens over first 10% of training"
```

### Training Stability and Debugging

```yaml
Training_Stability:
  common_issues:
    loss_spikes:
      what: "Sudden jump in loss (can be 10-100x normal)"
      causes:
        - "Bad data batch (corrupted, outlier, extreme values)"
        - "Learning rate too high for current stage"
        - "Gradient explosion (before clipping catches it)"
        - "Hardware error (GPU producing incorrect computation)"
      solutions:
        - "Gradient clipping (max_norm=1.0)"
        - "Skip bad batches (detect and skip if loss > threshold)"
        - "Reduce learning rate"
        - "Check hardware health (ECC errors, temp)"
        
    loss_divergence:
      what: "Loss increases continuously, model not learning"
      causes:
        - "Learning rate too high"
        - "Initialization issue"
        - "Data preprocessing bug"
        - "Precision issue (FP16 overflow)"
      solutions:
        - "Reduce learning rate by 2-10x"
        - "Check initialization (are gradients flowing?)"
        - "Verify data pipeline (log sample batches)"
        - "Switch to BF16 (wider dynamic range)"
        
    slow_convergence:
      what: "Loss decreases but very slowly"
      causes:
        - "Learning rate too low"
        - "Batch size too large (less noise = slower exploration)"
        - "Data quality issues (too much noise/duplication)"
      solutions:
        - "Increase learning rate"
        - "Reduce batch size or add noise"
        - "Improve data quality (filter, deduplicate)"
        
    nan_loss:
      what: "Loss becomes NaN (Not a Number)"
      causes:
        - "FP16 overflow (values > 65504)"
        - "Division by zero (in normalization)"
        - "Log of negative number"
        - "Gradient explosion (before any clipping)"
      solutions:
        - "Use BF16 instead of FP16"
        - "Add epsilon to denominators"
        - "Check for negative values before log operations"
        - "Lower learning rate, stronger gradient clipping"
        
  monitoring_during_training:
    metrics_to_watch:
      - "Training loss (smoothed, per-step)"
      - "Gradient norm (per-step — should be stable)"
      - "Learning rate (verify schedule is correct)"
      - "Throughput (tokens/sec — drops indicate problems)"
      - "GPU memory (approaching limit = future OOM)"
      - "GPU utilization (drops = communication/data bottleneck)"
      
    alerts:
      - "Loss spike > 5x running average"
      - "Gradient norm > 10x running average"
      - "Throughput drop > 20% from baseline"
      - "Any NaN in loss or gradient"
      - "GPU temperature > 85°C"
```

### Communication Optimization

```yaml
Communication_Optimization:
  overlap_strategies:
    computation_communication_overlap:
      what: "Communicate layer N gradients while computing layer N+1"
      how: "DDP backward hooks trigger AllReduce as soon as gradient is ready"
      benefit: "Communication is 'free' — hidden behind computation"
      requirement: "Bucketing gradients into appropriately sized chunks"
      
    prefetching:
      what: "Fetch next layer's parameters while current layer computes (FSDP/ZeRO-3)"
      how: "AllGather for layer N+1 issued before layer N forward completes"
      benefit: "Reduces communication stalls in FSDP"
      
  communication_compression:
    gradient_compression:
      what: "Compress gradients before communication (reduce volume)"
      methods:
        - "FP16/BF16 reduction (half the communication volume)"
        - "Top-K sparsification (send only largest K% of gradients)"
        - "PowerSGD (low-rank approximation of gradient tensor)"
        - "1-bit quantization (extreme compression, some quality loss)"
      typical: "BF16 reduction is standard; aggressive compression for slow networks"
      
    all_to_all_optimization:
      what: "Optimize collective operation patterns"
      techniques:
        - "Ring AllReduce (balanced bandwidth utilization)"
        - "Tree reduction (lower latency for small messages)"
        - "Hierarchical AllReduce (intra-node first, then inter-node)"
      
  network_topology_awareness:
    principle: "Match parallelism strategy to network topology"
    intra_node: "TP (high bandwidth: NVLink 900 GB/s)"
    across_racks: "PP (moderate bandwidth: InfiniBand 400 Gbps)"
    across_clusters: "DP only (lower bandwidth: cross-datacenter links)"
```

---

## How It Works in Practice

### Example: Training a 70B Model

```yaml
Example:
  scenario: "Startup training custom 70B model for domain-specific tasks"
  
  infrastructure:
    gpus: "64 × H100 80GB (8 nodes × 8 GPUs)"
    network: "InfiniBand NDR 400G (fully non-blocking fabric)"
    storage: "WekaFS (200TB, 100 GB/s aggregate throughput)"
    
  parallelism_config:
    tensor_parallel: 8
    reason: "One full node per TP group (NVLink)"
    pipeline_parallel: 4
    reason: "4 pipeline stages across 4 node-pairs"
    data_parallel: 2
    reason: "64 / (8 × 4) = 2 DP replicas"
    
  training_config:
    precision: "BF16 forward/backward, FP32 optimizer"
    sequence_length: 4096
    global_batch_size: 1024
    micro_batch_size: 1
    gradient_accumulation: "1024 / (2 DP × 1 micro × 8 PP_microbatches) = 64"
    optimizer: "AdamW (lr=3e-4, β1=0.9, β2=0.95, wd=0.1)"
    schedule: "Linear warmup 2000 steps → cosine decay"
    gradient_clipping: 1.0
    activation_checkpointing: "Every transformer layer"
    flash_attention: true
    
  training_run:
    total_tokens: "2 trillion"
    tokens_per_step: "1024 batch × 4096 seq = 4.2M tokens/step"
    total_steps: "~476,000 steps"
    throughput: "~180K tokens/sec (target MFU: 45%)"
    wall_time: "~130 days (continuous training)"
    cost: "$2.5M in GPU time (at $2.5/GPU-hour)"
    
  fault_tolerance:
    checkpointing: "Every 1000 steps (async, to WekaFS)"
    checkpoint_size: "~1.2 TB per checkpoint"
    expected_failures: "2-5 GPU failures over 130 days"
    recovery_time: "~15 minutes (load checkpoint + resume)"
    
  monitoring:
    dashboards: "W&B (real-time loss, throughput, GPU stats)"
    alerts: "Loss spike > 5x, throughput drop > 20%, any NaN"
    health_checks: "Hourly NCCL communication test, GPU ECC check"
```

---

## Interview Tip

> When asked about large model training: "For training models like 70B+ parameters, I use 3D parallelism: Tensor Parallelism within a node (8 GPUs connected by NVLink at 900 GB/s), Pipeline Parallelism across nodes (InfiniBand provides adequate bandwidth for once-per-stage communication), and Data Parallelism for throughput scaling. Memory is managed through ZeRO-3 (shard optimizer states across DP group), activation checkpointing (recompute instead of store), and BF16 precision. Key metrics I monitor: MFU (Model FLOPs Utilization — target >40%), throughput (tokens/sec), gradient norm (stability), and loss curve smoothness. For fault tolerance at this scale, I implement async checkpointing every 30-60 minutes, elastic training for node failure recovery, and hardware health monitoring (GPU ECC errors, temperature, communication latency). Training stability techniques: gradient clipping, learning rate warmup, batch size warmup, and data quality filtering. The most common failure mode is loss spikes from bad data batches — detected by monitoring gradient norm and handled by skipping or reducing learning rate."

---

## Common Mistakes

1. **Wrong parallelism assignment** — Putting tensor parallelism across nodes (Ethernet/InfiniBand) instead of within a node (NVLink). TP communicates every layer and needs >600 GB/s — InfiniBand at 50 GB/s makes TP 12x slower than with NVLink. Always: TP intra-node, PP inter-node.

2. **Not monitoring MFU** — Running a 64-GPU training job without tracking actual hardware utilization. Many setups achieve only 20-30% MFU due to communication overhead or pipeline bubbles — meaning 70-80% of expensive GPU time is wasted. Target >40% MFU.

3. **Checkpoint too infrequently** — Checkpointing every 24 hours on a job that can fail at any time. A failure 23 hours after the last checkpoint loses 23 hours × 64 GPUs × $2.5/hr = $3,680 of compute. Checkpoint every 30-60 minutes (async to avoid stalling training).

4. **Ignoring training instability at scale** — Assuming hyperparameters that work for 7B will work for 70B. Larger models are less stable — they need lower learning rates, more warmup, stricter gradient clipping, and careful initialization scaling. Test at smaller scale first, then carefully scale up.

5. **Not warming up batch size** — Starting with the full target batch size (e.g., 4M tokens). Large batch sizes in early training provide less gradient noise, which can hurt exploration and lead to sharp minima. Warm up from small batch (256K) to target over first 5-10% of training.

---

## Key Takeaways

- Large model memory: params + gradients + optimizer + activations — 70B model needs ~1.5TB total
- 3D Parallelism: TP (intra-node, NVLink) + PP (inter-node, InfiniBand) + DP (scale, AllReduce)
- ZeRO-3: shard everything across DP group — each GPU stores only 1/N of total state
- MFU (Model FLOPs Utilization): target >40% — below this means significant waste
- Training recipes: warmup LR + batch size, cosine decay, gradient clipping = 1.0, AdamW
- Fault tolerance: async checkpoint every 30-60 min — hardware WILL fail during multi-week runs
- Loss spikes: monitor gradient norm, skip bad batches, adjust LR if persistent
- Communication optimization: overlap compute+comm, hierarchical AllReduce, BF16 reduction
- Sequence length warmup: start short (2048), increase later — cheaper tokens early in training
- Scale testing: always validate training recipe at 1/10th scale before committing full cluster
