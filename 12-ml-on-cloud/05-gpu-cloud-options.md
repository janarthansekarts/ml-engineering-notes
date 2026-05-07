# GPU Cloud Options

## The Problem / Why This Matters

GPU compute is the most expensive and constrained resource in ML engineering. In 2026, the GPU landscape is more complex than ever: NVIDIA dominates with A100, H100, H200, and B200 GPUs, while cloud providers offer varying availability, pricing models, and networking options. Choosing the right GPU for your workload — and the right purchasing model (on-demand, reserved, spot/preemptible) — can mean the difference between a $10K and $100K training bill for the same model. Beyond just picking a GPU, you need to understand: memory bandwidth vs. compute (some workloads are memory-bound), interconnect bandwidth (multi-GPU training needs fast communication), availability zones (H100s are scarce), and alternative providers (CoreWeave, Lambda Labs, Together AI) that often offer better GPU availability and pricing than hyperscalers. This file covers: GPU hardware comparison (specs that actually matter for ML), cloud provider pricing and availability, purchasing strategies (spot vs. reserved vs. on-demand), and how to match GPU to workload.

---

## The Analogy

Think of GPU selection like choosing a vehicle for a delivery business:

- **A10G** = A reliable van. Handles most deliveries (inference, small training), affordable, widely available. Not fast for cross-country trips (large model training).
- **A100** = A cargo truck. Serious capacity (80GB HBM2e), handles heavy loads (large model training), good highway speed (900 GB/s memory bandwidth). The workhorse of 2023-2024.
- **H100** = A semi-truck with a turbocharger. Massive capacity (80GB HBM3), much faster (3.35 TB/s bandwidth), specialized engine for transformer workloads (Transformer Engine with FP8). The 2024-2025 standard.
- **H200** = The H100 with a bigger fuel tank. Same engine but 141GB HBM3e memory — fits larger models without splitting across GPUs.
- **B200** = Next-generation freight train. 192GB HBM3e, 8 TB/s bandwidth, 2× compute of H100. The 2025-2026 frontier.
- **Spot instances** = Renting vehicles only when available. 60-90% cheaper, but the rental company can recall them with 2-minute notice. Great if you can checkpoint your delivery progress.

---

## Deep Dive

### GPU Hardware Comparison

```yaml
GPU_Comparison_2026:
  nvidia_datacenter_gpus:
    A10G:
      generation: "Ampere (2021)"
      memory: "24 GB GDDR6X"
      memory_bandwidth: "600 GB/s"
      fp16_tflops: 125
      fp8_tflops: "N/A"
      tdp: "150W"
      interconnect: "PCIe Gen4 (no NVLink)"
      best_for: "Inference, fine-tuning small models, development"
      cloud_availability: "Widely available (all clouds)"
      typical_cost: "$0.75-1.50/hr (on-demand)"
      
    L4:
      generation: "Ada Lovelace (2023)"
      memory: "24 GB GDDR6"
      memory_bandwidth: "300 GB/s"
      fp16_tflops: 121
      fp8_tflops: 242
      int8_tops: 242
      tdp: "72W"
      interconnect: "PCIe Gen4"
      best_for: "Inference (excellent perf/watt), video processing"
      cloud_availability: "GCP (primary), AWS, Azure"
      typical_cost: "$0.50-1.00/hr"
      note: "Best inference GPU for cost-sensitive workloads"
      
    A100:
      generation: "Ampere (2020)"
      variants:
        40gb:
          memory: "40 GB HBM2e"
          memory_bandwidth: "1,555 GB/s"
        80gb:
          memory: "80 GB HBM2e"
          memory_bandwidth: "2,039 GB/s"
      fp16_tflops: 312
      tf32_tflops: 156
      fp8_tflops: "N/A (no FP8 support)"
      tdp: "400W"
      interconnect: "NVLink (600 GB/s), NVSwitch"
      best_for: "Training medium models (7B), general compute"
      cloud_availability: "Widely available, declining as H100 replaces"
      typical_cost: "$2.50-4.00/hr (80GB on-demand)"
      
    H100:
      generation: "Hopper (2023)"
      memory: "80 GB HBM3"
      memory_bandwidth: "3,350 GB/s"
      fp16_tflops: 990
      fp8_tflops: 1979
      tf32_tflops: 495
      tdp: "700W"
      interconnect: "NVLink (900 GB/s), NVSwitch"
      transformer_engine: "FP8 acceleration for attention and MLP layers"
      best_for: "Training large models (7B-70B), high-throughput inference"
      cloud_availability: "AWS (p5), GCP (a3-highgpu), Azure (ND H100)"
      typical_cost: "$4.00-8.00/hr (on-demand), $2.50-4.00/hr (spot)"
      note: "Dominant training GPU in 2025-2026"
      
    H200:
      generation: "Hopper refresh (2024)"
      memory: "141 GB HBM3e"
      memory_bandwidth: "4,800 GB/s"
      fp16_tflops: 990  # Same compute as H100
      fp8_tflops: 1979
      tdp: "700W"
      interconnect: "NVLink (900 GB/s)"
      best_for: "LLM inference (larger KV cache), training without model sharding"
      cloud_availability: "Limited (AWS p5en, GCP a3-megagpu)"
      typical_cost: "$6.00-10.00/hr (on-demand)"
      note: "Same compute as H100, 76% more memory, 43% more bandwidth"
      
    B200:
      generation: "Blackwell (2025)"
      memory: "192 GB HBM3e"
      memory_bandwidth: "8,000 GB/s"
      fp16_tflops: ~2250
      fp8_tflops: ~4500
      fp4_tflops: ~9000
      tdp: "1000W"
      interconnect: "NVLink 5th gen (1.8 TB/s)"
      transformer_engine: "FP4 acceleration (new in Blackwell)"
      best_for: "Frontier model training, high-throughput LLM inference"
      cloud_availability: "Rolling out 2025-2026 (very limited)"
      typical_cost: "$10.00-15.00/hr (estimated on-demand)"
      note: "2× H100 compute, 2.4× memory, 2.4× bandwidth"
      
  key_specs_for_ml:
    memory_capacity:
      why: "Determines max model size that fits on one GPU"
      rule_of_thumb: "Model parameters × 2 bytes (FP16) + optimizer states + activations"
      examples:
        - "7B model: ~14GB (FP16) — fits on A100 40GB with room for batch"
        - "13B model: ~26GB (FP16) — needs A100 80GB or H100"
        - "70B model: ~140GB (FP16) — needs H200 or multi-GPU sharding"
        
    memory_bandwidth:
      why: "Determines inference speed (LLM decoding is memory-bound)"
      rule_of_thumb: "Tokens/sec ≈ bandwidth / (model_size_bytes × 2)"
      impact: "H100 (3.35 TB/s) generates tokens 64% faster than A100 (2.04 TB/s)"
      
    compute_tflops:
      why: "Determines training speed (forward + backward pass is compute-bound)"
      impact: "H100 FP8 (1979 TFLOPS) is 3× faster than A100 FP16 (312 TFLOPS)"
      
    interconnect:
      why: "Determines multi-GPU scaling efficiency"
      rule_of_thumb: "If interconnect < 10× per-GPU bandwidth, multi-GPU is bottlenecked"
      nvlink_vs_pcie: "NVLink (900 GB/s) vs PCIe (64 GB/s) — 14× difference"
```

### Cloud Provider Comparison

```yaml
Cloud_GPU_Pricing_2026:
  aws:
    instances:
      p5_48xlarge:
        gpus: "8× H100 80GB"
        vcpu: 192
        memory: "2048 GB"
        networking: "3200 Gbps EFA"
        on_demand: "$98.32/hr"
        spot: "$35-65/hr (variable)"
        reserved_1yr: "$68/hr (~31% savings)"
        
      p5en_48xlarge:
        gpus: "8× H200 141GB"
        vcpu: 192
        memory: "2048 GB"
        networking: "3200 Gbps EFA"
        on_demand: "$120-140/hr (estimated)"
        availability: "Limited regions"
        
      p4d_24xlarge:
        gpus: "8× A100 40GB"
        vcpu: 96
        memory: "1152 GB"
        networking: "400 Gbps EFA"
        on_demand: "$32.77/hr"
        spot: "$12-20/hr"
        
      g5_xlarge:
        gpus: "1× A10G 24GB"
        vcpu: 4
        memory: "16 GB"
        on_demand: "$1.006/hr"
        spot: "$0.35-0.60/hr"
        best_for: "Development, inference, small training"
        
    networking:
      efa: "Elastic Fabric Adapter — low-latency RDMA"
      bandwidth: "Up to 3200 Gbps (p5)"
      placement_groups: "Cluster placement for minimal latency"
      
  gcp:
    instances:
      a3_highgpu_8g:
        gpus: "8× H100 80GB"
        vcpu: 208
        memory: "1872 GB"
        networking: "3200 Gbps GPUDirect-TCPX"
        on_demand: "$98.28/hr"
        spot: "$29.48/hr (70% savings)"
        
      a3_megagpu_8g:
        gpus: "8× H200 141GB"
        on_demand: "$120-135/hr"
        availability: "Select regions, capacity-limited"
        
      a2_ultragpu_8g:
        gpus: "8× A100 80GB"
        on_demand: "$40.22/hr"
        spot: "$12.07/hr (70% savings)"
        
      g2_standard_4:
        gpus: "1× L4 24GB"
        vcpu: 4
        memory: "16 GB"
        on_demand: "$0.84/hr"
        best_for: "Inference (excellent perf/watt)"
        
    tpus:
      v5e: "$1.20/chip/hr (training optimized)"
      v5p: "$4.20/chip/hr (highest performance)"
      advantage: "40-60% cheaper than H100 for transformer workloads"
      
  azure:
    instances:
      nd_h100_v5:
        gpus: "8× H100 80GB"
        vcpu: 96
        memory: "1900 GB"
        networking: "3200 Gbps InfiniBand"
        on_demand: "$98.56/hr"
        spot: "$29.57/hr (70% savings)"
        reserved_1yr: "$65/hr (34% savings)"
        
      nc_a100_v4:
        gpus: "1-4× A100 80GB"
        on_demand: "$3.67/hr (1× A100)"
        spot: "$1.10/hr (70% savings)"
        
  alternative_providers:
    coreweave:
      advantage: "GPU-native cloud — better availability, lower prices"
      h100_8gpu: "$85-95/hr (on-demand)"
      h100_spot: "$25-35/hr"
      differentiator: "InfiniBand clusters for large training, faster provisioning"
      
    lambda_labs:
      h100_8gpu: "$27.60/hr (on-demand, when available)"
      advantage: "Very competitive pricing, simple interface"
      limitation: "Limited availability, no enterprise features"
      
    together_ai:
      model: "Serverless and dedicated GPU hosting"
      advantage: "Optimized for LLM inference (vLLM-based)"
      pricing: "Per-token pricing for inference, hourly for dedicated"
      
    runpod:
      h100_single: "$3.89/hr (community cloud)"
      advantage: "Cheapest spot GPU access, good for experimentation"
      limitation: "Less enterprise security, variable availability"
```

### Matching GPU to Workload

```python
# GPU selection decision framework

"""
Framework for selecting the right GPU for different ML workloads.
Consider: memory needs, compute needs, budget, and availability.
"""

gpu_selection_framework = {
    "llm_inference": {
        "small_models": {  # 7B parameters
            "memory_needed": "14 GB (FP16) or 7 GB (INT8)",
            "bottleneck": "Memory bandwidth (decoding is memory-bound)",
            "recommended": ["L4 (INT8 quantized)", "A10G (FP16)", "T4 (INT8, budget)"],
            "cost_optimal": "L4 at $0.84/hr — best inference perf/$ for quantized models",
        },
        "medium_models": {  # 13B-30B parameters
            "memory_needed": "26-60 GB (FP16) or 13-30 GB (INT8)",
            "bottleneck": "Memory bandwidth",
            "recommended": ["A100 80GB (FP16)", "H100 (FP8, fastest)", "2× L4 (tensor parallel)"],
            "cost_optimal": "A100 80GB spot ($12-15/hr for high throughput)",
        },
        "large_models": {  # 70B+ parameters
            "memory_needed": "140+ GB (FP16) or 35-70 GB (INT4)",
            "bottleneck": "Memory capacity + bandwidth",
            "recommended": ["H200 (141GB, fits 70B FP16)", "2× H100 (tensor parallel)", "4× A100 80GB"],
            "cost_optimal": "H200 single GPU (no TP overhead) or 2× H100 spot",
        },
    },
    
    "model_training": {
        "small_training": {  # < 1B parameters, fine-tuning
            "memory_needed": "16-40 GB (model + optimizer + gradients)",
            "bottleneck": "Compute (forward/backward pass)",
            "recommended": ["A10G (LoRA fine-tuning)", "A100 40GB (full fine-tuning)"],
            "cost_optimal": "A10G spot ($0.35-0.60/hr) for LoRA, A100 spot for full",
        },
        "medium_training": {  # 7B-13B parameters
            "memory_needed": "40-160 GB (with optimizer states)",
            "bottleneck": "Compute + memory",
            "recommended": ["H100 (single, with DeepSpeed ZeRO)", "4× A100 80GB (FSDP)"],
            "cost_optimal": "4× A100 spot with FSDP (good price-performance)",
        },
        "large_training": {  # 70B+ parameters
            "memory_needed": "560+ GB (model + optimizer across GPUs)",
            "bottleneck": "Interconnect bandwidth (gradient sync)",
            "recommended": ["8× H100 (NVLink cluster)", "16× H100 (multi-node with InfiniBand)"],
            "cost_optimal": "Multi-node H100 with InfiniBand (EFA/RoCE/IB)",
            "critical": "NVLink required (PCIe too slow for gradient sync at this scale)",
        },
        "frontier_training": {  # 100B+ pre-training
            "memory_needed": "Terabytes (distributed across hundreds of GPUs)",
            "bottleneck": "All-reduce communication across nodes",
            "recommended": ["256-2048× H100 (InfiniBand cluster)", "TPU v5p pod (4096+ chips)"],
            "cost_optimal": "Reserved H100 clusters or TPU pods (negotiated enterprise pricing)",
            "alternatives": "CoreWeave InfiniBand clusters often cheaper than hyperscalers",
        },
    },
    
    "purchasing_strategy": {
        "spot_preemptible": {
            "savings": "60-90% vs. on-demand",
            "requirement": "Must implement checkpointing (training resumes after interruption)",
            "interruption_rate": "5-30% depending on instance type and region",
            "best_for": [
                "Training jobs > 1 hour",
                "Hyperparameter tuning (many short jobs)",
                "Batch inference (process in chunks)",
                "Development and experimentation",
            ],
            "not_for": [
                "Real-time inference (can't have interruptions)",
                "Time-critical training (deadline-bound)",
            ],
        },
        "reserved_committed": {
            "savings": "30-60% vs. on-demand (depends on term length)",
            "commitment": "1-3 year term",
            "best_for": [
                "Steady-state inference (always-on endpoints)",
                "Regular training workloads (predictable GPU usage)",
                "GPU utilization > 60% sustained",
            ],
            "strategy": "Reserve base capacity, use spot for burst",
        },
        "on_demand": {
            "savings": "0% (full price)",
            "best_for": [
                "Short experimentation (< 4 hours)",
                "Unpredictable burst needs",
                "When spot is unavailable",
            ],
            "strategy": "Minimize — always have spot or reserved alternative",
        },
    },
}


# Cost optimization patterns
cost_optimization = {
    "spot_training_pattern": {
        "description": "Use spot instances with automatic checkpointing and resume",
        "implementation": {
            "checkpoint_frequency": "Every 15 minutes or N steps",
            "checkpoint_storage": "Cloud object storage (S3/GCS)",
            "auto_resume": "Training script detects latest checkpoint on startup",
            "fallback": "If spot unavailable for 30 min, fall back to on-demand",
        },
        "savings": "60-90% for training",
    },
    
    "mixed_instance_serving": {
        "description": "Use different GPUs for different traffic tiers",
        "implementation": {
            "peak_traffic": "H100 instances (handle burst with low latency)",
            "baseline_traffic": "A10G/L4 instances (cost-efficient for steady load)",
            "off_peak": "Scale to minimum (1-2 instances) or serverless",
        },
    },
    
    "right_sizing": {
        "description": "Match GPU to actual workload needs",
        "common_mistake": "Using H100 for model that fits on A10G",
        "metrics_to_check": [
            "GPU memory utilization (should be > 60%)",
            "GPU compute utilization (should be > 70% during training)",
            "Memory bandwidth utilization (check nvidia-smi)",
        ],
        "rule": "If GPU memory util < 40%, downsize. If compute util < 30%, you're memory-bound — optimize batch size or use smaller GPU.",
    },
}
```

---

## How It Works in Practice

### GPU Cluster Sizing Example

```yaml
GPU_Cluster_Sizing:
  scenario: "Fine-tune Llama 3 70B, then serve at 500 requests/minute"
  
  training:
    model: "Llama 3 70B"
    method: "QLoRA (4-bit quantized base + LoRA adapters)"
    memory_calculation:
      model_4bit: "~35 GB (70B × 0.5 bytes/param)"
      lora_adapters: "~2 GB (r=16, target modules)"
      optimizer_states: "~4 GB (AdamW for LoRA params only)"
      activations: "~20 GB (batch_size=4, seq_len=4096)"
      total: "~61 GB"
    recommended: "1× H100 80GB or 1× H200 141GB"
    training_time: "~8 hours on 1× H100 (10K examples)"
    cost:
      h100_spot: "$4.00/hr × 8 hr = $32"
      h100_on_demand: "$8.00/hr × 8 hr = $64"
      
  serving:
    model: "Llama 3 70B (FP8 quantized for inference)"
    memory_needed: "~70 GB (FP8) + 30 GB KV cache"
    throughput_target: "500 req/min = ~8.3 req/sec"
    
    option_a:
      hardware: "1× H200 141GB"
      capacity: "~15 req/sec (single GPU, fits entire model)"
      instances_needed: 1
      cost: "$10/hr × 1 = $10/hr"
      
    option_b:
      hardware: "2× H100 80GB (tensor parallel)"
      capacity: "~20 req/sec (slightly higher due to TP overhead)"
      instances_needed: 1
      cost: "$16/hr × 1 = $16/hr"
      
    option_c:
      hardware: "4× A100 80GB (tensor parallel)"
      capacity: "~10 req/sec"
      instances_needed: 1
      cost: "$14/hr × 1 = $14/hr"
      
    recommendation: "Option A (H200) — simplest, lowest cost, no TP overhead"
    
  total_monthly_cost:
    training: "$32 (one-time fine-tuning, spot)"
    serving: "$7,200/month (H200, always-on)"
    with_autoscaling: "$4,000/month (scale down during off-peak)"
```

---

## Interview Tip

> When asked about GPU cloud options: "I match GPU to workload using three key metrics: memory capacity (what model size fits), memory bandwidth (inference speed), and compute TFLOPS (training speed). For inference: memory bandwidth dominates because LLM token generation is memory-bound. H100 generates tokens 64% faster than A100 purely from bandwidth difference (3.35 vs 2.04 TB/s). H200 adds 76% more memory (141GB vs 80GB) — fits 70B models on single GPU without tensor parallelism overhead. For training: compute TFLOPS dominates because forward/backward pass is compute-bound. H100's FP8 Transformer Engine (1979 TFLOPS) is 3× faster than A100 FP16 (312 TFLOPS). For multi-GPU: interconnect is critical — NVLink (900 GB/s) vs PCIe (64 GB/s) is 14× difference. Never do distributed training on PCIe-connected GPUs. For cost: spot instances save 60-90% with checkpointing. Reserved instances save 30-60% for steady workloads. The optimal strategy is reserved for baseline + spot for burst. Alternative providers (CoreWeave, Lambda Labs) often have better GPU availability and 20-40% lower prices than hyperscalers — worth evaluating for training-heavy workloads."

---

## Common Mistakes

1. **Using H100 for small model inference** — Deploying a 7B model on H100 ($8/hr) when A10G ($1/hr) or L4 ($0.84/hr) handles it fine with INT8 quantization. Solution: right-size GPU to model memory needs. 7B INT8 = 7 GB — doesn't need 80 GB HBM3.

2. **Ignoring memory bandwidth for inference** — Choosing GPU based on TFLOPS for inference workloads. LLM decoding is memory-bandwidth-bound, not compute-bound. Solution: compare memory bandwidth (GB/s) not TFLOPS when selecting inference GPUs. H100 > A100 > A10G for bandwidth.

3. **PCIe-connected multi-GPU training** — Trying distributed training on 4× A10G (PCIe only, 64 GB/s each). Gradient synchronization becomes the bottleneck. Solution: use NVLink-connected GPUs (A100, H100 multi-GPU instances) for distributed training. PCIe is fine for single-GPU or inference only.

4. **Not using spot instances for training** — Paying full on-demand price for 24-hour training runs. Solution: always use spot with checkpointing for training > 1 hour. The 60-90% savings is enormous at scale.

5. **Single-region dependency** — Requesting 16× H100 in a single region that has no availability. Job waits for days. Solution: implement multi-region training capability. Checkpoint to object storage, launch in whichever region has availability.

---

## Key Takeaways

- GPU hierarchy (2026): B200 > H200 > H100 > A100 > A10G/L4 (compute/memory)
- Inference is memory-bandwidth-bound: choose GPU by bandwidth, not TFLOPS
- Training is compute-bound: choose GPU by TFLOPS and interconnect bandwidth
- H200 vs H100: same compute, 76% more memory (141 vs 80 GB) — fits larger models on one GPU
- B200: 2× H100 compute, 2.4× memory (192 GB), FP4 support — the 2025-2026 frontier
- Spot instances: 60-90% savings with checkpointing — always use for training
- Reserved: 30-60% savings for steady workloads (serving endpoints)
- NVLink mandatory for multi-GPU training: 14× faster than PCIe for gradient sync
- Alternative providers: CoreWeave, Lambda Labs — often cheaper with better availability
- Right-sizing: measure GPU utilization — if memory util < 40%, downsize GPU
