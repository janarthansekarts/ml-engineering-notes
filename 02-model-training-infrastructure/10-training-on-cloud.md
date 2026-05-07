# Training on Cloud

## The Problem / Why This Matters

Most organizations don't own GPU clusters — they rent compute from cloud providers. Training ML (Machine Learning) models on cloud infrastructure introduces a different set of engineering challenges compared to on-premise: choosing between managed services (SageMaker, Vertex AI, Azure ML) vs raw GPU instances (EC2, GCE, AKS), managing costs (GPU instances cost $2-6/hour and bills escalate quickly), leveraging spot/preemptible instances (60-90% cheaper but can be terminated anytime), handling distributed training across cloud networking (different performance characteristics than InfiniBand), and integrating with cloud-native storage and orchestration. In 2026, cloud GPU availability has improved significantly but premium GPUs (H100, H200) still require capacity reservations or spot usage. Understanding cloud training infrastructure — pricing models, instance types, managed platforms, spot strategies, and cost optimization — is essential for ML engineers who need to train models within budget while maximizing quality and iteration speed.

---

## The Analogy

Think of cloud training like renting kitchen space for your catering business:

- **On-premise cluster** = Owning a commercial kitchen. High upfront cost, but you control everything and it's always available. Makes sense if you cook 16 hours/day.
- **Reserved cloud instances** = A 1-year kitchen lease. You get a guaranteed space at a discount, but you pay whether you cook or not. Good if you use it consistently.
- **On-demand instances** = Renting a kitchen by the hour. Flexible, available anytime, but expensive if you use it a lot. Good for unpredictable needs.
- **Spot instances** = Using someone else's kitchen when they're not cooking. 70% cheaper, but they can kick you out with 2 minutes notice. Great if you save your recipe (checkpoint) frequently.
- **Managed ML services** = A fully equipped test kitchen with a chef assistant. More expensive per hour, but someone else handles setup, equipment maintenance, and cleanup (infrastructure management).

---

## Deep Dive

### Cloud GPU Instance Types (2026)

```yaml
AWS_GPU_Instances:
  training_focused:
    p5_48xlarge:
      gpus: "8× H100 80GB"
      interconnect: "NVSwitch (intra-node), EFA 3200 Gbps (inter-node)"
      use: "Large model training, distributed training"
      cost: "$98/hour on-demand, $32/hour spot (when available)"
      
    p5e_48xlarge:
      gpus: "8× H200 141GB"
      interconnect: "NVSwitch, EFA 3200 Gbps"
      use: "Memory-intensive training (large batch, long sequences)"
      cost: "$130/hour on-demand"
      availability: "Limited (requires reservation)"
      
    p4d_24xlarge:
      gpus: "8× A100 40GB"
      interconnect: "NVSwitch, EFA 400 Gbps"
      use: "Standard training, fine-tuning"
      cost: "$32/hour on-demand, $10/hour spot"
      
  inference_training_versatile:
    g5_48xlarge:
      gpus: "8× A10G 24GB"
      use: "Fine-tuning small models, inference"
      cost: "$16/hour on-demand"
      
    g6_48xlarge:
      gpus: "8× L4 24GB"
      use: "Inference, small model training, QLoRA fine-tuning"
      cost: "$14/hour on-demand"

GCP_GPU_Instances:
  a3_mega:
    gpus: "8× H100 80GB"
    interconnect: "NVLink, GPUDirect-TCPX (inter-node)"
    use: "Large model training"
    cost: "$98/hour on-demand"
    
  a3_highgpu_8g:
    gpus: "8× H100 80GB"
    use: "Standard high-end training"
    cost: "$90/hour on-demand, $27/hour spot"
    
  a2_ultragpu_8g:
    gpus: "8× A100 80GB"
    use: "Training and fine-tuning"
    cost: "$40/hour on-demand"
    
  g2_standard_96:
    gpus: "8× L4 24GB"
    use: "Fine-tuning, inference"
    cost: "$12/hour on-demand"
    
  tpu_v5p:
    type: "TPU (Google's custom AI chip)"
    chips: "Up to 8960 chips per pod"
    use: "Very large model training (Google's Gemini trains on TPUs)"
    advantage: "Excellent interconnect, XLA optimization"
    cost: "Varies by region and commitment"

Azure_GPU_Instances:
  nd_h100_v5:
    gpus: "8× H100 80GB"
    interconnect: "NVLink, InfiniBand NDR"
    use: "Large model training"
    cost: "$96/hour on-demand"
    
  nd_a100_v4:
    gpus: "8× A100 80GB"
    interconnect: "NVLink, InfiniBand HDR"
    use: "Standard training"
    cost: "$37/hour on-demand"
    
  nc_a100_v4:
    gpus: "1-4× A100 80GB"
    use: "Fine-tuning, medium training"
    cost: "$5-18/hour on-demand"
```

### Managed ML Platforms

```yaml
Managed_Platforms:
  aws_sagemaker:
    what: "Fully managed ML platform (data → training → deployment)"
    training_features:
      distributed: "Built-in DDP, model parallelism, data parallelism"
      spot_training: "Managed spot with automatic checkpointing and resumption"
      hyperparameter_tuning: "Built-in Bayesian HPO (SageMaker Tuning Jobs)"
      managed_metrics: "Automatic CloudWatch metrics, logs, profiling"
    advantages:
      - "Integrated with AWS ecosystem (S3, ECR, CloudWatch)"
      - "Managed spot training (auto-resume from checkpoint on interruption)"
      - "SageMaker Training Compiler (XLA optimization, 10-30% speedup)"
      - "SageMaker Profiler (identify bottlenecks)"
    disadvantages:
      - "30-50% markup over raw EC2 instances"
      - "Less flexibility than raw instances"
      - "Vendor lock-in (SageMaker-specific APIs)"
    cost: "Instance cost + 15-30% SageMaker markup"
    
  gcp_vertex_ai:
    what: "Google's managed ML platform"
    training_features:
      custom_training: "Any container, any framework"
      distributed: "Multi-node, multi-GPU with NCCL optimization"
      tpu_training: "First-class TPU support (unique to GCP)"
      hyperparameter_tuning: "Vertex AI Vizier (Bayesian optimization)"
    advantages:
      - "TPU access (not available on AWS/Azure)"
      - "Vertex AI Vizier (advanced black-box optimization)"
      - "Integration with BigQuery, GCS, Vertex AI Pipelines"
      - "Vertex AI TensorBoard (managed experiment tracking)"
    disadvantages:
      - "Smaller GPU fleet than AWS"
      - "Less mature than SageMaker for some workflows"
    
  azure_ml:
    what: "Microsoft's managed ML platform"
    training_features:
      compute_clusters: "Auto-scaling GPU clusters"
      distributed: "Built-in DDP, DeepSpeed, FSDP support"
      designer: "No-code ML pipeline builder"
      mlflow_integration: "Native MLflow tracking and model registry"
    advantages:
      - "Best DeepSpeed integration (Microsoft develops both)"
      - "InfiniBand interconnect on ND-series (rare in cloud)"
      - "Native MLflow support"
      - "Azure OpenAI integration"
    disadvantages:
      - "Enterprise-focused (complex for small teams)"
      - "Documentation can be convoluted"
      
  when_to_use_managed:
    use_managed:
      - "Small team (1-3 ML engineers) — don't want to manage infrastructure"
      - "Need managed spot training (auto-resume)"
      - "Want integrated experiment tracking + deployment"
      - "Enterprise compliance requirements (managed security)"
    use_raw_instances:
      - "Maximum flexibility needed (custom networking, storage)"
      - "Cost-sensitive (save 30-50% vs managed markup)"
      - "Large ML platform team (can manage infrastructure)"
      - "Need custom distributed training setups"
```

### Spot/Preemptible Training

```yaml
Spot_Training:
  concept:
    what: "Use cloud provider's spare GPU capacity at 60-90% discount"
    risk: "Instance can be reclaimed with 30s-2min notice"
    key_insight: "Perfect for training (stateless if you checkpoint frequently)"
    
  pricing_comparison:
    aws_p5_48xlarge:
      on_demand: "$98/hour"
      spot: "$30-40/hour (when available)"
      savings: "60-70%"
    aws_p4d_24xlarge:
      on_demand: "$32/hour"
      spot: "$10-15/hour"
      savings: "55-70%"
    gcp_a3_highgpu:
      on_demand: "$90/hour"
      preemptible: "$27/hour"
      savings: "70%"
      
  implementation_strategy:
    checkpointing:
      frequency: "Every 15-30 minutes (balance between overhead and data loss)"
      type: "Async (don't pause training to save)"
      storage: "Cloud object storage (S3/GCS) — persists across instance termination"
      
    interruption_handling:
      detect: "Cloud provider sends termination signal (2min AWS, 30s GCP)"
      action_1: "Save emergency checkpoint immediately"
      action_2: "Request new spot instance"
      action_3: "Load checkpoint on new instance, resume training"
      automation: "Cloud-native (SageMaker Spot) or custom scripts"
      
    multi_zone_strategy:
      what: "Request spot instances across multiple availability zones"
      why: "Spot availability varies by zone — more chances of getting capacity"
      implementation: "Pool of instances across 3-4 zones"
      
  managed_spot_options:
    sagemaker_managed_spot:
      what: "SageMaker handles interruption, checkpointing, resume automatically"
      how: "Set use_spot_instances=True, checkpoint_s3_uri='s3://...'"
      savings: "Up to 90% with automatic management"
      
    gcp_spot_vms:
      what: "GCP provides spot VMs (30s termination notice)"
      handling: "Must implement own checkpointing and restart logic"
      
  best_practices:
    - "Checkpoint every 15-30 min to cloud storage"
    - "Use instance metadata to detect termination signal"
    - "Design training to be resumable (load checkpoint → continue)"
    - "Don't use spot for final hours of critical training (risk too high)"
    - "Mix spot + on-demand: spot for training, on-demand for the last 5-10%"
```

### Cloud Storage for Training

```yaml
Cloud_Storage:
  options:
    object_storage:
      services: "S3 (AWS), GCS (GCP), Azure Blob"
      throughput: "10-100 Gbps (depends on instance type and parallelism)"
      latency: "10-50ms per request"
      use: "Datasets, checkpoints, artifacts"
      cost: "$0.02-0.03/GB/month"
      
    managed_file_systems:
      aws_fsx_lustre:
        what: "Managed Lustre parallel file system"
        throughput: "Up to 1 TB/s"
        integration: "Auto-hydrates from S3"
        use: "High-performance training data access"
        cost: "$0.14/GB/month (persistent), $0.07/GB/month (scratch)"
        
      gcp_filestore:
        what: "Managed NFS"
        throughput: "Up to 100 GB/s (enterprise tier)"
        use: "Shared storage for multi-node training"
        
      azure_managed_lustre:
        what: "Managed Lustre on Azure"
        use: "HPC-style training workloads"
        
    local_nvme:
      what: "Instance-attached NVMe SSDs"
      throughput: "7-14 GB/s per instance"
      persistence: "Lost when instance terminates (ephemeral)"
      use: "Cache training data locally for max throughput"
      strategy: "Stage from S3 to local NVMe before training"
      
  data_access_patterns:
    small_dataset:
      strategy: "Download to local NVMe before training"
      tools: "aws s3 sync, gsutil rsync"
      when: "Dataset < instance storage (< 16TB)"
      
    large_dataset:
      strategy: "Stream from object storage or use managed file system"
      tools: "HuggingFace streaming, WebDataset, FSx Lustre"
      when: "Dataset > instance storage or frequently updated"
      
    checkpoint_strategy:
      save: "Write to local NVMe (fast), async copy to S3/GCS (durable)"
      load: "Read from S3/GCS to local NVMe, then load into GPU"
      why: "Local write is fast (no training stall), S3 copy provides durability"
```

### Cost Optimization

```yaml
Cost_Optimization:
  strategies:
    right_sizing:
      what: "Choose the cheapest instance that meets your requirements"
      common_mistake: "Using 8× H100 for a 7B model fine-tuning that fits on 1× A100"
      rule: "Start small, scale up only when justified by bottleneck"
      
    reserved_instances:
      what: "1-3 year commitment for 30-60% discount"
      when: "Consistent GPU usage (>60% utilization over the commitment period)"
      risk: "Locked in even if needs change"
      
    spot_instances:
      savings: "60-90% vs on-demand"
      when: "Fault-tolerant workloads (training with checkpointing)"
      not_for: "Production inference, deadline-critical jobs"
      
    training_efficiency:
      what: "Reduce total GPU-hours needed through algorithmic improvements"
      techniques:
        - "Mixed precision (2x throughput)"
        - "FlashAttention (2-4x faster attention)"
        - "Efficient data loading (no GPU idle time)"
        - "Optimal batch size (maximize throughput)"
      impact: "2-5x cost reduction from efficiency alone"
      
    auto_shutdown:
      what: "Automatically stop/terminate idle instances"
      implementation: "Lambda/Cloud Functions monitoring GPU utilization"
      saves: "Prevents $100s-$1000s from forgotten running instances"
      
    multi_cloud:
      what: "Use whichever cloud has cheapest spot GPUs at the moment"
      tools: "SkyPilot (UC Berkeley) — automatic multi-cloud job submission"
      challenge: "Data movement between clouds, different APIs"
      
  budget_planning:
    formula: "Cost = num_GPUs × hours × price_per_GPU_hour"
    examples:
      7b_fine_tuning:
        setup: "4× A100 spot instances on AWS"
        duration: "8 hours"
        cost: "4 × $3.50 × 8 = $112"
      70b_fine_tuning:
        setup: "8× H100 (1 node) on-demand"
        duration: "24 hours"
        cost: "1 × $98 × 24 = $2,352"
      7b_pretraining:
        setup: "32× H100 (4 nodes) spot"
        duration: "14 days"
        cost: "4 × $32 × 336 = $43,008"
```

---

## How It Works in Practice

### Example: Cloud Training Pipeline

```yaml
Example:
  scenario: "Startup fine-tuning 70B model on AWS (budget: $5000)"
  
  architecture:
    compute: "1× p5.48xlarge (8× H100) — spot instance"
    storage: 
      data: "S3 bucket (pre-tokenized dataset, 500GB)"
      cache: "Instance NVMe (stage data locally)"
      checkpoints: "S3 (durable checkpoint storage)"
    tracking: "W&B (cloud-hosted, free tier)"
    orchestration: "Custom launch script (torchrun)"
    
  workflow:
    step_1_prepare:
      - "Pre-tokenize dataset on cheaper instance (c5.4xlarge)"
      - "Upload tokenized shards to S3"
      - "Push training container to ECR"
      
    step_2_launch:
      - "Request p5.48xlarge spot instance"
      - "If unavailable: try alternative AZ, then on-demand as fallback"
      - "Stage data from S3 to local NVMe (15 min)"
      
    step_3_train:
      - "Launch training: torchrun --nproc_per_node=8 train.py"
      - "FSDP (ZeRO-3) across 8 GPUs"
      - "Checkpoint to S3 every 20 minutes"
      - "Monitor via W&B (loss, throughput, GPU util)"
      
    step_4_interruption_handling:
      - "Spot termination notice received (2 min warning)"
      - "Emergency checkpoint saved to S3"
      - "Script requests new spot instance"
      - "New instance starts, downloads last checkpoint, resumes"
      - "Total downtime: ~10 min per interruption"
      
    step_5_completion:
      - "Training complete after ~36 hours (including 3 interruptions)"
      - "Final model saved to S3"
      - "Instance terminated automatically"
      
  cost_breakdown:
    spot_compute: "36 hours × $32/hour (spot) = $1,152"
    interruption_overhead: "~2 hours wasted (restarting) × $32 = $64"
    data_staging: "c5.4xlarge × 2 hours = $1.40"
    storage: "S3 (500GB + checkpoints) = $15"
    total: "$1,232 (well under $5000 budget)"
    vs_on_demand: "Would have been $3,528 (65% savings from spot)"
```

---

## Interview Tip

> When asked about training on cloud: "My cloud training strategy balances cost, reliability, and performance. For compute: I use spot instances (60-90% cheaper) with frequent checkpointing (every 15-30 min to object storage) and automatic restart on termination. For storage: I stage training data to local NVMe for maximum throughput, keep checkpoints in durable cloud storage (S3/GCS), and use streaming for datasets too large to cache locally. For distributed training: I ensure instances have adequate networking — EFA on AWS, GPUDirect on GCP — without which multi-node training is communication-bound. Cost optimization: right-size instances (don't use 8 H100s for a 7B fine-tune), use mixed precision and FlashAttention to maximize throughput per dollar, and auto-terminate idle instances. For managed vs raw: SageMaker/Vertex AI for small teams that don't want to manage infrastructure (30-50% premium is worth the reduced ops burden), raw instances for teams with ML platform engineers who can squeeze more efficiency."

---

## Common Mistakes

1. **Forgetting to terminate instances** — Starting a GPU instance for experimentation and forgetting about it for a weekend. p5.48xlarge at $98/hour × 48 hours = $4,704 wasted. Always set auto-shutdown policies, budget alerts, and idle detection.

2. **Not using spot for training** — Running training jobs on-demand "because it's simpler." Spot instances save 60-90% and training is inherently fault-tolerant (checkpoint → resume). The 15 minutes of setup for spot handling saves thousands of dollars.

3. **Poor instance selection** — Using p5.48xlarge (8× H100, $98/hour) for QLoRA fine-tuning that only needs 1× A100 ($5/hour). Always right-size: calculate memory requirements, choose cheapest instance that fits.

4. **Data transfer bottleneck** — Training on cloud without staging data locally or using fast storage. Reading training data from S3 over network at 1 GB/s when GPUs can consume 50 GB/s means 98% GPU idle time. Stage to local NVMe or use FSx Lustre.

5. **Not accounting for total cost** — Budgeting only GPU hours without considering: data transfer (egress fees), storage, networking, HPO trials, failed runs, and idle time. Actual cost is typically 1.3-2x raw GPU cost. Always add 30% buffer to GPU-only estimates.

---

## Key Takeaways

- Cloud training trades capex for opex — no upfront cost but careful management needed to avoid waste
- Spot instances: 60-90% savings — checkpoint every 15-30 min, auto-resume on termination
- Right-size: match instance to workload — don't use 8× H100 for a job that needs 1× A100
- Managed platforms (SageMaker, Vertex AI, Azure ML): 30-50% premium for reduced ops burden
- Storage hierarchy: S3 (durable, cheap) → FSx Lustre (fast, shared) → local NVMe (fastest, ephemeral)
- Stage training data to local NVMe before training — network access to S3 is too slow for large models
- Auto-shutdown: always configure — forgotten GPU instances waste $100s-$1000s
- Multi-cloud tools: SkyPilot for finding cheapest spot GPUs across AWS/GCP/Azure
- Networking matters: ensure EFA/GPUDirect for multi-node — standard Ethernet kills distributed training performance
- Budget formula: (GPU-hours × price) × 1.3 (overhead for storage, transfer, failures, idle time)
