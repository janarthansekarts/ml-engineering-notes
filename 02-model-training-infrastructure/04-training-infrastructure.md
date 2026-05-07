# Training Infrastructure

## The Problem / Why This Matters

Training ML (Machine Learning) models — especially large ones — requires more than just GPUs. A GPU cluster is a complex system with interconnected components that all must work together: compute (GPUs), networking (moving data between GPUs), storage (feeding data to GPUs fast enough), orchestration (scheduling and managing training jobs), and fault tolerance (handling hardware failures during multi-week training runs). A training cluster for frontier models has hundreds to thousands of GPUs connected by high-speed networks, reading from petabyte-scale distributed storage, managed by sophisticated schedulers that maximize utilization. Understanding training infrastructure matters because: (1) the infrastructure determines your maximum model size and training speed, (2) network bottlenecks can waste 30-50% of your expensive GPU compute, (3) storage that can't feed data fast enough starves GPUs, and (4) a single GPU failure in a 1000-GPU job can waste hours of training if fault tolerance isn't properly configured. In 2026, as organizations build or rent GPU clusters costing $50M-$1B+, understanding training infrastructure has direct multi-million dollar implications.

---

## The Analogy

Think of a GPU training cluster like a Formula 1 pit crew operation:

- **GPUs** = The race car engines. Incredibly powerful, incredibly expensive, and useless if the support system isn't equally fast.
- **Network (NVLink/InfiniBand)** = The pit lane. How fast information flows between cars (GPUs). A slow pit lane means cars wait for each other instead of racing.
- **Storage** = The fuel delivery system. If fuel (data) can't reach the cars fast enough, they idle on track waiting. A $10M car sitting idle because of a $100K fuel pump is waste.
- **Orchestrator (Kubernetes/SLURM)** = The race director. Decides which cars race when, handles crashes (failed GPUs), ensures the track is shared fairly between teams.
- **Fault tolerance** = The backup systems. When a car breaks down mid-race (GPU failure), the team must recover instantly rather than restarting the entire race from scratch.

---

## Deep Dive

### GPU Cluster Architecture

```yaml
GPU_Cluster_Architecture:
  single_node:
    description: "One server with multiple GPUs"
    typical_config:
      gpus: "8x H100 80GB (or 8x H200 141GB)"
      intra_node_network: "NVLink (900 GB/s per GPU bidirectional)"
      cpu: "2x AMD EPYC or 2x Intel Xeon (128-256 cores total)"
      ram: "1-2 TB DDR5"
      local_storage: "8-16 TB NVMe SSD"
      host_network: "4-8x 400G InfiniBand (or 200G)"
    examples:
      nvidia_dgx_h100: "8x H100, NVLink, 4x 400G InfiniBand"
      nvidia_dgx_b200: "8x B200, NVLink, 8x 400G InfiniBand"
      cloud_equivalent: "AWS p5.48xlarge, Azure ND H100, GCP a3-highgpu-8g"
      
  multi_node_cluster:
    description: "Multiple servers connected by high-speed network"
    components:
      compute_nodes: "16-10,000 GPU servers"
      network_fabric: "InfiniBand or RoCE switches (leaf-spine topology)"
      storage_cluster: "Parallel file system (Lustre, GPFS, WekaFS)"
      management_plane: "Job scheduler, monitoring, logging"
      cooling: "Liquid cooling for dense GPU deployment"
      
  scale_examples:
    small_team: "4-16 GPUs (1-2 nodes) — fine-tuning, small training"
    mid_size_org: "64-256 GPUs (8-32 nodes) — 7B-70B training"
    large_lab: "1000-4000 GPUs — 70B-175B training"
    frontier: "16,000-100,000 GPUs — 400B-1T training"
```

### Networking for Distributed Training

```yaml
Networking:
  why_critical: "Distributed training requires frequent synchronization — network is often the bottleneck"
  
  intra_node_interconnect:
    nvlink:
      what: "NVIDIA's high-bandwidth GPU-to-GPU interconnect"
      bandwidth: "H100: 900 GB/s bidirectional per GPU"
      topology: "Fully connected (any GPU can communicate with any other at full speed)"
      purpose: "Tensor parallelism (high-frequency, high-volume communication)"
      
    nvswitch:
      what: "Switch chip connecting all GPUs in a node via NVLink"
      enables: "Full bisection bandwidth between any GPU pair"
      
  inter_node_network:
    infiniband:
      full_name: "InfiniBand (IB)"
      what: "High-bandwidth, low-latency network for HPC"
      speed: "NDR: 400 Gbps (50 GB/s) per port"
      latency: "~1 microsecond"
      features:
        rdma: "Remote Direct Memory Access — GPU writes directly to remote GPU memory"
        gdr: "GPUDirect RDMA — bypasses CPU entirely for GPU-to-GPU communication"
      topology: "Fat-tree or Dragonfly (non-blocking for collective operations)"
      use: "Standard for high-performance GPU clusters"
      
    roce:
      full_name: "RDMA over Converged Ethernet"
      what: "RDMA capabilities on standard Ethernet infrastructure"
      speed: "100-400 Gbps"
      advantage: "Uses existing Ethernet switches (cheaper infrastructure)"
      disadvantage: "Higher latency than InfiniBand, more complex congestion management"
      use: "Cloud providers (some), cost-sensitive deployments"
      
    ethernet:
      speed: "100-400 Gbps"
      latency: "~5-10 microseconds"
      use: "Budget clusters, pipeline parallelism (lower bandwidth OK)"
      limitation: "No RDMA (unless RoCE), higher latency"
      
  communication_patterns:
    allreduce:
      what: "Sum/average across all GPUs (gradient synchronization)"
      implementation: "Ring AllReduce or tree-based reduction"
      bandwidth_requirement: "High (model_size transferred per step)"
      
    allgather:
      what: "Each GPU sends data to all others (FSDP parameter gathering)"
      bandwidth_requirement: "Very high (model_size/N per GPU per forward/backward)"
      
    point_to_point:
      what: "Direct send/receive between two GPUs (pipeline parallelism)"
      bandwidth_requirement: "Moderate (activation_size per micro-batch)"
      
  network_performance_impact:
    good_network: "InfiniBand NDR (400G) → ~95% scaling efficiency at 64 GPUs"
    medium_network: "100G Ethernet → ~75% scaling efficiency at 64 GPUs"
    bad_network: "25G Ethernet → training becomes communication-bound (50% waste)"
```

### Storage for Training

```yaml
Storage_Systems:
  requirements:
    throughput: "Must feed data to GPUs faster than they consume it"
    capacity: "Store datasets (TB-PB) and checkpoints (TB)"
    latency: "Low enough for data loading not to bottleneck training"
    concurrent_access: "Hundreds of nodes reading simultaneously"
    
  storage_tiers:
    local_nvme:
      purpose: "Staging data, caching frequently accessed subsets"
      capacity: "8-16 TB per node"
      throughput: "7-14 GB/s per node"
      use_case: "Cache current epoch's data, store checkpoints temporarily"
      
    parallel_file_system:
      purpose: "Shared storage for datasets and checkpoints"
      examples:
        lustre: "Open-source parallel file system (most common in HPC)"
        gpfs: "IBM's General Parallel File System (high reliability)"
        wekafs: "High-performance flash-native parallel file system"
        vast_data: "Universal storage with NFS + parallel access"
      throughput: "100+ GB/s aggregate across cluster"
      capacity: "Petabytes"
      use_case: "Primary dataset storage, checkpoint storage"
      
    object_storage:
      purpose: "Long-term storage, archival, data lake"
      examples:
        s3: "AWS S3"
        gcs: "Google Cloud Storage"
        azure_blob: "Azure Blob Storage"
      throughput: "Limited per-request, high aggregate with parallelism"
      capacity: "Unlimited (cloud)"
      use_case: "Raw data, archived checkpoints, dataset distribution"
      
  data_loading_strategies:
    streaming:
      what: "Read data directly from storage during training (no pre-download)"
      libraries: "HuggingFace datasets (streaming mode), MosaicML Streaming, WebDataset"
      advantage: "No local storage needed, handle datasets larger than disk"
      risk: "Network hiccup → training stall"
      
    pre_staging:
      what: "Copy data to local NVMe before training"
      advantage: "Maximum throughput, no network dependency during training"
      risk: "Staging time adds to job startup, limited by local disk size"
      
    caching:
      what: "Read from remote, cache locally, serve from cache on subsequent reads"
      advantage: "First epoch remote, subsequent epochs local speed"
      libraries: "Alluxio, CacheLib"
      
  checkpoint_storage:
    frequency: "Every N steps or M minutes (typically every 1-4 hours)"
    size: "Model params + optimizer states = 2-4x model size"
    example: "70B model checkpoint: ~280GB (FP32 optimizer), every 2 hours"
    challenge: "1000 GPU job paused while writing checkpoint → 1000× GPU-hours wasted if slow"
    solution: "Async checkpointing — write checkpoint in background while training continues"
```

### Orchestration and Job Scheduling

```yaml
Orchestration:
  job_schedulers:
    slurm:
      full_name: "Simple Linux Utility for Resource Management"
      what: "Traditional HPC job scheduler"
      use: "On-premise GPU clusters, research labs"
      strengths:
        - "Mature, well-understood, battle-tested"
        - "Excellent GPU-aware scheduling"
        - "Multi-node job support (MPI-style launching)"
        - "Priority queues, fairshare scheduling"
      example_submit: |
        #SBATCH --job-name=train_llm
        #SBATCH --nodes=8
        #SBATCH --gpus-per-node=8
        #SBATCH --ntasks-per-node=8
        #SBATCH --time=72:00:00
        srun torchrun --nnodes=8 --nproc_per_node=8 train.py
        
    kubernetes:
      what: "Container orchestration platform adapted for ML"
      use: "Cloud-native teams, multi-tenant environments"
      ml_extensions:
        volcano: "Kubernetes batch scheduling (gang scheduling for multi-node)"
        kueue: "Kubernetes-native job queuing and resource management"
        kubeflow: "End-to-end ML platform on Kubernetes"
        ray_on_k8s: "Ray cluster operator for distributed computing"
      strengths:
        - "Multi-tenant isolation"
        - "Auto-scaling (spin up GPUs on demand)"
        - "Cloud-native (works across cloud providers)"
        - "Container-based reproducibility"
      challenges:
        - "Not designed for HPC-style multi-node jobs"
        - "GPU scheduling less mature than SLURM"
        - "Networking configuration for RDMA is complex"
        
    ray:
      what: "Distributed computing framework with cluster management"
      use: "Flexible distributed training, hyperparameter search"
      components:
        ray_train: "Distributed training (DDP, FSDP, DeepSpeed)"
        ray_tune: "Hyperparameter optimization"
        ray_cluster: "Cluster management and auto-scaling"
      strengths:
        - "Pythonic API (easy to use)"
        - "Flexible (training, inference, data processing)"
        - "Auto-scaling support"
        - "Good for heterogeneous workloads"
        
  gpu_utilization_optimization:
    challenge: "GPUs are expensive ($2-6/hour) — idle GPUs waste money"
    strategies:
      gang_scheduling: "Ensure all GPUs for a distributed job start simultaneously"
      preemption: "Pause low-priority jobs to free GPUs for high-priority work"
      bin_packing: "Fill partial nodes with smaller jobs"
      fractional_gpus: "Share one GPU between multiple jobs (MIG on H100)"
      spot_instances: "Use interruptible instances (60-90% cheaper) with checkpointing"
```

### Fault Tolerance

```yaml
Fault_Tolerance:
  why_critical:
    problem: "Large training runs (weeks-months) will encounter hardware failures"
    hardware_failure_rates:
      gpu: "~1-5% failure rate per year per GPU"
      network: "Link flaps, switch failures"
      storage: "SSD failures, file system issues"
      node: "PSU failure, memory ECC errors, overheating"
    implication: "1000-GPU job for 30 days → near-certain multiple hardware failures"
    cost_of_failure: "Each restart from last checkpoint wastes hours of multi-million-dollar compute"
    
  strategies:
    checkpointing:
      what: "Periodically save full training state (model, optimizer, scheduler, data position)"
      frequency: "Every 30-120 minutes (balance between safety and overhead)"
      async: "Write checkpoint without pausing training (background copy)"
      distributed: "Each rank saves its shard (parallel writes)"
      
    elastic_training:
      what: "Training continues even when nodes join or leave"
      how: "Auto-detect node failure, redistribute work to remaining nodes"
      frameworks:
        torch_elastic: "PyTorch's built-in elastic training (torchrun)"
        deepspeed_elastic: "DeepSpeed's elastic training support"
      limitation: "Efficiency drops when node count changes (need to reshard)"
      
    redundant_computation:
      what: "Multiple GPUs compute the same work (detect/correct errors)"
      when: "Critical training runs where silent data corruption is unacceptable"
      overhead: "2-3x compute cost (only for highest-stakes training)"
      
    health_monitoring:
      what: "Continuously check GPU health, detect failures early"
      checks:
        - "GPU temperature and power (NVML monitoring)"
        - "ECC (Error-Correcting Code) error counts"
        - "NCCL communication tests (periodic sanity checks)"
        - "Training loss anomaly detection (sudden spike = possible hardware issue)"
      tools: "DCGM (Data Center GPU Manager), custom monitoring"
      
    automatic_recovery:
      steps:
        1: "Detect failure (node unreachable, GPU error, NCCL timeout)"
        2: "Identify healthy nodes"
        3: "Allocate replacement node (if available)"
        4: "Load last checkpoint"
        5: "Resume training on healthy set of nodes"
      time_to_recover: "5-30 minutes (depends on checkpoint load time)"
```

---

## How It Works in Practice

### Example: Building a Training Cluster

```yaml
Example:
  scenario: "Startup building a cluster for training 7B-70B models"
  budget: "$5M (one-time hardware) + $500K/year operations"
  
  hardware_design:
    compute: "32x NVIDIA H100 80GB (4 DGX H100 nodes)"
    networking:
      intra_node: "NVLink (900 GB/s per GPU — included in DGX)"
      inter_node: "4x InfiniBand NDR 400G per node (16 ports total)"
      switch: "1x Quantum-2 InfiniBand switch (64 port NDR)"
    storage:
      hot: "WekaFS cluster — 200TB flash, 100 GB/s aggregate throughput"
      warm: "100TB NFS for archival checkpoints"
      cold: "S3-compatible object storage (unlimited)"
    management:
      scheduler: "SLURM (4 nodes, simple enough)"
      monitoring: "Prometheus + Grafana + DCGM exporter"
      containers: "Enroot + Pyxis (NVIDIA container runtime for SLURM)"
      
  training_capability:
    7b_model: "Full training on 32 GPUs in ~7 days"
    70b_model: "Full training on 32 GPUs in ~70 days (or QLoRA fine-tuning in hours)"
    utilization_target: "75%+ (accounting for maintenance, job gaps, failures)"
    
  operational_considerations:
    power: "~200kW (4 DGX nodes × 50kW each)"
    cooling: "Liquid cooling (direct-to-chip) — required for H100 density"
    location: "Colocation facility with adequate power and cooling"
    staffing: "2 ML infrastructure engineers (part-time) + cloud vendor support"
```

---

## Interview Tip

> When asked about training infrastructure: "I think about it in four layers. (1) Compute: GPU selection based on workload — H100/H200 for LLM training, A100 for general, L4 for inference. (2) Network: InfiniBand NDR (400 Gbps) is essential for tensor parallelism, NVLink (900 GB/s) within nodes. The network often determines scaling efficiency more than raw GPU power. (3) Storage: parallel file systems (Lustre, WekaFS) that can sustain 100+ GB/s to keep GPUs fed. (4) Orchestration: SLURM for HPC-style, Kubernetes + Volcano for cloud-native. Key insight: for multi-week training runs, fault tolerance becomes critical — async checkpointing every 30-60 minutes, elastic training to handle node failures, GPU health monitoring. The infrastructure cost for frontier model training (10,000+ GPUs) can exceed $100M, making efficiency optimization directly worth millions."

---

## Common Mistakes

1. **Networking afterthought** — Building a cluster with cheap Ethernet then wondering why distributed training scales at 50% efficiency. Network investment should be proportional to compute investment. InfiniBand pays for itself through better GPU utilization.

2. **Storage bottleneck** — Buying expensive GPUs but reading training data from NFS at 1 GB/s. When 32 GPUs can consume data at 50 GB/s, slow storage means GPUs idle 90% of the time during data loading. Always benchmark storage throughput against GPU consumption rate.

3. **No checkpointing strategy** — Starting a multi-day training run without configuring checkpointing. First hardware failure means restarting from scratch — potentially losing days of $10K+/hour compute time.

4. **Treating cloud like on-prem** — Running cloud GPU instances 24/7 like an on-premise cluster. Cloud's advantage is elasticity — use spot instances for training (60-90% cheaper), checkpoint frequently, scale to zero when not training.

5. **Ignoring GPU utilization** — Not monitoring whether GPUs are actually computing or waiting (for data, communication, synchronization). A cluster with 50% GPU utilization is wasting half its investment. Profile regularly with NVIDIA Nsight or PyTorch Profiler.

---

## Key Takeaways

- Training clusters: compute (GPUs) + networking (NVLink/InfiniBand) + storage (parallel FS) + orchestration (SLURM/K8s)
- NVLink: 900 GB/s intra-node (tensor parallelism), InfiniBand NDR: 400 Gbps inter-node (data/pipeline parallelism)
- Storage must sustain throughput to keep GPUs fed — parallel file systems (100+ GB/s) not NFS
- GPUDirect RDMA: GPU-to-GPU communication bypassing CPU — essential for low-latency collectives
- Fault tolerance critical for long runs: async checkpointing, elastic training, GPU health monitoring
- Schedulers: SLURM (HPC/on-prem), Kubernetes + Volcano/Kueue (cloud-native), Ray (flexible)
- GPU utilization target: >75% (idle GPUs waste expensive hardware)
- Spot/preemptible instances: 60-90% cheaper for training (with frequent checkpointing)
- Liquid cooling: required for modern GPU density (H100/B200 draw 700W+ each)
- Infrastructure decisions have multi-million dollar implications at scale
