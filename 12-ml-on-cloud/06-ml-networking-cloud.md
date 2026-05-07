# ML Networking in the Cloud

## The Problem / Why This Matters

Networking is the hidden bottleneck in distributed ML training. A single GPU can process data at hundreds of GB/s, but when you scale to multi-node training (8, 16, 64+ GPUs across machines), the network between nodes determines whether you get linear scaling or waste 50%+ of your GPU compute waiting for data. In 2026, frontier model training requires hundreds to thousands of GPUs communicating constantly — gradient synchronization (all-reduce), tensor parallelism (per-layer communication), pipeline parallelism (inter-stage transfers), and data loading (feeding GPUs fast enough). Standard cloud networking (25-100 Gbps Ethernet) is woefully inadequate for these workloads. High-performance alternatives include: InfiniBand (traditional HPC — 400-800 Gbps, sub-microsecond latency), RDMA over Converged Ethernet (RoCE — InfiniBand performance over Ethernet), EFA (AWS Elastic Fabric Adapter), and GPUDirect (bypass CPU for GPU-to-GPU communication). Understanding ML networking means knowing: when standard Ethernet is fine (inference, small training), when you need high-performance networking (distributed training >8 GPUs), and how to design VPC (Virtual Private Cloud) networks that support ML workloads without becoming security risks.

---

## The Analogy

Think of ML networking like a highway system connecting factories:

- **Standard Ethernet (25 Gbps)** = A two-lane road between factories. Fine for occasional deliveries (inference, data loading), but causes massive traffic jams when all 8 factories need to synchronize their output simultaneously (all-reduce).
- **InfiniBand (400 Gbps)** = A 16-lane expressway with no traffic lights. Deliveries arrive almost instantly because there's no congestion, no stops, and the road was built specifically for heavy freight.
- **GPUDirect RDMA** = Factories connected by direct conveyor belts — goods move from one factory floor to another without going through a loading dock (CPU). Zero overhead, maximum throughput.
- **VPC networking** = The city planning around the factories. You need walls (firewalls), security checkpoints (IAM), private roads (private subnets), and delivery routes (NAT gateways) that keep your industrial zone secure while allowing necessary traffic.

---

## Deep Dive

### ML Network Requirements

```yaml
ML_Network_Requirements:
  communication_patterns:
    all_reduce:
      what: "Average gradients across all GPUs (data parallelism)"
      frequency: "Every training step (hundreds of times per second)"
      data_volume: "Model parameters × 2 bytes × 2 (send + receive)"
      example: "7B model → 14GB per all-reduce (every ~500ms)"
      bandwidth_needed: "14GB / 0.5s = 28 GB/s (224 Gbps) minimum"
      
    tensor_parallelism:
      what: "Split single layer across GPUs (communicate every sub-layer)"
      frequency: "Multiple times per layer, per step (very frequent)"
      data_volume: "Activation tensors (batch × hidden_dim)"
      latency_sensitive: True  # Sub-millisecond communication needed
      requirement: "NVLink (within node) — Ethernet too slow for inter-layer"
      
    pipeline_parallelism:
      what: "Split model layers across nodes (forward/backward micro-batches)"
      frequency: "Per micro-batch (less frequent than TP)"
      data_volume: "Activation tensors between pipeline stages"
      latency_sensitive: "Moderate (buffer micro-batches to hide latency)"
      
    data_loading:
      what: "Feed training data from storage to GPU memory"
      bandwidth_needed: "Typically 1-10 GB/s per node"
      bottleneck: "Rarely the bottleneck with proper prefetching"
      solution: "Local NVMe SSD cache + async prefetch from object storage"
      
  bandwidth_requirements_by_workload:
    single_gpu_training:
      network: "Standard (25 Gbps sufficient)"
      reason: "No inter-GPU communication"
      
    multi_gpu_single_node:
      network: "NVLink (600-900 GB/s between GPUs in same machine)"
      reason: "Intra-node is always NVLink, no network config needed"
      
    multi_node_small: # 2-8 nodes (16-64 GPUs)
      network: "High-performance required (100-400 Gbps per node)"
      options: ["InfiniBand 200-400 Gbps", "RoCE 100-200 Gbps", "EFA 400 Gbps"]
      reason: "All-reduce with 64 GPUs needs high bandwidth + low latency"
      
    multi_node_large: # 16-256 nodes (128-2048 GPUs)
      network: "InfiniBand mandatory (400-800 Gbps per node)"
      topology: "Fat-tree or rail-optimized (minimize hops)"
      reason: "Communication overhead dominates without high-bandwidth interconnect"
      
    frontier_training: # 1000+ GPUs
      network: "Custom InfiniBand fabric (3200 Gbps aggregate per node)"
      topology: "Multi-rail InfiniBand (4-8 ports per node)"
      requirement: "Non-blocking fabric (full bisection bandwidth)"
```

### High-Performance Networking Technologies

```yaml
HP_Networking:
  infiniband:
    what: "Purpose-built HPC interconnect (Mellanox/NVIDIA)"
    speeds: "200 Gbps (HDR), 400 Gbps (NDR), 800 Gbps (XDR)"
    latency: "~1 microsecond (vs 20-50 μs for Ethernet)"
    features:
      rdma: "Remote Direct Memory Access — bypass OS/CPU entirely"
      gpudirect: "GPU-to-GPU without touching CPU memory"
      adaptive_routing: "Dynamic load balancing across fabric"
      congestion_control: "Hardware-level flow control"
    availability:
      aws: "Not directly (EFA provides similar capabilities)"
      gcp: "Not directly (GPUDirect-TCPX)"
      azure: "Available (ND-series H100 instances have InfiniBand)"
      coreweave: "Full InfiniBand clusters"
    best_for: "Large distributed training (>16 nodes)"
    
  efa:
    what: "AWS Elastic Fabric Adapter — high-performance network interface"
    bandwidth: "Up to 3200 Gbps (p5.48xlarge)"
    latency: "~5 microseconds (higher than IB but lower than standard Ethernet)"
    features:
      - "OS-bypass (applications communicate directly with network adapter)"
      - "SRD protocol (Scalable Reliable Datagram — AWS custom)"
      - "Integration with NCCL (NVIDIA Collective Communications Library)"
      - "No special switches needed (runs on AWS network fabric)"
    supported_instances: "p4d, p5, p5en, dl1, trn1"
    note: "AWS's answer to InfiniBand — not IB but close performance for ML"
    
  gpudirect_tcpx:
    what: "GCP's GPU-to-GPU networking (GPUDirect over TCP/IP eXtended)"
    bandwidth: "Up to 3200 Gbps (a3-highgpu instances)"
    features:
      - "GPU-to-network bypass (skip CPU for GPU packets)"
      - "Custom transport protocol optimized for collective operations"
      - "Integration with NCCL"
    supported: "A3 (H100) and A3-mega (H200) instances"
    
  roce:
    what: "RDMA over Converged Ethernet — InfiniBand protocol over Ethernet switches"
    versions:
      v1: "Layer 2 only (same subnet)"
      v2: "Layer 3 routable (across subnets)"
    bandwidth: "100-400 Gbps"
    latency: "~2-5 microseconds"
    advantage: "Works with standard Ethernet switches (cheaper infrastructure)"
    disadvantage: "Requires lossless Ethernet (PFC/ECN configuration)"
    
  nccl:
    what: "NVIDIA Collective Communications Library"
    role: "Abstraction layer for multi-GPU/multi-node communication"
    operations: "AllReduce, AllGather, ReduceScatter, Broadcast"
    backends:
      - "NVLink (intra-node)"
      - "InfiniBand (inter-node, native RDMA)"
      - "EFA (AWS, SRD protocol)"
      - "TCP/IP sockets (fallback, slowest)"
    tuning:
      NCCL_IB_DISABLE: "0 (enable InfiniBand)"
      NCCL_NET_GDR_LEVEL: "5 (enable GPUDirect RDMA)"
      NCCL_ALGO: "Ring, Tree, or CollNet"
      NCCL_PROTO: "LL, LL128, or Simple"
    importance: "Correct NCCL tuning can improve distributed training throughput 2-3×"
```

### VPC Architecture for ML

```yaml
VPC_For_ML:
  architecture:
    subnets:
      training_subnet:
        type: "Private subnet (no internet access)"
        purpose: "GPU training clusters"
        size: "/20 (4096 IPs — enough for large GPU clusters)"
        features:
          - "Placement groups (cluster — minimize network latency)"
          - "Enhanced networking enabled"
          - "EFA/InfiniBand interfaces"
        security_group:
          inbound: "All traffic from training_subnet (inter-node communication)"
          outbound: "S3/GCS endpoints, ECR/Artifact Registry, nothing else"
          
      serving_subnet:
        type: "Private subnet with NAT gateway"
        purpose: "Model serving endpoints"
        size: "/22 (1024 IPs)"
        features:
          - "Load balancer integration"
          - "Auto-scaling group support"
        security_group:
          inbound: "HTTPS from load balancer only"
          outbound: "Model registry, monitoring endpoints"
          
      data_subnet:
        type: "Private subnet"
        purpose: "Data processing, feature engineering"
        size: "/22 (1024 IPs)"
        security_group:
          inbound: "From training_subnet and serving_subnet"
          outbound: "Storage endpoints only"
          
    endpoints:
      vpc_endpoints:
        - "S3 gateway endpoint (free, no NAT needed for S3)"
        - "ECR endpoint (pull training containers privately)"
        - "CloudWatch endpoint (metrics/logs without internet)"
        - "SageMaker endpoint (if using managed services)"
      private_link:
        - "Model registry"
        - "Feature store"
        - "Secrets manager"
        
    security:
      no_public_ips: "Training nodes NEVER have public IPs"
      nat_gateway: "Only for serving subnet (pull model updates)"
      network_acls: "Restrict inter-subnet traffic"
      flow_logs: "Enable VPC flow logs for all subnets (audit)"
      
  multi_region_training:
    challenge: "Cross-region latency kills distributed training performance"
    solution: "Always train within single availability zone"
    data_replication: "Replicate training data to training region before starting"
    exception: "Federated learning (designed for high-latency communication)"
```

### Network Performance Optimization

```python
# Network optimization for distributed ML training

"""
Network configuration and optimization patterns for multi-node ML training.
Covers: NCCL tuning, topology-aware placement, and bandwidth optimization.
"""

network_optimization = {
    "nccl_environment_variables": {
        "description": "Critical NCCL settings for optimal distributed training",
        "settings": {
            # Network backend selection
            "NCCL_NET": "EFA or IB (depending on cloud)",
            "NCCL_IB_DISABLE": "0 (ensure InfiniBand/EFA is enabled)",
            
            # GPUDirect settings
            "NCCL_NET_GDR_LEVEL": "5 (enable GPU Direct RDMA — skip CPU)",
            "NCCL_NET_GDR_READ": "1 (GPU reads directly from network)",
            
            # Algorithm selection
            "NCCL_ALGO": "Ring or Tree",
            # Ring: better for large messages (gradient all-reduce)
            # Tree: better for small messages (barrier, broadcast)
            
            # Protocol selection
            "NCCL_PROTO": "Simple",
            # LL (Low Latency): small messages
            # LL128: medium messages
            # Simple: large messages (gradients)
            
            # Buffer sizes
            "NCCL_BUFFSIZE": "8388608",  # 8MB (increase for large clusters)
            "NCCL_NTHREADS": "512",  # Number of NCCL threads
            
            # Debugging
            "NCCL_DEBUG": "WARN",  # INFO for detailed logging, WARN for production
            "NCCL_DEBUG_SUBSYS": "NET",  # Network-specific debugging
        },
    },
    
    "topology_aware_placement": {
        "description": "Place training nodes to minimize network hops",
        "strategies": {
            "cluster_placement_group": {
                "what": "All instances in same rack/switch (AWS/GCP/Azure)",
                "benefit": "Minimal network latency (same top-of-rack switch)",
                "limitation": "Max cluster size limited by rack capacity",
                "use_when": "Training on 2-8 nodes",
            },
            "spread_placement_group": {
                "what": "Instances across different racks (fault tolerance)",
                "benefit": "No single-rack failure takes down training",
                "limitation": "Higher network latency between nodes",
                "use_when": "Long training runs where fault tolerance > performance",
            },
            "rail_optimized_topology": {
                "what": "Each GPU connects to dedicated NIC — parallel paths",
                "benefit": "Full bandwidth utilization for all GPUs simultaneously",
                "implementation": "H100 nodes have 8 NICs (one per GPU rail)",
                "use_when": "Large-scale training (16+ nodes)",
            },
        },
    },
    
    "bandwidth_optimization": {
        "gradient_compression": {
            "what": "Compress gradients before network transfer",
            "methods": [
                "FP16 gradients (2× compression vs FP32)",
                "PowerSGD (low-rank approximation — 10-100× compression)",
                "Top-K sparsification (send only largest K% of gradients)",
            ],
            "trade_off": "Reduces bandwidth but may slow convergence",
            "recommendation": "FP16 always; PowerSGD for bandwidth-constrained setups",
        },
        "overlap_computation_communication": {
            "what": "Compute next layer while communicating current layer's gradients",
            "implementation": "PyTorch DDP with bucket_cap_mb tuning",
            "benefit": "Hides communication latency behind computation",
            "setting": "bucket_cap_mb=25 (default) — tune based on model architecture",
        },
        "reduce_scatter_plus_all_gather": {
            "what": "FSDP pattern — reduce_scatter in backward, all_gather in forward",
            "benefit": "Each GPU only holds 1/N of parameters (memory savings)",
            "network_impact": "Same total bytes transferred but better pipelining",
        },
    },
    
    "data_loading_network": {
        "description": "Ensure data pipeline doesn't bottleneck GPU training",
        "patterns": {
            "local_ssd_cache": {
                "what": "Cache training data on local NVMe SSDs",
                "bandwidth": "3-7 GB/s read (NVMe) vs 5-10 Gbps (network)",
                "use_when": "Dataset fits on local disk (<10TB per node)",
            },
            "parallel_data_loading": {
                "what": "Multiple workers fetch data in parallel from storage",
                "config": "num_workers=4-8 per GPU, prefetch_factor=2-4",
                "storage_backend": "S3/GCS with VPC endpoint (no internet hop)",
            },
            "distributed_filesystem": {
                "what": "Lustre/FSx/Filestore for shared high-bandwidth storage",
                "bandwidth": "100+ GB/s aggregate for large clusters",
                "use_when": "Dataset too large for local SSD, random access needed",
                "aws": "FSx for Lustre",
                "gcp": "Filestore Enterprise",
                "azure": "Azure NetApp Files",
            },
        },
    },
}
```

---

## How It Works in Practice

### Multi-Node Training Network Design

```yaml
Multi_Node_Training_Design:
  scenario: "Train 70B model on 32× H100 GPUs (4 nodes × 8 GPUs)"
  
  network_architecture:
    intra_node:
      technology: "NVLink (4th gen)"
      bandwidth: "900 GB/s per GPU pair"
      topology: "NVSwitch — all-to-all within node"
      use: "Tensor parallelism (degree=8, within node)"
      
    inter_node:
      technology: "EFA (AWS) or InfiniBand (Azure/CoreWeave)"
      bandwidth: "3200 Gbps per node (8 NICs × 400 Gbps)"
      topology: "Rail-optimized (GPU[i] → NIC[i] on all nodes)"
      use: "Pipeline parallelism + data parallelism (between nodes)"
      
  parallelism_strategy:
    tensor_parallel: 8  # Within node (NVLink fast enough)
    pipeline_parallel: 2  # Between pairs of nodes
    data_parallel: 2  # Between pipeline-parallel groups
    total_gpus: "8 × 2 × 2 = 32 GPUs"
    
  network_traffic:
    tensor_parallel:
      volume: "~500 MB per forward pass (activations)"
      frequency: "Every sub-layer (very frequent)"
      bandwidth_used: "~400 GB/s (well within NVLink 900 GB/s)"
      
    pipeline_parallel:
      volume: "~100 MB per micro-batch (inter-stage activations)"
      frequency: "Per micro-batch"
      bandwidth_used: "~10 Gbps (well within EFA 3200 Gbps)"
      
    data_parallel:
      volume: "~35 GB per all-reduce (half the 70B model gradients)"
      frequency: "Every training step (~1 second)"
      bandwidth_used: "~280 Gbps (within EFA 3200 Gbps)"
      
  efficiency:
    compute_efficiency: "~90% (minimal communication bottleneck)"
    comparison_standard_ethernet: "~45% (network would be 7× bottleneck)"
    
  cost_impact:
    with_efa_infiniband: "4 days training = $32K"
    with_standard_25gbps: "9+ days (communication bound) = $72K+"
    savings: "56% cost savings from proper networking"
```

---

## Interview Tip

> When asked about ML networking: "Network architecture is the difference between 90% and 40% GPU efficiency in distributed training. I design based on three principles: (1) Tensor parallelism stays within a node (NVLink at 900 GB/s is fast enough — never cross the network for per-layer communication). (2) Pipeline and data parallelism go across nodes using high-bandwidth interconnect (EFA, InfiniBand, or GPUDirect-TCPX). (3) Bandwidth must exceed gradient volume per step — for a 7B model with 1-second step time, you need at least 14 GB/s × 2 = 224 Gbps for all-reduce. Standard 25 Gbps Ethernet is 9× too slow. For VPC design: training nodes in private subnets with cluster placement groups (same rack/switch), no public IPs, VPC endpoints for storage access, and NCCL tuned for the network backend (GPUDirect RDMA enabled, correct algorithm selection). Key optimization: overlap communication with computation (PyTorch DDP does this automatically with proper bucket sizing), use gradient compression (FP16 at minimum), and cache training data on local NVMe to avoid network bottleneck on data loading. The ROI is significant: proper networking at 3200 Gbps vs. standard 25 Gbps can cut training time (and cost) by 50-70% for large distributed jobs."

---

## Common Mistakes

1. **Ignoring placement groups** — Launching 4-node training job with instances in different availability zones. Inter-AZ latency (1-2ms) destroys gradient synchronization performance. Solution: always use cluster placement groups — all training nodes in same rack/switch.

2. **Standard networking for distributed training** — Using 25 Gbps Ethernet for 8-node training. GPUs spend 60%+ of time waiting for gradient sync. Solution: use EFA (AWS), InfiniBand (Azure), or GPUDirect-TCPX (GCP) instances for any training > 8 GPUs.

3. **Not tuning NCCL** — Using default NCCL settings with NCCL_IB_DISABLE=1 (InfiniBand disabled!). Performance drops 2-3× because NCCL falls back to TCP sockets. Solution: explicitly set NCCL_NET, NCCL_NET_GDR_LEVEL, and verify with NCCL_DEBUG=INFO.

4. **Public training nodes** — GPU training instances with public IPs for "easy SSH access." Exposes expensive infrastructure to the internet. Solution: use private subnets + bastion host or AWS SSM Session Manager. Training nodes should never have public IPs.

5. **Cross-region data loading** — Training data in us-east-1, GPU cluster in us-west-2. Every batch loads across regions (slow, expensive). Solution: replicate training data to same region/zone as training cluster. Use VPC endpoints for storage access (zero egress cost for S3/GCS).

---

## Key Takeaways

- Network is the bottleneck: determines whether distributed training scales linearly or wastes 50%+ GPU
- Standard Ethernet (25 Gbps): fine for inference and single-GPU training only
- High-performance networking: EFA (AWS), InfiniBand (Azure/CoreWeave), GPUDirect-TCPX (GCP)
- Tensor parallelism: always intra-node (NVLink 900 GB/s — never cross the network)
- Data/pipeline parallelism: inter-node — requires 100+ Gbps per node minimum
- NCCL tuning: correct environment variables can improve performance 2-3×
- Placement groups: cluster training nodes in same rack for minimal latency
- VPC design: private subnets, no public IPs, VPC endpoints for storage, flow logs
- Data loading: local NVMe cache or distributed filesystem (FSx/Lustre/Filestore)
- Cost impact: proper networking saves 50-70% on large training runs (faster = cheaper)
