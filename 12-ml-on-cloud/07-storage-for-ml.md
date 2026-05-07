# Storage for ML Workloads

## The Problem / Why This Matters

ML workloads have unique storage requirements that standard cloud storage configurations can't handle efficiently. Training pipelines read terabytes of data at high throughput, checkpoints write tens of gigabytes every few minutes, model artifacts need to be versioned and served from multiple regions, and feature stores require sub-millisecond latency for online serving. Getting storage wrong means: GPU idle time waiting for data (wasting $8-15/hr per GPU), training failures from checkpoint storage bottlenecks, slow model deployments from large artifact downloads, and exploding costs from storing redundant copies across services. In 2026, ML storage challenges have intensified with: larger datasets (10-100TB+ for pre-training), larger models (70B+ requiring 140GB+ checkpoints), more frequent checkpointing (every 5-15 minutes for spot instance recovery), and multi-modal data (images, video, audio alongside text). Understanding ML storage means knowing: which storage tier for which purpose, how to achieve maximum throughput for data loading, efficient checkpointing strategies, and cost optimization across the storage lifecycle.

---

## The Analogy

Think of ML storage like a restaurant's food storage system:

- **Object storage (S3/GCS/ADLS)** = The warehouse. Massive capacity, very cheap per unit. But getting items from the warehouse to the kitchen takes time (high latency). You store bulk ingredients (training datasets), archived recipes (old model versions), and backup supplies (checkpoints) here.
- **Block storage (EBS/Persistent Disk)** = The walk-in refrigerator. Right next to the kitchen, fast access, limited size. You keep today's ingredients (currently-training dataset) and work-in-progress (active checkpoints) here.
- **Local NVMe SSD** = The countertop prep station. Fastest access but smallest capacity. You keep the current dish's ingredients (current batch data) here — pre-loaded from the refrigerator.
- **High-performance filesystem (FSx/Lustre)** = A conveyor belt from warehouse to kitchen. Designed for continuous high-throughput delivery. Multiple chefs (GPUs) can grab items simultaneously without bottlenecking.
- **Feature Store online (Redis/Bigtable)** = The spice rack. Tiny amounts, but you need any spice instantly (sub-millisecond). Features served to models at inference time.

---

## Deep Dive

### Storage Tiers for ML

```yaml
Storage_Tiers_ML:
  object_storage:
    services:
      aws: "Amazon S3"
      gcp: "Google Cloud Storage (GCS)"
      azure: "Azure Data Lake Storage Gen2 (ADLS)"
    characteristics:
      latency: "20-100ms (first byte)"
      throughput: "Scales with parallelism (100+ GB/s with many connections)"
      capacity: "Unlimited"
      durability: "99.999999999% (11 nines)"
      cost: "$0.02-0.03/GB/month (standard tier)"
    ml_use_cases:
      - "Training datasets (bulk storage, read at training time)"
      - "Model artifacts (versioned model files)"
      - "Checkpoints (periodic saves during training)"
      - "Pipeline intermediate outputs"
      - "Experiment logs and metrics"
    optimization:
      parallelism: "Use multi-part upload/download (16-64 threads)"
      format: "Parquet/TFRecord (columnar, splittable)"
      prefetching: "DataLoader prefetch while GPU computes"
      caching: "Cache on local SSD for repeated epoch reads"
      
  block_storage:
    services:
      aws: "EBS (Elastic Block Store)"
      gcp: "Persistent Disk (PD)"
      azure: "Azure Managed Disks"
    characteristics:
      latency: "1-5ms"
      throughput: "Up to 4 GB/s (io2 Block Express)"
      capacity: "Up to 64 TB per volume"
      cost: "$0.08-0.12/GB/month (SSD), $0.05/GB for provisioned IOPS"
    ml_use_cases:
      - "OS and framework installation (boot volume)"
      - "Fast checkpointing (attached to training instance)"
      - "Shuffle buffer for training data"
    limitations:
      - "Single-instance attachment (can't share across nodes)"
      - "Region-locked (can't access from other regions easily)"
      
  local_nvme:
    services:
      aws: "Instance store (ephemeral, included with instance)"
      gcp: "Local SSD (ephemeral)"
      azure: "Temp disk / NVMe on certain instances"
    characteristics:
      latency: "~0.1ms (100 microseconds)"
      throughput: "3-14 GB/s (depends on instance)"
      capacity: "375 GB - 30 TB (depends on instance type)"
      cost: "Included with instance (no additional charge)"
      durability: "EPHEMERAL — data lost on instance stop/terminate"
    ml_use_cases:
      - "Training data cache (copy from S3/GCS for fast reads)"
      - "Shuffle buffer (random access during training)"
      - "Intermediate activations (gradient checkpointing)"
      - "Working directory for data processing"
    critical: "DATA IS LOST on stop/terminate — never store only copy here"
    
  high_performance_filesystem:
    services:
      aws: "FSx for Lustre"
      gcp: "Filestore / Cloud Storage FUSE"
      azure: "Azure NetApp Files"
    characteristics:
      latency: "~1ms"
      throughput: "100+ GB/s aggregate (scales with cluster size)"
      capacity: "Petabytes"
      shared: "Multiple nodes can access simultaneously (POSIX)"
      cost: "$0.14-0.35/GB/month (expensive at scale)"
    ml_use_cases:
      - "Shared training data for multi-node training"
      - "Checkpoint storage accessible from all nodes"
      - "Random-access datasets (can't easily parallelize from object store)"
    key_feature: "POSIX — works with any framework without code changes"
    
  feature_store_online:
    services:
      redis: "ElastiCache (AWS) / Memorystore (GCP) / Azure Cache for Redis"
      bigtable: "GCP Bigtable (Vertex AI Feature Store backend)"
      dynamodb: "AWS DynamoDB (SageMaker Feature Store backend)"
    characteristics:
      latency: "< 1ms (sub-millisecond)"
      throughput: "Millions of reads/sec"
      capacity: "Limited by memory/cost"
      cost: "$0.15-0.50/GB/month (Redis), varies by service"
    ml_use_cases:
      - "Real-time feature serving for inference"
      - "User/item embeddings for recommendation"
      - "Pre-computed features for online prediction"
```

### Data Loading Architecture

```yaml
Data_Loading_Architecture:
  challenge: "Feed 8× H100 GPUs (combined ~26 TB/s memory bandwidth) without stalling"
  
  pipeline:
    stage_1_storage:
      source: "Object storage (S3/GCS)"
      throughput: "~10-50 GB/s (with parallel readers)"
      optimization:
        - "Multi-threaded download (16-64 workers)"
        - "VPC endpoint (no internet hop)"
        - "Same region/AZ as compute"
        - "Optimal file size: 128MB-1GB (not too small, not too large)"
        
    stage_2_local_cache:
      destination: "Local NVMe SSD on training instance"
      throughput: "3-14 GB/s read"
      strategy:
        streaming: "Don't cache — read directly from storage each epoch"
        cached: "Download dataset once, read from local SSD for all epochs"
        hybrid: "Cache hot data locally, stream cold data from storage"
      decision: "Cache if: dataset < local disk AND epochs > 1"
      
    stage_3_cpu_processing:
      operations: "Decode, transform, augment, tokenize"
      parallelism: "num_workers=4-8 per GPU"
      optimization:
        - "Use efficient formats (WebDataset, Mosaic StreamingDataset)"
        - "Parallel decoding (multiple CPU cores)"
        - "Avoid GIL bottleneck (multiprocessing, not threading)"
        
    stage_4_gpu_prefetch:
      mechanism: "Async transfer CPU → GPU memory"
      pytorch: "pin_memory=True, non_blocking=True"
      overlap: "Prefetch next batch while GPU processes current batch"
      
  optimal_data_formats:
    for_text:
      format: "Pre-tokenized binary (numpy memmap or Arrow)"
      why: "Zero deserialization cost — direct memory mapping"
      examples: "HuggingFace datasets (Arrow), SlimPajama format"
      
    for_images:
      format: "WebDataset (tar archives) or TFRecord"
      why: "Sequential reads from object storage (no random access penalty)"
      optimization: "Resize/crop on CPU while GPU trains on previous batch"
      
    for_tabular:
      format: "Parquet (columnar, compressed, splittable)"
      why: "Read only needed columns, efficient compression"
      optimization: "Partition by training split (train/val/test)"
      
    for_multi_modal:
      format: "WebDataset or custom streaming format"
      why: "Keeps related modalities together (image + text pairs)"
      examples: "LAION format, DataComp format"
```

### Checkpointing Strategy

```python
# Checkpoint storage patterns for ML training

"""
Checkpoint strategies: frequency, storage tier, cost optimization.
Critical for spot instance training and fault tolerance.
"""

checkpoint_strategies = {
    "small_model_training": {
        "model_size": "< 1B parameters (~2 GB checkpoint)",
        "strategy": {
            "frequency": "Every epoch or every 1000 steps",
            "storage": "Object storage (S3/GCS) directly",
            "format": "PyTorch state_dict (compressed)",
            "retention": "Keep last 3 checkpoints + best checkpoint",
            "cost": "~$0.10/month for 4 × 2 GB checkpoints",
        },
    },
    
    "medium_model_training": {
        "model_size": "7B parameters (~14 GB checkpoint with optimizer)",
        "strategy": {
            "frequency": "Every 15 minutes (for spot instance recovery)",
            "storage": {
                "primary": "Local NVMe (fastest write — 7 GB/s)",
                "backup": "Object storage (async upload after local write)",
            },
            "format": "PyTorch FSDP sharded checkpoints (parallel save)",
            "retention": "Local: last 2; Object storage: last 5 + best",
            "cost": "~$5/month for 7 × 14 GB checkpoints in S3",
            "write_time": "~2 seconds local, ~30 seconds to S3 (background)",
        },
    },
    
    "large_model_training": {
        "model_size": "70B parameters (~140 GB checkpoint)",
        "strategy": {
            "frequency": "Every 15-30 minutes",
            "storage": {
                "primary": "High-performance filesystem (FSx for Lustre)",
                "backup": "Object storage (async, less frequent — every hour)",
            },
            "format": "Sharded across GPUs (each GPU saves its shard in parallel)",
            "retention": "Filesystem: last 2; Object storage: last 3 + best",
            "cost": "~$50/month for checkpoints in S3",
            "write_time": {
                "parallel_to_lustre": "~10 seconds (all GPUs write simultaneously)",
                "serial_to_s3": "~3 minutes (background upload)",
            },
        },
        "optimization": {
            "async_checkpointing": "Training continues while checkpoint uploads to S3",
            "incremental": "Only save changed parameters (delta from last checkpoint)",
            "compression": "LZ4 compression (fast, ~30% reduction)",
        },
    },
    
    "spot_instance_recovery": {
        "description": "Resume training after spot interruption",
        "flow": [
            "1. Spot interruption notification received (2-minute warning)",
            "2. Save emergency checkpoint to fastest available storage",
            "3. New spot instance launched (or fallback to on-demand)",
            "4. Load latest valid checkpoint from object storage",
            "5. Resume training from exact step (same optimizer state)",
        ],
        "requirements": {
            "checkpoint_must_include": [
                "Model parameters (state_dict)",
                "Optimizer state (momentum, variance for AdamW)",
                "Learning rate scheduler state",
                "Training step / epoch counter",
                "Data loader state (which samples have been seen)",
                "Random number generator states (reproducibility)",
            ],
            "storage_requirement": "Object storage (survives instance termination)",
        },
    },
}


# Cost optimization for ML storage
storage_cost_optimization = {
    "lifecycle_policies": {
        "description": "Automatically transition data to cheaper storage tiers",
        "rules": [
            {
                "data_type": "Training data (raw)",
                "active": "Standard tier (during active training)",
                "after_30_days": "Infrequent Access ($0.0125/GB/month)",
                "after_90_days": "Glacier/Archive ($0.004/GB/month)",
            },
            {
                "data_type": "Checkpoints",
                "active": "Standard tier (latest 3 checkpoints)",
                "older": "Delete after training completes (except best model)",
                "savings": "Don't accumulate checkpoints — they add up fast",
            },
            {
                "data_type": "Model artifacts",
                "active": "Standard tier (deployed models)",
                "inactive": "Infrequent Access (previous versions)",
                "archive": "Glacier Deep Archive (models > 6 months old)",
            },
            {
                "data_type": "Experiment logs",
                "active": "Standard (last 30 days)",
                "after_30_days": "Infrequent Access",
                "after_365_days": "Archive or delete",
            },
        ],
    },
    
    "deduplication": {
        "description": "Avoid storing same data multiple times",
        "common_waste": [
            "Training data copied to every experiment directory",
            "Model checkpoints with full optimizer state (often not needed for inference)",
            "Preprocessed data not cleaned up after training",
        ],
        "solutions": [
            "Centralized data registry (single source of truth)",
            "Symlinks or references instead of copies",
            "Automatic cleanup policies for intermediate data",
            "Separate model weights from optimizer state (inference only needs weights)",
        ],
    },
    
    "right_sizing_storage": {
        "block_storage": {
            "mistake": "Provisioning 1TB EBS for every training instance (pays $80/month)",
            "solution": "Use local NVMe (free with instance) + S3 for persistence",
            "when_ebs_needed": "Only if local NVMe is insufficient AND you need persistence",
        },
        "filesystem": {
            "mistake": "Keeping FSx for Lustre running 24/7 ($0.14/GB/month)",
            "solution": "Create FSx linked to S3 at training start, delete after training",
            "dynamic": "FSx can hydrate from S3 on-demand (lazy loading)",
        },
    },
}
```

### High-Performance Storage Patterns

```yaml
High_Performance_Storage:
  fsx_for_lustre:
    what: "AWS managed parallel filesystem (Lustre)"
    performance:
      throughput: "Up to 1000 MB/s per TiB of storage"
      iops: "Millions of IOPS"
      latency: "< 1ms"
    integration:
      s3_linked: "Automatically syncs with S3 bucket (data appears as files)"
      lazy_loading: "Files loaded from S3 on first access (no upfront copy)"
    sizing:
      rule: "Provision based on throughput needs, not just capacity"
      example: "1.2 TB filesystem → 1.2 GB/s throughput (enough for 8 GPU data loading)"
    cost: "$0.14/GB/month (Persistent SSD) or $0.07/GB/month (Persistent HDD)"
    
  gcs_fuse:
    what: "Mount GCS buckets as local filesystem (FUSE)"
    performance:
      throughput: "Limited by network bandwidth to GCS"
      latency: "Higher than native filesystem (FUSE overhead)"
    advantage: "Simple — no separate storage service to manage"
    limitation: "Not ideal for small random reads (high latency per operation)"
    optimization: "Large sequential reads with prefetching"
    
  streaming_datasets:
    mosaic_streaming:
      what: "Efficient streaming from object storage during training"
      features:
        - "Deterministic shuffling (reproducible across runs)"
        - "Multi-worker safe (no duplicate samples)"
        - "Elastic sharding (handles node failures gracefully)"
        - "No local storage needed (streams directly)"
      use_case: "Large datasets that don't fit on local disk (>10TB)"
      
    webdataset:
      what: "Sequential tar archives for efficient streaming"
      features:
        - "Optimal for object storage (large sequential reads)"
        - "Shuffling across shards + within buffers"
        - "Standard format (works with any training framework)"
      use_case: "Image/multimodal datasets (LAION, DataComp)"
      
    huggingface_datasets:
      what: "Apache Arrow backed with streaming mode"
      features:
        - "Memory-mapped access (zero-copy reads)"
        - "Streaming mode (no download needed)"
        - "Automatic caching and deduplication"
      use_case: "NLP datasets, text corpora"
```

---

## How It Works in Practice

### Production Storage Architecture

```yaml
Production_Storage_Architecture:
  scenario: "ML platform supporting 50+ models, 20 TB training data"
  
  architecture:
    data_lake:
      storage: "S3 (Standard tier)"
      organization:
        - "s3://ml-data-lake/raw/{dataset-name}/{version}/"
        - "s3://ml-data-lake/processed/{dataset-name}/{version}/"
        - "s3://ml-data-lake/features/{feature-group}/{version}/"
      access_control: "IAM policies per team/project"
      versioning: "S3 versioning enabled (rollback capability)"
      
    training_storage:
      hot: "Local NVMe on GPU instances (cache current dataset)"
      warm: "FSx for Lustre (shared across multi-node training)"
      cold: "S3 (persistent dataset storage)"
      flow: "S3 → FSx (lazy load) → NVMe cache → GPU"
      
    checkpoint_storage:
      primary: "S3 (all checkpoints persisted here)"
      fast_path: "Local NVMe (latest 2 checkpoints for fast resume)"
      lifecycle: "Delete after 7 days, keep only best model long-term"
      
    model_registry:
      storage: "S3 with versioning"
      format: "model-name/version/artifacts/"
      metadata: "MLflow or cloud-native model registry"
      serving: "CDN/replica in serving region for fast cold-starts"
      
    feature_store:
      offline: "S3 (Parquet partitioned by date)"
      online: "Redis cluster (sub-ms lookups)"
      sync: "Batch job materializes offline → online daily"
      
  cost_breakdown:
    data_lake: "$400/month (20 TB × $0.023/GB)"
    training_nvme: "$0 (included with GPU instances)"
    fsx_lustre: "$300/month (2 TB, active during training only)"
    checkpoints: "$50/month (lifecycle policy limits retention)"
    model_artifacts: "$100/month (4 TB across all versions)"
    feature_store_redis: "$800/month (50 GB high-memory cluster)"
    total_storage: "~$1,650/month"
    note: "Storage is <5% of total ML cost (GPUs dominate)"
```

---

## Interview Tip

> When asked about ML storage: "I design ML storage in tiers matched to access patterns. Training data lives in object storage (S3/GCS) — cheap, durable, unlimited. For multi-GPU training, I cache data on local NVMe SSDs (3-14 GB/s, free with instance) or use FSx for Lustre (shared parallel filesystem for multi-node). The key optimization: prefetch next batch to GPU while processing current batch — pipeline the data loading to hide storage latency. For checkpointing: frequency depends on spot instance strategy. With spot (60-90% savings), I checkpoint every 15 minutes to local NVMe (2-second write) then async-upload to S3 (background). Each checkpoint must include model weights, optimizer state, scheduler state, and data loader position for exact resume. For serving: model artifacts in S3 with CDN replication to serving regions (fast cold-start), features in Redis/Bigtable (sub-millisecond online lookups). Cost optimization: lifecycle policies (move old data to Glacier automatically), delete intermediate checkpoints after training, and right-size block storage (local NVMe is free — don't over-provision EBS). Storage is typically <5% of ML infrastructure cost, but poor storage architecture causes GPU idle time — which makes the 95% GPU cost go up."

---

## Common Mistakes

1. **GPU idle waiting for data** — DataLoader can't keep up with GPU training speed. GPUs sit idle 30%+ of the time. Solution: increase num_workers (4-8 per GPU), enable pin_memory=True, prefetch_factor=2+, cache dataset on local NVMe, use efficient formats (WebDataset, Parquet).

2. **Checkpoints only on local disk** — Training on spot instances, checkpoint to local NVMe. Instance gets preempted — all progress lost. Solution: always persist checkpoints to object storage (S3/GCS). Write locally for speed, then async-upload to durable storage.

3. **FSx for Lustre running 24/7** — Provisioned 10 TB FSx filesystem that's only used during training (8 hours/day). Paying $1,400/month for 16 hours of idle time daily. Solution: create FSx linked to S3 at training start, delete after training completes. Data persists in S3.

4. **Storing full optimizer states in model registry** — 70B model checkpoint: 140 GB (weights) + 280 GB (AdamW optimizer state). Storing both in model registry for serving. Solution: separate inference checkpoint (weights only, 140 GB) from training checkpoint (weights + optimizer, 420 GB). Serving only needs weights.

5. **No lifecycle policies** — Accumulating years of old checkpoints, intermediate datasets, and experiment artifacts. Storage costs grow linearly forever. Solution: automated lifecycle policies — delete checkpoints after 7 days, archive old datasets to Glacier after 90 days, clean up intermediate processing outputs immediately.

---

## Key Takeaways

- Storage tiers: object storage (bulk), block storage (attached), NVMe (cache), filesystem (shared), Redis (online serving)
- Data loading pipeline: object storage → local NVMe cache → CPU preprocessing → GPU prefetch
- Prefetching critical: overlap data loading with GPU computation (hide latency)
- Efficient formats: WebDataset (images), Parquet (tabular), pre-tokenized binary (text)
- Checkpointing: local NVMe for speed + async upload to object storage for durability
- Checkpoint frequency: every 15 min for spot instances (recovers from interruption)
- High-performance filesystem: FSx for Lustre (multi-node shared access, 1 GB/s per TiB)
- Feature store: offline in S3/BigQuery (training) + online in Redis/Bigtable (serving <1ms)
- Cost optimization: lifecycle policies, delete intermediates, right-size (NVMe is free)
- Storage is <5% of ML cost, but poor storage causes GPU idle (making GPU cost 2× higher)
