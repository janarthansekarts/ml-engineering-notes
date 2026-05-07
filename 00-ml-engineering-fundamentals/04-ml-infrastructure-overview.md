# ML Infrastructure Overview

## The Problem / Why This Matters

ML workloads have fundamentally different infrastructure requirements than traditional web applications. A web server needs CPU, memory, and network. ML training needs massive GPU (Graphics Processing Unit) compute with high-bandwidth interconnects, terabytes of fast storage, and specialized scheduling. ML serving needs low-latency GPU inference with dynamic scaling. In 2026, infrastructure costs dominate ML budgets — a single H100 GPU costs ~$30,000, and training a large model can cost millions. Choosing the wrong infrastructure means either burning cash on over-provisioned resources or suffering from bottlenecks that make experiments take days instead of hours. Understanding ML infrastructure means knowing what hardware exists, how to choose between options, how to orchestrate workloads efficiently, and how to optimize costs. This is the foundation everything else in ML engineering builds upon.

---

## The Analogy

Think of ML infrastructure like a Formula 1 racing team:

- **GPUs** = The engine. Raw power that determines top speed (training throughput). Different engines (H100 vs A100 vs consumer GPU) for different races (large model training vs fine-tuning vs inference).
- **Networking** (InfiniBand, RoCE) = The transmission. Connects engine power to the wheels. Distributed training is only as fast as GPUs can communicate.
- **Storage** (NVMe, object storage) = The fuel system. Data needs to flow to GPUs faster than they can consume it, or the engine starves.
- **Orchestration** (Kubernetes, Slurm) = The pit crew strategy. Deciding which car gets which driver, when to pit, how to optimize across the race.
- **Cost optimization** = The team budget cap. You can't just buy the best of everything — you need to allocate resources strategically.

---

## Deep Dive

### GPU Computing Landscape (2026)

```yaml
GPU_Landscape_2026:
  nvidia_data_center:
    b200:
      generation: "Blackwell (latest)"
      memory: "192 GB HBM3e"
      fp8_performance: "~4500 TFLOPS"
      interconnect: "NVLink 5 (1.8 TB/s)"
      use_case: "Large model training, frontier model inference"
      availability: "Limited — hyperscalers and well-funded startups"
      cost: "~$40,000-$50,000 per GPU"
      
    h200:
      generation: "Hopper (previous gen, widely available)"
      memory: "141 GB HBM3e"
      fp8_performance: "~3958 TFLOPS"
      interconnect: "NVLink 4 (900 GB/s)"
      use_case: "Model training, high-throughput inference"
      availability: "Available on major clouds"
      cloud_cost: "$4-6/hour (on-demand), $2-3/hour (spot/preemptible)"
      
    h100:
      generation: "Hopper (mature, best availability)"
      memory: "80 GB HBM3"
      fp8_performance: "~3958 TFLOPS"
      interconnect: "NVLink 4 (900 GB/s)"
      use_case: "General training and inference, most common in 2026"
      cloud_cost: "$3-4/hour (on-demand), $1.5-2.5/hour (spot)"
      
    l40s:
      generation: "Ada Lovelace"
      memory: "48 GB GDDR6X"
      use_case: "Inference, fine-tuning, smaller models"
      cloud_cost: "$1-2/hour"
      advantage: "Good price/performance for inference workloads"
      
    a100:
      generation: "Ampere (legacy but still widely used)"
      memory: "40 GB or 80 GB HBM2e"
      use_case: "Still used for training and inference, being phased out"
      cloud_cost: "$1.5-3/hour"
      note: "Many existing clusters still use A100s — not obsolete yet"

  alternatives_to_nvidia:
    amd_mi300x:
      memory: "192 GB HBM3"
      status: "Growing adoption, ROCm software stack improving"
      use_case: "Training and inference, price competitive"
      
    google_tpu_v5p:
      memory: "95 GB HBM"
      status: "Available on GCP, excellent for JAX/TensorFlow workloads"
      advantage: "Custom interconnects, integrated with Vertex AI"
      
    aws_trainium2:
      status: "AWS custom chip for training"
      advantage: "Cost-effective on AWS, integrated with SageMaker"
      limitation: "Vendor lock-in, limited framework support vs NVIDIA"
      
    intel_gaudi3:
      status: "Available on AWS and other clouds"
      use_case: "Training and inference, competitive pricing"
      
    apple_silicon:
      chips: "M4 Pro/Max/Ultra"
      use_case: "Local development, fine-tuning small models, on-device inference"
      memory: "Unified memory (up to 192 GB on Ultra)"
      advantage: "No GPU rental needed for experimentation"
```

### Storage for ML

```yaml
Storage_Architecture:
  training_data_storage:
    object_storage:
      services: "S3, GCS, Azure Blob Storage"
      use_case: "Raw data, datasets, model artifacts"
      performance: "High throughput, high latency (not ideal for random access)"
      cost: "$0.02-0.03/GB/month"
      
    high_performance_storage:
      services: "FSx for Lustre (AWS), Filestore (GCP), Azure NetApp Files"
      use_case: "Training data that GPUs read directly"
      performance: "High throughput + low latency, parallel file system"
      cost: "$0.15-0.30/GB/month"
      when_needed: "Large-scale distributed training with many GPUs reading same data"
      
    local_nvme:
      use_case: "Caching training data on GPU nodes"
      performance: "Fastest possible — data local to the machine"
      pattern: "Stage data from S3/GCS to local NVMe before training"
      
  feature_store_storage:
    offline_store: "Data warehouse (BigQuery, Redshift) or Parquet on object storage"
    online_store: "Redis, DynamoDB, Bigtable — low-latency key-value lookups"
    
  vector_database_storage:
    services: "Pinecone, Qdrant, Weaviate, Milvus, pgvector"
    use_case: "Embeddings for RAG, similarity search, recommendation"
    scale: "Millions to billions of vectors"
    
  model_artifact_storage:
    what: "Trained model weights, preprocessors, configs"
    where: "Model registry (MLflow) backed by object storage"
    sizes: "Small models: MBs. LLMs: 10s-100s of GBs. Frontier models: TBs"
```

### Orchestration and Scheduling

```yaml
Orchestration:
  kubernetes_for_ml:
    why: "Standard for container orchestration, extensible for ML workloads"
    ml_specific_components:
      gpu_operator: "NVIDIA GPU Operator — manages GPU drivers and device plugins"
      gpu_scheduling: "GPU sharing, MIG (Multi-Instance GPU), time-slicing"
      distributed_training: "Training operators (PyTorch, TensorFlow, MPI)"
      job_scheduling: "Volcano, Kueue — gang scheduling, fair queuing"
    tools:
      kubeflow: "ML platform on Kubernetes — pipelines, notebooks, training, serving"
      ray: "Distributed computing framework — Ray Train, Ray Serve, Ray Tune"
      kserve: "Model serving on Kubernetes — multi-framework, autoscaling"
      
  managed_services:
    aws_sagemaker: "Fully managed — training, endpoints, pipelines, feature store"
    gcp_vertex_ai: "Google's ML platform — AutoML, custom training, Model Garden"
    azure_ml: "Microsoft's platform — workspace, compute clusters, endpoints"
    
  hpc_schedulers:
    slurm: "Traditional HPC scheduler, used by research labs and large GPU clusters"
    use_case: "When you own/rent bare metal GPU servers"
    advantage: "Fine-grained control, optimized for multi-node GPU jobs"
    
  pipeline_orchestration:
    kubeflow_pipelines: "ML-specific DAG orchestration on Kubernetes"
    airflow: "General DAG orchestration, widely used for ML batch jobs"
    dagster: "Modern alternative to Airflow, better for data-aware orchestration"
    prefect: "Cloud-native workflow orchestration"
```

---

## How It Works in Practice

### Infrastructure Decision Framework

```yaml
Decision_Framework:
  scenario_1_startup_early:
    context: "Seed-stage AI startup, 2-3 engineers, building MVP"
    recommendation:
      training: "Cloud GPUs (spot instances) — RunPod, Lambda Cloud, or cloud provider"
      serving: "Serverless (Modal, Replicate) or small GPU instance"
      storage: "S3/GCS for everything"
      orchestration: "Simple scripts + GitHub Actions, maybe Dagster"
    monthly_cost: "$1,000-$5,000"
    principle: "Don't over-engineer. Ship fast, optimize later."
    
  scenario_2_growth_stage:
    context: "Series A/B company, 5-10 ML engineers, multiple models in production"
    recommendation:
      training: "Kubernetes cluster with GPU node pools (spot for training)"
      serving: "KServe or Ray Serve on Kubernetes"
      storage: "Feature store (Feast) + object storage + vector DB"
      orchestration: "Kubeflow Pipelines or Dagster"
      monitoring: "Evidently + Prometheus/Grafana"
    monthly_cost: "$20,000-$100,000"
    principle: "Invest in platform. Self-serve ML for data scientists."
    
  scenario_3_enterprise:
    context: "Large company, 50+ ML engineers, hundreds of models"
    recommendation:
      training: "Dedicated GPU clusters (reserved instances or on-prem)"
      serving: "Multi-model serving platform (Triton) with autoscaling"
      storage: "Enterprise feature store + data lake + multiple vector DBs"
      orchestration: "Custom ML platform built on Kubernetes"
      governance: "Full model registry, approval workflows, audit trails"
    monthly_cost: "$500,000-$5,000,000+"
    principle: "Standardize, govern, optimize for cost at scale."
```

---

## Interview Tip

> When asked about ML infrastructure choices: "I evaluate infrastructure across four dimensions: (1) Compute — what GPU generation, how many, shared vs dedicated. For training I prefer spot/preemptible for cost savings with checkpointing. For serving I need reliable on-demand with autoscaling. (2) Storage — tiered approach: object storage for raw data, high-performance parallel FS for active training, NVMe cache on nodes, feature store for serving. (3) Networking — for distributed training, inter-GPU bandwidth is critical (NVLink within node, InfiniBand/RoCE between nodes). (4) Orchestration — Kubernetes with GPU operators for flexibility, or managed services (SageMaker/Vertex AI) for simplicity. The key trade-off is always control vs operational overhead vs cost."

---

## Common Mistakes

1. **Using on-demand GPUs for training** — Training jobs can be checkpointed and resumed. Always use spot/preemptible instances (50-70% cheaper) with checkpoint-on-preemption for training. Reserve on-demand for serving.

2. **Ignoring data loading bottlenecks** — Buying expensive H100s but feeding them data from slow S3 reads. GPUs sit idle waiting for data. Profile the data pipeline first — use local NVMe caching and parallel data loading.

3. **Over-provisioning from day one** — Buying a massive Kubernetes cluster before you have workloads. Start small, autoscale up. Cloud elasticity is the advantage — use it.

4. **Not considering total cost of ownership** — Comparing only GPU hour costs while ignoring: networking, storage, engineering time, idle resources, and operational overhead. A cheaper GPU that takes 2x longer may cost more total.

5. **Vendor lock-in without awareness** — Building entirely on SageMaker or Vertex AI proprietary features without abstraction. Fine for now, expensive to migrate later. At minimum, keep training code framework-native (PyTorch).

---

## Key Takeaways

- GPU landscape 2026: B200 (frontier) → H200/H100 (mainstream) → L40S (inference) → A100 (legacy)
- Storage tiering: object store (cheap, bulk) → parallel FS (training throughput) → NVMe (cache) → Redis (serving)
- Use spot/preemptible instances for training (checkpoint + resume), on-demand for serving
- Kubernetes is the standard for ML orchestration (with GPU operators, training operators, KServe)
- Managed services (SageMaker, Vertex AI) trade control for operational simplicity
- Data loading is often the bottleneck — profile before buying more GPUs
- Cost optimization is a primary ML engineering responsibility — not an afterthought
- Alternative chips (AMD MI300X, Google TPU v5p, AWS Trainium2) are viable options in 2026
