# GCP Vertex AI

## The Problem / Why This Matters

Google Cloud's Vertex AI is the unified ML platform that brings together Google's decades of AI research (TensorFlow, TPUs, Transformer architecture, BERT, PaLM, Gemini) into a managed service for enterprise ML. In 2026, Vertex AI stands out for three reasons: (1) TPU (Tensor Processing Unit) access — Google's custom AI accelerators offer superior price-performance for large model training, (2) Gemini model integration — native access to Google's frontier model family with fine-tuning and grounding capabilities, and (3) MLOps maturity — Vertex AI Pipelines, Feature Store, Model Registry, and Experiments form a cohesive platform. For ML engineers, Vertex AI is particularly strong for: training at scale on TPUs (cheaper than H100s for many workloads), deploying Gemini-based applications with enterprise controls, and building end-to-end ML pipelines with Kubeflow Pipelines integration. The platform provides everything from AutoML (no-code model training) to custom containers (bring your own framework). Understanding Vertex AI means knowing when TPUs outperform GPUs, how to leverage Google's model garden (pre-trained models), and how to build production pipelines that scale.

---

## The Analogy

Think of Vertex AI like Google's self-driving car research applied to ML engineering:

- **Google's advantage** = They invented the map (Transformers, TPUs, TensorFlow). The roads (infrastructure), the navigation (MLOps), and the best vehicles (Gemini, PaLM) all come from the same company. Deep integration because it's all designed together.
- **Vertex AI** = You get access to Google's AI research vehicle fleet. Want a sedan (AutoML for simple tasks)? A truck (custom training on TPUs for heavy lifting)? A sports car (Gemini for frontier capabilities)? All maintained by Google's mechanics (managed infrastructure).
- **TPUs** = A purpose-built engine for AI. Not a general-purpose GPU adapted for ML — a chip designed from scratch for matrix multiplication and attention. Faster for specific workloads, different programming model.

---

## Deep Dive

### Vertex AI Architecture

```yaml
Vertex_AI_Architecture:
  model_garden:
    what: "Catalog of pre-trained models available for deployment and fine-tuning"
    categories:
      google_models:
        - "Gemini 2.5 (Pro, Flash, Ultra) — frontier multi-modal"
        - "PaLM 2 — text and code generation"
        - "Imagen 3 — image generation"
        - "Chirp — speech-to-text"
        - "Codey — code generation and completion"
      open_models:
        - "Llama 3 (Meta)"
        - "Mistral (Mistral AI)"
        - "Gemma 2 (Google open-weight)"
        - "Stable Diffusion"
      specialized:
        - "MedLM (healthcare)"
        - "Sec-PaLM (security)"
        
  training:
    custom_training:
      what: "Run any training code on managed infrastructure"
      frameworks: "PyTorch, TensorFlow, JAX, HuggingFace, custom containers"
      hardware:
        gpus:
          - "NVIDIA A100 40GB/80GB"
          - "NVIDIA H100 80GB"
          - "NVIDIA L4 (cost-optimized inference/training)"
        tpus:
          - "TPU v4 (per-chip: 275 TFLOPS BF16)"
          - "TPU v5e (cost-optimized training and inference)"
          - "TPU v5p (highest performance training)"
          - "TPU v6e (latest generation, 2026)"
      distribution:
        - "Multi-GPU (NCCL)"
        - "Multi-node GPU (Vertex AI distributed training)"
        - "Multi-TPU pod (up to thousands of chips)"
        
    automl:
      what: "No-code/low-code model training"
      types:
        tabular: "Classification, regression, forecasting"
        image: "Classification, object detection, segmentation"
        text: "Classification, entity extraction, sentiment"
        video: "Classification, object tracking"
      use_case: "Quick baselines, non-ML teams, simple problems"
      
    hyperparameter_tuning:
      what: "Managed hyperparameter search"
      algorithms: "Bayesian optimization, grid search, random search"
      features: "Early stopping, parallel trials, resumable studies"
      
  serving:
    endpoints:
      what: "Managed model serving with auto-scaling"
      features:
        - "Online prediction (real-time, < 100ms)"
        - "Batch prediction (large-scale offline)"
        - "Traffic splitting (A/B testing, canary deployment)"
        - "Model monitoring (drift detection)"
        - "Private endpoints (VPC-native)"
      hardware: "GPU, TPU, or CPU serving"
      
    model_optimization:
      compilation: "Vertex AI Model Optimizer (TensorRT, OpenVINO)"
      quantization: "INT8, INT4 quantization for inference"
      distillation: "Distill large models to smaller ones for serving"
      
  pipelines:
    what: "MLOps pipeline orchestration (Kubeflow Pipelines v2)"
    features:
      - "Pipeline templates and reusable components"
      - "Artifact tracking and lineage"
      - "Scheduling (cron-based or event-triggered)"
      - "Caching (skip unchanged steps)"
      - "Integration with Vertex AI components"
    language: "Python SDK (kfp — Kubeflow Pipelines)"
    
  feature_store:
    what: "Managed feature management"
    features:
      - "Online serving (Bigtable-backed, < 10ms)"
      - "Offline (BigQuery-backed, for training)"
      - "Point-in-time lookups (prevent data leakage)"
      - "Feature monitoring (drift detection)"
      - "Feature sharing across projects"
      
  experiments:
    what: "Experiment tracking and model comparison"
    features:
      - "Log metrics, parameters, artifacts"
      - "Compare runs visually"
      - "Integration with TensorBoard"
      - "Model lineage tracking"
      
  vector_search:
    what: "Managed vector similarity search (for RAG)"
    backend: "ScaNN (Scalable Nearest Neighbors)"
    features:
      - "Billion-scale vector index"
      - "Real-time updates"
      - "Filtering and hybrid search"
      - "Low latency (< 10ms at billion scale)"
```

### TPU Training

```yaml
TPU_Training:
  what: "Google's custom AI accelerators — designed specifically for ML workloads"
  
  generations:
    tpu_v4:
      bf16_tflops: 275
      hbm: "32 GB HBM2e"
      interconnect: "ICI (Inter-Chip Interconnect) — 4.8 Tbps"
      pod_size: "Up to 4096 chips"
      best_for: "Large model training (pre-training at scale)"
      
    tpu_v5e:
      bf16_tflops: 197
      hbm: "16 GB HBM2e"
      cost: "~50% lower than v4 per TFLOP"
      best_for: "Cost-optimized training and inference"
      
    tpu_v5p:
      bf16_tflops: 459
      hbm: "95 GB HBM3"
      pod_size: "Up to 8960 chips"
      best_for: "Highest performance training (frontier models)"
      
  tpu_vs_gpu:
    tpu_advantages:
      - "Higher throughput for transformer workloads (BF16 native)"
      - "Massive interconnect bandwidth (ICI >> NVLink for large pods)"
      - "Better price-performance for large-scale training"
      - "Native JAX support (Google's ML framework)"
      
    gpu_advantages:
      - "Broader framework support (PyTorch ecosystem is GPU-first)"
      - "More flexible (general compute, not just ML)"
      - "Easier debugging (CUDA tooling is mature)"
      - "Available on all clouds (TPU = GCP only)"
      
    when_tpu:
      - "Training >10B parameter models"
      - "JAX/Flax framework (native TPU support)"
      - "Transformer architectures (attention is TPU-optimized)"
      - "Google Cloud is primary cloud"
      - "Cost-sensitive large training runs"
      
    when_gpu:
      - "PyTorch-first teams (better GPU support)"
      - "Multi-cloud requirement"
      - "Custom CUDA kernels needed"
      - "Smaller models (< 1B parameters)"
      - "Mixed workloads (training + inference + general compute)"
      
  jax_for_tpus:
    what: "Google's ML framework designed for TPUs and distributed training"
    advantages:
      - "XLA compilation (automatic optimization for TPU/GPU)"
      - "Functional programming model (pure functions, explicit state)"
      - "pjit for distributed training (partition across TPU pod)"
      - "vmap for automatic batching"
    libraries:
      - "Flax (neural network library on JAX)"
      - "Optax (optimizers)"
      - "Orbax (checkpointing)"
      - "MaxText (Google's reference LLM implementation)"
```

### Gemini Integration

```yaml
Gemini_on_Vertex:
  what: "Deploy and customize Google's Gemini models through Vertex AI"
  
  capabilities:
    gemini_flash:
      context_window: "1M tokens"
      best_for: "Fast, cost-effective inference (chat, summarization)"
      latency: "~200ms first token"
      
    gemini_pro:
      context_window: "2M tokens"
      best_for: "Complex reasoning, code generation, analysis"
      multimodal: "Text, image, video, audio input"
      
  customization:
    fine_tuning:
      supervised: "Fine-tune on custom (input, output) pairs"
      rlhf: "Reinforcement learning from human feedback"
      distillation: "Distill Gemini Pro into smaller model for cost reduction"
      
    grounding:
      what: "Connect Gemini to external data sources"
      sources:
        - "Google Search (real-time web information)"
        - "Custom documents (your knowledge base)"
        - "Vertex AI Vector Search (RAG)"
        - "Enterprise data (BigQuery, Cloud Storage)"
      benefit: "Reduces hallucination by grounding in factual sources"
      
    function_calling:
      what: "Gemini invokes your APIs/functions as tools"
      use_case: "AI agents that take actions (query databases, call APIs)"
      
  enterprise_controls:
    - "VPC Service Controls (data doesn't leave your network)"
    - "Customer-managed encryption keys (CMEK)"
    - "Data residency (keep data in specific region)"
    - "Audit logging (who accessed what)"
    - "No training on customer data (enterprise commitment)"
```

### Implementation Patterns

```python
# Vertex AI implementation patterns

"""
GCP Vertex AI patterns: training, serving, pipelines, and Gemini integration.
"""

# Vertex AI Custom Training Job
vertex_training_config = {
    "gpu_training": {
        "display_name": "fraud-detection-training",
        "worker_pool_specs": [
            {
                "machine_spec": {
                    "machine_type": "n1-standard-8",
                    "accelerator_type": "NVIDIA_TESLA_A100",
                    "accelerator_count": 4,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3:latest",
                    "command": ["python", "train.py"],
                    "args": ["--epochs=50", "--batch-size=256"],
                },
                "disk_spec": {
                    "boot_disk_type": "pd-ssd",
                    "boot_disk_size_gb": 200,
                },
            }
        ],
    },
    
    "tpu_training": {
        "display_name": "llm-pretraining-tpu",
        "worker_pool_specs": [
            {
                "machine_spec": {
                    "machine_type": "cloud-tpu",
                    "accelerator_type": "TPU_V5E",
                    "accelerator_count": 8,  # TPU v5e-8 (8 chips)
                },
                "replica_count": 4,  # 4 hosts × 8 chips = 32 TPU chips
                "container_spec": {
                    "image_uri": "us-docker.pkg.dev/my-project/training/jax-tpu:latest",
                    "command": ["python", "pretrain.py"],
                    "args": [
                        "--model-size=7b",
                        "--batch-size=1024",
                        "--precision=bf16",
                        "--steps=100000",
                    ],
                },
            }
        ],
    },
    
    "distributed_gpu": {
        "display_name": "multi-node-training",
        "worker_pool_specs": [
            {
                "machine_spec": {
                    "machine_type": "a2-ultragpu-8g",  # 8× A100 80GB
                    "accelerator_type": "NVIDIA_A100_80GB",
                    "accelerator_count": 8,
                },
                "replica_count": 4,  # 32 GPUs total
                "container_spec": {
                    "image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3:latest",
                    "command": ["torchrun"],
                    "args": [
                        "--nproc_per_node=8",
                        "--nnodes=4",
                        "train_distributed.py",
                    ],
                },
            }
        ],
    },
}


# Vertex AI Pipeline (Kubeflow Pipelines v2)
vertex_pipeline = """
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics
from google_cloud_pipeline_components.v1 import vertex_notification_email

@component(base_image="python:3.11", packages_to_install=["pandas", "scikit-learn"])
def preprocess(
    input_data: Input[Dataset],
    output_train: Output[Dataset],
    output_test: Output[Dataset],
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(input_data.path)
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv(output_train.path, index=False)
    test.to_csv(output_test.path, index=False)


@component(base_image="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3:latest")
def train_model(
    train_data: Input[Dataset],
    model_output: Output[Model],
    metrics: Output[Metrics],
    epochs: int = 50,
    learning_rate: float = 0.001,
):
    # Training logic here
    pass


@component
def evaluate_model(
    model: Input[Model],
    test_data: Input[Dataset],
    metrics: Output[Metrics],
    threshold: float = 0.95,
) -> bool:
    # Evaluation logic
    accuracy = 0.97  # computed
    metrics.log_metric("accuracy", accuracy)
    return accuracy >= threshold


@pipeline(name="ml-training-pipeline")
def training_pipeline(
    input_uri: str,
    epochs: int = 50,
    accuracy_threshold: float = 0.95,
):
    preprocess_task = preprocess(input_data=input_uri)
    
    train_task = train_model(
        train_data=preprocess_task.outputs["output_train"],
        epochs=epochs,
    )
    
    eval_task = evaluate_model(
        model=train_task.outputs["model_output"],
        test_data=preprocess_task.outputs["output_test"],
        threshold=accuracy_threshold,
    )
    
    with dsl.If(eval_task.output == True):
        # Deploy model if evaluation passes
        pass
"""


# Vertex AI Gemini Integration
gemini_patterns = {
    "basic_generation": {
        "model": "gemini-2.5-pro",
        "config": {
            "temperature": 0.7,
            "max_output_tokens": 8192,
            "top_p": 0.95,
        },
    },
    
    "grounded_generation": {
        "model": "gemini-2.5-pro",
        "grounding": {
            "google_search": True,  # Ground with web search
            "vertex_ai_search": {  # Ground with enterprise docs
                "datastore": "projects/my-project/locations/global/collections/default/dataStores/my-docs",
            },
        },
    },
    
    "function_calling": {
        "model": "gemini-2.5-pro",
        "tools": [
            {
                "function_declarations": [
                    {
                        "name": "get_weather",
                        "description": "Get current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                            },
                            "required": ["city"],
                        },
                    }
                ]
            }
        ],
    },
}
```

---

## How It Works in Practice

### Large-Scale Training on TPUs

```yaml
TPU_Training_Production:
  scenario: "Pre-training a 7B parameter model on TPU v5e pod"
  
  setup:
    hardware: "TPU v5e-256 (256 chips, 32 hosts)"
    framework: "JAX + Flax (native TPU support)"
    data: "Cloud Storage bucket (100TB tokenized text)"
    config:
      precision: "BF16 (native on TPU)"
      batch_size: "2048 global (8 per chip × 256 chips)"
      sequence_length: 4096
      optimizer: "AdaFactor (memory-efficient for TPU)"
      
  cost_comparison:
    tpu_v5e_256:
      price_per_hour: "$2,048 (256 chips × $8/chip/hr)"
      training_time: "5 days"
      total_cost: "$245,760"
      
    gpu_h100_equivalent:
      hardware: "64× H100 (8 nodes × 8 GPUs)"
      price_per_hour: "$3,200 (64 × $50/GPU/hr on-demand)"
      training_time: "7 days (less interconnect bandwidth)"
      total_cost: "$537,600"
      
    savings: "54% cheaper on TPU for this workload"
    
  monitoring:
    tensorboard: "Vertex AI TensorBoard (managed, persistent)"
    metrics: "Loss, gradient norm, learning rate, throughput (tokens/sec)"
    alerts: "Cloud Monitoring → PagerDuty on training failure"
```

---

## Interview Tip

> When asked about GCP Vertex AI: "I choose Vertex AI for three main scenarios. First, TPU training — for large transformer models (>7B parameters), TPUs offer 40-60% cost savings vs. equivalent GPU setups because of superior interconnect bandwidth (ICI) and native BF16 support. I use JAX/Flax for TPU-native training with XLA compilation. Second, Gemini-based applications — Vertex AI provides enterprise-grade access to Google's frontier models with grounding (connect to your data), function calling (AI agents), and fine-tuning, all with VPC Service Controls and data residency guarantees. Third, end-to-end pipelines — Vertex AI Pipelines (Kubeflow v2) with native integration to Feature Store, Model Registry, and TensorBoard creates a cohesive MLOps experience. For comparison with SageMaker: SageMaker is stronger in broader AWS ecosystem integration and has more enterprise MLOps features. Vertex AI is stronger in: TPU access, Gemini integration, and price-performance for large-scale training. For serving: Vertex AI endpoints support auto-scaling, traffic splitting, and monitoring similar to SageMaker, but also offers Vertex AI Vector Search (ScaNN-based) for billion-scale similarity search — crucial for RAG systems. Cost optimization: preemptible VMs for training (80% savings), committed use discounts for sustained workloads, and Vertex AI Prediction auto-scaling to zero for serverless inference."

---

## Common Mistakes

1. **Using GPUs when TPUs are cheaper** — Training large transformer models on A100s when TPU v5e offers 50%+ cost savings. Solution: benchmark on TPUs for transformer architectures >1B parameters. JAX/Flax makes TPU training straightforward.

2. **Ignoring preemptible VMs for training** — Paying full on-demand price for multi-day training runs. Solution: use preemptible VMs (80% discount) with checkpointing. Vertex AI Custom Training supports automatic restart on preemption.

3. **Not using grounding for Gemini** — Using Gemini without grounding, getting hallucinations, then building complex RAG pipelines manually. Solution: Vertex AI grounding (Google Search or custom data) is built-in — reduces hallucination with minimal engineering effort.

4. **Overcomplicating pipelines** — Writing custom orchestration instead of using Vertex AI Pipelines (Kubeflow v2). Solution: use `kfp` SDK — lightweight Python components, automatic caching, artifact tracking, and native Vertex AI integration.

5. **Not leveraging Model Garden** — Training models from scratch when pre-trained alternatives exist in Model Garden. Solution: check Model Garden first — fine-tune Gemma, Llama, or domain-specific models instead of pre-training.

---

## Key Takeaways

- Vertex AI: Google's unified ML platform (training, serving, pipelines, feature store, model garden)
- TPU advantage: 40-60% cheaper than GPUs for large transformer training (native BF16, ICI interconnect)
- Gemini integration: enterprise access with grounding, function calling, fine-tuning, VPC controls
- Model Garden: 100+ pre-trained models (Google + open-source) for deployment and fine-tuning
- Pipelines: Kubeflow Pipelines v2 — Python components, caching, artifact lineage
- Vector Search: ScaNN-based billion-scale similarity search (< 10ms latency)
- Training options: GPU (A100, H100), TPU (v5e, v5p), preemptible (80% savings)
- JAX/Flax: Google's ML framework — native TPU support, XLA compilation, functional design
- Cost: preemptible VMs, committed use discounts, auto-scaling to zero
- Trade-off: TPU/GCP lock-in vs. significant cost savings for transformer workloads
