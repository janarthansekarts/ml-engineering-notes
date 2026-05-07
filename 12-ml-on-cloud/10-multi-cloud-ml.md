# Multi-Cloud ML

## The Problem / Why This Matters

Many enterprises operate across multiple cloud providers — either by strategic choice (avoiding vendor lock-in), through acquisitions (inherited infrastructure), regulatory requirements (data residency mandates), or capability gaps (GCP for TPUs, Azure for OpenAI, AWS for ecosystem breadth). Running ML across multiple clouds introduces challenges: fragmented tooling (SageMaker vs. Vertex AI vs. Azure ML have different APIs), data movement costs (egress fees for moving training data between clouds), inconsistent security models (IAM differs significantly across providers), and operational complexity (monitoring, logging, and alerting spread across platforms). In 2026, the multi-cloud ML landscape has matured with abstraction layers (ZenML, MLflow, Kubeflow), portable formats (ONNX, safetensors), and standardized serving protocols (OpenInference, KServe). But the fundamental trade-off remains: multi-cloud portability comes at the cost of losing cloud-native optimizations (SageMaker's tight S3 integration, Vertex AI's TPU access, Azure ML's OpenAI integration). Understanding multi-cloud ML means knowing: when multi-cloud is genuinely needed (vs. just desired), how to build portable pipelines without sacrificing performance, and where to accept cloud-specific optimization.

---

## The Analogy

Think of multi-cloud ML like a restaurant chain operating in multiple countries:

- **Single-cloud** = A restaurant in one country. Everything uses local suppliers, follows local regulations, and optimizes for local tastes. Maximum efficiency, but if that country bans your ingredient (vendor lock-in), you're stuck.
- **Multi-cloud** = A restaurant chain in 3 countries. Each location adapts to local suppliers (cloud services) but the recipes (ML models) and processes (MLOps pipelines) stay consistent. More complex operations, but if one country raises prices, you shift cooking to another.
- **The abstraction layer** = A standardized recipe book that works in any kitchen. The recipe says "sauté in oil" — doesn't specify whether it's olive oil (AWS S3) or sesame oil (GCS). The chef (orchestrator) picks the local equivalent.
- **The cost of portability** = You can't use the special local oven (TPU) if your recipe must work in any kitchen. You optimize for "works everywhere" rather than "works best here."

---

## Deep Dive

### Multi-Cloud ML Strategies

```yaml
Multi_Cloud_Strategies:
  strategy_1_primary_secondary:
    description: "One cloud primary (80% workloads), others for specific needs"
    example:
      primary: "AWS (most infrastructure, training, serving)"
      secondary_1: "GCP (TPU training for large transformers — 40% cheaper)"
      secondary_2: "Azure (OpenAI access — GPT-5, o3 for specific applications)"
    data_flow: "Training data replicated to secondary clouds as needed"
    tooling: "Cloud-native tools on primary, portable tools for cross-cloud"
    when_right:
      - "Need best-of-breed capabilities from different clouds"
      - "Large enough team to manage complexity"
      - "Specific cost/capability advantages justify multi-cloud"
      
  strategy_2_workload_split:
    description: "Different workloads on different clouds (permanent split)"
    example:
      training: "GCP (TPU access, better price-performance)"
      serving: "AWS (closest to application infrastructure, lower latency)"
      llm_apps: "Azure (OpenAI access, enterprise compliance)"
    connector: "Models exported in portable format (ONNX/safetensors)"
    when_right:
      - "Different workloads have different optimal platforms"
      - "Regulatory requirements mandate data in specific regions"
      - "Application infrastructure already multi-cloud"
      
  strategy_3_full_portable:
    description: "Cloud-agnostic stack that runs anywhere"
    stack:
      orchestration: "Kubeflow Pipelines or Argo Workflows on K8s"
      training: "Ray Train (runs on any cloud K8s)"
      serving: "KServe (Kubernetes-native model serving)"
      experiments: "MLflow (self-hosted, portable)"
      features: "Feast (open-source, multi-cloud)"
    when_right:
      - "Strong multi-cloud mandate from leadership"
      - "Large platform team that can maintain K8s on multiple clouds"
      - "Exit strategy from current cloud is required"
      - "Government/regulated workloads requiring true cloud independence"
    trade_offs:
      - "Lose cloud-native optimizations (SageMaker endpoints, Vertex TPU)"
      - "Higher operational complexity (K8s on 2-3 clouds)"
      - "Can't use proprietary services easily (need adapters)"
      
  anti_pattern_accidental_multi_cloud:
    description: "Ended up on multiple clouds without strategy"
    symptoms:
      - "Different teams chose different clouds independently"
      - "No shared tooling or standards"
      - "Data duplicated everywhere with no governance"
      - "3× operational complexity with no benefit"
    solution: "Consolidate to primary-secondary or establish standards"
```

### Portable ML Stack

```yaml
Portable_ML_Stack:
  abstraction_layers:
    pipeline_orchestration:
      zenml:
        what: "ML pipeline framework with pluggable orchestrators"
        supports: "Kubeflow, SageMaker, Vertex AI, Airflow, local"
        approach: "Write pipeline once, deploy on any orchestrator"
        limitation: "Abstraction adds complexity, may not expose all features"
        
      metaflow:
        what: "Netflix's ML infrastructure framework"
        supports: "AWS (native), Kubernetes, local"
        approach: "Pythonic pipeline definition with step decorators"
        limitation: "Strongest on AWS, K8s support is newer"
        
      kubeflow_pipelines:
        what: "Open-source ML pipelines on Kubernetes"
        supports: "Any K8s (EKS, GKE, AKS, bare metal)"
        approach: "Component-based pipelines, container-native"
        advantage: "True portability (K8s is the abstraction layer)"
        
    experiment_tracking:
      mlflow:
        what: "Open-source experiment tracking and model registry"
        deployment: "Self-hosted on any cloud K8s"
        portability: "No cloud dependency (S3/GCS/ADLS for artifact storage)"
        integration: "Works with all training frameworks"
        
    model_serving:
      kserve:
        what: "Kubernetes-native model serving (formerly KFServing)"
        supports: "Any K8s cluster"
        features: "Auto-scaling, canary, A/B testing, monitoring"
        inference_protocols: "V2 inference protocol (standardized REST/gRPC)"
        
      ray_serve:
        what: "Scalable model serving on Ray clusters"
        supports: "Any infrastructure where Ray runs"
        advantage: "Dynamic batching, model composition, multi-model"
        
    model_format:
      onnx:
        what: "Open Neural Network Exchange — portable model format"
        converts_from: "PyTorch, TensorFlow, scikit-learn, XGBoost"
        runs_on: "ONNX Runtime (CPU, GPU, all clouds)"
        advantage: "Write model once, optimize for any hardware"
        limitation: "Not all operations supported, complex models may not convert"
        
      safetensors:
        what: "Safe, fast model weight format (HuggingFace)"
        advantage: "No pickle (secure), fast loading, mmap support"
        portable: "Framework-agnostic (works with PyTorch, TF, JAX)"
        
    feature_store:
      feast:
        what: "Open-source feature store"
        online: "Redis, DynamoDB, BigQuery, or SQLite"
        offline: "S3, GCS, BigQuery, Redshift, Snowflake"
        portability: "Swap backends without changing feature definitions"
```

### Implementation Patterns

```python
# Multi-cloud ML implementation patterns

"""
Patterns for building ML systems that work across cloud providers.
Focus: portable pipelines, model formats, and cross-cloud deployment.
"""

multi_cloud_patterns = {
    "portable_training": {
        "principle": "Separate model code from infrastructure code",
        "implementation": {
            "model_code": {
                "location": "Git repository (shared across environments)",
                "requirements": [
                    "No cloud SDK imports in training logic",
                    "Configuration via environment variables or config files",
                    "Standard model checkpointing interface",
                    "Standard metrics logging interface",
                ],
                "example": """
# GOOD: Cloud-agnostic training code
import torch
import os

def train(config):
    model = create_model(config['model_type'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    
    # Data path from config (not hardcoded S3/GCS)
    data_path = os.environ['TRAINING_DATA_PATH']
    
    # Checkpoint path from config
    checkpoint_dir = os.environ['CHECKPOINT_DIR']
    
    for epoch in range(config['epochs']):
        # Training loop (pure PyTorch)
        ...
        # Save checkpoint to configured path
        torch.save(model.state_dict(), f"{checkpoint_dir}/epoch_{epoch}.pt")
""",
            },
            "infrastructure_code": {
                "per_cloud": True,
                "aws": "SageMaker Estimator wrapping the container",
                "gcp": "Vertex AI CustomJob with same container",
                "k8s": "Ray Train job spec referencing same container",
            },
        },
    },
    
    "cross_cloud_model_deployment": {
        "pattern": "Train on one cloud, serve on another",
        "flow": [
            "1. Train model on GCP (TPU) — save as safetensors",
            "2. Export to ONNX for optimized inference",
            "3. Store in portable model registry (MLflow)",
            "4. Deploy to AWS (SageMaker endpoint) or Azure (Azure ML endpoint)",
            "5. Inference uses ONNX Runtime (optimized for target hardware)",
        ],
        "model_format_choices": {
            "for_pytorch_serving": "safetensors (fast loading, secure)",
            "for_optimized_inference": "ONNX (hardware optimization via ONNX Runtime)",
            "for_edge_deployment": "TFLite, CoreML, or ONNX depending on target",
        },
        "data_sync": {
            "training_data": "Replicate to training cloud before job starts",
            "model_artifacts": "Push to serving cloud after training completes",
            "cost_consideration": "Egress fees: $0.08-0.12/GB between clouds",
        },
    },
    
    "unified_mlops": {
        "principle": "Single MLOps interface regardless of underlying cloud",
        "components": {
            "experiment_tracking": {
                "tool": "MLflow (self-hosted on Kubernetes)",
                "artifact_store": "Configured per cloud (S3 or GCS backend)",
                "unified": "Same MLflow UI/API regardless of where training runs",
            },
            "model_registry": {
                "tool": "MLflow Model Registry or custom",
                "requirements": [
                    "Single source of truth for model versions",
                    "Cloud-agnostic model lineage (data → training → model)",
                    "Deployment targets configurable per model",
                ],
            },
            "pipeline_orchestration": {
                "tool": "Kubeflow Pipelines (runs on K8s on any cloud)",
                "alternative": "ZenML with cloud-specific orchestrator plugins",
                "components": "Standardized containers (run identically anywhere)",
            },
            "monitoring": {
                "tool": "Prometheus + Grafana (self-hosted) or Datadog (SaaS)",
                "metrics": "OpenTelemetry standard (portable across observability tools)",
                "alerts": "PagerDuty or OpsGenie (cloud-agnostic)",
            },
        },
    },
    
    "vendor_lock_in_assessment": {
        "description": "Evaluate lock-in risk of cloud ML services",
        "high_lock_in": [
            "SageMaker Feature Store (proprietary API, no export)",
            "Vertex AI Pipelines (tightly coupled to GCP services)",
            "Azure ML Responsible AI (no equivalent elsewhere)",
            "Azure OpenAI (specific models only on Azure)",
            "GCP TPU (only available on GCP)",
        ],
        "medium_lock_in": [
            "SageMaker Training (can use custom containers — portable with effort)",
            "SageMaker Endpoints (standard container interface but deployment is proprietary)",
            "Vertex AI Endpoints (similar — portable model, proprietary deployment)",
        ],
        "low_lock_in": [
            "S3/GCS for data storage (standard APIs, easy to replicate)",
            "Container registries (ECR/GCR/ACR — push same image anywhere)",
            "Kubernetes services (EKS/GKE/AKS — nearly identical)",
            "MLflow on any cloud (fully portable)",
        ],
        "recommendation": "Accept high lock-in only for capabilities unavailable elsewhere (TPU, Azure OpenAI). Use portable alternatives for everything else.",
    },
}


# Data movement strategy
data_movement = {
    "egress_costs": {
        "aws": "$0.09/GB (to internet or other clouds)",
        "gcp": "$0.08-0.12/GB (tiered by volume)",
        "azure": "$0.087/GB (first 10TB/month)",
        "impact": "Moving 10 TB training data: $800-$1,200 per transfer",
    },
    
    "strategies": {
        "minimize_movement": {
            "principle": "Keep data close to compute",
            "implementation": [
                "Train where the data lives (don't move 10TB for one training run)",
                "Replicate only when establishing long-term multi-cloud",
                "Use feature stores with materialization (compute features locally)",
            ],
        },
        "one_time_replication": {
            "when": "Establishing new cloud presence",
            "method": "Bulk transfer (AWS DataSync, GCP Transfer Service)",
            "cost": "Pay egress once, then keep data synced incrementally",
        },
        "incremental_sync": {
            "what": "Only sync new/changed data daily",
            "tools": ["rclone", "Cloud-native transfer services", "Custom sync jobs"],
            "cost": "Much lower than full replication (delta only)",
        },
        "export_models_not_data": {
            "principle": "Move small model artifacts ($0.01) not large datasets ($1,000)",
            "flow": "Train on data cloud → export model (GB) → deploy on serving cloud",
        },
    },
}
```

---

## How It Works in Practice

### Multi-Cloud ML Architecture

```yaml
Multi_Cloud_ML_Architecture:
  scenario: "Enterprise with AWS primary, GCP for large training, Azure for LLM apps"
  
  architecture:
    aws_primary:
      role: "Main infrastructure — data lake, serving, MLOps"
      services:
        data: "S3 data lake (100+ TB training data)"
        serving: "SageMaker Endpoints (production inference)"
        mlops: "MLflow on EKS (experiment tracking)"
        pipelines: "Kubeflow on EKS (orchestration)"
        monitoring: "Prometheus + Grafana on EKS"
        
    gcp_training:
      role: "Large model training (TPU for transformers)"
      services:
        compute: "TPU v5e pods (pre-training and fine-tuning)"
        data: "GCS (training data replicated from AWS S3)"
        output: "Model artifacts pushed back to MLflow on AWS"
      data_sync: "Daily incremental from S3 → GCS (via rclone)"
      cost_benefit: "TPU v5e is 40% cheaper than H100 for transformer training"
      
    azure_llm:
      role: "LLM-powered applications (Azure OpenAI)"
      services:
        model: "Azure OpenAI (GPT-5, o3 — provisioned throughput)"
        orchestration: "Azure ML Prompt Flow"
        data: "Azure AI Search (RAG knowledge base)"
      integration: "Azure OpenAI API called from AWS-hosted applications"
      
  cross_cloud_glue:
    model_registry: "MLflow on AWS EKS (single source of truth)"
    model_format: "safetensors for storage, ONNX for optimized serving"
    networking: "VPN tunnels between AWS VPC and GCP VPC"
    identity: "Federated — AWS IAM trusts GCP service accounts for specific actions"
    secrets: "HashiCorp Vault (cloud-agnostic secret management)"
    
  data_flow:
    training_data:
      source: "AWS S3 (primary data lake)"
      sync_to_gcp: "Daily incremental (new data only, ~100 GB/day)"
      egress_cost: "$9/day for 100 GB sync"
      
    model_artifacts:
      trained_on: "GCP (TPU training output — safetensors)"
      stored_in: "AWS S3 via MLflow (model registry)"
      deployed_to: "SageMaker Endpoints (after conversion to ONNX)"
      artifact_size: "~14 GB (7B model) — one-time egress $1.26"
      
    llm_inference:
      application: "AWS-hosted microservice"
      calls: "Azure OpenAI API (HTTPS, pay-per-token)"
      no_data_movement: "Just API calls, no bulk transfer"
      
  cost:
    aws_infra: "$35,000/month (serving, MLOps, data lake)"
    gcp_training: "$15,000/month (TPU pods, spot pricing)"
    azure_openai: "$8,000/month (provisioned throughput for GPT-5)"
    data_sync: "$300/month (egress costs)"
    networking: "$500/month (VPN tunnels)"
    total: "$58,800/month"
    
    vs_all_aws: "$75,000/month (GPU training instead of TPU, no Azure OpenAI access)"
    savings: "22% savings + access to TPU and Azure OpenAI capabilities"
```

---

## Interview Tip

> When asked about multi-cloud ML: "I approach multi-cloud pragmatically — it should serve specific business or technical needs, not be ideology. Three valid reasons: (1) best-of-breed capabilities (GCP TPUs for large transformer training at 40% savings, Azure for OpenAI access, AWS for broadest ecosystem), (2) vendor negotiation leverage (credible exit option reduces pricing), and (3) regulatory requirements (data must stay in specific regions/providers). My architecture pattern: portable core (model code in containers, MLflow for tracking, Kubeflow/ZenML for pipelines, safetensors/ONNX for models) with strategic cloud-native services where the benefit is clear (TPUs on GCP, SageMaker endpoints for serving on AWS, Azure OpenAI for frontier LLMs). Key principles: separate model code from infrastructure code (same training script runs anywhere), minimize data movement (train where data lives — egress costs add up at $0.09/GB), and single model registry (MLflow as source of truth across clouds). Anti-pattern to avoid: accidental multi-cloud where different teams chose different clouds without coordination — creates 3× operational complexity with zero benefit. The sweet spot is usually primary-secondary: 80% on one cloud, 20% on another for specific capabilities."

---

## Common Mistakes

1. **Multi-cloud for the sake of multi-cloud** — Splitting workloads across clouds without clear benefit. Triple the operational complexity, triple the security surface, and no cost savings. Solution: have explicit justification per cloud (TPU access, specific model access, data residency). If none exists, consolidate.

2. **Moving large datasets repeatedly** — Syncing 50 TB training dataset between clouds for every training run. Egress costs: $4,500 per transfer. Solution: replicate data once to training cloud, keep synced incrementally. Better: train where the data lives.

3. **No portable model format** — Training on GCP with TensorFlow SavedModel, serving on AWS with PyTorch. Requires model conversion every time with potential accuracy loss. Solution: use ONNX for inference (converts from any framework) or safetensors (portable weight format).

4. **Cloud-specific code everywhere** — SageMaker SDK calls embedded throughout training scripts. Migration requires rewriting everything. Solution: containerize training code with no cloud SDK in the training logic. Infrastructure layer wraps containers for each cloud.

5. **No unified monitoring** — Different monitoring tools on each cloud (CloudWatch, Cloud Monitoring, Azure Monitor). No single view of ML system health. Solution: OpenTelemetry for metrics/traces (cloud-agnostic), Datadog or self-hosted Grafana for unified dashboards.

---

## Key Takeaways

- Multi-cloud needs justification: capabilities, cost, or regulation — not ideology
- Primary-secondary pattern: 80% on primary cloud, specific workloads on others
- Portable core: containers, MLflow, Kubeflow, ONNX/safetensors — cloud-agnostic
- Cloud-native where justified: TPUs (GCP only), Azure OpenAI (Azure only), SageMaker (AWS)
- Minimize data movement: egress costs $0.08-0.12/GB — train where data lives
- Model portability: safetensors for storage, ONNX for optimized serving
- Separate model from infrastructure: training code in containers, no cloud SDK in logic
- Unified observability: OpenTelemetry + Grafana/Datadog across all clouds
- Lock-in assessment: accept high lock-in only for irreplaceable capabilities
- Anti-pattern: accidental multi-cloud = 3× complexity, 0× benefit
