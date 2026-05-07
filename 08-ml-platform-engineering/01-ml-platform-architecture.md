# ML Platform Architecture

## The Problem / Why This Matters

When an organization scales beyond 5-10 ML models, individual teams building bespoke infrastructure for each model becomes unsustainable. Team A builds their own feature pipeline in Spark, Team B writes theirs in Pandas, Team C uses a different model registry — nobody can reuse work, every deployment is a snowflake, and the ML platform team spends all their time firefighting. An ML platform provides self-service infrastructure that enables data scientists and ML engineers to develop, train, deploy, and monitor models without building everything from scratch. The platform abstracts away infrastructure complexity (GPU scheduling, container orchestration, model serving) and provides standardized interfaces (APIs, SDKs, CLIs) for common ML workflows. In 2026, ML platforms have evolved from "just a model registry" to comprehensive internal developer platforms (IDPs) for ML — encompassing experiment tracking, feature stores, training infrastructure, model serving, monitoring, governance, and increasingly LLMOps (Large Language Model Operations) capabilities. The engineering challenge: building a platform that's opinionated enough to provide value (golden paths, templates, defaults) but flexible enough to not constrain innovation (custom models, novel architectures, research workflows).

---

## The Analogy

Think of an ML platform like an airport:

- **Without a platform** = Every airline builds its own runway, terminal, and air traffic control. Incredibly expensive, inconsistent safety standards, no shared infrastructure. Works for 1-2 airlines, collapses at scale.
- **With a platform** = Shared airport infrastructure. Airlines (ML teams) focus on their flights (models). The airport provides: runways (compute), terminals (deployment), air traffic control (orchestration), security (governance), baggage handling (data pipelines), and boarding gates (serving endpoints).

Airlines don't need to understand how runway concrete is poured. They need a gate, a runway slot, and fuel. Similarly, data scientists shouldn't need to understand Kubernetes — they need to train models, deploy endpoints, and monitor performance.

---

## Deep Dive

### Platform Architecture Layers

```yaml
Platform_Layers:
  layer_1_infrastructure:
    what: "Raw compute, storage, networking"
    components:
      - "GPU clusters (NVIDIA H100, H200, B200)"
      - "CPU pools (training preprocessing, serving)"
      - "Object storage (S3, GCS, ADLS — for datasets, artifacts)"
      - "Container orchestration (Kubernetes with GPU operators)"
      - "Networking (high-bandwidth for distributed training)"
    managed_by: "Platform/infra team"
    abstraction: "Users never interact with this directly"
    
  layer_2_core_services:
    what: "ML-specific services that multiple teams share"
    components:
      experiment_tracking:
        what: "Track experiments, metrics, hyperparameters, artifacts"
        tools: "MLflow, Weights & Biases, Neptune"
        
      feature_store:
        what: "Central repository for computed features"
        tools: "Feast, Tecton, custom on Redis/BigQuery"
        
      model_registry:
        what: "Version, stage, and manage model artifacts"
        tools: "MLflow Model Registry, custom on S3 + metadata DB"
        
      training_service:
        what: "Submit and manage training jobs"
        interface: "API/SDK: submit(config) → job_id"
        backend: "Kubernetes Jobs, Ray Train, or managed (SageMaker/Vertex)"
        
      serving_service:
        what: "Deploy models as endpoints"
        interface: "API/SDK: deploy(model, config) → endpoint_url"
        backend: "KServe, Seldon, TorchServe, vLLM, or managed"
        
      monitoring_service:
        what: "Monitor deployed models"
        interface: "Automatic (attach to deployed models) + custom metrics"
        backend: "Evidently, Arize, custom Prometheus metrics"
        
    managed_by: "ML platform team"
    abstraction: "Users interact via SDK/CLI/UI (not raw infrastructure)"
    
  layer_3_developer_experience:
    what: "How ML practitioners interact with the platform"
    components:
      sdk:
        what: "Python SDK for programmatic access"
        example: "platform.train(model_config) → returns job handle"
        
      cli:
        what: "Command-line tool for common operations"
        example: "mlplatform deploy --model my-model:v3 --replicas 2"
        
      web_ui:
        what: "Visual interface for monitoring, experiments, registry"
        example: "Dashboard showing all deployed models, their health, experiment results"
        
      notebooks:
        what: "Managed notebook environment (JupyterHub or VS Code)"
        integration: "Pre-configured with SDK, feature store access, GPU attach"
        
      templates:
        what: "Project scaffolding (cookiecutter/copier templates)"
        provides: "Standard project structure, CI/CD, Dockerfile, config"
        
    managed_by: "ML platform team (with input from users)"
    
  layer_4_governance:
    what: "Controls, policies, and compliance"
    components:
      - "Model approval workflows (staging → production gate)"
      - "Access control (who can deploy to production?)"
      - "Audit trail (who trained what, when, with which data)"
      - "Cost attribution (chargeback to teams)"
      - "Compliance checks (bias, fairness, privacy)"
    managed_by: "ML platform team + compliance/legal"
```

### Platform Team Structure

```yaml
Team_Structure:
  ml_platform_team:
    size: "5-15 engineers (depends on org size)"
    composition:
      platform_engineers: "60% — Build and maintain platform services"
      sre_engineers: "20% — Reliability, on-call, incident response"
      developer_experience: "20% — SDK, docs, onboarding, templates"
      
    responsibilities:
      - "Build and maintain core ML services (training, serving, monitoring)"
      - "Define and enforce standards (model packaging, API contracts)"
      - "On-call for platform issues (not individual model issues)"
      - "Developer experience (SDK quality, documentation, onboarding)"
      - "Cost optimization (spot instances, autoscaling, right-sizing)"
      - "Security and compliance (access control, audit, encryption)"
      
    does_not_do:
      - "Train or tune individual models (ML teams do this)"
      - "Own model accuracy (ML teams own their model performance)"
      - "Write feature engineering code (ML teams define features)"
      - "Make business decisions about which models to build"
      
  ml_teams_using_platform:
    interaction: "Self-service via SDK/CLI/UI"
    responsibilities:
      - "Define and train their models"
      - "Define features (platform stores and serves them)"
      - "Monitor their model's performance (platform provides tools)"
      - "Set retraining schedules and triggers"
      - "Respond to their model's alerts"
      
  ratio: "1 platform engineer : 5-10 ML practitioners supported"
```

### Self-Service Design Principles

```yaml
Design_Principles:
  golden_paths:
    what: "Opinionated, well-paved paths for common workflows"
    examples:
      - "Standard training pipeline: submit config → train → validate → register"
      - "Standard deployment: register → deploy staging → test → promote production"
      - "Standard monitoring: deploy → automatic drift monitoring + alerting"
    philosophy: "Make the right thing easy. Don't force people onto the path, but make it the easiest option."
    
  progressive_disclosure:
    what: "Simple by default, advanced when needed"
    example:
      simple: "platform.deploy(model='my-model:v3')  # Uses all defaults"
      advanced: |
        platform.deploy(
            model='my-model:v3',
            replicas=3,
            gpu='H100',
            autoscale=True,
            canary_percent=10,
            rollback_threshold=0.95,
        )
    principle: "2 parameters to get started, 20 parameters for full control"
    
  escape_hatches:
    what: "Allow power users to bypass platform abstractions when needed"
    examples:
      - "Custom Dockerfile (instead of platform-managed container)"
      - "Raw Kubernetes access (for novel workloads)"
      - "Custom serving runtime (for non-standard model types)"
    guard: "Escape hatches get less platform support (user owns ops)"
    
  consistency:
    what: "Same patterns everywhere"
    examples:
      - "All models deployed the same way (regardless of framework)"
      - "All experiments tracked in same system (regardless of team)"
      - "All features served through same store (regardless of who created them)"
      - "Same monitoring for all models (consistent alerting, dashboards)"
```

### Platform Components Integration

```python
# ML Platform SDK example — how users interact with the platform

"""
This is what the ML platform team provides to ML practitioners.
Clean, simple SDK that abstracts infrastructure complexity.
"""


class MLPlatform:
    """
    Main entry point for ML platform interaction.
    Data scientists use this SDK for all ML workflows.
    """
    
    def __init__(self, project: str):
        self.project = project
        self.training = TrainingService()
        self.serving = ServingService()
        self.features = FeatureService()
        self.registry = ModelRegistry()
        self.monitoring = MonitoringService()
    
    # --- Training ---
    def train(
        self,
        script: str,
        config: dict,
        gpu: str = "H100",
        num_gpus: int = 1,
        framework: str = "pytorch",
    ) -> "TrainingJob":
        """
        Submit a training job.
        
        Example:
            job = platform.train(
                script="train.py",
                config={"lr": 0.001, "epochs": 50},
                gpu="H100",
                num_gpus=4,
            )
            job.wait()  # Block until complete
            print(job.metrics)  # {"accuracy": 0.95, "loss": 0.12}
        """
        return self.training.submit(
            project=self.project,
            script=script,
            config=config,
            resources={"gpu": gpu, "num_gpus": num_gpus},
            framework=framework,
        )
    
    # --- Model Registry ---
    def register_model(
        self,
        model_path: str,
        name: str,
        metrics: dict,
        tags: dict = None,
    ) -> "RegisteredModel":
        """
        Register a trained model artifact.
        
        Example:
            model = platform.register_model(
                model_path="./outputs/model.pt",
                name="churn-predictor",
                metrics={"accuracy": 0.95, "f1": 0.91},
                tags={"team": "retention", "framework": "pytorch"},
            )
            print(model.version)  # "v7"
        """
        return self.registry.register(
            project=self.project,
            model_path=model_path,
            name=name,
            metrics=metrics,
            tags=tags or {},
        )
    
    # --- Serving ---
    def deploy(
        self,
        model: str,  # "model-name:version" or "model-name:latest"
        replicas: int = 2,
        gpu: str = None,
        autoscale: bool = True,
    ) -> "Endpoint":
        """
        Deploy a registered model as an endpoint.
        
        Example:
            endpoint = platform.deploy(
                model="churn-predictor:v7",
                replicas=2,
                autoscale=True,
            )
            print(endpoint.url)  # "https://serving.internal/churn-predictor/v7"
            
            # Make predictions
            result = endpoint.predict({"features": [1, 2, 3]})
        """
        return self.serving.deploy(
            project=self.project,
            model=model,
            replicas=replicas,
            gpu=gpu,
            autoscale=autoscale,
        )
    
    # --- Feature Store ---
    def get_features(
        self,
        entity_ids: list,
        feature_set: str,
    ) -> "DataFrame":
        """
        Retrieve features for given entities.
        
        Example:
            features = platform.get_features(
                entity_ids=["user_123", "user_456"],
                feature_set="user-engagement-features",
            )
        """
        return self.features.get_online(
            entity_ids=entity_ids,
            feature_set=feature_set,
        )
```

### Platform Maturity Model

```yaml
Maturity_Levels:
  level_1_manual:
    description: "Ad-hoc ML development, no shared infrastructure"
    characteristics:
      - "Each team manages their own infrastructure"
      - "Models deployed via SSH or manual scripts"
      - "No experiment tracking (or per-team tools)"
      - "No model registry"
    teams_supported: "1-3 ML practitioners"
    
  level_2_standardized:
    description: "Shared tools but limited automation"
    characteristics:
      - "Central experiment tracker (MLflow)"
      - "Shared model registry"
      - "Standardized deployment process (but manual)"
      - "Basic monitoring (availability only)"
    teams_supported: "5-15 ML practitioners"
    
  level_3_automated:
    description: "Self-service platform with automation"
    characteristics:
      - "SDK/CLI for common operations"
      - "Automated deployment pipelines"
      - "Feature store for shared features"
      - "Automated monitoring and alerting"
      - "CI/CD for model training"
    teams_supported: "15-50 ML practitioners"
    
  level_4_optimized:
    description: "Advanced platform with optimization"
    characteristics:
      - "Cost optimization (spot, autoscaling, right-sizing)"
      - "A/B testing infrastructure"
      - "Advanced governance (approval workflows, compliance)"
      - "Multi-model pipelines and orchestration"
      - "LLMOps capabilities (prompt management, evaluation)"
    teams_supported: "50-200+ ML practitioners"
    
  level_5_intelligent:
    description: "Self-optimizing platform"
    characteristics:
      - "AutoML and neural architecture search"
      - "Intelligent resource allocation"
      - "Automated model selection and routing"
      - "Self-healing (auto-rollback, auto-scale)"
      - "AI-assisted development (copilot for ML)"
    teams_supported: "200+ ML practitioners"
```

---

## How It Works in Practice

### Real-World Platform Architectures

```yaml
Example_Architectures:
  uber_michelangelo:
    training: "Custom on Apache Spark + Horovod"
    serving: "Custom serving layer (PyML)"
    features: "Custom feature store (Palette)"
    lessons: "Build custom at extreme scale, standardize model packaging"
    
  spotify_hendrix:
    training: "Kubeflow + custom orchestration"
    serving: "KServe + custom model server"
    features: "Custom feature store"
    lessons: "Kubernetes-native, focus on developer experience"
    
  netflix:
    training: "Custom Metaflow + AWS Batch/SageMaker"
    serving: "Custom serving (JVM-based)"
    features: "Custom feature store"
    lessons: "DAG-based ML workflow, strong notebook integration"
    
  typical_startup_2026:
    training: "Ray Train on Kubernetes (or managed: SageMaker/Vertex)"
    serving: "vLLM for LLMs + KServe for traditional ML"
    features: "Feast or Tecton"
    experiment_tracking: "Weights & Biases"
    orchestration: "Dagster or Prefect"
    monitoring: "Arize or Evidently + Grafana"
```

---

## Interview Tip

> When asked about ML platform architecture: "An ML platform has three main layers: (1) Infrastructure layer — GPU clusters on Kubernetes with scheduling, object storage for datasets and artifacts, high-bandwidth networking for distributed training. Users never interact with this directly. (2) Core services layer — experiment tracking (MLflow/W&B), feature store (Feast/Tecton), model registry, training service (submit jobs via API), serving service (deploy via API), and monitoring. These are the building blocks teams use. (3) Developer experience layer — Python SDK, CLI, web UI, notebook environments, and project templates. This is how ML practitioners actually interact with the platform. Design principles: golden paths (opinionated defaults for common workflows), progressive disclosure (2 params to start, 20 for full control), and escape hatches (power users can bypass abstractions). The key metric: 'time from trained model to production endpoint.' For a mature platform, this should be minutes (deploy command + validation), not weeks (custom infra work). Anti-pattern: building a platform nobody uses. Solution: start with the highest-pain workflow (usually deployment), solve it well, then expand. Platform team ratio: 1 platform engineer per 5-10 ML practitioners."

---

## Common Mistakes

1. **Building the platform before understanding users** — Spending 6 months building a feature store when the real bottleneck is deployment. Nobody uses the feature store. Solution: interview ML teams first. Find their #1 pain point. Build that first, then expand.

2. **Too many escape hatches** — Platform allows teams to bypass everything, so nobody uses the standardized paths. You end up with 10 different deployment methods, no consistency. Solution: make the golden path so good that teams WANT to use it. Limit escape hatches to genuine edge cases, not convenience.

3. **Platform team owns model performance** — Platform is blamed when a model's accuracy degrades. This is wrong — platform provides tools, ML teams own their model's quality. Solution: clear ownership boundary. Platform owns infrastructure reliability (uptime, latency SLAs). ML teams own model quality (accuracy, fairness, relevance).

4. **Over-engineering for future scale** — Building for 1000 ML practitioners when you have 10. Result: complex system that's hard to use and maintain, massive engineering investment with no users. Solution: build for 2-3× current scale, not 100×. You can always re-architect; you can't get back the 12 months of over-engineering.

5. **No documentation or onboarding** — Beautiful platform, zero adoption because nobody knows how to use it. "Just look at the code" doesn't work for data scientists. Solution: invest 20% of platform team time in documentation, tutorials, onboarding guides, and office hours.

---

## Key Takeaways

- ML platform = self-service infrastructure for ML workflows (train, deploy, monitor, govern)
- Three layers: infrastructure (compute/storage), core services (tracking/registry/serving), developer experience (SDK/CLI/UI)
- Design principles: golden paths (easy defaults), progressive disclosure (simple → advanced), escape hatches (flexibility)
- Platform team ratio: 1 platform engineer per 5-10 ML practitioners
- Start with highest pain point (usually deployment), solve it well, then expand
- Platform owns infra reliability; ML teams own model quality — clear boundary
- Maturity levels: manual → standardized → automated → optimized → intelligent
- Don't over-engineer: build for 2-3× current scale, not 100×
- Developer experience matters: invest 20% in docs, tutorials, onboarding
- 2026: LLMOps is now a first-class platform capability (prompt management, eval, agent hosting)
