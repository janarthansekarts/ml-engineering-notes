# Notebook Infrastructure

## The Problem / Why This Matters

Notebooks are where data scientists spend 60-80% of their time — exploring data, prototyping models, running experiments, and debugging. Yet notebook infrastructure is often an afterthought: data scientists run Jupyter locally on their laptops (no GPUs, limited RAM), share notebooks via email or Slack (no versioning), install conflicting packages (works on my machine), and have no access to production data or feature stores. A well-designed notebook infrastructure provides: managed environments with GPU access, pre-configured with organization's ML stack, connected to data sources and feature stores, with proper version control and collaboration. In 2026, notebook infrastructure has evolved beyond JupyterHub to include VS Code Server (remote development), cloud-managed notebooks (SageMaker Studio, Vertex AI Workbench, Azure ML Notebooks), collaborative editing (like Google Colab), and integration with experiment tracking and pipeline tools. The engineering challenge: providing a productive, consistent environment that data scientists love using while maintaining security (data access controls, network isolation), cost efficiency (GPU idle management), and reproducibility (environment pinning, artifact tracking).

---

## The Analogy

Think of notebook infrastructure like a well-equipped laboratory:

- **Local Jupyter** = Working in your kitchen. Fine for simple experiments, but no proper equipment, limited space, safety hazards, can't share with colleagues.
- **Managed notebooks** = Professional laboratory. Proper equipment (GPUs), safety controls (access management), shared resources (data connections), ventilation (isolation), and a lab notebook system (versioning).
- **Cloud notebooks** = Shared lab facility. Walk in, equipment is ready, all chemicals (data) available, proper disposal (cleanup), colleagues can observe and collaborate.

Scientists do their best work in well-equipped labs — not in kitchens. Same for data scientists.

---

## Deep Dive

### Notebook Platform Options

```yaml
Platform_Options:
  self_managed:
    jupyterhub:
      what: "Multi-user Jupyter server on Kubernetes"
      deployment: "Helm chart (Zero-to-JupyterHub)"
      features:
        - "Per-user notebook servers (isolated pods)"
        - "GPU access (request GPU in spawner profile)"
        - "Persistent storage (user home directories)"
        - "Customizable images (team-specific environments)"
        - "Idle culling (shut down idle notebooks)"
      scaling: "Kubernetes-based — scales to 100s of users"
      effort: "Medium — requires Kubernetes ops knowledge"
      
    code_server:
      what: "VS Code in browser (via code-server or VS Code Server)"
      advantage: "Full IDE experience (extensions, debugging, terminal)"
      deployment: "Kubernetes pod per user (similar to JupyterHub)"
      use_case: "Engineers who prefer IDE over notebook interface"
      
  cloud_managed:
    sagemaker_studio:
      what: "AWS managed notebook/IDE environment"
      features:
        - "JupyterLab or VS Code interface"
        - "Instant GPU/CPU switching"
        - "Built-in experiment tracking"
        - "Direct SageMaker integration (training, deployment)"
        - "Shared spaces for collaboration"
      pricing: "Per-hour compute (ml.g5.xlarge, ml.p4d.24xlarge, etc.)"
      
    vertex_ai_workbench:
      what: "GCP managed notebook environment"
      features:
        - "Managed JupyterLab instances"
        - "Pre-configured with GCP ML stack"
        - "BigQuery integration"
        - "Git integration"
        - "Scheduled execution"
      pricing: "Per-hour compute + persistent disk"
      
    azure_ml_notebooks:
      what: "Azure ML compute instances with notebook interface"
      features:
        - "JupyterLab or VS Code"
        - "Integrated with Azure ML workspace"
        - "Compute instance management"
        - "Collaboration via shared workspaces"
      pricing: "Per-hour compute instance"
      
    databricks:
      what: "Collaborative notebooks with Spark integration"
      features:
        - "Real-time collaboration (like Google Docs)"
        - "Native Spark/MLflow integration"
        - "MLflow experiment tracking built-in"
        - "Unity Catalog for data governance"
        - "Serverless compute"
      pricing: "DBU (Databricks Units) based"
      
  hybrid:
    approach: "Self-managed JupyterHub with cloud compute backends"
    benefit: "Custom UX + cloud GPU elasticity"
    example: "JupyterHub on-prem → spawns notebook pods on cloud GPU nodes"
```

### Architecture Design

```yaml
Architecture:
  hub:
    what: "Central management plane"
    responsibilities:
      - "User authentication (OIDC/SAML via Keycloak or cloud IAM)"
      - "Environment selection (which image/GPU/memory)"
      - "Resource allocation (spawn notebook pod with requested resources)"
      - "Idle management (track last activity, cull after timeout)"
      - "Access control (who can access which environments)"
      
  notebook_servers:
    what: "Per-user compute pods"
    isolation: "Each user gets their own Kubernetes pod (namespace isolation)"
    resources:
      cpu_only: "4-16 CPU cores, 16-64 GB RAM (exploration, small experiments)"
      gpu_small: "1× L40S or T4, 8 CPU cores, 32 GB RAM (prototyping)"
      gpu_large: "1× H100, 16 CPU cores, 128 GB RAM (training)"
      gpu_multi: "4-8× H100, 64 CPU cores, 512 GB RAM (large experiments)"
    lifecycle:
      start: "User clicks 'Start' → pod created (30-120s with GPU)"
      active: "User works, periodic idle checks"
      idle: "No activity for 30 min → warning notification"
      culled: "No activity for 60 min → pod terminated (state saved to PV)"
      
  storage:
    user_home:
      what: "Persistent home directory (survives pod restarts)"
      implementation: "PVC (Persistent Volume Claim) per user — NFS or EBS"
      size: "50-200 GB per user"
      content: "Notebooks, scripts, configs, small datasets"
      
    shared_data:
      what: "Team/org shared datasets and artifacts"
      implementation: "Read-only mount from object storage (S3, GCS)"
      access: "All team members can read, curated datasets"
      
    scratch:
      what: "Fast local storage for temporary data"
      implementation: "NVMe SSD on GPU node (ephemeral — gone when pod dies)"
      use_for: "Data loading, model checkpoints, temp processing"
      
  networking:
    data_access: "VPC peering to data lake/warehouse (BigQuery, Redshift)"
    feature_store: "Direct access to feature store (Feast/Tecton)"
    model_registry: "Access to MLflow/model registry"
    internet: "Restricted (allow pip/conda, block general internet for security)"
    internal_services: "Access to internal APIs (via service mesh)"
```

### Environment Management

```python
# Notebook environment management

"""
Manages reproducible environments for notebook infrastructure.
Ensures consistency across users and between development and production.
"""

from dataclasses import dataclass


@dataclass
class NotebookEnvironment:
    """Defines a notebook environment (Docker image + config)."""
    name: str
    description: str
    base_image: str
    python_version: str
    gpu_enabled: bool
    packages: list  # pip packages
    conda_packages: list  # conda packages
    system_packages: list  # apt packages
    env_vars: dict  # environment variables
    startup_scripts: list  # scripts to run on start


# Standard environments provided by platform
ENVIRONMENTS = {
    "ml-standard": NotebookEnvironment(
        name="ML Standard",
        description="General ML development — PyTorch, scikit-learn, pandas",
        base_image="nvidia/cuda:12.6-runtime-ubuntu22.04",
        python_version="3.12",
        gpu_enabled=True,
        packages=[
            "torch==2.5.*",
            "torchvision==0.20.*",
            "scikit-learn==1.6.*",
            "pandas==2.2.*",
            "numpy==2.1.*",
            "matplotlib==3.9.*",
            "seaborn==0.13.*",
            "mlflow==2.18.*",
            "evidently==0.5.*",
            "feast==0.40.*",
            "wandb==0.19.*",
        ],
        conda_packages=[],
        system_packages=["git", "curl", "vim"],
        env_vars={
            "MLFLOW_TRACKING_URI": "http://mlflow.internal:5000",
            "FEATURE_STORE_URL": "http://feast.internal:6566",
        },
        startup_scripts=["configure-git.sh", "setup-credentials.sh"],
    ),
    
    "llm-development": NotebookEnvironment(
        name="LLM Development",
        description="LLM fine-tuning and inference — transformers, vLLM, PEFT",
        base_image="nvidia/cuda:12.6-devel-ubuntu22.04",
        python_version="3.12",
        gpu_enabled=True,
        packages=[
            "torch==2.5.*",
            "transformers==4.47.*",
            "peft==0.14.*",          # LoRA/QLoRA fine-tuning
            "trl==0.13.*",           # RLHF training
            "vllm==0.6.*",           # Fast inference
            "datasets==3.2.*",
            "accelerate==1.2.*",
            "bitsandbytes==0.45.*",  # Quantization
            "langchain==0.3.*",
            "langsmith==0.2.*",
        ],
        conda_packages=[],
        system_packages=["git", "git-lfs", "curl"],
        env_vars={
            "HF_HOME": "/data/huggingface-cache",
            "TRANSFORMERS_CACHE": "/data/huggingface-cache/transformers",
        },
        startup_scripts=["configure-hf-auth.sh"],
    ),
    
    "data-engineering": NotebookEnvironment(
        name="Data Engineering",
        description="Data processing — Spark, Polars, DuckDB",
        base_image="python:3.12-slim",
        python_version="3.12",
        gpu_enabled=False,
        packages=[
            "pyspark==3.5.*",
            "polars==1.15.*",
            "duckdb==1.1.*",
            "pandas==2.2.*",
            "pyarrow==18.*",
            "great-expectations==1.3.*",
            "dbt-core==1.9.*",
        ],
        conda_packages=[],
        system_packages=["git", "openjdk-17-jre"],
        env_vars={
            "SPARK_HOME": "/opt/spark",
            "WAREHOUSE_URL": "bigquery://project-id",
        },
        startup_scripts=[],
    ),
}


class EnvironmentManager:
    """
    Manages notebook environments.
    Provides: image building, version pinning, and custom environments.
    """
    
    def build_image(self, env: NotebookEnvironment) -> str:
        """Build Docker image for notebook environment."""
        dockerfile = self._generate_dockerfile(env)
        tag = f"notebooks/{env.name}:{self._compute_hash(env)}"
        # Build and push to registry
        return tag
    
    def create_custom_environment(
        self,
        base: str,  # Name of base environment
        additional_packages: list,
        team: str,
    ) -> NotebookEnvironment:
        """
        Allow teams to create custom environments on top of standard base.
        Inherits base packages + adds team-specific ones.
        """
        base_env = ENVIRONMENTS[base]
        custom = NotebookEnvironment(
            name=f"{base}-{team}",
            description=f"Custom environment for {team} team",
            base_image=base_env.base_image,
            python_version=base_env.python_version,
            gpu_enabled=base_env.gpu_enabled,
            packages=base_env.packages + additional_packages,
            conda_packages=base_env.conda_packages,
            system_packages=base_env.system_packages,
            env_vars=base_env.env_vars,
            startup_scripts=base_env.startup_scripts,
        )
        return custom
```

### Collaboration and Versioning

```yaml
Collaboration:
  git_integration:
    what: "Notebooks stored in Git repositories"
    tools:
      - "nbstripout: Strip output cells before commit (clean diffs)"
      - "Jupytext: Sync .ipynb with .py (review-friendly format)"
      - "nbdime: Notebook-aware diff and merge"
    workflow:
      1: "User works in notebook (JupyterLab)"
      2: "On save: auto-sync to .py via Jupytext"
      3: "On commit: strip outputs (nbstripout pre-commit hook)"
      4: "PR review: review .py file (clean, readable diff)"
      5: "On merge: CI validates notebook executes cleanly"
      
  shared_workspaces:
    what: "Multiple users collaborating on same data/project"
    implementation:
      - "Shared project directories (read-write for team members)"
      - "Shared datasets (read-only mounts)"
      - "Shared experiment tracking (same MLflow project)"
    not_recommended: "Real-time collaborative editing (Databricks has this, but conflict resolution is complex)"
    
  notebook_to_production:
    what: "Pathway from prototype notebook to production pipeline"
    approaches:
      manual_refactor:
        what: "Engineer rewrites notebook as Python modules"
        when: "Complex logic that needs proper software engineering"
        
      notebook_as_pipeline_step:
        what: "Execute notebook as a pipeline step (Papermill, Ploomber)"
        tool: "Papermill — parameterize and execute notebooks programmatically"
        when: "Notebook IS the pipeline (exploratory analysis, reports)"
        
      auto_conversion:
        what: "Convert notebook to script/module automatically"
        tools: "nbconvert, Jupytext"
        when: "Quick productionization of simple notebooks"
```

### Cost and Idle Management

```yaml
Cost_Management:
  idle_detection:
    signals:
      - "No kernel activity (no cell execution) for N minutes"
      - "No browser/UI connection for N minutes"
      - "No file saves for N minutes"
    actions:
      warning_at: "30 minutes idle → notification to user"
      hibernate_at: "60 minutes idle → save state, stop compute (keep storage)"
      terminate_at: "24 hours hibernated → terminate (keep storage)"
      
  auto_shutdown:
    gpu_notebooks: "Cull after 30 min idle (GPUs are expensive: $3-30/hour)"
    cpu_notebooks: "Cull after 2 hours idle (CPU is cheap)"
    scheduled: "All notebooks terminated at midnight unless 'keep-alive' flag set"
    
  right_sizing:
    monitoring: "Track actual CPU/GPU/memory usage per notebook"
    recommendations: "Suggest smaller instance when consistently underutilized"
    enforcement: "For GPU notebooks: alert if GPU util <10% for >30 min"
    
  cost_visibility:
    per_user: "Dashboard showing each user's compute spend"
    per_team: "Team-level aggregation"
    budget_alerts: "Notify team lead when monthly budget approaches limit"
```

---

## How It Works in Practice

### Day-in-the-Life Workflow

```yaml
Data_Scientist_Workflow:
  morning:
    1: "Open platform web UI → select 'LLM Development' environment"
    2: "Choose compute: 1× H100 GPU, 64 GB RAM"
    3: "Notebook server starts in 45 seconds (cached image)"
    4: "Pre-configured: MLflow, feature store, HuggingFace connected"
    
  working:
    5: "Pull latest notebook from Git"
    6: "Load training data from feature store (one-line SDK call)"
    7: "Fine-tune model, track with MLflow (automatic logging)"
    8: "Push results to team experiment dashboard"
    
  end_of_day:
    9: "Commit notebook to Git (outputs stripped automatically)"
    10: "Close browser tab"
    11: "After 30 min idle → GPU notebook automatically terminated"
    12: "Cost: $24 (8 hours × $3/hour for GPU instance)"
```

---

## Interview Tip

> When asked about notebook infrastructure: "My notebook platform has three key design goals: (1) Instant productivity — data scientists select an environment profile (ML Standard, LLM Development, Data Engineering), choose compute size (CPU, single GPU, multi-GPU), and have a fully-configured notebook in under 60 seconds. Pre-loaded with org SDKs, connected to feature store, experiment tracking configured. Zero setup time. (2) Cost control — GPU notebooks are expensive ($3-30/hour). Idle detection terminates notebooks with no activity after 30-60 min. Right-sizing recommendations alert when GPU utilization is below 10%. Cost dashboards make spend visible per-user and per-team. Typical savings: 40-60% vs. unmanaged notebooks. (3) Reproducibility — standardized Docker images with pinned package versions. Git integration via Jupytext (notebooks as .py files for clean diffs). nbstripout removes outputs before commit. Teams can create custom environments on top of standard bases, but the base provides consistency. The production path: notebooks are for exploration and prototyping. When ready for production, either refactor into Python modules (for complex models) or run via Papermill (for report-style pipelines)."

---

## Common Mistakes

1. **No idle management for GPU notebooks** — Data scientist starts a GPU notebook Monday morning, goes to meetings, forgets about it. Runs all week unattended. Cost: $500+ for an idle GPU. Solution: aggressive idle detection for GPU instances (30 min → warn, 60 min → terminate). Make restart fast (30s) so termination isn't painful.

2. **Snowflake environments** — Each data scientist installs different package versions locally. Model works in Alice's notebook but breaks in Bob's (different pandas version). Solution: managed Docker images with pinned versions. Users use platform environments, not local installs.

3. **No path to production** — Great notebook environment but no bridge to production. Data scientists hand off "this notebook that works" to engineers who spend weeks converting it. Solution: clear pipeline from notebook → production (Papermill for simple cases, refactoring guides for complex, templates for standard models).

4. **Overly restrictive** — Platform locks down everything (no pip install, no internet, no custom packages). Data scientists hate it, find workarounds, or stop using the platform. Solution: allow custom packages within environments, provide internet access for package registries, and offer custom environment creation for teams with specific needs.

5. **No persistent storage** — Notebook pod is ephemeral. User's work disappears when pod is culled. Rage ensues. Solution: persistent home directories (PVC) that survive pod restarts. User state (notebooks, configs, small datasets) is always preserved.

---

## Key Takeaways

- Notebook infra = managed environments + GPU access + data connectivity + cost control
- Options: JupyterHub (self-managed), SageMaker Studio/Vertex Workbench (cloud), Databricks (collaborative)
- Environment management: standardized Docker images with pinned packages, team customization on top
- Idle management: terminate GPU notebooks after 30-60 min idle (saves 40-60% on compute costs)
- Storage: persistent home directory (PVC) + shared data mounts + fast local scratch
- Git integration: Jupytext (notebooks as .py), nbstripout (clean outputs), nbdime (merge)
- Notebook → Production: Papermill (execute parameterized), refactor (complex models), templates
- Cost visibility: per-user/team dashboards, budget alerts, right-sizing recommendations
- Security: network isolation, data access controls, no raw internet (only package registries)
- Fast startup: pre-built cached images → notebook ready in 30-60 seconds
