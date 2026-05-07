# Experiment Tracking Deep Dive

## The Problem / Why This Matters

ML (Machine Learning) development is inherently experimental — you try dozens to hundreds of configurations (hyperparameters, data subsets, model architectures, training recipes) to find what works. Without systematic experiment tracking, teams face: (1) "I got great results last week but can't reproduce them" — lost experiments, (2) "Which run had the best F1 score?" — no comparison across experiments, (3) "What hyperparameters did we use for the production model?" — no audit trail, (4) "Was this data bug present in training run #47?" — no debugging context. Experiment tracking platforms solve this by automatically logging everything: hyperparameters, metrics over time, artifacts (model weights, plots), code versions, environment details, and data references. In 2026, experiment tracking is as fundamental to ML engineering as version control is to software engineering. The two dominant platforms are MLflow (open-source, self-hostable) and W&B (Weights & Biases) (commercial, cloud-hosted, superior visualization). Understanding both — and knowing when to use each — is essential for ML engineers.

---

## The Analogy

Think of experiment tracking like a laboratory notebook:

- **Without tracking** = Scribbling results on sticky notes. You run experiments, get numbers, celebrate or move on. Two weeks later, you can't find which sticky note had the winning configuration. Your colleague asks "how did you get that 94% accuracy?" and you don't remember.
- **With tracking (MLflow)** = A well-organized lab notebook. Every experiment has a dated entry, clear parameters, results, and attached artifacts. You own the notebook (self-hosted), can organize it your way, and share it within your lab.
- **With tracking (W&B)** = A digital lab notebook with superpowers. Every entry auto-populates with experiment details, results auto-plot over time, you can compare any two experiments side-by-side with a click, and your entire team sees results in real-time on a shared dashboard.

---

## Deep Dive

### MLflow

```python
# MLflow - Open-source experiment tracking
import mlflow
from mlflow.tracking import MlflowClient

# Connect to MLflow tracking server
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("llm-fine-tuning-v2")

# Start an experiment run
with mlflow.start_run(run_name="llama4-8b-lora-r16"):
    # Log hyperparameters
    mlflow.log_params({
        "model": "meta-llama/Llama-4-8B",
        "method": "LoRA",
        "lora_rank": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-5,
        "batch_size": 4,
        "gradient_accumulation": 8,
        "num_epochs": 3,
        "max_seq_length": 2048,
        "dataset": "custom-instruction-v3",
        "precision": "bf16",
    })
    
    # Training loop
    for epoch in range(3):
        for step, batch in enumerate(dataloader):
            loss = train_step(model, batch)
            
            # Log metrics at each step
            mlflow.log_metrics({
                "train_loss": loss,
                "learning_rate": scheduler.get_last_lr()[0],
            }, step=global_step)
        
        # Log epoch-level metrics
        eval_metrics = evaluate(model, eval_dataset)
        mlflow.log_metrics({
            "eval_loss": eval_metrics["loss"],
            "eval_accuracy": eval_metrics["accuracy"],
            "eval_f1": eval_metrics["f1"],
        }, step=epoch)
    
    # Log artifacts
    mlflow.log_artifact("training_config.yaml")
    mlflow.log_artifact("eval_results.json")
    
    # Log model (with MLflow Model Registry)
    mlflow.pytorch.log_model(
        model, 
        "model",
        registered_model_name="llama4-8b-instruction"
    )
    
    # Log tags for organization
    mlflow.set_tags({
        "team": "nlp",
        "project": "customer-support-bot",
        "gpu": "H100",
        "framework": "transformers",
    })
```

```yaml
MLflow_Architecture:
  components:
    tracking_server:
      what: "Central server that stores experiment metadata"
      backend: "PostgreSQL, MySQL, or SQLite (metadata)"
      artifact_store: "S3, GCS, Azure Blob, HDFS, or local filesystem"
      
    model_registry:
      what: "Central repository for model versions (staging → production)"
      workflow: "Register model → stage (staging/production) → deploy"
      versioning: "Each model has versions with lineage to training run"
      
    mlflow_ui:
      what: "Web UI for viewing experiments, comparing runs, visualizing metrics"
      features: "Filter, sort, compare, download artifacts"
      
  deployment_options:
    local: "mlflow server --host 0.0.0.0 --port 5000 (development)"
    docker: "Official Docker image with PostgreSQL backend"
    managed: "Databricks (MLflow creators), AWS SageMaker (built-in), Azure ML"
    kubernetes: "Helm chart deployment for production"
    
  strengths:
    - "Open-source (no vendor lock-in)"
    - "Self-hostable (data stays on your infrastructure)"
    - "Model Registry (built-in model lifecycle management)"
    - "Language-agnostic (Python, R, Java, REST API)"
    - "Integrations (HuggingFace, PyTorch, scikit-learn, XGBoost)"
    - "MLflow Projects (reproducible ML code packaging)"
    
  limitations:
    - "Visualization limited compared to W&B (basic charts)"
    - "No built-in hyperparameter sweep management"
    - "Collaboration features less polished (no real-time dashboards)"
    - "Requires self-management (server, database, storage)"
```

### Weights & Biases (W&B)

```python
# Weights & Biases - Commercial experiment tracking
import wandb

# Initialize run
run = wandb.init(
    project="llm-fine-tuning",
    name="llama4-8b-lora-r16",
    config={
        "model": "meta-llama/Llama-4-8B",
        "method": "LoRA",
        "lora_rank": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-5,
        "batch_size": 4,
        "gradient_accumulation": 8,
        "num_epochs": 3,
        "max_seq_length": 2048,
        "dataset": "custom-instruction-v3",
        "precision": "bf16",
    },
    tags=["llm", "lora", "instruction-tuning"],
    group="hyperparameter-search-v2",  # Group related runs
)

# Training loop with automatic logging
for step, batch in enumerate(dataloader):
    loss = train_step(model, batch)
    
    # Log metrics (auto-creates charts)
    wandb.log({
        "train/loss": loss,
        "train/learning_rate": scheduler.get_last_lr()[0],
        "train/gradient_norm": grad_norm,
        "system/gpu_utilization": gpu_util,
        "system/gpu_memory": gpu_mem,
    }, step=step)

# Log evaluation results
eval_table = wandb.Table(
    columns=["input", "expected", "generated", "score"],
    data=eval_results
)
wandb.log({"eval/predictions": eval_table})

# Log model artifact
artifact = wandb.Artifact("llama4-8b-lora", type="model")
artifact.add_dir("./model_output")
run.log_artifact(artifact)

# Finish run
run.finish()
```

```yaml
WandB_Features:
  visualization:
    - "Auto-generated training curves (loss, metrics over time)"
    - "Run comparison (overlay metrics from multiple runs)"
    - "Parallel coordinates plot (hyperparameter → metric relationships)"
    - "Custom dashboards (drag-and-drop panels)"
    - "Tables (log structured data: predictions, examples, errors)"
    
  sweeps:
    what: "Built-in hyperparameter optimization"
    methods: "Grid, random, Bayesian (similar to Optuna)"
    advantage: "Integrated with tracking — all sweep runs auto-logged and compared"
    config: |
      sweep_config = {
          "method": "bayes",
          "metric": {"name": "eval/f1", "goal": "maximize"},
          "parameters": {
              "learning_rate": {"min": 1e-5, "max": 1e-3, "distribution": "log_uniform_values"},
              "lora_rank": {"values": [8, 16, 32, 64]},
              "batch_size": {"values": [4, 8, 16]},
          }
      }
      sweep_id = wandb.sweep(sweep_config, project="llm-sweep")
      wandb.agent(sweep_id, function=train, count=50)
      
  artifacts:
    what: "Versioned dataset and model management"
    features:
      - "Automatic lineage (which artifact produced which run)"
      - "Aliases (latest, best, production)"
      - "Deduplication (same file referenced, not copied)"
      
  reports:
    what: "Shareable documents combining text, charts, and run data"
    use: "Stakeholder updates, experiment write-ups, team documentation"
    
  alerts:
    what: "Notifications on run events (crash, metric threshold, completion)"
    channels: "Slack, email, webhook"
    
  strengths:
    - "Best-in-class visualization (interactive, real-time, beautiful)"
    - "Collaboration (real-time team dashboards, run sharing)"
    - "System metrics (auto-logs GPU utilization, memory, CPU)"
    - "Sweeps (built-in HPO with full tracking)"
    - "Reports (combine text + charts for stakeholders)"
    - "Minimal setup (pip install wandb, create account, done)"
    
  limitations:
    - "Commercial (free tier, paid for teams — $50+/user/month)"
    - "Data sent to W&B cloud (privacy concern for some enterprises)"
    - "Vendor lock-in (harder to migrate away)"
    - "Self-hosted option exists but expensive"
```

### Comparing MLflow vs W&B

```yaml
Comparison:
  dimension_by_dimension:
    setup_complexity:
      mlflow: "Moderate (need to provision server, database, storage)"
      wandb: "Minimal (pip install, login, done)"
      winner: "W&B"
      
    visualization:
      mlflow: "Basic charts (adequate but not inspiring)"
      wandb: "Excellent (interactive, real-time, comparison, parallel coordinates)"
      winner: "W&B (significantly)"
      
    collaboration:
      mlflow: "Shared server (everyone sees same runs, basic)"
      wandb: "Real-time dashboards, reports, team workspaces"
      winner: "W&B"
      
    cost:
      mlflow: "Free (open-source), self-host infrastructure cost"
      wandb: "Free tier (100GB), Teams $50/user/month, Enterprise custom"
      winner: "MLflow (for budget-constrained teams)"
      
    data_privacy:
      mlflow: "Full control (self-hosted, data never leaves your infra)"
      wandb: "Cloud-hosted by default (self-hosted option available)"
      winner: "MLflow"
      
    model_registry:
      mlflow: "Built-in, mature (staging → production lifecycle)"
      wandb: "Artifact versioning (less formal model lifecycle)"
      winner: "MLflow"
      
    integrations:
      mlflow: "Good (HuggingFace, PyTorch, Spark, scikit-learn)"
      wandb: "Excellent (HuggingFace, PyTorch, Lightning, Keras, XGBoost)"
      winner: "Tie (both excellent)"
      
    hpo_integration:
      mlflow: "Separate (use Optuna/Ray Tune, log to MLflow)"
      wandb: "Built-in Sweeps (HPO + tracking in one platform)"
      winner: "W&B"
      
  recommendation:
    startups_research: "W&B (fastest setup, best visualization, free tier generous)"
    enterprises_regulated: "MLflow (self-hosted, full data control, no vendor lock)"
    budget_constrained: "MLflow (free, self-host on existing infra)"
    ml_platform_teams: "MLflow (integrates with broader ML platform — deployment, registry)"
    use_both: "Many teams use W&B for experiment viz + MLflow for model registry/deployment"
```

### Advanced Experiment Patterns

```yaml
Advanced_Patterns:
  experiment_organization:
    projects:
      description: "Group experiments by project/objective"
      example: "customer-support-bot, fraud-detection, recommendation-v2"
    groups:
      description: "Group runs within a project (HPO sweep, ablation study)"
      example: "hyperparameter-sweep-lr, architecture-comparison"
    tags:
      description: "Flexible labels for filtering"
      examples: "['production', 'baseline', 'ablation', 'v2', 'h100']"
      
  comparison_strategies:
    baseline_comparison:
      what: "Always compare new runs against established baseline"
      implementation: "Pin baseline run, overlay new runs on same chart"
    ablation_studies:
      what: "Change ONE variable at a time to understand its impact"
      logging: "Tag with 'ablation-{variable}' for easy filtering"
    statistical_significance:
      what: "Run best configs 3-5 times with different seeds"
      reason: "Single run can be lucky/unlucky — mean ± std needed for confidence"
      
  reproducibility:
    track_everything:
      - "Random seeds (data shuffle, initialization, dropout)"
      - "Git commit hash (exact code version)"
      - "Environment (Python version, library versions — pip freeze)"
      - "Data version (DVC hash, dataset commit)"
      - "Hardware (GPU type, count, driver version)"
    mlflow: "mlflow.log_param('git_commit', get_git_hash())"
    wandb: "Automatically logs git state, requirements.txt, code"
    
  automated_tracking:
    huggingface_integration:
      mlflow: "TrainingArguments(report_to='mlflow')"
      wandb: "TrainingArguments(report_to='wandb')"
    pytorch_lightning: "Built-in MLflowLogger and WandbLogger"
    autologging: "mlflow.autolog() — automatically logs params, metrics, models"
```

---

## How It Works in Practice

### Example: Team Experiment Workflow

```yaml
Example:
  scenario: "5-person ML team fine-tuning LLMs for production chatbot"
  
  setup:
    tracking: "W&B (team workspace, real-time collaboration)"
    model_registry: "MLflow (production deployment pipeline)"
    why_both: "W&B for experiment viz, MLflow for deployment lifecycle"
    
  workflow:
    phase_1_exploration:
      who: "Individual ML engineers"
      what: "Try different approaches (LoRA ranks, datasets, base models)"
      tracking: "Each engineer logs to shared W&B project"
      output: "W&B report summarizing findings (shared with team)"
      
    phase_2_refinement:
      who: "Team collaboration"
      what: "HPO sweep on promising approaches"
      tracking: "W&B Sweep (Bayesian, 50 trials)"
      output: "Best configuration identified via W&B parallel coordinates"
      
    phase_3_validation:
      who: "ML engineer + evaluation team"
      what: "Run best config 5× with different seeds, full evaluation"
      tracking: "W&B group (statistical significance)"
      output: "Confirmed improvement over baseline (mean F1: 0.89 ± 0.01)"
      
    phase_4_deployment:
      who: "ML platform engineer"
      what: "Register model in MLflow, promote to production"
      tracking: "MLflow Model Registry (version 7 → staging → production)"
      output: "Model deployed to serving infrastructure"
      
  daily_standup:
    tool: "W&B team dashboard"
    shows: "Active runs, recent results, GPU utilization, cost tracking"
    action: "Team reviews progress, identifies promising/failing approaches"
```

---

## Interview Tip

> When asked about experiment tracking: "I use experiment tracking on every project — it's non-negotiable. My typical setup: W&B for experiment visualization and team collaboration (interactive charts, run comparison, HPO sweeps with full tracking), plus MLflow for model registry and deployment lifecycle (register model → staging → production). Key practices: (1) Log everything — hyperparameters, metrics at every step, git hash, data version, environment. (2) Organize by project/group/tags — makes it easy to filter thousands of runs. (3) Always have a baseline — every new run is compared against established baseline. (4) Statistical significance — run best configs 3-5 times, report mean ± std, not single lucky runs. (5) Artifact tracking — link model weights to exact training run for full lineage. The goal is that any model in production can be traced back to exact code, data, and hyperparameters that produced it."

---

## Common Mistakes

1. **Logging too infrequently** — Only logging epoch-level metrics when step-level information would reveal training instabilities (loss spikes, learning rate issues). Log loss every 10-50 steps, GPU metrics every step, eval metrics every few hundred steps.

2. **No baseline tracking** — Running dozens of experiments without a pinned baseline for comparison. Every new run should be explicitly compared against the current best. Without this, you can't tell if "loss 2.3" is good or bad.

3. **Not logging system metrics** — Only tracking model metrics (loss, accuracy) without GPU utilization, memory, and throughput. When training is slow, system metrics tell you WHY (data loading bottleneck? low GPU utilization? memory pressure?).

4. **Starting tracking after development** — "I'll add tracking once the code is stable." This means you lose all the exploration runs that inform future decisions. Add tracking from the first experiment — it takes 5 lines of code.

5. **Over-relying on single metrics** — Making decisions based on a single eval metric without checking for trade-offs (accuracy improved but latency doubled, or F1 improved on one dataset but regressed on another). Log multiple metrics, visualize all of them.

---

## Key Takeaways

- Experiment tracking is mandatory — every ML project should track from day one
- MLflow: open-source, self-hostable, strong model registry — best for enterprises and deployment pipelines
- W&B: superior visualization, real-time collaboration, built-in HPO — best for research and team productivity
- Log everything: hyperparameters, step-level metrics, system metrics, code version, data version, environment
- Organize: projects → groups → tags — enable filtering across hundreds of runs
- Always maintain a baseline: compare every new run against current best
- Statistical significance: run best configs 3-5× with different seeds before declaring improvement
- Full lineage: production model → training run → exact code + data + hyperparameters
- Use both tools: W&B for experiment exploration + MLflow for deployment pipeline (common pattern)
- HuggingFace integration: `report_to=['wandb', 'mlflow']` in TrainingArguments — logs to both automatically
