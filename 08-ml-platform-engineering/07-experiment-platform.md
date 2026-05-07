# Experiment Platform

## The Problem / Why This Matters

ML development is fundamentally experimental — data scientists run hundreds of experiments, varying hyperparameters, features, architectures, and data. Without a shared experiment platform, teams lose track of what they tried (which hyperparameters worked?), can't compare across experiments (is model A actually better than model B?), can't collaborate (what did my teammate try last week?), and waste time re-running failed experiments. An experiment platform provides: structured experiment tracking (log everything automatically), comparison tools (metrics, learning curves, artifacts side-by-side), collaboration features (share experiments, tag, annotate), reproducibility guarantees (record exact environment, code version, data version), and integration with the training pipeline (experiments → model registry → deployment). In 2026, experiment platforms also handle: LLM prompt experiments (tracking prompt versions, response quality), hyperparameter optimization (automated search), and experiment governance (approval workflows for production-bound experiments). The main tools: Weights & Biases (W&B, most feature-rich), MLflow Tracking (open-source standard), Neptune (metadata-focused), Comet ML (enterprise), and cloud-native options (SageMaker Experiments, Vertex AI Experiments).

---

## The Analogy

Think of an experiment platform like a scientific laboratory notebook system:

- **Without platform** = Each scientist uses their own notebook (some use paper, some Excel, some nothing). Results are scattered, unreproducible, and team members can't build on each other's work. Years of experiments lost when someone leaves.
- **With platform** = Electronic Lab Notebook (ELN) system. Every experiment automatically recorded with: hypothesis, materials used (data, code version), procedure (hyperparameters, training config), results (metrics, artifacts), and conclusions. Searchable, shareable, reproducible.

The difference between a world-class research lab and a chaotic one is often the quality of their experiment management system.

---

## Deep Dive

### Experiment Platform Comparison

```yaml
Platforms_2026:
  weights_and_biases:
    what: "Most feature-rich ML experiment tracking platform"
    strengths:
      - "Beautiful visualization (custom charts, parallel coordinates)"
      - "Sweeps (built-in hyperparameter optimization)"
      - "Artifacts (versioned datasets and models)"
      - "Reports (shareable experiment summaries)"
      - "Tables (interactive data exploration)"
      - "Launch (job scheduling from experiments)"
      - "Model registry (integrated)"
      - "LLM tracing (prompt tracking, evaluation)"
    pricing: "Free tier (100GB), Team ($50/user/month), Enterprise (custom)"
    adoption: "Very high in research and industry"
    integration: "PyTorch, TensorFlow, HuggingFace, LangChain, etc."
    
  mlflow:
    what: "Open-source ML lifecycle platform (by Databricks)"
    strengths:
      - "Open-source (self-host or managed)"
      - "Tracking + Models + Registry + Deployment in one tool"
      - "Language-agnostic (Python, R, Java, REST API)"
      - "Largest ecosystem of integrations"
      - "Autologging (automatic metric capture for major frameworks)"
      - "Model packaging standard (MLflow Model format)"
    limitations:
      - "UI less polished than W&B"
      - "Less real-time visualization"
      - "Self-hosted requires infrastructure management"
    pricing: "Free (self-hosted) or Databricks managed"
    adoption: "Very high (especially in enterprise with Databricks)"
    
  neptune:
    what: "Metadata store for ML experiments"
    strengths:
      - "Flexible metadata (any structure)"
      - "Fast querying (optimized for experiment comparison)"
      - "Integrations with 25+ frameworks"
      - "Clean comparison UI"
    pricing: "Free tier, then per-experiment"
    
  comet_ml:
    what: "Enterprise experiment management"
    strengths:
      - "Code diff tracking (see code changes between experiments)"
      - "Model production monitoring"
      - "Enterprise features (SSO, audit, compliance)"
    pricing: "Enterprise-focused"
    
  cloud_native:
    sagemaker_experiments:
      what: "AWS integrated experiment tracking"
      strengths: "Tight SageMaker integration, no setup"
      limitation: "AWS-only, less feature-rich than W&B/MLflow"
      
    vertex_experiments:
      what: "GCP integrated experiment tracking"
      strengths: "Tight Vertex AI integration"
      limitation: "GCP-only"
```

### Experiment Tracking Architecture

```python
# Experiment tracking infrastructure

"""
Standardized experiment tracking that captures everything needed
for reproducibility and comparison.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
import json


@dataclass
class Experiment:
    """
    Represents a complete ML experiment.
    Everything needed to reproduce and understand the experiment.
    """
    # Identity
    experiment_id: str
    name: str
    project: str
    team: str
    author: str
    
    # Timing
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Reproducibility
    code_version: str = ""           # Git commit SHA
    code_diff: str = ""              # Uncommitted changes
    environment: dict = field(default_factory=dict)  # conda/pip freeze
    random_seed: int = 42
    data_version: str = ""           # Dataset version used
    
    # Configuration
    hyperparameters: dict = field(default_factory=dict)
    model_architecture: str = ""
    training_config: dict = field(default_factory=dict)  # epochs, batch_size, etc.
    
    # Results
    metrics: dict = field(default_factory=dict)  # accuracy, loss, f1, etc.
    metric_history: list = field(default_factory=list)  # Per-epoch metrics
    
    # Artifacts
    model_path: str = ""
    artifacts: list = field(default_factory=list)  # Plots, configs, outputs
    
    # Metadata
    tags: list = field(default_factory=list)
    notes: str = ""
    status: str = "running"  # running, completed, failed, killed


class ExperimentTracker:
    """
    Platform's experiment tracking service.
    Provides automatic and manual logging for all experiments.
    """
    
    def __init__(self, project: str, backend: str = "mlflow"):
        self.project = project
        self.backend = backend  # "mlflow", "wandb", "neptune"
    
    def start_experiment(
        self,
        name: str,
        config: dict,
        tags: list = None,
    ) -> "ExperimentRun":
        """
        Start a new experiment run.
        
        Automatically captures:
        - Git commit SHA and diff
        - Python environment (packages + versions)
        - Hardware info (GPU type, count)
        - Start time
        
        Example:
            with tracker.start_experiment("lr-sweep-001", config={"lr": 0.001}) as run:
                model = train(config)
                run.log_metrics({"accuracy": 0.95})
                run.log_artifact(model_path)
        """
        run = ExperimentRun(
            tracker=self,
            name=name,
            config=config,
            tags=tags or [],
        )
        
        # Auto-capture environment
        run.log_system_info()
        run.log_git_info()
        run.log_environment()
        
        return run
    
    def compare_experiments(
        self,
        experiment_ids: list,
        metrics: list = None,
    ) -> "ComparisonReport":
        """
        Compare multiple experiments side-by-side.
        
        Returns structured comparison:
        - Metrics comparison (table)
        - Hyperparameter differences
        - Learning curves overlay
        - Statistical significance tests
        """
        experiments = [self._get_experiment(eid) for eid in experiment_ids]
        
        return ComparisonReport(
            experiments=experiments,
            metrics_table=self._build_metrics_table(experiments, metrics),
            param_diff=self._compute_param_diff(experiments),
            significance=self._statistical_comparison(experiments, metrics),
        )
    
    def find_best_experiment(
        self,
        project: str,
        metric: str,
        direction: str = "maximize",  # "maximize" or "minimize"
        filters: dict = None,
    ) -> Experiment:
        """
        Find the best experiment for a given metric.
        
        Example:
            best = tracker.find_best_experiment(
                project="churn-model",
                metric="val_f1",
                direction="maximize",
                filters={"tags": "production-candidate"},
            )
        """
        experiments = self._query_experiments(project, filters)
        
        if direction == "maximize":
            return max(experiments, key=lambda e: e.metrics.get(metric, float('-inf')))
        else:
            return min(experiments, key=lambda e: e.metrics.get(metric, float('inf')))


class ExperimentRun:
    """An active experiment run with logging capabilities."""
    
    def __init__(self, tracker, name, config, tags):
        self.tracker = tracker
        self.name = name
        self.config = config
        self.tags = tags
        self.step = 0
    
    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics (can be called multiple times for per-step tracking)."""
        self.step = step or self.step + 1
        # Metrics logged to backend (MLflow/W&B/etc.)
        self.tracker._log_metrics(self, metrics, self.step)
    
    def log_params(self, params: dict):
        """Log hyperparameters."""
        self.tracker._log_params(self, params)
    
    def log_artifact(self, path: str, name: str = None):
        """Log an artifact (model file, plot, config, etc.)."""
        self.tracker._log_artifact(self, path, name)
    
    def log_table(self, name: str, data: Any):
        """Log tabular data (predictions, feature importance, etc.)."""
        self.tracker._log_table(self, name, data)
    
    def log_image(self, name: str, image):
        """Log an image (confusion matrix, learning curve, etc.)."""
        self.tracker._log_image(self, name, image)
    
    def set_tags(self, tags: list):
        """Add tags to experiment (for filtering/grouping)."""
        self.tags.extend(tags)
    
    def finish(self, status: str = "completed"):
        """Mark experiment as finished."""
        self.tracker._finish_run(self, status)
```

### Hyperparameter Optimization

```yaml
HPO_Integration:
  what: "Automated hyperparameter search integrated with experiment tracking"
  
  methods:
    grid_search:
      what: "Try all combinations of specified values"
      use_when: "Few hyperparameters (2-3), small value ranges"
      integration: "Each combination = one experiment run"
      
    random_search:
      what: "Random sampling from distributions"
      use_when: "Many hyperparameters, large ranges"
      advantage: "Often finds good solutions faster than grid"
      
    bayesian_optimization:
      what: "Model the objective function, choose next point intelligently"
      tools: "Optuna, Ray Tune, W&B Sweeps"
      advantage: "Finds optima with fewer trials"
      
    population_based_training:
      what: "Evolve hyperparameters during training (not just between runs)"
      tool: "Ray Tune PBT"
      advantage: "Adapts hyperparameters during training (like adaptive learning rate)"
      
  platform_integration:
    wandb_sweeps:
      what: "W&B built-in HPO with Bayesian optimization"
      flow: "Define sweep config → W&B agent runs experiments → results tracked automatically"
      
    optuna_mlflow:
      what: "Optuna for search + MLflow for tracking"
      flow: "Optuna suggests params → train model → log to MLflow → Optuna uses results"
      
    ray_tune:
      what: "Distributed HPO with experiment tracking"
      flow: "Ray Tune distributes trials across GPUs → logs to W&B/MLflow"
      advantage: "Scales to hundreds of GPUs for parallel search"
```

### LLM Experiment Tracking

```yaml
LLM_Experiments:
  what: "Tracking experiments for LLM-based applications"
  
  what_to_track:
    prompts:
      - "System prompt (versioned)"
      - "User prompt template"
      - "Few-shot examples used"
      - "RAG context retrieval configuration"
    model:
      - "Model name and version (Claude 4 Sonnet, GPT-5, Llama-4-70B)"
      - "Temperature, top_p, max_tokens"
      - "Fine-tuning configuration (if fine-tuned)"
    evaluation:
      - "Response quality scores (relevance, coherence, accuracy)"
      - "Hallucination rate"
      - "Latency and cost per query"
      - "User feedback (if available)"
    context:
      - "Retrieved documents (for RAG)"
      - "Number of retrieved chunks"
      - "Retrieval scores"
      
  tools:
    langsmith:
      what: "LangChain's experiment tracking for LLM apps"
      tracks: "Full chain traces, prompt versions, evaluations"
      
    braintrust:
      what: "LLM evaluation and experimentation platform"
      tracks: "Prompts, model outputs, evaluations, A/B tests"
      
    wandb_prompts:
      what: "W&B tracing for LLM applications"
      tracks: "LLM calls, chain-of-thought, tool use, cost"
      
    arize_phoenix:
      what: "LLM observability and experimentation"
      tracks: "Traces, evaluations, embeddings, retrieval quality"
      
  experiment_patterns:
    prompt_comparison:
      what: "Compare different prompt versions on same eval set"
      flow: "Define prompts → run on eval dataset → compare quality metrics"
      
    model_comparison:
      what: "Compare different LLMs on same task"
      flow: "Same prompt → different models → compare quality/cost/latency"
      
    rag_optimization:
      what: "Optimize retrieval configuration"
      variables: "Chunk size, overlap, top-k, reranking model, embedding model"
      flow: "Vary config → evaluate retrieval quality → evaluate end-to-end"
```

---

## How It Works in Practice

### Daily Workflow

```yaml
Workflow:
  researcher_daily:
    morning:
      - "Check dashboard: review overnight training runs"
      - "Compare experiments: which hyperparameters worked best?"
      - "Tag promising experiments: 'production-candidate'"
      
    working:
      - "Start new experiment (auto-logs everything)"
      - "Iterate on model (each variation = new run)"
      - "Log intermediate results (loss curves, validation metrics)"
      - "Share findings with team (link to experiment dashboard)"
      
    end_of_day:
      - "Add notes to experiments (observations, hypotheses for tomorrow)"
      - "Clean up: kill stalled experiments, archive failed ones"
      
  team_weekly:
    - "Review experiment board: what did the team try this week?"
    - "Identify best-performing experiments"
    - "Decide: ready for production? (tag for deployment pipeline)"
    - "Plan next experiments based on learnings"
```

---

## Interview Tip

> When asked about experiment platforms: "My experiment platform provides three capabilities: (1) Automatic tracking — every training run logs: hyperparameters, metrics (per-step and final), code version (Git SHA + diff), environment (package versions), data version, hardware used, and training artifacts. Uses autologging (one line to enable) so data scientists don't need to manually log anything. I use MLflow for open-source or W&B when budget allows (superior visualization and sweeps). (2) Comparison and analysis — side-by-side comparison of experiments with statistical significance testing. 'Is model B actually better than model A, or is it within noise?' Parallel coordinates plots for hyperparameter sensitivity analysis. Interactive filtering and sorting. (3) Reproducibility guarantee — given any experiment ID, I can reproduce it exactly: same code (Git SHA), same data (dataset version), same environment (pip freeze), same random seed. This is critical for debugging production issues ('what exactly was model v3 trained on?'). For LLM applications, I extend tracking to: prompt versions, model configurations (temperature, max_tokens), RAG parameters (chunk size, top-k), and evaluation metrics (relevance, hallucination rate, cost per query). Tools like LangSmith or Braintrust integrate with the experiment platform for LLM-specific workflows."

---

## Common Mistakes

1. **Not tracking experiments at all** — "I'll remember what parameters I used." Three months later, best model is in production but nobody knows the exact hyperparameters, data version, or code that produced it. Solution: autologging from day one. MLflow's autolog captures everything with one line: `mlflow.autolog()`.

2. **Logging metrics but not artifacts** — Track accuracy and loss, but not the confusion matrix, learning curves, feature importance plots, or model predictions on edge cases. When investigating production issues, these artifacts are essential. Solution: log everything that helps understand model behavior — plots, predictions on test cases, error analysis.

3. **No experiment organization** — 10,000 experiments in a flat list. Finding "that promising experiment from 3 weeks ago" requires scrolling through pages. Solution: projects (group by model type), tags (production-candidate, failed, baseline), and notes on every experiment.

4. **Running same experiment twice** — Team member runs the exact same hyperparameters that were already tried (and failed) because they couldn't find the previous result. Solution: before starting a new experiment, query existing results. Platform should warn: "Similar experiment already exists (run_id: xyz, accuracy: 0.84)."

5. **No experiment → production pipeline** — Great experiment tracking, but when it's time to deploy, the model is manually extracted, repackaged, and deployed through a different system. Experiment context is lost. Solution: direct pipeline from experiment → model registry → deployment. Best experiment gets tagged "production-candidate" → promoted to registry → deployed.

---

## Key Takeaways

- Experiment platform = automatic tracking + comparison tools + reproducibility + collaboration
- Tool choice: W&B (most features, paid), MLflow (open-source standard), Neptune (flexible metadata)
- Autologging: capture everything automatically (params, metrics, code, env, data version)
- Comparison: side-by-side metrics, statistical significance, parallel coordinates for HPO
- Reproducibility: every experiment reproducible from experiment ID (code + data + env + seed)
- Organization: projects, tags, notes — find any experiment quickly
- HPO integration: Optuna/Ray Tune for search, experiment platform for tracking all trials
- LLM experiments: track prompts, model configs, RAG parameters, quality evaluations
- Experiment → Production: direct pipeline from best experiment → model registry → deployment
- Collaboration: share experiments, comment, annotate — build on team's collective knowledge
