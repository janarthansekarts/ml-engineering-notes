# Experiment Management

## The Problem / Why This Matters

ML development is inherently experimental. Unlike traditional software where you write deterministic code, ML involves trying hundreds of combinations — different features, architectures, hyperparameters, data subsets, preprocessing steps — to find what works. Without experiment management, teams face chaos: "Which model was the best one we trained last month?" "What hyperparameters did we use?" "Can we reproduce that result?" "Did we already try this approach?" In 2026, experiment management is even more critical because teams run experiments across traditional ML (XGBoost, neural networks), fine-tuning (LoRA/QLoRA on LLMs), prompt optimization (different prompt versions and strategies), and RAG (Retrieval-Augmented Generation) configurations. A single project might have thousands of tracked runs. Tools like MLflow and Weights & Biases (W&B) have become non-negotiable infrastructure, as essential as Git is for code versioning.

---

## The Analogy

Think of experiment management like a scientist's lab notebook:

- **Without a lab notebook** = You run experiments, get interesting results, but months later can't remember the exact conditions. Did you heat it to 150°C or 160°C? Was the sample from batch A or B? You can't reproduce your breakthrough.
- **With a lab notebook** = Every experiment is documented: conditions, observations, results, and your thoughts. Months later, anyone can pick up your notebook and reproduce exactly what you did.
- **Digital experiment tracking** = The lab notebook on steroids. Automatic, searchable, comparable, visualizable, and shared across the team. Plus it records things humans forget to note — exact software versions, random seeds, GPU type.

---

## Deep Dive

### What to Track

```yaml
Experiment_Tracking:
  what_to_log:
    parameters:
      model: "Architecture, hidden size, layers, dropout, activation functions"
      training: "Learning rate, batch size, epochs, optimizer, scheduler"
      data: "Dataset version, split ratios, preprocessing steps, augmentation"
      infrastructure: "GPU type, number of GPUs, distributed strategy"
      
    metrics:
      training_metrics: "Loss, accuracy/F1/AUC at each epoch — for learning curves"
      evaluation_metrics: "Final metrics on validation and test sets"
      per_class_metrics: "Precision/recall per class (not just aggregate)"
      business_metrics: "Expected impact on business KPIs"
      resource_metrics: "Training time, GPU memory used, cost"
      
    artifacts:
      model_files: "Trained weights, optimizer state, config"
      evaluation_reports: "Confusion matrix, ROC curves, calibration plots"
      data_samples: "Examples the model got right/wrong (for debugging)"
      plots: "Loss curves, feature importance, embedding visualizations"
      
    metadata:
      git_commit: "Exact code version used"
      data_version: "DVC hash or dataset tag"
      environment: "Docker image, Python version, package versions"
      author: "Who ran this experiment"
      notes: "Free-form observations and hypotheses"
      tags: "Categorization (baseline, ablation, final, failed)"
```

### MLflow

```yaml
MLflow:
  description: "Open-source ML lifecycle platform — experiment tracking, model registry, deployment"
  
  components:
    tracking:
      purpose: "Log parameters, metrics, and artifacts for each run"
      api_example: |
        import mlflow
        
        mlflow.set_experiment("fraud-detection-v2")
        
        with mlflow.start_run(run_name="xgboost-tuned"):
            # Log parameters
            mlflow.log_param("max_depth", 6)
            mlflow.log_param("learning_rate", 0.01)
            mlflow.log_param("n_estimators", 500)
            mlflow.log_param("data_version", "v2.3")
            
            # Train model
            model = train_model(params)
            
            # Log metrics
            mlflow.log_metric("auc", 0.94)
            mlflow.log_metric("precision_at_95_recall", 0.87)
            mlflow.log_metric("training_time_seconds", 342)
            
            # Log artifacts
            mlflow.log_artifact("confusion_matrix.png")
            mlflow.sklearn.log_model(model, "model")
      
    model_registry:
      purpose: "Version models, manage stages (staging → production), approval workflows"
      stages: "None → Staging → Production → Archived"
      features:
        - "Version history with full lineage"
        - "Annotations and descriptions"
        - "Transition approval (requires sign-off for production)"
        - "Aliases for deployment (e.g., 'champion', 'challenger')"
        
    deployment:
      purpose: "Serve models via REST API"
      options:
        - "Local serving (mlflow models serve)"
        - "Docker container generation"
        - "Cloud deployment (SageMaker, Azure ML, Databricks)"
        
  hosting_options:
    self_hosted: "MLflow server on your infrastructure (most control)"
    databricks: "Managed MLflow (tight Spark/Databricks integration)"
    cloud_managed: "Azure ML, SageMaker (built-in MLflow integration)"
    
  strengths: "Open-source, flexible, integrates with everything, model registry"
  weaknesses: "UI less polished than W&B, collaboration features basic"
```

### Weights & Biases (W&B)

```yaml
Weights_and_Biases:
  description: "Commercial experiment tracking platform — rich visualization, collaboration, LLM support"
  
  key_features:
    experiment_tracking:
      purpose: "Same as MLflow but with richer UI and collaboration"
      differentiator: "Real-time dashboards, team collaboration, automatic visualizations"
      
    sweeps:
      purpose: "Automated hyperparameter optimization"
      methods: "Grid, random, Bayesian optimization"
      advantage: "Distributed sweeps across multiple machines, early termination"
      
    tables:
      purpose: "Log and visualize tabular data (predictions, examples, comparisons)"
      use_case: "Log model predictions for qualitative analysis — see what the model gets wrong"
      
    artifacts:
      purpose: "Version datasets and model files with lineage tracking"
      advantage: "Track data → model → deployment lineage automatically"
      
    reports:
      purpose: "Shareable documents combining code, metrics, plots, and narrative"
      use_case: "Document experiment conclusions for stakeholders"
      
    weave:
      purpose: "LLM application tracing and evaluation (new in 2025-2026)"
      features:
        - "Trace LLM calls (prompt, response, tokens, latency)"
        - "Evaluate outputs (LLM-as-judge, custom scorers)"
        - "Compare prompt versions"
        - "Track RAG retrieval quality"
      differentiator: "Integrated with experiment tracking — same platform for traditional ML and LLM apps"
      
  strengths: "Beautiful UI, collaboration, LLM support (Weave), reports"
  weaknesses: "Commercial (free tier limited), data leaves your infra (unless self-hosted)"
  pricing: "Free for individuals, $50+/user/month for teams"
```

### Experiment Organization

```yaml
Experiment_Organization:
  naming_conventions:
    project_level: "fraud-detection, search-ranking, customer-support-agent"
    experiment_level: "baseline, feature-engineering-v2, architecture-search, production-candidate"
    run_level: "xgboost-depth6-lr001, bert-base-frozen-3epoch, llama4-lora-r16"
    
  tagging_strategy:
    purpose_tags: ["baseline", "ablation", "hyperopt", "final", "failed", "exploratory"]
    data_tags: ["full-dataset", "sampled-10pct", "augmented", "cleaned-v2"]
    status_tags: ["promising", "deployed", "deprecated", "reproducibility-verified"]
    
  comparison_workflows:
    ablation_study:
      purpose: "Determine contribution of each component"
      method: "Start with full model, remove one component at a time, measure impact"
      example: "Full model: 94% AUC. Without feature X: 91%. Feature X contributes 3%."
      
    hyperparameter_search:
      purpose: "Find optimal hyperparameters"
      method: "Sweep over parameter space, visualize metric landscape"
      tools: "W&B Sweeps, Optuna, Ray Tune"
      
    model_comparison:
      purpose: "Choose between approaches"
      method: "Compare multiple models on same evaluation set, same metrics"
      requirement: "Statistical significance — don't pick 'winner' based on noise"
```

---

## How It Works in Practice

### Experiment Workflow Example

```yaml
Example:
  project: "Improve customer churn prediction (currently 82% AUC)"
  
  experiment_plan:
    baseline: "Current production model (XGBoost, features v1)"
    hypothesis_1: "Adding behavioral features will improve prediction"
    hypothesis_2: "Deep learning (TabNet) will capture non-linear patterns better"
    hypothesis_3: "Ensemble of XGBoost + TabNet will beat both individually"
    
  tracked_experiments:
    run_1_baseline:
      params: { model: "xgboost", features: "v1", data: "2024-Q4" }
      result: { auc: 0.82, f1: 0.71 }
      
    run_2_new_features:
      params: { model: "xgboost", features: "v2_with_behavioral", data: "2024-Q4" }
      result: { auc: 0.86, f1: 0.76 }
      note: "Behavioral features (session_count, last_login_days) huge impact"
      
    run_3_tabnet:
      params: { model: "tabnet", features: "v2_with_behavioral", data: "2024-Q4" }
      result: { auc: 0.85, f1: 0.74 }
      note: "Slightly worse than XGBoost on this data size"
      
    run_4_ensemble:
      params: { model: "ensemble_xgb_tabnet", features: "v2", data: "2024-Q4" }
      result: { auc: 0.87, f1: 0.77 }
      note: "Marginal gain — complexity not worth it for 1% AUC"
      
  decision:
    promoted: "run_2 (XGBoost + new features) — best trade-off of performance vs complexity"
    reasoning: "4% AUC improvement from features alone. Ensemble adds only 1% more with 2x complexity."
    next_steps: "Deploy run_2, collect feedback, plan next feature iteration"
```

---

## Interview Tip

> When asked how you manage ML experiments: "I use a structured approach with either MLflow (open-source, self-hosted) or W&B (team collaboration, richer UI). Every run logs: all hyperparameters, data version (DVC hash), metrics at each step and final, model artifacts, and the git commit. I organize by project → experiment → individual runs, with tags for purpose (baseline, ablation, final). For LLM work, I use W&B Weave or LangSmith to trace prompt versions, token costs, and output quality. Key practices: (1) always have a tagged baseline run to compare against, (2) ablation studies to understand component contributions, (3) statistical significance testing before declaring winners, (4) reports documenting conclusions and rationale for the team."

---

## Common Mistakes

1. **Not tracking from day one** — "I'll add tracking later when the project is more mature." By then, you've lost weeks of experiment history and can't reproduce your best early results.

2. **Tracking metrics but not parameters** — Knowing a run achieved 94% AUC is useless if you don't know the hyperparameters, data version, and preprocessing that produced it.

3. **No baseline run** — Running experiments without a clearly tagged baseline. You end up comparing against different things at different times, making progress impossible to measure.

4. **Over-reading small differences** — Declaring a "0.2% improvement" as significant when it's within noise margins. Always run multiple seeds and compute confidence intervals.

5. **Abandoned experiments** — Running experiments and never documenting why they failed. Future team members (or your future self) will waste time trying the same failed approaches.

---

## Key Takeaways

- Track everything: parameters, metrics, artifacts, code version, data version, environment
- MLflow: open-source, self-hostable, model registry, wide integration
- W&B: richer UI, collaboration, sweeps, LLM support (Weave), reports
- Organization: project → experiment → run, with purpose tags and naming conventions
- Always maintain a tagged baseline run for comparison
- Ablation studies reveal what actually matters vs what's noise
- Statistical significance: don't declare winners based on single runs
- For LLMs: track prompt versions, token costs, latency, and quality scores alongside traditional metrics
- Experiment tracking is as essential as Git — non-negotiable infrastructure
