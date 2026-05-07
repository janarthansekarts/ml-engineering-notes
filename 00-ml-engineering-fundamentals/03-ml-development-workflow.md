# ML Development Workflow

## The Problem / Why This Matters

Traditional software development has a clear workflow: write code → test → deploy → monitor. ML development is fundamentally different because you're not just writing deterministic logic — you're conducting experiments where the outcome is uncertain. A code change in traditional software either works or doesn't. A model change might work better on one dataset but worse on another, work in testing but fail on production data distribution, or improve one metric while degrading another. Without a disciplined ML development workflow, teams waste months on experiments that can't be reproduced, models that can't be deployed, and improvements that can't be validated. In 2026, with teams running hundreds of experiments across traditional ML and LLM (Large Language Model) applications simultaneously, having a structured workflow isn't optional — it's the difference between shipping value and burning compute credits.

---

## The Analogy

Think of ML development like pharmaceutical drug development:

- **Hypothesis** = "This compound might cure disease X" (this model architecture might solve problem Y)
- **Lab testing** = Test in controlled conditions (offline evaluation on held-out data)
- **Clinical trials** = Test on real patients with controls (A/B testing in production)
- **FDA approval** = Regulatory sign-off before mass distribution (model governance, approval workflow)
- **Post-market surveillance** = Monitor for side effects after release (model monitoring in production)

You wouldn't ship a drug based solely on lab results. Similarly, you shouldn't ship a model based solely on offline metrics. The workflow ensures each stage validates what the previous stage claimed.

---

## Deep Dive

### The ML Development Lifecycle

```yaml
ML_Development_Lifecycle:
  stage_1_problem_definition:
    activities:
      - "Define the business problem clearly (not the ML problem)"
      - "Determine if ML is the right approach (vs rules, heuristics)"
      - "Define success metrics (business metrics, not just model metrics)"
      - "Establish baseline (current system performance, human performance)"
      - "Scope constraints (latency, cost, fairness requirements)"
    outputs:
      - "Problem statement document"
      - "Success criteria with measurable thresholds"
      - "Baseline performance numbers"
    common_failure: "Jumping to model training without defining what 'good enough' means"
    
  stage_2_data_preparation:
    activities:
      - "Identify and access data sources"
      - "Exploratory data analysis (EDA)"
      - "Data cleaning and validation"
      - "Feature engineering (initial)"
      - "Train/validation/test split (time-based for temporal data)"
      - "Data versioning (DVC, LakeFS, Delta Lake)"
    outputs:
      - "Versioned, validated dataset"
      - "Data documentation (schema, distributions, known issues)"
      - "Feature pipeline (reproducible)"
    common_failure: "Data leakage — using future information to predict the past"
    
  stage_3_experimentation:
    activities:
      - "Define experiment (hypothesis, metrics, approach)"
      - "Implement model/pipeline changes"
      - "Train with experiment tracking (MLflow, W&B)"
      - "Evaluate against baseline and previous experiments"
      - "Error analysis (where does the model fail?)"
      - "Document findings and decisions"
    outputs:
      - "Experiment results logged with full reproducibility"
      - "Best model candidate with evaluation report"
      - "Decision: proceed to production or iterate"
    common_failure: "Not logging experiment parameters — can't reproduce results later"
    
  stage_4_validation:
    activities:
      - "Offline evaluation on held-out test set (never seen during training)"
      - "Fairness and bias testing across subgroups"
      - "Robustness testing (adversarial, edge cases)"
      - "Performance profiling (latency, memory, throughput)"
      - "Model card creation (documentation for governance)"
      - "Peer review by another engineer or data scientist"
    outputs:
      - "Validated model registered in model registry"
      - "Model card with limitations, biases, performance characteristics"
      - "Sign-off from reviewer(s)"
    common_failure: "Evaluating only on aggregate metrics — missing failures on specific subgroups"
    
  stage_5_deployment:
    activities:
      - "Package model for serving (containerize, optimize)"
      - "Deploy to staging environment"
      - "Integration testing with downstream systems"
      - "Shadow deployment (serve alongside production, compare)"
      - "Canary deployment (serve to small % of traffic)"
      - "Full rollout with rollback plan"
    outputs:
      - "Model serving in production with monitoring"
      - "Deployment runbook (how to roll back)"
      - "Alerts configured for key metrics"
    common_failure: "No rollback plan — model degrades and team scrambles to fix"
    
  stage_6_monitoring_and_iteration:
    activities:
      - "Monitor prediction distribution, latency, errors"
      - "Detect data drift and concept drift"
      - "Collect ground truth for continuous evaluation"
      - "Trigger retraining when performance degrades"
      - "Plan next iteration based on production insights"
    outputs:
      - "Dashboards showing model health"
      - "Alerts firing on degradation"
      - "Retraining pipeline running on schedule/trigger"
    common_failure: "Deploy and forget — model slowly degrades without anyone noticing"
```

### Reproducibility in ML

```yaml
Reproducibility:
  why_it_matters: "If you can't reproduce a result, you can't trust it, debug it, or improve it"
  
  what_to_version:
    code: "Git — model code, training scripts, serving code, tests"
    data: "DVC or LakeFS — training data, validation data, test data"
    config: "Hydra/YAML — hyperparameters, feature config, model architecture"
    environment: "Docker — exact packages, CUDA version, Python version"
    model_artifacts: "MLflow/Model Registry — trained weights, preprocessing objects"
    
  experiment_tracking:
    tool: "MLflow or Weights & Biases (W&B)"
    what_to_log:
      parameters: "Learning rate, batch size, epochs, architecture choices"
      metrics: "Loss, accuracy, F1, custom metrics — at each step and final"
      artifacts: "Model files, plots, confusion matrices, evaluation reports"
      code_version: "Git commit hash"
      data_version: "DVC hash or dataset version tag"
      environment: "Python version, package versions, GPU type"
    best_practice: "Every experiment should be reproducible from its logged metadata alone"
    
  random_seed_management:
    set_seeds_for:
      - "Python random module"
      - "NumPy random"
      - "PyTorch (torch.manual_seed, torch.cuda.manual_seed_all)"
      - "Data shuffling and splitting"
    note: "Even with seeds, GPU non-determinism can cause small variations — document expected variance"
```

### ML Development Best Practices (2026)

```yaml
Best_Practices:
  start_simple:
    principle: "Always start with the simplest approach that could work"
    progression:
      step_1: "Rule-based baseline (if-else, regex, heuristics)"
      step_2: "Simple ML model (logistic regression, XGBoost)"
      step_3: "Complex ML model (deep learning, ensemble)"
      step_4: "LLM-based approach (if appropriate for the task)"
    why: "Each step gives you signal about where the problem difficulty lies"
    
  iterate_fast:
    principle: "Optimize for learning speed, not model accuracy in early stages"
    tactics:
      - "Use small data samples for initial experiments"
      - "Use simple evaluation first (before complex offline eval)"
      - "Ship to shadow mode quickly to get production signal"
      - "Time-box experiments (2 weeks max before decision point)"
    anti_pattern: "Spending 3 months optimizing a model without production validation"
    
  fail_fast:
    principle: "Kill bad ideas early, before they consume resources"
    tactics:
      - "Define kill criteria before starting ('if we can't beat baseline by X%, stop')"
      - "Validate data quality BEFORE model training"
      - "Test inference latency BEFORE optimizing accuracy"
      - "Check edge cases BEFORE full evaluation"
    anti_pattern: "Training for a week, then discovering the model can't meet latency requirements"
    
  automate_everything:
    principle: "If you do it twice, automate it"
    what_to_automate:
      - "Data validation pipeline"
      - "Training pipeline (not just training script)"
      - "Evaluation pipeline (automated test suite for models)"
      - "Deployment pipeline (CI/CD for models)"
      - "Monitoring and alerting setup"
    why: "Manual steps introduce errors, slow down iteration, and don't scale"
```

---

## How It Works in Practice

### Real ML Development Sprint

```yaml
Example_Sprint:
  context: "E-commerce search ranking team, 2-week sprint"
  
  week_1:
    monday:
      - "Sprint planning: prioritize experiments based on expected impact"
      - "Experiment 1: Add 'time since last purchase' feature"
      - "Experiment 2: Try cross-encoder reranker (instead of bi-encoder)"
    tue_wed:
      - "Implement features, run training with W&B logging"
      - "Evaluate: Experiment 1 → +0.8% NDCG (Normalized Discounted Cumulative Gain)"
      - "Evaluate: Experiment 2 → +2.1% NDCG but +40ms latency"
    thu_fri:
      - "Decision: Ship Experiment 1 (easy win), iterate on Experiment 2 (reduce latency)"
      - "Experiment 1: Push to model registry, request review"
      - "Experiment 2: Try distilled cross-encoder (smaller model)"
      
  week_2:
    monday:
      - "Code review passed for Experiment 1"
      - "Deploy Experiment 1 to shadow mode (compare against production)"
    tue_wed:
      - "Shadow results: Experiment 1 shows +0.6% CTR improvement (real traffic!)"
      - "Experiment 2 iteration: Distilled model → +1.5% NDCG, only +12ms latency"
    thu:
      - "Canary deployment: Experiment 1 to 5% of traffic"
      - "Submit Experiment 2 for review"
    friday:
      - "Canary looks good — promote Experiment 1 to 100%"
      - "Sprint retrospective: document learnings"
      - "Plan next sprint based on monitoring insights"
```

---

## Interview Tip

> When asked about your ML development workflow, demonstrate maturity by covering: "I follow a structured workflow: (1) Problem definition with clear success criteria and baseline — I always establish what we're trying to beat. (2) Data preparation with versioning — I use DVC so every experiment links to exact data. (3) Rapid experimentation with full tracking — W&B or MLflow, logging everything so results are reproducible. (4) Validation beyond aggregate metrics — subgroup analysis, fairness testing, latency profiling. (5) Staged deployment — shadow mode first, then canary with monitoring, then full rollout with rollback ready. (6) Continuous monitoring — drift detection, ground truth collection, automated retraining triggers. The key principle: optimize for learning speed early, then optimize for reliability as you approach production."

---

## Common Mistakes

1. **No baseline** — Starting complex model development without measuring what a simple approach (or the current system) achieves. You can't claim improvement without a baseline to compare against.

2. **Evaluating on training data** — Whether it's overfitting to the training set or data leakage from the test set, invalid evaluation gives false confidence. Always use truly held-out data.

3. **Optimizing the wrong metric** — Improving model accuracy while the business cares about precision at a specific recall threshold, or optimizing latency when the real bottleneck is data freshness.

4. **No time-boxing** — Letting experiments run indefinitely because "maybe one more hyperparameter will work." Set deadlines. If it's not working after the time-box, change your approach.

5. **Manual deployment** — Deploying models by SSH-ing into a server and copying files. This doesn't scale, isn't reproducible, and has no rollback. Use CI/CD from day one.

---

## Key Takeaways

- ML development lifecycle: Problem → Data → Experimentation → Validation → Deployment → Monitoring
- Reproducibility requires versioning: code (Git), data (DVC), config (Hydra), environment (Docker), models (Registry)
- Start simple (rules → simple ML → complex ML → LLM), iterate fast, fail fast
- Always establish a baseline before building complex models
- Time-box experiments — if it's not working in 2 weeks, change approach
- Staged deployment: shadow → canary → full rollout, always with rollback
- Automate early: data validation, training pipelines, evaluation, deployment, monitoring
- Log everything in experiment tracking — you will need to reproduce results months later
