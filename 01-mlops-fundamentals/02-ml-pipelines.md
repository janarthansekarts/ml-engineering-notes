# ML Pipelines

## The Problem / Why This Matters

An ML system in production isn't a single script — it's a complex DAG (Directed Acyclic Graph) of interdependent steps: data ingestion, validation, feature computation, model training, evaluation, registration, deployment, and monitoring. Without formal pipeline orchestration, these steps are manual notebooks, cron jobs, or shell scripts that break silently, can't be reproduced, and have no visibility into what ran when. ML pipelines formalize this workflow: each step is a self-contained unit with defined inputs/outputs, the pipeline tracks execution state, handles failures and retries, provides lineage tracking, and enables reproducibility. In 2026, ML pipelines have evolved beyond simple training automation to encompass the full ML lifecycle — including continuous training, LLM (Large Language Model) evaluation pipelines, RAG (Retrieval-Augmented Generation) index refresh workflows, and multi-model orchestration.

---

## The Analogy

Think of ML pipelines like an automotive assembly line:

- **Without a pipeline** = One person builds an entire car from scratch in their garage. They know the order, but if they're sick, production stops. If a part is defective, they might not notice until the car is "done." No record of what happened when.
- **With a pipeline** = An assembly line where each station has a specific job (chassis → engine → electrical → paint → quality check). Each station has clear inputs and outputs, quality gates between stages, and a digital record of everything. If a station fails, it retries or escalates. If a defective part enters, the next station catches it.
- **Orchestrator** = The factory floor manager. They know which stations are running, which are waiting, which failed, and what depends on what. They can restart a failed station without redoing the entire line.

---

## Deep Dive

### Types of ML Pipelines

```yaml
Pipeline_Types:
  training_pipeline:
    purpose: "End-to-end model training — data to registered model"
    steps:
      - "Data extraction (pull from source systems)"
      - "Data validation (schema, quality, freshness)"
      - "Feature engineering (compute features from raw data)"
      - "Data splitting (train/validation/test)"
      - "Model training (fit model on training data)"
      - "Model evaluation (metrics on test set)"
      - "Model registration (register in model registry if passes gates)"
    trigger: "Scheduled (weekly), event-driven (new data), or manual"
    frequency: "Daily to monthly depending on data velocity"
    
  data_pipeline:
    purpose: "Transform raw data into ML-ready features"
    steps:
      - "Ingest from sources (databases, APIs, event streams)"
      - "Clean and validate (handle nulls, outliers, schema drift)"
      - "Transform (aggregations, joins, normalization)"
      - "Store (feature store, data warehouse, object storage)"
    trigger: "Scheduled or event-driven (new data arrival)"
    relationship: "Feeds into training pipeline and feature serving"
    
  serving_pipeline:
    purpose: "Deploy and manage model serving"
    steps:
      - "Load model from registry"
      - "Package into serving container"
      - "Deploy to infrastructure (K8s, cloud endpoint)"
      - "Configure routing (canary, shadow, A/B)"
      - "Validate serving health (latency, error rate)"
    trigger: "New model promoted to production alias"
    
  evaluation_pipeline:
    purpose: "Continuously evaluate production model quality"
    steps:
      - "Collect ground truth labels (delayed feedback)"
      - "Join predictions with ground truth"
      - "Compute metrics (accuracy, AUC, business KPIs)"
      - "Compare against thresholds and historical performance"
      - "Alert if degradation detected"
    trigger: "Scheduled (daily/weekly) or continuous"
    
  llm_pipeline:
    purpose: "LLM application lifecycle — index refresh, evaluation, deployment"
    steps:
      - "Knowledge base update (new documents → chunk → embed → index)"
      - "Retrieval quality evaluation (test queries → check relevance)"
      - "Generation quality evaluation (LLM-as-judge on sample)"
      - "Prompt version management (A/B test new prompts)"
      - "Deploy updated RAG config"
    trigger: "New documents, prompt changes, model updates"
```

### Pipeline Orchestration Tools

```yaml
Orchestration_Tools:
  kubeflow_pipelines:
    description: "Kubernetes-native ML pipeline platform"
    strengths:
      - "Tight Kubernetes integration (each step = container)"
      - "Pipeline SDK (Python) for defining pipelines as code"
      - "Built-in experiment tracking and artifact management"
      - "Caching (skip unchanged steps)"
      - "ML-specific components (training, serving, hyperparameter tuning)"
    weaknesses:
      - "Complex to set up (requires Kubernetes cluster)"
      - "Heavyweight for simple pipelines"
      - "UI less polished than alternatives"
    best_for: "Teams already on Kubernetes with complex ML workflows"
    
  apache_airflow:
    description: "General-purpose workflow orchestration (most popular)"
    strengths:
      - "Mature ecosystem, large community"
      - "Rich operator library (cloud providers, databases, tools)"
      - "DAG definition in Python"
      - "Battle-tested scheduling and retry logic"
      - "Managed offerings (Astronomer, Cloud Composer, MWAA)"
    weaknesses:
      - "Not ML-specific (no built-in experiment tracking)"
      - "Heavy (overkill for simple pipelines)"
      - "Scheduler can become bottleneck at scale"
    best_for: "Organizations using Airflow for data pipelines — extend to ML"
    
  dagster:
    description: "Modern data-aware orchestration — assets over tasks"
    strengths:
      - "Software-defined assets (focus on WHAT, not just HOW)"
      - "Built-in data quality checks (asset checks)"
      - "Excellent developer experience (local testing, type system)"
      - "Native partitioning (handle time-partitioned data elegantly)"
      - "Integrated observability"
    weaknesses:
      - "Newer than Airflow (smaller community)"
      - "Different mental model (assets vs tasks) — learning curve"
    best_for: "Data-centric ML teams, new pipeline projects (2026 recommendation)"
    
  vertex_ai_pipelines:
    description: "Google Cloud managed pipeline service (KFP-based)"
    strengths:
      - "Fully managed (no infrastructure to manage)"
      - "Integrated with Vertex AI ecosystem (training, serving, feature store)"
      - "Pipeline component marketplace"
    best_for: "GCP-native ML teams"
    
  sagemaker_pipelines:
    description: "AWS managed ML pipeline service"
    strengths:
      - "Integrated with SageMaker ecosystem"
      - "Built-in model registry, endpoints, monitoring"
      - "Step caching and conditional execution"
    best_for: "AWS-native ML teams"
    
  prefect:
    description: "Modern workflow orchestration — lightweight, Python-native"
    strengths:
      - "Simple API (@flow, @task decorators)"
      - "Hybrid execution (local + cloud)"
      - "Built-in retries, caching, concurrency"
      - "Lightweight compared to Airflow"
    best_for: "Smaller teams wanting simple orchestration"
```

### Pipeline Design Patterns

```yaml
Pipeline_Patterns:
  component_reusability:
    principle: "Each pipeline step is a self-contained component with clear I/O"
    implementation:
      - "Docker container per step (language-agnostic)"
      - "Defined input/output artifacts (schemas)"
      - "Parameterized (configurable without code changes)"
      - "Reusable across pipelines"
    benefit: "Data validation step used in training, evaluation, and monitoring pipelines"
    
  pipeline_caching:
    principle: "Skip steps whose inputs haven't changed"
    how: "Hash inputs (data + code + params) → if hash matches previous run, reuse output"
    benefit: "Faster iteration — only re-run what changed"
    caveat: "Cache invalidation for time-dependent steps (freshness matters)"
    
  conditional_execution:
    principle: "Branch pipeline based on intermediate results"
    examples:
      - "If data quality check fails → skip training, alert team"
      - "If model performance improved → register model; else → skip"
      - "If drift detected → run full retraining; else → run incremental update"
      
  pipeline_parametrization:
    principle: "Same pipeline code, different configurations"
    examples:
      - "Training pipeline parameterized by: model_type, dataset_version, hyperparameters"
      - "One pipeline definition serves 10 different model training jobs"
    benefit: "Reduced code duplication, consistent workflow across models"
    
  idempotency:
    principle: "Running a pipeline step twice produces the same result"
    definition: "Idempotent (producing the same result when applied multiple times)"
    implementation:
      - "Overwrite outputs (not append)"
      - "Use unique run IDs for artifacts"
      - "Deterministic seeds for random operations"
    benefit: "Safe retries — failed pipeline can be re-run from any step"
    
  pipeline_testing:
    unit_tests: "Test individual steps with mock data"
    integration_tests: "Test step connections (output of A is valid input for B)"
    end_to_end_tests: "Run full pipeline on small test dataset"
    contract_tests: "Verify step I/O schemas match expectations"
```

---

## How It Works in Practice

### Production Pipeline Example

```yaml
Example:
  project: "Weekly retraining pipeline for product recommendation model"
  tool: "Dagster (assets-based orchestration)"
  
  pipeline_definition:
    asset_1_raw_data:
      source: "Data warehouse (BigQuery)"
      schedule: "Every Monday at 2 AM"
      output: "Raw interaction data (clicks, purchases) for trailing 90 days"
      validation: "Row count > 10M, no null user_ids, schema matches expected"
      
    asset_2_features:
      depends_on: "asset_1_raw_data"
      computation: "Spark job computing user features and item features"
      output: "Feature table with 200+ features per user-item pair"
      validation: "No features with >5% null, distributions within expected range"
      freshness: "Features must be <24 hours old at serving time"
      
    asset_3_training_data:
      depends_on: "asset_2_features"
      computation: "Join features with labels (click/purchase), split train/val/test"
      output: "Train (70%), validation (15%), test (15%) datasets"
      validation: "Label distribution consistent with historical (no sudden shifts)"
      
    asset_4_model:
      depends_on: "asset_3_training_data"
      computation: "Train two-tower neural model with hyperparameters from config"
      tracking: "Log to MLflow (params, metrics, model artifacts)"
      output: "Trained model weights + evaluation results"
      
    asset_5_evaluation:
      depends_on: "asset_4_model"
      computation: "Evaluate on test set, compare against production champion"
      metrics: "NDCG@10, MRR, coverage, diversity"
      gate: "NDCG@10 >= champion_ndcg - 0.01 (don't deploy if worse)"
      
    asset_6_registration:
      depends_on: "asset_5_evaluation (if gate passes)"
      action: "Register model in MLflow, assign 'challenger' alias"
      trigger: "Notify deployment pipeline to begin shadow deployment"
      
  failure_handling:
    data_quality_failure: "Alert data team, retry in 4 hours (data might be late)"
    training_failure: "Alert ML team, log error, keep current champion"
    evaluation_gate_failure: "Log comparison results, keep current champion, alert for review"
    
  monitoring:
    pipeline_health: "Dagster UI — execution status, duration, asset freshness"
    data_lineage: "Full trace from raw data → features → model → predictions"
    alerting: "PagerDuty for critical failures, Slack for warnings"
```

---

## Interview Tip

> When asked about ML pipelines: "I design ML pipelines with clear separation of concerns: data pipelines (ingestion, validation, feature computation), training pipelines (data → model), evaluation pipelines (continuous quality measurement), and serving pipelines (deployment and routing). Key principles: (1) Each step is idempotent and self-contained with defined I/O. (2) Quality gates between stages prevent bad data/models from propagating. (3) Caching skips unchanged steps for faster iteration. (4) Same pipeline runs in development and production (no 'works on my machine'). For tooling, I've used Dagster (modern, asset-focused, great DX), Kubeflow Pipelines (K8s-native), and Airflow (when it's already the org standard). The biggest anti-pattern is a monolithic script pretending to be a pipeline — true pipelines have independently restartable steps, lineage tracking, and failure isolation."

---

## Common Mistakes

1. **Monolithic training scripts** — One 2000-line Python script that does everything from data loading to deployment. Any failure requires rerunning from the beginning, there's no visibility into intermediate state, and no ability to cache completed work.

2. **No quality gates** — Pipeline runs all steps regardless of intermediate results. Bad data flows into training, bad models flow into production. Gates after data validation and model evaluation are non-negotiable.

3. **Development/production divergence** — Training pipeline runs differently in development (local, small data, different paths) vs production. This makes bugs discoverable only in production. Use the same pipeline code, just different configs.

4. **Ignoring pipeline testing** — Treating pipelines as "infrastructure" that doesn't need tests. Pipelines are code — they need unit tests (individual steps), integration tests (step connections), and end-to-end tests (small data through full pipeline).

5. **Over-engineering orchestration** — Setting up Kubeflow Pipelines on a Kubernetes cluster for a single weekly training job. Match tool complexity to actual needs — start with Dagster or even a simple script + scheduler, add orchestration as complexity warrants.

---

## Key Takeaways

- ML pipelines formalize the ML workflow: defined steps, clear I/O, quality gates, lineage tracking
- Pipeline types: data, training, serving, evaluation, LLM/RAG pipelines — each has distinct concerns
- Key patterns: component reusability, caching, conditional execution, idempotency, parameterization
- Quality gates between steps prevent bad data/models from propagating downstream
- Tools: Dagster (modern, recommended for new projects), Kubeflow (K8s), Airflow (established), managed (Vertex/SageMaker)
- Same pipeline code in dev and prod — only configs differ (data sources, scale, resources)
- Pipeline testing is mandatory: unit test steps, integration test connections, e2e test on small data
- Caching speeds iteration: only re-run steps whose inputs changed
- Pipeline monitoring: execution health, step durations, data freshness, failure rates
- Start simple (script + scheduler) and add orchestration as complexity grows
