# Pipeline Orchestration

## The Problem / Why This Matters

ML workflows are complex DAGs (Directed Acyclic Graphs) with dozens of interdependent steps: data extraction, validation, feature computation, training, evaluation, registration, deployment, and monitoring. These steps have dependencies (training can't start until features are computed), resource requirements (training needs GPUs, evaluation needs CPU), timing constraints (features must be fresh), and failure modes (any step can fail and needs retry/alerting). Without orchestration, teams use cron jobs and manual scripts that break silently, can't handle failures gracefully, provide no visibility, and make debugging impossible. Pipeline orchestration tools manage the complexity: scheduling, dependency resolution, resource allocation, retry logic, lineage tracking, and observability — allowing ML engineers to focus on ML logic rather than infrastructure plumbing. In 2026, orchestration has become more critical as pipelines span traditional ML training, LLM (Large Language Model) evaluation, RAG (Retrieval-Augmented Generation) index builds, and multi-model workflows.

---

## The Analogy

Think of pipeline orchestration like an air traffic control system:

- **Without orchestration** = Planes (pipeline steps) take off whenever pilots decide, with no coordination. Some crash into each other (resource conflicts), some circle forever waiting to land (deadlocks), and when something goes wrong, nobody knows where any plane is (no observability).
- **With orchestration** = Air traffic control knows every plane's schedule, manages takeoff/landing sequences (dependencies), handles delays (retries), reroutes on problems (failure handling), and provides a complete radar view (observability). Pilots focus on flying (ML logic), not on coordinating with other planes.
- **Different orchestrators** = Different airports. Airflow is the massive international hub (handles everything, complex). Dagster is the modern regional airport (efficient, great experience). Kubeflow is the military base (purpose-built for specific missions).

---

## Deep Dive

### Orchestration Core Concepts

```yaml
Orchestration_Concepts:
  dag:
    definition: "Directed Acyclic Graph — defines task execution order"
    nodes: "Individual tasks/steps"
    edges: "Dependencies between tasks"
    acyclic: "No circular dependencies (task A depends on B depends on A = invalid)"
    
  scheduling:
    cron: "Time-based triggers (daily at 2 AM, every 6 hours)"
    event_driven: "Trigger on external event (new data arrived, model registered)"
    manual: "Human-triggered execution"
    sensor: "Wait for condition (file exists, API returns success)"
    
  execution:
    sequential: "Tasks run one after another"
    parallel: "Independent tasks run simultaneously"
    conditional: "Tasks run based on conditions (if evaluation passes, continue)"
    dynamic: "DAG structure determined at runtime (variable number of tasks)"
    
  fault_tolerance:
    retry: "Automatically retry failed tasks (with backoff)"
    timeout: "Kill tasks that exceed time limit"
    alerting: "Notify on failure (PagerDuty, Slack)"
    skip: "Optionally skip non-critical failed tasks"
    
  observability:
    logs: "Per-task stdout/stderr captured and searchable"
    metrics: "Execution duration, success rate, resource usage"
    lineage: "Data lineage — what data flowed through which tasks"
    history: "Full execution history with outcomes"
```

### Kubeflow Pipelines

```yaml
Kubeflow_Pipelines:
  description: "Kubernetes-native ML pipeline platform — each step runs in a container"
  
  architecture:
    execution: "Each pipeline step = Kubernetes Pod (containerized)"
    metadata: "ML Metadata store tracks all artifacts and executions"
    ui: "Visual pipeline graph, run comparison, artifact viewer"
    
  defining_pipelines: |
    from kfp import dsl, compiler
    from kfp.dsl import Input, Output, Dataset, Model, Metrics
    
    @dsl.component(base_image="python:3.11")
    def preprocess_data(
        raw_data: Input[Dataset],
        processed_data: Output[Dataset]
    ):
        import pandas as pd
        df = pd.read_parquet(raw_data.path)
        # preprocessing logic
        df_processed = preprocess(df)
        df_processed.to_parquet(processed_data.path)
    
    @dsl.component(base_image="pytorch/pytorch:2.3-cuda12.1")
    def train_model(
        training_data: Input[Dataset],
        model: Output[Model],
        metrics: Output[Metrics],
        learning_rate: float = 0.001,
        epochs: int = 10
    ):
        # training logic
        trained_model = train(training_data.path, lr=learning_rate, epochs=epochs)
        save_model(trained_model, model.path)
        metrics.log_metric("accuracy", 0.94)
    
    @dsl.pipeline(name="training-pipeline")
    def training_pipeline(lr: float = 0.001):
        preprocess_task = preprocess_data(raw_data=get_data_task.outputs["dataset"])
        train_task = train_model(
            training_data=preprocess_task.outputs["processed_data"],
            learning_rate=lr
        )
    
    # Compile to YAML for submission
    compiler.Compiler().compile(training_pipeline, "pipeline.yaml")
    
  strengths:
    - "Container isolation (each step has its own environment)"
    - "GPU support via Kubernetes (request GPU resources per step)"
    - "Artifact tracking (inputs/outputs tracked automatically)"
    - "Pipeline caching (skip unchanged steps)"
    - "ML-specific components (training, serving, HPO built-in)"
    
  weaknesses:
    - "Requires Kubernetes (complex setup)"
    - "Heavier than alternatives for simple pipelines"
    - "Compilation step (pipeline → YAML → Argo workflow)"
    - "Debugging can be difficult (logs in pods)"
    
  managed_options:
    - "Vertex AI Pipelines (Google Cloud)"
    - "AWS SageMaker Pipelines (uses own format but KFP compatible)"
    - "Azure ML Pipelines"
```

### Apache Airflow

```yaml
Apache_Airflow:
  description: "Most widely adopted workflow orchestration platform — general purpose"
  
  architecture:
    scheduler: "Determines what to run and when"
    executor: "Runs tasks (local, Celery, Kubernetes, etc.)"
    webserver: "UI for monitoring and management"
    metadata_db: "PostgreSQL/MySQL storing DAG state, history"
    
  defining_dags: |
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.providers.amazon.operators.sagemaker import SageMakerTrainingOperator
    from datetime import datetime, timedelta
    
    default_args = {
        "owner": "ml-team",
        "retries": 3,
        "retry_delay": timedelta(minutes=5),
        "email_on_failure": True,
    }
    
    with DAG(
        dag_id="weekly_model_retraining",
        schedule_interval="0 2 * * 1",  # Every Monday at 2 AM
        start_date=datetime(2026, 1, 1),
        catchup=False,
        default_args=default_args,
    ) as dag:
        
        extract_data = PythonOperator(
            task_id="extract_data",
            python_callable=extract_training_data,
        )
        
        validate_data = PythonOperator(
            task_id="validate_data",
            python_callable=run_data_validation,
        )
        
        train_model = SageMakerTrainingOperator(
            task_id="train_model",
            config=training_config,
        )
        
        evaluate_model = PythonOperator(
            task_id="evaluate_model",
            python_callable=run_evaluation,
        )
        
        # Define dependencies
        extract_data >> validate_data >> train_model >> evaluate_model
    
  strengths:
    - "Massive ecosystem (hundreds of provider packages)"
    - "Battle-tested at scale (Airbnb, Spotify, Netflix)"
    - "Rich scheduling (cron, sensors, triggers)"
    - "Managed offerings (Astronomer, Cloud Composer, MWAA)"
    - "Extensive community and documentation"
    
  weaknesses:
    - "Not ML-specific (no built-in experiment tracking)"
    - "DAGs are Python scripts (but tasks run in separate processes)"
    - "Scheduler can be a bottleneck (many DAGs = slow parsing)"
    - "Testing is harder than modern alternatives"
    - "Data passing between tasks requires XCom (limited)"
    
  for_ml:
    pattern: "Airflow orchestrates, ML tools execute"
    example: "Airflow triggers SageMaker training → waits → triggers MLflow registration → triggers KServe deployment"
    role: "Scheduler and coordinator, not ML execution engine"
```

### Dagster

```yaml
Dagster:
  description: "Modern data-aware orchestration — software-defined assets"
  
  paradigm_shift:
    traditional: "Define TASKS (do this, then do that) — procedural"
    dagster: "Define ASSETS (this data exists and depends on that data) — declarative"
    benefit: "Focus on what data exists and its quality, not how to run things"
    
  defining_assets: |
    import dagster as dg
    from dagster import asset, AssetIn, MaterializeResult
    
    @asset(
        description="Raw transaction data from data warehouse",
        group_name="data_ingestion",
        metadata={"source": "BigQuery", "refresh": "daily"},
    )
    def raw_transactions() -> pd.DataFrame:
        """Pull latest transactions from data warehouse."""
        return pull_from_bigquery("SELECT * FROM transactions WHERE date > CURRENT_DATE - 90")
    
    @asset(
        description="Validated and cleaned transaction features",
        group_name="feature_engineering",
        ins={"transactions": AssetIn("raw_transactions")},
    )
    def transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
        """Compute features from raw transactions."""
        validate_schema(transactions)
        return compute_features(transactions)
    
    @asset(
        description="Trained fraud detection model",
        group_name="training",
        ins={"features": AssetIn("transaction_features")},
    )
    def fraud_model(features: pd.DataFrame) -> Model:
        """Train XGBoost fraud detection model."""
        X, y = split_features_labels(features)
        model = train_xgboost(X, y)
        log_to_mlflow(model)
        return model
    
    # Dagster automatically builds dependency graph from asset relationships
    
  key_features:
    asset_checks:
      description: "Built-in data quality validation"
      usage: "Define checks that run after asset materialization"
      
    partitions:
      description: "Native support for time-partitioned data"
      benefit: "Process daily/hourly partitions independently, backfill easily"
      
    schedules_and_sensors:
      description: "Time-based scheduling and event-driven triggers"
      
    io_managers:
      description: "Abstract storage — same code, different backends"
      benefit: "Use SQLite locally, BigQuery in production — no code change"
      
    testing:
      description: "First-class testing support — test assets as Python functions"
      benefit: "Easy unit testing without needing orchestration infrastructure"
      
  strengths:
    - "Modern developer experience (type system, testing, IDE support)"
    - "Asset-oriented (focus on WHAT, not HOW)"
    - "Built-in data quality (asset checks)"
    - "Excellent local development (no complex infrastructure needed)"
    - "Native partitioning (time-based data handled elegantly)"
    
  weaknesses:
    - "Newer ecosystem (fewer integrations than Airflow)"
    - "Different paradigm (learning curve for Airflow users)"
    - "Smaller community (growing rapidly)"
    
  recommendation: "For new ML projects in 2026, Dagster is the recommended starting point"
```

---

## How It Works in Practice

### Orchestration in Production

```yaml
Example:
  system: "Weekly model retraining with daily evaluation"
  tool: "Dagster"
  
  pipeline_structure:
    daily_pipeline:
      schedule: "Every day at 6 AM"
      assets:
        - "fresh_data (pull latest day from warehouse)"
        - "data_validation (quality checks on fresh data)"
        - "evaluation_data (join predictions with ground truth)"
        - "model_evaluation (compute metrics for current production model)"
        - "drift_detection (compare current distributions vs training)"
      alerts: "Slack notification if drift detected or metrics below threshold"
      
    weekly_pipeline:
      schedule: "Monday at 2 AM"
      assets:
        - "training_data (last 90 days, validated)"
        - "feature_computation (compute all features via Spark)"
        - "model_training (train new model version)"
        - "model_evaluation_comparison (compare new vs champion)"
        - "model_registration (register if better, as challenger)"
        - "shadow_deployment (deploy challenger in shadow mode)"
      gates: "Model must beat champion by >0.5% AND pass fairness AND pass latency"
      
    promotion_pipeline:
      trigger: "After 48 hours of successful shadow deployment"
      assets:
        - "shadow_comparison (analyze shadow results)"
        - "canary_deployment (route 5% to challenger)"
        - "canary_evaluation (monitor canary metrics)"
        - "full_promotion (if canary passes, promote to champion)"
      rollback: "Auto-revert if canary metrics degrade"
      
  monitoring:
    dagster_ui: "Visual asset lineage, materialization history, failure tracking"
    alerts: "PagerDuty for critical failures, Slack for warnings"
    metrics: "Pipeline duration, success rate, data freshness"
```

---

## Interview Tip

> When asked about pipeline orchestration: "I choose orchestration tools based on the problem: Dagster for new ML projects (asset-oriented, great DX, built-in quality checks), Airflow when it's already the organizational standard (massive ecosystem, battle-tested), and Kubeflow Pipelines for Kubernetes-native ML workflows with container isolation. Key principles: (1) Every step is idempotent — safe to retry without side effects. (2) Quality gates between stages prevent cascading failures. (3) Same pipeline code in development and production — only configs differ. (4) Full observability — I need to see what ran, when, how long it took, what data flowed through. For ML specifically, I design pipelines with two loops: a fast daily evaluation loop (detect problems) and a slower weekly/monthly retraining loop (improve models). The orchestrator coordinates both, managing dependencies and resources."

---

## Common Mistakes

1. **Cron jobs instead of orchestration** — Using crontab for ML pipelines. No dependency management, no failure handling, no visibility, no retry logic. One failure cascades silently through all downstream steps.

2. **Monolithic DAGs** — One massive DAG with 100+ tasks. Impossible to understand, slow to parse, difficult to debug. Break into smaller, composable pipelines with clear boundaries.

3. **No idempotency** — Pipeline steps that produce different results when retried (appending instead of overwriting, using current timestamp instead of partition timestamp). Every step should be safe to re-run.

4. **Wrong tool for scale** — Using Kubernetes-based orchestration (Kubeflow) for 3 simple weekly pipelines. Or using cron for 50 interdependent daily pipelines. Match tool complexity to actual needs.

5. **No pipeline testing** — Orchestration pipelines treated as "infrastructure" that doesn't need tests. Pipelines are code — test them: unit test individual tasks, integration test task connections, end-to-end test on sample data.

---

## Key Takeaways

- Pipeline orchestration manages: scheduling, dependencies, retries, resources, and observability for ML workflows
- Core concepts: DAGs, scheduling (cron/event/sensor), execution (sequential/parallel), fault tolerance (retry/alert)
- Kubeflow Pipelines: Kubernetes-native, container-per-step, ML-specific — for K8s environments
- Airflow: battle-tested, massive ecosystem, general-purpose — when already established in org
- Dagster: modern, asset-oriented, great DX — recommended for new ML projects in 2026
- Design principle: same pipeline code in dev and prod — only configurations differ
- Idempotency is non-negotiable — every step must be safe to retry
- Two loops for ML: daily evaluation (detect issues fast) + weekly/monthly retraining (improve model)
- Quality gates between stages prevent cascading failures (bad data → bad features → bad model)
- Pipeline testing: unit test tasks, integration test connections, e2e test on sample data
