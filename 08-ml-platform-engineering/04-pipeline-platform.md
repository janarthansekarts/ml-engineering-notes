# Pipeline Platform

## The Problem / Why This Matters

Every ML team builds pipelines — data preprocessing, feature computation, model training, evaluation, deployment. Without a shared pipeline platform, each team reinvents the wheel: Team A writes a Dagster pipeline, Team B uses Airflow, Team C has custom bash scripts. Nobody can reuse components, there's no consistency in monitoring or error handling, and the platform team can't provide shared infrastructure support. A pipeline platform provides: shared orchestration infrastructure (one system, not ten), reusable pipeline components (standardized data loading, validation, model training steps), templates for common patterns (train-evaluate-deploy, feature computation, batch inference), and centralized monitoring for all pipelines. In 2026, the pipeline platform landscape includes mature options: Dagster (asset-oriented, software-defined), Prefect (Pythonic, cloud-native), Kubeflow Pipelines (Kubernetes-native ML), Apache Airflow (battle-tested, largest ecosystem), ZenML (MLOps-focused), and Flyte (type-safe, scalable). The engineering challenge: choosing the right orchestrator, building reusable components that work across teams, standardizing pipeline patterns without constraining flexibility, and providing observability across hundreds of scheduled pipelines.

---

## The Analogy

Think of a pipeline platform like a car manufacturing assembly line system:

- **Without a platform** = Each car model has its own custom factory. Engine team builds their own conveyor belts, paint team builds their own spray booth, assembly team builds their own robots. Enormous duplication, no shared quality control.
- **With a platform** = Shared factory infrastructure. Standardized conveyor belts (orchestration), interchangeable robot arms (reusable components), quality checkpoints at every stage (validation), and a central control room monitoring all production lines (observability).

Teams still design their own cars (ML models), but the manufacturing infrastructure is shared and consistent.

---

## Deep Dive

### Orchestrator Comparison

```yaml
Orchestrators_2026:
  dagster:
    philosophy: "Software-defined assets — declare WHAT you want, system figures out HOW"
    strengths:
      - "Asset-oriented (data lineage built-in)"
      - "Strong typing and contracts between components"
      - "Excellent local development experience"
      - "Built-in partitioning (time, logical)"
      - "IO managers (abstract storage layer)"
      - "Sensors and schedules"
    weaknesses:
      - "Smaller community than Airflow"
      - "Different mental model (assets vs tasks)"
    best_for: "ML teams wanting type safety, data lineage, modern DX"
    scale: "Thousands of assets, production-proven"
    
  prefect:
    philosophy: "Python functions are pipelines — minimal decorator overhead"
    strengths:
      - "Most Pythonic (decorators on regular functions)"
      - "Dynamic workflows (create tasks at runtime)"
      - "Hybrid execution (orchestrate locally, compute anywhere)"
      - "Built-in retry, caching, concurrency"
      - "Good for ML experimentation (flexible)"
    weaknesses:
      - "Less opinionated about data assets"
      - "Cloud-centric (Prefect Cloud for production)"
    best_for: "ML teams wanting flexibility and Python-native feel"
    
  kubeflow_pipelines:
    philosophy: "Kubernetes-native ML pipelines with containers as steps"
    strengths:
      - "Each step is a container (perfect isolation)"
      - "GPU-native (Kubernetes GPU scheduling)"
      - "Integrates with KFP SDK for ML components"
      - "Versioned pipelines as YAML/JSON"
      - "Built for ML (training, serving, experiments)"
    weaknesses:
      - "Requires Kubernetes expertise"
      - "Heavier setup than Python-native tools"
      - "Container overhead for simple steps"
    best_for: "ML teams on Kubernetes wanting strong isolation and GPU support"
    
  airflow:
    philosophy: "DAGs of tasks — battle-tested workflow orchestration"
    strengths:
      - "Largest ecosystem (1000+ operators/providers)"
      - "Battle-tested at massive scale (Airbnb, Google, Spotify)"
      - "Managed options (MWAA, Cloud Composer, Astronomer)"
      - "Extensive community knowledge"
    weaknesses:
      - "DAG-only (no dynamic workflows without hacks)"
      - "Slow DAG parsing at scale"
      - "Less ML-specific features"
      - "Task-oriented (not asset/data-oriented)"
    best_for: "Organizations with existing Airflow investment or needing broad integrations"
    
  flyte:
    philosophy: "Type-safe, reproducible workflows for ML"
    strengths:
      - "Strong typing (enforced input/output types)"
      - "Built-in versioning and caching"
      - "Multi-tenancy and resource isolation"
      - "Map tasks (parallel execution)"
      - "Built for ML (GPU support, distributed training)"
    weaknesses:
      - "Smaller community"
      - "More complex setup"
    best_for: "ML teams wanting type safety and reproducibility at scale"
    
  zenml:
    philosophy: "MLOps-focused pipeline framework with integrations"
    strengths:
      - "ML-specific abstractions (Model, Artifact, Experiment)"
      - "Pluggable stack (swap orchestrator, artifact store, model deployer)"
      - "Simple decorator-based API"
      - "Built-in model registry integration"
    weaknesses:
      - "Newer, smaller community"
      - "Less battle-tested at hyperscale"
    best_for: "Small-medium ML teams wanting opinionated MLOps pipeline"
```

### Reusable Pipeline Components

```python
# Reusable ML pipeline components — platform provides these

"""
Standardized, tested components that any team can use in their pipelines.
Each component: validated inputs, standard outputs, proper error handling.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd


# --- Component Interface ---
@dataclass
class ComponentOutput:
    """Standard output from any pipeline component."""
    data: any  # Primary output (DataFrame, model, metrics)
    metadata: dict  # Metadata about the execution
    metrics: dict  # Quality metrics
    artifacts: list  # Paths to stored artifacts


# --- Data Validation Component ---
class DataValidationComponent:
    """
    Validates data quality before training or serving.
    Uses Great Expectations under the hood.
    
    Platform provides this so teams don't each build their own
    data validation logic.
    """
    
    def __init__(self, schema_path: str):
        self.schema = self._load_schema(schema_path)
    
    def validate(
        self,
        data: pd.DataFrame,
        fail_on_error: bool = True,
    ) -> ComponentOutput:
        """
        Validate data against schema.
        
        Checks:
        - Schema compliance (columns, types)
        - Null rates (within acceptable bounds)
        - Value distributions (no extreme outliers)
        - Row count (minimum viable dataset)
        """
        results = {
            "schema_valid": self._check_schema(data),
            "null_rates": self._check_nulls(data),
            "distributions": self._check_distributions(data),
            "row_count": len(data),
            "row_count_sufficient": len(data) >= self.schema.get("min_rows", 100),
        }
        
        is_valid = all([
            results["schema_valid"],
            results["row_count_sufficient"],
            not results["null_rates"]["has_violations"],
        ])
        
        if not is_valid and fail_on_error:
            raise DataValidationError(
                f"Data validation failed: {self._format_errors(results)}"
            )
        
        return ComponentOutput(
            data=data,
            metadata={"validation_passed": is_valid},
            metrics=results,
            artifacts=[],
        )


# --- Feature Engineering Component ---
class FeatureEngineeringComponent:
    """
    Loads and transforms features from the feature store.
    Standard interface for any model to get its features.
    """
    
    def __init__(self, feature_store_url: str):
        self.feature_store = self._connect(feature_store_url)
    
    def get_training_features(
        self,
        feature_set: str,
        entity_df: pd.DataFrame,
        timestamp_column: str = "event_timestamp",
    ) -> ComponentOutput:
        """
        Point-in-time correct feature retrieval for training.
        Prevents feature leakage by joining at correct timestamps.
        """
        features = self.feature_store.get_historical_features(
            feature_refs=self._get_feature_refs(feature_set),
            entity_df=entity_df,
            timestamp_column=timestamp_column,
        ).to_df()
        
        return ComponentOutput(
            data=features,
            metadata={
                "feature_set": feature_set,
                "num_features": len(features.columns),
                "num_rows": len(features),
            },
            metrics={
                "null_rate": features.isnull().mean().to_dict(),
                "feature_coverage": (1 - features.isnull().mean()).mean(),
            },
            artifacts=[],
        )


# --- Model Training Component ---
class ModelTrainingComponent:
    """
    Standardized model training wrapper.
    Handles: experiment tracking, checkpointing, distributed training.
    """
    
    def __init__(self, experiment_name: str, tracking_uri: str):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
    
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        model_config: dict,
        resources: dict = None,
    ) -> ComponentOutput:
        """
        Train model with full experiment tracking.
        
        Automatically:
        - Logs hyperparameters
        - Tracks training metrics
        - Saves best checkpoint
        - Records resource usage
        """
        import mlflow
        
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run():
            # Log configuration
            mlflow.log_params(model_config)
            
            # Train (framework-agnostic wrapper)
            model, metrics = self._train_model(
                train_data, val_data, model_config, resources
            )
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Save model artifact
            model_path = self._save_model(model)
            mlflow.log_artifact(model_path)
            
            return ComponentOutput(
                data=model,
                metadata={
                    "run_id": mlflow.active_run().info.run_id,
                    "model_path": model_path,
                },
                metrics=metrics,
                artifacts=[model_path],
            )


# --- Model Evaluation Component ---
class ModelEvaluationComponent:
    """
    Standardized model evaluation and comparison.
    Evaluates new model against current production model.
    """
    
    def evaluate(
        self,
        model: any,
        test_data: pd.DataFrame,
        production_model: Optional[any] = None,
        minimum_metrics: dict = None,
    ) -> ComponentOutput:
        """
        Evaluate model and optionally compare to production.
        
        Gates:
        - Must meet minimum metrics (configurable per model)
        - Must not regress on any metric vs production (configurable)
        """
        # Evaluate new model
        new_metrics = self._compute_metrics(model, test_data)
        
        comparison = None
        passed_gates = True
        gate_results = {}
        
        # Check minimum metrics gate
        if minimum_metrics:
            for metric_name, min_value in minimum_metrics.items():
                actual = new_metrics.get(metric_name, 0)
                passed = actual >= min_value
                gate_results[f"min_{metric_name}"] = {
                    "passed": passed,
                    "required": min_value,
                    "actual": actual,
                }
                if not passed:
                    passed_gates = False
        
        # Compare to production model
        if production_model:
            prod_metrics = self._compute_metrics(production_model, test_data)
            comparison = {
                metric: {
                    "new": new_metrics[metric],
                    "production": prod_metrics[metric],
                    "improvement": new_metrics[metric] - prod_metrics[metric],
                }
                for metric in new_metrics
            }
        
        return ComponentOutput(
            data={"passed_gates": passed_gates, "comparison": comparison},
            metadata={"gate_results": gate_results},
            metrics=new_metrics,
            artifacts=[],
        )
```

### Pipeline Templates

```yaml
Templates:
  train_evaluate_deploy:
    description: "Standard ML training pipeline"
    steps:
      1_data_validation: "Validate input data quality"
      2_feature_engineering: "Load features from feature store"
      3_data_split: "Train/validation/test split"
      4_model_training: "Train model with experiment tracking"
      5_evaluation: "Evaluate against production model"
      6_gates: "Pass quality gates (accuracy, fairness, latency)"
      7_registry: "Register model in model registry"
      8_deploy: "Deploy to staging, then canary, then production"
    parameters:
      - "model_config (hyperparameters)"
      - "data_source (feature set name)"
      - "minimum_metrics (gate thresholds)"
      - "deployment_strategy (canary %, rollback threshold)"
      
  batch_inference:
    description: "Scheduled batch prediction pipeline"
    steps:
      1_data_load: "Load entities for scoring"
      2_feature_fetch: "Get features from feature store"
      3_data_validation: "Validate features"
      4_inference: "Run model on all entities"
      5_post_process: "Apply business rules, thresholds"
      6_output: "Write predictions to output table"
      7_monitoring: "Log prediction distribution for drift monitoring"
    schedule: "Daily at 2 AM"
    
  feature_pipeline:
    description: "Feature computation and materialization"
    steps:
      1_source_data: "Read from data warehouse"
      2_transformations: "Apply feature transformations"
      3_validation: "Validate computed features"
      4_materialize: "Write to feature store (online + offline)"
    schedule: "Hourly or event-triggered"
    
  continuous_training:
    description: "Automated retraining triggered by monitoring"
    steps:
      1_trigger_check: "Evaluate retraining triggers"
      2_data_preparation: "Prepare training data (recent window)"
      3_training: "Train new model version"
      4_evaluation: "Compare to current production model"
      5_validation_gates: "Must pass all quality gates"
      6_deployment: "Shadow → canary → production (with rollback)"
    trigger: "Performance monitoring alert or scheduled"
```

### Pipeline Observability

```yaml
Observability:
  per_pipeline:
    metrics:
      - "Success/failure rate (7-day rolling)"
      - "Execution duration (p50, p95, p99)"
      - "Data volume processed"
      - "Resource usage (CPU, GPU, memory)"
      - "Cost per run"
    alerts:
      - "Pipeline failed (immediate notification)"
      - "Pipeline running longer than 2× historical average"
      - "Data volume anomaly (significantly more/less than expected)"
      
  across_pipelines:
    dashboard:
      - "Fleet health: % of pipelines healthy (all runs succeeded in last 24h)"
      - "SLA compliance: % of pipelines meeting their schedule"
      - "Resource utilization: cluster utilization by pipelines"
      - "Cost breakdown: daily/weekly spend by pipeline and team"
    alerts:
      - "Fleet health below 90%"
      - "Total pipeline compute spend exceeds budget"
      
  lineage:
    what: "Track data flow across pipelines"
    benefit: "When source data changes, know which pipelines and models are affected"
    implementation: "Dagster assets, OpenLineage, or custom metadata tracking"
```

---

## How It Works in Practice

### Team Onboarding

```yaml
Onboarding:
  step_1: "Team selects pipeline template (e.g., train-evaluate-deploy)"
  step_2: "Customize parameters (data source, model config, gates)"
  step_3: "Platform provides: orchestration, compute, monitoring, alerting"
  step_4: "Team focuses on: model code, feature definitions, evaluation criteria"
  
  time_to_first_pipeline:
    without_platform: "2-4 weeks (build infra, set up monitoring, handle deployment)"
    with_platform: "1-2 days (use template, fill in model-specific code)"
```

---

## Interview Tip

> When asked about pipeline platforms: "My pipeline platform provides three layers: (1) Shared orchestration — one orchestrator (Dagster or Kubeflow Pipelines) for all teams, centrally maintained, with proper scheduling, retry, and monitoring. Teams don't manage their own Airflow instances. (2) Reusable components — data validation (Great Expectations wrapper), feature store integration, experiment tracking wrapper, model evaluation with quality gates, and deployment automation. Each component handles logging, error handling, and observability. Teams compose these into pipelines. (3) Templates for common patterns — train-evaluate-deploy, batch inference, feature computation, continuous training. A new team goes from zero to production pipeline in 1-2 days (vs. 2-4 weeks without platform). Key design choices: components must be independently testable, each pipeline has automatic observability (duration, success rate, data volume, cost), and quality gates are mandatory (no deployment without validation). I chose Dagster for its asset-oriented model — you declare what data should exist and the system figures out what to compute. This gives you lineage for free: when upstream data changes, you know exactly which models are affected."

---

## Common Mistakes

1. **Each team runs their own orchestrator** — Team A has Airflow, Team B has Dagster, Team C has cron jobs. Platform team supports three different systems, no shared components, inconsistent monitoring. Solution: standardize on one orchestrator for the organization. Migration is painful but worth it.

2. **No pipeline templates** — Every team builds from scratch. First pipeline takes 3 weeks of boilerplate (orchestration setup, error handling, monitoring, alerting). Solution: templates that get teams to first pipeline in 1-2 days. Handle the infrastructure, let teams focus on model code.

3. **No quality gates** — Pipeline trains model and deploys directly. Bad training run (data issue, hyperparameter bug) → bad model in production. Solution: mandatory evaluation gates between training and deployment. New model must be >= production model performance.

4. **Monolithic pipelines** — Single giant pipeline does everything (data processing, feature engineering, training, evaluation, deployment) in one DAG. Impossible to test pieces independently, failure anywhere restarts everything. Solution: break into composable stages. Each stage independently testable and restartable.

5. **No cost visibility** — Pipelines run daily, nobody knows which ones cost the most. Monthly bill surprise: $50K, and the pipeline computing features nobody uses costs $15K. Solution: per-pipeline cost tracking, regular review of pipeline ROI (does the cost justify the value?).

---

## Key Takeaways

- Pipeline platform = shared orchestration + reusable components + templates + observability
- Orchestrator choice 2026: Dagster (assets, modern DX), Kubeflow (K8s-native ML), Prefect (Pythonic), Airflow (battle-tested)
- Reusable components: data validation, feature loading, model training wrapper, evaluation gates
- Templates: train-evaluate-deploy, batch inference, feature computation, continuous training
- Quality gates: mandatory validation between training and deployment
- Observability: per-pipeline metrics (duration, success rate, cost) + fleet-level health
- Lineage: track data flow across pipelines (know impact of upstream changes)
- Standardize: one orchestrator for the organization (not one per team)
- Time to first pipeline: 1-2 days with platform (vs. 2-4 weeks without)
- Cost tracking: per-pipeline cost attribution, regular ROI review
