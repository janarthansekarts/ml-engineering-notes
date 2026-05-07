# What is MLOps

## The Problem / Why This Matters

Building an ML model in a Jupyter notebook is easy. Getting that model into production, serving reliable predictions at scale, monitoring its performance, retraining it when it degrades, and doing all of this repeatably across dozens of models — that's where organizations fail. MLOps (Machine Learning Operations) is the set of practices, tools, and cultural norms that bridges the gap between ML development and ML in production. It's DevOps applied to ML, but with unique challenges: data is a first-class dependency, models degrade silently (no crash — just worse predictions), experimentation is non-deterministic, and the boundary between "code" and "learned behavior" (model weights) is blurry. Without MLOps, you get the "last mile" problem: data scientists build great models that never reach users, or models reach users but nobody knows when they break. In 2026, MLOps has matured from a buzzword to a concrete discipline with established tools (MLflow, Kubeflow, Feast, Evidently), well-defined maturity levels, and clear organizational patterns.

---

## The Analogy

Think of MLOps like the manufacturing process for a physical product:

- **Data science without MLOps** = A master craftsman making one-of-a-kind watches by hand. Beautiful, but slow, unreproducible, and doesn't scale.
- **ML with basic MLOps** = A small factory. Assembly line (pipeline), quality control (testing), inventory management (model registry). Produces consistent watches at modest scale.
- **Mature MLOps** = An automated factory with robotics. Raw materials flow in (data), products flow out (predictions), quality is monitored in real-time, defective batches are caught instantly, and the entire line can be reconfigured for a new product in hours (new model deployment).

DevOps automated software delivery. MLOps automates ML delivery — but with the added complexity that the "product" (model) changes even when the code doesn't (because the data changes).

---

## Deep Dive

### MLOps vs DevOps vs DataOps

```yaml
Comparison:
  DevOps:
    focus: "Automate software delivery and infrastructure"
    artifacts: "Code → binary/container"
    trigger: "Code commit"
    testing: "Unit, integration, e2e tests"
    monitoring: "Uptime, error rates, latency"
    versioning: "Git (code only)"
    
  MLOps:
    focus: "Automate ML model lifecycle — training, deployment, monitoring"
    artifacts: "Code + Data + Model → prediction service"
    trigger: "Code commit OR data change OR performance degradation"
    testing: "Code tests + data tests + model tests + fairness tests"
    monitoring: "All DevOps metrics + data drift + model performance + prediction quality"
    versioning: "Git (code) + DVC/LakeFS (data) + Model Registry (models)"
    unique_challenges:
      - "Data is a first-class dependency (not just code)"
      - "Models degrade silently (no exception — just worse predictions)"
      - "Experimentation is non-deterministic (same code + data can give different results)"
      - "Training is expensive (GPU hours, not just CPU seconds)"
      - "Model behavior is learned, not programmed (harder to test/debug)"
      
  DataOps:
    focus: "Automate data pipeline delivery and quality"
    artifacts: "Data pipelines → datasets"
    trigger: "Schedule, event, data arrival"
    testing: "Schema validation, data quality checks"
    monitoring: "Pipeline health, data freshness, completeness"
    versioning: "Schema versioning, data catalog"
    
  relationship: "MLOps builds ON DevOps and DataOps — it requires both as foundations"
```

### MLOps Maturity Levels

```yaml
Maturity_Levels:
  level_0_manual:
    description: "Everything done manually — notebooks, manual deployment, no automation"
    characteristics:
      - "Data scientists train models in Jupyter notebooks"
      - "Deployment is manual (copy files, restart service)"
      - "No pipeline — steps executed manually in order"
      - "No monitoring — discover problems from user complaints"
      - "No reproducibility — 'it worked on my machine'"
    pain_points:
      - "Months from model development to production"
      - "Can't reproduce past results"
      - "Model performance unknown until users complain"
      - "Data scientist is the only one who can retrain"
    typical_at: "Early-stage companies, first ML projects"
    
  level_1_pipeline:
    description: "Automated training pipeline, basic model serving, some monitoring"
    characteristics:
      - "Training pipeline automated (trigger → data → train → evaluate → register)"
      - "Model serving via standard framework (KServe, TorchServe)"
      - "Basic monitoring (latency, error rate — not model quality)"
      - "Experiment tracking (MLflow/W&B)"
      - "Some testing (unit tests on training code)"
    improvements_over_0:
      - "Reproducible training (pipeline can be re-run)"
      - "Multiple people can trigger training"
      - "Models deployed consistently"
    gaps:
      - "Retraining still manually triggered"
      - "No data drift monitoring"
      - "No automated model validation before deployment"
    typical_at: "Companies with 1-3 production models, dedicated ML team"
    
  level_2_automated:
    description: "Continuous training, comprehensive monitoring, automated deployment gates"
    characteristics:
      - "CI/CD for ML (code + data + model versioning)"
      - "Automated retraining triggers (schedule or drift-based)"
      - "Data quality gates (pipeline won't proceed with bad data)"
      - "Model validation gates (won't deploy if metrics below threshold)"
      - "Comprehensive monitoring (data drift, prediction quality, fairness)"
      - "Feature store for consistent feature serving"
      - "Automated A/B testing and progressive rollout"
    improvements_over_1:
      - "Models stay fresh automatically"
      - "Issues detected before users notice"
      - "Deployment is safe (validation gates)"
    typical_at: "Companies with 5-20 production models, dedicated ML platform team"
    
  level_3_full_automation:
    description: "Self-healing systems, automated feature engineering, ML-optimized infrastructure"
    characteristics:
      - "Automated model selection and hyperparameter optimization"
      - "Self-healing: drift detected → auto-retrain → auto-deploy"
      - "Feature discovery and automated feature engineering"
      - "Multi-model management with automated champion-challenger"
      - "Full lineage: data → features → model → predictions traceable"
      - "Cost-optimized infrastructure (auto-scaling, spot instances)"
    typical_at: "Large tech companies, ML-native organizations"
```

### Core MLOps Components

```yaml
MLOps_Components:
  source_control:
    what: "Version code, data, configs, and model artifacts"
    tools: "Git (code), DVC/LakeFS (data), MLflow Model Registry (models)"
    principle: "Everything reproducible from versioned inputs"
    
  training_pipeline:
    what: "Automated sequence: data prep → feature engineering → training → evaluation"
    tools: "Kubeflow Pipelines, Airflow, Dagster, Vertex AI Pipelines, SageMaker Pipelines"
    principle: "Same pipeline runs identically in development and production"
    
  continuous_integration:
    what: "Test code, data, and model on every change"
    tests:
      code_tests: "Unit tests, linting, type checking"
      data_tests: "Schema validation, distribution checks, freshness"
      model_tests: "Performance on held-out set, behavioral tests, fairness"
    tools: "GitHub Actions, GitLab CI, Jenkins + pytest + Great Expectations"
    
  continuous_delivery:
    what: "Deploy validated models to production safely"
    strategies: "Shadow → canary → blue-green → full rollout"
    gates: "Metric thresholds, approval workflows, compliance checks"
    tools: "ArgoCD, KServe, Seldon, SageMaker Endpoints"
    
  continuous_training:
    what: "Automatically retrain when conditions warrant"
    triggers:
      - "Schedule (daily, weekly, monthly)"
      - "Data drift detected (input distribution shifted)"
      - "Performance degradation (metrics below threshold)"
      - "New data available (significant volume increase)"
    tools: "Kubeflow Pipelines + scheduler, Vertex AI Continuous Training"
    
  monitoring:
    what: "Observe model behavior in production"
    dimensions:
      operational: "Latency, throughput, error rate, resource utilization"
      data_quality: "Input distribution, missing values, schema violations"
      model_quality: "Prediction distribution, drift metrics, ground truth comparison"
      business: "Business KPI impact (CTR, revenue, user satisfaction)"
    tools: "Evidently, Arize, WhyLabs, Fiddler, Prometheus + Grafana"
    
  feature_store:
    what: "Centralized feature management — compute once, serve consistently"
    purpose: "Eliminate training-serving skew, enable feature sharing"
    tools: "Feast, Tecton, Vertex AI Feature Store, SageMaker Feature Store"
    
  model_registry:
    what: "Single source of truth for all model artifacts and lifecycle"
    purpose: "Version, stage, approve, deploy, and rollback models"
    tools: "MLflow Model Registry, Vertex AI Model Registry, SageMaker Model Registry"
```

---

## How It Works in Practice

### MLOps Workflow Example

```yaml
Example:
  system: "Real-time fraud detection (Level 2 maturity)"
  
  workflow:
    trigger: "Weekly schedule OR drift alert"
    
    step_1_data:
      action: "Pull latest transaction data from data warehouse"
      validation: "Great Expectations checks (completeness, schema, distributions)"
      gate: "If data quality fails → alert team, skip training"
      
    step_2_features:
      action: "Compute features via feature pipeline (Spark + Feast)"
      validation: "Feature freshness check, completeness >99%"
      output: "Updated feature table in feature store"
      
    step_3_training:
      action: "Train XGBoost model on latest features + labels"
      tracking: "Log to MLflow (params, metrics, artifacts)"
      infrastructure: "CPU cluster (XGBoost doesn't need GPU)"
      
    step_4_evaluation:
      action: "Evaluate on held-out test set"
      metrics: "AUC, precision@95% recall, latency benchmark"
      gate: "AUC >= 0.93 AND precision >= 0.85 AND p99_latency < 10ms"
      failure: "If gate fails → alert team, keep current production model"
      
    step_5_registration:
      action: "Register model in MLflow, assign 'challenger' alias"
      metadata: "Data version, training date, evaluation results"
      
    step_6_deployment:
      action: "Deploy challenger in shadow mode"
      duration: "24 hours shadow deployment"
      comparison: "Compare challenger predictions vs champion on live traffic"
      
    step_7_promotion:
      action: "If shadow results positive → canary 5% → 25% → 100%"
      gate: "No degradation in live metrics during each stage"
      rollback: "Auto-rollback if live metrics drop >2% below champion"
      
  time_to_complete: "~6 hours automated (data → deployed), 24 hours shadow"
  human_involvement: "Only if gates fail or rollback triggered"
```

---

## Interview Tip

> When asked about MLOps: "MLOps is DevOps extended for ML — it addresses the unique challenges of ML systems: data as a first-class dependency, silent model degradation, expensive training compute, and the need for continuous retraining. I structure it across maturity levels: Level 0 (manual notebooks → manual deploy), Level 1 (automated pipelines, experiment tracking), Level 2 (continuous training, comprehensive monitoring, deployment gates), Level 3 (self-healing, auto-retraining). Key components: training pipelines (Kubeflow/Dagster), model registry (MLflow), feature store (Feast), monitoring (Evidently), and CI/CD adapted for ML (test data + model, not just code). The biggest mistake organizations make is under-investing in Level 1-2 infrastructure while over-investing in model complexity. A simple model with great MLOps beats a complex model deployed manually."

---

## Common Mistakes

1. **MLOps as an afterthought** — Building models first, then trying to "add MLOps later." By then, the model is entangled with notebooks, manual processes, and undocumented dependencies. Start with basic MLOps from the first model.

2. **Over-engineering at Level 0** — Trying to jump from nothing to Level 3 immediately. Start with Level 1 (automated training pipeline + experiment tracking), prove value, then incrementally add sophistication.

3. **Ignoring data ops** — Focusing all MLOps effort on model management while ignoring data quality, versioning, and freshness. Bad data = bad model regardless of how sophisticated your deployment pipeline is.

4. **Tool sprawl** — Adopting 10 different MLOps tools that don't integrate well. Better to start with fewer tools that work together (e.g., MLflow for tracking + registry, Kubeflow for pipelines, Feast for features) than a "best of breed" collection that requires custom glue.

5. **No monitoring = false confidence** — Deploying models without monitoring is driving blindfolded. Models degrade silently — the only way to know is to measure continuously.

---

## Key Takeaways

- MLOps = DevOps + DataOps applied to ML lifecycle (training, deployment, monitoring, retraining)
- Unique ML challenges: data dependency, silent degradation, expensive compute, non-determinism
- Maturity levels: 0 (manual) → 1 (pipeline) → 2 (automated) → 3 (self-healing) — progress incrementally
- Core components: pipelines, CI/CD, feature store, model registry, monitoring, continuous training
- Three triggers for action: code change, data change, performance degradation
- Testing pyramid for ML: data tests + code tests + model tests + integration tests + fairness tests
- Monitoring must cover: operational health, data quality, model quality, business impact
- Start with Level 1 quickly (pipeline + tracking), then build toward Level 2 with monitoring and gates
- Tools ecosystem: MLflow, Kubeflow, Feast, Evidently, DVC, KServe — choose cohesive stack
- MLOps investment has 10x more impact on production ML success than model architecture choices
