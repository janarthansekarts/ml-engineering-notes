# Model Registry

## The Problem / Why This Matters

In a production ML system, you have dozens of models across different versions, stages, and environments. Without a model registry, answering basic questions becomes impossible: "Which model is currently serving production traffic?" "What data was this model trained on?" "Who approved this model for deployment?" "If this model fails, which version should we roll back to?" "Does this model meet our fairness requirements?" A model registry is the single source of truth for all trained models — it stores versioned model artifacts with rich metadata, manages lifecycle stages (development → staging → production → archived), enforces governance workflows (approval gates, compliance checks), and provides lineage tracking (which data, code, and experiments produced this model). In 2026, with teams managing traditional ML models, fine-tuned LLMs (Large Language Models), LoRA adapters, and prompt configurations simultaneously, a model registry isn't just useful — it's a compliance requirement for many regulated industries.

---

## The Analogy

Think of a model registry like a pharmaceutical drug database:

- **Drug identity** = Model artifact (the actual trained weights and configuration)
- **Version history** = Each reformulation or dosage change (model v1, v2, v3)
- **Clinical trial data** = Evaluation metrics and test results (offline/online performance)
- **FDA approval status** = Lifecycle stage (development → staging → production)
- **Manufacturing record** = Lineage (which lab, which batch of ingredients, which process)
- **Recall capability** = Rollback (if side effects found, immediately pull from market)

Without this database, you'd have drugs with unknown compositions, untested dosages, and no way to recall dangerous batches. Same risk applies to ML models without a registry.

---

## Deep Dive

### Model Registry Core Concepts

```yaml
Model_Registry_Concepts:
  model:
    definition: "A named, registered ML model (e.g., 'fraud-detector', 'search-ranker')"
    contains: "Multiple versions over time"
    metadata: "Owner, team, description, use case, SLA requirements"
    
  model_version:
    definition: "A specific trained instance of a model"
    contains:
      artifacts: "Model weights, preprocessing objects, config files"
      metadata:
        training_data: "Dataset version, date range, size"
        code_version: "Git commit hash"
        experiment_run: "Link to experiment tracking run"
        metrics: "Evaluation results on test set"
        hardware: "GPU type, training duration"
      lineage: "Full trace from data → features → training → evaluation"
    
  lifecycle_stages:
    none: "Just registered — no stage assigned"
    development: "Under active development, not ready for testing"
    staging: "Ready for integration testing and shadow deployment"
    production: "Actively serving live traffic"
    archived: "Retired — kept for lineage, not serving"
    
  aliases:
    purpose: "Human-readable pointers to specific versions"
    examples:
      champion: "Current production model"
      challenger: "Candidate model being tested against champion"
      latest: "Most recently trained version"
      rollback_target: "Last known good version"
    advantage: "Deployment config references alias, not version number — swap without config change"
    
  governance:
    approval_workflow: "Model can't move to production without sign-off"
    approvers: "ML lead, product owner, compliance (for regulated domains)"
    required_checks:
      - "Evaluation metrics above threshold"
      - "Fairness testing passed"
      - "Latency profiling within SLA"
      - "Model card completed"
      - "No data quality alerts on training data"
```

### MLflow Model Registry

```yaml
MLflow_Model_Registry:
  description: "Most widely used open-source model registry (part of MLflow platform)"
  
  registration_workflow:
    step_1: "Train model and log to MLflow experiment tracking"
    step_2: "Register model (creates named model in registry)"
    step_3: "Add metadata, descriptions, tags"
    step_4: "Transition through stages with optional approval"
    
  code_example: |
    import mlflow
    from mlflow import MlflowClient
    
    # During training — log the model
    with mlflow.start_run() as run:
        model = train_model(params)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metrics({"auc": 0.94, "f1": 0.87})
    
    # Register the model
    model_uri = f"runs:/{run.info.run_id}/model"
    mv = mlflow.register_model(model_uri, "fraud-detector")
    
    # Manage lifecycle with client
    client = MlflowClient()
    
    # Add description and tags
    client.update_model_version(
        name="fraud-detector",
        version=mv.version,
        description="XGBoost v2 with behavioral features, trained on 2024-Q4 data"
    )
    client.set_model_version_tag(
        name="fraud-detector", version=mv.version,
        key="validation_status", value="passed"
    )
    
    # Set alias for deployment
    client.set_registered_model_alias(
        name="fraud-detector", alias="champion", version=mv.version
    )
    
  deployment_integration: |
    # Serving code loads by alias — no hardcoded version
    model = mlflow.pyfunc.load_model("models:/fraud-detector@champion")
    prediction = model.predict(features)
    
  strengths:
    - "Open source, self-hostable"
    - "Integrates with MLflow tracking (full lineage)"
    - "Alias system for zero-config deployment updates"
    - "Supports any framework (sklearn, pytorch, transformers, custom)"
    
  limitations:
    - "Basic approval workflow (no built-in multi-step approval)"
    - "Limited UI for governance (compared to enterprise solutions)"
    - "No built-in model card generation"
```

### Model Cards and Documentation

```yaml
Model_Cards:
  purpose: "Structured documentation about a model's capabilities, limitations, and appropriate use"
  origin: "Google's 'Model Cards for Model Reporting' (2018) — now industry standard"
  
  required_sections:
    model_details:
      - "Model name and version"
      - "Architecture and framework"
      - "Training date and data date range"
      - "Intended use cases"
      - "Out-of-scope use cases (things it shouldn't be used for)"
      
    training_data:
      - "Dataset description and size"
      - "Data collection methodology"
      - "Known biases or gaps in data"
      - "Preprocessing steps applied"
      
    evaluation:
      - "Test datasets used"
      - "Metrics and results (with confidence intervals)"
      - "Performance across subgroups (fairness)"
      - "Failure modes and edge cases"
      
    ethical_considerations:
      - "Potential harms and mitigations"
      - "Bias analysis results"
      - "Privacy considerations"
      - "Environmental impact (training compute)"
      
    limitations:
      - "Known limitations and failure modes"
      - "Scenarios where the model should NOT be trusted"
      - "Recommended human oversight level"
      
  automation:
    tools:
      - "Hugging Face Model Cards (integrated with Hub)"
      - "Google's Model Card Toolkit"
      - "Custom templates in model registry CI/CD"
    best_practice: "Auto-generate from experiment tracking data, human-fill limitations and ethics sections"
```

### Advanced Registry Patterns

```yaml
Advanced_Patterns:
  multi_model_registry:
    scenario: "Organization with 50+ models across 10 teams"
    structure:
      organization_level: "Central registry with all models"
      team_level: "Team-specific namespaces (team-fraud/detector, team-search/ranker)"
      access_control: "Teams own their models, platform team has read access to all"
      
  llm_in_registry:
    challenge: "LLM artifacts are 10-100+ GB, plus adapters, prompts, and configs"
    approach:
      base_model: "Register reference to base model (e.g., 'meta-llama/Llama-4-8B')"
      adapter: "Register LoRA adapter weights (small, <1 GB typically)"
      prompt_config: "Version prompt templates alongside model"
      rag_config: "Version retrieval settings (chunk size, top-k, embedding model)"
    versioning: "Base + Adapter + Prompt + RAG config = single deployable unit"
    
  automated_promotion:
    workflow:
      trigger: "New model version registered"
      step_1: "Automated evaluation pipeline runs (test set, fairness, latency)"
      step_2: "If passes thresholds → auto-promote to staging"
      step_3: "Shadow deployment for 24-48 hours (compare against champion)"
      step_4: "If shadow results positive → notify approver for production promotion"
      step_5: "Approver reviews and promotes (or rejects with reason)"
    tools: "GitHub Actions / GitLab CI + MLflow API + custom promotion logic"
    
  rollback_strategy:
    principle: "Every model in production must have a clear rollback target"
    implementation:
      - "Always maintain 'rollback_target' alias pointing to previous champion"
      - "Deployment system supports instant alias swap (no rebuild needed)"
      - "Automated rollback trigger: if live metrics drop below threshold, auto-revert"
      - "Rollback runbook documented and tested"
    speed: "Rollback should take <5 minutes from detection to recovery"
```

---

## How It Works in Practice

### Model Lifecycle Example

```yaml
Example:
  model: "Customer Churn Predictor"
  team: "Growth ML"
  
  lifecycle:
    version_1:
      trained: "2025-10-01"
      data: "Customer data 2024-01 to 2025-09"
      metrics: { auc: 0.82, f1: 0.71 }
      status: "Archived (superseded by v2)"
      
    version_2:
      trained: "2026-01-15"
      data: "Customer data 2024-06 to 2025-12 + behavioral features"
      metrics: { auc: 0.87, f1: 0.78 }
      status: "Production (champion alias)"
      approved_by: "ML Lead + Product Manager"
      model_card: "Complete with fairness analysis across customer segments"
      
    version_3:
      trained: "2026-04-20"
      data: "Customer data 2025-01 to 2026-03 + engagement features"
      metrics: { auc: 0.89, f1: 0.80 }
      status: "Staging (challenger — shadow deployment)"
      shadow_results: "2% improvement in prediction accuracy on live traffic"
      next_step: "Submit for production approval after 1 week shadow"
      
  rollback_plan:
    current_champion: "version_2"
    rollback_target: "version_1 (last known good before v2)"
    trigger: "AUC drops below 0.83 on daily evaluation"
    procedure: "Swap champion alias to rollback_target, page on-call engineer"
```

---

## Interview Tip

> When asked about model management: "I use a model registry as the single source of truth for all model artifacts and lifecycle. Key components: (1) Versioned artifacts — model weights, preprocessors, and configs stored with full lineage to training data and code. (2) Lifecycle stages — development → staging → production → archived, with governance gates between stages. (3) Aliases — deployment references 'champion' alias, not version numbers. Promotion is an alias swap, rollback is instant. (4) Model cards — structured documentation including performance, limitations, fairness analysis, and intended use. (5) Automated promotion — CI/CD pipeline evaluates new versions, auto-promotes to staging if metrics pass, then human approves for production. For LLM applications, I register the combination of base model + adapter + prompt config + RAG settings as a single deployable unit."

---

## Common Mistakes

1. **No registry at all** — Models stored as files on someone's laptop or a random S3 bucket. Nobody knows which version is in production, and rollback means "ask the person who deployed it."

2. **Registering without metadata** — Storing model artifacts without linking to training data, code, or metrics. The model becomes a black box — you can serve it but can't explain, reproduce, or improve it.

3. **No approval workflow** — Anyone can push any model to production without review. One bad model update can degrade the product for millions of users before anyone notices.

4. **Forgetting rollback** — Deploying a new model without maintaining a rollback path. When it fails in production (and it will eventually), recovery time is hours instead of minutes.

5. **Treating LLMs differently** — Not version-controlling prompt templates and RAG configurations alongside model artifacts. A "model" in an LLM application is the entire stack (base model + adapter + prompt + retrieval config), not just the weights.

---

## Key Takeaways

- Model registry = single source of truth for all model artifacts, metadata, and lifecycle
- Core concepts: model (named entity) → versions (specific trained instances) → stages (lifecycle) → aliases (deployment pointers)
- MLflow Model Registry: open-source standard, alias-based deployment, full lineage tracking
- Lifecycle: development → staging → production → archived, with governance gates
- Aliases enable instant rollback (swap "champion" pointer, no rebuild needed)
- Model cards: mandatory documentation of capabilities, limitations, fairness, and intended use
- For LLMs: register base model + adapter + prompt config + RAG settings as one unit
- Automated promotion: CI/CD evaluates → auto-stage → shadow test → human approves → production
- Always maintain a rollback target alias — recovery should take <5 minutes
