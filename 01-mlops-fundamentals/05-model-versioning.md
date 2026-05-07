# Model Versioning

## The Problem / Why This Matters

In traditional software, you version code with Git and that's sufficient — the code deterministically produces the binary. In ML, the "binary" (trained model) depends on code AND data AND configuration AND random seeds AND hardware. Two identical training runs with the same code can produce different models if the data changed, the random seed differed, or the training ran on different hardware. Model versioning means tracking not just the model artifact (weights, config), but the complete provenance — every input that produced this model. Without it, you can't answer: "What changed between model v7 and v8?" "Can we reproduce the model from 3 months ago?" "Which data version was used for the model currently in production?" In 2026, with teams managing hundreds of model versions across traditional ML and LLM (Large Language Model) fine-tuned adapters (LoRA weights, prompt configurations), robust model versioning is the foundation of ML governance and reproducibility.

---

## The Analogy

Think of model versioning like a wine cellar's records:

- **Without records** = You have 200 bottles with no labels. You know some are great and some are bad, but you can't tell which vineyard, which harvest year, which fermentation process produced each one. You can't replicate a good year.
- **With records** = Every bottle is labeled with: vineyard (data source), harvest year (data version), grape variety (model architecture), fermentation process (training config), barrel type (hardware), and tasting notes (evaluation metrics). You can trace any bottle back to its origins and replicate the process.
- **The model registry** = The cellar catalog. Not just the bottles themselves, but which bottles are being served at the restaurant (production), which are aging (staging), and which have been retired.

---

## Deep Dive

### What Constitutes a Model Version

```yaml
Model_Version_Components:
  artifacts:
    model_weights: "The trained parameters (serialized tensors, pickle, ONNX)"
    preprocessing: "Scalers, encoders, tokenizers — transformations applied to input"
    configuration: "Hyperparameters, architecture definition, feature list"
    inference_code: "Code needed to load model and generate predictions"
    
  provenance:
    code_version: "Git commit hash — exact code used for training"
    data_version: "DVC hash, dataset tag, or data warehouse snapshot ID"
    environment: "Docker image digest, Python version, package versions (pip freeze)"
    training_config: "All hyperparameters, random seeds, hardware spec"
    
  evaluation:
    metrics: "All evaluation metrics on test set"
    sliced_metrics: "Performance broken down by subgroups"
    comparison: "How this version compares to previous and to champion"
    
  metadata:
    author: "Who triggered the training"
    timestamp: "When training completed"
    experiment_run: "Link to MLflow/W&B run with full training details"
    model_card: "Documentation of capabilities, limitations, fairness analysis"
    
  lifecycle:
    stage: "Development → Staging → Production → Archived"
    aliases: "champion, challenger, rollback_target, latest"
    approvals: "Who approved stage transitions, when, with what justification"
```

### Model Versioning Tools

```yaml
Versioning_Tools:
  mlflow_model_registry:
    description: "Integrated with MLflow experiment tracking — version + lifecycle management"
    workflow:
      - "Train model, log to MLflow experiment"
      - "Register model (creates versioned artifact)"
      - "Manage stages via API or UI"
      - "Deploy using alias-based loading"
    storage: "Model artifacts in artifact store (S3, GCS, HDFS, local)"
    lineage: "Automatic link to experiment run (params, metrics, code)"
    code_example: |
      import mlflow
      
      # Register model (creates version automatically)
      mlflow.register_model(
          model_uri=f"runs:/{run_id}/model",
          name="churn-predictor"
      )
      
      # Load specific version
      model_v3 = mlflow.pyfunc.load_model("models:/churn-predictor/3")
      
      # Load by alias (production pointer)
      champion = mlflow.pyfunc.load_model("models:/churn-predictor@champion")
      
  dvc:
    description: "Data Version Control — version large files (data + models) alongside Git"
    how: "Stores large files in remote storage, Git tracks metadata (hash + remote)"
    workflow:
      - "dvc add model.pkl → creates model.pkl.dvc (tracked in Git)"
      - "git commit → versions the metadata"
      - "dvc push → uploads artifact to remote (S3, GCS)"
      - "dvc pull → downloads artifact from remote"
    strengths:
      - "Git-native workflow (model versions tied to code commits)"
      - "Supports large files (GB-scale models)"
      - "Reproducibility (checkout Git commit → dvc pull → exact model)"
      - "Works with any file format"
    limitation: "No lifecycle management (stages, aliases) — pair with model registry"
    
  hugging_face_hub:
    description: "Model hosting and versioning platform — dominant for LLMs and transformers"
    workflow:
      - "Push model to Hub (auto-versions with Git LFS)"
      - "Tag versions (v1.0, production, latest)"
      - "Pull specific versions for serving"
    strengths:
      - "Largest model ecosystem (millions of models)"
      - "Built-in model cards"
      - "Easy sharing and collaboration"
      - "Integrated with transformers library"
    use_case: "LLM fine-tuned adapters, transformer models, embeddings"
    
  onnx:
    description: "Open Neural Network Exchange — framework-agnostic model format"
    purpose: "Convert models from training framework (PyTorch) to portable format"
    versioning: "ONNX files versioned via DVC, MLflow, or model registry"
    benefit: "Decouple training framework from serving framework"
```

### Versioning Strategies

```yaml
Versioning_Strategies:
  semantic_versioning:
    pattern: "MAJOR.MINOR.PATCH"
    ml_interpretation:
      major: "Architecture change, feature set change, breaking API change"
      minor: "Retrained on new data, hyperparameter tuning, minor improvement"
      patch: "Bug fix in preprocessing, config correction"
    example: "v2.3.1 = 2nd architecture, 3rd retraining, 1st bug fix"
    benefit: "Communicates scope of change to consumers"
    
  auto_incrementing:
    pattern: "Integer version (v1, v2, v3, ...)"
    when: "Every training run produces a new version"
    benefit: "Simple, no ambiguity about which is newer"
    limitation: "Doesn't communicate what changed"
    
  hash_based:
    pattern: "Content-addressed hash of model artifact"
    when: "Exact reproducibility is critical (regulated environments)"
    benefit: "Identical inputs always produce same version identifier"
    limitation: "Not human-readable"
    
  alias_based:
    pattern: "Human-readable names pointing to versions"
    examples: "champion, challenger, canary, rollback, latest, staging"
    benefit: "Deployment config references alias, not version number"
    key_advantage: "Promote model by updating alias — no deployment config change needed"
    
  recommended_approach:
    strategy: "Auto-incrementing versions + semantic aliases"
    implementation:
      - "Every training creates version N (auto-increment)"
      - "Aliases 'champion', 'challenger', 'rollback' point to specific versions"
      - "Deployment loads models:/model-name@champion"
      - "Promotion = move alias to new version"
      - "Rollback = move alias back to previous version"
```

### LLM Model Versioning (2026 Patterns)

```yaml
LLM_Versioning:
  challenge: "LLM 'model' is composite: base model + adapter + prompt + RAG config"
  
  what_to_version:
    base_model:
      what: "Foundation model reference (not the weights — too large to store)"
      example: "meta-llama/Llama-4-8B (reference, not copy)"
      versioning: "Version string or model card link"
      
    adapter_weights:
      what: "LoRA/QLoRA fine-tuned adapter (small: 10-500 MB)"
      storage: "MLflow, Hugging Face Hub, or model registry"
      versioning: "Each fine-tuning run produces new adapter version"
      
    prompt_templates:
      what: "System prompts, few-shot examples, instruction templates"
      storage: "Version controlled in Git or prompt management platform"
      challenge: "Small text changes can dramatically affect output"
      best_practice: "Version prompts like code — PR review, testing, staged rollout"
      
    rag_configuration:
      what: "Embedding model, chunk strategy, retrieval parameters, knowledge base version"
      storage: "Configuration versioned in Git, index versioned separately"
      components:
        - "Embedding model version"
        - "Chunk size and overlap settings"
        - "Top-k retrieval parameter"
        - "Re-ranking model version"
        - "Knowledge base document versions"
        
  composite_versioning:
    approach: "Register deployable unit that combines all components"
    example: |
      model_version:
        name: "customer-support-agent"
        version: "v12"
        components:
          base_model: "meta-llama/Llama-4-8B"
          adapter: "adapters/customer-support-v3.safetensors"
          system_prompt: "prompts/support-v7.txt"
          rag_config:
            embedding_model: "text-embedding-3-small"
            index_version: "2026-04-15"
            top_k: 5
            chunk_size: 512
    benefit: "Single version ID = complete deployable configuration"
```

---

## How It Works in Practice

### Model Versioning Workflow

```yaml
Example:
  project: "Search ranking model"
  
  versioning_setup:
    registry: "MLflow Model Registry"
    data_versioning: "DVC (training data tracked alongside code)"
    code: "Git (feature engineering, training script, config)"
    
  workflow:
    development:
      - "Data scientist creates branch, modifies feature engineering"
      - "DVC tracks new training data version"
      - "Training produces new model → logged to MLflow experiment"
      - "Model passes evaluation gates → registered as new version (v15)"
      
    promotion:
      - "v15 assigned 'challenger' alias"
      - "Shadow deployment: challenger predictions logged alongside champion (v12)"
      - "After 48 hours: challenger shows 3% NDCG improvement"
      - "ML lead approves: v15 promoted to 'champion' alias"
      - "v12 reassigned to 'rollback_target' alias"
      
    serving:
      - "Serving infrastructure loads: models:/search-ranker@champion"
      - "Resolves to v15 automatically"
      - "No code change, no redeployment — alias swap only"
      
    rollback:
      - "Day 3: latency spike detected on v15"
      - "Auto-rollback: 'champion' alias moved back to v12 (rollback_target)"
      - "Recovery time: 2 minutes (alias swap, model already cached)"
      - "Team investigates v15 latency issue"
      
  lineage_query:
    question: "What produced the current production model?"
    answer:
      model: "v12 (rolled back from v15)"
      training_run: "mlflow-run-abc123"
      code: "git commit 7f8e9a2"
      data: "dvc version hash 4d5e6f7 (transactions 2025-07 to 2026-03)"
      metrics: "NDCG@10: 0.78, MRR: 0.65"
      trained_on: "4x A100 GPUs, 6 hours"
```

---

## Interview Tip

> When asked about model versioning: "I version models with full provenance — not just the weights, but the complete reproducibility chain: code commit (Git), data version (DVC), training configuration, environment (Docker image hash), and evaluation metrics. For lifecycle management, I use MLflow Model Registry with alias-based deployment: 'champion' points to production model, 'challenger' to the candidate being tested, 'rollback_target' to the previous good version. Promotion is an alias swap — no code change needed in serving infrastructure. For LLM applications, the 'model version' is composite: base model reference + adapter weights + prompt template + RAG config — all versioned together as a single deployable unit. Key principles: every model must be reproducible from versioned inputs, every stage transition must be auditable, and rollback must be instant (alias swap, not retraining)."

---

## Common Mistakes

1. **Versioning model files without provenance** — Storing model.pkl v1, v2, v3 without recording what data, code, and config produced each version. Models become black boxes — you can serve them but can't reproduce, explain, or improve them.

2. **Hardcoded version numbers in deployment** — Serving code says `load_model("models:/ranker/7")`. Every promotion requires a code change and redeployment. Use aliases (`@champion`) so promotion is just updating a pointer.

3. **Not versioning data alongside models** — Versioning model artifacts without the data that produced them. You can't reproduce the model if the training data isn't versioned and linked.

4. **No rollback plan** — Deploying new model versions without maintaining a rollback path. When production model fails, recovery means retraining (hours) instead of alias swap (seconds).

5. **Treating LLM prompts as informal text** — Not versioning prompt templates with the same rigor as model weights. A one-word prompt change can dramatically alter model behavior. Version prompts, test them, and roll them out progressively.

---

## Key Takeaways

- Model version = artifact (weights) + provenance (code, data, config, environment) + evaluation (metrics)
- MLflow Model Registry: standard for lifecycle management with alias-based deployment
- DVC: version large data/model files alongside Git (reproducibility foundation)
- Alias-based deployment: `@champion`, `@challenger`, `@rollback` — promotion is pointer swap, not redeployment
- Full reproducibility: given a version, reproduce exact model from versioned inputs (code + data + config)
- LLM versioning is composite: base model + adapter + prompt + RAG config = one deployable unit
- Every stage transition (dev → staging → production) should be auditable (who, when, why)
- Rollback must be instant (alias swap, cached models) — never require retraining
- Version prompts with same rigor as model weights — small text changes have large behavior effects
- Lineage tracking: trace any production prediction back to exact data, code, and training run
