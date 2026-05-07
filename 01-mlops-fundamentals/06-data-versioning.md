# Data Versioning

## The Problem / Why This Matters

In ML, data is as important as code — often more so. A model's behavior is determined by the data it was trained on. Yet most teams version their code meticulously (Git) while treating data as mutable state: training datasets get overwritten, preprocessing logic changes without record, and nobody can answer "what data did model v5 train on?" Data versioning provides Git-like capabilities for datasets: track changes over time, reproduce any historical state, branch and experiment without affecting production data, and maintain full lineage from raw data through transformations to training-ready features. Without it, ML reproducibility is impossible, debugging data issues is guesswork, and regulatory compliance (explaining model decisions) fails. In 2026, data versioning has become especially critical as teams manage not just training data, but also RAG (Retrieval-Augmented Generation) knowledge bases, embedding indexes, evaluation datasets, and synthetic data — all of which need version control and lineage tracking.

---

## The Analogy

Think of data versioning like version control for a recipe book:

- **Code versioning only** = You track changes to the recipe instructions (Git), but not to the ingredient list. The instructions say "add flour" but you don't know if it was all-purpose flour (v1) or almond flour (v2). The cake comes out different each time, and you can't figure out why.
- **Data versioning** = You track both the recipe instructions AND the specific ingredient batches used. "Cake baked on Jan 15 used flour batch #A7, eggs batch #B3, oven temp 350°F." Months later, you can reproduce that exact cake or understand why February's batch tasted different (flour supplier changed).
- **DVC** = Git for ingredients. Each ingredient batch gets a tracking label (hash), the labels live in your recipe book (Git repo), and the actual ingredients are stored in the pantry (remote storage).

---

## Deep Dive

### Why Git Alone Isn't Enough for Data

```yaml
Git_Limitations_for_Data:
  file_size: "Git is designed for text files (KB). ML datasets are GB-TB. Git LFS helps but has limits."
  binary_diffs: "Git can't show meaningful diffs for binary data (Parquet, images, embeddings)"
  storage_cost: "Storing every version of a 50GB dataset in Git history is impractical"
  performance: "Clone, checkout, and diff operations become unusably slow with large files"
  
  what_we_need:
    - "Track large files (GB-TB) without bloating the repository"
    - "Show meaningful changes between versions (row counts, schema changes, distribution shifts)"
    - "Store actual data in cheap object storage (S3, GCS) — only metadata in Git"
    - "Support branching (experiment with data changes without affecting main)"
    - "Link data versions to code commits (reproducibility)"
```

### DVC (Data Version Control)

```yaml
DVC:
  description: "Open-source version control for data and ML pipelines — Git extension for large files"
  
  how_it_works:
    principle: "Git tracks metadata (hash pointers), DVC tracks actual data in remote storage"
    workflow:
      step_1: "dvc init → initializes DVC in Git repo"
      step_2: "dvc add data/train.parquet → computes hash, creates .dvc file"
      step_3: "git add data/train.parquet.dvc → Git tracks the pointer"
      step_4: "dvc push → uploads data to remote storage (S3, GCS, Azure Blob)"
      step_5: "git commit → data version is tied to code commit"
      
  file_structure:
    git_tracks: "data/train.parquet.dvc (small YAML file with hash + metadata)"
    dvc_remote: "Actual data stored in S3/GCS, addressed by content hash"
    gitignore: "data/train.parquet is in .gitignore (not tracked by Git)"
    
  key_features:
    content_addressing:
      description: "Files identified by content hash (SHA-256)"
      benefit: "Same data = same hash. Deduplication across versions."
      
    pipeline_tracking:
      description: "DVC can track full pipelines (data → preprocessing → features → model)"
      usage: "dvc.yaml defines pipeline stages, dvc repro runs changed stages"
      benefit: "Reproduce any historical pipeline state"
      
    experiments:
      description: "Branch, experiment, and compare data + model variations"
      usage: "dvc exp run → creates experiment with modified params/data"
      benefit: "Lightweight experiments without full Git branches"
      
    remote_storage:
      options: "S3, GCS, Azure Blob, SSH, HTTP, local directory"
      benefit: "Cheap, scalable storage for large datasets"
      
  commands: |
    # Initialize DVC
    dvc init
    
    # Track a large dataset
    dvc add data/training_data.parquet
    git add data/training_data.parquet.dvc .gitignore
    git commit -m "Add training data v1"
    dvc push  # Upload to remote
    
    # Later: update the data
    # (new data lands in data/training_data.parquet)
    dvc add data/training_data.parquet  # Updates hash
    git add data/training_data.parquet.dvc
    git commit -m "Update training data to v2 (added March 2026 data)"
    dvc push
    
    # Checkout historical version
    git checkout abc123  # Go back to previous commit
    dvc checkout         # Pull data that matches that commit
    # Now you have exact code + data from that point in time
    
  strengths:
    - "Git-native workflow (data versions tied to code commits)"
    - "Language/framework agnostic (works with any file)"
    - "Pipeline reproduction (dvc repro)"
    - "Lightweight experiments (dvc exp)"
    - "Large community, well-documented"
    
  limitations:
    - "No built-in data catalog/discovery"
    - "Branch-based (not table-level time travel)"
    - "Manual process (must remember to dvc add after data changes)"
```

### LakeFS

```yaml
LakeFS:
  description: "Git-like version control for data lakes — branch, commit, merge at scale"
  
  how_it_works:
    principle: "Git semantics (branch, commit, merge, diff) applied to object storage"
    architecture: "Sits between your application and S3/GCS, intercepts operations"
    access: "Standard S3 API — existing tools (Spark, Pandas) work without changes"
    
  key_features:
    branching:
      description: "Create branches of entire data lake"
      use_case: "Experiment with data transformations without affecting production"
      implementation: "Copy-on-write (branches are cheap — only changed objects are duplicated)"
      
    commits:
      description: "Atomic snapshots of data state"
      use_case: "Point-in-time reproducibility, audit trail"
      guarantee: "Either all changes in a commit succeed or none do (atomic)"
      
    merge:
      description: "Merge data changes from branch to main"
      use_case: "After validating data quality on branch, merge to production"
      conflict_handling: "Detects and reports conflicts for resolution"
      
    diff:
      description: "See what changed between commits/branches"
      granularity: "Object-level (which files added/modified/deleted)"
      
  workflow_example: |
    # Create branch for data experiment
    lakectl branch create lakefs://repo/experiment-new-features \
        --source lakefs://repo/main
    
    # Write experimental data to branch (standard S3 API)
    spark.write.parquet("s3a://repo/experiment-new-features/features/new_feature.parquet")
    
    # Validate quality on branch
    validate_data("lakefs://repo/experiment-new-features/features/")
    
    # If valid, merge to main
    lakectl merge lakefs://repo/experiment-new-features lakefs://repo/main
    
    # Commit with message
    lakectl commit lakefs://repo/main -m "Add new behavioral features"
    
  strengths:
    - "Zero code changes (S3 API compatible)"
    - "Works with any tool (Spark, Trino, dbt, pandas)"
    - "Branch-level isolation for experiments"
    - "Atomic operations (no partial updates)"
    - "Pre-commit hooks (run quality checks before commit)"
    
  best_for: "Data lake environments (S3/GCS) with multiple teams and large-scale data"
```

### Delta Lake Time Travel

```yaml
Delta_Lake:
  description: "Open-source storage layer with ACID transactions and time travel"
  
  time_travel:
    description: "Query any historical version of a Delta table"
    implementation: "Transaction log records every change — read any version by replaying log"
    
    usage: |
      # Read current version
      df = spark.read.format("delta").load("/data/features/")
      
      # Read as of specific version
      df_v5 = spark.read.format("delta") \
          .option("versionAsOf", 5) \
          .load("/data/features/")
      
      # Read as of timestamp
      df_historical = spark.read.format("delta") \
          .option("timestampAsOf", "2026-01-15") \
          .load("/data/features/")
      
      # Show version history
      history = spark.sql("DESCRIBE HISTORY delta.`/data/features/`")
      
  strengths:
    - "ACID transactions (reliable concurrent writes)"
    - "Time travel without extra storage (transaction log)"
    - "Schema enforcement and evolution"
    - "Unified batch and streaming"
    - "Native Spark/Databricks integration"
    
  for_ml:
    training_data: "Reproduce exact training dataset from any point in time"
    feature_store: "Query feature values as of specific timestamps"
    evaluation: "Compare model performance on different data versions"
    
  limitation: "Table-level versioning (not arbitrary file versioning like DVC)"
```

### Data Versioning for LLM Applications

```yaml
LLM_Data_Versioning:
  knowledge_base_versioning:
    what: "Version the documents that feed RAG systems"
    why: "Knowledge base changes affect LLM outputs — must track what content was available"
    approach:
      - "Version document corpus (DVC or LakeFS)"
      - "Version embedding index (tied to document version + embedding model version)"
      - "Track: which documents, which chunking strategy, which embedding model"
    reproduction: "Given a query + timestamp, reproduce exact retrieval results"
    
  evaluation_dataset_versioning:
    what: "Version test sets used to evaluate LLM quality"
    why: "Evaluation results only comparable if using same test set"
    approach:
      - "Version evaluation sets with DVC"
      - "Tag evaluation sets (v1-initial, v2-expanded, v3-adversarial)"
      - "Record which eval set was used for each model evaluation"
    contamination: "Track to ensure eval data never leaked into training/RAG"
    
  synthetic_data_versioning:
    what: "Version LLM-generated training data"
    why: "Synthetic data quality varies by generation model, prompt, and filtering"
    approach:
      - "Version generation prompts + model used + filtering criteria"
      - "Version output datasets with DVC"
      - "Track lineage: generation model → raw synthetic → filtered → training"
```

---

## How It Works in Practice

### Data Versioning Workflow

```yaml
Example:
  project: "Credit risk model with quarterly retraining"
  tools: "DVC (data versioning) + Git (code) + MLflow (model registry)"
  
  quarterly_workflow:
    q1_2026:
      data_update: "Add Q4 2025 loan applications to training set"
      steps:
        - "Pull new data from data warehouse into data/raw/"
        - "dvc add data/raw/loans_q4_2025.parquet"
        - "Run preprocessing pipeline: dvc repro"
        - "git commit -m 'Add Q4 2025 data, update features'"
        - "dvc push (uploads new data to S3)"
      result: "Git commit abc123 is tied to this exact data state"
      model_trained: "Model v8 trained on this data, registered in MLflow"
      
    q2_2026:
      data_update: "Add Q1 2026 + fix data quality issue found in Q3 2025"
      steps:
        - "Pull Q1 2026 data"
        - "Fix Q3 2025 labeling error (retroactive correction)"
        - "dvc add data/raw/ data/processed/"
        - "dvc repro (rerun pipeline with corrected data)"
        - "git commit -m 'Add Q1 2026, fix Q3 2025 label issue'"
      result: "New data state, full history preserved"
      model_trained: "Model v9 — trained on corrected + new data"
      
  reproducibility_test:
    question: "Reproduce model v8 exactly"
    answer:
      - "git checkout abc123 (code + DVC pointers for v8)"
      - "dvc checkout (pull exact data that v8 was trained on)"
      - "dvc repro (run pipeline — produces identical model)"
      - "Compare: reproduced model metrics match original within tolerance"
      
  debugging:
    question: "Model v9 performance dropped — why?"
    investigation:
      - "dvc diff abc123 def456 → shows which data files changed"
      - "Compare distributions: Q3 2025 label correction affected 2% of training data"
      - "Root cause: label correction removed some previously-correct labels (overcorrection)"
```

---

## Interview Tip

> When asked about data versioning: "I version data with the same rigor as code because model behavior is determined by training data. My approach: DVC for dataset versioning (Git-native workflow — data versions tied to code commits, actual data in S3), Delta Lake for table-level time travel (training feature tables), and LakeFS for data lake branching (safe experimentation). Key principle: given any historical model version, I can reproduce the exact data state used to train it — code (Git) + data (DVC) + config (Git) = full reproducibility. For LLM applications, I also version knowledge bases, embedding indexes, evaluation datasets, and generation prompts. The biggest anti-pattern is mutable datasets — if your training data can be silently modified by upstream processes, you've lost reproducibility. Data contracts + versioning + validation = data you can trust."

---

## Common Mistakes

1. **Mutable datasets** — Training data stored in a database table that gets updated in place. When the model performs differently after retraining, you can't tell if it's because of code changes, new data, or data corrections made silently by another team.

2. **Not linking data versions to model versions** — Versioning data and versioning models independently, without connecting them. You have model v7 and data v12 and training code commit abc123, but no record of which combination produced the production model.

3. **Versioning processed data only** — Tracking the final training dataset but not the raw data and transformation logic. When the transformation has a bug, you can't reproduce the original raw data to reprocess correctly.

4. **Ignoring evaluation dataset versioning** — Changing evaluation data without tracking versions. "Model v8 got 92% accuracy" and "Model v9 got 91% accuracy" are incomparable if they used different test sets.

5. **No data validation in version pipeline** — Accepting any data version without quality checks. Garbage data gets versioned and used for training, producing garbage models. Always validate before committing a new data version.

---

## Key Takeaways

- Data is as important as code in ML — version it with the same rigor
- DVC: Git-native data versioning (pointers in Git, data in S3/GCS) — most common for ML teams
- LakeFS: Git semantics for data lakes (branch, commit, merge on S3) — for large-scale data operations
- Delta Lake: Table-level time travel (query any historical version) — for feature tables and structured data
- Key linkage: Git commit + DVC hash + MLflow run = full reproducibility chain
- Point-in-time queries: retrieve exact data state for any historical model version
- Version everything: raw data, processed data, features, evaluation sets, knowledge bases
- Data contracts prevent upstream teams from silently changing your training data
- For LLMs: version knowledge bases, embedding indexes, eval datasets, and generation prompts
- Reproducibility test: can you reproduce any historical model from versioned inputs alone?
