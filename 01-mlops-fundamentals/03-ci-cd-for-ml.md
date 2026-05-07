# CI/CD for ML

## The Problem / Why This Matters

CI/CD (Continuous Integration / Continuous Delivery) in traditional software is well-established: commit code → run tests → build → deploy. For ML systems, CI/CD is fundamentally more complex because you have three axes of change instead of one: code changes (new features, bug fixes), data changes (new training data, schema updates), and model changes (retrained weights, new architectures). A code change to a feature engineering function might not break any unit tests but could degrade model performance. A data distribution shift might not violate any schema but could make the model unreliable. Traditional CI/CD catches code bugs; ML CI/CD must also catch data bugs and model quality regressions. In 2026, ML CI/CD has matured with established patterns for continuous training (CT), model validation gates, automated A/B testing, and progressive deployment — but many organizations still deploy models with less rigor than they deploy a CSS change.

---

## The Analogy

Think of ML CI/CD like quality control in pharmaceutical manufacturing:

- **Traditional software CI/CD** = Testing that the pill-making machine is assembled correctly (code compiles, tests pass, no mechanical defects). If the machine works, the pills will be fine.
- **ML CI/CD** = Testing the machine AND the ingredients AND the final pill. The machine might be perfect (code is correct), but if the raw material changed (data shifted), the pill might be ineffective (model degraded). You need quality checks at every level: ingredients (data validation), process (training pipeline), and product (model evaluation).
- **Continuous Training** = Not just making pills once — continuously making them as new ingredient batches arrive, with full quality control on each batch.

---

## Deep Dive

### The Three Pillars of ML CI/CD

```yaml
ML_CICD_Pillars:
  continuous_integration:
    traditional_ci:
      triggers: "Code commit to repository"
      checks:
        - "Lint and format (ruff, black, isort)"
        - "Type checking (mypy, pyright)"
        - "Unit tests (pytest)"
        - "Integration tests"
        - "Security scanning (bandit, safety)"
      tools: "GitHub Actions, GitLab CI, Jenkins"
      
    ml_specific_ci:
      additional_triggers:
        - "Data schema change"
        - "Pipeline configuration change"
        - "Feature definition change"
      additional_checks:
        - "Data contract validation (schema, distributions)"
        - "Feature engineering unit tests (transform correctness)"
        - "Model behavioral tests (sanity checks on small data)"
        - "Pipeline integration test (full pipeline on test dataset)"
        - "Training smoke test (train for 1 epoch, verify no errors)"
      tools: "GitHub Actions + pytest + Great Expectations + custom model tests"
      
  continuous_delivery:
    what: "Automated, gated deployment of validated models to production"
    ml_specific:
      - "Model validation gate (metrics above threshold)"
      - "Fairness check gate (no bias regression)"
      - "Latency profiling gate (meets SLA requirements)"
      - "Shadow deployment (new model runs alongside production)"
      - "Progressive rollout (canary → gradual → full)"
      - "Automated rollback (if live metrics degrade)"
    deployment_strategies:
      shadow: "Serve both models, compare outputs, don't affect users"
      canary: "Route 1-5% of traffic to new model, monitor"
      blue_green: "Two environments, instant switch"
      gradual: "5% → 25% → 50% → 100% with gates at each stage"
      
  continuous_training:
    what: "Automatically retrain models when conditions warrant"
    triggers:
      scheduled: "Retrain every N hours/days/weeks on latest data"
      data_driven: "Retrain when new data exceeds threshold volume"
      drift_driven: "Retrain when drift detection signals degradation"
      performance_driven: "Retrain when live metrics drop below threshold"
    workflow:
      - "Trigger fires → pull latest data → validate data quality"
      - "Train model → evaluate against champion"
      - "If better → register as challenger → progressive deployment"
      - "If worse → log results, keep current champion"
    principle: "Models stay fresh automatically without human intervention"
```

### ML Testing Pyramid

```yaml
ML_Testing_Pyramid:
  level_1_unit_tests:
    what: "Test individual functions in isolation"
    examples:
      - "Feature transformation produces correct output for known input"
      - "Data preprocessing handles edge cases (nulls, empty strings)"
      - "Custom loss function computes correctly"
      - "Metric calculation matches expected values"
    speed: "Fast (milliseconds per test)"
    when: "Every commit (CI)"
    
  level_2_data_tests:
    what: "Validate data quality and schema"
    examples:
      - "Schema hasn't changed (column names, types)"
      - "No unexpected nulls in required columns"
      - "Value distributions within expected ranges"
      - "Label distribution consistent with historical"
      - "No data leakage between train and test sets"
    speed: "Medium (seconds to minutes)"
    when: "Every pipeline run, every commit that changes data logic"
    tools: "Great Expectations, Pandera, TFDV (TensorFlow Data Validation)"
    
  level_3_model_tests:
    what: "Validate model behavior and quality"
    subcategories:
      training_tests:
        - "Model trains without errors (smoke test on small data)"
        - "Loss decreases over epochs (model is learning)"
        - "Model converges within expected time"
      behavioral_tests:
        - "Invariance: prediction doesn't change for irrelevant input changes"
        - "Directional: known input changes produce expected output direction"
        - "Minimum functionality: model handles known-correct examples"
      performance_tests:
        - "Metrics above minimum threshold on test set"
        - "No regression vs previous version beyond tolerance"
        - "Per-slice metrics (performance across subgroups)"
      fairness_tests:
        - "Demographic parity within tolerance"
        - "Equal opportunity across protected groups"
        - "No regression in fairness metrics vs previous version"
    speed: "Slow (minutes to hours — requires training or evaluation)"
    when: "Before model promotion, scheduled evaluation"
    
  level_4_integration_tests:
    what: "Test full pipeline end-to-end"
    examples:
      - "Pipeline runs successfully on test dataset"
      - "Model loads and serves predictions correctly"
      - "Feature store serves correct values for test inputs"
      - "Monitoring correctly detects synthetic drift"
    speed: "Slow (minutes to hours)"
    when: "Before deployment, periodic validation"
    
  level_5_system_tests:
    what: "Test the complete system in production-like environment"
    examples:
      - "Shadow deployment produces reasonable predictions"
      - "Latency meets SLA under load"
      - "Fallback triggers correctly when model errors"
      - "Rollback procedure works end-to-end"
    speed: "Very slow (hours to days)"
    when: "Before production deployment"
```

### ML CI/CD Pipeline Definition

```yaml
CICD_Pipeline_Example:
  name: "ML CI/CD for Fraud Detection Model"
  tool: "GitHub Actions"
  
  on_code_commit:
    stage_1_code_quality:
      - "ruff check . (linting)"
      - "mypy src/ (type checking)"
      - "pytest tests/unit/ (unit tests)"
      duration: "2-3 minutes"
      
    stage_2_data_tests:
      - "pytest tests/data/ (data validation on test dataset)"
      - "Great Expectations checkpoint (schema + distributions)"
      duration: "5-10 minutes"
      
    stage_3_model_smoke_test:
      - "Train model for 1 epoch on small dataset (verify no errors)"
      - "Run behavioral tests (invariance, directional)"
      duration: "10-15 minutes"
      
    stage_4_integration_test:
      - "Run full pipeline on test dataset (subset of production data)"
      - "Verify model meets minimum metrics on test set"
      - "Verify model can be served (load + predict)"
      duration: "20-30 minutes"
      
  on_training_trigger:
    stage_5_full_training:
      - "Pull latest production data"
      - "Run full data validation"
      - "Train model on full dataset"
      - "Full evaluation on held-out test set"
      duration: "1-4 hours"
      
    stage_6_model_validation:
      gates:
        performance: "AUC >= 0.93 AND F1 >= 0.85"
        fairness: "Demographic parity ratio >= 0.8 across groups"
        latency: "p99 inference latency < 10ms"
        comparison: "Not worse than current champion by > 0.5%"
      action_on_pass: "Register model, trigger deployment"
      action_on_fail: "Alert team, log results, keep current model"
      
  on_model_validated:
    stage_7_deployment:
      - "Deploy as shadow (24 hours)"
      - "Compare shadow predictions vs champion"
      - "If shadow OK → canary 5% (24 hours)"
      - "If canary OK → gradual rollout (25% → 50% → 100%)"
      - "Auto-rollback if live metrics drop"
      duration: "3-5 days total rollout"
```

### Model Validation Gates

```yaml
Validation_Gates:
  purpose: "Automated quality gates that prevent bad models from reaching production"
  
  gate_types:
    performance_gate:
      metrics: "AUC, F1, precision, recall, NDCG — domain-specific"
      comparison:
        absolute: "Metric > minimum_threshold (e.g., AUC > 0.90)"
        relative: "Metric >= champion_metric - tolerance (e.g., within 1% of champion)"
      implementation: "Evaluate on held-out test set, compare against threshold"
      
    fairness_gate:
      metrics: "Demographic parity, equalized odds, calibration across groups"
      threshold: "Disparity < allowed_threshold (e.g., max 20% difference between groups)"
      implementation: "Evaluate per-subgroup metrics, flag if any group underserved"
      
    latency_gate:
      metrics: "p50, p95, p99 inference latency"
      threshold: "Must meet SLA (e.g., p99 < 50ms)"
      implementation: "Benchmark model on representative hardware with realistic batch sizes"
      
    size_gate:
      metrics: "Model file size, memory footprint"
      threshold: "Must fit deployment constraints (e.g., < 2GB for edge deployment)"
      
    regression_gate:
      metrics: "Performance on specific critical slices"
      threshold: "No slice can degrade more than X% vs previous version"
      purpose: "Prevent overall improvement that hides specific-group degradation"
      
  gate_outcomes:
    all_pass: "Model registered as challenger, deployment triggered"
    soft_fail: "Warning logged, human review requested, deployment paused"
    hard_fail: "Model rejected, team alerted, current champion stays"
```

---

## How It Works in Practice

### Complete ML CI/CD Flow

```yaml
Example:
  scenario: "ML engineer improves feature engineering for churn prediction model"
  
  day_1:
    action: "Engineer adds new feature: 'support_tickets_last_30d'"
    commits: "Feature code + unit tests + updated pipeline config"
    ci_triggers:
      code_quality: "PASS (lint, types, unit tests)"
      data_tests: "PASS (new feature validates correctly)"
      smoke_test: "PASS (model trains with new feature, no errors)"
      integration: "PASS (full pipeline on test data, metrics improved by 2%)"
    result: "PR approved and merged to main"
    
  day_2:
    action: "Continuous training trigger fires (weekly schedule)"
    training_pipeline:
      data_pull: "Latest 6 months of data with new feature computed"
      validation: "PASS (data quality checks, schema valid)"
      training: "Model trained with new feature included"
      evaluation: "AUC: 0.89 (champion: 0.87) — 2% improvement"
    validation_gates:
      performance: "PASS (0.89 > threshold 0.85, better than champion)"
      fairness: "PASS (no group degradation)"
      latency: "PASS (p99: 8ms < 10ms SLA)"
    result: "Model registered as challenger, shadow deployment triggered"
    
  day_3_4:
    action: "Shadow deployment running"
    monitoring: "Challenger predictions logged alongside champion predictions"
    comparison: "Challenger would have caught 4% more churning customers"
    result: "Shadow results positive, canary triggered"
    
  day_5:
    action: "Canary deployment (5% traffic)"
    monitoring: "Real predictions going to 5% of users"
    metrics: "No degradation in production metrics, slight improvement visible"
    result: "Auto-promoted to 25% → 50% → 100% over 48 hours"
    
  day_7:
    action: "Full deployment complete"
    status: "New model is now champion"
    documentation: "Model card updated, experiment logged, team notified"
```

---

## Interview Tip

> When asked about CI/CD for ML: "ML CI/CD extends traditional CI/CD with three additional dimensions: data validation, model testing, and continuous training. My pipeline has progressive checks: (1) On every commit — code quality, unit tests, data tests, training smoke test. (2) On training trigger — full training with model validation gates (performance, fairness, latency). (3) On model validation pass — progressive deployment (shadow → canary → gradual → full). Key differences from software CI/CD: tests include behavioral model tests (not just functional), gates include statistical comparisons (not just pass/fail), and there's a continuous training loop triggered by data changes or drift detection. I always implement auto-rollback — if live metrics degrade during deployment, the system automatically reverts to the previous champion. The hardest part isn't the tooling; it's defining good validation gates and getting the right balance between velocity and safety."

---

## Common Mistakes

1. **Testing only code, not models** — Standard unit tests pass, but nobody evaluates whether the model still performs well. ML CI must include model-level tests (behavioral, performance, fairness) in addition to code tests.

2. **No gates between training and deployment** — The pipeline trains a model and immediately deploys it. If training data was corrupted or the model degraded, you've pushed a bad model to production. Always have evaluation gates.

3. **Binary pass/fail without comparison** — Gate says "AUC > 0.80" but doesn't compare against the current champion. A model with 0.81 AUC might be worse than the current 0.89. Always compare against the incumbent.

4. **Skipping progressive deployment** — Going from "passed validation gates" straight to 100% production traffic. Shadow → canary → gradual rollout gives you multiple chances to catch problems that offline evaluation missed.

5. **No automated rollback** — Manual rollback means slow recovery. If live metrics degrade, the system should auto-rollback within minutes, not wait for a human to notice hours later.

---

## Key Takeaways

- ML CI/CD has three axes: code changes, data changes, model changes — all need testing
- ML testing pyramid: unit tests → data tests → model tests → integration tests → system tests
- Continuous Training (CT) extends CI/CD with automated retraining triggers (schedule, drift, performance)
- Model validation gates: performance, fairness, latency, regression — all must pass before deployment
- Progressive deployment: shadow → canary → gradual → full, with auto-rollback at each stage
- Compare against champion, not just absolute thresholds (relative performance matters more)
- Same pipeline code in CI and production — test with small data, deploy with full data
- Auto-rollback is non-negotiable — live metric degradation triggers immediate revert
- Gate failures should log detailed comparison (not just "failed") for debugging
- Balance velocity and safety: strict gates prevent bad models but shouldn't block good ones
