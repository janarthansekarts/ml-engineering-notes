# ML Project Lifecycle

## The Problem / Why This Matters

Most ML projects fail — industry estimates suggest 85-90% of ML projects never reach production. The reason isn't bad algorithms; it's bad project management. Teams jump straight to modeling without understanding the problem, collect data without validating quality, build complex architectures before proving a simple baseline works, and then discover in deployment that the model's predictions don't actually help the business. The ML project lifecycle is fundamentally different from traditional software development because it has more uncertainty (will ML even work for this problem?), more iteration (data → model → evaluation loops), and more maintenance burden (models degrade over time). In 2026, the lifecycle has expanded further to include LLM (Large Language Model) evaluation, prompt engineering phases, human feedback loops for RLHF (Reinforcement Learning from Human Feedback), and continuous fine-tuning — making structured lifecycle management even more critical.

---

## The Analogy

Think of an ML project lifecycle like building a house:

1. **Scoping** = Meeting with the client to understand what they want (mansion? cottage? budget?)
2. **Data** = Surveying the land and sourcing materials (is the ground solid? are materials available?)
3. **Modeling** = Architecture and construction (design blueprints, build structure)
4. **Deployment** = Moving the family in (does everything actually work when people live there?)
5. **Maintenance** = Ongoing upkeep (fixing leaks, updating wiring, adapting to changing needs)

Skipping the survey phase (data) leads to foundations on swamp. Skipping the client meeting (scoping) leads to building something nobody wanted. Most failed ML projects are beautiful houses built on the wrong land for the wrong client.

---

## Deep Dive

### Phase 1: Problem Scoping and Feasibility

```yaml
Phase_1_Scoping:
  purpose: "Determine IF ML is the right solution and define success criteria"
  duration: "1-2 weeks"
  
  key_questions:
    business_alignment:
      - "What business metric will this improve? (Revenue, retention, cost, time)"
      - "What's the current baseline? (Rule-based system, human process, nothing)"
      - "What improvement justifies the ML investment?"
      - "What's the cost of wrong predictions? (Asymmetric: false positive vs false negative)"
      
    feasibility_assessment:
      - "Is there data available? How much? How clean?"
      - "Has anyone solved a similar problem with ML? (Literature, competition results)"
      - "What's the minimum useful accuracy? (If 95% required but state-of-art is 80%, don't start)"
      - "Are there ethical/legal constraints? (Protected attributes, GDPR, industry regulations)"
      
    ml_vs_alternatives:
      consider_alternatives:
        - "Rule-based systems (if logic is known and stable)"
        - "Simple heuristics (if pattern is obvious)"
        - "Human-in-the-loop (if volume is low enough)"
        - "LLM prompting without fine-tuning (if accuracy requirements are moderate)"
      ml_justified_when:
        - "Pattern is complex and non-obvious"
        - "Data volume makes manual rules impractical"
        - "Pattern changes over time (model can be retrained)"
        - "Marginal accuracy improvement has high business value"
        
  deliverable:
    problem_statement: "Clear definition of what ML will predict/classify/generate"
    success_metrics: "Specific thresholds (AUC > 0.90, latency < 50ms, cost < $X/month)"
    timeline: "Realistic milestones with go/no-go checkpoints"
    risks: "Identified risks and mitigation strategies"
```

### Phase 2: Data Engineering

```yaml
Phase_2_Data:
  purpose: "Collect, validate, clean, and prepare data for modeling"
  duration: "2-8 weeks (often longest phase — 60-80% of project time)"
  
  sub_phases:
    data_collection:
      tasks:
        - "Identify data sources (databases, APIs, logs, third-party)"
        - "Assess data availability and access permissions"
        - "Set up data pipelines (ETL/ELT)"
        - "Define data contracts with upstream teams"
      pitfalls:
        - "Data exists but isn't accessible (permissions, format, latency)"
        - "Label collection is expensive/slow (may need annotation pipeline)"
        - "Historical data doesn't represent future patterns"
        
    data_validation:
      tasks:
        - "Profile data (distributions, missing values, outliers)"
        - "Validate schema consistency across sources"
        - "Check for label quality (noise, disagreement)"
        - "Identify data drift vs training distribution"
      tools: "Great Expectations, Pandera, TensorFlow Data Validation (TFDV), Evidently"
      
    data_preparation:
      tasks:
        - "Feature engineering (domain-specific transformations)"
        - "Handle missing values (imputation strategy)"
        - "Split data (train/validation/test — no leakage)"
        - "Version data (DVC, LakeFS, Delta Lake)"
      critical_rule: "Test set is sacred — never look at it during development"
      
    data_labeling:
      when_needed: "Supervised learning without existing labels"
      approaches:
        manual: "Human annotators (Labelbox, Scale AI, Amazon SageMaker Ground Truth)"
        semi_supervised: "Label small subset, propagate to similar examples"
        weak_supervision: "Programmatic labeling rules (Snorkel, LabelStudio)"
        llm_assisted: "Use LLMs to generate initial labels, human verifies (2026 common pattern)"
      quality: "Inter-annotator agreement, quality audits, disagreement resolution"
```

### Phase 3: Model Development

```yaml
Phase_3_Modeling:
  purpose: "Build, train, and evaluate models — iterative experimentation"
  duration: "2-6 weeks"
  
  sub_phases:
    baseline_establishment:
      principle: "Always start simple — complex models must beat simple baselines"
      progression:
        step_1: "Heuristic baseline (majority class, average, simple rules)"
        step_2: "Simple ML model (logistic regression, decision tree)"
        step_3: "Stronger ML model (XGBoost, random forest)"
        step_4: "Deep learning (only if step 3 insufficient)"
        step_5: "LLM/foundation model (if text/multi-modal, step 3/4 insufficient)"
      purpose: "Understand problem difficulty and ensure ML adds value over simple approaches"
      
    iterative_improvement:
      experiment_axes:
        features: "Add/remove/transform features"
        architecture: "Model type, layers, hidden size"
        hyperparameters: "Learning rate, regularization, batch size"
        data: "More data, better labels, augmentation, resampling"
        loss_function: "Custom loss for business objective"
      tracking: "All experiments logged in MLflow/W&B with parameters, metrics, artifacts"
      
    evaluation:
      offline_metrics:
        classification: "AUC-ROC, precision, recall, F1, calibration"
        regression: "MAE, RMSE, MAPE, quantile losses"
        ranking: "NDCG, MAP, MRR"
        generation: "BLEU, ROUGE, BERTScore, human evaluation"
      beyond_aggregate:
        - "Per-class metrics (not just average)"
        - "Slice-based evaluation (performance across subgroups)"
        - "Error analysis (what patterns does the model get wrong?)"
        - "Calibration (are probabilities meaningful?)"
        - "Fairness metrics (demographic parity, equalized odds)"
        
    llm_specific_development:
      prompt_engineering: "Systematic prompt optimization (few-shot, chain-of-thought)"
      rag_development: "Retrieval pipeline tuning (chunking, embedding model, re-ranking)"
      fine_tuning: "LoRA/QLoRA adaptation for domain-specific performance"
      evaluation: "LLM-as-judge, human evaluation, task-specific benchmarks"
```

### Phase 4: Deployment

```yaml
Phase_4_Deployment:
  purpose: "Serve model predictions to users/systems reliably"
  duration: "1-4 weeks"
  
  deployment_strategies:
    shadow_deployment:
      description: "New model runs alongside production but doesn't affect users"
      purpose: "Validate performance on live data without risk"
      duration: "1-2 weeks typically"
      
    canary_deployment:
      description: "Route small percentage of traffic (1-5%) to new model"
      purpose: "Detect issues before full rollout"
      monitoring: "Compare new model metrics against control"
      
    blue_green_deployment:
      description: "Two identical environments — switch traffic between them"
      purpose: "Instant rollback capability"
      
    gradual_rollout:
      description: "Slowly increase traffic to new model (5% → 25% → 50% → 100%)"
      purpose: "Build confidence with increasing exposure"
      
  deployment_requirements:
    latency: "Model must meet SLA (Service Level Agreement) requirements"
    throughput: "Handle expected QPS (Queries Per Second)"
    reliability: "99.9%+ uptime with fallback on failure"
    monitoring: "Alerts on prediction quality, latency, error rates"
    rollback: "Ability to revert to previous model in <5 minutes"
```

### Phase 5: Monitoring and Maintenance

```yaml
Phase_5_Maintenance:
  purpose: "Ensure model continues to perform well over time"
  duration: "Ongoing (80% of total ML lifecycle cost)"
  
  monitoring:
    what_to_monitor:
      - "Prediction quality (offline evaluation on fresh data)"
      - "Data drift (input distribution changes)"
      - "Concept drift (relationship between features and target changes)"
      - "System metrics (latency, error rate, throughput)"
      - "Business metrics (does model still improve the KPI?)"
    tools: "Evidently AI, Arize, WhyLabs, Fiddler"
    
  retraining:
    triggers:
      - "Scheduled (weekly/monthly retraining on latest data)"
      - "Performance-based (metrics drop below threshold)"
      - "Data-driven (significant drift detected)"
      - "Event-driven (known world change — e.g., COVID, new product launch)"
    automation: "CI/CD pipeline: trigger → retrain → evaluate → promote if better"
    
  model_decay:
    causes:
      - "World changes (user behavior shifts, new products, economic changes)"
      - "Data pipeline issues (upstream schema change, missing data source)"
      - "Feature staleness (feature data stops updating)"
      - "Population shift (new user segment with different behavior)"
    mitigation:
      - "Regular retraining on recent data"
      - "Monitoring for drift and performance degradation"
      - "Feature freshness checks"
      - "Periodic full model rebuild (not just retraining — re-evaluate features, architecture)"
```

---

## How It Works in Practice

### Real Project Timeline

```yaml
Example:
  project: "Product recommendation engine for e-commerce"
  
  week_1_2:
    phase: "Scoping"
    activities:
      - "Met with product team to define objectives"
      - "Agreed on metrics: CTR improvement >15% and revenue per session >10%"
      - "Reviewed existing rule-based recommendations (baseline: 2.1% CTR)"
      - "Assessed data availability: 18 months of click/purchase history"
    decision: "Proceed — data sufficient, business impact clear"
    
  week_3_6:
    phase: "Data"
    activities:
      - "Built ETL pipeline from clickstream + purchase databases"
      - "Created user features (browsing history, purchase frequency, session patterns)"
      - "Created item features (category, price, popularity, co-purchase patterns)"
      - "Data quality issues: 12% missing user sessions (mobile app bug) — worked with eng to fix"
    blocker: "Item metadata incomplete for 30% of catalog — escalated and resolved"
    
  week_7_10:
    phase: "Modeling"
    activities:
      - "Baseline: popularity-based recommendations (CTR: 2.1%)"
      - "Collaborative filtering (CTR: 3.4% offline estimate)"
      - "Two-tower neural model (CTR: 3.9% offline estimate)"
      - "XGBoost re-ranker on top (CTR: 4.2% offline estimate)"
    decision: "Two-tower + XGBoost re-ranker for production (2x baseline)"
    
  week_11_12:
    phase: "Deployment"
    activities:
      - "Deployed shadow mode — verified latency <50ms at p99"
      - "Canary 5% traffic — CTR 4.0% (slightly below offline estimate, normal)"
      - "Gradual rollout to 100% over 1 week"
      - "Final result: CTR 3.8% (80% improvement over baseline)"
      
  ongoing:
    phase: "Maintenance"
    activities:
      - "Daily retraining on latest interaction data"
      - "Weekly evaluation on fresh test set"
      - "Monthly full pipeline review"
      - "Quarterly architecture reassessment"
```

---

## Interview Tip

> When asked about ML project lifecycle: "I follow a structured lifecycle: (1) Scoping — validate ML is appropriate, define success metrics, establish baselines. This prevents building solutions for non-problems. (2) Data — 60-80% of project time. Collection, validation, cleaning, labeling, versioning. Bad data guarantees bad models regardless of architecture. (3) Modeling — start simple (logistic regression/XGBoost), only add complexity when simpler approaches are insufficient. Every experiment tracked in MLflow/W&B. (4) Deployment — progressive rollout (shadow → canary → gradual → full) with instant rollback capability. (5) Maintenance — continuous monitoring for drift, automated retraining triggers, periodic architecture reviews. The key insight is that maintenance is 80% of the total cost and effort. Most teams over-invest in modeling (phase 3) and under-invest in data quality (phase 2) and maintenance (phase 5)."

---

## Common Mistakes

1. **Skipping scoping** — Jumping to modeling without validating that ML is the right approach or defining success criteria. Result: a technically impressive model that doesn't solve the actual business problem.

2. **Underestimating data work** — Assuming clean, labeled data exists. In reality, data collection and preparation take 60-80% of project time. Plan accordingly.

3. **No baselines** — Building a complex deep learning model without first trying simple approaches. If XGBoost gets 92% accuracy and your complex model gets 93%, the complexity isn't justified.

4. **Big bang deployment** — Going from development straight to 100% production traffic. Always use progressive rollout (shadow → canary → gradual). One bad model update can lose millions in revenue.

5. **Ignoring maintenance** — Treating deployment as the finish line. Models degrade over time as data distributions shift. Without monitoring and retraining, production model quality slowly degrades until someone notices a business metric has cratered.

---

## Key Takeaways

- ML lifecycle: Scoping → Data → Modeling → Deployment → Maintenance
- 85-90% of ML projects fail, primarily due to poor scoping and data issues (not bad algorithms)
- Data phase takes 60-80% of project time — plan for it
- Always start with simple baselines — complexity must justify itself
- Progressive deployment: shadow → canary → gradual rollout → full
- Maintenance is 80% of total lifecycle cost — budget accordingly
- Every phase has explicit go/no-go criteria and deliverables
- Regular retraining (scheduled + triggered) prevents model decay
- For LLMs: lifecycle includes prompt engineering, RAG tuning, and fine-tuning phases
- Document decisions and rationale at each phase for team knowledge
