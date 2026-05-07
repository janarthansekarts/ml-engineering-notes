# Data-Centric AI

## The Problem / Why This Matters

For decades, the ML community focused on model-centric AI — improving results by building better architectures, bigger models, and smarter algorithms while keeping data fixed. Andrew Ng's data-centric AI movement flipped this: keep the model fixed and systematically improve the data. In 2026, this principle is proven: for most production ML systems, improving data quality yields 2-10x more impact than model architecture changes. Why? Because in production, data issues cause 90% of failures: mislabeled examples, distribution shifts, missing values, inconsistent annotations, stale features, and biased samples. A perfect model trained on bad data produces bad predictions. A simple model trained on excellent data often outperforms a complex model on messy data. For LLM (Large Language Model) applications, data-centricity manifests as: quality of RAG (Retrieval-Augmented Generation) documents, quality of fine-tuning data, quality of evaluation datasets, and quality of prompt examples. Data is the product — the model is just the lens that focuses it.

---

## The Analogy

Think of a chef (model) and ingredients (data):

- **Model-centric approach** = Hire progressively better chefs (GPT-3 → GPT-4 → GPT-5) while using the same mediocre ingredients. Results improve, but there's a ceiling.
- **Data-centric approach** = Keep a good chef, but systematically upgrade ingredients. Fresh instead of frozen. Organic instead of processed. Properly sourced and stored.
- **Reality** = The best restaurants in the world obsess over ingredient quality AND have great chefs. But between the two, ingredient quality matters more. A mediocre chef with amazing ingredients produces better food than a great chef with rotten ingredients.

In ML: a well-tuned XGBoost on perfectly labeled, clean data often beats a poorly-fed neural network on noisy data.

---

## Deep Dive

### The Data Quality Hierarchy

```yaml
Data_Quality_Hierarchy:
  level_1_availability:
    question: "Do we have data at all?"
    actions:
      - "Identify all potential data sources"
      - "Assess accessibility (permissions, APIs, legal)"
      - "Determine if synthetic data can supplement"
    common_issue: "Assuming data exists when it doesn't — validate before committing to a project"
    
  level_2_correctness:
    question: "Are the labels/values actually correct?"
    actions:
      - "Audit a sample of labels (human review)"
      - "Measure inter-annotator agreement"
      - "Identify systematic labeling errors"
      - "Fix or remove incorrect examples"
    common_issue: "Assuming labels are ground truth — in reality, human labels have 5-20% error rates"
    metrics:
      inter_annotator_agreement: "Cohen's Kappa > 0.8 for reliable labels"
      label_error_rate: "Target < 2% for production training data"
    
  level_3_consistency:
    question: "Are similar examples labeled the same way?"
    actions:
      - "Create detailed labeling guidelines"
      - "Review boundary cases with annotators"
      - "Use consensus labeling (multiple annotators per example)"
      - "Automated consistency checks (find conflicting labels)"
    common_issue: "Different annotators interpret guidelines differently — causes noisy labels"
    
  level_4_completeness:
    question: "Does the data represent all scenarios the model will encounter?"
    actions:
      - "Analyze coverage gaps (demographics, edge cases, rare events)"
      - "Collect additional data for underrepresented scenarios"
      - "Augment data to improve coverage"
      - "Monitor production for inputs not well-represented in training"
    common_issue: "Model works great on common cases but fails on tail distribution"
    
  level_5_freshness:
    question: "Is the data current and relevant?"
    actions:
      - "Track data age and decay patterns"
      - "Set up continuous data collection"
      - "Detect when training data no longer represents production distribution"
      - "Implement automated retraining on fresh data"
    common_issue: "Model trained on 2024 data serving 2026 users with different behavior"
    
  level_6_relevance:
    question: "Does every feature/example actually help the model?"
    actions:
      - "Feature importance analysis"
      - "Remove features that don't improve performance"
      - "Prune noisy or confusing examples from training set"
      - "Focus on high-signal data over high-volume data"
    common_issue: "Adding more features/data without checking if it helps — sometimes it hurts"
```

### Data Flywheel

```yaml
Data_Flywheel:
  concept: "A self-reinforcing cycle where model usage generates data that improves the model"
  
  cycle:
    step_1_deploy: "Ship model to production"
    step_2_collect: "Collect user interactions, feedback, and outcomes"
    step_3_label: "Convert interactions into labeled training data"
    step_4_improve: "Retrain model on enriched dataset"
    step_5_deploy: "Ship improved model → attracts more users → more data"
    
  examples:
    recommendation_system:
      signal: "User clicks, purchases, time-spent"
      label: "Implicit feedback (click = relevant, skip = not relevant)"
      flywheel: "Better recommendations → more engagement → more signal → better recommendations"
      
    search_ranking:
      signal: "Click-through, dwell time, query reformulations"
      label: "Position-debiased click models"
      flywheel: "Better ranking → users trust results → cleaner signals → better ranking"
      
    llm_application:
      signal: "User thumbs up/down, message edits, task completion"
      label: "Preference pairs for RLHF/DPO"
      flywheel: "Better responses → more usage → more feedback → better responses"
      
    spam_detection:
      signal: "User reports, false positive complaints"
      label: "Confirmed spam/not-spam from user actions"
      flywheel: "Fewer false positives → users trust system → more reports → better detection"
      
  building_the_flywheel:
    requirements:
      - "Instrument your application to collect signals (clicks, ratings, corrections)"
      - "Build pipelines to convert signals into training data"
      - "Automate retraining on fresh data (continuous training)"
      - "Monitor for improvements (offline eval + online A/B tests)"
    time_to_value: "Typically 3-6 months before flywheel effect is visible"
    moat: "The longer it runs, the harder it is for competitors to catch up"
```

### Data Labeling at Scale

```yaml
Data_Labeling:
  approaches:
    human_annotation:
      when: "Ground truth requires human judgment (sentiment, intent, quality)"
      tools: "Label Studio, Labelbox, Scale AI, Amazon SageMaker Ground Truth"
      best_practices:
        - "Write detailed guidelines with examples of edge cases"
        - "Use 3+ annotators per example for consensus (majority vote)"
        - "Measure inter-annotator agreement (Cohen's Kappa)"
        - "Review disagreements — they reveal ambiguity in your task definition"
        - "Pay annotators fairly — quality correlates with compensation"
      cost: "$0.05-$5.00 per label depending on complexity"
      
    programmatic_labeling:
      when: "Heuristics can approximate labels (weak supervision)"
      tools: "Snorkel, Cleanlab"
      approach:
        - "Write labeling functions (rules that label subsets of data)"
        - "Combine multiple noisy labeling functions"
        - "Model learns despite individual function errors (noise-aware training)"
      example: "If email contains 'unsubscribe' → likely newsletter (not spam)"
      
    llm_labeling:
      when: "Task is well-defined and LLM judgment aligns with human judgment"
      approach:
        - "Use GPT-5/Claude 4 to label data with structured prompts"
        - "Validate LLM labels against human labels on a sample"
        - "Use LLM labels for initial training, refine with human labels"
      cost: "10-100x cheaper than human annotation"
      risk: "LLM biases propagate into your model — always validate"
      
    active_learning:
      when: "Labeling budget is limited — maximize label efficiency"
      approach:
        - "Train initial model on small labeled set"
        - "Model identifies examples it's most uncertain about"
        - "Human labels those specific examples (maximum information gain)"
        - "Retrain model and repeat"
      benefit: "Same accuracy with 50-80% fewer labels"
      
    synthetic_data:
      when: "Real data is scarce, private, or expensive to collect"
      approaches:
        - "LLM-generated training examples (text, code, Q&A pairs)"
        - "Augmentation (image flips, text paraphrasing, noise injection)"
        - "Simulation (robotics, self-driving, game environments)"
        - "GANs/Diffusion models (image generation for rare classes)"
      validation: "ALWAYS validate synthetic data against real-world performance"
```

---

## How It Works in Practice

### Data Improvement Sprint

```yaml
Example:
  context: "Customer intent classification model — accuracy stuck at 87%"
  
  diagnosis:
    step_1: "Audit 200 random misclassified examples"
    findings:
      - "42% of errors: ambiguous labels (example could be 2+ intents)"
      - "28% of errors: missing intent category (users asking something we don't handle)"
      - "18% of errors: data quality issues (truncated text, encoding errors)"
      - "12% of errors: genuinely hard cases (model limitation)"
      
  data_improvements:
    fix_ambiguity:
      action: "Rewrite labeling guidelines, add decision tree for border cases"
      impact: "Re-labeled 500 ambiguous examples with consensus"
    add_category:
      action: "Created 'billing_dispute' intent (was being labeled as 'complaint')"
      impact: "Collected 300 examples for new category"
    fix_quality:
      action: "Fixed text preprocessing pipeline (was truncating at 256 chars)"
      impact: "15% of training examples now have full context"
    
  result:
    before: "87% accuracy with complex BERT model"
    after: "93% accuracy with SAME model, only data improved"
    model_change: "None — same architecture, same hyperparameters"
    time_spent: "2 weeks on data vs 6 weeks previously spent on model experiments"
    lesson: "Data improvements yielded 6% gain; 6 weeks of model work yielded 0.5% gain"
```

---

## Interview Tip

> When asked about improving model performance: "My first instinct is to look at the data, not the model. I follow this process: (1) Error analysis — manually inspect 100-200 misclassified examples and categorize failure modes. (2) Root cause — determine if failures are data issues (wrong labels, missing coverage, quality) or model limitations (genuinely hard cases). (3) Data improvements — fix labels, add coverage for underrepresented cases, improve preprocessing. (4) Evaluate — measure impact of data changes before touching the model. In my experience, data improvements give 2-10x more impact per engineering hour than model architecture changes. Only after exhausting data improvements do I explore model changes."

---

## Common Mistakes

1. **Assuming labels are correct** — Taking training labels at face value. In reality, human annotators have 5-20% error rates. Always audit labels, measure agreement, and treat labels as noisy.

2. **More data without quality** — Collecting 10x more data when the existing data has quality issues. 1,000 perfectly labeled examples often outperform 100,000 noisy examples. Quality over quantity.

3. **Ignoring the tail distribution** — Optimizing for common cases while rare cases (which matter most — fraud, safety, edge cases) have terrible data coverage. Actively collect and label tail cases.

4. **Static datasets** — Training on a fixed dataset and never updating it. Production data distributions shift constantly. Build pipelines for continuous data collection and refreshing.

5. **Not instrumenting for data collection** — Deploying a model without building the infrastructure to collect user feedback, corrections, and outcomes. You miss the data flywheel opportunity.

---

## Key Takeaways

- Data quality improvements yield 2-10x more impact than model architecture changes for most production systems
- Data quality hierarchy: Availability → Correctness → Consistency → Completeness → Freshness → Relevance
- Data flywheel: deploy → collect signals → label → retrain → deploy better model → more signals
- Error analysis first: manually inspect failures before trying model changes
- Labeling approaches: human, programmatic (Snorkel), LLM-generated, active learning, synthetic
- LLM-generated labels are 10-100x cheaper but must be validated against human judgment
- Always measure inter-annotator agreement — if humans disagree, the model will too
- For LLM applications: data quality = RAG document quality + fine-tuning data quality + eval set quality
