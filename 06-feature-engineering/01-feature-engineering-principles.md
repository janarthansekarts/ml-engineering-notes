# Feature Engineering Principles

## The Problem / Why This Matters

Features are the inputs your model uses to make predictions — and feature quality is the single biggest determinant of model performance. You can have the most sophisticated model architecture in the world, but if your features are noisy, stale, leaked, or poorly designed, your model will underperform a simpler model with better features. Feature engineering is the discipline of transforming raw data into meaningful signals that models can learn from. The challenge in production ML systems isn't just creating good features — it's creating features that are: (1) predictive (actually correlate with what you're predicting), (2) available at prediction time (no leakage from the future), (3) computable at low latency (milliseconds for real-time serving), (4) maintainable (don't break when source data changes), and (5) reusable (other teams can benefit from your work). Most ML engineers spend 60-80% of their time on feature engineering — not because it's tedious, but because it's where model performance is actually won or lost.

---

## The Analogy

Think of feature engineering like building a case for a jury:

- **Raw data** = All the evidence collected (thousands of documents, recordings, photos). Overwhelming and unusable in raw form.
- **Feature engineering** = The lawyer's work of selecting, organizing, and presenting evidence. Which facts are relevant? How should they be framed? What context helps the jury (model) make a good decision?
- **Good features** = Clear, relevant evidence presented at the right time. "The defendant's credit card was used at location X at time Y" — specific, verifiable, predictive.
- **Bad features** = Irrelevant noise ("the defendant likes blue"), leaked information ("the verdict was guilty" — you can't use the answer as input!), or stale data ("10 years ago the defendant lived here" — no longer relevant).
- **The model** = The jury. It can only decide based on what's presented to it. If you present good evidence (features), it makes good decisions. If you present noise, it can't perform regardless of how intelligent the jurors are.

---

## Deep Dive

### Feature Design Principles

```yaml
Core_Principles:
  signal_vs_noise:
    signal: "Information that helps predict the target"
    noise: "Irrelevant variation that confuses the model"
    principle: "Every feature should increase signal-to-noise ratio"
    test: "Does this feature have a logical, causal relationship with the target?"
    anti_pattern: "Adding every available column as a feature (throws noise at the model)"
    
  leakage_prevention:
    what: "Feature that contains information from the future (unavailable at prediction time)"
    types:
      target_leakage: "Feature derived from the target variable"
      temporal_leakage: "Feature that uses future data"
      data_contamination: "Training data includes test examples"
    examples:
      leaked: "Using 'account_closed_date' to predict churn (you only know this AFTER churn)"
      safe: "Using 'days_since_last_login' to predict churn (available before the event)"
    prevention: "Always ask: 'Would I have this information at the exact moment I need to make the prediction?'"
    
  point_in_time_correctness:
    what: "Features must reflect the state of the world AT prediction time, not after"
    violation: "Feature uses customer_lifetime_value computed over ALL time (includes future purchases)"
    correct: "Feature uses customer_lifetime_value computed up to prediction point"
    importance: "Critical for training (must simulate production environment) and production (must be current)"
    
  feature_freshness:
    what: "How current does this feature need to be?"
    spectrum:
      static: "Rarely changes (date of birth, account creation date)"
      slowly_changing: "Changes over weeks/months (address, subscription tier)"
      fast_changing: "Changes hourly/daily (recent purchases, page views)"
      real_time: "Changes per second (current session actions, live price)"
    design_implication: "Match computation frequency to freshness requirement"
    
  computational_feasibility:
    what: "Can you compute this feature at serving time within latency budget?"
    trade_offs:
      complex_features: "More predictive but slower to compute"
      simple_features: "Faster but less informative"
    solutions:
      pre_computation: "Compute expensive features in batch, serve from cache"
      approximation: "Use simpler proxy features for real-time"
      tiered: "Batch features for first pass, real-time features for re-ranking"
```

### Feature Categories

```yaml
Feature_Types:
  numerical:
    raw: "Use directly (age, price, temperature)"
    transformed:
      log_transform: "log(value) — reduce skewness for long-tailed distributions"
      normalization: "Scale to [0,1] or z-score — required for distance-based models"
      binning: "Convert continuous to categorical (age groups)"
      polynomial: "x², x³ — capture non-linear relationships"
      
  categorical:
    encoding:
      one_hot: "Binary column per category — for low cardinality (<50 values)"
      ordinal: "Integer encoding — for ordered categories (low/medium/high)"
      target_encoding: "Replace category with mean of target — for high cardinality"
      embedding: "Learned dense vector — for very high cardinality (user_id, product_id)"
    hashing: "Hash categories to fixed-size vector — handles unseen categories"
    
  temporal:
    raw: "Timestamp → extract: hour, day_of_week, month, is_weekend, is_holiday"
    relative: "time_since_event, time_until_event, recency"
    cyclical: "sin/cos encoding for periodic features (hour of day, day of week)"
    
  text:
    traditional: "TF-IDF, bag of words, n-grams"
    embeddings: "Dense vector from pre-trained model (sentence-transformers, OpenAI embeddings)"
    extracted: "Named entities, sentiment score, text length, reading level"
    
  aggregated:
    what: "Summarize multiple records into one feature"
    examples:
      count: "Number of purchases in last 30 days"
      sum: "Total spend in last 90 days"
      mean: "Average order value"
      max: "Maximum single transaction"
      distinct_count: "Number of unique products purchased"
      recency: "Days since last activity"
    windows: "Compute over time windows: 1h, 24h, 7d, 30d, 90d, all-time"
    
  interaction:
    what: "Combine two features to capture joint effects"
    examples:
      ratio: "price / average_category_price (relative price)"
      difference: "user_age - product_target_age (age fit)"
      product: "time_on_page * pages_viewed (engagement intensity)"
    caution: "Can explode feature space — be selective"
```

### Feature Importance and Selection

```python
# Feature importance analysis

"""
Not all features contribute equally. Feature selection removes noise
and speeds up training/inference.
"""

# Methods for measuring feature importance
FEATURE_IMPORTANCE_METHODS = {
    "model_based": {
        "tree_feature_importance": {
            "how": "Split gain / impurity reduction from tree models",
            "pros": "Fast, built into XGBoost/LightGBM/RandomForest",
            "cons": "Biased toward high-cardinality features",
        },
        "permutation_importance": {
            "how": "Shuffle feature values, measure prediction degradation",
            "pros": "Model-agnostic, captures non-linear importance",
            "cons": "Slow (requires multiple model evaluations)",
        },
        "shap_values": {
            "how": "Game-theoretic contribution of each feature",
            "pros": "Most rigorous, shows direction of contribution",
            "cons": "Computationally expensive for large models",
        },
    },
    "statistical": {
        "correlation": {
            "how": "Pearson/Spearman correlation with target",
            "pros": "Fast, simple",
            "cons": "Misses non-linear relationships",
        },
        "mutual_information": {
            "how": "Information-theoretic dependency measure",
            "pros": "Captures non-linear relationships",
            "cons": "Sensitive to binning/discretization",
        },
        "chi_squared": {
            "how": "Statistical test for categorical feature - target association",
            "pros": "Well-understood, fast",
            "cons": "Only for categorical features",
        },
    },
}


# Feature selection strategies
def select_features(X, y, method="combined", target_count=50):
    """Select top features using combined methods."""
    
    # 1. Remove zero-variance features
    # Features with no variation provide no signal
    
    # 2. Remove highly correlated features  
    # If two features are >0.95 correlated, keep the one with higher target correlation
    
    # 3. Statistical filtering (fast initial pass)
    # Remove features with near-zero mutual information with target
    
    # 4. Model-based ranking (more accurate)
    # Train LightGBM, get feature importances
    
    # 5. Final selection
    # Keep top N features by model importance
    # Verify no leakage in remaining features
    
    pass
```

### Feature Engineering Best Practices

```yaml
Best_Practices:
  domain_knowledge_first:
    principle: "Domain expertise > automated feature generation"
    why: "A domain expert knows that 'time since last order' matters for churn prediction"
    approach:
      1: "Talk to domain experts (product managers, analysts)"
      2: "Understand the business process behind the prediction"
      3: "Identify what information a human uses for this decision"
      4: "Translate that knowledge into computable features"
      
  start_simple:
    principle: "Begin with obvious features, add complexity incrementally"
    approach:
      tier_1: "Raw features + basic transformations (log, normalize)"
      tier_2: "Aggregations over time windows (counts, sums, averages)"
      tier_3: "Interaction features and ratios"
      tier_4: "Embedding features and complex computations"
    validate: "Measure model improvement at each tier — stop when marginal gain < 1%"
    
  test_for_leakage:
    methods:
      - "Feature importance: if one feature dominates (>80%), likely leakage"
      - "Temporal validation: train on past, test on future (leakage causes massive drop)"
      - "Logic check: 'Would I have this at prediction time?'"
      - "Remove suspect features: if accuracy drops 30%+, was likely leaked"
      
  document_features:
    what_to_document:
      - "Feature name and description"
      - "Business logic / rationale"
      - "Data sources used"
      - "Computation logic (SQL, code)"
      - "Expected distribution (range, nulls)"
      - "Freshness requirements"
      - "Known limitations or caveats"
    why: "Other teams need to understand, reuse, and maintain features"
```

---

## How It Works in Practice

### Production Feature Engineering Workflow

```yaml
Workflow:
  1_understand:
    - "Define prediction target clearly"
    - "Identify available data sources"
    - "Map the timeline (when prediction needed vs when data available)"
    - "Talk to domain experts about signals"
    
  2_explore:
    - "EDA (Exploratory Data Analysis) on raw data"
    - "Check distributions, null rates, cardinality"
    - "Look for obvious signals (correlation with target)"
    - "Identify temporal patterns"
    
  3_build:
    - "Start with simple features (tier 1)"
    - "Add aggregations (tier 2)"
    - "Add interactions (tier 3)"
    - "Add embeddings/complex features (tier 4)"
    
  4_validate:
    - "Check for leakage (temporal split test)"
    - "Measure feature importance"
    - "Remove low-importance features"
    - "Verify production computability"
    
  5_productionize:
    - "Implement in feature store (batch + online)"
    - "Set up monitoring (drift, staleness)"
    - "Document in feature catalog"
    - "Register for reuse by other teams"
```

---

## Interview Tip

> When asked about feature engineering: "I approach feature engineering systematically: (1) Start with domain knowledge — what information would a human expert use for this decision? That translates directly to features. (2) Layer complexity progressively — raw features first, then aggregations (counts, sums over time windows), then interactions, then embeddings. Measure improvement at each layer to avoid over-engineering. (3) Validate rigorously — temporal train/test split catches leakage (any feature unavailable at prediction time), feature importance analysis identifies noise (remove features that don't help). (4) Design for production from day one — every feature I build, I ask: 'Can I compute this within my latency budget at serving time?' Pre-compute expensive features in batch, serve simple features real-time. (5) Productionize in a feature store — single definition shared between training and serving (avoids train-serve skew), monitored for drift and staleness. The most impactful features I've built aren't complex — they're domain-informed aggregations over the right time windows. For example, 'ratio of this week's activity to average weekly activity' captures behavioral change better than any single metric."

---

## Common Mistakes

1. **Feature leakage** — Accidentally using information that wouldn't be available at prediction time. Example: using "claim_amount" to predict "is_fraudulent" when claim_amount is determined AFTER fraud investigation. Solution: strict temporal ordering — map the timeline of when each piece of data becomes available, and only use data available BEFORE the prediction moment.

2. **Too many features without validation** — Adding 500 features because "more is better." Most add noise, slow training, and make the model overfit. Solution: feature importance analysis after each batch. Remove features below threshold. Typical production models: 50-200 well-chosen features outperform 1000 random features.

3. **Training-serving skew** — Feature computed differently in training vs production. Training uses exact SQL on data warehouse; production uses an approximation. Model performs great offline, poorly online. Solution: single feature definition (feature store), or at minimum, validate that training features and serving features have the same distribution.

4. **Ignoring feature freshness** — Using a batch-computed feature (updated daily) for a real-time fraud model. By the time the feature is stale, the fraud pattern has already happened. Solution: match feature freshness to decision cadence. Real-time decisions need real-time features (or at least hourly aggregations).

5. **Not documenting feature logic** — Feature named "feature_42" with no documentation. Original creator left the company. Nobody knows what it means, whether it's still correct, or if the source data changed. Solution: feature catalog with: name, description, business rationale, SQL/code, data source, owner, freshness, known issues.

---

## Key Takeaways

- Feature quality > model complexity — good features + simple model beats bad features + complex model
- Leakage is the #1 killer: always ask "would I have this at prediction time?"
- Point-in-time correctness: features must reflect state BEFORE the prediction moment
- Feature freshness must match decision cadence (real-time decisions need real-time features)
- Start simple, add complexity incrementally — measure improvement at each step
- Domain knowledge produces the best features (talk to experts, understand the business)
- Feature importance analysis: remove features that don't help (reduces noise and latency)
- Document everything: name, logic, source, freshness, owner — features are shared assets
- Production constraint: feature must be computable within latency budget at serving time
- Feature store: single definition for training and serving eliminates train-serve skew
