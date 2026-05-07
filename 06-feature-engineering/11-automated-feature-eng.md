# Automated Feature Engineering

## The Problem / Why This Matters

Manual feature engineering is the bottleneck in most ML projects — a senior ML engineer might spend 60-80% of their time crafting features, trying aggregations, testing interactions, and iterating on transformations. Automated Feature Engineering (AutoFE) uses algorithms to systematically generate, evaluate, and select features without manual intervention. This includes: (1) automatic generation — algorithmically create hundreds or thousands of candidate features from raw data (all possible aggregations, interactions, transformations), (2) automatic selection — identify which generated features actually improve model performance and discard the rest, (3) automatic transformation — find optimal encodings and transformations for each feature type. The landscape in 2026: tools like Featuretools (Deep Feature Synthesis), tsfresh (time series), AutoML platforms (Google Vertex AI AutoML, H2O Driverless AI, AWS AutoGluon), and increasingly LLM-powered feature suggestion (Claude 4, GPT-5 suggesting feature ideas from data descriptions). The engineering challenge: generated features need to be production-ready (efficient to compute, stable over time, interpretable enough to debug), not just good in offline evaluation.

---

## The Analogy

Think of automated feature engineering like a kitchen robot vs. a human chef:

**Manual feature engineering** = Expert chef who knows which spice combinations work, what temperature brings out flavors, how long to marinate. Produces excellent results but takes years of experience and hours per dish.

**Automated feature engineering** = Kitchen robot that systematically tries every possible spice combination, every temperature, every duration — runs 10,000 experiments in parallel, then serves you only the combinations that taste great. It discovers things the chef never thought to try (unexpected flavor pairings), but also generates a lot of garbage that needs to be filtered out.

The best approach in 2026: the robot generates candidates (speed and breadth), the chef curates and selects (domain knowledge and judgment).

---

## Deep Dive

### Feature Generation Techniques

```yaml
Generation_Methods:
  deep_feature_synthesis:
    tool: "Featuretools"
    what: "Automatically generate features by stacking aggregation primitives"
    how:
      - "Define entity relationships (users → orders → items)"
      - "Apply transformation primitives to each entity (mean, count, max, min, std)"
      - "Stack primitives across relationships (mean of sum of item prices per user)"
      - "Generate hundreds of candidate features automatically"
    example:
      input: "users table, orders table, items table"
      output:
        - "users.COUNT(orders)" — how many orders per user
        - "users.MEAN(orders.total)" — average order value per user
        - "users.MAX(orders.COUNT(items))" — largest cart size per user
        - "users.STD(orders.total)" — spending variability per user
        - "users.MEAN(orders.items.MEAN(price))" — avg of avg item price
    depth: "Depth=1 (direct aggs), Depth=2 (nested), Depth=3 (deep nested)"
    trade_off: "Higher depth = more features but most are useless noise"
    
  time_series_features:
    tool: "tsfresh"
    what: "Automatically extract features from time series data"
    categories:
      statistical: "Mean, variance, skewness, kurtosis, quantiles"
      frequency: "FFT coefficients, spectral density"
      temporal: "Autocorrelation, partial autocorrelation, trend"
      complexity: "Approximate entropy, sample entropy"
      peak_detection: "Number of peaks, peak heights"
    volume: "800+ features extracted per time series"
    
  interaction_features:
    what: "Combinations of existing features"
    types:
      multiplicative: "feature_A × feature_B"
      ratio: "feature_A / feature_B"
      difference: "feature_A - feature_B"
      polynomial: "feature_A², feature_A³"
    when_useful: "When the relationship between features is non-additive"
    risk: "Explosion of features (N features → N² interactions)"
    
  transformation_features:
    what: "Mathematical transforms of existing features"
    transforms:
      - "log(x), sqrt(x), 1/x (handle skewness)"
      - "sin(x), cos(x) (cyclical features like hour-of-day)"
      - "binning (continuous → categorical)"
      - "quantile transform (normalize distribution)"
      - "power transform (Box-Cox, Yeo-Johnson)"
```

### Automated Feature Selection

```python
# Feature selection: pick the best from hundreds of generated features

"""
After generating 500+ candidate features, select the ones that:
1. Improve model performance
2. Are not redundant (correlated with other selected features)
3. Are stable (work on train AND test data)
4. Are computationally feasible for production
"""

import numpy as np
from sklearn.feature_selection import (
    mutual_info_classif, mutual_info_regression,
    SelectKBest, RFE, SequentialFeatureSelector,
)
from sklearn.ensemble import RandomForestClassifier


class AutomatedFeatureSelector:
    """Multi-stage feature selection pipeline."""
    
    def __init__(self, task: str = "classification", max_features: int = 50):
        self.task = task
        self.max_features = max_features
    
    def select(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> list:
        """
        Multi-stage selection:
        Stage 1: Remove zero-variance and near-constant features
        Stage 2: Remove highly correlated features (redundancy)
        Stage 3: Statistical relevance (mutual information)
        Stage 4: Model-based importance (permutation importance)
        Stage 5: Stability selection (consistent across subsamples)
        """
        
        selected = feature_names.copy()
        
        # Stage 1: Remove zero/near-zero variance
        selected = self._filter_low_variance(X, selected, threshold=0.01)
        print(f"After variance filter: {len(selected)} features")
        
        # Stage 2: Remove highly correlated features
        selected = self._filter_correlated(X, selected, threshold=0.95)
        print(f"After correlation filter: {len(selected)} features")
        
        # Stage 3: Statistical relevance (mutual information)
        selected = self._filter_mutual_info(X, y, selected, top_k=min(200, len(selected)))
        print(f"After MI filter: {len(selected)} features")
        
        # Stage 4: Model-based importance
        selected = self._filter_model_importance(X, y, selected, top_k=self.max_features)
        print(f"After model importance: {len(selected)} features")
        
        # Stage 5: Stability check (features that survive across subsamples)
        selected = self._stability_selection(X, y, selected)
        print(f"After stability check: {len(selected)} features (FINAL)")
        
        return selected
    
    def _filter_low_variance(self, X, features, threshold):
        """Remove features with near-zero variance."""
        variances = np.var(X, axis=0)
        mask = variances > threshold
        return [f for f, keep in zip(features, mask) if keep]
    
    def _filter_correlated(self, X, features, threshold):
        """Remove one of each pair of highly correlated features."""
        corr_matrix = np.abs(np.corrcoef(X.T))
        to_remove = set()
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                if corr_matrix[i, j] > threshold:
                    to_remove.add(features[j])  # Remove the second one
        return [f for f in features if f not in to_remove]
    
    def _filter_mutual_info(self, X, y, features, top_k):
        """Keep features with highest mutual information with target."""
        if self.task == "classification":
            mi_scores = mutual_info_classif(X, y)
        else:
            mi_scores = mutual_info_regression(X, y)
        
        top_indices = np.argsort(mi_scores)[-top_k:]
        return [features[i] for i in top_indices]
    
    def _filter_model_importance(self, X, y, features, top_k):
        """Keep features with highest model-based importance."""
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        model.fit(X, y)
        
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-top_k:]
        return [features[i] for i in top_indices]
    
    def _stability_selection(self, X, y, features, n_subsamples=20, threshold=0.7):
        """Keep features selected in >70% of subsampled runs."""
        selection_counts = {f: 0 for f in features}
        
        for _ in range(n_subsamples):
            # Subsample 80% of data
            indices = np.random.choice(len(X), size=int(0.8 * len(X)), replace=False)
            X_sub, y_sub = X[indices], y[indices]
            
            # Fit model and get top features
            model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
            model.fit(X_sub, y_sub)
            
            top_k = min(len(features) // 2, self.max_features)
            top_indices = np.argsort(model.feature_importances_)[-top_k:]
            for idx in top_indices:
                selection_counts[features[idx]] += 1
        
        # Keep features selected in >threshold fraction of runs
        stable_features = [
            f for f, count in selection_counts.items() 
            if count / n_subsamples >= threshold
        ]
        return stable_features
```

### AutoML Feature Engineering

```yaml
AutoML_Platforms:
  h2o_driverless_ai:
    approach: "Genetic algorithm to evolve feature transformations"
    capabilities:
      - "Automatic interaction features"
      - "Target encoding with regularization"
      - "Time series lag/rolling features"
      - "Text features (TF-IDF, embeddings)"
      - "Image features (pre-trained CNN outputs)"
    unique: "Evolutionary approach finds non-obvious transformations"
    
  google_vertex_automl:
    approach: "Neural Architecture Search + feature preprocessing"
    capabilities:
      - "Automatic type detection and encoding"
      - "Missing value imputation"
      - "Feature crosses for tabular data"
      - "Embedding learning for categorical features"
    unique: "Neural architecture designed for the specific features"
    
  aws_autogluon:
    approach: "Multi-model ensemble with automatic feature processing"
    capabilities:
      - "Automatic categorical encoding"
      - "Text and image feature extraction"
      - "Time-aware feature generation"
      - "Stacked ensembles"
    unique: "No feature engineering needed — model handles raw data"
    
  featuretools:
    approach: "Deep Feature Synthesis (DFS) — systematic aggregation stacking"
    capabilities:
      - "Entity relationship modeling"
      - "Primitive stacking (depth 1, 2, 3)"
      - "Custom primitive definition"
      - "Cutoff-time aware (point-in-time correct)"
    unique: "Transparent, reproducible, production-ready features"
    
  tsfresh:
    approach: "Exhaustive time series feature extraction"
    capabilities:
      - "800+ feature calculators"
      - "Automatic relevance filtering"
      - "Parallelized computation"
      - "Custom feature calculators"
    unique: "Specialized for time series — covers statistical, frequency, and complexity features"
```

### LLM-Powered Feature Engineering

```yaml
LLM_Feature_Engineering:
  what: "Use LLMs (Large Language Models) to suggest and generate features"
  approach: "Describe your data and task → LLM suggests feature ideas → validate"
  
  workflow:
    1_describe_data:
      prompt: |
        I have these tables for churn prediction:
        - users (user_id, signup_date, plan_type, country)
        - orders (order_id, user_id, date, amount, status)
        - sessions (session_id, user_id, timestamp, duration, pages_viewed)
        - support_tickets (ticket_id, user_id, date, category, resolution_time)
        
    2_request_features:
      prompt: "Suggest 50 features that would be predictive of user churn. 
               For each, provide: name, computation logic (SQL), and reasoning."
               
    3_llm_output:
      examples:
        - name: "days_since_last_session"
          sql: "DATEDIFF(CURRENT_DATE, MAX(sessions.timestamp)) WHERE user_id = ?"
          reasoning: "Longer absence → higher churn risk"
          
        - name: "session_duration_trend_30d"
          sql: "REGR_SLOPE(duration, ROW_NUMBER) OVER (PARTITION BY user_id ORDER BY timestamp ROWS 30 PRECEDING)"
          reasoning: "Declining session duration = disengagement"
          
        - name: "support_unresolved_ratio"
          sql: "COUNT(CASE WHEN resolution_time IS NULL THEN 1 END) / COUNT(*)"
          reasoning: "High unresolved tickets = frustration = churn"
          
    4_validate:
      - "Check if feature is computable from available data"
      - "Check if feature has point-in-time correctness issues"
      - "Test on historical data (does it actually correlate with target?)"
      - "Estimate computation cost (can we compute this in production?)"
      
  benefits:
    - "Domain knowledge injection (LLM knows common churn signals)"
    - "Faster ideation (50 ideas in 30 seconds vs. 50 ideas in 2 days)"
    - "Novel combinations (suggestions humans might not think of)"
    
  limitations:
    - "LLM doesn't know YOUR specific data quality issues"
    - "Suggestions may not be feasible with your infrastructure"
    - "Still need validation (not all suggestions are useful)"
    - "May suggest features with leakage (needs human review)"
    
  tools_2026:
    - "GitHub Copilot agent mode — suggests features in notebook context"
    - "Claude 4 / GPT-5 — multi-step feature engineering workflows"
    - "Specialized AutoFE agents — fine-tuned for feature ideation"
```

### Production Considerations for Auto-Generated Features

```yaml
Production_Reality:
  challenges:
    interpretability:
      problem: "Auto-generated feature: MEAN(MAX(items.price) WHERE orders.date > -30d)"
      issue: "Hard to explain to stakeholders why model uses this"
      solution: "Keep interpretable features for regulated domains; use auto features for non-regulated"
      
    computation_cost:
      problem: "Deep Feature Synthesis with depth=3 generates 5000 features"
      issue: "Computing all at serving time is too slow/expensive"
      solution: "Select top N features, precompute heavy ones, optimize SQL"
      
    stability:
      problem: "Auto-generated feature works great on training data but is noisy on new data"
      issue: "Overfitting to training distribution, unstable on small counts"
      solution: "Stability selection, minimum support thresholds, regularization"
      
    maintenance:
      problem: "500 auto-generated features with opaque names and no documentation"
      issue: "Nobody understands what breaks when upstream data changes"
      solution: "Auto-generate documentation, add lineage tracking, name features descriptively"
      
  best_practices:
    generate_broadly: "Create thousands of candidates (cheap in offline setting)"
    select_ruthlessly: "Keep only features that pass ALL selection stages"
    name_descriptively: "Auto-name based on computation: entity_agg_field_window"
    document_automatically: "Generate docs from feature definition (computation + stats)"
    monitor_in_production: "Track feature distribution drift even for auto-features"
    set_maximum: "Cap at 50-100 features per model (diminishing returns beyond)"
```

---

## How It Works in Practice

### End-to-End AutoFE Pipeline

```yaml
Pipeline:
  step_1_generate:
    tool: "Featuretools Deep Feature Synthesis"
    config: "Depth=2, all numeric primitives, cutoff_time for point-in-time"
    output: "800 candidate features"
    time: "2 hours on Spark cluster"
    
  step_2_filter:
    tool: "Custom selection pipeline"
    stages:
      - "Remove zero-variance → 650 remain"
      - "Remove >0.95 correlated → 300 remain"
      - "Mutual information filter (top 150) → 150 remain"
      - "Model importance (top 80) → 80 remain"
      - "Stability selection (>70% subsample) → 45 remain"
    output: "45 production-ready features"
    time: "1 hour"
    
  step_3_validate:
    checks:
      - "Point-in-time correctness (no future leakage)"
      - "Computation feasibility (can compute in <30s at serving time)"
      - "Distribution stability (consistent across time periods)"
    output: "42 validated features (3 removed for leakage risk)"
    
  step_4_productionize:
    - "Generate feature pipeline code (Spark SQL)"
    - "Register in feature catalog with auto-documentation"
    - "Add to daily feature pipeline"
    - "Deploy to online feature store"
    
  outcome:
    time_saved: "Manual feature engineering: 3 weeks → AutoFE: 2 days"
    quality: "AutoFE found 5 features humans missed (interaction effects)"
    model_improvement: "+3% AUC over manually-engineered baseline"
```

---

## Interview Tip

> When asked about automated feature engineering: "I use AutoFE as a complement to manual feature engineering, not a replacement. My approach: (1) Generate broadly using Deep Feature Synthesis (Featuretools) — systematically create all possible aggregations across entity relationships at depth 2. This produces 500-1000 candidate features in hours, including interactions I might not think of manually. (2) Select ruthlessly through a multi-stage pipeline: remove near-zero variance, remove correlated pairs (>0.95), filter by mutual information, rank by model importance, then stability selection (features that survive across 80% of subsamples). This typically reduces 800 candidates to 40-50 production features. (3) Validate for production: check point-in-time correctness (no leakage), computation cost (can we compute at serving time?), and temporal stability (consistent across time periods). In 2026, I also use LLMs for feature ideation — describe the dataset and task to Claude 4 or GPT-5, get 50 feature suggestions in 30 seconds, validate which are feasible and useful. The key engineering insight: auto-generated features need the same production rigor as manual features — pipeline code, monitoring, documentation, and freshness SLAs. Without that, they're research artifacts, not production features."

---

## Common Mistakes

1. **Using all generated features without selection** — Deep Feature Synthesis produces 2000 features, team dumps all into the model. Result: overfitting, slow training, impossible to debug, fragile pipeline. Solution: aggressive multi-stage selection. 2000 → 50. More features ≠ better model.

2. **No stability check** — Feature looks great on one train/test split but is noise on different subsamples. Model's performance varies wildly on different data slices. Solution: stability selection — only keep features that are consistently important across many subsampled runs.

3. **Auto-features with data leakage** — Automated process generates feature that uses future information (e.g., aggregation without time cutoff). Looks amazing in evaluation, fails in production. Solution: point-in-time validation is NON-NEGOTIABLE for auto-features. Use cutoff_time in Featuretools.

4. **Ignoring computation cost** — Auto-generated feature requires joining 5 tables with window functions over 90 days. Works fine for batch training but takes 30 seconds to compute at serving time. Solution: evaluate computation cost as part of selection. Features must be computable within latency SLA.

5. **Replacing domain knowledge entirely** — Team uses only AutoFE, skips domain expert input. Misses obvious features that domain expert would suggest immediately (e.g., "day of week is important for this business"). Solution: combine — let domain experts suggest key features, then use AutoFE to find interactions and non-obvious signals humans miss.

---

## Key Takeaways

- Automated Feature Engineering: generate broadly, select ruthlessly, validate for production
- Deep Feature Synthesis: systematic aggregation stacking across entity relationships (Featuretools)
- Time series: tsfresh extracts 800+ features automatically from temporal data
- Multi-stage selection: variance → correlation → mutual info → model importance → stability
- LLM-powered ideation: describe data to Claude 4/GPT-5, get feature suggestions, validate
- Point-in-time correctness: auto-generated features MUST respect temporal boundaries
- Computation cost: select only features computable within serving latency SLA
- Stability: features must survive across multiple data subsamples (not just one split)
- Combine with domain knowledge: AutoFE finds interactions, humans provide business logic
- Production rigor: auto-features need same pipeline, monitoring, and documentation as manual ones
- Cap features: 50-100 per model (diminishing returns beyond, increases maintenance burden)
