# Data Drift Detection

## The Problem / Why This Matters

Data drift (also called covariate shift or feature drift) occurs when the statistical distribution of input features changes between training and production. Your model learned patterns from training data with specific distributions — when production data looks different, the model is operating outside its training domain, and predictions become unreliable. Examples: a fraud model trained on US transactions starts receiving transactions from a new market (different amounts, merchants, patterns); a recommendation model trained pre-holiday encounters holiday shopping behavior; a credit scoring model sees a surge of first-time applicants with no credit history. Detecting data drift early — before it causes performance degradation — is the primary value of drift monitoring. You can act (retrain, adjust, investigate) before users or business metrics are impacted. The challenge: not all drift matters equally. Some features drift constantly without affecting predictions (benign drift), while small shifts in high-importance features can cause significant degradation. Effective drift detection distinguishes signal from noise.

---

## The Analogy

Think of data drift detection like a weather station monitoring climate change:

- **Reference climate** = Your training data. You know what "normal" looks like — temperature ranges, rainfall patterns, wind speeds for your region.
- **Current readings** = Production data. You're continuously measuring today's conditions.
- **Drift detection** = Comparing current readings to historical norms. "Is today's data consistent with what we've seen before?"
- **Not all change is alarming** = Temperature varies daily (noise), but if the 30-day average is 5°C higher than the historical average, that's meaningful drift.
- **Impact assessment** = "The temperature changed, but do our crops still grow?" = "The feature drifted, but does the model still perform well?"

You need the station (monitoring), the historical baseline (reference), the comparison method (statistical tests), and the impact assessment (correlation with performance).

---

## Deep Dive

### Statistical Tests for Drift Detection

```yaml
Statistical_Tests:
  kolmogorov_smirnov_test:
    abbreviation: "KS test"
    type: "Non-parametric, continuous features"
    what: "Maximum difference between two CDFs (Cumulative Distribution Functions)"
    statistic: "D = max|F_reference(x) - F_current(x)|"
    p_value: "Probability of observing this difference by chance"
    thresholds:
      p_value_less_than_0.05: "Statistically significant drift"
      p_value_less_than_0.01: "Highly significant drift"
    pros: "No assumptions about distribution shape, well-understood"
    cons: "Sensitive to sample size (large samples always find 'drift'), only for continuous"
    
  population_stability_index:
    abbreviation: "PSI"
    type: "Binned comparison, continuous or categorical"
    what: "Measures how much a distribution has shifted relative to reference"
    formula: "PSI = Σ (P_current_i - P_reference_i) × ln(P_current_i / P_reference_i)"
    thresholds:
      less_than_0.1: "No significant change"
      between_0.1_and_0.2: "Moderate change (investigate)"
      greater_than_0.2: "Significant change (action needed)"
    pros: "Intuitive scale, works for binned data, industry standard"
    cons: "Sensitive to bin choice, zero bins need smoothing"
    
  chi_squared_test:
    abbreviation: "χ² test"
    type: "Categorical features"
    what: "Tests if observed category frequencies differ from expected"
    formula: "χ² = Σ (O_i - E_i)² / E_i"
    use: "Compare category distribution in current window vs reference"
    pros: "Standard test for categorical data"
    cons: "Requires sufficient counts per category, doesn't work for rare categories"
    
  wasserstein_distance:
    abbreviation: "Earth Mover's Distance (EMD)"
    type: "Continuous features"
    what: "Minimum 'work' to transform one distribution into another"
    intuition: "If distributions are piles of dirt, how much dirt must you move?"
    pros: "Intuitive physical meaning, captures magnitude of shift"
    cons: "Scale-dependent (normalize features first)"
    
  jensen_shannon_divergence:
    abbreviation: "JSD"
    type: "Both continuous and categorical"
    what: "Symmetric version of KL (Kullback-Leibler) divergence"
    range: "[0, 1] where 0 = identical distributions"
    pros: "Bounded, symmetric, works for both types"
    cons: "Requires binning for continuous features"
    
  page_hinkley_test:
    type: "Sequential / streaming"
    what: "Detects change point in a stream of values"
    use: "Online drift detection (detect when drift starts, not just that it exists)"
    pros: "Works with streaming data, detects change point"
    cons: "Requires tuning threshold parameter"
```

### Implementation

```python
# Data drift detection implementation

"""
Comprehensive data drift detection for production models.
Compares current feature distributions against reference (training-time) distributions.
"""

import numpy as np
from scipy import stats
from typing import Optional
from dataclasses import dataclass


@dataclass
class DriftResult:
    """Result of a drift check for a single feature."""
    feature_name: str
    test_name: str
    statistic: float
    p_value: Optional[float]
    is_drifted: bool
    severity: str  # "none", "low", "moderate", "high"
    reference_stats: dict
    current_stats: dict


class DataDriftDetector:
    """
    Multi-method drift detection for feature distributions.
    
    Usage:
        detector = DataDriftDetector(reference_data)
        results = detector.check_drift(current_data)
    """
    
    def __init__(
        self, 
        reference_data: dict,  # {feature_name: np.array of reference values}
        continuous_features: list[str] = None,
        categorical_features: list[str] = None,
        psi_threshold: float = 0.2,
        ks_p_value_threshold: float = 0.01,
    ):
        self.reference = reference_data
        self.continuous_features = continuous_features or []
        self.categorical_features = categorical_features or []
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_p_value_threshold
        
        # Pre-compute reference statistics
        self.reference_stats = self._compute_reference_stats()
    
    def check_drift(self, current_data: dict) -> list[DriftResult]:
        """
        Check all features for drift against reference.
        Returns list of DriftResult (one per feature).
        """
        results = []
        
        for feature_name in self.reference:
            if feature_name not in current_data:
                continue
                
            current_values = np.array(current_data[feature_name])
            reference_values = np.array(self.reference[feature_name])
            
            if feature_name in self.continuous_features:
                result = self._check_continuous_drift(
                    feature_name, reference_values, current_values
                )
            elif feature_name in self.categorical_features:
                result = self._check_categorical_drift(
                    feature_name, reference_values, current_values
                )
            else:
                # Auto-detect: if few unique values → categorical, else continuous
                if len(np.unique(current_values)) < 20:
                    result = self._check_categorical_drift(
                        feature_name, reference_values, current_values
                    )
                else:
                    result = self._check_continuous_drift(
                        feature_name, reference_values, current_values
                    )
            
            results.append(result)
        
        return results
    
    def _check_continuous_drift(
        self, name: str, reference: np.ndarray, current: np.ndarray
    ) -> DriftResult:
        """Check drift for continuous feature using KS test + PSI."""
        
        # KS test
        ks_stat, ks_p_value = stats.ks_2samp(reference, current)
        
        # PSI
        psi_value = self._compute_psi(reference, current)
        
        # Determine severity based on PSI (more interpretable)
        if psi_value < 0.1:
            severity = "none"
            is_drifted = False
        elif psi_value < 0.2:
            severity = "low"
            is_drifted = False  # Investigate but don't alert
        elif psi_value < 0.5:
            severity = "moderate"
            is_drifted = True
        else:
            severity = "high"
            is_drifted = True
        
        return DriftResult(
            feature_name=name,
            test_name="KS + PSI",
            statistic=psi_value,
            p_value=float(ks_p_value),
            is_drifted=is_drifted,
            severity=severity,
            reference_stats={
                "mean": float(np.mean(reference)),
                "std": float(np.std(reference)),
                "median": float(np.median(reference)),
            },
            current_stats={
                "mean": float(np.mean(current)),
                "std": float(np.std(current)),
                "median": float(np.median(current)),
            },
        )
    
    def _check_categorical_drift(
        self, name: str, reference: np.ndarray, current: np.ndarray
    ) -> DriftResult:
        """Check drift for categorical feature using chi-squared test."""
        
        # Get all categories from both
        all_categories = list(set(reference) | set(current))
        
        # Count frequencies
        ref_counts = np.array([np.sum(reference == cat) for cat in all_categories])
        cur_counts = np.array([np.sum(current == cat) for cat in all_categories])
        
        # Normalize to proportions
        ref_proportions = ref_counts / max(ref_counts.sum(), 1)
        cur_proportions = cur_counts / max(cur_counts.sum(), 1)
        
        # Chi-squared test
        # Scale current proportions to expected counts based on reference
        expected = ref_proportions * cur_counts.sum()
        expected = np.clip(expected, 0.001, None)  # Avoid zeros
        
        chi2_stat, chi2_p = stats.chisquare(cur_counts, f_exp=expected)
        
        # PSI for categorical
        psi_value = self._compute_psi_categorical(ref_proportions, cur_proportions)
        
        severity = "none" if psi_value < 0.1 else "low" if psi_value < 0.2 else "moderate" if psi_value < 0.5 else "high"
        
        return DriftResult(
            feature_name=name,
            test_name="Chi-squared + PSI",
            statistic=psi_value,
            p_value=float(chi2_p),
            is_drifted=psi_value >= self.psi_threshold,
            severity=severity,
            reference_stats={"distribution": dict(zip(all_categories, ref_proportions.tolist()))},
            current_stats={"distribution": dict(zip(all_categories, cur_proportions.tolist()))},
        )
    
    def _compute_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Compute PSI for continuous feature."""
        # Create bins from reference
        breakpoints = np.quantile(reference, np.linspace(0, 1, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        ref_percents = np.histogram(reference, bins=breakpoints)[0] / len(reference)
        cur_percents = np.histogram(current, bins=breakpoints)[0] / len(current)
        
        # Smoothing to avoid log(0)
        ref_percents = np.clip(ref_percents, 0.0001, None)
        cur_percents = np.clip(cur_percents, 0.0001, None)
        
        psi = np.sum((cur_percents - ref_percents) * np.log(cur_percents / ref_percents))
        return float(psi)
    
    def _compute_psi_categorical(
        self, ref_proportions: np.ndarray, cur_proportions: np.ndarray
    ) -> float:
        """Compute PSI for categorical feature."""
        ref_p = np.clip(ref_proportions, 0.0001, None)
        cur_p = np.clip(cur_proportions, 0.0001, None)
        return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
    
    def _compute_reference_stats(self) -> dict:
        """Pre-compute statistics on reference data for dashboards."""
        stats_dict = {}
        for name, values in self.reference.items():
            values = np.array(values)
            stats_dict[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "quantiles": {
                    "p5": float(np.quantile(values, 0.05)),
                    "p25": float(np.quantile(values, 0.25)),
                    "p50": float(np.quantile(values, 0.50)),
                    "p75": float(np.quantile(values, 0.75)),
                    "p95": float(np.quantile(values, 0.95)),
                },
            }
        return stats_dict
```

### Windowing Strategies

```yaml
Windowing:
  fixed_window:
    what: "Compare last N hours/days against reference"
    example: "Compare last 24 hours of data vs training distribution"
    pros: "Simple, predictable compute cost"
    cons: "Misses gradual drift within the window"
    
  sliding_window:
    what: "Rolling window that advances over time"
    example: "Compare last 1000 predictions vs reference, recompute every 100 predictions"
    pros: "Smoother signal, catches gradual changes"
    cons: "More computation, needs careful deduplication"
    
  expanding_window:
    what: "Window grows over time (all data since deployment)"
    example: "Compare ALL production data since deploy vs training"
    pros: "Global view, not fooled by short-term fluctuations"
    cons: "Dilutes recent drift signal (old data dominates)"
    
  adaptive_window:
    what: "Window size adjusts based on drift detection"
    example: "Normally 24h window; when drift detected, shrink to 1h to localize change point"
    pros: "Responsive to actual changes"
    cons: "More complex implementation"
    
  recommended:
    for_alerting: "Fixed 1-hour window, checked hourly"
    for_dashboards: "Sliding 24-hour window, recomputed every hour"
    for_reports: "Weekly window for trend analysis"
```

### Multivariate Drift Detection

```yaml
Multivariate_Drift:
  why_needed: "Individual features may not drift, but their JOINT distribution changes"
  example: "Age and income individually stable, but the age-income RELATIONSHIP changed (younger high-earners appearing)"
  
  methods:
    maximum_mean_discrepancy:
      abbreviation: "MMD"
      what: "Kernel-based test comparing joint distributions"
      how: "Map data to high-dimensional kernel space, compare means"
      pros: "Captures complex multi-dimensional drift"
      cons: "Computationally expensive, hard to interpret"
      
    domain_classifier:
      what: "Train a model to distinguish reference from current data"
      how: "If a classifier can separate old vs new data → drift exists"
      metric: "AUC of the domain classifier (0.5 = no drift, 1.0 = complete drift)"
      pros: "Intuitive, captures complex patterns, shows WHICH features contribute"
      cons: "Requires training a model (adds complexity)"
      
    pca_reconstruction_error:
      abbreviation: "PCA (Principal Component Analysis)"
      what: "Project data through reference PCA, measure reconstruction error"
      how: "High reconstruction error on current data = data is 'out of domain'"
      pros: "Fast, captures multivariate structure"
      cons: "Linear assumption may miss non-linear drift"
      
    embedding_drift:
      what: "Compare embedding space distributions"
      how: "Encode features through model's embedding layer, compare distributions in that space"
      pros: "Detects drift in the model's representation space (most relevant)"
      cons: "Model-specific, requires access to intermediate layers"
```

---

## How It Works in Practice

### Drift Detection Pipeline

```yaml
Pipeline:
  data_collection:
    - "Log all prediction requests with full feature vectors"
    - "Store in time-partitioned table (hourly/daily partitions)"
    
  reference_creation:
    - "At deployment: save training data distributions"
    - "Compute and store: histograms, quantiles, means, stds per feature"
    - "Store reference sample (10K-100K representative examples)"
    
  drift_computation:
    schedule: "Every hour"
    steps:
      1: "Read last hour's prediction logs (features only)"
      2: "For each feature: compute PSI vs reference"
      3: "For top important features: run KS test"
      4: "Multivariate: compute domain classifier AUC"
      5: "Write results to monitoring DB"
      
  alerting:
    info: "PSI between 0.1-0.2 on any feature → log to dashboard"
    warning: "PSI > 0.2 on important feature → Slack notification"
    critical: "PSI > 0.2 on 3+ features AND performance drop → PagerDuty"
    
  dashboard:
    - "PSI heatmap over time (features × time windows)"
    - "Top drifted features (ranked by PSI)"
    - "Feature distribution comparison (reference vs current histogram)"
    - "Drift timeline (when did drift start?)"
```

---

## Interview Tip

> When asked about data drift detection: "I use a layered approach to drift detection. For individual continuous features, I compute PSI (Population Stability Index) — it's industry standard, interpretable (0.1 = investigate, 0.2 = action), and works well in practice. For categorical features, chi-squared test with PSI. I run these hourly on a 1-hour window of production data compared against reference distributions saved at deployment time. Key nuance: I weight drift detection by feature importance. If a low-importance feature drifts (PSI=0.3) but the top-10 features are stable, I investigate but don't alert. If a top-3 importance feature drifts even slightly (PSI=0.15), I investigate immediately. For multivariate drift (joint distribution changes not visible in individual features), I use a domain classifier — train a simple model to distinguish 'reference' vs 'current' data. AUC > 0.65 suggests meaningful multivariate drift. The key engineering decision: how to avoid alert fatigue. Not all drift matters. I correlate drift signals with downstream performance metrics — only page on-call when drift + performance drop are both observed. This reduces false alerts by 80%+ while still catching real issues."

---

## Common Mistakes

1. **Same threshold for all features** — Using PSI > 0.2 as the alert threshold for every feature equally. Low-importance features trigger alerts that don't matter. Solution: weight thresholds by feature importance. Top features get lower thresholds (more sensitive), low-importance features get higher thresholds (less sensitive).

2. **Too small reference sample** — Using 100 training examples as reference. Any current sample will appear "different" just from sampling noise. Solution: use 10K-100K reference samples (or the full training set statistics). Larger reference = more stable comparison.

3. **Not accounting for known patterns** — Model serves e-commerce users → weekday vs. weekend distributions are naturally different. Every Monday, drift detector fires. Solution: compare against time-matched reference (Monday vs. historical Mondays), or account for known seasonal patterns in thresholds.

4. **Univariate only** — Checking each feature independently, missing that the joint distribution changed. Age stable, income stable, but age-income correlation flipped (different population). Solution: add multivariate drift detection (domain classifier, MMD, or PCA reconstruction error) alongside univariate checks.

5. **Acting on drift without checking performance** — "Feature X drifted! Emergency retrain!" But model performance is unchanged — the drift was in a feature the model barely uses. Solution: always correlate drift with performance. Drift alone is informational; drift + performance drop = actionable.

---

## Key Takeaways

- Data drift: input feature distributions change between training and production
- PSI (Population Stability Index): <0.1 stable, 0.1-0.2 investigate, >0.2 act
- KS test: non-parametric, detects any distribution difference (continuous features)
- Chi-squared: standard test for categorical feature drift
- Weight detection by feature importance: drift in top features matters most
- Multivariate drift: domain classifier (AUC > 0.65 = drift) catches joint distribution changes
- Reference data: save training-time distributions at deployment (10K-100K samples)
- Windowing: hourly fixed window for alerting, daily sliding for dashboards
- Correlate with performance: drift alone is informational, drift + performance drop = alert
- Avoid alert fatigue: not all drift is actionable — distinguish benign from harmful drift
