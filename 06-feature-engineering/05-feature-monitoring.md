# Feature Monitoring

## The Problem / Why This Matters

Features degrade silently. A pipeline change upstream removes a field — your feature becomes all nulls. A business process changes — customer behavior shifts and your feature's distribution drifts. A data source goes stale — your "real-time" feature hasn't updated in 3 days. A schema migration changes a column type — your numeric feature becomes strings. Without monitoring, you don't know features have degraded until model performance drops in production (days to weeks later), customers complain, or revenue decreases. By then, the damage is done and root cause analysis is expensive. Feature monitoring catches problems at the feature level — before they propagate to model predictions. It answers: Are features still being computed? (staleness) Are feature values within expected ranges? (quality) Has the distribution shifted from training? (drift) Are null rates acceptable? (completeness) Feature monitoring is your early warning system — it detects problems days before they become model failures.

---

## The Analogy

Think of feature monitoring like quality control in manufacturing:

- **Without monitoring** = Factory produces widgets. No inspection at any stage. Bad raw materials come in, defective products go out. You only discover the problem when customers return products (model performance drops weeks later).
- **With monitoring** = Factory has quality checkpoints at every stage:
  - **Incoming inspection** (data source monitoring) = Check raw materials quality before they enter the factory
  - **In-process inspection** (computation monitoring) = Verify each manufacturing step produces expected output
  - **Final inspection** (serving monitoring) = Check finished product before shipping to customer
  - **Statistical process control** (drift detection) = Monitor that processes stay within acceptable variation over time

If any checkpoint fails, production stops immediately (alerts fire) rather than producing defective products for days.

---

## Deep Dive

### What to Monitor

```yaml
Monitoring_Dimensions:
  freshness:
    what: "How recently was this feature updated?"
    metric: "time_since_last_update"
    alert: "Feature older than SLA (hourly feature > 2 hours old)"
    examples:
      stale_pipeline: "Airflow DAG failed silently → features not updated for 3 days"
      slow_materialization: "Batch compute finished but Redis write is stuck"
    monitoring:
      check: "Compare last_updated_timestamp against expected cadence"
      threshold: "2× expected update interval (hourly feature → alert if > 2h stale)"
      
  completeness:
    what: "What percentage of values are null/missing?"
    metric: "null_rate (percentage of entities with null value)"
    alert: "Null rate exceeds threshold (e.g., > 5% when historically < 1%)"
    examples:
      upstream_change: "Data source removed a column → feature becomes 100% null"
      join_failure: "Join key mismatch → 80% of entities have no features"
      new_entities: "Burst of new users → no historical features computed yet"
    monitoring:
      baseline: "Measure null rate during stable period"
      threshold: "Alert if null_rate > baseline + 2 standard deviations"
      
  distribution_drift:
    what: "Has the feature's value distribution changed from training?"
    metric: "Statistical distance between current and reference distribution"
    alert: "Distribution shift exceeds threshold"
    examples:
      business_change: "New pricing tier → 'average_order_value' distribution shifts"
      bug_introduced: "Feature computation bug → values 10x higher than normal"
      seasonality: "Holiday season → purchase count spikes (expected drift)"
    methods:
      ks_test: "Kolmogorov-Smirnov test (numerical features)"
      psi: "Population Stability Index (binned comparison)"
      js_divergence: "Jensen-Shannon divergence (probability distributions)"
      chi_squared: "Chi-squared test (categorical features)"
      
  value_range:
    what: "Are feature values within expected bounds?"
    metric: "min, max, mean, percentiles"
    alert: "Values outside expected range"
    examples:
      overflow: "Integer overflow → negative values for a count feature"
      unit_change: "Source changed from dollars to cents → 100x increase"
      impossible_values: "Age: -5, or percentage: 150%"
    monitoring:
      bounds: "Define expected min/max per feature"
      statistical: "Alert if mean deviates > 3σ from historical mean"
      
  cardinality:
    what: "For categorical features, how many distinct values?"
    metric: "Number of unique values"
    alert: "Cardinality increase/decrease beyond threshold"
    examples:
      data_quality: "Country code feature suddenly has 500 values (normally 195)"
      category_explosion: "Free-text field leaking into categorical feature"
      data_loss: "Only 3 categories remain (normally 50)"
      
  correlation:
    what: "Has the relationship between features and target changed?"
    metric: "Feature-target correlation over time"
    alert: "Correlation drops significantly"
    examples:
      concept_drift: "Feature that was highly predictive becomes irrelevant"
      proxy_failure: "Proxy feature no longer correlates with true signal"
```

### Statistical Tests for Drift Detection

```python
# Feature drift detection implementation

import numpy as np
from scipy import stats
from typing import Tuple


class FeatureDriftDetector:
    """Detect distribution drift in features."""
    
    def __init__(self, reference_data: np.ndarray, feature_type: str = "numerical"):
        """
        Args:
            reference_data: Training/reference distribution
            feature_type: "numerical" or "categorical"
        """
        self.reference = reference_data
        self.feature_type = feature_type
        
        # Pre-compute reference statistics
        if feature_type == "numerical":
            self.ref_mean = np.mean(reference_data)
            self.ref_std = np.std(reference_data)
            self.ref_percentiles = np.percentile(reference_data, [5, 25, 50, 75, 95])
        elif feature_type == "categorical":
            unique, counts = np.unique(reference_data, return_counts=True)
            self.ref_distribution = dict(zip(unique, counts / counts.sum()))
    
    def detect_drift(self, current_data: np.ndarray, threshold: float = 0.05) -> dict:
        """
        Detect if current data has drifted from reference.
        
        Returns:
            dict with drift_detected (bool), score, method, details
        """
        if self.feature_type == "numerical":
            return self._numerical_drift(current_data, threshold)
        else:
            return self._categorical_drift(current_data, threshold)
    
    def _numerical_drift(self, current: np.ndarray, threshold: float) -> dict:
        """Detect drift in numerical features using multiple methods."""
        
        results = {}
        
        # Method 1: KS Test (Kolmogorov-Smirnov)
        # Tests if two samples come from same distribution
        ks_stat, ks_pvalue = stats.ks_2samp(self.reference, current)
        results["ks_test"] = {
            "statistic": float(ks_stat),
            "p_value": float(ks_pvalue),
            "drift_detected": ks_pvalue < threshold,
        }
        
        # Method 2: PSI (Population Stability Index)
        psi_value = self._compute_psi(self.reference, current, buckets=10)
        results["psi"] = {
            "value": float(psi_value),
            "drift_detected": psi_value > 0.2,  # PSI > 0.2 = significant drift
            "interpretation": self._interpret_psi(psi_value),
        }
        
        # Method 3: Wasserstein distance (Earth Mover's Distance)
        wasserstein = stats.wasserstein_distance(self.reference, current)
        # Normalize by reference std for interpretability
        normalized_wasserstein = wasserstein / max(self.ref_std, 1e-10)
        results["wasserstein"] = {
            "value": float(wasserstein),
            "normalized": float(normalized_wasserstein),
            "drift_detected": normalized_wasserstein > 0.5,
        }
        
        # Overall decision (conservative: drift if ANY method detects)
        drift_detected = any(r["drift_detected"] for r in results.values())
        
        return {
            "drift_detected": drift_detected,
            "methods": results,
            "current_stats": {
                "mean": float(np.mean(current)),
                "std": float(np.std(current)),
                "null_rate": float(np.isnan(current).mean()),
            },
            "reference_stats": {
                "mean": float(self.ref_mean),
                "std": float(self.ref_std),
            },
        }
    
    def _compute_psi(self, reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
        """
        Compute PSI (Population Stability Index).
        
        PSI interpretation:
        - < 0.1: No significant drift
        - 0.1 - 0.2: Moderate drift (monitor)
        - > 0.2: Significant drift (investigate)
        """
        # Create buckets from reference distribution
        breakpoints = np.percentile(reference, np.linspace(0, 100, buckets + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        # Bin both distributions
        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        cur_counts = np.histogram(current, bins=breakpoints)[0]
        
        # Convert to proportions (with smoothing to avoid log(0))
        ref_pct = (ref_counts + 1) / (len(reference) + buckets)
        cur_pct = (cur_counts + 1) / (len(current) + buckets)
        
        # PSI formula
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        return psi
    
    def _interpret_psi(self, psi: float) -> str:
        if psi < 0.1:
            return "no_significant_change"
        elif psi < 0.2:
            return "moderate_drift_monitor"
        else:
            return "significant_drift_investigate"
    
    def _categorical_drift(self, current: np.ndarray, threshold: float) -> dict:
        """Detect drift in categorical features."""
        
        # Current distribution
        unique, counts = np.unique(current, return_counts=True)
        cur_distribution = dict(zip(unique, counts / counts.sum()))
        
        # Chi-squared test
        # Align categories between reference and current
        all_categories = set(self.ref_distribution.keys()) | set(cur_distribution.keys())
        
        ref_probs = [self.ref_distribution.get(c, 0.001) for c in all_categories]
        cur_counts_aligned = [cur_distribution.get(c, 0) * len(current) for c in all_categories]
        
        chi2_stat, chi2_pvalue = stats.chisquare(cur_counts_aligned, f_exp=[p * len(current) for p in ref_probs])
        
        # New categories appeared
        new_categories = set(cur_distribution.keys()) - set(self.ref_distribution.keys())
        disappeared_categories = set(self.ref_distribution.keys()) - set(cur_distribution.keys())
        
        return {
            "drift_detected": chi2_pvalue < threshold or len(new_categories) > 0,
            "chi2_statistic": float(chi2_stat),
            "chi2_p_value": float(chi2_pvalue),
            "new_categories": list(new_categories),
            "disappeared_categories": list(disappeared_categories),
            "current_cardinality": len(cur_distribution),
            "reference_cardinality": len(self.ref_distribution),
        }
```

### Monitoring Pipeline

```yaml
Monitoring_Pipeline:
  architecture:
    data_collection:
      source: "Feature store online/offline + serving logs"
      frequency: "Hourly (detailed), daily (comprehensive)"
      what_to_collect:
        - "Feature value distributions (sample 10K entities)"
        - "Null rates per feature"
        - "Last update timestamps"
        - "Value range statistics (min, max, p5, p50, p95)"
        
    drift_computation:
      frequency: "Hourly or daily"
      reference: "Training data distribution (snapshot at model training time)"
      current: "Last N hours/days of production feature values"
      output: "Drift score per feature + alert decisions"
      
    alerting:
      channels: "PagerDuty (critical), Slack (warning), dashboard (info)"
      severity_levels:
        critical: "Feature completely missing (100% null) or pipeline down"
        warning: "Significant drift detected or freshness SLA violated"
        info: "Moderate drift, requires monitoring"
        
    dashboards:
      feature_health: "Overview of all features: green/yellow/red status"
      drift_trends: "Drift scores over time per feature"
      freshness_map: "Time since last update for all features"
      null_rates: "Null rate trends per feature"
      
  tools:
    evidently:
      what: "Open-source ML monitoring (feature drift, model monitoring)"
      features: "Pre-built drift tests, visual reports, alerting"
      
    whylabs:
      what: "Managed feature monitoring platform"
      features: "Automatic drift detection, data profiling, anomaly detection"
      
    arize:
      what: "ML observability platform"
      features: "Feature importance drift, embedding monitoring, custom metrics"
      
    custom:
      what: "Build your own with Great Expectations + Prometheus + Grafana"
      when: "Specific requirements, budget constraints, existing infrastructure"
```

---

## How It Works in Practice

### Production Monitoring Setup

```python
# Feature monitoring job (runs hourly)

async def monitor_features():
    """Hourly feature monitoring job."""
    
    # Load reference distributions (from training time)
    reference_profiles = load_reference_profiles("model_v2.3")
    
    # Get current feature values (sample)
    current_values = await feature_store.sample_online_features(
        sample_size=10000,
        features=list(reference_profiles.keys()),
    )
    
    alerts = []
    
    for feature_name, ref_profile in reference_profiles.items():
        current = current_values[feature_name]
        
        # Check freshness
        freshness = await feature_store.get_freshness(feature_name)
        if freshness.hours_since_update > ref_profile.freshness_sla_hours * 2:
            alerts.append(Alert(
                severity="critical",
                feature=feature_name,
                type="stale",
                message=f"Feature {feature_name} not updated in {freshness.hours_since_update}h",
            ))
        
        # Check null rate
        null_rate = np.isnan(current).mean()
        if null_rate > ref_profile.max_null_rate:
            alerts.append(Alert(
                severity="warning",
                feature=feature_name,
                type="high_null_rate",
                message=f"Null rate {null_rate:.1%} exceeds threshold {ref_profile.max_null_rate:.1%}",
            ))
        
        # Check drift
        detector = FeatureDriftDetector(ref_profile.values, ref_profile.dtype)
        drift_result = detector.detect_drift(current[~np.isnan(current)])
        
        if drift_result["drift_detected"]:
            alerts.append(Alert(
                severity="warning",
                feature=feature_name,
                type="drift",
                message=f"Distribution drift detected (PSI={drift_result['methods']['psi']['value']:.3f})",
            ))
    
    # Send alerts
    for alert in alerts:
        await alerting.send(alert)
    
    # Update dashboard metrics
    await metrics.update_feature_health(current_values, alerts)
```

---

## Interview Tip

> When asked about feature monitoring: "I monitor features across four dimensions: (1) Freshness — is the feature being updated on schedule? I track `time_since_last_update` and alert at 2× the expected cadence (hourly feature alerts if >2h stale). Catches silent pipeline failures. (2) Completeness — null rate monitoring. I baseline null rates from training, alert if current null rate exceeds baseline + 2σ. Catches upstream schema changes and join failures. (3) Distribution drift — PSI (Population Stability Index) and KS tests comparing current distribution against training reference. PSI > 0.2 = significant drift requiring investigation. I run hourly on sampled production features. (4) Value range — bounds checking (min/max) and statistical anomaly detection. Catches data type changes, unit changes, and computation bugs. Operationally: I use Evidently for drift computation, Prometheus/Grafana for metrics and dashboards, and PagerDuty for critical alerts (feature completely missing). Key insight: feature monitoring catches problems DAYS before model performance degrades — model accuracy drops slowly as features degrade, but feature anomalies are immediately detectable."

---

## Common Mistakes

1. **No reference distribution** — Running drift detection without a clear baseline. What are you comparing against? Solution: snapshot feature distributions at model training time. Store as reference profiles. Compare production features against these specific snapshots.

2. **Alerting on every fluctuation** — Setting drift thresholds too sensitive (PSI > 0.05). Natural variation causes alerts 10 times per day. Team ignores all alerts (alert fatigue). Solution: calibrate thresholds based on historical variation. Start with PSI > 0.2 for drift, adjust based on false positive rate. Use warning (monitor) vs critical (investigate) tiers.

3. **Not distinguishing expected vs unexpected drift** — Holiday season causes purchase count to spike. Alert fires: "drift detected!" But this is expected seasonal behavior. Solution: maintain seasonal baselines (compare this December to last December, not to July). Annotate known events (sales, holidays, marketing campaigns).

4. **Monitoring aggregate only** — Checking average null rate across all features (2% average = "looks fine"). But one critical feature is 100% null (hidden by average). Solution: monitor EACH feature independently. Dashboard should show per-feature health with red/yellow/green status.

5. **No action plan for drift** — Alert fires: "Feature X has drifted." Then what? Nobody knows what to do. Alert is acknowledged and ignored. Solution: every alert should have a runbook. Drift detected → (1) check pipeline health, (2) check upstream data source, (3) evaluate model performance impact, (4) decide: retrain model, fix pipeline, or update reference.

---

## Key Takeaways

- Monitor four dimensions: freshness (staleness), completeness (nulls), drift (distribution), range (bounds)
- Reference distributions: snapshot at training time, compare production features against training baseline
- PSI (Population Stability Index): < 0.1 fine, 0.1-0.2 monitor, > 0.2 investigate
- Freshness SLA: alert at 2× expected update interval (hourly feature → alert at 2h stale)
- Per-feature monitoring: don't aggregate — one critical feature failing can hide in averages
- Alert severity: critical (pipeline down, 100% null) vs warning (drift, elevated nulls)
- Seasonal awareness: compare against same-period baseline, annotate known events
- Runbooks: every alert must have a documented response procedure
- Tools: Evidently (open-source), WhyLabs/Arize (managed), custom (Great Expectations + Prometheus)
- Feature monitoring catches problems DAYS before model metrics degrade
