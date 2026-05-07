# Model Monitoring Fundamentals

## The Problem / Why This Matters

Models degrade in production. Not if — when. A model that achieved 95% accuracy at deployment will silently deteriorate as the world changes: user behavior shifts, new products launch, economic conditions change, upstream data pipelines break, and the patterns the model learned become stale. This is called model rot (gradual performance decline due to changing real-world conditions). Without monitoring, you discover degradation the worst way: users complain, revenue drops, or (in regulated domains) compliance violations are flagged. Model monitoring is the practice of continuously measuring model health in production — tracking input distributions, output distributions, prediction quality, and system health — to detect problems early and trigger corrective action before impact is significant. Traditional software monitoring (is the server up? is latency acceptable?) is necessary but insufficient for ML: a model can return predictions with perfect latency while being completely wrong. You need ML-specific monitoring that understands the statistical properties of what the model is doing.

---

## The Analogy

Think of model monitoring like health monitoring for a person:

- **System monitoring** (traditional) = Checking if the person is alive: pulse, breathing, temperature. Important, but "alive" doesn't mean "healthy."
- **Data drift monitoring** = Checking if the person's environment changed: moved to a new climate, changed diet, new stressors. Environment changes often precede health issues.
- **Prediction monitoring** = Checking vital signs: blood pressure, cholesterol, blood sugar. These are leading indicators — problems show here before symptoms appear.
- **Performance monitoring** = Checking outcomes: can the person run, think clearly, perform daily tasks? This is what ultimately matters, but by the time performance drops, damage may already be done.

A good monitoring system watches all layers: environment (data), vitals (predictions), and outcomes (performance) — catching problems at the earliest possible stage.

---

## Deep Dive

### Types of Model Degradation

```yaml
Degradation_Types:
  data_drift:
    what: "Distribution of input features changes over time"
    also_called: "Covariate shift, feature drift"
    example: "E-commerce model trained on urban users, but rural user base grew 5x"
    detection: "Statistical comparison of current feature distributions vs training distributions"
    urgency: "Medium — drift may or may not affect performance"
    
  concept_drift:
    what: "Relationship between features and target changes"
    also_called: "Target drift, posterior drift"
    example: "During COVID, purchase patterns changed — same features, different outcomes"
    types:
      sudden: "Abrupt change (new regulation, platform change)"
      gradual: "Slow shift over months (changing user preferences)"
      seasonal: "Recurring patterns (holiday shopping, tax season)"
      recurring: "Periodic shifts between states"
    detection: "Performance metric decline on recent data with ground truth"
    urgency: "High — model is making wrong predictions"
    
  upstream_data_issues:
    what: "Data pipeline problems causing bad inputs"
    examples:
      - "Feature pipeline broke → features are stale (yesterday's values)"
      - "Schema change → column shifted, wrong values in wrong features"
      - "Upstream service outage → null values for real-time features"
      - "ETL bug → duplicate records inflating aggregation features"
    detection: "Data quality checks (freshness, completeness, schema validation)"
    urgency: "Critical — immediate impact, usually fixable quickly"
    
  model_staleness:
    what: "Model was trained long ago, world has moved on"
    example: "Model trained on 2023 data serving in 2026 — 3 years of change"
    detection: "Track time since last training + performance trend"
    urgency: "Low-medium — gradual, predictable"
    
  feedback_loops:
    what: "Model's own predictions influence future training data"
    example: "Recommendation model shows item A → users click A → model learns A is popular → shows A even more (self-reinforcing)"
    detection: "Diversity metrics declining, model entropy decreasing"
    urgency: "Medium — insidious, hard to detect"
```

### Monitoring Architecture

```yaml
Architecture:
  layers:
    system_layer:
      what: "Infrastructure and application health"
      metrics:
        - "Prediction latency (p50, p95, p99)"
        - "Throughput (predictions/second)"
        - "Error rate (failed predictions / total)"
        - "CPU/GPU utilization"
        - "Memory usage"
        - "Queue depth (for async predictions)"
      tools: "Prometheus, Grafana, DataDog, CloudWatch"
      alert_on: "Latency SLA breach, error rate spike, resource exhaustion"
      
    data_layer:
      what: "Input data quality and distribution"
      metrics:
        - "Feature distributions (mean, std, quantiles)"
        - "Null rates per feature"
        - "Type violations"
        - "Feature value ranges"
        - "Data freshness"
        - "Volume (requests per time window)"
      tools: "Evidently AI, Whylabs, Great Expectations, custom checks"
      alert_on: "Distribution shift detected, null rate spike, schema violation"
      
    model_layer:
      what: "Model output behavior"
      metrics:
        - "Prediction distribution (mean, std, quantiles)"
        - "Confidence distribution"
        - "Prediction class balance"
        - "Prediction entropy"
        - "Output range violations"
      tools: "Evidently AI, Arize, custom logging"
      alert_on: "Prediction distribution shift, confidence drop, unusual class ratios"
      
    performance_layer:
      what: "Actual model accuracy (requires ground truth)"
      metrics:
        - "Accuracy, precision, recall, F1"
        - "AUC-ROC, AUC-PR"
        - "RMSE, MAE (regression)"
        - "Business metrics (CTR, revenue, conversion)"
      challenge: "Ground truth often delayed (hours/days/weeks)"
      tools: "Custom dashboards, A/B testing platforms"
      alert_on: "Performance below threshold, significant degradation from baseline"
```

### Monitoring Implementation

```python
# Model monitoring system implementation

"""
Production model monitoring: log predictions, compute metrics, detect drift, alert.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from scipy import stats


@dataclass
class PredictionLog:
    """Single prediction log entry."""
    request_id: str
    timestamp: datetime
    model_version: str
    features: dict
    prediction: float
    confidence: float
    latency_ms: float
    ground_truth: float = None  # Filled later when available


class ModelMonitor:
    """
    Production model monitor.
    
    Responsibilities:
    1. Log all predictions
    2. Compute running statistics
    3. Detect data and prediction drift
    4. Alert on anomalies
    """
    
    def __init__(
        self, 
        model_name: str,
        reference_data: dict,  # Training data statistics for comparison
        alert_callback: callable = None,
    ):
        self.model_name = model_name
        self.reference = reference_data
        self.alert_callback = alert_callback
        self.prediction_buffer = []
        
    def log_prediction(self, log: PredictionLog):
        """Log a prediction and check for anomalies."""
        self.prediction_buffer.append(log)
        
        # Real-time checks (per-prediction)
        self._check_output_range(log)
        self._check_latency(log)
        self._check_feature_values(log)
        
    def run_periodic_checks(self, window_hours: int = 1):
        """Run statistical checks on recent prediction window."""
        
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        recent = [p for p in self.prediction_buffer if p.timestamp > cutoff]
        
        if len(recent) < 100:
            return  # Not enough data for statistical tests
        
        results = {}
        
        # Check 1: Prediction distribution drift
        results["prediction_drift"] = self._check_prediction_drift(recent)
        
        # Check 2: Feature drift (per feature)
        results["feature_drift"] = self._check_feature_drift(recent)
        
        # Check 3: Volume anomaly
        results["volume_anomaly"] = self._check_volume(recent, window_hours)
        
        # Check 4: Confidence distribution
        results["confidence_drift"] = self._check_confidence_drift(recent)
        
        # Alert if any check fails
        for check_name, result in results.items():
            if result.get("alert", False):
                self._send_alert(check_name, result)
        
        return results
    
    def _check_prediction_drift(self, recent: list) -> dict:
        """Compare recent prediction distribution vs reference."""
        recent_preds = [p.prediction for p in recent]
        reference_preds = self.reference["prediction_distribution"]
        
        # KS test (Kolmogorov-Smirnov): are distributions different?
        ks_stat, p_value = stats.ks_2samp(recent_preds, reference_preds)
        
        return {
            "test": "KS test on predictions",
            "statistic": float(ks_stat),
            "p_value": float(p_value),
            "alert": p_value < 0.01,  # Significant drift if p < 0.01
            "recent_mean": float(np.mean(recent_preds)),
            "reference_mean": float(np.mean(reference_preds)),
        }
    
    def _check_feature_drift(self, recent: list) -> dict:
        """Check each feature for distribution drift."""
        drift_results = {}
        
        for feature_name in self.reference["feature_distributions"]:
            recent_values = [p.features.get(feature_name) for p in recent if feature_name in p.features]
            
            if not recent_values or None in recent_values:
                continue
                
            reference_values = self.reference["feature_distributions"][feature_name]
            
            # PSI (Population Stability Index) for drift detection
            psi = self._compute_psi(reference_values, recent_values)
            
            drift_results[feature_name] = {
                "psi": float(psi),
                "drifted": psi > 0.2,  # PSI > 0.2 = significant drift
            }
        
        drifted_features = [f for f, r in drift_results.items() if r["drifted"]]
        
        return {
            "total_features": len(drift_results),
            "drifted_features": drifted_features,
            "drifted_count": len(drifted_features),
            "alert": len(drifted_features) > 3,  # Alert if >3 features drifting
        }
    
    def _compute_psi(self, reference: list, current: list, bins: int = 10) -> float:
        """
        Compute PSI (Population Stability Index).
        PSI < 0.1: no significant change
        PSI 0.1-0.2: moderate change (investigate)
        PSI > 0.2: significant change (alert)
        """
        # Create bins from reference distribution
        breakpoints = np.quantile(reference, np.linspace(0, 1, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
        cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current)
        
        # Avoid division by zero
        ref_counts = np.clip(ref_counts, 0.001, None)
        cur_counts = np.clip(cur_counts, 0.001, None)
        
        psi = np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts))
        return psi
    
    def _check_output_range(self, log: PredictionLog):
        """Real-time check: is prediction within expected range?"""
        expected_min = self.reference.get("prediction_min", 0)
        expected_max = self.reference.get("prediction_max", 1)
        
        if log.prediction < expected_min or log.prediction > expected_max:
            self._send_alert("output_range_violation", {
                "prediction": log.prediction,
                "expected_range": [expected_min, expected_max],
            })
    
    def _send_alert(self, check_name: str, details: dict):
        """Send alert via configured callback."""
        if self.alert_callback:
            self.alert_callback(
                model=self.model_name,
                check=check_name,
                details=details,
                timestamp=datetime.utcnow(),
            )
```

### What to Monitor for Different Model Types

```yaml
Model_Type_Monitoring:
  classification:
    inputs: "Feature distributions, null rates, class balance of inputs"
    outputs: "Predicted class ratios, confidence distribution, entropy"
    performance: "Accuracy, precision, recall, F1, AUC by class"
    red_flags:
      - "Class prediction ratio changes suddenly (80/20 → 60/40)"
      - "Average confidence drops below threshold"
      - "One class disappears from predictions entirely"
      
  regression:
    inputs: "Feature distributions, outlier rates, value ranges"
    outputs: "Prediction mean, variance, range, outliers"
    performance: "RMSE, MAE, R², residual distribution"
    red_flags:
      - "Prediction mean shifts significantly"
      - "Prediction variance increases (model becoming uncertain)"
      - "Extreme predictions (orders of magnitude off)"
      
  recommendation:
    inputs: "User-item distributions, item catalog changes"
    outputs: "Recommendation diversity, popularity bias, coverage"
    performance: "CTR (Click-Through Rate), engagement, relevance scores"
    red_flags:
      - "Diversity drops (recommending same items to everyone)"
      - "New items never get recommended (cold start failure)"
      - "CTR declining over time (staleness)"
      
  llm_applications:
    inputs: "Query distribution, topic shifts, token counts"
    outputs: "Response length distribution, confidence, hallucination scores"
    performance: "User satisfaction, thumbs up/down ratio, regeneration rate"
    red_flags:
      - "Response length increasing unexpectedly (verbose mode)"
      - "Hallucination rate spiking"
      - "Regeneration rate increasing (users not satisfied)"
      - "New topic category appearing that model wasn't trained for"
```

---

## How It Works in Practice

### Production Monitoring Workflow

```yaml
Workflow:
  setup:
    1: "Define reference distributions (from training/validation data)"
    2: "Deploy prediction logging (every prediction gets logged)"
    3: "Configure drift detectors (KS test, PSI, chi-squared)"
    4: "Set alert thresholds (PSI > 0.2, accuracy < 0.9, latency > 200ms)"
    5: "Create dashboards (Grafana/custom — feature drift, prediction drift, performance)"
    
  daily_operation:
    continuous: "Log all predictions (features + outputs + metadata)"
    hourly: "Run statistical drift checks on 1-hour windows"
    daily: "Compute performance metrics (where ground truth available)"
    weekly: "Generate monitoring report (trends, anomalies, recommendations)"
    
  incident_response:
    alert_fired: "PSI > 0.2 on 3 features detected"
    triage:
      1: "Check: is this real drift or pipeline bug?"
      2: "Examine which features drifted and by how much"
      3: "Check model performance (has accuracy degraded?)"
      4: "Decide: retrain, fix pipeline, or acceptable drift?"
    resolution:
      - "Pipeline fix → redeploy pipeline, features recover"
      - "Real drift + performance down → trigger retraining"
      - "Real drift + performance stable → monitor, don't retrain"
```

---

## Interview Tip

> When asked about model monitoring: "I monitor models at four layers: (1) System health — latency, throughput, error rates. A model returning predictions in 500ms when SLA is 100ms is broken regardless of accuracy. (2) Data quality — feature freshness, null rates, schema violations. Upstream pipeline failures are the #1 cause of model issues in my experience. (3) Statistical drift — I compare current feature and prediction distributions against reference (training-time) distributions using PSI (Population Stability Index) and KS tests. PSI > 0.2 triggers investigation. But drift alone isn't an alert — I've seen significant drift with zero performance impact. (4) Performance — actual accuracy, precision, recall on ground truth data. This is the gold standard but often delayed (labels arrive days/weeks later). My alert hierarchy: System health → immediate page. Data quality → immediate investigation. Drift → investigate within hours. Performance degradation → trigger retraining pipeline. The key insight: most teams monitor only system health (is the API up?) but miss that the model can serve garbage predictions with perfect latency. ML-specific monitoring catches the silent failures."

---

## Common Mistakes

1. **Only monitoring system health** — "API returns 200, latency under 100ms, model is healthy." But model has been returning the same prediction for all inputs because a feature is stuck at null. Solution: monitor model outputs (prediction distribution, confidence) in addition to system health.

2. **Alerting on all drift** — "PSI > 0.1 → page on-call." Feature distributions shift constantly (seasonal patterns, organic growth). Most drift is benign. Result: alert fatigue, team ignores real problems. Solution: use tiered thresholds (0.1 = log, 0.2 = investigate, 0.3+ with performance drop = alert). Only page on confirmed performance impact.

3. **No reference distribution** — "We detect drift... compared to what?" No baseline saved from training time. Can't tell if current distribution is normal. Solution: save reference distributions at model deployment time. Store training data statistics (mean, std, quantiles, histograms per feature).

4. **Monitoring means but not tails** — "Average prediction is stable." But 5% of predictions are wildly wrong (extreme values) hiding in the average. Solution: monitor quantiles (p5, p25, p50, p75, p95), not just means. Monitor prediction range and outlier rates.

5. **No ground truth collection** — "We can't measure accuracy because we never collect labels." Performance could have degraded 30% and nobody knows. Solution: implement ground truth collection (even partial/sampled). Proxy labels, delayed feedback, human evaluation on samples — anything is better than flying blind.

---

## Key Takeaways

- Models degrade silently: concept drift, data drift, pipeline failures, staleness
- Four monitoring layers: system health, data quality, statistical drift, performance metrics
- PSI (Population Stability Index): <0.1 stable, 0.1-0.2 investigate, >0.2 alert
- KS test (Kolmogorov-Smirnov): statistical test comparing two distributions
- Reference distributions: save training-time statistics at deployment for comparison
- Monitor outputs (prediction distribution, confidence) not just inputs
- Ground truth collection is critical but often delayed — use proxy metrics in the meantime
- Alert hierarchy: system (immediate), data quality (investigate), drift (triage), performance (retrain)
- Most drift is benign — only alert when drift correlates with performance degradation
- Tools: Evidently AI, Whylabs, Arize, custom Prometheus + Grafana dashboards
