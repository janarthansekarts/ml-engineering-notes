# Prediction Monitoring

## The Problem / Why This Matters

Prediction monitoring watches what your model outputs in production — the distribution of predictions, confidence scores, class balances, and anomalous outputs. Unlike performance monitoring (which needs ground truth labels), prediction monitoring works immediately with zero delay because you observe predictions the instant they're made. If a model suddenly starts predicting "high risk" for 90% of users (when historically it was 15%), you know something is wrong — without waiting weeks for actual default data. Prediction monitoring is your fastest early warning system: it detects (1) model failures (bug in feature pipeline → model outputs garbage), (2) data issues (null features → model defaults to one prediction), (3) distribution shift effects (model behavior changed because inputs changed), and (4) serving issues (wrong model version deployed, stale cache). In production systems handling millions of predictions daily, even a 5% shift in output distribution can mean thousands of incorrect decisions — lost revenue, bad user experiences, or compliance violations.

---

## The Analogy

Think of prediction monitoring like monitoring a factory's output quality:

You don't need to wait for customer returns (ground truth) to know something's wrong. If the factory normally produces 80% blue widgets and 20% red widgets, and suddenly it's producing 50/50 — something changed in the production line. You don't need customer feedback to investigate. The shift in output ratio IS the signal.

Similarly:
- **Output distribution shift** = Factory producing different mix of products
- **Confidence drop** = Quality inspector marking more items as "uncertain" 
- **Extreme predictions** = Factory producing oversized/undersized items (out of spec)
- **Prediction stuck** = Factory producing only one type (machine jammed)

You catch these immediately, investigate the cause, and fix it — all before any customer is impacted.

---

## Deep Dive

### What to Monitor in Predictions

```yaml
Prediction_Metrics:
  classification:
    class_distribution:
      what: "Proportion of each predicted class over time"
      baseline: "Training data class balance (or recent production balance)"
      alert_on: "Significant shift from baseline (e.g., positive class goes from 10% to 40%)"
      check: "Rolling 1-hour window vs 7-day baseline"
      
    confidence_distribution:
      what: "Distribution of model confidence/probability scores"
      metrics:
        mean_confidence: "Average confidence across predictions"
        low_confidence_ratio: "% of predictions below confidence threshold"
        high_confidence_ratio: "% of predictions above 0.95 (potential overconfidence)"
        bimodal_check: "Is confidence bimodal (certain) or uniform (uncertain)?"
      alert_on: "Mean confidence drops, or low-confidence ratio spikes"
      
    prediction_entropy:
      what: "How uncertain is the model overall? (Shannon entropy of prediction distribution)"
      low_entropy: "Model is very certain (always predicts same class) — possibly stuck"
      high_entropy: "Model is very uncertain — possibly confused by new data patterns"
      alert_on: "Entropy changes significantly from baseline"
      
    edge_case_rate:
      what: "% of predictions near decision boundary (e.g., probability between 0.45-0.55)"
      meaning: "Many borderline predictions → model struggling to discriminate"
      alert_on: "Edge case rate increases (model becoming less decisive)"
      
  regression:
    prediction_distribution:
      what: "Distribution of predicted values (mean, std, quantiles)"
      alert_on: "Mean shift, variance change, new modes appearing"
      
    outlier_rate:
      what: "% of predictions outside historical range"
      alert_on: "Outlier rate exceeds threshold (>1% if historically <0.1%)"
      
    prediction_range:
      what: "Min and max predictions in time window"
      alert_on: "Predictions exceeding physical/business constraints (negative price, age > 200)"
      
  recommendation:
    coverage:
      what: "% of catalog items being recommended (in time window)"
      alert_on: "Coverage dropping (recommending fewer and fewer items)"
      
    popularity_bias:
      what: "% of recommendations going to top-N items"
      alert_on: "Popularity concentration increasing (less diversity)"
      
    novelty:
      what: "How often model recommends items users haven't seen before"
      alert_on: "Novelty dropping (showing same items repeatedly)"
```

### Implementation

```python
# Prediction monitoring implementation

"""
Real-time prediction monitoring: detect anomalies in model outputs
without waiting for ground truth labels.
"""

import numpy as np
from collections import deque
from datetime import datetime, timedelta
from typing import Optional


class PredictionMonitor:
    """
    Monitor model predictions for anomalies.
    
    Detects:
    - Output distribution shift
    - Confidence anomalies
    - Stuck predictions (same output repeated)
    - Extreme/out-of-range predictions
    """
    
    def __init__(
        self,
        model_name: str,
        task_type: str,  # "classification" or "regression"
        reference_predictions: np.ndarray,  # Reference prediction distribution
        window_size: int = 1000,
        alert_callback: callable = None,
    ):
        self.model_name = model_name
        self.task_type = task_type
        self.window_size = window_size
        self.alert_callback = alert_callback
        
        # Reference statistics
        self.ref_mean = float(np.mean(reference_predictions))
        self.ref_std = float(np.std(reference_predictions))
        self.ref_min = float(np.min(reference_predictions))
        self.ref_max = float(np.max(reference_predictions))
        self.ref_quantiles = {
            "p5": float(np.quantile(reference_predictions, 0.05)),
            "p25": float(np.quantile(reference_predictions, 0.25)),
            "p50": float(np.quantile(reference_predictions, 0.50)),
            "p75": float(np.quantile(reference_predictions, 0.75)),
            "p95": float(np.quantile(reference_predictions, 0.95)),
        }
        
        if task_type == "classification":
            # Reference class distribution
            unique, counts = np.unique(reference_predictions.round(), return_counts=True)
            self.ref_class_distribution = dict(zip(unique.tolist(), (counts / counts.sum()).tolist()))
        
        # Rolling window of recent predictions
        self.recent_predictions = deque(maxlen=window_size)
        self.recent_confidences = deque(maxlen=window_size)
        
        # Anomaly counters
        self.consecutive_same = 0
        self.last_prediction = None
    
    def log_prediction(
        self, 
        prediction: float, 
        confidence: Optional[float] = None,
    ) -> list[dict]:
        """
        Log a prediction and run real-time checks.
        Returns list of anomaly alerts (empty if everything normal).
        """
        alerts = []
        
        self.recent_predictions.append(prediction)
        if confidence is not None:
            self.recent_confidences.append(confidence)
        
        # Real-time checks (per prediction)
        
        # Check 1: Out-of-range prediction
        if prediction < self.ref_min * 0.5 or prediction > self.ref_max * 1.5:
            alerts.append({
                "type": "out_of_range",
                "severity": "warning",
                "details": {
                    "prediction": prediction,
                    "expected_range": [self.ref_min, self.ref_max],
                },
            })
        
        # Check 2: Stuck prediction (same output N times in a row)
        if prediction == self.last_prediction:
            self.consecutive_same += 1
            if self.consecutive_same > 50:
                alerts.append({
                    "type": "stuck_prediction",
                    "severity": "critical",
                    "details": {
                        "repeated_value": prediction,
                        "consecutive_count": self.consecutive_same,
                    },
                })
        else:
            self.consecutive_same = 0
        self.last_prediction = prediction
        
        # Check 3: Extremely low confidence
        if confidence is not None and confidence < 0.3:
            # Don't alert on individual low confidence, but track rate
            pass
        
        # Send alerts
        for alert in alerts:
            if self.alert_callback:
                self.alert_callback(self.model_name, alert)
        
        return alerts
    
    def run_window_checks(self) -> dict:
        """
        Run statistical checks on the full prediction window.
        Call periodically (every 5 minutes or every N predictions).
        """
        if len(self.recent_predictions) < self.window_size * 0.5:
            return {"status": "insufficient_data"}
        
        predictions = np.array(list(self.recent_predictions))
        results = {}
        
        # Check 1: Mean shift
        current_mean = float(np.mean(predictions))
        mean_shift = abs(current_mean - self.ref_mean) / max(self.ref_std, 0.001)
        results["mean_shift_zscore"] = mean_shift
        results["mean_shift_alert"] = mean_shift > 3.0  # More than 3 stds from reference mean
        
        # Check 2: Variance change
        current_std = float(np.std(predictions))
        variance_ratio = current_std / max(self.ref_std, 0.001)
        results["variance_ratio"] = variance_ratio
        results["variance_alert"] = variance_ratio > 2.0 or variance_ratio < 0.5
        
        # Check 3: Class distribution (classification)
        if self.task_type == "classification":
            unique, counts = np.unique(predictions.round(), return_counts=True)
            current_dist = dict(zip(unique.tolist(), (counts / counts.sum()).tolist()))
            
            max_class_shift = 0
            for cls, ref_rate in self.ref_class_distribution.items():
                current_rate = current_dist.get(cls, 0)
                shift = abs(current_rate - ref_rate)
                max_class_shift = max(max_class_shift, shift)
            
            results["max_class_distribution_shift"] = max_class_shift
            results["class_distribution_alert"] = max_class_shift > 0.1  # >10% shift
        
        # Check 4: Confidence analysis
        if len(self.recent_confidences) > 100:
            confidences = np.array(list(self.recent_confidences))
            results["mean_confidence"] = float(np.mean(confidences))
            results["low_confidence_ratio"] = float(np.mean(confidences < 0.5))
            results["confidence_alert"] = results["low_confidence_ratio"] > 0.3
        
        # Check 5: Quantile drift
        current_p50 = float(np.median(predictions))
        p50_shift = abs(current_p50 - self.ref_quantiles["p50"]) / max(self.ref_std, 0.001)
        results["median_shift_zscore"] = p50_shift
        
        # Overall status
        alert_count = sum(1 for k, v in results.items() if k.endswith("_alert") and v)
        results["overall_status"] = (
            "critical" if alert_count >= 3 else
            "warning" if alert_count >= 1 else
            "healthy"
        )
        
        return results
```

### Confidence Calibration Monitoring

```yaml
Confidence_Calibration:
  what: "Is the model's stated confidence aligned with actual accuracy?"
  example: "Model says 90% confident → should be correct 90% of the time"
  
  calibration_metrics:
    expected_calibration_error:
      abbreviation: "ECE"
      what: "Average difference between confidence and accuracy across bins"
      formula: "ECE = Σ (|bin_size / N|) × |accuracy_in_bin - avg_confidence_in_bin|"
      ideal: "0 (perfect calibration)"
      concerning: "> 0.05"
      
    reliability_diagram:
      what: "Plot of confidence (x-axis) vs actual accuracy (y-axis)"
      ideal: "Diagonal line (confidence = accuracy)"
      overconfident: "Below diagonal (says 90% but only 70% correct)"
      underconfident: "Above diagonal (says 60% but actually 85% correct)"
      
  monitoring_approach:
    - "Compute ECE on rolling window (7 days) with ground truth"
    - "Compare current calibration to deployment-time calibration"
    - "Alert if calibration degrades (model becoming overconfident or underconfident)"
    - "Especially important after retraining (new model may not be calibrated)"
    
  recalibration:
    when: "Calibration degrades but model discrimination is still good"
    methods:
      - "Platt scaling (fit sigmoid on validation predictions)"
      - "Isotonic regression (non-parametric calibration)"
      - "Temperature scaling (divide logits by learned temperature)"
    benefit: "Fix confidence without retraining the model"
```

### Anomaly Detection on Predictions

```yaml
Prediction_Anomalies:
  individual_anomalies:
    what: "Single predictions that are unusual"
    detection:
      - "Z-score > 4 from reference mean"
      - "Outside [min, max] of training predictions"
      - "Confidence < 0.1 (model extremely uncertain)"
    action: "Log for investigation, don't necessarily alert"
    
  collective_anomalies:
    what: "Groups of predictions that are unusual together"
    detection:
      - "Sudden spike in prediction volume (10x normal)"
      - "Burst of identical predictions (model stuck)"
      - "All predictions in one class for sustained period"
    action: "Alert immediately — usually indicates system issue"
    
  contextual_anomalies:
    what: "Predictions unusual given their context"
    detection:
      - "Prediction for returning customer = 'new user' class"
      - "Predicted price = $0 for premium product"
      - "Risk score = 0.99 for verified customer with 10-year history"
    action: "Flag for review, check if feature pipeline issue"
```

---

## How It Works in Practice

### Production Monitoring Dashboard

```yaml
Dashboard_Panels:
  real_time:
    - "Prediction throughput (predictions/sec)"
    - "Mean confidence (1-min rolling)"
    - "Class distribution (last 1 hour pie chart)"
    - "Anomaly count (last 1 hour)"
    
  trends:
    - "Prediction mean over time (7-day line chart)"
    - "Class balance over time (stacked area chart)"
    - "Confidence distribution over time (box plot per day)"
    - "Outlier rate over time (line chart)"
    
  alerts:
    - "Active alerts with severity and details"
    - "Alert history (last 30 days)"
    - "Time-to-detection for recent issues"
    
  comparison:
    - "Current vs reference distribution (histogram overlay)"
    - "Current vs last week (same weekday comparison)"
    - "Current vs last month (trend comparison)"
```

---

## Interview Tip

> When asked about prediction monitoring: "Prediction monitoring is my fastest early warning system because it requires zero ground truth — I see every prediction the instant it's made. I monitor four things: (1) Output distribution — class ratios for classification, prediction mean/variance for regression. If my fraud model suddenly flags 50% of transactions as fraud (historically 2%), something's very wrong. (2) Confidence distribution — mean confidence, low-confidence ratio, confidence shift. A confidence drop often precedes performance drop. (3) Stuck predictions — is the model outputting the same value repeatedly? This usually means a critical feature is null or stale, so the model defaults to one output. I detect this with a consecutive-same-output counter. (4) Individual anomalies — predictions outside physical constraints (negative prices, impossible ages). These indicate feature pipeline bugs. The key insight: prediction monitoring doesn't tell you WHY something's wrong, but it tells you IMMEDIATELY that something IS wrong. Then I investigate: check feature freshness, look at specific requests, compare to data drift signals. Response time: prediction monitoring catches issues in minutes. Performance monitoring (needs labels) catches them in days/weeks."

---

## Common Mistakes

1. **Only monitoring average predictions** — "Mean prediction is stable, all good." But the distribution became bimodal (half predictions at 0, half at 1) while the mean stayed at 0.5. Solution: monitor distribution shape (quantiles, histogram), not just mean. Check for new modes, bimodality, and distributional changes.

2. **No stuck-prediction detection** — Feature pipeline broke, one feature is null for all users, model always outputs the same prediction. Nobody notices for 3 hours (predictions still look "confident"). Solution: track consecutive identical predictions. More than N in a row (adjusted for your use case) = immediate alert.

3. **Treating confidence at face value** — "Model confidence is 95%, predictions must be right." But model is poorly calibrated — 95% confidence actually means 70% accuracy. Solution: monitor calibration (ECE) over time, not just raw confidence values. If calibration degrades, recalibrate (temperature scaling / Platt scaling).

4. **Not accounting for expected variation** — Alert fires every Monday because weekend prediction patterns differ from weekday. Team ignores all Monday alerts. Then a real issue on Monday gets missed. Solution: time-aware baselines (compare Monday to previous Mondays, not to weekday baseline). Account for known patterns.

5. **Monitoring predictions but not investigating** — "Prediction distribution shifted 15%." → Team says "noted" and does nothing. A week later, revenue drops 8%. Solution: prediction monitoring must connect to investigation workflow. Significant shifts → automated investigation: check feature freshness, recent deployments, upstream data quality. Shift + explanation = OK. Shift + no explanation = escalate.

---

## Key Takeaways

- Prediction monitoring: fastest early warning (no ground truth needed, instant signal)
- Monitor: output distribution, confidence scores, class balance, stuck predictions, outliers
- Stuck prediction detection: consecutive identical outputs = critical alert (usually pipeline failure)
- Confidence monitoring: mean confidence drop often precedes performance drop
- Calibration: verify model's stated confidence matches actual accuracy (ECE metric)
- Distribution checks: monitor quantiles and shape, not just mean (catches bimodal shifts)
- Time-aware baselines: compare to same day-of-week, not just rolling average
- Investigation workflow: prediction shift → check features → check deployments → check upstream
- Prediction anomalies: individual (log), collective (alert), contextual (investigate)
- Combine with data drift: prediction shift + feature drift = strong signal for action
