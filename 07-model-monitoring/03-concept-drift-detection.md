# Concept Drift Detection

## The Problem / Why This Matters

Concept drift is the most dangerous form of model degradation: the underlying relationship between inputs and outputs changes over time. Unlike data drift (where inputs change but the mapping stays the same), concept drift means the "correct answer" for the same input has changed. A customer with features [age=30, income=80K, account_age=2y] might have been low-risk in 2023 but high-risk in 2025 because economic conditions changed the relationship between these features and default probability. The model's learned mapping is now wrong — not because inputs look different, but because the world changed. Concept drift is harder to detect than data drift because: (1) you need ground truth labels to confirm it (which are often delayed), (2) input features may look perfectly normal (no data drift signal), and (3) it can be gradual (slowly accumulating error that's hard to notice day-to-day). Production systems must detect concept drift through performance monitoring, proxy metrics, and statistical change-point detection — then trigger retraining or model adaptation.

---

## The Analogy

Think of concept drift like a teacher's grading standards changing:

- **No drift** = Teacher consistently grades A for scores > 90, B for 80-89, etc. Same input (score), same output (grade). Your model learned this mapping perfectly.
- **Data drift** = Students start scoring differently (more high scores or low scores), but the grading standards stay the same. Input distribution changed, but the mapping is unchanged.
- **Concept drift** = The teacher changes grading standards mid-semester. Now A requires > 95 instead of > 90. Same input (score of 92) → different correct output (used to be A, now B). Your model still predicts A for 92, but the right answer changed.

The insidious part: you can't detect concept drift just by looking at inputs (students' scores look normal). You only discover it when you compare your predictions (model says A) against actual outcomes (teacher gave B). That comparison requires ground truth — which often arrives with delay.

---

## Deep Dive

### Types of Concept Drift

```yaml
Concept_Drift_Types:
  sudden:
    what: "Abrupt change in the input-output relationship"
    examples:
      - "New regulation changes what counts as fraud (overnight)"
      - "Platform redesign changes user engagement patterns"
      - "COVID lockdown instantly changed purchase behavior"
      - "Competitor launches, changing customer churn dynamics"
    detection: "Performance drops sharply at a specific point in time"
    response: "Immediate retraining on recent data"
    
  gradual:
    what: "Slow, continuous change over weeks/months"
    examples:
      - "User preferences evolving over time"
      - "Gradual market shift (new demographics)"
      - "Technology adoption changing behavior patterns"
      - "Seasonal transitions (winter → spring shopping)"
    detection: "Performance slowly degrades; hard to pinpoint start"
    response: "Regular retraining schedule, shorter training windows"
    
  incremental:
    what: "Old concept gradually replaced by new concept"
    examples:
      - "Customer base slowly shifting from Gen X to Millennial"
      - "Product catalog evolving (old products retired, new ones added)"
    detection: "Performance on recent data worse than on older data"
    response: "Weight recent data more in training, sliding window training"
    
  recurring:
    what: "Concept alternates between known states"
    examples:
      - "Seasonal patterns (holiday vs normal shopping)"
      - "Work week vs weekend behavior"
      - "Marketing campaign periods vs organic periods"
    detection: "Performance degrades cyclically at predictable times"
    response: "Ensemble of models for different states, or calendar-aware features"
    
  reoccurring:
    what: "Concept returns to a previous known state after change"
    examples:
      - "Post-pandemic behavior returning to pre-pandemic patterns"
      - "After product launch hype dies down, normal patterns resume"
    detection: "Performance recovers without retraining (drift was temporary)"
    response: "Don't retrain hastily — wait to see if drift is permanent"
```

### Detection Methods

```python
# Concept drift detection methods

"""
Detect when the relationship between features and target changes.
Requires some form of ground truth (even delayed or partial).
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class DriftAlert:
    """Alert from drift detector."""
    detected: bool
    method: str
    timestamp: str
    severity: str  # "warning", "critical"
    details: dict


class PerformanceBasedDetector:
    """
    Detect concept drift by monitoring model performance over time.
    
    Approach: Track rolling performance metric. If it drops significantly
    below baseline, concept drift is likely.
    """
    
    def __init__(
        self,
        baseline_metric: float,  # Performance at deployment
        metric_name: str = "accuracy",
        window_size: int = 1000,  # Predictions to accumulate before checking
        warning_threshold: float = 0.05,  # 5% relative drop = warning
        critical_threshold: float = 0.10,  # 10% relative drop = critical
    ):
        self.baseline = baseline_metric
        self.metric_name = metric_name
        self.window_size = window_size
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.predictions = deque(maxlen=window_size)
        self.labels = deque(maxlen=window_size)
    
    def add_prediction(self, prediction: float, ground_truth: Optional[float] = None):
        """Add a prediction and (optionally) its ground truth label."""
        self.predictions.append(prediction)
        if ground_truth is not None:
            self.labels.append((prediction, ground_truth))
    
    def check(self) -> DriftAlert:
        """Check if performance has degraded significantly."""
        if len(self.labels) < self.window_size * 0.5:
            return DriftAlert(detected=False, method="performance", 
                            timestamp="", severity="none", details={"reason": "insufficient labels"})
        
        # Compute current performance
        preds = [p for p, _ in self.labels]
        truths = [t for _, t in self.labels]
        current_metric = self._compute_metric(preds, truths)
        
        # Compare to baseline
        relative_drop = (self.baseline - current_metric) / self.baseline
        
        if relative_drop >= self.critical_threshold:
            return DriftAlert(
                detected=True,
                method="performance_monitoring",
                timestamp=str(np.datetime64("now")),
                severity="critical",
                details={
                    "baseline": self.baseline,
                    "current": current_metric,
                    "relative_drop": relative_drop,
                    "window_size": len(self.labels),
                },
            )
        elif relative_drop >= self.warning_threshold:
            return DriftAlert(
                detected=True,
                method="performance_monitoring", 
                timestamp=str(np.datetime64("now")),
                severity="warning",
                details={
                    "baseline": self.baseline,
                    "current": current_metric,
                    "relative_drop": relative_drop,
                },
            )
        
        return DriftAlert(detected=False, method="performance", 
                        timestamp="", severity="none", details={})
    
    def _compute_metric(self, predictions, truths):
        """Compute performance metric."""
        if self.metric_name == "accuracy":
            return np.mean(np.array(predictions).round() == np.array(truths))
        elif self.metric_name == "rmse":
            return -np.sqrt(np.mean((np.array(predictions) - np.array(truths))**2))
        # Add more metrics as needed


class DDMDetector:
    """
    DDM (Drift Detection Method) — statistical process control approach.
    
    Monitors error rate as a Bernoulli process.
    Alerts when error rate exceeds control limits.
    
    Based on: Gama et al., 2004 "Learning with Drift Detection"
    """
    
    def __init__(self, warning_level: float = 2.0, drift_level: float = 3.0):
        self.warning_level = warning_level  # Standard deviations for warning
        self.drift_level = drift_level  # Standard deviations for drift
        self.n = 0
        self.p = 0  # Running error rate
        self.s = 0  # Running standard deviation
        self.p_min = float('inf')  # Minimum error rate seen
        self.s_min = float('inf')  # Std at minimum error rate
        self.in_warning = False
    
    def add_element(self, is_error: bool) -> DriftAlert:
        """
        Add a single prediction result (correct=0, error=1).
        Returns drift alert if detected.
        """
        self.n += 1
        
        # Update running statistics
        self.p += (int(is_error) - self.p) / self.n
        self.s = np.sqrt(self.p * (1 - self.p) / self.n)
        
        # Update minimums
        if self.p + self.s < self.p_min + self.s_min:
            self.p_min = self.p
            self.s_min = self.s
        
        # Check control limits
        if self.p + self.s > self.p_min + self.drift_level * self.s_min:
            # DRIFT detected — reset statistics
            alert = DriftAlert(
                detected=True,
                method="DDM",
                timestamp=str(np.datetime64("now")),
                severity="critical",
                details={
                    "current_error_rate": self.p,
                    "min_error_rate": self.p_min,
                    "n_samples": self.n,
                },
            )
            self._reset()
            return alert
        
        elif self.p + self.s > self.p_min + self.warning_level * self.s_min:
            self.in_warning = True
            return DriftAlert(
                detected=True,
                method="DDM",
                timestamp=str(np.datetime64("now")),
                severity="warning",
                details={
                    "current_error_rate": self.p,
                    "min_error_rate": self.p_min,
                },
            )
        
        return DriftAlert(detected=False, method="DDM", 
                        timestamp="", severity="none", details={})
    
    def _reset(self):
        """Reset after drift detection."""
        self.n = 0
        self.p = 0
        self.s = 0
        self.p_min = float('inf')
        self.s_min = float('inf')
        self.in_warning = False


class ADWINDetector:
    """
    ADWIN (ADaptive WINdowing) — automatically adjusts window size.
    
    Maintains a variable-length window of recent observations.
    Drops old data when statistical difference is detected between
    old and recent subwindows.
    
    Based on: Bifet & Gavalda, 2007
    """
    
    def __init__(self, delta: float = 0.002):
        """
        Args:
            delta: Confidence parameter (lower = fewer false alerts, slower detection)
        """
        self.delta = delta
        self.window = []
        self.total = 0
        self.variance = 0
        self.width = 0
    
    def add_element(self, value: float) -> DriftAlert:
        """Add new observation, check if old data should be dropped."""
        self.window.append(value)
        self.width += 1
        self.total += value
        
        if self.width < 10:
            return DriftAlert(detected=False, method="ADWIN",
                            timestamp="", severity="none", details={})
        
        # Check if we should drop old elements (concept changed)
        drift_detected = self._check_for_cut()
        
        if drift_detected:
            return DriftAlert(
                detected=True,
                method="ADWIN",
                timestamp=str(np.datetime64("now")),
                severity="critical",
                details={
                    "window_size": self.width,
                    "recent_mean": np.mean(self.window[-self.width//2:]),
                    "old_mean": np.mean(self.window[:self.width//2]),
                },
            )
        
        return DriftAlert(detected=False, method="ADWIN",
                        timestamp="", severity="none", details={})
    
    def _check_for_cut(self) -> bool:
        """Check if cutting the window improves the model."""
        found_cut = False
        
        # Try different cut points
        for i in range(1, self.width):
            # Compare mean of left (old) vs right (recent) subwindows
            left = self.window[:i]
            right = self.window[i:]
            
            if len(left) < 5 or len(right) < 5:
                continue
            
            mean_diff = abs(np.mean(left) - np.mean(right))
            
            # Hoeffding bound for significance
            n1, n2 = len(left), len(right)
            epsilon = np.sqrt(
                (1/(2*n1) + 1/(2*n2)) * np.log(4/self.delta)
            )
            
            if mean_diff > epsilon:
                # Significant difference found — drop old data
                self.window = right
                self.width = len(right)
                self.total = sum(right)
                found_cut = True
                break
        
        return found_cut
```

### Adaptation Strategies

```yaml
Adaptation_Strategies:
  periodic_retraining:
    what: "Retrain on schedule regardless of drift detection"
    schedule: "Weekly/monthly/quarterly depending on domain"
    data: "Most recent N months of data"
    pros: "Simple, predictable, catches gradual drift"
    cons: "May retrain unnecessarily (waste), may miss sudden drift between cycles"
    
  triggered_retraining:
    what: "Retrain only when drift is detected"
    trigger: "Performance drop > threshold OR drift detector fires"
    data: "Recent data (post-drift) only, or mixed old + new"
    pros: "Efficient (only retrain when needed), responsive to change"
    cons: "Requires reliable detection (false negatives miss drift)"
    
  online_learning:
    what: "Continuously update model with each new example"
    implementation: "Incremental learning algorithms (SGD, online forests)"
    pros: "Immediately adapts to any concept change"
    cons: "Not all models support it, risk of catastrophic forgetting, harder to validate"
    
  ensemble_adaptation:
    what: "Maintain ensemble of models trained on different time windows"
    implementation: "Dynamic Weighted Majority — weight models by recent performance"
    pros: "Handles recurring drift (old model becomes useful again)"
    cons: "Higher serving cost (multiple models), complexity"
    
  windowed_training:
    what: "Always train on most recent N months only (sliding window)"
    implementation: "Discard old training data, only use recent"
    pros: "Naturally adapts to gradual drift, forgets outdated patterns"
    cons: "Loses long-term patterns, needs enough recent data"
    
  importance_weighting:
    what: "Weight training examples by recency (recent examples weighted higher)"
    implementation: "Exponential decay weights: w = exp(-λ × age_in_days)"
    pros: "Smooth adaptation, uses all data but emphasizes recent"
    cons: "Need to tune decay parameter λ"
```

---

## How It Works in Practice

### Concept Drift Response Workflow

```yaml
Workflow:
  detection:
    layer_1_proxy: "Performance on proxy metrics drops (e.g., engagement drops before labels arrive)"
    layer_2_delayed: "Ground truth arrives (days/weeks later) → confirm actual performance drop"
    layer_3_change_point: "DDM/ADWIN detect change in error stream"
    
  diagnosis:
    questions:
      - "Is this sudden (specific point in time) or gradual?"
      - "Is it affecting all segments or specific subpopulations?"
      - "Is it correlated with a known event (product launch, season change)?"
      - "Is there also data drift, or only concept drift?"
    actions:
      - "Plot performance over time (identify change point)"
      - "Segment analysis (which user groups degraded most?)"
      - "Check for external events (look at calendar, news, product changes)"
      
  response:
    sudden_drift:
      immediate: "If impact is large → rollback to rule-based system or previous model"
      short_term: "Retrain on post-drift data (may need accelerated labeling)"
      long_term: "Add drift-causing features, build more robust model"
      
    gradual_drift:
      immediate: "Increase monitoring frequency"
      short_term: "Retrain with sliding window (drop old data)"
      long_term: "Implement regular retraining schedule (weekly/monthly)"
      
    recurring_drift:
      immediate: "Switch to seasonal model (if you have one)"
      long_term: "Build seasonal model ensemble or add temporal features"
```

---

## Interview Tip

> When asked about concept drift: "Concept drift is when the relationship between inputs and outputs changes — the 'correct answer' for the same input is different now than when the model was trained. I detect it primarily through performance monitoring with ground truth labels. For real-time detection without labels, I use the DDM (Drift Detection Method) algorithm — it monitors the error stream as a Bernoulli process and fires when error rate exceeds statistical control limits. For windowed detection, I use ADWIN (Adaptive Windowing) which automatically adjusts its window size to isolate the change point. The challenge: ground truth is often delayed (fraud labels arrive 30 days later, loan defaults after 6 months). In this gap, I use proxy metrics — engagement rates, user complaints, downstream business metrics — as early warning signals. My adaptation strategy depends on drift type: sudden drift → immediate retrain on new data (or rollback if severe). Gradual drift → sliding window training (train only on recent N months). Recurring drift → ensemble with models trained on different time periods, dynamically weighted by recent performance. The key insight: not all concept drift requires retraining. Sometimes it's seasonal (will resolve itself) or temporary (event-driven). I validate drift is persistent before committing to expensive retraining."

---

## Common Mistakes

1. **Retraining on every detected drift** — DDM fires, team retrains immediately. But drift was seasonal (Black Friday) — model was fine a week later. Wasted retraining on temporary drift. Solution: wait to confirm drift is persistent (monitor for 1-2 more windows), check if it correlates with known seasonal patterns, and only retrain if performance doesn't recover.

2. **No proxy metrics for delayed ground truth** — Ground truth arrives 30 days later. For 30 days, you're blind to concept drift. Performance could have degraded 20% and you won't know for a month. Solution: identify proxy metrics (user engagement, downstream business metrics, prediction confidence) that correlate with true performance and respond faster.

3. **Training on all historical data after drift** — Concept drifted (new behavior patterns), but you retrain on full history including years of pre-drift data. Old patterns dominate → model doesn't learn new concept well. Solution: after confirmed concept drift, train primarily on post-drift data. Use sliding window or importance weighting (recent data weighted higher).

4. **Confusing data drift with concept drift** — Features drifted → assume model performance degraded → retrain. But actually the model handles the new distribution fine (it generalizes). Solution: always verify concept drift through performance metrics, not just input distribution changes. Data drift without performance drop = no concept drift.

5. **Single detection method** — Using only one drift detector. DDM is good for sudden drift but slow for gradual. ADWIN is good for gradual but sensitive. Solution: combine methods — DDM for sudden detection, ADWIN for gradual, performance monitoring as ground truth. Alert when multiple detectors agree.

---

## Key Takeaways

- Concept drift: the relationship between inputs and outputs changes (not just input distribution)
- Types: sudden (regulation change), gradual (user evolution), recurring (seasonal), incremental
- Detection requires ground truth (or proxy metrics for delayed labels)
- DDM (Drift Detection Method): monitors error rate as Bernoulli process, fires on control limit breach
- ADWIN (Adaptive Windowing): automatically adjusts window size to detect change points
- Performance monitoring: simplest and most reliable — track accuracy/AUC over time
- Proxy metrics: engagement, business metrics as early warning before labels arrive
- Adaptation: sudden → retrain on new data; gradual → sliding window; recurring → ensemble
- Not all drift requires retraining: seasonal drift resolves, temporary drift recovers
- Combine detection methods: DDM (sudden) + ADWIN (gradual) + performance (ground truth)
