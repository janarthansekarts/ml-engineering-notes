# Alerting for ML Systems

## The Problem / Why This Matters

ML alerting is harder than traditional software alerting because ML failures are statistical, not binary. A web server is either up or down — clear alert. But a model can be "slightly wrong" or "degrading gradually" or "drifting in ways that might matter." This leads to two failure modes: (1) Alert fatigue — every minor drift triggers alerts, team ignores them, and misses the real problems buried in noise. (2) Silent failures — alerts are too conservative, model degrades 20% before anyone notices. Effective ML alerting requires understanding that: some drift is benign (seasonal variation, natural evolution), alert thresholds must be tuned empirically (not just "p < 0.05"), different issues have different urgency (data pipeline break → minutes to respond; gradual concept drift → days to respond), and alerts must be actionable (telling someone "PSI is 0.23" without context isn't useful). The goal: alert on issues that actually impact users or business, at the right urgency, with enough context to diagnose and act — while suppressing noise that wastes the team's attention.

---

## The Analogy

Think of ML alerting like a hospital's patient monitoring system:

- **Too sensitive** = Heart monitor beeps every time heartbeat varies by 1 BPM. Nurses disable the alarm. When a real cardiac event happens, nobody's paying attention. (Alert fatigue.)
- **Too conservative** = Heart monitor only alerts for complete cardiac arrest. Patient has dangerous arrhythmia for 6 hours before it becomes critical. (Silent failures.)
- **Well-calibrated** = Different thresholds for different conditions. Minor irregularity → nurse check in 30 minutes. Dangerous pattern → immediate nurse response. Cardiac arrest → code blue, all hands.

The key: severity-based thresholds matched to the actual risk, with clear escalation paths and enough context for the responder to act effectively.

---

## Deep Dive

### Alert Design Principles for ML

```yaml
Alert_Principles:
  actionable:
    rule: "Every alert must have a clear action the responder can take"
    bad: "PSI is 0.23" (what should I do?)
    good: "Feature user_purchase_count_30d drifted significantly (PSI=0.23). Model performance may be impacted. Runbook: check feature pipeline, verify upstream data, assess if retraining needed."
    
  severity_tiered:
    rule: "Different issues need different response times"
    tiers:
      critical: "Respond in <15 minutes (system down, model serving errors)"
      high: "Respond in <1 hour (confirmed performance degradation)"
      medium: "Respond in <4 hours (significant drift, unclear impact)"
      low: "Respond in <1 day (minor drift, informational)"
      
  contextual:
    rule: "Alert includes enough context to diagnose without investigation"
    includes:
      - "What changed (which metric, from what value to what value)"
      - "Since when (when did the change start)"
      - "Impact scope (how many predictions affected)"
      - "Related signals (are other metrics also abnormal?)"
      - "Possible causes (recent deployments? upstream changes?)"
      - "Runbook link (step-by-step investigation guide)"
      
  deduplicated:
    rule: "Same underlying issue shouldn't generate multiple independent alerts"
    example: "One broken feature pipeline causes 5 features to drift → ONE alert about the pipeline, not 5 separate drift alerts"
    implementation: "Alert grouping by root cause, suppression of child alerts"
    
  self_resolving:
    rule: "Alerts should auto-resolve when the condition clears"
    example: "Drift detected at 2PM, data normalized by 4PM → alert auto-resolves, don't keep paging"
    implementation: "Re-check condition periodically, close alert if condition no longer met"
```

### SLOs for ML Models

```yaml
ML_SLOs:
  what: "Service Level Objectives — quantitative targets for model service quality"
  
  system_slos:
    availability:
      target: "99.9% (8.7 hours downtime/year)"
      measurement: "% of prediction requests that return successfully"
      
    latency:
      target: "p95 < 100ms, p99 < 500ms"
      measurement: "End-to-end prediction latency (feature lookup + inference)"
      
    throughput:
      target: ">10,000 predictions/second"
      measurement: "Sustainable prediction rate under peak load"
      
  data_quality_slos:
    feature_freshness:
      target: "Features updated within 2 hours of new data"
      measurement: "max(current_time - latest_feature_computation_time)"
      alert: "Any feature >4 hours stale"
      
    feature_completeness:
      target: ">99% of prediction requests have all features populated"
      measurement: "1 - (null_features / total_features) per request"
      alert: "Completeness drops below 95%"
      
  model_quality_slos:
    prediction_drift:
      target: "PSI < 0.2 for output distribution (hourly window vs reference)"
      measurement: "Population Stability Index of predictions"
      alert: "PSI > 0.2 for 2+ consecutive hours"
      
    performance:
      target: "Accuracy > 0.92 (evaluated on labeled data, 7-day rolling)"
      measurement: "Accuracy on ground-truth-joined predictions"
      alert: "7-day accuracy drops below 0.90"
      
    confidence:
      target: "Mean confidence > 0.7 (hourly window)"
      measurement: "Average prediction confidence"
      alert: "Mean confidence drops below 0.6 for 2+ hours"
      
  business_slos:
    false_positive_rate:
      target: "< 5% (for fraud detection: legitimate transactions wrongly blocked)"
      measurement: "False positives / (True negatives + False positives)"
      alert: "FPR exceeds 7% (measured on labeled data)"
      
    business_metric:
      target: "Model-driven conversion rate within 10% of 30-day average"
      measurement: "Conversion rate for model-influenced interactions"
      alert: "Conversion rate drops >15% from 30-day average"
```

### Alert Implementation

```python
# ML alerting system implementation

"""
Production alerting for ML models: severity-tiered, contextual, deduplicated.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional


class AlertSeverity(Enum):
    CRITICAL = "critical"  # Respond in <15 min
    HIGH = "high"          # Respond in <1 hour
    MEDIUM = "medium"      # Respond in <4 hours
    LOW = "low"            # Respond in <1 day


@dataclass
class MLAlert:
    """An ML model alert with full context."""
    alert_id: str
    model_name: str
    severity: AlertSeverity
    title: str
    description: str
    
    # Context for diagnosis
    metric_name: str
    current_value: float
    threshold: float
    baseline_value: float
    
    # Timing
    detected_at: datetime
    condition_started: Optional[datetime] = None
    
    # Impact
    affected_predictions: int = 0
    affected_percentage: float = 0.0
    
    # Related signals
    related_signals: list = field(default_factory=list)
    possible_causes: list = field(default_factory=list)
    
    # Actionability
    runbook_url: str = ""
    suggested_actions: list = field(default_factory=list)
    
    # State
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MLAlertingSystem:
    """
    Alerting system for ML models.
    
    Features:
    - Severity-based routing (critical → PagerDuty, low → Slack)
    - Alert deduplication (same root cause → one alert)
    - Auto-resolution (condition clears → alert resolves)
    - Contextual enrichment (add related signals, possible causes)
    """
    
    def __init__(self, model_name: str, notification_channels: dict):
        self.model_name = model_name
        self.channels = notification_channels  # {severity: [channel1, channel2]}
        self.active_alerts = {}  # alert_id → MLAlert
        self.suppression_rules = []
        self.alert_history = []
    
    def evaluate_condition(
        self,
        condition_name: str,
        current_value: float,
        threshold: float,
        severity: AlertSeverity,
        context: dict = None,
    ) -> Optional[MLAlert]:
        """
        Evaluate an alert condition. Creates or updates alert if triggered.
        """
        
        triggered = self._is_triggered(condition_name, current_value, threshold)
        
        existing_alert = self._find_active_alert(condition_name)
        
        if triggered and not existing_alert:
            # New alert
            alert = self._create_alert(condition_name, current_value, threshold, severity, context)
            
            # Check suppression rules
            if not self._is_suppressed(alert):
                self.active_alerts[alert.alert_id] = alert
                self._notify(alert)
                return alert
                
        elif triggered and existing_alert:
            # Update existing (escalate if worsening)
            self._update_alert(existing_alert, current_value)
            
        elif not triggered and existing_alert:
            # Condition cleared → auto-resolve
            self._resolve_alert(existing_alert)
        
        return None
    
    def _create_alert(
        self, 
        condition: str, 
        value: float, 
        threshold: float,
        severity: AlertSeverity,
        context: dict,
    ) -> MLAlert:
        """Create alert with full context."""
        
        # Enrich with related signals
        related = self._gather_related_signals(condition)
        causes = self._infer_possible_causes(condition, related)
        actions = self._suggest_actions(condition, severity)
        
        return MLAlert(
            alert_id=f"{self.model_name}_{condition}_{datetime.utcnow().isoformat()}",
            model_name=self.model_name,
            severity=severity,
            title=f"[{severity.value.upper()}] {self.model_name}: {condition}",
            description=self._format_description(condition, value, threshold, context),
            metric_name=condition,
            current_value=value,
            threshold=threshold,
            baseline_value=context.get("baseline", 0) if context else 0,
            detected_at=datetime.utcnow(),
            related_signals=related,
            possible_causes=causes,
            suggested_actions=actions,
            runbook_url=f"https://wiki.company.com/ml-runbooks/{condition}",
        )
    
    def _infer_possible_causes(self, condition: str, related_signals: list) -> list:
        """Infer possible root causes based on condition and related signals."""
        causes = []
        
        if condition == "feature_drift":
            causes.append("Upstream data pipeline change or failure")
            causes.append("New user segment entering the system")
            causes.append("Seasonal pattern shift")
            
        elif condition == "prediction_distribution_shift":
            causes.append("Feature drift propagating to predictions")
            causes.append("Model serving wrong version")
            causes.append("Feature store returning stale values")
            
        elif condition == "performance_degradation":
            causes.append("Concept drift (real-world relationship changed)")
            causes.append("Data quality issue affecting features")
            causes.append("Model staleness (trained too long ago)")
            
        elif condition == "latency_spike":
            causes.append("Increased traffic load")
            causes.append("Feature store latency issue")
            causes.append("Model server resource contention")
        
        # Add related signal info to causes
        for signal in related_signals:
            if signal.get("abnormal"):
                causes.append(f"Related: {signal['name']} is also abnormal ({signal['details']})")
        
        return causes
    
    def _suggest_actions(self, condition: str, severity: AlertSeverity) -> list:
        """Suggest immediate actions based on condition."""
        actions = {
            "feature_drift": [
                "Check feature pipeline health in Dagster/Airflow",
                "Verify upstream data source is healthy",
                "Compare drifted feature values to expectations",
                "Check if model performance is impacted (performance dashboard)",
            ],
            "prediction_distribution_shift": [
                "Verify correct model version is serving",
                "Check feature store for stale values",
                "Compare with data drift signals",
                "If confirmed: consider triggering retraining",
            ],
            "performance_degradation": [
                "Verify ground truth labels are correct (not a labeling issue)",
                "Check which segments are most degraded",
                "Compare with data/concept drift signals",
                "If confirmed: initiate retraining with recent data",
            ],
            "latency_spike": [
                "Check model server resources (CPU/GPU/memory)",
                "Check feature store latency",
                "Check if traffic spike caused overload",
                "Consider scaling up serving infrastructure",
            ],
        }
        return actions.get(condition, ["Investigate using model monitoring dashboard"])
    
    def _notify(self, alert: MLAlert):
        """Route notification based on severity."""
        channels = self.channels.get(alert.severity, [])
        
        for channel in channels:
            if channel == "pagerduty":
                self._page_oncall(alert)
            elif channel == "slack":
                self._send_slack(alert)
            elif channel == "email":
                self._send_email(alert)
    
    def _is_suppressed(self, alert: MLAlert) -> bool:
        """Check if alert should be suppressed (deduplication, maintenance windows)."""
        # Deduplicate: same condition within last hour
        for active in self.active_alerts.values():
            if (active.metric_name == alert.metric_name and 
                not active.resolved and
                (datetime.utcnow() - active.detected_at) < timedelta(hours=1)):
                return True  # Already alerting for this condition
        
        return False
```

### Alert Escalation

```yaml
Escalation_Policies:
  ml_model_alert:
    level_1:
      who: "ML engineer on-call"
      when: "Alert fires"
      timeout: "30 minutes (critical), 2 hours (high), 8 hours (medium)"
      
    level_2:
      who: "Senior ML engineer / ML team lead"
      when: "Level 1 doesn't acknowledge within timeout"
      action: "Page directly + Slack mention"
      
    level_3:
      who: "Engineering manager + product owner"
      when: "Issue not resolved within 2x timeout OR business impact confirmed"
      action: "Incident declared, war room opened"
      
  false_positive_handling:
    threshold: "If >30% of alerts in past week were false positives → adjust thresholds"
    review: "Weekly alert quality review: which alerts were actionable?"
    tuning: "Tighten/loosen thresholds based on signal-to-noise ratio"
```

### Alert Threshold Tuning

```yaml
Threshold_Tuning:
  initial_setting:
    approach: "Start with industry defaults, tune based on experience"
    defaults:
      psi_drift: "Alert at 0.2, warn at 0.1"
      performance_drop: "Alert at 10% relative drop, warn at 5%"
      latency: "Alert at 2x p95 baseline, warn at 1.5x"
      null_rate: "Alert at 5%, warn at 2%"
      
  tuning_process:
    1: "Run for 2 weeks with logging (no alerting)"
    2: "Review all threshold breaches — which would have been actionable?"
    3: "Adjust thresholds: raise for noisy signals, lower for missed issues"
    4: "Enable alerting with tuned thresholds"
    5: "Weekly review: false positive rate should be <20%"
    
  adaptive_thresholds:
    what: "Thresholds that adjust based on recent behavior"
    implementation: "Alert if current value > (30-day rolling mean + 3 × rolling std)"
    benefit: "Automatically adapts to seasonal patterns and gradual changes"
    risk: "May adapt to drift (normalizing bad behavior)"
    mitigation: "Keep absolute thresholds as backstop alongside adaptive ones"
```

---

## How It Works in Practice

### Alert Routing Configuration

```yaml
Routing:
  critical:
    conditions:
      - "Model serving 5xx error rate > 5%"
      - "Prediction latency p99 > 5 seconds"
      - "Model returning same prediction for >100 consecutive requests"
      - "Feature store completely unavailable"
    channels: ["pagerduty", "slack_urgent"]
    response_time: "15 minutes"
    
  high:
    conditions:
      - "Performance (accuracy) dropped >10% from baseline"
      - "Multiple high-importance features drifting (PSI > 0.3)"
      - "Prediction distribution shifted >20% from baseline"
    channels: ["slack_ml_alerts", "email_oncall"]
    response_time: "1 hour"
    
  medium:
    conditions:
      - "Single feature drift detected (PSI > 0.2)"
      - "Confidence distribution shifted"
      - "Prediction volume anomaly (>50% from expected)"
    channels: ["slack_ml_monitoring"]
    response_time: "4 hours"
    
  low:
    conditions:
      - "Minor drift (PSI 0.1-0.2)"
      - "Performance within normal range but trending down"
      - "Model >90 days since last training"
    channels: ["weekly_digest_email"]
    response_time: "1 business day"
```

---

## Interview Tip

> When asked about ML alerting: "ML alerting is fundamentally different from software alerting because ML failures are statistical, not binary. A server is up or down — clear. A model can be 'slightly degraded' or 'drifting in a way that might matter.' My approach: (1) Severity tiers — critical (system failure, 15-min response), high (confirmed performance drop, 1-hour response), medium (drift detected/unclear impact, 4-hour response), low (informational, next business day). (2) SLOs (Service Level Objectives) for ML — not just availability and latency, but model-specific: prediction drift PSI < 0.2, feature freshness < 2 hours, accuracy > 92% on rolling labeled data. (3) Contextual alerts — every alert includes: what changed, since when, impact scope, related signals, possible causes, and runbook link. A bare 'PSI is 0.23' is useless without context. (4) Deduplication — one broken pipeline causing 5 features to drift = one alert about the pipeline, not 5 independent drift alerts. (5) Alert quality management — weekly review of alert signal-to-noise ratio. Target: <20% false positive rate. If too many false positives, tighten thresholds. If missed real issues, loosen them. The biggest anti-pattern I've seen: alerting on every statistical test p < 0.05, which guarantees alert fatigue and ensures the team ignores everything."

---

## Common Mistakes

1. **Alerting on every drift signal** — KS test p < 0.05 on ANY feature → alert. With 50 features checked hourly, you get multiple false alerts daily (multiple testing problem). Team ignores all alerts. Solution: adjust for multiple comparisons (Bonferroni), use PSI with higher thresholds (0.2 not 0.05), and require drift + impact confirmation.

2. **No runbook for alerts** — Alert fires at 3 AM: "Model performance degraded." On-call engineer has no idea what to check, who to contact, or what the expected behavior is. Solution: every alert condition gets a runbook: investigation steps, common causes, remediation actions, escalation contacts. Link runbook in the alert.

3. **Same urgency for everything** — All ML alerts go to PagerDuty. Minor drift at midnight wakes someone up. By week 3, team disables PagerDuty for ML alerts. Then a critical model failure gets no response. Solution: strict severity tiers. Only system failures and confirmed severe degradation page on-call. Drift and informational signals go to Slack/email.

4. **Static thresholds that never update** — Alert threshold set at deployment. 6 months later, the model and data have naturally evolved — the old threshold triggers constantly because the baseline moved. Solution: adaptive thresholds (rolling mean + N*std), periodic threshold reviews, and separate absolute vs. relative thresholds.

5. **No alert resolution** — Alert fires and stays active forever. After the issue is fixed, alert still shows as "active." Team loses track of which alerts represent real current problems. Solution: auto-resolution (re-check condition periodically, close alert if condition clears) + manual resolution (on-call marks fixed with root cause notes).

---

## Key Takeaways

- ML alerting differs from software: statistical failures, not binary up/down
- Severity tiers: critical (15 min), high (1 hour), medium (4 hours), low (next day)
- SLOs for ML: availability + latency + feature freshness + prediction drift + performance
- Contextual alerts: what changed, since when, impact scope, causes, runbook link
- Alert deduplication: one root cause = one alert (not N alerts per drifted feature)
- Threshold tuning: start with defaults, run 2 weeks in logging mode, tune based on signal-to-noise
- Adaptive thresholds: rolling mean + N×std (auto-adapts to seasonal patterns)
- Alert quality: target <20% false positive rate, weekly review of alert actionability
- Escalation: on-call → senior engineer → manager (with defined timeouts)
- Auto-resolution: close alerts when condition clears (don't accumulate stale alerts)
