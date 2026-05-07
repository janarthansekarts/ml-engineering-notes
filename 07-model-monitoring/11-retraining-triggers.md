# Retraining Triggers

## The Problem / Why This Matters

Models need retraining when they degrade — but when exactly? Too early: you waste compute and engineering time retraining a model that's still fine. Too late: degraded model serves bad predictions for weeks before anyone acts. Retraining triggers are the decision rules that determine when a model should be retrained: scheduled (every N days), performance-based (accuracy drops below threshold), drift-based (data distribution changes significantly), or event-based (new data source available, business requirements change). The engineering challenge: designing triggers that are sensitive enough to catch real degradation but not so sensitive they fire constantly (trigger fatigue ≈ alert fatigue). In 2026, most production ML teams use a combination: a scheduled baseline (retrain monthly regardless) plus performance-triggered retraining (retrain immediately if accuracy drops more than 5%) plus drift-triggered investigation (check if retraining would help). Automated retraining pipelines — continuous training — make this feasible at scale, but they introduce their own risks: automatically deploying a model trained on bad data, or retraining during a transient data anomaly.

---

## The Analogy

Think of retraining triggers like car maintenance schedules:

- **Scheduled retraining** = Oil change every 5,000 miles. Simple rule, prevents most problems, but sometimes wasteful (oil was still fine) and sometimes too late (oil degraded faster than expected due to conditions).
- **Performance-triggered** = "Check engine" light. Only triggers when something is actually wrong. Reactive — by the time the light is on, some damage may have occurred.
- **Drift-triggered** = Tire pressure warning. Detects conditions that MIGHT cause problems. Proactive — warns before performance degrades, but sometimes the tire is fine.
- **Continuous training** = Hybrid/EV that continuously self-tunes. Always adapting. No maintenance visits needed — but you need to trust the self-tuning system isn't going wrong.

Best approach: scheduled maintenance (prevents most issues) + warning lights (catches what schedules miss) + continuous monitoring (early detection of emerging problems).

---

## Deep Dive

### Trigger Types

```yaml
Trigger_Types:
  scheduled:
    what: "Retrain on fixed schedule regardless of performance"
    schedules:
      daily: "High-frequency domains (ad ranking, fraud, recommendation)"
      weekly: "Medium-frequency domains (churn, personalization)"
      monthly: "Stable domains (credit scoring, insurance pricing)"
      quarterly: "Very stable domains (healthcare risk models with regulatory review)"
    pros: "Simple, predictable, prevents staleness, no monitoring dependency"
    cons: "May retrain unnecessarily (waste) or too late (between schedules)"
    implementation: "Cron job or orchestrator schedule (Airflow/Dagster)"
    
  performance_triggered:
    what: "Retrain when model performance drops below threshold"
    metrics:
      - "Accuracy/F1 below minimum acceptable level"
      - "AUC-ROC drops >5% from deployment baseline"
      - "Business metric degrades (CTR/conversion/revenue)"
    pros: "Only retrains when actually needed, evidence-based"
    cons: "Requires ground truth labels (often delayed), reactive not proactive"
    implementation: "Monitor performance metrics → threshold check → trigger pipeline"
    
  drift_triggered:
    what: "Retrain when input data distribution shifts significantly"
    signals:
      - "PSI > 0.2 on important features for 2+ consecutive windows"
      - "Multiple features drifting simultaneously"
      - "Prediction distribution shift"
    pros: "Proactive (catches issues before performance degrades)"
    cons: "Not all drift causes performance degradation (may retrain unnecessarily)"
    implementation: "Drift detection → validate with performance check → conditional retrain"
    
  event_triggered:
    what: "Retrain based on external events"
    events:
      - "New data source becomes available"
      - "Business requirements change (new features needed)"
      - "Upstream schema change (features recomputed)"
      - "Regulatory requirement (annual model review)"
      - "New product launch (model must adapt)"
      - "Major external event (pandemic, market crash)"
    pros: "Responds to known changes immediately"
    cons: "Requires human judgment about which events matter"
    
  data_volume_triggered:
    what: "Retrain when sufficient new labeled data accumulates"
    logic: "If new_labeled_data > N since last training → retrain"
    example: "Retrain churn model when 10,000 new churn/non-churn labels available"
    pros: "Ensures model benefits from latest labeled data"
    cons: "Doesn't account for quality — new data might not add value"
```

### Trigger Decision Framework

```python
# Retraining trigger decision system

"""
Combines multiple signals to make retrain/no-retrain decisions.
Avoids both under-retraining (model degrades) and over-retraining (waste).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional


class RetrainDecision(Enum):
    RETRAIN_NOW = "retrain_now"          # Immediate retraining needed
    RETRAIN_SCHEDULED = "retrain_scheduled"  # Wait for next scheduled slot
    INVESTIGATE = "investigate"           # Something's off, human should check
    NO_ACTION = "no_action"              # Model is fine


@dataclass
class TriggerSignals:
    """All signals that inform the retrain decision."""
    # Performance signals
    current_accuracy: Optional[float] = None
    baseline_accuracy: Optional[float] = None
    accuracy_trend: Optional[str] = None  # "stable", "declining", "improving"
    
    # Drift signals
    max_feature_psi: float = 0.0
    num_drifted_features: int = 0
    prediction_distribution_psi: float = 0.0
    
    # Timing signals
    days_since_last_training: int = 0
    scheduled_retrain_due: bool = False
    
    # Data signals
    new_labeled_samples: int = 0
    data_quality_issues: bool = False
    
    # External signals
    known_event_occurred: bool = False
    event_description: str = ""


class RetrainingTriggerSystem:
    """
    Multi-signal retraining trigger system.
    
    Decision logic:
    1. If performance critically degraded → RETRAIN NOW
    2. If significant drift + performance decline → RETRAIN NOW
    3. If drift only (no performance impact) → INVESTIGATE
    4. If scheduled and enough new data → RETRAIN SCHEDULED
    5. If everything stable → NO ACTION
    """
    
    def __init__(
        self,
        min_accuracy: float = 0.88,          # Below this: immediate retrain
        warning_accuracy_drop: float = 0.05,  # 5% relative drop: investigate
        critical_accuracy_drop: float = 0.10, # 10% relative drop: retrain now
        max_days_between_training: int = 30,  # Scheduled retrain maximum
        min_new_samples_for_retrain: int = 5000,  # Minimum new data
        drift_psi_threshold: float = 0.2,
    ):
        self.min_accuracy = min_accuracy
        self.warning_drop = warning_accuracy_drop
        self.critical_drop = critical_accuracy_drop
        self.max_days = max_days_between_training
        self.min_samples = min_new_samples_for_retrain
        self.drift_threshold = drift_psi_threshold
        
        self.retrain_history = []  # Track past retrains
    
    def evaluate(self, signals: TriggerSignals) -> dict:
        """
        Evaluate all signals and make a retrain decision.
        Returns decision + reasoning.
        """
        reasons = []
        
        # Rule 1: Critical performance degradation
        if signals.current_accuracy is not None:
            if signals.current_accuracy < self.min_accuracy:
                return {
                    "decision": RetrainDecision.RETRAIN_NOW,
                    "reason": f"Accuracy ({signals.current_accuracy:.3f}) below minimum ({self.min_accuracy})",
                    "urgency": "critical",
                    "confidence": "high",
                }
            
            if signals.baseline_accuracy is not None:
                relative_drop = (signals.baseline_accuracy - signals.current_accuracy) / signals.baseline_accuracy
                
                if relative_drop >= self.critical_drop:
                    return {
                        "decision": RetrainDecision.RETRAIN_NOW,
                        "reason": f"Accuracy dropped {relative_drop:.1%} from baseline ({signals.baseline_accuracy:.3f} → {signals.current_accuracy:.3f})",
                        "urgency": "high",
                        "confidence": "high",
                    }
                
                if relative_drop >= self.warning_drop:
                    reasons.append(f"Accuracy declining ({relative_drop:.1%} from baseline)")
        
        # Rule 2: Significant drift + any performance decline
        has_significant_drift = (
            signals.max_feature_psi > self.drift_threshold or
            signals.num_drifted_features >= 3 or
            signals.prediction_distribution_psi > self.drift_threshold
        )
        
        if has_significant_drift:
            drift_details = f"PSI={signals.max_feature_psi:.3f}, {signals.num_drifted_features} features drifted"
            
            if signals.accuracy_trend == "declining":
                return {
                    "decision": RetrainDecision.RETRAIN_NOW,
                    "reason": f"Significant drift ({drift_details}) + declining performance",
                    "urgency": "high",
                    "confidence": "high",
                }
            else:
                reasons.append(f"Drift detected ({drift_details}) but performance stable — investigate")
        
        # Rule 3: Data quality issues
        if signals.data_quality_issues:
            return {
                "decision": RetrainDecision.INVESTIGATE,
                "reason": "Data quality issues detected — fix data before retraining",
                "urgency": "medium",
                "confidence": "medium",
            }
        
        # Rule 4: Known external event
        if signals.known_event_occurred:
            return {
                "decision": RetrainDecision.RETRAIN_NOW,
                "reason": f"External event: {signals.event_description}",
                "urgency": "medium",
                "confidence": "medium",
            }
        
        # Rule 5: Scheduled retrain (time-based)
        if signals.days_since_last_training >= self.max_days:
            if signals.new_labeled_samples >= self.min_samples:
                return {
                    "decision": RetrainDecision.RETRAIN_SCHEDULED,
                    "reason": f"Scheduled: {signals.days_since_last_training} days since last training, {signals.new_labeled_samples} new samples available",
                    "urgency": "low",
                    "confidence": "high",
                }
            else:
                reasons.append(f"Due for scheduled retrain but insufficient new data ({signals.new_labeled_samples} < {self.min_samples})")
        
        # Rule 6: Investigation needed (multiple minor signals)
        if len(reasons) >= 2:
            return {
                "decision": RetrainDecision.INVESTIGATE,
                "reason": "; ".join(reasons),
                "urgency": "medium",
                "confidence": "medium",
            }
        
        # Default: no action
        return {
            "decision": RetrainDecision.NO_ACTION,
            "reason": "All signals within normal range",
            "urgency": "none",
            "confidence": "high",
            "notes": reasons if reasons else ["Model healthy"],
        }
```

### Continuous Training Pipeline

```yaml
Continuous_Training:
  what: "Automated pipeline that retrains, validates, and deploys without human intervention"
  
  pipeline_steps:
    1_trigger:
      input: "Trigger signal (scheduled, performance, drift)"
      output: "Retrain decision"
      
    2_data_preparation:
      actions:
        - "Collect training data (recent N months from feature store)"
        - "Validate data quality (Great Expectations checks)"
        - "Split into train/validation/test"
      safeguard: "If data quality fails → abort and alert (don't train on bad data)"
      
    3_training:
      actions:
        - "Train model with same hyperparameters as current production model"
        - "Optionally: hyperparameter tuning on validation set"
        - "Log experiment to MLflow/W&B"
      safeguard: "Training must complete within time budget (detect hung jobs)"
      
    4_validation:
      actions:
        - "Evaluate on held-out test set"
        - "Compare against current production model performance"
        - "Run fairness checks (must pass fairness criteria)"
        - "Check model size and inference latency"
      safeguards:
        performance_gate: "New model must be >= current model on test set"
        regression_gate: "New model must not degrade on any segment by >2%"
        fairness_gate: "Must pass 4/5ths rule on all protected groups"
        latency_gate: "Inference must be within latency SLA"
      on_failure: "Reject new model, keep current, alert team"
      
    5_deployment:
      actions:
        - "Register model in model registry"
        - "Deploy to shadow (100% traffic, no user impact)"
        - "Compare shadow predictions vs current model"
        - "If shadow performance good: promote to canary (5% traffic)"
        - "If canary good: full rollout (100% traffic)"
      rollback: "Automatic rollback if canary metrics degrade"
      
    6_post_deployment:
      actions:
        - "Update monitoring reference distributions"
        - "Log retrain event (what triggered it, new model version, performance delta)"
        - "Notify team of successful retrain"
        
  safeguards:
    max_retrain_frequency: "No more than once per day (prevent thrashing)"
    human_approval_required: "For regulated models (credit, healthcare)"
    rollback_always_available: "Previous model version deployable in <5 minutes"
```

### Training Data Strategy for Retraining

```yaml
Data_Strategy:
  sliding_window:
    what: "Train on most recent N months only"
    benefit: "Forgets outdated patterns, adapts to recent changes"
    risk: "May lose long-term patterns (seasonal, rare events)"
    typical: "3-12 months depending on domain"
    
  expanding_window:
    what: "Train on all historical data (growing over time)"
    benefit: "Maximum data, captures rare events, long-term patterns"
    risk: "Old patterns may not represent current reality (concept drift)"
    typical: "Used when data is expensive to collect or rare events matter"
    
  weighted_window:
    what: "Use all data but weight recent data higher"
    implementation: "Sample weight = exp(-λ × age_in_days)"
    benefit: "Best of both: retains old patterns but emphasizes recent"
    typical: "When both historical patterns and recent changes matter"
    
  hybrid:
    what: "Recent window + sampled historical (for rare events)"
    example: "Last 6 months full + 10% sample from older data (stratified by rare events)"
    benefit: "Captures recent patterns + doesn't forget rare but important cases"
    typical: "Fraud detection (recent patterns + historical rare fraud examples)"
```

---

## How It Works in Practice

### Production Retraining Workflow

```yaml
Workflow:
  daily:
    check_triggers:
      - "Is performance below threshold? (ground truth from 30 days ago)"
      - "Is drift above threshold? (from hourly drift monitoring)"
      - "Is scheduled retrain due? (check calendar)"
    if_triggered:
      - "Start continuous training pipeline (Dagster job)"
      - "Pipeline handles: data prep → train → validate → deploy (with gates)"
      
  monitoring_during_retrain:
    - "Track training progress (loss curves, convergence)"
    - "Validate intermediate checkpoints"
    - "Alert if training takes longer than expected"
    
  post_retrain:
    - "Shadow deploy new model (compare to current)"
    - "Canary deploy if shadow is good (5% traffic)"
    - "Full rollout if canary holds for 24 hours"
    - "Update reference distributions for monitoring"
    
  record_keeping:
    - "When was model retrained?"
    - "What triggered the retrain?"
    - "What data was used?"
    - "How did new model compare to old?"
    - "Any issues during deployment?"
```

---

## Interview Tip

> When asked about retraining triggers: "I use a multi-signal approach combining scheduled, performance-based, and drift-based triggers. Scheduled retraining (weekly or monthly) provides a safety net — prevents models from becoming stale even if monitoring misses something. Performance-triggered retraining fires when accuracy drops more than 5% from baseline (requires ground truth, which may be delayed 30+ days). Drift-triggered investigation fires when PSI > 0.2 on important features or multiple features drift simultaneously — but drift alone isn't sufficient to retrain (drift might be benign), so I verify performance impact before committing to retraining. The critical safeguard: validation gates before deployment. Even if retraining was triggered correctly, the new model must pass: (1) performance gate (at least as good as current model on test set), (2) fairness gate (passes 4/5ths rule), (3) latency gate (within serving SLA), and (4) regression gate (no segment degrades >2%). If any gate fails, the retrain is rejected and the current model continues serving. For continuous training at scale, I cap retrain frequency (no more than once per day to prevent thrashing), and implement automatic rollback if the newly deployed model degrades within the first 24 hours. The anti-pattern I avoid: retraining during a transient data anomaly — the new model learns the anomaly as 'normal' and performs worse when data returns to actual normal."

---

## Common Mistakes

1. **Retraining during data anomalies** — Drift detected (pipeline bug caused bad data for 2 hours) → trigger fires → model retrains on bad data → new model is worse. Solution: validate data quality BEFORE retraining. If recent data has quality issues, fix the data first. Never train on data you haven't validated.

2. **No validation gate** — Retrain triggered → new model deployed automatically → new model is actually worse (bad training run, insufficient data, etc.) → degraded predictions for hours. Solution: always validate new model against current model before deployment. New model must be equal or better; otherwise, keep current model.

3. **Too-frequent retraining (thrashing)** — Model retrained daily because drift signal oscillates around threshold. Each retrain produces slightly different model → inconsistent user experience, high compute cost, hard to track which version is which. Solution: minimum cooldown between retrains (e.g., 7 days unless critical degradation), confirm trigger is persistent (not transient noise).

4. **Only scheduled retraining** — Model retrained monthly on the 1st. Concept drift causes 15% accuracy drop on the 5th. Users suffer degraded predictions for 25 days until next scheduled retrain. Solution: combine scheduled with event-driven triggers. Scheduled = safety net. Performance/drift = responsive to actual problems.

5. **Not updating monitoring reference after retrain** — New model deployed with different prediction distribution (because it learned new patterns). Old monitoring reference → false drift alerts immediately. Solution: always update reference distributions after deploying a retrained model. New model = new baseline for monitoring.

---

## Key Takeaways

- Multiple trigger types: scheduled (safety net), performance (reactive), drift (proactive), event-driven
- Scheduled retraining: prevents staleness, simple to implement, but may be wasteful or too late
- Performance triggers: most reliable but require ground truth (often delayed)
- Drift triggers: proactive but not all drift causes degradation — validate before retraining
- Validation gates: new model must pass performance, fairness, latency, and regression checks
- Continuous training: automated pipeline with safeguards (data quality → train → validate → deploy)
- Cooldown period: minimum time between retrains to prevent thrashing
- Data anomaly protection: validate data quality before training (don't train on bad data)
- Training data strategy: sliding window (recent), expanding (all), or weighted hybrid
- Post-retrain: update monitoring reference distributions, shadow → canary → full rollout
