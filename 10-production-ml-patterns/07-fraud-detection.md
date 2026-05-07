# Fraud Detection

## The Problem / Why This Matters

Fraud detection is one of the most challenging production ML applications — it combines real-time inference requirements (decisions in milliseconds), extreme class imbalance (< 0.1% of transactions are fraudulent), adversarial dynamics (fraudsters actively adapt to evade detection), and severe consequences of errors (false positives block legitimate customers, false negatives cause financial loss). In 2026, financial fraud costs exceed $40 billion annually, and ML is the primary defense. A production fraud system must: score every transaction in real-time (< 100ms), maintain high precision (avoid blocking good customers) while catching most fraud (high recall), adapt to new fraud patterns within hours (not weeks), handle concept drift as fraudster tactics evolve, and provide explainable decisions for compliance (regulatory requirement under EU AI Act and similar frameworks). The engineering patterns — real-time feature computation, cascade architecture, feedback loops, ensemble models for robustness, and graph-based detection — represent some of the most sophisticated production ML patterns in industry.

---

## The Analogy

Think of fraud detection like airport security:

- **Rules-based** (legacy) = A list of banned items. Easy to check, but sophisticated threats evolve faster than the list updates. Fraudsters learn what's on the list and adapt.
- **ML-based** = Trained security officers who recognize suspicious patterns — nervous behavior, unusual routes, inconsistent stories. They adapt to new tactics by learning from experience. But they must also avoid blocking innocent travelers (false positives create terrible experiences).
- **Multi-layer defense** = The full airport security system: initial screening (rules), X-ray machines (automated ML), and human officers for uncertain cases (human-in-the-loop). No single layer catches everything — defense in depth.

---

## Deep Dive

### Fraud Detection System Architecture

```yaml
Architecture:
  multi_layer_defense:
    layer_1_rules:
      latency: "< 1ms"
      catches: "Known fraud patterns, blacklists, velocity limits"
      examples:
        - "Card used in 2 countries within 1 hour → block"
        - "Amount > $10,000 from new device → flag"
        - "IP on known fraud list → block"
      false_positive_rate: "Low (rules are precise)"
      coverage: "Catches ~30% of fraud (obvious patterns)"
      
    layer_2_ml_model:
      latency: "5-20ms"
      catches: "Complex patterns rules can't express"
      model_types:
        gradient_boosted_trees: "XGBoost/LightGBM — fast, interpretable, handles tabular data well"
        neural_network: "Deep learning for sequential behavior patterns"
        ensemble: "Multiple models vote (robustness against adversarial manipulation)"
      coverage: "Catches ~60% of remaining fraud"
      
    layer_3_graph_analysis:
      latency: "50-200ms"
      catches: "Fraud rings, connected suspicious entities"
      approach: "Graph neural networks on transaction/entity graph"
      examples: "Multiple accounts sharing same device, shipping address, or payment method"
      
    layer_4_human_review:
      latency: "Minutes to hours"
      catches: "Cases where models are uncertain"
      when: "Confidence between 0.3-0.7 (not clearly fraud or legitimate)"
      feedback: "Analyst decisions become training labels"
      
  real_time_features:
    streaming_aggregates:
      - "transaction_count_5min: # transactions in last 5 minutes"
      - "unique_merchants_1hr: # distinct merchants in last hour"
      - "amount_sum_24hr: total spend in last 24 hours"
      - "country_count_24hr: # distinct countries in last 24 hours"
      - "velocity_change: current vs. historical transaction frequency"
      
    profile_features:
      - "account_age_days: how old is the account"
      - "avg_transaction_amount: historical average"
      - "typical_merchants: usual merchant categories"
      - "typical_locations: normal geographic patterns"
      - "device_history: known devices for this user"
      
    contextual_features:
      - "time_since_last_transaction: seconds since previous"
      - "amount_vs_average: how unusual is this amount"
      - "new_merchant: first time at this merchant?"
      - "new_device: first time from this device?"
      - "distance_from_last: geographic distance from last transaction"
```

### Implementation

```python
# Production fraud detection system

"""
Real-time fraud detection with multi-layer architecture,
streaming features, and adaptive learning.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    """A payment transaction to score."""
    transaction_id: str
    user_id: str
    amount: float
    merchant_id: str
    merchant_category: str
    device_id: str
    ip_address: str
    location: Tuple[float, float]  # (lat, lon)
    timestamp: float


@dataclass
class FraudDecision:
    """Fraud scoring decision."""
    transaction_id: str
    risk_score: float  # 0.0 (legitimate) to 1.0 (definitely fraud)
    decision: str  # "approve", "decline", "review"
    reasons: List[str]
    layer_triggered: str  # Which layer made the decision
    latency_ms: float
    model_version: str


class FraudDetectionSystem:
    """
    Multi-layer fraud detection system.
    
    Cascade architecture:
    1. Rules (fast, precise) → catches obvious fraud
    2. ML model (medium) → catches complex patterns
    3. Graph analysis (slow) → catches fraud rings
    4. Human review (slowest) → handles uncertain cases
    """
    
    def __init__(
        self,
        rules_engine,
        ml_model,
        graph_analyzer,
        feature_store,
        streaming_features,
        thresholds: Dict[str, float] = None,
    ):
        self.rules_engine = rules_engine
        self.ml_model = ml_model
        self.graph_analyzer = graph_analyzer
        self.feature_store = feature_store
        self.streaming_features = streaming_features
        
        # Decision thresholds
        self.thresholds = thresholds or {
            "auto_decline": 0.9,   # Very high confidence → decline
            "review": 0.5,         # Medium confidence → human review
            "auto_approve": 0.1,   # Very low risk → approve
        }
    
    def score_transaction(self, transaction: Transaction) -> FraudDecision:
        """
        Score a transaction through the fraud detection cascade.
        
        Returns decision within ~50ms for most transactions.
        """
        start_time = time.perf_counter()
        reasons = []
        
        # Layer 1: Rules (< 1ms)
        rule_result = self.rules_engine.check(transaction)
        if rule_result["blocked"]:
            return FraudDecision(
                transaction_id=transaction.transaction_id,
                risk_score=1.0,
                decision="decline",
                reasons=rule_result["reasons"],
                layer_triggered="rules",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                model_version="rules_v1",
            )
        
        if rule_result["risk_signals"]:
            reasons.extend(rule_result["risk_signals"])
        
        # Layer 2: ML Model (5-20ms)
        features = self._get_features(transaction)
        ml_score = self.ml_model.predict_proba(features)
        
        # Quick decision if very confident
        if ml_score >= self.thresholds["auto_decline"]:
            reasons.append(f"ML model high risk: {ml_score:.3f}")
            return FraudDecision(
                transaction_id=transaction.transaction_id,
                risk_score=ml_score,
                decision="decline",
                reasons=reasons,
                layer_triggered="ml_model",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                model_version=self.ml_model.version,
            )
        
        if ml_score <= self.thresholds["auto_approve"]:
            return FraudDecision(
                transaction_id=transaction.transaction_id,
                risk_score=ml_score,
                decision="approve",
                reasons=["Low risk score"],
                layer_triggered="ml_model",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                model_version=self.ml_model.version,
            )
        
        # Layer 3: Graph analysis for medium-risk cases (50-200ms)
        if ml_score >= 0.3:
            graph_score = self.graph_analyzer.analyze(
                user_id=transaction.user_id,
                device_id=transaction.device_id,
                ip_address=transaction.ip_address,
            )
            # Combine ML and graph scores
            combined_score = 0.7 * ml_score + 0.3 * graph_score
        else:
            combined_score = ml_score
        
        # Final decision
        if combined_score >= self.thresholds["auto_decline"]:
            decision = "decline"
        elif combined_score >= self.thresholds["review"]:
            decision = "review"
        else:
            decision = "approve"
        
        return FraudDecision(
            transaction_id=transaction.transaction_id,
            risk_score=combined_score,
            decision=decision,
            reasons=reasons,
            layer_triggered="combined",
            latency_ms=(time.perf_counter() - start_time) * 1000,
            model_version=self.ml_model.version,
        )
    
    def _get_features(self, transaction: Transaction) -> Dict[str, float]:
        """
        Combine batch features (feature store) + streaming features (real-time).
        """
        # Pre-computed user profile features
        profile_features = self.feature_store.get_user_features(transaction.user_id)
        
        # Real-time streaming aggregates
        stream_features = self.streaming_features.get(
            user_id=transaction.user_id,
            current_timestamp=transaction.timestamp
        )
        
        # Transaction-specific features
        tx_features = {
            "amount": transaction.amount,
            "amount_vs_avg": transaction.amount / max(profile_features.get("avg_amount", 50), 1),
            "is_new_merchant": float(transaction.merchant_id not in profile_features.get("known_merchants", set())),
            "is_new_device": float(transaction.device_id not in profile_features.get("known_devices", set())),
            "hour_of_day": int((transaction.timestamp % 86400) / 3600),
            "is_weekend": float(time.gmtime(transaction.timestamp).tm_wday >= 5),
        }
        
        # Combine all features
        return {**profile_features, **stream_features, **tx_features}


class AdaptiveFraudModel:
    """
    Fraud model that adapts to evolving fraud patterns.
    
    Key challenge: fraud patterns change weekly as fraudsters adapt.
    Model trained on last month's data may miss new attack patterns.
    
    Solution:
    - Retrain frequently (daily or weekly)
    - Monitor for concept drift (new patterns model hasn't seen)
    - Fast feedback loop (analyst labels → training data → retrained model)
    """
    
    def __init__(self, base_model, drift_detector, retraining_pipeline):
        self.model = base_model
        self.drift_detector = drift_detector
        self.retraining_pipeline = retraining_pipeline
        self.version = base_model.version
    
    def should_retrain(self) -> Tuple[bool, str]:
        """
        Check if model needs retraining.
        
        Signals:
        1. Performance degradation (precision/recall dropping)
        2. Feature drift (input distribution shifted)
        3. Scheduled (weekly regardless of drift)
        4. New fraud pattern detected (manual trigger)
        """
        # Check performance metrics
        recent_precision = self._get_recent_precision()
        if recent_precision < 0.8:  # Below acceptable threshold
            return True, f"Precision dropped to {recent_precision:.2f}"
        
        # Check feature drift
        drift_detected = self.drift_detector.check()
        if drift_detected:
            return True, "Feature distribution drift detected"
        
        # Check time since last training
        days_since_training = self._days_since_last_train()
        if days_since_training >= 7:
            return True, f"Scheduled retrain ({days_since_training} days since last)"
        
        return False, "No retrain needed"
    
    def retrain(self, include_recent_labels: bool = True):
        """
        Retrain model on latest data.
        
        Includes:
        - Historical labeled data (confirmed fraud/legitimate)
        - Recent analyst decisions (manual reviews)
        - Synthetic oversampling of rare fraud types (SMOTE)
        """
        training_data = self.retraining_pipeline.prepare_training_data(
            include_recent_labels=include_recent_labels
        )
        
        # Train new model
        new_model = self.retraining_pipeline.train(training_data)
        
        # Validate: new model must be better than current
        validation = self.retraining_pipeline.validate(
            new_model=new_model,
            current_model=self.model
        )
        
        if validation["new_is_better"]:
            self.model = new_model
            self.version = new_model.version
            logger.info(f"Model updated to {self.version}")
        else:
            logger.warning("New model not better than current, keeping old model")
    
    def _get_recent_precision(self) -> float:
        """Compute precision from recent decisions with known outcomes."""
        # In production: query decisions table joined with outcomes
        return 0.85  # Placeholder
    
    def _days_since_last_train(self) -> int:
        """Days since model was last retrained."""
        return 5  # Placeholder


class ClassImbalanceHandler:
    """
    Handle extreme class imbalance in fraud detection.
    
    Challenge: < 0.1% of transactions are fraud.
    Naive training: model predicts "legitimate" for everything (99.9% accuracy!).
    But catches 0% of fraud (useless).
    
    Strategies:
    1. Oversampling minority class (SMOTE)
    2. Undersampling majority class
    3. Class weights (penalize missed fraud more)
    4. Anomaly detection (model "normal", flag deviations)
    5. Focal loss (focus training on hard examples)
    """
    
    @staticmethod
    def prepare_balanced_training(
        X_train, y_train,
        strategy: str = "class_weight",
        fraud_weight_multiplier: float = 100.0
    ):
        """
        Prepare training data with class imbalance handling.
        """
        fraud_count = sum(y_train)
        legit_count = len(y_train) - fraud_count
        imbalance_ratio = legit_count / max(fraud_count, 1)
        
        logger.info(
            f"Class imbalance: {fraud_count} fraud / {legit_count} legit "
            f"(ratio: 1:{imbalance_ratio:.0f})"
        )
        
        if strategy == "class_weight":
            # Weight fraud examples higher in loss function
            sample_weights = np.where(
                y_train == 1,
                fraud_weight_multiplier,
                1.0
            )
            return X_train, y_train, sample_weights
        
        elif strategy == "undersample":
            # Reduce legitimate examples to match fraud count
            legit_indices = np.where(y_train == 0)[0]
            fraud_indices = np.where(y_train == 1)[0]
            
            # Keep all fraud, sample legitimate
            sample_size = min(len(legit_indices), len(fraud_indices) * 10)
            selected_legit = np.random.choice(legit_indices, sample_size, replace=False)
            
            all_indices = np.concatenate([fraud_indices, selected_legit])
            np.random.shuffle(all_indices)
            
            return X_train[all_indices], y_train[all_indices], None
        
        elif strategy == "smote":
            # Synthetic Minority Oversampling Technique
            # Generate synthetic fraud examples by interpolating between real ones
            # In production: use imblearn.over_sampling.SMOTE
            logger.info("Using SMOTE oversampling for minority class")
            # Placeholder: in production use imblearn
            return X_train, y_train, None
```

### Feedback Loops

```yaml
Feedback_Loop:
  challenge: "Labels arrive with delay (fraud confirmed days/weeks later)"
  
  label_sources:
    chargebacks: "Customer disputes charge (2-90 days delay)"
    analyst_review: "Human reviews flagged transactions (hours delay)"
    account_takeover: "Customer reports unauthorized access (days delay)"
    self_reported: "Merchant reports fraud (1-7 days delay)"
    
  delayed_label_handling:
    approach_1: "Wait for labels (retrain weekly with last week's confirmed labels)"
    approach_2: "Semi-supervised (use unlabeled data with high-confidence predictions)"
    approach_3: "Active learning (prioritize review of most informative cases)"
    
  feedback_loop_risk:
    problem: "Model blocks fraud → less fraud in training data → model learns fraud is rare → model stops detecting"
    solution: "Randomized holdout: let 0.1% of flagged transactions through for ground truth"
    alternative: "Synthetic fraud injection: generate realistic fraud patterns for training"
```

---

## How It Works in Practice

### Real-Time Transaction Scoring

```yaml
Transaction_Scoring:
  volume: "10,000 transactions per second"
  latency_sla: "< 50ms P99"
  
  flow:
    t_0ms: "Transaction received from payment gateway"
    t_1ms: "Rules engine check (blacklists, velocity limits)"
    t_5ms: "Streaming features fetched (Redis: tx count last 5min, amount sum 1hr)"
    t_8ms: "Profile features fetched (feature store: account age, avg amount)"
    t_15ms: "Feature vector assembled"
    t_20ms: "ML model inference (XGBoost ensemble)"
    t_22ms: "Decision: approve/decline/review"
    t_25ms: "Response sent to payment gateway"
    t_30ms: "Prediction logged (async, for monitoring)"
    
  model_performance:
    precision: "95% (5% of declines are legitimate users → bad UX)"
    recall: "85% (catches 85% of fraud)"
    false_positive_rate: "0.5% (1 in 200 legitimate transactions incorrectly blocked)"
    
  business_impact:
    fraud_prevented: "$50M per month"
    false_positive_cost: "$2M per month (lost revenue from blocked good customers)"
    model_improvement: "5% recall improvement = $5M more fraud caught per month"
```

---

## Interview Tip

> When asked about fraud detection: "I design fraud detection as a multi-layer cascade optimized for different latency/accuracy trade-offs. Layer 1 — Rules (< 1ms): blacklists, velocity limits, impossible travel. Catches 30% of fraud with near-zero false positives. Layer 2 — ML model (< 20ms): XGBoost ensemble on 100+ features combining pre-computed profile features (from feature store) with real-time streaming features (transaction velocity in last 5 minutes from Kafka/Redis). Handles class imbalance with class weighting (100x penalty for missed fraud) and trains on both confirmed fraud and analyst-reviewed labels. Layer 3 — Graph analysis (< 200ms) for medium-risk cases: detects fraud rings by analyzing shared devices, IP addresses, and shipping addresses across accounts. Key engineering challenges: (1) Streaming features — compute transaction velocity, location anomaly in real-time via Flink/Kafka, materialize to Redis for sub-5ms serving. (2) Feedback loops — labels arrive days later (chargebacks), so I retrain weekly. Use randomized holdout (0.1% of flagged transactions approved) to prevent the feedback loop problem (blocking all fraud → no fraud in training data → model degrades). (3) Adversarial adaptation — fraudsters learn model behavior. I retrain frequently, monitor for concept drift, and use ensemble of diverse models (harder for adversaries to game all simultaneously). (4) Explainability — regulatory compliance requires explaining why a transaction was declined. I use SHAP values to show top features driving the decision."

---

## Common Mistakes

1. **Optimizing accuracy instead of precision/recall** — 99.9% accuracy is trivial (just predict "legitimate" always). Solution: optimize for recall at a fixed precision (e.g., catch 85% of fraud while keeping precision > 95%). Use precision-recall curve, not ROC.

2. **Ignoring the feedback loop** — Model blocks fraud → less fraud in training data → model learns fraud is rarer than it is → model blocks less fraud → fraud increases. Solution: randomized holdout (approve 0.1% of flagged transactions to get ground truth labels).

3. **Training on static data** — Model trained on 6-month-old data. Fraud patterns changed. New account-takeover technique not in training data. Solution: retrain weekly on latest labeled data. Monitor precision/recall in production. Alert on drift.

4. **No streaming features** — Using only batch features (computed daily). Fraudster makes 20 transactions in 5 minutes before daily batch catches the pattern. Solution: streaming features (transaction count/amount in last 5/15/60 minutes) computed in real-time via Kafka/Flink.

5. **Single model without ensemble** — One model is easier to game. Adversary figures out what triggers the model and avoids those patterns. Solution: ensemble of diverse models (different algorithms, different feature sets). Harder for adversary to evade all simultaneously.

---

## Key Takeaways

- Fraud detection: multi-layer cascade (rules → ML → graph → human review)
- Real-time features: streaming aggregates (velocity, amount sums) via Kafka → Redis
- Class imbalance: < 0.1% fraud — handle with class weighting, SMOTE, or anomaly detection
- Adversarial dynamics: fraudsters adapt — retrain frequently, use diverse ensembles
- Feedback loop: blocking fraud removes it from training data — use randomized holdout
- Delayed labels: fraud confirmed days/weeks later — retrain weekly with confirmed labels
- Metrics: precision-recall (not accuracy), recall at fixed precision threshold
- Graph analysis: detect fraud rings (shared devices, IPs, addresses across accounts)
- Explainability: SHAP values for regulatory compliance (why was transaction declined?)
- Latency: < 50ms P99 for real-time scoring (rules < 1ms, ML < 20ms, graph < 200ms)
