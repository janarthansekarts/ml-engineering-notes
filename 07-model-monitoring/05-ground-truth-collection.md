# Ground Truth Collection

## The Problem / Why This Matters

Ground truth (actual outcomes / labels) is the only way to definitively measure model performance in production. Without it, you're flying blind — drift detection and prediction monitoring give early warnings, but only ground truth confirms whether the model is actually making correct decisions. The challenge: ground truth is almost never available immediately. Fraud labels arrive 30-60 days after the transaction (after investigation completes). Loan default takes 6-24 months to manifest. Ad click happens seconds after impression, but conversion takes days. Medical diagnosis confirmation requires follow-up visits. This creates a "label lag" — a gap between when the model makes a prediction and when you know if it was right. During this gap, the model could be degrading and you wouldn't know. Production ML systems must implement strategies to collect ground truth as quickly as possible, handle delayed feedback gracefully, use proxy labels as interim signals, and design systems that eventually join predictions with outcomes for retrospective performance evaluation. Without deliberate ground truth collection, most teams never know their model's actual production accuracy — they only discover problems when business metrics tank.

---

## The Analogy

Think of ground truth collection like a student getting exam results:

- **Immediate feedback** = Pop quiz graded on the spot. You know instantly if you got it right. (Like ad click prediction — you know within seconds if the user clicked.)
- **Delayed feedback** = Term paper grade arrives 3 weeks later. You've already taken the next exam, but now you know how you did. (Like fraud detection — outcome arrives 30 days later.)
- **Partial feedback** = You get comments on your paper but not the final grade. Useful but incomplete. (Like proxy labels — user engagement signals without conversion data.)
- **No feedback** = You submit the paper but never hear back. You have no idea how you did. (Like many production models — predictions made but outcomes never collected.)

The worst scenario is no feedback. Even delayed feedback (3 weeks) is vastly better than nothing — it lets you identify degradation, measure drift impact, and validate that retraining helped.

---

## Deep Dive

### Label Lag Across Domains

```yaml
Label_Lag_Spectrum:
  immediate_seconds:
    examples:
      - "Click prediction (did user click? → known in seconds)"
      - "Real-time bidding (did we win the auction? → instant)"
      - "Spell correction (did user accept suggestion? → instant)"
    strategy: "Stream labels back, compute metrics in near real-time"
    
  minutes_to_hours:
    examples:
      - "Content recommendation (did user engage? → minutes)"
      - "Email open prediction (opened within first hour? → hours)"
      - "Session prediction (did user convert in this session? → ~30 min)"
    strategy: "Batch collect labels every few hours, compute daily metrics"
    
  days_to_weeks:
    examples:
      - "Ad conversion (clicked ad → purchased? → 1-7 days)"
      - "Churn prediction (did user churn in next 30 days? → 30 days)"
      - "Content moderation (was post reported/removed? → hours to days)"
    strategy: "Delayed join pipeline; use proxy labels for interim monitoring"
    
  weeks_to_months:
    examples:
      - "Fraud detection (investigation completes → 30-60 days)"
      - "Credit scoring (defaulted? → 6-24 months)"
      - "Insurance claims (claim filed? → months)"
      - "Drug efficacy (outcome observed? → months to years)"
    strategy: "Proxy labels essential; partial feedback; extended evaluation windows"
    
  never_or_difficult:
    examples:
      - "Recommendation (user never saw the non-recommended items)"
      - "Ad ranking (counterfactual: would user have clicked different ad?)"
      - "Preventive actions (fraud prevented → was it actually fraud?)"
    strategy: "Counterfactual estimation, A/B testing, randomized exploration"
```

### Collection Strategies

```python
# Ground truth collection implementation

"""
Systems for collecting and joining ground truth labels with predictions.
"""

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import asyncio


@dataclass
class PredictionRecord:
    """A stored prediction waiting for ground truth."""
    prediction_id: str
    model_version: str
    timestamp: datetime
    entity_id: str
    features: dict
    prediction: float
    confidence: float
    ground_truth: Optional[float] = None
    label_timestamp: Optional[datetime] = None
    label_source: Optional[str] = None


class GroundTruthCollector:
    """
    Collects and joins ground truth labels with predictions.
    
    Handles:
    - Direct label collection (explicit feedback)
    - Delayed label joining (outcome arrives later)
    - Proxy label computation (interim signals)
    - Partial labeling (sample-based evaluation)
    """
    
    def __init__(self, prediction_store, label_store, metrics_store):
        self.predictions = prediction_store
        self.labels = label_store
        self.metrics = metrics_store
    
    # Strategy 1: Direct label collection (immediate feedback)
    async def collect_direct_label(
        self,
        prediction_id: str,
        ground_truth: float,
        source: str = "direct",
    ):
        """
        Collect label that arrives immediately or shortly after prediction.
        Example: User clicked/didn't click after recommendation.
        """
        record = await self.predictions.get(prediction_id)
        if record:
            record.ground_truth = ground_truth
            record.label_timestamp = datetime.utcnow()
            record.label_source = source
            await self.predictions.update(record)
            
            # Immediately contribute to performance metrics
            await self.metrics.add_labeled_prediction(record)
    
    # Strategy 2: Delayed label joining (batch)
    async def run_delayed_label_join(
        self,
        label_delay_days: int = 30,
        batch_size: int = 10000,
    ):
        """
        Join delayed labels with historical predictions.
        Run as daily batch job.
        
        Example: Fraud labels arrive 30 days after transaction.
        Join today's labels with predictions from 30 days ago.
        """
        # Get labels that arrived today
        new_labels = await self.labels.get_recent(days=1)
        
        joined_count = 0
        for label in new_labels:
            # Find the prediction this label corresponds to
            prediction = await self.predictions.find(
                entity_id=label.entity_id,
                timestamp_range=(
                    label.event_timestamp - timedelta(hours=1),
                    label.event_timestamp + timedelta(hours=1),
                ),
            )
            
            if prediction and prediction.ground_truth is None:
                prediction.ground_truth = label.outcome
                prediction.label_timestamp = datetime.utcnow()
                prediction.label_source = "delayed_join"
                await self.predictions.update(prediction)
                joined_count += 1
        
        # Recompute metrics for the newly-labeled window
        labeled_window_start = datetime.utcnow() - timedelta(days=label_delay_days + 1)
        labeled_window_end = datetime.utcnow() - timedelta(days=label_delay_days)
        await self.metrics.recompute_window(labeled_window_start, labeled_window_end)
        
        return {"joined": joined_count, "total_labels": len(new_labels)}
    
    # Strategy 3: Proxy label computation
    async def compute_proxy_labels(self):
        """
        Compute proxy labels from behavioral signals when true labels
        are unavailable or delayed.
        
        Example: For churn prediction (true label in 30 days),
        use engagement signals as proxy (low engagement ≈ likely churning).
        """
        # Get recent predictions without labels
        unlabeled = await self.predictions.get_unlabeled(days=7)
        
        for record in unlabeled:
            # Compute proxy label from available signals
            proxy = await self._compute_proxy(record)
            
            if proxy is not None:
                record.ground_truth = proxy
                record.label_source = "proxy"
                record.label_timestamp = datetime.utcnow()
                await self.predictions.update(record)
    
    async def _compute_proxy(self, record: PredictionRecord) -> Optional[float]:
        """
        Compute proxy label based on available signals.
        Must be validated against true labels (when they eventually arrive).
        """
        entity_id = record.entity_id
        
        # Example proxy for churn: user hasn't logged in for 14 days
        last_activity = await self.get_last_activity(entity_id)
        if last_activity:
            days_inactive = (datetime.utcnow() - last_activity).days
            if days_inactive > 14:
                return 1.0  # Proxy: likely churned
            elif days_inactive < 3:
                return 0.0  # Proxy: likely active
        
        return None  # Can't determine proxy
    
    # Strategy 4: Human evaluation (sampling)
    async def request_human_evaluation(
        self,
        sample_size: int = 100,
        sampling_strategy: str = "stratified",
    ):
        """
        Sample predictions for human evaluation.
        Used when labels can't be collected automatically.
        
        Example: LLM response quality — humans rate a sample.
        """
        # Get recent unlabeled predictions
        unlabeled = await self.predictions.get_unlabeled(days=1)
        
        if sampling_strategy == "stratified":
            # Sample across confidence levels (more low-confidence samples)
            sample = self._stratified_sample(unlabeled, sample_size)
        elif sampling_strategy == "uncertainty":
            # Focus on uncertain predictions (most informative)
            sample = sorted(unlabeled, key=lambda p: abs(p.confidence - 0.5))[:sample_size]
        else:
            # Random sample
            sample = np.random.choice(unlabeled, min(sample_size, len(unlabeled)), replace=False)
        
        # Queue for human review
        review_batch = await self.create_review_batch(sample)
        return review_batch
```

### Proxy Labels

```yaml
Proxy_Labels:
  what: "Intermediate signals that approximate the true outcome"
  why: "Bridge the label lag gap — provide feedback before true labels arrive"
  
  examples:
    churn_prediction:
      true_label: "Did user cancel subscription? (known in 30 days)"
      proxy_labels:
        - "Days since last login (>14 days ≈ likely churning)"
        - "Feature usage declining (engagement drop)"
        - "Support tickets increasing (frustration signal)"
      proxy_quality: "~75% correlation with true churn label"
      
    fraud_detection:
      true_label: "Was transaction confirmed fraudulent? (known in 30-60 days)"
      proxy_labels:
        - "User disputed transaction (strong proxy, ~90% accuracy)"
        - "Card blocked by bank (strong proxy)"
        - "Multiple rapid transactions (behavioral proxy)"
      proxy_quality: "Dispute = ~90% correlation; behavioral = ~60% correlation"
      
    content_recommendation:
      true_label: "Did user find content valuable? (hard to define)"
      proxy_labels:
        - "Time spent on content (>30 seconds = engaged)"
        - "Scrolled to bottom (consumed full content)"
        - "Shared content (strong positive signal)"
        - "Bounced immediately (<5 seconds = not relevant)"
      proxy_quality: "Time-based proxy = ~65% correlation with explicit ratings"
      
    credit_scoring:
      true_label: "Did borrower default? (known in 6-24 months)"
      proxy_labels:
        - "Missed first payment (strong early signal, ~80% correlation)"
        - "Payment amount below minimum (early warning)"
        - "Account flagged for collections (strong proxy)"
      proxy_quality: "Missed first payment = ~80% of eventual defaults"
      
  validation:
    requirement: "ALWAYS validate proxy labels against true labels (when available)"
    how: "For historical data where true labels exist, compute proxy → true label correlation"
    frequency: "Re-validate monthly (proxy quality can degrade)"
    red_flag: "If proxy-true correlation drops below 0.6, proxy is unreliable"
```

### Human-in-the-Loop Evaluation

```yaml
Human_Evaluation:
  when_needed:
    - "LLM responses (no automatic 'correct' answer)"
    - "Creative content generation (subjective quality)"
    - "Complex predictions where automated signals are unreliable"
    - "New model validation before full deployment"
    - "Dispute resolution (conflicting signals)"
    
  sampling_strategies:
    random:
      what: "Uniform random sample of predictions"
      pros: "Unbiased estimate of overall performance"
      cons: "Expensive — many samples are 'obviously correct'"
      sample_size: "100-500 per evaluation cycle"
      
    uncertainty:
      what: "Sample predictions with lowest confidence"
      pros: "Finds failures efficiently (model isn't sure = likely wrong)"
      cons: "Biased toward difficult cases (overestimates error rate)"
      correction: "Weight results by inverse sampling probability"
      
    stratified:
      what: "Sample from each confidence/segment bucket"
      pros: "Balanced view across all prediction types"
      approach: "20% from high-confidence, 40% from medium, 40% from low"
      
    adversarial:
      what: "Sample cases where model disagrees with simple rules"
      pros: "Finds systematic failures and edge cases"
      approach: "Model predicts X but business rule predicts Y → evaluate"
      
  evaluation_design:
    for_classification:
      task: "Is this prediction correct? (yes/no)"
      guidelines: "Clear labeling criteria with examples"
      agreement: "Minimum 2 annotators, resolve disagreements with 3rd"
      
    for_llm_responses:
      task: "Rate response quality on multiple dimensions"
      dimensions:
        - "Relevance (1-5): Does response answer the question?"
        - "Accuracy (1-5): Is information factually correct?"
        - "Completeness (1-5): Is anything missing?"
        - "Clarity (1-5): Is response well-written and clear?"
      guidelines: "Detailed rubric with examples for each score level"
      
  platforms:
    - "Scale AI (managed labeling workforce)"
    - "Labelbox (labeling platform with ML-assisted tooling)"
    - "Internal evaluation queues (for sensitive data)"
    - "LLM-as-judge (Claude 4 / GPT-5 evaluating other LLM outputs)"
```

### Joining Predictions with Outcomes

```yaml
Join_Architecture:
  event_driven:
    what: "Labels stream in as events → join with stored predictions"
    implementation:
      - "Predictions stored in time-partitioned table (S3/BigQuery)"
      - "Labels arrive as events (Kafka/database triggers)"
      - "Join job runs continuously or hourly"
      - "Matched prediction-label pairs written to evaluation table"
    challenge: "Matching — which prediction corresponds to which label?"
    solution: "Store prediction_id in the system that generates labels"
    
  batch_join:
    what: "Periodically join all new labels with historical predictions"
    implementation:
      - "Daily batch job reads new labels + unlabeled predictions"
      - "Join on (entity_id, timestamp window)"
      - "Write joined records to evaluation table"
      - "Recompute performance metrics"
    schedule: "Daily (for fast-labeling domains) or weekly"
    
  label_propagation:
    what: "One label applies to multiple predictions"
    example: "User churned → all churn predictions for this user in last 30 days get labeled"
    implementation: "When label arrives, find all predictions for entity in relevant window"
    
  join_keys:
    primary: "prediction_id (most reliable, stored with the outcome)"
    fallback: "entity_id + timestamp_window (when prediction_id not stored)"
    challenge: "Multiple predictions for same entity (which one gets the label?)"
    resolution: "Use the prediction closest in time to the label-triggering event"
```

---

## How It Works in Practice

### Production Ground Truth Pipeline

```yaml
Pipeline:
  for_fast_labels_seconds:
    example: "Click prediction"
    flow: "Prediction logged → user event streams → join in <1 minute → metrics updated"
    freshness: "Real-time performance metrics (1-5 minute delay)"
    
  for_medium_labels_days:
    example: "Conversion prediction"  
    flow: "Prediction logged → daily batch job → join labels from 7 days ago → metrics"
    freshness: "Performance metrics with 7-day lag"
    interim: "Proxy labels (engagement signals) for monitoring in the gap"
    
  for_slow_labels_months:
    example: "Credit default prediction"
    flow: "Prediction logged → monthly batch → join labels from 6-12 months ago → metrics"
    freshness: "Performance metrics with 6-12 month lag"
    interim: "Early proxy labels (missed first payment) + human evaluation samples"
    
  for_subjective_labels:
    example: "LLM response quality"
    flow: "Predictions sampled → human evaluation queue → daily ratings → metrics"
    freshness: "2-3 day turnaround for sampled evaluations"
    coverage: "~1-5% of predictions labeled by humans"
    supplemental: "LLM-as-judge (automated evaluation) for 100% coverage"
```

---

## Interview Tip

> When asked about ground truth collection: "Ground truth collection is the foundation of production model monitoring — without it, you can't measure actual performance. My approach depends on label lag: For fast labels (clicks, <1 minute): stream labels back in real-time, join with predictions, compute live performance metrics. For medium delay (conversions, 1-7 days): daily batch join pipeline, plus proxy labels for interim monitoring. For slow labels (fraud: 30 days, credit default: 6+ months): I implement a three-layer strategy: (1) proxy labels for fast signal (missed first payment correlates ~80% with eventual default), (2) human evaluation on sampled predictions (stratified by confidence — more low-confidence samples), and (3) eventual batch join when true labels arrive for retrospective analysis. The proxy labels are critical — I validate them against true labels (when those eventually arrive) and monitor the proxy-truth correlation over time. If it degrades below 0.6, the proxy is unreliable. For LLM applications where 'correct' is subjective, I use LLM-as-judge (Claude 4 evaluating responses) for automated 100% coverage, validated against periodic human evaluation (~100 samples/week). The key engineering: always store a prediction_id that can be joined with the outcome later, design the label join pipeline from day one (not as an afterthought), and validate that proxy labels actually correlate with true outcomes."

---

## Common Mistakes

1. **Never collecting ground truth** — "We deployed the model 6 months ago and never measured actual accuracy in production." Team assumes training accuracy holds forever. Solution: implement ground truth collection from day one — even if labels are delayed, the join pipeline must exist. Any feedback (even 1% of predictions labeled) is infinitely better than zero.

2. **Trusting proxy labels without validation** — "Users who don't log in for 7 days are churned" (proxy). But only 50% of 7-day-inactive users actually cancel. Proxy gives inflated churn rate → false sense of model failure. Solution: validate proxy labels against true labels. Compute correlation, precision, and recall of the proxy. Document proxy limitations.

3. **Ignoring selection bias in labels** — Only getting labels for predictions that triggered action. Fraud model flags transaction → investigation happens → label collected. Transactions not flagged → never investigated → no label. You only know accuracy on the flagged set (biased). Solution: randomly sample unflagged predictions for investigation (even if expensive). Use exploration strategies to collect unbiased labels.

4. **Not accounting for label lag in metrics** — Computing "last 7 days accuracy" when labels have 30-day lag. Result: you're looking at accuracy from 30+ days ago and calling it "current." Solution: clearly timestamp when labels were available, not when predictions were made. Dashboard should show: "Performance for predictions made 30-37 days ago (labels just arrived)."

5. **No human evaluation for subjective tasks** — LLM response quality measured only by automated metrics (BLEU, ROUGE) that don't correlate well with actual quality. Solution: regular human evaluation (weekly samples), use LLM-as-judge for coverage, validate automated metrics against human ratings.

---

## Key Takeaways

- Ground truth is the only definitive measure of production model performance
- Label lag varies: seconds (clicks), days (conversions), months (fraud, credit)
- Proxy labels bridge the gap: behavioral signals that approximate true outcomes
- Always validate proxies against true labels (monitor proxy-truth correlation)
- Human evaluation: essential for subjective tasks (LLM quality, creative content)
- Sampling strategies: random (unbiased), uncertainty (efficient), stratified (balanced)
- Join pipeline: predictions stored with ID → labels arrive → batch/stream join → metrics
- Selection bias: labels only for actioned predictions → biased performance estimates
- LLM-as-judge: automated evaluation at scale, validated against human ratings
- Design ground truth collection from day one — not as an afterthought
