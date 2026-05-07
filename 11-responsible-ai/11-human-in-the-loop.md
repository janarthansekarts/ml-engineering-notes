# Human-in-the-Loop

## The Problem / Why This Matters

Not every decision should be fully automated. A model with 95% accuracy still gets 1 in 20 decisions wrong — and in high-stakes domains (healthcare, criminal justice, lending, hiring), those errors can destroy lives. Human-in-the-loop (HITL) ML is the practice of strategically involving humans in the ML pipeline — during labeling, training, inference, and evaluation — to catch errors, handle edge cases, provide feedback, and maintain accountability. In 2026, HITL isn't just about safety — it's a legal requirement. The EU AI Act mandates human oversight for high-risk AI systems. GDPR (General Data Protection Regulation) Article 22 gives individuals the right not to be subject to purely automated decisions. Financial regulations require human review of credit denials. But HITL done wrong is worse than no HITL: rubber-stamping (human approves everything without review), automation bias (humans defer to model even when they shouldn't), alert fatigue (too many escalations, human stops paying attention). ML engineers must design HITL systems that are genuinely effective — routing the right cases to humans, presenting information to support good decisions, and measuring whether human involvement actually improves outcomes.

---

## The Analogy

Think of human-in-the-loop like a semi-autonomous car:

- **Full automation (no human)** = Self-driving car with no steering wheel. Works great on highways (easy cases). Catastrophic in construction zones (edge cases). If it fails, there's no backup.
- **Human-in-the-loop** = Tesla Autopilot. Car handles routine driving (model handles easy decisions). When confused or in dangerous situations, it alerts the driver to take over. Driver is the safety net.
- **The problem** = Drivers who trust autopilot too much fall asleep. They're "in the loop" but not actually supervising. This is automation bias — the human is technically there but provides zero value.
- **Effective HITL** = Airline cockpit. Two pilots cross-check each other AND the automation. Clear protocols for when human overrides automation. Regular training on edge cases. Alerts designed to be actionable, not overwhelming.

---

## Deep Dive

### When to Involve Humans

```yaml
When_To_Involve_Humans:
  always_human:
    scenarios:
      - "Irreversible decisions (death penalty, organ transplant priority)"
      - "Novel situations with no training data precedent"
      - "High liability decisions (medical diagnosis, criminal sentencing)"
      - "Cases where model confidence is below threshold"
      - "First deployment (shadow mode — human decides, model learns)"
      
  human_review:
    scenarios:
      - "Medium-confidence predictions (model unsure)"
      - "Decisions affecting vulnerable populations"
      - "Random sample for quality assurance"
      - "Appeals/disputes from affected individuals"
      - "Model flagged potential errors"
      
  full_automation:
    scenarios:
      - "High-confidence, low-stakes decisions (spam filtering)"
      - "Speed-critical with reversible outcomes (content recommendation)"
      - "Volume too high for human review (millions of decisions/second)"
      - "Model performance verified and monitored continuously"
      
  decision_framework:
    criteria:
      stakes: "What's the cost of a wrong decision?"
      reversibility: "Can the decision be undone?"
      volume: "How many decisions per day?"
      model_confidence: "How certain is the model?"
      regulatory: "Does regulation require human oversight?"
      novelty: "Is this situation similar to training data?"
```

### Confidence Thresholds

```yaml
Confidence_Thresholds:
  concept: |
    Route decisions based on model confidence:
    - High confidence → automate
    - Medium confidence → human review
    - Low confidence → reject/escalate
    
  calibration_requirement: |
    CRITICAL: Confidence thresholds only work if model is CALIBRATED.
    Calibration = when model says 80% confident, it's correct 80% of the time.
    Uncalibrated model with threshold = dangerous false sense of security.
    
  threshold_design:
    two_threshold:
      above_upper: "Auto-approve (high confidence)"
      between: "Route to human review (uncertain)"
      below_lower: "Auto-reject or escalate (low confidence/high risk)"
      
    example_lending:
      auto_approve: "Model confidence > 0.95 (very likely to repay)"
      human_review: "Model confidence 0.4 - 0.95 (uncertain cases)"
      auto_deny: "Model confidence < 0.4 (very likely to default)"
      
    tuning:
      - "Upper threshold: set based on desired automation rate + acceptable error"
      - "Lower threshold: set based on maximum false positive tolerance"
      - "Between: this is the human workload — must be manageable"
      
  workload_management:
    problem: "If 60% of cases fall in 'human review' zone, humans can't keep up"
    solutions:
      - "Tighten thresholds (more automation, less human review)"
      - "Prioritize within human queue (highest uncertainty first)"
      - "Active learning: use human decisions to retrain and narrow uncertain zone"
      - "Batch similar cases for faster human processing"
```

### Implementation

```python
# Human-in-the-loop ML system implementation

"""
Human-in-the-loop system: confidence-based routing, human review queues,
feedback collection, and active learning integration.
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import uuid

logger = logging.getLogger(__name__)


class RoutingDecision(Enum):
    """Where to route a prediction."""
    AUTOMATE = "automate"  # Model decides, no human
    HUMAN_REVIEW = "human_review"  # Human reviews model prediction
    HUMAN_DECIDE = "human_decide"  # Human makes decision (model provides info only)
    REJECT = "reject"  # Auto-reject (model very confident negative)
    ESCALATE = "escalate"  # Complex case, needs senior/specialist


@dataclass
class ReviewItem:
    """An item in the human review queue."""
    item_id: str
    prediction: float
    confidence: float
    features: Dict
    model_explanation: str
    routing_reason: str
    priority: int  # 1 (highest) to 5 (lowest)
    created_at: str
    assigned_to: Optional[str] = None
    human_decision: Optional[str] = None
    decision_time: Optional[str] = None
    feedback: Optional[str] = None


class HITLRouter:
    """
    Routes predictions to automation or human review based on confidence.
    
    Key design principles:
    1. Route based on calibrated confidence (must verify calibration)
    2. Manage human workload (don't overwhelm reviewers)
    3. Prioritize by impact and uncertainty
    4. Collect human decisions for model improvement
    """
    
    def __init__(
        self,
        auto_approve_threshold: float = 0.95,
        auto_reject_threshold: float = 0.15,
        max_human_queue_size: int = 1000,
        priority_function: Optional[Callable] = None,
    ):
        self.auto_approve_threshold = auto_approve_threshold
        self.auto_reject_threshold = auto_reject_threshold
        self.max_queue_size = max_human_queue_size
        self.priority_fn = priority_function or self._default_priority
        
        self.review_queue: List[ReviewItem] = []
        self.metrics = {
            "total_predictions": 0,
            "automated": 0,
            "human_reviewed": 0,
            "rejected": 0,
            "escalated": 0,
        }
    
    def route(
        self,
        prediction: float,
        confidence: float,
        features: Dict,
        explanation: str = "",
        context: Dict = None,
    ) -> Dict:
        """
        Route a prediction to automation or human review.
        
        Returns routing decision and, if automated, the final decision.
        """
        self.metrics["total_predictions"] += 1
        
        # Check for mandatory human review conditions
        if self._requires_mandatory_review(features, context):
            routing = RoutingDecision.HUMAN_DECIDE
            reason = "Mandatory review (regulatory/policy requirement)"
        
        # High confidence positive → automate
        elif confidence >= self.auto_approve_threshold:
            routing = RoutingDecision.AUTOMATE
            reason = f"High confidence ({confidence:.3f} >= {self.auto_approve_threshold})"
        
        # High confidence negative → auto-reject
        elif confidence <= self.auto_reject_threshold:
            routing = RoutingDecision.REJECT
            reason = f"High confidence negative ({confidence:.3f} <= {self.auto_reject_threshold})"
        
        # Uncertain → human review
        else:
            routing = RoutingDecision.HUMAN_REVIEW
            reason = f"Uncertain ({self.auto_reject_threshold} < {confidence:.3f} < {self.auto_approve_threshold})"
        
        # Handle routing
        if routing in (RoutingDecision.HUMAN_REVIEW, RoutingDecision.HUMAN_DECIDE):
            item = self._add_to_queue(prediction, confidence, features, explanation, reason)
            self.metrics["human_reviewed"] += 1
            return {
                "routing": routing.value,
                "reason": reason,
                "queue_item_id": item.item_id,
                "queue_position": len(self.review_queue),
            }
        elif routing == RoutingDecision.AUTOMATE:
            self.metrics["automated"] += 1
            return {
                "routing": routing.value,
                "reason": reason,
                "decision": "approve",
                "confidence": confidence,
            }
        else:  # REJECT
            self.metrics["rejected"] += 1
            return {
                "routing": routing.value,
                "reason": reason,
                "decision": "deny",
                "confidence": confidence,
            }
    
    def _requires_mandatory_review(self, features: Dict, context: Dict) -> bool:
        """Check if case requires mandatory human review."""
        if not context:
            return False
        
        # Examples of mandatory review conditions
        mandatory_conditions = [
            context.get("regulatory_review_required"),
            context.get("amount") and context["amount"] > 100000,  # High-value
            context.get("vulnerable_population"),
            context.get("first_time_scenario"),
        ]
        return any(mandatory_conditions)
    
    def _add_to_queue(
        self,
        prediction: float,
        confidence: float,
        features: Dict,
        explanation: str,
        reason: str,
    ) -> ReviewItem:
        """Add item to human review queue."""
        priority = self.priority_fn(prediction, confidence, features)
        
        item = ReviewItem(
            item_id=str(uuid.uuid4()),
            prediction=prediction,
            confidence=confidence,
            features=features,
            model_explanation=explanation,
            routing_reason=reason,
            priority=priority,
            created_at=datetime.now().isoformat(),
        )
        
        # Insert in priority order
        self.review_queue.append(item)
        self.review_queue.sort(key=lambda x: x.priority)
        
        # Manage queue size
        if len(self.review_queue) > self.max_queue_size:
            # Auto-approve lowest-priority items to manage backlog
            overflow = self.review_queue[self.max_queue_size:]
            self.review_queue = self.review_queue[:self.max_queue_size]
            logger.warning(
                f"Queue overflow: auto-approved {len(overflow)} lowest-priority items"
            )
        
        return item
    
    def _default_priority(
        self, prediction: float, confidence: float, features: Dict
    ) -> int:
        """Default priority: lower confidence = higher priority (review first)."""
        if confidence < 0.3:
            return 1  # Highest priority
        elif confidence < 0.5:
            return 2
        elif confidence < 0.7:
            return 3
        elif confidence < 0.85:
            return 4
        else:
            return 5  # Lowest priority
    
    def get_automation_rate(self) -> float:
        """Get current automation rate (fraction handled without human)."""
        total = self.metrics["total_predictions"] or 1
        automated = self.metrics["automated"] + self.metrics["rejected"]
        return automated / total


class HumanReviewInterface:
    """
    Interface for human reviewers to process queue items.
    
    Design principles:
    - Show model prediction AND explanation (but don't anchor on it)
    - Show relevant context to make informed decision
    - Collect structured feedback for model improvement
    - Track reviewer accuracy and agreement
    """
    
    def __init__(self, router: HITLRouter):
        self.router = router
        self.completed_reviews: List[Dict] = []
    
    def get_next_item(self, reviewer_id: str) -> Optional[ReviewItem]:
        """Get next item from queue for reviewer."""
        for item in self.router.review_queue:
            if item.assigned_to is None:
                item.assigned_to = reviewer_id
                return item
        return None
    
    def submit_review(
        self,
        item_id: str,
        reviewer_id: str,
        decision: str,
        confidence: float,
        reasoning: str = "",
        disagrees_with_model: bool = False,
    ) -> Dict:
        """
        Submit human review decision.
        
        Collects:
        - Decision (approve/deny/escalate)
        - Reviewer confidence
        - Whether reviewer agrees/disagrees with model
        - Reasoning for disagreement (valuable training signal)
        """
        # Find and remove from queue
        item = None
        for i, queued in enumerate(self.router.review_queue):
            if queued.item_id == item_id:
                item = self.router.review_queue.pop(i)
                break
        
        if not item:
            return {"error": "Item not found in queue"}
        
        item.human_decision = decision
        item.decision_time = datetime.now().isoformat()
        
        review_record = {
            "item_id": item_id,
            "model_prediction": item.prediction,
            "model_confidence": item.confidence,
            "human_decision": decision,
            "human_confidence": confidence,
            "agrees_with_model": not disagrees_with_model,
            "reasoning": reasoning,
            "reviewer_id": reviewer_id,
            "review_duration_seconds": self._compute_duration(item),
        }
        
        self.completed_reviews.append(review_record)
        
        return {
            "status": "submitted",
            "decision": decision,
            "record_id": item_id,
        }
    
    def get_reviewer_metrics(self, reviewer_id: str) -> Dict:
        """Get metrics for a specific reviewer."""
        reviewer_reviews = [
            r for r in self.completed_reviews if r["reviewer_id"] == reviewer_id
        ]
        
        if not reviewer_reviews:
            return {"total_reviews": 0}
        
        return {
            "total_reviews": len(reviewer_reviews),
            "avg_review_time_seconds": sum(
                r["review_duration_seconds"] for r in reviewer_reviews
            ) / len(reviewer_reviews),
            "agreement_with_model": sum(
                1 for r in reviewer_reviews if r["agrees_with_model"]
            ) / len(reviewer_reviews),
            "override_rate": sum(
                1 for r in reviewer_reviews if not r["agrees_with_model"]
            ) / len(reviewer_reviews),
        }
    
    def _compute_duration(self, item: ReviewItem) -> float:
        """Compute review duration in seconds."""
        # In production: compute from assignment time to submission time
        return 30.0  # Placeholder


class ActiveLearningLoop:
    """
    Use human decisions to improve the model (active learning).
    
    The human review queue is also a training signal:
    - Cases human reviewed = cases model was uncertain about
    - Human decisions on these cases = high-value labels
    - Retraining on these cases reduces the uncertain zone over time
    """
    
    def __init__(self, review_interface: HumanReviewInterface):
        self.interface = review_interface
        self.training_buffer: List[Dict] = []
        self.buffer_size_threshold = 100  # Retrain when buffer reaches this size
    
    def collect_training_signal(self):
        """
        Collect human decisions as training data.
        
        These are the MOST valuable training examples because they're
        exactly the cases the model struggles with.
        """
        for review in self.interface.completed_reviews:
            training_example = {
                "features": review.get("features", {}),
                "label": 1 if review["human_decision"] == "approve" else 0,
                "weight": self._compute_sample_weight(review),
                "source": "human_review",
            }
            self.training_buffer.append(training_example)
        
        # Clear processed reviews
        self.interface.completed_reviews = []
    
    def should_retrain(self) -> bool:
        """Check if enough training signal has accumulated for retraining."""
        return len(self.training_buffer) >= self.buffer_size_threshold
    
    def get_training_data(self) -> List[Dict]:
        """Get accumulated training data and clear buffer."""
        data = self.training_buffer.copy()
        self.training_buffer = []
        return data
    
    def _compute_sample_weight(self, review: Dict) -> float:
        """
        Weight samples by informativeness.
        
        Higher weight for:
        - Cases where human disagreed with model (corrective signal)
        - Cases with low model confidence (uncertain boundary)
        """
        weight = 1.0
        
        # Higher weight for disagreements
        if not review["agrees_with_model"]:
            weight *= 3.0
        
        # Higher weight for low-confidence predictions
        model_conf = review.get("model_confidence", 0.5)
        uncertainty = 1 - abs(model_conf - 0.5) * 2  # Max at 0.5 confidence
        weight *= (1 + uncertainty)
        
        return weight
```

### HITL Anti-Patterns

```yaml
HITL_Anti_Patterns:
  automation_bias:
    what: "Humans defer to model prediction even when they have contradicting evidence"
    cause: "Model presented as authoritative, human effort to override is high"
    solution:
      - "Don't show model prediction first (let human form opinion, then compare)"
      - "Randomize model display (sometimes show, sometimes don't)"
      - "Track override rate — too low = rubber-stamping"
      - "Train reviewers on cases where model was wrong"
      
  alert_fatigue:
    what: "Too many cases routed to humans — they stop paying attention"
    cause: "Thresholds too wide, model calibration poor, too many edge cases"
    solution:
      - "Tighten thresholds (accept more automation risk)"
      - "Prioritize queue (only show highest-impact cases first)"
      - "Batch similar cases for faster processing"
      - "Set daily review limits per person"
      
  rubber_stamping:
    what: "Human approves model prediction without genuine review"
    cause: "High volume, time pressure, trust in model, interface doesn't encourage review"
    solution:
      - "Track review time (too fast = likely not reviewing)"
      - "Insert known-wrong predictions to detect inattention (trap items)"
      - "Require specific justification for agreement (not just 'approve')"
      - "Show cases where model was wrong to maintain vigilance"
      
  feedback_not_used:
    what: "Human decisions collected but never fed back to model"
    cause: "No active learning pipeline, engineering effort to retrain"
    solution:
      - "Automated active learning loop (retrain on human decisions)"
      - "Track if model accuracy improves over time (validates HITL value)"
      - "Monitor the uncertain zone — should shrink over time"
```

---

## How It Works in Practice

### HITL Lending System

```yaml
HITL_Lending_System:
  context: "Auto-loan approval with human oversight for uncertain cases"
  
  routing_thresholds:
    auto_approve: "Model confidence > 0.92 AND loan amount < $50K"
    auto_deny: "Model confidence < 0.15 (very high default risk)"
    human_review: "Everything else (0.15 - 0.92 confidence)"
    mandatory_review: "Loan > $100K, OR first-time applicant, OR regulatory flag"
    
  volume:
    total_daily: "5,000 applications"
    auto_approved: "3,200 (64%)"
    auto_denied: "400 (8%)"
    human_review: "1,400 (28%)"
    
  reviewer_interface:
    shows:
      - "Model prediction and confidence score"
      - "Top 5 factors (SHAP explanation)"
      - "Applicant information (income, employment, credit history)"
      - "Similar past applications and their outcomes"
      - "Counterfactual: what would change the model's prediction"
    
    doesn't_show_first:
      - "Model recommendation displayed AFTER reviewer forms initial opinion"
      - "Prevents automation bias (anchoring on model's answer)"
      
  feedback_loop:
    - "Human decisions collected daily"
    - "Model retrained monthly on human-reviewed cases"
    - "Uncertain zone has shrunk 15% in 6 months (fewer cases need human review)"
    - "Human review time: average 4 minutes per application"
    
  quality_controls:
    - "5% of auto-approved checked by human (spot-check)"
    - "10% of queue items are 'trap items' (known answer) to detect rubber-stamping"
    - "Inter-reviewer agreement tracked (if <80%, calibration session needed)"
    - "Override rate monitored (if <5%, likely automation bias)"
```

---

## Interview Tip

> When asked about human-in-the-loop: "I design HITL systems around three questions: WHEN to involve humans (routing), HOW to present information (interface), and WHAT to do with human decisions (feedback loop). Routing: confidence-based thresholds — auto-approve above 0.92, auto-deny below 0.15, human review in between. Critical requirement: the model must be calibrated (80% confidence = 80% correct) — otherwise thresholds are meaningless. Interface: I deliberately don't show the model's recommendation first. Research shows this causes automation bias — humans anchor on the model's answer and rubber-stamp. Instead: show features and context first, let the human form an opinion, then reveal model prediction for comparison. This dramatically increases genuine review quality. Feedback loop: every human decision is training signal. Cases routed to humans are exactly the cases the model is uncertain about — the most valuable training examples. I implement active learning: human decisions are collected, weighted by disagreement with model (3x weight), and used for monthly retraining. Over time, the uncertain zone shrinks and automation rate increases. Anti-patterns I guard against: automation bias (track override rate — should be 10-20%, not 2%), alert fatigue (cap daily review volume per person), rubber-stamping (trap items with known-wrong predictions), and feedback not used (active learning pipeline must be automated). The measure of a good HITL system: outcomes with human involvement are measurably better than full automation, AND the automation rate increases over time as the model improves."

---

## Common Mistakes

1. **Showing model prediction first** — Displaying "Model recommends: APPROVE (95%)" before the human reviews the case. This anchors the human on the model's answer (automation bias). Solution: show context/features first, let human form independent opinion, then reveal model prediction.

2. **Not calibrating the model** — Using confidence thresholds on uncalibrated model. Model says 80% confident but is only correct 60% of the time. Threshold-based routing is meaningless. Solution: calibrate model (Platt scaling, isotonic regression) and verify calibration regularly.

3. **Unsustainable human workload** — Routing 60% of cases to human review. Queue grows infinitely, quality drops, reviewers burn out. Solution: set thresholds for manageable workload (target <30% human review), use active learning to reduce over time.

4. **No quality metrics on human review** — Assuming human review is always correct. Humans make errors, especially under time pressure or with automation bias. Solution: track inter-reviewer agreement, use trap items to detect inattention, measure human accuracy against ground truth.

5. **HITL as checkbox compliance** — Adding human review because regulation requires it, but designing it so humans rubber-stamp without genuine review. Solution: measure override rate (should be 10-20%), track review time (too fast = not reviewing), and design interface to support genuine decision-making.

---

## Key Takeaways

- Route by confidence: high → automate, medium → human review, low → reject/escalate
- Calibration required: thresholds only work if model confidence is calibrated (80% = 80% correct)
- Automation bias: don't show model prediction first — let human form independent opinion
- Alert fatigue: cap human workload (<30% of volume), prioritize by impact and uncertainty
- Active learning: human decisions are high-value training data (model was uncertain on these)
- Trap items: insert known-answer cases to detect rubber-stamping (quality assurance)
- Override rate: monitor — too low (<5%) = automation bias, too high (>50%) = model issues
- Feedback loop: human decisions retrain model → uncertain zone shrinks → automation increases
- Regulatory: EU AI Act requires human oversight for high-risk AI systems
- Design for effectiveness: HITL should measurably improve outcomes vs. full automation
