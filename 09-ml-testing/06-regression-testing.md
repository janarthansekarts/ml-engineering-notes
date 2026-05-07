# Regression Testing for ML

## The Problem / Why This Matters

In traditional software, regression testing verifies that new changes don't break existing functionality — run the same test suite, verify same outputs. In ML, "regression" has a dual meaning: (1) statistical regression (the model type) and (2) quality regression (new model is worse than old model). Model regression testing verifies that a newly trained model doesn't degrade compared to the currently deployed model. This is critical because ML models are retrained regularly (daily/weekly) and any retrain can introduce regressions — lower accuracy on specific segments, new failure modes, or degraded performance on edge cases that the previous model handled correctly. Without systematic regression testing, teams deploy models that silently get worse on subsets of traffic. A model might improve overall accuracy by 0.5% but regress 10% on a critical customer segment. Golden datasets (curated sets of known-correct predictions) are the foundation of ML regression testing — they represent the "contract" of what the model must get right.

---

## The Analogy

Think of ML regression testing like a restaurant maintaining recipe quality when a chef changes:

- **No regression testing** = New chef arrives, changes recipes based on "overall taste." Majority of dishes improve, but three signature dishes (that loyal customers come for) are now different. Customers leave, and no one knows why.
- **Regression testing** = Before the new chef starts, document exactly how each signature dish should taste (golden dataset). After the new chef's changes, taste-test every signature dish against the documented standard. If any regresses, fix before opening.

The golden dataset is your "recipe book" — the documented standard that must be maintained regardless of other improvements.

---

## Deep Dive

### Types of ML Regression

```yaml
ML_Regression_Types:
  overall_regression:
    what: "New model has worse aggregate metrics than current model"
    example: "New model accuracy: 93.2%, current: 94.1% → overall regression"
    detection: "Simple: compare overall metric (accuracy, F1, AUC)"
    severity: "High — clear signal, easy to detect, blocks deployment"
    
  segment_regression:
    what: "New model worse on a specific segment, even if overall improved"
    example: "Overall: +0.5%, but age>65 segment: -3.2%"
    detection: "Per-segment evaluation (demographic, geographic, product category)"
    severity: "Critical — hidden by overall metrics, affects real users"
    
  edge_case_regression:
    what: "New model fails on specific known-important cases"
    example: "Fraud model no longer catches pattern that caused $1M loss last year"
    detection: "Golden dataset with curated edge cases"
    severity: "Critical — high-value cases that must always work"
    
  behavioral_regression:
    what: "Model behavior changes in unexpected ways"
    example: "Model now treats 'not bad' as negative (previously neutral)"
    detection: "Behavioral tests (invariance, directional, negation)"
    severity: "Medium — indicates training instability"
    
  latency_regression:
    what: "New model is slower than current model"
    example: "p95 latency: 120ms (was 80ms) after adding new features"
    detection: "Performance benchmark against baseline"
    severity: "Medium-High — affects user experience and cost"
```

### Golden Datasets

```python
# Golden dataset management for regression testing

"""
Golden datasets are curated, labeled examples that represent
critical model behavior. They are the contract of what the model
MUST get right — regardless of other improvements.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import hashlib
from datetime import datetime


@dataclass
class GoldenExample:
    """A single example in the golden dataset."""
    id: str
    input_data: Dict[str, Any]
    expected_output: Any  # Can be class label, score range, or behavior
    category: str  # "edge_case", "critical_customer", "known_failure", etc.
    added_date: str
    reason: str  # Why this example is in the golden set
    tolerance: float = 0.0  # Allowed deviation from expected output
    
    # For classification: exact match or within confidence range
    match_type: str = "exact"  # "exact", "range", "top_k"


@dataclass
class GoldenDataset:
    """
    Curated dataset for regression testing.
    
    Sources of golden examples:
    1. Production incidents (add the failure case to prevent recurrence)
    2. Critical business cases (high-value predictions that must be correct)
    3. Edge cases from manual testing (known difficult inputs)
    4. Fairness-critical examples (predictions that must be equitable)
    5. Regulatory requirements (cases that must meet legal standards)
    """
    name: str
    version: str
    model_name: str
    examples: List[GoldenExample] = field(default_factory=list)
    created_date: str = ""
    
    def add_example(
        self,
        input_data: Dict,
        expected_output: Any,
        category: str,
        reason: str,
        **kwargs
    ):
        """Add a new golden example."""
        example_id = hashlib.md5(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()[:12]
        
        self.examples.append(GoldenExample(
            id=example_id,
            input_data=input_data,
            expected_output=expected_output,
            category=category,
            added_date=datetime.now().isoformat(),
            reason=reason,
            **kwargs
        ))
    
    def run_regression_test(self, model) -> Dict:
        """
        Run the golden dataset against a model.
        
        Returns detailed results:
        - Overall pass rate
        - Per-category pass rates
        - Specific failures with details
        """
        results = []
        
        for example in self.examples:
            prediction = model.predict(example.input_data)
            passed = self._check_match(prediction, example)
            
            results.append({
                "id": example.id,
                "category": example.category,
                "passed": passed,
                "expected": example.expected_output,
                "actual": prediction,
                "reason": example.reason,
            })
        
        # Aggregate results
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        failures = [r for r in results if not r["passed"]]
        
        # Per-category results
        categories = set(e.category for e in self.examples)
        per_category = {}
        for cat in categories:
            cat_results = [r for r in results if r["category"] == cat]
            cat_passed = sum(1 for r in cat_results if r["passed"])
            per_category[cat] = {
                "total": len(cat_results),
                "passed": cat_passed,
                "pass_rate": cat_passed / len(cat_results) if cat_results else 0,
            }
        
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "per_category": per_category,
            "failures": failures,
            "can_deploy": len(failures) == 0,  # Zero tolerance on golden set
        }
    
    def _check_match(self, prediction, example: GoldenExample) -> bool:
        """Check if prediction matches expected output."""
        if example.match_type == "exact":
            return prediction == example.expected_output
        elif example.match_type == "range":
            low, high = example.expected_output
            return low <= prediction <= high
        elif example.match_type == "top_k":
            # Expected output is in top-k predictions
            return example.expected_output in prediction[:example.tolerance]
        return False


# Example: Building a golden dataset from production incidents
def build_golden_dataset_from_incidents():
    """
    Create golden dataset from historical production incidents.
    Each incident becomes a test case to prevent recurrence.
    """
    golden = GoldenDataset(
        name="fraud_detection_golden_v3",
        version="3.0",
        model_name="fraud_detector",
        created_date="2026-01-15",
    )
    
    # Incident 1: Model missed $500K fraud pattern
    golden.add_example(
        input_data={
            "transaction_amount": 499999,
            "transaction_count_1h": 1,
            "merchant_category": "wire_transfer",
            "account_age_days": 3,
            "previous_max_transaction": 500,
        },
        expected_output="fraud",
        category="critical_incident",
        reason="INC-2025-1234: Model missed $500K fraud. New account, massive wire transfer.",
    )
    
    # Incident 2: False positive on legitimate high-value customer
    golden.add_example(
        input_data={
            "transaction_amount": 250000,
            "transaction_count_1h": 1,
            "merchant_category": "real_estate",
            "account_age_days": 3650,  # 10 years
            "previous_max_transaction": 200000,
        },
        expected_output="legitimate",
        category="false_positive_prevention",
        reason="INC-2025-5678: High-value customer blocked. 10-year account, normal for them.",
    )
    
    # Edge case: Exact threshold amount
    golden.add_example(
        input_data={
            "transaction_amount": 10000,  # Reporting threshold
            "transaction_count_1h": 3,
            "merchant_category": "cash_advance",
            "account_age_days": 30,
            "previous_max_transaction": 9999,  # Structuring pattern
        },
        expected_output="fraud",
        category="edge_case",
        reason="Structuring: multiple transactions just below reporting threshold.",
    )
    
    return golden
```

### Model Comparison Testing

```python
# Compare new model against current production model

"""
A/B model comparison before deployment.
New model must be better overall AND not regress on any critical segment.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


class ModelComparison:
    """
    Compare candidate model against current production model.
    
    Decision criteria:
    1. Overall metrics: candidate >= current (with statistical significance)
    2. Per-segment: no segment regresses > threshold
    3. Golden dataset: 100% pass rate
    4. Performance: latency within budget
    """
    
    def __init__(
        self,
        current_model,
        candidate_model,
        test_data: pd.DataFrame,
        label_column: str = "label"
    ):
        self.current = current_model
        self.candidate = candidate_model
        self.test_data = test_data
        self.label_col = label_column
        
        # Generate predictions from both models
        features = test_data.drop(columns=[label_column])
        self.current_preds = current_model.predict(features)
        self.candidate_preds = candidate_model.predict(features)
        self.labels = test_data[label_column].values
    
    def overall_comparison(self) -> Dict:
        """Compare overall metrics between models."""
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        metrics = {}
        for name, metric_fn in [
            ("accuracy", accuracy_score),
            ("f1_weighted", lambda y, p: f1_score(y, p, average="weighted")),
        ]:
            current_score = metric_fn(self.labels, self.current_preds)
            candidate_score = metric_fn(self.labels, self.candidate_preds)
            
            metrics[name] = {
                "current": current_score,
                "candidate": candidate_score,
                "delta": candidate_score - current_score,
                "improved": candidate_score > current_score,
            }
        
        return metrics
    
    def segment_comparison(
        self,
        segment_column: str,
        max_regression: float = 0.02  # Max 2% regression per segment
    ) -> Dict:
        """
        Compare metrics per segment.
        Flag any segment with regression > threshold.
        """
        from sklearn.metrics import accuracy_score
        
        segments = self.test_data[segment_column].unique()
        results = {}
        regressions = []
        
        for segment in segments:
            mask = self.test_data[segment_column] == segment
            segment_labels = self.labels[mask]
            
            current_score = accuracy_score(segment_labels, self.current_preds[mask])
            candidate_score = accuracy_score(segment_labels, self.candidate_preds[mask])
            
            delta = candidate_score - current_score
            regressed = delta < -max_regression
            
            results[str(segment)] = {
                "current": current_score,
                "candidate": candidate_score,
                "delta": delta,
                "regressed": regressed,
                "n_samples": int(mask.sum()),
            }
            
            if regressed:
                regressions.append({
                    "segment": segment,
                    "regression": abs(delta),
                    "current": current_score,
                    "candidate": candidate_score,
                })
        
        return {
            "segments": results,
            "regressions": regressions,
            "any_regression": len(regressions) > 0,
        }
    
    def statistical_significance(
        self,
        metric: str = "accuracy",
        alpha: float = 0.05
    ) -> Dict:
        """
        Test if candidate improvement is statistically significant.
        Uses McNemar's test for paired binary classification.
        """
        # McNemar's test: compare disagreements between models
        current_correct = (self.current_preds == self.labels)
        candidate_correct = (self.candidate_preds == self.labels)
        
        # Contingency table
        # both_correct: both right
        # current_only: current right, candidate wrong
        # candidate_only: candidate right, current wrong
        # both_wrong: both wrong
        
        current_only = np.sum(current_correct & ~candidate_correct)
        candidate_only = np.sum(~current_correct & candidate_correct)
        
        # McNemar's test (with continuity correction)
        if current_only + candidate_only == 0:
            p_value = 1.0  # No disagreements
        else:
            chi2 = (abs(candidate_only - current_only) - 1) ** 2 / (candidate_only + current_only)
            p_value = 1 - stats.chi2.cdf(chi2, df=1)
        
        return {
            "test": "mcnemar",
            "current_only_correct": int(current_only),
            "candidate_only_correct": int(candidate_only),
            "p_value": p_value,
            "significant": p_value < alpha,
            "candidate_better": candidate_only > current_only,
            "conclusion": (
                "Candidate significantly better" if (p_value < alpha and candidate_only > current_only)
                else "Candidate significantly worse" if (p_value < alpha and current_only > candidate_only)
                else "No significant difference"
            ),
        }
    
    def deployment_decision(
        self,
        golden_results: Dict,
        segment_column: str = None,
    ) -> Dict:
        """
        Make deployment decision based on all criteria.
        
        Criteria:
        1. Golden dataset: 100% pass (HARD REQUIREMENT)
        2. Overall metrics: not worse (soft — statistical significance required)
        3. Segments: no segment regresses > 2% (HARD REQUIREMENT)
        """
        decision = {
            "criteria": [],
            "can_deploy": True,
        }
        
        # Criterion 1: Golden dataset
        golden_pass = golden_results.get("can_deploy", False)
        decision["criteria"].append({
            "name": "golden_dataset",
            "passed": golden_pass,
            "blocking": True,
            "details": f"Pass rate: {golden_results.get('pass_rate', 0):.1%}",
        })
        if not golden_pass:
            decision["can_deploy"] = False
        
        # Criterion 2: Overall metrics
        overall = self.overall_comparison()
        overall_pass = all(m["candidate"] >= m["current"] - 0.001 for m in overall.values())
        decision["criteria"].append({
            "name": "overall_metrics",
            "passed": overall_pass,
            "blocking": True,
            "details": overall,
        })
        if not overall_pass:
            decision["can_deploy"] = False
        
        # Criterion 3: Segment regression
        if segment_column:
            segments = self.segment_comparison(segment_column)
            segment_pass = not segments["any_regression"]
            decision["criteria"].append({
                "name": "segment_regression",
                "passed": segment_pass,
                "blocking": True,
                "details": segments.get("regressions", []),
            })
            if not segment_pass:
                decision["can_deploy"] = False
        
        return decision
```

### Continuous Regression Monitoring

```yaml
Continuous_Monitoring:
  what: "Detect regressions in production (not just pre-deployment)"
  
  approach:
    baseline_window:
      what: "Rolling window of recent model performance (e.g., last 7 days)"
      metric: "Daily accuracy, daily per-segment accuracy"
      
    regression_detection:
      method_1: "Compare today's metric to 7-day rolling average"
      method_2: "Statistical process control (SPC) charts"
      method_3: "CUSUM (Cumulative Sum) for detecting sustained degradation"
      
    alert_thresholds:
      warning: "Metric drops > 1 standard deviation below baseline"
      critical: "Metric drops > 2 standard deviations OR below hard threshold"
      
  automated_response:
    on_warning: "Alert team, increase monitoring frequency"
    on_critical: "Alert + auto-rollback to previous model version"
```

---

## How It Works in Practice

### Regression Testing in CI/CD

```yaml
CI_CD_Flow:
  model_retrain_triggered:
    step_1: "Train new model on updated data"
    step_2: "Run golden dataset test → must be 100% pass"
    step_3: "Run overall metric comparison → must not regress"
    step_4: "Run per-segment comparison → no segment regresses > 2%"
    step_5: "Run statistical significance test → must be significant if different"
    step_6: "Generate comparison report (visual diff: scatter plot, confusion matrix diff)"
    step_7: "If all pass → deploy as canary (5% traffic)"
    step_8: "Monitor canary for 24h → if stable, promote to 100%"
    
  on_golden_failure:
    action: "BLOCK deployment"
    investigation: "Which examples failed? Why? Is the training data missing this pattern?"
    resolution: "Fix training data/features, retrain, re-test"
    
  on_segment_regression:
    action: "BLOCK deployment"
    investigation: "Which segment regressed? What changed in their data?"
    resolution: "Add segment-specific examples to training, or use segment-specific model"
```

---

## Interview Tip

> When asked about ML regression testing: "I use three layers of regression testing to prevent model quality degradation: (1) Golden dataset testing — a curated set of 100-500 critical examples that the model MUST get right: past production incidents (prevent recurrence), high-value business cases, known edge cases, and fairness-critical examples. Zero tolerance — any failure blocks deployment. I add new examples whenever we discover a production issue. (2) Model comparison testing — compare candidate model vs. current production model on held-out test data. Overall metrics must not regress (with statistical significance via McNemar's test), AND no segment/subgroup can regress more than 2% even if overall improves. This catches the 'improved average but degraded minority' pattern. (3) Continuous regression monitoring — in production, track daily metrics per-segment against a rolling baseline. Use statistical process control (CUSUM or SPC charts) to detect sustained degradation early. Auto-rollback on critical regression. The key insight: overall accuracy improvement can mask segment-level regression. A model that's +0.5% overall but -5% for elderly users is a regression, not an improvement."

---

## Common Mistakes

1. **Only comparing overall accuracy** — New model: 94.5% (up from 94.0%). Ship it! But performance for users age 65+ dropped from 91% to 82%. Solution: always evaluate per-segment alongside overall. Block if any critical segment regresses.

2. **Golden dataset too small or not maintained** — Created 20 golden examples at launch, never added more. Doesn't cover patterns discovered in 2 years of production. Solution: add to golden dataset after every production incident. Review quarterly. Target 200-500 examples.

3. **No statistical significance testing** — New model is 0.1% better on test set. Could be noise. Deploy anyway. Later: it's actually the same (or worse). Solution: McNemar's test or bootstrap confidence intervals. Don't deploy on noise.

4. **Testing only at retraining time** — Model passes regression test at deploy time. Three weeks later, data distribution shifts and model degrades gradually. Solution: continuous regression monitoring in production (daily metric tracking, automated alerts on sustained degradation).

5. **Golden dataset contamination** — Golden examples accidentally appear in training data. Model memorizes them. Golden test passes 100% but model hasn't actually learned the pattern. Solution: strict separation. Golden dataset is NEVER included in training data. Verify with data lineage tracking.

---

## Key Takeaways

- Golden dataset: curated examples that model MUST get right — zero tolerance on failures
- Sources: production incidents, critical business cases, edge cases, fairness examples
- Model comparison: candidate must match/beat current model overall AND per-segment
- Statistical significance: use McNemar's test — don't deploy on noise
- Segment regression: overall improvement can mask segment-level degradation
- Continuous monitoring: track metrics per-segment daily, auto-rollback on critical regression
- Golden dataset maintenance: add new examples after every production incident
- Deployment criteria: golden 100% + overall not worse + no segment regresses > 2%
- Never include golden dataset in training data (prevents memorization)
- Regression = deployment blocker, not just a warning
