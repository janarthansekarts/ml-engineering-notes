# Fairness Testing

## The Problem / Why This Matters

ML models can perpetuate, amplify, or create discrimination — even without intentional bias. A credit scoring model trained on historical data learns that zip code correlates with default risk, but zip code is a proxy for race due to historical redlining. A hiring model trained on past decisions learns to downrank female candidates because the training data reflects historical gender bias in hiring. A healthcare model allocates fewer resources to Black patients because it uses healthcare spending as a proxy for health need, but spending is lower due to systemic access barriers. Fairness testing systematically evaluates whether an ML model treats different groups equitably. In 2026, this isn't just ethical — it's legal. The EU AI Act requires fairness assessments for high-risk AI systems. The US EEOC (Equal Employment Opportunity Commission) investigates AI-driven hiring discrimination. Financial regulators require adverse impact analysis for automated lending decisions. Fairness testing quantifies: Does the model perform differently for different groups? Does it make decisions at different rates? Are its errors distributed equitably? Which groups bear disproportionate burden?

---

## The Analogy

Think of fairness testing like checking if a teacher grades fairly:

- **No fairness testing** = Teacher gives grades, everyone assumes they're fair because the rubric is the same for everyone. Nobody checks if students from certain neighborhoods consistently get lower grades for equivalent work.
- **Fairness testing** = Auditor reviews grades by student demographics. Discovers: same quality work gets A from suburban students and B from urban students. Same errors penalized 2x more for male students. Teacher isn't consciously biased — the grading rubric implicitly favors certain backgrounds.

The rubric (model) can be technically "the same for everyone" while producing systematically different outcomes for different groups. Fairness testing measures these outcome differences.

---

## Deep Dive

### Fairness Metrics

```yaml
Fairness_Metrics:
  group_fairness:
    demographic_parity:
      definition: "P(positive outcome) is same across groups"
      formula: "P(Ŷ=1 | A=a) = P(Ŷ=1 | A=b) for all groups a, b"
      interpretation: "Same acceptance/approval rate across demographics"
      example: "Loan approval rate: 60% for Group A, 60% for Group B"
      limitation: "Ignores whether groups have different base rates"
      
    equal_opportunity:
      definition: "P(true positive) is same across groups"
      formula: "P(Ŷ=1 | Y=1, A=a) = P(Ŷ=1 | Y=1, A=b)"
      interpretation: "Among qualified individuals, same acceptance rate"
      example: "Among creditworthy applicants: 90% approved regardless of group"
      preference: "Often preferred over demographic parity (rewards qualification)"
      
    predictive_equality:
      definition: "P(false positive) is same across groups"
      formula: "P(Ŷ=1 | Y=0, A=a) = P(Ŷ=1 | Y=0, A=b)"
      interpretation: "Among unqualified individuals, same error rate"
      example: "Among defaulters: 10% incorrectly approved regardless of group"
      
    equalized_odds:
      definition: "Both TPR and FPR are same across groups"
      formula: "Equal opportunity + predictive equality combined"
      interpretation: "Model errors are equally distributed across groups"
      note: "Strictest group fairness metric — hardest to satisfy"
      
    calibration:
      definition: "Predicted probability matches actual probability per group"
      formula: "P(Y=1 | Ŷ=p, A=a) = p for all groups a"
      interpretation: "When model says '70% likely', it IS 70% likely for ALL groups"
      
  individual_fairness:
    definition: "Similar individuals receive similar predictions"
    formula: "d(Ŷ(x), Ŷ(x')) ≤ L * d(x, x') for similar x, x'"
    challenge: "Defining 'similar' (what features matter, which are protected)"
    
  four_fifths_rule:
    definition: "Adverse impact ratio (selection rate of minority ÷ majority) ≥ 0.8"
    origin: "US EEOC Uniform Guidelines on Employee Selection (1978)"
    formula: "min_group_rate / max_group_rate >= 0.8"
    legal_status: "Standard threshold for employment discrimination claims"
    
  impossibility_theorem:
    what: "Cannot satisfy all fairness metrics simultaneously (except trivially)"
    implication: "Must CHOOSE which fairness definition is appropriate for context"
    guidance: "Choice depends on domain, legal requirements, and ethical priorities"
```

### Fairness Testing Implementation

```python
# Comprehensive fairness testing framework

"""
Systematic fairness evaluation for ML models.
Tests model across multiple fairness metrics and protected attributes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class FairnessResult:
    """Result of a fairness test."""
    metric: str
    protected_attribute: str
    group_values: Dict[str, float]
    disparity: float  # Ratio or difference between groups
    threshold: float  # Threshold for passing
    passed: bool
    details: str


class FairnessTester:
    """
    Comprehensive fairness testing for ML models.
    
    Tests:
    1. Demographic parity (equal acceptance rates)
    2. Equal opportunity (equal TPR)
    3. Predictive equality (equal FPR)
    4. Four-fifths rule (adverse impact)
    5. Calibration per group
    6. Subgroup performance analysis
    """
    
    def __init__(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        protected_attributes: pd.DataFrame,
        prediction_proba: Optional[np.ndarray] = None
    ):
        """
        Args:
            predictions: Model predictions (binary or multiclass)
            labels: Ground truth labels
            protected_attributes: DataFrame with protected attribute columns
            prediction_proba: Prediction probabilities (for calibration tests)
        """
        self.predictions = predictions
        self.labels = labels
        self.protected = protected_attributes
        self.proba = prediction_proba
        self.results: List[FairnessResult] = []
    
    def test_demographic_parity(
        self,
        attribute: str,
        threshold: float = 0.1  # Max allowed difference in selection rate
    ) -> FairnessResult:
        """
        Test demographic parity: P(Ŷ=1) should be same across groups.
        
        Passes if: |P(Ŷ=1|A=a) - P(Ŷ=1|A=b)| < threshold for all pairs.
        """
        groups = self.protected[attribute].unique()
        selection_rates = {}
        
        for group in groups:
            mask = self.protected[attribute] == group
            rate = self.predictions[mask].mean()
            selection_rates[str(group)] = float(rate)
        
        max_rate = max(selection_rates.values())
        min_rate = min(selection_rates.values())
        disparity = max_rate - min_rate
        
        result = FairnessResult(
            metric="demographic_parity",
            protected_attribute=attribute,
            group_values=selection_rates,
            disparity=disparity,
            threshold=threshold,
            passed=disparity < threshold,
            details=f"Selection rates: {selection_rates}. Max disparity: {disparity:.4f}",
        )
        self.results.append(result)
        return result
    
    def test_equal_opportunity(
        self,
        attribute: str,
        threshold: float = 0.1
    ) -> FairnessResult:
        """
        Test equal opportunity: TPR (True Positive Rate) same across groups.
        
        Among positive-label individuals, acceptance rate should be equal.
        """
        groups = self.protected[attribute].unique()
        tpr_values = {}
        
        for group in groups:
            mask = (self.protected[attribute] == group) & (self.labels == 1)
            if mask.sum() > 0:
                tpr = self.predictions[mask].mean()
                tpr_values[str(group)] = float(tpr)
        
        if len(tpr_values) < 2:
            return FairnessResult(
                metric="equal_opportunity",
                protected_attribute=attribute,
                group_values=tpr_values,
                disparity=0.0,
                threshold=threshold,
                passed=True,
                details="Insufficient data for comparison",
            )
        
        max_tpr = max(tpr_values.values())
        min_tpr = min(tpr_values.values())
        disparity = max_tpr - min_tpr
        
        result = FairnessResult(
            metric="equal_opportunity",
            protected_attribute=attribute,
            group_values=tpr_values,
            disparity=disparity,
            threshold=threshold,
            passed=disparity < threshold,
            details=f"TPR by group: {tpr_values}. Max disparity: {disparity:.4f}",
        )
        self.results.append(result)
        return result
    
    def test_predictive_equality(
        self,
        attribute: str,
        threshold: float = 0.1
    ) -> FairnessResult:
        """
        Test predictive equality: FPR (False Positive Rate) same across groups.
        
        Among negative-label individuals, false acceptance rate should be equal.
        """
        groups = self.protected[attribute].unique()
        fpr_values = {}
        
        for group in groups:
            mask = (self.protected[attribute] == group) & (self.labels == 0)
            if mask.sum() > 0:
                fpr = self.predictions[mask].mean()
                fpr_values[str(group)] = float(fpr)
        
        max_fpr = max(fpr_values.values()) if fpr_values else 0
        min_fpr = min(fpr_values.values()) if fpr_values else 0
        disparity = max_fpr - min_fpr
        
        result = FairnessResult(
            metric="predictive_equality",
            protected_attribute=attribute,
            group_values=fpr_values,
            disparity=disparity,
            threshold=threshold,
            passed=disparity < threshold,
            details=f"FPR by group: {fpr_values}. Max disparity: {disparity:.4f}",
        )
        self.results.append(result)
        return result
    
    def test_four_fifths_rule(
        self,
        attribute: str
    ) -> FairnessResult:
        """
        Test four-fifths (80%) rule for adverse impact.
        
        Selection rate of any group must be >= 80% of the highest group.
        Legal standard for employment discrimination in the US.
        """
        groups = self.protected[attribute].unique()
        selection_rates = {}
        
        for group in groups:
            mask = self.protected[attribute] == group
            rate = self.predictions[mask].mean()
            selection_rates[str(group)] = float(rate)
        
        max_rate = max(selection_rates.values())
        
        # Adverse impact ratio for each group
        impact_ratios = {
            group: rate / max_rate if max_rate > 0 else 1.0
            for group, rate in selection_rates.items()
        }
        
        min_ratio = min(impact_ratios.values())
        worst_group = min(impact_ratios, key=impact_ratios.get)
        
        result = FairnessResult(
            metric="four_fifths_rule",
            protected_attribute=attribute,
            group_values=impact_ratios,
            disparity=1.0 - min_ratio,  # How far from passing
            threshold=0.8,
            passed=min_ratio >= 0.8,
            details=f"Impact ratios: {impact_ratios}. Worst group: {worst_group} ({min_ratio:.3f})",
        )
        self.results.append(result)
        return result
    
    def test_subgroup_performance(
        self,
        attribute: str,
        metric_fn,
        metric_name: str = "accuracy",
        min_performance: float = 0.8
    ) -> FairnessResult:
        """
        Test that no subgroup has performance below minimum threshold.
        
        Even if overall performance is high, individual subgroups
        must meet minimum quality bar.
        """
        groups = self.protected[attribute].unique()
        group_performance = {}
        
        for group in groups:
            mask = self.protected[attribute] == group
            if mask.sum() > 10:  # Minimum sample size
                score = metric_fn(self.labels[mask], self.predictions[mask])
                group_performance[str(group)] = float(score)
        
        min_perf = min(group_performance.values()) if group_performance else 0
        worst_group = min(group_performance, key=group_performance.get) if group_performance else "N/A"
        
        result = FairnessResult(
            metric=f"subgroup_{metric_name}",
            protected_attribute=attribute,
            group_values=group_performance,
            disparity=max(group_performance.values()) - min_perf if group_performance else 0,
            threshold=min_performance,
            passed=min_perf >= min_performance,
            details=f"Per-group {metric_name}: {group_performance}. Worst: {worst_group} ({min_perf:.4f})",
        )
        self.results.append(result)
        return result
    
    def run_full_suite(
        self,
        attributes: List[str],
        blocking_metrics: List[str] = None
    ) -> Dict:
        """
        Run complete fairness test suite across all protected attributes.
        
        Returns comprehensive report with pass/fail per metric per attribute.
        """
        if blocking_metrics is None:
            blocking_metrics = ["four_fifths_rule", "equal_opportunity"]
        
        all_results = []
        
        for attr in attributes:
            all_results.append(self.test_demographic_parity(attr))
            all_results.append(self.test_equal_opportunity(attr))
            all_results.append(self.test_predictive_equality(attr))
            all_results.append(self.test_four_fifths_rule(attr))
            
            from sklearn.metrics import accuracy_score
            all_results.append(self.test_subgroup_performance(
                attr, accuracy_score, "accuracy", 0.80
            ))
        
        # Determine if deployment should be blocked
        blocking_failures = [
            r for r in all_results
            if r.metric in blocking_metrics and not r.passed
        ]
        
        return {
            "total_tests": len(all_results),
            "passed": sum(1 for r in all_results if r.passed),
            "failed": sum(1 for r in all_results if not r.passed),
            "blocking_failures": [
                {"metric": r.metric, "attribute": r.protected_attribute, "details": r.details}
                for r in blocking_failures
            ],
            "can_deploy": len(blocking_failures) == 0,
            "all_results": all_results,
        }
```

### Counterfactual Fairness Testing

```python
# Counterfactual testing: what if a protected attribute were different?

"""
Counterfactual fairness: if we changed ONLY the protected attribute
(e.g., gender), would the prediction change?

If yes → model is using the protected attribute (directly or via proxies).
"""


class CounterfactualTester:
    """
    Test counterfactual fairness by flipping protected attributes.
    
    For each example:
    1. Get prediction with original attributes
    2. Flip protected attribute (male → female, etc.)
    3. Get prediction with flipped attributes
    4. If predictions differ → model is sensitive to protected attribute
    """
    
    def __init__(self, model_predict):
        self.predict = model_predict
    
    def test_counterfactual(
        self,
        examples: List[Dict],
        protected_attribute: str,
        counterfactual_values: Dict  # Original → counterfactual mapping
    ) -> Dict:
        """
        Test if flipping protected attribute changes predictions.
        
        Example:
            protected_attribute: "gender"
            counterfactual_values: {"male": "female", "female": "male"}
        """
        changes = 0
        total = 0
        change_details = []
        
        for example in examples:
            original_value = example.get(protected_attribute)
            if original_value not in counterfactual_values:
                continue
            
            # Original prediction
            original_pred = self.predict(example)
            
            # Counterfactual prediction (flip only protected attribute)
            counterfactual = {**example}
            counterfactual[protected_attribute] = counterfactual_values[original_value]
            counterfactual_pred = self.predict(counterfactual)
            
            total += 1
            if original_pred != counterfactual_pred:
                changes += 1
                change_details.append({
                    "original": {protected_attribute: original_value, "prediction": original_pred},
                    "counterfactual": {protected_attribute: counterfactual_values[original_value], "prediction": counterfactual_pred},
                })
        
        change_rate = changes / total if total > 0 else 0
        
        return {
            "attribute": protected_attribute,
            "total_tested": total,
            "predictions_changed": changes,
            "change_rate": change_rate,
            "passed": change_rate < 0.05,  # Less than 5% of predictions change
            "threshold": 0.05,
            "details": change_details[:10],  # First 10 examples that changed
        }
    
    def test_proxy_detection(
        self,
        examples: List[Dict],
        protected_attribute: str,
        proxy_candidates: List[str]
    ) -> Dict:
        """
        Detect if non-protected features serve as proxies.
        
        Remove protected attribute → see if proxy features encode same info.
        If removing proxy also changes prediction → it's a proxy.
        """
        proxy_results = {}
        
        for proxy in proxy_candidates:
            # Correlation between proxy and protected attribute
            proxy_values = [e.get(proxy) for e in examples]
            protected_values = [e.get(protected_attribute) for e in examples]
            
            # For categorical: mutual information or chi-squared
            # For numeric: correlation
            correlation = self._compute_association(proxy_values, protected_values)
            
            proxy_results[proxy] = {
                "association_with_protected": correlation,
                "likely_proxy": correlation > 0.3,  # Threshold for proxy detection
            }
        
        return {
            "protected_attribute": protected_attribute,
            "proxy_analysis": proxy_results,
            "likely_proxies": [p for p, v in proxy_results.items() if v["likely_proxy"]],
        }
    
    def _compute_association(self, values_a, values_b) -> float:
        """Compute association between two sets of values."""
        # Simplified — in practice use mutual information or Cramér's V
        try:
            from sklearn.metrics import mutual_info_score
            return mutual_info_score(values_a, values_b)
        except:
            return 0.0
```

### Intersectional Fairness

```yaml
Intersectional_Fairness:
  what: "Test fairness at intersections of protected attributes (not just individually)"
  why: "Model may be fair for gender AND race individually, but biased against Black women specifically"
  
  example:
    individual_test:
      gender_fair: "Male approval: 65%, Female approval: 63% → PASSES"
      race_fair: "White approval: 64%, Black approval: 62% → PASSES"
    intersectional_test:
      white_male: "68%"
      white_female: "65%"
      black_male: "62%"
      black_female: "52%"  # ← FAILS (hidden by individual tests)
      
  implementation:
    approach: "Test all combinations of protected attributes"
    challenge: "Small sample sizes in intersectional groups"
    minimum_samples: "At least 50 examples per intersectional group for meaningful testing"
    reporting: "Flag ANY intersectional group with performance > 5% below overall"
```

---

## How It Works in Practice

### Fairness Testing Pipeline

```yaml
Pipeline:
  pre_training:
    - "Audit training data for historical bias (label distribution per group)"
    - "Check for underrepresentation (minimum samples per group)"
    - "Identify proxy variables (correlated with protected attributes)"
    
  post_training:
    - "Run full fairness test suite (all metrics × all protected attributes)"
    - "Test intersectional fairness (combinations of attributes)"
    - "Run counterfactual tests (flip protected attribute → same prediction?)"
    - "Generate fairness report (Model Card format)"
    
  deployment_gate:
    - "Four-fifths rule: MUST pass (legal requirement)"
    - "Equal opportunity: MUST pass (critical for high-risk systems)"
    - "Subgroup performance: no group < 80% of overall performance"
    - "Counterfactual: < 5% of predictions change when flipping protected attribute"
    
  monitoring:
    - "Track per-group metrics daily in production"
    - "Alert if any group metric degrades (even if overall is stable)"
    - "Quarterly fairness audit (manual review + extended testing)"
```

---

## Interview Tip

> When asked about fairness testing: "I test fairness across multiple metrics because no single metric captures all aspects of fairness (the impossibility theorem proves you can't satisfy all simultaneously). My standard suite: (1) Four-fifths rule — the legal standard: selection rate of any group must be ≥ 80% of the highest group's rate. Required for employment and lending. (2) Equal opportunity — among qualified individuals (positive labels), acceptance rate should be equal across groups. I prefer this over demographic parity because it respects qualification differences while ensuring equal treatment of equally-qualified people. (3) Predictive equality — false positive rates equal across groups. Important because false positives have different costs (false arrest, denied service). (4) Subgroup performance — no demographic group should have accuracy/F1 below a minimum threshold. Overall accuracy can hide poor performance for minorities. (5) Counterfactual testing — flip only the protected attribute and check if prediction changes. If it does, the model is directly or indirectly using that attribute. I also test intersectional fairness (race × gender combinations) because individual attribute tests can hide intersectional bias. Key implementation detail: I make four-fifths rule and equal opportunity BLOCKING for deployment in high-risk domains (credit, hiring, healthcare). Other metrics are warnings that trigger human review."

---

## Common Mistakes

1. **Testing one fairness metric and declaring 'fair'** — Model passes demographic parity (equal selection rates) but has very different false positive rates across groups. Solution: test MULTIPLE metrics (they measure different things). Choose which to prioritize based on domain and legal requirements.

2. **Only testing individual attributes, not intersections** — Fair for gender AND fair for race individually, but biased against the intersection (e.g., Black women). Solution: test intersectional groups — all combinations of protected attributes with sufficient sample size.

3. **Using overall performance as fairness proxy** — "Model is 95% accurate so it must be fair." But accuracy is 98% for majority group and 82% for minority group. Solution: always evaluate per-group performance. Overall metrics hide subgroup disparities.

4. **Testing fairness once at launch, never again** — Model was fair at launch. Data distribution shifts over time. Groups' representation changes. Model becomes unfair in production. Solution: continuous fairness monitoring — track per-group metrics daily, alert on degradation.

5. **Removing protected attributes and assuming fairness** — Remove gender from features. Model uses zip code, name, and shopping habits as proxies for gender — still biased. Solution: removing features doesn't remove bias. Test with counterfactual analysis and proxy detection.

---

## Key Takeaways

- Fairness testing measures whether ML models treat different groups equitably
- Multiple metrics needed: demographic parity, equal opportunity, predictive equality, four-fifths rule
- Impossibility theorem: can't satisfy all fairness metrics simultaneously — choose based on context
- Four-fifths rule: legal standard for employment/lending (min group rate ≥ 80% of max)
- Equal opportunity: same true positive rate across groups (equal treatment of qualified)
- Counterfactual testing: flip protected attribute, check if prediction changes
- Intersectional fairness: test combinations (race × gender) — individual tests hide compound bias
- Proxy detection: removing protected attributes doesn't remove bias (proxies exist)
- Deployment gate: critical fairness tests BLOCK deployment (not just warnings)
- Continuous monitoring: fairness can degrade over time as data distribution shifts
