# Fairness Monitoring

## The Problem / Why This Matters

ML models can systematically discriminate against protected groups — not because engineers intended bias, but because historical data encodes societal inequities, proxy features correlate with protected attributes, and optimization objectives can amplify disparities. A hiring model trained on historical hire data learns to penalize resumes with women's college names. A loan model trained on historical approvals perpetuates redlining patterns. A recidivism model assigns higher risk to Black defendants because historical arrest data reflects policing patterns, not actual crime rates. Fairness monitoring is the continuous measurement of model behavior across demographic groups to detect and quantify discriminatory patterns. In production, this means: (1) tracking performance metrics broken down by protected attributes (race, gender, age, disability, etc.), (2) detecting when model behavior differs significantly across groups, (3) measuring specific fairness metrics (demographic parity, equalized odds, predictive parity), and (4) alerting when disparities exceed acceptable thresholds. Regulations increasingly mandate this: the EU AI Act requires bias testing for high-risk systems, US fair lending laws require adverse impact analysis, and NYC Local Law 144 requires annual bias audits for automated employment decision tools.

---

## The Analogy

Think of fairness monitoring like quality control in a factory that serves diverse customers:

The factory (model) produces products (predictions). If you only measure average quality, you might miss that products for one region consistently have defects while others are perfect. Average looks good; disaggregated data reveals a problem.

Fairness monitoring = inspecting product quality SEPARATELY for each customer group. Not just "overall accuracy is 95%" but "accuracy is 97% for group A, 93% for group B, and 82% for group C." That 82% is unacceptable — and you'd never find it by looking at the overall average.

The tricky part: what constitutes "fair"? Equal accuracy across groups? Equal approval rates? Equal error rates? These definitions often conflict with each other (mathematical impossibility result). You have to choose which fairness criterion matters most for your specific application.

---

## Deep Dive

### Fairness Metrics

```yaml
Fairness_Metrics:
  demographic_parity:
    also_called: "Statistical parity, independence"
    definition: "P(positive prediction | Group A) = P(positive prediction | Group B)"
    meaning: "Each group gets the same rate of positive predictions"
    when_appropriate: "When you want equal outcomes regardless of actual qualification rates"
    limitation: "Ignores actual differences in base rates (may be unfair if groups genuinely differ)"
    example: "Loan approval rate should be similar for men and women"
    
  equalized_odds:
    also_called: "Separation"
    definition: "P(positive prediction | actual positive, Group A) = P(positive prediction | actual positive, Group B) AND same for negatives"
    meaning: "True positive rate AND false positive rate are equal across groups"
    when_appropriate: "When you want the model to be equally accurate for all groups"
    limitation: "Requires ground truth labels (often delayed)"
    example: "Fraud detection should catch fraud equally well regardless of race"
    
  equal_opportunity:
    definition: "P(positive prediction | actual positive, Group A) = P(positive prediction | actual positive, Group B)"
    meaning: "True positive rate is equal across groups (only for positive class)"
    when_appropriate: "When false negatives are the primary concern (missing qualified people)"
    example: "Qualified applicants should have equal chance of being approved regardless of gender"
    
  predictive_parity:
    also_called: "Sufficiency"
    definition: "P(actual positive | positive prediction, Group A) = P(actual positive | positive prediction, Group B)"
    meaning: "Precision is equal across groups (predictions are equally trustworthy)"
    when_appropriate: "When acting on predictions and you want them to mean the same thing for all groups"
    limitation: "May allow different error rates across groups"
    
  calibration:
    definition: "Among all predictions with confidence X, actual positive rate is X for all groups"
    meaning: "Model is equally well-calibrated across groups"
    when_appropriate: "When confidence scores are used for downstream decisions"
    
  treatment_equality:
    definition: "FP/FN ratio is equal across groups"
    meaning: "Errors are distributed the same way (not more false accusations for one group)"
    when_appropriate: "When both types of errors have consequences"
    
  impossible_together:
    fact: "Demographic parity + equalized odds + predictive parity CANNOT all hold simultaneously (unless base rates are equal across groups)"
    implication: "You MUST choose which fairness criterion matters most for your application"
    guidance:
      lending: "Equal opportunity or predictive parity (qualified people approved equally)"
      criminal_justice: "Equalized odds (equal error rates across groups)"
      hiring: "Demographic parity or equal opportunity (depending on legal framework)"
```

### Protected Attributes

```yaml
Protected_Attributes:
  common:
    - "Race/ethnicity"
    - "Gender/sex"
    - "Age"
    - "Religion"
    - "National origin"
    - "Disability status"
    - "Marital status"
    - "Sexual orientation"
    - "Veteran status"
    
  challenges_in_production:
    data_availability:
      problem: "Often don't have protected attributes in production data"
      solutions:
        - "Self-reported demographics (if available from user profiles)"
        - "BISG (Bayesian Improved Surname Geocoding) for race proxy"
        - "Name-based gender inference (imperfect but common)"
      caution: "Imputed demographics add noise — results are estimates"
      
    intersectionality:
      problem: "Bias may affect intersectional groups (Black women) differently than individual groups"
      solution: "Monitor intersectional subgroups, not just individual attributes"
      challenge: "Small sample sizes for intersectional groups → noisy estimates"
      
    proxy_features:
      problem: "ZIP code correlates with race; university name correlates with socioeconomic status"
      implication: "Removing protected attributes doesn't prevent discrimination"
      solution: "Monitor fairness metrics even when protected attributes aren't model inputs"
```

### Implementation

```python
# Fairness monitoring implementation

"""
Production fairness monitoring: compute fairness metrics across groups,
detect disparities, and alert on threshold violations.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class FairnessResult:
    """Result of fairness evaluation for one metric."""
    metric_name: str
    overall_value: float
    group_values: dict  # {group_name: metric_value}
    max_disparity: float  # Maximum difference between any two groups
    disparity_ratio: float  # Min group value / max group value
    passes_threshold: bool
    threshold: float
    flagged_groups: list[str]


class FairnessMonitor:
    """
    Monitor model fairness across protected groups.
    
    Computes multiple fairness metrics, tracks over time,
    and alerts on disparities exceeding thresholds.
    """
    
    def __init__(
        self,
        protected_attribute: str,
        groups: list[str],
        disparity_threshold: float = 0.8,  # 4/5ths rule (80% rule)
        min_group_size: int = 30,
    ):
        """
        Args:
            protected_attribute: Name of the protected attribute (e.g., "gender")
            groups: List of group values (e.g., ["male", "female", "non_binary"])
            disparity_threshold: Minimum acceptable ratio (4/5ths rule = 0.8)
            min_group_size: Minimum samples per group for reliable metrics
        """
        self.attribute = protected_attribute
        self.groups = groups
        self.threshold = disparity_threshold
        self.min_group_size = min_group_size
    
    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: Optional[np.ndarray],
        group_labels: np.ndarray,
    ) -> list[FairnessResult]:
        """
        Compute all fairness metrics across groups.
        
        Args:
            predictions: Model predictions (0/1 for binary, or probabilities)
            ground_truth: Actual labels (None if not available yet)
            group_labels: Protected attribute values for each prediction
        """
        results = []
        
        # Metric 1: Demographic Parity (no ground truth needed)
        results.append(self._demographic_parity(predictions, group_labels))
        
        # Metrics requiring ground truth
        if ground_truth is not None:
            results.append(self._equal_opportunity(predictions, ground_truth, group_labels))
            results.append(self._equalized_odds(predictions, ground_truth, group_labels))
            results.append(self._predictive_parity(predictions, ground_truth, group_labels))
        
        return results
    
    def _demographic_parity(
        self, predictions: np.ndarray, groups: np.ndarray
    ) -> FairnessResult:
        """
        Demographic parity: positive prediction rate should be equal across groups.
        Also known as the "4/5ths rule" (80% rule) in disparate impact analysis.
        """
        group_rates = {}
        
        for group in self.groups:
            mask = groups == group
            if mask.sum() < self.min_group_size:
                continue
            group_preds = predictions[mask]
            # Positive prediction rate
            group_rates[group] = float(np.mean(group_preds > 0.5))
        
        if not group_rates:
            return FairnessResult(
                metric_name="demographic_parity",
                overall_value=float(np.mean(predictions > 0.5)),
                group_values={},
                max_disparity=0,
                disparity_ratio=1.0,
                passes_threshold=True,
                threshold=self.threshold,
                flagged_groups=[],
            )
        
        max_rate = max(group_rates.values())
        min_rate = min(group_rates.values())
        disparity_ratio = min_rate / max_rate if max_rate > 0 else 0
        
        flagged = [g for g, r in group_rates.items() if r < max_rate * self.threshold]
        
        return FairnessResult(
            metric_name="demographic_parity",
            overall_value=float(np.mean(predictions > 0.5)),
            group_values=group_rates,
            max_disparity=max_rate - min_rate,
            disparity_ratio=disparity_ratio,
            passes_threshold=disparity_ratio >= self.threshold,
            threshold=self.threshold,
            flagged_groups=flagged,
        )
    
    def _equal_opportunity(
        self, predictions: np.ndarray, truth: np.ndarray, groups: np.ndarray
    ) -> FairnessResult:
        """
        Equal opportunity: true positive rate should be equal across groups.
        Among actually positive cases, each group should have equal chance of
        being correctly predicted positive.
        """
        group_tpr = {}
        
        for group in self.groups:
            mask = (groups == group) & (truth == 1)
            if mask.sum() < self.min_group_size:
                continue
            group_preds = predictions[mask]
            group_tpr[group] = float(np.mean(group_preds > 0.5))
        
        if not group_tpr:
            return FairnessResult(
                metric_name="equal_opportunity", overall_value=0,
                group_values={}, max_disparity=0, disparity_ratio=1.0,
                passes_threshold=True, threshold=self.threshold, flagged_groups=[],
            )
        
        max_tpr = max(group_tpr.values())
        min_tpr = min(group_tpr.values())
        disparity_ratio = min_tpr / max_tpr if max_tpr > 0 else 0
        
        flagged = [g for g, r in group_tpr.items() if r < max_tpr * self.threshold]
        
        return FairnessResult(
            metric_name="equal_opportunity",
            overall_value=float(np.mean(predictions[truth == 1] > 0.5)),
            group_values=group_tpr,
            max_disparity=max_tpr - min_tpr,
            disparity_ratio=disparity_ratio,
            passes_threshold=disparity_ratio >= self.threshold,
            threshold=self.threshold,
            flagged_groups=flagged,
        )
    
    def _equalized_odds(
        self, predictions: np.ndarray, truth: np.ndarray, groups: np.ndarray
    ) -> FairnessResult:
        """Equalized odds: both TPR and FPR equal across groups."""
        group_metrics = {}
        
        for group in self.groups:
            group_mask = groups == group
            if group_mask.sum() < self.min_group_size:
                continue
            
            # TPR (True Positive Rate)
            pos_mask = group_mask & (truth == 1)
            tpr = float(np.mean(predictions[pos_mask] > 0.5)) if pos_mask.sum() > 0 else 0
            
            # FPR (False Positive Rate)
            neg_mask = group_mask & (truth == 0)
            fpr = float(np.mean(predictions[neg_mask] > 0.5)) if neg_mask.sum() > 0 else 0
            
            group_metrics[group] = {"tpr": tpr, "fpr": fpr}
        
        # Compute max disparity across both TPR and FPR
        tprs = [m["tpr"] for m in group_metrics.values()]
        fprs = [m["fpr"] for m in group_metrics.values()]
        
        tpr_disparity = max(tprs) - min(tprs) if tprs else 0
        fpr_disparity = max(fprs) - min(fprs) if fprs else 0
        max_disparity = max(tpr_disparity, fpr_disparity)
        
        return FairnessResult(
            metric_name="equalized_odds",
            overall_value=0,
            group_values=group_metrics,
            max_disparity=max_disparity,
            disparity_ratio=1.0 - max_disparity,  # Simplified
            passes_threshold=max_disparity < (1 - self.threshold),
            threshold=self.threshold,
            flagged_groups=[g for g, m in group_metrics.items() 
                          if abs(m["tpr"] - np.mean(tprs)) > 0.1],
        )
    
    def _predictive_parity(
        self, predictions: np.ndarray, truth: np.ndarray, groups: np.ndarray
    ) -> FairnessResult:
        """Predictive parity: precision equal across groups."""
        group_precision = {}
        
        for group in self.groups:
            mask = (groups == group) & (predictions > 0.5)
            if mask.sum() < self.min_group_size:
                continue
            group_precision[group] = float(np.mean(truth[mask] == 1))
        
        if not group_precision:
            return FairnessResult(
                metric_name="predictive_parity", overall_value=0,
                group_values={}, max_disparity=0, disparity_ratio=1.0,
                passes_threshold=True, threshold=self.threshold, flagged_groups=[],
            )
        
        max_prec = max(group_precision.values())
        min_prec = min(group_precision.values())
        disparity_ratio = min_prec / max_prec if max_prec > 0 else 0
        
        return FairnessResult(
            metric_name="predictive_parity",
            overall_value=float(np.mean(truth[predictions > 0.5] == 1)),
            group_values=group_precision,
            max_disparity=max_prec - min_prec,
            disparity_ratio=disparity_ratio,
            passes_threshold=disparity_ratio >= self.threshold,
            threshold=self.threshold,
            flagged_groups=[g for g, p in group_precision.items() if p < max_prec * self.threshold],
        )
```

### Fairness Monitoring Dashboard

```yaml
Dashboard:
  overview_panel:
    - "Overall fairness status (PASS / WARN / FAIL)"
    - "Worst disparity ratio across all metrics"
    - "Number of flagged groups"
    - "Trend: disparity over last 30 days"
    
  per_metric_panels:
    demographic_parity:
      - "Positive prediction rate by group (bar chart)"
      - "4/5ths rule line (80% of highest group)"
      - "Trend over time"
    equal_opportunity:
      - "TPR by group (bar chart)"
      - "Disparity from best-performing group"
    equalized_odds:
      - "TPR and FPR by group (grouped bar chart)"
      - "ROC curves by group (overlay)"
      
  intersectional_panel:
    - "Performance by intersectional groups (race × gender)"
    - "Small group warning (insufficient data for reliable metrics)"
    
  trend_panel:
    - "Disparity ratio over time (line chart per metric)"
    - "Alert: disparity worsening over consecutive windows"
```

---

## How It Works in Practice

### Production Fairness Monitoring Workflow

```yaml
Workflow:
  data_collection:
    - "Log predictions with group labels (from user profiles or imputed)"
    - "Collect ground truth labels (with same group information)"
    - "Store in time-partitioned fairness evaluation table"
    
  evaluation:
    frequency: "Weekly (needs sufficient samples per group)"
    compute: "All fairness metrics across all monitored groups"
    output: "FairnessResult objects stored in monitoring database"
    
  alerting:
    threshold: "4/5ths rule (disparity ratio < 0.8)"
    alert_on: "Any metric failing for any group"
    severity: "HIGH (regulatory risk)"
    notify: "ML team + legal/compliance team"
    
  response:
    investigation:
      - "Which groups are disadvantaged?"
      - "Is this a model issue or a data issue?"
      - "Has the disparity been growing or is it new?"
    mitigation:
      - "Adjust decision threshold per group (post-processing)"
      - "Retrain with fairness constraints"
      - "Add features that reduce proxy correlation"
      - "Use adversarial debiasing"
```

---

## Interview Tip

> When asked about fairness monitoring: "I monitor fairness continuously in production using multiple metrics because no single fairness metric captures everything — and they can conflict mathematically (Chouldechova's impossibility theorem). My standard approach: (1) Demographic parity — are positive prediction rates equal across groups? This is the 4/5ths rule in disparate impact analysis. (2) Equal opportunity — is the true positive rate equal? (Qualified people from every group should have equal chance of positive prediction.) (3) Predictive parity — is precision equal? (Positive predictions should mean the same thing regardless of group.) I compute these weekly on rolling labeled data, broken down by protected attributes (and intersections — Black women may face different disparities than Black men or White women individually). Key implementation challenges: (a) Getting protected attribute data in production — often requires imputation (BISG for race), which introduces noise. (b) Small sample sizes for minority groups — need minimum thresholds before computing metrics (I use n≥30). (c) Choosing which metric to optimize — this is a values decision, not a technical one. I involve stakeholders (legal, product, ethics) in selecting the appropriate fairness criterion for each use case. When a disparity is detected: first diagnose (is it model-caused or reflecting upstream data bias?), then mitigate (threshold adjustment, constrained retraining, or adversarial debiasing depending on severity)."

---

## Common Mistakes

1. **Only checking overall accuracy** — "Model accuracy is 94% — it's fair!" But accuracy is 97% for majority group and 78% for minority group. The average hides disparities. Solution: ALWAYS disaggregate metrics by protected groups. Overall performance says nothing about fairness.

2. **Removing protected attributes and assuming fairness** — "We removed race and gender from the model inputs, so it can't discriminate." But ZIP code, university name, income level all correlate with protected attributes. Model still discriminates through proxies. Solution: monitor fairness outcomes regardless of whether protected attributes are model inputs.

3. **Using only one fairness metric** — "We pass demographic parity!" But equal opportunity is violated (qualified minority applicants are rejected at higher rates). Solution: monitor multiple metrics. They can conflict — choose which matters most for your application, but monitor all to understand the full picture.

4. **Insufficient sample sizes** — Computing fairness metrics for a group with 5 examples. Random variation dominates — metric is meaningless noise. Solution: require minimum sample size (30+) before computing metrics. For small groups, accumulate data over longer windows. Flag when sample is too small for reliable measurement.

5. **One-time audit instead of continuous monitoring** — "We checked fairness at deployment and it passed." But model behavior changes over time (drift affects groups differently), population changes, and new data patterns emerge. Solution: continuous fairness monitoring (weekly at minimum) — not just a one-time pre-deployment check.

---

## Key Takeaways

- Fairness metrics: demographic parity, equal opportunity, equalized odds, predictive parity
- Impossibility: these metrics cannot all be satisfied simultaneously (choose based on application)
- 4/5ths rule: disparate impact if any group's rate < 80% of the highest group's rate
- Monitor intersectional groups: race × gender may reveal disparities invisible in individual attributes
- Protected attribute proxies: removing race/gender doesn't prevent discrimination (ZIP code, name, etc.)
- Minimum sample sizes: need 30+ per group for reliable fairness metrics
- Continuous monitoring: weekly evaluation, not just one-time pre-deployment audit
- Regulatory requirements: EU AI Act, GDPR Article 22, US fair lending, NYC Local Law 144
- Mitigation options: threshold adjustment, constrained retraining, adversarial debiasing
- Stakeholder involvement: fairness criterion selection is a values decision, not just technical
