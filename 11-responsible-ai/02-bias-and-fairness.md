# Bias and Fairness

## The Problem / Why This Matters

ML models learn patterns from data — including patterns of historical discrimination. A hiring model trained on past decisions will learn that women were hired less for engineering roles (because of past bias, not lack of qualification). A lending model trained on historical approvals will learn that certain zip codes (proxies for race) predict rejection. A facial recognition system trained primarily on lighter-skinned faces will fail on darker skin tones. Bias in ML isn't a bug that can be fixed with one technique — it's a systemic challenge that manifests at every stage: data collection (who's represented?), feature selection (which attributes encode bias?), model training (which patterns are amplified?), evaluation (which groups are tested?), and deployment (who is affected?). In 2026, with AI making consequential decisions about hiring, lending, healthcare, criminal justice, and housing, understanding and mitigating bias is a core ML engineering competency. The challenge: there's no single definition of "fair," fairness metrics can mathematically conflict with each other, and removing sensitive attributes doesn't remove bias (proxies exist). ML engineers must navigate these trade-offs with rigor, transparency, and continuous measurement.

---

## The Analogy

Think of bias in ML like bias in a mirror:

- **Biased training data** = A funhouse mirror that makes some people look thinner and others larger. The mirror faithfully reflects, but the reflection is distorted from the start. Training on historical data means learning historical biases.
- **Model amplification** = The funhouse mirror gets more extreme over time. ML models don't just learn bias — they can amplify it. A 60/40 bias in data can become 80/20 in predictions because the model optimizes for patterns, including biased ones.
- **Fairness intervention** = Adding corrective lenses. Different lenses fix different distortions (some make height fair, others make width fair). No single lens makes everything simultaneously perfect — you choose which distortion matters most for your context.

---

## Deep Dive

### Types of Bias

```yaml
Bias_Types:
  data_bias:
    representation_bias:
      what: "Some groups underrepresented in training data"
      example: "Medical dataset 80% male → model worse at diagnosing female patients"
      fix: "Audit representation, oversample underrepresented groups, collect more data"
      
    historical_bias:
      what: "Data reflects past discrimination"
      example: "Hiring data shows women rejected from engineering (past discrimination, not qualification)"
      fix: "Don't use historical decisions as ground truth. Redefine target variable."
      
    measurement_bias:
      what: "Features measured differently across groups"
      example: "Credit scores more available for affluent populations → model biased toward them"
      fix: "Audit feature availability by group, use robust features, multiple data sources"
      
    label_bias:
      what: "Labels (ground truth) are themselves biased"
      example: "Recidivism labels biased by policing patterns (more arrests where more police)"
      fix: "Acknowledge label limitations, use debiased labels, or alternative outcome measures"
      
  algorithm_bias:
    optimization_bias:
      what: "Model optimizes for majority group performance (larger gradient signal)"
      example: "99% non-fraud, 1% fraud → model optimized for majority, ignores minority patterns"
      fix: "Group-aware optimization, fairness constraints in loss function"
      
    proxy_discrimination:
      what: "Model uses features correlated with protected attributes"
      example: "Zip code → race proxy. Removing race doesn't remove racial bias."
      fix: "Audit for proxy features, adversarial debiasing, causal modeling"
      
    amplification:
      what: "Model amplifies existing biases beyond training data levels"
      example: "Training: 60% male doctors. Model predictions: 80% male for 'doctor' queries"
      fix: "Post-processing calibration, fairness constraints, debiased embeddings"
      
  deployment_bias:
    population_shift:
      what: "Production population differs from training population"
      example: "Model trained in US deployed globally without validation"
      fix: "Validate on deployment population, monitor performance by region/demographic"
      
    feedback_loop:
      what: "Model predictions influence future training data"
      example: "Loan denial → no repayment data → confirms 'risky' → perpetuates denial"
      fix: "Exploration (approve some predicted-denials for ground truth), causal analysis"
```

### Fairness Metrics

```yaml
Fairness_Metrics:
  group_fairness:
    demographic_parity:
      definition: "P(positive outcome | Group A) = P(positive outcome | Group B)"
      meaning: "Equal acceptance rates across groups"
      when_appropriate: "When equal representation in outcome is the goal"
      limitation: "Ignores qualifications — forces equal outcomes regardless of eligibility"
      
    equal_opportunity:
      definition: "P(predict positive | actually positive, Group A) = P(predict positive | actually positive, Group B)"
      meaning: "Equal true positive rates (qualified people treated equally)"
      when_appropriate: "When you want qualified members of all groups to have equal chance"
      limitation: "Doesn't constrain false positive rates"
      
    equalized_odds:
      definition: "Equal TPR AND equal FPR across groups"
      meaning: "Both qualified and unqualified treated equally across groups"
      when_appropriate: "When both types of errors matter (criminal justice)"
      limitation: "Hard to satisfy, often requires significant accuracy trade-off"
      
    predictive_parity:
      definition: "P(actually positive | predict positive, Group A) = P(actually positive | predict positive, Group B)"
      meaning: "Precision equal across groups (positive predictions equally reliable)"
      when_appropriate: "When you want users to trust predictions equally regardless of group"
      
    calibration:
      definition: "Among those predicted 70% risk, 70% actually default — for ALL groups"
      meaning: "Predicted probabilities are accurate within each group"
      when_appropriate: "When probability estimates guide decisions (insurance, lending)"
      
  individual_fairness:
    definition: "Similar individuals receive similar predictions"
    challenge: "Defining 'similar' is subjective and domain-dependent"
    approach: "Metric learning to define similarity, then ensure nearby individuals get similar scores"
    
  impossibility_theorem:
    what: "Except in trivial cases, you CANNOT satisfy all fairness metrics simultaneously"
    implication: "Must CHOOSE which fairness definition matters most for your context"
    reference: "Chouldechova (2017), Kleinberg et al. (2016)"
    guidance: "Choose based on: who is harmed by which type of error?"
```

### Implementation

```python
# Bias measurement and mitigation implementation

"""
Fairness measurement and bias mitigation for production ML.
Covers pre-processing, in-processing, and post-processing approaches.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class FairnessReport:
    """Fairness evaluation report for a model."""
    model_name: str
    model_version: str
    protected_attribute: str
    groups: List[str]
    metrics: Dict[str, Dict[str, float]]  # metric_name → {group: value}
    disparities: Dict[str, float]  # metric_name → max disparity
    passes_threshold: bool
    threshold: float
    recommendations: List[str]


class FairnessEvaluator:
    """
    Evaluate ML model fairness across demographic groups.
    
    Computes multiple fairness metrics and identifies disparities.
    """
    
    def __init__(self, disparity_threshold: float = 0.1):
        """
        Args:
            disparity_threshold: Maximum acceptable difference between groups.
                               0.1 = max 10% difference in metric values.
        """
        self.threshold = disparity_threshold
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        protected_attribute: np.ndarray,
        model_name: str = "model",
        model_version: str = "1.0",
    ) -> FairnessReport:
        """
        Comprehensive fairness evaluation.
        
        Args:
            y_true: Ground truth labels (0/1)
            y_pred: Predicted labels (0/1)
            y_prob: Predicted probabilities
            protected_attribute: Group membership (e.g., "male", "female")
            
        Returns:
            FairnessReport with metrics per group and disparity analysis.
        """
        groups = np.unique(protected_attribute)
        metrics = {}
        
        # Compute metrics per group
        metrics["selection_rate"] = {}
        metrics["tpr"] = {}  # True Positive Rate (recall)
        metrics["fpr"] = {}  # False Positive Rate
        metrics["precision"] = {}
        metrics["accuracy"] = {}
        
        for group in groups:
            mask = protected_attribute == group
            group_true = y_true[mask]
            group_pred = y_pred[mask]
            
            # Selection rate (demographic parity)
            metrics["selection_rate"][group] = float(np.mean(group_pred))
            
            # True Positive Rate (equal opportunity)
            positives = group_true == 1
            if positives.sum() > 0:
                metrics["tpr"][group] = float(
                    np.mean(group_pred[positives] == 1)
                )
            else:
                metrics["tpr"][group] = 0.0
            
            # False Positive Rate
            negatives = group_true == 0
            if negatives.sum() > 0:
                metrics["fpr"][group] = float(
                    np.mean(group_pred[negatives] == 1)
                )
            else:
                metrics["fpr"][group] = 0.0
            
            # Precision
            predicted_pos = group_pred == 1
            if predicted_pos.sum() > 0:
                metrics["precision"][group] = float(
                    np.mean(group_true[predicted_pos] == 1)
                )
            else:
                metrics["precision"][group] = 0.0
            
            # Accuracy
            metrics["accuracy"][group] = float(np.mean(group_true == group_pred))
        
        # Compute disparities (max difference between any two groups)
        disparities = {}
        for metric_name, group_values in metrics.items():
            values = list(group_values.values())
            if values:
                disparities[metric_name] = max(values) - min(values)
        
        # Check if passes threshold
        passes = all(d <= self.threshold for d in disparities.values())
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, disparities)
        
        return FairnessReport(
            model_name=model_name,
            model_version=model_version,
            protected_attribute=str(protected_attribute[0].__class__.__name__),
            groups=[str(g) for g in groups],
            metrics=metrics,
            disparities=disparities,
            passes_threshold=passes,
            threshold=self.threshold,
            recommendations=recommendations,
        )
    
    def _generate_recommendations(
        self, metrics: Dict, disparities: Dict
    ) -> List[str]:
        """Generate actionable recommendations based on fairness analysis."""
        recommendations = []
        
        # Selection rate disparity (demographic parity)
        if disparities.get("selection_rate", 0) > self.threshold:
            recommendations.append(
                f"Selection rate disparity ({disparities['selection_rate']:.3f}) exceeds threshold. "
                "Consider: threshold adjustment per group, or re-weighting training data."
            )
        
        # TPR disparity (equal opportunity)
        if disparities.get("tpr", 0) > self.threshold:
            recommendations.append(
                f"True Positive Rate disparity ({disparities['tpr']:.3f}) exceeds threshold. "
                "Qualified members of disadvantaged group are being missed. "
                "Consider: collecting more data for underperforming group, or in-processing constraints."
            )
        
        # FPR disparity
        if disparities.get("fpr", 0) > self.threshold:
            recommendations.append(
                f"False Positive Rate disparity ({disparities['fpr']:.3f}) exceeds threshold. "
                "One group is experiencing more false accusations/rejections. "
                "Consider: group-specific threshold calibration."
            )
        
        if not recommendations:
            recommendations.append("All fairness metrics within acceptable thresholds.")
        
        return recommendations


class BiasM mitigation:
    """
    Bias mitigation strategies at different pipeline stages.
    
    Three categories:
    1. Pre-processing: modify training data before training
    2. In-processing: add fairness constraints during training
    3. Post-processing: adjust predictions after training
    """
    
    @staticmethod
    def reweighting(
        y_true: np.ndarray,
        protected_attribute: np.ndarray,
    ) -> np.ndarray:
        """
        Pre-processing: Reweight training samples.
        
        Upweight underrepresented (group, label) combinations.
        
        Example: If women are under-approved in training data,
        upweight (female, approved) examples to counteract bias.
        """
        groups = np.unique(protected_attribute)
        labels = np.unique(y_true)
        n = len(y_true)
        
        weights = np.ones(n)
        
        for group in groups:
            for label in labels:
                # Expected proportion (if fair)
                group_mask = protected_attribute == group
                label_mask = y_true == label
                
                p_group = group_mask.sum() / n
                p_label = label_mask.sum() / n
                expected = p_group * p_label * n
                
                # Actual count
                actual = (group_mask & label_mask).sum()
                
                if actual > 0:
                    weight = expected / actual
                    weights[group_mask & label_mask] = weight
        
        return weights
    
    @staticmethod
    def threshold_adjustment(
        y_prob: np.ndarray,
        protected_attribute: np.ndarray,
        target_metric: str = "equal_opportunity",
        target_rate: float = None,
    ) -> Dict[str, float]:
        """
        Post-processing: Different thresholds per group.
        
        Find threshold for each group that achieves target fairness metric.
        
        Example: If Group A needs threshold 0.5 for 80% TPR,
        and Group B needs threshold 0.4 for 80% TPR,
        use different thresholds to equalize TPR.
        
        Note: This is controversial — treating groups differently.
        May be appropriate for equal opportunity, less so for others.
        """
        groups = np.unique(protected_attribute)
        thresholds = {}
        
        for group in groups:
            mask = protected_attribute == group
            group_probs = y_prob[mask]
            
            # Find threshold that achieves target rate for this group
            # Simple binary search over thresholds
            best_threshold = 0.5  # default
            
            if target_rate is not None:
                for t in np.arange(0.01, 0.99, 0.01):
                    group_pred = (group_probs >= t).astype(int)
                    rate = np.mean(group_pred)  # Selection rate
                    if abs(rate - target_rate) < 0.02:
                        best_threshold = t
                        break
            
            thresholds[str(group)] = best_threshold
        
        return thresholds
    
    @staticmethod
    def adversarial_debiasing_concept():
        """
        In-processing: Adversarial debiasing.
        
        Architecture:
        - Main model: predicts target variable (y)
        - Adversary model: tries to predict protected attribute from main model's predictions
        - Training: main model optimized to predict y AND fool adversary
        
        Result: predictions that are accurate but DON'T contain information
        about protected attribute (adversary can't guess group from prediction).
        
        Implementation: Use fairlearn or AIF360 libraries.
        """
        return {
            "approach": "adversarial_debiasing",
            "how_it_works": (
                "Train main model (predict target) with adversary (predict group from predictions). "
                "Main model learns to be accurate while making predictions independent of group membership."
            ),
            "libraries": ["fairlearn", "aif360", "tensorflow-constrained-optimization"],
            "trade_off": "May reduce overall accuracy to achieve fairness",
        }


class FairnessMonitoring:
    """
    Continuous fairness monitoring in production.
    
    Even if a model is fair at deployment, fairness can degrade:
    - Population shift (different demographics in production)
    - Feature drift (feature distributions change)
    - Feedback loops (model affects future data)
    """
    
    def __init__(self, evaluator: FairnessEvaluator, alert_threshold: float = 0.15):
        self.evaluator = evaluator
        self.alert_threshold = alert_threshold
        self.history: List[FairnessReport] = []
    
    def check(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        protected_attribute: np.ndarray,
        window: str = "daily",
    ) -> Dict:
        """
        Run fairness check on recent predictions.
        
        Alert if fairness degrades beyond threshold.
        """
        report = self.evaluator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            protected_attribute=protected_attribute,
        )
        
        self.history.append(report)
        
        # Check for alerting conditions
        alerts = []
        for metric, disparity in report.disparities.items():
            if disparity > self.alert_threshold:
                alerts.append({
                    "metric": metric,
                    "disparity": disparity,
                    "threshold": self.alert_threshold,
                    "severity": "critical" if disparity > 0.2 else "warning",
                })
        
        # Check for degradation trend
        if len(self.history) >= 7:
            recent_disparities = [
                max(r.disparities.values()) for r in self.history[-7:]
            ]
            trend = np.polyfit(range(7), recent_disparities, 1)[0]
            if trend > 0.01:  # Disparity increasing
                alerts.append({
                    "metric": "trend",
                    "message": f"Fairness degrading: +{trend:.4f}/day",
                    "severity": "warning",
                })
        
        return {
            "report": report,
            "alerts": alerts,
            "status": "ALERT" if alerts else "OK",
        }
```

### Fairness in LLMs

```yaml
LLM_Fairness:
  challenges:
    training_data: "Internet data contains stereotypes, biases, toxic content"
    amplification: "LLMs can amplify biases in creative/generative tasks"
    evaluation: "No single ground truth — bias manifests in subtle language choices"
    intersectionality: "Bias compounds across multiple attributes (race + gender + age)"
    
  measurement:
    stereotype_benchmarks: "BBQ, WinoBias, StereoSet — measure stereotypical associations"
    toxicity: "RealToxicityPrompts — measure harmful generation likelihood"
    representation: "Who is depicted how? (occupational stereotypes, sentiment by group)"
    
  mitigation:
    rlhf: "RLHF (Reinforcement Learning from Human Feedback) with diverse annotators"
    constitutional_ai: "Principles-based self-correction (Anthropic's approach)"
    red_teaming: "Adversarial testing for biased outputs"
    output_filtering: "Post-generation bias detection and filtering"
    prompt_engineering: "System prompts that instruct balanced, unbiased responses"
```

---

## How It Works in Practice

### Fairness Audit for a Lending Model

```yaml
Lending_Model_Audit:
  context: "Credit decision model (approve/deny loan applications)"
  protected_attributes: "Race, gender, age"
  
  findings:
    selection_rate:
      white: 0.72
      black: 0.51
      hispanic: 0.58
      disparity: "21% gap (white vs. black) — FAILS threshold of 10%"
      
    equal_opportunity:
      white_tpr: 0.85  # 85% of qualified white applicants approved
      black_tpr: 0.68  # 68% of qualified Black applicants approved
      disparity: "17% gap — FAILS"
      
    root_cause_analysis:
      - "Zip code feature is proxy for race (strong correlation)"
      - "Credit history length biased (younger applicants/recent immigrants disadvantaged)"
      - "Training data reflects historical discrimination in lending"
      
  mitigation_applied:
    1: "Removed zip code (proxy for race)"
    2: "Reweighted training data to equalize (group, approval) combinations"
    3: "Added fairness constraint in training (max 5% TPR gap)"
    4: "Post-processing threshold adjustment per group"
    
  after_mitigation:
    selection_rate_gap: "8% (within threshold)"
    equal_opportunity_gap: "5% (within threshold)"
    accuracy_trade_off: "Overall accuracy dropped 2% (from 87% to 85%)"
    
  ongoing:
    - "Monthly fairness monitoring (alert if gaps grow)"
    - "Quarterly external audit"
    - "Feedback loop monitoring (denied applicants tracked for outcomes)"
```

---

## Interview Tip

> When asked about bias and fairness: "I approach fairness as a measurable engineering requirement, not an abstract value. First, I identify which fairness definition applies — this depends on the domain. For hiring: equal opportunity (qualified candidates from all groups should have equal acceptance chance). For lending: calibration (a 70% predicted repayment probability should mean 70% actually repay, regardless of race). For criminal justice: equalized odds (equal TPR AND FPR across groups). I know these can mathematically conflict (impossibility theorem), so I choose based on which errors cause most harm. Measurement: I compute fairness metrics across protected groups in every evaluation. Threshold: max 10% disparity between groups (configurable per domain). If it fails, I apply mitigation: (1) Pre-processing — reweight training data to counteract historical bias. (2) In-processing — fairness constraints in the loss function (fairlearn library). (3) Post-processing — group-specific thresholds to equalize target metric. Critical insight: removing the sensitive attribute (race, gender) doesn't remove bias — proxies exist (zip code correlates with race, name correlates with gender). I audit for proxy features using correlation analysis. In production: continuous fairness monitoring. Disparities can grow due to population shift or feedback loops. Alert when gaps exceed threshold. Quarterly re-evaluation with fresh data."

---

## Common Mistakes

1. **Removing sensitive attribute and assuming fairness** — Dropping 'race' or 'gender' from features. Model still discriminates through proxies (zip code, name, school). Solution: audit for proxy discrimination using correlation analysis and counterfactual testing.

2. **Using one fairness metric** — Optimizing demographic parity alone. Might force equal outcomes for unequally qualified groups (lowering bar for one, raising for another). Solution: choose the fairness metric that matches your domain's harm model. Document the trade-off.

3. **Not disaggregating evaluation** — Reporting overall accuracy (87%) without checking per-group performance. Overall accuracy hides disparities (92% for majority, 68% for minority). Solution: always report metrics disaggregated by protected attributes.

4. **Fairness only at deployment** — Checking fairness once before launch, never again. Population shifts, feedback loops cause fairness degradation over time. Solution: continuous fairness monitoring in production with automated alerts.

5. **Ignoring intersectionality** — Checking fairness for gender and race separately, but not for intersections (Black women, elderly Hispanic men). Compound bias at intersections can be severe. Solution: evaluate at intersectional subgroups, not just individual attributes.

---

## Key Takeaways

- Bias types: representation, historical, measurement, label, optimization, proxy, amplification
- Fairness metrics: demographic parity, equal opportunity, equalized odds, calibration
- Impossibility theorem: can't satisfy all fairness metrics simultaneously — must choose
- Mitigation: pre-processing (reweighting), in-processing (fairness constraints), post-processing (threshold adjustment)
- Proxy discrimination: removing sensitive attributes doesn't remove bias (proxies exist)
- Continuous monitoring: fairness degrades over time (drift, feedback loops) — alert on increases
- Intersectionality: evaluate at intersections of protected attributes (not just individually)
- Trade-off: fairness improvements may reduce overall accuracy (2-5% typical)
- Tools: fairlearn, AIF360, Responsible AI Toolbox, custom evaluation pipelines
- Not optional: regulatory requirement (EU AI Act), legal liability, and ethical imperative
