# A/B Test Monitoring for ML Models

## The Problem / Why This Matters

A/B testing (online experimentation) is the gold standard for evaluating ML model changes in production — you split traffic between the current model (control) and a new model (treatment), then measure which performs better on business metrics. But A/B tests for ML models are harder than traditional A/B tests for UI changes because: (1) ML models affect complex, multi-step user journeys (a recommendation model change affects browsing → cart → purchase → return), (2) effects can be delayed (a churn model improvement shows impact over weeks, not minutes), (3) guardrail metrics are critical (new model might improve the target metric but degrade another), (4) novelty effects confuse results (users initially engage more with any change), and (5) sample size requirements are larger for small effect sizes typical of ML improvements. A/B test monitoring means continuously tracking experiment health, detecting issues early (sample ratio mismatch, metric degradation), computing statistical significance correctly, and making trustworthy ship/no-ship decisions.

---

## The Analogy

Think of A/B testing ML models like a clinical drug trial:

- **Control group** = Patients on existing treatment (current model serving traffic)
- **Treatment group** = Patients on new drug (new model serving traffic)
- **Primary endpoint** = The main outcome you're measuring (revenue, engagement, accuracy)
- **Guardrail metrics** = Safety monitoring (new drug shouldn't cause harmful side effects)
- **Interim analysis** = Checking periodically if results are clear (or if you should stop early because treatment is harmful)
- **Statistical significance** = Being confident the observed difference is real, not random noise

You don't just compare averages — you need rigorous statistical methods to distinguish real effects from noise, monitor for harmful side effects, and decide when you have enough evidence to conclude.

---

## Deep Dive

### A/B Test Architecture for ML

```yaml
Architecture:
  traffic_splitting:
    method: "Hash user_id (or session_id) → consistent assignment"
    consistency: "Same user always sees same model version (within experiment duration)"
    granularity: "Usually user-level (not request-level) to avoid inconsistent experience"
    tools: "LaunchDarkly, Optimizely, Statsig, internal experimentation platform"
    
  experiment_design:
    unit: "User (most common), session, or request"
    split: "50/50 (fastest), 90/10 (lower risk), or multi-arm"
    duration: "1-4 weeks typical (need enough data for significance)"
    power_analysis:
      what: "Calculate minimum sample size BEFORE starting"
      inputs: "Baseline metric, minimum detectable effect (MDE), significance level (α), power (1-β)"
      typical_ml: "MDE of 1-2% on conversion → need 50K-200K users per variant"
    
  metrics_framework:
    primary_metric:
      what: "The ONE metric you're trying to improve"
      examples: "Conversion rate, revenue per user, CTR (Click-Through Rate), engagement"
      decision: "Ship if primary metric improves with statistical significance"
      
    secondary_metrics:
      what: "Additional metrics you hope to improve (but won't block shipping)"
      examples: "Session duration, pages viewed, satisfaction score"
      
    guardrail_metrics:
      what: "Metrics that MUST NOT degrade (regardless of primary metric improvement)"
      examples: "Page load time, error rate, unsubscribe rate, customer complaints"
      decision: "DO NOT ship if any guardrail degrades significantly"
      
    counter_metrics:
      what: "Metrics expected to move in opposite direction (acceptable trade-off)"
      examples: "If improving precision, recall may decrease (known trade-off)"
```

### Statistical Methods

```python
# A/B test statistical analysis for ML experiments

"""
Statistical methods for evaluating ML model A/B tests.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentResult:
    """Result of A/B test statistical analysis."""
    metric_name: str
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift: float
    p_value: float
    confidence_interval: tuple  # (lower, upper) for lift
    is_significant: bool
    significance_level: float
    power: float
    sample_size_control: int
    sample_size_treatment: int


class ABTestAnalyzer:
    """
    Statistical analysis for ML model A/B tests.
    
    Supports:
    - Frequentist (t-test, z-test)
    - Sequential testing (for early stopping)
    - CUPED variance reduction
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level
    
    def analyze_continuous_metric(
        self,
        control_values: np.ndarray,
        treatment_values: np.ndarray,
        metric_name: str = "metric",
    ) -> ExperimentResult:
        """
        Analyze a continuous metric (revenue, session duration, etc.)
        using Welch's t-test.
        """
        n_control = len(control_values)
        n_treatment = len(treatment_values)
        
        control_mean = float(np.mean(control_values))
        treatment_mean = float(np.mean(treatment_values))
        
        # Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(
            treatment_values, control_values, equal_var=False
        )
        
        # Lift
        absolute_lift = treatment_mean - control_mean
        relative_lift = absolute_lift / control_mean if control_mean != 0 else 0
        
        # Confidence interval for the difference
        se = np.sqrt(
            np.var(control_values) / n_control + 
            np.var(treatment_values) / n_treatment
        )
        ci_lower = absolute_lift - 1.96 * se
        ci_upper = absolute_lift + 1.96 * se
        
        return ExperimentResult(
            metric_name=metric_name,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            absolute_lift=absolute_lift,
            relative_lift=relative_lift,
            p_value=float(p_value),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            is_significant=p_value < self.alpha,
            significance_level=self.alpha,
            power=self._compute_power(control_values, treatment_values),
            sample_size_control=n_control,
            sample_size_treatment=n_treatment,
        )
    
    def analyze_proportion_metric(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
        metric_name: str = "conversion_rate",
    ) -> ExperimentResult:
        """
        Analyze a proportion metric (conversion rate, CTR, etc.)
        using z-test for proportions.
        """
        p_control = control_successes / control_total
        p_treatment = treatment_successes / treatment_total
        
        # Pooled proportion
        p_pooled = (control_successes + treatment_successes) / (control_total + treatment_total)
        
        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
        
        # Z-statistic and p-value
        z_stat = (p_treatment - p_control) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-sided
        
        # Confidence interval
        se_diff = np.sqrt(
            p_control * (1 - p_control) / control_total +
            p_treatment * (1 - p_treatment) / treatment_total
        )
        lift = p_treatment - p_control
        ci_lower = lift - 1.96 * se_diff
        ci_upper = lift + 1.96 * se_diff
        
        return ExperimentResult(
            metric_name=metric_name,
            control_mean=p_control,
            treatment_mean=p_treatment,
            absolute_lift=lift,
            relative_lift=lift / p_control if p_control > 0 else 0,
            p_value=float(p_value),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            is_significant=p_value < self.alpha,
            significance_level=self.alpha,
            power=0,  # Compute separately
            sample_size_control=control_total,
            sample_size_treatment=treatment_total,
        )
    
    def cuped_analysis(
        self,
        control_values: np.ndarray,
        treatment_values: np.ndarray,
        control_pre_values: np.ndarray,
        treatment_pre_values: np.ndarray,
        metric_name: str = "metric_cuped",
    ) -> ExperimentResult:
        """
        CUPED (Controlled-experiment Using Pre-Experiment Data).
        
        Reduces variance by adjusting for pre-experiment behavior.
        Can reduce required sample size by 50%+.
        
        Idea: if user's pre-experiment behavior predicts their experiment behavior,
        we can subtract that prediction and reduce noise.
        """
        # Compute theta (regression coefficient)
        all_post = np.concatenate([control_values, treatment_values])
        all_pre = np.concatenate([control_pre_values, treatment_pre_values])
        
        cov = np.cov(all_post, all_pre)[0, 1]
        var_pre = np.var(all_pre)
        theta = cov / var_pre if var_pre > 0 else 0
        
        # Adjust values: Y_adjusted = Y - theta * (X - mean(X))
        pre_mean = np.mean(all_pre)
        control_adjusted = control_values - theta * (control_pre_values - pre_mean)
        treatment_adjusted = treatment_values - theta * (treatment_pre_values - pre_mean)
        
        # Run standard t-test on adjusted values
        return self.analyze_continuous_metric(
            control_adjusted, treatment_adjusted, metric_name
        )
    
    def _compute_power(
        self, control: np.ndarray, treatment: np.ndarray
    ) -> float:
        """Compute statistical power of the test."""
        effect_size = abs(np.mean(treatment) - np.mean(control)) / np.std(control)
        n = min(len(control), len(treatment))
        
        # Approximate power using normal approximation
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        power = stats.norm.cdf(
            effect_size * np.sqrt(n) - z_alpha
        )
        return float(power)
    
    def sequential_test(
        self,
        control_values: np.ndarray,
        treatment_values: np.ndarray,
        max_sample_size: int,
        spending_function: str = "obrien_fleming",
    ) -> dict:
        """
        Sequential testing: test at interim analyses with controlled Type I error.
        
        Allows early stopping if effect is clear (saves time and resources).
        Uses alpha spending to control overall false positive rate.
        """
        current_n = len(control_values)
        info_fraction = current_n / max_sample_size
        
        # O'Brien-Fleming spending function (conservative early, liberal late)
        if spending_function == "obrien_fleming":
            alpha_spent = 2 * (1 - stats.norm.cdf(
                stats.norm.ppf(1 - self.alpha / 2) / np.sqrt(info_fraction)
            ))
        else:  # Pocock (uniform spending)
            alpha_spent = self.alpha * info_fraction
        
        # Run test at current alpha level
        t_stat, p_value = stats.ttest_ind(treatment_values, control_values, equal_var=False)
        
        return {
            "information_fraction": info_fraction,
            "alpha_spent_so_far": alpha_spent,
            "current_p_value": float(p_value),
            "can_reject": p_value < alpha_spent,
            "recommendation": (
                "STOP: reject null (treatment wins)" if p_value < alpha_spent
                else "CONTINUE: insufficient evidence"
            ),
        }
```

### Experiment Monitoring

```yaml
Monitoring_During_Experiment:
  health_checks:
    sample_ratio_mismatch:
      what: "Verify traffic split is as expected (e.g., 50/50)"
      test: "Chi-squared test on observed split vs expected"
      alert: "If p < 0.001, something is wrong with randomization"
      causes: "Bot traffic, caching issues, broken randomization, redirects"
      action: "STOP experiment, investigate — results are unreliable"
      
    metric_degradation:
      what: "Is treatment causing significant HARM?"
      implementation: "Sequential test for guardrail metrics"
      alert: "Guardrail metric significantly worse in treatment"
      action: "Consider stopping early to prevent user harm"
      
    novelty_effect:
      what: "Initial spike in engagement due to newness (not real improvement)"
      detection: "Effect size decreases over time (plot lift by day)"
      handling: "Exclude first 2-3 days from analysis, or run longer"
      
    data_quality:
      what: "Are metrics being collected correctly for both groups?"
      checks:
        - "No NULLs in metric columns"
        - "Metric ranges are reasonable"
        - "Event logging is working for both variants"
      
  dashboard_metrics:
    real_time:
      - "Traffic split (actual vs expected)"
      - "Primary metric running average (control vs treatment)"
      - "Guardrail metrics status (all green?)"
      - "Sample size progress (% of target reached)"
      
    daily:
      - "P-value trend (converging or diverging?)"
      - "Effect size with confidence intervals (shrinking CI = more data)"
      - "Segment breakdowns (is effect consistent across segments?)"
      
    end_of_experiment:
      - "Final statistical analysis (all metrics)"
      - "Power analysis (was test adequately powered?)"
      - "Heterogeneous effects (did some segments benefit more?)"
```

### ML-Specific A/B Test Considerations

```yaml
ML_Specific:
  model_warm_up:
    problem: "New model may perform poorly initially (cold caches, no recent predictions)"
    solution: "Warm-up period: route small traffic to treatment first, then ramp up"
    implementation: "0.1% → 1% → 10% → 50% over 3 days"
    
  feature_store_consistency:
    problem: "Both models must see same features for fair comparison"
    solution: "Both models read from same feature store snapshot"
    anti_pattern: "Treatment model uses new features unavailable to control"
    
  delayed_outcomes:
    problem: "Revenue/conversion happens days after recommendation"
    solution: "Run experiment long enough for outcomes to materialize"
    typical_duration: "2-4 weeks for most metrics"
    
  interaction_effects:
    problem: "Multiple models being A/B tested simultaneously (recommendation + ranking + pricing)"
    solution: "Layer-based experimentation (each model has its own randomization layer)"
    tool: "Google's Overlapping Experiment Infrastructure (OEIL) pattern"
    
  feedback_loops:
    problem: "Model A's predictions affect what data Model B learns from"
    example: "If treatment model recommends different items, future training data changes"
    mitigation: "Use held-out exploration set not influenced by model choices"
    
  model_confidence:
    problem: "Treatment model is less confident → more varied predictions → different user behavior"
    monitoring: "Track confidence distribution in both variants"
    interpretation: "Lower confidence + higher metrics = model is usefully uncertain"
```

---

## How It Works in Practice

### End-to-End Experiment Workflow

```yaml
Workflow:
  pre_experiment:
    1: "Define hypothesis: 'New model will improve CTR by 2%'"
    2: "Power analysis: need 100K users per variant for MDE=2%, α=0.05, power=0.8"
    3: "Identify metrics: primary (CTR), guardrails (page load, error rate), secondary (session duration)"
    4: "Set up experiment in platform (traffic split, metric definitions, duration)"
    
  during_experiment:
    day_1: "Verify traffic split (sample ratio check), verify logging works"
    daily: "Check guardrail metrics, monitor for degradation"
    weekly: "Interim analysis (sequential test), review segment breakdowns"
    
  post_experiment:
    analysis:
      - "Final statistical test on primary metric"
      - "Check ALL guardrail metrics"
      - "Segment analysis (did all user groups benefit?)"
      - "Long-term effect check (exclude novelty period)"
    decision:
      ship: "Primary metric significantly positive AND no guardrail degradation"
      dont_ship: "Primary metric not significant OR guardrail degraded"
      extend: "Promising direction but insufficient power — run longer"
```

---

## Interview Tip

> When asked about A/B testing ML models: "ML A/B tests have unique challenges beyond traditional web experiments. My approach: (1) Pre-experiment: power analysis to determine sample size (ML improvements are typically small — 1-3% lift — so you need large samples, often 50K-200K users per variant). I use CUPED (Controlled-experiment Using Pre-Experiment Data) to reduce variance by 30-50%, effectively requiring fewer users for the same power. (2) During experiment: continuous monitoring for experiment health — sample ratio mismatch (if split isn't 50/50, something's wrong with randomization), guardrail metrics (new model must not degrade latency, error rate, or user satisfaction), and novelty effects (initial engagement spike that fades). (3) Analysis: I use sequential testing (O'Brien-Fleming spending function) to enable early stopping when effects are clearly positive or negative — this saves weeks compared to fixed-duration tests. (4) ML-specific concerns: model warm-up (ramp traffic gradually to avoid cold-start effects), feature consistency (both variants must use same feature store), delayed outcomes (run long enough for conversions to materialize, typically 2-4 weeks), and interaction effects (use layered experimentation when multiple models are tested simultaneously). The ship decision requires BOTH statistical significance on primary metric AND no degradation in guardrail metrics — one without the other is insufficient."

---

## Common Mistakes

1. **Peeking at results and stopping early** — Checking p-value daily and shipping when it crosses 0.05. But p-values fluctuate — early significance often disappears. This inflates false positive rate from 5% to 20-30%. Solution: either commit to fixed duration OR use sequential testing with alpha-spending functions that explicitly control the overall false positive rate.

2. **Ignoring guardrail metrics** — "Primary metric improved 3%! Ship!" But page load time increased 200ms, and unsubscribe rate doubled. Solution: define guardrail metrics BEFORE experiment starts. Ship decision requires no significant degradation in ANY guardrail, even if primary metric improves.

3. **Underpowered experiments** — Running experiment with 1000 users per variant when you need 100K for the expected effect size. "No significant difference" → conclude model is same. But you didn't have power to detect a 2% improvement even if it existed. Solution: power analysis before starting. If you can't get enough traffic, consider composite metrics or longer duration.

4. **Not checking sample ratio** — Experiment says 50/50 split, but actual traffic is 52/48 due to a bug. Results are biased. Solution: first check in any experiment: is the actual traffic split what you expect? Chi-squared test on sample counts. If SRM (Sample Ratio Mismatch) detected (p < 0.001), stop and investigate before analyzing results.

5. **Confusing novelty effect with real improvement** — New model shows 10% CTR lift in first 3 days, but users are just exploring the "new" experience. Effect fades to 1% after a week. Solution: run experiments for 2+ weeks. Analyze effect by day — is it stable or declining? Exclude first few days from final analysis.

---

## Key Takeaways

- A/B testing: gold standard for evaluating ML model changes in production
- Power analysis: calculate required sample size BEFORE starting (MDE, α, power)
- CUPED: reduce variance by 30-50% using pre-experiment data (smaller sample needed)
- Sequential testing: enable early stopping while controlling false positive rate
- Guardrail metrics: must NOT degrade (page load, error rate, satisfaction)
- Sample Ratio Mismatch: first health check — if split is wrong, results are invalid
- Novelty effect: initial spike that fades — run 2+ weeks, exclude first few days
- ML-specific: model warm-up, feature consistency, delayed outcomes, interaction effects
- Ship decision: significant primary improvement + no guardrail degradation
- Layered experiments: separate randomization layers for multiple simultaneous model tests
