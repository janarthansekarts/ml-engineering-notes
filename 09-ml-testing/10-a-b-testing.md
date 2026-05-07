# A/B Testing for ML

## The Problem / Why This Matters

Offline evaluation (testing on held-out data) tells you how a model performs on historical data. But production is different — user behavior changes, feedback loops emerge, and the model interacts with the full system in ways offline tests can't capture. A/B testing (also called online experimentation) is the gold standard for evaluating ML models in production. It exposes real users to different model versions and measures actual business outcomes (revenue, engagement, conversion, satisfaction). In 2026, sophisticated A/B testing for ML handles unique challenges: longer-term effects (a recommendation model might boost short-term clicks but reduce long-term engagement), network effects (a social media algorithm's impact depends on what friends see), non-stationarity (user behavior changes during the experiment), and statistical complexity (multiple metrics, multiple models, correlated outcomes). Organizations like Netflix, Spotify, and Uber run thousands of ML experiments simultaneously. Getting A/B testing wrong means either shipping worse models (false positive) or failing to ship better ones (insufficient power/sensitivity).

---

## The Analogy

Think of A/B testing for ML like a clinical drug trial:

- **Offline evaluation** = Lab testing. The drug kills cancer cells in a petri dish (model beats baseline on test set). Necessary but insufficient — petri dish ≠ human body.
- **A/B testing** = Clinical trial. Give the drug to real patients (real users), compare outcomes against placebo (current model), measure actual health outcomes (business metrics), with proper randomization and statistical rigor.

Just like a drug might work in the lab but fail in clinical trials (side effects, interactions, compliance), a model might win offline but lose online (unexpected user behavior, system interactions, long-term effects).

---

## Deep Dive

### A/B Testing Architecture for ML

```yaml
Architecture:
  components:
    traffic_splitter:
      what: "Randomly assign users to control (current model) or treatment (new model)"
      requirements:
        - "Consistent: same user always sees same variant (sticky assignment)"
        - "Random: no selection bias between groups"
        - "Balanced: equal sample sizes (or intentional split ratios)"
      implementation: "Hash(user_id + experiment_id) mod 100 → bucket assignment"
      
    model_router:
      what: "Route prediction requests to correct model based on assignment"
      implementation: "Feature flag / experiment SDK checks user's bucket"
      
    metric_collector:
      what: "Collect outcome metrics for both groups"
      metrics:
        primary: "The ONE metric the experiment is designed to move (e.g., conversion)"
        secondary: "Related metrics (engagement, revenue, satisfaction)"
        guardrail: "Metrics that must NOT degrade (latency, crash rate, complaint rate)"
      
    statistical_engine:
      what: "Analyze results with proper statistical methodology"
      methods:
        frequentist: "p-values, confidence intervals, power analysis"
        bayesian: "Posterior probability of improvement, credible intervals"
        sequential: "Continuous monitoring with valid stopping rules"
```

### Experiment Design

```python
# A/B test design and analysis for ML models

"""
Design, run, and analyze A/B tests for ML model deployments.
Handles: sample size calculation, randomization, and statistical analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment."""
    experiment_id: str
    model_control: str  # Control model version
    model_treatment: str  # Treatment model version
    traffic_split: float  # Fraction to treatment (e.g., 0.5 = 50/50)
    primary_metric: str  # The key metric to evaluate
    guardrail_metrics: List[str]  # Metrics that must not degrade
    minimum_detectable_effect: float  # MDE (minimum meaningful difference)
    significance_level: float = 0.05  # Alpha (false positive rate)
    power: float = 0.80  # 1 - beta (probability of detecting true effect)
    duration_days: int = 14  # Minimum experiment duration


class ExperimentDesigner:
    """
    Design A/B experiments with proper statistical rigor.
    
    Key decisions:
    1. Sample size (power analysis)
    2. Duration (novelty effects, day-of-week effects)
    3. Randomization unit (user, session, request)
    4. Metrics (primary, secondary, guardrail)
    """
    
    def calculate_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> int:
        """
        Calculate required sample size per group for a proportion test.
        
        Args:
            baseline_rate: Current conversion rate (e.g., 0.05 for 5%)
            minimum_detectable_effect: Minimum meaningful relative change (e.g., 0.05 for 5% relative)
            alpha: Significance level (false positive rate)
            power: Statistical power (1 - false negative rate)
            
        Returns:
            Required sample size per group
        """
        # Effect in absolute terms
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)
        
        # Pooled proportion
        p_avg = (p1 + p2) / 2
        
        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha / 2)  # Two-sided
        z_beta = stats.norm.ppf(power)
        
        # Sample size formula for proportions
        n = (
            (z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) +
             z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        ) / (p2 - p1) ** 2
        
        return int(np.ceil(n))
    
    def calculate_duration(
        self,
        required_sample_size: int,
        daily_traffic: int,
        traffic_split: float = 0.5,
        min_days: int = 7  # At least one full week (day-of-week effects)
    ) -> int:
        """
        Calculate experiment duration based on required sample size and traffic.
        
        Minimum 7 days to capture day-of-week effects.
        Minimum 14 days recommended to capture novelty effects wearing off.
        """
        per_group_traffic = daily_traffic * traffic_split
        days_needed = int(np.ceil(required_sample_size / per_group_traffic))
        
        return max(days_needed, min_days)
    
    def design_experiment(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        daily_traffic: int,
        primary_metric: str,
        guardrail_metrics: List[str],
    ) -> Dict:
        """
        Complete experiment design with all parameters.
        
        Returns design document with:
        - Sample size
        - Duration
        - Recommended traffic split
        - Power analysis details
        """
        sample_size = self.calculate_sample_size(
            baseline_rate, minimum_detectable_effect
        )
        
        duration = self.calculate_duration(
            sample_size, daily_traffic
        )
        
        return {
            "sample_size_per_group": sample_size,
            "total_sample_needed": sample_size * 2,
            "duration_days": duration,
            "traffic_split": "50/50",
            "primary_metric": primary_metric,
            "guardrail_metrics": guardrail_metrics,
            "baseline_rate": baseline_rate,
            "minimum_detectable_effect": minimum_detectable_effect,
            "significance_level": 0.05,
            "power": 0.80,
            "randomization_unit": "user_id",
            "notes": [
                "Run for at least full duration (don't peek and stop early)",
                "Check guardrail metrics before celebrating primary metric win",
                "Watch for novelty effects (metrics inflated early, then decay)",
            ],
        }


class TrafficSplitter:
    """
    Deterministic traffic splitting for A/B experiments.
    
    Key properties:
    - Consistent: same user always in same bucket
    - Independent: different experiments are independent
    - Uniform: buckets are evenly distributed
    """
    
    def assign_bucket(
        self,
        user_id: str,
        experiment_id: str,
        num_buckets: int = 100
    ) -> int:
        """
        Assign user to a bucket for an experiment.
        
        Uses deterministic hashing for consistency:
        - Same user + same experiment → always same bucket
        - Different experiments → independent assignments
        """
        hash_input = f"{user_id}:{experiment_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        return hash_value % num_buckets
    
    def get_variant(
        self,
        user_id: str,
        experiment_id: str,
        treatment_fraction: float = 0.5
    ) -> str:
        """
        Get user's variant assignment.
        
        Returns "control" or "treatment".
        """
        bucket = self.assign_bucket(user_id, experiment_id)
        threshold = int(treatment_fraction * 100)
        
        return "treatment" if bucket < threshold else "control"


class ExperimentAnalyzer:
    """
    Statistical analysis of A/B experiment results.
    
    Supports:
    - Frequentist analysis (p-values, confidence intervals)
    - Bayesian analysis (posterior probability of improvement)
    - Sequential testing (valid peeking with alpha spending)
    """
    
    def analyze_proportion(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
        alpha: float = 0.05
    ) -> Dict:
        """
        Analyze A/B test for a proportion metric (e.g., conversion rate).
        
        Returns: p-value, confidence interval, effect size, and recommendation.
        """
        # Rates
        p_control = control_successes / control_total
        p_treatment = treatment_successes / treatment_total
        
        # Relative lift
        relative_lift = (p_treatment - p_control) / p_control if p_control > 0 else 0
        
        # Z-test for proportions
        p_pooled = (control_successes + treatment_successes) / (control_total + treatment_total)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
        
        z_stat = (p_treatment - p_control) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-sided
        
        # Confidence interval for difference
        se_diff = np.sqrt(
            p_control * (1 - p_control) / control_total +
            p_treatment * (1 - p_treatment) / treatment_total
        )
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_lower = (p_treatment - p_control) - z_crit * se_diff
        ci_upper = (p_treatment - p_control) + z_crit * se_diff
        
        # Decision
        significant = p_value < alpha
        treatment_better = p_treatment > p_control
        
        return {
            "control_rate": p_control,
            "treatment_rate": p_treatment,
            "absolute_lift": p_treatment - p_control,
            "relative_lift": relative_lift,
            "p_value": p_value,
            "significant": significant,
            "confidence_interval": (ci_lower, ci_upper),
            "recommendation": (
                "SHIP treatment" if significant and treatment_better
                else "KEEP control" if significant and not treatment_better
                else "INCONCLUSIVE (need more data or larger effect)"
            ),
            "sample_sizes": {"control": control_total, "treatment": treatment_total},
        }
    
    def analyze_continuous(
        self,
        control_values: np.ndarray,
        treatment_values: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """
        Analyze A/B test for a continuous metric (e.g., revenue per user).
        
        Uses Welch's t-test (doesn't assume equal variances).
        """
        # Means
        mean_control = np.mean(control_values)
        mean_treatment = np.mean(treatment_values)
        
        # Welch's t-test
        t_stat, p_value = stats.ttest_ind(
            treatment_values, control_values, equal_var=False
        )
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(control_values) + np.var(treatment_values)) / 2
        )
        cohens_d = (mean_treatment - mean_control) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval
        se = np.sqrt(
            np.var(control_values) / len(control_values) +
            np.var(treatment_values) / len(treatment_values)
        )
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_lower = (mean_treatment - mean_control) - z_crit * se
        ci_upper = (mean_treatment - mean_control) + z_crit * se
        
        significant = p_value < alpha
        treatment_better = mean_treatment > mean_control
        
        return {
            "control_mean": mean_control,
            "treatment_mean": mean_treatment,
            "absolute_lift": mean_treatment - mean_control,
            "relative_lift": (mean_treatment - mean_control) / mean_control if mean_control != 0 else 0,
            "p_value": p_value,
            "significant": significant,
            "cohens_d": cohens_d,
            "confidence_interval": (ci_lower, ci_upper),
            "recommendation": (
                "SHIP treatment" if significant and treatment_better
                else "KEEP control" if significant and not treatment_better
                else "INCONCLUSIVE"
            ),
        }
    
    def bayesian_analysis(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_simulations: int = 100000,
    ) -> Dict:
        """
        Bayesian A/B test analysis.
        
        Returns: probability that treatment is better than control.
        More intuitive than p-values for decision making.
        """
        # Posterior distributions (Beta-Binomial conjugate)
        control_alpha = prior_alpha + control_successes
        control_beta = prior_beta + (control_total - control_successes)
        
        treatment_alpha = prior_alpha + treatment_successes
        treatment_beta = prior_beta + (treatment_total - treatment_successes)
        
        # Monte Carlo simulation
        control_samples = np.random.beta(control_alpha, control_beta, n_simulations)
        treatment_samples = np.random.beta(treatment_alpha, treatment_beta, n_simulations)
        
        # P(treatment > control)
        prob_treatment_better = np.mean(treatment_samples > control_samples)
        
        # Expected lift
        lift_samples = (treatment_samples - control_samples) / control_samples
        expected_lift = np.mean(lift_samples)
        
        # Credible interval for lift
        ci_lower = np.percentile(lift_samples, 2.5)
        ci_upper = np.percentile(lift_samples, 97.5)
        
        return {
            "prob_treatment_better": prob_treatment_better,
            "expected_relative_lift": expected_lift,
            "credible_interval_95": (ci_lower, ci_upper),
            "recommendation": (
                "SHIP treatment" if prob_treatment_better > 0.95
                else "KEEP control" if prob_treatment_better < 0.05
                else "CONTINUE experiment (insufficient certainty)"
            ),
            "risk_of_shipping_worse": 1 - prob_treatment_better,
        }
```

### ML-Specific A/B Challenges

```yaml
ML_Specific_Challenges:
  novelty_effect:
    what: "New model gets better metrics initially because it's 'new/different'"
    example: "New recommendations show different items → users click more (curiosity)"
    danger: "Metric decays after novelty wears off → you shipped based on temporary boost"
    mitigation: "Run experiment for 2+ weeks, look for metric decay over time"
    
  primacy_effect:
    what: "Users stick with what they know (old model) — new model looks worse initially"
    example: "New UI/recommendations confuse users at first → engagement drops"
    danger: "Kill a good model too early because metrics dipped in first days"
    mitigation: "Run for 2+ weeks, look for metric improvement over time"
    
  feedback_loops:
    what: "Model influences what data it sees, creating self-reinforcing cycles"
    example: "Recommendation model promotes item A → A gets more clicks → model learns A is popular → promotes A more"
    danger: "Treatment model creates its own positive signal (circular validation)"
    mitigation: "Use counterfactual evaluation, long-term holdout groups"
    
  network_effects:
    what: "Treatment user's experience depends on what control users see"
    example: "Social feed algorithm shows user A different content → A posts differently → affects user B in control group"
    danger: "SUTVA (Stable Unit Treatment Value Assumption) violated — groups not independent"
    mitigation: "Cluster randomization (randomize at network cluster level, not user level)"
    
  metric_sensitivity:
    what: "The metric you're testing might not be sensitive enough to detect ML improvements"
    example: "Recommendation model improves from 89% to 91% accuracy, but conversion rate barely moves"
    danger: "Declare 'no difference' when model IS better but metric can't detect it"
    mitigation: "Use more sensitive proxy metrics, increase sample size, use better evaluation metrics"
    
  multiple_testing:
    what: "Testing many metrics/models simultaneously inflates false positive rate"
    example: "Test 20 metrics → expect 1 'significant' by chance at α=0.05"
    danger: "Celebrate a win that's actually statistical noise"
    mitigation: "Bonferroni correction, FDR (False Discovery Rate) control, pre-register primary metric"
```

### Guardrail Metrics

```yaml
Guardrail_Metrics:
  what: "Metrics that must NOT degrade during an experiment"
  purpose: "Catch unintended negative consequences of model changes"
  
  common_guardrails:
    latency: "p95 latency must not increase > 10%"
    error_rate: "Error rate must not increase > 0.1%"
    crash_rate: "App crash rate must not increase"
    revenue: "Revenue must not decrease > 1% (for non-revenue experiments)"
    satisfaction: "User satisfaction proxy must not decrease"
    fairness: "Per-group metrics must not diverge"
    
  implementation:
    check: "After experiment concludes, verify ALL guardrails before shipping"
    action: "If primary metric improves but guardrail degrades → DO NOT SHIP"
    exception: "Only ship with guardrail violation if explicitly accepted by stakeholder"
```

---

## How It Works in Practice

### Experiment Lifecycle

```yaml
Lifecycle:
  1_design:
    - "Define hypothesis: 'New model improves conversion rate by ≥ 3%'"
    - "Choose primary metric, guardrails, and success criteria"
    - "Calculate sample size (power analysis)"
    - "Determine duration (≥ 14 days recommended)"
    
  2_launch:
    - "Deploy both models (control + treatment)"
    - "Start traffic split (gradually ramp: 1% → 5% → 50%)"
    - "Verify metrics are being collected correctly"
    - "Monitor for crashes/errors in first hours"
    
  3_monitor:
    - "Daily: check guardrail metrics (no degradation?)"
    - "Daily: check primary metric direction (trending positive?)"
    - "DO NOT make ship/kill decisions before minimum duration"
    - "Watch for novelty effects (metric decay)"
    
  4_analyze:
    - "After minimum duration: run full statistical analysis"
    - "Check significance (p < 0.05 or Bayesian P(better) > 0.95)"
    - "Check all guardrails pass"
    - "Check for segment-level effects (is one group hurt?)"
    
  5_decide:
    - "SHIP: significant improvement + guardrails pass + no segment regression"
    - "KILL: significant degradation OR guardrail violated"
    - "EXTEND: inconclusive (need more data/time)"
```

---

## Interview Tip

> When asked about A/B testing for ML: "A/B testing is how I validate ML models in production because offline metrics don't capture real-world effects (user behavior, feedback loops, system interactions). My process: (1) Design — power analysis to determine sample size. For a 5% baseline conversion rate and 3% MDE (Minimum Detectable Effect), I calculate required samples using proportion tests. I require at least 14 days to capture day-of-week effects and novelty wearing off. (2) Randomization — deterministic hashing (hash(user_id + experiment_id) mod 100) for consistent, independent assignment. Users always see the same variant. (3) Metrics — one pre-registered PRIMARY metric for the ship/kill decision, plus guardrail metrics that must not degrade (latency, error rate, revenue). Testing 20 metrics and celebrating whichever is significant is p-hacking. (4) Analysis — both frequentist (p-values, confidence intervals) and Bayesian (P(treatment > control)). Bayesian is more intuitive: 'There's a 97% probability the new model is better' vs. 'p < 0.05.' (5) ML-specific gotchas — novelty effects (new model looks great initially but decays), feedback loops (model influences its own training data), and network effects (control and treatment users influence each other). I mitigate by running longer, using counterfactual evaluation, and cluster randomization. Key rule: NEVER peek at results and stop early — that inflates false positive rates. If I need to monitor continuously, I use sequential testing with alpha spending functions."

---

## Common Mistakes

1. **Stopping early when results look good (peeking)** — Check results daily, stop as soon as p < 0.05. This inflates false positive rate from 5% to 20-30% (multiple comparisons over time). Solution: pre-commit to experiment duration. If you must peek, use sequential testing (O'Brien-Fleming boundaries or alpha spending).

2. **No power analysis (undersized experiment)** — Run experiment for 3 days with 1000 users. Results are "insignificant" so you keep the old model. But you never had enough power to detect a real 3% improvement. Solution: calculate required sample size BEFORE starting. If your daily traffic can't reach required sample size in 14 days, accept that you need a longer experiment.

3. **Celebrating guardrail violations** — New model increases conversion +5% but latency increases 3x. Ship it because conversion is the primary metric! Users start abandoning due to slow responses. Solution: guardrails are non-negotiable. Primary metric improvement with guardrail failure = DO NOT SHIP.

4. **Testing too many metrics** — Define 30 metrics, test all of them, find 2 are "significant." By chance alone (5% false positive rate × 30 tests), you'd expect 1.5 false positives. Solution: one pre-registered primary metric. Secondary metrics inform but don't drive the ship decision. Apply Bonferroni or FDR correction for multiple testing.

5. **Ignoring novelty and primacy effects** — Ship after 3 days because metrics look great. Actually: users were curious about new recommendations (novelty) and clicked more. After 2 weeks, engagement returns to (or below) baseline. Solution: minimum 14-day experiments. Look at metric trends over time (day 1-3 vs. day 10-14).

---

## Key Takeaways

- A/B testing validates ML models in production with real users and real outcomes
- Power analysis: calculate sample size BEFORE starting (based on baseline rate and MDE)
- Minimum duration: 14 days (day-of-week effects, novelty wearing off)
- Primary metric: ONE pre-registered metric for the ship/kill decision
- Guardrail metrics: must NOT degrade (latency, error rate, revenue, fairness)
- Randomization: deterministic hashing for consistent, independent user assignment
- Don't peek and stop early: use sequential testing if continuous monitoring is needed
- Bayesian analysis: "97% probability treatment is better" — more intuitive than p-values
- ML-specific: watch for novelty effects, feedback loops, and network effects
- Multiple testing correction: Bonferroni or FDR when evaluating many metrics
