# A/B Testing Models in Production

## The Problem / Why This Matters

You've trained a new model version that shows 3% improvement on your offline evaluation set. Should you deploy it to 100% of traffic immediately? Absolutely not. Offline metrics (accuracy, perplexity, BLEU) don't always correlate with business outcomes (revenue, user engagement, retention). A model with higher accuracy might generate longer responses that users don't read, produce more "correct" but less helpful answers, or introduce unexpected failure modes on edge cases not in your eval set. A/B testing (also called split testing or online experimentation) deploys the new model to a small percentage of users while the majority continues using the current model. By comparing real user behavior between groups, you measure the actual business impact before committing to the new version. For ML models specifically, A/B testing is complicated by: (1) non-deterministic outputs (same input, different output each time), (2) delayed feedback (did the recommendation lead to a purchase days later?), (3) statistical complexity (continuous metrics vs binary click/no-click), and (4) network effects (one user's results affecting others). In 2026, sophisticated experimentation is what separates companies that iterate rapidly from those that ship regressions they don't notice for months.

---

## The Analogy

Think of A/B testing models like clinical drug trials:

- **Phase 1 (canary)** = Give the new drug to 5 patients, monitor intensively for adverse reactions. If anyone gets seriously sick, stop immediately. This is deploying to 1% of traffic with automatic rollback on error rate spikes.
- **Phase 2 (limited rollout)** = If Phase 1 is safe, expand to 100 patients. Measure both safety AND efficacy. This is 10-25% traffic with full metric comparison.
- **Phase 3 (full trial)** = Large-scale randomized trial (1000+ patients) with statistical rigor. Proves the drug works better than placebo. This is 50/50 traffic split with statistical significance testing.
- **FDA approval (full rollout)** = After Phase 3 proves efficacy, approve for everyone. This is 100% traffic on the new model.
- **The control group** = Patients who get the placebo (current model). Without them, you can't prove the drug (new model) actually helped — maybe patients would have improved anyway (maybe metrics would have improved due to seasonality).

---

## Deep Dive

### Traffic Splitting Strategies

```yaml
Traffic_Splitting:
  canary_deployment:
    what: "Route small % of traffic to new model, rest to current model"
    typical_split: "1-5% canary, 95-99% stable"
    purpose: "Detect catastrophic failures (errors, latency spikes, crashes)"
    duration: "Hours to 1 day"
    success_criteria: "No increase in error rate, latency within bounds"
    rollback: "Automatic if error rate exceeds threshold"
    
  gradual_rollout:
    what: "Progressively increase traffic to new model"
    stages: "1% → 5% → 10% → 25% → 50% → 100%"
    purpose: "Build confidence progressively, catch issues at each stage"
    duration: "Days to weeks"
    gate_at_each_stage: "Statistical confidence that key metrics aren't degraded"
    
  a_b_test:
    what: "True randomized experiment with equal-ish groups for statistical measurement"
    typical_split: "50/50 or 80/20 (depends on risk tolerance)"
    purpose: "Measure CAUSAL impact of new model on business metrics"
    duration: "1-4 weeks (enough for statistical significance)"
    key_requirement: "Random, consistent user assignment (same user always sees same variant)"
    
  multi_armed_bandit:
    what: "Dynamically allocate more traffic to better-performing variant"
    how: "Start 50/50, observe rewards, shift traffic toward winner"
    algorithms: "Thompson Sampling, UCB (Upper Confidence Bound), Epsilon-Greedy"
    advantage: "Reduces regret (less traffic to losing variant)"
    disadvantage: "Harder to reach statistical significance (non-fixed allocation)"
    use_when: "High opportunity cost of sending traffic to inferior variant"
```

### User Assignment

```yaml
User_Assignment:
  key_requirement: |
    Each user must CONSISTENTLY see the same model variant across their session.
    If user A sees model_v2 on request 1, they must see model_v2 on request 2.
    Otherwise: confounded results (can't attribute outcomes to a single variant).
    
  methods:
    hash_based:
      how: "hash(user_id + experiment_id) % 100 → bucket 0-99"
      assign: "bucket 0-4 → variant B (5%), bucket 5-99 → variant A (95%)"
      advantage: "Deterministic — same user always gets same variant, no storage needed"
      implementation: |
        import hashlib
        
        def get_variant(user_id: str, experiment_id: str, traffic_pct: int) -> str:
            """Deterministic variant assignment."""
            key = f"{user_id}:{experiment_id}"
            hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
            bucket = hash_val % 100
            return "treatment" if bucket < traffic_pct else "control"
            
    session_based:
      how: "Assign variant at session start, store in session cookie/token"
      advantage: "Works for anonymous users (no user_id needed)"
      disadvantage: "Different sessions may get different variants"
      
    feature_flag_service:
      how: "Centralized service (LaunchDarkly, Unleash, Flagsmith) manages assignment"
      advantage: "Rich targeting rules (by region, device, user segment)"
      tools: "LaunchDarkly, Statsig, Unleash, GrowthBook, custom"
      
  stratification:
    what: "Ensure experiment groups are balanced on key dimensions"
    dimensions: "Region, device type, user tenure, historical activity level"
    why: |
      If variant B accidentally gets 80% power users and variant A gets 80% new users,
      differences in metrics are due to user composition, not model quality.
    implementation: "Stratified random assignment or post-hoc adjustment"
```

### Statistical Analysis

```yaml
Statistical_Analysis:
  frequentist_approach:
    hypothesis_test:
      h0: "New model has same or worse performance than current model"
      h1: "New model has better performance"
      test: "Two-sample t-test (continuous metrics) or chi-square (binary metrics)"
      
    key_concepts:
      p_value:
        what: "Probability of observing this result if H0 is true"
        threshold: "p < 0.05 (5% significance level — industry standard)"
        meaning: "< 5% chance the improvement is due to random chance"
        
      statistical_power:
        what: "Probability of detecting a real effect if one exists"
        target: "80% power (standard)"
        meaning: "80% chance of catching a true 2% improvement"
        
      sample_size:
        what: "Number of observations needed for desired power"
        formula: "Depends on effect size, variance, significance level, power"
        practical: |
          For detecting 2% relative improvement in conversion rate (baseline 5%):
          ~50,000 users per variant (at 80% power, 5% significance)
          
      mde:
        name: "MDE (Minimum Detectable Effect)"
        what: "Smallest improvement you can reliably detect given your sample size"
        trade_off: "Smaller MDE requires more samples (longer experiment duration)"
        
    multiple_comparisons:
      problem: "Testing 10 metrics simultaneously → 40% chance of at least one false positive"
      solutions:
        - "Bonferroni correction (divide alpha by number of tests — conservative)"
        - "FDR (False Discovery Rate) control (Benjamini-Hochberg — less conservative)"
        - "Pre-register primary metric (one primary, rest exploratory)"
        
  bayesian_approach:
    what: "Compute probability that variant B is better than variant A"
    output: "P(B > A) = 95% (95% probability new model is better)"
    advantage: "Intuitive interpretation, works with smaller samples"
    disadvantage: "Requires prior specification, computationally heavier"
    tools: "PyMC, Stan, Statsig (built-in Bayesian analysis)"
    
  sequential_testing:
    what: "Check results continuously (not just at end) with proper statistical correction"
    why: "Business wants to stop early if winner is clear or loser is obvious"
    methods: "Sequential probability ratio test (SPRT), always-valid confidence intervals"
    advantage: "Can stop experiment early while maintaining statistical validity"
    danger: "Peeking without correction inflates false positive rate massively"
```

### ML-Specific A/B Testing Challenges

```yaml
ML_Specific_Challenges:
  non_determinism:
    problem: "Same prompt to same model gives different outputs (temperature > 0)"
    impact: "Higher variance in metrics → need more samples for significance"
    mitigation: "Fix temperature=0 for experiment (deterministic outputs)"
    alternative: "Accept higher variance, run experiment longer"
    
  delayed_feedback:
    problem: "Recommendation shown today → purchase happens in 3 days"
    impact: "Can't evaluate model for days after exposure"
    mitigation: "Use proxy metrics (clicks, add-to-cart) for fast signal + delayed metrics (purchase, retention) for final decision"
    
  long_tail_effects:
    problem: "Model performs well on average but terribly on 1% of inputs"
    impact: "Average metrics look good but 1% of users have awful experience"
    mitigation: "Monitor P95/P99 of quality metrics, not just mean. Slice by user segment."
    
  novelty_effect:
    problem: "Users engage more with new model simply because it's different"
    impact: "Initial metrics inflated, decay to true level over days/weeks"
    mitigation: "Run experiment for 2+ weeks, analyze excluding first 3 days"
    
  carryover_effects:
    problem: "User exposed to model A develops habits, then switched to model B"
    impact: "Model B's metrics reflect habits learned from model A"
    mitigation: "Only include new users or users with wash-out period between variants"
    
  network_effects:
    problem: "User A's recommendations affect User B (social networks, marketplaces)"
    impact: "Treatment and control groups contaminate each other"
    mitigation: "Cluster randomization (assign groups of connected users together)"
```

### Multi-Armed Bandit (MAB)

```python
# Thompson Sampling for model selection
# Dynamically allocate more traffic to better-performing model

import numpy as np
from dataclasses import dataclass

@dataclass
class ModelVariant:
    name: str
    # Beta distribution parameters (for binary reward: click/no-click)
    alpha: float = 1.0  # Successes + 1 (prior)
    beta: float = 1.0   # Failures + 1 (prior)
    
    def sample(self) -> float:
        """Sample from posterior distribution."""
        return np.random.beta(self.alpha, self.beta)
    
    def update(self, reward: bool):
        """Update posterior with observed reward."""
        if reward:
            self.alpha += 1
        else:
            self.beta += 1
    
    @property
    def mean_reward(self) -> float:
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def total_samples(self) -> int:
        return int(self.alpha + self.beta - 2)  # Subtract prior


class ThompsonSamplingRouter:
    """Route requests to models using Thompson Sampling."""
    
    def __init__(self, model_names: list[str]):
        self.variants = {name: ModelVariant(name=name) for name in model_names}
    
    def select_model(self) -> str:
        """Select model for next request (Thompson Sampling)."""
        # Sample from each model's posterior
        samples = {name: variant.sample() for name, variant in self.variants.items()}
        # Route to model with highest sample
        return max(samples, key=samples.get)
    
    def record_outcome(self, model_name: str, reward: bool):
        """Record whether the interaction was successful."""
        self.variants[model_name].update(reward)
    
    def get_stats(self) -> dict:
        """Get current allocation statistics."""
        return {
            name: {
                "mean_reward": f"{v.mean_reward:.4f}",
                "samples": v.total_samples,
                "traffic_share": f"{v.total_samples / sum(vv.total_samples for vv in self.variants.values()) * 100:.1f}%"
            }
            for name, v in self.variants.items()
        }


# Usage
router = ThompsonSamplingRouter(["model_v1", "model_v2", "model_v3"])

# Over time, Thompson Sampling automatically:
# 1. Explores all variants initially (roughly equal traffic)
# 2. Exploits better-performing variants (shifts traffic toward winner)
# 3. Balances exploration/exploitation optimally

for request in incoming_requests:
    selected_model = router.select_model()
    response = serve_with_model(request, selected_model)
    reward = collect_user_feedback(response)  # Click, thumbs up, purchase, etc.
    router.record_outcome(selected_model, reward)
```

### Infrastructure for A/B Testing

```yaml
Infrastructure:
  traffic_routing:
    istio_virtual_service:
      what: "Kubernetes service mesh with traffic splitting"
      config: |
        apiVersion: networking.istio.io/v1beta1
        kind: VirtualService
        metadata:
          name: llm-service
        spec:
          hosts:
            - llm-service
          http:
            - match:
                - headers:
                    x-experiment-group:
                      exact: "treatment"
              route:
                - destination:
                    host: llm-service-v2
                    port:
                      number: 8080
            - route:
                - destination:
                    host: llm-service-v1
                    port:
                      number: 8080
                  weight: 90
                - destination:
                    host: llm-service-v2
                    port:
                      number: 8080
                  weight: 10
                  
    application_level:
      what: "Route in application code based on user assignment"
      advantage: "More flexible targeting (user attributes, context)"
      disadvantage: "Both models must be accessible from same service"
      
  metrics_collection:
    what: "Track per-variant metrics for statistical comparison"
    events:
      - "model_variant: which model served this request"
      - "response_time: latency (TTFT + generation time)"
      - "response_quality: user rating, thumbs up/down"
      - "engagement: clicks, time spent, follow-up questions"
      - "business: conversion, revenue, retention"
    tools: "Statsig, Amplitude, Mixpanel, custom event pipeline"
    
  guardrails:
    automatic_rollback:
      triggers:
        - "Error rate > 5% (vs control's error rate)"
        - "P95 latency > 2× control's P95"
        - "Safety filter triggers > 3× control rate"
      action: "Route 100% traffic back to control immediately"
      alert: "Page on-call engineer"
```

---

## How It Works in Practice

### End-to-End Experimentation Workflow

```yaml
Workflow:
  step_1_define:
    hypothesis: "Model v2 (fine-tuned on recent data) improves user satisfaction"
    primary_metric: "User thumbs-up rate"
    guardrail_metrics: ["Error rate", "P95 latency", "Safety violations"]
    sample_size_calculation: "Need 20,000 users per variant for 2% MDE at 80% power"
    duration: "14 days (accounts for weekly patterns)"
    
  step_2_canary:
    traffic: "1% to model v2"
    duration: "24 hours"
    check: "No error rate increase, latency within bounds"
    gate: "Pass → proceed to A/B test"
    
  step_3_ab_test:
    traffic: "10% treatment (model v2), 90% control (model v1)"
    duration: "14 days"
    monitoring: "Daily check on guardrail metrics"
    analysis: "Sequential testing with alpha spending"
    
  step_4_decision:
    if_significant_positive: "Ramp to 50%, confirm, then 100%"
    if_not_significant: "Extend experiment or accept no difference"
    if_significant_negative: "Roll back, investigate why offline metrics didn't predict"
    
  step_5_post_mortem:
    document: "What offline metrics predicted vs actual online impact"
    learn: "Update evaluation methodology for future experiments"
```

---

## Interview Tip

> When asked about A/B testing ML models: "I follow a multi-stage approach: (1) Canary deployment (1% traffic, 24 hours) — catch catastrophic failures (error spikes, latency regression, safety violations) with automatic rollback. (2) A/B test (10-50% traffic, 1-4 weeks) — measure causal impact on business metrics with proper statistical rigor. Key implementation details: hash-based user assignment (deterministic, no storage needed — hash(user_id + experiment_id) % 100), stratification by key dimensions to ensure balanced groups, and sequential testing to allow valid early stopping. ML-specific challenges I account for: non-determinism (fix temperature for experiment), delayed feedback (proxy metrics for fast signal, business metrics for final decision), novelty effect (run 2+ weeks, exclude first 3 days from analysis), and P99 monitoring (average can be good while 1% of users have terrible experience). For multi-variant testing with high opportunity cost, I use Thompson Sampling (multi-armed bandit) — it automatically shifts traffic toward the better variant while maintaining exploration."

---

## Common Mistakes

1. **Deploying to 100% based on offline metrics alone** — Offline eval showed 5% accuracy improvement, so you ship to everyone. But the new model generates 40% longer responses, increasing latency and cost, while users actually preferred concise answers. Always validate with online A/B test.

2. **Peeking at results daily without statistical correction** — Checking p-values every day and declaring a winner as soon as p < 0.05. This massively inflates false positive rates (20-30% instead of 5%). Use sequential testing methods (always-valid confidence intervals) if you need to check frequently.

3. **Inconsistent user assignment** — User sees model A on Monday, model B on Tuesday. You can't attribute their weekly behavior to either model. Use hash-based assignment: same user always gets same variant for the experiment duration.

4. **Only monitoring average metrics** — Average response quality is 4.2/5 for both variants, so you call it neutral. But variant B has a bimodal distribution: 80% of responses score 4.8 and 20% score 2.0 (vs variant A's consistent 4.2). The 20% having terrible experiences will churn. Always check P95/P99 and distribution shape.

5. **Not accounting for sample size in experiment duration** — Running an experiment for "2 weeks" by convention when your traffic volume could reach significance in 3 days or might need 6 weeks. Calculate required sample size FIRST, then determine duration from your traffic volume.

---

## Key Takeaways

- Never deploy new models to 100% based on offline metrics alone — A/B test in production
- Multi-stage: canary (1%, catch crashes) → A/B test (10-50%, measure business impact) → full rollout
- Hash-based user assignment: deterministic, consistent, no storage — hash(user_id + experiment_id) % 100
- Primary metric: business outcome (revenue, engagement), not model metric (accuracy, perplexity)
- Statistical rigor: pre-calculate sample size, use sequential testing for early stopping, correct for multiple comparisons
- ML-specific: account for non-determinism, delayed feedback, novelty effect, and P99 monitoring
- Multi-armed bandit: Thompson Sampling for dynamic traffic allocation when opportunity cost is high
- Automatic rollback: guardrail metrics (error rate, latency, safety) trigger immediate rollback
- Run experiments 2+ weeks to capture weekly patterns and wash out novelty effects
- Document findings: build institutional knowledge of what offline metrics predict online impact
