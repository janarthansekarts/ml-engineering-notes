# ML Monitoring

## The Problem / Why This Matters

Traditional software either works or it doesn't — a broken service returns errors, crashes, or times out. ML systems fail silently. A model can serve predictions with perfect uptime, sub-millisecond latency, and zero errors — while those predictions are completely wrong because the data distribution shifted. This is why ML monitoring is fundamentally different from application monitoring. You need to watch not just system health (is the service up?), but also data health (is the input what the model expects?), model health (are predictions still accurate?), and business health (is the model still helping the KPI?). Without ML monitoring, model performance degrades for weeks or months before someone notices via business metrics — by then, significant damage is done. In 2026, monitoring has become even more complex as systems include traditional ML models, LLM (Large Language Model) outputs, RAG (Retrieval-Augmented Generation) retrieval quality, and agent behavior — each requiring different monitoring approaches.

---

## The Analogy

Think of ML monitoring like health monitoring for a patient:

- **System monitoring (traditional)** = Checking vital signs (heart rate, blood pressure, temperature). If these are normal, the patient is "alive and stable." Equivalent to: service uptime, latency, error rate.
- **Data drift monitoring** = Checking diet and lifestyle changes. The patient's vitals might be fine today, but their diet changed from vegetables to fast food. Eventually, vitals will deteriorate. Equivalent to: input distribution shifts that haven't affected output yet.
- **Model performance monitoring** = Regular health screenings (blood tests, imaging). Detect problems before symptoms appear. Equivalent to: measuring prediction accuracy against ground truth.
- **Business metric monitoring** = Quality of life assessment. The patient might be technically healthy by all tests, but their quality of life has decreased. Equivalent to: model is "accurate" but isn't improving the business KPI anymore.

---

## Deep Dive

### Monitoring Layers

```yaml
Monitoring_Layers:
  layer_1_system_operational:
    what: "Is the ML service running correctly?"
    metrics:
      - "Service uptime / availability"
      - "Inference latency (p50, p95, p99)"
      - "Throughput (predictions/second)"
      - "Error rate (failed predictions / total)"
      - "Resource utilization (GPU, CPU, memory)"
      - "Queue depth (if async prediction)"
    tools: "Prometheus, Grafana, Datadog, CloudWatch"
    alert_examples:
      - "p99 latency > 100ms for 5 minutes"
      - "Error rate > 1% for 3 minutes"
      - "GPU utilization < 10% for 30 minutes (waste)"
    detection_speed: "Immediate (seconds to minutes)"
    
  layer_2_data_quality:
    what: "Is the model receiving the data it expects?"
    metrics:
      - "Feature completeness (% missing values)"
      - "Feature freshness (time since last update)"
      - "Schema validation (unexpected types or values)"
      - "Volume anomalies (sudden spike or drop in request rate)"
    tools: "Great Expectations, Monte Carlo, Evidently (data checks)"
    alert_examples:
      - "Feature X missing for >5% of requests (was <1%)"
      - "Feature Y hasn't updated in 24 hours (should be hourly)"
      - "Request volume dropped 50% from expected"
    detection_speed: "Minutes to hours"
    
  layer_3_data_drift:
    what: "Has the input data distribution changed from training?"
    metrics:
      - "Distribution comparison (KS test, PSI, chi-squared)"
      - "Per-feature drift scores"
      - "Multivariate drift (joint distribution changes)"
      - "Embedding drift (for text/image features)"
    tools: "Evidently, Arize, WhyLabs, NannyML"
    alert_examples:
      - "PSI (Population Stability Index) > 0.2 for feature X"
      - "KS statistic > 0.1 for 3+ features simultaneously"
    detection_speed: "Hours to days"
    note: "Drift doesn't always mean degradation — but it's a leading indicator"
    
  layer_4_model_performance:
    what: "Is the model's prediction quality degrading?"
    metrics:
      - "Accuracy, AUC, F1, precision, recall (vs ground truth)"
      - "Prediction distribution shift (output distribution changed)"
      - "Confidence calibration (are probabilities still meaningful?)"
      - "Per-segment performance (any group degrading?)"
    challenge: "Ground truth is often DELAYED (fraud confirmed weeks later)"
    tools: "Evidently, Arize, NannyML, custom evaluation pipelines"
    alert_examples:
      - "AUC dropped from 0.94 to 0.89 on latest evaluation"
      - "Prediction distribution shifted (more high-confidence predictions)"
    detection_speed: "Days to weeks (depends on ground truth delay)"
    
  layer_5_business_impact:
    what: "Is the model still helping the business?"
    metrics:
      - "Business KPI (CTR, conversion rate, revenue per session)"
      - "User behavior (engagement, complaints, overrides)"
      - "Downstream impact (if model feeds another system)"
    tools: "Analytics platforms (Amplitude, Mixpanel), A/B test platforms"
    alert_examples:
      - "CTR dropped 10% week-over-week"
      - "Customer complaint rate about recommendations doubled"
    detection_speed: "Days to weeks"
```

### Data Drift Detection

```yaml
Data_Drift:
  definition: "Input data distribution in production differs from training data distribution"
  
  types:
    feature_drift:
      what: "Individual feature distributions changed"
      example: "Average transaction amount shifted from $50 to $150 (holiday season)"
      detection: "Per-feature statistical tests"
      
    concept_drift:
      what: "Relationship between features and target changed"
      example: "Previously, high transaction amount = high fraud risk. Now it doesn't (genuine luxury purchases increased)"
      detection: "Performance degradation even without feature drift"
      harder: "Can only detect with ground truth labels"
      
    covariate_shift:
      what: "Input distribution changed but conditional distribution P(Y|X) unchanged"
      example: "New customer demographic (younger users), but fraud patterns within each group unchanged"
      
  detection_methods:
    statistical_tests:
      ks_test:
        full_name: "Kolmogorov-Smirnov Test"
        type: "Numerical features"
        measures: "Maximum distance between two cumulative distributions"
        threshold: "p-value < 0.05 or D-statistic > 0.1"
        
      psi:
        full_name: "Population Stability Index"
        type: "Any feature (binned)"
        measures: "Divergence between reference and current distributions"
        interpretation: "PSI < 0.1: stable, 0.1-0.2: moderate shift, > 0.2: significant drift"
        
      chi_squared:
        full_name: "Chi-Squared Test"
        type: "Categorical features"
        measures: "Whether category frequencies differ from expected"
        
      wasserstein_distance:
        type: "Numerical features"
        measures: "Earth mover's distance between distributions"
        advantage: "Captures magnitude of shift (not just presence)"
        
    windowed_comparison:
      reference: "Training data distribution (or production baseline window)"
      current: "Latest production data (sliding window: last hour/day/week)"
      comparison: "Apply statistical tests between reference and current"
      frequency: "Hourly or daily depending on traffic volume"
```

### Monitoring Tools

```yaml
Monitoring_Tools:
  evidently:
    description: "Open-source ML monitoring and observability"
    capabilities:
      - "Data drift detection (statistical tests, visualizations)"
      - "Model quality monitoring (when ground truth available)"
      - "Data quality checks (missing, outliers, schema)"
      - "Target drift (output distribution monitoring)"
    deployment: "Library (generate reports) or service (continuous monitoring)"
    strengths: "Open-source, comprehensive, good visualizations, easy to start"
    
  arize:
    description: "Commercial ML observability platform"
    capabilities:
      - "Embedding drift (for NLP and computer vision models)"
      - "Performance tracing (individual prediction debugging)"
      - "Automatic drift detection with smart alerting"
      - "LLM monitoring (hallucination detection, quality scoring)"
    strengths: "Rich UI, automatic insights, LLM support, enterprise features"
    
  whylabs:
    description: "AI observability platform — profile-based monitoring"
    capabilities:
      - "Data profiling (statistical profiles of each feature)"
      - "Drift detection (profile comparison over time)"
      - "Anomaly detection on profiles"
      - "Lightweight integration (just log profiles)"
    strengths: "Low overhead (profiles vs raw data), privacy-preserving"
    
  nannyml:
    description: "Performance estimation WITHOUT ground truth"
    capabilities:
      - "CBPE (Confidence-Based Performance Estimation)"
      - "DLE (Direct Loss Estimation)"
      - "Estimate performance degradation before labels arrive"
    strengths: "Critical for systems with delayed ground truth (fraud, churn)"
    
  custom_monitoring:
    when: "Specific business logic, custom metrics, integration with existing stack"
    stack: "Prometheus (metrics) + Grafana (dashboards) + custom scripts (drift computation)"
    implementation:
      - "Instrument model serving code to emit custom metrics"
      - "Periodic batch jobs compute drift and performance metrics"
      - "Results pushed to Prometheus or stored in data warehouse"
      - "Grafana dashboards with alerting rules"
```

---

## How It Works in Practice

### Monitoring Setup Example

```yaml
Example:
  model: "Real-time fraud detection"
  monitoring_stack: "Evidently + Prometheus + Grafana + PagerDuty"
  
  layer_1_operational:
    metrics:
      - "fraud_model_latency_ms (histogram)"
      - "fraud_model_requests_total (counter)"
      - "fraud_model_errors_total (counter)"
      - "fraud_model_gpu_utilization (gauge)"
    alerts:
      - "p99 latency > 50ms for 3 min → Slack warning"
      - "Error rate > 0.1% for 2 min → PagerDuty page"
      
  layer_2_data_quality:
    checks:
      - "Feature completeness: all 15 features present in >99% of requests"
      - "Feature freshness: streaming features <30s old, batch features <24h old"
      - "Volume: request rate within 2 standard deviations of expected"
    frequency: "Every 5 minutes"
    alerts:
      - "Any feature missing in >5% of requests → Slack alert"
      - "Streaming feature stale >5 minutes → PagerDuty page"
      
  layer_3_drift:
    reference: "Training data distribution (updated each retraining)"
    current_window: "Last 1 hour of production data"
    metrics:
      - "PSI per feature (15 features)"
      - "Multivariate drift score (embedding-based)"
    frequency: "Hourly"
    alerts:
      - "PSI > 0.2 for any feature → Slack alert + investigation"
      - "3+ features drifting simultaneously → Slack alert + trigger evaluation"
      
  layer_4_performance:
    challenge: "Fraud labels confirmed 7-30 days after transaction"
    approach:
      proxy_metrics:
        - "Prediction score distribution (immediate — no labels needed)"
        - "Alert rate (% flagged as fraud — should be stable)"
        - "NannyML performance estimation (CBPE)"
      delayed_ground_truth:
        - "Weekly evaluation on confirmed fraud labels (7-day-old data)"
        - "Monthly deep evaluation on 30-day-old data"
    alerts:
      - "Alert rate changed >20% from baseline → investigate"
      - "Weekly AUC dropped >2% → trigger retraining"
      - "NannyML estimates performance below 0.90 → early warning"
      
  layer_5_business:
    metrics:
      - "Fraud caught (true positives) — should increase or stay stable"
      - "False blocks (false positives) — should decrease or stay stable"
      - "Customer friction (legitimate users blocked)"
      - "Dollar amount of fraud prevented"
    frequency: "Daily report, weekly deep dive"
    alerts: "Fraud losses increase >15% week-over-week → executive escalation"
```

---

## Interview Tip

> When asked about ML monitoring: "I implement monitoring across five layers: (1) Operational — latency, throughput, errors, resource utilization (catches infrastructure failures). (2) Data quality — feature completeness, freshness, schema violations (catches pipeline issues). (3) Data drift — statistical tests (PSI, KS) comparing production vs training distributions (leading indicator of degradation). (4) Model performance — accuracy metrics against ground truth when available, proxy metrics and estimation (NannyML) when ground truth is delayed. (5) Business impact — KPI tracking to confirm the model is still helping. The key insight: layer 1 detects if the service is broken, layer 3 predicts FUTURE degradation (drift), and layer 4 confirms ACTUAL degradation. I use Evidently for drift detection, Prometheus/Grafana for operational metrics, and custom evaluation pipelines for performance tracking. Alert design is critical — too many alerts = alert fatigue, too few = missed problems. I set different severity levels: drift alone = warning (investigate), drift + performance drop = critical (retrain)."

---

## Common Mistakes

1. **Only monitoring operational metrics** — Latency is 5ms, throughput is 10K QPS, zero errors. Everything looks great! Except the model's accuracy dropped from 94% to 82% because data drifted. Operational health ≠ prediction health.

2. **Alerting on every drift signal** — Data distributions naturally fluctuate (weekday vs weekend, seasonal patterns). If you alert on every statistical test that fires, you get alert fatigue. Set meaningful thresholds and use windowed comparisons.

3. **No ground truth strategy** — "We'll monitor performance" but no plan for how to get labels. In many systems, ground truth is delayed (fraud confirmed weeks later) or never available (recommendation quality). Plan for proxy metrics and delayed evaluation.

4. **Reference distribution never updated** — Comparing current data against training data from 12 months ago. Natural drift accumulates. Update reference distribution after each retraining cycle.

5. **Monitoring model in isolation** — Model performance dropped, but you don't know why because you're not monitoring the upstream data pipelines. Was it a data pipeline failure? Schema change? Missing data source? Monitor the entire system, not just the model.

---

## Key Takeaways

- ML monitoring has 5 layers: operational → data quality → drift → performance → business impact
- Models fail silently — good latency and zero errors doesn't mean good predictions
- Data drift is a leading indicator of performance degradation — detect it before metrics drop
- PSI (Population Stability Index) and KS test are standard drift detection methods
- Ground truth delay is the biggest challenge — use proxy metrics and estimation (NannyML) as early warning
- Tools: Evidently (open-source, comprehensive), Arize (commercial, LLM support), WhyLabs (profile-based)
- Update reference distribution after each retraining — don't compare against year-old baselines
- Alert design: drift alone = investigate, drift + performance = act, performance alone = urgent
- Monitor the full pipeline (data sources → features → model → predictions), not just the model
- For LLMs: monitor hallucination rates, response quality scores, token costs, and retrieval relevance
