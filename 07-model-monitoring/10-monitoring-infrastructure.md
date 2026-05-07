# Monitoring Infrastructure and Tools

## The Problem / Why This Matters

Building ML monitoring from scratch is expensive and error-prone — you need prediction logging, statistical drift detection, dashboard visualization, alerting integration, and ground truth joining. A mature monitoring infrastructure combines specialized tools (Evidently AI for drift detection, Whylabs for data profiling, Arize for observability) with general-purpose platforms (Prometheus for metrics, Grafana for dashboards, PagerDuty for alerting). The challenge: choosing the right tools for your scale and needs, integrating them into a coherent system, avoiding vendor lock-in while getting value quickly, and managing the cost of storing and processing millions of predictions. In 2026, the landscape has matured significantly — open-source tools like Evidently AI provide production-ready drift detection, managed platforms like Arize and Whylabs offer full-stack observability, and cloud providers (SageMaker Model Monitor, Vertex AI Model Monitoring) integrate monitoring into their ML platforms. The engineering decision: build vs. buy vs. open-source composition — depends on your scale, budget, and team capabilities.

---

## The Analogy

Think of monitoring infrastructure like the instrumentation in an aircraft cockpit:

- **Prediction logging** = Flight data recorder (black box). Records everything — altitude, speed, heading, engine parameters — so you can reconstruct what happened.
- **Drift detection** = Autopilot anomaly detection. "We're deviating from expected flight path." Not a crisis yet, but worth investigating.
- **Dashboards** = Instrument panel. Pilot glances at it to confirm everything's normal. Gauges show current state at a glance.
- **Alerting** = Warning systems (GPWS, TCAS). Only activates for genuine threats. Fires too often → pilot ignores it. Fires too rarely → crash.
- **Investigation tools** = Black box analysis after an incident. Deep dive into what went wrong, when, and why.

You need all of these working together, integrated into one coherent system where each component feeds the others.

---

## Deep Dive

### Tool Landscape (2026)

```yaml
Tools:
  open_source:
    evidently_ai:
      what: "ML monitoring and testing — drift detection, data quality, model performance"
      strengths:
        - "Comprehensive drift detection (PSI, KS, Wasserstein, Jensen-Shannon)"
        - "Beautiful HTML reports and dashboards"
        - "Test suites for CI/CD (validate before deploy)"
        - "Works with any model framework"
        - "Lightweight (pip install evidently)"
      deployment: "Python library, self-hosted dashboard, or Evidently Cloud"
      best_for: "Teams wanting open-source drift detection with minimal setup"
      integration: "Pandas DataFrames in, reports/metrics out"
      
    whylogs:
      what: "Data logging and profiling library (by WhyLabs)"
      strengths:
        - "Lightweight statistical profiling (mergeable summaries)"
        - "Constant memory regardless of data size"
        - "Schema inference, distribution tracking"
        - "Integration with WhyLabs platform"
      deployment: "Python library (logging) + WhyLabs (visualization/alerting)"
      best_for: "High-volume data profiling without storing raw data"
      
    prometheus_grafana:
      what: "General-purpose metrics collection and visualization"
      strengths:
        - "Industry standard for operational monitoring"
        - "Flexible alerting (Alertmanager)"
        - "Mature ecosystem (exporters for everything)"
      limitation: "Not ML-specific — you build the ML logic yourself"
      use_for: "System metrics (latency, throughput, errors), custom ML metrics"
      
    mlflow:
      what: "ML lifecycle platform (tracking, models, deployment)"
      monitoring: "Basic model tracking, metric logging, artifact versioning"
      limitation: "Not a monitoring tool per se — more for experiment tracking"
      complement: "Pair with Evidently or custom monitoring for production observability"
      
  managed_platforms:
    arize:
      what: "ML observability platform — monitoring, debugging, root cause analysis"
      strengths:
        - "Automatic drift detection with smart alerting"
        - "Embedding drift visualization (UMAP projections)"
        - "Performance tracing (trace individual predictions through pipeline)"
        - "LLM monitoring (hallucination, toxicity, relevance)"
        - "Integrated with major ML frameworks"
      pricing: "Usage-based (per prediction logged)"
      best_for: "Teams wanting full-stack managed observability"
      
    whylabs:
      what: "AI observability platform built on whylogs profiles"
      strengths:
        - "Efficient profiling (no raw data storage needed)"
        - "Data quality monitoring"
        - "ML monitoring with customizable policies"
        - "Privacy-preserving (profiles, not raw data)"
      pricing: "Based on profile volume"
      best_for: "Privacy-conscious teams, high-volume data monitoring"
      
    datadog_ml:
      what: "ML monitoring as part of Datadog observability platform"
      strengths:
        - "Unified with application monitoring (traces, logs, metrics)"
        - "ML-specific dashboards and alerts"
        - "Infrastructure monitoring included"
      limitation: "Less ML-specific depth than Arize/WhyLabs"
      best_for: "Teams already on Datadog wanting ML monitoring in same platform"
      
  cloud_provider:
    sagemaker_model_monitor:
      what: "AWS native model monitoring"
      capabilities: "Data quality, model quality, bias drift, feature attribution drift"
      integration: "Tight with SageMaker endpoints"
      limitation: "AWS-only, less flexible than open-source alternatives"
      
    vertex_ai_model_monitoring:
      what: "GCP native model monitoring"
      capabilities: "Training-serving skew, prediction drift, feature attribution"
      integration: "Tight with Vertex AI endpoints"
      limitation: "GCP-only, less customizable"
      
    azure_ml_monitoring:
      what: "Azure native model monitoring"
      capabilities: "Data drift, model performance, data quality"
      integration: "Azure ML workspace"
      limitation: "Azure-only"
```

### Architecture Patterns

```yaml
Architecture_Patterns:
  pattern_1_lightweight:
    name: "Evidently + Prometheus + Grafana"
    description: "Open-source stack for medium-scale teams"
    components:
      logging: "Custom prediction logger → S3/BigQuery"
      drift_detection: "Evidently (scheduled job computes metrics)"
      metrics_store: "Prometheus (custom metrics pushed)"
      dashboards: "Grafana (ML-specific panels)"
      alerting: "Alertmanager → Slack/PagerDuty"
    scale: "Up to ~1M predictions/day"
    cost: "Infrastructure only (no licensing)"
    effort: "Medium setup effort, ongoing maintenance"
    
  pattern_2_managed:
    name: "Arize (or WhyLabs) + existing observability"
    description: "Managed platform for fast setup"
    components:
      logging: "Arize SDK (log predictions + features + labels)"
      drift_detection: "Arize platform (automatic)"
      dashboards: "Arize web UI"
      alerting: "Arize alerts → Slack/PagerDuty"
      system_monitoring: "Keep existing Datadog/New Relic for infra"
    scale: "Any scale (managed)"
    cost: "Platform fees ($$-$$$)"
    effort: "Low setup effort, minimal maintenance"
    
  pattern_3_custom:
    name: "Custom pipeline + composable tools"
    description: "Large-scale custom system"
    components:
      logging: "Kafka → Spark Streaming → Data Lake"
      drift_detection: "Custom Spark jobs using scipy/numpy"
      metrics_store: "Custom metrics tables (BigQuery/Redshift)"
      dashboards: "Custom (Looker/Superset/Streamlit)"
      alerting: "Custom alerting service → PagerDuty"
    scale: "Billions of predictions/day"
    cost: "Engineering time (expensive)"
    effort: "High setup effort, significant maintenance"
    when: "Very large scale, specific requirements, or regulated industries"
```

### Prediction Logging Infrastructure

```python
# Production prediction logging system

"""
Log every prediction with full context for downstream monitoring.
Must be: fast (not add latency), reliable (don't lose data), and complete.
"""

import json
from datetime import datetime
from typing import Optional
import asyncio


class PredictionLogger:
    """
    Async prediction logger that captures full prediction context.
    
    Design principles:
    - Non-blocking (don't add latency to prediction path)
    - Buffered (batch writes for efficiency)
    - Schema-validated (consistent structure)
    - Fault-tolerant (failures don't affect serving)
    """
    
    def __init__(
        self,
        sink: str = "kafka",  # "kafka", "kinesis", "bigquery", "s3"
        buffer_size: int = 100,
        flush_interval_seconds: int = 5,
    ):
        self.sink = sink
        self.buffer = []
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval_seconds
        
    async def log(
        self,
        request_id: str,
        model_name: str,
        model_version: str,
        features: dict,
        prediction: float,
        confidence: Optional[float] = None,
        latency_ms: float = 0,
        metadata: Optional[dict] = None,
    ):
        """
        Log a prediction asynchronously (non-blocking).
        Called from the prediction serving path.
        """
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": model_name,
            "model_version": model_version,
            "features": features,
            "prediction": prediction,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "metadata": metadata or {},
        }
        
        self.buffer.append(log_entry)
        
        # Flush if buffer is full
        if len(self.buffer) >= self.buffer_size:
            await self._flush()
    
    async def _flush(self):
        """Write buffered logs to sink."""
        if not self.buffer:
            return
            
        batch = self.buffer.copy()
        self.buffer.clear()
        
        try:
            if self.sink == "kafka":
                await self._write_kafka(batch)
            elif self.sink == "s3":
                await self._write_s3(batch)
            elif self.sink == "bigquery":
                await self._write_bigquery(batch)
        except Exception as e:
            # Log failures should NEVER affect serving
            # Write to dead letter queue for retry
            await self._write_dlq(batch, error=str(e))
    
    async def _write_kafka(self, batch: list):
        """Write prediction logs to Kafka topic."""
        for entry in batch:
            await self.kafka_producer.send(
                topic="ml-predictions",
                key=entry["request_id"].encode(),
                value=json.dumps(entry).encode(),
            )
    
    async def _write_s3(self, batch: list):
        """Write prediction logs to S3 (Parquet, partitioned by hour)."""
        # Group by hour partition
        hour = datetime.utcnow().strftime("%Y/%m/%d/%H")
        key = f"predictions/{hour}/batch_{datetime.utcnow().timestamp()}.json"
        
        await self.s3_client.put_object(
            Bucket="ml-prediction-logs",
            Key=key,
            Body=json.dumps(batch).encode(),
        )


# Downstream: monitoring pipeline reads from prediction logs
class MonitoringPipeline:
    """
    Consumes prediction logs, computes monitoring metrics.
    Runs as a scheduled job (hourly) or streaming (Flink/Spark Streaming).
    """
    
    def __init__(self, drift_detector, alert_system, metrics_store):
        self.drift_detector = drift_detector
        self.alert_system = alert_system
        self.metrics_store = metrics_store
    
    async def process_window(self, window_start: datetime, window_end: datetime):
        """Process one time window of predictions."""
        
        # 1. Read prediction logs for this window
        predictions = await self.read_predictions(window_start, window_end)
        
        # 2. Compute drift metrics
        drift_results = self.drift_detector.check_drift(
            current_data=self.extract_features(predictions)
        )
        
        # 3. Compute prediction distribution metrics
        pred_metrics = self.compute_prediction_metrics(predictions)
        
        # 4. Store metrics
        await self.metrics_store.write(
            timestamp=window_end,
            drift_results=drift_results,
            prediction_metrics=pred_metrics,
        )
        
        # 5. Evaluate alerts
        await self.alert_system.evaluate_all(drift_results, pred_metrics)
```

### Cost Management

```yaml
Cost_Considerations:
  storage:
    raw_predictions:
      volume: "1M predictions/day × 2KB each = 2 GB/day = 60 GB/month"
      retention: "90 days raw, then aggregate and archive"
      cost: "~$2/month on S3, more on managed databases"
      
    features_with_predictions:
      volume: "1M predictions/day × 10KB (with features) = 10 GB/day = 300 GB/month"
      strategy: "Store features for recent window only (7-30 days)"
      cost: "~$10-50/month on S3"
      
    metrics_aggregates:
      volume: "Tiny (hourly/daily summaries)"
      retention: "Indefinite (for trend analysis)"
      cost: "Negligible"
      
  compute:
    drift_detection:
      frequency: "Hourly"
      cost: "Small Spark job or single machine — $50-200/month"
      
    embedding_drift:
      frequency: "Daily"
      cost: "GPU needed for embedding — $100-500/month"
      
    statistical_tests:
      frequency: "Hourly"
      cost: "Minimal (scipy on single machine)"
      
  managed_platform_costs:
    arize: "$500-5000/month depending on prediction volume"
    whylabs: "$300-3000/month depending on profile volume"
    datadog_ml: "Included in existing Datadog plan (ML addon)"
    
  optimization:
    sampling: "Log 100% but compute metrics on 10% sample (for high volume)"
    profiling: "Use whylogs profiles instead of raw data (constant memory)"
    retention: "Aggregate after 7 days, archive after 90 days"
    tiered: "High-frequency monitoring for critical models, daily for others"
```

---

## How It Works in Practice

### Reference Architecture

```yaml
Reference_Architecture:
  prediction_path:
    flow: "Request → Model Server → Prediction + Explanation → Response"
    logging: "Async: prediction logged to Kafka (non-blocking)"
    latency_impact: "< 1ms (async fire-and-forget)"
    
  monitoring_path:
    flow: "Kafka → Spark Streaming → Compute Metrics → Write to TimescaleDB/Prometheus"
    frequency: "Near real-time (5-minute micro-batches)"
    outputs: "PSI scores, prediction stats, volume counts, latency percentiles"
    
  alerting_path:
    flow: "Metrics → Alert Rules Engine → PagerDuty/Slack"
    evaluation: "Every 5 minutes (matches monitoring frequency)"
    deduplication: "Same condition doesn't re-alert within 1 hour"
    
  dashboard_path:
    flow: "TimescaleDB → Grafana dashboards"
    refresh: "Auto-refresh every 1 minute"
    panels: "Drift heatmap, prediction distribution, latency, volume, alerts"
    
  investigation_path:
    flow: "Alert → Arize/Evidently UI → Drill down → Root cause"
    capability: "Filter by time, segment, feature. Compare windows. Trace individual predictions."
```

---

## Interview Tip

> When asked about monitoring infrastructure: "My monitoring architecture has three layers: (1) Prediction logging — async, non-blocking logger that captures every prediction with full feature context. Written to Kafka or directly to S3 Parquet (partitioned by hour). Critical: logging must NEVER add latency to serving or fail in a way that affects predictions. (2) Metrics computation — scheduled jobs (hourly for drift, daily for performance) that read prediction logs, compute PSI, KS tests, distribution statistics, and write aggregated metrics to Prometheus/TimescaleDB. For high-volume systems (10M+ predictions/day), I sample 10% for statistical tests (more than sufficient for reliable drift detection). (3) Dashboard + Alerting — Grafana dashboards for real-time visibility, Alertmanager for severity-based routing to Slack/PagerDuty. Tool selection depends on scale and team: small-medium teams → Evidently AI (open-source, excellent drift detection, minimal setup). Large teams with budget → Arize or WhyLabs (managed, LLM monitoring, embedding visualization). Enterprise on cloud → complement with SageMaker/Vertex Model Monitor (tight integration). Key cost optimization: don't store raw features forever. Keep 7-30 days raw (for investigation), aggregate to metrics after that, archive to cold storage after 90 days."

---

## Common Mistakes

1. **Logging blocks the prediction path** — Synchronous prediction logging where if Kafka is slow, prediction latency spikes. Solution: always log asynchronously (fire-and-forget with local buffer). If logging fails, predictions still serve correctly. Use dead letter queues for retry.

2. **Storing everything forever** — Logging every prediction with full features indefinitely. After 6 months: 500TB of data costing $10K/month in storage, and nobody queries data older than 30 days. Solution: retention policy — raw data for 7-30 days, aggregated metrics for 1 year, archive or delete after that.

3. **Building custom when tools exist** — Spending 3 months building a drift detection system from scratch when Evidently AI does the same thing with `pip install evidently`. Solution: evaluate existing tools first. Build custom only when existing tools don't meet specific requirements (scale, privacy, integration).

4. **Separate monitoring for each model** — Each team builds their own monitoring dashboard and alerting. 10 models = 10 different systems, inconsistent quality, duplicated effort. Solution: platform approach — shared monitoring infrastructure that any model can plug into. Standard logging format, shared dashboards, consistent alerting.

5. **Monitoring without investigation tools** — Great at detecting "something is wrong" but no tools to figure out WHAT is wrong or WHY. Alert fires, engineer spends 3 hours manually querying logs. Solution: invest in investigation UX — ability to filter predictions by time, segment, feature values. Compare distributions side by side. Trace individual predictions through the pipeline.

---

## Key Takeaways

- Tool landscape: Evidently (OSS drift), Arize (managed observability), WhyLabs (profiling), cloud-native monitors
- Prediction logging: async, non-blocking, buffered — never impact serving path
- Architecture: logging → Kafka/S3 → metrics computation (hourly) → dashboards + alerting
- Cost management: sample for high-volume, tiered retention (raw 7-30 days, aggregated 1 year)
- Build vs. buy: small teams → OSS (Evidently + Prometheus + Grafana); large/budget → managed (Arize)
- Platform approach: shared monitoring infrastructure that all models plug into (not per-model custom)
- Investigation tools: equally important as detection — need to drill down into WHY
- Retention policy: raw predictions (7-30 days), metrics (1 year), archive (beyond)
- Storage: 1M predictions/day ≈ 2-10 GB/day depending on feature logging verbosity
- Integration: monitoring feeds into alerting, alerting links to runbooks, runbooks guide investigation
