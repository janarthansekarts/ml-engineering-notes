# ML for Operations (AIOps)

## The Problem / Why This Matters

AIOps (Artificial Intelligence for IT Operations) applies ML to operational data — logs, metrics, traces, alerts — to automate incident detection, root cause analysis, capacity planning, and remediation. Modern infrastructure generates overwhelming data: thousands of microservices each producing metrics, millions of log lines per minute, and cascading alerts that bury signal in noise. In 2026, AIOps has evolved from basic anomaly detection to intelligent systems that: predict failures before they happen (predictive maintenance), automatically identify root causes from symptoms (causal inference on dependency graphs), suppress alert noise (correlate related alerts into single incidents), auto-remediate known issues (runbook automation), and assist on-call engineers with contextual diagnosis (LLM-powered analysis). The shift from reactive (fix after break) to predictive (fix before break) operations represents a fundamental change in how infrastructure is managed. ML for Ops handles unique challenges: extremely high data volume (TB/day of telemetry), need for real-time processing (anomaly must be detected in seconds, not hours), dynamic baselines (normal behavior changes with deployments, traffic patterns), and the requirement for explainability (on-call engineer needs to understand WHY the system thinks something is wrong).

---

## The Analogy

Think of AIOps like a doctor monitoring patients in an ICU (Intensive Care Unit):

- **Manual monitoring** (traditional ops) = A doctor checking each patient's vitals by walking the floor every hour. If something goes wrong between checks, it's missed. If 100 monitors alarm simultaneously, the doctor is overwhelmed.
- **Basic alerting** (threshold-based) = Setting alarm thresholds on each monitor. CPU > 90%? Alert! But the alarm rings constantly (false positives), and complex failures (CPU fine, but latency climbing) go undetected.
- **AIOps** = An AI assistant that knows each patient's normal baseline, correlates signals across monitors (heart rate + blood pressure + oxygen together mean X), predicts deterioration before it happens, and summarizes: "Patient 7 is developing sepsis — here's the evidence and recommended action."

---

## Deep Dive

### AIOps Capabilities

```yaml
AIOps_Capabilities:
  anomaly_detection:
    what: "Detect unusual behavior in metrics, logs, and traces"
    types:
      metric_anomaly: "Latency spike, error rate increase, memory leak"
      log_anomaly: "Unusual log patterns, new error types, volume changes"
      trace_anomaly: "Slow spans, new dependency patterns, missing spans"
    challenge: "Dynamic baselines (deployments, traffic patterns, seasonality)"
    
  alert_correlation:
    what: "Group related alerts into single incidents"
    problem: "One root cause → 50 symptoms → 50 alerts → alert fatigue"
    approach: "Temporal correlation, topology-aware grouping, causal inference"
    result: "50 alerts → 1 incident with root cause hypothesis"
    
  root_cause_analysis:
    what: "Identify the underlying cause of an incident"
    methods:
      topology_based: "Walk service dependency graph from symptom to cause"
      statistical: "Correlate metric changes with deployment/config events"
      causal: "Causal inference to distinguish cause from effect"
      llm_powered: "LLM analyzes logs + metrics + topology for diagnosis"
    
  predictive:
    failure_prediction: "Predict disk failure, node failure, capacity exhaustion"
    capacity_planning: "Forecast resource needs based on traffic trends"
    deployment_risk: "Predict probability of incidents from deployment characteristics"
    
  remediation:
    auto_remediation: "Automatically fix known issues (restart, scale, rollback)"
    runbook_automation: "Execute runbook steps automatically when triggered"
    assisted_diagnosis: "LLM-powered assistant helps on-call diagnose issues"
```

### Architecture

```yaml
AIOps_Architecture:
  data_ingestion:
    metrics:
      sources: "Prometheus, Datadog, CloudWatch, custom metrics"
      volume: "Millions of time series, 15-second resolution"
      storage: "Time-series database (InfluxDB, Thanos, Cortex)"
      
    logs:
      sources: "Application logs, system logs, audit logs"
      volume: "TB per day for large organizations"
      processing: "Structured extraction, pattern recognition"
      storage: "Elasticsearch, Loki, CloudWatch Logs"
      
    traces:
      sources: "OpenTelemetry, Jaeger, Zipkin"
      volume: "Millions of spans per minute"
      analysis: "Latency breakdown, error attribution, dependency mapping"
      
  processing_pipeline:
    streaming:
      what: "Real-time anomaly detection on metrics/logs"
      tool: "Flink, Kafka Streams, custom stream processors"
      latency: "Detect anomaly within 30 seconds of occurrence"
      
    batch:
      what: "Model training, pattern mining, capacity forecasting"
      tool: "Spark, scheduled ML pipelines"
      frequency: "Daily model retraining, hourly forecast updates"
      
  ml_models:
    anomaly_detection:
      metric_level: "Per-metric anomaly scoring (statistical + ML)"
      service_level: "Multi-metric anomaly (across related metrics of a service)"
      
    correlation:
      temporal: "Events close in time likely related"
      topological: "Events on connected services likely related"
      causal: "Granger causality, causal discovery algorithms"
      
    prediction:
      time_series: "Forecast metric values (for predictive alerting)"
      classification: "Predict incident severity, type, duration"
```

### Implementation

```python
# AIOps ML system implementation

"""
AIOps system for anomaly detection, alert correlation,
root cause analysis, and predictive operations.
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import numpy as np
import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class MetricDataPoint:
    """A single metric observation."""
    metric_name: str
    service: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """A triggered alert."""
    alert_id: str
    service: str
    metric: str
    severity: str  # critical, warning, info
    message: str
    timestamp: float
    value: float
    threshold: float


@dataclass
class Incident:
    """Correlated incident (grouped alerts)."""
    incident_id: str
    alerts: List[Alert]
    root_cause_hypothesis: str
    affected_services: Set[str]
    severity: str
    start_time: float
    suggested_actions: List[str]


class MetricAnomalyDetector:
    """
    Anomaly detection for infrastructure metrics.
    
    Challenge: "normal" changes constantly
    - Deployments shift baseline
    - Traffic has daily/weekly patterns
    - Scaling events change resource usage
    
    Approach: dynamic baseline that adapts to:
    - Seasonality (daily/weekly patterns)
    - Trend (gradual shifts)
    - Recent behavior (quick adaptation to new normal)
    """
    
    def __init__(
        self,
        seasonality_period: int = 1440,  # Minutes in a day
        sensitivity: float = 3.0,
        min_data_points: int = 100,
    ):
        self.seasonality_period = seasonality_period
        self.sensitivity = sensitivity
        self.min_data_points = min_data_points
        
        # Per-metric state
        self.metric_models: Dict[str, MetricModel] = {}
    
    def score(self, data_point: MetricDataPoint) -> Dict:
        """
        Score a metric data point for anomaly.
        
        Returns anomaly score and whether it exceeds threshold.
        """
        key = f"{data_point.service}/{data_point.metric_name}"
        
        # Get or create metric model
        if key not in self.metric_models:
            self.metric_models[key] = MetricModel(
                seasonality_period=self.seasonality_period
            )
        
        model = self.metric_models[key]
        
        # Update model with new data point
        model.add_observation(data_point.value, data_point.timestamp)
        
        # Score anomaly
        if model.observation_count < self.min_data_points:
            return {"is_anomaly": False, "score": 0, "reason": "warming_up"}
        
        expected = model.predict(data_point.timestamp)
        residual_std = model.residual_std()
        
        if residual_std == 0:
            return {"is_anomaly": False, "score": 0, "reason": "constant_metric"}
        
        z_score = abs(data_point.value - expected) / residual_std
        is_anomaly = z_score > self.sensitivity
        
        return {
            "is_anomaly": is_anomaly,
            "score": float(z_score),
            "threshold": self.sensitivity,
            "expected": float(expected),
            "actual": data_point.value,
            "metric": key,
        }


class MetricModel:
    """Per-metric statistical model for dynamic baseline."""
    
    def __init__(self, seasonality_period: int = 1440, max_history: int = 10080):
        self.seasonality_period = seasonality_period
        self.max_history = max_history  # 7 days in minutes
        self.values: List[float] = []
        self.timestamps: List[float] = []
        self.observation_count = 0
    
    def add_observation(self, value: float, timestamp: float):
        """Add new observation, maintain rolling window."""
        self.values.append(value)
        self.timestamps.append(timestamp)
        self.observation_count += 1
        
        # Trim to max history
        if len(self.values) > self.max_history:
            self.values = self.values[-self.max_history:]
            self.timestamps = self.timestamps[-self.max_history:]
    
    def predict(self, timestamp: float) -> float:
        """
        Predict expected value at timestamp.
        
        Combines:
        - Recent mean (last hour)
        - Seasonal component (same time yesterday/last week)
        """
        if not self.values:
            return 0.0
        
        # Recent mean (last 60 observations ≈ 1 hour at 1min resolution)
        recent = self.values[-60:] if len(self.values) >= 60 else self.values
        recent_mean = np.mean(recent)
        
        # Seasonal: same time period (look back 1 day and 1 week)
        seasonal_values = []
        for lookback in [self.seasonality_period, self.seasonality_period * 7]:
            idx = len(self.values) - lookback
            if 0 <= idx < len(self.values):
                seasonal_values.append(self.values[idx])
        
        if seasonal_values:
            seasonal_mean = np.mean(seasonal_values)
            # Blend recent and seasonal (70% recent, 30% seasonal)
            return 0.7 * recent_mean + 0.3 * seasonal_mean
        
        return recent_mean
    
    def residual_std(self) -> float:
        """Compute standard deviation of prediction residuals."""
        if len(self.values) < 10:
            return 1.0
        
        recent = self.values[-100:]  # Last 100 observations
        return float(np.std(recent)) or 1.0


class AlertCorrelator:
    """
    Correlate related alerts into incidents.
    
    Problem: One root cause triggers cascade of alerts across services.
    - Database slow → API timeout → Frontend error → User complaints
    - Results in 50+ alerts for one issue
    
    Solution: Group alerts that are:
    1. Close in time (within correlation window)
    2. On connected services (topology-aware)
    3. Causally related (downstream effects of upstream failure)
    """
    
    def __init__(
        self,
        service_topology: Dict[str, List[str]],  # service → dependencies
        time_window_seconds: float = 300,  # 5 minute correlation window
    ):
        self.topology = service_topology
        self.time_window = time_window_seconds
        self.active_incidents: List[Incident] = []
        self.alert_buffer: List[Alert] = []
    
    def process_alert(self, alert: Alert) -> Optional[Incident]:
        """
        Process new alert: correlate with existing incidents or create new one.
        """
        self.alert_buffer.append(alert)
        
        # Try to correlate with existing incident
        for incident in self.active_incidents:
            if self._correlates_with(alert, incident):
                incident.alerts.append(alert)
                incident.affected_services.add(alert.service)
                # Update root cause hypothesis
                incident.root_cause_hypothesis = self._infer_root_cause(incident)
                return incident
        
        # Check if correlates with recent alerts (new incident)
        correlated_alerts = self._find_correlated_alerts(alert)
        
        if correlated_alerts:
            # Create new incident
            all_alerts = correlated_alerts + [alert]
            incident = Incident(
                incident_id=f"INC-{int(time.time())}",
                alerts=all_alerts,
                root_cause_hypothesis=self._infer_root_cause_from_alerts(all_alerts),
                affected_services={a.service for a in all_alerts},
                severity=max(a.severity for a in all_alerts),
                start_time=min(a.timestamp for a in all_alerts),
                suggested_actions=self._suggest_actions(all_alerts),
            )
            self.active_incidents.append(incident)
            return incident
        
        return None  # Standalone alert (no correlation found yet)
    
    def _correlates_with(self, alert: Alert, incident: Incident) -> bool:
        """Check if alert correlates with existing incident."""
        # Time proximity
        time_diff = alert.timestamp - incident.start_time
        if time_diff > self.time_window:
            return False
        
        # Topological proximity (alert's service connected to incident services)
        alert_deps = set(self.topology.get(alert.service, []))
        incident_services = incident.affected_services
        
        if alert.service in incident_services:
            return True  # Same service
        if alert_deps & incident_services:
            return True  # Alert's service depends on incident's services
        if any(alert.service in self.topology.get(s, []) for s in incident_services):
            return True  # Incident's services depend on alert's service
        
        return False
    
    def _find_correlated_alerts(self, alert: Alert) -> List[Alert]:
        """Find recent alerts that correlate with this one."""
        correlated = []
        for buffered in self.alert_buffer[-50:]:  # Check last 50 alerts
            if buffered.alert_id == alert.alert_id:
                continue
            time_diff = abs(alert.timestamp - buffered.timestamp)
            if time_diff <= self.time_window:
                if self._services_related(alert.service, buffered.service):
                    correlated.append(buffered)
        return correlated
    
    def _services_related(self, service_a: str, service_b: str) -> bool:
        """Check if two services are topologically related."""
        deps_a = set(self.topology.get(service_a, []))
        deps_b = set(self.topology.get(service_b, []))
        return (service_b in deps_a or service_a in deps_b or
                bool(deps_a & deps_b))  # Shared dependency
    
    def _infer_root_cause(self, incident: Incident) -> str:
        """
        Infer root cause from correlated alerts.
        
        Heuristic: the service that is upstream (depended upon) and
        alerted first is likely the root cause.
        """
        # Find the earliest alert on an upstream service
        sorted_alerts = sorted(incident.alerts, key=lambda a: a.timestamp)
        
        for alert in sorted_alerts:
            # Check if this service is depended upon by others in the incident
            dependents = [
                s for s in incident.affected_services
                if alert.service in self.topology.get(s, [])
            ]
            if dependents:
                return f"Likely root cause: {alert.service} ({alert.message})"
        
        return f"Root cause unclear — earliest alert: {sorted_alerts[0].service}"
    
    def _infer_root_cause_from_alerts(self, alerts: List[Alert]) -> str:
        """Infer root cause from a list of correlated alerts."""
        if not alerts:
            return "Unknown"
        earliest = min(alerts, key=lambda a: a.timestamp)
        return f"Likely: {earliest.service} — {earliest.message}"
    
    def _suggest_actions(self, alerts: List[Alert]) -> List[str]:
        """Suggest remediation actions based on alert patterns."""
        actions = []
        services = {a.service for a in alerts}
        metrics = {a.metric for a in alerts}
        
        if "cpu_utilization" in metrics:
            actions.append("Consider scaling up or investigating CPU-intensive processes")
        if "error_rate" in metrics:
            actions.append("Check recent deployments for regression")
        if "memory_utilization" in metrics:
            actions.append("Investigate memory leaks, consider restart or scaling")
        if "latency_p99" in metrics:
            actions.append("Check downstream dependencies and database queries")
        
        return actions or ["Investigate correlated services and recent changes"]


class PredictiveOps:
    """
    Predictive operations: forecast failures before they happen.
    
    Examples:
    - Disk will be full in 3 days (based on growth trend)
    - This deployment will cause incidents (based on characteristics)
    - CPU will breach threshold in 2 hours (traffic forecast)
    """
    
    def __init__(self, forecaster, threshold_config: Dict):
        self.forecaster = forecaster
        self.thresholds = threshold_config  # metric → threshold
    
    def predict_threshold_breach(
        self, metric_name: str, current_values: List[float], horizon_hours: int = 24
    ) -> Optional[Dict]:
        """
        Predict when a metric will breach its threshold.
        
        Returns time-to-breach and recommended action.
        """
        threshold = self.thresholds.get(metric_name)
        if not threshold:
            return None
        
        # Forecast next N hours
        forecast = self.forecaster.predict(current_values, horizon=horizon_hours)
        
        # Find when forecast exceeds threshold
        for i, predicted_value in enumerate(forecast):
            if predicted_value >= threshold:
                hours_to_breach = i + 1
                return {
                    "metric": metric_name,
                    "current_value": current_values[-1],
                    "threshold": threshold,
                    "predicted_breach_hours": hours_to_breach,
                    "predicted_breach_value": predicted_value,
                    "urgency": "critical" if hours_to_breach < 4 else "warning",
                    "action": f"Scale or address {metric_name} before breach in ~{hours_to_breach}h",
                }
        
        return None  # No breach predicted within horizon
```

### LLM-Powered Diagnosis

```yaml
LLM_Diagnosis:
  what: "Use LLM to analyze incidents and suggest diagnosis"
  
  input_to_llm:
    - "Alert details (metric, service, threshold, current value)"
    - "Recent logs from affected services (last 30 minutes)"
    - "Recent deployments (what changed?)"
    - "Service topology (what depends on what?)"
    - "Historical incidents (similar past events and their resolutions)"
    
  output:
    - "Root cause hypothesis with confidence"
    - "Evidence from logs/metrics supporting hypothesis"
    - "Suggested remediation steps"
    - "Similar past incidents and how they were resolved"
    
  implementation:
    tool: "Claude 4 / GPT-5 with RAG on runbooks + incident history"
    context: "MCP (Model Context Protocol) to pull live telemetry data"
    integration: "Slack/PagerDuty bot that assists on-call engineer"
    
  example_prompt: |
    You are an SRE assistant. Analyze this incident:
    - Alert: API latency P99 > 2s (normal: 200ms)
    - Affected: payment-service, checkout-service
    - Recent deploy: payment-service v3.2.1 (30 minutes ago)
    - Logs: "connection pool exhausted" appearing 100x/min
    - Database: connection count at 95% of max
    
    Provide: root cause, evidence, and remediation steps.
```

---

## How It Works in Practice

### AIOps Incident Workflow

```yaml
Incident_Workflow:
  t_0: "Database connection pool exhaustion begins"
  t_30s: "Anomaly detector scores database connection metric as anomalous (z=4.2)"
  t_45s: "API latency anomaly detected (upstream of database)"
  t_60s: "Alert correlator groups: database + 3 API services → single incident"
  t_90s: "Root cause inference: database connection pool is upstream cause"
  t_120s: "LLM diagnosis: 'Connection pool exhausted due to slow queries introduced in deploy v3.2.1. Evidence: new query pattern in logs, deploy 30min ago.'"
  t_150s: "Auto-remediation: increase connection pool limit (known safe action)"
  t_180s: "PagerDuty notification with full context (not 50 separate alerts)"
  
  without_aiops:
    t_0: "Same failure begins"
    t_300s: "First threshold alert fires (5 min delay for threshold to breach)"
    t_360s: "50+ alerts fire across services (alert storm)"
    t_600s: "On-call paged, starts investigating (which alert first?)"
    t_1800s: "After 30 min of manual investigation, identifies database as root cause"
    t_2400s: "Fix applied (40 min total resolution)"
```

---

## Interview Tip

> When asked about AIOps/ML for Ops: "I build AIOps systems with three layers: detection, correlation, and action. Detection: anomaly detection on infrastructure metrics using dynamic baselines (not static thresholds). Each metric has a model that adapts to seasonality (daily/weekly patterns), trend (gradual shifts), and recent behavior (deployment-caused changes). I use z-score on prediction residuals — flagging when actual deviates > 3σ from expected. Correlation: when one failure cascades (database slow → API timeout → frontend error), I get 50 alerts. Alert correlator groups them using temporal proximity (within 5 minutes) AND service topology (connectivity graph). Result: one incident with root cause hypothesis, not 50 independent alerts. Root cause inference uses topology: the upstream service that alerted first is likely the cause. Action: predictive operations forecast metric trajectories — 'disk full in 3 days based on growth rate.' LLM-powered diagnosis (Claude 4 / GPT-5 with RAG on runbooks) analyzes logs + metrics + recent deployments and suggests remediation. Auto-remediation executes known-safe actions (scale up, increase pool, rollback) when confidence is high. Key challenge: dynamic baselines. Traditional thresholds (CPU > 80%) cause alert fatigue. ML baselines adapt to 'what's normal for this service at this time on this day' — reducing false positives by 60-80% while catching more real incidents."

---

## Common Mistakes

1. **Static thresholds** — Alerting on CPU > 80% when normal is 75% on Mondays (legitimate traffic spike). Constant false alarms. Solution: dynamic baselines that learn normal patterns per service per time-of-day/week.

2. **No alert correlation** — Every symptom generates a separate page. On-call gets 50 notifications for one incident. Alert fatigue → real alerts ignored. Solution: correlate alerts by time + topology → one incident per root cause.

3. **Training on clean data** — Anomaly detection model trained on 6 months of "normal" data. But the data contains unlabeled incidents. Model learns incidents are normal. Solution: careful data curation, or unsupervised methods that don't require labeled "normal."

4. **Ignoring topology** — Treating each service independently. Can't trace cascade from root cause to symptoms. Solution: maintain service dependency graph (from traces, service mesh). Use topology for root cause inference.

5. **Automating too aggressively** — Auto-remediation runs without guard rails, causes cascading failures (auto-scaling into budget overrun, auto-restart loops). Solution: implement blast radius limits, human approval for high-impact actions, circuit breakers on auto-remediation.

---

## Key Takeaways

- AIOps: ML for anomaly detection, alert correlation, root cause analysis, predictive ops
- Dynamic baselines: adapt to seasonality, trend, deployments (not static thresholds)
- Alert correlation: group related alerts using time proximity + service topology
- Root cause inference: upstream service that alerted first is likely the cause
- Predictive ops: forecast metrics, predict threshold breaches before they happen
- LLM-powered diagnosis: analyze logs + metrics + topology + runbooks for suggestions
- Auto-remediation: known-safe actions (scale, restart, rollback) with guard rails
- Reduce alert fatigue: correlation reduces 50 alerts → 1 incident (80% noise reduction)
- Data challenges: high volume (TB/day), real-time processing (detect in 30 seconds)
- Tools (2026): Datadog AI, Dynatrace Davis AI, PagerDuty AIOps, custom Flink pipelines
