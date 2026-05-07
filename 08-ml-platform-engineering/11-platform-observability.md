# Platform Observability

## The Problem / Why This Matters

An ML platform serves dozens of teams running hundreds of models — if the platform itself has reliability issues, every team is affected. Platform observability means monitoring the platform's own health (not individual model performance — that's the model owner's responsibility). Without platform observability: silent failures cascade (feature store is slow → all real-time models degrade → nobody knows why), capacity problems hit without warning (GPU queue grows to 3 days overnight), user experience degrades gradually (SDK response times creep from 200ms to 5s over months), and cost anomalies go undetected (a misconfiguration causes 3× normal cloud spend for a week). Platform observability encompasses: infrastructure metrics (GPU cluster health, storage capacity, network throughput), service metrics (feature store latency, model registry availability, training service queue depth), user-facing metrics (SDK response time, deployment success rate, onboarding completion rate), and capacity planning (when do we need more GPUs? storage? compute?). In 2026, with ML platforms supporting mission-critical applications (fraud detection, real-time recommendations, autonomous systems), platform reliability is measured in SLOs (Service Level Objectives) and SLAs (Service Level Agreements) — just like any other production infrastructure.

---

## The Analogy

Think of platform observability like the monitoring systems in a hospital:

- **Individual patient monitors** = Model monitoring (heart rate, blood pressure for each patient/model). This is the model owner's responsibility.
- **Hospital infrastructure monitoring** = Platform observability (power systems, HVAC, medical gas supply, elevator status, network connectivity). If the hospital infrastructure fails, ALL patients are affected, not just one.

You need both: patient monitors (model monitoring) AND hospital infrastructure monitoring (platform observability). A hospital that only monitors patients but not its own power systems will eventually have a building-wide failure that affects every patient simultaneously.

---

## Deep Dive

### Platform Metrics Taxonomy

```yaml
Metrics_Taxonomy:
  infrastructure_layer:
    gpu_cluster:
      - "GPU utilization (per-node, cluster-wide)"
      - "GPU memory usage (allocated vs. available)"
      - "GPU temperature and throttling events"
      - "Node health (up/down, kubelet status)"
      - "InfiniBand link status (for distributed training)"
      - "NVLink bandwidth utilization"
    storage:
      - "Object storage (S3/GCS) throughput and latency"
      - "Persistent volume utilization (% full)"
      - "Feature store online read latency (p50, p95, p99)"
      - "Model artifact storage capacity"
    compute:
      - "CPU utilization (platform services)"
      - "Memory usage (platform services)"
      - "Pod scheduling latency (time from request to running)"
      - "Kubernetes API server latency"
    networking:
      - "Network throughput (intra-cluster, cross-cluster)"
      - "DNS resolution latency"
      - "Service mesh (Istio/Envoy) latency overhead"
      
  service_layer:
    training_service:
      - "Job queue depth (waiting jobs)"
      - "Job queue wait time (p50, p95)"
      - "Job success rate (% completed without error)"
      - "Job scheduling latency (submitted → started)"
      - "Active training jobs (gauge)"
      - "GPU allocation efficiency (requested vs. used)"
    serving_service:
      - "Deployment success rate"
      - "Deployment rollback rate"
      - "Model endpoint availability (% uptime)"
      - "Endpoint p95 latency (across all endpoints)"
      - "Autoscaling responsiveness (time to scale up)"
    feature_store:
      - "Online read latency (p50, p95, p99)"
      - "Online read throughput (QPS)"
      - "Offline query execution time"
      - "Feature freshness (time since last update)"
      - "Feature materialization job success rate"
    model_registry:
      - "API latency (registration, lookup)"
      - "Storage capacity (total artifacts)"
      - "API error rate"
    experiment_tracking:
      - "Log write latency"
      - "Dashboard load time"
      - "Storage usage"
      
  user_experience_layer:
    sdk_metrics:
      - "SDK response time (end-to-end for each operation)"
      - "SDK error rate (% of calls that fail)"
      - "SDK adoption (active users per week)"
    deployment_experience:
      - "Time to deploy (command → endpoint ready)"
      - "First-time deployment success rate"
      - "Rollback frequency (indicates quality issues)"
    support:
      - "Support ticket volume (leading indicator of problems)"
      - "Mean time to resolution (MTTR)"
      - "Ticket severity distribution"
    user_satisfaction:
      - "NPS (Net Promoter Score) — quarterly survey"
      - "Feature request volume and themes"
      - "Platform abandonment (teams leaving platform)"
      
  business_layer:
    cost:
      - "Total platform cost (monthly)"
      - "Cost per active user"
      - "Cost per deployed model"
      - "Cost efficiency trend (cost / ML value delivered)"
    capacity:
      - "GPU utilization trend (approaching capacity?)"
      - "Storage growth rate"
      - "User growth rate"
```

### SLOs and SLAs

```yaml
SLOs:
  what: "SLO (Service Level Objective) — internal target for platform reliability"
  
  critical_slos:
    feature_store_online:
      metric: "p99 read latency"
      target: "< 10ms"
      measurement: "Rolling 7-day window"
      consequence: "If breached → P1 incident, dedicated team fix"
      
    model_serving_availability:
      metric: "Endpoint uptime (successful responses / total requests)"
      target: "99.9% (43 min downtime/month allowed)"
      measurement: "Rolling 30-day window per endpoint"
      consequence: "If breached → postmortem, action items"
      
    training_job_scheduling:
      metric: "Time from job submission to job start (standard priority)"
      target: "p95 < 15 minutes"
      measurement: "Rolling 7-day window"
      consequence: "If breached → add capacity or optimize scheduling"
      
    deployment_success:
      metric: "% of deployment requests that succeed"
      target: "99%"
      measurement: "Rolling 7-day window"
      consequence: "If breached → investigate failure causes"
      
    sdk_availability:
      metric: "% of SDK calls that succeed (2xx response)"
      target: "99.9%"
      measurement: "Rolling 7-day window"
      
  error_budgets:
    what: "How much unreliability is tolerable before action required"
    example:
      target: "99.9% availability (30-day)"
      budget: "43 minutes of downtime per month"
      current_burn: "If burning budget faster than 1/30 per day → slow down changes"
      exhausted: "If budget exhausted → freeze non-critical deployments until next window"
```

### Alerting Strategy

```python
# Platform alerting configuration

"""
Alerting strategy for ML platform observability.
Principle: Alert on symptoms (user impact), not causes (internal metrics).
"""


class PlatformAlerts:
    """
    Multi-tier alerting for ML platform.
    
    Severity levels:
    - P1 (Critical): Platform-wide outage or data loss risk
    - P2 (High): Major feature degradation affecting multiple teams
    - P3 (Medium): Single service degradation, workaround available
    - P4 (Low): Performance degradation, non-urgent investigation
    """
    
    alerts = {
        # P1 — Critical (page on-call immediately)
        "p1_feature_store_down": {
            "severity": "P1",
            "condition": "feature_store_error_rate > 50% for 5 minutes",
            "impact": "All real-time models serving stale/no features",
            "action": "Page on-call, escalate to team lead",
            "runbook": "https://wiki.internal/runbooks/feature-store-outage",
        },
        "p1_gpu_cluster_unhealthy": {
            "severity": "P1",
            "condition": "available_gpu_nodes < 20% of total for 10 minutes",
            "impact": "Training jobs failing, serving may degrade",
            "action": "Page on-call, check Kubernetes nodes",
            "runbook": "https://wiki.internal/runbooks/gpu-cluster-health",
        },
        "p1_model_serving_outage": {
            "severity": "P1",
            "condition": "serving_platform_error_rate > 10% across all endpoints for 5 min",
            "impact": "Multiple production models returning errors",
            "action": "Page on-call, check ingress/service mesh",
        },
        
        # P2 — High (alert Slack + page if not acknowledged in 30 min)
        "p2_feature_store_slow": {
            "severity": "P2",
            "condition": "feature_store_p99_latency > 50ms for 15 minutes",
            "impact": "Real-time predictions slower, possible timeouts",
            "action": "Alert Slack, investigate Redis/network",
        },
        "p2_training_queue_long": {
            "severity": "P2",
            "condition": "training_queue_wait_p95 > 60 minutes for 30 minutes",
            "impact": "ML teams blocked, productivity loss",
            "action": "Alert Slack, check cluster capacity, preempt low-priority",
        },
        "p2_deployment_failures": {
            "severity": "P2",
            "condition": "deployment_failure_rate > 20% over last 1 hour",
            "impact": "Teams cannot deploy models",
            "action": "Alert Slack, check deployment service health",
        },
        
        # P3 — Medium (Slack notification during business hours)
        "p3_storage_capacity": {
            "severity": "P3",
            "condition": "storage_utilization > 80%",
            "impact": "May run out of storage within days/weeks",
            "action": "Plan capacity expansion or cleanup",
        },
        "p3_sdk_slow": {
            "severity": "P3",
            "condition": "sdk_response_time_p95 > 2s for 1 hour",
            "impact": "Developer experience degraded",
            "action": "Investigate API server performance",
        },
        
        # P4 — Low (daily digest, non-urgent)
        "p4_gpu_utilization_low": {
            "severity": "P4",
            "condition": "cluster_gpu_utilization < 50% for 24 hours",
            "impact": "Wasted resources, cost inefficiency",
            "action": "Review scheduling, consider reducing capacity",
        },
        "p4_documentation_404s": {
            "severity": "P4",
            "condition": "documentation_404_rate > 5%",
            "impact": "Users hitting broken links",
            "action": "Fix broken documentation links",
        },
    }
    
    def evaluate_alert(self, alert_name: str, current_metrics: dict) -> dict:
        """Evaluate if alert should fire based on current metrics."""
        alert = self.alerts[alert_name]
        # Evaluate condition against metrics
        # Return: {should_fire, severity, context}
        pass
```

### Capacity Planning

```yaml
Capacity_Planning:
  gpu_capacity:
    metrics_tracked:
      - "Current GPU utilization (rolling 7d average)"
      - "Peak GPU utilization (daily max)"
      - "Queue depth trend (growing or stable?)"
      - "New teams onboarding (forecast demand)"
    thresholds:
      green: "Average util < 60%, peak < 80% — healthy headroom"
      yellow: "Average util 60-75%, peak 80-90% — plan expansion"
      red: "Average util > 75%, peak > 90% — immediate action needed"
    lead_time: "GPU node provisioning: 2-4 weeks (cloud), 3-6 months (on-prem)"
    
  storage_capacity:
    growth_rate: "Track daily storage growth rate"
    projection: "At current rate, when does storage hit 80%?"
    actions:
      - "Data lifecycle policies (archive old datasets)"
      - "Experiment artifact cleanup (delete old failed runs)"
      - "Increase storage quota"
      
  compute_capacity:
    scaling: "Horizontal (more pods/nodes) and vertical (bigger instances)"
    autoscaling: "Platform services autoscale within limits"
    limits: "Set max scale to prevent runaway costs"
    
  forecasting:
    method: "Linear regression on growth metrics + seasonal adjustments"
    frequency: "Monthly capacity review"
    output: "Projected need for next quarter → procurement decision"
```

### Incident Management

```yaml
Incident_Management:
  detection:
    automated: "Alerts fire based on SLO breach or metric anomaly"
    user_reported: "User opens support ticket or reports in Slack"
    
  response:
    p1_p2:
      1: "Alert fires → on-call paged"
      2: "On-call acknowledges within 5 min"
      3: "Initial diagnosis (check dashboard, recent changes)"
      4: "Mitigation (if possible — rollback, restart, failover)"
      5: "Communication (status page update, Slack notification to affected teams)"
      6: "Root cause investigation (after mitigation)"
      7: "Post-incident review (blameless postmortem within 48 hours)"
      
  postmortem:
    template:
      - "Incident summary (what happened, duration, impact)"
      - "Timeline (when detected, mitigated, resolved)"
      - "Root cause (why it happened)"
      - "Detection gap (why didn't we detect sooner?)"
      - "Action items (prevent recurrence, improve detection)"
    output: "Published action items tracked to completion"
    
  on_call:
    rotation: "Weekly rotation among platform engineers"
    tooling: "PagerDuty for alerting, runbooks for common issues"
    escalation: "If not resolved in 30 min → escalate to team lead"
```

---

## How It Works in Practice

### Daily Operations

```yaml
Daily_Ops:
  morning_check:
    - "Check overnight alerts (any P1/P2 fired?)"
    - "Review SLO dashboard (any budget burn?)"
    - "Check GPU queue depth (wait time acceptable?)"
    - "Review cost dashboard (any anomalies?)"
    
  continuous:
    - "Automated alerts route to on-call"
    - "SLO dashboard visible to all teams"
    - "Capacity dashboard for long-term planning"
    
  weekly:
    - "Review platform metrics trend (getting better or worse?)"
    - "Review support tickets (common issues = platform improvements needed)"
    - "Capacity review (GPU utilization trend, storage growth)"
    
  monthly:
    - "SLO report (achieved targets? error budget status?)"
    - "Capacity planning review"
    - "Cost review and optimization"
    - "Incident review (recurring themes?)"
```

---

## Interview Tip

> When asked about platform observability: "I observe the ML platform at four layers: (1) Infrastructure — GPU cluster health (node availability, utilization, temperature), storage capacity (utilization trends, growth rate), and network (InfiniBand link status, throughput). DCGM (Data Center GPU Manager) exporter feeds Prometheus for GPU metrics. (2) Services — each platform service has SLOs: feature store online reads < 10ms p99, training job queue wait < 15 min p95, deployment success > 99%. I track error budgets — if a service burns its budget faster than expected, we freeze non-critical changes until reliability improves. (3) User experience — SDK response times, deployment success rate, support ticket volume. These are leading indicators — SDK slowness today means frustrated users tomorrow. (4) Business — cost per deployed model, GPU utilization efficiency, capacity projections. For alerting: I alert on symptoms (user impact) not causes (internal metrics). P1 = platform-wide outage (page immediately), P2 = major feature degradation (alert + 30-min ack deadline), P3 = single-service degradation (business hours notification), P4 = non-urgent trends (daily digest). Every P1/P2 gets a blameless postmortem with action items. Capacity planning: monthly review of utilization trends with 2-4 week lead time for GPU procurement."

---

## Common Mistakes

1. **Alert on every metric, not symptoms** — 500 alerts for internal metrics (CPU spike, pod restart, GC pause). Noise overwhelms the on-call engineer, real issues get missed. Solution: alert on user-facing symptoms (feature store slow, deployments failing, queue growing). Internal metrics are for diagnosis, not alerting.

2. **No SLOs defined** — Platform "feels" reliable but nobody measures. When it degrades, there's no shared understanding of "how bad is this?" Solution: define SLOs for critical paths (feature store latency, serving availability, queue wait time). SLOs provide objective measure: breached = must act, healthy = can invest in features.

3. **Capacity planning as reactive** — Run out of GPU capacity on Monday. Emergency procurement takes 3 weeks. ML teams blocked. Solution: monthly capacity review with projections. Track utilization trends. When approaching 75%, start procurement (2-4 week lead time for cloud, longer for on-prem).

4. **Platform metrics mixed with model metrics** — Platform team gets paged for individual model degradation (which is the model team's problem). Solution: clear ownership boundary. Platform monitors its own SLOs (infrastructure, services, availability). Model teams monitor their model's quality (accuracy, drift, business metrics).

5. **No runbooks** — Alert fires at 3 AM, on-call engineer doesn't know what to do. Spends 45 minutes reading code instead of following a procedure. Solution: every alert links to a runbook (diagnosis steps, mitigation steps, escalation criteria). Runbooks maintained by the team that owns the service.

---

## Key Takeaways

- Platform observability = monitoring the platform itself (not individual models)
- Four layers: infrastructure (GPU/storage/network), services (feature store/serving/training), UX (SDK latency, success rate), business (cost, capacity)
- SLOs: define reliability targets for critical services (feature store <10ms p99, serving 99.9% uptime)
- Error budgets: quantify allowable unreliability — when exhausted, prioritize reliability over features
- Alert on symptoms (user impact), not causes (internal metrics) — reduces noise
- Severity levels: P1 (platform outage, page immediately) → P4 (trends, daily digest)
- Capacity planning: monthly review, projections, 2-4 week lead time for GPU provisioning
- Incident management: detect → mitigate → communicate → root cause → postmortem → action items
- Clear ownership: platform team owns infra reliability, model teams own model quality
- Runbooks: every alert links to step-by-step diagnosis and mitigation guide
