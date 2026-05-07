# Chaos Testing for ML

## The Problem / Why This Matters

ML systems in production depend on a complex web of services: feature stores, model registries, vector databases, GPU clusters, monitoring pipelines, and external APIs. When any of these fail — and they will — what happens to predictions? Without chaos testing, you don't know. Most teams discover their failure modes in production: the feature store goes down and the model serves predictions with stale features (wrong results), the GPU runs out of memory and the inference service crashes (no results), or the monitoring pipeline breaks and nobody notices the model drifting (silent degradation). Chaos testing (also called resilience testing or fault injection) intentionally introduces failures in controlled environments to verify that ML systems degrade gracefully. The goal isn't to prevent failures — it's to ensure failures don't cascade into catastrophic outcomes. In 2026, with ML systems making real-time decisions in finance, healthcare, and autonomous vehicles, chaos testing is essential for operational resilience. Netflix pioneered chaos engineering with Chaos Monkey; ML chaos engineering extends these principles to ML-specific failure modes: data failures, model failures, feature failures, and infrastructure failures.

---

## The Analogy

Think of chaos testing like fire drills and emergency preparedness:

- **No chaos testing** = Building has fire exits and extinguishers, but never tested. During actual fire: exits are locked, extinguishers are expired, nobody knows the evacuation route, elevator fills with smoke. Chaos.
- **Chaos testing** = Monthly fire drills. Intentionally trigger alarm, verify: exits unlock automatically, sprinklers activate, people know routes, elevator returns to ground floor. Discover and fix problems BEFORE the real emergency.

You don't cause real fires — you simulate fire conditions to test response. Similarly, chaos testing simulates failures to test system response without causing production outages.

---

## Deep Dive

### ML-Specific Failure Modes

```yaml
ML_Failure_Modes:
  data_failures:
    stale_features:
      what: "Feature store returns old data (not updated in 24h)"
      impact: "Model predicts based on yesterday's reality"
      example: "Fraud model uses 'transactions_last_1h = 0' when user actually has 50"
      severity: "HIGH — silent incorrect predictions"
      
    missing_features:
      what: "Feature store returns partial data (some features null)"
      impact: "Model receives incomplete input"
      example: "5 of 20 features return null → model uses default values"
      severity: "HIGH — degraded predictions with no clear signal"
      
    schema_change:
      what: "Upstream data source changes schema without notice"
      impact: "Feature pipeline breaks or produces wrong values"
      example: "Column 'amount' renamed to 'transaction_amount' upstream"
      severity: "CRITICAL — pipeline crash or silent wrong features"
      
    data_corruption:
      what: "Feature values are wrong (negative ages, future dates)"
      impact: "Model produces confident but wrong predictions"
      example: "All 'income' values returned as 0 due to ETL bug"
      severity: "HIGH — affects all predictions until detected"
      
  model_failures:
    model_loading_failure:
      what: "Model artifact fails to load (corrupt file, version mismatch)"
      impact: "Inference service can't serve predictions"
      example: "New deployment, model file is 0 bytes due to failed download"
      severity: "CRITICAL — no predictions at all"
      
    oom_under_load:
      what: "GPU runs out of memory at high concurrency"
      impact: "Inference service crashes, restarts, serves errors"
      example: "KV cache fills GPU memory with 50 concurrent LLM requests"
      severity: "HIGH — intermittent failures under load"
      
    model_timeout:
      what: "Model inference takes too long (complex input, GPU contention)"
      impact: "Requests timeout, users get errors"
      example: "LLM generates 2000 tokens → takes 60s → times out at 30s"
      severity: "MEDIUM — some requests fail under load"
      
    nan_predictions:
      what: "Model outputs NaN or infinity (numerical instability)"
      impact: "Downstream systems receive invalid predictions"
      example: "Very large feature value → overflow → NaN prediction"
      severity: "HIGH — corrupts downstream decisions"
      
  infrastructure_failures:
    gpu_failure:
      what: "GPU hardware failure or driver crash"
      impact: "Pod crashes, requests rerouted or dropped"
      frequency: "1-2% of GPUs per month in large clusters"
      
    network_partition:
      what: "Feature store or model registry unreachable"
      impact: "Can't compute features or load model updates"
      
    disk_full:
      what: "Log/cache disk fills up on inference node"
      impact: "Can't write predictions to cache, logging fails"
      
    dependency_failure:
      what: "External API (for feature enrichment) is down"
      impact: "Can't compute features that depend on external data"
```

### Chaos Testing Framework

```python
# ML Chaos Testing Framework

"""
Systematically inject failures into ML systems to verify resilience.
Run in staging/pre-production environments.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import time
import random


@dataclass
class ChaosExperiment:
    """Definition of a chaos experiment."""
    name: str
    category: str  # "data", "model", "infrastructure"
    description: str
    inject_fault: Callable  # Function that injects the fault
    verify_behavior: Callable  # Function that verifies expected behavior
    cleanup: Callable  # Function that restores normal state
    severity: str  # "low", "medium", "high", "critical"
    blast_radius: str  # "single_request", "single_pod", "service", "cluster"


class MLChaosRunner:
    """
    Execute chaos experiments against ML systems.
    
    Safety principles:
    1. Always run in staging/pre-production first
    2. Have automatic rollback (cleanup always runs)
    3. Start with smallest blast radius
    4. Monitor system health during experiment
    5. Stop immediately if unexpected cascading failure
    """
    
    def __init__(self, system_health_check: Callable):
        self.health_check = system_health_check
        self.results: List[Dict] = []
    
    def run_experiment(self, experiment: ChaosExperiment) -> Dict:
        """
        Execute a chaos experiment safely.
        
        Steps:
        1. Verify system is healthy (baseline)
        2. Inject fault
        3. Verify expected behavior (graceful degradation)
        4. Cleanup (restore normal state)
        5. Verify system recovers
        """
        result = {
            "name": experiment.name,
            "category": experiment.category,
            "severity": experiment.severity,
        }
        
        # 1. Pre-check: system must be healthy before injecting fault
        if not self.health_check():
            result["status"] = "SKIPPED"
            result["reason"] = "System not healthy before experiment"
            return result
        
        try:
            # 2. Inject fault
            experiment.inject_fault()
            
            # 3. Wait for fault to propagate
            time.sleep(5)
            
            # 4. Verify expected behavior
            behavior_correct = experiment.verify_behavior()
            
            result["behavior_correct"] = behavior_correct
            result["status"] = "PASSED" if behavior_correct else "FAILED"
            
        except Exception as e:
            result["status"] = "ERROR"
            result["error"] = str(e)
            
        finally:
            # 5. Always cleanup
            experiment.cleanup()
            
            # 6. Verify recovery
            time.sleep(10)
            recovered = self.health_check()
            result["recovered"] = recovered
            
            if not recovered:
                result["status"] = "CRITICAL_FAILURE"
                result["alert"] = "System did not recover after cleanup!"
        
        self.results.append(result)
        return result


# === Concrete Chaos Experiments ===

class DataChaosExperiments:
    """Chaos experiments for data-related failures."""
    
    @staticmethod
    def stale_features_experiment(
        feature_store_client,
        inference_client,
        test_request: Dict
    ) -> ChaosExperiment:
        """
        Simulate stale features: feature store returns old data.
        
        Expected behavior: system should either:
        - Detect staleness and return degraded/fallback prediction
        - Log staleness alert
        - NOT silently serve predictions on stale data
        """
        def inject():
            # Pause feature store updates (simulate pipeline failure)
            feature_store_client.pause_updates()
            # Advance system clock to make features appear stale
            feature_store_client.set_feature_timestamp(
                hours_ago=24  # Features are 24 hours old
            )
        
        def verify():
            # Send prediction request
            response = inference_client.predict(test_request)
            
            # Expected behaviors (at least one should be true):
            checks = [
                response.get("warning") == "stale_features",  # Warned about staleness
                response.get("confidence_degraded") is True,  # Reduced confidence
                response.get("fallback_used") is True,  # Used fallback prediction
                response.status_code == 503,  # Service unavailable (strict mode)
            ]
            
            return any(checks)
        
        def cleanup():
            feature_store_client.resume_updates()
            feature_store_client.reset_timestamps()
        
        return ChaosExperiment(
            name="stale_features_24h",
            category="data",
            description="Feature store returns 24-hour-old features",
            inject_fault=inject,
            verify_behavior=verify,
            cleanup=cleanup,
            severity="high",
            blast_radius="service",
        )
    
    @staticmethod
    def missing_features_experiment(
        feature_store_client,
        inference_client,
        test_request: Dict,
        features_to_remove: List[str]
    ) -> ChaosExperiment:
        """
        Simulate missing features: some features return null.
        
        Expected behavior:
        - System handles nulls gracefully (imputation or fallback)
        - Does NOT crash or return NaN
        - Logs which features were missing
        """
        def inject():
            for feature in features_to_remove:
                feature_store_client.set_feature_null(feature)
        
        def verify():
            response = inference_client.predict(test_request)
            
            # Should not crash
            if response.status_code >= 500:
                return False
            
            # Should return valid prediction (not NaN)
            prediction = response.json().get("prediction")
            if prediction is None or str(prediction) == "nan":
                return False
            
            # Should log missing features
            logs = inference_client.get_recent_logs()
            missing_logged = any("missing_feature" in log for log in logs)
            
            return True  # Didn't crash and returned valid prediction
        
        def cleanup():
            feature_store_client.restore_all_features()
        
        return ChaosExperiment(
            name="missing_features",
            category="data",
            description=f"Features {features_to_remove} return null",
            inject_fault=inject,
            verify_behavior=verify,
            cleanup=cleanup,
            severity="high",
            blast_radius="service",
        )


class ModelChaosExperiments:
    """Chaos experiments for model-related failures."""
    
    @staticmethod
    def model_loading_failure(
        model_registry,
        inference_client,
        test_request: Dict
    ) -> ChaosExperiment:
        """
        Simulate model loading failure.
        
        Expected behavior:
        - Fall back to previous model version
        - OR return graceful error (not 500 crash)
        - Alert operations team
        """
        def inject():
            # Corrupt the model artifact
            model_registry.corrupt_latest_artifact()
        
        def verify():
            # Trigger model reload
            inference_client.reload_model()
            time.sleep(5)
            
            # Check system state
            health = inference_client.health()
            
            # Should either:
            # 1. Be serving with fallback model
            # 2. Return clear error (not crash)
            if health.get("model_loaded") and health.get("using_fallback"):
                return True
            
            # Try a prediction
            response = inference_client.predict(test_request)
            return response.status_code != 500  # Shouldn't crash
        
        def cleanup():
            model_registry.restore_artifact()
            inference_client.reload_model()
        
        return ChaosExperiment(
            name="model_loading_failure",
            category="model",
            description="Model artifact is corrupted, can't load",
            inject_fault=inject,
            verify_behavior=verify,
            cleanup=cleanup,
            severity="critical",
            blast_radius="service",
        )
    
    @staticmethod
    def high_latency_simulation(
        inference_client,
        test_requests: List[Dict],
        concurrent_requests: int = 100
    ) -> ChaosExperiment:
        """
        Simulate high load causing latency spikes.
        
        Expected behavior:
        - Requests timeout gracefully (not hang forever)
        - Circuit breaker activates
        - Queue management prevents cascade
        """
        def inject():
            # Flood with concurrent requests
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                futures = [
                    executor.submit(inference_client.predict, req)
                    for req in test_requests[:concurrent_requests]
                ]
                # Don't wait for all — just inject load
        
        def verify():
            # Send a normal request during high load
            start = time.time()
            response = inference_client.predict(test_requests[0], timeout=10)
            latency = time.time() - start
            
            # Should either:
            # 1. Complete within timeout (with degraded latency)
            # 2. Return 429 (rate limited) or 503 (overloaded)
            if response.status_code in [200, 429, 503]:
                return True
            
            # Should NOT hang forever
            return latency < 30
        
        def cleanup():
            time.sleep(10)  # Let load dissipate
        
        return ChaosExperiment(
            name="high_load_latency",
            category="infrastructure",
            description=f"Flood system with {concurrent_requests} concurrent requests",
            inject_fault=inject,
            verify_behavior=verify,
            cleanup=cleanup,
            severity="medium",
            blast_radius="service",
        )


class InfrastructureChaosExperiments:
    """Chaos experiments for infrastructure failures."""
    
    @staticmethod
    def gpu_failure_experiment(
        kubernetes_client,
        inference_client,
        test_request: Dict
    ) -> ChaosExperiment:
        """
        Simulate GPU node failure.
        
        Expected behavior:
        - Kubernetes reschedules pod to healthy node
        - Requests rerouted to remaining replicas
        - No dropped requests (if multiple replicas)
        """
        def inject():
            # Kill one GPU pod (simulate hardware failure)
            kubernetes_client.delete_pod(
                namespace="ml-serving",
                label_selector="app=inference-server",
                count=1  # Kill one pod
            )
        
        def verify():
            # Wait for Kubernetes to notice
            time.sleep(15)
            
            # System should still serve (from remaining replicas)
            response = inference_client.predict(test_request, timeout=30)
            return response.status_code == 200
        
        def cleanup():
            # Kubernetes auto-restarts — just verify recovery
            time.sleep(60)
        
        return ChaosExperiment(
            name="gpu_node_failure",
            category="infrastructure",
            description="Kill one GPU inference pod",
            inject_fault=inject,
            verify_behavior=verify,
            cleanup=cleanup,
            severity="high",
            blast_radius="single_pod",
        )
```

### Chaos Test Scenarios Catalog

```yaml
Chaos_Catalog:
  tier_1_mandatory:
    description: "Must pass before any production deployment"
    experiments:
      - "Feature store unavailable → fallback predictions served"
      - "Model loading failure → previous version used"
      - "Single pod crash → remaining pods serve traffic"
      - "High latency → timeouts trigger gracefully"
      
  tier_2_important:
    description: "Should pass, blocking for high-risk systems"
    experiments:
      - "Stale features (24h old) → detected and flagged"
      - "Partial features (50% null) → graceful degradation"
      - "GPU OOM under load → queue management, no crash loop"
      - "Network partition to monitoring → predictions still served"
      - "Model returns NaN → detected, fallback used"
      
  tier_3_advanced:
    description: "Nice to have, informational"
    experiments:
      - "All replicas crash simultaneously → recovery time measured"
      - "Data corruption (wrong values) → detection latency measured"
      - "Cascading failure simulation → blast radius measured"
      - "Multi-zone failure → cross-zone recovery verified"
```

---

## How It Works in Practice

### Chaos Testing Schedule

```yaml
Schedule:
  pre_deployment:
    what: "Tier 1 chaos tests in staging"
    frequency: "Every deployment"
    blocking: true
    duration: "30-60 minutes"
    
  weekly:
    what: "Tier 1 + Tier 2 in staging"
    frequency: "Weekly (automated)"
    blocking: false
    duration: "2-4 hours"
    
  monthly:
    what: "Full chaos catalog in staging"
    frequency: "Monthly"
    includes: "Multi-failure scenarios, cascading failures"
    duration: "Half day"
    
  game_day:
    what: "Controlled chaos in production (small blast radius)"
    frequency: "Quarterly"
    requires: "On-call team ready, rollback plan prepared"
    examples: "Kill one production pod, inject stale features for 1% of traffic"
```

---

## Interview Tip

> When asked about chaos testing for ML: "I apply chaos engineering principles specifically to ML systems because ML has unique failure modes that traditional chaos testing doesn't cover. My approach: (1) Data chaos — inject stale features (24-hour-old data), null features (partial data), and corrupted values. Expected behavior: system must DETECT the issue (log alerts, degrade confidence score) and NOT silently serve wrong predictions. Most ML production incidents are data issues, so I test these first. (2) Model chaos — simulate model loading failures (corrupt artifact), OOM under load (fill GPU memory), and NaN outputs (numerical instability). Expected behavior: fall back to previous model version, use circuit breakers, return graceful errors instead of crashing. (3) Infrastructure chaos — kill GPU pods, introduce network partitions to feature stores, fill disks. Expected behavior: Kubernetes reschedules, remaining replicas absorb traffic, system recovers within SLA. I organize experiments by severity: Tier 1 (mandatory before deployment), Tier 2 (blocking for high-risk systems), Tier 3 (informational). Key principle: chaos tests are run in staging first, with automatic cleanup, smallest possible blast radius, and a kill switch. You want to discover failure modes in controlled conditions, not in production at 3 AM."

---

## Common Mistakes

1. **No fallback when feature store fails** — Feature store goes down, model receives null features, serves garbage predictions confidently. Solution: detect missing/stale features, use fallback predictions (cached, default, or degrade gracefully).

2. **Single replica with no redundancy** — One inference pod. It crashes. Zero predictions until Kubernetes restarts it (30-60 seconds). Solution: minimum 2 replicas. Chaos test: kill one pod, verify remaining pod serves traffic with acceptable latency increase.

3. **No circuit breaker on model inference** — Model takes 60 seconds for complex input. Requests queue up. Every subsequent request waits behind the slow one. Cascade failure. Solution: timeout per request (e.g., 5 seconds), circuit breaker that returns fallback after N failures.

4. **Chaos testing only in theory (never actually run)** — Team knows they "should" chaos test but never schedules it. First real failure is the first failure they've ever experienced. Solution: automate Tier 1 chaos tests in CI/CD pipeline. They run every deployment. No human effort required.

5. **Running chaos tests in production without preparation** — Inject failure in production without a rollback plan, without the on-call team aware, without monitoring the blast radius. Chaos turns into a real outage. Solution: start in staging. Production chaos (game days) requires: on-call ready, rollback plan, blast radius containment, monitoring dashboards open.

---

## Key Takeaways

- Chaos testing for ML: intentionally inject failures to verify graceful degradation
- ML-specific failures: stale features, missing features, model loading failures, GPU OOM, NaN outputs
- Data chaos: most critical — 60-80% of ML production issues are data-related
- Expected behaviors: detect issues, log alerts, use fallbacks, never serve silently wrong predictions
- Tier system: Tier 1 (mandatory/blocking), Tier 2 (important), Tier 3 (informational)
- Safety: always run in staging first, automatic cleanup, smallest blast radius, kill switch
- Fallback strategy: every ML system needs a fallback (cached predictions, default values, previous model)
- Circuit breakers: prevent cascade failures when model inference is slow or failing
- Automate Tier 1: run in CI/CD so chaos tests happen without human effort
- Game days: quarterly controlled chaos in production (with preparation and on-call ready)
