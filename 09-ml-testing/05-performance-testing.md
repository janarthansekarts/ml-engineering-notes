# Performance Testing for ML

## The Problem / Why This Matters

An ML model can have 99% accuracy but be completely useless if inference takes 10 seconds when the SLA (Service Level Agreement) requires 50ms. Performance testing for ML systems validates that models meet latency, throughput, and resource requirements under realistic load conditions. This is different from traditional API performance testing because ML inference has unique characteristics: GPU memory constraints (only N concurrent requests fit on one GPU), batch size trade-offs (larger batches are more efficient but add latency), model loading time (large models take seconds to load, affecting cold starts), and variable input complexity (a 10-token prompt vs. 10,000-token prompt have vastly different latencies). In 2026, with LLMs (Large Language Models) serving requests that can take 5-60 seconds (generating hundreds of tokens) and GPU resources costing $2-10/hour, performance testing is critical for both user experience and cost management. A poorly optimized serving configuration can cost 10x more than necessary or deliver unacceptable response times.

---

## The Analogy

Think of ML performance testing like testing a new highway before opening it to traffic:

- **Unit testing** = Testing that each car can drive (model produces correct output). Individual car works fine.
- **Performance testing** = Testing the highway at capacity. How many cars per minute? What happens at rush hour (peak load)? Where are the bottlenecks (GPU saturation)? What's the average travel time under normal and heavy traffic? Does adding another lane (GPU) actually help?

A highway that handles one car perfectly might gridlock at 1000 cars. Similarly, a model that infers one request in 10ms might take 500ms at 100 concurrent requests due to GPU memory contention.

---

## Deep Dive

### Performance Metrics for ML

```yaml
Performance_Metrics:
  latency:
    p50: "Median latency (typical user experience)"
    p95: "95th percentile (most users' worst case)"
    p99: "99th percentile (tail latency, worst case)"
    TTFT: "Time to First Token (LLM streaming — when user sees first word)"
    TPS: "Tokens Per Second (LLM generation speed)"
    cold_start: "Time from request to response when model isn't loaded"
    
  throughput:
    QPS: "Queries Per Second (how many requests per second)"
    tokens_per_second: "Total token generation rate (LLMs)"
    batch_throughput: "Requests processed per batch cycle"
    concurrent_users: "Maximum simultaneous users without degradation"
    
  resource_utilization:
    gpu_utilization: "Percentage of GPU compute used"
    gpu_memory: "VRAM usage vs. total available"
    cpu_utilization: "CPU usage (preprocessing often CPU-bound)"
    memory: "RAM usage (feature computation, data loading)"
    network: "Bandwidth utilization (feature retrieval, large payloads)"
    
  efficiency:
    cost_per_prediction: "Infrastructure cost ÷ predictions served"
    cost_per_token: "Infrastructure cost ÷ tokens generated (LLMs)"
    utilization_efficiency: "Actual throughput ÷ theoretical maximum"
```

### Load Testing Infrastructure

```python
# ML performance testing with Locust

"""
Performance testing for ML inference endpoints.
Uses Locust for load generation with ML-specific patterns.
"""

from locust import HttpUser, task, between, events
import time
import json
import random
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class PerformanceResult:
    """Individual request performance measurement."""
    latency_ms: float
    status_code: int
    model_version: str
    tokens_generated: int = 0
    time_to_first_token_ms: float = 0.0


class MLInferenceUser(HttpUser):
    """
    Simulates users making prediction requests.
    
    Key ML-specific considerations:
    - Variable input sizes (affects latency)
    - Different request types (simple vs. complex)
    - Realistic request patterns (not uniform)
    """
    wait_time = between(0.1, 1.0)  # Between requests
    
    # Sample inputs of varying complexity
    simple_inputs = [
        {"features": [random.random() for _ in range(10)]}
        for _ in range(100)
    ]
    complex_inputs = [
        {"features": [random.random() for _ in range(1000)]}
        for _ in range(100)
    ]
    
    @task(7)  # 70% of traffic is simple requests
    def simple_prediction(self):
        """Test simple prediction (small input)."""
        payload = random.choice(self.simple_inputs)
        
        start = time.time()
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="predict_simple"
        ) as response:
            latency_ms = (time.time() - start) * 1000
            
            if response.status_code == 200:
                result = response.json()
                if latency_ms > 100:  # SLA: 100ms for simple
                    response.failure(f"Latency {latency_ms:.0f}ms > 100ms SLA")
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(3)  # 30% of traffic is complex requests
    def complex_prediction(self):
        """Test complex prediction (large input)."""
        payload = random.choice(self.complex_inputs)
        
        start = time.time()
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="predict_complex"
        ) as response:
            latency_ms = (time.time() - start) * 1000
            
            if response.status_code == 200:
                if latency_ms > 500:  # SLA: 500ms for complex
                    response.failure(f"Latency {latency_ms:.0f}ms > 500ms SLA")
            else:
                response.failure(f"Status {response.status_code}")


class LLMInferenceUser(HttpUser):
    """
    Simulates users making LLM requests.
    
    LLM-specific considerations:
    - Streaming responses (time to first token matters)
    - Variable output length (generation time varies)
    - Long-running requests (seconds, not milliseconds)
    - Token-based metrics (tokens/sec, not just latency)
    """
    wait_time = between(1.0, 5.0)  # LLM users wait longer between requests
    
    prompts = [
        {"messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 50},
        {"messages": [{"role": "user", "content": "Explain quantum computing in 3 paragraphs."}], "max_tokens": 500},
        {"messages": [{"role": "user", "content": "Write a Python function to sort a linked list."}], "max_tokens": 1000},
    ]
    
    @task(5)
    def short_generation(self):
        """Short LLM response (< 100 tokens)."""
        payload = {
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "max_tokens": 50,
            "stream": False,
        }
        
        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            catch_response=True,
            name="llm_short"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                tokens = result.get("usage", {}).get("completion_tokens", 0)
                # Verify reasonable generation speed
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(3)
    def long_generation(self):
        """Long LLM response (500+ tokens)."""
        payload = {
            "messages": [{"role": "user", "content": "Write a detailed essay about climate change."}],
            "max_tokens": 1000,
            "stream": False,
        }
        
        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            catch_response=True,
            timeout=60,  # Long timeout for generation
            name="llm_long"
        ) as response:
            if response.status_code == 200:
                result = response.json()
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(2)
    def streaming_generation(self):
        """Streaming LLM response (measure TTFT)."""
        payload = {
            "messages": [{"role": "user", "content": "Explain recursion."}],
            "max_tokens": 200,
            "stream": True,
        }
        
        start = time.time()
        first_token_time = None
        total_tokens = 0
        
        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            stream=True,
            catch_response=True,
            name="llm_stream"
        ) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line and first_token_time is None:
                        first_token_time = time.time()
                    if line:
                        total_tokens += 1
                
                ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0
                total_ms = (time.time() - start) * 1000
                
                if ttft_ms > 2000:  # TTFT SLA: 2 seconds
                    response.failure(f"TTFT {ttft_ms:.0f}ms > 2000ms SLA")
            else:
                response.failure(f"Status {response.status_code}")
```

### Latency Profiling

```python
# Latency breakdown analysis for ML inference

"""
Profile where time is spent in the ML inference pipeline.
Essential for identifying bottlenecks.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class LatencyProfile:
    """Breakdown of inference latency by component."""
    total_ms: float = 0.0
    stages: Dict[str, float] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Human-readable latency breakdown."""
        lines = [f"Total: {self.total_ms:.1f}ms"]
        for stage, duration in sorted(self.stages.items(), key=lambda x: -x[1]):
            pct = (duration / self.total_ms * 100) if self.total_ms > 0 else 0
            lines.append(f"  {stage}: {duration:.1f}ms ({pct:.0f}%)")
        return "\n".join(lines)


class LatencyProfiler:
    """
    Profile inference latency broken down by pipeline stage.
    
    Typical breakdown for traditional ML:
    - Feature retrieval: 5-20ms (network to feature store)
    - Preprocessing: 1-5ms (normalization, encoding)
    - Model inference: 2-50ms (depends on model complexity)
    - Postprocessing: 1-5ms (threshold, formatting)
    - Overhead: 1-5ms (serialization, routing)
    
    Typical breakdown for LLM:
    - Prompt processing: 10-100ms (tokenization, prefill)
    - Token generation: 100ms-60s (autoregressive decoding)
    - Output processing: 1-10ms (detokenization, formatting)
    """
    
    def __init__(self):
        self.profiles: List[LatencyProfile] = []
    
    @contextmanager
    def measure(self, profile: LatencyProfile, stage: str):
        """Context manager to measure a pipeline stage."""
        start = time.perf_counter()
        yield
        duration_ms = (time.perf_counter() - start) * 1000
        profile.stages[stage] = duration_ms
    
    def profile_request(self, request: dict) -> LatencyProfile:
        """Profile a single inference request."""
        profile = LatencyProfile()
        total_start = time.perf_counter()
        
        # Stage 1: Feature retrieval
        with self.measure(profile, "feature_retrieval"):
            features = self._retrieve_features(request)
        
        # Stage 2: Preprocessing
        with self.measure(profile, "preprocessing"):
            processed = self._preprocess(features)
        
        # Stage 3: Model inference
        with self.measure(profile, "model_inference"):
            raw_prediction = self._infer(processed)
        
        # Stage 4: Postprocessing
        with self.measure(profile, "postprocessing"):
            result = self._postprocess(raw_prediction)
        
        profile.total_ms = (time.perf_counter() - total_start) * 1000
        self.profiles.append(profile)
        
        return profile
    
    def aggregate_report(self) -> Dict[str, Dict[str, float]]:
        """
        Aggregate latency report across all profiled requests.
        
        Returns p50, p95, p99 for each stage.
        """
        if not self.profiles:
            return {}
        
        stages = self.profiles[0].stages.keys()
        report = {}
        
        for stage in stages:
            values = [p.stages.get(stage, 0) for p in self.profiles]
            report[stage] = {
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "mean": float(np.mean(values)),
            }
        
        # Total
        totals = [p.total_ms for p in self.profiles]
        report["total"] = {
            "p50": float(np.percentile(totals, 50)),
            "p95": float(np.percentile(totals, 95)),
            "p99": float(np.percentile(totals, 99)),
            "mean": float(np.mean(totals)),
        }
        
        return report
```

### GPU Performance Testing

```yaml
GPU_Performance_Testing:
  metrics_to_measure:
    gpu_utilization:
      what: "Percentage of GPU compute cycles used"
      healthy: "> 70% during load (below = underutilized, wasting $)"
      tool: "nvidia-smi, DCGM (Data Center GPU Manager)"
      
    gpu_memory:
      what: "VRAM usage vs. total"
      healthy: "< 90% (leave headroom for batch variation)"
      concern: "OOM (Out of Memory) at high concurrency"
      
    batch_efficiency:
      what: "Throughput vs. batch size"
      test: "Measure throughput at batch sizes: 1, 2, 4, 8, 16, 32, 64"
      finding: "Usually optimal around 8-32 (GPU saturated but latency acceptable)"
      
    scaling:
      what: "Throughput vs. number of GPUs"
      test: "Measure throughput with 1, 2, 4, 8 GPUs"
      finding: "Should scale near-linearly (if not → bottleneck elsewhere)"
      
  llm_specific:
    prefill_speed:
      what: "Tokens/sec during prompt processing"
      depends_on: "Input length, model size, GPU type"
      example: "Llama 3 70B on H100: ~5000 tokens/sec prefill"
      
    decode_speed:
      what: "Tokens/sec during generation"
      depends_on: "Batch size, KV cache size, model size"
      example: "Llama 3 70B on H100: ~40-80 tokens/sec/request (batched)"
      
    kv_cache_capacity:
      what: "How many concurrent sequences can be served"
      depends_on: "Model size, sequence length, GPU memory"
      test: "Increase concurrent requests until OOM or throughput drops"
```

### Performance Test Automation

```python
# Automated performance regression testing in CI/CD

"""
Detect performance regressions automatically.
Run after model changes to verify latency/throughput still meet SLA.
"""

import json
from pathlib import Path


class PerformanceRegressionTest:
    """
    Compare current performance against baseline.
    Fail CI if performance regresses beyond threshold.
    """
    
    def __init__(self, baseline_path: str = "perf_baseline.json"):
        self.baseline = self._load_baseline(baseline_path)
    
    def _load_baseline(self, path: str) -> dict:
        """Load performance baseline (from previous release)."""
        with open(path) as f:
            return json.load(f)
    
    def check_regression(
        self,
        current_metrics: dict,
        regression_threshold: float = 0.10  # 10% regression allowed
    ) -> dict:
        """
        Compare current metrics against baseline.
        
        Args:
            current_metrics: Current performance measurements
            regression_threshold: Maximum allowed regression (0.10 = 10%)
            
        Returns:
            Dict with pass/fail and details for each metric
        """
        results = {}
        
        for metric, current_value in current_metrics.items():
            if metric not in self.baseline:
                results[metric] = {"status": "new", "value": current_value}
                continue
            
            baseline_value = self.baseline[metric]
            
            # For latency metrics: higher is worse
            if "latency" in metric or "p50" in metric or "p95" in metric or "p99" in metric:
                regression = (current_value - baseline_value) / baseline_value
                passed = regression <= regression_threshold
            # For throughput metrics: lower is worse
            elif "throughput" in metric or "qps" in metric:
                regression = (baseline_value - current_value) / baseline_value
                passed = regression <= regression_threshold
            else:
                regression = 0
                passed = True
            
            results[metric] = {
                "status": "pass" if passed else "FAIL",
                "baseline": baseline_value,
                "current": current_value,
                "regression_pct": regression * 100,
                "threshold_pct": regression_threshold * 100,
            }
        
        return results
    
    def update_baseline(self, new_metrics: dict, output_path: str):
        """Save new baseline after successful deployment."""
        with open(output_path, "w") as f:
            json.dump(new_metrics, f, indent=2)


# CI/CD integration
def run_performance_gate():
    """
    Performance gate for CI/CD pipeline.
    Returns exit code 0 (pass) or 1 (fail).
    """
    # Run load test
    current = run_load_test(duration_seconds=300, users=50)
    
    # Check regression
    checker = PerformanceRegressionTest("perf_baseline.json")
    results = checker.check_regression(current)
    
    # Report
    failures = [k for k, v in results.items() if v.get("status") == "FAIL"]
    
    if failures:
        print(f"PERFORMANCE REGRESSION DETECTED in: {failures}")
        for f in failures:
            r = results[f]
            print(f"  {f}: {r['baseline']:.1f} → {r['current']:.1f} "
                  f"({r['regression_pct']:.1f}% regression, threshold: {r['threshold_pct']}%)")
        return 1
    
    print("Performance check passed")
    return 0
```

---

## How It Works in Practice

### Performance Testing Workflow

```yaml
Workflow:
  pre_deployment:
    1_baseline: "Establish performance baseline (current production model)"
    2_load_test: "Run load test against new model (staging environment)"
    3_compare: "Compare against baseline (allow 10% regression max)"
    4_gate: "Block deployment if regression exceeds threshold"
    
  performance_budget:
    total_latency: "200ms (end-to-end, p95)"
    breakdown:
      network: "10ms"
      feature_retrieval: "30ms"
      preprocessing: "10ms"
      model_inference: "100ms"
      postprocessing: "10ms"
      overhead: "40ms (serialization, routing, buffering)"
```

---

## Interview Tip

> When asked about performance testing for ML: "I test ML inference performance across three dimensions: latency, throughput, and resource efficiency. For latency, I measure p50, p95, and p99 (not just mean — tail latency matters for user experience) at realistic load levels. For LLMs, I also measure TTFT (Time To First Token) because streaming UX depends on how quickly the first word appears, not total generation time. For throughput, I measure QPS (Queries Per Second) at the latency SLA — not maximum possible QPS which might violate latency targets. I use Locust with ML-specific patterns: variable input sizes (a 10-feature request vs. 1000-feature request), realistic traffic mix (70% simple, 30% complex), and LLM-specific metrics (tokens/second). Key optimization insights: batch inference improves throughput but hurts latency (finding the right batch size is critical), GPU utilization should be 70-85% (below 70% = wasting money, above 85% = queuing adds latency), and cold starts matter for autoscaled systems. I automate performance regression testing in CI/CD — every model change is load-tested against the baseline, and deployment is blocked if p95 latency regresses more than 10%."

---

## Common Mistakes

1. **Testing with single requests (no concurrency)** — Model handles 1 request in 10ms, so team assumes p95 will be 10ms. Under 100 concurrent requests, p95 is 500ms due to GPU contention. Solution: always test under realistic concurrent load.

2. **Measuring mean latency, not percentiles** — Mean is 50ms but p99 is 5 seconds (a few very slow requests). Mean hides tail latency that affects real users. Solution: always measure and report p50, p95, p99.

3. **Testing with uniform input sizes** — All test inputs are the same size. In production, 5% of requests have 10x larger inputs that dominate tail latency. Solution: use realistic input size distribution in load tests.

4. **No cold start testing** — System tests run after model is loaded and warm. In production, autoscaler adds new pods with cold models that take 30 seconds to load. Solution: test cold start latency separately, ensure it meets timeout requirements.

5. **Ignoring GPU memory under load** — Model fits in GPU memory with one request. Under 32 concurrent requests, KV (Key-Value) cache fills GPU memory → OOM crash. Solution: test at maximum expected concurrency, monitor GPU memory during load test.

---

## Key Takeaways

- Measure latency as percentiles (p50, p95, p99), not mean — tail latency matters
- Throughput testing: measure QPS at SLA latency, not maximum possible QPS
- TTFT (Time To First Token): critical metric for LLM streaming user experience
- Load test with realistic patterns: variable input sizes, mixed complexity, concurrent users
- GPU testing: utilization, memory under load, batch size optimization, scaling efficiency
- Latency profiling: break down where time is spent (feature retrieval vs. inference vs. preprocessing)
- Performance regression testing: automate in CI/CD, block deployment on regression > threshold
- Cold start testing: autoscaled systems need model loading time within timeout SLA
- Batch size trade-off: larger batches = higher throughput but higher per-request latency
- Performance budget: allocate latency budget across pipeline stages, track each independently
