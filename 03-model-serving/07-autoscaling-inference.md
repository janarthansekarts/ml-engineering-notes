# Autoscaling Inference

## The Problem / Why This Matters

ML inference traffic is rarely constant. A chatbot might handle 100 requests/minute at 3 AM and 10,000 requests/minute at noon. A recommendation system spikes during product launches. An image generation service surges when social media trends drive traffic. Static provisioning forces a terrible choice: over-provision (waste 80% of GPU cost during quiet hours) or under-provision (drop requests during peaks, violating SLA). GPU inference autoscaling is uniquely challenging because: (1) GPU instances take 2-5 minutes to start and load models (vs 10-30 seconds for CPU), (2) GPU memory is the binding constraint (not CPU/RAM as in traditional web services), (3) tokens-per-second doesn't map cleanly to requests-per-second (variable input/output lengths), and (4) scale-to-zero saves massive cost but imposes cold start penalties. In 2026, effective GPU autoscaling is the difference between a $50K/month and a $500K/month serving bill for the same workload. The techniques: predictive scaling (anticipate demand), reactive scaling (respond to metrics), scale-to-zero (eliminate idle cost), and right-sizing (optimal instance type per traffic pattern).

---

## The Analogy

Think of inference autoscaling like managing a fleet of specialized trucks (GPUs):

- **No scaling (fixed fleet)** = Owning 100 trucks whether you need them or not. During holidays, you're short 50 trucks (dropped deliveries). On Tuesdays, 80 trucks sit in the parking lot burning insurance costs.
- **Reactive scaling** = Watching the delivery queue. When it backs up beyond 10 minutes, call in 5 more trucks. When trucks are idle for 30 minutes, send them home. Problem: calling trucks takes 5 minutes — by then the queue is already painful.
- **Predictive scaling** = Looking at historical data: "Mondays always spike at 9 AM." Pre-position trucks at 8:45 AM before the rush hits. Reactive + predictive together handle both expected and unexpected demand.
- **Scale-to-zero** = Don't keep any trucks overnight (park at a rental lot). First morning delivery waits 5 minutes while you pick up a truck. Saves all overnight costs but first customer waits.
- **Right-sizing** = Using minivans for small packages and semi-trucks for large freight. Don't waste a semi-truck's capacity on a single envelope.

---

## Deep Dive

### Scaling Metrics for GPU Inference

```yaml
Scaling_Metrics:
  key_insight: |
    Traditional web services scale on CPU utilization or request count.
    GPU inference requires DIFFERENT metrics because:
    1. GPU utilization % doesn't tell you about queue depth (GPU might be 80% busy but no queue)
    2. Request count doesn't capture variable compute cost (1 request with 10K tokens ≠ 1 with 100 tokens)
    3. Memory pressure (KV cache fullness) determines true capacity, not compute utilization
    
  recommended_metrics:
    queue_depth:
      what: "Number of requests waiting to be processed"
      why: "Directly measures user-visible latency impact"
      target: "Queue depth < 5 → comfortable, > 20 → scale up"
      advantage: "Universal — works for any model type"
      
    time_to_first_token:
      what: "TTFT (Time To First Token) — time from request arrival to first generated token"
      why: "Directly measures user-perceived responsiveness for LLMs"
      target: "< 500ms for interactive chat, < 2s for batch"
      scaling_rule: "If P95 TTFT > 1s → scale up"
      
    tokens_per_second_per_instance:
      what: "Total output tokens generated per second by one instance"
      why: "Measures actual throughput capacity utilization"
      scaling_rule: "If actual TPS (Tokens Per Second) > 80% of max TPS → scale up"
      
    kv_cache_utilization:
      what: "Percentage of available KV cache memory currently in use"
      why: "When KV cache is full, new requests are queued or rejected"
      target: "< 80% → comfortable, > 90% → scale up"
      advantage: "Accounts for variable context lengths"
      
    gpu_memory_utilization:
      what: "Total GPU memory in use (model + KV cache + overhead)"
      caution: "Less useful than KV cache utilization (model is constant)"
      
    pending_requests:
      what: "Requests received but not yet started processing"
      why: "Leading indicator — scales before latency degrades"
      
  NOT_recommended:
    gpu_compute_utilization:
      why_not: |
        LLM decode is memory-bandwidth bound, not compute bound.
        GPU compute utilization might show 30% while the GPU is actually saturated
        (waiting for memory reads). Misleading metric for LLM workloads.
```

### Reactive Autoscaling

```yaml
Reactive_Scaling:
  what: "Monitor metrics in real-time, add/remove instances based on thresholds"
  
  kubernetes_hpa:
    name: "HPA (Horizontal Pod Autoscaler)"
    default_metrics: "CPU/memory utilization"
    custom_metrics: "Queue depth, TTFT, TPS (via Prometheus adapter)"
    
    example_yaml: |
      apiVersion: autoscaling/v2
      kind: HorizontalPodAutoscaler
      metadata:
        name: llm-serving-hpa
      spec:
        scaleTargetRef:
          apiVersion: apps/v1
          kind: Deployment
          name: vllm-server
        minReplicas: 2        # Never go below 2 (redundancy)
        maxReplicas: 20       # Cost cap
        metrics:
          - type: Pods
            pods:
              metric:
                name: pending_requests
              target:
                type: AverageValue
                averageValue: "5"     # Scale when > 5 pending per pod
          - type: Pods
            pods:
              metric:
                name: kv_cache_utilization
              target:
                type: AverageValue
                averageValue: "75"    # Scale when KV cache > 75% full
        behavior:
          scaleUp:
            stabilizationWindowSeconds: 60    # Don't scale up on brief spikes
            policies:
              - type: Pods
                value: 4              # Add max 4 pods at once
                periodSeconds: 120    # Every 2 minutes at most
          scaleDown:
            stabilizationWindowSeconds: 300   # Wait 5 min before scaling down
            policies:
              - type: Pods
                value: 2              # Remove max 2 pods at once
                periodSeconds: 300    # Every 5 minutes at most
                
  challenges:
    cold_start: |
      GPU instance startup: 1-3 minutes (instance provisioning)
      Model loading: 30-120 seconds (load weights to GPU)
      Total cold start: 2-5 minutes before new instance serves traffic
      During this time: existing instances are overloaded
      
    scaling_lag: |
      Detect overload → decide to scale → provision instance → load model → ready
      Total lag: 3-7 minutes
      If traffic spike is sudden (viral content), users experience degradation for 3-7 minutes
      
    oscillation: |
      Scale up → load decreases → scale down → load increases → scale up → ...
      Solution: stabilization windows (don't scale down immediately after scaling up)
      Cooldown: wait 5-10 minutes between scale events
```

### Predictive Scaling

```yaml
Predictive_Scaling:
  what: "Use historical traffic patterns to pre-scale BEFORE demand arrives"
  
  approaches:
    time_based:
      what: "Scale based on time-of-day / day-of-week patterns"
      example: |
        - Weekdays 9 AM: scale to 10 instances (morning rush)
        - Weekdays 6 PM: scale to 5 instances (evening drop)
        - Weekends: scale to 3 instances (lower traffic)
      implementation: "Kubernetes CronHPA, AWS Predictive Scaling"
      
    ml_based_prediction:
      what: "Train a model on historical traffic to forecast future load"
      inputs: "Time, day, holidays, recent trend, external events"
      output: "Predicted request volume for next 15/30/60 minutes"
      implementation: "AWS Predictive Scaling, custom Prophet/ARIMA model"
      advantage: "Captures complex patterns (holiday effects, gradual growth)"
      
    event_driven:
      what: "Pre-scale based on known upcoming events"
      examples:
        - "Marketing campaign launching at 2 PM → pre-scale at 1:45 PM"
        - "Product launch → pre-scale 30 minutes before"
        - "Black Friday → pre-scale night before"
      implementation: "Manual trigger or integration with event calendar"
      
  combined_strategy:
    baseline: "Predictive scaling sets expected capacity"
    reactive: "HPA handles unexpected spikes above prediction"
    result: "No cold start for expected load + rapid response to unexpected"
    
  aws_predictive_scaling:
    what: "Built-in AWS service that analyzes 14 days of history"
    how: "Automatically creates scaling schedules based on detected patterns"
    lead_time: "Scales 5 minutes before predicted demand"
    integration: "Works with Auto Scaling Groups"
```

### Scale-to-Zero

```yaml
Scale_to_Zero:
  what: "Remove all instances when there's no traffic (pay nothing for idle time)"
  
  motivation: |
    Many ML models serve sporadic traffic:
    - Internal tools: used during business hours only (8 hours of 24)
    - Development/staging: rarely used
    - Long-tail models: 10 requests/day
    GPU cost: $2-4/hour × 24 hours × 30 days = $1,440-2,880/month for ONE idle GPU
    Scale-to-zero: pay only for actual usage
    
  implementations:
    kserve_knative:
      what: "KServe on Knative with serverless scaling"
      cold_start: "30-120 seconds (container start + model load for small models)"
      config: |
        annotations:
          autoscaling.knative.dev/target: "10"           # Scale at 10 concurrent requests
          autoscaling.knative.dev/scale-to-zero-grace-period: "30m"  # Wait 30 min idle before zero
          autoscaling.knative.dev/min-scale: "0"         # Allow zero pods
          
    serverless_gpu:
      what: "Cloud serverless GPU (Modal, Replicate, RunPod Serverless)"
      cold_start: "5-30 seconds (pre-baked images, fast model loading)"
      advantage: "No infrastructure management, per-second billing"
      providers:
        modal: "Pre-warm containers, fast cold start (~3-10s)"
        replicate: "Serverless model hosting, auto-scale"
        runpod_serverless: "GPU workers spin up on-demand"
        
    aws_sagemaker_serverless:
      what: "SageMaker Serverless Inference"
      limitation: "4 GB memory max (only small models)"
      cold_start: "30-60 seconds"
      use: "Small models (BERT, embeddings) with low traffic"
      
  cold_start_mitigation:
    keep_warm:
      what: "Keep 1 instance running (min_replicas=1) for latency-sensitive services"
      cost: "Pay for 1 instance 24/7, but no cold start for initial requests"
      
    pre_warming:
      what: "Ping the endpoint periodically to prevent scale-to-zero"
      implementation: "Health check every 5 minutes (cheaper than keeping GPU warm)"
      limitation: "Only prevents scale-to-zero, doesn't help with scale-up"
      
    model_caching:
      what: "Cache model on local SSD — faster reload on scale-up"
      benefit: "Reduce cold start from 120s (S3 download) to 30s (local load)"
      
    container_pre_warming:
      what: "Keep container image cached, only load model weights on scale-up"
      benefit: "Skip container pull (30-60s savings)"
```

### GPU-Specific Scaling Challenges

```yaml
GPU_Scaling_Challenges:
  instance_availability:
    problem: "GPU instances (especially H100) have limited availability"
    impact: "Scale-up request may fail — no capacity in region"
    solutions:
      - "Multi-region deployment (try us-east-1 if us-west-2 is full)"
      - "Multi-instance-type (fall back to A100 if H100 unavailable)"
      - "Capacity reservations (reserve instances in advance)"
      - "Spot/preemptible instances (cheaper but can be interrupted)"
      
  model_loading_time:
    problem: "Large model takes 1-5 minutes to load into GPU memory"
    impact: "Scale-up response time is minutes, not seconds"
    solutions:
      - "Pre-loaded warm pool (keep spare instances ready)"
      - "Model sharding (load from multiple disks in parallel)"
      - "Smaller quantized models (less data to load)"
      - "Model on local NVMe (skip network download)"
      
  memory_vs_compute_scaling:
    problem: |
      Adding a GPU instance adds both memory (for KV cache) and compute (for generation).
      But the bottleneck might be only ONE of these:
      - Long-context requests: need more memory (KV cache), not more compute
      - Short-context high-volume: need more compute, have plenty of memory
    solution: |
      Right-size instances:
      - Memory-bound: Use H200 (141 GB) instead of 2× L4 (24 GB each)
      - Compute-bound: Use multiple L4s in parallel (more total TFLOPS per dollar)
```

### Traffic Shaping and Load Shedding

```python
# Load shedding strategy for GPU inference
# When at capacity, gracefully degrade instead of crashing

from fastapi import FastAPI, HTTPException
from asyncio import Semaphore, wait_for
import time

app = FastAPI()

# Configuration
MAX_CONCURRENT_REQUESTS = 50  # Based on GPU memory / KV cache capacity
MAX_QUEUE_SIZE = 100          # Don't let queue grow unbounded
REQUEST_TIMEOUT = 30.0        # Max time to wait in queue

# Semaphore limits concurrent processing
processing_semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
current_queue_size = 0

@app.post("/v1/chat/completions")
async def generate(request: ChatRequest):
    global current_queue_size
    
    # Load shedding: reject immediately if queue is too long
    if current_queue_size >= MAX_QUEUE_SIZE:
        raise HTTPException(
            status_code=503,
            detail="Server at capacity. Retry after 5 seconds.",
            headers={"Retry-After": "5"}
        )
    
    current_queue_size += 1
    try:
        # Wait for a processing slot (with timeout)
        await wait_for(
            processing_semaphore.acquire(),
            timeout=REQUEST_TIMEOUT
        )
    except TimeoutError:
        current_queue_size -= 1
        raise HTTPException(
            status_code=504,
            detail="Request timed out waiting for capacity."
        )
    
    try:
        # Process request (actual inference)
        result = await run_inference(request)
        return result
    finally:
        processing_semaphore.release()
        current_queue_size -= 1

# Priority-based load shedding
# Premium users get priority, free tier gets shed first
@app.post("/v1/chat/completions/priority")
async def generate_priority(request: ChatRequest, tier: str = "free"):
    if tier == "free" and current_queue_size > MAX_QUEUE_SIZE * 0.7:
        # Shed free-tier traffic at 70% capacity
        raise HTTPException(status_code=503, detail="Capacity limited. Upgrade to premium.")
    
    if tier == "premium" and current_queue_size > MAX_QUEUE_SIZE * 0.95:
        # Shed premium only at 95% capacity
        raise HTTPException(status_code=503, detail="Extreme load. Retry shortly.")
    
    # ... proceed with inference
```

---

## How It Works in Practice

### Production Autoscaling Architecture

```yaml
Production_Architecture:
  multi_layer_scaling:
    layer_1_predictive:
      what: "Scheduled scaling based on historical patterns"
      timing: "Pre-scales 5-10 minutes before expected demand"
      handles: "Regular daily/weekly traffic patterns"
      
    layer_2_reactive:
      what: "HPA based on queue depth and KV cache utilization"
      timing: "Reacts within 1-2 minutes of metric threshold"
      handles: "Unexpected traffic spikes"
      
    layer_3_load_shedding:
      what: "Reject requests gracefully when at absolute capacity"
      timing: "Immediate (milliseconds)"
      handles: "Extreme overload while scaling is in progress"
      
  scaling_configuration:
    min_replicas: 2
    target_replicas: "Based on predictive model"
    max_replicas: 20
    scale_up_trigger: "Queue depth > 10 OR KV utilization > 80%"
    scale_down_trigger: "Queue depth < 2 AND KV utilization < 40% for 5+ minutes"
    cooldown_up: "60 seconds (allow frequent scale-up)"
    cooldown_down: "300 seconds (don't scale down too eagerly)"
```

---

## Interview Tip

> When asked about autoscaling GPU inference: "GPU autoscaling is harder than web service scaling for three reasons: (1) cold start is 2-5 minutes (instance + model loading), not 10 seconds; (2) the right metric isn't GPU utilization (LLM decode shows low compute utilization even when saturated); (3) the binding constraint is KV cache memory, not compute. My scaling strategy is three-layered: predictive scaling (pre-scale based on historical patterns, handles 80% of traffic), reactive HPA on queue depth and KV cache utilization (handles unexpected spikes), and load shedding (graceful rejection with Retry-After headers when all else fails). Key metric: pending requests or P95 TTFT rather than GPU utilization. For cost optimization: scale-to-zero for low-traffic models (using KServe/Knative or serverless GPU platforms like Modal), capacity reservations for baseline load, and spot instances for burst capacity. The biggest win: right-sizing — using H200 (141GB, 4.8 TB/s) for memory-bound long-context workloads instead of 2× H100 (80GB each, 3.35 TB/s)."

---

## Common Mistakes

1. **Scaling on GPU compute utilization** — LLM decode is memory-bandwidth bound. GPU SM utilization might show 30% even when the GPU is completely saturated (waiting for memory reads). This metric would say "don't scale up" when requests are already queuing. Use queue depth, KV cache utilization, or TTFT instead.

2. **No stabilization window** — Scale up at 80% utilization, new instance reduces utilization to 40%, scale down, utilization goes back to 80%, scale up again. Oscillation wastes money (instance startup/shutdown) and causes instability. Use 5-10 minute stabilization windows.

3. **Setting min_replicas = 0 for production services** — Scale-to-zero saves cost but the first user after idle period waits 2-5 minutes. Acceptable for internal tools, unacceptable for customer-facing production. Keep min_replicas ≥ 1 (preferably 2 for redundancy) for latency-sensitive services.

4. **Not pre-warming new instances** — A new GPU instance joins the load balancer before the model finishes loading. It receives requests but returns errors for 60-120 seconds. Always use readiness probes: only route traffic after model is loaded and health check passes.

5. **Single-region GPU deployment** — When your region runs out of GPU capacity (common for H100), scale-up requests fail entirely. Deploy across 2-3 regions with DNS-based routing. If primary region can't scale, overflow routes to secondary region.

---

## Key Takeaways

- GPU inference scaling is harder: 2-5 minute cold start, KV cache memory as binding constraint
- Scale on queue depth or KV cache utilization, NOT GPU compute utilization (misleading for LLMs)
- Three-layer strategy: predictive (handles expected load) + reactive (handles spikes) + load shedding (prevents crash)
- Cold start mitigation: pre-warm instances, local model storage, minimum replicas > 0
- Scale-to-zero: saves massive cost for low-traffic models (use KServe/Knative or serverless GPU)
- Stabilization windows prevent oscillation: 60s up, 300s down
- Right-size instances: H200 for memory-bound, multiple L4s for compute-bound
- Multi-region: protect against GPU capacity limits in single region
- Readiness probes: only route traffic after model is loaded and serving
- Load shedding: return 503 with Retry-After header instead of crashing under extreme load
