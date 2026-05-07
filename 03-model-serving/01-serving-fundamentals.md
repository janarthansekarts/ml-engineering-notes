# Serving Fundamentals

## The Problem / Why This Matters

A trained model sitting on disk is useless — it only creates value when it's serving predictions to users, applications, or downstream systems. Model serving is the discipline of making trained models available for inference in production: handling requests, managing compute resources, meeting latency requirements, and maintaining reliability. The challenges are fundamentally different from training: training optimizes for throughput (process as many samples as possible), but serving often optimizes for latency (respond to a single request within milliseconds). Training runs on dedicated GPU clusters for hours/days; serving runs 24/7 on shared infrastructure handling unpredictable traffic patterns. In 2026, with LLMs (Large Language Models) generating tokens one at a time (autoregressive), serving has become even more complex — a single inference call can take seconds and consume significant GPU memory for the KV (Key-Value) cache. Understanding serving patterns (online vs batch, latency vs throughput), SLAs (Service Level Agreements), deployment strategies, and optimization techniques is essential for ML engineers who need to deliver models to production users reliably and cost-effectively.

---

## The Analogy

Think of model serving like running a restaurant:

- **Batch inference** = Catering. You prepare 1000 meals in advance, deliver them at a scheduled time. Efficient (cook in bulk) but not responsive (can't serve someone who walks in now).
- **Online inference** = Restaurant service. Each customer orders individually, expects food within 10 minutes. Must handle variable traffic (lunch rush vs quiet evening), maintain quality under pressure, and never keep someone waiting too long.
- **Streaming inference** = Sushi conveyor belt. Food is continuously generated and delivered piece by piece (token by token for LLMs). The customer starts eating before the full order is complete.
- **Latency SLA** = Your restaurant guarantee: "Served within 10 minutes or it's free." Every infrastructure decision optimizes for meeting this guarantee.

---

## Deep Dive

### Serving Patterns

```yaml
Serving_Patterns:
  online_serving:
    description: "Synchronous request-response (user waits for prediction)"
    characteristics:
      latency: "Critical (p50 < 100ms for traditional ML, < 2s for LLMs)"
      throughput: "Variable (depends on traffic patterns)"
      availability: "Must be always-up (99.9%+ SLA)"
      scaling: "Must handle traffic spikes (auto-scaling)"
    examples:
      - "Search ranking (result in <200ms)"
      - "Fraud detection (decision before transaction completes)"
      - "Chatbot (response within 2-5 seconds)"
      - "Recommendation (personalized in <100ms)"
    infrastructure:
      - "Load balancer → model server fleet → GPU/CPU"
      - "Auto-scaling based on request rate and latency"
      - "Health checks and automatic failover"
      
  batch_inference:
    description: "Process large volumes of data offline (no user waiting)"
    characteristics:
      latency: "Not critical (minutes to hours acceptable)"
      throughput: "Maximize (process as much data as possible)"
      scheduling: "Run periodically (hourly, daily) or triggered"
      cost: "Optimize for throughput/dollar (spot instances)"
    examples:
      - "Nightly recommendation recomputation for all users"
      - "Document classification of new uploads"
      - "Embedding generation for vector search index"
      - "Monthly churn prediction for all customers"
    infrastructure:
      - "Job scheduler (Airflow, Kubernetes Jobs)"
      - "Large batch sizes (maximize GPU utilization)"
      - "Spot/preemptible instances (cheaper for non-urgent work)"
      
  streaming_inference:
    description: "Real-time inference on continuous data streams"
    characteristics:
      latency: "Low (sub-second for each event)"
      ordering: "May need to respect event order"
      state: "May require maintaining state across events"
    examples:
      - "Fraud detection on transaction stream"
      - "Anomaly detection on IoT sensor data"
      - "Real-time content moderation on social media"
    infrastructure:
      - "Kafka/Kinesis → model service → output stream"
      - "Stateful inference (sliding windows, session context)"
      
  llm_streaming:
    description: "Token-by-token generation streamed to client (SSE/WebSocket)"
    characteristics:
      ttft: "Time To First Token (critical UX metric: <500ms)"
      throughput: "Tokens per second per request"
      total_time: "Can be seconds-minutes for long generations"
    examples:
      - "ChatGPT-style interfaces (tokens appear as generated)"
      - "Code completion (suggestions stream as you type)"
    infrastructure:
      - "Server-Sent Events (SSE) or WebSocket"
      - "Continuous batching (process multiple requests simultaneously)"
```

### Latency vs Throughput

```yaml
Latency_vs_Throughput:
  definitions:
    latency: "Time from request received to response sent (single request)"
    throughput: "Total requests processed per unit time (system capacity)"
    trade_off: "Optimizing one often hurts the other"
    
  the_trade_off:
    batching_example:
      no_batching: "Latency: 10ms, Throughput: 100 req/s (process one at a time)"
      batch_8: "Latency: 30ms (wait to fill batch), Throughput: 400 req/s (GPU parallelism)"
      batch_32: "Latency: 100ms (longer wait), Throughput: 800 req/s"
    insight: "Larger batches improve throughput but increase latency (waiting + processing time)"
    
  sla_definitions:
    p50_latency: "Median — 50% of requests complete faster than this"
    p95_latency: "95th percentile — only 5% of requests are slower"
    p99_latency: "99th percentile — only 1% of requests are slower"
    why_p99_matters: "Users remember the worst experiences. P99 affects user perception more than P50"
    
  typical_sla_targets:
    traditional_ml:
      search_ranking: "p99 < 200ms"
      recommendation: "p99 < 100ms"
      fraud_detection: "p99 < 50ms"
    llm_serving:
      ttft: "p95 < 500ms (time to first token)"
      generation_speed: ">30 tokens/sec per request"
      total_response: "p95 < 5s for short responses"
      
  optimization_approaches:
    for_latency:
      - "Model quantization (INT8/INT4 — faster computation)"
      - "Smaller models (fewer parameters = faster forward pass)"
      - "Caching (serve cached responses for repeated queries)"
      - "Speculative decoding (predict ahead, verify in parallel)"
      - "Hardware (faster GPUs, inference-specific chips)"
    for_throughput:
      - "Batching (process multiple requests simultaneously)"
      - "Continuous batching (don't wait for batch to fill)"
      - "Model parallelism (split model across GPUs for larger batch)"
      - "Replication (multiple model instances behind load balancer)"
      - "Async processing (decouple request from processing)"
```

### Model Serving Architecture

```yaml
Serving_Architecture:
  components:
    load_balancer:
      purpose: "Distribute requests across model server replicas"
      options: "NGINX, Envoy, cloud LB (ALB, GCP LB)"
      routing: "Round-robin, least-connections, or model-aware routing"
      
    api_gateway:
      purpose: "Authentication, rate limiting, request validation"
      options: "Kong, AWS API Gateway, Envoy"
      features: "API keys, usage quotas, request/response transformation"
      
    model_server:
      purpose: "Load model, execute inference, return predictions"
      options: "vLLM, TGI, Triton, TorchServe, custom FastAPI"
      responsibilities:
        - "Model loading and GPU memory management"
        - "Request batching"
        - "Pre/post processing"
        - "Health checking"
        
    model_registry:
      purpose: "Store and version model artifacts"
      options: "MLflow Model Registry, S3/GCS, HuggingFace Hub"
      enables: "Versioned deployment, rollback, A/B testing"
      
    monitoring:
      purpose: "Track latency, errors, throughput, model quality"
      options: "Prometheus + Grafana, Datadog, custom dashboards"
      metrics:
        - "Request latency (p50, p95, p99)"
        - "Throughput (requests/sec)"
        - "Error rate"
        - "GPU utilization"
        - "Queue depth"
        
  deployment_patterns:
    single_model:
      description: "One model per service (simplest)"
      when: "Single model serving one API endpoint"
      
    multi_model:
      description: "Multiple models in one service (shared resources)"
      when: "Many small models, or model routing (ensemble, cascade)"
      
    model_mesh:
      description: "Distributed model serving across a cluster"
      when: "Hundreds of models with variable traffic"
      tools: "KServe ModelMesh, Seldon"
```

### Cost of Serving

```yaml
Serving_Cost:
  cost_drivers:
    compute:
      description: "GPU/CPU instances running model servers"
      typical: "$2-6/hour per H100, $0.50-1/hour per L4"
      optimization: "Right-size GPUs, auto-scale, use cheaper hardware where possible"
      
    always_on:
      description: "Servers must run 24/7 for online serving (unlike training)"
      impact: "Monthly cost = hourly_rate × 720 hours"
      example: "1× H100 always-on: $2.50/hr × 720 = $1,800/month"
      optimization: "Scale-to-zero during low traffic, serverless for bursty"
      
    over_provisioning:
      description: "Must provision for PEAK traffic, not average"
      impact: "If peak is 5x average, 80% of capacity is wasted at average load"
      optimization: "Auto-scaling, traffic prediction, queue-based smoothing"
      
  cost_per_inference:
    calculation: "server_cost_per_hour / requests_per_hour"
    example_traditional_ml:
      hardware: "1× L4 GPU ($0.50/hour)"
      throughput: "10,000 req/sec = 36M req/hour"
      cost_per_request: "$0.50 / 36,000,000 = $0.000014 (negligible)"
    example_llm:
      hardware: "1× H100 ($2.50/hour)"
      throughput: "20 concurrent requests × 50 tok/sec = 1000 tok/sec"
      cost_per_1000_tokens: "$2.50 / 3600 × 1000 = $0.0007"
      context: "OpenAI charges ~$0.01-0.06 per 1K tokens (their margin)"
      
  cost_optimization:
    quantization: "INT4 model uses 4x less memory → fit more on one GPU → cheaper per request"
    batching: "Process 32 requests simultaneously → 10x more throughput per dollar"
    caching: "Repeated queries return cached response → zero inference cost"
    right_sizing: "Use L4 instead of H100 if latency requirements allow"
    scale_to_zero: "Pay nothing during zero-traffic periods"
```

---

## How It Works in Practice

### Example: Serving Architecture for a Chatbot

```yaml
Example:
  application: "Enterprise chatbot (10K concurrent users, <2s response time)"
  
  architecture:
    frontend: "Web app → API Gateway (rate limiting, auth)"
    routing: "Request classifier → route to appropriate model"
    
    serving_layer:
      simple_queries:
        model: "Llama-4-8B (quantized INT4)"
        hardware: "2× L4 GPU"
        throughput: "200 req/s"
        latency: "TTFT < 200ms"
        traffic_share: "70% of queries"
        
      complex_queries:
        model: "Llama-4-70B (INT4 quantized)"
        hardware: "2× H100 (tensor parallel)"
        throughput: "20 req/s"
        latency: "TTFT < 500ms"
        traffic_share: "30% of queries"
        
    caching:
      exact_cache: "Redis — cache exact query matches"
      semantic_cache: "Embedding similarity — cache similar queries"
      hit_rate: "~30% (saves compute on repeated queries)"
      
    auto_scaling:
      metric: "Queue depth + p95 latency"
      min_replicas: 2
      max_replicas: 10
      scale_up: "When p95 > 1.5s for 2 minutes"
      scale_down: "When GPU util < 30% for 10 minutes"
      
  monthly_cost:
    compute: "2× L4 ($720) + 2× H100 ($3,600) = $4,320/month baseline"
    with_scaling: "Average ~$6,000/month (peak hours scale up)"
    caching_savings: "30% fewer inference calls = ~$1,800 saved"
    net_cost: "~$4,200/month for 10K concurrent users"
```

---

## Interview Tip

> When asked about model serving: "I think about serving across four dimensions: (1) Pattern — online (low-latency per-request), batch (high-throughput offline), or streaming (continuous event processing). LLMs add streaming generation (token-by-token SSE). (2) SLA — p50/p95/p99 latency targets and availability requirements drive architecture decisions. (3) Cost — the trade-off between always-on capacity (expensive) and auto-scaling (complex). Scale-to-zero for low-traffic models, right-size GPU for the workload, batch for throughput. (4) Architecture — load balancer → API gateway → model server (vLLM, Triton) → monitoring. For LLM serving specifically: Time To First Token (TTFT) is the critical UX metric, continuous batching maximizes GPU utilization, and quantization (INT4) reduces cost 4x with minimal quality loss."

---

## Common Mistakes

1. **No latency budgeting** — Building a serving pipeline (preprocessing → model → postprocessing) without allocating latency budget to each step. If SLA is 200ms total and you spend 150ms on preprocessing, the model has only 50ms — which may require a smaller/faster model than planned.

2. **Always-on without traffic analysis** — Running GPU servers 24/7 when traffic is only active 8 hours/day. This wastes 66% of GPU cost. Implement scale-to-zero or scheduled scaling for predictable traffic patterns.

3. **Optimizing for wrong metric** — Maximizing throughput when users care about latency. Adding aggressive batching improves throughput but hurts individual request latency. Always optimize for the metric that matches your SLA.

4. **No graceful degradation** — When the system is overloaded, requests queue up and ALL users get terrible latency. Better: shed load (reject excess with 429), use simpler/faster model under load, or serve cached responses.

5. **Ignoring cold start** — First request after model loading takes much longer (seconds-minutes for large models). Users hitting cold-start see terrible latency. Solutions: keep warm instances, pre-warm on schedule, or use serverless with provisioned concurrency.

---

## Key Takeaways

- Online serving: low-latency, always-available — the core pattern for user-facing ML
- Batch inference: high-throughput, offline — for background processing (recommendations, embeddings)
- LLM streaming: token-by-token generation — TTFT (Time To First Token) is the critical UX metric
- Latency vs throughput trade-off: batching improves throughput but increases per-request latency
- SLA targets: define p50/p95/p99 latency, availability (99.9%+), and error rate budgets
- Cost: always-on GPU is expensive ($1,800/month for one H100) — auto-scale, cache, right-size
- Architecture: load balancer → API gateway → model server → monitoring → auto-scaling
- Serving cost per request: traditional ML ($0.00001) vs LLM ($0.001-0.01) — 100-1000x difference
- Graceful degradation: reject excess load > queue indefinitely (better: 429 vs infinite queue)
- Cold start: first request after model load is slow — pre-warm or keep minimum warm instances
