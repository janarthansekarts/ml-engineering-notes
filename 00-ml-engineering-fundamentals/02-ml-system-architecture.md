# ML System Architecture

## The Problem / Why This Matters

Most ML tutorials show you how to train a model. Very few show you the **system** that model lives in. In production, the model is typically less than 10% of the total system code. The rest is data pipelines, feature computation, serving infrastructure, monitoring, retraining triggers, and orchestration. Google's famous 2015 paper "Hidden Technical Debt in Machine Learning Systems" illustrated this — the tiny "ML Code" box surrounded by massive infrastructure. In 2026, this is even more true: LLM (Large Language Model) applications require additional layers for prompt management, RAG (Retrieval-Augmented Generation) pipelines, vector databases, guardrails, evaluation, and multi-model routing. Understanding ML system architecture means understanding how all these components connect, what data flows where, and what fails when something breaks. Without this architectural view, you build fragile systems that work in demos but collapse under real traffic.

---

## The Analogy

Think of an ML system like a modern hospital:

- **The model** = The surgeon. Highly skilled, does the critical work. But the surgeon is useless without the rest of the hospital.
- **Data pipelines** = The ambulance system and intake process. Getting patients (data) to the right place, cleaned and prepped.
- **Feature store** = The patient's medical records. Pre-computed, instantly available, consistent across departments.
- **Serving infrastructure** = The operating room. Sterile environment, monitoring equipment, proper tools, ready to go.
- **Monitoring** = ICU (Intensive Care Unit) post-surgery. Watching vital signs, alerting when something goes wrong.
- **Retraining pipeline** = Continuing medical education. The surgeon keeps learning from new cases.

A hospital that only invests in great surgeons but has terrible intake, broken equipment, and no post-op monitoring will kill patients. Same with ML systems.

---

## Deep Dive

### The ML System Lifecycle

```yaml
ML_System_Lifecycle:
  phase_1_data:
    description: "Collect, validate, transform, and store training data"
    components:
      data_sources: "Databases, APIs, event streams, logs, third-party data"
      data_validation: "Schema checks, statistical tests, freshness verification"
      data_transformation: "Cleaning, normalization, encoding, augmentation"
      data_storage: "Data lakes (S3/GCS), feature stores, vector databases"
    failures_here: "Garbage in, garbage out — most ML failures trace back to data issues"
    
  phase_2_training:
    description: "Train, evaluate, and select the best model"
    components:
      experiment_tracking: "MLflow, Weights & Biases — log params, metrics, artifacts"
      training_infrastructure: "GPU clusters (H100/H200/B200), distributed training"
      hyperparameter_optimization: "Optuna, Ray Tune, Bayesian optimization"
      model_evaluation: "Offline metrics, fairness analysis, error analysis"
      model_registry: "Versioned model storage with metadata and approval workflows"
    failures_here: "Training-serving skew, non-reproducible experiments, resource waste"
    
  phase_3_serving:
    description: "Deploy model and serve predictions to production traffic"
    components:
      model_server: "vLLM, TGI, Triton Inference Server, TorchServe, KServe"
      inference_optimization: "Quantization (INT4/INT8), batching, caching, speculative decoding"
      deployment_strategy: "Canary, blue-green, shadow mode, A/B testing"
      api_gateway: "Rate limiting, authentication, routing, load balancing"
      autoscaling: "Scale GPU pods based on queue depth, latency, or traffic"
    failures_here: "Latency spikes, cost overruns, cold starts, model loading delays"
    
  phase_4_monitoring:
    description: "Continuously observe model behavior and system health"
    components:
      data_drift: "Monitor input feature distributions vs training distribution"
      model_drift: "Track prediction accuracy over time (when ground truth available)"
      system_metrics: "Latency (p50/p95/p99), throughput, error rates, GPU utilization"
      business_metrics: "Click-through rate, conversion, revenue impact"
      alerting: "PagerDuty/Opsgenie integration, SLO (Service Level Objective) violations"
    failures_here: "Silent degradation — model accuracy drops but nobody notices for weeks"
    
  phase_5_feedback_and_retraining:
    description: "Close the loop — use production data to improve the model"
    components:
      ground_truth_collection: "User feedback, delayed labels, human evaluation"
      trigger_evaluation: "Drift threshold exceeded? Schedule-based? Performance-based?"
      automated_retraining: "Triggered pipeline that trains, evaluates, and promotes new model"
      champion_challenger: "New model serves shadow traffic, compared against current model"
    failures_here: "Feedback loops that amplify bias, stale models serving outdated predictions"
```

### Reference Architecture (2026)

```yaml
Reference_Architecture:
  data_layer:
    batch_data:
      source: "Data warehouse (BigQuery, Snowflake, Redshift)"
      pipeline: "Airflow/Dagster → transform → feature store (offline)"
      refresh: "Daily or hourly batch jobs"
    streaming_data:
      source: "Kafka/Kinesis event streams"
      pipeline: "Flink/Spark Streaming → feature store (online)"
      refresh: "Real-time (seconds to minutes)"
    vector_data:
      source: "Documents, embeddings, knowledge bases"
      pipeline: "Chunking → embedding model → vector DB (Pinecone/Qdrant/pgvector)"
      refresh: "Event-driven (new documents trigger re-indexing)"

  training_layer:
    orchestration: "Kubeflow Pipelines / Vertex AI Pipelines / SageMaker Pipelines"
    compute: "GPU clusters (H100/H200), spot instances for cost optimization"
    experiment_tracking: "MLflow or Weights & Biases"
    model_registry: "MLflow Model Registry / Vertex AI Model Registry"
    artifact_storage: "S3/GCS with versioning"

  serving_layer:
    traditional_ml:
      server: "Triton Inference Server or KServe on Kubernetes"
      format: "ONNX, TorchScript, or native framework"
      optimization: "Quantization, graph optimization, batching"
    llm_serving:
      server: "vLLM or TGI (Text Generation Inference) on GPU nodes"
      optimization: "PagedAttention, continuous batching, speculative decoding"
      caching: "Semantic caching, KV cache management"
    gateway:
      tool: "Kong, Envoy, or custom LLM gateway (LiteLLM, Portkey)"
      features: "Rate limiting, model routing, fallback, cost tracking"

  monitoring_layer:
    system_monitoring: "Prometheus + Grafana (latency, throughput, errors)"
    ml_monitoring: "Evidently AI, Arize, Whylabs (drift, quality)"
    llm_monitoring: "LangSmith, Langfuse (traces, token usage, quality scores)"
    alerting: "PagerDuty / Opsgenie with SLO-based alerts"
    dashboards: "Grafana dashboards per model, per feature, per business metric"

  feedback_layer:
    user_signals: "Thumbs up/down, corrections, implicit signals (clicks, time-on-page)"
    evaluation: "LLM-as-judge, human evaluation queues, automated benchmarks"
    retraining: "Scheduled or drift-triggered pipeline → train → evaluate → promote"
```

### Architecture Patterns

```yaml
Architecture_Patterns:
  pattern_1_batch_prediction:
    description: "Pre-compute predictions for all entities, store results"
    when_to_use: "Predictions needed for known entities, latency not critical"
    example: "Nightly churn scores for all customers"
    architecture: "Scheduled pipeline → model inference → results DB → application reads"
    pros: "Simple, cheap, no real-time serving needed"
    cons: "Stale predictions, can't handle new entities"
    
  pattern_2_online_prediction:
    description: "Real-time inference on each request"
    when_to_use: "Low-latency needed, dynamic inputs, personalization"
    example: "Real-time fraud detection on each transaction"
    architecture: "Request → feature store (online) → model server → response"
    pros: "Fresh predictions, handles any input"
    cons: "Complex infrastructure, latency constraints, cost at scale"
    
  pattern_3_llm_application:
    description: "Foundation model + context + tools for intelligent responses"
    when_to_use: "Open-ended tasks, natural language, reasoning required"
    example: "Customer support agent, document analysis, code generation"
    architecture: "Request → RAG retrieval → prompt construction → LLM → guardrails → response"
    pros: "Flexible, powerful, handles unstructured tasks"
    cons: "Expensive (token costs), unpredictable latency, hallucination risk"
    
  pattern_4_hybrid:
    description: "Traditional ML for fast/cheap tasks + LLM for complex reasoning"
    when_to_use: "Mixed workload — some tasks need speed, others need intelligence"
    example: "ML classifier routes tickets → simple ones auto-resolved → complex ones sent to LLM agent"
    architecture: "Router (fast ML model) → branch to cheap/fast path OR expensive/smart path"
    pros: "Cost-efficient, best-of-both-worlds"
    cons: "More complex to build and maintain"
```

---

## How It Works in Practice

### System Architecture at a Real Company

```yaml
Example:
  company: "E-commerce platform (10M daily users)"
  ml_systems:
    recommendation_engine:
      architecture: "Online prediction"
      serving: "Triton on K8s, 50ms p99 latency"
      features: "Feature store (Feast) — user history, item embeddings, real-time session"
      model: "Two-tower retrieval + ranking model, retrained daily"
      monitoring: "Recommendation CTR (click-through rate), diversity metrics"
      
    search_ranking:
      architecture: "Online prediction + LLM reranking"
      serving: "Stage 1: BM25 retrieval (Elasticsearch), Stage 2: ML ranker (Triton)"
      llm_component: "Top-10 results reranked by Gemini 2.5 Flash for relevance"
      latency_budget: "BM25: 20ms, ML rank: 30ms, LLM rerank: 150ms"
      
    customer_support:
      architecture: "LLM application (agent-based)"
      serving: "LangGraph agent on vLLM (self-hosted Llama 4)"
      rag: "Product docs + order history in Qdrant vector DB"
      guardrails: "Input/output filtering, hallucination check, escalation rules"
      fallback: "Route to human agent if confidence < threshold"
      
    fraud_detection:
      architecture: "Online prediction (streaming)"
      serving: "Custom model on Triton, 10ms p99 (hard requirement)"
      features: "Real-time transaction velocity, device fingerprint, location"
      model: "XGBoost ensemble, retrained every 4 hours on new fraud labels"
```

---

## Interview Tip

> When asked "Describe an ML system architecture," structure your answer around the lifecycle: "I'd design this with five layers: (1) Data layer — batch features from the warehouse plus streaming features for real-time signals, feeding a feature store for training/serving consistency. (2) Training layer — orchestrated pipelines with experiment tracking, running on GPU clusters with automated hyperparameter tuning. (3) Serving layer — model server with dynamic batching and quantization, behind an API gateway with rate limiting. (4) Monitoring layer — data drift detection, model performance tracking, system metrics, with SLO-based alerts. (5) Feedback layer — ground truth collection feeding automated retraining when drift is detected. For LLM applications, I'd add a RAG pipeline, prompt management, guardrails, and LLM-specific observability." This shows you think in systems, not just models.

---

## Common Mistakes

1. **Designing for the model, not the system** — Spending weeks optimizing model accuracy from 94% to 95%, while the serving infrastructure has 5% error rates and 2-second latency. Fix the system first.

2. **No training-serving parity** — Computing features differently in training (Python/Pandas) vs serving (Java/real-time). This "training-serving skew" silently destroys model performance.

3. **Monolithic architecture** — Putting data processing, training, serving, and monitoring in one giant codebase. When one component needs to change, everything breaks.

4. **Ignoring the data layer** — Building elaborate model infrastructure while data arrives late, with missing values, or schema changes. Data quality infrastructure should be built FIRST.

5. **No feedback loop** — Deploying a model and never closing the loop. Without ground truth collection and retraining, models decay silently over months.

---

## Key Takeaways

- The model is <10% of the production ML system — the rest is data, serving, monitoring, and orchestration
- Five lifecycle phases: Data → Training → Serving → Monitoring → Feedback/Retraining
- Architecture patterns: batch prediction, online prediction, LLM application, hybrid
- Training-serving skew is the silent killer — use feature stores for consistency
- LLM applications (2026) add: RAG pipelines, vector databases, prompt management, guardrails, LLM-specific monitoring
- Always design for failure: fallbacks, graceful degradation, human escalation paths
- Monitoring is not optional — silent model degradation is the norm without active drift detection
