# Online Prediction

## The Problem / Why This Matters

Online prediction (also called real-time inference) is the pattern where a model runs inference on each incoming request individually, returning predictions in milliseconds. Unlike batch prediction where results are pre-computed, online prediction responds to the current state of the world — the user's current session, the transaction happening right now, the search query just typed. In 2026, online prediction powers the most interactive ML experiences: real-time search ranking, instant fraud detection, dynamic pricing, conversational AI, and personalized content feeds. The engineering challenge is immense: you must maintain model availability at 99.99% uptime, keep latency under 50-100ms (including feature retrieval), handle traffic spikes gracefully, deploy model updates without downtime, and do all this cost-effectively. Getting online prediction wrong means slow user experiences, missed fraud events, lost revenue, and degraded trust. The pattern requires careful attention to model optimization, infrastructure scaling, feature freshness, and operational reliability.

---

## The Analogy

Think of online prediction like a surgeon in an emergency room:

- **Batch prediction** = A scheduled surgery. Everything is prepared in advance (instruments sterilized, room booked, team assembled). Efficient but inflexible — you can't handle surprise patients this way.
- **Online prediction** = Emergency room. Patient arrives unexpectedly, surgeon must assess and act immediately with whatever information is available. Fast response time is critical, but you have limited time to prepare.

The ER needs: pre-staged equipment (model loaded in memory), trained staff on standby (GPU/CPU allocated), triage protocols (request routing), and backup plans (fallback predictions). Without these, response time suffers and outcomes worsen.

---

## Deep Dive

### When to Use Online Prediction

```yaml
Use_Online_When:
  real_time_context_matters:
    - "Prediction depends on what's happening RIGHT NOW"
    - "Current search query, live transaction details, ongoing session"
    - "Cannot pre-compute: infinite possible inputs (free-text, new combinations)"
    
  freshness_critical:
    - "Stale predictions are dangerous or useless"
    - "Fraud detection: must catch fraud AS IT HAPPENS, not tomorrow"
    - "Ad bidding: must bid based on current auction, not yesterday's estimates"
    
  entity_set_unknown:
    - "Can't enumerate all possible inputs ahead of time"
    - "Search queries: infinite possible queries (can't pre-compute all)"
    - "New users: no pre-computed prediction exists (cold start)"
    
  personalization_per_interaction:
    - "Each interaction requires unique prediction"
    - "Chatbot: response depends on exact user message"
    - "Dynamic pricing: price depends on current demand, inventory, user"

Dont_Use_Online_When:
  - "Predictions don't change between interactions (use batch)"
  - "All entities known in advance (pre-compute cheaper)"
  - "Latency tolerance is > 1 minute (batch is simpler)"
  - "Model is extremely expensive and inputs are enumerable (pre-compute)"
```

### Online Prediction Architecture

```yaml
Architecture:
  model_serving_layer:
    model_server:
      options:
        triton_inference_server: "NVIDIA, multi-framework, GPU optimized, dynamic batching"
        vllm: "LLM-specific, PagedAttention, continuous batching, 2-5x throughput"
        tgi: "HuggingFace Text Generation Inference, LoRA support, production-ready"
        tensorflow_serving: "TensorFlow models, gRPC/REST, versioning"
        torchserve: "PyTorch models, custom handlers, model management"
        onnx_runtime: "Cross-framework, CPU/GPU, quantization support"
        ray_serve: "Python-native, composable, auto-scaling, multi-model"
      
    optimization:
      model_compilation: "torch.compile, TensorRT, OpenVINO for inference speedup"
      quantization: "FP16, INT8, INT4 (2-4x speedup, minimal accuracy loss)"
      batching: "Dynamic batching (wait a few ms, batch requests, amortize GPU overhead)"
      caching: "Response cache for identical inputs (exact match or semantic similarity)"
      
  feature_retrieval:
    pattern: "Features fetched at prediction time from feature store"
    sources:
      pre_computed: "Feature store (batch features, < 5ms lookup)"
      real_time: "Computed on-the-fly from request (session features)"
      streaming: "Materialized from event stream (last-N-minutes aggregates)"
    total_budget: "Feature retrieval must complete within latency budget (e.g., < 20ms)"
    
  request_flow:
    1: "Client sends prediction request (user_id, context, real-time features)"
    2: "API gateway routes to model service"
    3: "Service fetches pre-computed features from feature store"
    4: "Service computes real-time features from request context"
    5: "Combines features → model inference"
    6: "Returns prediction + metadata (confidence, model version)"
    7: "Logs request + prediction for monitoring and future training"
    
  scaling:
    horizontal: "Multiple replicas behind load balancer"
    auto_scaling: "Scale based on QPS, latency percentiles, GPU utilization"
    gpu_sharing: "Multiple models on one GPU (MPS, time-sharing)"
    traffic_management: "Rate limiting, circuit breakers, graceful degradation"
```

### Implementation

```python
# Online prediction service implementation

"""
Production online prediction service.
Handles real-time inference with feature retrieval, caching, and monitoring.
"""

import time
import hashlib
import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class PredictionRequest:
    """Incoming prediction request."""
    entity_id: str
    real_time_features: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    request_id: str = ""


@dataclass
class PredictionResponse:
    """Prediction response with metadata."""
    prediction: Any
    confidence: float
    model_version: str
    latency_ms: float
    features_used: List[str]
    cache_hit: bool = False
    fallback_used: bool = False


class OnlinePredictionService:
    """
    Online prediction service with:
    - Feature retrieval from feature store
    - Response caching (LRU)
    - Graceful degradation (fallback on failure)
    - Latency monitoring
    - Request logging for training data
    """
    
    def __init__(
        self,
        model,
        feature_store,
        cache_size: int = 10000,
        latency_budget_ms: float = 100.0,
        fallback_prediction: Any = None,
    ):
        self.model = model
        self.feature_store = feature_store
        self.cache = LRUCache(max_size=cache_size)
        self.latency_budget_ms = latency_budget_ms
        self.fallback_prediction = fallback_prediction
        
        # Metrics
        self.request_count = 0
        self.cache_hits = 0
        self.fallback_count = 0
        self.latency_histogram = []
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Handle a single prediction request.
        
        Steps:
        1. Check cache (fast path)
        2. Retrieve pre-computed features from feature store
        3. Combine with real-time features
        4. Run model inference
        5. Cache result, log for monitoring
        """
        start_time = time.perf_counter()
        self.request_count += 1
        
        try:
            # Step 1: Check cache
            cache_key = self._make_cache_key(request)
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.cache_hits += 1
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return PredictionResponse(
                    prediction=cached["prediction"],
                    confidence=cached["confidence"],
                    model_version=self.model.version,
                    latency_ms=elapsed_ms,
                    features_used=cached["features_used"],
                    cache_hit=True,
                )
            
            # Step 2: Retrieve features from feature store
            stored_features = self.feature_store.get_online_features(
                entity_id=request.entity_id,
                feature_names=self.model.expected_features
            )
            
            # Step 3: Combine features
            all_features = {**stored_features, **request.real_time_features}
            
            # Validate features (check for missing required features)
            missing = [f for f in self.model.required_features if f not in all_features]
            if missing:
                logger.warning(f"Missing features for {request.entity_id}: {missing}")
                # Use defaults for missing features
                for feat in missing:
                    all_features[feat] = self.model.feature_defaults.get(feat, 0)
            
            # Step 4: Model inference
            prediction = self.model.predict(all_features)
            confidence = self.model.predict_confidence(all_features)
            
            # Step 5: Cache and log
            self.cache.put(cache_key, {
                "prediction": prediction,
                "confidence": confidence,
                "features_used": list(all_features.keys()),
            })
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.latency_histogram.append(elapsed_ms)
            
            # Log for monitoring and training data generation
            self._log_prediction(request, prediction, confidence, all_features, elapsed_ms)
            
            # Check latency budget
            if elapsed_ms > self.latency_budget_ms:
                logger.warning(
                    f"Prediction exceeded latency budget: {elapsed_ms:.1f}ms > {self.latency_budget_ms}ms"
                )
            
            return PredictionResponse(
                prediction=prediction,
                confidence=confidence,
                model_version=self.model.version,
                latency_ms=elapsed_ms,
                features_used=list(all_features.keys()),
            )
            
        except Exception as e:
            # Graceful degradation: return fallback prediction
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.fallback_count += 1
            logger.error(f"Prediction failed for {request.entity_id}: {e}, using fallback")
            
            return PredictionResponse(
                prediction=self.fallback_prediction,
                confidence=0.0,
                model_version=self.model.version,
                latency_ms=elapsed_ms,
                features_used=[],
                fallback_used=True,
            )
    
    def _make_cache_key(self, request: PredictionRequest) -> str:
        """Create cache key from request (entity + real-time features)."""
        key_data = f"{request.entity_id}:{sorted(request.real_time_features.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def _log_prediction(
        self, request, prediction, confidence, features, latency_ms
    ):
        """
        Log prediction for:
        1. Monitoring (detect drift, latency degradation)
        2. Training data generation (predictions + outcomes → labels)
        """
        log_entry = {
            "request_id": request.request_id,
            "entity_id": request.entity_id,
            "prediction": prediction,
            "confidence": confidence,
            "model_version": self.model.version,
            "latency_ms": latency_ms,
            "timestamp": time.time(),
            "features": features,  # For debugging (consider PII)
        }
        # In production: send to Kafka/Kinesis for async processing
        # self.prediction_logger.log(log_entry)
    
    def get_metrics(self) -> Dict:
        """Return service metrics."""
        return {
            "total_requests": self.request_count,
            "cache_hit_rate": self.cache_hits / max(self.request_count, 1),
            "fallback_rate": self.fallback_count / max(self.request_count, 1),
            "p50_latency_ms": sorted(self.latency_histogram)[len(self.latency_histogram)//2] if self.latency_histogram else 0,
            "p99_latency_ms": sorted(self.latency_histogram)[int(len(self.latency_histogram)*0.99)] if self.latency_histogram else 0,
        }


class LRUCache:
    """Simple LRU (Least Recently Used) cache for prediction responses."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Dict):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)


class PrecomputeHybrid:
    """
    Hybrid pattern: pre-compute what you can, compute the rest online.
    
    Example: Recommendation system
    - Pre-computed (batch): user embeddings, item embeddings, static features
    - Real-time (online): session features, recency features, context
    - At serving: combine both → final prediction
    
    Benefits:
    - Faster than computing everything online
    - Fresher than pure batch (incorporates real-time signals)
    """
    
    def __init__(self, model, batch_store, feature_store):
        self.model = model
        self.batch_store = batch_store  # Pre-computed batch predictions/features
        self.feature_store = feature_store
    
    def predict(self, entity_id: str, session_context: Dict) -> Dict:
        """
        Combine pre-computed signals with real-time context.
        """
        # Get pre-computed base prediction/embedding (from last batch run)
        batch_signal = self.batch_store.get(entity_id)
        
        # Get real-time features
        real_time_features = self._compute_real_time_features(session_context)
        
        # Combine: batch embeddings + real-time features → final prediction
        combined_features = {
            "batch_embedding": batch_signal["embedding"],
            "batch_score": batch_signal["base_score"],
            **real_time_features,
        }
        
        final_prediction = self.model.predict(combined_features)
        
        return {
            "prediction": final_prediction,
            "batch_freshness": batch_signal["computed_at"],
            "real_time_features": list(real_time_features.keys()),
        }
    
    def _compute_real_time_features(self, session_context: Dict) -> Dict:
        """Compute features from current session/request."""
        return {
            "time_of_day": session_context.get("hour", 12),
            "session_length": session_context.get("pages_viewed", 0),
            "device_type": session_context.get("device", "desktop"),
            "last_interaction_seconds": session_context.get("seconds_since_last_action", 0),
        }
```

### Latency Optimization

```yaml
Latency_Optimization:
  model_level:
    quantization:
      what: "Reduce model precision (FP32 → FP16 → INT8)"
      speedup: "2-4x inference speedup"
      accuracy_loss: "< 1% typically"
      tools: "ONNX Runtime, TensorRT, OpenVINO"
      
    compilation:
      what: "Compile model graph for target hardware"
      speedup: "1.5-3x"
      tools: "torch.compile (PyTorch 2.x), XLA (JAX), TensorRT (NVIDIA)"
      
    distillation:
      what: "Train smaller model to mimic larger model"
      speedup: "10-100x (much smaller model)"
      accuracy_loss: "2-5% (acceptable for serving)"
      
  serving_level:
    dynamic_batching:
      what: "Collect requests, batch them, process together"
      benefit: "Higher GPU utilization (amortize fixed overhead)"
      trade_off: "Adds latency (wait time for batch to fill)"
      config: "Batch window 5-10ms, max batch size 32-64"
      
    model_warmup:
      what: "Pre-load model and run dummy inference on startup"
      benefit: "Avoid cold start latency on first real request"
      
    connection_pooling:
      what: "Reuse connections to feature store, model server"
      benefit: "Avoid TCP handshake latency (saves 10-50ms)"
      
  infrastructure_level:
    gpu_selection:
      inference_cards: "NVIDIA L4 (cost-efficient), A10G (balanced), A100/H100 (throughput)"
      specialized: "AWS Inferentia2, Google TPU v5e (cost-effective for specific models)"
      
    placement:
      what: "Deploy model close to data and users"
      strategy: "Feature store in same region, low-latency networking"
      
    auto_scaling:
      metric: "Scale on P99 latency or GPU utilization (not just CPU)"
      warmup: "Pre-warm instances (model loading takes 30s-2min)"
```

---

## How It Works in Practice

### Real-World Online Prediction System

```yaml
Example_Search_Ranking:
  user_query: "best running shoes"
  
  step_1_retrieval:
    latency: "5ms"
    action: "ANN search in product embedding index → 500 candidates"
    
  step_2_feature_retrieval:
    latency: "10ms"
    action: "Fetch user features (preferences, history) from feature store"
    
  step_3_feature_computation:
    latency: "5ms"
    action: "Compute real-time features (query-product similarity, session context)"
    
  step_4_ranking:
    latency: "15ms"
    action: "Score 500 candidates with ranking model (batched GPU inference)"
    
  step_5_post_processing:
    latency: "5ms"
    action: "Apply business rules (diversity, availability), select top 20"
    
  total_latency: "~40ms (well within 100ms budget)"
  
  scaling:
    qps: "10,000 queries per second"
    infrastructure: "8 GPU replicas (A10G), auto-scale 4-16 based on traffic"
    availability: "99.99% (multi-AZ, health checks, automatic failover)"
```

---

## Interview Tip

> When asked about online prediction: "Online prediction serves requests in real-time — typically under 100ms end-to-end. My architecture: (1) Model serving — I use Triton Inference Server for multi-framework support with dynamic batching, or vLLM for LLM serving (PagedAttention gives 2-5x throughput). Models are quantized to FP16 or INT8 using TensorRT for 2-4x speedup. (2) Feature retrieval — pre-computed features from a feature store (Redis, < 5ms lookup) combined with real-time features computed from the request. Total feature budget: < 20ms. (3) Scaling — horizontal auto-scaling based on P99 latency (not just QPS). Pre-warm instances because model loading takes 30s-2min. GPU sharing for smaller models (MPS). (4) Reliability — graceful degradation with fallback predictions, circuit breakers to feature store, response caching for repeated inputs (LRU cache can handle 30-50% hit rate for search). (5) Latency budget — I decompose the 100ms budget: 10ms feature retrieval + 5ms feature computation + 15ms model inference + 5ms post-processing + 15ms network = 50ms typical, 65ms buffer for P99. The key optimization insight: most latency comes from feature retrieval, not model inference. So I invest heavily in feature store performance (Redis in same AZ, connection pooling)."

---

## Common Mistakes

1. **Ignoring feature retrieval latency** — Optimizing model inference to 5ms but feature retrieval takes 80ms. Most online prediction latency is in feature retrieval, not model computation. Solution: put feature store in same AZ, use connection pooling, cache hot features, pre-compute what you can.

2. **No fallback on failure** — Model server goes down → 500 errors to all users. Solution: graceful degradation — return cached prediction, default prediction, or rule-based fallback. Users get a slightly worse prediction rather than an error.

3. **Cold start on deployment** — New model version deployed, first requests take 5-10 seconds (model loading, JIT compilation, cache empty). Solution: model warmup (run dummy inference on startup), canary deployment (route 1% traffic first), readiness probes (don't route traffic until warm).

4. **Not logging predictions** — Running inference but not recording what was predicted. Can't debug issues, can't compute model metrics, can't generate training data from outcomes. Solution: log every prediction asynchronously (Kafka → data warehouse). Include features for debugging.

5. **Scaling on wrong metric** — Auto-scaling on CPU utilization, but GPU is the bottleneck (or vice versa). Solution: scale on the actual constraint — GPU utilization for GPU models, P99 latency as the user-facing metric, QPS for planning capacity.

---

## Key Takeaways

- Online prediction: real-time model inference per request (< 100ms end-to-end)
- Use when: prediction depends on current context, inputs can't be enumerated, freshness critical
- Architecture: feature retrieval + model inference + caching + fallback
- Latency optimization: quantization (2-4x), compilation (1.5-3x), dynamic batching, distillation
- Feature retrieval is usually the latency bottleneck (not model inference)
- Graceful degradation: always have a fallback (cached, default, or rule-based prediction)
- Scaling: auto-scale on P99 latency or GPU utilization, pre-warm instances
- Hybrid pattern: pre-compute what you can (batch), compute the rest online (real-time signals)
- Logging: record every prediction for monitoring, debugging, and training data generation
- GPU selection: L4 (cost-efficient inference), A10G (balanced), H100 (max throughput for LLMs)
