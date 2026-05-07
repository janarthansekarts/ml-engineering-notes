# Feature Serving

## The Problem / Why This Matters

Feature serving is the critical-path operation that retrieves feature values at prediction time — it directly determines whether your model can meet its latency SLA (Service Level Agreement). When a user clicks "Buy Now" and your fraud detection model needs to score the transaction, the model needs 50+ features (user history, transaction velocity, device fingerprint, merchant risk) delivered in under 10 milliseconds. If feature serving is slow, your entire prediction pipeline is slow. If it's unreliable, your model can't score at all (degraded experience or blocked transactions). The engineering challenges: (1) low latency — features must be retrieved from storage in single-digit milliseconds, (2) high throughput — serving thousands to millions of predictions per second, (3) high availability — if the feature store is down, predictions fail, (4) consistency — features served at prediction time must match what was used in training, and (5) cost efficiency — storing and serving billions of feature values isn't free. Feature serving is the bridge between your carefully engineered features and your production model — and it's where many ML systems have their worst bottlenecks.

---

## The Analogy

Think of feature serving like a fast-food kitchen's order assembly:

- **The order** = Prediction request ("score this transaction for fraud"). Needs specific ingredients (features) assembled quickly.
- **The hot holding station** = Online feature store (Redis). Pre-cooked items kept warm and ready. Grab items by name instantly. This is where 90% of features come from.
- **The prep station** = On-demand computation. Some items made fresh per order (request-context features). Fast but adds time.
- **The walk-in freezer** = Offline store. Has everything historically but takes too long to access during rush hour. Used for prep work (training), not live orders (serving).
- **The assembly line** = Feature serving layer. Gathers all ingredients (pre-made + fresh), assembles the complete plate (feature vector), passes to the chef (model) for final check.
- **The SLA** = "Customer gets their food in 30 seconds." Feature serving must complete its part in under 10ms to leave room for model inference.

---

## Deep Dive

### Feature Serving Architecture

```yaml
Serving_Architecture:
  request_flow:
    1: "Prediction request arrives (entity_id + context)"
    2: "Feature serving layer receives request"
    3: "Parallel retrieval:"
      a: "Lookup pre-computed features from online store (Redis/DynamoDB)"
      b: "Compute on-demand features from request context"
    4: "Merge all features into feature vector"
    5: "Apply transformations (normalization, encoding)"
    6: "Return feature vector to model serving layer"
    
  latency_breakdown:
    typical_budget: "50ms total prediction latency"
    allocation:
      network_overhead: "5ms (request/response)"
      feature_retrieval: "5-10ms"
      feature_transformation: "1-2ms"
      model_inference: "10-30ms"
      post_processing: "2-5ms"
    implication: "Feature serving gets 5-10ms — every millisecond counts"
    
  components:
    online_store:
      what: "Low-latency key-value store with pre-computed features"
      technology: "Redis Cluster, DynamoDB, Bigtable, ScyllaDB"
      data_model: "entity_id → {feature_name: value, ...}"
      access_pattern: "Multi-get (retrieve multiple features for one entity)"
      
    feature_serving_api:
      what: "Service that handles feature requests from models"
      technology: "gRPC service (lowest latency) or REST API"
      operations:
        get_features: "Retrieve features for specific entities"
        batch_get: "Retrieve features for multiple entities (batch prediction)"
        
    transformation_layer:
      what: "Apply feature transformations at serving time"
      operations:
        - "Normalization (z-score, min-max)"
        - "Encoding (one-hot, ordinal)"
        - "Imputation (fill missing values)"
        - "Feature crossing (combine features)"
      requirement: "Must be identical to training transformations"
```

### Low-Latency Serving Patterns

```yaml
Latency_Optimization:
  redis_patterns:
    single_key_per_entity:
      what: "Store all features for an entity in one Redis hash"
      key: "features:{entity_type}:{entity_id}"
      value: "Hash with field per feature"
      read: "HMGET features:user:1001 purchase_count_30d total_spend_30d ..."
      latency: "~1ms for local Redis, ~3ms cross-AZ"
      pros: "Single round trip, simple"
      cons: "All features must be from same entity"
      
    pipeline_multi_entity:
      what: "Batch multiple entity lookups in one Redis pipeline"
      when: "Need features from multiple entities (user + item + merchant)"
      implementation: "Redis PIPELINE with multiple HMGET commands"
      latency: "Same as single lookup (one round trip for all)"
      
    cluster_locality:
      what: "Co-locate related features on same Redis shard"
      how: "Use hash tags: {user:1001}:activity, {user:1001}:demographics"
      benefit: "Related features on same shard — no cross-shard coordination"
      
  caching_strategies:
    request_level_cache:
      what: "Cache complete feature vectors for recently seen entities"
      technology: "In-process cache (LRU) in the feature serving service"
      hit_rate: "30-60% for popular entities (power law distribution)"
      benefit: "Zero-latency for cache hits (no Redis round trip)"
      ttl: "5-60 seconds (depends on feature freshness requirements)"
      
    pre_computation_cache:
      what: "Pre-compute and cache feature vectors for entities likely to be queried"
      trigger: "When entity becomes 'active' (login, session start)"
      benefit: "Features ready before first prediction request"
      
    negative_caching:
      what: "Cache 'not found' results to avoid repeated misses"
      why: "New entities have no features yet — don't hit Redis every time"
      ttl: "Short (30s) — features may be computed soon"
      
  pre_computation:
    what: "Compute expensive features in advance, serve from fast storage"
    when: "Feature is too expensive to compute per-request but needs freshness"
    examples:
      - "User embeddings (recomputed hourly, served from Redis)"
      - "Item similarity scores (recomputed daily, served from cache)"
      - "Risk scores (recomputed every 15 min, served immediately)"
    pattern: "Async pipeline computes → writes to Redis → serving reads instantly"
```

### Feature Serving Implementation

```python
# Production feature serving service

import asyncio
import redis.asyncio as redis
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class FeatureRequest:
    """Request for features."""
    entity_type: str  # "user", "item", "merchant"
    entity_id: str
    feature_names: list[str]


@dataclass
class FeatureResponse:
    """Response with feature values."""
    features: dict[str, any]
    metadata: dict  # freshness, source, etc.
    latency_ms: float


class FeatureServingService:
    """Low-latency feature serving with caching and fallbacks."""
    
    def __init__(self, redis_pool: redis.ConnectionPool):
        self.redis = redis.Redis(connection_pool=redis_pool)
        self.local_cache = LRUCache(max_size=10000, ttl_seconds=30)
        self.metrics = MetricsCollector()
        
    async def get_features(
        self,
        entities: list[FeatureRequest],
        on_demand_context: Optional[dict] = None,
    ) -> list[FeatureResponse]:
        """Get features for multiple entities in parallel."""
        
        start = time.perf_counter()
        
        # 1. Check local cache
        cached_results, cache_misses = self._check_cache(entities)
        
        # 2. Fetch from Redis (only misses)
        if cache_misses:
            redis_results = await self._fetch_from_redis(cache_misses)
            # Update cache
            for entity, result in zip(cache_misses, redis_results):
                self._update_cache(entity, result)
        else:
            redis_results = []
        
        # 3. Compute on-demand features
        on_demand_features = {}
        if on_demand_context:
            on_demand_features = self._compute_on_demand(on_demand_context)
        
        # 4. Merge all features
        all_results = self._merge_results(
            entities, cached_results, redis_results, on_demand_features
        )
        
        # 5. Handle missing features (imputation)
        final_results = self._impute_missing(all_results)
        
        # 6. Record metrics
        latency = (time.perf_counter() - start) * 1000
        self.metrics.record("feature_serving_latency_ms", latency)
        self.metrics.record("cache_hit_rate", len(cached_results) / len(entities))
        
        return final_results
    
    async def _fetch_from_redis(self, entities: list[FeatureRequest]) -> list[dict]:
        """Batch fetch from Redis using pipeline."""
        
        pipe = self.redis.pipeline(transaction=False)
        
        for entity in entities:
            key = f"features:{entity.entity_type}:{entity.entity_id}"
            pipe.hmget(key, *entity.feature_names)
        
        # Single round trip for all entities
        results = await pipe.execute()
        
        # Convert to dicts
        feature_dicts = []
        for entity, values in zip(entities, results):
            feature_dict = {}
            for name, value in zip(entity.feature_names, values):
                if value is not None:
                    feature_dict[name] = self._deserialize(value)
                else:
                    feature_dict[name] = None  # Missing feature
            feature_dicts.append(feature_dict)
        
        return feature_dicts
    
    def _compute_on_demand(self, context: dict) -> dict:
        """Compute request-time features."""
        features = {}
        
        now = time.time()
        features["hour_of_day"] = time.gmtime(now).tm_hour
        features["day_of_week"] = time.gmtime(now).tm_wday
        features["is_weekend"] = features["day_of_week"] >= 5
        
        if "latitude" in context and "item_latitude" in context:
            features["distance_km"] = self._haversine(
                context["latitude"], context["longitude"],
                context["item_latitude"], context["item_longitude"],
            )
        
        return features
    
    def _impute_missing(self, results: list[FeatureResponse]) -> list[FeatureResponse]:
        """Handle missing feature values with defaults."""
        
        # Strategy: use pre-defined defaults per feature
        defaults = {
            "purchase_count_30d": 0,
            "total_spend_30d": 0.0,
            "days_since_last_purchase": 365,  # Conservative: assume inactive
            "avg_session_duration_30d": 0.0,
        }
        
        for result in results:
            for feature_name, value in result.features.items():
                if value is None:
                    result.features[feature_name] = defaults.get(feature_name, 0)
                    self.metrics.increment("feature_imputation_count", tags={"feature": feature_name})
        
        return results
```

### High Availability Patterns

```yaml
High_Availability:
  challenge: "If feature store is down, no predictions possible"
  
  strategies:
    redis_cluster:
      what: "Multi-node Redis with replication and automatic failover"
      configuration:
        nodes: "6 minimum (3 masters + 3 replicas)"
        replication: "Each master has 1+ replica"
        failover: "Automatic (replica promotes in <5s)"
      availability: "99.99% (single node failure handled automatically)"
      
    multi_az_deployment:
      what: "Deploy feature store replicas across availability zones"
      benefit: "Survive entire AZ failure"
      trade_off: "Higher cross-AZ latency (~1-2ms additional)"
      
    read_replicas:
      what: "Multiple read replicas for scaling read throughput"
      when: "High read volume (>100K requests/sec)"
      configuration: "Route reads to replicas, writes to primary"
      
    fallback_strategies:
      local_cache:
        what: "In-memory cache with longer TTL as fallback"
        when: "Redis unavailable"
        trade_off: "Staler features (better than no features)"
        
      default_features:
        what: "Use population-average features when entity features unavailable"
        when: "New entity with no computed features"
        implementation: "Pre-compute median/mean per feature, serve as defaults"
        
      degraded_mode:
        what: "Serve predictions with reduced feature set"
        when: "Some feature sources unavailable"
        approach: "Model trained to handle missing features (via feature dropout during training)"
        
    circuit_breaker:
      what: "Stop calling failing dependency, use fallback"
      implementation: "If Redis errors > 50% in 10s window → switch to local cache"
      recovery: "After 30s, try Redis again (half-open state)"
```

### Batch Serving (Offline Predictions)

```yaml
Batch_Serving:
  what: "Generate predictions for many entities at once (not real-time)"
  when:
    - "Daily recommendation emails (compute for all users)"
    - "Batch scoring (score all accounts for risk)"
    - "Pre-compute predictions (serve from cache)"
    
  architecture:
    approach: "Read features from offline store, score in batch"
    technology: "Spark + model loading (or batch inference API)"
    
  optimization:
    point_in_time_join:
      what: "Join entity timestamps with feature table"
      key_operation: "For each (entity, timestamp), get features valid at that time"
      
    partition_parallel:
      what: "Partition entities, process in parallel"
      implementation: "Spark partitions → each partition fetches features + runs model"
      
    pre_fetch:
      what: "Fetch all needed features in bulk, then score"
      benefit: "Amortize feature retrieval overhead across many predictions"
      
  vs_online_serving:
    latency: "Minutes/hours (batch) vs milliseconds (online)"
    throughput: "Millions of predictions/run (batch) vs thousands/sec (online)"
    freshness: "Stale by hours/day (batch) vs real-time (online)"
    cost: "Cheaper per prediction (batch) vs more expensive per prediction (online)"
```

---

## How It Works in Practice

### Production Serving Configuration

```yaml
Production_Config:
  redis_cluster:
    nodes: 6
    memory_per_node: "64GB"
    total_capacity: "~200 million entity-feature-sets"
    replication_factor: 1
    eviction_policy: "volatile-lru (evict stale features first)"
    max_memory_policy: "allkeys-lru"
    
  serving_service:
    replicas: 5
    resources:
      cpu: "4 cores"
      memory: "8GB (for local cache)"
    local_cache:
      max_entries: 50000
      ttl: "30 seconds"
    timeouts:
      redis_timeout: "5ms"
      total_timeout: "15ms"
    circuit_breaker:
      failure_threshold: "50% in 10s"
      recovery_timeout: "30s"
      
  monitoring:
    metrics:
      - "p50/p95/p99 feature retrieval latency"
      - "Cache hit rate"
      - "Feature missing rate (per feature)"
      - "Redis connection errors"
      - "Imputation count (per feature)"
    alerts:
      - "p99 latency > 10ms"
      - "Cache hit rate < 20%"
      - "Missing rate > 5% for any feature"
      - "Redis unavailability > 10s"
```

---

## Interview Tip

> When asked about feature serving: "Feature serving must deliver 50+ features in under 10ms to meet prediction latency SLAs. My architecture: (1) Online store — Redis Cluster (6+ nodes, multi-AZ) storing latest feature values keyed by entity_id. Single pipeline round trip fetches all features for an entity. (2) Local cache — in-process LRU cache (30s TTL) in the serving service. Typical hit rate 30-50% for popular entities, effectively zero latency on hits. (3) On-demand computation — request-context features (time of day, distance calculations) computed during serving in <2ms. (4) High availability — circuit breaker pattern: if Redis errors > 50%, fall back to local cache (staler but available). Model trained with feature dropout handles missing features gracefully. (5) Batch serving — for offline use cases (daily emails, pre-computation), read from offline store via Spark, much cheaper than online. Critical design principle: the feature serving layer must produce IDENTICAL features to what was used in training. Same transformations, same imputation logic, same encoding. Any discrepancy is training-serving skew and degrades model performance."

---

## Common Mistakes

1. **No fallback when online store is down** — Redis goes down for maintenance → all predictions fail → user experience degrades catastrophically. Solution: multi-layer fallback: local cache (30s stale) → population defaults → degraded model (fewer features). Model should be trained with feature dropout to handle missing features gracefully.

2. **Serving features one-at-a-time** — Making separate Redis calls for each feature (50 features = 50 round trips = 150ms). Solution: batch all features into one pipeline request. HMGET retrieves all features for an entity in one command. Redis PIPELINE batches multiple entities into one network round trip.

3. **Transformation mismatch** — Training normalizes features with z-score using training-set statistics. Serving uses different statistics (or doesn't normalize at all). Model gets unnormalized inputs → garbage predictions. Solution: serialize transformation parameters (mean, std, encoding maps) as artifacts alongside the model. Apply identical transformations at serving time.

4. **No monitoring of feature freshness at serving time** — Features materialized to Redis 3 days ago (pipeline failed silently). Model serves predictions on stale features without anyone knowing. Solution: store `last_updated_timestamp` alongside features in Redis. Feature serving layer checks freshness — alert if features exceed staleness threshold.

5. **Over-sizing online store** — Storing ALL historical feature values in Redis (expensive). Online store grows to 500GB, costs $thousands/month. Solution: online store holds ONLY latest values per entity. TTL-based eviction for inactive entities. Historical values stay in cheap offline storage.

---

## Key Takeaways

- Feature serving budget: 5-10ms to retrieve all features (within 50ms total prediction latency)
- Redis Cluster: primary online store, multi-AZ, automatic failover, pipeline for batch reads
- Local cache: in-process LRU (30s TTL), 30-50% hit rate, zero-latency for popular entities
- On-demand features: computed at request time for context-dependent calculations (<2ms)
- Fallback chain: Redis → local cache → population defaults → degraded model
- Feature imputation: pre-defined defaults for missing features, model trained to handle gaps
- Transformation consistency: identical normalization/encoding between training and serving
- Batch serving: offline store + Spark for high-volume, non-real-time predictions
- Monitor: latency (p99 < 10ms), cache hit rate, missing rate per feature, freshness
- Cost: online store holds ONLY latest values, TTL-evict inactive entities
