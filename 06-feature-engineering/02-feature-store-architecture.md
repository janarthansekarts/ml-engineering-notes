# Feature Store Architecture

## The Problem / Why This Matters

Without a feature store, every ML project reinvents feature computation, storage, and serving. Team A calculates "user_purchase_count_30d" in their training notebook with SQL. Team B needs the same feature but writes different SQL (slightly different logic, different time zones, different null handling). Both teams serve features differently — one reads from a cache, the other recomputes on every request. Results: inconsistency (training vs serving skew), duplication (same feature computed 5 ways), latency issues (expensive features recomputed on every prediction), and no governance (nobody knows what features exist, who owns them, or if they're still valid). A feature store solves this with: a single definition per feature (one source of truth), offline storage for training (historical feature values), online storage for serving (low-latency current values), and a catalog for discovery (what features exist and what they mean). The architectural challenge is bridging the offline world (batch computation on data warehouses, historical point-in-time joins) with the online world (sub-millisecond serving from Redis/DynamoDB for real-time predictions).

---

## The Analogy

Think of a feature store like a restaurant supply chain:

- **Without a feature store** = Every chef (ML engineer) goes to the market (data warehouse) individually, picks ingredients (raw data), prepares them (feature engineering) from scratch for every dish (model). Inconsistent quality, wasted time, no sharing.
- **With a feature store** = Central kitchen (feature store) receives bulk ingredients (raw data), prepares and portions them (feature computation), stores them properly (offline: freezer for historical, online: ready-to-serve station for current), and any chef can grab pre-prepared ingredients instantly. Consistent quality, shared work, fast service.
- **Offline store** = The freezer. Contains historical ingredients (feature values) for recipe development (model training). You can go back in time: "What did this ingredient look like last Tuesday?"
- **Online store** = The hot holding station. Current, fresh, immediately available ingredients (feature values) for serving customers (making predictions). Must be fast (milliseconds).

---

## Deep Dive

### Feature Store Architecture

```yaml
Core_Architecture:
  components:
    feature_definitions:
      what: "Code that defines how features are computed"
      format: "Python/SQL definitions with metadata"
      example:
        name: "user_purchase_count_30d"
        description: "Number of purchases in last 30 days"
        entity: "user_id"
        dtype: "int64"
        source: "orders table"
        computation: "COUNT(*) WHERE order_date > NOW() - 30d"
        freshness: "computed every hour"
        owner: "recommendation-team"
        
    offline_store:
      what: "Historical feature values for model training"
      storage: "Data warehouse (BigQuery, Snowflake, Redshift) or data lake (S3/Parquet)"
      use_case: "Training dataset generation with point-in-time correctness"
      query_pattern: "Give me features for these entities at these historical timestamps"
      scale: "Billions of rows, terabytes of data"
      latency: "Seconds to minutes (batch queries, acceptable for training)"
      
    online_store:
      what: "Latest feature values for real-time prediction"
      storage: "Low-latency store (Redis, DynamoDB, Bigtable)"
      use_case: "Feature serving at prediction time"
      query_pattern: "Give me current features for this entity RIGHT NOW"
      scale: "Millions of entities, each with latest feature values"
      latency: "Single-digit milliseconds (sub-10ms)"
      
    feature_computation:
      what: "Pipelines that compute and materialize features"
      types:
        batch: "Scheduled jobs (hourly/daily) computing features over historical data"
        streaming: "Real-time computation from event streams (Kafka/Kinesis)"
        on_demand: "Computed at request time (no pre-computation)"
        
    feature_registry:
      what: "Metadata catalog of all features"
      contains:
        - "Feature name, description, owner"
        - "Computation logic (transformations applied)"
        - "Data source and lineage"
        - "Statistics (distribution, null rate, freshness)"
        - "Consumers (which models use this feature)"
        - "Quality metrics (drift, staleness)"
```

### Feast Architecture

```yaml
Feast:
  what: "Open-source feature store (most popular, created by Tecton founders)"
  
  components:
    feature_repository:
      what: "Git repo containing feature definitions"
      files:
        - "Entity definitions (entity.py)"
        - "Feature view definitions (features.py)"
        - "Data source definitions (sources.py)"
        
    offline_store:
      backends:
        - "BigQuery (GCP)"
        - "Snowflake"
        - "Redshift (AWS)"
        - "File-based (Parquet for development)"
      operation: "Point-in-time joins for training data generation"
      
    online_store:
      backends:
        - "Redis (most common)"
        - "DynamoDB"
        - "Bigtable"
        - "SQLite (development)"
      operation: "Key-value lookup (entity_id → feature values)"
      
    materialization:
      what: "Process that moves features from offline → online store"
      trigger: "Scheduled (cron) or manual"
      process: "Read from offline, write to online store"
      
  example_definitions:
    code: |
      from feast import Entity, Feature, FeatureView, ValueType
      from feast import BigQuerySource
      from datetime import timedelta
      
      # Define entity
      user = Entity(
          name="user_id",
          value_type=ValueType.INT64,
          description="Unique user identifier",
      )
      
      # Define data source
      user_activity_source = BigQuerySource(
          table="project.dataset.user_activity_features",
          timestamp_field="event_timestamp",
      )
      
      # Define feature view
      user_activity_fv = FeatureView(
          name="user_activity",
          entities=[user],
          ttl=timedelta(days=1),  # Features older than 1 day are stale
          features=[
              Feature(name="purchase_count_30d", dtype=ValueType.INT64),
              Feature(name="total_spend_30d", dtype=ValueType.FLOAT),
              Feature(name="avg_session_duration_7d", dtype=ValueType.FLOAT),
              Feature(name="last_login_days_ago", dtype=ValueType.INT64),
              Feature(name="favorite_category", dtype=ValueType.STRING),
          ],
          source=user_activity_source,
      )
      
  workflow:
    training:
      code: |
        # Get training data with point-in-time correct features
        from feast import FeatureStore
        
        store = FeatureStore(repo_path="feature_repo/")
        
        # Entity DataFrame: entities + timestamps (when prediction would happen)
        entity_df = pd.DataFrame({
            "user_id": [1001, 1002, 1003, ...],
            "event_timestamp": ["2024-01-15", "2024-01-16", ...],
        })
        
        # Get historical features (point-in-time correct)
        training_df = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "user_activity:purchase_count_30d",
                "user_activity:total_spend_30d",
                "user_activity:last_login_days_ago",
            ],
        ).to_df()
    
    serving:
      code: |
        # Get features for real-time prediction
        feature_vector = store.get_online_features(
            features=[
                "user_activity:purchase_count_30d",
                "user_activity:total_spend_30d",
                "user_activity:last_login_days_ago",
            ],
            entity_rows=[{"user_id": 1001}],
        ).to_dict()
        
        # Feed to model
        prediction = model.predict(feature_vector)
```

### Tecton Architecture

```yaml
Tecton:
  what: "Enterprise managed feature platform (commercial, by Feast creators)"
  
  differentiators_vs_feast:
    real_time_features:
      what: "Stream processing built-in (not just batch + online)"
      how: "Define features that compute in real-time from Kafka/Kinesis events"
      example: "user_transactions_last_5_min — computed live from payment stream"
      
    managed_infrastructure:
      what: "No need to manage Redis, compute clusters, or pipelines yourself"
      benefit: "Focus on feature logic, not infrastructure"
      
    feature_pipelines:
      what: "Built-in orchestration for feature computation"
      features:
        - "Automatic backfilling (compute features for historical data)"
        - "Incremental computation (only process new data)"
        - "Data quality checks (built-in validation)"
        
    monitoring:
      what: "Built-in feature quality monitoring"
      tracks: "Drift, staleness, null rates, distribution changes"
      
  feature_types:
    batch:
      what: "Computed on schedule (hourly/daily)"
      source: "Data warehouse (BigQuery/Snowflake)"
      example: "user_lifetime_value — computed daily from all-time order history"
      
    streaming:
      what: "Computed in real-time from event streams"
      source: "Kafka, Kinesis"
      example: "user_clicks_last_5_min — computed from clickstream events"
      latency: "Seconds (from event to feature update)"
      
    real_time:
      what: "Computed at request time (on-demand)"
      source: "Request data + context"
      example: "time_since_page_load — computed when prediction is requested"
      latency: "Milliseconds (computed during request)"
```

### Online/Offline Architecture Deep Dive

```python
# Understanding the online/offline split

"""
The fundamental tension in feature stores:

TRAINING needs:
- Historical data (what were features like 6 months ago?)
- Point-in-time correctness (no future leakage)
- Large volumes (millions of training examples)
- Acceptable latency (minutes/hours for batch training)

SERVING needs:
- Current data (what are features RIGHT NOW?)
- Low latency (sub-10ms)
- High throughput (10,000+ requests/sec)
- High availability (99.99% uptime)

These are fundamentally different access patterns → different storage systems.
"""

# Point-in-time join (the key offline operation)
"""
Problem: For each training example (entity, timestamp), get the feature
values that would have been available at that exact timestamp.

Example:
  Training example: user_1001, prediction_time = 2024-03-15 10:00:00
  Feature: purchase_count_30d
  
  Correct: Count purchases between 2024-02-13 and 2024-03-15 (30 days before prediction)
  WRONG: Count purchases including after 2024-03-15 (temporal leakage!)

Implementation: For each (entity, timestamp) pair, find the most recent
feature value computed BEFORE that timestamp.
"""

# SQL implementation of point-in-time join
POINT_IN_TIME_JOIN_SQL = """
SELECT 
    e.entity_id,
    e.event_timestamp,
    f.purchase_count_30d,
    f.total_spend_30d
FROM entity_df e
ASOF JOIN feature_table f
    ON e.entity_id = f.entity_id
    AND f.feature_timestamp <= e.event_timestamp
    -- Get the most recent feature value BEFORE the event timestamp
"""

# Online store data model
"""
Online store is a simple key-value store:
  Key: (feature_view_name, entity_id)
  Value: {feature_name: feature_value, ...} + timestamp
  
Example:
  Key: ("user_activity", "user_1001")
  Value: {
      "purchase_count_30d": 7,
      "total_spend_30d": 342.50,
      "last_login_days_ago": 2,
      "computed_at": "2024-03-15T09:00:00Z"
  }
  
Access pattern: GET by entity_id (simple, fast, Redis-friendly)
"""
```

### Feature Store Selection Guide

```yaml
Selection_Guide:
  feast_open_source:
    when:
      - "Tight budget (no managed service fees)"
      - "Primarily batch features (not heavy streaming)"
      - "Team has infrastructure expertise"
      - "Need flexibility (custom backends)"
    limitations:
      - "Streaming features require additional setup"
      - "No built-in monitoring"
      - "Self-managed infrastructure"
      
  tecton:
    when:
      - "Need real-time/streaming features"
      - "Want managed infrastructure"
      - "Enterprise scale (millions of entities)"
      - "Budget allows managed service"
    strengths:
      - "Best streaming feature support"
      - "Built-in monitoring and quality"
      - "Managed backfilling"
      
  databricks_feature_store:
    when:
      - "Already on Databricks"
      - "Unity Catalog for governance"
      - "Spark-based feature computation"
    strengths:
      - "Tight integration with Databricks ML"
      - "Lineage tracking via Unity Catalog"
      
  sagemaker_feature_store:
    when:
      - "All-in on AWS / SageMaker"
      - "Want tight integration with SageMaker training"
    strengths:
      - "Managed by AWS"
      - "Direct integration with SageMaker"
      
  vertex_feature_store:
    when:
      - "All-in on GCP / Vertex AI"
      - "BigQuery as primary data warehouse"
    strengths:
      - "Tight BigQuery integration"
      - "Managed by GCP"
      
  custom_build:
    when:
      - "Very specific requirements (custom storage, unique access patterns)"
      - "Extremely low latency requirements (<1ms)"
      - "Already have strong internal infrastructure"
    components:
      - "Redis/DynamoDB for online serving"
      - "Parquet/BigQuery for offline storage"
      - "Airflow/Dagster for computation orchestration"
      - "Custom API for serving"
```

---

## How It Works in Practice

### Production Feature Store Deployment

```yaml
Production_Deployment:
  architecture:
    compute:
      batch: "Spark on Kubernetes (daily/hourly feature computation)"
      streaming: "Flink on Kubernetes (real-time feature computation)"
      
    storage:
      offline: "BigQuery (historical features, PIT joins)"
      online: "Redis Cluster (current features, sub-5ms reads)"
      
    orchestration:
      scheduler: "Airflow (trigger batch computations)"
      streaming: "Kafka → Flink → Redis (continuous pipeline)"
      
    serving:
      api: "gRPC service for feature retrieval"
      sdk: "Python SDK for training data generation"
      
    monitoring:
      freshness: "Alert if features older than SLA"
      quality: "Alert on null rate spikes, distribution shifts"
      latency: "p99 online serving latency < 10ms"
```

---

## Interview Tip

> When asked about feature stores: "I think of feature stores as solving three core problems: (1) Training-serving skew — single feature definition used for both training (point-in-time correct historical values from the offline store) and serving (current values from the online store). This eliminates the #1 source of model degradation in production. (2) Feature reuse — instead of every team computing 'user_purchase_count_30d' independently with slightly different logic, it's defined once, computed once, served consistently. Feature catalog enables discovery. (3) Operational reliability — features are monitored (drift, staleness, null rates), versioned, and served with SLAs (sub-10ms latency, 99.99% availability). Architecture: offline store (BigQuery/Snowflake) for training with point-in-time joins, online store (Redis/DynamoDB) for serving, materialization pipeline moving batch features from offline to online, and streaming pipeline computing real-time features from Kafka events. I'd choose Feast for open-source flexibility with primarily batch features, Tecton for managed real-time features at scale, or Databricks Feature Store if already on that platform."

---

## Common Mistakes

1. **Skipping point-in-time joins** — Using the LATEST feature values for historical training examples. Training on features that include future information → massive leakage → model fails in production. Solution: always use point-in-time correct historical features (feature value that existed AT the prediction timestamp, not after).

2. **Different logic for training vs serving** — Training computes features with complex SQL on the data warehouse. Serving approximates with simpler code. Features don't match → training-serving skew → model degrades. Solution: single feature definition in feature store, used by both training and serving paths.

3. **No freshness monitoring** — Feature materialization job silently fails on Friday night. Online store serves stale (3-day-old) features all weekend. Model makes predictions on outdated information. Solution: monitor feature freshness (time since last update). Alert if older than SLA (e.g., alert if hourly feature is >2h stale).

4. **Online store as data warehouse** — Storing full historical feature data in Redis (online store). Costs explode, Redis runs out of memory. Solution: online store holds ONLY the latest value per entity. Historical values stay in the offline store (cheap, scalable storage).

5. **Building before proving value** — Spending 6 months building a feature store before having any models in production. Solution: prove feature store value incrementally — start with Feast or a simple Redis cache for your first production model, expand to full platform as you add more models.

---

## Key Takeaways

- Feature store: single source of truth for feature definitions, computation, storage, and serving
- Offline store (BigQuery/Snowflake): historical values, point-in-time joins, training data generation
- Online store (Redis/DynamoDB): current values, sub-10ms serving, high availability
- Point-in-time correctness: feature values at training must match what existed at prediction time
- Eliminates training-serving skew: same definition used for both paths
- Feature computation: batch (Spark/SQL, hourly/daily), streaming (Flink/Kafka, real-time), on-demand
- Materialization: pipeline moving computed features from offline → online store
- Feature catalog: metadata, discovery, documentation, ownership, quality metrics
- Start simple: Feast or custom Redis cache for first model, expand to platform as needed
- Monitor: freshness (staleness alerts), quality (drift detection), coverage (null rates)
