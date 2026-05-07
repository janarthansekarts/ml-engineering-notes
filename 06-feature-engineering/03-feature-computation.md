# Feature Computation

## The Problem / Why This Matters

Features don't appear magically — they must be computed from raw data, and HOW you compute them determines their freshness, cost, and reliability. A feature like "user_transaction_count_last_5_minutes" requires real-time stream processing (Flink/Spark Streaming processing events as they arrive). A feature like "user_lifetime_value" requires batch computation (scanning all historical orders). A feature like "time_since_page_load" is computed on-demand at request time. Each computation paradigm has radically different infrastructure, cost profiles, and failure modes. The ML engineering challenge: choosing the right computation paradigm for each feature, building reliable pipelines that handle late data, failures, and schema changes, and doing it at scale (millions of entities, thousands of features, petabytes of source data). Most feature computation problems come from: (1) computing expensive features too frequently (wasting compute), (2) computing time-sensitive features too infrequently (stale data), or (3) not handling edge cases (late events, duplicates, null values).

---

## The Analogy

Think of feature computation like food preparation at different restaurant types:

- **Batch computation** = Catering service. Prepares large quantities in advance on a schedule. You cook everything Sunday night for the week's events. Efficient but not fresh — Monday's food is great, Friday's is stale.
- **Streaming computation** = Sushi conveyor belt restaurant. Food is prepared continuously as ingredients arrive. Always fresh, but expensive (chef always working) and complex (what if a rare ingredient doesn't arrive on time?).
- **On-demand computation** = Made-to-order diner. Nothing is pre-prepared — customer orders, chef cooks. Freshest possible, but slowest and can't handle high volume (each order blocks).
- **The optimal strategy** = Combine all three. Pre-prepare common sides (batch: features that change slowly), continuously refresh popular items (streaming: fast-changing signals), and make specialized requests to order (on-demand: request-specific calculations).

---

## Deep Dive

### Batch Feature Computation

```yaml
Batch_Computation:
  what: "Scheduled jobs that compute features over large datasets"
  when: "Features that don't need real-time freshness (hourly or daily is fine)"
  
  technology:
    primary: "Apache Spark (PySpark) on Kubernetes or managed (Databricks, EMR, Dataproc)"
    alternatives:
      - "SQL-based (dbt + BigQuery/Snowflake) — simplest for SQL-heavy features"
      - "Pandas/Polars on single machine — for smaller datasets (<100GB)"
      - "Ray — for Python-heavy computation that doesn't fit SQL"
      
  architecture:
    source: "Data warehouse (BigQuery, Snowflake) or data lake (S3/Parquet)"
    compute: "Spark cluster (auto-scaling)"
    destination: "Feature store (offline store + materialize to online store)"
    scheduler: "Airflow, Dagster, or Prefect"
    
  patterns:
    full_recompute:
      what: "Recompute all features for all entities from scratch"
      when: "Simple, logic changes frequently, dataset is small"
      cost: "High for large datasets (scan all data every time)"
      
    incremental:
      what: "Only compute features for entities with new data since last run"
      when: "Large datasets, features based on recent activity"
      implementation:
        - "Track watermark (last processed timestamp)"
        - "Only scan new events since watermark"
        - "Update affected entity features only"
      cost: "Much lower (only process delta)"
      complexity: "Higher (must handle late-arriving data)"
      
    window_based:
      what: "Compute aggregations over sliding time windows"
      implementation: "For each entity, aggregate events in window (last 7d, 30d, 90d)"
      challenge: "Efficient window computation at scale"
```

```python
# Batch feature computation with PySpark

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime, timedelta


def compute_user_features(spark: SparkSession, computation_date: str):
    """Compute user activity features for a given date."""
    
    # Read raw events
    events = spark.read.parquet("s3://data-lake/events/")
    orders = spark.read.parquet("s3://data-lake/orders/")
    
    comp_date = datetime.strptime(computation_date, "%Y-%m-%d")
    
    # Define time windows
    windows = {
        "7d": comp_date - timedelta(days=7),
        "30d": comp_date - timedelta(days=30),
        "90d": comp_date - timedelta(days=90),
    }
    
    # Compute purchase features per window
    purchase_features = []
    for window_name, start_date in windows.items():
        window_orders = orders.filter(
            (F.col("order_date") >= start_date) &
            (F.col("order_date") < comp_date)  # Strict: before computation date
        )
        
        features = window_orders.groupBy("user_id").agg(
            F.count("*").alias(f"purchase_count_{window_name}"),
            F.sum("order_total").alias(f"total_spend_{window_name}"),
            F.avg("order_total").alias(f"avg_order_value_{window_name}"),
            F.max("order_total").alias(f"max_order_value_{window_name}"),
            F.countDistinct("product_category").alias(f"unique_categories_{window_name}"),
        )
        purchase_features.append(features)
    
    # Join all window features
    result = purchase_features[0]
    for df in purchase_features[1:]:
        result = result.join(df, on="user_id", how="outer")
    
    # Compute recency features
    recency = orders.filter(F.col("order_date") < comp_date).groupBy("user_id").agg(
        F.datediff(F.lit(comp_date), F.max("order_date")).alias("days_since_last_purchase"),
        F.datediff(F.lit(comp_date), F.min("order_date")).alias("days_since_first_purchase"),
    )
    
    result = result.join(recency, on="user_id", how="outer")
    
    # Compute engagement features from events
    engagement = events.filter(
        (F.col("event_timestamp") >= windows["30d"]) &
        (F.col("event_timestamp") < comp_date)
    ).groupBy("user_id").agg(
        F.count("*").alias("total_events_30d"),
        F.countDistinct(F.date_format("event_timestamp", "yyyy-MM-dd")).alias("active_days_30d"),
        F.avg("session_duration_sec").alias("avg_session_duration_30d"),
    )
    
    result = result.join(engagement, on="user_id", how="outer")
    
    # Add computation metadata
    result = result.withColumn("feature_timestamp", F.lit(comp_date))
    result = result.withColumn("computation_id", F.lit(f"batch_{computation_date}"))
    
    # Write to feature store offline storage
    result.write.mode("overwrite").partitionBy("feature_timestamp").parquet(
        f"s3://feature-store/offline/user_activity/date={computation_date}/"
    )
    
    # Materialize to online store (latest values)
    materialize_to_redis(result)
    
    return result


def materialize_to_redis(features_df):
    """Write latest feature values to Redis online store."""
    # Convert to key-value format and write to Redis
    # Key: "user_activity:{user_id}"
    # Value: JSON of all feature values + timestamp
    pass
```

### Streaming Feature Computation

```yaml
Streaming_Computation:
  what: "Continuous real-time feature computation from event streams"
  when: "Features that need to reflect events within seconds/minutes"
  
  technology:
    primary: "Apache Flink (most capable stream processor for ML features)"
    alternatives:
      - "Spark Structured Streaming — if already on Spark (simpler, less flexible)"
      - "Apache Kafka Streams — lightweight, no separate cluster needed"
      - "Bytewax — Python-native stream processing (easier for ML engineers)"
      - "Materialize — SQL over streaming data (incremental materialized views)"
      
  architecture:
    source: "Kafka/Kinesis (event stream)"
    compute: "Flink/Spark Streaming cluster"
    destination: "Online store (Redis/DynamoDB) — updated in real-time"
    
  common_operations:
    windowed_aggregations:
      what: "Aggregate events over sliding or tumbling time windows"
      examples:
        - "Count of transactions in last 5 minutes"
        - "Sum of purchases in last 1 hour"
        - "Average response time in last 10 minutes"
      window_types:
        tumbling: "Fixed non-overlapping windows (every 5 min: 0-5, 5-10, 10-15)"
        sliding: "Overlapping windows (every minute, look back 5 min)"
        session: "Dynamic windows based on activity gaps"
        
    event_counting:
      what: "Count specific event types per entity"
      examples:
        - "Failed login attempts in last 10 minutes (fraud detection)"
        - "Page views in current session (engagement)"
        - "API errors in last 5 minutes (reliability)"
        
    state_tracking:
      what: "Track entity state changes in real-time"
      examples:
        - "Current cart total (updated on add/remove events)"
        - "Is user currently active? (last event < 5 min ago)"
        - "Running balance (updated on each transaction)"
```

```python
# Streaming feature computation with Flink (Python/PyFlink)

"""
Example: Real-time fraud detection features
Compute transaction velocity features from payment event stream.
"""

from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from pyflink.table.expressions import col, lit
from pyflink.table.window import Slide


def setup_streaming_features():
    """Configure streaming feature computation pipeline."""
    
    env_settings = EnvironmentSettings.in_streaming_mode()
    t_env = StreamTableEnvironment.create(environment_settings=env_settings)
    
    # Define source: Kafka topic with transaction events
    t_env.execute_sql("""
        CREATE TABLE transactions (
            user_id STRING,
            amount DECIMAL(10, 2),
            merchant_category STRING,
            transaction_time TIMESTAMP(3),
            WATERMARK FOR transaction_time AS transaction_time - INTERVAL '5' SECOND
        ) WITH (
            'connector' = 'kafka',
            'topic' = 'payment-events',
            'properties.bootstrap.servers' = 'kafka:9092',
            'format' = 'json',
            'scan.startup.mode' = 'latest-offset'
        )
    """)
    
    # Compute windowed features
    # Feature: transaction_count_5min, total_amount_5min, avg_amount_5min
    t_env.execute_sql("""
        CREATE VIEW user_txn_features_5min AS
        SELECT
            user_id,
            COUNT(*) as txn_count_5min,
            SUM(amount) as total_amount_5min,
            AVG(amount) as avg_amount_5min,
            MAX(amount) as max_amount_5min,
            COUNT(DISTINCT merchant_category) as unique_merchants_5min,
            window_end as feature_time
        FROM TABLE(
            HOP(TABLE transactions, DESCRIPTOR(transaction_time), 
                INTERVAL '1' MINUTE, INTERVAL '5' MINUTE)
        )
        GROUP BY user_id, window_start, window_end
    """)
    
    # Write to Redis (online store) via Redis sink
    t_env.execute_sql("""
        CREATE TABLE redis_sink (
            user_id STRING,
            txn_count_5min BIGINT,
            total_amount_5min DECIMAL(10, 2),
            avg_amount_5min DECIMAL(10, 2),
            max_amount_5min DECIMAL(10, 2),
            unique_merchants_5min BIGINT,
            feature_time TIMESTAMP(3),
            PRIMARY KEY (user_id) NOT ENFORCED
        ) WITH (
            'connector' = 'redis',
            'host' = 'redis-cluster:6379',
            'key-prefix' = 'fraud_features:'
        )
    """)
    
    t_env.execute_sql("""
        INSERT INTO redis_sink
        SELECT * FROM user_txn_features_5min
    """)
```

### On-Demand Feature Computation

```yaml
On_Demand_Computation:
  what: "Features computed at the moment of prediction request"
  when: "Feature depends on request context (not pre-computable)"
  
  examples:
    request_context:
      - "Time of day of the request (hour, is_weekend)"
      - "User's current device type"
      - "Geographic distance between user and item"
      - "Time since last activity (requires current time)"
      
    cross_features:
      - "Ratio of item price to user's average purchase (combines stored features)"
      - "Similarity between current search query and user's history"
      
    expensive_but_necessary:
      - "Real-time embedding similarity computation"
      - "LLM-generated features (summarize recent conversation)"
      
  implementation:
    approach: "Compute during the feature retrieval call"
    latency_budget: "5-20ms (part of the overall serving latency)"
    
    code: |
      # On-demand feature computation at serving time
      
      def compute_on_demand_features(request_context: dict, stored_features: dict) -> dict:
          """Compute features that depend on request context."""
          
          on_demand = {}
          
          # Time-based features
          now = datetime.utcnow()
          on_demand["hour_of_day"] = now.hour
          on_demand["is_weekend"] = now.weekday() >= 5
          on_demand["is_business_hours"] = 9 <= now.hour <= 17
          
          # Recency (needs current time)
          last_login = stored_features.get("last_login_timestamp")
          if last_login:
              on_demand["minutes_since_last_login"] = (now - last_login).total_seconds() / 60
          
          # Cross-features (combine stored + context)
          item_price = request_context.get("item_price", 0)
          avg_purchase = stored_features.get("avg_order_value_30d", 0)
          if avg_purchase > 0:
              on_demand["price_to_avg_ratio"] = item_price / avg_purchase
          
          # Geographic
          user_lat = request_context.get("user_latitude")
          item_lat = stored_features.get("item_latitude")
          if user_lat and item_lat:
              on_demand["distance_km"] = haversine(user_lat, user_lon, item_lat, item_lon)
          
          return on_demand
      
  trade_offs:
    pros:
      - "Always fresh (computed at request time)"
      - "No storage needed"
      - "Can use request context (impossible to pre-compute)"
    cons:
      - "Adds latency to serving"
      - "Must be fast (can't do heavy computation)"
      - "Compute cost scales with request volume"
```

### Hybrid Computation Strategy

```yaml
Hybrid_Strategy:
  principle: "Match computation paradigm to feature freshness requirements"
  
  decision_matrix:
    batch:
      freshness: "Hourly to daily"
      examples:
        - "User lifetime value (changes slowly)"
        - "User segment (updated daily)"
        - "Product popularity score (daily)"
        - "Historical aggregations (30d, 90d windows)"
      compute_schedule: "Every 1-24 hours"
      
    streaming:
      freshness: "Seconds to minutes"
      examples:
        - "Transaction count last 5 min (fraud detection)"
        - "Session page views (engagement)"
        - "Error rate last 10 min (anomaly detection)"
        - "Cart total (real-time personalization)"
      compute_schedule: "Continuous (always processing)"
      
    on_demand:
      freshness: "Instant (request time)"
      examples:
        - "Time features (hour, day_of_week)"
        - "Distance calculations (user to item)"
        - "Ratio features (combine batch + context)"
        - "Request-specific context"
      compute_schedule: "Per request"
      
  typical_production_model:
    fraud_detection:
      batch_features: "user_avg_transaction_amount, user_account_age_days, user_typical_merchants"
      streaming_features: "txn_count_5min, total_amount_1h, unique_merchants_10min"
      on_demand_features: "time_since_last_txn, distance_from_home, amount_to_avg_ratio"
      
    recommendation:
      batch_features: "user_preferences, item_popularity, user_segment"
      streaming_features: "recent_views_1h, cart_contents, session_interests"
      on_demand_features: "time_of_day_affinity, device_type, geo_relevance"
```

---

## How It Works in Practice

### Production Pipeline Architecture

```yaml
Pipeline_Architecture:
  batch_pipeline:
    scheduler: "Airflow DAG running every hour"
    flow:
      1: "Read new events from data lake (incremental)"
      2: "Compute features with Spark"
      3: "Write to offline store (append)"
      4: "Materialize latest to online store (upsert)"
      5: "Run quality checks (null rate, distribution)"
    monitoring: "Airflow alerts on failure, quality alerts on drift"
    
  streaming_pipeline:
    flow:
      1: "Kafka consumer reads events"
      2: "Flink processes with windowed aggregations"
      3: "Writes to Redis (online store) continuously"
      4: "Also writes to offline store (for training consistency)"
    monitoring: "Consumer lag, processing latency, error rate"
    recovery: "Checkpoint to S3, resume from checkpoint on failure"
    
  serving:
    flow:
      1: "Prediction request arrives"
      2: "Retrieve batch + streaming features from online store (Redis)"
      3: "Compute on-demand features from request context"
      4: "Combine all features into feature vector"
      5: "Send to model for prediction"
    latency_budget: "Total < 50ms (feature retrieval < 10ms)"
```

---

## Interview Tip

> When asked about feature computation: "I use three computation paradigms matched to feature freshness requirements: (1) Batch (Spark/SQL) — for slowly-changing features computed hourly or daily. User lifetime value, 30-day aggregations, segment membership. Scheduled via Airflow, stored in offline store, materialized to online store. I use incremental computation (only process new events since last watermark) to keep costs manageable. (2) Streaming (Flink) — for features needing second/minute-level freshness. Transaction count in last 5 minutes (fraud), session activity (engagement), error rates (monitoring). Continuous processing from Kafka, writes directly to Redis. Key challenge is handling late-arriving events (watermarks + allowed lateness). (3) On-demand — for features that depend on request context (can't pre-compute). Time features, distance calculations, ratio features combining stored values with request data. Computed during serving within 5ms budget. A typical production model uses all three: batch for stable signals, streaming for real-time signals, on-demand for contextual signals. The critical engineering challenge is ensuring all three paths produce consistent results for training vs serving (same feature logic, same semantics, different computation schedules)."

---

## Common Mistakes

1. **Computing everything in batch** — Features like "transactions in last 5 minutes" computed hourly. By the time the feature updates, the fraud has already happened. Solution: match computation frequency to feature freshness requirement. If the feature needs minute-level freshness, use streaming computation.

2. **Full recompute every time** — Daily Spark job scans ALL historical data to compute 30-day features. As data grows, job takes 6 hours and costs $500/day. Solution: incremental computation — track the watermark (last processed event), only process new events, update affected entities. Reduces compute by 90%+.

3. **Ignoring late-arriving data** — Streaming pipeline processes events as they arrive. Some events arrive 5 minutes late (network delays, client batching). Late events are dropped → features are inaccurate. Solution: configure watermarks and allowed lateness in Flink/Spark Streaming. Process late events when they arrive, update affected windows.

4. **Not handling duplicates** — Same event processed twice (Kafka rebalance, producer retry) → feature values double-counted. "Transaction count: 4" is actually 2 real transactions counted twice. Solution: deduplication at the stream processing layer (event IDs + exactly-once semantics) or idempotent aggregations (MAX is naturally idempotent, COUNT is not).

5. **Streaming without backfill** — Streaming pipeline starts computing features from "now." But training needs historical feature values. No historical data in online store. Solution: every streaming feature should also have a batch backfill job that computes historical values from the data lake, ensuring training can access point-in-time correct features.

---

## Key Takeaways

- Three paradigms: batch (hourly/daily, Spark), streaming (seconds/minutes, Flink), on-demand (request time)
- Match computation frequency to feature freshness requirement
- Batch: use incremental computation (delta processing) to control cost at scale
- Streaming: handle late events (watermarks), duplicates (dedup), and failures (checkpoints)
- On-demand: keep under 5ms latency, only for request-context-dependent features
- Hybrid is the production reality: most models use all three paradigms together
- Consistency: ensure batch and streaming compute same semantics (same results for same data)
- Backfill: streaming features need historical backfill for training data generation
- Monitor: pipeline health (lag, failures), data quality (nulls, distributions), freshness (staleness)
- Cost management: incremental > full recompute, right-size Spark clusters, auto-scale streaming
