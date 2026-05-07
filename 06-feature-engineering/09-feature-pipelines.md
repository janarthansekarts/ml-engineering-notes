# Feature Pipelines

## The Problem / Why This Matters

Features don't compute themselves — they require orchestrated pipelines that reliably transform raw data into model-ready features on schedule, handle failures gracefully, maintain point-in-time correctness, and scale with data growth. A feature pipeline is the production system that: (1) reads from data sources (databases, event streams, APIs), (2) applies transformations (aggregations, joins, encodings), (3) validates quality (null checks, range checks, schema validation), (4) writes to feature stores (offline for training, online for serving), and (5) handles operational concerns (retries, backfilling, monitoring, alerting). The engineering challenge: feature pipelines must be correct (point-in-time, no leakage), reliable (automatic retry, alert on failure), efficient (don't reprocess everything on every run), maintainable (easy to debug, modify, test), and observable (you know when something's wrong before it affects models). Most ML system outages trace back to feature pipeline failures — a broken pipeline means stale features means degraded predictions means unhappy users.

---

## The Analogy

Think of feature pipelines like a factory assembly line:

- **Raw materials** (data sources) arrive at the loading dock → **Processing stations** (transformations) shape them → **Quality control** (validation) inspects each piece → **Shipping** (feature store writes) delivers finished products → **Monitoring** (observability) ensures the line is running smoothly.

Without the assembly line:
- **No schedule** = Workers show up randomly, some days nothing gets produced
- **No quality control** = Defective products (bad features) ship to customers (models)
- **No error handling** = One broken machine (failed transform) stops the entire factory indefinitely
- **No monitoring** = Factory is on fire but nobody notices for 3 days

A well-built pipeline is invisible when working correctly and loud when something's wrong.

---

## Deep Dive

### Pipeline Architecture

```yaml
Pipeline_Architecture:
  components:
    scheduler:
      what: "Triggers pipeline execution on schedule or events"
      tools: "Airflow, Dagster, Prefect, Temporal"
      patterns:
        time_based: "Run every hour/day at specific time"
        event_driven: "Run when new data arrives (S3 event, Kafka message)"
        dependency_based: "Run after upstream pipeline completes"
        
    data_reader:
      what: "Reads from source systems with exactly-once semantics"
      sources:
        - "Data warehouse (BigQuery, Snowflake, Redshift)"
        - "Data lake (S3 Parquet, Delta Lake)"
        - "Databases (PostgreSQL, MySQL via CDC)"
        - "Event streams (Kafka, Kinesis)"
        - "APIs (external data providers)"
      patterns:
        full_read: "Read all data (simple but expensive)"
        incremental: "Read only data since last watermark"
        change_data_capture: "Read only changed records"
        
    transformer:
      what: "Applies feature computation logic"
      compute_engines:
        - "Spark (large scale batch)"
        - "Flink (streaming)"
        - "dbt (SQL-based transformations)"
        - "Pandas/Polars (small scale)"
        - "Ray (distributed Python)"
        
    validator:
      what: "Checks output quality before writing"
      checks:
        schema: "Correct columns, types, and structure"
        completeness: "Null rate within bounds"
        freshness: "Data is recent enough"
        distribution: "Values within expected ranges"
        volume: "Row count matches expectations"
        
    writer:
      what: "Writes validated features to storage"
      destinations:
        offline_store: "Append to historical feature table"
        online_store: "Upsert latest values to Redis/DynamoDB"
        
    monitor:
      what: "Tracks pipeline health and data quality"
      metrics:
        - "Pipeline duration (trend over time)"
        - "Records processed (expected vs actual)"
        - "Validation pass/fail rates"
        - "Feature freshness after write"
```

### Orchestration with Airflow/Dagster

```python
# Feature pipeline with Dagster (modern orchestration)

"""
Dagster example: daily user feature computation pipeline.
Dagster provides: scheduling, retries, observability, data lineage, testing.
"""

from dagster import (
    asset, op, job, schedule, 
    AssetIn, Output, DailyPartitionsDefinition,
    RetryPolicy, Backoff
)
from datetime import datetime


# Define partitions (one per day)
daily_partitions = DailyPartitionsDefinition(start_date="2024-01-01")


@asset(
    partitions_def=daily_partitions,
    retry_policy=RetryPolicy(max_retries=3, delay=60, backoff=Backoff.EXPONENTIAL),
)
def raw_events(context) -> Output:
    """Read raw events for the partition date."""
    partition_date = context.partition_key
    
    # Read only this day's data (incremental)
    events = spark.read.parquet(
        f"s3://data-lake/events/date={partition_date}/"
    )
    
    context.log.info(f"Read {events.count()} events for {partition_date}")
    
    # Validate: minimum expected volume
    if events.count() < 1000:
        raise ValueError(f"Unexpectedly low event count: {events.count()}")
    
    return Output(events, metadata={"row_count": events.count()})


@asset(
    ins={"events": AssetIn("raw_events")},
    partitions_def=daily_partitions,
    retry_policy=RetryPolicy(max_retries=2, delay=30),
)
def user_activity_features(context, events) -> Output:
    """Compute user activity features from raw events."""
    partition_date = context.partition_key
    
    # Compute windowed aggregations
    features = compute_user_features(events, partition_date)
    
    # Validate output
    null_rate = features.select(
        [F.mean(F.col(c).isNull().cast("int")).alias(c) for c in features.columns]
    ).first()
    
    for col_name, rate in null_rate.asDict().items():
        if rate > 0.05 and col_name != "user_id":
            context.log.warning(f"High null rate for {col_name}: {rate:.2%}")
    
    context.log.info(f"Computed features for {features.count()} users")
    
    return Output(features, metadata={
        "user_count": features.count(),
        "partition_date": partition_date,
    })


@asset(
    ins={"features": AssetIn("user_activity_features")},
    partitions_def=daily_partitions,
)
def offline_store_write(context, features) -> Output:
    """Write features to offline store (data warehouse)."""
    partition_date = context.partition_key
    
    features.write.mode("overwrite").partitionBy("feature_date").parquet(
        f"s3://feature-store/offline/user_activity/date={partition_date}/"
    )
    
    context.log.info(f"Written to offline store: {partition_date}")
    return Output(None, metadata={"destination": "offline_store"})


@asset(
    ins={"features": AssetIn("user_activity_features")},
    partitions_def=daily_partitions,
)
def online_store_materialize(context, features) -> Output:
    """Materialize latest features to online store (Redis)."""
    
    # Convert to dict and write to Redis
    feature_rows = features.collect()
    
    pipe = redis_client.pipeline(transaction=False)
    for row in feature_rows:
        key = f"user_activity:{row['user_id']}"
        value = serialize_features(row)
        pipe.hset(key, mapping=value)
        pipe.expire(key, 86400 * 7)  # TTL: 7 days
    
    pipe.execute()
    
    context.log.info(f"Materialized {len(feature_rows)} entities to online store")
    return Output(None, metadata={"entities_written": len(feature_rows)})


# Schedule: run daily at 3:00 AM UTC
@schedule(cron_schedule="0 3 * * *", job=user_feature_pipeline_job)
def daily_feature_schedule(context):
    """Trigger daily feature computation."""
    return {}
```

### Point-in-Time Correctness

```yaml
Point_in_Time:
  what: "Features must reflect the state of the world AT the prediction timestamp"
  why: "Training on features that include future information = leakage = production failure"
  
  implementation:
    offline_training:
      approach: "For each training example (entity, timestamp), join features that existed BEFORE that timestamp"
      sql: |
        SELECT 
            t.entity_id,
            t.event_timestamp AS prediction_time,
            f.*
        FROM training_labels t
        ASOF JOIN features f
            ON t.entity_id = f.entity_id
            AND f.feature_timestamp <= t.event_timestamp
        -- Get the most recent feature computed BEFORE the prediction would happen
      
    production_serving:
      approach: "Features in online store are always 'current' — they were computed from past data"
      verification: "Feature timestamp < prediction time (always true by construction)"
      
  common_violations:
    using_full_history: "Aggregating ALL data (including future) for historical training examples"
    wrong_join: "Inner join instead of ASOF join (gets closest-in-time, may be future)"
    stale_label_timing: "Using label timestamp instead of prediction timestamp for join"
    
  validation:
    test: "Train model with temporal split (train on past, test on future)"
    signal: "If temporal-split performance matches random-split → no leakage"
    red_flag: "If random split >> temporal split → likely leakage"
```

### Backfilling

```yaml
Backfilling:
  what: "Computing features for historical dates (not just today)"
  when_needed:
    - "New feature created → need historical values for training"
    - "Bug fix in feature logic → need to recompute past values"
    - "New entity type added → backfill from available history"
    - "Feature store migration → populate new storage from scratch"
    
  challenges:
    computation_cost: "Reprocessing months of data for millions of entities"
    point_in_time: "Must compute features AS IF running on that historical date"
    data_availability: "Historical source data might be in different format/location"
    idempotency: "Running backfill multiple times should produce same result"
    
  implementation_patterns:
    date_range_backfill:
      what: "Run pipeline for each historical date in sequence"
      approach: "Loop over dates, compute features as if that date were 'today'"
      optimization: "Parallelize across dates (each date is independent)"
      
    partition_based:
      what: "Process one partition per pipeline run"
      orchestration: "Dagster/Airflow with date-partitioned assets"
      benefit: "Built-in parallelism, retry per partition, progress tracking"
      
    full_recompute:
      what: "One big Spark job that computes all historical features at once"
      when: "Small-medium data, or one-time migration"
      implementation: "Window functions over full history, output partitioned by date"
      
  best_practices:
    - "Use same pipeline code for backfill and daily runs (avoid divergence)"
    - "Validate backfilled features against known good values (spot check)"
    - "Run backfill with lower priority / separate cluster (don't impact production)"
    - "Maintain idempotency: re-running for same date produces same output"
    - "Track backfill progress: which dates are done, which failed"
```

### Pipeline Testing

```python
# Testing feature pipelines

"""
Feature pipelines need tests at multiple levels:
1. Unit tests: individual transformation functions
2. Integration tests: full pipeline on sample data
3. Data quality tests: validation rules on output
"""

import pytest
from datetime import date


class TestUserActivityFeatures:
    """Unit tests for user activity feature computation."""
    
    def test_purchase_count_basic(self, spark_session):
        """Test that purchase count is computed correctly."""
        # Arrange: create test data
        orders = spark_session.createDataFrame([
            {"user_id": 1, "order_date": date(2024, 3, 1), "amount": 50.0},
            {"user_id": 1, "order_date": date(2024, 3, 10), "amount": 30.0},
            {"user_id": 1, "order_date": date(2024, 3, 15), "amount": 70.0},
            {"user_id": 2, "order_date": date(2024, 3, 5), "amount": 100.0},
        ])
        
        # Act: compute features for March 20
        result = compute_user_features(orders, computation_date="2024-03-20")
        
        # Assert
        user_1 = result.filter(F.col("user_id") == 1).first()
        assert user_1["purchase_count_30d"] == 3
        assert user_1["total_spend_30d"] == 150.0
        
    def test_no_future_leakage(self, spark_session):
        """Test that features don't include future data."""
        orders = spark_session.createDataFrame([
            {"user_id": 1, "order_date": date(2024, 3, 1), "amount": 50.0},   # Before
            {"user_id": 1, "order_date": date(2024, 3, 20), "amount": 100.0},  # ON the date
            {"user_id": 1, "order_date": date(2024, 3, 25), "amount": 200.0},  # AFTER
        ])
        
        # Computation date: March 20 (should NOT include March 20 or 25)
        result = compute_user_features(orders, computation_date="2024-03-20")
        
        user_1 = result.filter(F.col("user_id") == 1).first()
        # Only the March 1 order should be included (strictly before March 20)
        assert user_1["purchase_count_30d"] == 1
        assert user_1["total_spend_30d"] == 50.0
    
    def test_handles_null_gracefully(self, spark_session):
        """Test that pipeline handles missing data without crashing."""
        orders = spark_session.createDataFrame([
            {"user_id": 1, "order_date": date(2024, 3, 1), "amount": None},
        ])
        
        result = compute_user_features(orders, computation_date="2024-03-20")
        # Should not crash, amount-based features should be null or 0
        assert result.count() > 0


class TestPipelineIntegration:
    """Integration tests: full pipeline on realistic sample data."""
    
    def test_full_pipeline_runs(self, sample_data, temp_feature_store):
        """Test that the full pipeline executes without errors."""
        result = run_feature_pipeline(
            source_data=sample_data,
            computation_date="2024-03-20",
            output_path=temp_feature_store,
        )
        
        assert result.status == "success"
        assert result.entities_processed > 0
        
    def test_idempotency(self, sample_data, temp_feature_store):
        """Test that running pipeline twice produces same results."""
        result_1 = run_feature_pipeline(sample_data, "2024-03-20", temp_feature_store)
        result_2 = run_feature_pipeline(sample_data, "2024-03-20", temp_feature_store)
        
        # Same output both times
        assert result_1.output_hash == result_2.output_hash
```

---

## How It Works in Practice

### Production Pipeline Operations

```yaml
Operations:
  daily_run:
    schedule: "3:00 AM UTC (after upstream ETL completes)"
    duration: "45 minutes typical"
    monitoring: "Airflow/Dagster UI + PagerDuty on failure"
    
  failure_handling:
    automatic_retry: "3 retries with exponential backoff"
    partial_failure: "If one entity partition fails, others continue"
    manual_intervention: "After 3 retries, page on-call engineer"
    root_cause: "Check: source data available? Spark resources? Schema changed?"
    
  backfill_process:
    trigger: "New feature added, bug fix, or data migration"
    execution: "Separate cluster (don't impact daily pipeline)"
    monitoring: "Progress dashboard (which dates done, which pending)"
    validation: "Spot-check 5 random dates against manual calculation"
```

---

## Interview Tip

> When asked about feature pipelines: "I build feature pipelines with five properties: (1) Incremental — process only new data since last watermark. A daily pipeline reading 90-day windows would be 90× too expensive if it reprocessed everything daily. Track watermarks, process deltas. (2) Point-in-time correct — the pipeline code ensures features use only data from BEFORE the computation timestamp. Same code works for both daily production runs and historical backfills. (3) Validated — every pipeline run validates output before writing: schema checks (correct types), completeness (null rate within bounds), volume (row count matches expectations), and distribution (values in expected range). Validation failures block the write and alert. (4) Idempotent — running the pipeline twice for the same date produces identical output. This is critical for backfills and retry-after-failure. (5) Observable — I track: pipeline duration (trending up = growing data or degrading performance), records processed, validation pass rates, and feature freshness post-write. Dagster is my preferred orchestrator for feature pipelines — its asset-based model maps naturally to features, and partitioned assets handle backfills elegantly."

---

## Common Mistakes

1. **No validation before write** — Pipeline computes features, immediately writes to feature store. Feature computation had a bug → bad values in store → model serves garbage predictions for hours. Solution: validation gate between computation and write. Check null rates, value ranges, row counts, and data types before writing anywhere.

2. **Non-idempotent pipelines** — Running pipeline twice for the same date produces different results (due to randomness, non-deterministic joins, or append-mode writes). Backfill produces inconsistent data. Solution: design pipelines to be idempotent — same inputs always produce same outputs. Use overwrite mode (not append) for date partitions.

3. **Tightly coupled to source schema** — Source database team renames a column → feature pipeline breaks → features go stale → models degrade. Solution: data contracts between source and feature pipelines. Define expected schema explicitly, validate on read, alert immediately on schema changes (don't just fail silently).

4. **No backfill strategy** — Creating a new feature but having no way to compute historical values. Training data has null for this feature for all historical examples → feature is useless for training. Solution: design every feature pipeline to support backfilling from day one. Same code, parameterized by date, run for historical range.

5. **Pipeline and feature logic diverge** — "Production pipeline" and "training notebook" compute the same feature with slightly different logic. Training data looks different from production data → training-serving skew. Solution: single source of truth for feature computation logic. Training and serving use the same pipeline code/output.

---

## Key Takeaways

- Feature pipelines: orchestrated systems that reliably compute, validate, and serve features
- Incremental processing: track watermarks, process only new data (don't recompute everything)
- Point-in-time correctness: same code handles daily runs and historical backfills correctly
- Validation gates: schema, completeness, volume, distribution — check BEFORE writing
- Idempotency: same input → same output (critical for retries and backfills)
- Backfill support: every feature pipeline must support historical computation from day one
- Orchestration: Dagster (asset-based, partition-native) or Airflow (widely adopted, mature)
- Testing: unit tests (transformations), integration tests (full pipeline), data quality tests (output)
- Single source of truth: training and serving use same pipeline code (no logic divergence)
- Observability: duration trends, records processed, validation rates, freshness post-write
