# Batch Prediction

## The Problem / Why This Matters

Not every ML prediction needs to happen in real-time. Many use cases require predictions for all entities (users, products, transactions) computed ahead of time and stored for fast lookup. Batch prediction pre-computes predictions on a schedule (hourly, daily, weekly) and stores results in a database or cache. When a user makes a request, the system simply looks up the pre-computed prediction instead of running model inference. This approach offers massive advantages: simpler infrastructure (no GPU serving endpoint needed), lower cost (one computation covers all lookups), lower latency (database read vs. model inference), and higher reliability (predictions available even if model service is down). In 2026, batch prediction remains the workhouse pattern for most production ML — recommendation scores, risk assessments, marketing segmentation, propensity scores, and any scenario where predictions don't need to reflect real-time changes. The trade-off is freshness: batch predictions are only as current as the last batch run.

---

## The Analogy

Think of batch prediction like a restaurant's daily specials vs. cooking to order:

- **Online prediction** = Cooking to order. Customer arrives, chef prepares their specific meal from scratch. Fresh, customized, but slow and expensive (chef works per customer).
- **Batch prediction** = Daily specials. Chef prepares 50 portions of each special before the restaurant opens. Customer arrives, gets served immediately from pre-prepared stock. Fast, efficient, but limited to what was prepared earlier.

If the daily special is popular and covers 90% of orders, pre-cooking is far more efficient. Only unusual orders (real-time edge cases) need to be cooked fresh.

---

## Deep Dive

### When to Use Batch Prediction

```yaml
Use_Batch_When:
  freshness_tolerance:
    - "Predictions valid for hours or days (don't change with every interaction)"
    - "Example: credit risk score (recalculate daily, not per transaction)"
    - "Example: product recommendations (update nightly based on browsing history)"
    
  scale_needs:
    - "Need predictions for ALL entities (millions of users, products)"
    - "Example: send personalized emails to 10M users (score all users overnight)"
    - "Example: rank all products by relevance for each user segment"
    
  latency_requirements:
    - "Serving latency must be < 10ms (pre-computed lookup faster than inference)"
    - "Example: ad pre-ranking (pre-score all ad-user pairs for fast retrieval)"
    
  cost_optimization:
    - "Inference cost is high (large models) but predictions don't change frequently"
    - "Example: LLM-generated product descriptions (generate once, serve many times)"
    
  infrastructure_simplicity:
    - "No GPU serving infrastructure needed (just batch compute + database)"
    - "Simpler operational burden (no real-time SLA, just pipeline SLA)"

Dont_Use_Batch_When:
  - "Prediction depends on real-time context (current session, live data)"
  - "Entity set is unknown (can't pre-compute for users who haven't signed up)"
  - "Feature values change rapidly (last 5 minutes of behavior matters)"
  - "Freshness is critical (fraud detection — need to catch fraud immediately)"
```

### Batch Prediction Architecture

```yaml
Architecture:
  pipeline:
    1_feature_computation:
      what: "Compute features for all entities"
      source: "Data warehouse, feature store historical data"
      output: "Feature matrix (entity_id × features)"
      schedule: "Match prediction schedule (daily features for daily predictions)"
      
    2_model_inference:
      what: "Run model on all entity features"
      scale: "Process millions of entities (distributed compute)"
      optimization: "Batch processing (GPU batch size optimization)"
      output: "Predictions table (entity_id × prediction × metadata)"
      
    3_result_storage:
      what: "Store predictions for fast serving-time lookup"
      options:
        redis: "Sub-millisecond lookup, limited by memory"
        dynamodb: "Millisecond lookup, scalable, managed"
        postgresql: "Simple, good for < 10M entities"
        bigquery: "Analytical queries on predictions (reporting)"
      schema: "entity_id, prediction, confidence, model_version, computed_at"
      
    4_serving:
      what: "API reads pre-computed prediction from store"
      latency: "< 5ms (database read, no model inference)"
      fallback: "If prediction not found → use default or trigger online inference"
      
  scheduling:
    frequency: "Depends on data freshness requirements"
    options:
      hourly: "For rapidly changing data (trending content)"
      daily: "Most common (user preferences, risk scores)"
      weekly: "Slowly changing signals (long-term preferences)"
    orchestration: "Airflow, Prefect, Dagster, or cloud-native (AWS Step Functions)"
    
  monitoring:
    pipeline_health: "Did the batch job complete successfully?"
    prediction_quality: "Are predictions within expected range?"
    coverage: "How many entities got predictions? (target: 100%)"
    freshness: "How old are the current predictions?"
```

### Implementation

```python
# Batch prediction pipeline implementation

"""
Production batch prediction pipeline.
Computes predictions for all entities on schedule.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BatchPredictionPipeline:
    """
    Batch prediction pipeline that:
    1. Loads features for all entities
    2. Runs model inference in batches
    3. Stores predictions for serving
    4. Validates results
    """
    
    def __init__(
        self,
        model,
        feature_store,
        prediction_store,
        batch_size: int = 10000
    ):
        self.model = model
        self.feature_store = feature_store
        self.prediction_store = prediction_store
        self.batch_size = batch_size
    
    def run(self, prediction_date: Optional[str] = None) -> Dict:
        """
        Execute full batch prediction pipeline.
        
        Args:
            prediction_date: Date for point-in-time features (default: today)
            
        Returns:
            Pipeline execution summary
        """
        start_time = datetime.now()
        prediction_date = prediction_date or datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Starting batch prediction for date: {prediction_date}")
        
        # Step 1: Load all entity IDs
        entity_ids = self.feature_store.get_all_entity_ids()
        total_entities = len(entity_ids)
        logger.info(f"Total entities to predict: {total_entities}")
        
        # Step 2: Process in batches
        all_predictions = []
        failed_entities = []
        
        for batch_start in range(0, total_entities, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_entities)
            batch_ids = entity_ids[batch_start:batch_end]
            
            try:
                # Get features for batch
                features_df = self.feature_store.get_batch_features(
                    entity_ids=batch_ids,
                    point_in_time=prediction_date,
                    feature_names=self.model.expected_features
                )
                
                # Handle missing features (imputation)
                features_df = self._impute_missing(features_df)
                
                # Run inference
                predictions = self.model.predict_batch(features_df)
                confidences = self.model.predict_proba_batch(features_df)
                
                # Collect results
                for entity_id, pred, conf in zip(batch_ids, predictions, confidences):
                    all_predictions.append({
                        "entity_id": entity_id,
                        "prediction": pred,
                        "confidence": float(conf.max()),
                        "prediction_date": prediction_date,
                        "model_version": self.model.version,
                        "computed_at": datetime.now().isoformat(),
                    })
                    
            except Exception as e:
                logger.error(f"Batch {batch_start}-{batch_end} failed: {e}")
                failed_entities.extend(batch_ids)
        
        # Step 3: Validate predictions
        validation = self._validate_predictions(all_predictions)
        
        if not validation["passed"]:
            logger.error(f"Validation failed: {validation['errors']}")
            return {"status": "FAILED", "validation": validation}
        
        # Step 4: Store predictions (atomic swap)
        self._store_predictions(all_predictions, prediction_date)
        
        # Step 5: Summary
        duration = (datetime.now() - start_time).total_seconds()
        summary = {
            "status": "SUCCESS",
            "total_entities": total_entities,
            "predictions_generated": len(all_predictions),
            "failed_entities": len(failed_entities),
            "coverage": len(all_predictions) / total_entities,
            "duration_seconds": duration,
            "prediction_date": prediction_date,
            "model_version": self.model.version,
            "validation": validation,
        }
        
        logger.info(f"Batch prediction complete: {summary}")
        return summary
    
    def _impute_missing(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing features with sensible defaults."""
        # Use training-time statistics for imputation
        for col in features_df.columns:
            if features_df[col].isnull().any():
                default = self.model.feature_defaults.get(col, 0)
                features_df[col] = features_df[col].fillna(default)
        return features_df
    
    def _validate_predictions(self, predictions: list) -> Dict:
        """
        Validate batch predictions before storing.
        
        Checks:
        1. No NaN predictions
        2. Predictions within expected range
        3. Distribution not drastically different from last run
        4. Coverage meets minimum threshold
        """
        errors = []
        
        pred_values = [p["prediction"] for p in predictions]
        conf_values = [p["confidence"] for p in predictions]
        
        # Check for NaN
        nan_count = sum(1 for p in pred_values if p is None or str(p) == "nan")
        if nan_count > 0:
            errors.append(f"Found {nan_count} NaN predictions")
        
        # Check confidence range
        invalid_conf = sum(1 for c in conf_values if c < 0 or c > 1)
        if invalid_conf > 0:
            errors.append(f"Found {invalid_conf} invalid confidence scores")
        
        # Check prediction distribution (compare to last run)
        last_run = self.prediction_store.get_last_run_stats()
        if last_run:
            current_mean = np.mean([p for p in pred_values if isinstance(p, (int, float))])
            last_mean = last_run.get("mean_prediction", current_mean)
            
            if abs(current_mean - last_mean) / (last_mean + 1e-10) > 0.2:
                errors.append(
                    f"Prediction mean shifted >20%: {last_mean:.4f} → {current_mean:.4f}"
                )
        
        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "stats": {
                "count": len(predictions),
                "nan_count": nan_count,
                "mean_confidence": np.mean(conf_values),
            },
        }
    
    def _store_predictions(self, predictions: list, prediction_date: str):
        """
        Store predictions atomically (swap old with new).
        
        Pattern: write to staging table → validate → swap with production table.
        This ensures serving always reads complete, valid predictions.
        """
        # Write to staging
        self.prediction_store.write_staging(predictions)
        
        # Validate staging matches expectations
        staging_count = self.prediction_store.staging_count()
        if staging_count == len(predictions):
            # Atomic swap: staging → production
            self.prediction_store.swap_to_production(prediction_date)
            logger.info(f"Swapped {staging_count} predictions to production")
        else:
            raise ValueError(
                f"Staging count mismatch: expected {len(predictions)}, got {staging_count}"
            )


class IncrementalBatchPrediction:
    """
    Incremental batch: only re-predict entities whose features changed.
    
    Much faster than full batch when only 5-10% of entities change daily.
    
    Approach:
    1. Detect which entities have updated features since last run
    2. Predict only those entities
    3. Merge new predictions with existing (unchanged) predictions
    """
    
    def __init__(self, model, feature_store, prediction_store):
        self.model = model
        self.feature_store = feature_store
        self.prediction_store = prediction_store
    
    def run_incremental(self) -> Dict:
        """Run incremental batch (only changed entities)."""
        
        # Find entities with changed features since last prediction
        last_run_time = self.prediction_store.get_last_run_time()
        changed_entities = self.feature_store.get_changed_entities(
            since=last_run_time
        )
        
        logger.info(f"Incremental batch: {len(changed_entities)} entities changed")
        
        if not changed_entities:
            return {"status": "NO_CHANGES", "entities_updated": 0}
        
        # Predict only changed entities
        features = self.feature_store.get_batch_features(
            entity_ids=changed_entities,
            feature_names=self.model.expected_features
        )
        
        predictions = self.model.predict_batch(features)
        
        # Update only changed predictions (not full swap)
        self.prediction_store.upsert(
            entity_ids=changed_entities,
            predictions=predictions,
            model_version=self.model.version
        )
        
        return {
            "status": "SUCCESS",
            "entities_updated": len(changed_entities),
            "total_entities": self.prediction_store.total_count(),
            "update_percentage": len(changed_entities) / self.prediction_store.total_count(),
        }
```

### Batch Prediction at Scale

```yaml
Scaling_Strategies:
  horizontal_scaling:
    what: "Distribute prediction across many workers"
    implementation: "Spark, Ray, Dask — partition entities across workers"
    example: "10M entities ÷ 100 workers = 100K per worker"
    benefit: "Linear scaling with worker count"
    
  gpu_batch_optimization:
    what: "Maximize GPU utilization with optimal batch sizes"
    implementation: "Batch size tuning (32, 64, 128, 256) to maximize throughput"
    benefit: "10-100x throughput vs. single-item inference"
    caveat: "Larger batch = more GPU memory needed"
    
  incremental_processing:
    what: "Only re-predict entities whose features changed"
    implementation: "Change detection on feature store → filter to changed entities"
    benefit: "90% fewer predictions when only 10% of entities change daily"
    
  caching_hot_entities:
    what: "Cache most-accessed predictions in Redis (hot entities)"
    implementation: "Redis LRU cache in front of prediction store"
    benefit: "< 1ms for frequently accessed entities"
    
  materialized_views:
    what: "Pre-join predictions with entity metadata for fast queries"
    implementation: "Database materialized view or denormalized table"
    benefit: "Avoid joins at query time"
```

### Handling Freshness

```yaml
Freshness_Strategies:
  full_refresh:
    what: "Re-predict all entities on schedule"
    when: "Features change globally (market conditions, seasonal)"
    frequency: "Daily or weekly"
    
  incremental_refresh:
    what: "Re-predict only changed entities"
    when: "Most entities stable, few change daily"
    frequency: "Hourly or more"
    
  hybrid:
    what: "Incremental hourly + full refresh daily"
    benefit: "Low latency for changes + guarantee of completeness"
    
  staleness_detection:
    what: "Monitor how old predictions are, alert on excessive staleness"
    metrics:
      max_age: "Oldest prediction (should be < 2x schedule interval)"
      pct_stale: "% of predictions older than threshold"
    alert: "If > 10% predictions are stale → pipeline failure suspected"
```

---

## How It Works in Practice

### Daily Batch Prediction Workflow

```yaml
Daily_Workflow:
  02_00_am: "Feature pipeline runs (compute features from yesterday's data)"
  03_00_am: "Feature pipeline complete → trigger batch prediction"
  03_30_am: "Batch prediction starts (10M entities)"
  05_00_am: "Batch prediction complete (validated)"
  05_05_am: "Atomic swap: new predictions → production table"
  05_10_am: "Monitoring: verify prediction quality metrics normal"
  06_00_am: "Users wake up → served fresh predictions from new batch"
  
  if_failure:
    - "Predictions from yesterday remain in production (stale but functional)"
    - "Alert on-call engineer"
    - "Manual investigation and re-run"
    - "SLA: predictions must be refreshed within 24h (one retry window)"
```

---

## Interview Tip

> When asked about batch prediction: "Batch prediction is my default pattern for ML serving because it's simpler, cheaper, and more reliable than real-time inference — and most use cases tolerate predictions that are hours old. My approach: (1) Architecture — feature pipeline computes features for all entities (Spark/BigQuery), model predicts in optimized batches (GPU batch size tuning for throughput), results stored in a serving store (Redis for sub-millisecond lookup, DynamoDB for scalability). (2) Atomic deployment — write predictions to staging table, validate (no NaN, distribution stable, coverage > 99%), then atomic swap to production. This ensures serving always sees complete, valid predictions. (3) Incremental optimization — detect which entities' features changed since last run, only re-predict those (90% compute savings when 10% of entities change daily). Full refresh periodically for completeness. (4) Monitoring — track freshness (how old are predictions?), coverage (how many entities have predictions?), and quality (prediction distribution stable?). Alert if predictions are staler than 2x the schedule interval. Key trade-off: batch predictions are stale by design. I use batch when predictions don't need to reflect real-time behavior — which is most cases (daily recommendations, weekly risk scores, nightly marketing segments). Only escalate to online serving when real-time freshness is truly required."

---

## Common Mistakes

1. **Not validating before serving** — Batch job completes, predictions deployed immediately. But predictions are all zeros (bug in feature pipeline). Users served garbage for hours before anyone notices. Solution: validate predictions before swapping to production — distribution stability, no NaN, coverage check.

2. **No atomic swap** — Write predictions directly to production table as they're computed. Mid-write: some entities have new predictions, some have old. Inconsistent serving. Solution: write to staging → validate → atomic swap (rename tables or redirect pointer).

3. **Full refresh when incremental suffices** — Recompute 10M predictions daily when only 100K entities changed. Waste of compute. Solution: change detection → incremental prediction → merge with unchanged predictions. Full refresh weekly for completeness.

4. **No fallback for missing predictions** — New user signs up at 3 PM. Batch ran at 2 AM. No prediction exists for this user. Application crashes or returns empty. Solution: have a fallback (default prediction, popular items, rule-based backup) for entities not in the batch prediction store.

5. **Not monitoring staleness** — Batch pipeline silently fails for 3 days. Predictions are 3 days stale. Nobody notices because the serving layer still returns predictions (just old ones). Solution: monitor prediction age. Alert if max staleness exceeds 2x schedule interval.

---

## Key Takeaways

- Batch prediction: pre-compute predictions for all entities, serve from database lookup
- Advantages: simpler infrastructure, lower cost, lower latency (DB read vs. model inference), higher reliability
- Trade-off: predictions are stale by design (as old as last batch run)
- Use when: freshness tolerance > hours, need predictions for all entities, latency < 10ms required
- Atomic deployment: staging → validate → swap (never serve partially-computed results)
- Incremental optimization: only re-predict entities with changed features (90% savings)
- Validation: distribution stability, no NaN, coverage check before production swap
- Freshness monitoring: track prediction age, alert on excessive staleness
- Fallback: always have a backup for entities not in prediction store (new users)
- Scale: Spark/Ray for distributed computation, GPU batch optimization for throughput
