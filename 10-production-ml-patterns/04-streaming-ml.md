# Streaming ML

## The Problem / Why This Matters

Streaming ML sits between batch and online prediction — processing data as events arrive continuously rather than in scheduled batches or per-request. In a streaming ML system, models consume event streams (Kafka, Kinesis, Pub/Sub), run inference on each event (or micro-batch of events), and produce predictions that feed into downstream systems. This pattern is critical for use cases where: (1) data arrives continuously and must be processed immediately but not necessarily per-user-request (IoT sensors, clickstreams, log events), (2) features depend on recent event aggregates (last 5 minutes of clicks, rolling averages), or (3) models need to update incrementally as new data arrives (online learning). In 2026, streaming ML powers real-time personalization, anomaly detection on infrastructure metrics, continuous fraud scoring, real-time recommendation updates, and event-driven ML pipelines. The engineering challenge is handling backpressure, ensuring exactly-once semantics, managing stateful computations (windowed aggregates), and maintaining model freshness — all while processing millions of events per second with sub-second latency.

---

## The Analogy

Think of streaming ML like a river monitoring system:

- **Batch prediction** = Taking water samples once a week, sending to lab, getting results in 3 days. By the time you detect contamination, it's been flowing downstream for days.
- **Online prediction** = Someone asks "is THIS glass of water safe?" and you test it immediately. Fast per-request, but you only test when someone asks.
- **Streaming ML** = Continuous sensor in the river, testing every second automatically. Detects contamination within seconds of it entering the water system. No one has to ask — it continuously monitors and alerts.

The streaming sensor doesn't wait for someone to ask (online) or for a scheduled collection (batch) — it continuously processes the flow and reacts in real-time.

---

## Deep Dive

### Streaming ML Architecture

```yaml
Architecture:
  event_source:
    kafka: "Most common, high-throughput, durable, exactly-once semantics"
    kinesis: "AWS managed, simpler ops, auto-scaling"
    pubsub: "GCP, serverless, global routing"
    flink_cdc: "Database change streams (Change Data Capture)"
    
  stream_processor:
    apache_flink: "Gold standard for stateful streaming, exactly-once, complex windows"
    kafka_streams: "Lightweight, library (not cluster), good for simpler processing"
    spark_structured_streaming: "Micro-batch, familiar Spark API, good for batch+stream"
    bytewax: "Python-native streaming (Rust backend), ML-friendly API"
    
  ml_integration:
    embedded_model:
      what: "Model loaded inside stream processor"
      latency: "Lowest (no network hop)"
      update: "Redeploy stream processor to update model"
      
    model_service_call:
      what: "Stream processor calls external model service"
      latency: "Higher (network hop)"
      update: "Update model service independently (decoupled)"
      
    feature_materialization:
      what: "Stream computes features, stores in feature store for online serving"
      pattern: "Stream → aggregate → feature store → online model reads"
      
  output:
    predictions_topic: "Write predictions to another Kafka topic"
    feature_store: "Materialize streaming features for online serving"
    database: "Write predictions to serving database"
    alerting: "Trigger alerts based on prediction thresholds"
```

### Streaming Feature Computation

```yaml
Streaming_Features:
  windowed_aggregates:
    description: "Aggregate events over sliding or tumbling windows"
    examples:
      - "Count of transactions in last 5 minutes (fraud detection)"
      - "Average response time in last 1 minute (anomaly detection)"
      - "Number of page views in last 30 seconds (engagement scoring)"
    window_types:
      tumbling: "Fixed-size, non-overlapping (every 5 min, reset)"
      sliding: "Fixed-size, overlapping (last 5 min from NOW, updated continuously)"
      session: "Dynamic size, based on activity gaps (user session = events with < 30s gap)"
      
  event_patterns:
    description: "Detect patterns across multiple events"
    examples:
      - "Login from new device followed by large transfer (fraud pattern)"
      - "Three failed requests in 10 seconds (service degradation pattern)"
      - "User viewed product 3 times without purchasing (intent pattern)"
      
  real_time_embeddings:
    description: "Compute embeddings on streaming text/events"
    examples:
      - "Embed each chat message for real-time topic detection"
      - "Encode user behavior sequence for next-action prediction"
```

### Implementation

```python
# Streaming ML pipeline implementation

"""
Streaming ML system using Apache Flink (via PyFlink) and Kafka.
Demonstrates: windowed feature computation, model inference on streams,
and real-time anomaly detection.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """An event from the stream."""
    event_id: str
    entity_id: str
    event_type: str
    timestamp: float
    payload: Dict


class StreamingMLPipeline:
    """
    Streaming ML pipeline that:
    1. Consumes events from Kafka
    2. Computes windowed features (stateful)
    3. Runs model inference on each event (with features)
    4. Produces predictions to output topic
    """
    
    def __init__(
        self,
        model,
        feature_windows: Dict[str, int],  # feature_name → window_seconds
        prediction_topic: str = "predictions",
    ):
        self.model = model
        self.feature_windows = feature_windows
        self.prediction_topic = prediction_topic
        
        # State: per-entity event buffers for windowed features
        # In production: managed by Flink state backend (RocksDB)
        self.entity_state: Dict[str, EntityState] = {}
    
    def process_event(self, event: StreamEvent) -> Optional[Dict]:
        """
        Process a single event from the stream.
        
        Steps:
        1. Update entity state (add event to window)
        2. Compute windowed features
        3. Run model inference
        4. Return prediction (or None if no prediction needed)
        """
        # Step 1: Update state
        state = self._get_or_create_state(event.entity_id)
        state.add_event(event)
        
        # Step 2: Compute windowed features
        features = self._compute_windowed_features(state, event)
        
        # Step 3: Model inference
        prediction = self.model.predict(features)
        confidence = self.model.predict_confidence(features)
        
        # Step 4: Produce prediction
        result = {
            "entity_id": event.entity_id,
            "event_id": event.event_id,
            "prediction": prediction,
            "confidence": confidence,
            "features": features,
            "model_version": self.model.version,
            "timestamp": time.time(),
        }
        
        return result
    
    def _get_or_create_state(self, entity_id: str) -> "EntityState":
        """Get or create stateful window for entity."""
        if entity_id not in self.entity_state:
            self.entity_state[entity_id] = EntityState(
                entity_id=entity_id,
                max_window_seconds=max(self.feature_windows.values())
            )
        return self.entity_state[entity_id]
    
    def _compute_windowed_features(
        self, state: "EntityState", current_event: StreamEvent
    ) -> Dict:
        """
        Compute features from windowed event history.
        
        Example features:
        - transaction_count_5min: how many transactions in last 5 minutes
        - avg_amount_1hr: average transaction amount in last hour
        - unique_locations_24hr: distinct locations in last 24 hours
        """
        now = current_event.timestamp
        features = {}
        
        for feature_name, window_seconds in self.feature_windows.items():
            window_events = state.get_events_in_window(now - window_seconds, now)
            
            if feature_name.startswith("count_"):
                features[feature_name] = len(window_events)
            elif feature_name.startswith("avg_"):
                field = feature_name.replace("avg_", "").split("_")[0]
                values = [e.payload.get(field, 0) for e in window_events]
                features[feature_name] = sum(values) / max(len(values), 1)
            elif feature_name.startswith("max_"):
                field = feature_name.replace("max_", "").split("_")[0]
                values = [e.payload.get(field, 0) for e in window_events]
                features[feature_name] = max(values) if values else 0
            elif feature_name.startswith("unique_"):
                field = feature_name.replace("unique_", "").split("_")[0]
                values = [e.payload.get(field) for e in window_events]
                features[feature_name] = len(set(v for v in values if v))
        
        # Add current event features
        features["current_amount"] = current_event.payload.get("amount", 0)
        features["current_type"] = current_event.payload.get("type", "unknown")
        features["hour_of_day"] = int((now % 86400) / 3600)
        
        return features


class EntityState:
    """
    Stateful event buffer for one entity.
    
    Maintains a sliding window of recent events.
    In production: backed by Flink state (RocksDB, checkpointed).
    """
    
    def __init__(self, entity_id: str, max_window_seconds: int = 86400):
        self.entity_id = entity_id
        self.max_window_seconds = max_window_seconds
        self.events: List[StreamEvent] = []
    
    def add_event(self, event: StreamEvent):
        """Add event and evict expired events."""
        self.events.append(event)
        self._evict_expired(event.timestamp)
    
    def get_events_in_window(
        self, start_time: float, end_time: float
    ) -> List[StreamEvent]:
        """Get events within time window."""
        return [
            e for e in self.events
            if start_time <= e.timestamp <= end_time
        ]
    
    def _evict_expired(self, current_time: float):
        """Remove events older than max window."""
        cutoff = current_time - self.max_window_seconds
        self.events = [e for e in self.events if e.timestamp >= cutoff]


class OnlineLearningPipeline:
    """
    Online learning: model updates incrementally on each event.
    
    Instead of batch retraining (expensive, infrequent),
    the model learns from each new labeled event immediately.
    
    Use cases:
    - Ad click prediction (learn from each click/no-click)
    - Spam detection (learn from each user report)
    - News ranking (learn from engagement signals)
    
    Key challenges:
    - Catastrophic forgetting (model forgets old patterns)
    - Label delay (outcome known minutes/hours after prediction)
    - Concept drift (what's spam today wasn't spam yesterday)
    """
    
    def __init__(self, model, learning_rate: float = 0.001):
        self.model = model  # Must support incremental/online updates
        self.learning_rate = learning_rate
        self.update_count = 0
        self.performance_buffer = []  # Track recent performance
    
    def predict_and_learn(
        self, features: Dict, label: Optional[float] = None
    ) -> Dict:
        """
        Two modes:
        1. Predict only (label unknown yet)
        2. Learn from labeled example (label arrived later)
        """
        # Predict
        prediction = self.model.predict(features)
        
        if label is not None:
            # Label available → update model
            loss = self.model.partial_fit(features, label, lr=self.learning_rate)
            self.update_count += 1
            
            # Track performance
            self.performance_buffer.append({
                "prediction": prediction,
                "actual": label,
                "loss": loss,
            })
            
            # Keep only last 1000 for monitoring
            if len(self.performance_buffer) > 1000:
                self.performance_buffer = self.performance_buffer[-1000:]
        
        return {
            "prediction": prediction,
            "model_updates": self.update_count,
            "recent_accuracy": self._recent_accuracy(),
        }
    
    def _recent_accuracy(self) -> float:
        """Compute accuracy over recent predictions."""
        if not self.performance_buffer:
            return 0.0
        correct = sum(
            1 for p in self.performance_buffer
            if round(p["prediction"]) == p["actual"]
        )
        return correct / len(self.performance_buffer)


class StreamingAnomalyDetector:
    """
    Streaming anomaly detection on metrics/events.
    
    Pattern: maintain statistical model of normal behavior,
    flag events that deviate significantly.
    
    Approaches:
    1. Statistical: z-score on rolling statistics
    2. ML-based: isolation forest updated incrementally
    3. Deep learning: autoencoder reconstruction error
    """
    
    def __init__(
        self,
        sensitivity: float = 3.0,  # Number of standard deviations
        window_size: int = 1000,   # Rolling window for statistics
    ):
        self.sensitivity = sensitivity
        self.window_size = window_size
        self.values: List[float] = []
    
    def detect(self, value: float) -> Dict:
        """
        Check if value is anomalous given recent history.
        
        Returns anomaly score and whether it exceeds threshold.
        """
        self.values.append(value)
        
        # Need minimum history
        if len(self.values) < 30:
            return {"is_anomaly": False, "score": 0.0, "reason": "insufficient_history"}
        
        # Keep rolling window
        if len(self.values) > self.window_size:
            self.values = self.values[-self.window_size:]
        
        # Compute statistics (excluding current value)
        history = self.values[:-1]
        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std = variance ** 0.5
        
        # Z-score
        if std == 0:
            z_score = 0.0
        else:
            z_score = abs(value - mean) / std
        
        is_anomaly = z_score > self.sensitivity
        
        return {
            "is_anomaly": is_anomaly,
            "score": z_score,
            "threshold": self.sensitivity,
            "value": value,
            "expected_range": (mean - self.sensitivity * std, mean + self.sensitivity * std),
            "reason": f"z_score={z_score:.2f} > {self.sensitivity}" if is_anomaly else "within_normal",
        }
```

### Exactly-Once Semantics (EOS)

```yaml
Exactly_Once_Semantics:
  problem: "Stream fails mid-processing → duplicate or lost predictions"
  
  solutions:
    kafka_transactions:
      how: "Kafka transactional producer + consumer offset commit in same transaction"
      ensures: "Each event processed exactly once (no duplicates, no loss)"
      framework: "Flink handles this automatically with checkpointing"
      
    idempotent_writes:
      how: "Predictions written with event_id as key (duplicate writes overwrite same key)"
      ensures: "Even if processed twice, result is the same"
      simpler: "Doesn't require distributed transactions"
      
    checkpointing:
      how: "Flink periodically snapshots state + offsets"
      recovery: "On failure, restore from last checkpoint, replay from offset"
      overhead: "Small latency cost (checkpoint barriers), configurable interval"
      
  trade_offs:
    exactly_once: "Highest correctness, highest complexity, slight latency overhead"
    at_least_once: "May process duplicates, simpler, use with idempotent writes"
    at_most_once: "May lose events, simplest, acceptable if loss is tolerable"
```

### Backpressure Handling

```yaml
Backpressure:
  what: "Upstream produces events faster than downstream can process"
  symptoms: "Growing lag, increasing latency, OOM (Out of Memory) errors"
  
  strategies:
    rate_limiting:
      how: "Limit consumer rate to sustainable throughput"
      trade_off: "Lag grows during bursts"
      
    auto_scaling:
      how: "Add more consumers/workers when lag grows"
      trade_off: "Scale-up latency (minutes), cost when burst is short"
      
    load_shedding:
      how: "Drop low-priority events when overwhelmed"
      trade_off: "Data loss (acceptable if not all events are critical)"
      
    buffering:
      how: "Kafka itself acts as buffer (events wait in topic)"
      benefit: "No data loss, consumers catch up when burst ends"
      limit: "Kafka retention (default 7 days, configurable)"
      
    adaptive_processing:
      how: "Reduce processing quality under load (skip expensive features)"
      trade_off: "Slightly worse predictions, but no lag growth"
```

---

## How It Works in Practice

### Real-Time Fraud Detection System

```yaml
Fraud_Detection_Streaming:
  input: "Transaction events (Kafka topic: transactions)"
  throughput: "50,000 events/second"
  latency: "< 200ms from event to prediction"
  
  pipeline:
    step_1: "Consume transaction event"
    step_2: "Compute streaming features (transactions_last_5min, unique_merchants_1hr)"
    step_3: "Fetch user profile features from feature store"
    step_4: "Run fraud model (XGBoost, embedded in Flink job)"
    step_5: "If score > 0.8 → produce to 'high_risk' topic (trigger block)"
    step_6: "If score 0.5-0.8 → produce to 'review' topic (human review)"
    step_7: "All predictions → 'predictions' topic (for monitoring)"
    
  state_management:
    what: "Per-user transaction history (last 24 hours)"
    backend: "Flink RocksDB state backend (disk-backed, survives failures)"
    checkpoint_interval: "Every 30 seconds"
    
  exactly_once: "Kafka transactions + Flink checkpointing"
```

---

## Interview Tip

> When asked about streaming ML: "Streaming ML processes events continuously — between batch (scheduled, stale) and online (per-request). I use it for three main patterns: (1) Streaming feature computation — aggregate events in time windows (transactions in last 5 minutes, clicks in last 30 seconds) and materialize to feature store. This gives online serving models access to near-real-time features without computing at request time. (2) Event-driven inference — run model on each event as it arrives (Flink + embedded model). For fraud detection: score each transaction within 200ms using windowed features (transaction velocity, location anomaly). (3) Online learning — update model incrementally on each labeled event. For ad click prediction: learn from each click/no-click to adapt to changing user behavior within minutes, not days. Key engineering concerns: exactly-once semantics (EOS) via Kafka transactions + Flink checkpointing (prevent duplicate predictions), backpressure handling (buffer in Kafka, auto-scale consumers), state management (per-entity windows backed by RocksDB, checkpointed for fault tolerance). For tools in 2026: Apache Flink for complex stateful streaming, Bytewax for Python-native streaming ML, Kafka Streams for lightweight processing. The main trade-off vs. batch: streaming is operationally complex (state management, exactly-once, ordering) — only use when latency of hours (batch) isn't acceptable but millisecond latency (online) isn't required."

---

## Common Mistakes

1. **Using streaming when batch suffices** — Building a complex Flink pipeline for features that don't change within hours. Solution: if your model is retrained daily and features are daily aggregates, batch is simpler and cheaper. Only use streaming when minutes-level freshness matters.

2. **Ignoring state size** — Keeping all events forever in state. State grows unbounded → OOM. Solution: define TTL (Time To Live) for state (e.g., evict events older than 24 hours). Use RocksDB state backend for disk-based state.

3. **Not handling late events** — Events arrive out of order (network delays). Window computes on incomplete data. Solution: allowed lateness in Flink (wait N seconds after window closes for late events). Accept that some late events will be missed.

4. **Exactly-once is expensive** — Using exactly-once semantics when at-least-once with idempotent writes would suffice. Solution: if your output store supports upserts (keyed by event_id), at-least-once is simpler and nearly equivalent.

5. **No monitoring of consumer lag** — Stream processing falls behind (backpressure) and nobody notices until predictions are minutes stale. Solution: monitor consumer lag (Kafka lag metric). Alert when lag exceeds threshold (e.g., > 10,000 events behind).

---

## Key Takeaways

- Streaming ML: continuous event processing for near-real-time features and predictions
- Three patterns: streaming features (→ feature store), event-driven inference, online learning
- Architecture: Kafka (events) → Flink/Bytewax (processing) → feature store or prediction topic
- Windowed aggregates: tumbling, sliding, session windows for temporal features
- Exactly-once semantics: Kafka transactions + Flink checkpointing (prevents duplicates)
- Backpressure: buffer in Kafka, auto-scale consumers, shed low-priority events under load
- State management: per-entity windows, RocksDB backend, TTL for bounded state
- Online learning: model updates incrementally on each event (fast adaptation to drift)
- Trade-off: more complex than batch, simpler than per-request online serving
- Use when: minutes-level freshness needed, features depend on recent event aggregates
