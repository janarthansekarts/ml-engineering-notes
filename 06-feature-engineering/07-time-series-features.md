# Time Series Features

## The Problem / Why This Matters

Time series data — sequences of values ordered by time — is the backbone of most production ML features. User activity over time, financial transactions, sensor readings, web traffic patterns, stock prices — all are time series. The challenge: raw timestamps and event sequences aren't useful to models directly. You need to engineer features that capture temporal patterns: trends (is this metric going up or down?), seasonality (does behavior differ by day of week?), velocity (how fast are things changing?), and anomalies (is this current value unusual given history?). Time series feature engineering is notoriously tricky because of: (1) temporal leakage (accidentally using future data), (2) irregular time series (events don't arrive at fixed intervals), (3) multiple time scales (pattern visible at hourly scale isn't visible at daily), and (4) computational cost (windowed aggregations over millions of entities are expensive). Getting time series features right — fresh, point-in-time correct, efficiently computed — is often the difference between a model that detects fraud in real-time and one that catches it a day too late.

---

## The Analogy

Think of time series features like analyzing a patient's medical chart:

- **Raw time series** = The complete chart with every vital sign measurement, every lab result, every note. Overwhelming in raw form.
- **Windowed aggregations** = "Average blood pressure over the last week." Summarizes recent history into a useful signal.
- **Lag features** = "What was blood pressure yesterday vs today?" Captures change.
- **Rolling statistics** = "Has blood pressure been trending up over the last month?" Captures direction of change.
- **Seasonality** = "Blood pressure is always higher in morning than evening — so this morning's reading is normal even though it's higher than last night's." Context matters.
- **Rate of change** = "Blood pressure jumped 30 points in 2 hours — this is an emergency" vs "Blood pressure rose 30 points over 3 months — this is gradual concern." Speed of change matters as much as absolute value.

---

## Deep Dive

### Windowed Aggregations

```yaml
Windowed_Aggregations:
  what: "Summarize events/values within a time window"
  windows: "1h, 6h, 24h, 7d, 30d, 90d — choose based on signal"
  
  aggregation_functions:
    basic:
      count: "Number of events in window"
      sum: "Total value in window"
      mean: "Average value in window"
      median: "Middle value (robust to outliers)"
      min_max: "Extreme values in window"
      std: "Variability in window"
      
    advanced:
      distinct_count: "Unique values in window (e.g., unique merchants)"
      ratio: "Proportion (e.g., failed_logins / total_logins in window)"
      percentile: "p90, p95, p99 values in window"
      first_last: "First and last values in window"
      mode: "Most common value in window"
      
  multi_window_pattern:
    concept: "Same metric over multiple windows reveals different signals"
    example:
      feature_1: "transaction_count_1h"    # Very recent activity
      feature_2: "transaction_count_24h"   # Today's activity
      feature_3: "transaction_count_7d"    # This week's activity
      feature_4: "transaction_count_30d"   # This month's activity
    why: "Comparing windows reveals acceleration/deceleration"
    derived: "transaction_count_1h / (transaction_count_24h / 24) = hourly ratio vs daily average"
```

### Lag Features

```python
# Lag features: comparing current to past values

import pandas as pd
import numpy as np


def compute_lag_features(df: pd.DataFrame, value_col: str, 
                         entity_col: str, time_col: str,
                         lags: list = [1, 7, 14, 30]) -> pd.DataFrame:
    """
    Compute lag features for time series data.
    
    Lag features answer: "What was the value N time periods ago?"
    This captures patterns like: "Today's value vs yesterday's value"
    """
    
    # Sort by entity and time
    df = df.sort_values([entity_col, time_col])
    
    for lag in lags:
        # Shift within each entity group
        df[f"{value_col}_lag_{lag}d"] = df.groupby(entity_col)[value_col].shift(lag)
        
        # Difference from lag (absolute change)
        df[f"{value_col}_diff_{lag}d"] = df[value_col] - df[f"{value_col}_lag_{lag}d"]
        
        # Percentage change from lag
        df[f"{value_col}_pct_change_{lag}d"] = (
            df[f"{value_col}_diff_{lag}d"] / df[f"{value_col}_lag_{lag}d"].clip(lower=1e-10)
        )
    
    return df


def compute_rolling_features(df: pd.DataFrame, value_col: str,
                             entity_col: str, time_col: str,
                             windows: list = [7, 14, 30]) -> pd.DataFrame:
    """
    Compute rolling window statistics.
    
    Rolling features answer: "What's the trend over the last N periods?"
    """
    
    df = df.sort_values([entity_col, time_col])
    
    for window in windows:
        group = df.groupby(entity_col)[value_col]
        
        # Rolling mean (trend)
        df[f"{value_col}_rolling_mean_{window}d"] = group.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        
        # Rolling std (volatility)
        df[f"{value_col}_rolling_std_{window}d"] = group.transform(
            lambda x: x.rolling(window, min_periods=2).std()
        )
        
        # Rolling min/max
        df[f"{value_col}_rolling_min_{window}d"] = group.transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )
        df[f"{value_col}_rolling_max_{window}d"] = group.transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
        
        # Value relative to rolling window (z-score within window)
        df[f"{value_col}_zscore_{window}d"] = (
            (df[value_col] - df[f"{value_col}_rolling_mean_{window}d"]) / 
            df[f"{value_col}_rolling_std_{window}d"].clip(lower=1e-10)
        )
        
        # Value relative to rolling range (0-1 normalized within window)
        range_val = df[f"{value_col}_rolling_max_{window}d"] - df[f"{value_col}_rolling_min_{window}d"]
        df[f"{value_col}_range_position_{window}d"] = (
            (df[value_col] - df[f"{value_col}_rolling_min_{window}d"]) / 
            range_val.clip(lower=1e-10)
        )
    
    return df
```

### Trend and Velocity Features

```python
# Trend and velocity: capturing direction and speed of change

def compute_trend_features(df: pd.DataFrame, value_col: str,
                           entity_col: str, time_col: str) -> pd.DataFrame:
    """
    Compute trend features that capture direction and acceleration.
    """
    
    df = df.sort_values([entity_col, time_col])
    group = df.groupby(entity_col)[value_col]
    
    # Exponentially weighted mean (recent values weighted more)
    # Captures recent trend better than simple rolling mean
    df[f"{value_col}_ewm_7d"] = group.transform(
        lambda x: x.ewm(span=7, min_periods=1).mean()
    )
    
    # Trend direction: comparing short-term to long-term average
    # If short > long → uptrend, short < long → downtrend
    short_window = 7
    long_window = 30
    
    df[f"{value_col}_short_avg"] = group.transform(
        lambda x: x.rolling(short_window, min_periods=1).mean()
    )
    df[f"{value_col}_long_avg"] = group.transform(
        lambda x: x.rolling(long_window, min_periods=1).mean()
    )
    
    # Trend ratio: > 1 means uptrend, < 1 means downtrend
    df[f"{value_col}_trend_ratio"] = (
        df[f"{value_col}_short_avg"] / df[f"{value_col}_long_avg"].clip(lower=1e-10)
    )
    
    # Velocity: rate of change per time period
    # First derivative of the time series
    df[f"{value_col}_velocity"] = group.transform(lambda x: x.diff())
    
    # Acceleration: rate of change of the rate of change
    # Second derivative — is velocity increasing or decreasing?
    df[f"{value_col}_acceleration"] = group.transform(lambda x: x.diff().diff())
    
    # Momentum: sum of recent changes (like financial momentum indicators)
    df[f"{value_col}_momentum_7d"] = group.transform(
        lambda x: x.diff().rolling(7, min_periods=1).sum()
    )
    
    return df


def compute_seasonality_features(timestamp_series: pd.Series) -> pd.DataFrame:
    """
    Extract cyclical time features that capture seasonality.
    
    Uses sin/cos encoding so the model understands that
    hour 23 and hour 0 are adjacent (not 23 units apart).
    """
    
    features = pd.DataFrame(index=timestamp_series.index)
    
    # Hour of day (cyclical)
    hour = timestamp_series.dt.hour
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    
    # Day of week (cyclical)
    dow = timestamp_series.dt.dayofweek
    features["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    
    # Month of year (cyclical)
    month = timestamp_series.dt.month
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    
    # Day of month (cyclical)
    day = timestamp_series.dt.day
    features["dom_sin"] = np.sin(2 * np.pi * day / 31)
    features["dom_cos"] = np.cos(2 * np.pi * day / 31)
    
    # Binary features
    features["is_weekend"] = (dow >= 5).astype(int)
    features["is_month_start"] = timestamp_series.dt.is_month_start.astype(int)
    features["is_month_end"] = timestamp_series.dt.is_month_end.astype(int)
    
    return features
```

### Event-Based Time Series Features

```yaml
Event_Based_Features:
  what: "Features derived from sequences of discrete events (not regular time series)"
  
  recency_features:
    time_since_last_event: "Seconds/minutes/hours since last event of this type"
    time_since_first_event: "Account age / entity lifetime"
    time_between_events: "Average/median inter-event interval"
    examples:
      - "seconds_since_last_purchase"
      - "hours_since_last_login"
      - "days_since_account_creation"
      - "avg_days_between_orders"
      
  frequency_features:
    event_count_per_window: "How many events in the last N hours/days"
    event_rate: "Events per hour/day (normalized by time period)"
    burst_detection: "Is current activity rate significantly above normal?"
    examples:
      - "logins_last_24h"
      - "purchases_per_week"
      - "api_calls_per_minute"
      - "current_rate_vs_avg_rate_ratio"
      
  sequence_features:
    last_n_events: "Types of the last N events (ordered)"
    transition_probabilities: "P(event_B | previous_event_A)"
    sequence_length: "How many events in current session/sequence"
    examples:
      - "last_3_page_types: [product, cart, checkout]"
      - "session_length_events: 12"
      - "checkout_after_search_probability"
      
  pattern_features:
    regularity: "How regular is the event pattern? (entropy of inter-event times)"
    periodicity: "Does the entity have periodic behavior? (weekly login pattern)"
    anomaly_score: "How unusual is this event given history?"
    examples:
      - "login_regularity_score (0=random, 1=perfectly periodic)"
      - "is_unusual_hour (login at 3am when normally 9am)"
      - "activity_deviation_from_pattern"
```

### Efficient Computation at Scale

```yaml
Efficient_Computation:
  challenges:
    data_volume: "Millions of entities × years of history = billions of events"
    window_overlap: "7d and 30d windows share most data — don't compute twice"
    real_time_need: "Some features need second-level freshness"
    
  optimization_strategies:
    pre_aggregation:
      what: "Aggregate to intermediate granularity before computing features"
      example: "Aggregate events to daily summaries first, then compute 30d rolling mean"
      benefit: "Process 30 daily records instead of 10,000 raw events per window"
      
    incremental_updates:
      what: "Update features by adding new data and dropping old data"
      example: "7d count: add today's events, subtract events from 8 days ago"
      benefit: "O(new_events) instead of O(all_events_in_window)"
      implementation: "Maintain running state (count, sum) with event queue for window management"
      
    approximate_aggregations:
      what: "Use approximate algorithms for expensive operations"
      examples:
        count_distinct: "HyperLogLog (estimate unique count with tiny memory)"
        percentile: "T-Digest (approximate percentiles in streaming fashion)"
        heavy_hitters: "Count-Min Sketch (approximate frequent items)"
      when: "Exact values not critical, scale is massive"
      
    spark_optimization:
      window_functions: "Use Spark window functions with proper partitioning"
      partition_by_entity: "Partition data by entity_id for parallel processing"
      broadcast_calendar: "Small lookup tables (holidays, etc.) broadcast to all nodes"
```

---

## How It Works in Practice

### Production Time Series Feature Pipeline

```yaml
Pipeline:
  batch_features:
    schedule: "Daily at 2:00 AM UTC"
    computation:
      1: "Read last 90 days of events from data lake"
      2: "Pre-aggregate to daily summaries per entity"
      3: "Compute windowed features (7d, 30d, 90d)"
      4: "Compute lag and trend features"
      5: "Write to offline store + materialize to online store"
    technology: "Spark on Kubernetes, 50 executors"
    runtime: "~45 minutes for 10M entities"
    
  streaming_features:
    source: "Kafka topic (real-time events)"
    computation:
      - "Sliding window counts (5min, 1h)"
      - "Time since last event (updated on every event)"
      - "Session-level aggregations"
    technology: "Flink with RocksDB state backend"
    latency: "Event → feature update in <5 seconds"
    
  feature_vector:
    serving_example:
      user_1001:
        batch: "purchase_count_30d=7, avg_order_90d=45.20, trend_ratio=1.3"
        streaming: "events_last_5min=3, seconds_since_last_event=42"
        on_demand: "hour_sin=0.87, hour_cos=0.5, is_weekend=0"
```

---

## Interview Tip

> When asked about time series features: "I engineer time series features in four categories: (1) Windowed aggregations — count, sum, mean, std over multiple time windows (1h, 24h, 7d, 30d). Multiple windows capture different signals: short windows detect bursts, long windows capture trends. (2) Lag and difference features — value today vs value 7 days ago (captures week-over-week change), percentage change over 30 days (captures monthly trends). (3) Velocity and acceleration — first and second derivatives of the time series. Velocity = rate of change (is it going up?), acceleration = is the rate of change increasing? (4) Cyclical features — hour of day and day of week encoded with sin/cos (so hour 23 is close to hour 0). Key engineering practices: point-in-time correctness (features computed from data BEFORE the prediction timestamp), incremental computation (update by adding new data and dropping expired data, not full recompute), and multi-window ratios (event_count_1h / avg_event_count_1h_over_30d tells you if current activity is unusual relative to the entity's baseline). For fraud detection specifically, the ratio features (current velocity vs historical baseline) are often the most predictive signals."

---

## Common Mistakes

1. **Not using multiple time windows** — Only computing "purchase_count_30d." Miss that the user made 5 purchases in the last hour (burst activity, fraud signal). Solution: always compute at multiple scales (1h, 24h, 7d, 30d) and derive ratio features (current_rate / historical_rate).

2. **Temporal leakage in lag features** — Computing lag features using the wrong time reference. Training uses "current date" instead of "prediction timestamp" for the training example. Solution: always compute lags relative to the prediction timestamp for that training example, not relative to the pipeline run time.

3. **Not handling irregular time series** — Assuming events arrive at regular intervals. Computing "velocity" by dividing value difference by 1 (assuming daily). But events 2 days apart → velocity is halved. Solution: divide by actual time elapsed between events, not assumed fixed interval.

4. **Integer overflow on long windows** — Sum of transactions over 90 days for a high-volume merchant → integer overflow or precision loss. Solution: use appropriate numeric types (INT64 for counts, FLOAT64 for sums). Validate that computed values are within expected ranges.

5. **Expensive full recomputation** — Every hour, recompute 30-day rolling mean by reading all 30 days of data for all entities. O(entities × 30 days × events_per_day). Solution: incremental computation — maintain running sum and count, add new data, subtract expired data. Or pre-aggregate to daily summaries first.

---

## Key Takeaways

- Multiple time windows: 1h, 24h, 7d, 30d, 90d — each captures different temporal signal
- Lag features: compare current to past (captures change direction and magnitude)
- Rolling statistics: mean, std, min, max over windows (captures trend and volatility)
- Velocity and acceleration: first/second derivatives (speed and direction of change)
- Cyclical encoding: sin/cos for hour, day_of_week, month (captures periodicity correctly)
- Ratio features: current_rate / historical_baseline (detects anomalous behavior)
- Point-in-time correctness: compute relative to prediction timestamp, not current time
- Incremental computation: update windows by adding new / dropping old (not full recompute)
- Irregular time series: divide by actual elapsed time, not assumed fixed interval
- Pre-aggregate: daily summaries first, then compute weekly/monthly features from summaries
