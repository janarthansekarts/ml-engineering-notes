# Time Series ML

## The Problem / Why This Matters

Time series ML — predicting future values or detecting anomalies in sequential data ordered by time — is one of the most common yet challenging production ML applications. Unlike tabular or image data, time series has temporal dependencies (what happened yesterday matters for today), seasonal patterns (daily, weekly, yearly cycles), trend components (long-term growth or decline), and non-stationarity (the statistical properties change over time). In 2026, production time series ML powers demand forecasting (retail, supply chain), infrastructure monitoring (server metrics, network traffic), financial trading (price prediction, risk), energy management (load forecasting, renewable optimization), and IoT (sensor anomaly detection). The engineering challenges are unique: handling irregular timestamps, managing missing values (sensor gaps), multi-horizon forecasting (predict next hour AND next month), combining many related time series (hierarchical forecasting), and providing uncertainty estimates (confidence intervals). Modern approaches combine statistical methods (ARIMA, exponential smoothing), gradient-boosted trees (LightGBM on engineered temporal features), deep learning (Temporal Fusion Transformer, N-BEATS, PatchTST), and foundation models for time series (TimeGPT, Chronos, Lag-Llama).

---

## The Analogy

Think of time series forecasting like weather prediction:

- **Statistical methods (ARIMA)** = A farmer who predicts weather based on decades of local patterns. "It rained last three Tuesdays, so probably rain next Tuesday." Works for stable, repeating patterns but fails when something unprecedented happens.
- **ML methods (gradient boosting)** = A meteorologist with radar, satellite, barometric data, and historical patterns. Combines many signals systematically. Better at capturing complex relationships.
- **Deep learning (transformers)** = A supercomputer processing global weather data, finding patterns across thousands of locations and decades of history simultaneously. Best for complex systems but needs massive data and compute.
- **Foundation models (TimeGPT/Chronos)** = A pre-trained model that has "seen" millions of time series. Can forecast a new series with minimal data because it learned universal temporal patterns.

---

## Deep Dive

### Time Series Tasks

```yaml
Time_Series_Tasks:
  forecasting:
    what: "Predict future values given past observations"
    horizons:
      short_term: "Next hour/day (operational planning, real-time actions)"
      medium_term: "Next week/month (inventory, staffing, budgeting)"
      long_term: "Next quarter/year (strategic planning, capacity)"
    types:
      point_forecast: "Single predicted value (most likely outcome)"
      probabilistic: "Distribution of possible outcomes (confidence intervals)"
      hierarchical: "Forecast at multiple levels (store → region → country)"
      
  anomaly_detection:
    what: "Identify unusual patterns in time series data"
    types:
      point_anomaly: "Single unusually high/low value"
      contextual: "Normal value but wrong context (high sales on holiday when store is closed)"
      collective: "Sequence of values that together are anomalous"
    applications: "Infrastructure monitoring, fraud, quality control, security"
    
  classification:
    what: "Classify an entire time series (or segment) into categories"
    examples: "ECG classification, activity recognition, vibration analysis"
    
  change_point_detection:
    what: "Detect when the underlying process changes fundamentally"
    examples: "Market regime change, equipment degradation onset, user behavior shift"
```

### Forecasting Approaches

```yaml
Forecasting_Methods:
  statistical:
    arima: "Auto-regressive + moving average. Good for single univariate series."
    exponential_smoothing: "Weighted average of past observations. Handles trend + seasonality."
    prophet: "Facebook/Meta. Handles holidays, changepoints, multiple seasonalities."
    when: "Few series, well-understood patterns, interpretability needed"
    
  gradient_boosted_trees:
    approach: "Engineer temporal features → train XGBoost/LightGBM"
    features:
      - "Lags (value at t-1, t-7, t-30)"
      - "Rolling statistics (mean/std of last 7 days)"
      - "Calendar features (day of week, month, holiday)"
      - "Fourier features (encode seasonality as sin/cos)"
    strengths: "Fast training, handles mixed feature types, good with limited data"
    when: "Tabular features available, moderate number of series"
    
  deep_learning:
    temporal_fusion_transformer:
      what: "Attention-based model for multi-horizon forecasting"
      strengths: "Multi-horizon, handles static + temporal features, interpretable attention"
    nbeats:
      what: "Neural Basis Expansion Analysis"
      strengths: "Purely univariate, no feature engineering needed"
    patchtst:
      what: "Transformer on time series patches (like ViT for time)"
      strengths: "Long-range dependencies, efficient with patches"
    informer:
      what: "Efficient transformer for long sequences"
      strengths: "Handles very long input sequences (1000+ timesteps)"
    when: "Many related time series, long histories, complex interactions"
    
  foundation_models:
    timegpt:
      what: "Pre-trained on 100B+ data points, zero-shot forecasting"
      api: "Nixtla TimeGPT API"
    chronos:
      what: "Amazon's time series foundation model (T5-based)"
      approach: "Tokenize real values → language model forecasting"
    lag_llama:
      what: "Open-source time series foundation model"
      benefit: "Fine-tunable, runs locally, probabilistic forecasts"
    when: "Limited historical data, cold start, quick prototyping"
```

### Implementation

```python
# Production time series ML pipeline

"""
Time series forecasting and anomaly detection system.
Combines feature engineering, model training, and production serving.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesData:
    """Time series data container."""
    series_id: str
    timestamps: List[float]  # Unix timestamps
    values: List[float]
    metadata: Dict = None  # Static features (location, category, etc.)


@dataclass
class ForecastResult:
    """Forecast output with uncertainty."""
    series_id: str
    timestamps: List[float]  # Future timestamps
    predictions: List[float]  # Point forecasts
    lower_bound: List[float]  # Confidence interval lower
    upper_bound: List[float]  # Confidence interval upper
    model_used: str
    metrics: Dict = None


class TimeSeriesFeatureEngineering:
    """
    Engineer temporal features for gradient-boosted tree forecasting.
    
    Transforms raw time series into tabular features suitable for
    XGBoost/LightGBM training.
    """
    
    def __init__(self, lags: List[int] = None, rolling_windows: List[int] = None):
        self.lags = lags or [1, 7, 14, 28, 30, 60, 90, 365]
        self.rolling_windows = rolling_windows or [7, 14, 30, 60, 90]
    
    def extract_features(
        self, series: TimeSeriesData, target_timestamp: float
    ) -> Dict[str, float]:
        """
        Extract features for a specific prediction timestamp.
        
        Feature categories:
        1. Lag features (past values)
        2. Rolling statistics (mean, std, min, max over windows)
        3. Calendar features (day of week, month, hour, holiday)
        4. Trend features (slope, momentum)
        5. Fourier features (encode seasonality)
        """
        features = {}
        values = np.array(series.values)
        timestamps = np.array(series.timestamps)
        
        # Most recent value index before target timestamp
        valid_idx = np.where(timestamps < target_timestamp)[0]
        if len(valid_idx) == 0:
            return features  # No history available
        
        last_idx = valid_idx[-1]
        
        # 1. Lag features
        for lag in self.lags:
            lag_idx = last_idx - lag + 1
            if lag_idx >= 0:
                features[f"lag_{lag}"] = values[lag_idx]
            else:
                features[f"lag_{lag}"] = np.nan
        
        # 2. Rolling statistics
        for window in self.rolling_windows:
            start_idx = max(0, last_idx - window + 1)
            window_values = values[start_idx:last_idx + 1]
            
            if len(window_values) > 0:
                features[f"rolling_mean_{window}"] = float(np.mean(window_values))
                features[f"rolling_std_{window}"] = float(np.std(window_values))
                features[f"rolling_min_{window}"] = float(np.min(window_values))
                features[f"rolling_max_{window}"] = float(np.max(window_values))
            else:
                for stat in ["mean", "std", "min", "max"]:
                    features[f"rolling_{stat}_{window}"] = np.nan
        
        # 3. Calendar features
        import datetime
        dt = datetime.datetime.fromtimestamp(target_timestamp)
        features["day_of_week"] = dt.weekday()
        features["month"] = dt.month
        features["hour"] = dt.hour
        features["day_of_month"] = dt.day
        features["week_of_year"] = dt.isocalendar()[1]
        features["is_weekend"] = float(dt.weekday() >= 5)
        
        # 4. Trend features
        if last_idx >= 7:
            recent = values[last_idx - 6:last_idx + 1]
            features["trend_7d"] = float(recent[-1] - recent[0])
            features["momentum"] = float(np.mean(np.diff(recent)))
        
        # 5. Fourier features (capture seasonality)
        day_progress = (dt.hour * 3600 + dt.minute * 60) / 86400.0
        features["sin_daily"] = float(np.sin(2 * np.pi * day_progress))
        features["cos_daily"] = float(np.cos(2 * np.pi * day_progress))
        
        week_progress = dt.weekday() / 7.0
        features["sin_weekly"] = float(np.sin(2 * np.pi * week_progress))
        features["cos_weekly"] = float(np.cos(2 * np.pi * week_progress))
        
        # 6. Static metadata features
        if series.metadata:
            for key, value in series.metadata.items():
                features[f"meta_{key}"] = value
        
        return features


class TimeSeriesAnomalyDetector:
    """
    Anomaly detection for time series data.
    
    Approaches:
    1. Statistical: z-score, IQR on residuals after seasonal decomposition
    2. Forecast-based: predict expected value, flag large deviations
    3. Isolation forest: unsupervised anomaly scoring on temporal features
    """
    
    def __init__(
        self,
        forecaster,
        sensitivity: float = 3.0,
        min_history: int = 30,
    ):
        self.forecaster = forecaster
        self.sensitivity = sensitivity  # Standard deviations for threshold
        self.min_history = min_history
    
    def detect(
        self, series: TimeSeriesData, current_value: float, current_timestamp: float
    ) -> Dict:
        """
        Detect if current value is anomalous.
        
        Method: forecast expected value → compute residual → threshold.
        """
        values = np.array(series.values)
        
        if len(values) < self.min_history:
            return {"is_anomaly": False, "reason": "insufficient_history"}
        
        # Forecast what value SHOULD be
        expected = self.forecaster.predict_next(series)
        
        # Compute historical residuals (how much do predictions typically deviate?)
        historical_residuals = self._compute_historical_residuals(series)
        residual_std = np.std(historical_residuals)
        
        # Current residual
        current_residual = abs(current_value - expected)
        
        # Z-score of current residual
        if residual_std > 0:
            z_score = current_residual / residual_std
        else:
            z_score = 0.0
        
        is_anomaly = z_score > self.sensitivity
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": float(z_score),
            "threshold": self.sensitivity,
            "expected_value": float(expected),
            "actual_value": current_value,
            "deviation": float(current_residual),
            "expected_range": (
                float(expected - self.sensitivity * residual_std),
                float(expected + self.sensitivity * residual_std)
            ),
        }
    
    def _compute_historical_residuals(self, series: TimeSeriesData) -> np.ndarray:
        """Compute residuals from historical predictions."""
        values = np.array(series.values)
        # Simple approach: residuals from moving average
        window = min(7, len(values) // 2)
        if window < 2:
            return np.array([0.0])
        
        # Moving average as baseline prediction
        moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
        residuals = values[window-1:] - moving_avg
        
        return residuals


class HierarchicalForecasting:
    """
    Hierarchical forecasting: consistent forecasts at multiple aggregation levels.
    
    Example: Retail demand
    - Level 0: Total company demand
    - Level 1: Region demand (North, South, East, West)
    - Level 2: Store demand (100 stores)
    - Level 3: Product-store demand (10,000 combinations)
    
    Challenge: Independent forecasts at each level don't add up!
    Store forecasts might sum to more than region forecast.
    
    Solution: Reconciliation — adjust forecasts to be coherent across levels.
    """
    
    def __init__(self, hierarchy: Dict, reconciliation: str = "mint"):
        """
        Args:
            hierarchy: Tree structure defining aggregation relationships
            reconciliation: Method to make forecasts coherent
                          "bottom_up": aggregate from bottom level
                          "top_down": distribute from top level
                          "mint": optimal combination (minimum trace)
        """
        self.hierarchy = hierarchy
        self.reconciliation = reconciliation
    
    def forecast_hierarchical(
        self, series_dict: Dict[str, TimeSeriesData], horizon: int
    ) -> Dict[str, ForecastResult]:
        """
        Generate coherent forecasts across all hierarchy levels.
        
        Steps:
        1. Generate base forecasts at each level independently
        2. Reconcile to ensure coherence (forecasts add up correctly)
        """
        # Step 1: Base forecasts (independent per series)
        base_forecasts = {}
        for series_id, series in series_dict.items():
            forecast = self._base_forecast(series, horizon)
            base_forecasts[series_id] = forecast
        
        # Step 2: Reconciliation
        reconciled = self._reconcile(base_forecasts)
        
        return reconciled
    
    def _base_forecast(self, series: TimeSeriesData, horizon: int) -> List[float]:
        """Generate base forecast for a single series."""
        # In production: use best model for this series
        # Simplified: exponential smoothing
        values = np.array(series.values)
        alpha = 0.3
        level = values[-1]
        forecasts = []
        for _ in range(horizon):
            forecasts.append(level)
            # Level stays constant (simplification)
        return forecasts
    
    def _reconcile(self, base_forecasts: Dict) -> Dict[str, ForecastResult]:
        """
        Reconcile forecasts to be coherent across hierarchy.
        
        In production: use statsforecast or hierarchicalforecast library.
        """
        # Bottom-up reconciliation (simplest)
        reconciled = {}
        for series_id, forecast_values in base_forecasts.items():
            reconciled[series_id] = ForecastResult(
                series_id=series_id,
                timestamps=[],  # Would be filled with actual timestamps
                predictions=forecast_values,
                lower_bound=[v * 0.8 for v in forecast_values],
                upper_bound=[v * 1.2 for v in forecast_values],
                model_used="hierarchical_reconciled",
            )
        return reconciled
```

### Production Time Series Challenges

```yaml
Production_Challenges:
  missing_data:
    problem: "Sensors fail, data collection gaps, irregular timestamps"
    solutions:
      forward_fill: "Use last known value (most common)"
      interpolation: "Linear or spline between known values"
      model_imputation: "Predict missing values from other series"
    engineering: "Track % missing per series, alert on high missing rates"
    
  cold_start:
    problem: "New entity (new product, new store) with no history"
    solutions:
      similar_series: "Find similar entities, use their history as proxy"
      hierarchical: "Use parent-level forecast as starting point"
      foundation_model: "TimeGPT/Chronos can forecast with minimal history"
      
  concept_drift:
    problem: "Patterns change over time (COVID, seasonality shifts, trends)"
    solutions:
      adaptive_windows: "Only train on recent data (last 6 months, not 5 years)"
      drift_detection: "Monitor forecast errors, retrain when errors increase"
      regime_detection: "Detect structural breaks, separate pre/post models"
      
  multiple_seasonalities:
    problem: "Hourly (daily pattern), daily (weekly pattern), weekly (yearly pattern)"
    solutions:
      fourier_features: "Sin/cos features at multiple frequencies"
      prophet: "Handles multiple seasonalities explicitly"
      deep_learning: "Learns multiple periodicities from data (TFT, N-BEATS)"
      
  scale:
    problem: "Millions of time series to forecast (one per product-store)"
    solutions:
      global_model: "Single model trained on all series (share patterns)"
      hierarchical: "Forecast at aggregate level, disaggregate"
      parallel: "Spark/Ray for distributed training on millions of series"
```

---

## How It Works in Practice

### Retail Demand Forecasting

```yaml
Retail_Forecasting:
  scale: "100,000 product-store combinations, daily forecasts"
  horizon: "14 days ahead (for inventory ordering)"
  
  pipeline:
    daily_at_2am:
      step_1: "Extract features: lags, rolling stats, calendar, promotions, weather"
      step_2: "Run LightGBM global model (trained on all series)"
      step_3: "Generate point forecasts + 80% prediction intervals"
      step_4: "Hierarchical reconciliation (product → category → store → region)"
      step_5: "Write forecasts to serving table"
      step_6: "Inventory system reads forecasts for ordering decisions"
      
  features:
    temporal: "Lags (1,7,14,28), rolling mean/std (7,14,28), trend, momentum"
    calendar: "Day of week, month, holiday, school vacation, payday proximity"
    external: "Weather forecast, promotions, competitor actions, local events"
    product: "Category, price, shelf life, substitutability"
    
  model:
    type: "LightGBM (global model across all series)"
    training: "Last 2 years of daily sales data"
    retraining: "Weekly (captures recent pattern shifts)"
    
  accuracy:
    metric: "WAPE (Weighted Absolute Percentage Error)"
    target: "< 25% WAPE at product-store level"
    improvement: "30% better than statistical baselines (ARIMA, ETS)"
```

---

## Interview Tip

> When asked about time series ML: "I approach time series forecasting in production with a model selection strategy based on data characteristics and scale. For moderate scale (< 1000 series): statistical methods (Prophet for multiple seasonalities with holidays, ARIMA for simple univariate). For large scale (100K+ series): LightGBM on engineered temporal features — lags, rolling statistics, calendar features, Fourier components for seasonality. A single global model trained across all series outperforms individual models because it shares patterns. For deep learning (when data is abundant): Temporal Fusion Transformer (TFT) for interpretable multi-horizon forecasting with static covariates. For cold start (new series, minimal history): foundation models like Chronos or TimeGPT — pre-trained on billions of data points, can forecast zero-shot. Key engineering concerns: (1) Feature engineering — lags, rolling windows, calendar features, external signals (weather, promotions). Training features must ONLY use past data (no future leakage). (2) Missing data — forward-fill for short gaps, track missing % per series, alert on high missing rates. (3) Hierarchical reconciliation — independent forecasts at different aggregation levels don't sum correctly. Use MinT (Minimum Trace) reconciliation for coherent forecasts. (4) Uncertainty — always provide prediction intervals, not just point forecasts. Downstream systems need to plan for uncertainty (safety stock = f(forecast uncertainty)). (5) Evaluation — use time-series cross-validation (expanding window, not random split) to avoid data leakage."

---

## Common Mistakes

1. **Data leakage in training** — Using future information in features (e.g., next week's weather as a feature to predict this week's sales). Model performs great in backtesting, terribly in production. Solution: strict temporal splits. Features at time T can only use data from before T.

2. **Random train/test split** — Splitting time series randomly (not temporally). Model sees future data in training, reports inflated accuracy. Solution: time-based split (train on past, test on future). Use expanding-window cross-validation.

3. **Ignoring prediction intervals** — Only providing point forecasts. Downstream systems have no sense of uncertainty. When forecast is wrong, there's no buffer. Solution: always provide confidence intervals. Use quantile regression or conformal prediction.

4. **Not handling missing data explicitly** — NaN values silently dropped or filled with zeros. Model trains on corrupted data. Solution: explicit missing data strategy — forward-fill (most common), track % missing, flag series with > 20% missing for review.

5. **Single model for all horizons** — Using the same model to predict tomorrow AND next month. Short-term and long-term have different dynamics (short = autocorrelation, long = trend + seasonality). Solution: separate models for different horizons, or multi-horizon models (TFT) that handle this natively.

---

## Key Takeaways

- Time series ML: forecasting, anomaly detection, classification, change point detection
- Feature engineering: lags, rolling statistics, calendar features, Fourier (seasonality), external signals
- Model selection: statistical (few series, interpretable) → GBT (scale, tabular features) → deep learning (complex, abundant data) → foundation models (cold start)
- Foundation models (2026): Chronos, TimeGPT, Lag-Llama — zero-shot forecasting on new series
- Anomaly detection: forecast expected value, flag large deviations (z-score on residuals)
- Hierarchical: reconcile forecasts across aggregation levels (product → store → region)
- Data leakage: features must only use past data (strict temporal ordering)
- Prediction intervals: always provide uncertainty estimates (not just point forecasts)
- Missing data: explicit strategy (forward-fill, interpolation, alert on high missing %)
- Scale: global model (single LightGBM on all series) often beats individual models
