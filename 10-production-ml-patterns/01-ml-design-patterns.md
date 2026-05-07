# ML Design Patterns

## The Problem / Why This Matters

Building ML systems from scratch every time is inefficient and error-prone. Just as software engineering has design patterns (singleton, observer, factory), ML engineering has recurring architectural patterns that solve common problems. These patterns represent proven solutions to challenges that every ML team encounters: how to serve predictions at different latency requirements, how to handle data that changes over time, how to combine multiple models, how to retrain without downtime, and how to handle the gap between training and serving. In 2026, with ML systems becoming more complex (multiple models, LLM chains, agent architectures, real-time features), understanding these patterns is essential for building production systems that are maintainable, scalable, and reliable. Choosing the wrong pattern leads to over-engineering simple problems or under-engineering complex ones — both result in systems that are expensive to operate and difficult to maintain. The right pattern depends on your specific constraints: latency requirements, data freshness needs, traffic patterns, model complexity, and operational maturity.

---

## The Analogy

Think of ML design patterns like architectural blueprints for buildings:

- **No patterns** = Every building designed from scratch. Each architect invents new ways to handle plumbing, electricity, and structural support. Some work, many fail. Every project is unique and hard to maintain.
- **Design patterns** = Standard blueprints for common building types. A "restaurant pattern" solves kitchen-to-dining flow. A "hospital pattern" solves patient-to-operating-room flow. You choose the right blueprint for your needs, then customize.

Patterns don't eliminate creativity — they provide a proven foundation so you focus your creativity on what's unique to your problem, not on reinventing basic infrastructure.

---

## Deep Dive

### Pattern Catalog

```yaml
ML_Design_Patterns:
  serving_patterns:
    batch_prediction:
      when: "Latency tolerance > minutes, predictions needed for all entities"
      how: "Pre-compute predictions on schedule, store in database"
      examples: "Daily recommendations, weekly risk scores, nightly fraud scores"
      
    online_prediction:
      when: "Real-time response needed (< 100ms)"
      how: "Model loaded in memory, inference on each request"
      examples: "Search ranking, ad bidding, real-time fraud detection"
      
    streaming_prediction:
      when: "Near-real-time on event streams"
      how: "Model processes events as they arrive (Kafka/Flink)"
      examples: "IoT anomaly detection, real-time personalization, clickstream analysis"
      
    hybrid_serving:
      when: "Some features need pre-computation, some need real-time"
      how: "Batch features + real-time features combined at serving time"
      examples: "Recommendations (batch user embeddings + real-time session context)"
      
  training_patterns:
    retraining_on_schedule:
      when: "Data distribution changes gradually"
      how: "Retrain daily/weekly on latest data, automated pipeline"
      examples: "Most production models — fraud, recommendations, pricing"
      
    continuous_training:
      when: "Data distribution changes rapidly"
      how: "Retrain on every new batch of data (hourly or on-event)"
      examples: "Ad click prediction, trending content ranking"
      
    transfer_learning:
      when: "Limited labeled data, pre-trained model available"
      how: "Start from pre-trained model, fine-tune on your data"
      examples: "NLP tasks with BERT/GPT, vision tasks with ResNet/ViT"
      
    federated_learning:
      when: "Data can't leave source (privacy, regulation)"
      how: "Train on-device, aggregate model updates centrally"
      examples: "Keyboard prediction (mobile), hospital data (HIPAA)"
      
  architecture_patterns:
    model_ensemble:
      when: "Single model not accurate enough, or need robustness"
      how: "Multiple models vote or blend predictions"
      examples: "Random forest, stacking, model committee for critical decisions"
      
    cascade_model:
      when: "Most inputs are easy, few are hard"
      how: "Cheap model first → if uncertain → expensive model"
      examples: "Spam filter (rule → ML → human review)"
      
    two_tower:
      when: "Need to match items to queries (retrieval + ranking)"
      how: "Encode query and items separately, compute similarity"
      examples: "Search, recommendations, ad matching"
      
    champion_challenger:
      when: "Deploying new model safely"
      how: "Current model (champion) serves most traffic, new model (challenger) gets small %"
      examples: "Any model deployment (canary pattern for ML)"
```

### Pattern Selection Guide

```yaml
Selection_Guide:
  by_latency_requirement:
    sub_10ms: "Pre-computed batch predictions (read from cache/DB)"
    10_100ms: "Online prediction (model in memory, optimized inference)"
    100ms_1s: "Online prediction with feature retrieval"
    1_60s: "LLM generation, complex multi-model pipelines"
    
  by_data_freshness:
    stale_ok: "Batch prediction (compute nightly, serve from cache)"
    minutes_old: "Near-real-time (streaming prediction, feature store with low-latency materialization)"
    seconds_old: "Online prediction with real-time features"
    
  by_traffic_pattern:
    uniform: "Standard auto-scaling based on QPS"
    bursty: "Pre-warm instances, aggressive scaling, batch processing during bursts"
    predictable_peaks: "Scheduled scaling (e.g., scale up before Black Friday)"
    
  by_model_complexity:
    simple_model: "Online serving in any framework (scikit-learn, XGBoost)"
    medium_model: "Dedicated inference server (TensorFlow Serving, Triton)"
    large_model: "GPU serving, batching, model parallelism (vLLM, TGI)"
    multi_model: "Orchestration layer (pipeline of models, routing)"
```

### Common Pattern Implementations

```python
# ML Design Pattern implementations

"""
Common production ML patterns implemented in Python.
These show the structural approach — production implementations
would include error handling, monitoring, and scaling.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import time


# === Cascade Pattern ===

class CascadeModel:
    """
    Cascade pattern: cheap model first, expensive model for uncertain cases.
    
    Saves compute: 80% of requests handled by fast model.
    Only hard cases go to expensive model.
    
    Example: Spam detection
    - Level 1: Rule-based filter (< 1ms) → catches obvious spam
    - Level 2: Lightweight ML model (5ms) → catches most remaining
    - Level 3: Large model (50ms) → handles ambiguous cases
    - Level 4: Human review → handles model-uncertain cases
    """
    
    def __init__(self, models: List, confidence_thresholds: List[float]):
        """
        Args:
            models: List of models from cheapest to most expensive.
            confidence_thresholds: Confidence above which we trust each model.
        """
        self.models = models
        self.thresholds = confidence_thresholds
    
    def predict(self, input_data: Any) -> Dict:
        """
        Run cascade: try cheap models first, escalate if uncertain.
        """
        for i, (model, threshold) in enumerate(zip(self.models, self.thresholds)):
            prediction, confidence = model.predict_with_confidence(input_data)
            
            if confidence >= threshold:
                return {
                    "prediction": prediction,
                    "confidence": confidence,
                    "model_level": i,
                    "model_name": model.name,
                }
        
        # All models uncertain — use last model's prediction
        return {
            "prediction": prediction,
            "confidence": confidence,
            "model_level": len(self.models) - 1,
            "model_name": self.models[-1].name,
            "low_confidence": True,
        }


# === Two-Tower Pattern ===

class TwoTowerRetrieval:
    """
    Two-tower pattern for retrieval + ranking.
    
    Separate encoders for queries and items:
    - Query tower: encode user query into embedding
    - Item tower: pre-encode all items into embeddings (offline)
    - At serving time: find nearest items to query (ANN search)
    
    Used in: search, recommendations, ad matching.
    """
    
    def __init__(self, query_encoder, item_encoder, vector_index):
        self.query_encoder = query_encoder
        self.item_encoder = item_encoder
        self.index = vector_index  # Pre-built ANN index of item embeddings
    
    def retrieve(self, query: str, top_k: int = 100) -> List[Dict]:
        """
        Retrieve top-k items for a query.
        
        Steps:
        1. Encode query → embedding
        2. ANN search in pre-computed item embeddings
        3. Return top-k candidates (for re-ranking)
        """
        # Encode query (real-time, ~5ms)
        query_embedding = self.query_encoder.encode(query)
        
        # ANN search in item index (pre-computed, ~1ms for millions of items)
        candidates = self.index.search(query_embedding, top_k=top_k)
        
        return candidates
    
    def rank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Re-rank candidates with a more expensive cross-encoder.
        
        The two-tower retrieves broadly (fast but approximate).
        The ranker scores precisely (slow but accurate).
        """
        # Cross-encoder scoring (more expensive, ~50ms for 100 candidates)
        scored = []
        for candidate in candidates:
            score = self.ranker.score(query, candidate["content"])
            scored.append({**candidate, "relevance_score": score})
        
        return sorted(scored, key=lambda x: -x["relevance_score"])


# === Champion-Challenger Pattern ===

class ChampionChallenger:
    """
    Champion-challenger: safe deployment of new models.
    
    Champion (current production model) serves most traffic.
    Challenger (new candidate model) serves small percentage.
    Compare metrics → if challenger wins → promote to champion.
    
    This IS A/B testing for models, implemented as a serving pattern.
    """
    
    def __init__(
        self,
        champion_model,
        challenger_model,
        challenger_traffic_pct: float = 0.05  # 5% to challenger
    ):
        self.champion = champion_model
        self.challenger = challenger_model
        self.challenger_pct = challenger_traffic_pct
    
    def predict(self, input_data: Any, user_id: str) -> Dict:
        """
        Route request to champion or challenger based on user assignment.
        """
        import hashlib
        bucket = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
        
        if bucket < self.challenger_pct * 100:
            # Challenger
            prediction = self.challenger.predict(input_data)
            return {"prediction": prediction, "model": "challenger", "version": self.challenger.version}
        else:
            # Champion
            prediction = self.champion.predict(input_data)
            return {"prediction": prediction, "model": "champion", "version": self.champion.version}


# === Feature Store Pattern ===

class FeatureStorePattern:
    """
    Feature Store pattern: separate feature computation from model serving.
    
    Problem: features computed differently in training vs. serving (skew).
    Solution: single source of truth for features (feature store).
    
    - Training: read historical features from feature store (point-in-time)
    - Serving: read latest features from feature store (real-time)
    - Same computation logic → no training-serving skew.
    """
    
    def __init__(self, feature_store_client, model):
        self.feature_store = feature_store_client
        self.model = model
    
    def predict(self, entity_id: str, real_time_features: Dict = None) -> Dict:
        """
        Get prediction using feature store.
        
        Combines:
        - Pre-computed features from feature store (batch-computed)
        - Real-time features from request (computed on-the-fly)
        """
        # Get pre-computed features (from feature store)
        stored_features = self.feature_store.get_features(
            entity_id=entity_id,
            feature_names=["credit_score", "avg_transaction_30d", "account_age"]
        )
        
        # Combine with real-time features
        all_features = {**stored_features}
        if real_time_features:
            all_features.update(real_time_features)
        
        # Predict
        prediction = self.model.predict(all_features)
        
        return {
            "prediction": prediction,
            "features_used": list(all_features.keys()),
            "feature_freshness": stored_features.get("_timestamp"),
        }
```

### Anti-Patterns

```yaml
Anti_Patterns:
  train_serve_skew:
    what: "Different feature computation in training vs. serving"
    cause: "Training in notebook (pandas), serving in Java/Go (custom code)"
    fix: "Feature store pattern — single source of truth for features"
    
  big_model_for_simple_task:
    what: "Using a 70B parameter LLM for classification that logistic regression handles"
    cause: "LLM hype — reaching for ChatGPT when simpler model suffices"
    fix: "Start simple (rules → classical ML → deep learning → LLM). Only escalate if simpler fails."
    
  no_fallback:
    what: "System crashes when model is unavailable (no graceful degradation)"
    cause: "Only designed the happy path"
    fix: "Always have a fallback: cached prediction, rule-based backup, graceful error"
    
  monolithic_pipeline:
    what: "Training, feature engineering, serving, monitoring all in one codebase"
    cause: "Started as notebook, grew into production system"
    fix: "Separate concerns: feature pipeline, training pipeline, serving service, monitoring"
    
  over_engineering:
    what: "Building ML platform for 100 models when you have 2"
    cause: "Premature optimization based on anticipated scale"
    fix: "Build what you need now. Add infrastructure when pain points appear."
```

---

## How It Works in Practice

### Pattern Combinations

```yaml
Typical_Production_Stack:
  recommendation_system:
    patterns:
      - "Two-tower (retrieval): encode user + items, ANN search"
      - "Cascade (ranking): lightweight ranker → heavy re-ranker"
      - "Batch (embeddings): pre-compute item embeddings nightly"
      - "Feature store: user features (batch) + session features (real-time)"
      - "Champion-challenger: new model gets 5% traffic"
      
  fraud_detection:
    patterns:
      - "Online prediction (real-time): every transaction scored"
      - "Feature store: historical features + real-time aggregates"
      - "Cascade: rules → ML → human review"
      - "Continuous training: retrain on each day's labeled data"
      - "Ensemble: multiple models vote (robustness against adversaries)"
```

---

## Interview Tip

> When asked about ML design patterns: "I choose patterns based on three constraints: latency, data freshness, and model complexity. For latency: if sub-10ms needed, I pre-compute predictions in batch and serve from cache. For 10-100ms, online serving with model in memory. For longer (LLMs), async generation with streaming. For cost efficiency, I use the cascade pattern: cheap model handles 80% of traffic (obvious cases), expensive model only for uncertain cases. This typically reduces compute cost by 5-10x. For retrieval systems (search, recommendations), I use the two-tower pattern: encode queries and items separately, use ANN (Approximate Nearest Neighbor) search for fast retrieval (~1ms for millions of items), then re-rank top candidates with a more expensive cross-encoder. For safe deployment, I use champion-challenger: current model serves 95% of traffic, new model gets 5%, compare metrics with statistical significance before promoting. The most important anti-pattern to avoid is training-serving skew — features computed differently in training and production. I solve this with a feature store: one source of truth for feature computation, used by both training (historical) and serving (real-time)."

---

## Common Mistakes

1. **Choosing real-time when batch suffices** — Building a real-time inference endpoint (complex, expensive, high-ops) when predictions are only consumed once daily. Solution: start with batch. Only move to real-time when latency requirement demands it.

2. **Using LLM when classical ML works** — Deploying GPT-5 for binary classification that XGBoost handles at 1% of the cost and 100x the speed. Solution: start with the simplest model that meets requirements. Escalate only when simpler approaches demonstrably fail.

3. **No cascade for heterogeneous difficulty** — All requests go to the expensive model. 80% are trivial (could be handled by rules or a tiny model). Solution: cascade pattern — cheap filter → moderate model → expensive model. Most requests handled cheaply.

4. **Skipping the feature store** — Compute features in training notebook, rewrite in production service. Eventually they diverge. Solution: feature store from day one if you have > 1 model or > 10 features.

5. **Building everything before having a single model in production** — Spending 6 months building an ML platform before deploying one model. Solution: get one model in production simply. Then invest in platform when pain points emerge from real operational experience.

---

## Key Takeaways

- ML design patterns: proven solutions to recurring production ML challenges
- Serving: batch (pre-compute), online (real-time), streaming (event-driven), hybrid
- Cascade: cheap model first, expensive model for uncertain cases (5-10x cost savings)
- Two-tower: separate query/item encoding for fast retrieval + expensive re-ranking
- Champion-challenger: safe deployment (new model gets small % traffic, compare metrics)
- Feature store: single source of truth for features (eliminates training-serving skew)
- Pattern selection: driven by latency requirements, data freshness needs, and model complexity
- Anti-patterns: training-serving skew, LLM for simple tasks, no fallback, over-engineering
- Start simple: rules → classical ML → deep learning → LLM. Only escalate when simpler fails
- Combine patterns: real systems use multiple patterns together (two-tower + cascade + feature store)
