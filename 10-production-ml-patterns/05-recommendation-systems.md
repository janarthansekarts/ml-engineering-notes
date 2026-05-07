# Recommendation Systems

## The Problem / Why This Matters

Recommendation systems are the most commercially impactful application of ML — they drive 35% of Amazon's revenue, 80% of Netflix views, and dominate engagement across social media, music streaming, and e-commerce. Building a production recommendation system isn't just training a model that predicts ratings — it's a complex engineering system involving retrieval (finding candidates from millions of items), ranking (scoring candidates), real-time personalization (adapting to current session), diversity and freshness (avoiding filter bubbles), cold start handling (new users/items), and serving at massive scale (millions of users, millisecond latency). In 2026, recommendation systems incorporate LLM (Large Language Model) embeddings, multi-modal signals (text + images + video), real-time session context, and conversational interfaces. The engineering patterns — two-tower retrieval, multi-stage ranking, feature stores for user/item signals, and real-time feature computation — are foundational ML engineering skills that transfer to search, ads, and any system that matches entities.

---

## The Analogy

Think of a recommendation system like a personal shopping assistant in a massive department store:

- **Retrieval stage** = The assistant narrows millions of products down to a few hundred relevant ones. They don't examine every product in detail — they quickly filter by department, category, and what they know about your style.
- **Ranking stage** = From those few hundred, the assistant carefully evaluates each one considering your specific preferences, budget, occasion, and what's on sale. They arrange the top 20 in order.
- **Re-ranking/business logic** = The assistant makes final adjustments — ensuring variety (not all blue shirts), removing out-of-stock items, boosting new arrivals, and mixing familiar brands with discoveries.

Without the retrieval stage, the assistant would spend hours examining every product. Without ranking, they'd just show random relevant items. The multi-stage pipeline is what makes recommendations both relevant and fast.

---

## Deep Dive

### Multi-Stage Architecture

```yaml
Recommendation_Pipeline:
  stage_1_candidate_generation:
    purpose: "Reduce millions of items to hundreds of candidates"
    latency: "< 10ms"
    methods:
      two_tower_embedding:
        what: "Encode user and items into same embedding space"
        retrieval: "ANN (Approximate Nearest Neighbor) search (FAISS, ScaNN, Pinecone)"
        scale: "Billions of items, sub-10ms retrieval"
        
      collaborative_filtering:
        what: "Users who liked similar items → suggest their other likes"
        implementation: "Matrix factorization, item-item similarity"
        
      content_based:
        what: "Items similar to what user previously engaged with"
        features: "Item text embeddings, category, attributes"
        
      popularity_based:
        what: "Trending items, new arrivals, bestsellers"
        use: "Cold start fallback, diversity injection"
        
    output: "200-1000 candidates from multiple sources"
    
  stage_2_ranking:
    purpose: "Score each candidate with user-specific features"
    latency: "< 30ms for hundreds of candidates"
    model: "Deep neural network (DNN) or gradient-boosted tree (GBT)"
    features:
      user_features: "Demographics, preferences, history embedding"
      item_features: "Category, price, popularity, freshness"
      context_features: "Time of day, device, location, session history"
      cross_features: "User-item interaction history, affinity scores"
    output: "Scored and ranked candidates"
    
  stage_3_reranking:
    purpose: "Apply business rules, diversity, freshness"
    operations:
      diversity: "Ensure variety (not all same category/brand)"
      freshness: "Boost recently added items"
      business_rules: "Promoted items, inventory constraints, margin optimization"
      exploration: "Include some uncertain items (explore vs. exploit)"
      filtering: "Remove already-seen, out-of-stock, blocked items"
    output: "Final ordered list (top 10-50 items shown to user)"
```

### Implementation

```python
# Production recommendation system implementation

"""
Multi-stage recommendation system:
1. Candidate retrieval (two-tower + multiple sources)
2. Ranking (feature-rich model)
3. Re-ranking (diversity, business rules)
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class RecommendationRequest:
    """User recommendation request."""
    user_id: str
    context: Dict  # device, time, session history
    num_items: int = 20
    exclude_items: List[str] = None


@dataclass
class ScoredItem:
    """Item with recommendation score."""
    item_id: str
    score: float
    source: str  # Which retrieval source produced this candidate
    features: Dict = None


class RecommendationSystem:
    """
    Production recommendation system with multi-stage pipeline.
    """
    
    def __init__(
        self,
        retrieval_sources: List["RetrievalSource"],
        ranking_model,
        feature_store,
        diversity_config: Dict = None,
    ):
        self.retrieval_sources = retrieval_sources
        self.ranking_model = ranking_model
        self.feature_store = feature_store
        self.diversity_config = diversity_config or {"category_max_pct": 0.4}
    
    def recommend(self, request: RecommendationRequest) -> List[ScoredItem]:
        """
        Generate recommendations through multi-stage pipeline.
        """
        start = time.perf_counter()
        
        # Stage 1: Candidate Retrieval (multiple sources)
        candidates = self._retrieve_candidates(request)
        retrieval_time = time.perf_counter() - start
        
        # Stage 2: Ranking
        ranked = self._rank_candidates(candidates, request)
        ranking_time = time.perf_counter() - start - retrieval_time
        
        # Stage 3: Re-ranking (diversity, business rules)
        final = self._rerank(ranked, request)
        rerank_time = time.perf_counter() - start - retrieval_time - ranking_time
        
        total_time = (time.perf_counter() - start) * 1000
        logger.info(
            f"Recommendation for {request.user_id}: "
            f"retrieval={retrieval_time*1000:.1f}ms, "
            f"ranking={ranking_time*1000:.1f}ms, "
            f"rerank={rerank_time*1000:.1f}ms, "
            f"total={total_time:.1f}ms, "
            f"candidates={len(candidates)}, final={len(final)}"
        )
        
        return final[:request.num_items]
    
    def _retrieve_candidates(self, request: RecommendationRequest) -> List[ScoredItem]:
        """
        Retrieve candidates from multiple sources.
        
        Merge and deduplicate. Each source contributes different signal:
        - Embedding similarity (personalized)
        - Collaborative filtering (social signal)
        - Popularity (safe fallback)
        - Content-based (attribute match)
        """
        all_candidates = {}  # item_id → ScoredItem (deduplicate)
        
        for source in self.retrieval_sources:
            try:
                source_candidates = source.retrieve(
                    user_id=request.user_id,
                    context=request.context,
                    num_candidates=200
                )
                for item in source_candidates:
                    if item.item_id not in all_candidates:
                        all_candidates[item.item_id] = item
                    else:
                        # Item from multiple sources → boost score
                        existing = all_candidates[item.item_id]
                        existing.score = max(existing.score, item.score)
                        
            except Exception as e:
                logger.warning(f"Retrieval source {source.name} failed: {e}")
                # Continue with other sources (graceful degradation)
        
        # Filter excluded items
        if request.exclude_items:
            exclude_set = set(request.exclude_items)
            all_candidates = {
                k: v for k, v in all_candidates.items()
                if k not in exclude_set
            }
        
        return list(all_candidates.values())
    
    def _rank_candidates(
        self, candidates: List[ScoredItem], request: RecommendationRequest
    ) -> List[ScoredItem]:
        """
        Score each candidate with ranking model.
        
        Features:
        - User features (from feature store)
        - Item features (from feature store)
        - Context features (from request)
        - Cross features (user-item interaction history)
        """
        # Get user features
        user_features = self.feature_store.get_user_features(request.user_id)
        
        # Get item features (batched for efficiency)
        item_ids = [c.item_id for c in candidates]
        item_features_batch = self.feature_store.get_item_features_batch(item_ids)
        
        # Score each candidate
        scored = []
        for candidate in candidates:
            item_features = item_features_batch.get(candidate.item_id, {})
            
            # Combine all features for ranking model
            features = {
                **user_features,
                **item_features,
                "context_hour": request.context.get("hour", 12),
                "context_device": request.context.get("device", "mobile"),
                "retrieval_score": candidate.score,
                "retrieval_source": candidate.source,
            }
            
            # Ranking model predicts engagement probability
            rank_score = self.ranking_model.predict(features)
            candidate.score = rank_score
            candidate.features = features
            scored.append(candidate)
        
        # Sort by ranking score
        scored.sort(key=lambda x: -x.score)
        return scored
    
    def _rerank(
        self, ranked: List[ScoredItem], request: RecommendationRequest
    ) -> List[ScoredItem]:
        """
        Re-rank for diversity, business rules, and exploration.
        
        Ensures final list isn't all same category/brand.
        Injects exploration items for learning.
        """
        final = []
        category_counts = {}
        max_pct = self.diversity_config.get("category_max_pct", 0.4)
        target_size = request.num_items * 2  # Get 2x to have buffer
        
        for item in ranked:
            category = item.features.get("category", "unknown") if item.features else "unknown"
            
            # Check diversity constraint
            current_pct = category_counts.get(category, 0) / max(len(final), 1)
            if current_pct >= max_pct and len(final) > 5:
                continue  # Skip: too many from this category
            
            final.append(item)
            category_counts[category] = category_counts.get(category, 0) + 1
            
            if len(final) >= target_size:
                break
        
        return final


class TwoTowerRetrieval:
    """
    Two-tower (dual encoder) retrieval.
    
    Architecture:
    - User tower: encode user features → user embedding (d=128)
    - Item tower: encode item features → item embedding (d=128)
    - Similarity: dot product or cosine similarity
    
    At serving time:
    - Item embeddings pre-computed and indexed (ANN)
    - User embedding computed online
    - ANN search for nearest items
    """
    
    def __init__(self, user_encoder, item_index, name="two_tower"):
        self.user_encoder = user_encoder
        self.item_index = item_index  # ANN index (FAISS, ScaNN)
        self.name = name
    
    def retrieve(
        self, user_id: str, context: Dict, num_candidates: int = 200
    ) -> List[ScoredItem]:
        """Retrieve similar items using embedding similarity."""
        # Encode user into embedding (online)
        user_embedding = self.user_encoder.encode(user_id, context)
        
        # ANN search (pre-computed item embeddings)
        item_ids, scores = self.item_index.search(
            query_vector=user_embedding,
            top_k=num_candidates
        )
        
        return [
            ScoredItem(item_id=iid, score=score, source=self.name)
            for iid, score in zip(item_ids, scores)
        ]


class ColdStartHandler:
    """
    Handle cold start: new users or new items with no interaction history.
    
    Strategies:
    - New user: use popularity-based recs + content-based on provided preferences
    - New item: use content similarity to existing items + boost for exploration
    - Warm-up: transition from cold start to personalized as interactions accumulate
    """
    
    def __init__(self, popularity_source, content_source, min_interactions: int = 5):
        self.popularity_source = popularity_source
        self.content_source = content_source
        self.min_interactions = min_interactions
    
    def get_recommendations(
        self, user_id: str, interaction_count: int, preferences: Dict = None
    ) -> Tuple[List[ScoredItem], str]:
        """
        Get recommendations with cold start handling.
        
        Returns: (items, strategy_used)
        """
        if interaction_count == 0:
            # Brand new user: popularity + stated preferences
            items = self.popularity_source.get_popular(limit=50)
            if preferences:
                items = self._filter_by_preferences(items, preferences)
            return items, "cold_start_popularity"
        
        elif interaction_count < self.min_interactions:
            # Few interactions: blend popularity with content-based
            popular = self.popularity_source.get_popular(limit=25)
            content = self.content_source.get_similar_to_history(
                user_id=user_id, limit=25
            )
            items = self._blend(popular, content, content_weight=interaction_count / self.min_interactions)
            return items, "cold_start_blended"
        
        else:
            # Enough interactions: use full personalized pipeline
            return None, "personalized"  # Caller uses main pipeline
    
    def _filter_by_preferences(self, items, preferences):
        """Filter items matching stated preferences."""
        # Example: user said they like "sci-fi" and "action"
        preferred_categories = preferences.get("categories", [])
        if not preferred_categories:
            return items
        return [i for i in items if i.features.get("category") in preferred_categories] or items
    
    def _blend(self, popular, content, content_weight):
        """Blend popular and content-based with weight."""
        # Simple interleaving with weight
        blended = []
        p_idx, c_idx = 0, 0
        while len(blended) < len(popular) + len(content):
            if np.random.random() < content_weight and c_idx < len(content):
                blended.append(content[c_idx])
                c_idx += 1
            elif p_idx < len(popular):
                blended.append(popular[p_idx])
                p_idx += 1
            else:
                break
        return blended
```

### Real-Time Session Personalization

```yaml
Session_Personalization:
  what: "Adapt recommendations based on CURRENT session (not just history)"
  
  signals:
    clicked_items: "User clicked shoes → show more shoes"
    search_queries: "User searched 'birthday gift' → gift-related items"
    time_spent: "User spent 30s on an item → strong interest signal"
    cart_additions: "User added item to cart → similar/complementary items"
    scroll_depth: "User scrolled past 50 items → show different category"
    
  implementation:
    approach: "Streaming features computed from session events"
    feature_store: "Session features materialized in real-time (Redis)"
    update_frequency: "After each user interaction (click, view, add-to-cart)"
    
  architecture:
    event_stream: "User events → Kafka → session feature computation → Redis"
    serving: "Ranking model reads session features from Redis at prediction time"
    latency: "Feature updated within 1 second of user action"
```

---

## How It Works in Practice

### Netflix-Scale Recommendation Pipeline

```yaml
Netflix_Scale:
  users: "200M+ subscribers"
  items: "50,000+ titles (movies, shows, episodes)"
  
  pipeline:
    retrieval:
      - "Collaborative filtering: users with similar watch history"
      - "Content-based: similar genres, actors, directors (text + image embeddings)"
      - "Trending: what's popular in user's region today"
      - "Continue watching: unfinished content"
      candidates: "~2000 per user"
    
    ranking:
      model: "Deep neural network (wide & deep architecture)"
      features: "800+ features (user, item, context, cross)"
      prediction: "P(user watches > 70% of title)"
      
    personalization:
      artwork: "Different thumbnail selected per user (A/B tested)"
      row_ordering: "Personalized row order (action row first for action fans)"
      explanations: "'Because you watched X' — builds trust"
      
  batch_vs_online:
    batch: "Heavy embeddings, collaborative filtering (computed nightly)"
    online: "Session-based re-ranking (what you just browsed)"
    hybrid: "Batch candidates + online re-ranking"
```

---

## Interview Tip

> When asked about recommendation systems: "I design recommendations as a multi-stage pipeline optimized for different concerns at each stage. Stage 1 — Candidate Retrieval: narrow millions of items to hundreds using cheap methods. I use a two-tower architecture (user encoder + item encoder) with ANN (Approximate Nearest Neighbor) search via FAISS or ScaNN — retrieves 200 candidates in < 5ms from millions. Multiple retrieval sources (embedding similarity, collaborative filtering, popularity) merged for coverage. Stage 2 — Ranking: score each candidate with a feature-rich model. Features from feature store (user history embedding, item attributes, interaction features) combined with real-time context (session, device, time). Deep ranking model predicts engagement probability. Stage 3 — Re-ranking: enforce diversity (max 40% from one category), freshness (boost new items), business rules (promoted content, availability), and exploration (epsilon-greedy for learning). For cold start (new users): blend popularity-based and content-based recs, transition to personalized as interactions accumulate (5+ interactions). For real-time personalization: session features (clicked items, search queries) computed via streaming pipeline (Kafka → Redis) and consumed by ranking model within 1 second of user action. Key metrics: engagement rate (clicks, watches), diversity (unique categories shown), coverage (% of catalog recommended), and long-term retention (not just short-term clicks)."

---

## Common Mistakes

1. **Optimizing only for clicks** — Maximizing click-through rate leads to clickbait. Users click but don't engage deeply, and long-term satisfaction drops. Solution: multi-objective optimization — predict both click probability AND completion/satisfaction. Blend objectives.

2. **No diversity in results** — Showing 10 very similar items from the same category. User sees a filter bubble. Solution: MMR (Maximal Marginal Relevance) or category constraints in re-ranking. Ensure variety in final results.

3. **Ignoring cold start** — New users get empty or random recommendations. Bad first impression. Solution: explicit cold start strategy — use popularity, stated preferences, onboarding quiz. Transition to personalized after 5-10 interactions.

4. **Stale recommendations** — Batch predictions updated daily. User's interests shift intra-day (they're shopping for gifts, not themselves). Solution: real-time session features that adapt recommendations within seconds of user behavior change.

5. **Retrieval-ranking disconnect** — Training ranking model on random negatives, but at serving time it only sees retrieval candidates. Different distribution. Solution: train ranking model on candidates actually produced by retrieval (hard negatives from retrieval pipeline, not random items).

---

## Key Takeaways

- Recommendation systems: multi-stage pipeline (retrieval → ranking → re-ranking)
- Retrieval: narrow millions to hundreds cheaply (two-tower + ANN, < 10ms)
- Ranking: score candidates with feature-rich model (user + item + context features)
- Re-ranking: enforce diversity, freshness, business rules, exploration
- Two-tower: encode user and items separately, dot-product similarity, ANN search
- Cold start: popularity + content-based → transition to personalized (5+ interactions)
- Real-time personalization: session features via streaming (Kafka → Redis → model)
- Multi-objective: don't just optimize clicks — include completion, satisfaction, diversity
- Feature store: pre-computed user/item features + real-time session features
- Scale: batch for heavy computation (embeddings), online for session-based re-ranking
