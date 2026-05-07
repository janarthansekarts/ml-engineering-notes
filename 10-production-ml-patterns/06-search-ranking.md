# Search and Ranking

## The Problem / Why This Matters

Search ranking is the art of ordering results so the most relevant items appear first. Unlike recommendations (proactive suggestions), search responds to explicit user intent — a query. The engineering challenge: given a query and millions of potential results, return the top 10 most relevant items in under 100ms. This requires a multi-stage pipeline: retrieval (find candidate documents from millions), ranking (score candidates by relevance), and re-ranking (apply business logic and personalization). In 2026, search has evolved from keyword matching (BM25) to semantic understanding (dense embeddings, LLMs), hybrid approaches (sparse + dense retrieval), and learned ranking (Learning to Rank / LTR). Modern search systems combine traditional IR (Information Retrieval) signals with neural embeddings, user behavior features, and LLM-powered query understanding. Production search powers e-commerce (product search), enterprise (document search), and web search — each with different relevance definitions and engineering constraints. Getting ranking wrong means users can't find what they need, directly impacting revenue and user satisfaction.

---

## The Analogy

Think of search ranking like a librarian helping you find books:

- **Retrieval (Stage 1)** = The librarian quickly narrows from 10 million books to 500 potentially relevant ones. They use the card catalog (inverted index), subject classification (categories), and their general knowledge of the collection (embeddings). Speed matters — they don't read every book.
- **Ranking (Stage 2)** = The librarian examines those 500 books more carefully. They consider how well the title and content match your question, how popular the book is, how recent it is, whether you've liked similar books before. They arrange the top 20 in order.
- **Re-ranking (Stage 3)** = Final adjustments: put the one you previously checked out on top (personalization), ensure variety (not all from the same author), and note which ones are currently available (business constraints).

---

## Deep Dive

### Search System Architecture

```yaml
Search_Architecture:
  query_understanding:
    purpose: "Interpret user intent before retrieval"
    techniques:
      spell_correction: "Autocorrect typos ('runnig shoes' → 'running shoes')"
      query_expansion: "Add synonyms ('laptop' → 'laptop OR notebook OR computer')"
      intent_classification: "Is this navigational, informational, or transactional?"
      entity_recognition: "Extract entities ('red Nike shoes size 10')"
      query_rewriting: "LLM rewrites ambiguous query for better retrieval"
      
  stage_1_retrieval:
    purpose: "Find candidate documents (from millions → hundreds)"
    latency: "< 20ms"
    
    sparse_retrieval:
      bm25: "Traditional keyword matching (term frequency, document frequency)"
      inverted_index: "Elasticsearch, Lucene, Apache Solr"
      strength: "Exact keyword match, rare terms, proper nouns"
      weakness: "Misses semantic similarity ('car' won't match 'automobile')"
      
    dense_retrieval:
      embeddings: "Encode query and documents into vector space"
      models: "E5-large, BGE, Cohere Embed v3, OpenAI text-embedding-3-large"
      index: "FAISS, ScaNN, Pinecone, Weaviate, Qdrant"
      strength: "Semantic understanding ('car' matches 'automobile')"
      weakness: "Can miss exact keyword matches, expensive to index"
      
    hybrid_retrieval:
      what: "Combine sparse (BM25) + dense (embeddings)"
      fusion: "Reciprocal Rank Fusion (RRF) or learned fusion weights"
      benefit: "Best of both worlds: keyword precision + semantic recall"
      implementation: "Elasticsearch with kNN, Vespa, Qdrant hybrid search"
      
  stage_2_ranking:
    purpose: "Score candidates by relevance (hundreds → top 10)"
    latency: "< 50ms for hundreds of candidates"
    
    learning_to_rank:
      what: "ML model trained to order results by relevance"
      approaches:
        pointwise: "Predict relevance score for each (query, doc) pair"
        pairwise: "Predict which of two docs is more relevant (RankNet, LambdaRank)"
        listwise: "Optimize entire list ordering (LambdaMART, ListNet)"
      models:
        xgboost: "Fast, interpretable, good with tabular features (most common in production)"
        cross_encoder: "Transformer scores (query, doc) pair — accurate but expensive"
        two_tower: "Separate query/doc encoders — fast but less accurate"
        
    features:
      query_features: "Query length, intent type, entity count"
      document_features: "Popularity, freshness, quality score, content length"
      query_document_features: "BM25 score, embedding similarity, exact match count"
      user_features: "Click history, purchase history, preferences"
      context_features: "Device, location, time of day"
      
  stage_3_reranking:
    purpose: "Final adjustments for business needs"
    techniques:
      personalization: "Boost items matching user's history/preferences"
      diversity: "Ensure variety in results (not all same brand/category)"
      freshness: "Boost recent content for news/trending queries"
      business_rules: "Promoted listings, availability filtering, geo-restrictions"
```

### Learning to Rank (LTR) Implementation

```python
# Learning to Rank implementation

"""
Production Learning to Rank (LTR) system.
Trains a ranking model from user interaction data (clicks, purchases).
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A search result with features and score."""
    doc_id: str
    features: Dict[str, float]
    relevance_score: float = 0.0
    position: int = 0


class LearningToRankPipeline:
    """
    End-to-end LTR pipeline:
    1. Feature extraction
    2. Model training (LambdaMART via XGBoost)
    3. Serving (score and rank candidates)
    """
    
    def __init__(self, feature_extractor, ranking_model, feature_store):
        self.feature_extractor = feature_extractor
        self.ranking_model = ranking_model
        self.feature_store = feature_store
    
    def rank(self, query: str, candidates: List[Dict], user_id: str = None) -> List[SearchResult]:
        """
        Rank candidates for a query using LTR model.
        
        Args:
            query: User search query
            candidates: Retrieved candidate documents
            user_id: Optional user ID for personalization
            
        Returns:
            Ranked results (highest relevance first)
        """
        # Extract features for each (query, document) pair
        results = []
        for candidate in candidates:
            features = self.feature_extractor.extract(
                query=query,
                document=candidate,
                user_id=user_id
            )
            results.append(SearchResult(
                doc_id=candidate["id"],
                features=features,
            ))
        
        # Score with ranking model (batch prediction)
        feature_matrix = [list(r.features.values()) for r in results]
        scores = self.ranking_model.predict(feature_matrix)
        
        # Assign scores and sort
        for result, score in zip(results, scores):
            result.relevance_score = float(score)
        
        results.sort(key=lambda r: -r.relevance_score)
        
        # Assign positions
        for i, result in enumerate(results):
            result.position = i + 1
        
        return results


class FeatureExtractor:
    """
    Extract features for (query, document) pairs.
    
    Features used by the LTR model to predict relevance.
    """
    
    def __init__(self, embedding_model, feature_store):
        self.embedding_model = embedding_model
        self.feature_store = feature_store
    
    def extract(self, query: str, document: Dict, user_id: str = None) -> Dict[str, float]:
        """
        Extract ranking features.
        
        Feature categories:
        1. Text match features (lexical)
        2. Semantic features (embedding similarity)
        3. Document quality features
        4. User personalization features
        5. Context features
        """
        features = {}
        
        # 1. Text match features
        features["bm25_score"] = self._bm25_score(query, document)
        features["title_exact_match"] = float(query.lower() in document.get("title", "").lower())
        features["query_term_coverage"] = self._term_coverage(query, document)
        features["title_term_overlap"] = self._title_overlap(query, document)
        
        # 2. Semantic features
        query_embedding = self.embedding_model.encode(query)
        doc_embedding = self.embedding_model.encode(document.get("title", "") + " " + document.get("description", ""))
        features["embedding_similarity"] = float(np.dot(query_embedding, doc_embedding))
        
        # 3. Document quality features
        features["popularity_score"] = document.get("popularity", 0)
        features["freshness_days"] = document.get("age_days", 365)
        features["content_length"] = len(document.get("description", ""))
        features["rating"] = document.get("avg_rating", 0)
        features["review_count"] = document.get("review_count", 0)
        
        # 4. User personalization features (if user_id provided)
        if user_id:
            user_features = self.feature_store.get_user_features(user_id)
            features["user_category_affinity"] = user_features.get(
                f"category_affinity_{document.get('category', 'unknown')}", 0
            )
            features["user_brand_affinity"] = user_features.get(
                f"brand_affinity_{document.get('brand', 'unknown')}", 0
            )
            features["user_price_match"] = self._price_preference_match(
                user_features, document
            )
        
        # 5. Context features
        features["is_mobile"] = 0.0  # Set from request context
        features["hour_of_day"] = 12.0  # Set from request context
        
        return features
    
    def _bm25_score(self, query: str, document: Dict) -> float:
        """Compute BM25 relevance score."""
        # In production: retrieved from Elasticsearch score
        # Simplified implementation for illustration
        text = (document.get("title", "") + " " + document.get("description", "")).lower()
        query_terms = query.lower().split()
        score = sum(1.0 for term in query_terms if term in text)
        return score / max(len(query_terms), 1)
    
    def _term_coverage(self, query: str, document: Dict) -> float:
        """What fraction of query terms appear in document."""
        text = (document.get("title", "") + " " + document.get("description", "")).lower()
        query_terms = query.lower().split()
        covered = sum(1 for term in query_terms if term in text)
        return covered / max(len(query_terms), 1)
    
    def _title_overlap(self, query: str, document: Dict) -> float:
        """Overlap between query and title."""
        title_words = set(document.get("title", "").lower().split())
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0
        return len(title_words & query_words) / len(query_words)
    
    def _price_preference_match(self, user_features: Dict, document: Dict) -> float:
        """How well document price matches user's price preference."""
        user_avg_price = user_features.get("avg_purchase_price", 50)
        doc_price = document.get("price", 50)
        # Score higher when close to user's typical price
        diff = abs(doc_price - user_avg_price) / max(user_avg_price, 1)
        return max(0, 1 - diff)  # 1.0 = exact match, 0 = very different


class TrainingDataGenerator:
    """
    Generate LTR training data from user interactions.
    
    Implicit feedback (clicks, purchases) → relevance labels.
    
    Challenges:
    - Position bias: users click top results more (not because they're better)
    - Presentation bias: can only learn from shown results
    - Sparse labels: most results get no interaction
    """
    
    def __init__(self, click_log_store, feature_extractor):
        self.click_log_store = click_log_store
        self.feature_extractor = feature_extractor
    
    def generate_training_data(
        self, date_range: Tuple[str, str]
    ) -> List[Dict]:
        """
        Generate training examples from click logs.
        
        Approach: treat clicks as positive, non-clicks at same position as negative.
        Apply position debiasing (inverse propensity weighting).
        """
        training_examples = []
        
        # Get click logs
        click_logs = self.click_log_store.get_logs(date_range)
        
        for log in click_logs:
            query = log["query"]
            results = log["results"]  # List of shown results with click info
            
            for result in results:
                # Extract features
                features = self.feature_extractor.extract(
                    query=query,
                    document=result["document"],
                    user_id=log.get("user_id")
                )
                
                # Relevance label
                if result.get("purchased"):
                    relevance = 4  # Highest
                elif result.get("add_to_cart"):
                    relevance = 3
                elif result.get("clicked"):
                    relevance = 2
                elif result.get("shown"):
                    relevance = 0  # Shown but not clicked
                else:
                    relevance = 0
                
                # Position debiasing weight
                # Items at position 1 get more clicks regardless of relevance
                position = result.get("position", 1)
                position_weight = self._position_debias_weight(position)
                
                training_examples.append({
                    "query_id": log["query_id"],
                    "features": features,
                    "relevance": relevance,
                    "weight": position_weight,
                })
        
        return training_examples
    
    def _position_debias_weight(self, position: int) -> float:
        """
        Inverse propensity scoring (IPS) for position bias.
        
        Position 1 items are clicked 10x more than position 10
        regardless of relevance. Weight lower-position clicks higher.
        """
        # Examination probability decreases with position
        # Based on eye-tracking studies: P(examine) ∝ 1/position^0.5
        examination_prob = 1.0 / (position ** 0.5)
        # IPS weight: upweight items examined less
        return 1.0 / max(examination_prob, 0.01)
```

### Hybrid Search (Sparse + Dense)

```yaml
Hybrid_Search:
  why: "Neither sparse nor dense is universally better"
  
  sparse_strengths:
    - "Exact keyword matching (product codes, names)"
    - "Rare terms (unusual proper nouns)"
    - "No training data needed (works out of the box)"
    
  dense_strengths:
    - "Semantic understanding ('car' matches 'automobile')"
    - "Multilingual (same model handles English and Spanish)"
    - "Handles typos and variations naturally"
    
  hybrid_fusion:
    reciprocal_rank_fusion:
      what: "Combine rankings from both systems"
      formula: "Score = Σ 1/(k + rank_in_system_i)"
      benefit: "No tuning needed, robust"
      
    learned_fusion:
      what: "ML model learns optimal weight for each signal"
      features: "BM25 score, embedding score, query length, query type"
      benefit: "Adapts weight based on query characteristics"
      
  implementation:
    elasticsearch_knn: "Native hybrid in Elasticsearch 8.x (BM25 + kNN)"
    vespa: "Built for hybrid search (BM25 + embedding + LTR in one system)"
    qdrant_hybrid: "Sparse + dense vectors in same index"
```

### Evaluation Metrics

```yaml
Search_Metrics:
  relevance_metrics:
    ndcg: "Normalized Discounted Cumulative Gain — order-sensitive relevance"
    map: "Mean Average Precision — precision at each relevant result"
    mrr: "Mean Reciprocal Rank — position of first relevant result"
    
  online_metrics:
    click_through_rate: "% of queries with at least one click"
    success_rate: "% of queries where user found what they needed"
    abandonment_rate: "% of queries with no interaction (bad sign)"
    time_to_first_click: "How long before user clicks (faster = better results)"
    
  position_metrics:
    clicks_at_1: "% of clicks on first result (higher = better top result)"
    pogo_sticking: "Click → immediate back → click another (bad relevance)"
```

---

## How It Works in Practice

### E-Commerce Search Pipeline

```yaml
Ecommerce_Search:
  query: "wireless noise cancelling headphones under $200"
  
  query_understanding:
    entities: "{product_type: headphones, features: [wireless, noise_cancelling], price_max: 200}"
    intent: "transactional (user wants to buy)"
    
  retrieval:
    bm25: "1500 products matching keywords (Elasticsearch)"
    dense: "800 products semantically similar (embedding search)"
    hybrid_fusion: "1200 unique candidates (RRF fusion, deduplicated)"
    filter: "Apply price < $200, in_stock = true → 600 candidates"
    
  ranking:
    model: "XGBoost LambdaMART (trained on click/purchase logs)"
    features: "BM25 score, embedding similarity, popularity, rating, price_match, brand_affinity"
    output: "600 candidates scored and sorted"
    
  reranking:
    diversity: "Max 3 from same brand in top 10"
    boosting: "Sponsored products at positions 2, 5, 8 (clearly labeled)"
    freshness: "New releases boosted slightly"
    
  result: "Top 20 products returned in 65ms total"
```

---

## Interview Tip

> When asked about search ranking: "I design search as a multi-stage pipeline: retrieval → ranking → re-ranking. For retrieval, I use hybrid search — BM25 (keyword precision for exact matches, product codes) combined with dense retrieval (semantic understanding via embeddings like E5 or Cohere Embed v3). Fusion via Reciprocal Rank Fusion (RRF) gives best of both. For ranking, I use Learning to Rank (LTR) — specifically LambdaMART (XGBoost with listwise objective). Features include BM25 score, embedding similarity, document popularity, freshness, user personalization features (category affinity, brand affinity, price preference). Training data comes from click logs with position debiasing (inverse propensity weighting to correct for users clicking top results regardless of relevance). For evaluation: NDCG (Normalized Discounted Cumulative Gain) for offline metrics (position-sensitive relevance), click-through rate and abandonment rate for online metrics. Key engineering decisions: (1) embedding model choice — I use E5-large or Cohere for multi-lingual support and semantic quality, (2) re-ranking with cross-encoder for top 50 results (expensive but accurate — BERT-based scoring of (query, doc) pairs), (3) real-time personalization from session features (what user just searched/clicked)."

---

## Common Mistakes

1. **Only using dense retrieval** — Dense embeddings miss exact keyword matches (product codes, rare proper nouns). BM25 catches these. Solution: hybrid retrieval (sparse + dense) with fusion. Evaluate both independently and combined.

2. **Not debiasing click data** — Training on raw clicks. Position 1 gets 10x more clicks than position 5, regardless of relevance. Model learns "top results are best" instead of actual relevance. Solution: inverse propensity weighting — upweight clicks at lower positions (they're more meaningful signals).

3. **Ignoring query understanding** — Same ranking for "Nike" (navigational — user wants Nike.com) and "running shoes" (informational — user wants options). Solution: classify query intent, adapt retrieval and ranking strategy per intent type.

4. **No offline evaluation before deployment** — Deploying new ranking model without measuring NDCG on held-out data. Model might rank worse than the baseline. Solution: rigorous offline evaluation on labeled test set, then A/B test online if offline metrics improve.

5. **Stale ranking model** — Ranking model trained on last year's data. Product catalog changed, user behavior shifted. Solution: retrain weekly/monthly on recent click data. Monitor online metrics for degradation.

---

## Key Takeaways

- Search ranking: multi-stage pipeline (query understanding → retrieval → ranking → re-ranking)
- Hybrid retrieval: BM25 (keyword precision) + dense embeddings (semantic understanding) + RRF fusion
- Learning to Rank: ML model (LambdaMART/XGBoost) trained on click logs with position debiasing
- Features: text match + semantic similarity + document quality + user personalization + context
- Position bias: users click top results more — correct with inverse propensity weighting
- Evaluation: NDCG (offline, position-sensitive), CTR and abandonment rate (online)
- Query understanding: spell correction, entity extraction, intent classification, query rewriting
- Dense retrieval models: E5-large, BGE, Cohere Embed v3 (2026 state-of-art)
- Hybrid search tools: Elasticsearch kNN, Vespa, Qdrant (combine sparse + dense in one system)
- Re-ranking: diversity, freshness, personalization, business rules (sponsored results)
