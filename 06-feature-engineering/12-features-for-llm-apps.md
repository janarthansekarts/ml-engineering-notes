# Features for LLM Applications

## The Problem / Why This Matters

LLM (Large Language Model) applications — chatbots, RAG (Retrieval-Augmented Generation) systems, AI agents, content generators — are production ML systems that need features just like traditional ML. But the features are different: instead of user_purchase_count_30d, you need embedding similarity scores, retrieval relevance signals, conversation context features, user preference vectors, and behavioral signals that determine how the LLM should respond. Feature engineering for LLM apps is still emerging but already critical: deciding which documents to retrieve, personalizing responses, routing queries to the right model, detecting hallucinations, managing costs, and improving quality — all require features computed from user behavior, query characteristics, document metadata, and conversation history. In 2026, this is the fastest-growing area of feature engineering, as every company shipping LLM products discovers they need the same production ML infrastructure they built for traditional models, now applied to their AI applications.

---

## The Analogy

Think of LLM features like a personal assistant's context:

Without features, the LLM is like an assistant who starts fresh every conversation — no memory of what you prefer, no idea what documents are relevant to you, no sense of urgency, no awareness of your expertise level.

With features, the assistant knows: "This user is a senior engineer (expertise_level=expert), they've asked about Kubernetes 5 times this week (topic_affinity=k8s:high), their last 3 questions were follow-ups (conversation_depth=3), they prefer concise answers (style_preference=brief), and the most relevant document for this query has 0.92 similarity (retrieval_confidence=high)."

Features turn a generic LLM into a personalized, context-aware system that makes smart decisions about what to retrieve, how to respond, and when to escalate.

---

## Deep Dive

### Feature Categories for LLM Apps

```yaml
Feature_Categories:
  retrieval_features:
    what: "Signals about document relevance for RAG"
    examples:
      embedding_similarity: "Cosine similarity between query and document embeddings"
      bm25_score: "Lexical match score (BM25 algorithm)"
      reranker_score: "Cross-encoder reranking score (more accurate but slower)"
      document_freshness: "Days since document was last updated"
      document_popularity: "How often this document is retrieved for similar queries"
      chunk_position: "Position of chunk within document (beginning often more relevant)"
      metadata_match: "Query metadata matches document metadata (same category, same product)"
    purpose: "Decide which documents to include in LLM context"
    
  query_features:
    what: "Characteristics of the user's query"
    examples:
      query_length: "Word count (short queries often need clarification)"
      query_complexity: "Estimated difficulty / required expertise"
      query_intent: "Classified intent (question, instruction, conversation, creative)"
      query_language: "Detected language"
      query_specificity: "Specific (exact error message) vs vague ('it doesn't work')"
      contains_code: "Does query contain code snippets"
      question_type: "Factual, opinion, how-to, comparison, troubleshooting"
      topic_category: "Classified topic (billing, technical, product, general)"
    purpose: "Route queries, select retrieval strategy, determine response format"
    
  user_features:
    what: "User profile and behavioral signals"
    examples:
      expertise_level: "Beginner/intermediate/expert (inferred from past interactions)"
      topic_affinity: "Topics this user frequently asks about"
      response_preference: "Preferred response length, format, detail level"
      satisfaction_history: "Average rating of past responses"
      session_count: "How many sessions this user has had"
      last_active: "Recency of engagement"
      plan_tier: "Free/pro/enterprise (determines model/feature access)"
    purpose: "Personalize responses, match expertise level, adapt communication style"
    
  conversation_features:
    what: "Signals from the current conversation"
    examples:
      turn_count: "Number of turns in this conversation"
      topic_shifts: "How many topic changes in this conversation"
      follow_up_depth: "Consecutive follow-up questions (drill-down behavior)"
      sentiment_trend: "Is user getting frustrated? (sentiment declining)"
      tokens_used: "Total tokens consumed in this session"
      unresolved_questions: "Questions asked but not answered satisfactorily"
      context_window_usage: "How full is the context window"
    purpose: "Manage context window, detect frustration, trigger escalation"
    
  response_quality_features:
    what: "Features for evaluating/improving response quality"
    examples:
      source_coverage: "% of response claims backed by retrieved sources"
      hallucination_risk: "Estimated probability response contains fabricated info"
      confidence_score: "Model's self-assessed confidence (logprob-based)"
      factual_consistency: "Does response contradict retrieved documents"
      response_relevance: "How relevant is the response to the query"
    purpose: "Quality monitoring, guardrails, automatic re-generation"
```

### Embedding Features for RAG

```python
# Embedding-based features for RAG systems

"""
RAG systems need rich features to decide:
1. What to retrieve (retrieval features)
2. How to rank retrieved results (reranking features)
3. Whether retrieved context is sufficient (confidence features)
"""

import numpy as np
from typing import Optional


class RAGFeatureExtractor:
    """Extract features for RAG retrieval and quality decisions."""
    
    def __init__(self, embedding_model, reranker_model=None):
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
    
    def extract_retrieval_features(
        self, 
        query: str, 
        retrieved_chunks: list[dict],
    ) -> dict:
        """
        Extract features about the retrieval results.
        Used by downstream model to decide: 
        - Is retrieval quality sufficient to answer?
        - Should we try different retrieval strategy?
        - Which chunks to include in context?
        """
        
        query_embedding = self.embedding_model.encode(query)
        
        features = {}
        
        # Query characteristics
        features["query_word_count"] = len(query.split())
        features["query_has_question_mark"] = int("?" in query)
        features["query_embedding_norm"] = float(np.linalg.norm(query_embedding))
        
        # Retrieval quality signals
        similarities = []
        for chunk in retrieved_chunks:
            sim = np.dot(query_embedding, chunk["embedding"])
            similarities.append(sim)
        
        features["top_1_similarity"] = similarities[0] if similarities else 0.0
        features["top_3_avg_similarity"] = np.mean(similarities[:3]) if similarities else 0.0
        features["top_5_avg_similarity"] = np.mean(similarities[:5]) if similarities else 0.0
        features["similarity_gap"] = (
            similarities[0] - similarities[4] if len(similarities) >= 5 else 0.0
        )  # Large gap = clear winner; small gap = ambiguous
        
        # Diversity of results
        features["source_diversity"] = len(set(
            c.get("source_document") for c in retrieved_chunks[:5]
        ))  # More diverse sources = broader coverage
        
        # Freshness of retrieved content
        features["newest_chunk_age_days"] = min(
            c.get("age_days", 9999) for c in retrieved_chunks[:5]
        ) if retrieved_chunks else 9999
        
        features["oldest_chunk_age_days"] = max(
            c.get("age_days", 0) for c in retrieved_chunks[:5]
        ) if retrieved_chunks else 0
        
        # Retrieval confidence
        features["confident_retrieval"] = int(features["top_1_similarity"] > 0.8)
        features["ambiguous_retrieval"] = int(
            features["similarity_gap"] < 0.05 and features["top_1_similarity"] < 0.7
        )
        
        return features
    
    def extract_response_features(
        self,
        query: str,
        response: str,
        retrieved_chunks: list[dict],
        logprobs: Optional[list[float]] = None,
    ) -> dict:
        """
        Extract features about the generated response.
        Used for quality monitoring, hallucination detection, routing.
        """
        
        features = {}
        
        # Response characteristics
        features["response_word_count"] = len(response.split())
        features["response_sentence_count"] = response.count('.') + response.count('!') + response.count('?')
        features["contains_code_block"] = int("```" in response)
        features["contains_caveat"] = int(any(
            phrase in response.lower() 
            for phrase in ["i'm not sure", "i don't have", "cannot confirm", "may not be accurate"]
        ))
        
        # Confidence from logprobs (if available)
        if logprobs:
            features["avg_logprob"] = np.mean(logprobs)
            features["min_logprob"] = np.min(logprobs)
            features["low_confidence_token_ratio"] = np.mean([
                1 for lp in logprobs if lp < -2.0
            ]) / max(len(logprobs), 1)
        
        # Source attribution
        response_lower = response.lower()
        cited_sources = sum(
            1 for chunk in retrieved_chunks 
            if any(sent.lower() in response_lower for sent in chunk.get("key_sentences", []))
        )
        features["source_coverage_ratio"] = cited_sources / max(len(retrieved_chunks[:5]), 1)
        
        # Response-query relevance
        query_embedding = self.embedding_model.encode(query)
        response_embedding = self.embedding_model.encode(response)
        features["query_response_similarity"] = float(
            np.dot(query_embedding, response_embedding)
        )
        
        return features
```

### User Behavior Features for LLM Apps

```yaml
User_Behavior_Features:
  engagement_signals:
    thumbs_up_rate: "% of responses user gives positive feedback"
    copy_rate: "% of responses user copies (indicates usefulness)"
    follow_up_rate: "% of responses that lead to follow-up questions"
    session_duration: "Average time spent per session"
    messages_per_session: "Average messages before user leaves"
    
  preference_signals:
    preferred_length: "Median response length for positively-rated responses"
    preferred_format: "Does user prefer bullet points, prose, code examples?"
    topic_distribution: "Distribution of topics user asks about"
    time_of_day: "When does user typically interact (timezone inference)"
    
  quality_signals:
    regeneration_rate: "How often user clicks 'regenerate'"
    edit_after_response: "Does user frequently modify AI-suggested content"
    escalation_rate: "How often user gives up and contacts human support"
    
  computed_features:
    expertise_score:
      computation: "Based on question complexity, vocabulary, topics"
      levels: "beginner (needs explanations), intermediate (knows concepts), expert (needs specifics)"
      update: "Rolling average over last 30 interactions"
      
    satisfaction_score:
      computation: "Weighted combination: thumbs_up (0.4) + copy (0.3) + no_regenerate (0.3)"
      range: "[0, 1]"
      use: "Low score → route to better model or trigger human handoff"
      
    cost_tier:
      computation: "Based on plan + usage patterns + token consumption"
      use: "Route expensive queries to cheaper models for free-tier users"
```

### Routing and Model Selection Features

```python
# Features for intelligent routing in LLM systems

"""
Multi-model architectures need features to decide:
- Which model handles this query? (GPT-4o-mini vs Claude 4 vs local model)
- Should we use RAG or direct generation?
- How many retrieval results to include?
"""


class QueryRoutingFeatures:
    """Extract features for query routing decisions."""
    
    def extract(self, query: str, user_context: dict, system_state: dict) -> dict:
        """
        Features that determine query routing:
        - Simple factual → cheap model (GPT-4o-mini)
        - Complex reasoning → expensive model (Claude 4 Opus, o3)
        - Code generation → code-specialized model
        - Need retrieval → RAG pipeline
        - No retrieval needed → direct generation
        """
        
        features = {}
        
        # Query complexity signals
        features["query_word_count"] = len(query.split())
        features["contains_code"] = int("```" in query or "def " in query or "function" in query)
        features["contains_math"] = int(any(c in query for c in "∫∑∏√±"))
        features["multi_step_indicators"] = sum(
            1 for phrase in ["step by step", "compare", "analyze", "pros and cons", "trade-offs"]
            if phrase in query.lower()
        )
        
        # Intent classification (pre-computed by lightweight classifier)
        features["intent_factual"] = float(user_context.get("intent_scores", {}).get("factual", 0))
        features["intent_creative"] = float(user_context.get("intent_scores", {}).get("creative", 0))
        features["intent_reasoning"] = float(user_context.get("intent_scores", {}).get("reasoning", 0))
        features["intent_code"] = float(user_context.get("intent_scores", {}).get("code", 0))
        
        # User tier (determines available models)
        features["user_tier_free"] = int(user_context.get("plan") == "free")
        features["user_tier_pro"] = int(user_context.get("plan") == "pro")
        features["user_tier_enterprise"] = int(user_context.get("plan") == "enterprise")
        
        # Cost signals
        features["estimated_input_tokens"] = len(query.split()) * 1.3  # rough estimate
        features["user_monthly_token_usage"] = user_context.get("monthly_tokens", 0)
        features["user_budget_remaining_pct"] = user_context.get("budget_remaining_pct", 100)
        
        # System state (capacity-based routing)
        features["primary_model_latency_p50"] = system_state.get("claude4_latency_p50", 0)
        features["primary_model_queue_depth"] = system_state.get("claude4_queue", 0)
        features["fallback_model_available"] = int(system_state.get("gpt4o_mini_healthy", True))
        
        # Retrieval signals
        features["needs_current_info"] = int(any(
            w in query.lower() for w in ["latest", "current", "today", "recent", "2025", "2026"]
        ))
        features["needs_internal_data"] = int(any(
            w in query.lower() for w in ["our", "my", "company", "internal", "project"]
        ))
        
        return features


class ResponseQualityPredictor:
    """Predict response quality BEFORE generation (for routing decisions)."""
    
    def predict_quality(self, routing_features: dict, retrieval_features: dict) -> dict:
        """
        Predict expected quality for different model choices.
        Used to make routing decision: which model will give best quality?
        """
        
        predictions = {}
        
        # If retrieval confidence is high + simple query → cheap model is sufficient
        if (retrieval_features.get("top_1_similarity", 0) > 0.85 and 
            routing_features.get("multi_step_indicators", 0) == 0):
            predictions["recommended_model"] = "gpt-4o-mini"
            predictions["confidence"] = 0.9
            predictions["reasoning"] = "High retrieval confidence + simple query"
            
        # Complex reasoning → expensive model
        elif routing_features.get("intent_reasoning", 0) > 0.7:
            predictions["recommended_model"] = "claude-4-opus"
            predictions["confidence"] = 0.8
            predictions["reasoning"] = "Complex reasoning detected"
            
        # Code generation → code model
        elif routing_features.get("intent_code", 0) > 0.8:
            predictions["recommended_model"] = "claude-4-sonnet"  # Good at code
            predictions["confidence"] = 0.85
            predictions["reasoning"] = "Code generation task"
            
        else:
            predictions["recommended_model"] = "claude-4-sonnet"
            predictions["confidence"] = 0.6
            predictions["reasoning"] = "Default routing"
        
        return predictions
```

### Cost and Token Management Features

```yaml
Cost_Features:
  per_request:
    estimated_input_tokens: "Tokens in query + context (before generation)"
    estimated_output_tokens: "Predicted output length (based on query type)"
    retrieval_cost: "Embedding computation + vector DB query cost"
    total_estimated_cost: "Sum of all costs for this request"
    
  per_user:
    daily_token_usage: "Tokens consumed today"
    monthly_token_usage: "Tokens consumed this billing period"
    budget_remaining: "Remaining budget before throttling/upgrading"
    cost_per_interaction: "Average cost per conversation turn"
    
  system_level:
    hourly_token_rate: "System-wide token consumption rate"
    model_cost_breakdown: "Spend per model (claude-4 vs gpt-4o-mini)"
    cache_hit_rate: "% of requests served from cache (cost savings)"
    
  features_for_cost_optimization:
    query_can_be_cached: "Similar query answered recently → serve cached response"
    context_can_be_shortened: "Low-relevance chunks → remove to save tokens"
    response_length_prediction: "Predict if response will be long → maybe use cheaper model"
    multi_turn_context_compression: "Older turns can be summarized → save tokens"
```

---

## How It Works in Practice

### Feature Pipeline for LLM Application

```yaml
Pipeline:
  real_time_features:
    computed_at: "Request time (must be fast: <50ms)"
    features:
      - "Query embedding (10ms with cached model)"
      - "Query word count, intent classification (5ms)"
      - "Top-K retrieval similarity scores (20ms)"
      - "Context window tokens used (instant)"
    
  near_real_time:
    computed_at: "After previous interaction completes (async)"
    features:
      - "Conversation turn count (updated per message)"
      - "Sentiment trend (updated per message)"
      - "Topic tracking (what's being discussed)"
      
  batch_computed:
    computed_at: "Daily batch pipeline"
    features:
      - "User expertise score (rolling 30-day)"
      - "User satisfaction score (rolling 30-day)"
      - "Topic affinity distribution"
      - "Response preference profile"
      - "Monthly token usage"
      
  serving:
    user_features: "Pre-computed daily → Redis lookup (1ms)"
    query_features: "Computed at request time (10-20ms)"
    retrieval_features: "Computed during retrieval step (20ms)"
    conversation_features: "Updated per turn, stored in session cache"
```

---

## Interview Tip

> When asked about features for LLM applications: "LLM apps need feature engineering just like traditional ML, but the features serve different decisions. I organize them into four categories: (1) Retrieval features — embedding similarity, BM25 scores, document freshness, source diversity. These drive RAG quality: which chunks make it into context. (2) Query features — complexity, intent classification, whether it needs current information. These drive routing: which model handles it, whether to use RAG. (3) User features — expertise level, topic affinity, satisfaction history, plan tier. These drive personalization and cost management. (4) Response quality features — source coverage, confidence scores from logprobs, hallucination risk indicators. These drive guardrails and regeneration decisions. The architecture: user features are batch-computed daily and served from Redis. Query and retrieval features are computed at request time (must be <50ms). Conversation features are session-scoped and updated per turn. Model selection/routing uses a lightweight classifier trained on these features to decide: cheap model vs. expensive model, RAG vs. direct generation, short vs. detailed response. The key insight: every 'if-else' in your LLM app is secretly a feature engineering problem — turn those heuristics into learned decisions with proper features."

---

## Common Mistakes

1. **Hardcoded routing logic instead of feature-based** — "If query contains 'code' → use code model, else → use general model." Brittle rules that don't generalize. Solution: extract query features, train lightweight routing model that learns optimal routing from outcome data (which model produced highest-rated responses for which query types).

2. **No user personalization features** — Every user gets the same experience regardless of expertise, preferences, or history. Expert gets "Let me explain what a variable is..." Solution: compute and serve user features (expertise_level, topic_affinity, response_preference). Use these in system prompts and retrieval strategies.

3. **Ignoring retrieval confidence** — Always include retrieved context regardless of relevance. Top result has 0.3 similarity (garbage) but gets shoved into context anyway → confuses the LLM. Solution: retrieval confidence features (top similarity, similarity gap, source diversity) drive decisions: high confidence → use context, low confidence → generate without context or ask for clarification.

4. **No cost features** — Free-tier users consuming $50/day in Claude 4 Opus calls because there's no cost-aware routing. Solution: user tier + token usage + estimated cost features drive model selection. Free tier → smaller model. Enterprise → best model. Budget nearly spent → cheaper model + notify.

5. **Treating LLM app as stateless** — Every request handled independently, no memory of conversation context or user patterns. Same user asks the same question 5 times because each response is wrong, but system doesn't notice. Solution: conversation features (turn count, sentiment trend, regeneration rate) trigger quality interventions. Rising frustration → escalate to better model or human.

---

## Key Takeaways

- LLM apps need four feature categories: retrieval, query, user, and response quality
- Retrieval features (similarity scores, freshness, diversity) determine RAG context quality
- Query features (complexity, intent, language) drive model routing and retrieval strategy
- User features (expertise, preferences, satisfaction) enable personalization and cost control
- Response quality features (confidence, source coverage, hallucination risk) power guardrails
- User features: batch-computed daily, served from Redis (<1ms lookup)
- Query/retrieval features: computed at request time (target <50ms total)
- Conversation features: session-scoped, updated per turn, stored in session cache
- Replace hardcoded routing rules with feature-based learned routing
- Every heuristic in your LLM app is a feature engineering problem waiting to be solved
- Cost features: user tier + usage + estimated cost → intelligent model selection
- LLM feature engineering is the fastest-growing area in ML engineering (2026)
