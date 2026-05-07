# Text and NLP Features

## The Problem / Why This Matters

Text data is everywhere in production systems — product descriptions, customer reviews, support tickets, search queries, chat messages, emails, code — but models can't process raw text directly. You need to convert text into numerical representations (features) that capture meaning, intent, and relevant signals. The landscape has evolved dramatically: from bag-of-words and TF-IDF (Term Frequency-Inverse Document Frequency) in the 2010s, to word embeddings (Word2Vec, GloVe) in the mid-2010s, to transformer-based sentence embeddings (2019+), to LLM-generated features in 2024-2026. The engineering challenge: choosing the right text representation for your use case (semantic search needs dense embeddings, spam detection might work fine with TF-IDF), managing computational costs (embedding 10M documents with a large model is expensive), handling multilingual text, dealing with text that changes over time, and serving text features at low latency. Most production ML systems now use pre-trained sentence embeddings (384-1536 dimensions) as the primary text feature, supplemented by extracted signals (sentiment, entities, length, readability) for interpretability.

---

## The Analogy

Think of text features like describing a painting to someone who can't see it:

- **Bag of Words / TF-IDF** = Inventory list. "Contains: red (3 times), woman (1), flowers (5), garden (2)." You know what's IN the painting but not how things relate or what it means.
- **Word Embeddings** = Individual color swatches. Each word is a rich color, but you're looking at each word in isolation — you can't see the full picture.
- **Sentence Embeddings** = A photograph of the painting reduced to a thumbnail. It captures the overall composition, mood, and meaning in a compact form. Two similar paintings have similar thumbnails.
- **LLM-Generated Features** = An art critic's review. "This impressionist garden scene evokes serenity, features Monet's signature light play, and depicts a woman in contemplation." Rich, structured understanding extracted by an expert.

Each level gives more semantic understanding but costs more to produce.

---

## Deep Dive

### Traditional Text Features

```yaml
Traditional_Methods:
  bag_of_words:
    what: "Count occurrences of each word in the text"
    output: "Sparse vector (size = vocabulary)"
    pros: "Simple, fast, interpretable"
    cons: "No semantic understanding ('car' and 'automobile' are unrelated)"
    use_when: "Very simple classification, keyword matching"
    
  tf_idf:
    what: "Term Frequency × Inverse Document Frequency"
    formula: "TF(word) × log(N_documents / N_documents_containing_word)"
    intuition: "Words important to THIS document but rare across ALL documents get high weight"
    output: "Sparse vector (size = vocabulary)"
    pros: "Better than raw counts (downweights common words)"
    cons: "Still no semantic understanding, high dimensionality"
    use_when: "Document similarity, simple search, baseline models"
    
  n_grams:
    what: "Sequences of N consecutive words"
    types:
      unigrams: "Individual words (same as bag-of-words)"
      bigrams: "'machine learning', 'deep learning', 'neural network'"
      trigrams: "'natural language processing', 'large language model'"
    benefit: "Captures multi-word phrases (improves over single words)"
    
  extracted_features:
    what: "Specific signals extracted from text (not representations)"
    examples:
      text_length: "Character count, word count, sentence count"
      readability: "Flesch-Kincaid grade level, average word length"
      capitalization: "% uppercase words (angry email signal)"
      punctuation: "! count, ? count (urgency/questions)"
      language: "Detected language code"
      entities: "Named entities extracted (people, places, organizations)"
      sentiment: "Positive/negative/neutral score"
      keywords: "Top TF-IDF keywords"
```

### Dense Embeddings (Modern Approach)

```python
# Text embedding approaches for production features

"""
Modern text features: encode text into dense vectors using pre-trained models.
Similar texts → similar vectors (close in embedding space).
"""

from sentence_transformers import SentenceTransformer
import numpy as np


# Approach 1: Open-source sentence transformers (fast, free, self-hosted)
class LocalTextEmbedder:
    """Self-hosted text embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Model options (2026):
        - all-MiniLM-L6-v2: 384d, fast, good quality (general purpose)
        - all-mpnet-base-v2: 768d, slower, better quality
        - bge-large-en-v1.5: 1024d, state-of-the-art open-source
        - gte-large-en-v1.5: 1024d, excellent for retrieval
        - nomic-embed-text-v1.5: 768d, long context (8192 tokens)
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a batch of texts."""
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )
        return embeddings  # Shape: (len(texts), embedding_dim)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text (for serving)."""
        return self.model.encode([text], normalize_embeddings=True)[0]


# Approach 2: API-based embeddings (higher quality, simpler, costs money)
class APITextEmbedder:
    """API-based text embeddings (OpenAI, Cohere, Voyage)."""
    
    def __init__(self, provider: str = "openai"):
        """
        Provider options (2026):
        - OpenAI text-embedding-3-small: 1536d, $0.02/1M tokens
        - OpenAI text-embedding-3-large: 3072d, $0.13/1M tokens
        - Cohere embed-english-v3.0: 1024d, good multilingual
        - Voyage voyage-3: 1024d, excellent for code + general text
        """
        self.provider = provider
        
    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts via API (with batching and rate limiting)."""
        # Batch into groups of 100 (API limits)
        all_embeddings = []
        for batch in chunk(texts, 100):
            response = await self.client.embeddings.create(
                input=batch,
                model="text-embedding-3-small",
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        return np.array(all_embeddings)


# Approach 3: Hybrid features (embeddings + extracted signals)
class HybridTextFeatures:
    """Combine dense embeddings with interpretable extracted features."""
    
    def __init__(self):
        self.embedder = LocalTextEmbedder("all-MiniLM-L6-v2")
        
    def extract_features(self, text: str) -> dict:
        """Extract comprehensive text features."""
        
        features = {}
        
        # Dense embedding (primary semantic signal)
        features["embedding"] = self.embedder.embed_single(text)  # 384d vector
        
        # Extracted features (interpretable, cheap)
        features["char_count"] = len(text)
        features["word_count"] = len(text.split())
        features["sentence_count"] = text.count('.') + text.count('!') + text.count('?')
        features["avg_word_length"] = np.mean([len(w) for w in text.split()]) if text else 0
        
        # Punctuation features
        features["exclamation_count"] = text.count('!')
        features["question_count"] = text.count('?')
        features["uppercase_ratio"] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Content signals
        features["has_url"] = int("http" in text.lower())
        features["has_email"] = int("@" in text and "." in text)
        features["numeric_ratio"] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        
        return features
```

### LLM-Generated Features

```yaml
LLM_Features:
  what: "Use LLMs to extract structured features from text"
  when: "Need high-quality structured extraction that rules/regex can't handle"
  
  extraction_types:
    sentiment_analysis:
      prompt: "Rate the sentiment of this review: 1 (very negative) to 5 (very positive)"
      output: "Integer score"
      cost: "~$0.001 per text (GPT-4o-mini)"
      
    intent_classification:
      prompt: "Classify this support ticket: billing, technical, account, feedback, other"
      output: "Category string"
      
    entity_extraction:
      prompt: "Extract: product_mentioned, issue_type, urgency (high/medium/low)"
      output: "Structured JSON"
      
    summarization:
      prompt: "Summarize this document in one sentence"
      output: "String (then embed for feature)"
      
    topic_extraction:
      prompt: "What are the main topics discussed? Return as list"
      output: "List of topics"
      
  production_considerations:
    cost: "LLM extraction is expensive at scale — batch process offline"
    latency: "Too slow for real-time serving — pre-compute and cache"
    consistency: "LLMs can be inconsistent — use structured outputs / function calling"
    implementation:
      1: "Batch extract features for all historical text (offline job)"
      2: "Store extracted features in feature store"
      3: "For new text: extract on ingestion (async), not at serving time"
      4: "Cache aggressively — same text should produce same features"
```

### Text Features for Different Use Cases

```yaml
Use_Case_Features:
  search_ranking:
    primary: "Dense embeddings (query + document similarity via dot product)"
    supplementary:
      - "BM25 score (lexical match, handles exact keywords)"
      - "Query-document term overlap"
      - "Document freshness"
      - "Document length"
    architecture: "Hybrid retrieval (dense + sparse) → re-ranking with cross-encoder"
    
  spam_detection:
    primary: "TF-IDF (still effective for spam patterns)"
    supplementary:
      - "Uppercase ratio (spam tends to SHOUT)"
      - "URL count (spam has many links)"
      - "Specific keyword presence (urgency words, financial terms)"
      - "Character repetition (!!!!!)"
      - "Short sentence ratio"
    why_not_embeddings: "Spam detection needs keyword patterns, not semantics"
    
  customer_support_routing:
    primary: "LLM-extracted intent + product mentions"
    supplementary:
      - "Sentiment score (angry → priority routing)"
      - "Text length (verbose usually means complex issue)"
      - "Question count (many questions = needs detailed help)"
      - "Previous interaction count (repeat caller)"
    
  content_recommendation:
    primary: "Document embeddings (content similarity)"
    supplementary:
      - "Topic extraction (for topic-based diversity)"
      - "Reading level (match user's reading level)"
      - "Document length (some users prefer short/long content)"
      - "Freshness (recency of publication)"
      
  fraud_detection:
    primary: "Extracted features (not semantic similarity)"
    features:
      - "Does description match category? (semantic mismatch = suspicious)"
      - "Description length (too short = lazy fake listing)"
      - "Copy-paste detection (same text across multiple listings)"
      - "Language quality score (machine translated text patterns)"
      - "Keyword stuffing score"
```

### Serving Text Features

```yaml
Serving_Architecture:
  pre_computed_embeddings:
    what: "Embed text at ingestion time, serve embedding from feature store"
    when: "Text is known before prediction time (product descriptions, user bios)"
    pipeline:
      1: "New text created/updated → trigger embedding computation"
      2: "Compute embedding (async, GPU-accelerated batch)"
      3: "Store in online feature store (Redis)"
      4: "At serving time: lookup embedding by entity_id (<1ms)"
    benefit: "Zero embedding latency at serving time"
    
  real_time_embedding:
    what: "Embed text at prediction time (for dynamic/new text)"
    when: "Text is part of the prediction request (search query, chat message)"
    infrastructure:
      embedding_service: "GPU-backed gRPC service with sentence-transformers"
      scaling: "Horizontal scaling with GPU instances"
      batching: "Dynamic batching (collect requests, process as batch)"
      latency: "10-30ms per text (acceptable for search, not for high-throughput)"
      caching: "LRU cache on exact text matches (popular queries cached)"
    optimization:
      model_choice: "Smaller model (MiniLM: 384d, 5ms) vs larger (mpnet: 768d, 15ms)"
      quantization: "INT8 quantized model for 2x speedup"
      onnx: "ONNX Runtime for optimized inference"
```

---

## How It Works in Practice

### End-to-End Text Feature Pipeline

```yaml
Pipeline:
  ingestion:
    trigger: "New text created (product listing, review, support ticket)"
    processing:
      1: "Clean text (normalize unicode, strip HTML, lowercase)"
      2: "Compute embedding (sentence-transformers GPU batch)"
      3: "Extract features (length, language, entities)"
      4: "Store all in feature store"
      
  batch_refresh:
    schedule: "Weekly (re-embed with latest model version)"
    scope: "All documents in corpus"
    
  serving:
    lookup: "Fetch pre-computed embedding + extracted features from Redis"
    real_time: "Embed new text (search query) via embedding service"
    
  monitoring:
    embedding_quality: "Spot-check semantic similarity of known-similar pairs"
    model_drift: "Compare embedding distributions across model versions"
    coverage: "% of entities with computed embeddings"
```

---

## Interview Tip

> When asked about text/NLP features: "My approach depends on the use case. For semantic understanding (search, recommendation, similarity): I use pre-trained sentence embeddings — OpenAI text-embedding-3-small for quality, or self-hosted sentence-transformers (all-MiniLM-L6-v2) for cost control. These produce 384-1536 dimensional dense vectors where similar text is close in embedding space. For classification tasks (spam, routing): I often combine lightweight features (text length, keyword counts, TF-IDF on specific n-grams) with embeddings. TF-IDF is still surprisingly effective for keyword-pattern tasks like spam detection. For structured extraction (entities, intent, sentiment): LLM-based extraction with structured outputs, processed in batch (not real-time due to cost/latency). The key production engineering: embed text at ingestion time (not serving time). Store pre-computed embeddings in Redis, serve in <1ms. For real-time text (search queries), I run a GPU-backed embedding service with dynamic batching and LRU caching for popular queries (10-30ms latency). Hybrid features (embedding + extracted signals) give the best of both: semantic understanding + interpretable signals."

---

## Common Mistakes

1. **Embedding at serving time for static text** — Product descriptions don't change, yet you re-embed them on every prediction request. Wastes GPU, adds 15ms latency. Solution: embed once at ingestion, store in feature store, serve from Redis (<1ms).

2. **Using heavy embeddings for simple tasks** — Running a 768-dimensional model for spam detection (where TF-IDF + keyword features work fine). Costs more, not necessarily better. Solution: match embedding complexity to task requirements. Start with simple features, add embeddings only if they improve metrics.

3. **Ignoring text preprocessing** — Feeding raw text with HTML tags, unicode artifacts, inconsistent casing to the embedding model. Embeddings of `<p>Hello World</p>` vs `hello world` are different for no good reason. Solution: standardize preprocessing: strip HTML, normalize unicode, handle encoding issues, consistent casing (or let model handle casing).

4. **Not handling multilingual text** — System encounters French text, embedding model only handles English → garbage embeddings → bad predictions. Solution: use multilingual embedding models (multilingual-e5-large, Cohere embed multilingual) or detect language first and route to appropriate model.

5. **Treating LLM extraction as real-time** — Calling GPT-4o to extract sentiment at serving time (500ms+ latency, $0.01 per call × 1000 requests/sec = $864/day). Solution: LLM-extracted features must be pre-computed in batch. Extract on ingestion, store results, serve from cache. Only extract real-time for low-volume, high-value use cases.

---

## Key Takeaways

- Dense embeddings (sentence-transformers or OpenAI) are the default text feature for semantic tasks
- TF-IDF still effective for keyword-pattern tasks (spam, exact matching, simple classification)
- Hybrid features: embedding + extracted signals (length, entities, sentiment) = best results
- Pre-compute embeddings at ingestion time, serve from Redis (<1ms) at prediction time
- Real-time embedding service: GPU-backed, dynamic batching, LRU cache for popular inputs
- LLM-generated features: batch extract (expensive), cache (deterministic), never real-time at scale
- Model selection: MiniLM (384d, fast, cheap) vs larger models (1024d, better quality, slower)
- Multilingual: use multilingual models or detect language and route appropriately
- Use case matters: search (dense embeddings), spam (TF-IDF), routing (LLM extraction)
- Monitor embedding quality: spot-check similar pairs, track downstream metric impact
