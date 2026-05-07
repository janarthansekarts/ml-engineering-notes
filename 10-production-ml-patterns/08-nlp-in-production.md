# NLP in Production

## The Problem / Why This Matters

Natural Language Processing (NLP) in production goes far beyond training a text classification model in a notebook. Production NLP systems must handle: high-throughput text processing (millions of documents daily), multi-language support, evolving vocabulary, latency constraints (real-time classification in < 50ms), and the challenge of combining traditional NLP techniques with modern LLM (Large Language Model) capabilities. In 2026, production NLP spans two worlds: (1) classical NLP tasks at scale — NER (Named Entity Recognition), text classification, sentiment analysis, information extraction — where speed and cost efficiency demand smaller specialized models, and (2) LLM-powered NLP — where foundation models handle complex reasoning, summarization, and generation but at higher cost and latency. The engineering challenge is choosing the right approach for each task: a fine-tuned BERT model classifies text at 10,000 requests/second for $0.001 per 1000 requests, while GPT-5 does it at 50 requests/second for $0.50 per 1000 requests. Production NLP engineering is about knowing when to use each approach, building efficient pipelines that combine both, and handling the unique challenges of text data (tokenization, encoding, language detection, text normalization).

---

## The Analogy

Think of production NLP like a document processing center at a large company:

- **Small specialized models** = Expert clerks who each handle one task extremely fast. One clerk only stamps "urgent" or "normal" on letters (classification). Another extracts names and addresses (NER). They're fast because they only do one thing.
- **LLMs** = A senior executive who can handle any document task — summarize, translate, analyze, draft responses. Brilliant but expensive and slow. You don't send every letter to the CEO.
- **Production NLP system** = An efficient organization. Clerks handle routine processing (90% of volume). The executive handles complex cases (10%). Smart routing decides who gets what.

---

## Deep Dive

### NLP Task Categories in Production

```yaml
Production_NLP_Tasks:
  classification:
    tasks:
      sentiment: "Positive/negative/neutral (customer reviews, social media)"
      intent: "What does user want? (chatbot routing, support ticket triage)"
      topic: "What category? (news categorization, document routing)"
      spam: "Is this spam? (email filtering, content moderation)"
      toxicity: "Is this harmful? (content safety, comment moderation)"
    
    approach_2026:
      high_volume: "Fine-tuned DistilBERT or deberta-v3 (fast, cheap, 95%+ accuracy)"
      complex: "LLM with few-shot prompting (flexible, no training data needed)"
      
  extraction:
    tasks:
      ner: "Extract entities: names, organizations, locations, dates, amounts"
      relation: "Extract relationships between entities"
      key_value: "Extract structured data from unstructured text"
      event: "Extract events (who did what, when, where)"
    
    approach_2026:
      high_volume: "Fine-tuned token classification model (SpaCy, HuggingFace)"
      complex: "LLM structured extraction with JSON schema output"
      
  generation:
    tasks:
      summarization: "Condense long documents into key points"
      translation: "Cross-language translation"
      paraphrase: "Rewrite text (simplification, style transfer)"
      completion: "Complete partial text (autocomplete, drafting)"
    
    approach_2026:
      all: "LLM-based (GPT-5, Claude 4, Gemini 2.5) — generation is LLM territory"
      optimization: "Smaller models (Phi-4, Gemma-3) for high-volume, lower-quality needs"
      
  embeddings:
    tasks:
      semantic_search: "Find similar documents/passages"
      clustering: "Group similar texts together"
      deduplication: "Find near-duplicate content"
      rag_retrieval: "Retrieve relevant context for LLM"
    
    approach_2026:
      models: "E5-large, BGE-M3, Cohere Embed v3, text-embedding-3-large"
      serving: "Batch compute embeddings → vector index (Pinecone, Qdrant, Weaviate)"
```

### Model Selection: When to Use What

```yaml
Model_Selection:
  use_small_specialized_model:
    when:
      - "High volume (> 10K requests/second)"
      - "Simple task (binary classification, NER with known entities)"
      - "Cost-sensitive (< $0.01 per 1000 requests needed)"
      - "Latency-critical (< 10ms per request)"
      - "Enough training data available (> 1000 labeled examples)"
    models:
      - "DistilBERT (6 layers, 66M params, fast inference)"
      - "DeBERTa-v3-base (good accuracy/speed trade-off)"
      - "SpaCy (fast NER, dependency parsing)"
      - "Sentence-transformers (embeddings)"
    
  use_llm:
    when:
      - "Complex reasoning required (nuanced classification)"
      - "Zero/few-shot (no training data available)"
      - "Generation task (summarization, translation, drafting)"
      - "Flexible schema (extraction varies per document type)"
      - "Multi-step reasoning (chain-of-thought needed)"
    models:
      - "GPT-5 / Claude 4 (highest capability, highest cost)"
      - "GPT-4.1 / Claude 3.5 Sonnet (good balance)"
      - "Phi-4 / Gemma-3 (smaller, self-hosted, lower cost)"
      
  use_hybrid:
    when:
      - "Routing: classify first (fast model), then process (LLM for complex cases)"
      - "Enrichment: NER extraction (fast model) + summarization (LLM)"
      - "Validation: LLM generates + small model validates"
```

### Implementation

```python
# Production NLP pipeline implementation

"""
Production NLP system combining specialized models and LLMs.
Handles high-throughput text processing with intelligent routing.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import logging
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class TextDocument:
    """Input document for NLP processing."""
    doc_id: str
    text: str
    language: str = "en"
    metadata: Dict = None


@dataclass
class NLPResult:
    """NLP processing result."""
    doc_id: str
    task: str
    result: Any
    confidence: float
    model_used: str
    latency_ms: float
    cost_estimate: float = 0.0


class ProductionNLPPipeline:
    """
    Production NLP pipeline with:
    - Language detection
    - Task routing (specialized model vs. LLM)
    - Batching for throughput
    - Caching for repeated texts
    - Fallback handling
    """
    
    def __init__(
        self,
        classifier_model,
        ner_model,
        embedding_model,
        llm_client,
        cache_store,
    ):
        self.classifier = classifier_model
        self.ner = ner_model
        self.embedder = embedding_model
        self.llm = llm_client
        self.cache = cache_store
    
    def classify(
        self, documents: List[TextDocument], task: str = "sentiment"
    ) -> List[NLPResult]:
        """
        Classify documents using appropriate model.
        
        Routing:
        - Short, standard text → specialized model (fast, cheap)
        - Long, complex, or low-confidence → LLM (accurate, expensive)
        """
        results = []
        llm_batch = []  # Documents that need LLM processing
        
        for doc in documents:
            # Check cache
            cache_key = self._cache_key(doc.text, task)
            cached = self.cache.get(cache_key)
            if cached:
                results.append(NLPResult(
                    doc_id=doc.doc_id,
                    task=task,
                    result=cached["result"],
                    confidence=cached["confidence"],
                    model_used="cache",
                    latency_ms=0.1,
                ))
                continue
            
            # Try specialized model first
            start = time.perf_counter()
            prediction, confidence = self.classifier.predict(doc.text, task=task)
            latency = (time.perf_counter() - start) * 1000
            
            if confidence >= 0.85:
                # High confidence → use specialized model result
                result = NLPResult(
                    doc_id=doc.doc_id,
                    task=task,
                    result=prediction,
                    confidence=confidence,
                    model_used="specialized_classifier",
                    latency_ms=latency,
                    cost_estimate=0.000001,
                )
                results.append(result)
                self.cache.put(cache_key, {"result": prediction, "confidence": confidence})
            else:
                # Low confidence → escalate to LLM
                llm_batch.append(doc)
        
        # Process LLM batch (if any)
        if llm_batch:
            llm_results = self._classify_with_llm(llm_batch, task)
            results.extend(llm_results)
        
        return results
    
    def extract_entities(
        self, documents: List[TextDocument]
    ) -> List[NLPResult]:
        """
        Extract named entities from documents.
        
        Uses SpaCy/transformer NER for standard entities.
        Falls back to LLM for domain-specific extraction.
        """
        results = []
        
        for doc in documents:
            start = time.perf_counter()
            
            # Standard NER
            entities = self.ner.extract(doc.text, language=doc.language)
            latency = (time.perf_counter() - start) * 1000
            
            results.append(NLPResult(
                doc_id=doc.doc_id,
                task="ner",
                result=entities,
                confidence=0.9,
                model_used="ner_model",
                latency_ms=latency,
                cost_estimate=0.000005,
            ))
        
        return results
    
    def embed(
        self, documents: List[TextDocument], batch_size: int = 64
    ) -> List[NLPResult]:
        """
        Compute embeddings for documents (batched for GPU efficiency).
        """
        results = []
        
        # Process in batches for GPU efficiency
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc.text for doc in batch]
            
            start = time.perf_counter()
            embeddings = self.embedder.encode(texts)
            latency = (time.perf_counter() - start) * 1000
            
            per_doc_latency = latency / len(batch)
            
            for doc, embedding in zip(batch, embeddings):
                results.append(NLPResult(
                    doc_id=doc.doc_id,
                    task="embedding",
                    result=embedding.tolist(),
                    confidence=1.0,
                    model_used="embedding_model",
                    latency_ms=per_doc_latency,
                    cost_estimate=0.00001,
                ))
        
        return results
    
    def summarize(
        self, documents: List[TextDocument], max_length: int = 200
    ) -> List[NLPResult]:
        """
        Summarize documents using LLM.
        Generation tasks always use LLM.
        """
        results = []
        
        for doc in documents:
            start = time.perf_counter()
            
            summary = self.llm.generate(
                prompt=f"Summarize the following text in {max_length} words or less:\n\n{doc.text}",
                max_tokens=max_length * 2,  # Tokens ≈ 0.75 words
                temperature=0.3,
            )
            
            latency = (time.perf_counter() - start) * 1000
            
            results.append(NLPResult(
                doc_id=doc.doc_id,
                task="summarization",
                result=summary,
                confidence=0.9,
                model_used="llm",
                latency_ms=latency,
                cost_estimate=0.001,  # ~1000 tokens
            ))
        
        return results
    
    def _classify_with_llm(
        self, documents: List[TextDocument], task: str
    ) -> List[NLPResult]:
        """Classify using LLM (for complex/uncertain cases)."""
        results = []
        
        task_prompts = {
            "sentiment": "Classify the sentiment as positive, negative, or neutral",
            "intent": "Classify the user intent into one of: question, complaint, request, feedback",
            "topic": "Classify the topic of this text",
        }
        
        for doc in documents:
            start = time.perf_counter()
            
            response = self.llm.generate(
                prompt=f"{task_prompts.get(task, 'Classify this text')}:\n\n{doc.text}\n\nClassification:",
                max_tokens=20,
                temperature=0.0,
            )
            
            latency = (time.perf_counter() - start) * 1000
            
            results.append(NLPResult(
                doc_id=doc.doc_id,
                task=task,
                result=response.strip(),
                confidence=0.8,  # LLM confidence estimation is separate
                model_used="llm_classifier",
                latency_ms=latency,
                cost_estimate=0.0005,
            ))
        
        return results
    
    def _cache_key(self, text: str, task: str) -> str:
        """Generate cache key from text content and task."""
        return hashlib.sha256(f"{task}:{text[:500]}".encode()).hexdigest()[:20]


class TextPreprocessingPipeline:
    """
    Text preprocessing for production NLP.
    
    Handles the messy reality of production text:
    - Mixed encodings
    - HTML/markdown artifacts
    - Multiple languages
    - PII (Personally Identifiable Information) that must be redacted
    - Varying lengths (too short = noise, too long = truncation)
    """
    
    def __init__(self, max_length: int = 512, language_detector=None):
        self.max_length = max_length
        self.language_detector = language_detector
    
    def preprocess(self, text: str) -> Dict:
        """
        Clean and normalize text for model input.
        """
        # Step 1: Encoding normalization
        text = self._normalize_encoding(text)
        
        # Step 2: Remove HTML/markdown
        text = self._strip_markup(text)
        
        # Step 3: Language detection
        language = "en"
        if self.language_detector:
            language = self.language_detector.detect(text)
        
        # Step 4: Length handling
        is_truncated = False
        if len(text.split()) > self.max_length:
            text = " ".join(text.split()[:self.max_length])
            is_truncated = True
        
        # Step 5: Basic normalization
        text = self._normalize_whitespace(text)
        
        return {
            "text": text,
            "language": language,
            "is_truncated": is_truncated,
            "word_count": len(text.split()),
        }
    
    def _normalize_encoding(self, text: str) -> str:
        """Fix encoding issues (mojibake, etc.)."""
        # Handle common encoding errors
        try:
            # Fix double-encoded UTF-8
            text = text.encode("utf-8").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
        return text
    
    def _strip_markup(self, text: str) -> str:
        """Remove HTML tags and markdown formatting."""
        import re
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove markdown formatting
        text = re.sub(r'[*_~`#]', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '[URL]', text)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Collapse multiple spaces/newlines."""
        import re
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
```

### High-Throughput NLP Serving

```yaml
Serving_Patterns:
  batch_processing:
    when: "Process millions of documents (no latency requirement)"
    approach: "GPU batch inference (batch_size=128-512)"
    throughput: "10,000-50,000 documents/minute per GPU"
    cost: "~$0.001 per 1000 documents (small model)"
    
  online_serving:
    when: "Real-time classification needed (< 50ms)"
    approach: "ONNX Runtime or TensorRT-optimized model"
    optimization:
      quantization: "INT8 (2-4x speedup)"
      distillation: "DistilBERT instead of BERT (2x faster, 97% accuracy)"
      caching: "Cache frequent inputs (30-50% hit rate for common queries)"
    throughput: "5,000-20,000 requests/second per GPU"
    
  embedding_at_scale:
    when: "Embed millions of documents for vector search"
    approach: "Batch GPU inference → store in vector database"
    pre_compute: "Embed all documents offline (batch)"
    online: "Only embed queries at serving time (single embedding, fast)"
    
  llm_serving:
    when: "Generation tasks (summarization, complex classification)"
    approach: "vLLM or TGI (Text Generation Inference) with continuous batching"
    optimization:
      kv_cache: "PagedAttention for efficient memory"
      speculative: "Speculative decoding for faster generation"
      quantization: "AWQ/GPTQ for reduced memory (4-bit)"
    throughput: "50-500 requests/second depending on model size"
```

---

## How It Works in Practice

### Customer Support Ticket Processing

```yaml
Support_Pipeline:
  input: "10,000 support tickets per day"
  
  pipeline:
    step_1_preprocessing:
      action: "Clean text, detect language, redact PII"
      output: "Normalized text ready for models"
      
    step_2_classification:
      action: "Classify intent (billing, technical, account, general)"
      model: "Fine-tuned DeBERTa-v3-base (98% accuracy)"
      latency: "< 10ms"
      
    step_3_entity_extraction:
      action: "Extract: order_id, product_name, issue_type"
      model: "SpaCy + custom NER model"
      latency: "< 15ms"
      
    step_4_sentiment:
      action: "Detect customer emotion (frustrated, neutral, positive)"
      model: "Fine-tuned DistilBERT"
      latency: "< 5ms"
      
    step_5_priority:
      action: "Combine intent + sentiment + entity → priority score"
      model: "XGBoost on extracted features"
      
    step_6_routing:
      action: "Route to correct team based on classification"
      
    step_7_draft_response:
      action: "LLM generates draft response for agent"
      model: "GPT-4.1 with template and context"
      latency: "2-5 seconds (async, not blocking)"
```

---

## Interview Tip

> When asked about NLP in production: "I design NLP systems with a model selection strategy based on volume, latency, and complexity. For high-volume classification (> 10K req/sec): fine-tuned DistilBERT or DeBERTa-v3-base — these handle sentiment, intent, and spam detection at < 10ms latency and $0.001/1000 requests. For NER (Named Entity Recognition) at scale: SpaCy transformer pipeline or fine-tuned token-classification model. For embeddings: batch-compute with E5-large or BGE-M3, store in vector index (Qdrant/Pinecone). For generation (summarization, drafting): LLMs with vLLM serving. Key engineering patterns: (1) Cascade routing — specialized model classifies with confidence threshold. High-confidence (> 0.85): use specialized model result. Low-confidence: escalate to LLM. This handles 85% of traffic cheaply, only 15% needs expensive LLM. (2) Batching — group texts for GPU efficiency (batch_size=128 gives 10x throughput vs. single inference). (3) Preprocessing pipeline — encoding normalization, HTML stripping, language detection, PII redaction, length truncation. Production text is messy — models trained on clean data fail without preprocessing. (4) Caching — many texts repeat (common queries, standard messages). 30-50% cache hit rate saves significant compute."

---

## Common Mistakes

1. **Using LLM for everything** — Calling GPT-5 for binary sentiment classification at $0.50/1000 requests when DistilBERT does it at $0.001/1000 with 97% accuracy. Solution: right-size models. Use LLM only when specialized model can't handle the complexity.

2. **Ignoring preprocessing** — Raw user text goes straight to model. Fails on HTML fragments, special characters, excessive whitespace, wrong encoding. Solution: robust preprocessing pipeline before any model — normalize encoding, strip markup, detect language, handle length.

3. **Not handling language diversity** — English-only model deployed globally. Fails on Spanish/French/Chinese inputs. Solution: language detection → route to appropriate model (multilingual or language-specific). Multilingual models (XLM-RoBERTa, mBERT) handle 100+ languages.

4. **No batching for throughput** — Processing one document at a time on GPU. GPU utilization at 5%. Solution: batch texts (batch_size 64-512 depending on model/GPU). 10x throughput improvement from batching alone.

5. **Not versioning training data** — Model retrained on new data, performance drops. Can't reproduce old model or identify what changed. Solution: version training data alongside model (DVC, MLflow datasets). Track which data each model version was trained on.

---

## Key Takeaways

- Production NLP: specialize models by task — don't use LLM for everything
- Classification at scale: fine-tuned DistilBERT/DeBERTa (< 10ms, $0.001/1K requests)
- NER: SpaCy or HuggingFace token classification (fast, domain-specific)
- Embeddings: E5-large/BGE-M3 (batch compute → vector index, serve online)
- Generation: LLMs only (summarization, translation, drafting — vLLM serving)
- Cascade routing: specialized model first (85% traffic), LLM for uncertain cases (15%)
- Preprocessing: normalize encoding, strip markup, detect language, handle length, redact PII
- Batching: GPU efficiency requires batching (batch_size 64-512 for 10x throughput)
- Caching: 30-50% of texts repeat — cache results for instant response
- Cost efficiency: specialized model at $0.001/1K vs. LLM at $0.50/1K (500x difference)
