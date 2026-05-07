# RAG Operations

## The Problem / Why This Matters

RAG (Retrieval-Augmented Generation) combines the knowledge retrieval of a search engine with the language generation of an LLM — but deploying RAG reliably is far harder than the tutorial version suggests. The tutorial: chunk documents, embed them, retrieve top-K, stuff into prompt, generate answer. Production reality: documents become stale (your knowledge base updated yesterday but the index wasn't rebuilt), retrieval quality degrades silently (embeddings drift, new document types don't chunk well), answers hallucinate from irrelevant retrieved chunks, and there's no monitoring to catch any of this until users complain. RAG operations encompasses: document ingestion pipelines (processing, chunking, embedding at scale), retrieval quality monitoring (are we finding the right documents?), freshness management (is the knowledge base current?), index operations (rebuilding, updating, versioning), and end-to-end quality tracking (are the final answers correct?). In 2026, RAG is the most common LLM application pattern — used for customer support bots, internal knowledge assistants, code documentation search, and enterprise Q&A. Getting RAG ops right means the difference between a useful assistant that builds trust and a hallucinating liability that erodes it.

---

## The Analogy

Think of RAG operations like running a library's reference desk:

- **Document ingestion** = Cataloging new books. Every new book must be processed (chunked into chapters), tagged (embedded), and shelved in the right section (indexed). If cataloging is wrong, librarians can never find it for patrons.
- **Retrieval** = A patron asks a question, and the librarian searches the catalog to find relevant books/chapters. Poor catalog organization (bad chunking/embedding) means the librarian brings irrelevant books.
- **Generation** = The librarian reads the relevant chapters and synthesizes an answer for the patron. If the retrieved books are wrong, the librarian gives wrong answers (hallucination from bad context).
- **Freshness** = New editions replace old ones, new books arrive daily. If the catalog isn't updated, the librarian works with outdated information (stale index).
- **Monitoring** = Tracking patron satisfaction. Are they finding what they need? Are answers correct? If satisfaction drops, something in the catalog/retrieval/synthesis chain is broken.

---

## Deep Dive

### RAG Pipeline Architecture

```yaml
RAG_Architecture:
  ingestion_pipeline:
    source_connectors:
      - "Document stores (Google Drive, SharePoint, Confluence, Notion)"
      - "Code repositories (GitHub, GitLab)"
      - "Databases (SQL, NoSQL)"
      - "APIs (Zendesk, Salesforce, custom)"
      - "Web scrapers (documentation sites)"
      
    document_processing:
      extraction: "Convert to text (PDF parsing, OCR, table extraction)"
      cleaning: "Remove headers/footers, navigation, duplicates"
      chunking: "Split into retrieval-sized pieces"
      metadata: "Extract title, date, author, section, source URL"
      
    embedding:
      models: "text-embedding-3-large (OpenAI), E5-Mistral, BGE-M3, Cohere Embed v3"
      batch_processing: "Embed thousands of chunks efficiently"
      storage: "Vector database (Pinecone, Weaviate, Qdrant, Milvus, pgvector)"
      
  retrieval_pipeline:
    query_processing:
      - "Query expansion (add synonyms, related terms)"
      - "Query rewriting (LLM reformulates for better retrieval)"
      - "Multi-query (generate multiple search angles)"
      
    retrieval_strategies:
      dense_retrieval: "Embed query, find nearest vectors (semantic similarity)"
      sparse_retrieval: "BM25 keyword matching (good for exact terms, names, codes)"
      hybrid: "Combine dense + sparse with Reciprocal Rank Fusion (RRF)"
      
    post_retrieval:
      reranking: "Cross-encoder model scores relevance of each retrieved chunk"
      filtering: "Remove low-relevance, duplicate, or stale results"
      context_assembly: "Combine top chunks into generation context"
      
  generation:
    prompt: "System prompt + retrieved context + user query → LLM generates answer"
    citation: "Include source references in the answer"
    confidence: "Indicate when information might be incomplete"
```

### Chunking Strategies

```yaml
Chunking:
  why_it_matters: |
    Chunk too small: lose context, retrieval finds fragments without meaning
    Chunk too large: dilute relevant info with noise, exceed context window
    Chunk poorly: split mid-sentence, break tables, separate heading from content
    
  strategies:
    fixed_size:
      what: "Split every N tokens/characters with overlap"
      params: "chunk_size=512 tokens, overlap=50 tokens"
      pros: "Simple, predictable"
      cons: "Splits mid-paragraph, ignores document structure"
      
    recursive_text_splitting:
      what: "Split by hierarchy: sections → paragraphs → sentences → characters"
      logic: |
        Try splitting by "\n\n" (paragraphs)
        If chunks too large, split by "\n" (lines)
        If still too large, split by ". " (sentences)
        Maintain overlap between chunks
      tools: "LangChain RecursiveCharacterTextSplitter"
      
    semantic_chunking:
      what: "Use embedding similarity to find natural break points"
      logic: |
        Embed each sentence
        Find sentences where embedding similarity drops (topic shift)
        Split at those boundaries
      benefit: "Chunks align with topic boundaries"
      tools: "LangChain SemanticChunker, LlamaIndex SentenceWindowRetriever"
      
    document_structure:
      what: "Respect document hierarchy (sections, subsections, headers)"
      logic: |
        Split by headers (H1 → major sections, H2 → subsections)
        Keep each section as one chunk (if fits in context)
        If section too large, split at paragraph boundaries
      benefit: "Preserves logical structure, heading provides context"
      tools: "MarkdownHeaderTextSplitter, HTML section parser"
      
    parent_document_retriever:
      what: "Store small chunks for retrieval but return larger parent context"
      logic: |
        Split into small chunks (128 tokens) for precise retrieval
        When a small chunk matches, return its parent section (1024 tokens)
      benefit: "Precise retrieval + sufficient generation context"
      
  operational_considerations:
    chunk_size_tuning: "Test different sizes on YOUR retrieval evaluation set"
    overlap: "10-20% overlap prevents losing context at chunk boundaries"
    metadata_enrichment: "Attach source, date, section title to each chunk"
    deduplication: "Detect and remove near-duplicate chunks across sources"
```

### Retrieval Quality Operations

```yaml
Retrieval_Quality:
  evaluation_metrics:
    recall_at_k:
      what: "Of all relevant documents, what fraction is in the top-K retrieved?"
      formula: "relevant_in_top_k / total_relevant"
      target: "Recall@10 > 0.9 (find 90% of relevant docs in top 10)"
      
    precision_at_k:
      what: "Of the top-K retrieved, what fraction is actually relevant?"
      formula: "relevant_in_top_k / k"
      target: "Precision@5 > 0.7 (70% of top 5 are relevant)"
      
    mrr:
      name: "MRR (Mean Reciprocal Rank)"
      what: "How high is the first relevant result ranked?"
      formula: "1 / rank_of_first_relevant_result (averaged over queries)"
      target: "MRR > 0.7 (relevant result usually in top 2)"
      
    ndcg:
      name: "NDCG (Normalized Discounted Cumulative Gain)"
      what: "How well are results ordered by relevance?"
      target: "NDCG@10 > 0.8"
      
  evaluation_dataset:
    structure: |
      queries:
        - query: "What is our refund policy for digital products?"
          relevant_doc_ids: ["doc_456", "doc_789"]
          relevant_chunks: ["chunk_456_3", "chunk_789_1"]
          
        - query: "How do I configure SSO for enterprise accounts?"
          relevant_doc_ids: ["doc_123"]
          relevant_chunks: ["chunk_123_7", "chunk_123_8"]
          
    creation_methods:
      manual: "Domain experts label relevant documents for sample queries"
      synthetic: "LLM generates questions from existing documents (then human validates)"
      user_feedback: "Track which retrieved docs users click/rate useful"
      
  monitoring:
    retrieval_metrics:
      - "Similarity score distribution (are scores trending lower?)"
      - "Empty result rate (queries returning no relevant results)"
      - "Diversity of sources (are results coming from one document repeatedly?)"
      - "Latency (vector search time, reranking time)"
      
    signals_of_degradation:
      - "Average similarity score dropping week-over-week"
      - "Increasing 'I don't know' responses from LLM"
      - "User thumbs-down rate increasing"
      - "Retrieved chunks increasingly stale (old dates)"
```

### Document Freshness and Index Management

```yaml
Freshness:
  problem: |
    Knowledge bases change: documents are updated, new documents added, old ones deprecated.
    If the index isn't kept current, RAG answers become stale/wrong.
    Example: Policy changed on Jan 15. Index last rebuilt on Jan 1. 
    All answers about that policy are WRONG until index is updated.
    
  freshness_strategies:
    full_rebuild:
      what: "Delete and rebuild entire index periodically"
      frequency: "Daily or weekly"
      pros: "Simple, guaranteed consistency"
      cons: "Expensive for large collections, downtime during rebuild"
      use: "Small-medium collections (< 100K documents)"
      
    incremental_updates:
      what: "Only process new/changed/deleted documents"
      implementation: |
        1. Track document modification timestamps
        2. On schedule (hourly): query for changed docs since last sync
        3. Re-chunk and re-embed only changed documents
        4. Upsert new vectors, delete removed document vectors
      pros: "Efficient for large collections"
      cons: "More complex, potential for stale vectors if sync fails"
      
    streaming_updates:
      what: "Process document changes in real-time via event stream"
      implementation: |
        - Source system emits events: doc_created, doc_updated, doc_deleted
        - Processing pipeline: extract → chunk → embed → upsert (within minutes)
      tools: "Kafka/Pub-Sub for events, Airflow/Dagster for pipeline"
      pros: "Near-real-time freshness (minutes, not hours)"
      cons: "Complex infrastructure, potential ordering issues"
      
  staleness_monitoring:
    metrics:
      - "Time since last index update (should be < threshold for freshness SLA)"
      - "Percentage of documents modified but not re-indexed"
      - "Average document age in index vs average document age in source"
    alerts:
      - "Index staleness > 24 hours → P2 alert"
      - "More than 10% of docs out-of-sync → P1 alert"
      - "Ingestion pipeline failure → immediate alert"
      
  index_versioning:
    what: "Maintain multiple index versions for safe updates"
    pattern: |
      1. Production serves from index_v1
      2. Build new index_v2 (full rebuild or major changes)
      3. Evaluate v2 quality against golden dataset
      4. If quality ≥ v1: swap production to v2 (atomic switch)
      5. Keep v1 available for rollback (7 days)
    benefit: "Zero-downtime index updates, safe rollback"
```

### End-to-End RAG Monitoring

```python
# RAG monitoring pipeline — track quality at every stage

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class RAGTrace:
    """Full trace of a RAG request for monitoring."""
    trace_id: str
    timestamp: datetime
    
    # Query stage
    original_query: str
    rewritten_query: Optional[str]
    
    # Retrieval stage
    retrieved_chunks: list[dict]  # [{id, text, score, source, date}]
    retrieval_latency_ms: float
    top_similarity_score: float
    num_results: int
    
    # Reranking stage
    reranked_chunks: list[dict]
    reranking_latency_ms: float
    
    # Generation stage
    context_tokens: int
    generated_answer: str
    generation_latency_ms: float
    model_used: str
    total_tokens: int
    
    # Quality signals
    user_feedback: Optional[str]  # "thumbs_up", "thumbs_down", None
    cited_sources: list[str]
    answer_confidence: Optional[float]


class RAGMonitor:
    """Monitor RAG quality across all stages."""
    
    def analyze_traces(self, traces: list[RAGTrace]) -> dict:
        """Compute aggregate metrics from traces."""
        return {
            "retrieval": {
                "avg_top_score": mean([t.top_similarity_score for t in traces]),
                "empty_result_rate": sum(1 for t in traces if t.num_results == 0) / len(traces),
                "avg_retrieval_latency_ms": mean([t.retrieval_latency_ms for t in traces]),
            },
            "generation": {
                "avg_generation_latency_ms": mean([t.generation_latency_ms for t in traces]),
                "avg_context_tokens": mean([t.context_tokens for t in traces]),
                "avg_total_tokens": mean([t.total_tokens for t in traces]),
            },
            "quality": {
                "thumbs_up_rate": self._feedback_rate(traces, "thumbs_up"),
                "thumbs_down_rate": self._feedback_rate(traces, "thumbs_down"),
                "no_citation_rate": sum(1 for t in traces if not t.cited_sources) / len(traces),
            },
        }
    
    def detect_degradation(self, current_metrics: dict, baseline_metrics: dict) -> list[str]:
        """Alert on quality degradation vs baseline."""
        alerts = []
        
        if current_metrics["retrieval"]["avg_top_score"] < baseline_metrics["retrieval"]["avg_top_score"] * 0.9:
            alerts.append("Retrieval quality degraded: avg similarity score dropped >10%")
            
        if current_metrics["quality"]["thumbs_down_rate"] > baseline_metrics["quality"]["thumbs_down_rate"] * 1.5:
            alerts.append("User dissatisfaction increased: thumbs-down rate up >50%")
            
        if current_metrics["retrieval"]["empty_result_rate"] > 0.1:
            alerts.append("High empty result rate: >10% of queries return no results")
            
        return alerts
```

### Common RAG Failure Modes

```yaml
Failure_Modes:
  retrieval_failures:
    wrong_chunks_retrieved:
      symptom: "Answer is about wrong topic despite query being clear"
      causes:
        - "Embedding model doesn't capture domain semantics"
        - "Chunks too large (relevant info diluted)"
        - "No metadata filtering (returning docs from wrong category)"
      fix: "Better embedding model, smaller chunks, metadata filters, reranking"
      
    relevant_doc_not_found:
      symptom: "LLM says 'I don't have information about...' when doc exists"
      causes:
        - "Document not yet indexed (freshness lag)"
        - "Query terms don't match document semantics"
        - "Chunk boundary splits relevant content"
      fix: "Check freshness, query rewriting, overlap in chunking"
      
    too_many_irrelevant_results:
      symptom: "Context is stuffed with noise, answer quality is poor"
      causes:
        - "Similarity threshold too low"
        - "No reranking step (top-K by vector distance alone)"
      fix: "Add reranker (cross-encoder), raise similarity threshold"
      
  generation_failures:
    hallucination_from_context:
      symptom: "Answer states something NOT in retrieved context"
      cause: "LLM uses parametric knowledge instead of context"
      fix: "Stronger instruction: 'Only answer from the provided context'"
      monitoring: "Check if answer claims are supported by retrieved text"
      
    ignoring_relevant_context:
      symptom: "Relevant info is in context but LLM ignores it"
      cause: "Context too long, relevant info in the middle (lost in the middle problem)"
      fix: "Put most relevant chunks first, use reranking, limit context size"
      
    attribution_failure:
      symptom: "Answer is correct but doesn't cite sources"
      cause: "Prompt doesn't explicitly require citations"
      fix: "Add instruction: 'Cite sources using [Source: title] format'"
```

---

## How It Works in Practice

### RAG Operations Checklist

```yaml
Operations_Checklist:
  daily:
    - "Check ingestion pipeline health (any failures?)"
    - "Monitor retrieval score trends (degrading?)"
    - "Review user feedback (thumbs down rate?)"
    - "Verify freshness SLA met (index updated within threshold)"
    
  weekly:
    - "Run retrieval evaluation on golden dataset"
    - "Analyze failure patterns (what types of queries fail?)"
    - "Check embedding model performance (drift?)"
    - "Review cost (token usage, vector DB cost, embedding cost)"
    
  monthly:
    - "Full index rebuild and quality comparison"
    - "Evaluate new embedding models (has a better one been released?)"
    - "Review and update chunking strategy (new document types?)"
    - "Update golden evaluation dataset (new query patterns?)"
```

---

## Interview Tip

> When asked about RAG operations: "Production RAG has three operational dimensions: (1) Ingestion ops — chunking strategy (I use recursive splitting with 512-token chunks and 50-token overlap for most documents, semantic chunking for unstructured content), incremental updates for freshness (detect changed docs via timestamps, re-embed only changes, staleness SLA < 24 hours), and metadata enrichment (source, date, category for filtering). (2) Retrieval quality — hybrid search (dense embeddings + BM25 sparse retrieval combined via Reciprocal Rank Fusion), cross-encoder reranking (reorders top 20 candidates by relevance), and monitoring (track similarity score distribution, empty result rate, retrieval latency). (3) End-to-end quality — RAG-specific evaluation metrics (faithfulness: does answer match context, relevance: is context relevant to query, answer quality: is answer helpful), tracing every request (query → retrieved chunks → generated answer → user feedback), and detecting degradation (similarity scores dropping, thumbs-down rate increasing). Key failure modes I watch for: stale index (freshness lag), lost-in-the-middle (relevant context ignored when too much context is provided), and hallucination from parametric knowledge (model ignoring retrieved context)."

---

## Common Mistakes

1. **No retrieval quality evaluation** — Building RAG and only evaluating the final answer. If retrieval is bad, no amount of prompt engineering fixes the answer. Create a retrieval evaluation set (query → relevant chunks) and measure Recall@K, MRR, and NDCG separately from generation quality.

2. **One-size-fits-all chunking** — Using 512-token fixed chunks for every document type. A FAQ page should be chunked by question-answer pair. A legal document should be chunked by section. A table should never be split across chunks. Adapt chunking strategy to document structure.

3. **No freshness monitoring** — Index was built last month. Documents updated since then contain the correct answers. Users get outdated information and lose trust. Implement staleness tracking and automated re-indexing on source changes.

4. **Stuffing too much context** — Retrieving 20 chunks (5000 tokens of context) when 3-5 relevant chunks (1000 tokens) would suffice. More context means: higher cost, slower generation, and "lost in the middle" where the model ignores information in the middle of long contexts. Use reranking and take only the top 3-5 most relevant chunks.

5. **Not implementing citations** — Users can't verify the answer against source documents. This makes the system untrustable for important decisions. Always include source references (document title, URL, section) in the generated answer so users can verify.

---

## Key Takeaways

- RAG ops has three dimensions: ingestion (chunking, embedding, freshness), retrieval (quality, monitoring), generation (faithfulness, citations)
- Chunking matters enormously: recursive text splitting for general docs, semantic chunking for unstructured, structure-aware for formatted docs
- Hybrid retrieval (dense + sparse + reranking) consistently outperforms pure vector search
- Freshness: implement incremental updates with staleness SLA (<24 hours for most applications)
- Monitor retrieval quality separately from generation quality (track Recall@K, MRR, similarity distributions)
- Common failure: stale index, lost-in-the-middle, hallucination from parametric knowledge
- Evaluation: create golden dataset (query → relevant chunks → expected answer), run regularly
- Reranking (cross-encoder): essential for production — reorders top-K by true relevance
- Citations: always include source references for user trust and verifiability
- RAG is an operational system: monitor daily, evaluate weekly, rebuild/upgrade monthly
