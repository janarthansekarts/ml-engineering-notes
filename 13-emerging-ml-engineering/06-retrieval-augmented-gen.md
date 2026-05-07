# Retrieval-Augmented Generation Engineering

## The Problem / Why This Matters

LLMs (Large Language Models) have a fundamental limitation: they only know what was in their training data. If you ask about events after the training cutoff, proprietary company data, or rapidly changing information, the model either hallucinates a confident-sounding wrong answer or admits ignorance. RAG (Retrieval-Augmented Generation) solves this by giving the LLM access to external knowledge at inference time — retrieve relevant documents, include them in the prompt, and generate an answer grounded in retrieved evidence. In 2026, RAG is the single most deployed LLM pattern in enterprise AI: every company building AI assistants, search systems, or knowledge bases uses RAG. But building production RAG that actually works well is surprisingly hard. Naive RAG (chunk documents → embed → retrieve top-K → generate) achieves maybe 60-70% accuracy. Production RAG with proper engineering (hybrid search, re-ranking, query expansion, agentic retrieval, evaluation) achieves 90%+. The gap between naive and production RAG is entirely engineering — chunking strategy, embedding model choice, retrieval pipeline design, prompt engineering, evaluation methodology, and handling edge cases (no relevant documents, conflicting information, multi-hop reasoning). RAG engineering is now a core competency for ML engineers building LLM-powered systems.

---

## The Analogy

Think of RAG like a university student taking an open-book exam:

- **Closed-book (vanilla LLM)** = Student relies entirely on memory. Great for well-known facts but fails on specific details, recent events, or obscure topics. Might confidently give wrong answers (hallucination = false memory).
- **Open-book (RAG)** = Student can consult their textbook during the exam. Much more accurate on specific details. Quality depends on: Can they find the right page quickly (retrieval)? Do they understand what they read (comprehension)? Do they integrate information across multiple pages (multi-hop reasoning)?
- **Bad RAG** = Student with a poorly organized textbook and bad index. Spends most of the exam flipping through irrelevant pages, sometimes finds related-but-wrong content, or gives up and answers from memory anyway.
- **Good RAG** = Student with a well-organized textbook, color-coded tabs, a comprehensive index, and practice using it efficiently. Quickly finds exactly the right section, cross-references related content, and produces well-grounded answers.

---

## Deep Dive

### RAG Architecture Layers

```yaml
RAG_Architecture:
  layer_1_data_ingestion:
    document_loading:
      sources: ["PDFs", "HTML", "Docs", "Confluence", "Notion", "Databases", "APIs"]
      parsers: ["Unstructured.io", "LlamaParse", "Azure Document Intelligence"]
      challenge: "Tables, images, code blocks require special handling"
      
    chunking:
      strategies:
        fixed_size:
          what: "Split by token count (512-1024 tokens)"
          pro: "Simple, predictable"
          con: "Cuts across sentence/topic boundaries"
          
        semantic:
          what: "Split by topic/section (using embeddings or structure)"
          pro: "Each chunk is a coherent topic"
          con: "Variable size, harder to implement"
          method: "Embedding similarity between sentences — split when similarity drops"
          
        recursive:
          what: "Try large chunks, recursively split if too big"
          pro: "Respects document structure (headings, paragraphs)"
          tools: "LangChain RecursiveCharacterTextSplitter"
          
        document_structure:
          what: "Use headings, sections, paragraphs as natural boundaries"
          pro: "Preserves author's intended structure"
          con: "Requires structured documents"
          
      overlap:
        purpose: "Context at chunk boundaries isn't lost"
        typical: "100-200 tokens overlap between chunks"
        
      metadata:
        what: "Attach source info to each chunk"
        fields: ["document_title", "section_heading", "page_number", "url", "date"]
        purpose: "Enable filtering, attribution, freshness ranking"
        
  layer_2_embedding:
    models:
      text_embedding_3_large:
        provider: "OpenAI"
        dimensions: "3072 (or 256-1536 via Matryoshka)"
        quality: "Best overall quality (2026)"
        cost: "$0.13/M tokens"
        
      voyage_3:
        provider: "Voyage AI"
        strengths: "Code and technical documents"
        cost: "$0.06/M tokens"
        
      bge_m3:
        provider: "BAAI (open-source)"
        strengths: "Multilingual, hybrid (dense + sparse)"
        cost: "Free (self-hosted)"
        
      cohere_embed_v4:
        provider: "Cohere"
        strengths: "Multi-modal (text + images)"
        cost: "$0.10/M tokens"
        
    considerations:
      - "Match embedding model at index time and query time (must be same model)"
      - "Higher dimensions = better quality but more storage/compute"
      - "Matryoshka embeddings: truncate dimensions for speed (256 for filtering, 3072 for final)"
      
  layer_3_vector_storage:
    databases:
      pinecone:
        type: "Managed serverless"
        strength: "Zero-ops, auto-scaling, namespace isolation"
        scale: "Billions of vectors"
        cost: "Pay-per-query (serverless) or $70/month+ (pods)"
        
      weaviate:
        type: "Open-source / managed"
        strength: "Hybrid search (vector + BM25), multi-modal"
        deployment: "Self-hosted or Weaviate Cloud"
        
      pgvector:
        type: "PostgreSQL extension"
        strength: "Familiar SQL + vector search combined"
        when: "Already using Postgres, <10M vectors"
        
      qdrant:
        type: "Open-source / managed"
        strength: "Advanced filtering, high performance"
        deployment: "Self-hosted or Qdrant Cloud"
        
      chroma:
        type: "Open-source (embedded)"
        strength: "Simple, good for prototyping"
        when: "Development, small datasets"
        
  layer_4_retrieval:
    basic_retrieval:
      method: "Cosine similarity between query embedding and document embeddings"
      top_k: "3-10 chunks (balance context length vs. noise)"
      
    advanced_retrieval:
      hybrid_search:
        what: "Combine dense vectors + sparse BM25 (keyword matching)"
        benefit: "Catches exact terms that semantic search misses"
        ratio: "Typically 0.7 semantic + 0.3 keyword (tune per use case)"
        
      re_ranking:
        what: "Retrieve top-20, re-rank with cross-encoder to top-5"
        models: ["Cohere Rerank v3", "BGE Reranker", "Jina Reranker"]
        benefit: "10-20% accuracy improvement (cross-encoder is more precise)"
        cost: "50-100ms additional latency"
        
      query_expansion:
        what: "Generate multiple query versions, retrieve for each"
        methods:
          - "LLM rewrites query in different phrasings"
          - "HyDE: LLM generates hypothetical answer, search with that"
          - "Sub-queries: break complex question into simpler sub-questions"
        benefit: "Broader coverage (user's phrasing may not match document phrasing)"
        
      metadata_filtering:
        what: "Pre-filter by metadata before vector search"
        examples:
          - "date > 2025-01-01 (only recent documents)"
          - "department = 'engineering' (scope to relevant team)"
          - "document_type = 'policy' (only official docs)"
        benefit: "Reduces search space, improves relevance"
        
  layer_5_generation:
    prompt_construction:
      structure: |
        System: You are a helpful assistant. Answer based ONLY on the provided context.
        If the context doesn't contain the answer, say "I don't have that information."
        
        Context: {retrieved_chunks}
        
        Question: {user_query}
        
        Instructions: Cite your sources using [Source: document_name].
      
    strategies:
      stuff: "Put all chunks in one prompt (works for small context)"
      map_reduce: "Summarize each chunk, then summarize summaries"
      refine: "Iterate: initial answer → refine with each chunk"
      
    citation:
      what: "Link generated claims to source documents"
      method: "Instruct model to cite [Source N] for each claim"
      verification: "Post-process: check cited source actually supports claim"
```

### Advanced RAG Patterns

```python
# Advanced RAG engineering patterns

"""
Production RAG patterns beyond naive retrieve-and-generate:
agentic RAG, multi-hop, evaluation, and optimization.
"""

advanced_rag_patterns = {
    "agentic_rag": {
        "what": "AI agent decides when and what to retrieve iteratively",
        "why": "Complex questions need multiple retrieval steps",
        "flow": [
            "1. Agent receives user question",
            "2. Agent decides: 'Do I need to retrieve information?' (sometimes not needed)",
            "3. If yes: formulates optimal search query (not just user's words)",
            "4. Reviews retrieved documents: 'Is this sufficient to answer?'",
            "5. If not: reformulates query, retrieves again (different angle)",
            "6. If yes: generates answer with citations",
            "7. Self-checks: 'Does my answer match the retrieved evidence?'",
        ],
        "advantage": "Handles multi-hop questions, adapts retrieval strategy",
        "example": {
            "question": "How did Q3 2025 revenue compare to our main competitor?",
            "agent_steps": [
                "Search 1: 'Q3 2025 revenue report internal' → finds our revenue",
                "Search 2: 'competitor X Q3 2025 earnings' → finds competitor revenue",
                "Synthesize: Compare numbers, calculate difference, answer",
            ],
        },
        "tools": ["LangGraph (agent orchestration)", "LlamaIndex agents"],
    },
    
    "multi_hop_rag": {
        "what": "Answer questions requiring information from multiple documents",
        "challenge": "No single chunk contains the full answer",
        "example": {
            "question": "Which manager approved the largest budget increase in 2025?",
            "requires": [
                "Document 1: Budget change records (amounts)",
                "Document 2: Approval hierarchy (who approved what)",
                "Reasoning: Find largest increase → look up approver",
            ],
        },
        "techniques": {
            "iterative_retrieval": "Retrieve → partial answer → retrieve more → complete answer",
            "graph_rag": "Build knowledge graph from documents, traverse for multi-hop",
            "decomposition": "Break complex question into sub-questions, answer each, combine",
        },
    },
    
    "corrective_rag": {
        "what": "Detect and fix low-quality retrieval before generation",
        "mechanism": [
            "1. Retrieve top-K documents",
            "2. Score relevance of each document (is it actually relevant?)",
            "3. If all scores low: query is unanswerable from knowledge base",
            "4. If some relevant: filter out irrelevant ones, generate from relevant only",
            "5. If answer quality is low: try different retrieval strategy and regenerate",
        ],
        "benefit": "Reduces hallucination when retrieval returns irrelevant content",
    },
    
    "contextual_retrieval": {
        "what": "Add document-level context to each chunk before embedding",
        "problem": "Isolated chunks lose context (a chunk about 'the policy' without knowing WHICH policy)",
        "solution": [
            "For each chunk: ask LLM to generate contextual header",
            "Header includes: document title, section, topic summary",
            "Prepend to chunk before embedding",
        ],
        "improvement": "20-30% retrieval accuracy improvement (Anthropic's research)",
    },
    
    "evaluation_framework": {
        "what": "Measure RAG quality systematically",
        "dimensions": {
            "retrieval_quality": {
                "metrics": [
                    "Recall@K: What % of relevant docs are in top-K?",
                    "Precision@K: What % of top-K docs are relevant?",
                    "MRR (Mean Reciprocal Rank): How high is first relevant doc?",
                    "NDCG: Graded relevance of retrieved documents",
                ],
                "method": "Requires annotated relevance judgments (per query, per doc)",
            },
            "generation_quality": {
                "metrics": [
                    "Faithfulness: Does answer match retrieved evidence? (no hallucination)",
                    "Relevance: Does answer address the question?",
                    "Completeness: Does answer cover all aspects of the question?",
                    "Conciseness: Is answer appropriately detailed (not too long/short)?",
                ],
                "method": "LLM-as-judge (rate on 1-5 scale) or human evaluation",
            },
            "end_to_end": {
                "metrics": [
                    "Answer accuracy: Is the factual content correct?",
                    "Citation accuracy: Do citations point to supporting evidence?",
                    "Refusal accuracy: Does model refuse when answer isn't in context?",
                ],
                "method": "Comparison against gold-standard answers",
            },
        },
        "tools": {
            "ragas": "Open-source RAG evaluation framework",
            "trulens": "LLM-based evaluation with explanations",
            "langsmith": "LangChain's tracing and evaluation platform",
            "custom": "Build task-specific evaluation (often best)",
        },
        "evaluation_dataset": {
            "size": "100-500 question-answer-evidence triples",
            "coverage": "Easy questions, hard questions, unanswerable questions",
            "source": "Expert-written (gold standard) or LLM-generated + human-verified",
        },
    },
    
    "optimization_techniques": {
        "latency_optimization": {
            "pre_filter": "Metadata filter before vector search (reduce search space)",
            "matryoshka_embeddings": "Use 256-dim for first pass, 3072 for re-ranking",
            "cache": "Cache frequent queries (same question → same retrieval result)",
            "streaming": "Stream generation while retrieval runs in parallel",
            "pre_computed": "Pre-embed common queries at index time",
        },
        "quality_optimization": {
            "chunk_size_tuning": "Experiment with 256, 512, 1024 tokens (task-dependent)",
            "parent_document_retriever": "Retrieve small chunk, return surrounding context",
            "ensemble_retrieval": "Run multiple retrievers, merge/deduplicate results",
            "fine_tuned_embeddings": "Fine-tune embedding model on your domain data",
            "colbert": "Late interaction retrieval (token-level matching, higher quality)",
        },
        "cost_optimization": {
            "fewer_chunks": "Reduce K from 10 to 5 (fewer input tokens)",
            "compression": "Extract key sentences from chunks (discard noise)",
            "smaller_model": "GPT-4o-mini for routine queries, GPT-4o for complex",
            "caching": "Cache generation results (exact query match)",
        },
    },
}


# RAG anti-patterns and fixes
rag_anti_patterns = {
    "lost_in_the_middle": {
        "problem": "LLMs ignore content in the middle of long context",
        "research": "Information at start/end of context is used 2× more than middle",
        "fix": [
            "Put most relevant chunks at START of context",
            "Limit to 3-5 chunks (less middle to get lost in)",
            "Re-rank so best evidence is first",
        ],
    },
    
    "retrieval_noise": {
        "problem": "Retrieved chunks are related but don't answer the question",
        "example": "Question about 'pricing' retrieves 'pricing page redesign' discussion",
        "fix": [
            "Re-ranking with cross-encoder (discriminates better)",
            "Relevance threshold (drop chunks below score X)",
            "Query expansion (retrieve from multiple angles)",
        ],
    },
    
    "chunk_boundary_issues": {
        "problem": "Answer spans across two chunks, neither is complete",
        "fix": [
            "Increase overlap between chunks (200+ tokens)",
            "Use parent-document retriever (small chunk for matching, return full section)",
            "Semantic chunking (split on topic boundaries, not fixed size)",
        ],
    },
}
```

---

## How It Works in Practice

### Production RAG System

```yaml
Production_RAG_System:
  scenario: "Enterprise knowledge assistant for 10K employees"
  
  requirements:
    sources: "50K documents (Confluence, Notion, Google Drive, Slack)"
    queries: "20K queries/day (technical questions, policy lookup, onboarding)"
    accuracy: ">90% answer correctness"
    latency: "< 3 seconds end-to-end"
    freshness: "Documents updated within 1 hour of source change"
    
  architecture:
    ingestion:
      pipeline: "Apache Airflow DAG (runs every 30 minutes)"
      steps:
        - "Pull new/updated documents from sources (incremental sync)"
        - "Parse with Unstructured.io (handles PDF, DOCX, HTML)"
        - "Chunk semantically (500-token target, respect headings)"
        - "Add context header per chunk (LLM-generated document summary prefix)"
        - "Embed with text-embedding-3-large (3072 dimensions)"
        - "Upsert to Pinecone (with metadata: source, date, author, department)"
      scale: "Process 500 document updates/hour"
      
    retrieval:
      step_1: "Metadata pre-filter (department, date range if specified)"
      step_2: "Hybrid search (0.7 semantic + 0.3 BM25)"
      step_3: "Retrieve top-20 chunks"
      step_4: "Re-rank with Cohere Rerank v3 → top-5"
      step_5: "Contextual compression (extract relevant sentences only)"
      latency: "200-400ms total retrieval"
      
    generation:
      model: "Claude 4 Sonnet (strong instruction following, long context)"
      prompt: |
        Answer the question based ONLY on the provided context.
        If context doesn't contain the answer, say "I couldn't find this in our knowledge base."
        Cite sources as [Source: document_title, section].
      response_time: "1-2 seconds (streaming)"
      
    evaluation:
      automated:
        - "Daily: run 500-question eval set, track accuracy/faithfulness"
        - "Real-time: detect 'I don't know' rate (should be 5-15%, not 50%)"
        - "Citation verification: automated check that cited source supports claim"
      human:
        - "Weekly: expert reviews 50 random Q&A pairs"
        - "Thumbs up/down from users (3-5% response rate)"
        
  metrics:
    accuracy: "93% on eval set (vs 67% with naive RAG)"
    latency_p50: "1.8 seconds"
    latency_p99: "4.2 seconds"
    cost: "$8K/month (embedding + vector DB + LLM API)"
    user_satisfaction: "4.2/5 (from feedback)"
```

---

## Interview Tip

> When asked about RAG: "RAG engineering is where naive implementation differs most from production quality. My approach: (1) Chunking matters enormously — I use semantic chunking (split by topic, not fixed size) with contextual headers (LLM-generated summary of parent document prepended to each chunk). This alone improves retrieval 20-30%. (2) Retrieval pipeline: hybrid search (dense embeddings + BM25 keyword matching) → retrieve top-20 → re-rank with cross-encoder to top-5. Re-ranking adds 50-100ms latency but 10-20% accuracy. (3) Generation: strict system prompt (answer ONLY from context, cite sources). Include refusal instruction (say 'I don't know' when evidence is insufficient). (4) Evaluation: I maintain a 500-question eval set with gold-standard answers. Track retrieval quality (recall@5, MRR) and generation quality (faithfulness, relevance) separately. If retrieval is bad, improving generation won't help. (5) Advanced patterns: for complex questions requiring multiple documents, I use agentic RAG (LangGraph agent that iteratively retrieves, checks sufficiency, retrieves more if needed). (6) Common failure mode: 'lost in the middle' — LLMs ignore content in the middle of long context. Fix: put most relevant chunks FIRST, limit to 3-5 chunks, use re-ranking. Production numbers: hybrid search + reranking + contextual chunking achieves 90%+ accuracy vs. 60-70% with naive RAG."

---

## Common Mistakes

1. **Fixed-size chunking without overlap** — Splitting documents every 512 tokens regardless of content boundaries. Answers that span chunk boundaries are lost. Solution: semantic chunking (split by topic/section) with 100-200 token overlap. Use document structure (headings, paragraphs) as natural break points.

2. **No re-ranking** — Using raw vector similarity scores as final ranking. Bi-encoder similarity is good for recall but mediocre for precision (retrieves related-but-not-relevant content). Solution: always add a cross-encoder re-ranker. Retrieve more (top-20) and re-rank to fewer (top-5). Worth the 50-100ms latency.

3. **Stuffing too many chunks into context** — Retrieving top-20 chunks and including all in prompt. Causes "lost in the middle" (model ignores middle content), increases cost, and adds noise. Solution: retrieve many, re-rank aggressively, include only top-3-5 highest relevance chunks.

4. **No evaluation pipeline** — Building RAG without measuring quality. No way to know if changes help or hurt. Solution: build eval set of 100-500 questions with known answers BEFORE building RAG. Run after every change. Track retrieval and generation metrics separately.

5. **Ignoring the 'no answer' case** — Model hallucinates an answer when retrieved documents don't contain relevant information. Users trust confident-sounding wrong answers. Solution: explicit system prompt instruction to refuse when evidence is insufficient. Track refusal rate (should be 5-15% for a well-scoped knowledge base). Test with questions that SHOULDN'T be answerable.

---

## Key Takeaways

- RAG: retrieve relevant documents → include in prompt → generate grounded answer
- Naive RAG: 60-70% accuracy. Production RAG (with proper engineering): 90%+
- Chunking: semantic (by topic) > fixed-size. Add contextual headers. 100-200 token overlap
- Retrieval: hybrid search (vector + BM25) → retrieve top-20 → re-rank to top-5
- Re-ranking: cross-encoder models add 10-20% accuracy for 50-100ms latency
- Generation: strict instructions (answer from context only, cite sources, refuse if unsure)
- Evaluation: separate retrieval quality (recall@K) from generation quality (faithfulness)
- Agentic RAG: LLM agent iteratively retrieves, checks sufficiency, re-retrieves (complex questions)
- "Lost in the middle": put most relevant content FIRST in context, limit chunks
- Vector databases: Pinecone (managed), Weaviate (hybrid), pgvector (PostgreSQL), Qdrant (filtering)
- Cost drivers: embedding (one-time), storage (ongoing), re-ranking (per query), generation (per query)
