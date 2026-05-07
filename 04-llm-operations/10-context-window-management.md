# Context Window Management

## The Problem / Why This Matters

Every LLM has a finite context window — the maximum number of tokens it can process in a single request (input + output combined). In 2026, context windows range from 8K tokens (small models) to 2M tokens (Gemini 2.5 Pro), but "can fit" doesn't mean "works well." Models experience quality degradation with very long contexts: the "lost in the middle" phenomenon where information in the middle of long documents is less likely to be retrieved accurately. Context window management is the engineering discipline of: deciding what information goes into the prompt (context selection), how to structure it for best recall (context positioning), when to use context vs other approaches (RAG vs fine-tuning vs long context), and how to handle conversations that exceed the window (context compression, summarization). This matters because: tokens cost money (a 200K context prompt costs 100× more than a 2K prompt), longer contexts are slower (prefill time scales with input length), and quality degrades with irrelevant context (more noise = worse signal). The best LLM engineers don't just throw everything into the context — they carefully curate what the model sees to maximize quality while minimizing cost and latency.

---

## The Analogy

Think of the context window like a desk workspace:

- **Small context (8K tokens)** = Tiny desk. You can only have 2-3 pages in front of you. Must carefully choose which documents to reference — can't spread everything out. Forces focused, curated context.
- **Medium context (128K tokens)** = Large desk. Can spread out a full report with appendices. Enough for most single-document tasks. But if you pile 50 documents on it, you start losing track of which page has what information.
- **Large context (1-2M tokens)** = Entire room of filing cabinets. Technically everything fits, but finding a specific fact in cabinet 47 takes longer, and you might miss things buried deep in the middle. The room is available but navigating it efficiently requires strategy.
- **Context management** = A skilled executive assistant who prepares a brief folder with exactly the right documents for each meeting. They don't hand you the entire filing cabinet — they select, organize, and highlight what matters for THIS specific question.

---

## Deep Dive

### Context Window Sizes (2026)

```yaml
Context_Windows_2026:
  models:
    gemini_2_5_pro:
      window: "2M tokens (~1.5M words, ~3000 pages)"
      effective: "Good recall up to ~500K, degradation at extreme lengths"
      
    claude_4_opus:
      window: "200K tokens (~150K words, ~300 pages)"
      effective: "Excellent recall throughout (near-perfect needle-in-haystack)"
      
    gpt_5:
      window: "256K tokens (~192K words, ~380 pages)"
      effective: "Strong recall, slight degradation at extremes"
      
    llama_4_70b:
      window: "128K tokens (~96K words, ~190 pages)"
      effective: "Good up to ~64K, noticeable degradation beyond"
      
    small_models:
      window: "8K-32K tokens"
      effective: "Must be very selective about context content"
      
  token_to_content_conversion:
    1_token: "~4 characters or ~0.75 words (English)"
    1000_tokens: "~750 words (about 1.5 pages)"
    128K_tokens: "~1 medium-length book"
    1M_tokens: "~8 books worth of text"
    
  cost_implications:
    example: "200K tokens input to Claude 4 Opus = $3.00 per request"
    concern: "10 users × 10 requests/hour × 200K tokens = $600/hour = $14,400/day"
    takeaway: "Long context is expensive — only include what's necessary"
```

### Context Selection Strategies

```yaml
Context_Selection:
  strategy_1_rag_retrieval:
    what: "Retrieve only relevant chunks (not entire documents)"
    approach: |
      1. Embed user query
      2. Find top-K most similar chunks from knowledge base
      3. Include only those chunks in context (typically 3-10 chunks, 2K-8K tokens)
    benefit: "Focused context, low cost, fast"
    limitation: "May miss relevant info if retrieval imperfect"
    best_for: "Large knowledge bases where <1% of content is relevant per query"
    
  strategy_2_full_document:
    what: "Include entire document(s) in context"
    approach: "Send full document + question to model"
    benefit: "No retrieval errors, model sees complete picture"
    limitation: "Expensive, slow, limited by context window"
    best_for: "Single-document QA, contract analysis, code review"
    
  strategy_3_hierarchical:
    what: "Multi-level retrieval: broad → narrow"
    approach: |
      1. First pass: identify which documents are relevant (titles/summaries)
      2. Second pass: retrieve specific sections from relevant documents
      3. Include only those sections in context
    benefit: "Better precision than flat retrieval"
    cost: "Two retrieval passes (slightly more latency)"
    
  strategy_4_map_reduce:
    what: "Process long content in chunks, then combine"
    approach: |
      1. Split document into N chunks
      2. Process each chunk independently (map step)
      3. Combine results (reduce step)
    use_case: "Summarizing a 500-page document that doesn't fit in context"
    limitation: "Loses cross-chunk relationships"
    
  strategy_5_iterative_refinement:
    what: "Start with broad context, refine based on model's needs"
    approach: |
      1. Send query with minimal context
      2. If model says "I need more information about X"
      3. Retrieve context about X, send follow-up
    benefit: "Minimizes tokens used (only fetch what's needed)"
    limitation: "Multiple round trips (higher latency)"
```

### Context Positioning and Structure

```yaml
Context_Positioning:
  lost_in_the_middle:
    problem: |
      Research shows models recall information better from the beginning 
      and end of the context, with lower recall for middle sections.
      A fact at position 50% into a 128K context may be missed.
    mitigation:
      - "Put most important information at the BEGINNING of context"
      - "Put instructions/query at the END (recent attention)"
      - "Use clear section headers and separators"
      - "Shorter, focused context > long, comprehensive context"
      
  optimal_structure:
    template: |
      [SYSTEM INSTRUCTIONS - beginning, always attended to]
      
      [MOST RELEVANT CONTEXT - immediately after instructions]
      ## Highly Relevant Information
      {top_3_most_relevant_chunks}
      
      [SUPPORTING CONTEXT - middle section]
      ## Additional Context
      {less_critical_but_potentially_useful_chunks}
      
      [USER QUERY - at the end, strong recency attention]
      ## Question
      {user_question}
      
  chunking_best_practices:
    chunk_size: "500-1000 tokens (enough for coherent content, small enough for precision)"
    overlap: "100-200 tokens overlap between chunks (preserve context across boundaries)"
    semantic_boundaries: "Split at paragraph/section breaks, not mid-sentence"
    metadata: "Include source, page number, section title with each chunk"
    
  context_window_budget:
    allocation:
      system_prompt: "500-2000 tokens (fixed cost per request)"
      retrieved_context: "2000-8000 tokens (varies by query)"
      conversation_history: "2000-10000 tokens (grows over conversation)"
      user_query: "50-500 tokens"
      output_reserved: "1000-4000 tokens (must leave room for response)"
    total_example: "128K window, budget: 2K system + 8K context + 8K history + 500 query + 4K output = 22.5K used"
    observation: "You rarely need to use the full context window"
```

### Conversation History Management

```python
# Context window management for multi-turn conversations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str
    tokens: int
    timestamp: float
    importance: float = 1.0  # For prioritized truncation


class ConversationManager:
    """Manages conversation history within context window limits."""
    
    def __init__(
        self,
        max_context_tokens: int = 128000,
        max_history_tokens: int = 10000,
        reserved_output_tokens: int = 4000,
        system_prompt_tokens: int = 1500,
    ):
        self.max_context = max_context_tokens
        self.max_history = max_history_tokens
        self.reserved_output = reserved_output_tokens
        self.system_prompt_tokens = system_prompt_tokens
        self.messages: list[Message] = []
        
    def available_for_context(self) -> int:
        """Calculate how many tokens available for retrieved context."""
        history_tokens = sum(m.tokens for m in self._get_active_history())
        used = self.system_prompt_tokens + history_tokens + self.reserved_output
        return self.max_context - used
    
    def add_message(self, message: Message):
        """Add message and manage window overflow."""
        self.messages.append(message)
        self._manage_overflow()
    
    def get_messages_for_prompt(self) -> list[Message]:
        """Get messages that fit within the context window."""
        return self._get_active_history()
    
    def _manage_overflow(self):
        """Handle conversation exceeding max_history_tokens."""
        total_history_tokens = sum(m.tokens for m in self.messages if m.role != "system")
        
        if total_history_tokens <= self.max_history:
            return  # No overflow
        
        # Strategy: summarize old messages, keep recent ones
        self._apply_truncation_strategy()
    
    def _apply_truncation_strategy(self):
        """Apply context management strategy when history overflows."""
        # Strategy 1: Sliding window (drop oldest messages)
        # Strategy 2: Summarize old messages into a condensed form
        # Strategy 3: Keep first + last N messages (preserve beginning context + recency)
        
        # Implementation: summarize-and-compress approach
        history = [m for m in self.messages if m.role != "system"]
        
        if len(history) <= 4:
            return  # Too few messages to compress
        
        # Keep last 4 messages (recent context) in full
        recent = history[-4:]
        older = history[:-4]
        
        # Summarize older messages
        summary_content = self._summarize_messages(older)
        summary_message = Message(
            role="system",
            content=f"[Conversation summary: {summary_content}]",
            tokens=self._count_tokens(summary_content),
            timestamp=older[-1].timestamp,
            importance=0.5,
        )
        
        # Replace older messages with summary
        self.messages = [m for m in self.messages if m.role == "system"]
        self.messages.append(summary_message)
        self.messages.extend(recent)
    
    def _summarize_messages(self, messages: list[Message]) -> str:
        """Summarize a sequence of messages into condensed form."""
        # In production: use a fast model (Gemini Flash) to summarize
        # Here: simple extraction of key points
        key_points = []
        for msg in messages:
            if msg.role == "user":
                key_points.append(f"User asked about: {msg.content[:100]}")
            elif msg.role == "assistant":
                key_points.append(f"Assistant provided: {msg.content[:100]}")
        return " | ".join(key_points[-5:])  # Last 5 key points
    
    def _get_active_history(self) -> list[Message]:
        """Get messages currently in the active window."""
        return [m for m in self.messages if m.role != "system" or "summary" not in m.content.lower()]
    
    def _count_tokens(self, text: str) -> int:
        """Approximate token count."""
        return len(text) // 4  # Rough approximation


class ContextWindowOptimizer:
    """Optimize context usage for quality and cost."""
    
    def __init__(self, model_context_limit: int):
        self.limit = model_context_limit
        
    def build_optimal_context(
        self,
        system_prompt: str,
        retrieved_chunks: list[dict],  # {"text": ..., "score": ..., "source": ...}
        conversation_history: list[Message],
        user_query: str,
        max_output_tokens: int = 4000,
    ) -> list[dict]:
        """Build the optimal prompt within context constraints."""
        
        # Calculate token budgets
        system_tokens = len(system_prompt) // 4
        query_tokens = len(user_query) // 4
        
        available_for_context_and_history = (
            self.limit - system_tokens - query_tokens - max_output_tokens
        )
        
        # Allocate: 60% to retrieved context, 40% to conversation history
        context_budget = int(available_for_context_and_history * 0.6)
        history_budget = int(available_for_context_and_history * 0.4)
        
        # Select chunks that fit within context budget
        selected_chunks = self._select_chunks(retrieved_chunks, context_budget)
        
        # Trim history to fit within history budget
        trimmed_history = self._trim_history(conversation_history, history_budget)
        
        # Construct final prompt with optimal positioning
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Add context at the beginning (high attention zone)
        if selected_chunks:
            context_text = "\n\n".join(
                f"[Source: {c['source']}]\n{c['text']}" for c in selected_chunks
            )
            messages.append({"role": "system", "content": f"Relevant context:\n{context_text}"})
        
        # Add conversation history
        for msg in trimmed_history:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add current query at the end (recency attention)
        messages.append({"role": "user", "content": user_query})
        
        return messages
    
    def _select_chunks(self, chunks: list[dict], budget: int) -> list[dict]:
        """Select highest-relevance chunks that fit in budget."""
        # Sort by relevance score (descending)
        sorted_chunks = sorted(chunks, key=lambda c: c["score"], reverse=True)
        
        selected = []
        tokens_used = 0
        
        for chunk in sorted_chunks:
            chunk_tokens = len(chunk["text"]) // 4
            if tokens_used + chunk_tokens <= budget:
                selected.append(chunk)
                tokens_used += chunk_tokens
            else:
                break
        
        return selected
    
    def _trim_history(self, history: list[Message], budget: int) -> list[Message]:
        """Keep most recent messages that fit in budget."""
        # Start from most recent, work backwards
        selected = []
        tokens_used = 0
        
        for msg in reversed(history):
            if tokens_used + msg.tokens <= budget:
                selected.insert(0, msg)
                tokens_used += msg.tokens
            else:
                break
        
        return selected
```

### When to Use Long Context vs RAG vs Fine-Tuning

```yaml
Decision_Matrix:
  long_context:
    use_when:
      - "Analyzing a single large document (contract review, code review)"
      - "Content changes frequently (can't pre-embed)"
      - "Need complete document understanding (not just fact retrieval)"
      - "Cross-referencing multiple sections of the same document"
    limitations:
      - "Expensive (pay for all tokens even if most are irrelevant)"
      - "Slower (prefill time proportional to input length)"
      - "Quality degrades with very long, unfocused content"
      - "Lost-in-the-middle phenomenon"
    cost: "High per-request (all tokens processed every time)"
    
  rag:
    use_when:
      - "Large knowledge base (1000s of documents)"
      - "Only small fraction relevant per query (needle in haystack)"
      - "Knowledge updates frequently (embed new docs incrementally)"
      - "Need citations/source attribution"
      - "Budget-constrained (only process relevant chunks)"
    limitations:
      - "Retrieval errors (wrong chunks retrieved → wrong answers)"
      - "Doesn't understand document structure (chunks lose context)"
      - "Requires embedding pipeline infrastructure"
    cost: "Low per-request (only relevant chunks processed)"
    
  fine_tuning:
    use_when:
      - "Behavior/style that should be automatic (not from context)"
      - "Domain terminology and conventions"
      - "Consistent output format across all requests"
      - "Reducing prompt size (bake knowledge into weights)"
    limitations:
      - "Stale knowledge (can't update without retraining)"
      - "Expensive to update (retraining takes hours)"
      - "Can't cite sources (knowledge is in weights)"
    cost: "Low per-request (no context tokens needed), high upfront (training)"
    
  hybrid_approaches:
    fine_tune_plus_rag:
      what: "Fine-tune for style/behavior + RAG for factual grounding"
      example: "Fine-tune for medical report format + RAG for patient-specific data"
      
    long_context_plus_rag:
      what: "Retrieve relevant documents, then process them as full long context"
      example: "RAG finds 5 relevant documents, all 5 sent as full text in context"
      benefit: "Best of both: focused retrieval + complete document understanding"
      
    context_plus_fine_tune:
      what: "Fine-tuned model + minimal context for per-request specifics"
      example: "Fine-tuned customer service model + context with customer's order history"
```

---

## How It Works in Practice

### Context Management Operations

```yaml
Operations:
  monitoring:
    metrics:
      - "Average context utilization (% of window used)"
      - "Context tokens per request (cost driver)"
      - "Retrieval relevance scores (are right chunks selected?)"
      - "Conversation length before summarization triggers"
      - "Quality vs context length correlation (is longer context helping?)"
    
  optimization_cycle:
    weekly:
      - "Review average context length vs quality scores"
      - "Identify over-contexted requests (full window but no quality benefit)"
      - "Tune chunk size and retrieval count"
      
    monthly:
      - "Evaluate RAG vs long context tradeoff (cost vs quality)"
      - "Update chunking strategy based on failure analysis"
      - "Test new models with larger context windows"
      
  common_patterns:
    chat_application:
      strategy: "Sliding window + summarization"
      implementation: "Keep last 10 messages in full, summarize older into 500-token summary"
      
    document_qa:
      strategy: "RAG for collection, full-document for single-doc"
      implementation: "If query targets one document → full context. If across many → RAG retrieval."
      
    code_assistant:
      strategy: "Relevant files + current file context"
      implementation: "Include current file + imports + referenced files (up to budget)"
```

---

## Interview Tip

> When asked about context window management: "I treat context window as a precious resource — not 'how much can I fit' but 'what's the minimum context for maximum quality': (1) Context budgeting — I allocate: system prompt (fixed, 1-2K tokens), retrieved context (variable, 2-8K), conversation history (capped, 8-10K), reserved output (4K). Total well under the 128K limit — longer isn't better. (2) Selection strategy — RAG for large knowledge bases (retrieve top-K relevant chunks), full-document for single-doc analysis, hybrid (retrieve documents then include them fully) when I need both precision and completeness. (3) Positioning — most important context at the beginning (strongest attention), user query at the end (recency bias), avoid critical info in the middle (lost-in-the-middle phenomenon). (4) Conversation management — sliding window keeps last N messages, older messages get summarized (by a fast model) into a condensed summary that preserves key context. (5) Cost optimization — 200K tokens to Claude 4 Opus = $3/request. If I can achieve same quality with 20K tokens via smart retrieval, that's 10× savings. Key insight: I choose strategy per query type: factual lookup → RAG (small context, fast, cheap), document analysis → full document in context, multi-session chat → summarize + recent messages, code tasks → relevant files only."

---

## Common Mistakes

1. **Stuffing the entire context window** — "Model has 128K context, let's use all of it!" Sending every document, full conversation history, and verbose system prompt. Result: higher cost, slower latency, quality degradation from noise. Solution: include only what's relevant. 5K of focused context often beats 100K of everything.

2. **Ignoring lost-in-the-middle** — Placing critical information in the middle of a 50K token context. Model misses it and gives wrong answer. Solution: put critical information at the beginning or end of context. Use clear section headers. Test recall at different positions.

3. **No conversation history management** — Chat applications that keep appending messages until the context window is full, then crash or silently truncate. Solution: implement explicit history management (summarization, sliding window) that triggers well before the limit.

4. **Same strategy for all query types** — Using RAG for everything (including single-document QA where full document fits in context). Or using long context for everything (including knowledge base with 10,000 documents). Solution: choose strategy based on task: single-doc → full context, large corpus → RAG, hybrid when needed.

5. **Not measuring context efficiency** — No visibility into how much context is actually helping. Maybe you're paying for 8K tokens of retrieved context but only 2K is relevant. Solution: track quality vs context length, experiment with fewer chunks, measure marginal quality gain of additional context.

---

## Key Takeaways

- Context window is a budget: allocate tokens intentionally (system + context + history + output)
- Longer context ≠ better quality: focused, relevant context outperforms comprehensive noise
- Lost-in-the-middle: place critical information at beginning or end, not middle
- Strategy per task: RAG (large corpus, specific facts), full-document (single-doc analysis), fine-tuning (behavior/style)
- Conversation management: summarize old messages, keep recent in full, trigger before overflow
- Hybrid approaches: RAG retrieval + full-document context = precision + completeness
- Cost awareness: 200K tokens to Claude Opus = $3/request. Smart selection saves 10×
- Monitor context efficiency: track quality vs context length, optimize the tradeoff
- Chunking matters: 500-1000 tokens per chunk, overlap at boundaries, split at semantic breaks
- Context windows keep growing but the engineering challenge remains: selecting what matters
