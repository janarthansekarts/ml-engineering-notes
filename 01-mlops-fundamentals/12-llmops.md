# LLMOps

## The Problem / Why This Matters

LLMOps (Large Language Model Operations) is the emerging discipline of operationalizing LLM-powered applications in production. While traditional MLOps focuses on custom-trained models (data → training → deployment → monitoring), LLMOps deals with a fundamentally different operational surface: prompt management (prompts ARE the model behavior), token cost optimization (billing per token instead of per inference), output quality monitoring (non-deterministic text outputs), RAG (Retrieval-Augmented Generation) pipeline maintenance, provider management (multi-model routing, fallbacks), safety guardrails (content filtering, prompt injection defense), and evaluation without ground truth (LLM-as-judge patterns). Traditional MLOps practices still apply (CI/CD, monitoring, versioning), but LLMOps adds entirely new operational concerns. In 2026, as organizations move from LLM prototypes to production systems serving millions of users, LLMOps has become the critical discipline separating hobby projects from reliable AI applications.

---

## The Analogy

Think of the difference between MLOps and LLMOps like managing a factory vs managing a creative agency:

- **MLOps (factory)** = Predictable production line. You know the inputs (features), the process (model), and the outputs (scores). Quality control is measuring if the score is correct. Problems are systematic (data drift shifts all predictions).
- **LLMOps (creative agency)** = Variable creative output. Same brief (prompt) can produce different creative work each time. Quality is subjective (was the response helpful? accurate? safe?). Problems are subtle (model occasionally hallucinations, tone shifts with provider updates). Cost is per-word instead of per-unit. You manage multiple creative partners (LLM providers) and route work based on budget and complexity.

---

## Deep Dive

### LLMOps vs Traditional MLOps

```yaml
LLMOps_vs_MLOps:
  key_differences:
    model_development:
      mlops: "Train custom model (data → training → evaluation)"
      llmops: "Prompt engineering + RAG + optional fine-tuning (no training from scratch)"
      
    versioning:
      mlops: "Version model weights, data, code"
      llmops: "Version prompts, RAG configs, adapter weights, evaluation sets"
      
    deployment:
      mlops: "Deploy model binary to serving infrastructure"
      llmops: "Configure LLM provider (API) or deploy open model (vLLM/TGI)"
      
    cost_model:
      mlops: "Fixed cost (GPU running 24/7 for serving)"
      llmops: "Variable cost (per-token pricing, scales with usage)"
      
    quality_measurement:
      mlops: "Ground truth comparison (accuracy, AUC, F1)"
      llmops: "Subjective quality (LLM-as-judge, human eval, task-specific rubrics)"
      
    failure_modes:
      mlops: "Data drift → performance degradation (gradual, measurable)"
      llmops: "Hallucination, prompt injection, safety violations (sporadic, hard to predict)"
      
    provider_dependency:
      mlops: "Self-hosted model (full control)"
      llmops: "Often dependent on external API (rate limits, pricing changes, model deprecation)"
```

### Prompt Management

```yaml
Prompt_Management:
  why: "Prompts determine model behavior — they ARE the 'code' in LLM applications"
  
  prompt_lifecycle:
    development:
      - "Write initial prompt (system prompt, user template, few-shot examples)"
      - "Test against evaluation set (does it produce good outputs?)"
      - "Iterate (refine instructions, add/remove examples, adjust tone)"
    versioning:
      - "Store prompts in version control (Git or prompt management platform)"
      - "Tag versions with performance metrics"
      - "Track which prompt version is deployed where"
    testing:
      - "Regression tests (output quality doesn't degrade)"
      - "Edge case tests (handles unusual inputs)"
      - "Safety tests (doesn't produce harmful content)"
    deployment:
      - "Progressive rollout (A/B test new prompt vs current)"
      - "Rollback capability (instant revert to previous prompt)"
      - "Feature flags (test new prompts with subset of users)"
      
  prompt_versioning_tools:
    git_based: "Store prompts as files in repository (simple, reviewable)"
    langfuse: "Open-source prompt management with versioning, metrics, A/B"
    humanloop: "Commercial prompt management and evaluation platform"
    langsmith: "LangChain's tracing + prompt management"
    custom: "Database-backed prompt store with versioning and rollback"
    
  best_practices:
    - "Treat prompts as code — review, test, version, deploy progressively"
    - "Separate prompt content from application logic (not hardcoded strings)"
    - "Test prompts against new model versions before provider upgrades"
    - "Maintain a prompt changelog (what changed, why, what was the impact)"
    - "Use structured prompts (system instructions + context template + output schema)"
```

### LLM Observability

```yaml
LLM_Observability:
  what_to_trace:
    per_request:
      - "Input prompt (full text or template + variables)"
      - "Output response (full text)"
      - "Model used (provider, model name, version)"
      - "Token counts (input, output, total)"
      - "Latency (time to first token, total response time)"
      - "Cost (calculated from token counts + pricing)"
      - "Temperature and other generation parameters"
    per_session:
      - "Conversation history (multi-turn context)"
      - "Total tokens consumed"
      - "Total cost"
      - "User satisfaction signals (thumbs up/down, follow-up questions)"
    for_rag:
      - "Query embedding"
      - "Retrieved documents (which chunks, relevance scores)"
      - "Retrieval latency"
      - "Whether retrieved context was actually used in response"
    for_agents:
      - "Planning steps (reasoning trace)"
      - "Tool calls (which tool, arguments, results)"
      - "Total LLM calls per task"
      - "Task completion (success/failure/partial)"
      
  tools:
    langfuse:
      description: "Open-source LLM observability (tracing, evaluation, prompt management)"
      strengths: "Self-hostable, comprehensive tracing, cost tracking, evaluation"
    langsmith:
      description: "LangChain's commercial tracing and evaluation platform"
      strengths: "Deep LangChain integration, prompt playground"
    arize_phoenix:
      description: "Open-source LLM tracing and evaluation"
      strengths: "Embedding visualization, retrieval evaluation"
    braintrust:
      description: "LLM evaluation and prompt development platform"
      strengths: "Eval-first approach, prompt playground, scoring"
    wandb_weave:
      description: "W&B's LLM tracing and evaluation (integrated with experiment tracking)"
      strengths: "Unified platform for ML + LLM, evaluation framework"
```

### LLM Evaluation

```yaml
LLM_Evaluation:
  challenge: "No single accuracy metric — output quality is multi-dimensional and subjective"
  
  evaluation_approaches:
    llm_as_judge:
      description: "Use a strong LLM to evaluate another LLM's outputs"
      how: "Feed output + rubric to judge model, get score + explanation"
      advantages: "Scalable, consistent, covers many quality dimensions"
      limitations: "Judge can be wrong, biased toward verbose/fluent text"
      tools: "Custom rubrics + OpenAI/Claude API, Braintrust, RAGAS"
      
    human_evaluation:
      description: "Human raters score LLM outputs"
      advantages: "Gold standard, catches nuances LLM judges miss"
      limitations: "Expensive, slow, subjective (inter-rater disagreement)"
      when: "Critical decisions, periodic calibration of automated metrics"
      
    task_specific_metrics:
      extraction: "Precision, recall, F1 on extracted entities"
      classification: "Accuracy, confusion matrix"
      summarization: "ROUGE, BERTScore, faithfulness"
      code_generation: "Pass rate on unit tests, functional correctness"
      qa: "Exact match, semantic similarity, faithfulness to source"
      
    comparative_evaluation:
      description: "Pairwise comparison (is output A or B better?)"
      advantages: "Easier for humans than absolute scoring, more reliable"
      tools: "Chatbot Arena methodology, custom comparison frameworks"
      
  evaluation_dimensions:
    correctness: "Is the information factually accurate?"
    helpfulness: "Does it address the user's actual need?"
    safety: "Is the output free from harmful content?"
    coherence: "Is the text well-structured and logical?"
    faithfulness: "Is it grounded in provided context (for RAG)?"
    completeness: "Does it cover all relevant aspects?"
    conciseness: "Is it appropriately detailed without padding?"
```

### LLM Cost Management

```yaml
Cost_Management:
  strategies:
    model_routing:
      description: "Route queries to appropriate model based on complexity"
      implementation:
        - "Classifier determines query complexity (simple/medium/complex)"
        - "Simple → GPT-4o-mini or self-hosted small model"
        - "Complex → Claude 4 Sonnet or GPT-4o"
        - "Critical → Claude 4 Opus (only when stakes are highest)"
      savings: "50-70% (most queries are simple)"
      
    prompt_optimization:
      description: "Reduce token count without quality loss"
      techniques:
        - "Compress verbose system prompts"
        - "Remove redundant instructions"
        - "Use concise few-shot examples"
        - "Structured output reduces output tokens"
      savings: "20-40%"
      
    caching:
      description: "Cache responses for repeated or similar queries"
      types:
        exact_cache: "Same input → same output (deterministic caching)"
        semantic_cache: "Similar inputs → reuse previous output (embedding similarity)"
      tools: "Redis, GPTCache, Langchain caching"
      savings: "30-60% depending on query repetition"
      
    batch_processing:
      description: "Collect requests, process in batch (cheaper API pricing)"
      when: "Non-real-time workloads (document processing, evaluation, reporting)"
      savings: "50% with OpenAI batch API"
      
    self_hosting:
      description: "Run open models instead of API calls"
      when: "Monthly API spend >$10K-50K (break-even point)"
      models: "Llama 4, Mistral, Qwen, DeepSeek"
      infrastructure: "vLLM on GPU cluster (H100/A100)"
      savings: "60-90% at scale vs API pricing"
```

### LLM Safety and Guardrails

```yaml
Safety_Guardrails:
  input_guardrails:
    prompt_injection_detection:
      what: "Detect attempts to override system instructions"
      methods:
        - "Input classifier (trained to detect injection patterns)"
        - "Perplexity-based detection (injections often have unusual patterns)"
        - "Canary tokens (detect if system prompt is leaked)"
    content_filtering:
      what: "Block harmful, inappropriate, or off-topic inputs"
      methods:
        - "Keyword filtering (basic)"
        - "Classification model (toxic, PII, off-topic)"
        - "LLM-based input screening (use small model to check input)"
    rate_limiting:
      what: "Prevent abuse through excessive usage"
      implementation: "Per-user, per-session, per-minute limits"
      
  output_guardrails:
    content_safety:
      what: "Ensure outputs don't contain harmful content"
      methods:
        - "Output classifier (toxicity, bias, violence)"
        - "LLM-based output review (second model checks first model's output)"
        - "Regex-based PII detection and redaction"
    factuality_checking:
      what: "Verify output claims against known sources"
      methods:
        - "RAG-grounding check (is response supported by retrieved context?)"
        - "Confidence scoring (flag low-confidence claims)"
        - "Citation verification (do cited sources exist and support claims?)"
    format_validation:
      what: "Ensure output meets structural requirements"
      methods:
        - "JSON schema validation (for structured outputs)"
        - "Length constraints (min/max response length)"
        - "Required element checking (response must contain X)"
        
  tools:
    guardrails_ai: "Framework for adding guardrails to LLM applications"
    nemo_guardrails: "NVIDIA's toolkit for LLM safety rails"
    lakera: "Commercial prompt injection detection"
    rebuff: "Open-source prompt injection detection"
```

---

## How It Works in Practice

### LLMOps Stack Example

```yaml
Example:
  application: "Customer support chatbot (enterprise, 50K conversations/day)"
  
  stack:
    llm_providers:
      primary: "Claude 4 Sonnet (complex queries, high quality)"
      secondary: "GPT-4o-mini (simple FAQ queries, 70% of traffic)"
      fallback: "Self-hosted Llama-4-8B (if providers down)"
      
    rag:
      vector_db: "Qdrant (product knowledge base, support articles)"
      embedding: "text-embedding-3-small (OpenAI)"
      chunking: "512 tokens, 50 token overlap"
      refresh: "Daily knowledge base re-indexing"
      
    orchestration:
      framework: "LangGraph (conversation flow, tool use)"
      tools: "Order lookup, ticket creation, FAQ search, escalation"
      
    observability:
      tracing: "Langfuse (all LLM calls traced with tokens, cost, latency)"
      evaluation: "Weekly LLM-as-judge evaluation on 500 sample conversations"
      metrics: "Token cost/day, avg latency, resolution rate, escalation rate"
      
    safety:
      input: "Prompt injection detection (classifier), PII redaction"
      output: "Toxicity check, brand safety, hallucination flagging"
      human_escalation: "Low confidence (< 0.7) → route to human agent"
      
    cost_management:
      routing: "Complexity classifier routes 70% to GPT-4o-mini"
      caching: "FAQ responses cached (semantic similarity > 0.95)"
      budget: "$15K/month target, alert at $12K (80% threshold)"
      
  operational_metrics:
    daily_dashboard:
      - "Total conversations: 50K"
      - "Token cost: $480 (target: <$500)"
      - "Avg response latency: 2.1s"
      - "Resolution rate: 78% (without human)"
      - "Escalation rate: 22%"
      - "Safety flag rate: 0.3%"
      - "User satisfaction: 4.2/5"
      
  incident_response:
    hallucination_spike: "Auto-detected by evaluation → increase RAG top-k → redeploy"
    provider_outage: "Auto-failover to Llama-4-8B → page on-call → manual override"
    cost_spike: "Budget alert → investigate (traffic spike? prompt issue?) → adjust routing"
```

---

## Interview Tip

> When asked about LLMOps: "LLMOps extends MLOps for LLM-powered applications with unique operational concerns: (1) Prompt management — prompts as versioned code with testing, A/B deployment, and rollback. (2) Cost management — model routing (simple→cheap model, complex→expensive), caching, prompt optimization. (3) Quality monitoring — LLM-as-judge evaluation, hallucination detection, user satisfaction tracking. (4) Safety — prompt injection defense, output content filtering, PII redaction, human escalation for low confidence. (5) Provider management — multi-model routing, fallbacks, rate limiting. (6) RAG operations — knowledge base freshness, retrieval quality monitoring, index refresh. Key tools: Langfuse for observability, vLLM for self-hosting, Guardrails AI for safety, and custom evaluation pipelines with LLM-as-judge scoring. The biggest operational challenge is evaluation without ground truth — you can't compute accuracy for a chatbot response like you can for a fraud score."

---

## Common Mistakes

1. **No prompt versioning** — Editing prompts directly in application code without tracking versions. When output quality degrades, you can't identify which prompt change caused it or roll back to a working version.

2. **Ignoring cost until the bill arrives** — Not monitoring token usage during development. A feature that costs $100/month in testing with 100 users can cost $100K/month with 100K users. Calculate production cost projections early.

3. **Single provider dependency** — Building entire application against one LLM provider with no fallback. Provider outage = complete application outage. Always have a fallback route (second provider or self-hosted model).

4. **No output evaluation** — Deploying LLM features without systematic quality measurement. "It seemed to work in demos" is not evaluation. Set up automated scoring (LLM-as-judge) and periodic human evaluation.

5. **Security as afterthought** — Not implementing prompt injection defense, PII filtering, or output safety checks until a public incident. LLM applications are uniquely vulnerable to injection attacks — build guardrails from day one.

---

## Key Takeaways

- LLMOps extends MLOps for LLM applications: prompt management, token cost, output quality, safety, provider management
- Prompts ARE the model behavior — version, test, and deploy them like code
- Cost management: model routing (50-70% savings), caching (30-60%), prompt optimization (20-40%)
- Evaluation without ground truth: LLM-as-judge, human eval, task-specific metrics, comparative
- Safety guardrails: prompt injection defense (input), content filtering (output), PII redaction, human escalation
- Observability: trace every LLM call (prompt, response, tokens, cost, latency)
- Provider management: multi-model routing, fallbacks, rate limiting, budget controls
- RAG operations: knowledge base freshness, retrieval quality monitoring, index refresh cycles
- Self-hosting break-even: >$10K-50K/month API spend → consider vLLM + open models
- Tools: Langfuse (observability), Guardrails AI (safety), vLLM (serving), Braintrust (evaluation)
