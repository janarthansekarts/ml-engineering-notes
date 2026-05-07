# AI Platform 2025+

## The Problem / Why This Matters

The ML platform built for traditional ML (train scikit-learn model → deploy as REST endpoint) is insufficient for the AI landscape of 2025-2026. Organizations now need: LLM hosting (serving 70B+ parameter models with GPU-intensive inference), prompt management (versioning, testing, and deploying prompts as first-class artifacts), RAG (Retrieval-Augmented Generation) infrastructure (vector databases, retrieval pipelines, context management), agent hosting (autonomous AI agents with tool use, memory, and multi-step reasoning), evaluation pipelines (automated quality assessment for non-deterministic outputs), and guardrails systems (safety filters, content moderation, PII detection in real-time). This isn't just "add LLM support to existing platform" — it's a fundamental architectural shift. Traditional ML platforms optimize for throughput and batch processing. AI/LLM platforms optimize for: token-level streaming, long-running agent sessions, multi-turn conversations with memory, tool/function calling, and real-time safety filtering. In 2026, the platform must support: Claude 4 (Opus/Sonnet), GPT-5, Gemini 2.5, Llama 4, Mistral Large, and dozens of specialized models — often routing between them based on task complexity, cost constraints, and quality requirements.

---

## The Analogy

Think of the evolution from traditional ML platform to AI platform like the evolution from a library to a university:

- **Traditional ML platform** = Library. Store books (models), check them out (deploy), return them (retire). Simple transactions, well-defined items, predictable workflows.
- **AI platform 2025+** = University. Libraries (model serving) still exist, but now you also need: lecture halls (agent sessions), research labs (prompt experimentation), tutoring centers (RAG), security guards (guardrails), student counselors (orchestration), and collaborative spaces (multi-agent systems). Much more complex, diverse workloads, unpredictable interactions.

The library didn't disappear — it's still needed. But the university adds many new capabilities on top.

---

## Deep Dive

### AI Platform Architecture

```yaml
Architecture:
  llm_serving_layer:
    what: "Host and serve large language models efficiently"
    components:
      model_hosting:
        vllm: "Primary LLM inference engine (PagedAttention, continuous batching)"
        tgi: "HuggingFace Text Generation Inference (alternative)"
        tensorrt_llm: "NVIDIA TensorRT-LLM (maximum throughput on NVIDIA GPUs)"
      model_routing:
        what: "Route requests to appropriate model based on complexity/cost/quality"
        logic: "Simple query → GPT-4o-mini/Claude Haiku; Complex → GPT-5/Claude Opus"
        benefit: "80% cost reduction (most queries handled by cheaper models)"
      multi_lora:
        what: "Serve multiple fine-tuned variants from single base model"
        implementation: "vLLM LoRA adapter loading (dynamic per-request)"
        benefit: "One GPU cluster serves 10+ specialized models"
      scaling:
        gpu: "H100/H200 for large models (70B+), L40S for smaller (7-13B)"
        autoscaling: "Scale based on token queue depth + request rate"
        
  prompt_management:
    what: "Version, test, and deploy prompts as first-class artifacts"
    components:
      prompt_registry:
        what: "Store prompt templates with versions"
        tracks: "Template text, variables, model, temperature, max_tokens"
        versioning: "Semantic versioning (v1.2.3), rollback support"
        
      prompt_testing:
        what: "Evaluate prompt versions before deployment"
        methods:
          - "Run against eval dataset (golden set of queries + expected answers)"
          - "A/B test prompt versions in production"
          - "LLM-as-judge scoring (quality, relevance, safety)"
          
      prompt_deployment:
        what: "Deploy prompt changes without code deployment"
        mechanism: "Config-driven (change prompt version → takes effect immediately)"
        rollback: "Instant rollback to previous prompt version"
        
  rag_infrastructure:
    what: "Retrieval-Augmented Generation pipeline platform"
    components:
      document_ingestion:
        what: "Process and index documents for retrieval"
        pipeline: "Document → chunk → embed → index in vector DB"
        tools: "LlamaIndex, LangChain, custom Spark pipelines"
        
      vector_store:
        what: "Store and query document embeddings"
        options: "pgvector, Qdrant, Pinecone, Weaviate"
        scale: "Millions to billions of embeddings"
        
      retrieval_service:
        what: "Find relevant documents for a given query"
        methods:
          - "Dense retrieval (embedding similarity)"
          - "Sparse retrieval (BM25/keyword)"
          - "Hybrid (dense + sparse with reciprocal rank fusion)"
          - "Reranking (Cohere Rerank, cross-encoder)"
        serving: "API endpoint: query → top-k relevant chunks"
        
      context_management:
        what: "Assemble context window from retrieved documents"
        challenges: "Context length limits, relevance filtering, deduplication"
        strategies: "Stuff all, map-reduce, refine, tree-summarize"
        
  agent_hosting:
    what: "Run autonomous AI agents in production"
    components:
      agent_runtime:
        what: "Execute agent reasoning loops"
        frameworks: "LangGraph, CrewAI, AutoGen, custom"
        features:
          - "Tool calling (function execution based on LLM decisions)"
          - "Memory (short-term conversation, long-term knowledge)"
          - "Multi-step reasoning (plan → execute → observe → re-plan)"
          - "Parallel tool execution (concurrent actions)"
          
      tool_registry:
        what: "Catalog of tools available to agents"
        tools:
          - "API calls (internal and external services)"
          - "Database queries (read-only by default)"
          - "Code execution (sandboxed)"
          - "File operations (within boundaries)"
          - "Web search (for information retrieval)"
        security: "Per-agent tool permissions, rate limits, sandboxing"
        
      session_management:
        what: "Manage agent sessions (potentially long-running)"
        challenges:
          - "Sessions can last minutes to hours"
          - "Memory must persist across interactions"
          - "Multiple concurrent sessions per user"
          - "State must be recoverable after failures"
        implementation: "Stateful session store (Redis) + durable execution (Temporal)"
        
      mcp_integration:
        what: "MCP (Model Context Protocol) for tool connectivity"
        purpose: "Standard interface between AI agents and external tools/data"
        benefit: "Agent can connect to any MCP server (databases, APIs, file systems)"
        
  guardrails_system:
    what: "Safety filters and content moderation for all AI outputs"
    components:
      input_guardrails:
        - "Prompt injection detection (classify adversarial inputs)"
        - "PII detection (block or redact personal info in prompts)"
        - "Topic filtering (block off-topic or prohibited queries)"
        - "Rate limiting (prevent abuse)"
        
      output_guardrails:
        - "Toxicity detection (Llama Guard 3, OpenAI Moderation API)"
        - "Hallucination detection (claim verification against sources)"
        - "PII leakage prevention (scan output for personal data)"
        - "Factual grounding check (ensure responses cite sources)"
        - "Brand safety (filter inappropriate content)"
        
      implementation:
        real_time: "< 100ms overhead per request (parallel safety checks)"
        deployment: "Sidecar pattern (guardrails container alongside model container)"
        configurability: "Per-application safety policies (stricter for healthcare, looser for creative)"
```

### LLM Gateway

```python
# AI Platform LLM Gateway — unified interface to multiple LLM providers

"""
The LLM Gateway provides a unified API across multiple LLM providers,
with built-in routing, caching, guardrails, and cost management.
"""

from typing import Optional, AsyncIterator
from dataclasses import dataclass


@dataclass
class LLMRequest:
    """Standard request to the LLM Gateway."""
    messages: list  # [{"role": "user", "content": "..."}]
    model: str = "auto"  # "auto" for intelligent routing, or specific model
    max_tokens: int = 1024
    temperature: float = 0.7
    stream: bool = False
    tools: list = None  # Function definitions for tool calling
    
    # Routing hints
    complexity: str = "auto"  # "simple", "moderate", "complex", "auto"
    cost_limit: float = None  # Max cost per request in $
    latency_target_ms: int = None  # Target response time
    
    # Safety
    safety_policy: str = "default"  # "strict", "default", "permissive"


@dataclass  
class LLMResponse:
    """Standard response from the LLM Gateway."""
    content: str
    model_used: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    cached: bool
    safety_checks: dict


class LLMGateway:
    """
    Unified LLM Gateway for the AI Platform.
    
    Features:
    - Multi-provider: OpenAI, Anthropic, Google, self-hosted (vLLM)
    - Intelligent routing: auto-select model based on query complexity
    - Semantic caching: cache similar queries
    - Guardrails: input/output safety filtering
    - Cost management: per-request and per-team budgets
    - Fallback: if primary model fails, route to backup
    - Observability: full tracing of every request
    """
    
    def __init__(self):
        self.providers = {
            "anthropic": AnthropicProvider(),  # Claude 4 Opus, Sonnet
            "openai": OpenAIProvider(),        # GPT-5, GPT-4o
            "google": GoogleProvider(),        # Gemini 2.5 Pro
            "self_hosted": VLLMProvider(),     # Llama 4, Mistral
        }
        self.router = ModelRouter()
        self.cache = SemanticCache()
        self.guardrails = GuardrailsService()
        self.cost_tracker = CostTracker()
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Process an LLM request through the full pipeline:
        1. Input guardrails
        2. Cache check
        3. Model routing
        4. LLM call
        5. Output guardrails
        6. Cost tracking
        """
        # 1. Input safety check
        input_check = await self.guardrails.check_input(
            request.messages, request.safety_policy
        )
        if not input_check.safe:
            return LLMResponse(
                content="I can't help with that request.",
                model_used="guardrails",
                input_tokens=0, output_tokens=0,
                cost_usd=0, latency_ms=input_check.latency_ms,
                cached=False, safety_checks=input_check.details,
            )
        
        # 2. Semantic cache lookup
        cached = await self.cache.lookup(request.messages)
        if cached:
            return LLMResponse(
                content=cached.content,
                model_used=cached.model,
                input_tokens=0, output_tokens=0,
                cost_usd=0, latency_ms=5,
                cached=True, safety_checks={},
            )
        
        # 3. Route to appropriate model
        if request.model == "auto":
            selected_model = await self.router.select(
                messages=request.messages,
                complexity=request.complexity,
                cost_limit=request.cost_limit,
                latency_target=request.latency_target_ms,
            )
        else:
            selected_model = request.model
        
        # 4. Call LLM
        provider = self._get_provider(selected_model)
        response = await provider.generate(request, model=selected_model)
        
        # 5. Output safety check
        output_check = await self.guardrails.check_output(
            response.content, request.safety_policy
        )
        if not output_check.safe:
            response.content = output_check.filtered_content
        
        # 6. Track cost
        await self.cost_tracker.record(
            model=selected_model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=response.cost_usd,
        )
        
        # 7. Cache response
        await self.cache.store(request.messages, response)
        
        return response
    
    async def generate_stream(
        self, request: LLMRequest
    ) -> AsyncIterator[str]:
        """Stream response tokens (for real-time UX)."""
        # Input guardrails (same as above)
        # Route to model
        # Stream from provider
        # Output guardrails applied per-chunk or post-stream
        pass


class ModelRouter:
    """
    Intelligent model routing based on query characteristics.
    
    Routing logic:
    - Simple factual queries → cheapest model (GPT-4o-mini, Claude Haiku)
    - Moderate complexity → mid-tier (Claude Sonnet, GPT-4o)
    - Complex reasoning/coding → premium (Claude Opus, GPT-5, o3)
    - Specialized tasks → fine-tuned models or specific providers
    """
    
    async def select(
        self,
        messages: list,
        complexity: str = "auto",
        cost_limit: float = None,
        latency_target: int = None,
    ) -> str:
        """Select the best model for this request."""
        
        if complexity == "auto":
            # Classify query complexity (using a small classifier)
            complexity = await self._classify_complexity(messages)
        
        model_candidates = self._get_candidates(complexity)
        
        # Filter by constraints
        if cost_limit:
            model_candidates = [m for m in model_candidates if m.estimated_cost <= cost_limit]
        if latency_target:
            model_candidates = [m for m in model_candidates if m.estimated_latency_ms <= latency_target]
        
        # Select best remaining candidate (optimize for quality within constraints)
        return model_candidates[0].name if model_candidates else "claude-4-sonnet"
    
    def _get_candidates(self, complexity: str) -> list:
        """Get model candidates for a complexity level."""
        routing_table = {
            "simple": ["gpt-4o-mini", "claude-haiku", "gemini-2.5-flash"],
            "moderate": ["claude-4-sonnet", "gpt-4o", "gemini-2.5-pro"],
            "complex": ["claude-4-opus", "gpt-5", "o3"],
            "code": ["claude-4-opus", "gpt-5", "deepseek-coder"],
        }
        return routing_table.get(complexity, routing_table["moderate"])
```

### Evaluation Infrastructure

```yaml
Evaluation_Infrastructure:
  what: "Automated quality assessment for LLM/AI outputs"
  
  eval_types:
    offline_eval:
      what: "Evaluate model/prompt on curated dataset before deployment"
      pipeline: "Eval dataset → run inference → compute metrics → report"
      metrics: "Accuracy, relevance, faithfulness, safety scores"
      tools: "RAGAS, DeepEval, custom eval pipelines"
      frequency: "Before every prompt/model change"
      
    online_eval:
      what: "Continuous quality monitoring in production"
      pipeline: "Sample production traffic → evaluate with judge → alert on degradation"
      sampling: "5-10% of traffic (cost management)"
      tools: "LangSmith, Arize Phoenix, Braintrust"
      frequency: "Continuous (hourly/daily aggregation)"
      
    human_eval:
      what: "Human annotators rate response quality"
      pipeline: "Sample responses → human ratings → calibration → metrics"
      frequency: "Weekly or monthly (expensive, high quality)"
      purpose: "Calibrate automated evaluations against human judgment"
      
  eval_dataset_management:
    versioning: "Eval sets versioned and immutable (changes = new version)"
    contamination: "Ensure eval data never appears in training/fine-tuning data"
    coverage: "Eval set covers: happy path, edge cases, adversarial, multi-turn"
    freshness: "Update eval set quarterly (add new failure modes discovered in production)"
```

### Cost Management for AI

```yaml
AI_Cost_Management:
  model_costs_2026:
    claude_4_opus: "$15/M input, $75/M output"
    claude_4_sonnet: "$3/M input, $15/M output"
    gpt_5: "$10/M input, $30/M output"
    gpt_4o_mini: "$0.15/M input, $0.60/M output"
    gemini_2_5_pro: "$7/M input, $21/M output"
    gemini_2_5_flash: "$0.075/M input, $0.30/M output"
    self_hosted_llama_4_70b: "~$4/hour GPU cost ÷ throughput"
    
  optimization:
    model_routing:
      savings: "60-80% (most queries handled by cheap models)"
      implementation: "Complexity classifier → route to appropriate tier"
      
    semantic_caching:
      savings: "20-40% (cache hits avoid LLM calls entirely)"
      implementation: "Embedding similarity → if cached response exists, return it"
      ttl: "Cache for 1-24 hours (depends on freshness requirements)"
      
    prompt_caching:
      savings: "50-90% on input tokens (for repeated system prompts)"
      provider: "Anthropic (native prompt caching), OpenAI (seed-based)"
      
    batch_api:
      savings: "50% (OpenAI Batch API for non-real-time)"
      use_for: "Evaluation pipelines, content generation, batch summarization"
      
    context_optimization:
      savings: "30-50% on input tokens"
      techniques:
        - "Compress retrieved context (summarize before sending to LLM)"
        - "Limit retrieval top-k (fewer documents = fewer input tokens)"
        - "Use shorter system prompts"
```

---

## How It Works in Practice

### AI Application Deployment

```yaml
Deployment_Flow:
  traditional_ml: "Model artifact → container → endpoint (stateless, fast)"
  llm_application:
    components:
      - "LLM (served via gateway)"
      - "Prompt template (deployed from prompt registry)"
      - "RAG pipeline (retrieval + vector DB)"
      - "Guardrails (safety filters)"
      - "Agent logic (if applicable)"
    deployment: "Application container + LLM Gateway + vector DB + guardrails sidecar"
    update: "Change prompt version (no code redeploy) or update RAG index (no model redeploy)"
```

---

## Interview Tip

> When asked about AI platform in 2025+: "The AI platform extends the traditional ML platform with five new capabilities: (1) LLM Gateway — unified API across multiple providers (Claude 4, GPT-5, Gemini 2.5, self-hosted Llama 4) with intelligent routing (simple queries → cheap models, complex → premium), semantic caching (20-40% cost savings), and automatic fallback if a provider is down. (2) Prompt management — prompts are first-class artifacts with versioning, A/B testing, and instant deployment without code changes. Like feature flags for AI behavior. (3) RAG infrastructure — document ingestion pipeline (chunk → embed → index), vector database (Qdrant/pgvector), hybrid retrieval (dense + sparse + reranking), and context management. Platform provides this so teams don't each build their own RAG pipeline. (4) Agent hosting — runtime for autonomous AI agents with tool calling, memory management, and multi-step reasoning. Using LangGraph or CrewAI. Key challenge: sessions can run for minutes/hours (not milliseconds like traditional inference), requiring stateful session management and durable execution (Temporal). (5) Guardrails — real-time safety filtering on both input (prompt injection, PII) and output (toxicity, hallucination, PII leakage). < 100ms overhead, configurable per application. The key architectural difference: traditional ML is request-response (stateless, milliseconds). AI applications are often multi-turn, stateful, long-running, and non-deterministic — requiring fundamentally different infrastructure patterns."

---

## Common Mistakes

1. **Treating LLM apps like traditional ML** — Deploying LLM application as a stateless REST endpoint. But agents need memory, conversations need state, and RAG needs retrieval context. Solution: platform must support stateful sessions, persistent memory, and multi-step execution — not just request-response.

2. **No model routing (using expensive model for everything)** — Every query goes to Claude 4 Opus at $75/M output tokens. 80% of queries are simple and could use Haiku at $1.25/M output tokens. Solution: intelligent routing — classify complexity, route appropriately. Most platforms save 60-80% this way.

3. **Building RAG from scratch per team** — Each team builds their own chunking, embedding, retrieval, and context assembly. Different approaches, duplicated infrastructure, inconsistent quality. Solution: platform provides RAG-as-a-service: ingest documents → configure retrieval → get relevant context via API.

4. **No guardrails in production** — LLM deployed without safety checks. First adversarial user extracts system prompts, second generates harmful content, third triggers PII leakage. Solution: guardrails are mandatory, not optional. Input AND output filtering. Platform enforces minimum safety policy.

5. **Ignoring evaluation until production** — Deploy LLM app, discover hallucination rate is 15% when users complain. Solution: mandatory evaluation pipeline before deployment (offline eval on curated dataset). Continuous online evaluation in production (sample 5-10%, judge with another LLM). Alert when quality degrades.

---

## Key Takeaways

- AI platform 2025+ adds: LLM gateway, prompt management, RAG infra, agent hosting, guardrails
- LLM Gateway: unified API, intelligent routing (60-80% cost savings), semantic caching, fallback
- Prompt management: version, test, deploy prompts as first-class artifacts (no code redeploy)
- RAG infrastructure: platform-provided document ingestion → retrieval → context assembly
- Agent hosting: stateful sessions, tool calling, memory, durable execution (Temporal)
- Guardrails: input + output safety filtering, < 100ms overhead, configurable per application
- Model routing: classify complexity → route to appropriate model tier (Haiku vs Opus)
- MCP (Model Context Protocol): standard interface for agent-tool connectivity
- Evaluation: mandatory offline eval before deploy + continuous online eval in production
- Architectural shift: traditional ML is stateless/fast; AI apps are stateful/long-running/non-deterministic
