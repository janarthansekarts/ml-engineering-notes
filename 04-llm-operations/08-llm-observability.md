# LLM Observability

## The Problem / Why This Matters

LLM applications are black boxes by default — you send a prompt, get a response, but have no visibility into what happened in between. When a user reports "the AI gave me a wrong answer," how do you debug it? Which model was used? What was the full prompt (system + context + user message)? How many tokens were consumed? What was the latency breakdown? Did guardrails modify the response? Was RAG (Retrieval-Augmented Generation) context retrieved successfully? Traditional APM (Application Performance Monitoring) tools track HTTP status codes and latency, but LLM applications need deeper observability: prompt/response logging, token-level cost tracking, quality scoring, chain-of-thought visibility, and multi-step trace correlation (especially for agents that make multiple LLM calls). In 2026, LLM observability has become its own category with dedicated tools (LangSmith, Langfuse, Arize Phoenix, Weights & Biases Weave) because generic monitoring tools don't understand LLM-specific concerns: prompt versioning, token economics, generation quality, and hallucination detection. Without proper observability, you're operating blind — unable to debug issues, optimize costs, improve quality, or detect degradation.

---

## The Analogy

Think of LLM observability like a flight recorder (black box) on an aircraft:

- **Traditional monitoring** = Knowing the plane took off and landed (HTTP 200). No details about what happened during the flight.
- **LLM observability** = Full flight data recorder: every control input (prompt), every sensor reading (model confidence), fuel consumption (tokens), altitude changes (chain-of-thought steps), weather conditions (context retrieved), and landing quality (response quality score).
- **Traces** = Flight path recording. For multi-step agents, you see the complete journey: first LLM call → tool use → second LLM call → final answer. Like tracking the plane through multiple waypoints.
- **When something goes wrong** = Investigators pull the black box and reconstruct exactly what happened. Without it, you're guessing why the crash occurred.

---

## Deep Dive

### LLM Observability Dimensions

```yaml
Observability_Dimensions:
  traces:
    what: "End-to-end visibility into multi-step LLM operations"
    captures:
      - "Full request lifecycle (user input → preprocessing → model call → postprocessing → response)"
      - "Multi-step chains (RAG: embed query → retrieve → construct prompt → generate)"
      - "Agent loops (tool selection → tool call → observation → next step → final answer)"
      - "Parent-child relationships between spans"
    why: "Debug complex failures, understand latency breakdown, identify bottlenecks"
    
  spans:
    what: "Individual operations within a trace"
    types:
      llm_call:
        attributes:
          - "Model name and version"
          - "Full prompt (system + user + few-shot)"
          - "Full response (raw model output)"
          - "Input tokens, output tokens, total tokens"
          - "Latency (TTFT, total generation time)"
          - "Temperature, max_tokens, top_p (generation parameters)"
          - "Cost (calculated from tokens × pricing)"
          
      retrieval:
        attributes:
          - "Query embedding"
          - "Number of documents retrieved"
          - "Relevance scores"
          - "Retrieved text chunks"
          - "Vector DB (Database) latency"
          
      tool_call:
        attributes:
          - "Tool name and parameters"
          - "Tool response"
          - "Execution time"
          - "Success/failure status"
          
      guardrail:
        attributes:
          - "Guardrail type (input/output)"
          - "Decision (allow/block/modify)"
          - "Confidence score"
          - "Triggered rules"
          
  metrics:
    latency:
      - "TTFT (Time To First Token): prompt processing time"
      - "TPS (Tokens Per Second): generation speed"
      - "E2E (End-to-End) latency: total request time"
      - "Per-step latency breakdown"
      
    cost:
      - "Cost per request (input + output tokens × pricing)"
      - "Cost per user session"
      - "Cost per feature"
      - "Daily/weekly/monthly spend"
      
    quality:
      - "LLM-as-judge scores (sampled)"
      - "User feedback (thumbs up/down)"
      - "Format compliance rate"
      - "Hallucination rate (sampled)"
      
    volume:
      - "Requests per second/minute/hour"
      - "Tokens consumed (input vs output)"
      - "Cache hit rate"
      - "Error rate (timeouts, rate limits, safety blocks)"
```

### Tracing Implementation

```python
# LLM observability with OpenTelemetry + custom LLM spans

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from dataclasses import dataclass
from typing import Optional
import time

# Initialize OpenTelemetry tracer
provider = TracerProvider()
exporter = OTLPSpanExporter(endpoint="http://observability-backend:4317")
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("llm-application")


@dataclass
class LLMSpanAttributes:
    """Standard attributes for LLM observability spans."""
    model: str
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    ttft_ms: float
    total_latency_ms: float
    temperature: float
    max_tokens: int
    finish_reason: str  # "stop", "length", "content_filter"


class LLMObservability:
    """Comprehensive LLM observability wrapper."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.tracer = trace.get_tracer(service_name)
    
    async def trace_llm_call(
        self,
        model: str,
        messages: list[dict],
        generation_params: dict,
        parent_span=None,
    ) -> tuple[str, LLMSpanAttributes]:
        """Trace a single LLM API call with full observability."""
        
        context = trace.set_span_in_context(parent_span) if parent_span else None
        
        with self.tracer.start_as_current_span(
            name=f"llm.{model}",
            context=context,
            attributes={
                "llm.model": model,
                "llm.temperature": generation_params.get("temperature", 1.0),
                "llm.max_tokens": generation_params.get("max_tokens", 4096),
                "llm.prompt_tokens": 0,  # Updated after call
            }
        ) as span:
            start_time = time.perf_counter()
            
            try:
                # Make LLM API call
                response = await self._call_llm(model, messages, generation_params)
                
                total_latency = (time.perf_counter() - start_time) * 1000
                
                # Extract token usage
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                cost = self._calculate_cost(model, input_tokens, output_tokens)
                
                # Set span attributes
                span.set_attribute("llm.input_tokens", input_tokens)
                span.set_attribute("llm.output_tokens", output_tokens)
                span.set_attribute("llm.total_tokens", input_tokens + output_tokens)
                span.set_attribute("llm.cost_usd", cost)
                span.set_attribute("llm.latency_ms", total_latency)
                span.set_attribute("llm.finish_reason", response.finish_reason)
                span.set_attribute("llm.response_length", len(response.content))
                
                # Log prompt and response (careful with PII)
                span.set_attribute("llm.prompt", self._sanitize_for_logging(str(messages)))
                span.set_attribute("llm.response", self._sanitize_for_logging(response.content))
                
                span.set_status(trace.StatusCode.OK)
                
                return response.content, LLMSpanAttributes(
                    model=model,
                    prompt=str(messages),
                    response=response.content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                    ttft_ms=response.ttft_ms,
                    total_latency_ms=total_latency,
                    temperature=generation_params.get("temperature", 1.0),
                    max_tokens=generation_params.get("max_tokens", 4096),
                    finish_reason=response.finish_reason,
                )
                
            except Exception as e:
                span.set_status(trace.StatusCode.ERROR, str(e))
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                raise
    
    async def trace_rag_pipeline(
        self,
        query: str,
        retriever_config: dict,
    ):
        """Trace a full RAG pipeline with sub-spans."""
        
        with self.tracer.start_as_current_span("rag.pipeline") as pipeline_span:
            pipeline_span.set_attribute("rag.query", query)
            
            # Span 1: Query embedding
            with self.tracer.start_as_current_span("rag.embed_query") as embed_span:
                embedding = await self._embed_query(query)
                embed_span.set_attribute("embedding.dimensions", len(embedding))
                embed_span.set_attribute("embedding.model", "text-embedding-3-large")
            
            # Span 2: Vector retrieval
            with self.tracer.start_as_current_span("rag.retrieve") as retrieve_span:
                documents = await self._retrieve(embedding, retriever_config)
                retrieve_span.set_attribute("retrieval.num_results", len(documents))
                retrieve_span.set_attribute("retrieval.top_score", documents[0].score if documents else 0)
                retrieve_span.set_attribute("retrieval.min_score", documents[-1].score if documents else 0)
            
            # Span 3: Context construction
            with self.tracer.start_as_current_span("rag.build_context") as context_span:
                context = self._build_context(documents)
                context_span.set_attribute("context.total_tokens", self._count_tokens(context))
                context_span.set_attribute("context.num_chunks", len(documents))
            
            # Span 4: LLM generation
            messages = [
                {"role": "system", "content": "Answer based on the provided context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
            ]
            response, attrs = await self.trace_llm_call(
                model="claude-4-sonnet",
                messages=messages,
                generation_params={"temperature": 0.1, "max_tokens": 1024},
                parent_span=pipeline_span,
            )
            
            pipeline_span.set_attribute("rag.response", response)
            pipeline_span.set_attribute("rag.total_cost", attrs.cost_usd)
            
            return response
    
    def _sanitize_for_logging(self, text: str) -> str:
        """Remove PII before logging (critical for compliance)."""
        import re
        # Redact common PII patterns
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        # Truncate if too long
        if len(text) > 10000:
            text = text[:10000] + "...[TRUNCATED]"
        return text
```

### LLM Observability Platforms

```yaml
Platforms:
  langsmith:
    provider: "LangChain"
    strengths:
      - "Native LangChain/LangGraph integration"
      - "Trace visualization for complex chains"
      - "Dataset management for evaluation"
      - "Prompt versioning and testing"
      - "Online evaluation (LLM-as-judge on traces)"
    best_for: "Teams using LangChain/LangGraph"
    pricing: "Free tier (5K traces/month), paid starts $39/month"
    
  langfuse:
    provider: "Open source (self-hosted or cloud)"
    strengths:
      - "Framework-agnostic (works with any LLM framework)"
      - "Self-hostable (data stays on your infrastructure)"
      - "Cost tracking with custom pricing tables"
      - "Prompt management and versioning"
      - "Session tracking (group traces by user session)"
      - "Score/annotate traces for quality tracking"
    best_for: "Teams wanting open-source, self-hosted observability"
    pricing: "Free (self-hosted), cloud starts $59/month"
    
  arize_phoenix:
    provider: "Arize AI"
    strengths:
      - "Embedding visualization (see retrieval quality)"
      - "Trace analysis with automatic insights"
      - "Drift detection (prompt/response distribution shifts)"
      - "LLM evaluation built-in"
    best_for: "Teams focused on RAG quality and retrieval analysis"
    
  weights_and_biases_weave:
    provider: "Weights & Biases"
    strengths:
      - "Integrated with W&B experiment tracking"
      - "Model evaluation and comparison"
      - "Dataset versioning"
      - "Collaborative annotation"
    best_for: "Teams already using W&B for ML experiment tracking"
    
  opentelemetry_llm:
    what: "Standard tracing protocol with LLM semantic conventions"
    strengths:
      - "Vendor-neutral (send to any backend)"
      - "Standard span attributes for LLM operations"
      - "Works with existing observability stack (Jaeger, Grafana Tempo)"
    best_for: "Teams with existing OTel (OpenTelemetry) infrastructure"
    standard: "OpenTelemetry GenAI Semantic Conventions (2026 stable)"
```

### Dashboards and Alerts

```yaml
Dashboards:
  operational_dashboard:
    panels:
      - title: "Request Volume"
        metric: "requests_per_minute by model"
        
      - title: "Latency Distribution"
        metric: "P50, P95, P99 latency by model"
        alert: "P95 > 5s"
        
      - title: "Error Rate"
        metric: "errors / total_requests"
        alert: "> 1%"
        breakdown: "By error type (timeout, rate_limit, content_filter, model_error)"
        
      - title: "Token Usage"
        metric: "total_tokens by model (input vs output)"
        
      - title: "Cost (Real-time)"
        metric: "cumulative_cost_today vs budget"
        alert: "80% of daily budget reached"
        
  quality_dashboard:
    panels:
      - title: "Quality Scores (LLM-as-Judge)"
        metric: "avg_quality_score (sampled 5%)"
        alert: "Drops > 10% from 7-day baseline"
        
      - title: "User Feedback"
        metric: "thumbs_up_rate by feature"
        alert: "Drops below 70%"
        
      - title: "Format Compliance"
        metric: "valid_json_rate, length_compliance_rate"
        
      - title: "Safety Triggers"
        metric: "guardrail_block_rate by category"
        alert: "Spike > 2× baseline"
        
      - title: "Cache Hit Rate"
        metric: "cache_hits / total_requests"
        target: "> 30%"
        
  debugging_view:
    features:
      - "Search traces by user_id, session_id, or content"
      - "Filter by error status, high latency, low quality"
      - "View full prompt and response for any trace"
      - "Trace timeline (visual span waterfall)"
      - "Compare responses across model versions"
      
  alerts:
    critical:
      - "Error rate > 5% for 5 minutes → Page on-call"
      - "All models returning errors → Page on-call"
      - "Safety violation detected → Immediate review"
    warning:
      - "P95 latency > 10s → Slack notification"
      - "Daily cost > 80% budget → Slack notification"
      - "Quality score drop > 10% → Slack notification"
      - "Cache hit rate < 10% → Engineering review"
```

---

## How It Works in Practice

### Production Observability Setup

```yaml
Production_Setup:
  data_flow:
    1: "Application wraps LLM calls with trace instrumentation"
    2: "Spans exported to observability backend (Langfuse/LangSmith/OTel collector)"
    3: "Backend stores traces, computes metrics, indexes for search"
    4: "Dashboards display real-time operational metrics"
    5: "Alerts fire on threshold breaches"
    6: "Engineers debug issues by searching and inspecting traces"
    
  privacy_considerations:
    problem: "Logging full prompts/responses may contain PII"
    solutions:
      - "PII redaction before logging (regex + NER model)"
      - "Sampling (log only 10% of traces in detail)"
      - "Separate PII-containing fields with restricted access"
      - "Retention policies (delete detailed logs after 30 days)"
      - "Encryption at rest for trace storage"
    compliance: "GDPR, HIPAA, SOC2 requirements for log retention and access"
    
  retention:
    detailed_traces: "30 days (full prompt/response)"
    aggregated_metrics: "13 months (for year-over-year comparison)"
    cost_data: "Indefinite (for budgeting and forecasting)"
    
  sampling:
    strategy: "Head-based sampling with tail-based upgrade"
    default_rate: "10% of requests get full trace detail"
    upgrade_conditions:
      - "Errors: always trace in full"
      - "High latency (> P95): always trace in full"
      - "Low quality score: always trace in full"
      - "New model/prompt version: 100% for first 24 hours"
```

---

## Interview Tip

> When asked about LLM observability: "I implement observability at three levels: (1) Trace-level — every LLM request gets a trace with spans for each step (embedding, retrieval, generation, guardrails). Spans capture: model, full prompt/response (PII-redacted), token counts, latency, cost, finish reason. For agents, traces show the complete reasoning loop (tool calls, observations, decisions). I use OpenTelemetry GenAI semantic conventions so traces work with any backend. (2) Metric-level — real-time dashboards showing: request volume, P50/P95/P99 latency, error rate by type (timeout, rate limit, content filter), token usage, cost accumulation vs budget, cache hit rate. Quality metrics: LLM-as-judge scores (sampled 5%), user feedback rate, format compliance. (3) Debugging workflows — when a user reports a bad response, I search traces by user_id/session_id, find the exact trace, see the full prompt (with retrieved context), model response, guardrail decisions, and quality score. Can reproduce and fix in minutes vs hours of guessing. Key operational patterns: sample 10% for detailed traces (100% for errors/slow/new versions), PII redaction before logging (compliance), 30-day retention for detailed traces, alerts on quality degradation and cost spikes. Platform choice: Langfuse for self-hosted/open-source needs, LangSmith for LangChain-heavy teams, OTel for teams with existing observability infrastructure."

---

## Common Mistakes

1. **Logging prompts/responses without PII redaction** — Full prompt logs contain customer emails, account numbers, medical information. One log export or security breach exposes all of it. Solution: PII detection and redaction BEFORE logging. Separate PII fields with access controls.

2. **Only monitoring latency and errors** — Traditional APM metrics tell you the system is "up" but not whether it's producing good outputs. A model returning garbage at 200ms latency shows green on traditional dashboards. Solution: add quality metrics (LLM-as-judge sampling, user feedback, format compliance).

3. **No trace correlation for multi-step agents** — Agent makes 5 LLM calls + 3 tool calls, but each logged independently. When debugging, you can't reconstruct the full reasoning chain. Solution: proper parent-child span relationships, trace IDs propagated through all steps.

4. **100% logging without sampling** — Logging every trace in full detail at 10K RPM (Requests Per Minute). Storage costs explode, query performance degrades, and 99% of traces are never looked at. Solution: sample 10% for detailed traces, always capture errors/slow/anomalous at 100%.

5. **No baseline for quality alerts** — Setting a static alert threshold (quality < 4.0) without understanding normal variance. Results in alert fatigue (too many false alerts) or missed degradation (threshold too low). Solution: use rolling 7-day baseline with percentage deviation alerts (> 10% drop from baseline → alert).

---

## Key Takeaways

- LLM observability goes beyond APM: need prompt/response logging, token tracking, quality metrics
- Traces with spans: capture full lifecycle (embedding → retrieval → generation → guardrails)
- Essential metrics: latency (TTFT, E2E), cost (per request, daily), quality (judge scores, feedback), errors
- PII redaction before logging: compliance requirement, not optional
- Sampling strategy: 10% detailed + 100% for errors/slow/new versions
- Platforms: Langfuse (open-source), LangSmith (LangChain ecosystem), OTel (vendor-neutral)
- Dashboards: operational (latency, errors, cost) + quality (scores, feedback, compliance)
- Debugging workflow: search by user/session → inspect full trace → reproduce → fix
- Alert on quality degradation (rolling baseline), not just infrastructure metrics
- Retention: 30 days detailed, 13 months aggregated, comply with GDPR/HIPAA
