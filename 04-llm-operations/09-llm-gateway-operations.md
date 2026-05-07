# LLM Gateway Operations

## The Problem / Why This Matters

Production LLM applications rarely use a single model from a single provider. You use GPT-5 for complex reasoning, Claude 4 Sonnet for long-context tasks, Gemini 2.5 Flash for cost-sensitive queries, and self-hosted Llama for data-sensitive workloads. Each provider has different APIs, authentication, rate limits, pricing, and failure modes. Without a gateway layer, you end up with provider-specific code scattered across your application, no unified rate limiting, no automatic failover when a provider goes down, inconsistent logging, and no ability to switch models without code changes. An LLM gateway is a centralized proxy layer that sits between your application and LLM providers, providing: unified API (one interface regardless of backend model), routing (send requests to appropriate providers based on rules), failover (automatic retry with different provider on failure), rate limiting (prevent any single consumer from exhausting quotas), cost controls (enforce budgets, track spending), and observability (unified logging across all providers). In 2026, LLM gateways are essential infrastructure for any organization using multiple models — they're the control plane for your AI operations.

---

## The Analogy

Think of an LLM gateway like a CDN (Content Delivery Network) or load balancer, but for AI:

- **Without gateway** = Every application directly calls different restaurants for food delivery. Each restaurant has its own phone number, menu format, payment system, and delivery terms. Your app needs to know all the details of every restaurant.
- **With gateway** = A food delivery platform (DoorDash/Uber Eats). One app, one payment method, unified tracking. The platform handles: routing to the right restaurant, switching to another if one is closed, rate limiting orders during peak hours, cost tracking across all orders.
- **Failover** = If your primary restaurant is closed (provider outage), the platform automatically routes to an equivalent restaurant (fallback model).
- **Rate limiting** = The platform limits how many orders one person can place per hour (prevents one consumer from exhausting shared quotas).
- **API key management** = You use one platform account regardless of how many restaurants you order from.

---

## Deep Dive

### Gateway Architecture

```yaml
Gateway_Architecture:
  components:
    request_router:
      what: "Determines which provider/model handles each request"
      routing_strategies:
        priority_based: "Try provider A first, B on failure, C as last resort"
        load_balanced: "Distribute across providers based on weights"
        content_based: "Route based on request characteristics (complexity, length)"
        cost_based: "Route to cheapest provider that meets quality requirements"
        latency_based: "Route to fastest available provider"
        geography_based: "Route to provider in user's region (data residency)"
        
    rate_limiter:
      what: "Controls request flow to prevent quota exhaustion"
      levels:
        per_user: "Max 100 requests/minute per user"
        per_team: "Max 1000 requests/minute per team"
        per_provider: "Stay within provider's rate limits"
        global: "Total system capacity cap"
      algorithms:
        token_bucket: "Burst-friendly, refills at steady rate"
        sliding_window: "Precise per-window limiting"
        
    failover_manager:
      what: "Handles provider failures with automatic fallback"
      strategies:
        immediate_retry: "Same provider, immediate retry (for transient errors)"
        provider_fallback: "Different provider (for sustained outages)"
        model_downgrade: "Cheaper model if primary unavailable"
        circuit_breaker: "Stop calling failed provider, auto-recover after cooldown"
        
    cost_controller:
      what: "Enforces spending limits and optimizes costs"
      features:
        - "Per-team daily/monthly budgets"
        - "Per-request cost estimation (before calling)"
        - "Automatic model downgrade at budget thresholds"
        - "Cost-based routing (cheapest viable option)"
        
    api_key_vault:
      what: "Centralized secret management for provider API keys"
      benefits:
        - "Applications don't hold provider keys directly"
        - "Key rotation without application changes"
        - "Audit trail for key usage"
        - "Per-team key isolation"
        
    request_transformer:
      what: "Translates between unified gateway API and provider-specific formats"
      handles:
        - "Message format differences (OpenAI vs Anthropic vs Google)"
        - "Parameter naming (max_tokens vs maxOutputTokens)"
        - "Streaming format differences (SSE variations)"
        - "Tool/function calling format differences"
```

### Gateway Implementation

```python
# LLM Gateway with routing, failover, and rate limiting

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import asyncio
import time
from collections import defaultdict


class ProviderStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


@dataclass
class ProviderConfig:
    name: str
    models: list[str]
    endpoint: str
    api_key: str
    rate_limit_rpm: int      # Requests per minute
    rate_limit_tpm: int      # Tokens per minute
    priority: int            # Lower = higher priority
    cost_per_1m_input: float
    cost_per_1m_output: float


@dataclass
class RoutingDecision:
    provider: str
    model: str
    reason: str


class CircuitBreaker:
    """Prevents repeated calls to failing providers."""
    
    def __init__(self, failure_threshold: int = 5, recovery_time_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time_seconds
        self.failure_counts: dict[str, int] = defaultdict(int)
        self.last_failure_time: dict[str, float] = {}
        self.status: dict[str, ProviderStatus] = {}
        
    def record_success(self, provider: str):
        """Reset failure count on success."""
        self.failure_counts[provider] = 0
        self.status[provider] = ProviderStatus.HEALTHY
        
    def record_failure(self, provider: str):
        """Increment failure count, trip breaker if threshold reached."""
        self.failure_counts[provider] += 1
        self.last_failure_time[provider] = time.time()
        
        if self.failure_counts[provider] >= self.failure_threshold:
            self.status[provider] = ProviderStatus.DOWN
            
    def is_available(self, provider: str) -> bool:
        """Check if provider is available (not circuit-broken)."""
        status = self.status.get(provider, ProviderStatus.HEALTHY)
        
        if status == ProviderStatus.HEALTHY:
            return True
            
        if status == ProviderStatus.DOWN:
            # Check if recovery time has elapsed
            last_failure = self.last_failure_time.get(provider, 0)
            if time.time() - last_failure > self.recovery_time:
                # Allow one test request (half-open state)
                self.status[provider] = ProviderStatus.DEGRADED
                return True
            return False
            
        # DEGRADED: allow requests (testing recovery)
        return True


class TokenBucketRateLimiter:
    """Per-consumer rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # Tokens per second
        self.buckets: dict[str, dict] = {}
        
    def allow_request(self, consumer_id: str, tokens_needed: int = 1) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()
        
        if consumer_id not in self.buckets:
            self.buckets[consumer_id] = {
                "tokens": self.capacity,
                "last_refill": now,
            }
        
        bucket = self.buckets[consumer_id]
        
        # Refill tokens based on elapsed time
        elapsed = now - bucket["last_refill"]
        bucket["tokens"] = min(
            self.capacity,
            bucket["tokens"] + elapsed * self.refill_rate
        )
        bucket["last_refill"] = now
        
        # Check if enough tokens
        if bucket["tokens"] >= tokens_needed:
            bucket["tokens"] -= tokens_needed
            return True
        return False


class LLMGateway:
    """Production LLM gateway with routing, failover, and controls."""
    
    def __init__(self, providers: list[ProviderConfig]):
        self.providers = {p.name: p for p in providers}
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = TokenBucketRateLimiter(capacity=100, refill_rate=2)
        
    async def route_request(
        self,
        messages: list[dict],
        model_preference: Optional[str] = None,
        consumer_id: str = "default",
        routing_strategy: str = "priority",
    ) -> tuple[str, dict]:
        """Route request to appropriate provider with failover."""
        
        # 1. Rate limit check
        if not self.rate_limiter.allow_request(consumer_id):
            raise RateLimitError(f"Rate limit exceeded for {consumer_id}")
        
        # 2. Determine routing order
        candidates = self._get_routing_candidates(model_preference, routing_strategy)
        
        # 3. Try each candidate with failover
        last_error = None
        for candidate in candidates:
            provider = self.providers[candidate.provider]
            
            # Check circuit breaker
            if not self.circuit_breaker.is_available(candidate.provider):
                continue
            
            try:
                response = await self._call_provider(
                    provider=provider,
                    model=candidate.model,
                    messages=messages,
                )
                self.circuit_breaker.record_success(candidate.provider)
                return response, {"provider": candidate.provider, "model": candidate.model}
                
            except ProviderError as e:
                self.circuit_breaker.record_failure(candidate.provider)
                last_error = e
                continue  # Try next candidate
        
        # All candidates exhausted
        raise AllProvidersFailedError(f"All providers failed. Last error: {last_error}")
    
    def _get_routing_candidates(
        self, 
        model_preference: Optional[str],
        strategy: str
    ) -> list[RoutingDecision]:
        """Determine provider ordering based on strategy."""
        
        available = [
            p for p in self.providers.values()
            if self.circuit_breaker.is_available(p.name)
        ]
        
        if strategy == "priority":
            available.sort(key=lambda p: p.priority)
        elif strategy == "cost":
            available.sort(key=lambda p: p.cost_per_1m_input + p.cost_per_1m_output)
        elif strategy == "round_robin":
            # Rotate through available providers
            pass
        
        candidates = []
        for provider in available:
            model = model_preference if model_preference in provider.models else provider.models[0]
            candidates.append(RoutingDecision(
                provider=provider.name,
                model=model,
                reason=f"{strategy} routing"
            ))
        
        return candidates
    
    async def _call_provider(self, provider: ProviderConfig, model: str, messages: list[dict]):
        """Make actual API call to provider (implementation per provider)."""
        # Provider-specific API call logic
        pass


class RateLimitError(Exception):
    pass

class ProviderError(Exception):
    pass

class AllProvidersFailedError(Exception):
    pass
```

### Gateway Solutions (2026)

```yaml
Gateway_Solutions:
  open_source:
    litellm:
      what: "Python proxy that provides OpenAI-compatible API for 100+ models"
      features:
        - "Unified API (OpenAI format) → translates to Anthropic, Google, etc."
        - "Load balancing across providers"
        - "Fallbacks (provider A fails → try provider B)"
        - "Budget management (per-team spending limits)"
        - "Rate limiting (per-user, per-team)"
        - "Spend tracking and logging"
        - "Virtual keys (abstract away provider API keys)"
      deployment: "Docker container or Python package"
      use_when: "Need quick multi-provider proxy with budget controls"
      
    portkey:
      what: "AI gateway with advanced routing and observability"
      features:
        - "Conditional routing (route based on request properties)"
        - "Automatic retries with exponential backoff"
        - "Request caching (semantic and exact)"
        - "Guardrails integration"
        - "Load balancing with health checks"
      deployment: "Cloud-hosted or self-hosted (open-source gateway)"
      
    kong_ai_gateway:
      what: "Kong API gateway extended for AI workloads"
      features:
        - "Built on Kong (mature API gateway)"
        - "Rate limiting, authentication, logging"
        - "AI-specific plugins (token counting, prompt templating)"
        - "Multi-provider routing"
      use_when: "Already using Kong for API management"
      
  commercial:
    azure_ai_gateway:
      what: "Azure API Management with AI capabilities"
      features:
        - "Token-based rate limiting"
        - "Multi-model routing (Azure OpenAI + others)"
        - "Built-in content safety"
        - "Enterprise compliance (SOC2, HIPAA)"
        
    aws_bedrock_gateway:
      what: "AWS Bedrock as unified multi-model access"
      features:
        - "Single API for Claude, Llama, Mistral, Titan, etc."
        - "Guardrails for Bedrock (built-in safety)"
        - "Model evaluation capabilities"
        - "Fine-tuning and provisioned throughput"
        
  build_vs_buy:
    build_your_own:
      when:
        - "Highly custom routing logic"
        - "Specific compliance requirements"
        - "Deep integration with internal systems"
      effort: "2-4 weeks for basic gateway, 2-3 months for production-grade"
      
    use_existing:
      when:
        - "Standard routing/failover needs"
        - "Want to move fast (deploy in days)"
        - "Small team (can't maintain custom infrastructure)"
      recommendation: "Start with LiteLLM (free, open-source, covers 90% of use cases)"
```

### Advanced Routing Patterns

```yaml
Advanced_Routing:
  semantic_routing:
    what: "Route based on the content/intent of the request"
    example: |
      - Medical questions → Claude 4 Opus (best at nuanced reasoning)
      - Code generation → GPT-5 (best at code)
      - Simple Q&A → Gemini Flash (cheapest, fast enough)
      - Creative writing → Claude 4 Sonnet (best creative output)
    implementation: "Intent classifier (fine-tuned model) determines routing"
    
  quality_aware_routing:
    what: "Route based on historical quality scores per model per task type"
    implementation: |
      Track quality scores (LLM-as-judge) per model per task category.
      Route new requests to model with highest historical quality for that category.
      Periodically re-evaluate (A/B test) to detect model improvements.
      
  latency_budget_routing:
    what: "Route based on available time budget"
    example: |
      - Streaming chat (user waiting): fastest available model
      - Background processing: cheapest model (latency doesn't matter)
      - SLA-bound (Service Level Agreement): model that meets latency SLA
      
  canary_routing:
    what: "Send small % of traffic to new model/version for evaluation"
    implementation: |
      - 90% → current production model
      - 10% → candidate model (new version or different provider)
      - Compare quality metrics between groups
      - Promote candidate if quality ≥ production
      
  data_residency_routing:
    what: "Route based on data sovereignty requirements"
    example: |
      - EU user data → EU-hosted model (Azure EU, self-hosted in EU)
      - US user data → US providers (OpenAI, Anthropic)
      - Sensitive data → self-hosted model (never leaves your infrastructure)
```

---

## How It Works in Practice

### Production Gateway Operations

```yaml
Operations:
  deployment:
    topology: "Gateway runs as a stateless service (Kubernetes deployment)"
    scaling: "Horizontal scaling (add replicas for more throughput)"
    latency_overhead: "< 5ms added latency (proxy routing + rate limit check)"
    availability: "99.99% (gateway itself is critical path)"
    
  configuration:
    routing_rules: "YAML/JSON config, hot-reloaded without restart"
    provider_keys: "Stored in secrets manager (AWS Secrets Manager, Vault)"
    rate_limits: "Configurable per team/user, adjustable without deploy"
    budgets: "Per-team monthly budgets, alerts at 80%/100%"
    
  incident_response:
    provider_outage:
      detection: "Circuit breaker trips after 5 consecutive failures"
      response: "Automatic failover to next priority provider"
      notification: "Alert team that primary is down, traffic rerouted"
      recovery: "Circuit breaker half-opens after 60s, tests with single request"
      
    rate_limit_hit:
      detection: "Provider returns 429 (Too Many Requests)"
      response: "Queue requests, retry with exponential backoff"
      escalation: "If sustained, shift traffic to secondary provider"
      
    cost_spike:
      detection: "Daily spend exceeds 120% of average"
      response: "Auto-downgrade to cheaper models for non-critical traffic"
      investigation: "Identify source (runaway script, attack, organic growth)"
      
  monitoring:
    gateway_health:
      - "Requests per second (throughput)"
      - "Added latency (gateway overhead)"
      - "Error rate (gateway-level failures)"
      - "Provider availability (per-provider health)"
    routing_metrics:
      - "Requests per provider (distribution)"
      - "Failover events (frequency, from→to)"
      - "Rate limit rejections (per consumer)"
      - "Cost per provider (spend distribution)"
```

---

## Interview Tip

> When asked about LLM gateway operations: "I use an LLM gateway as the control plane for all model interactions: (1) Unified API — applications call one endpoint with one format (OpenAI-compatible). The gateway translates to each provider's specific API (Anthropic, Google, Azure, self-hosted). This means switching models requires zero application code changes. (2) Intelligent routing — priority-based (primary provider first, fallback on failure), cost-based (cheapest model meeting quality threshold), content-based (different models for different task types), and canary routing (10% to new model for A/B testing). (3) Failover with circuit breakers — if a provider fails 5 consecutive requests, circuit breaker opens (stops trying), traffic routes to fallback, breaker half-opens after 60s to test recovery. Zero manual intervention for provider outages. (4) Rate limiting and cost control — token bucket per consumer (team/user), per-provider rate limit respect, daily budget enforcement with automatic model downgrade at thresholds. (5) Security — API keys centralized in secrets manager (teams get virtual keys, never see provider keys), audit trail for all requests, data residency routing for compliance. Tool choice: LiteLLM for quick setup (covers 90% of use cases), custom gateway only if you need unique routing logic or deep integration with internal systems."

---

## Common Mistakes

1. **Hardcoding provider APIs in application code** — Direct OpenAI/Anthropic SDK calls scattered across 50 files. When you need to switch providers, add failover, or change routing, you modify application code everywhere. Solution: centralize all LLM calls through a gateway layer — applications call one endpoint, gateway handles the rest.

2. **No failover strategy** — Using a single provider. When they have an outage (happens to every provider multiple times per year), your entire AI system goes down. Solution: configure at least two providers with automatic failover. Test failover regularly (chaos engineering for LLM infrastructure).

3. **Rate limiting at provider level only** — Relying on OpenAI's rate limits to protect you. But when you hit their limit, ALL your users get errors simultaneously. Solution: implement your own rate limiting at the gateway (per-user, per-team) so one runaway consumer doesn't exhaust quotas for everyone.

4. **No cost controls in the gateway** — Gateway routes requests without budget awareness. A misconfigured batch job sends 1 million requests to GPT-5 overnight → $50K unexpected bill. Solution: per-team budgets enforced at the gateway level, automatic model downgrade or request rejection when budget is reached.

5. **Gateway as single point of failure** — Running one gateway instance. If it goes down, no LLM calls work. Solution: deploy gateway as stateless service with multiple replicas behind a load balancer. Health checks and auto-scaling. The gateway must be more available than the providers it fronts.

---

## Key Takeaways

- LLM gateway = centralized proxy for all model interactions (routing, failover, rate limiting, cost control)
- Unified API: one interface for applications regardless of backend provider
- Routing strategies: priority-based, cost-based, content-based, latency-based, canary
- Failover: circuit breaker pattern (trip after N failures, auto-recover after cooldown)
- Rate limiting: per-consumer token bucket (prevent quota exhaustion by one consumer)
- Cost control: per-team budgets, automatic model downgrade at thresholds
- API key management: centralized in secrets manager, teams get virtual keys
- Tools: LiteLLM (quick open-source), Portkey (advanced routing), Azure/AWS (enterprise)
- Gateway overhead: < 5ms latency, must be highly available (99.99%)
- Test failover regularly: provider outages happen, verify your fallbacks work
