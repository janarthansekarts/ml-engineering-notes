# LLM Cost Management

## The Problem / Why This Matters

LLM API costs scale linearly with usage and can become the dominant line item in your cloud bill. A single GPT-5 API call costs $0.01-0.10 depending on token count. At 1 million requests per day, that's $10,000-$100,000 per day — $3.6-$36 million per year. Many companies have been surprised by LLM bills 10-100× their initial estimates because they didn't plan for scale. Cost management isn't about being cheap — it's about building sustainable AI products. The key levers: model selection (use the cheapest model that meets quality requirements), prompt optimization (shorter prompts = fewer tokens = lower cost), caching (identical or similar queries reuse previous responses), routing (send simple queries to cheap models, complex to expensive ones), and batching (batch APIs are 50% cheaper). In 2026, with multiple providers (OpenAI, Anthropic, Google, open-source via vLLM), the cost optimization landscape is rich: you can achieve 80-90% cost reduction while maintaining quality through intelligent routing, caching, and model selection. The companies winning in AI are not the ones spending the most — they're the ones spending efficiently.

---

## The Analogy

Think of LLM cost management like managing a law firm's billing:

- **GPT-5 / Claude 4 Opus** = Senior partner ($1,000/hour). Brilliant at complex cases but absurdly expensive for routine work. You wouldn't have them draft a standard letter.
- **GPT-5-mini / Claude 4 Sonnet** = Associate ($300/hour). Great for most work, handles 80% of tasks at a fraction of the cost.
- **Llama-4-8B / Mistral** = Paralegal ($50/hour). Handles routine, well-defined tasks perfectly. Classification, extraction, simple generation.
- **Caching** = Template library. Why redraft a standard NDA from scratch every time? Use the template (cached response) and customize only when needed.
- **Routing** = Reception triaging cases. Simple questions go to the paralegal, medium complexity to associates, only truly novel/complex cases to the senior partner.
- **Batching** = Batch processing overnight. Non-urgent work (reports, analysis) gets done in off-peak hours at discounted rates.

---

## Deep Dive

### Token Economics (2026 Pricing)

```yaml
Token_Pricing:
  understanding_tokens:
    what: "LLMs charge per token (roughly 4 characters or 0.75 words in English)"
    example: "A 500-word response ≈ 670 tokens"
    pricing: "Most providers charge differently for input vs output tokens"
    output_expensive: "Output tokens cost 3-5× more than input tokens (generation is harder)"
    
  current_pricing_2026:
    frontier_models:
      gpt_5:
        input: "$10/million tokens"
        output: "$30/million tokens"
        context: "256K tokens"
        
      claude_4_opus:
        input: "$15/million tokens"
        output: "$75/million tokens"
        context: "200K tokens"
        
      gemini_2_5_pro:
        input: "$7/million tokens"
        output: "$21/million tokens"
        context: "2M tokens"
        
    mid_tier_models:
      gpt_5_mini:
        input: "$1/million tokens"
        output: "$3/million tokens"
        quality: "90% of GPT-5 on most tasks"
        
      claude_4_sonnet:
        input: "$3/million tokens"
        output: "$15/million tokens"
        quality: "85-95% of Opus on most tasks"
        
      gemini_2_5_flash:
        input: "$0.50/million tokens"
        output: "$1.50/million tokens"
        quality: "Good for straightforward tasks"
        
    self_hosted_open_source:
      llama_4_70b:
        cost: "$0.30-0.80/million tokens (on H100, depends on batch size)"
        quality: "Comparable to mid-tier commercial models"
        
      llama_4_8b:
        cost: "$0.05-0.15/million tokens"
        quality: "Good for classification, extraction, simple generation"
        
    batch_apis:
      openai_batch: "50% discount, results within 24 hours"
      anthropic_batch: "50% discount, async processing"
      use_for: "Non-real-time workloads: analytics, content generation, evaluation"
      
  cost_calculation_example:
    scenario: "Customer support chatbot, 100K conversations/day"
    avg_conversation: "5 turns, 200 tokens input + 300 tokens output per turn"
    per_conversation:
      input_tokens: "200 × 5 = 1,000"
      output_tokens: "300 × 5 = 1,500"
    daily_tokens:
      input: "100K × 1,000 = 100M input tokens"
      output: "100K × 1,500 = 150M output tokens"
    daily_cost:
      gpt_5: "$100M × $10/M + $150M × $30/M = $1,000 + $4,500 = $5,500/day"
      gpt_5_mini: "$100M × $1/M + $150M × $3/M = $100 + $450 = $550/day"
      self_hosted_llama_8b: "$250M × $0.10/M = $25/day"
    annual_cost:
      gpt_5: "$2,007,500/year"
      gpt_5_mini: "$200,750/year"
      self_hosted_llama_8b: "$9,125/year + infrastructure"
    insight: "10× cost difference between model tiers. Smart routing saves millions."
```

### Cost Optimization Strategies

```yaml
Strategy_1_Model_Routing:
  what: "Route queries to the cheapest model that can handle them"
  
  implementation:
    simple_router:
      approach: "Classify query complexity, route accordingly"
      tiers:
        simple: "Classification, yes/no, extraction → Llama-8B ($0.10/M)"
        medium: "Summarization, standard Q&A, drafting → GPT-5-mini ($2/M avg)"
        complex: "Reasoning, creative writing, code → GPT-5 ($20/M avg)"
        
    complexity_classifier:
      approach: "Train a small classifier on query difficulty"
      features:
        - "Query length"
        - "Domain keywords"
        - "Question type (factual vs reasoning)"
        - "Required output format complexity"
      cost: "Classifier runs on CPU in < 5ms, costs nearly nothing"
      
    cascade_routing:
      approach: "Try cheap model first, escalate if quality is low"
      flow: |
        1. Send query to Llama-8B
        2. Check output quality (confidence score, format validity)
        3. If quality < threshold → resend to GPT-5-mini
        4. If still insufficient → resend to GPT-5
      benefit: "80% of queries handled by cheapest model"
      risk: "Added latency for escalated queries (2-3× latency)"
      
  savings: "60-80% cost reduction vs sending everything to frontier model"

Strategy_2_Prompt_Optimization:
  what: "Reduce token count in prompts without losing quality"
  
  techniques:
    system_prompt_compression:
      before: "You are an extremely helpful, knowledgeable, and friendly customer service representative for Acme Corporation. You should always be polite and professional. Always greet the customer warmly..." (150 tokens)
      after: "Acme support agent. Be helpful, professional, concise." (12 tokens)
      saving: "138 tokens per request × 100K requests = 13.8M tokens/day saved"
      
    few_shot_reduction:
      before: "10 few-shot examples in every prompt (2000 tokens)"
      after: "Fine-tune model on examples → 0 few-shot tokens needed"
      saving: "2000 tokens per request (major for high-volume)"
      
    output_length_control:
      technique: "Set max_tokens parameter, instruct concise responses"
      example: "Respond in 2-3 sentences maximum."
      saving: "Output tokens cost 3-5× input — controlling length has outsized impact"
      
    structured_output:
      technique: "Request JSON/structured output instead of verbose prose"
      saving: "Structured outputs typically 50-70% fewer tokens than prose"

Strategy_3_Caching:
  what: "Store and reuse previous responses for identical/similar queries"
  
  levels:
    exact_match_cache:
      what: "Hash the full prompt, return cached response if seen before"
      hit_rate: "10-30% for support bots (common questions repeat)"
      implementation: "Redis/Memcached with TTL (Time To Live)"
      cost: "~$0 per cached response (Redis is pennies)"
      
    semantic_cache:
      what: "Find similar (not identical) queries, reuse response"
      implementation: "Embed query → vector similarity search → if >0.95 similar, return cached"
      hit_rate: "30-50% (paraphrases of same question)"
      risk: "May return stale or slightly wrong answer for edge cases"
      ttl: "Set TTL based on content volatility (prices: 1hr, policies: 24hr, facts: 7d)"
      
    prompt_caching_provider:
      what: "Provider-side prompt prefix caching"
      anthropic: "Cached prompt prefix: 90% discount on repeated prefix tokens"
      openai: "Automatic prefix caching: 50% discount on cached prefix"
      benefit: "Long system prompts + few-shot examples cached server-side"
      
  savings: "30-60% cost reduction (depends on query repetitiveness)"

Strategy_4_Batching:
  what: "Accumulate non-urgent requests and process in bulk at discount"
  
  batch_api_details:
    openai_batch_api:
      discount: "50% off standard pricing"
      sla: "Results within 24 hours"
      use_cases:
        - "Nightly content generation"
        - "Batch evaluation/scoring"
        - "Data classification/labeling"
        - "Document summarization pipeline"
        
    self_hosted_batching:
      technique: "Accumulate requests, process with high batch size"
      benefit: "Higher GPU utilization = lower per-token cost"
      batch_size: "32-128 concurrent requests (continuous batching in vLLM)"
      
  savings: "50% for batch-eligible workloads"
```

### Cost Monitoring and Budgets

```python
# LLM cost tracking and budget management

from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict


@dataclass
class LLMUsageRecord:
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    user_id: str
    feature: str       # Which product feature generated this call
    cached: bool       # Was this served from cache?
    routed_from: str   # Original model before routing (if downgraded)


class CostTracker:
    """Track and control LLM spending."""
    
    # Pricing per million tokens (2026)
    PRICING = {
        "gpt-5": {"input": 10.0, "output": 30.0},
        "gpt-5-mini": {"input": 1.0, "output": 3.0},
        "claude-4-opus": {"input": 15.0, "output": 75.0},
        "claude-4-sonnet": {"input": 3.0, "output": 15.0},
        "gemini-2.5-flash": {"input": 0.5, "output": 1.5},
        "llama-4-8b": {"input": 0.10, "output": 0.10},
    }
    
    def __init__(self, daily_budget_usd: float, alert_threshold: float = 0.8):
        self.daily_budget = daily_budget_usd
        self.alert_threshold = alert_threshold
        self.usage_records: list[LLMUsageRecord] = []
        self.daily_spend: defaultdict = defaultdict(float)
        
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a single LLM call."""
        pricing = self.PRICING.get(model)
        if not pricing:
            raise ValueError(f"Unknown model: {model}")
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    def record_usage(self, record: LLMUsageRecord) -> dict:
        """Record usage and check budget."""
        self.usage_records.append(record)
        
        today = record.timestamp.strftime("%Y-%m-%d")
        self.daily_spend[today] += record.cost_usd
        
        # Budget check
        current_spend = self.daily_spend[today]
        budget_utilization = current_spend / self.daily_budget
        
        result = {"recorded": True, "daily_spend": current_spend}
        
        if budget_utilization >= 1.0:
            result["alert"] = "BUDGET_EXCEEDED"
            result["action"] = "Switch to cheapest model or reject requests"
        elif budget_utilization >= self.alert_threshold:
            result["alert"] = "BUDGET_WARNING"
            result["action"] = "Consider switching to cheaper models"
            
        return result
    
    def get_cost_breakdown(self, days: int = 7) -> dict:
        """Get cost breakdown by model, feature, and user."""
        cutoff = datetime.now() - timedelta(days=days)
        recent = [r for r in self.usage_records if r.timestamp > cutoff]
        
        by_model = defaultdict(float)
        by_feature = defaultdict(float)
        by_user = defaultdict(float)
        cache_savings = 0.0
        
        for record in recent:
            by_model[record.model] += record.cost_usd
            by_feature[record.feature] += record.cost_usd
            by_user[record.user_id] += record.cost_usd
            if record.cached:
                # Estimate savings from caching
                cache_savings += record.cost_usd  # Would have cost this much
                
        return {
            "total_cost": sum(r.cost_usd for r in recent),
            "by_model": dict(by_model),
            "by_feature": dict(by_feature),
            "top_users": sorted(by_user.items(), key=lambda x: x[1], reverse=True)[:10],
            "cache_savings": cache_savings,
            "avg_daily": sum(r.cost_usd for r in recent) / days,
        }


class BudgetEnforcer:
    """Enforce spending limits with automatic model downgrade."""
    
    def __init__(self, tracker: CostTracker):
        self.tracker = tracker
        
    def get_allowed_model(self, requested_model: str, feature: str) -> str:
        """Determine which model to actually use based on budget."""
        today = datetime.now().strftime("%Y-%m-%d")
        current_spend = self.tracker.daily_spend[today]
        utilization = current_spend / self.tracker.daily_budget
        
        # Budget tiers
        if utilization < 0.5:
            return requested_model  # Use whatever was requested
        elif utilization < 0.8:
            # Downgrade frontier to mid-tier
            downgrades = {
                "gpt-5": "gpt-5-mini",
                "claude-4-opus": "claude-4-sonnet",
            }
            return downgrades.get(requested_model, requested_model)
        elif utilization < 1.0:
            # Force cheapest models
            return "gemini-2.5-flash"
        else:
            # Over budget — reject or use cached only
            return "BUDGET_EXCEEDED"
```

### ROI (Return on Investment) Calculation

```yaml
ROI_Framework:
  cost_of_llm:
    direct: "API costs (tokens consumed)"
    infrastructure: "GPU instances for self-hosted models"
    engineering: "Team time building and maintaining LLM features"
    guardrails: "Safety systems, evaluation, monitoring"
    
  value_generated:
    cost_savings:
      example: "Customer support: LLM handles 60% of tickets that previously needed humans"
      calculation: |
        Human agent cost: $25/hour, handles 8 tickets/hour = $3.12/ticket
        LLM cost: $0.05/ticket (500 tokens × $0.10/1K)
        Savings per ticket: $3.07
        At 10,000 tickets/day × 60% automated = 6,000 tickets
        Daily savings: $18,420
        Annual savings: $6.7M
        Annual LLM cost: 6,000 × $0.05 × 365 = $109,500
        ROI: 61× return
        
    revenue_increase:
      example: "AI-powered product recommendations increase conversion 15%"
      
    speed_improvement:
      example: "Code review that took 2 hours now takes 10 minutes"
      
  optimization_priority:
    highest_impact:
      - "Model routing (60-80% savings)"
      - "Caching (30-60% savings)"
      - "Prompt optimization (20-40% savings)"
    medium_impact:
      - "Batch processing (50% on eligible workloads)"
      - "Fine-tuning small models to replace large ones (70-90% savings per query)"
    lower_impact:
      - "Token count micro-optimization"
      - "Provider negotiation (10-20% at scale)"
```

---

## How It Works in Practice

### Cost Management Operations

```yaml
Operations:
  daily:
    - "Monitor spend vs budget (automated alerts at 80%, 100%)"
    - "Check cache hit rate (should be >30% for repeat-heavy workloads)"
    - "Review top cost drivers by feature and user"
    
  weekly:
    - "Cost breakdown report by model, feature, team"
    - "Identify optimization opportunities (expensive queries that could be routed cheaper)"
    - "Review routing decisions (are queries going to appropriate tier?)"
    
  monthly:
    - "Full cost optimization review"
    - "Evaluate new models (cheaper models that maintain quality)"
    - "Fine-tuning ROI analysis (could fine-tuned small model replace expensive API?)"
    - "Negotiate provider pricing (if spend > $50K/month, get custom pricing)"
    
  quarterly:
    - "Architecture review (self-host vs API? Multi-provider strategy?)"
    - "Budget forecasting for next quarter"
    - "ROI review of all LLM-powered features"
```

---

## Interview Tip

> When asked about LLM cost management: "I treat LLM costs like any engineering optimization — measure, then optimize the biggest levers: (1) Model routing — I classify query complexity and route to the cheapest model that meets quality requirements. Simple queries (classification, extraction) go to Llama-8B at $0.10/M tokens, medium queries to GPT-5-mini at $2/M, only complex reasoning to GPT-5 at $20/M. This alone saves 60-80% vs sending everything to frontier. (2) Caching — exact-match cache (Redis, 10-30% hit rate) + semantic cache (embedding similarity, 30-50% hit rate) + provider prompt caching (Anthropic: 90% off cached prefix). (3) Prompt optimization — compress system prompts, minimize few-shot examples (or fine-tune to eliminate them), control output length. Output tokens cost 3-5× input, so shorter responses have outsized savings. (4) Batching — non-real-time workloads (evaluation, classification, content generation) use batch APIs at 50% discount. Operationally: I set daily budgets with automatic model downgrade (budget 80% → force mid-tier, 100% → force cheapest), track cost by feature/user/model with dashboards, and review weekly to find optimization opportunities. Real example: reduced a chatbot from $5,500/day to $550/day by routing 80% of queries to GPT-5-mini (quality difference was negligible for those query types)."

---

## Common Mistakes

1. **Using frontier models for everything** — Sending classification tasks ("is this spam? yes/no") to GPT-5 at $20/M tokens when Llama-8B at $0.10/M gives identical accuracy for binary classification. Solution: benchmark multiple models on YOUR tasks and use the cheapest that meets your quality bar.

2. **Ignoring output token costs** — Focusing on input optimization while letting the model generate verbose 2000-token responses. Output tokens cost 3-5× input tokens. Solution: set max_tokens, instruct concise responses, use structured output (JSON is more token-efficient than prose).

3. **No caching strategy** — Processing every request from scratch even when 30% are identical or near-identical. Solution: implement exact-match cache (Redis, trivial to add) + semantic cache (embedding similarity) + use provider prompt caching features.

4. **Not tracking costs by feature** — Monthly bill is $50K but you don't know which product features are driving it. Can't optimize what you can't measure. Solution: tag every API call with feature name, user ID, and log costs per-call. Build dashboards showing cost by feature.

5. **Over-optimizing prematurely** — Building complex routing and caching systems before product-market fit. At 100 requests/day, your LLM bill is $5. Engineering time on optimization costs more than the savings. Solution: optimize when costs exceed $1K/month or when you have validated scaling plans.

---

## Key Takeaways

- Model routing saves 60-80%: classify complexity, route simple→cheap, complex→expensive
- Caching saves 30-60%: exact-match (Redis) + semantic (embedding similarity) + provider prefix caching
- Output tokens cost 3-5× input: controlling response length has outsized impact
- Batch APIs save 50%: use for non-real-time workloads (evaluation, classification, generation)
- Budget enforcement: automatic model downgrade when approaching limits, alerts at 80%
- Track by feature/user/model: can't optimize what you can't measure
- Fine-tuning ROI: small fine-tuned model can replace expensive API for specific tasks (90% savings)
- Provider prompt caching: Anthropic (90% off cached prefix), OpenAI (50% off cached prefix)
- At scale ($50K+/month): negotiate custom pricing, consider self-hosting for stable workloads
- Self-hosted open-source: 10-100× cheaper per token for high-volume, predictable workloads
