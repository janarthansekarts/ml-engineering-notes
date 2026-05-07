# LLM Monitoring

## The Problem / Why This Matters

LLM (Large Language Model) monitoring is fundamentally different from traditional ML monitoring. Traditional models output numbers (probability of churn: 0.73) — easy to monitor with statistical tests. LLMs output free-form text — monitoring requires evaluating response quality, factual accuracy, toxicity, relevance, and hallucination detection. You can't compute PSI (Population Stability Index) on text outputs. You need new metrics: semantic similarity to expected answers, faithfulness to source documents, response coherence, latency per token, cost per query, and safety compliance. In 2026, with Claude 4, GPT-5, Gemini 2.5, and open-source models (Llama 4, Mistral Large) powering production applications, LLM monitoring is a top priority. The stakes: a hallucinating LLM in a customer-facing application can provide dangerous medical advice, fabricate legal citations, or leak private information from its context window. RAG (Retrieval-Augmented Generation) applications add another monitoring dimension: retrieval quality (did we find the right documents?) and grounding (did the model actually use the retrieved context?). The tooling landscape has matured rapidly — Arize Phoenix, LangSmith, Braintrust, and custom evaluation pipelines provide production-grade LLM observability.

---

## The Analogy

Think of LLM monitoring like quality control for a customer service team:

- **Traditional ML monitoring** = Monitoring a calculator. Did it compute the right number? Easy to check — compare output to expected answer.
- **LLM monitoring** = Monitoring a human customer service agent. Did they answer correctly? Were they polite? Did they make stuff up? Did they reveal confidential information? Did they stay on topic? Much harder to evaluate — requires judgment, not just comparison.

You need multiple evaluation dimensions: accuracy (correct information), safety (nothing harmful), relevance (answers the actual question), groundedness (based on real facts, not hallucination), and efficiency (reasonable response time and cost).

---

## Deep Dive

### LLM Monitoring Dimensions

```yaml
Monitoring_Dimensions:
  response_quality:
    metrics:
      - "Relevance: Does the response answer the user's question?"
      - "Coherence: Is the response logically structured and readable?"
      - "Completeness: Does it cover all aspects of the question?"
      - "Conciseness: Is it appropriately brief without unnecessary verbosity?"
    measurement: "LLM-as-judge (GPT-4/Claude evaluates responses on 1-5 scale)"
    
  factual_accuracy:
    metrics:
      - "Hallucination rate: % of responses containing fabricated information"
      - "Faithfulness: Does response only contain info from provided context?"
      - "Citation accuracy: Are referenced sources real and correctly quoted?"
    measurement: "NLI (Natural Language Inference) models, source verification"
    
  safety:
    metrics:
      - "Toxicity score: Hate speech, profanity, harmful content"
      - "PII leakage: Personal information exposed in responses"
      - "Jailbreak attempts: Adversarial prompts that bypass safety filters"
      - "Off-topic rate: Responses outside the model's intended scope"
    measurement: "Safety classifiers (Llama Guard 3, OpenAI Moderation API)"
    
  rag_quality:
    metrics:
      - "Retrieval precision: % of retrieved documents actually relevant"
      - "Retrieval recall: % of relevant documents actually retrieved"
      - "Groundedness: % of response claims supported by retrieved documents"
      - "Context utilization: How much of the retrieved context was used?"
    measurement: "RAGAS framework, custom evaluation pipelines"
    
  operational:
    metrics:
      - "Latency: Time to first token (TTFT), total response time"
      - "Token usage: Input tokens, output tokens, total cost per query"
      - "Throughput: Queries per second, tokens per second"
      - "Error rate: API failures, timeout rate, rate limit hits"
      - "Cache hit rate: % of queries served from semantic cache"
    measurement: "Standard operational metrics (Prometheus, Datadog)"
    
  user_experience:
    metrics:
      - "User satisfaction: Thumbs up/down, explicit ratings"
      - "Regeneration rate: How often users ask for a different response"
      - "Follow-up questions: Indicates incomplete/unclear first response"
      - "Session abandonment: Users leaving without getting answer"
    measurement: "Product analytics, in-app feedback collection"
```

### Hallucination Detection

```python
# Production hallucination detection system

"""
Detects when LLM generates information not supported by provided context.
Critical for RAG applications where responses must be grounded in documents.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HallucinationResult:
    """Result of hallucination check."""
    is_hallucinated: bool
    confidence: float  # 0-1
    hallucinated_claims: list  # Specific claims not in context
    grounded_claims: list  # Claims supported by context
    groundedness_score: float  # 0-1 (% claims grounded)


class HallucinationDetector:
    """
    Multi-method hallucination detection.
    
    Approaches:
    1. NLI-based: Check if response is entailed by context
    2. Claim decomposition: Break response into claims, verify each
    3. LLM-as-judge: Ask another LLM to evaluate groundedness
    """
    
    def __init__(self, method: str = "hybrid"):
        self.method = method
        # NLI model for entailment checking
        self.nli_model = self._load_nli_model()
        # LLM for claim decomposition and judgment
        self.judge_llm = self._setup_judge_llm()
    
    async def check(
        self,
        response: str,
        context: str,
        query: str,
    ) -> HallucinationResult:
        """
        Check if response is grounded in the provided context.
        """
        if self.method == "nli":
            return await self._nli_check(response, context)
        elif self.method == "claim_decomposition":
            return await self._claim_decomposition_check(response, context)
        elif self.method == "hybrid":
            return await self._hybrid_check(response, context, query)
    
    async def _claim_decomposition_check(
        self, response: str, context: str
    ) -> HallucinationResult:
        """
        1. Decompose response into atomic claims
        2. For each claim, check if it's supported by context
        3. Report groundedness score
        """
        # Step 1: Extract claims from response
        claims = await self._extract_claims(response)
        
        grounded = []
        hallucinated = []
        
        # Step 2: Verify each claim against context
        for claim in claims:
            is_supported = await self._verify_claim(claim, context)
            if is_supported:
                grounded.append(claim)
            else:
                hallucinated.append(claim)
        
        total_claims = len(claims)
        groundedness = len(grounded) / total_claims if total_claims > 0 else 1.0
        
        return HallucinationResult(
            is_hallucinated=len(hallucinated) > 0,
            confidence=0.8,  # Claim decomposition is fairly reliable
            hallucinated_claims=hallucinated,
            grounded_claims=grounded,
            groundedness_score=groundedness,
        )
    
    async def _extract_claims(self, response: str) -> list:
        """Use LLM to decompose response into atomic, verifiable claims."""
        prompt = f"""Decompose the following response into atomic claims.
Each claim should be a single factual statement that can be verified.

Response: {response}

Return each claim on a separate line, numbered:
1. [claim]
2. [claim]
..."""
        
        result = await self.judge_llm.generate(prompt)
        claims = [line.split(". ", 1)[1] for line in result.strip().split("\n") if ". " in line]
        return claims
    
    async def _verify_claim(self, claim: str, context: str) -> bool:
        """Check if a single claim is supported by context using NLI."""
        # NLI: premise=context, hypothesis=claim
        # entailment → supported, contradiction/neutral → not supported
        result = self.nli_model.predict(
            premise=context,
            hypothesis=claim,
        )
        return result.label == "entailment" and result.score > 0.7


# RAGAS-style evaluation for RAG pipelines
class RAGEvaluator:
    """
    Evaluates RAG pipeline quality using RAGAS-inspired metrics.
    
    Metrics:
    - Faithfulness: Is the answer grounded in retrieved context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Are retrieved documents actually relevant?
    - Context Recall: Did retrieval find all needed information?
    """
    
    async def evaluate(
        self,
        query: str,
        response: str,
        retrieved_contexts: list,
        ground_truth: Optional[str] = None,
    ) -> dict:
        """Compute all RAG quality metrics."""
        
        faithfulness = await self._compute_faithfulness(response, retrieved_contexts)
        answer_relevancy = await self._compute_answer_relevancy(query, response)
        context_precision = await self._compute_context_precision(query, retrieved_contexts)
        
        results = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
        }
        
        if ground_truth:
            context_recall = await self._compute_context_recall(
                ground_truth, retrieved_contexts
            )
            results["context_recall"] = context_recall
        
        # Overall RAG quality score (weighted average)
        results["overall_score"] = (
            faithfulness * 0.4 +     # Hallucination prevention is critical
            answer_relevancy * 0.3 + # Must answer the question
            context_precision * 0.3  # Retrieval must be relevant
        )
        
        return results
    
    async def _compute_faithfulness(self, response: str, contexts: list) -> float:
        """What fraction of claims in the response are supported by context?"""
        combined_context = "\n".join(contexts)
        detector = HallucinationDetector(method="claim_decomposition")
        result = await detector.check(response, combined_context, "")
        return result.groundedness_score
    
    async def _compute_answer_relevancy(self, query: str, response: str) -> float:
        """Does the response actually answer the question?"""
        prompt = f"""Rate how well the response answers the question.
Score 0-1 where 1 = perfectly answers the question, 0 = completely irrelevant.

Question: {query}
Response: {response}

Score (0-1):"""
        
        score = await self.judge_llm.generate(prompt)
        return float(score.strip())
    
    async def _compute_context_precision(self, query: str, contexts: list) -> float:
        """What fraction of retrieved documents are actually relevant to the query?"""
        relevant_count = 0
        for context in contexts:
            is_relevant = await self._is_context_relevant(query, context)
            if is_relevant:
                relevant_count += 1
        return relevant_count / len(contexts) if contexts else 0.0
```

### LLM Monitoring Architecture

```yaml
Architecture:
  tracing:
    what: "End-to-end trace of every LLM interaction"
    captures:
      - "User query (original)"
      - "Prompt template + variables"
      - "Full prompt sent to LLM (after template rendering)"
      - "Retrieved contexts (for RAG)"
      - "LLM response (raw)"
      - "Post-processed response (after guardrails)"
      - "Latency breakdown (retrieval, LLM inference, post-processing)"
      - "Token counts (input, output)"
      - "Model name and version"
      - "Cost per query"
    tools: "LangSmith, Arize Phoenix, Braintrust, OpenTelemetry with LLM spans"
    
  evaluation_pipeline:
    what: "Automated quality assessment of LLM responses"
    approaches:
      llm_as_judge:
        what: "Use a strong LLM to evaluate weaker LLM responses"
        judge_model: "Claude 4 Opus or GPT-5 (strongest available)"
        evaluated_model: "Production model (possibly smaller/cheaper)"
        criteria: "Relevance, accuracy, helpfulness, safety"
        sampling: "Evaluate 5-10% of production traffic (cost management)"
        
      reference_based:
        what: "Compare response to known-good answers"
        metrics: "ROUGE, BERTScore, semantic similarity"
        limitation: "Only works when you have reference answers"
        use_for: "QA systems with known correct answers"
        
      classifier_based:
        what: "Fine-tuned classifiers for specific quality dimensions"
        examples:
          - "Toxicity classifier (Llama Guard 3)"
          - "Relevance classifier (fine-tuned on domain data)"
          - "Completeness classifier"
        advantage: "Fast, cheap, deterministic (no LLM cost for evaluation)"
        
  alerting:
    what: "Alert when LLM quality degrades"
    metrics_monitored:
      - "Hallucination rate > 5% (measured by claim verification)"
      - "Toxicity rate > 1% (measured by safety classifier)"
      - "Average relevance score < 0.7 (measured by LLM-as-judge)"
      - "Retrieval precision < 0.6 (measured by context precision)"
      - "TTFT (Time to First Token) > 2s (measured directly)"
      - "Cost per query > budget threshold (measured from token counts)"
    
  dashboards:
    panels:
      - "Response quality trend (daily average scores)"
      - "Hallucination rate over time"
      - "Token usage and cost (daily/weekly)"
      - "Latency distribution (p50, p95, p99)"
      - "User feedback distribution (thumbs up/down)"
      - "Top failing queries (lowest quality scores)"
      - "Retrieval quality breakdown (precision, recall)"
```

### Cost Monitoring

```yaml
Cost_Monitoring:
  why_critical: "LLM API costs can explode quickly — $0.01/query × 1M queries = $10K/day"
  
  metrics:
    per_query:
      - "Input tokens (context + prompt)"
      - "Output tokens (response)"
      - "Cost = (input_tokens × input_price) + (output_tokens × output_price)"
    aggregate:
      - "Daily cost (total spend)"
      - "Cost per user / per session"
      - "Cost by model (if using multiple)"
      - "Cost by use case / feature"
      
  pricing_2026:
    claude_4_opus: "$15/M input, $75/M output"
    claude_4_sonnet: "$3/M input, $15/M output"
    gpt_5: "$10/M input, $30/M output"
    gemini_2_5_pro: "$7/M input, $21/M output"
    llama_4_self_hosted: "$0 API cost, but GPU cost ~$2-5/hour per H100"
    
  optimization:
    prompt_caching: "Cache repeated system prompts (Claude prompt caching: 90% discount)"
    semantic_caching: "Cache similar queries → return cached response (save 30-60% calls)"
    model_routing: "Simple queries → small model, complex → large model"
    response_length: "Set max_tokens appropriately (don't generate 4K tokens for yes/no)"
    batching: "Batch non-urgent requests for lower per-token pricing"
    
  alerts:
    - "Daily cost exceeds 2× normal"
    - "Single user consuming disproportionate cost"
    - "Average tokens per query increasing (prompt injection attempt?)"
    - "Cost per successful interaction increasing"
```

### Safety Monitoring

```python
# LLM safety monitoring system

"""
Monitors LLM outputs for safety violations:
- Toxicity and harmful content
- PII (Personally Identifiable Information) leakage
- Jailbreak attempts and prompt injection
- Off-topic responses (scope violations)
"""


class LLMSafetyMonitor:
    """
    Real-time safety monitoring for LLM responses.
    Runs as post-processing step (before response reaches user)
    AND as async batch analysis (for aggregate safety metrics).
    """
    
    def __init__(self):
        self.toxicity_model = self._load_toxicity_model()  # Llama Guard 3
        self.pii_detector = self._load_pii_detector()       # Presidio or custom NER
        self.jailbreak_detector = self._load_jailbreak_detector()
        
    async def check_response(
        self,
        response: str,
        user_query: str,
        context: dict,
    ) -> dict:
        """
        Check LLM response for safety violations.
        Called in real-time (must be fast: <100ms).
        """
        results = {
            "safe": True,
            "violations": [],
        }
        
        # Check 1: Toxicity
        toxicity = await self.toxicity_model.predict(response)
        if toxicity.score > 0.7:
            results["safe"] = False
            results["violations"].append({
                "type": "toxicity",
                "score": toxicity.score,
                "category": toxicity.category,
            })
        
        # Check 2: PII leakage
        pii_found = self.pii_detector.detect(response)
        if pii_found:
            results["safe"] = False
            results["violations"].append({
                "type": "pii_leakage",
                "entities": [
                    {"type": e.type, "redacted": True}
                    for e in pii_found
                ],
            })
        
        # Check 3: Jailbreak in input (detect adversarial prompts)
        jailbreak = self.jailbreak_detector.check(user_query)
        if jailbreak.detected:
            results["safe"] = False
            results["violations"].append({
                "type": "jailbreak_attempt",
                "technique": jailbreak.technique,
                "confidence": jailbreak.confidence,
            })
        
        return results
    
    async def aggregate_safety_metrics(
        self,
        interactions: list,
        time_window: str = "1h",
    ) -> dict:
        """
        Compute aggregate safety metrics for dashboards and alerting.
        Run as batch job (hourly).
        """
        total = len(interactions)
        
        metrics = {
            "total_interactions": total,
            "time_window": time_window,
            "toxicity_rate": 0.0,
            "pii_leak_rate": 0.0,
            "jailbreak_attempt_rate": 0.0,
            "blocked_response_rate": 0.0,
        }
        
        toxicity_count = sum(1 for i in interactions if i.get("toxicity_detected"))
        pii_count = sum(1 for i in interactions if i.get("pii_detected"))
        jailbreak_count = sum(1 for i in interactions if i.get("jailbreak_detected"))
        blocked_count = sum(1 for i in interactions if i.get("response_blocked"))
        
        if total > 0:
            metrics["toxicity_rate"] = toxicity_count / total
            metrics["pii_leak_rate"] = pii_count / total
            metrics["jailbreak_attempt_rate"] = jailbreak_count / total
            metrics["blocked_response_rate"] = blocked_count / total
        
        return metrics
```

---

## How It Works in Practice

### LLM Monitoring Stack

```yaml
Production_Stack:
  tracing: "LangSmith or Arize Phoenix — captures full LLM interaction chains"
  evaluation: "RAGAS + LLM-as-judge (Claude 4 Opus evaluates 10% of production traffic)"
  safety: "Llama Guard 3 (real-time) + custom PII detector (Presidio)"
  cost: "Custom token tracking → BigQuery → Looker dashboards"
  alerting: "PagerDuty for safety violations, Slack for quality degradation"
  
  typical_monitoring_budget:
    evaluation_cost: "10% of production traffic × judge LLM cost ≈ 10-15% of total LLM spend"
    safety_models: "Llama Guard self-hosted on shared GPU — $200-500/month"
    tracing_platform: "LangSmith/Phoenix — $500-2000/month depending on volume"
    total: "15-25% additional cost on top of base LLM spend"
```

---

## Interview Tip

> When asked about LLM monitoring: "LLM monitoring is fundamentally different from traditional ML because outputs are free-form text, not numbers. I monitor five dimensions: (1) Response quality — using LLM-as-judge (Claude 4 Opus evaluates 10% of production responses on relevance, coherence, completeness) plus user feedback (thumbs up/down). (2) Hallucination detection — claim decomposition (break response into atomic claims, verify each against retrieved context using NLI). For RAG applications, I use RAGAS metrics: faithfulness (are claims grounded?), context precision (is retrieval relevant?), answer relevancy (does it answer the question?). (3) Safety — Llama Guard 3 for toxicity detection (real-time, <50ms), Presidio for PII detection, custom jailbreak detector. Unsafe responses are blocked before reaching users. (4) Cost — token tracking per query, daily/weekly spend, cost per use-case. Alert when daily cost exceeds 2× normal (might indicate prompt injection inflating token usage). (5) Operational — TTFT (time to first token), total latency, error rate, rate limit hits. The tradeoff: evaluation costs 10-15% of base LLM spend (because you're using another LLM to judge), so I sample strategically — 100% safety monitoring (non-negotiable), 10% quality evaluation (sufficient for statistical significance), and aggregate cost metrics on everything."

---

## Common Mistakes

1. **No hallucination monitoring** — Deploying RAG application without measuring whether responses are grounded in retrieved documents. Users report wrong answers weeks later. Solution: automated faithfulness checking (claim decomposition + NLI) on sampled traffic from day one.

2. **Using the same model as judge** — GPT-4 evaluating GPT-4 responses (or same model class). The judge has the same blind spots as the production model. Solution: use a different model family as judge (Claude judges GPT, or vice versa). Cross-model evaluation catches more issues.

3. **Not monitoring retrieval quality separately** — Only monitoring final response quality. When quality degrades, you can't tell if it's a retrieval problem (wrong documents) or a generation problem (model hallucinating). Solution: monitor retrieval precision and recall independently. Log which documents were retrieved for each query.

4. **Ignoring cost until the bill arrives** — No token tracking, no cost attribution. Monthly cloud bill is $50K and nobody knows why. Solution: track tokens per query from day one. Attribute cost to features/use-cases. Set budget alerts at 50%, 80%, 100% of expected spend.

5. **Safety monitoring only on outputs** — Checking responses for toxicity but not checking inputs for adversarial prompts. Jailbreak attempts succeed because input was never validated. Solution: monitor BOTH input (jailbreak/injection detection) and output (toxicity/PII). Log and alert on suspicious input patterns even if the model handles them correctly.

---

## Key Takeaways

- LLM monitoring has five dimensions: quality, factual accuracy, safety, RAG quality, and operational
- Hallucination detection: claim decomposition + NLI (Natural Language Inference) verification against context
- RAGAS metrics for RAG: faithfulness, answer relevancy, context precision, context recall
- LLM-as-judge: strong model evaluates weaker model (sample 5-10% of traffic to manage cost)
- Safety monitoring: Llama Guard 3 (toxicity), Presidio (PII), custom jailbreak detection — real-time
- Cost monitoring: track tokens per query, set budget alerts, attribute cost to features
- Evaluation cost overhead: 10-15% of base LLM spend (trade-off: monitoring cost vs. quality assurance)
- Tracing: capture full interaction chains (query → prompt → retrieval → response → post-processing)
- User feedback: thumbs up/down is the cheapest and most direct quality signal
- Safety is non-negotiable: 100% of responses must pass safety checks before reaching users
