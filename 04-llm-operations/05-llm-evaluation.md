# LLM Evaluation

## The Problem / Why This Matters

How do you know if your LLM application is good? Traditional ML has clear metrics — accuracy, F1, AUC (Area Under the Curve). But LLM outputs are free-form text where "correct" has many valid forms. A summarization might be factually correct but miss the key point. A code generation might compile but use deprecated patterns. A chatbot response might be accurate but rude. LLM evaluation is the hardest unsolved problem in applied AI — there is no single metric that captures "quality." You need a multi-dimensional evaluation framework: correctness (factually right), helpfulness (actually useful), safety (no harmful content), format compliance (follows instructions), and domain-specific criteria (medical accuracy, legal precision, code correctness). In 2026, the evaluation landscape includes: automated benchmarks (MMLU, HumanEval), LLM-as-judge (using one model to evaluate another), human evaluation (gold standard but expensive), and task-specific metrics (ROUGE for summarization, pass@k for code). Effective evaluation is what lets you make data-driven decisions about model selection, prompt changes, fine-tuning, and deployment — without it, you're flying blind in production.

---

## The Analogy

Think of LLM evaluation like grading essays vs grading math tests:

- **Math test (traditional ML)** = One right answer. Either the student got 42 or they didn't. Easy to score automatically. Accuracy = correct answers / total questions.
- **Essay grading (LLM evaluation)** = Multiple valid approaches, subjective quality dimensions. Is the argument coherent? Is the evidence strong? Is the writing clear? Different graders may disagree. No single number captures essay quality.
- **LLM-as-judge** = Having a senior professor grade the essays (using another LLM to evaluate). Reasonably calibrated, scalable, but may have blind spots or biases.
- **Human evaluation** = Having multiple expert humans grade each essay. Gold standard but expensive (100 essays × 3 graders × $5/grade = $1,500 per evaluation round).
- **Automated metrics** = Word count, vocabulary level, grammar score. Measurable but doesn't capture true quality. A grammatically perfect essay can be completely wrong.

---

## Deep Dive

### Evaluation Dimensions

```yaml
Evaluation_Dimensions:
  correctness:
    what: "Is the output factually accurate?"
    measures:
      - "Factual accuracy (verifiable claims are true)"
      - "No hallucinations (doesn't invent information)"
      - "Logical consistency (reasoning is valid)"
    metrics: "Factual accuracy score, hallucination rate"
    
  helpfulness:
    what: "Does it actually solve the user's problem?"
    measures:
      - "Answers the actual question asked"
      - "Actionable (user can follow the advice)"
      - "Complete (doesn't omit important information)"
      - "Appropriate depth (not too brief, not too verbose)"
    metrics: "Task completion rate, user satisfaction score"
    
  safety:
    what: "Is the output safe and appropriate?"
    measures:
      - "No harmful instructions (weapons, self-harm, illegal)"
      - "No bias or discrimination"
      - "No PII (Personally Identifiable Information) leakage"
      - "Respects content boundaries"
    metrics: "Safety violation rate, toxicity score"
    
  format_compliance:
    what: "Does it follow the specified output format?"
    measures:
      - "JSON validity (if JSON requested)"
      - "Length constraints (within min/max tokens)"
      - "Structure (has required sections)"
      - "Formatting (markdown, bullets, etc.)"
    metrics: "Format compliance rate, parse success rate"
    
  instruction_following:
    what: "Does it adhere to all instructions in the prompt?"
    measures:
      - "Persona maintained (stays in character)"
      - "Constraints respected (no discussion of competitors, etc.)"
      - "Style adherence (tone, language level)"
    metrics: "Constraint violation rate"
    
  domain_specific:
    code: "Compiles, passes tests, follows style, no security vulnerabilities"
    medical: "Clinically accurate, appropriate disclaimers, cites evidence"
    legal: "Legally sound, proper citations, jurisdictionally appropriate"
    customer_service: "Resolves issue, empathetic tone, escalates appropriately"
```

### Automated Benchmarks

```yaml
Benchmarks:
  general_knowledge:
    mmlu:
      name: "MMLU (Massive Multitask Language Understanding)"
      what: "57 subjects, 14K multiple-choice questions"
      measures: "Broad factual knowledge and reasoning"
      limitation: "Multiple choice — doesn't test generation quality"
      
    mmlu_pro:
      what: "Harder version of MMLU with 10 choices (vs 4)"
      measures: "Same as MMLU but more discriminating"
      
  reasoning:
    gsm8k:
      what: "Grade-school math word problems"
      measures: "Multi-step mathematical reasoning"
      scoring: "Exact match on final numerical answer"
      
    math:
      what: "Competition-level math problems"
      measures: "Advanced mathematical reasoning"
      
    arc:
      name: "ARC (AI2 Reasoning Challenge)"
      what: "Science questions requiring reasoning"
      
  code:
    humaneval:
      what: "164 Python function completion problems"
      measures: "Code generation from docstrings"
      scoring: "pass@k (passes unit tests at k attempts)"
      limitation: "Only Python, simple functions, may be contaminated in training data"
      
    swe_bench:
      what: "Real GitHub issues requiring code changes"
      measures: "Practical software engineering ability"
      scoring: "Resolution rate (does the generated patch fix the issue?)"
      
    mbpp:
      name: "MBPP (Mostly Basic Python Problems)"
      what: "1000 entry-level Python problems"
      
  instruction_following:
    ifeval:
      name: "IFEval (Instruction Following Evaluation)"
      what: "500 verifiable instruction-following tasks"
      measures: "Can the model follow specific formatting/content constraints?"
      example: "Write a response with exactly 3 paragraphs, each starting with a capital letter"
      
  chat_quality:
    mt_bench:
      name: "MT-Bench (Multi-Turn Benchmark)"
      what: "80 multi-turn questions judged by GPT-4"
      measures: "Conversation quality across turns"
      scoring: "1-10 score from LLM judge"
      
    chatbot_arena:
      what: "Human preference between two model responses (blind)"
      measures: "Overall human preference"
      scoring: "Elo rating system"
      status: "Most trusted open evaluation (LMSYS)"
      
  limitations_of_benchmarks:
    - "Test contamination (models may have seen test questions in training)"
    - "Multiple choice doesn't test generation"
    - "Static benchmarks don't capture production-specific quality"
    - "Public benchmarks are gamed (optimized for without real improvement)"
    - "Don't measure YOUR domain-specific quality"
```

### LLM-as-Judge

```yaml
LLM_as_Judge:
  what: "Use a strong LLM (GPT-5, Claude 4 Opus) to evaluate another model's output"
  
  approaches:
    pointwise:
      what: "Rate a single response on a scale"
      prompt: |
        Rate this response on a 1-5 scale for helpfulness.
        
        User query: {query}
        Response: {response}
        
        Score (1-5):
        Reasoning:
        
    pairwise:
      what: "Compare two responses and pick the better one"
      prompt: |
        Which response better answers the user's query?
        
        User query: {query}
        Response A: {response_a}
        Response B: {response_b}
        
        Better response (A or B):
        Reasoning:
      advantage: "More reliable than absolute scoring (relative judgment is easier)"
      
    reference_based:
      what: "Compare response against a reference answer"
      prompt: |
        Given the reference answer, rate the candidate response for correctness.
        
        Query: {query}
        Reference answer: {reference}
        Candidate response: {candidate}
        
        Correctness (1-5):
        
  multi_dimensional:
    prompt: |
      Evaluate this response on the following dimensions (1-5 each):
      
      Query: {query}
      Response: {response}
      
      1. Correctness: Is the information factually accurate?
      2. Helpfulness: Does it solve the user's problem?
      3. Clarity: Is it well-written and easy to understand?
      4. Safety: Is it appropriate and harmless?
      5. Completeness: Does it cover all relevant aspects?
      
      For each dimension, provide a score and brief reasoning.
      
  best_practices:
    use_structured_output: "Request JSON output for reliable parsing"
    multiple_judges: "Run 3 judges, take majority (reduces noise)"
    calibration: "Include known-quality examples to verify judge accuracy"
    position_bias: "In pairwise, randomize which response is A vs B (judges prefer A)"
    verbosity_bias: "Judges may prefer longer responses regardless of quality"
    
  reliability:
    agreement_with_humans: "70-85% agreement (decent but not perfect)"
    systematic_biases:
      - "Prefers longer responses (verbosity bias)"
      - "Prefers response A in pairwise (position bias)"
      - "Favors its own model family outputs"
      - "Misses subtle factual errors"
    mitigation:
      - "Swap positions and average scores"
      - "Penalize length explicitly"
      - "Use a different model family as judge"
      - "Validate with human labels on subset"
```

### Human Evaluation

```yaml
Human_Evaluation:
  when_needed:
    - "Validating LLM-as-judge reliability (calibration)"
    - "Subtle quality dimensions (tone, empathy, creativity)"
    - "High-stakes domains (medical, legal — need expert judgment)"
    - "Debugging failure modes (understanding WHY something is bad)"
    
  setup:
    evaluators: "3-5 per sample (for inter-annotator agreement)"
    rubric: "Clear scoring criteria with examples for each score level"
    interface: "Labeling platform (Scale AI, Surge AI, Argilla, Label Studio)"
    volume: "100-500 samples per evaluation round (statistical power)"
    
  rubric_example:
    dimension: "Helpfulness"
    scale:
      5: "Completely resolves the user's question with actionable detail"
      4: "Mostly resolves the question, minor gaps"
      3: "Partially helpful but missing key information"
      2: "Tangentially related but doesn't address the core question"
      1: "Completely unhelpful or off-topic"
    anchor_examples:
      5_example: "User asks 'How do I fix error X?' → Response provides exact steps that work"
      1_example: "User asks 'How do I fix error X?' → Response discusses unrelated topic"
      
  inter_annotator_agreement:
    metric: "Cohen's Kappa or Krippendorff's Alpha"
    target: "κ > 0.6 (substantial agreement)"
    low_agreement_response: "Revise rubric, provide more examples, retrain evaluators"
    
  cost:
    calculation: "500 samples × 3 evaluators × $2/evaluation = $3,000 per round"
    frequency: "Monthly for ongoing monitoring, per-release for model changes"
```

### Task-Specific Evaluation Metrics

```python
# Comprehensive LLM evaluation framework

from dataclasses import dataclass
from typing import Optional

@dataclass
class EvaluationResult:
    query: str
    response: str
    reference: Optional[str]
    
    # Automated metrics
    format_valid: bool       # Did it output valid format (JSON, markdown, etc.)
    length_compliant: bool   # Within specified length constraints
    
    # LLM-judge scores (1-5)
    correctness: float
    helpfulness: float
    safety: float
    
    # Task-specific
    task_metric: Optional[float]  # e.g., code passes tests, factual accuracy
    
    # Metadata
    model: str
    prompt_version: str
    latency_ms: float
    tokens_used: int


class LLMEvaluator:
    """Production LLM evaluation pipeline."""
    
    def __init__(self, judge_model: str = "claude-4-opus"):
        self.judge_model = judge_model
        
    async def evaluate_batch(self, samples: list[dict]) -> list[EvaluationResult]:
        """Evaluate a batch of model outputs."""
        results = []
        for sample in samples:
            # 1. Automated checks (instant, free)
            format_valid = self._check_format(sample["response"], sample.get("format_spec"))
            length_ok = self._check_length(sample["response"], sample.get("max_tokens"))
            
            # 2. LLM-as-judge (costs tokens but scalable)
            judge_scores = await self._llm_judge(
                query=sample["query"],
                response=sample["response"],
                reference=sample.get("reference"),
            )
            
            # 3. Task-specific metric
            task_score = self._task_metric(sample)
            
            results.append(EvaluationResult(
                query=sample["query"],
                response=sample["response"],
                reference=sample.get("reference"),
                format_valid=format_valid,
                length_compliant=length_ok,
                correctness=judge_scores["correctness"],
                helpfulness=judge_scores["helpfulness"],
                safety=judge_scores["safety"],
                task_metric=task_score,
                model=sample["model"],
                prompt_version=sample["prompt_version"],
                latency_ms=sample["latency_ms"],
                tokens_used=sample["tokens_used"],
            ))
        return results
    
    def _check_format(self, response: str, format_spec: Optional[str]) -> bool:
        """Check if response matches expected format."""
        if format_spec == "json":
            try:
                import json
                json.loads(response)
                return True
            except json.JSONDecodeError:
                return False
        return True
    
    async def _llm_judge(self, query: str, response: str, reference: Optional[str]) -> dict:
        """Use LLM to judge response quality."""
        # Implementation: call judge model with structured evaluation prompt
        # Returns: {"correctness": 4.0, "helpfulness": 5.0, "safety": 5.0}
        pass
    
    def aggregate_results(self, results: list[EvaluationResult]) -> dict:
        """Compute aggregate metrics for reporting."""
        return {
            "format_compliance_rate": sum(r.format_valid for r in results) / len(results),
            "avg_correctness": sum(r.correctness for r in results) / len(results),
            "avg_helpfulness": sum(r.helpfulness for r in results) / len(results),
            "safety_violation_rate": sum(1 for r in results if r.safety < 3) / len(results),
            "p50_latency_ms": sorted(r.latency_ms for r in results)[len(results) // 2],
            "avg_tokens": sum(r.tokens_used for r in results) / len(results),
        }
```

---

## How It Works in Practice

### Production Evaluation Strategy

```yaml
Evaluation_Strategy:
  pre_deployment:
    what: "Run before any model/prompt change goes to production"
    suite:
      - "Golden test set (200+ curated examples with expected outputs)"
      - "Format compliance check (100% must pass)"
      - "Safety adversarial tests (0% safety violations)"
      - "LLM-as-judge scores (must be ≥ current production version)"
    gate: "All checks pass → proceed to canary deployment"
    
  canary_evaluation:
    what: "Continuous evaluation during canary deployment (5-10% traffic)"
    metrics:
      - "Real-time user feedback (thumbs up/down)"
      - "Error rate (parse failures, timeout, safety triggers)"
      - "LLM-judge sampling (score 5% of responses automatically)"
    duration: "24-72 hours"
    gate: "Metrics ≥ control group → proceed to full rollout"
    
  production_monitoring:
    what: "Ongoing evaluation of production responses"
    sample_rate: "Evaluate 1-5% of all responses via LLM-judge"
    alerts: "If avg_quality drops > 10% from baseline → alert + investigate"
    human_review: "Weekly: human reviews 50-100 low-scored responses"
    
  periodic_deep_eval:
    frequency: "Monthly"
    what: "Full evaluation suite including human evaluation"
    scope: "Golden test set + new test cases from production failures"
    output: "Quality report, trend analysis, improvement recommendations"
```

---

## Interview Tip

> When asked about LLM evaluation: "I use a multi-layered approach because no single metric captures LLM quality: (1) Automated checks — format compliance, length constraints, safety keyword filtering. These are binary gates (must pass 100%). (2) LLM-as-judge — Claude 4 Opus rates responses on correctness, helpfulness, and safety (1-5 scale). I use pairwise comparison for model selection (more reliable than absolute scoring), with position randomization to avoid bias. (3) Task-specific metrics — code: pass@k on test suites; summarization: factual consistency checked by NLI model; RAG: faithfulness (answer supported by context). (4) Human evaluation — gold standard for calibrating LLM judges and catching subtle quality issues (run monthly, 500 samples × 3 evaluators). Key operational patterns: golden test set (200+ curated examples) runs on every change as a gate, 1-5% of production responses evaluated by LLM-judge for continuous monitoring, and weekly human review of lowest-scored responses to identify systematic failure modes. Critical insight: public benchmarks (MMLU, HumanEval) don't predict YOUR application quality. Build domain-specific evaluation sets that reflect your actual use cases."

---

## Common Mistakes

1. **Relying solely on public benchmarks** — Model scores 90% on MMLU, so you deploy it. But MMLU doesn't test your domain (medical coding, legal research, customer service). Build a domain-specific evaluation set that tests the actual tasks your model will perform.

2. **LLM-as-judge without calibration** — Trusting judge scores without validating against human judgment. The judge might consistently rate verbose responses higher (verbosity bias) or miss factual errors in your domain. Always validate LLM-judge scores against human labels on a subset (100+ samples).

3. **Evaluating only average quality** — Average score is 4.2/5. Looks great! But 5% of responses score 1/5 (completely wrong or harmful). Those 5% cause user trust destruction and potential liability. Always monitor the tail: P5 score, worst-case examples, safety violation rate.

4. **Not versioning evaluation sets** — Your evaluation set is static from 6 months ago. New product features, new edge cases, and new failure modes aren't tested. Update evaluation sets quarterly: add cases from production failures, new features, and evolving requirements.

5. **Single-dimension evaluation** — Evaluating only "correctness" and deploying. But the model is correct in a rude, verbose, and formatting-broken way. Evaluate multiple dimensions: correctness + helpfulness + safety + format + style. A response must pass ALL dimensions, not just one.

---

## Key Takeaways

- LLM evaluation requires multiple dimensions: correctness, helpfulness, safety, format, domain-specific
- No single metric captures quality — use automated checks + LLM-as-judge + human evaluation
- LLM-as-judge: scalable (1000s of evaluations), 70-85% agreement with humans, has known biases
- Pairwise comparison is more reliable than pointwise scoring (relative judgment is easier)
- Human evaluation: gold standard but expensive ($3K per round) — use for calibration and edge cases
- Build domain-specific evaluation sets (public benchmarks don't predict YOUR quality)
- Golden test set (200+ examples): runs as automated gate on every change
- Monitor production: sample 1-5% of responses for continuous LLM-judge evaluation
- Track P5/P99, not just average — tail quality drives user trust and safety
- Version evaluation sets: update quarterly with new failure modes and requirements
