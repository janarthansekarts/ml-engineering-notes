# Reasoning Models

## The Problem / Why This Matters

Standard LLMs generate responses token-by-token using pattern matching — fast but shallow. For complex tasks requiring multi-step reasoning (mathematical proofs, code architecture decisions, strategic planning, scientific analysis), pattern matching fails. Reasoning models — OpenAI's o3/o4, Anthropic's extended thinking, Google's Gemini 2.5 Thinking — introduce a fundamentally different paradigm: the model "thinks" before answering, breaking problems into steps, considering multiple approaches, verifying intermediate results, and self-correcting errors. This CoT (Chain-of-Thought) reasoning dramatically improves performance on hard problems (o3 scores 96.7% on AIME math vs GPT-4's 13%). But reasoning models come with engineering tradeoffs: they're 5-50× slower (thinking takes time), 3-10× more expensive (thinking tokens cost money), and require different prompting strategies (explicit reasoning inhibits their internal reasoning process). In 2026, reasoning models are essential for: complex code generation (architecture decisions, debugging), research synthesis (connecting disparate findings), strategic planning (multi-step decision making), and agentic workflows (deciding which tools to use and in what order). Operating reasoning models effectively means understanding when to use them (hard problems only), how to prompt them (differently from standard models), how to manage cost (thinking tokens add up), and how to orchestrate them in production systems.

---

## The Analogy

Think of the difference between standard and reasoning models like fast vs slow thinking (Daniel Kahneman's System 1 vs System 2):

- **Standard LLM (System 1)** = Instant intuition. "What's the capital of France?" → "Paris" (immediate, pattern matching). Fast, cheap, handles 90% of everyday questions. But for complex multi-step problems, intuition gives wrong answers.
- **Reasoning model (System 2)** = Deliberate analytical thinking. "Prove this theorem" → *Let me consider the premises... I'll try approach A... that leads to a contradiction... let me try approach B... this works because...* → Correct answer. Slow, expensive, but handles problems that intuition can't.
- **Thinking tokens** = Internal monologue. Like a math student showing their work (intermediate steps) vs just writing the final answer. The work is essential for correctness but takes time and resources.
- **When to use System 2** = Only for genuinely hard problems. Using deliberate reasoning for "What's 2+2?" is wasteful. Using intuition for "Prove the Riemann hypothesis" is ineffective. Routing is key.

---

## Deep Dive

### Reasoning Model Landscape (2026)

```yaml
Reasoning_Models:
  openai:
    o4:
      what: "Latest reasoning model, successor to o3"
      capabilities:
        - "PhD-level science and math reasoning"
        - "Complex code generation and debugging"
        - "Strategic planning and decision making"
        - "Self-verification (checks own reasoning)"
      pricing:
        input: "$15/million tokens"
        output: "$60/million tokens"
        thinking: "Thinking tokens billed at output rate"
      context: "200K tokens"
      latency: "10-60 seconds for complex problems (thinking time)"
      
    o4_mini:
      what: "Efficient reasoning model (cost-optimized)"
      capabilities: "Strong reasoning at fraction of o4 cost"
      pricing:
        input: "$3/million tokens"
        output: "$12/million tokens"
      use: "When reasoning is needed but o4 is too expensive"
      
  anthropic:
    extended_thinking:
      what: "Claude 4 with explicit chain-of-thought (thinking budget)"
      capabilities:
        - "User-visible reasoning trace"
        - "Configurable thinking budget (tokens allocated to reasoning)"
        - "Strong at code, math, analysis"
      features:
        thinking_budget: "Set max thinking tokens (1K-100K)"
        visible_thinking: "Can show reasoning to user (transparency)"
      pricing: "Thinking tokens at standard output rate"
      
  google:
    gemini_thinking:
      what: "Gemini 2.5 with thinking mode"
      capabilities:
        - "Multi-modal reasoning (think about images/video)"
        - "Long-context reasoning (reason over large documents)"
      context: "2M tokens with reasoning"
      
  open_source:
    qwen_qwq:
      what: "QwQ-32B reasoning model"
      capability: "Competitive reasoning for self-hosted deployment"
      
    deepseek_r1:
      what: "DeepSeek-R1 open reasoning model"
      capability: "Strong math/code reasoning, fully open weights"
      benefit: "Self-hostable, no API dependency"
```

### How Reasoning Models Work

```yaml
Chain_of_Thought:
  standard_model_behavior:
    input: "What is 287 × 453?"
    output: "130,011"  # May be wrong (pattern matching)
    process: "Direct token prediction (no intermediate steps)"
    
  reasoning_model_behavior:
    input: "What is 287 × 453?"
    thinking: |
      Let me break this down:
      287 × 453
      = 287 × 400 + 287 × 50 + 287 × 3
      = 114,800 + 14,350 + 861
      = 114,800 + 14,350 = 129,150
      = 129,150 + 861 = 130,011
      Let me verify: 130,011 / 287 = 453.17... 
      Hmm, let me recheck: 287 × 453
      Actually 287 × 3 = 861 ✓
      287 × 50 = 14,350 ✓  
      287 × 400 = 114,800 ✓
      114,800 + 14,350 + 861 = 130,011 ✓
    output: "130,011"  # Correct (verified through reasoning)
    
  key_properties:
    self_verification: "Model checks its own work (catches errors)"
    backtracking: "Can abandon wrong approaches and try alternatives"
    decomposition: "Breaks complex problems into manageable sub-problems"
    exploration: "Considers multiple solution paths before committing"
    
  thinking_tokens:
    what: "Tokens generated during internal reasoning (may or may not be shown to user)"
    cost: "Billed at output token rate (expensive)"
    volume: "Complex problems may use 5,000-50,000 thinking tokens"
    example: |
      Simple question: 50-200 thinking tokens
      Medium complexity: 500-2,000 thinking tokens
      Hard math/code problem: 5,000-30,000 thinking tokens
      Very complex research: 30,000-100,000 thinking tokens
```

### Prompting Reasoning Models

```yaml
Prompting_Differences:
  key_insight: |
    Reasoning models think INTERNALLY. Adding explicit chain-of-thought 
    instructions ("think step by step") can HURT performance because it 
    conflicts with the model's internal reasoning process.
    
  standard_model_prompting:
    good_practice: "Think step by step. First analyze X, then consider Y, finally conclude Z."
    why: "Explicit CoT guides the model to reason (it wouldn't otherwise)"
    
  reasoning_model_prompting:
    anti_pattern: "Think step by step. First analyze X, then..."
    why_bad: "Conflicts with model's trained reasoning approach. It already thinks step-by-step internally."
    
    good_practice:
      - "State the problem clearly and completely"
      - "Provide all necessary context and constraints"
      - "Specify what a correct answer looks like"
      - "Don't prescribe HOW to think — just WHAT to solve"
      
  examples:
    bad_prompt: |
      Think step by step about this problem.
      First, identify the key variables.
      Then, set up the equations.
      Then, solve the equations.
      Finally, verify your answer.
      
      Problem: A train leaves station A at 60 mph...
      
    good_prompt: |
      Problem: A train leaves station A at 60 mph heading east.
      Another train leaves station B (300 miles east of A) at 45 mph heading west.
      When and where do they meet?
      
      Provide the exact time and distance from station A.
      
  configuration:
    thinking_budget:
      what: "Control how many tokens the model can use for thinking"
      low_budget: "1,000-5,000 tokens — for simpler reasoning tasks"
      medium_budget: "10,000-30,000 tokens — for standard complex tasks"
      high_budget: "50,000-100,000 tokens — for very hard problems"
      tradeoff: "Higher budget = better answers but more cost and latency"
      
    reasoning_effort:
      what: "Some APIs allow setting reasoning effort level"
      levels:
        low: "Quick reasoning, less verification (faster, cheaper)"
        medium: "Standard reasoning depth (balanced)"
        high: "Deep reasoning with extensive verification (slowest, best quality)"
```

### Operating Reasoning Models in Production

```python
# Production reasoning model operations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import asyncio
import time


class ReasoningEffort(Enum):
    LOW = "low"       # Quick answer, minimal thinking
    MEDIUM = "medium" # Standard reasoning
    HIGH = "high"     # Deep reasoning, maximum quality


@dataclass
class ReasoningConfig:
    model: str
    effort: ReasoningEffort
    max_thinking_tokens: int
    timeout_seconds: int
    show_thinking: bool = False  # Whether to return thinking trace to user


@dataclass 
class ReasoningResult:
    answer: str
    thinking: Optional[str]     # Internal reasoning trace (if show_thinking=True)
    thinking_tokens: int
    output_tokens: int
    total_cost: float
    latency_seconds: float
    reasoning_steps: int        # Number of distinct reasoning steps


class ReasoningModelRouter:
    """Route queries to reasoning vs standard models based on complexity."""
    
    # Heuristics for when reasoning models are beneficial
    REASONING_INDICATORS = [
        "prove", "derive", "analyze the tradeoffs", "compare and contrast",
        "what are all the ways", "find the bug", "optimize this code",
        "design a system", "what's wrong with this approach",
        "solve this problem", "calculate", "determine",
    ]
    
    def should_use_reasoning(self, query: str, context: dict) -> bool:
        """Determine if a query benefits from a reasoning model."""
        
        # 1. Check explicit indicators
        query_lower = query.lower()
        has_indicator = any(ind in query_lower for ind in self.REASONING_INDICATORS)
        
        # 2. Check query complexity (length, structure)
        is_complex = len(query.split()) > 50 or "```" in query
        
        # 3. Check task type from context
        task_type = context.get("task_type", "general")
        reasoning_tasks = {"code_review", "architecture", "math", "debugging", "planning"}
        is_reasoning_task = task_type in reasoning_tasks
        
        # 4. Check if previous attempt with standard model failed
        previous_failed = context.get("standard_model_failed", False)
        
        return (has_indicator and is_complex) or is_reasoning_task or previous_failed
    
    def select_config(self, query: str, context: dict) -> ReasoningConfig:
        """Select appropriate reasoning configuration."""
        
        if not self.should_use_reasoning(query, context):
            # Use standard model (fast, cheap)
            return ReasoningConfig(
                model="claude-4-sonnet",
                effort=ReasoningEffort.LOW,
                max_thinking_tokens=0,
                timeout_seconds=30,
            )
        
        # Determine effort level
        task_type = context.get("task_type", "general")
        
        if task_type in {"math", "formal_proof"}:
            return ReasoningConfig(
                model="o4",
                effort=ReasoningEffort.HIGH,
                max_thinking_tokens=50000,
                timeout_seconds=120,
            )
        elif task_type in {"code_review", "debugging", "architecture"}:
            return ReasoningConfig(
                model="o4-mini",
                effort=ReasoningEffort.MEDIUM,
                max_thinking_tokens=20000,
                timeout_seconds=60,
            )
        else:
            return ReasoningConfig(
                model="claude-4-opus",  # Extended thinking
                effort=ReasoningEffort.MEDIUM,
                max_thinking_tokens=10000,
                timeout_seconds=45,
                show_thinking=True,  # Show reasoning to user
            )


class ReasoningModelService:
    """Service layer for reasoning model interactions."""
    
    def __init__(self):
        self.router = ReasoningModelRouter()
        
    async def solve(
        self, 
        query: str, 
        context: dict,
        config_override: Optional[ReasoningConfig] = None,
    ) -> ReasoningResult:
        """Solve a problem using appropriate reasoning model."""
        
        config = config_override or self.router.select_config(query, context)
        
        start_time = time.time()
        
        # Call reasoning model with timeout
        try:
            response = await asyncio.wait_for(
                self._call_reasoning_model(query, config),
                timeout=config.timeout_seconds
            )
        except asyncio.TimeoutError:
            # Reasoning exceeded time budget — return partial or fallback
            return await self._handle_timeout(query, config)
        
        latency = time.time() - start_time
        
        return ReasoningResult(
            answer=response.content,
            thinking=response.thinking if config.show_thinking else None,
            thinking_tokens=response.thinking_tokens,
            output_tokens=response.output_tokens,
            total_cost=self._calculate_cost(config.model, response),
            latency_seconds=latency,
            reasoning_steps=self._count_reasoning_steps(response.thinking),
        )
    
    async def _call_reasoning_model(self, query: str, config: ReasoningConfig):
        """Make API call to reasoning model."""
        # Implementation varies by provider
        pass
    
    async def _handle_timeout(self, query: str, config: ReasoningConfig) -> ReasoningResult:
        """Handle timeout — either retry with less thinking or return best effort."""
        # Option 1: Retry with lower effort
        # Option 2: Return partial result if available
        # Option 3: Fall back to standard model
        pass
    
    def _calculate_cost(self, model: str, response) -> float:
        """Calculate total cost including thinking tokens."""
        pricing = {
            "o4": {"input": 15, "output": 60},
            "o4-mini": {"input": 3, "output": 12},
            "claude-4-opus": {"input": 15, "output": 75},
        }
        rates = pricing.get(model, {"input": 10, "output": 30})
        
        # Thinking tokens billed at output rate
        thinking_cost = (response.thinking_tokens / 1_000_000) * rates["output"]
        output_cost = (response.output_tokens / 1_000_000) * rates["output"]
        input_cost = (response.input_tokens / 1_000_000) * rates["input"]
        
        return thinking_cost + output_cost + input_cost
```

### Structured Output from Reasoning Models

```yaml
Structured_Output:
  challenge: |
    Reasoning models think deeply but may produce verbose, unstructured output.
    For production systems, you need structured (JSON) output for downstream processing.
    
  approaches:
    think_then_format:
      technique: |
        Let the model reason freely in thinking phase,
        then request structured output in the final response.
      prompt: |
        Analyze this code for security vulnerabilities.
        After your analysis, provide your response as JSON:
        {"vulnerabilities": [{"severity": "high|medium|low", "description": "...", "location": "...", "fix": "..."}]}
        
    separate_reasoning_and_extraction:
      technique: |
        1. Call reasoning model for deep analysis (get free-form thinking)
        2. Call standard model to extract structured data from the analysis
      benefit: "Best reasoning quality + reliable structure"
      cost: "Two API calls"
      
    native_structured_output:
      technique: "Use provider's structured output feature (JSON mode, function calling)"
      openai: "response_format: {type: 'json_schema', schema: {...}}"
      anthropic: "Tool use with result schema"
      benefit: "Guaranteed valid JSON output"
      
  agentic_reasoning:
    what: "Reasoning models deciding which tools to use and in what order"
    pattern: |
      Reasoning model receives: task + available tools + constraints
      Thinking phase: "I need to first check the database, then call the API, 
                       then validate the result. Let me plan the sequence..."
      Actions: Tool calls in optimal order based on reasoning
    benefit: "Better tool selection and sequencing than standard models"
    use_cases:
      - "Complex research tasks (multi-source verification)"
      - "Code debugging (systematic hypothesis testing)"
      - "Data analysis (choosing which analyses to run)"
```

### Cost Management for Reasoning Models

```yaml
Cost_Management:
  the_problem: |
    Reasoning models are 5-50× more expensive per query due to thinking tokens.
    A complex problem might use 30,000 thinking tokens at output rate ($60/M) = $1.80 per query.
    At 1000 queries/day = $1,800/day = $657K/year.
    
  optimization_strategies:
    route_intelligently:
      what: "Only use reasoning for problems that need it"
      savings: "80-90% of queries don't need deep reasoning"
      implementation: "Complexity classifier routes simple→standard, complex→reasoning"
      
    effort_budgeting:
      what: "Set thinking token limits per query type"
      simple_reasoning: "2,000 thinking tokens max"
      medium_reasoning: "10,000 thinking tokens max"
      hard_problems: "50,000 thinking tokens max"
      
    cascade_approach:
      what: "Try standard model first, escalate to reasoning only if needed"
      flow: |
        1. Standard model answers (fast, cheap)
        2. Verify answer quality (automated check or confidence score)
        3. If quality insufficient → resend to reasoning model
      savings: "70% of queries resolved by standard model"
      
    cache_reasoning_results:
      what: "Cache answers to previously-reasoned questions"
      benefit: "Identical problem → instant answer (no re-reasoning)"
      hit_rate: "Lower than standard caching (reasoning queries are more unique)"
      
    batch_reasoning:
      what: "Accumulate problems, process in batch at 50% discount"
      use_for: "Code review, document analysis, non-real-time tasks"
```

---

## How It Works in Practice

### Production Reasoning Operations

```yaml
Operations:
  deployment_pattern:
    architecture: |
      User request → Complexity Router → 
        Simple: Standard model (90% of requests)
        Complex: Reasoning model (10% of requests)
    
  monitoring:
    metrics:
      - "Thinking tokens per request (cost driver)"
      - "Reasoning latency (seconds, not milliseconds)"
      - "Reasoning quality score (are longer-reasoned answers better?)"
      - "Timeout rate (queries exceeding thinking budget)"
      - "Routing accuracy (are we sending right queries to reasoning?)"
      
  user_experience:
    latency_management:
      problem: "Reasoning takes 10-60 seconds — users expect instant responses"
      solutions:
        - "Streaming: show 'Thinking...' indicator with progress"
        - "Show reasoning steps as they're generated (transparency)"
        - "Parallel: start standard model response while reasoning runs"
        - "Async: for long reasoning, notify when done (email, notification)"
        
    setting_expectations:
      ui_patterns:
        - "'Deep thinking about your problem...' with elapsed time"
        - "Show reasoning steps as expandable section"
        - "Offer 'Quick answer' vs 'Deep analysis' toggle"
```

---

## Interview Tip

> When asked about reasoning models: "I use reasoning models as precision tools — not for everything, but for the 10% of queries where standard models struggle: complex code architecture decisions, multi-step mathematical problems, debugging, and strategic planning. Key operational considerations: (1) Routing — complexity classifier determines which queries go to reasoning (saves 80% cost by not reasoning on simple queries). Indicators: multi-step problems, 'prove/analyze/compare' language, code review tasks, previous standard model failure. (2) Cost management — thinking tokens are expensive (billed at output rate, can be 5,000-50,000 tokens per query). I set thinking budgets per task type and cascade: try standard model first, escalate to reasoning only if quality check fails. (3) Prompting — counterintuitively, DON'T say 'think step by step' to reasoning models (it conflicts with their internal reasoning process). Instead: state the problem clearly, provide all constraints, specify what a correct answer looks like, let the model figure out HOW to reason. (4) Latency — reasoning takes 10-60 seconds. For UX: stream 'thinking' indicator, show reasoning steps progressively, offer quick vs deep toggle. (5) Production pattern — reasoning models are excellent for agentic workflows (deciding which tools to use and in what order). Their planning ability makes them ideal orchestrators for complex multi-step tasks."

---

## Common Mistakes

1. **Using reasoning models for simple queries** — Sending "What's the weather like?" to o4 (10 seconds, $0.50) when GPT-5-mini gives identical answer in 0.5 seconds for $0.001. Solution: route to reasoning models ONLY for genuinely complex problems. Build a complexity classifier as the front door.

2. **Adding "think step by step" to reasoning model prompts** — Standard CoT (Chain-of-Thought) prompting that works great for GPT-4 actually hurts o3/o4 performance. The model already thinks step-by-step internally — explicit instructions conflict with trained reasoning behavior. Solution: state the problem clearly, let the model reason on its own.

3. **No thinking budget (unlimited tokens)** — Letting the reasoning model think indefinitely on every problem. Some problems trigger 100K+ thinking tokens ($6+ per query) without proportional quality improvement. Solution: set max thinking token limits per task type. Monitor thinking-tokens-vs-quality correlation.

4. **Not handling timeout gracefully** — Reasoning model takes 90 seconds thinking about a complex problem, user gives up or connection drops. No fallback, no partial result. Solution: set timeout limits, implement graceful degradation (return partial reasoning, fall back to standard model, or async with notification).

5. **Expecting reasoning to fix bad context** — Giving a reasoning model incomplete information and expecting it to reason its way to the right answer. Reasoning models are better at processing complete information, not at guessing missing information. Solution: provide all necessary context, constraints, and criteria. Better input = better reasoning.

---

## Key Takeaways

- Reasoning models (o4, Claude extended thinking, Gemini thinking) add deliberate multi-step reasoning
- 5-50× slower and 3-10× more expensive than standard models — use only for complex problems
- Route intelligently: 90% of queries go to standard models, 10% to reasoning
- DON'T prompt with "think step by step" — reasoning models think internally, explicit CoT conflicts
- DO: state problem clearly, provide all context, specify success criteria, let model reason
- Thinking tokens: 5,000-50,000 per complex query, billed at output rate
- Cost management: thinking budgets, cascade (standard first → reasoning if needed), caching
- Latency UX: streaming "Thinking..." indicator, show reasoning steps, async for long tasks
- Best use cases: math/proofs, code architecture, debugging, multi-step planning, agentic tool selection
- Structured output: let model reason freely in thinking phase, then format final answer as JSON
