# Agent Evaluation

## The Problem / Why This Matters

How do you know if your agent actually works? Unlike traditional software (deterministic: same input → same output), agents are non-deterministic, multi-step, and their quality is subjective. An agent might complete a task via 10 different valid paths. It might use 3 tool calls or 15. It might produce output that's correct but verbose, or concise but missing nuance. Traditional metrics (latency, error rate) don't capture agent quality. You need specialized evaluation: task completion rate (did it achieve the goal?), tool use efficiency (did it use the right tools, in the right order, with correct parameters?), reasoning quality (did it make good decisions at each step?), safety (did it stay within boundaries?), and cost efficiency (how many tokens/dollars per successful task?). Without proper evaluation, you can't: compare agent architectures, validate changes (did the new prompt improve things?), detect regressions, or build confidence for production deployment. Agent evaluation is harder than LLM evaluation because you're evaluating an entire workflow, not a single response.

---

## The Analogy

Think of agent evaluation like evaluating an employee:

- **Task completion** = Did they finish the assigned project? (Binary: yes/no, but also quality of the finished product)
- **Efficiency** = Did they finish in reasonable time with reasonable resources? (An agent that makes 50 API calls for a 3-call task is inefficient)
- **Decision quality** = Did they make good choices along the way? (Even if the end result is OK, were individual decisions sound?)
- **Safety compliance** = Did they follow company policies? (No unauthorized access, no data leaks, no harmful actions)
- **Growth/learning** = Are they getting better over time? (Comparing performance across versions)
- **The challenge** = You can't just look at the final output — you need to evaluate the entire process. A correct final answer achieved through unsafe means is a failure.

---

## Deep Dive

### Evaluation Dimensions

```yaml
Evaluation_Dimensions:
  task_completion:
    what: "Did the agent successfully achieve the stated goal?"
    metrics:
      success_rate: "% of tasks completed successfully"
      partial_completion: "% of sub-goals achieved (0-100%)"
      output_quality: "Quality score of final output (LLM-as-judge, human eval)"
    measurement:
      binary: "Pass/fail on verifiable tasks (code passes tests, answer matches ground truth)"
      graded: "Quality score (1-5) on open-ended tasks"
      
  tool_use_efficiency:
    what: "How well did the agent use available tools?"
    metrics:
      tool_selection_accuracy: "% of tool calls that were appropriate for the step"
      parameter_accuracy: "% of tool calls with correct parameters"
      unnecessary_calls: "Number of redundant or irrelevant tool calls"
      optimal_path_ratio: "Agent's tool calls / minimum necessary tool calls"
    ideal: "Agent uses minimum tools necessary, selects correctly, provides valid parameters"
    
  reasoning_quality:
    what: "Quality of decision-making at each step"
    metrics:
      plan_quality: "Is the plan logical and efficient? (for plan-and-execute agents)"
      adaptation: "Does agent adapt when results are unexpected?"
      error_recovery: "Does agent handle tool failures gracefully?"
      goal_alignment: "Do actions move toward the goal (vs tangential exploration)?"
    measurement: "LLM-as-judge on reasoning traces"
    
  safety:
    what: "Did the agent stay within defined boundaries?"
    metrics:
      boundary_violations: "Attempted actions outside allowed scope"
      data_leakage: "PII or sensitive data in outputs"
      destructive_actions: "Unauthorized modifications or deletions"
      injection_resistance: "Did agent resist prompt injection in tool results?"
    measurement: "Automated safety checks + adversarial testing"
    
  efficiency:
    what: "Resource usage per successful task"
    metrics:
      total_tokens: "Input + output tokens consumed"
      llm_calls: "Number of LLM invocations per task"
      tool_calls: "Number of tool calls per task"
      wall_time: "Total execution time"
      cost: "Total $ spent per task"
    benchmarks: "Compare against baseline (previous version, simpler approach)"
    
  reliability:
    what: "Consistency of performance across similar tasks"
    metrics:
      variance: "How much does performance vary for similar inputs?"
      failure_modes: "Distribution of failure types"
      degradation: "Performance under load, with noisy inputs, at edge cases"
```

### Evaluation Frameworks

```yaml
Evaluation_Approaches:
  golden_test_set:
    what: "Curated set of tasks with known correct outcomes"
    structure:
      - task: "Find the stock price of Apple and calculate the P/E ratio"
        expected_tools: ["get_stock_price", "get_financials", "calculate"]
        expected_answer_contains: ["P/E ratio", numeric_value]
        max_steps: 5
        max_cost: "$0.50"
    size: "50-200 test cases covering task categories and edge cases"
    use: "Run on every agent change (prompt, model, tool update)"
    
  trajectory_evaluation:
    what: "Evaluate the agent's step-by-step trajectory, not just final output"
    evaluates:
      - "Was each step necessary?"
      - "Were tools selected appropriately?"
      - "Were parameters correct?"
      - "Was reasoning sound?"
    implementation: "LLM-as-judge reviews the full trace"
    
  comparison_evaluation:
    what: "Compare agent against baselines"
    baselines:
      - "Previous version of the agent"
      - "Simpler architecture (ReAct vs plan-and-execute)"
      - "Human performance on same tasks"
      - "Different model (same agent, different LLM)"
    metrics: "Success rate, cost, latency, quality — side by side"
    
  adversarial_evaluation:
    what: "Test agent under challenging conditions"
    scenarios:
      ambiguous_tasks: "Unclear instructions (does agent ask for clarification?)"
      tool_failures: "Tools return errors (does agent recover gracefully?)"
      conflicting_information: "Tool results contradict each other"
      prompt_injection: "Malicious content in tool results"
      impossible_tasks: "Tasks that can't be completed (does agent recognize this?)"
      resource_limits: "Budget exhausted mid-task"
```

### Evaluation Implementation

```python
# Agent evaluation framework

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import json


@dataclass
class EvalCase:
    """A single evaluation test case."""
    id: str
    task: str
    category: str
    difficulty: str  # "easy", "medium", "hard"
    
    # Expected outcomes
    expected_result_contains: list[str] = field(default_factory=list)
    expected_tools_used: list[str] = field(default_factory=list)
    expected_tools_not_used: list[str] = field(default_factory=list)
    max_steps: int = 10
    max_cost_usd: float = 1.0
    
    # Ground truth (if available)
    ground_truth: Optional[str] = None


@dataclass
class EvalResult:
    """Result of evaluating an agent on a single test case."""
    case_id: str
    
    # Completion
    completed: bool
    final_output: str
    
    # Trajectory
    steps_taken: int
    tools_called: list[dict]  # [{name, args, result, success}]
    reasoning_trace: list[str]
    
    # Metrics
    total_tokens: int
    total_cost_usd: float
    wall_time_seconds: float
    
    # Quality scores
    output_quality: float = 0.0  # LLM-as-judge score (0-1)
    tool_efficiency: float = 0.0
    safety_score: float = 0.0
    
    # Issues
    errors: list[str] = field(default_factory=list)
    safety_violations: list[str] = field(default_factory=list)


class AgentEvaluator:
    """Comprehensive agent evaluation system."""
    
    def __init__(self, agent, judge_model: str = "claude-4-opus"):
        self.agent = agent
        self.judge_model = judge_model
        
    async def run_evaluation(self, test_cases: list[EvalCase]) -> dict:
        """Run full evaluation suite."""
        results = []
        
        for case in test_cases:
            result = await self._evaluate_single(case)
            results.append(result)
        
        return self._aggregate_results(results, test_cases)
    
    async def _evaluate_single(self, case: EvalCase) -> EvalResult:
        """Evaluate agent on a single test case."""
        
        # Run agent with instrumentation
        trace = await self.agent.run_instrumented(case.task)
        
        # Calculate metrics
        result = EvalResult(
            case_id=case.id,
            completed=trace.completed,
            final_output=trace.final_output,
            steps_taken=trace.step_count,
            tools_called=trace.tool_calls,
            reasoning_trace=trace.reasoning_steps,
            total_tokens=trace.total_tokens,
            total_cost_usd=trace.total_cost,
            wall_time_seconds=trace.wall_time,
        )
        
        # Score output quality (LLM-as-judge)
        result.output_quality = await self._judge_output_quality(
            task=case.task,
            output=trace.final_output,
            ground_truth=case.ground_truth,
        )
        
        # Score tool efficiency
        result.tool_efficiency = self._score_tool_efficiency(
            tools_called=trace.tool_calls,
            expected_tools=case.expected_tools_used,
            max_steps=case.max_steps,
        )
        
        # Check safety
        result.safety_score, result.safety_violations = await self._check_safety(trace)
        
        # Check constraints
        if trace.step_count > case.max_steps:
            result.errors.append(f"Exceeded max steps: {trace.step_count} > {case.max_steps}")
        if trace.total_cost > case.max_cost_usd:
            result.errors.append(f"Exceeded cost budget: ${trace.total_cost:.2f} > ${case.max_cost_usd}")
        
        return result
    
    async def _judge_output_quality(
        self, task: str, output: str, ground_truth: Optional[str]
    ) -> float:
        """Use LLM-as-judge to score output quality."""
        
        prompt = f"""Rate the quality of this agent's output (0.0 to 1.0):

Task: {task}
Agent Output: {output}
{"Expected Answer: " + ground_truth if ground_truth else ""}

Scoring criteria:
- 1.0: Perfectly correct, complete, well-formatted
- 0.8: Correct with minor imperfections
- 0.6: Mostly correct but missing some information
- 0.4: Partially correct with significant gaps
- 0.2: Mostly incorrect
- 0.0: Completely wrong or empty

Score (0.0-1.0):"""
        
        # Call judge model
        # Return float score
        pass
    
    def _score_tool_efficiency(
        self, tools_called: list, expected_tools: list, max_steps: int
    ) -> float:
        """Score how efficiently the agent used tools."""
        
        if not tools_called:
            return 0.0 if expected_tools else 1.0
        
        # Check if expected tools were used
        tools_used_names = [t["name"] for t in tools_called]
        expected_coverage = sum(1 for t in expected_tools if t in tools_used_names) / max(len(expected_tools), 1)
        
        # Penalize unnecessary tool calls
        efficiency_ratio = min(len(expected_tools), len(tools_called)) / max(len(tools_called), 1)
        
        # Penalize failed tool calls
        success_rate = sum(1 for t in tools_called if t["success"]) / len(tools_called)
        
        return (expected_coverage * 0.4 + efficiency_ratio * 0.3 + success_rate * 0.3)
    
    def _aggregate_results(self, results: list[EvalResult], cases: list[EvalCase]) -> dict:
        """Aggregate individual results into summary metrics."""
        
        completed = [r for r in results if r.completed]
        
        return {
            "total_cases": len(results),
            "completion_rate": len(completed) / len(results),
            "avg_output_quality": sum(r.output_quality for r in results) / len(results),
            "avg_tool_efficiency": sum(r.tool_efficiency for r in results) / len(results),
            "avg_safety_score": sum(r.safety_score for r in results) / len(results),
            "safety_violations": sum(len(r.safety_violations) for r in results),
            "avg_steps": sum(r.steps_taken for r in results) / len(results),
            "avg_cost": sum(r.total_cost_usd for r in results) / len(results),
            "avg_latency_s": sum(r.wall_time_seconds for r in results) / len(results),
            "by_difficulty": {
                diff: {
                    "completion_rate": len([r for r, c in zip(results, cases) if c.difficulty == diff and r.completed]) / max(len([c for c in cases if c.difficulty == diff]), 1),
                    "avg_quality": sum(r.output_quality for r, c in zip(results, cases) if c.difficulty == diff) / max(len([c for c in cases if c.difficulty == diff]), 1),
                }
                for diff in ["easy", "medium", "hard"]
            },
            "budget_violations": sum(1 for r in results if r.total_cost_usd > next(c.max_cost_usd for c in cases if c.id == r.case_id)),
        }
```

### Benchmarks for Agent Evaluation

```yaml
Benchmarks:
  swe_bench:
    what: "Real GitHub issues that agents must resolve"
    measures: "Software engineering capability (read code, diagnose, fix)"
    scoring: "% of issues resolved (patch passes tests)"
    leaders_2026: "Claude agent mode ~55%, Devin ~48%, Copilot ~42%"
    
  webarena:
    what: "Web-based tasks on real websites (shopping, forums, maps)"
    measures: "Web navigation and task completion"
    scoring: "Task completion rate"
    
  gaia:
    what: "General AI Assistants benchmark (multi-step reasoning + tools)"
    measures: "Real-world assistant capabilities"
    levels:
      level_1: "Simple (1-2 steps)"
      level_2: "Medium (3-5 steps, multiple tools)"
      level_3: "Hard (complex reasoning + many tools)"
      
  agent_bench:
    what: "8 environments for evaluating agents (OS, DB, web, code)"
    measures: "General agent competency across domains"
    
  custom_benchmarks:
    why: "Public benchmarks don't test YOUR specific agent's domain"
    how:
      - "Define 50-200 test cases from real user requests"
      - "Include: simple (30%), medium (40%), hard (20%), adversarial (10%)"
      - "Tag by category, difficulty, and required capabilities"
      - "Include known failure modes from production"
      - "Update quarterly with new edge cases"
```

---

## How It Works in Practice

### Production Evaluation Pipeline

```yaml
Evaluation_Pipeline:
  pre_deployment:
    trigger: "Every agent change (prompt, model, tool, code)"
    suite: "Full golden test set (200 cases)"
    gate: "Must pass: completion_rate >= 85%, safety_violations == 0, avg_quality >= 0.7"
    
  canary_evaluation:
    trigger: "During canary deployment (5% traffic)"
    metrics: "Real user satisfaction, task completion, error rate"
    comparison: "Against current production agent (A/B test)"
    duration: "24-72 hours"
    
  production_monitoring:
    ongoing: "Continuous evaluation of production agent"
    sample: "5-10% of completed tasks evaluated by LLM-as-judge"
    metrics:
      - "Task completion rate (weekly rolling average)"
      - "Average quality score (weekly)"
      - "Cost per task (trending up or down?)"
      - "Safety violations (any = investigate immediately)"
    alerts:
      - "Completion rate drops > 10% from baseline"
      - "Quality score drops > 0.1 from baseline"
      - "Any safety violation"
      - "Cost per task increases > 50%"
      
  regression_testing:
    trigger: "Nightly (automated)"
    suite: "Subset of golden test cases (50 most important)"
    purpose: "Catch regressions quickly (model updates, drift)"
```

---

## Interview Tip

> When asked about agent evaluation: "I evaluate agents across five dimensions: (1) Task completion — binary for verifiable tasks (code passes tests), graded (1-5) for open-ended tasks via LLM-as-judge. My golden test set has 200 cases across difficulty levels. (2) Tool use efficiency — did it pick the right tools, provide correct parameters, and avoid unnecessary calls? I compare against the optimal path (minimum tools needed). (3) Reasoning quality — LLM-as-judge evaluates the trajectory (not just output). Were decisions logical? Did it adapt to unexpected results? (4) Safety — automated checks for boundary violations, data leakage, destructive actions. Plus adversarial testing: prompt injection in tool results, impossible tasks, tool failures. (5) Efficiency — tokens, cost, latency per task. I track these over time to detect regression. Operationally: the golden test set runs as a gate on every agent change (completion_rate >= 85%, zero safety violations). In production, 5-10% of tasks get LLM-as-judge evaluation with alerts on quality drops. Critical insight: evaluating just the final output misses dangerous failures — I evaluate the full trajectory because a correct answer achieved unsafely is still a failure."

---

## Common Mistakes

1. **Evaluating only final output** — Agent produces correct answer, you mark it as success. But it accessed unauthorized data, made 50 unnecessary API calls, and leaked PII in intermediate steps. Solution: evaluate the full trajectory (tool calls, reasoning, intermediate outputs) not just the final answer.

2. **No adversarial testing** — Agent works great on happy-path test cases. First production user sends ambiguous input → agent enters infinite loop. Solution: include adversarial cases (ambiguous tasks, tool failures, prompt injection, impossible tasks) in your test suite. 10% of test cases should be adversarial.

3. **Static evaluation set** — Same 50 test cases from 6 months ago. New features, new failure modes, new attack vectors not tested. Solution: update evaluation set quarterly — add cases from production failures, new capabilities, and emerging threats.

4. **Evaluating with same model as agent** — Agent uses Claude 4 Sonnet, evaluator also uses Claude 4 Sonnet. Evaluator has same blind spots as agent (systematic bias). Solution: use a different (ideally stronger) model as judge. Or use human evaluation for calibration.

5. **No cost/efficiency tracking** — Agent gets better at task completion (85% → 92%) but cost went from $0.10 to $2.50 per task (25× increase). Is that acceptable? You don't know because you don't track it. Solution: always measure efficiency alongside quality. Report quality-per-dollar, not just quality.

---

## Key Takeaways

- Evaluate five dimensions: completion, tool efficiency, reasoning, safety, cost
- Golden test set (200 cases): runs as gate on every agent change
- Trajectory evaluation: judge the full path (tool calls, reasoning), not just final output
- LLM-as-judge for quality scoring, but validate against human labels periodically
- Adversarial testing: ambiguous tasks, tool failures, injection, impossible tasks (10% of test cases)
- Benchmarks: SWE-bench (code), WebArena (web tasks), GAIA (multi-step) — but build custom for YOUR domain
- Production monitoring: 5-10% of tasks evaluated continuously, alerts on quality drops
- Efficiency matters: track quality-per-dollar, not just quality alone
- Compare against baselines: previous version, simpler architecture, human performance
- Update evaluation set quarterly: add new failure modes, edge cases, capabilities
