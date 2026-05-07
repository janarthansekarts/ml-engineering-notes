# Agent Architectures

## The Problem / Why This Matters

Not all agent tasks are the same. A simple "look up the weather" task needs a reactive agent (one tool call, done). But "research the competitive landscape, analyze pricing strategies, and produce a 20-page report with recommendations" needs planning, coordination, self-reflection, and potentially multiple specialized agents working together. Choosing the wrong architecture for a task leads to: over-engineering (complex architecture for simple tasks — slow and expensive), under-engineering (simple reactive loop for complex tasks — gets stuck, produces poor results), or fragility (architecture that can't handle unexpected results or failures). In 2026, several well-established agent architectures exist: ReAct (reactive reasoning), Plan-and-Execute (planning then execution), Reflexion (self-improvement through reflection), LATS (Language Agent Tree Search — exploring multiple paths), and multi-agent systems (specialized agents collaborating). Understanding these architectures lets you choose the right one for each use case, implement it correctly, and combine patterns for production-grade agentic systems. The architecture you choose determines your agent's reliability, cost, latency, and ability to handle complex tasks.

---

## The Analogy

Think of agent architectures like different project management styles:

- **ReAct** = Agile sprint. Do the next most important thing, see what happens, adapt. Great for tasks where you can't plan everything upfront (requirements change, discoveries happen). Fast iteration.
- **Plan-and-Execute** = Waterfall planning. Create a detailed project plan with milestones, then execute each step. Great when the task is well-understood and predictable. Risk: plan becomes obsolete if reality differs.
- **Reflexion** = Post-sprint retrospective. Do the work → review what went wrong → improve approach → try again. Critical for tasks where first attempt is unlikely to be perfect. Learning from failure.
- **LATS (Tree Search)** = Parallel prototyping. Try multiple approaches simultaneously, evaluate each, pursue the most promising. Expensive but finds optimal solutions.
- **Multi-Agent** = Specialized team. Researcher researches, writer writes, reviewer reviews. Each expert does what they're best at. Complex coordination but high-quality output.

---

## Deep Dive

### ReAct (Reasoning + Acting)

```yaml
ReAct:
  what: "The simplest and most common agent architecture"
  pattern: "Think → Act → Observe → Think → Act → Observe → ... → Final Answer"
  
  strengths:
    - "Simple to implement"
    - "Works well for 1-5 step tasks"
    - "Flexible (adapts to unexpected results)"
    - "Low overhead (no upfront planning cost)"
    
  weaknesses:
    - "Myopic (only thinks one step ahead)"
    - "Can loop (repeats same failed actions)"
    - "No global plan (may miss more efficient paths)"
    - "Quality degrades with longer chains (> 10 steps)"
    
  best_for:
    - "Simple tool-use tasks (lookup, calculation, API calls)"
    - "Interactive Q&A with tools"
    - "Tasks with 1-5 steps"
    
  implementation:
    system_prompt: |
      You have access to the following tools: {tool_descriptions}
      
      For each step:
      1. Think about what you need to do next
      2. Use a tool if needed
      3. Analyze the result
      4. Decide if you're done or need more steps
      
      When you have enough information to answer the user's question,
      provide your final answer directly (without calling a tool).
```

### Plan-and-Execute

```yaml
Plan_and_Execute:
  what: "Separate planning from execution: first create a plan, then execute steps"
  pattern: |
    1. Planner LLM creates a step-by-step plan
    2. Executor LLM executes each step (with tools)
    3. After each step, optionally re-plan (adapt to new information)
    
  architecture:
    planner:
      role: "Create and revise the plan"
      model: "Strong reasoning model (o4, Claude Opus) — plans better"
      output: "Ordered list of steps with descriptions and success criteria"
      
    executor:
      role: "Execute individual steps using tools"
      model: "Efficient model (Claude Sonnet, GPT-5-mini) — cheaper for execution"
      output: "Tool calls and results for the current step"
      
    replanner:
      role: "Revise plan based on execution results"
      trigger: "After each step, or on unexpected results"
      decision: "Continue plan as-is, modify remaining steps, or abandon and replan"
      
  strengths:
    - "Better for complex multi-step tasks (10-30 steps)"
    - "Global perspective (considers entire task before acting)"
    - "Can use different models for planning vs execution (cost optimization)"
    - "Easier to monitor progress (explicit plan shows what's happening)"
    
  weaknesses:
    - "Higher upfront cost (planning takes tokens)"
    - "Plan may be wrong (wasted effort if replanning is needed)"
    - "More complex implementation"
    - "Slower start (must plan before first action)"
    
  best_for:
    - "Complex research tasks"
    - "Multi-step code generation (design → implement → test → refactor)"
    - "Document creation (outline → draft sections → review → revise)"
    - "Data analysis workflows"

  example:
    user_goal: "Analyze our Q4 sales data and create a report with visualizations"
    
    plan:
      - step: "Load Q4 sales data from database"
        tools: ["query_database"]
        success: "DataFrame with Q4 sales records loaded"
      - step: "Clean and preprocess data (handle missing values, outliers)"
        tools: ["execute_python"]
        success: "Clean dataset ready for analysis"
      - step: "Compute key metrics (revenue, growth, top products, regional breakdown)"
        tools: ["execute_python"]
        success: "Metrics calculated and stored"
      - step: "Generate visualizations (bar charts, trend lines, heatmap)"
        tools: ["execute_python", "save_file"]
        success: "Charts saved as PNG files"
      - step: "Write report narrative combining metrics and insights"
        tools: ["execute_python", "save_file"]
        success: "Report.md created with analysis"
```

### Reflexion

```yaml
Reflexion:
  what: "Agent that reflects on its failures and improves on subsequent attempts"
  pattern: |
    1. Attempt the task
    2. Evaluate the result (did it succeed? what went wrong?)
    3. Generate reflection (lessons learned, what to do differently)
    4. Retry with reflection as additional context
    
  architecture:
    actor:
      role: "Attempts the task"
      input: "Task + previous reflections (if any)"
      
    evaluator:
      role: "Determines if the result is satisfactory"
      methods:
        - "Unit tests (for code generation)"
        - "LLM-as-judge (for open-ended tasks)"
        - "Automated checks (format, constraints)"
        
    reflector:
      role: "Analyzes failures and generates improvement insights"
      output: "What went wrong, what to try differently, key observations"
      
  strengths:
    - "Self-improving (gets better with each attempt)"
    - "Handles hard problems where first attempt often fails"
    - "Explicit learning from mistakes (reflection is reusable)"
    
  weaknesses:
    - "Expensive (multiple full attempts)"
    - "Slow (N attempts × cost per attempt)"
    - "May not converge (some problems don't get better with retries)"
    - "Needs good evaluation function"
    
  best_for:
    - "Code generation (try → test → fix → try again)"
    - "Creative writing (draft → critique → revise)"
    - "Problem solving with verifiable answers"
    
  example:
    task: "Write a function that solves the 8-queens problem"
    
    attempt_1:
      code: "[initial implementation - incorrect]"
      evaluation: "Fails test case: returns solutions with queens attacking diagonally"
      reflection: "I didn't check diagonal attacks properly. Need to verify both diagonals (row+col and row-col) are unique."
      
    attempt_2:
      code: "[improved implementation - uses reflection]"
      evaluation: "Passes all test cases"
      result: "Success after 2 attempts (reflection guided the fix)"
```

### LATS (Language Agent Tree Search)

```yaml
LATS:
  full_name: "Language Agent Tree Search"
  what: "Explores multiple solution paths simultaneously, selects the best one"
  pattern: |
    1. Generate multiple possible next actions (branching)
    2. Evaluate each branch (score or simulate forward)
    3. Select most promising branch
    4. Expand from selected branch
    5. Repeat until solution found
    
  architecture:
    similar_to: "Monte Carlo Tree Search (MCTS) but with LLM reasoning"
    
    expansion:
      what: "Generate N different possible next actions from current state"
      implementation: "Call LLM with temperature > 0, get N diverse continuations"
      
    evaluation:
      what: "Score each branch for likelihood of reaching the goal"
      methods:
        - "LLM-as-judge (ask model to rate each approach)"
        - "Heuristic scoring (based on progress indicators)"
        - "Simulation (execute forward and check result)"
        
    selection:
      what: "Choose which branch to expand next"
      strategy: "UCB1 (balances exploration vs exploitation) or greedy (best score)"
      
    backpropagation:
      what: "Update scores of parent nodes based on child results"
      
  strengths:
    - "Finds optimal solutions (explores multiple paths)"
    - "Recovers from bad decisions (can backtrack)"
    - "Produces higher quality results than greedy approaches"
    
  weaknesses:
    - "Very expensive (N branches × evaluation cost per branch)"
    - "Slow (many LLM calls for exploration)"
    - "Complex implementation"
    - "Diminishing returns for simple tasks"
    
  best_for:
    - "Hard problems with multiple valid approaches"
    - "Tasks where quality matters more than speed/cost"
    - "Code generation for complex algorithms"
    - "Mathematical proofs"
```

### Multi-Agent Systems

```yaml
Multi_Agent:
  what: "Multiple specialized agents collaborating on a task"
  
  patterns:
    supervisor:
      what: "One orchestrator agent delegates to specialist agents"
      flow: |
        User → Supervisor → routes to:
          → Research Agent (web search, data gathering)
          → Analysis Agent (computation, reasoning)
          → Writing Agent (content generation, formatting)
        Supervisor combines results → User
      benefit: "Clear control flow, easy to debug"
      
    peer_collaboration:
      what: "Agents communicate as equals, negotiating and building on each other's work"
      flow: |
        Agent A (Researcher): "Here's what I found about the market..."
        Agent B (Analyst): "Based on that data, here's my analysis..."
        Agent C (Writer): "I'll draft the report incorporating both..."
        Agent A: "Actually, I found a contradiction — let me verify..."
      benefit: "More thorough (agents challenge each other)"
      
    assembly_line:
      what: "Sequential pipeline — each agent's output is the next agent's input"
      flow: |
        Agent 1 (Draft) → Agent 2 (Review) → Agent 3 (Refine) → Agent 4 (Format)
      benefit: "Simple, predictable, easy to test each stage"
      
    debate:
      what: "Agents argue opposing positions, reach consensus"
      flow: |
        Agent Pro: "We should invest because..."
        Agent Con: "The risks are..."
        Judge Agent: "Based on both arguments, my recommendation is..."
      benefit: "Reduces bias, considers multiple perspectives"
      
  frameworks:
    crewai:
      what: "Framework for building multi-agent teams"
      concepts: "Agents (roles), Tasks (assignments), Crews (teams)"
      
    autogen:
      what: "Microsoft's multi-agent conversation framework"
      concepts: "Agents communicate via messages, code execution built-in"
      
    langgraph:
      what: "LangChain's graph-based agent orchestration"
      concepts: "Nodes (agents/functions), edges (transitions), state (shared memory)"
```

### Choosing the Right Architecture

```python
# Architecture selection logic

from enum import Enum
from dataclasses import dataclass


class Architecture(Enum):
    REACT = "react"
    PLAN_AND_EXECUTE = "plan_and_execute"
    REFLEXION = "reflexion"
    LATS = "lats"
    MULTI_AGENT = "multi_agent"


@dataclass
class TaskAnalysis:
    estimated_steps: int
    has_verifiable_output: bool
    quality_critical: bool
    time_sensitive: bool
    requires_multiple_skills: bool
    complexity: str  # "low", "medium", "high"


def select_architecture(task: TaskAnalysis) -> Architecture:
    """Select optimal agent architecture based on task characteristics."""
    
    # Simple tasks: use ReAct
    if task.estimated_steps <= 5 and task.complexity == "low":
        return Architecture.REACT
    
    # Complex multi-step with predictable structure: Plan-and-Execute
    if task.estimated_steps > 5 and not task.quality_critical:
        return Architecture.PLAN_AND_EXECUTE
    
    # Tasks with verifiable output where retry is cheap: Reflexion
    if task.has_verifiable_output and not task.time_sensitive:
        return Architecture.REFLEXION
    
    # Quality-critical with no time pressure: LATS
    if task.quality_critical and not task.time_sensitive:
        return Architecture.LATS
    
    # Requires diverse skills: Multi-Agent
    if task.requires_multiple_skills and task.estimated_steps > 5:
        return Architecture.MULTI_AGENT
    
    # Default: Plan-and-Execute (good general choice)
    return Architecture.PLAN_AND_EXECUTE
```

---

## How It Works in Practice

### Production Architecture Selection

```yaml
Production_Patterns:
  customer_service_agent:
    architecture: "ReAct (simple, fast, interactive)"
    reasoning: "Most queries need 1-3 tool calls. User is waiting."
    tools: ["lookup_order", "check_status", "initiate_refund", "create_ticket"]
    model: "Claude 4 Sonnet (fast, good enough)"
    
  code_generation_agent:
    architecture: "Reflexion (try → test → fix → retry)"
    reasoning: "Code has verifiable correctness (tests). First attempt often has bugs."
    tools: ["write_file", "run_tests", "read_file", "search_code"]
    model: "Claude 4 Opus for generation, Sonnet for reflection"
    
  research_agent:
    architecture: "Plan-and-Execute (structured research phases)"
    reasoning: "Research has clear phases: gather → analyze → synthesize. Plan helps."
    tools: ["web_search", "read_document", "execute_python", "write_report"]
    model: "o4 for planning, GPT-5-mini for execution steps"
    
  enterprise_workflow_agent:
    architecture: "Multi-Agent (diverse skills needed)"
    reasoning: "Involves data analysis + document generation + communication"
    agents:
      - "Data Agent (SQL, Python, analysis)"
      - "Document Agent (writing, formatting)"
      - "Communication Agent (email, Slack)"
    orchestrator: "Supervisor pattern with LangGraph"
```

---

## Interview Tip

> When asked about agent architectures: "I select architecture based on task characteristics: (1) ReAct — for simple 1-5 step interactive tasks (customer service, Q&A). Lowest overhead, fast iteration. (2) Plan-and-Execute — for complex multi-step tasks (research, data analysis, code projects). Planner creates the strategy (using a reasoning model), executor handles individual steps (cheaper model). Key benefit: I can monitor progress against the plan and detect when the agent is stuck. (3) Reflexion — when output is verifiable (code with tests, data with validation). Agent attempts → evaluates → reflects on failure → retries with lessons learned. Critical for code generation where first attempt has bugs. (4) Multi-agent — when the task requires diverse skills (research + analysis + writing). Each agent specializes with focused tools and prompts. I use supervisor pattern (one orchestrator routes to specialists) for control, or LangGraph for complex state machines. Architecture choice tradeoffs: ReAct (fast, simple, myopic), Plan-and-Execute (thorough, slower start, better for 10+ steps), Reflexion (higher quality, expensive — N attempts × cost), Multi-agent (highest quality for complex tasks, hardest to debug). In production, I often combine: Plan-and-Execute at the top level, with Reflexion for individual code-generation steps within the plan."

---

## Common Mistakes

1. **Using LATS/multi-agent for simple tasks** — Over-engineering a 2-step lookup task with tree search or 5 specialized agents. Result: 10× cost, 5× latency, same quality. Solution: start with ReAct. Only upgrade architecture when simple approaches demonstrably fail.

2. **No replanning in Plan-and-Execute** — Agent creates a plan, then stubbornly executes all steps even when step 3 revealed the plan is wrong. Solution: after each step, check if results match expectations. If not, trigger replanning (adapt remaining steps based on new information).

3. **Reflexion without a good evaluator** — Agent "reflects" but has no way to know if the new attempt is better. Without objective evaluation (tests, metrics, structured checks), reflection is just noise. Solution: invest in evaluation before implementing reflexion. Code → unit tests. Content → LLM-as-judge with rubric.

4. **Multi-agent without clear boundaries** — Agents with overlapping responsibilities, unclear handoff points, no shared state management. They duplicate work, contradict each other, or drop information between handoffs. Solution: clear role definitions, explicit input/output contracts between agents, shared state store for coordination.

5. **Ignoring cost of complex architectures** — LATS exploring 5 branches with 10 evaluations each = 50 LLM calls per decision point. For a 10-step task, that's 500 LLM calls at $0.05 each = $25 per task. Solution: profile cost vs quality improvement. Is LATS's 5% quality improvement worth 10× the cost?

---

## Key Takeaways

- ReAct: simplest architecture, Think→Act→Observe loop, best for 1-5 step tasks
- Plan-and-Execute: separate planning from execution, best for 10+ step complex tasks
- Reflexion: try→evaluate→reflect→retry, best when output is verifiable (code, structured data)
- LATS: explore multiple paths, best when quality is critical and cost isn't a concern
- Multi-agent: specialized agents collaborate, best when task requires diverse skills
- Start simple (ReAct), upgrade only when simpler approaches demonstrably fail
- Combine architectures: Plan-and-Execute with Reflexion for individual steps is powerful
- Plan-and-Execute can use different models: expensive model for planning, cheap for execution
- Multi-agent needs clear boundaries: role definitions, input/output contracts, shared state
- Architecture choice determines cost, latency, quality — optimize for your constraints
