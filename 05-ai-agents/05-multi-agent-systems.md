# Multi-Agent Systems

## The Problem / Why This Matters

Single agents hit a ceiling of complexity. When a task requires deep expertise across multiple domains (research + data analysis + code generation + report writing), a single agent with 50 tools and a massive system prompt becomes unreliable — tool selection accuracy drops, context gets diluted, and the agent's "identity" becomes confused (is it a researcher? a coder? a writer?). Multi-agent systems solve this by decomposing complex tasks across specialized agents that collaborate. Each agent has a focused role, limited tools, and clear expertise — like a team of specialists rather than one overloaded generalist. In 2026, multi-agent systems power: autonomous software engineering teams (one agent architects, another implements, another tests), customer service escalation chains (L1 agent → L2 specialist → human), research synthesis workflows (multiple researchers + one synthesizer), and enterprise automation (data agent + communication agent + approval agent). The engineering challenge is coordination: how do agents communicate, share state, avoid conflicts, handle disagreements, and produce coherent output? Multi-agent systems are more capable but also more complex — the coordination overhead must be justified by the quality improvement.

---

## The Analogy

Think of multi-agent systems like a hospital emergency department:

- **Single agent** = One doctor doing everything: triage, diagnosis, surgery, pharmacy, paperwork. Possible for simple cases but impossible for complex trauma. Quality suffers, burnout is inevitable.
- **Multi-agent team** = The ER team: triage nurse (routes patients), ER doctor (diagnosis), surgeon (complex procedures), radiologist (imaging interpretation), pharmacist (medications), administrator (paperwork). Each specialist is excellent at their focused role.
- **Supervisor pattern** = The attending physician coordinating the team: decides who handles what, resolves conflicts, ensures coherent patient care.
- **Communication** = The patient's medical record: shared state that all specialists can read and write. Everyone stays coordinated without having to be in the same room.
- **Handoff** = "Surgery is done, transferring to post-op care" — clear transition between agents with context preservation.

---

## Deep Dive

### Multi-Agent Patterns

```yaml
Patterns:
  supervisor:
    what: "One orchestrator agent manages specialist agents"
    flow: |
      User → Supervisor Agent (decides routing)
        → Routes to Agent A (research)
        → Routes to Agent B (analysis)  
        → Routes to Agent C (writing)
      Supervisor combines results → User
    supervisor_role:
      - "Decompose task into sub-tasks"
      - "Assign sub-tasks to appropriate agents"
      - "Monitor progress and quality"
      - "Handle failures (reassign, retry)"
      - "Combine outputs into coherent result"
    best_for: "Clear hierarchical workflows, predictable task decomposition"
    
  hierarchical:
    what: "Multi-level supervision (team leads → specialists)"
    flow: |
      User → Executive Agent
        → Research Lead → [Web Researcher, DB Researcher, Doc Researcher]
        → Engineering Lead → [Architect, Implementer, Tester]
        → Comms Lead → [Writer, Editor, Designer]
    best_for: "Large complex projects with many agents"
    
  collaborative:
    what: "Agents work as peers, communicating directly"
    flow: |
      Agent A: "I found these market trends..."
      Agent B: "Based on those trends, here's my analysis..."
      Agent C: "I disagree with conclusion X because..."
      Agent A: "Let me verify with additional data..."
      [Continue until consensus or moderator intervenes]
    best_for: "Creative tasks, research synthesis, strategy development"
    
  pipeline:
    what: "Sequential chain — each agent's output feeds the next"
    flow: |
      Agent 1 (Research) → Agent 2 (Analysis) → Agent 3 (Draft) → Agent 4 (Review)
    best_for: "Well-defined sequential workflows"
    advantage: "Simple, predictable, easy to test each stage"
    limitation: "Can't parallelize, single point of failure per stage"
    
  debate:
    what: "Agents argue opposing positions, judge decides"
    flow: |
      Agent Pro: argues FOR the proposal
      Agent Con: argues AGAINST the proposal
      Judge Agent: weighs both sides, makes recommendation
    best_for: "Decision making, bias reduction, thorough analysis"
    advantage: "Reduces confirmation bias, explores counterarguments"
```

### Multi-Agent Frameworks (2026)

```yaml
Frameworks:
  crewai:
    what: "Framework for orchestrating teams of AI agents"
    concepts:
      agents: "Role-based agents with specific expertise and tools"
      tasks: "Defined work items with expected outputs"
      crews: "Teams of agents working together"
      processes: "Sequential or hierarchical workflow"
    features:
      - "Role-based agent definition"
      - "Task delegation and dependency management"
      - "Built-in collaboration patterns"
      - "Memory sharing between agents"
      - "Human-in-the-loop integration"
    best_for: "Business workflows, content creation, research"
    
  autogen:
    what: "Microsoft's multi-agent conversation framework"
    concepts:
      agents: "Conversable agents that communicate via messages"
      group_chat: "Multiple agents in a shared conversation"
      code_execution: "Built-in code execution capability"
    features:
      - "Flexible conversation patterns"
      - "Code execution in sandboxed environments"
      - "Human proxy agent (human as agent in the loop)"
      - "Nested agent conversations"
    best_for: "Code generation, data analysis, complex reasoning"
    
  langgraph:
    what: "LangChain's graph-based orchestration framework"
    concepts:
      nodes: "Functions or agents (processing steps)"
      edges: "Transitions between nodes (routing logic)"
      state: "Shared state object passed through the graph"
      checkpointing: "Save and resume execution state"
    features:
      - "Cyclic graphs (loops, retries, human approval)"
      - "Conditional routing (dynamic path selection)"
      - "Persistent state (survive across sessions)"
      - "Streaming (real-time output as agents work)"
      - "Sub-graphs (composable agent teams)"
    best_for: "Complex stateful workflows, production deployments"
    
  openai_swarm:
    what: "Lightweight multi-agent handoff framework"
    concepts:
      agents: "Minimal agents with instructions and tools"
      handoff: "Agent transfers conversation to another agent"
    features:
      - "Extremely simple API"
      - "Agent-to-agent handoff"
      - "Lightweight (no complex orchestration)"
    best_for: "Simple routing/handoff scenarios"
```

### Implementation with LangGraph

```python
# Multi-agent system with LangGraph

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Literal
from operator import add


class TeamState(TypedDict):
    """Shared state between all agents."""
    task: str
    research_results: list[str]
    analysis: str
    draft: str
    review_feedback: str
    final_output: str
    messages: Annotated[list, add]  # Append-only message history
    current_agent: str
    iteration: int


class ResearchAgent:
    """Agent specialized in information gathering."""
    
    def __init__(self, model, tools):
        self.model = model  # Fast model (GPT-5-mini)
        self.tools = tools  # [web_search, read_document, query_db]
        
    async def run(self, state: TeamState) -> TeamState:
        """Conduct research based on the task."""
        # Use tools to gather information
        results = await self._research(state["task"])
        
        return {
            **state,
            "research_results": results,
            "current_agent": "research",
            "messages": [{"agent": "researcher", "content": f"Found {len(results)} relevant sources"}],
        }


class AnalysisAgent:
    """Agent specialized in data analysis and reasoning."""
    
    def __init__(self, model, tools):
        self.model = model  # Reasoning model (o4-mini)
        self.tools = tools  # [execute_python, calculate]
        
    async def run(self, state: TeamState) -> TeamState:
        """Analyze research results."""
        analysis = await self._analyze(state["research_results"], state["task"])
        
        return {
            **state,
            "analysis": analysis,
            "current_agent": "analyst",
            "messages": [{"agent": "analyst", "content": "Analysis complete"}],
        }


class WritingAgent:
    """Agent specialized in content creation."""
    
    def __init__(self, model):
        self.model = model  # Creative model (Claude 4 Sonnet)
        
    async def run(self, state: TeamState) -> TeamState:
        """Draft output based on analysis."""
        draft = await self._write(state["task"], state["analysis"], state["research_results"])
        
        return {
            **state,
            "draft": draft,
            "current_agent": "writer",
            "messages": [{"agent": "writer", "content": "Draft complete"}],
        }


class ReviewAgent:
    """Agent that reviews and provides feedback."""
    
    def __init__(self, model):
        self.model = model  # Critical model (Claude 4 Opus)
        
    async def run(self, state: TeamState) -> TeamState:
        """Review draft and provide feedback."""
        feedback = await self._review(state["draft"], state["task"])
        
        return {
            **state,
            "review_feedback": feedback,
            "current_agent": "reviewer",
            "messages": [{"agent": "reviewer", "content": f"Feedback: {feedback[:100]}..."}],
            "iteration": state.get("iteration", 0) + 1,
        }


def should_revise(state: TeamState) -> Literal["revise", "finalize"]:
    """Decide whether to revise or finalize based on feedback."""
    if "APPROVED" in state["review_feedback"]:
        return "finalize"
    if state.get("iteration", 0) >= 3:
        return "finalize"  # Max iterations reached
    return "revise"


def build_research_team() -> StateGraph:
    """Build a multi-agent research team graph."""
    
    # Initialize agents
    researcher = ResearchAgent(model="gpt-5-mini", tools=["web_search", "read_doc"])
    analyst = AnalysisAgent(model="o4-mini", tools=["execute_python"])
    writer = WritingAgent(model="claude-4-sonnet")
    reviewer = ReviewAgent(model="claude-4-opus")
    
    # Build graph
    graph = StateGraph(TeamState)
    
    # Add nodes (agents)
    graph.add_node("research", researcher.run)
    graph.add_node("analyze", analyst.run)
    graph.add_node("write", writer.run)
    graph.add_node("review", reviewer.run)
    graph.add_node("finalize", lambda s: {**s, "final_output": s["draft"]})
    
    # Add edges (workflow)
    graph.set_entry_point("research")
    graph.add_edge("research", "analyze")
    graph.add_edge("analyze", "write")
    graph.add_edge("write", "review")
    
    # Conditional: revise or finalize based on review
    graph.add_conditional_edges(
        "review",
        should_revise,
        {"revise": "write", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)
    
    return graph.compile(checkpointer=MemorySaver())
```

### Communication Patterns

```yaml
Communication:
  message_passing:
    what: "Agents communicate by sending structured messages"
    format:
      sender: "Agent ID"
      recipient: "Agent ID or 'broadcast'"
      type: "request | response | information | feedback"
      content: "Message content"
      metadata: "Priority, timestamp, thread_id"
    advantage: "Clear, auditable, async-friendly"
    
  shared_state:
    what: "All agents read/write a shared state object"
    format: "Typed dictionary (TypedDict) with fields per agent's domain"
    advantage: "Simple, no explicit message handling"
    risk: "Race conditions if agents run in parallel"
    mitigation: "Agent-specific fields (researcher writes research_results, writer writes draft)"
    
  blackboard:
    what: "Shared workspace where agents post findings and read others' posts"
    how: "Like a shared document — agents add sections, reference others' work"
    advantage: "Flexible, good for creative collaboration"
    
  structured_handoff:
    what: "Explicit handoff with context transfer"
    format:
      from_agent: "Research Agent"
      to_agent: "Analysis Agent"
      handoff_content:
        summary: "What was accomplished"
        artifacts: ["research_results.json", "sources.md"]
        next_steps: "Suggested focus areas for analysis"
        constraints: "Time limit, scope boundaries"
    advantage: "Minimal information loss between agents"
```

### Coordination Challenges

```yaml
Challenges:
  conflict_resolution:
    problem: "Two agents produce contradictory outputs"
    solutions:
      voting: "Multiple agents vote, majority wins"
      authority: "Designated expert agent's opinion prevails for their domain"
      judge: "Separate judge agent evaluates conflicting outputs"
      human: "Escalate to human for resolution"
      
  coordination_overhead:
    problem: "More agents = more communication = more cost and latency"
    measurement: "Useful work tokens / total tokens spent (include coordination)"
    acceptable: "> 60% useful work (< 40% coordination overhead)"
    too_high: "If 80% of tokens are agents talking to each other, simplify"
    
  state_consistency:
    problem: "Agent B reads state before Agent A finishes writing"
    solutions:
      sequential: "Agents execute one at a time (simplest)"
      barriers: "Sync points where all agents must complete before next phase"
      locks: "Agent acquires lock on state fields before writing"
      
  infinite_loops:
    problem: "Reviewer keeps rejecting, writer keeps revising, forever"
    solutions:
      max_iterations: "Hard limit on revision cycles (3 max)"
      escalation: "After N failures, escalate to human or accept best attempt"
      diminishing_returns: "Track quality score per iteration, stop if not improving"
      
  cost_explosion:
    problem: "5 agents × 10 iterations × 5000 tokens each = 250K tokens per task"
    solutions:
      budget_per_task: "Set max total tokens per multi-agent workflow"
      efficient_agents: "Use cheapest model per role (research=mini, review=opus)"
      parallel_where_possible: "Independent research agents run simultaneously"
```

---

## How It Works in Practice

### Production Multi-Agent Deployment

```yaml
Production:
  example_enterprise_workflow:
    task: "Monthly competitive intelligence report"
    team:
      research_agent:
        model: "GPT-5-mini"
        tools: ["web_search", "news_api", "sec_filings"]
        role: "Gather competitor data from multiple sources"
        
      data_agent:
        model: "Claude 4 Sonnet"
        tools: ["execute_python", "query_database"]
        role: "Analyze pricing, market share, feature comparisons"
        
      writing_agent:
        model: "Claude 4 Opus"
        tools: ["create_document", "generate_chart"]
        role: "Produce executive-ready report with visualizations"
        
      review_agent:
        model: "Claude 4 Opus"
        tools: ["fact_check", "style_check"]
        role: "Verify claims, check formatting, ensure completeness"
        
    orchestration: "LangGraph with checkpointing (resume on failure)"
    execution_time: "15-30 minutes"
    cost: "$5-15 per report (depending on research depth)"
    
  monitoring:
    per_agent:
      - "Task completion rate"
      - "Average tokens used"
      - "Average execution time"
      - "Error rate"
    system_level:
      - "End-to-end completion rate"
      - "Total cost per workflow"
      - "Coordination overhead ratio"
      - "Human escalation rate"
```

---

## Interview Tip

> When asked about multi-agent systems: "I use multi-agent systems when a task requires diverse expertise that doesn't fit in one agent's context (research + code + analysis + writing). My preferred architecture is supervisor pattern with LangGraph: (1) Supervisor agent — decomposes the task, routes to specialists, monitors quality, combines results. Uses a strong reasoning model (Claude Opus) for planning. (2) Specialist agents — each has focused tools and a narrow role. Research agent (web search, documents), code agent (execute Python, write files), writing agent (drafting, formatting). Each uses the cheapest model that handles their specialty well. (3) Shared state — TypedDict flowing through the graph. Each agent writes to their domain fields, reads from others. Checkpointing enables resume-on-failure. (4) Quality loop — review agent evaluates output, either approves or requests revision (max 3 iterations). Key considerations: coordination overhead (if >40% of tokens are agents coordinating vs doing useful work, the system is over-engineered — simplify), cost management (budget per workflow, cheapest model per role), and failure handling (LangGraph checkpoints let me resume from last successful step, not restart from scratch). I DON'T use multi-agent for tasks a single agent handles well — the overhead isn't justified unless task complexity truly demands specialization."

---

## Common Mistakes

1. **Multi-agent for simple tasks** — Building a 5-agent team to answer a FAQ question. Coordination overhead exceeds the value of specialization. A single agent with 5 tools handles this in 2 seconds for $0.01. Solution: multi-agent only when task genuinely requires diverse expertise AND exceeds single-agent capability.

2. **No coordination budget** — Agents chatting endlessly among themselves (passing messages back and forth without making progress). 80% of tokens spent on coordination, 20% on actual work. Solution: monitor coordination ratio, set max messages between agents, time-box each phase.

3. **Unclear agent boundaries** — Two agents have overlapping responsibilities ("who handles data analysis — the researcher or the analyst?"). They duplicate work or drop tasks that both assume the other handles. Solution: explicit, non-overlapping role definitions. Each capability assigned to exactly one agent.

4. **No failure isolation** — One agent fails (API error, bad output) and the entire multi-agent workflow crashes. All previous work is lost. Solution: checkpointing (LangGraph persistence), retry individual agents, graceful degradation (skip failed agent's contribution if non-critical).

5. **Sequential when parallel is possible** — Three research agents that could run simultaneously execute one after another (3× latency). Solution: identify independent sub-tasks and run them in parallel. Only serialize when there are actual dependencies (Agent B needs Agent A's output).

---

## Key Takeaways

- Multi-agent = specialized agents collaborating on tasks too complex for one agent
- Patterns: supervisor (hierarchical), collaborative (peer), pipeline (sequential), debate (adversarial)
- Frameworks: LangGraph (stateful graphs, production-grade), CrewAI (role-based teams), AutoGen (conversations)
- Supervisor pattern: orchestrator routes to specialists, monitors quality, combines results
- Shared state: TypedDict with per-agent fields, checkpointed for resume-on-failure
- Communication: message passing (explicit) or shared state (implicit)
- Coordination overhead: monitor ratio of useful work vs coordination tokens (target >60% useful)
- Cost management: cheapest model per agent role, budget per workflow, parallel where possible
- Failure handling: checkpoint + retry individual agents, don't restart entire workflow
- Only use multi-agent when single agent demonstrably can't handle the complexity
