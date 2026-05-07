# AI Agent Fundamentals

## The Problem / Why This Matters

Traditional LLM applications are reactive — they respond to a single prompt and produce a single output. Ask a question, get an answer. But real-world tasks aren't single-step: "Research competitor pricing, update our pricing spreadsheet, draft an email to the team about the changes, and schedule a follow-up meeting." This requires multiple steps, decisions at each step, accessing external tools (web search, spreadsheets, email, calendar), and adapting when things don't go as planned. AI agents solve this by combining LLMs with reasoning, planning, tool use, and memory — creating autonomous systems that can accomplish complex multi-step goals. An agent isn't just an LLM; it's an LLM augmented with: (1) reasoning (breaking goals into steps), (2) actions (calling tools and APIs), (3) observation (processing results of actions), and (4) memory (maintaining context across steps). In 2026, agents are transforming software engineering (GitHub Copilot agent mode, Devin), customer service (autonomous resolution), data analysis (self-directed exploration), and enterprise workflows (end-to-end process automation). Understanding agent architecture is essential because agents are the primary way LLMs will be deployed for complex, real-world tasks — not as single-turn chatbots but as autonomous workers that accomplish goals.

---

## The Analogy

Think of the difference between a chatbot and an agent like the difference between a search engine and a personal assistant:

- **Chatbot (standard LLM)** = A search engine. You ask a question, it gives you information. One query, one response. If the answer requires multiple steps, YOU do the coordination.
- **AI agent** = A personal executive assistant. You say "Plan my trip to Tokyo." The assistant: researches flights → checks your calendar for conflicts → books optimal flights → finds hotels near your meetings → reserves the hotel → creates a detailed itinerary → sends confirmations to your email. Multiple steps, multiple tools, autonomous decision-making, handles surprises ("that flight is full, let me check alternatives").
- **Tools** = The assistant's resources (computer for booking, phone for calling, calendar app, email). Without tools, the assistant can only talk about planning — can't actually DO anything.
- **Memory** = The assistant remembering your preferences (aisle seat, 4-star hotels, no red-eyes). Without memory, they ask the same questions every time.
- **Reasoning** = The assistant deciding WHAT to do next (not you dictating every step). "Flight is full → check alternative → adjust hotel dates → notify you of change."

---

## Deep Dive

### Agent Architecture (ReAct Pattern)

```yaml
Agent_Core_Loop:
  what: "Reasoning + Acting: the agent thinks about what to do, does it, observes the result, and repeats"
  
  loop:
    step_1_observe: "Take in the current state (user goal, previous actions, tool results)"
    step_2_think: "Reason about what to do next (which tool? what input? is the goal achieved?)"
    step_3_act: "Execute the chosen action (call a tool, make an API request, generate output)"
    step_4_observe_result: "Process the result of the action"
    step_5_decide: "Is the goal achieved? If yes → return final answer. If no → back to step 1"
    
  example:
    user_goal: "What's the current stock price of NVIDIA and how has it changed this week?"
    
    iteration_1:
      thought: "I need to look up NVIDIA's current stock price. I'll use the stock_price tool."
      action: "call stock_price(symbol='NVDA')"
      observation: "NVDA: $185.42, last updated 2026-03-15 16:00 EST"
      
    iteration_2:
      thought: "I have the current price. Now I need the price from a week ago to calculate the change."
      action: "call stock_history(symbol='NVDA', period='1w')"
      observation: "NVDA 1-week: open $178.30, high $187.50, low $176.80, close $185.42"
      
    iteration_3:
      thought: "I now have both pieces of information. Current: $185.42, week ago open: $178.30. Change: +$7.12 (+4.0%). I can answer the user's question."
      action: "return_final_answer"
      observation: null
      
    final_answer: "NVIDIA (NVDA) is currently at $185.42. This week it's up $7.12 (+4.0%), opening at $178.30 on Monday."
```

### Agent Components

```yaml
Components:
  llm_brain:
    role: "The reasoning engine — decides what to do"
    requirements:
      - "Strong instruction following (understands tool descriptions)"
      - "Good reasoning (can plan multi-step workflows)"
      - "Reliable structured output (generates valid tool calls)"
    recommended_models:
      complex_agents: "Claude 4 Opus, GPT-5, o4 (reasoning model for hard planning)"
      simple_agents: "Claude 4 Sonnet, GPT-5-mini (sufficient for routine tasks)"
      self_hosted: "Llama-4-70B, Mistral Large (for data-sensitive workloads)"
      
  tools:
    what: "External capabilities the agent can invoke"
    types:
      information_retrieval:
        - "Web search (Google, Bing, Tavily)"
        - "Database queries (SQL, vector search)"
        - "File reading (read documents, code)"
        - "API calls (weather, stock, CRM data)"
        
      actions:
        - "Write files (code, documents, configs)"
        - "Send communications (email, Slack, notifications)"
        - "Execute code (Python, bash)"
        - "Modify data (update records, create tickets)"
        
      computation:
        - "Calculator (precise math)"
        - "Code interpreter (run Python for analysis)"
        - "Data transformation (parse, format, convert)"
        
    definition:
      format: |
        Each tool needs:
        - Name: unique identifier
        - Description: what it does (LLM reads this to decide when to use it)
        - Parameters: JSON schema of inputs
        - Returns: what the tool outputs
      example:
        name: "search_web"
        description: "Search the web for current information. Use when you need up-to-date facts, news, or information not in your training data."
        parameters:
          query: {type: "string", description: "The search query"}
          num_results: {type: "integer", default: 5}
        returns: "List of search results with title, URL, and snippet"
        
  memory:
    short_term:
      what: "Conversation context and recent actions (in the context window)"
      stores: "Current goal, action history, tool results"
      limitation: "Context window size (lost after window is full)"
      
    long_term:
      what: "Persistent knowledge across sessions"
      implementation: "Vector database, key-value store, or structured database"
      stores: "User preferences, past interactions, learned facts"
      
    working_memory:
      what: "Scratchpad for intermediate computations"
      implementation: "Structured state that persists across agent loop iterations"
      stores: "Partial results, hypotheses being tested, sub-goal progress"
      
  planning:
    what: "Breaking complex goals into sub-goals"
    approaches:
      no_planning:
        what: "Agent decides next action reactively (based on current state)"
        use: "Simple tasks with 1-3 steps"
        
      upfront_planning:
        what: "Agent creates a full plan before executing any steps"
        use: "Predictable multi-step tasks"
        risk: "Plan may be wrong (real-world is unpredictable)"
        
      plan_and_adapt:
        what: "Create initial plan, execute steps, revise plan based on results"
        use: "Complex tasks with uncertainty"
        benefit: "Robust to unexpected results"
```

### Function Calling (Tool Use)

```python
# How agents call tools (OpenAI function calling pattern)

import json
from typing import Any


# Define available tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code and return the output",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    },
]


class SimpleAgent:
    """Basic ReAct agent with tool use."""
    
    def __init__(self, model: str, tools: list[dict], system_prompt: str):
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = 10  # Safety limit
        
    async def run(self, user_goal: str) -> str:
        """Execute agent loop until goal is achieved or limit reached."""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_goal},
        ]
        
        for iteration in range(self.max_iterations):
            # Call LLM with tools available
            response = await self._call_llm(messages)
            
            # Check if agent wants to call a tool
            if response.tool_calls:
                # Execute each tool call
                for tool_call in response.tool_calls:
                    result = await self._execute_tool(tool_call)
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
            else:
                # No tool call — agent is returning final answer
                return response.content
        
        return "Agent reached maximum iterations without completing the goal."
    
    async def _execute_tool(self, tool_call) -> Any:
        """Execute a tool and return the result."""
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        # Route to appropriate tool implementation
        if name == "search_web":
            return await self._search_web(args["query"], args.get("num_results", 5))
        elif name == "execute_python":
            return await self._execute_python(args["code"])
        else:
            return {"error": f"Unknown tool: {name}"}
    
    async def _search_web(self, query: str, num_results: int) -> dict:
        """Web search implementation."""
        # Call search API (Tavily, SerpAPI, etc.)
        pass
    
    async def _execute_python(self, code: str) -> dict:
        """Sandboxed Python execution."""
        # Execute in sandbox (Docker container, serverless function)
        pass
```

### MCP (Model Context Protocol)

```yaml
MCP:
  full_name: "Model Context Protocol"
  creator: "Anthropic (open standard)"
  what: |
    A standard protocol for connecting LLMs to external tools and data sources.
    Like USB for AI — a universal connector between agents and capabilities.
    
  why_needed: |
    Before MCP: every tool integration was custom. 
    Each agent framework (LangChain, CrewAI) defined tools differently.
    Each tool needed separate integration for each framework.
    N frameworks × M tools = N×M integrations.
    
    With MCP: tools implement one standard protocol.
    Any MCP-compatible agent can use any MCP-compatible tool.
    N frameworks + M tools = N+M integrations.
    
  architecture:
    mcp_server:
      what: "A service that exposes tools via MCP protocol"
      provides:
        - "Tool definitions (name, description, parameters)"
        - "Tool execution endpoints"
        - "Resource access (files, databases)"
        - "Prompt templates"
      examples:
        - "File system MCP server (read/write files)"
        - "Database MCP server (query PostgreSQL, MySQL)"
        - "GitHub MCP server (issues, PRs, code search)"
        - "Slack MCP server (send messages, read channels)"
        
    mcp_client:
      what: "The agent/application that connects to MCP servers"
      does:
        - "Discovers available tools from servers"
        - "Presents tools to LLM for selection"
        - "Executes tool calls via protocol"
        - "Returns results to LLM"
      implementations: "Claude Desktop, VS Code Copilot, Cursor, custom agents"
      
  protocol:
    transport: "JSON-RPC over stdio (local) or HTTP/SSE (remote)"
    messages:
      initialize: "Client ↔ Server handshake (capabilities negotiation)"
      tools_list: "Client asks: what tools do you have?"
      tools_call: "Client says: execute this tool with these parameters"
      resources_list: "Client asks: what data sources are available?"
      resources_read: "Client says: give me this data"
```

---

## How It Works in Practice

### Agent Development Workflow

```yaml
Development_Workflow:
  step_1_define_scope:
    what: "Clearly define what the agent should and shouldn't do"
    outputs:
      - "Goal description (what tasks the agent handles)"
      - "Tool inventory (what capabilities it needs)"
      - "Safety boundaries (what it must NEVER do)"
      - "Success criteria (how you measure if it's working)"
      
  step_2_select_tools:
    what: "Choose and implement the tools the agent needs"
    principles:
      - "Minimum viable tools (start with fewest necessary)"
      - "Clear descriptions (LLM selects tools based on descriptions)"
      - "Deterministic where possible (tools should behave consistently)"
      - "Error handling (tools should return helpful errors, not crash)"
      
  step_3_design_system_prompt:
    what: "Instruct the agent on behavior, constraints, and strategy"
    includes:
      - "Role and personality"
      - "Available tools and when to use each"
      - "Constraints (never do X, always verify Y)"
      - "Strategy hints (check before modifying, ask before deleting)"
      
  step_4_test_iteratively:
    what: "Run the agent on test scenarios, identify failures, refine"
    approach:
      - "Start with simple tasks (1-2 tool calls)"
      - "Progress to complex tasks (5-10 steps)"
      - "Adversarial testing (edge cases, confusing inputs)"
      - "Safety testing (can it be tricked into unsafe actions?)"
      
  step_5_deploy_with_guardrails:
    what: "Deploy with safety controls and monitoring"
    controls:
      - "Human-in-the-loop for high-risk actions"
      - "Budget limits (max API calls per task)"
      - "Timeout limits (max execution time)"
      - "Action allowlists (can only call approved tools)"
```

---

## Interview Tip

> When asked about AI agents: "An agent is an LLM augmented with reasoning, tool use, and memory — enabling it to accomplish multi-step goals autonomously. The core loop is ReAct: Reason (decide what to do next) → Act (call a tool) → Observe (process the result) → repeat until goal is achieved. Key architectural components: (1) LLM brain — I use Claude 4 Opus or o4 for complex agents (strong reasoning and tool selection), Sonnet/GPT-5-mini for routine agents. (2) Tools — defined with clear descriptions so the LLM knows when to use each. I follow MCP (Model Context Protocol) for standardized tool integration — any MCP server works with any MCP client. (3) Memory — short-term (context window, current task state), long-term (vector store for past interactions, user preferences), working memory (scratchpad for intermediate results). (4) Safety — maximum iteration limits (prevent infinite loops), action allowlists (only approved tools), human-in-the-loop for destructive actions, spending limits. The critical engineering challenge isn't making agents work — it's making them reliable, safe, and observable. I instrument every step (trace which tools were called, what the LLM reasoned, what results came back) so I can debug failures. I also implement graceful degradation: if the agent is stuck after N iterations, it returns what it has so far rather than looping forever."

---

## Common Mistakes

1. **Too many tools** — Giving the agent 50 tools and expecting it to choose correctly. With more options, the LLM makes worse selections (paradox of choice). Solution: start with 5-10 focused tools per agent. If you need more, create specialized sub-agents with focused toolsets.

2. **No iteration limit** — Agent enters an infinite loop (keeps trying the same failed approach). Without limits, this burns through API budget and never completes. Solution: hard limit on iterations (10-20 typical), with graceful exit ("I was unable to complete the task. Here's what I accomplished so far: ...").

3. **Poor tool descriptions** — Tool named "process_data" with description "Processes data." The LLM has no idea when to use this vs other tools. Solution: descriptive names + detailed descriptions explaining WHEN to use the tool and what it returns. "Search the web for current real-time information when you need facts newer than your training data."

4. **No observability** — Agent produces wrong answer but you can't see WHY. Was it bad reasoning? Wrong tool selection? Tool returned bad data? Solution: log every step: the thought process, tool selection, tool arguments, tool results, and reasoning about results. Full trace for every agent run.

5. **Trusting agent actions blindly** — Agent deletes a production database because user said "clean up old data" and it interpreted this as DROP TABLE. Solution: human-in-the-loop for destructive actions, sandboxed execution environments, action confirmation for irreversible operations.

---

## Key Takeaways

- Agent = LLM + reasoning + tool use + memory (not just a chatbot)
- Core loop: Observe → Think → Act → Observe result → repeat until done
- Tools: defined with clear descriptions, minimum necessary set per agent
- MCP: universal protocol for tool integration (like USB for AI tools)
- Memory: short-term (context), long-term (vector store), working (scratchpad)
- Planning: reactive (simple tasks), upfront (predictable), plan-and-adapt (complex/uncertain)
- Safety: iteration limits, action allowlists, human-in-the-loop, spending limits
- Observability: trace every step (reasoning, tool calls, results) for debugging
- Model selection: complex agents → Opus/o4 (reasoning), routine agents → Sonnet/mini
- Start simple: 5-10 tools, 10 iteration limit, expand only when proven reliable
