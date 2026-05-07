# Agent Frameworks

## The Problem / Why This Matters

Building agents from scratch — implementing the reasoning loop, tool calling, state management, memory, error handling, streaming, checkpointing, multi-agent coordination — is thousands of lines of infrastructure code before you write any business logic. Agent frameworks solve this by providing battle-tested abstractions for common agent patterns. But the framework landscape in 2026 is fragmented and fast-moving: LangGraph dominates production workflows, CrewAI leads for multi-agent teams, OpenAI Assistants offers the simplest API, Anthropic's tool use is the most reliable, and Vercel AI SDK targets full-stack JavaScript applications. Choosing the wrong framework means: hitting limitations that require migration (expensive), coupling to one provider's ecosystem (vendor lock-in), or using a framework that's abandoned within a year (community matters). Understanding the strengths, limitations, and ideal use cases of each framework lets you make an informed choice that serves your project for years, not weeks.

---

## The Analogy

Think of agent frameworks like web frameworks:

- **LangGraph** = Django/Rails. Full-featured, production-ready, opinionated about architecture (graph-based). Handles complex state management, persistence, and deployment. The "enterprise" choice.
- **CrewAI** = Team collaboration tools (like Asana/Notion for AI). Designed specifically for multi-agent teams. Simple role-based agent definition, task assignment, team coordination.
- **OpenAI Assistants** = Firebase/Supabase. Managed backend — OpenAI handles state, file storage, code execution. Fastest to prototype but most vendor-locked.
- **Vercel AI SDK** = Next.js for AI. Full-stack JavaScript/TypeScript, streaming-first, React integration, edge-ready. The frontend-developer's choice.
- **Raw function calling** = Express.js/Flask. Minimal abstraction, maximum control. Build exactly what you need, nothing more. For teams that want full control over every decision.

---

## Deep Dive

### LangGraph

```yaml
LangGraph:
  creator: "LangChain"
  philosophy: "Agents as graphs: nodes (steps), edges (transitions), state (shared context)"
  
  key_concepts:
    state:
      what: "Typed dictionary passed through the graph"
      features:
        - "Type-safe (TypedDict)"
        - "Reducer functions (how state updates merge)"
        - "Checkpointed (persisted, resumable)"
        
    nodes:
      what: "Functions or agent calls that transform state"
      types:
        - "LLM calls (agent reasoning)"
        - "Tool execution"
        - "Human input (wait for user)"
        - "Sub-graphs (nested workflows)"
        
    edges:
      what: "Transitions between nodes"
      types:
        - "Static: always go from A to B"
        - "Conditional: go to B or C based on state"
        
    checkpointing:
      what: "Persist graph execution state"
      enables:
        - "Resume after failure (restart from last checkpoint)"
        - "Human-in-the-loop (pause, wait for input, resume)"
        - "Time-travel debugging (replay from any point)"
        - "Long-running workflows (days/weeks)"
        
  strengths:
    - "Production-grade (used by large companies)"
    - "Supports complex stateful workflows"
    - "Cyclic graphs (loops, retries)"
    - "Streaming support (real-time UI updates)"
    - "Human-in-the-loop built-in"
    - "LangGraph Platform (managed deployment)"
    - "Sub-graphs (composable, reusable agent teams)"
    
  limitations:
    - "Steeper learning curve (graph abstraction)"
    - "LangChain ecosystem dependency"
    - "Can be verbose for simple agents"
    
  best_for:
    - "Complex multi-step workflows"
    - "Production deployments needing persistence"
    - "Multi-agent orchestration"
    - "Workflows requiring human approval"
    
  example:
    code: |
      from langgraph.graph import StateGraph, END
      from langgraph.checkpoint.postgres import PostgresSaver
      
      # Define state
      class AgentState(TypedDict):
          messages: Annotated[list, add]
          next_action: str
          
      # Define nodes
      async def call_model(state):
          response = await model.invoke(state["messages"])
          return {"messages": [response]}
          
      async def call_tool(state):
          result = await execute_tool(state["messages"][-1])
          return {"messages": [result]}
      
      # Build graph
      graph = StateGraph(AgentState)
      graph.add_node("agent", call_model)
      graph.add_node("tools", call_tool)
      graph.add_conditional_edges("agent", route_decision)
      graph.add_edge("tools", "agent")
      
      # Compile with persistence
      app = graph.compile(checkpointer=PostgresSaver(conn))
```

### CrewAI

```yaml
CrewAI:
  creator: "CrewAI (open-source)"
  philosophy: "Multi-agent teams defined by roles, tasks, and processes"
  
  key_concepts:
    agents:
      what: "AI entities with specific roles and expertise"
      definition:
        role: "Senior Data Analyst"
        goal: "Analyze data to provide actionable insights"
        backstory: "Expert data scientist with 10 years experience..."
        tools: ["query_database", "execute_python"]
        llm: "claude-4-sonnet"
        
    tasks:
      what: "Specific work items assigned to agents"
      definition:
        description: "Analyze Q4 revenue data and identify growth trends"
        agent: "data_analyst"
        expected_output: "Report with 5 key findings and recommendations"
        dependencies: ["research_task"]  # Must complete first
        
    crews:
      what: "Teams of agents working together"
      definition:
        agents: ["researcher", "analyst", "writer"]
        tasks: ["research", "analyze", "write_report"]
        process: "sequential"  # or "hierarchical"
        
    processes:
      sequential: "Tasks execute one after another"
      hierarchical: "Manager agent delegates and coordinates"
      
  strengths:
    - "Intuitive role-based agent definition"
    - "Simple API (few lines to define a team)"
    - "Built-in collaboration patterns"
    - "Memory sharing between agents"
    - "Active community and ecosystem"
    
  limitations:
    - "Less control over execution flow (vs LangGraph)"
    - "Limited persistence/checkpointing"
    - "More suited to content generation than complex stateful workflows"
    
  best_for:
    - "Multi-agent content creation workflows"
    - "Research and analysis teams"
    - "Business process automation"
    - "Quick prototyping of agent teams"
    
  example:
    code: |
      from crewai import Agent, Task, Crew, Process
      
      researcher = Agent(
          role="Market Researcher",
          goal="Find comprehensive competitive intelligence",
          tools=[web_search, news_api],
          llm="gpt-5-mini",
      )
      
      analyst = Agent(
          role="Business Analyst",
          goal="Synthesize research into actionable insights",
          tools=[execute_python],
          llm="claude-4-sonnet",
      )
      
      research_task = Task(
          description="Research top 5 competitors' pricing strategies",
          agent=researcher,
          expected_output="Detailed pricing comparison document",
      )
      
      analysis_task = Task(
          description="Analyze pricing data and recommend our strategy",
          agent=analyst,
          expected_output="Strategy recommendation with supporting data",
          dependencies=[research_task],
      )
      
      crew = Crew(
          agents=[researcher, analyst],
          tasks=[research_task, analysis_task],
          process=Process.sequential,
      )
      
      result = crew.kickoff()
```

### OpenAI Assistants API

```yaml
OpenAI_Assistants:
  what: "Managed agent runtime by OpenAI"
  philosophy: "Stateful AI assistants with built-in tools and file handling"
  
  key_concepts:
    assistants:
      what: "Configured AI agents with instructions, tools, and model"
      features:
        - "Persistent instructions (system prompt)"
        - "Built-in tools (code interpreter, file search, function calling)"
        - "File handling (upload and process documents)"
        
    threads:
      what: "Conversations with persistent state (managed by OpenAI)"
      features:
        - "Message history maintained automatically"
        - "File attachments per message"
        - "No context window management needed (handled automatically)"
        
    runs:
      what: "Invocation of an assistant on a thread"
      features:
        - "Async execution with polling"
        - "Streaming support"
        - "Tool call handling"
        - "Cancellation support"
        
  built_in_tools:
    code_interpreter:
      what: "Sandboxed Python execution (pandas, matplotlib, etc.)"
      features: "File generation, data analysis, chart creation"
      
    file_search:
      what: "RAG over uploaded documents (managed by OpenAI)"
      features: "Automatic chunking, embedding, retrieval"
      
    function_calling:
      what: "Custom tool integration"
      features: "Same as standard function calling but within assistant framework"
      
  strengths:
    - "Fastest time to prototype (zero infrastructure)"
    - "Managed state (no database needed)"
    - "Code interpreter (free sandboxed execution)"
    - "File search (built-in RAG)"
    - "Simple API"
    
  limitations:
    - "Vendor lock-in (OpenAI only)"
    - "Limited control over internals"
    - "Higher cost (managed infrastructure)"
    - "No custom memory strategies"
    - "Can't use non-OpenAI models"
    - "Limited multi-agent support"
    
  best_for:
    - "Rapid prototyping"
    - "Single-agent assistants with file/code needs"
    - "Teams without ML infrastructure expertise"
```

### Anthropic Tool Use

```yaml
Anthropic_Tools:
  what: "Claude's native function calling and tool use"
  philosophy: "Reliable, safe tool use with detailed thinking"
  
  strengths:
    - "Most reliable tool selection (lowest hallucination of tool calls)"
    - "Excellent at following complex tool-use instructions"
    - "Extended thinking mode for complex planning"
    - "Strong safety defaults (refuses dangerous tool use)"
    - "Computer use (UI automation via screenshots)"
    
  unique_features:
    extended_thinking:
      what: "Claude reasons about tool selection before acting"
      benefit: "Better planning for multi-step tool workflows"
      
    computer_use:
      what: "Agent that can interact with computer UI (click, type, screenshot)"
      use_cases: "Legacy system automation, UI testing, desktop workflows"
      
    batch_tool_calls:
      what: "Multiple independent tool calls in one response"
      benefit: "Parallel execution of independent tools"
      
  limitations:
    - "No managed runtime (you build the execution loop)"
    - "No built-in persistence"
    - "Single provider (can't mix models)"
    
  best_for:
    - "Applications where tool-use reliability is critical"
    - "Complex multi-tool workflows requiring planning"
    - "Safety-sensitive domains (medical, financial)"
```

### Vercel AI SDK

```yaml
Vercel_AI_SDK:
  what: "TypeScript SDK for building AI applications (agents, chatbots, generative UI)"
  philosophy: "Full-stack AI for JavaScript/TypeScript developers"
  
  key_features:
    streaming:
      what: "First-class streaming support for all operations"
      benefit: "Real-time UI updates, token-by-token response rendering"
      
    provider_agnostic:
      what: "Unified interface for OpenAI, Anthropic, Google, open-source"
      benefit: "Switch models without code changes"
      
    structured_output:
      what: "Type-safe structured generation with Zod schemas"
      benefit: "Guaranteed valid output matching TypeScript types"
      
    tool_calling:
      what: "Provider-agnostic tool use with type-safe definitions"
      
    react_integration:
      what: "React hooks for chat, completion, and AI UI"
      hooks: ["useChat", "useCompletion", "useAssistant"]
      
    generative_ui:
      what: "AI that generates React components (not just text)"
      benefit: "Dynamic, interactive AI responses with UI elements"
      
  strengths:
    - "Best DX (Developer Experience) for TypeScript/JavaScript devs"
    - "Streaming-first architecture"
    - "React/Next.js integration"
    - "Edge-deployable (Vercel Edge Functions)"
    - "Provider-agnostic (easy model switching)"
    - "Type-safe (Zod schemas, TypeScript)"
    
  limitations:
    - "JavaScript/TypeScript only (no Python)"
    - "Less suited for complex multi-agent workflows"
    - "Smaller ecosystem than LangChain/LangGraph"
    
  best_for:
    - "Full-stack JS/TS applications"
    - "AI chatbots with rich UI"
    - "Applications deployed on Vercel/Edge"
    - "Projects needing provider flexibility"
    
  example:
    code: |
      // TypeScript agent with Vercel AI SDK
      import { generateText, tool } from 'ai';
      import { openai } from '@ai-sdk/openai';
      import { z } from 'zod';
      
      const result = await generateText({
        model: openai('gpt-5'),
        tools: {
          weather: tool({
            description: 'Get weather for a city',
            parameters: z.object({
              city: z.string().describe('City name'),
            }),
            execute: async ({ city }) => {
              return await fetchWeather(city);
            },
          }),
        },
        maxSteps: 5,  // Allow up to 5 tool calls
        prompt: 'What is the weather in Tokyo and New York?',
      });
```

### Agent-Specialized Open-Weight Models (Self-Hosted)

```yaml
Agent_Models:
  why: "Not every agent needs a $0.03/call API — self-hosted agent models 
        reduce cost 30×, eliminate API dependency, and keep data private"
  
  hermes_3:
    provider: "Nous Research"
    what: "The gold standard open-weight model for AI agents"
    base_model: "Llama 3 (fine-tuned specifically for tool use and agent behavior)"
    sizes: ["8B (single GPU agent)", "70B (production quality)", "405B (frontier open)"]
    
    why_hermes_for_agents:
      - "Trained specifically on function-calling and tool-use datasets"
      - "ChatML format with dedicated <tool_call> / <tool_response> tokens"
      - "95%+ valid JSON in function calls (matches GPT-4o reliability)"
      - "Excellent system prompt adherence (roles, constraints, boundaries)"
      - "Multi-turn tool use (plans across turns, remembers results)"
      - "Knows WHEN to call tools vs answer directly (tool selection intelligence)"
      
    how_to_use:
      prompt_format: |
        <|im_start|>system
        You are a helpful assistant with access to the following tools:
        
        {tools_json_schema}
        
        When you need to use a tool, respond with:
        <tool_call>
        {"name": "function_name", "arguments": {"param": "value"}}
        </tool_call>
        
        When you have enough information, respond normally without tool calls.
        <|im_end|>
        <|im_start|>user
        What's the weather in London and should I bring an umbrella?
        <|im_end|>
        <|im_start|>assistant
        I'll check the weather for you.
        <tool_call>
        {"name": "get_weather", "arguments": {"city": "London", "units": "celsius"}}
        </tool_call>
        <|im_end|>
        <|im_start|>tool
        {"temperature": 12, "condition": "rainy", "humidity": 85}
        <|im_end|>
        <|im_start|>assistant
        The weather in London is 12°C and rainy with 85% humidity.
        Yes, definitely bring an umbrella!
        <|im_end|>
      
      tool_definition_format: |
        # Define tools as JSON Schema (same format as OpenAI function calling)
        tools = [
          {
            "type": "function",
            "function": {
              "name": "get_weather",
              "description": "Get current weather for a city",
              "parameters": {
                "type": "object",
                "properties": {
                  "city": {"type": "string", "description": "City name"},
                  "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
              }
            }
          }
        ]
      
      with_vllm: |
        # Serve Hermes 3 with vLLM (production serving)
        # Install: pip install vllm
        
        # Start server:
        # vllm serve NousResearch/Hermes-3-Llama-3.1-8B \
        #   --dtype auto --max-model-len 8192 \
        #   --tool-call-parser hermes
        
        # vLLM has NATIVE Hermes tool-call parsing support!
        # Client code (OpenAI-compatible API):
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
        
        response = client.chat.completions.create(
            model="NousResearch/Hermes-3-Llama-3.1-8B",
            messages=[
                {"role": "system", "content": "You are a helpful agent with tool access."},
                {"role": "user", "content": "What's the weather in Tokyo?"},
            ],
            tools=tools,  # Same format as OpenAI tools
            tool_choice="auto",
        )
        
        # Response includes tool_calls just like OpenAI API!
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            print(f"Call: {tool_call.function.name}({tool_call.function.arguments})")
      
      with_langchain: |
        # Use Hermes with LangChain/LangGraph (same as any chat model)
        from langchain_openai import ChatOpenAI
        from langchain.tools import tool
        
        # Point to local vLLM server running Hermes
        llm = ChatOpenAI(
            base_url="http://localhost:8000/v1",
            api_key="not-needed",
            model="NousResearch/Hermes-3-Llama-3.1-8B",
        )
        
        @tool
        def search_database(query: str) -> str:
            """Search the customer database by name or email."""
            return db.search(query)
        
        # Bind tools — works exactly like GPT-4o/Claude
        llm_with_tools = llm.bind_tools([search_database])
        response = llm_with_tools.invoke("Find customer John Smith")
      
      with_ollama: |
        # Simplest local setup — Ollama (one command)
        # Install Ollama: https://ollama.ai
        
        # Pull Hermes model:
        # ollama pull adrienbrault/nous-hermes2pro:Q4_K_M
        
        # Or for Hermes 3:
        # ollama pull hermes3:8b
        
        # Use via API (OpenAI compatible):
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        
        response = client.chat.completions.create(
            model="hermes3:8b",
            messages=[{"role": "user", "content": "Search for flights to NYC"}],
            tools=tools,
        )
      
      with_llama_cpp: |
        # CPU inference (no GPU needed) — good for development
        # Download GGUF: NousResearch/Hermes-3-Llama-3.1-8B-GGUF
        # Use Q4_K_M quantization (best quality/size balance)
        
        # Run with llama-server:
        # llama-server -m hermes-3-8b-Q4_K_M.gguf \
        #   --port 8080 --n-gpu-layers 0
        
        # Same OpenAI-compatible API as vLLM/Ollama
    
    practical_tips:
      - "Always include tool descriptions in system prompt (Hermes uses them for selection)"
      - "Use ChatML format (<|im_start|> tokens) for best results"
      - "For complex agents: Hermes 70B matches Claude Sonnet on tool use"
      - "For simple tool routing: Hermes 8B is sufficient (fast + cheap)"
      - "GGUF Q4_K_M works well for development (CPU, no GPU needed)"
      - "vLLM --tool-call-parser hermes gives OpenAI-compatible tool_calls in response"
      - "Works with LangChain, LangGraph, CrewAI via standard OpenAI-compatible interface"
      
    cost_comparison:
      hermes_8b_self_hosted:
        hardware: "1× L4 ($0.80/hr)"
        throughput: "100 agent calls/minute"
        cost_per_call: "$0.00013"
      hermes_70b_self_hosted:
        hardware: "1× H100 ($8/hr)"
        throughput: "30 agent calls/minute"
        cost_per_call: "$0.004"
      gpt_4o_api:
        cost_per_call: "$0.03-0.10 (depending on tokens)"
        comparison: "Hermes 8B is 230× cheaper, 70B is 7-25× cheaper"
      
  other_agent_models:
    functionary:
      provider: "MeetKai"
      what: "Trained on OpenAI function-calling format (drop-in replacement)"
      strength: "Exact same JSON format as OpenAI tools API"
      
    gorilla:
      provider: "UC Berkeley (Gorilla LLM)"
      what: "Trained on API documentation (calls real-world APIs)"
      strength: "Can call APIs it hasn't seen (generalizes from docs)"
      
    nexus_raven:
      provider: "Nexusflow"
      what: "Function-calling model with zero-shot tool use"
      strength: "Works with unseen tools (just from description)"
      
  model_selection_for_agents:
    simple_routing: "Hermes 8B / Functionary 7B (fast, cheap, single tool choice)"
    complex_planning: "Hermes 70B (multi-step plans with tools)"
    maximum_quality: "Claude 4 / GPT-5 API (when cost isn't the constraint)"
    hybrid: "Hermes 8B for routing + frontier model for complex reasoning"
```

### Framework Selection Guide

```python
# Decision framework for selecting agent framework

def select_framework(requirements: dict) -> str:
    """Select the best agent framework based on requirements."""
    
    # Factors
    language = requirements.get("language", "python")
    complexity = requirements.get("complexity", "medium")  # simple, medium, complex
    multi_agent = requirements.get("multi_agent", False)
    persistence_needed = requirements.get("persistence", False)
    provider_flexibility = requirements.get("multi_provider", True)
    prototype_speed = requirements.get("speed_priority", False)
    production_grade = requirements.get("production", True)
    
    # TypeScript → Vercel AI SDK
    if language == "typescript":
        return "Vercel AI SDK"
    
    # Self-hosted / cost-sensitive / privacy-required → Hermes
    if requirements.get("self_hosted", False) or requirements.get("data_privacy", False):
        volume = requirements.get("daily_calls", 0)
        if volume > 10000 or requirements.get("air_gapped", False):
            return "Hermes 3 (8B/70B) + vLLM + LangGraph"
    
    # Quick prototype with file/code needs → OpenAI Assistants
    if prototype_speed and not provider_flexibility and not multi_agent:
        return "OpenAI Assistants API"
    
    # Multi-agent content/research workflows → CrewAI
    if multi_agent and complexity in ["simple", "medium"]:
        return "CrewAI"
    
    # Complex stateful workflows, production → LangGraph
    if production_grade and (persistence_needed or complexity == "complex"):
        return "LangGraph"
    
    # Simple single-agent, maximum control → Raw function calling
    if complexity == "simple" and not multi_agent:
        return "Raw function calling (no framework)"
    
    # Default for Python production
    return "LangGraph"
```

---

## How It Works in Practice

### Framework Comparison Matrix

```yaml
Comparison:
  dimensions:
    learning_curve:
      raw_function_calling: "Low (you know the APIs)"
      openai_assistants: "Low (managed, simple API)"
      crewai: "Low-Medium (intuitive concepts)"
      vercel_ai_sdk: "Medium (TypeScript + AI concepts)"
      langgraph: "Medium-High (graph abstraction)"
      
    production_readiness:
      raw_function_calling: "You build it (varies)"
      openai_assistants: "Medium (managed but limited)"
      crewai: "Medium (good for batch, less for real-time)"
      vercel_ai_sdk: "High (built for production web apps)"
      langgraph: "High (persistence, streaming, deployment platform)"
      
    multi_agent:
      raw_function_calling: "Manual implementation"
      openai_assistants: "Limited (single assistant per thread)"
      crewai: "Excellent (designed for this)"
      vercel_ai_sdk: "Basic (manual orchestration)"
      langgraph: "Excellent (sub-graphs, shared state)"
      
    provider_flexibility:
      raw_function_calling: "Full (you choose)"
      openai_assistants: "None (OpenAI only)"
      crewai: "Good (supports multiple providers)"
      vercel_ai_sdk: "Excellent (provider-agnostic by design)"
      langgraph: "Good (via LangChain providers)"
      
    cost:
      raw_function_calling: "Lowest (no overhead)"
      openai_assistants: "Higher (managed infrastructure fee)"
      crewai: "Medium (multiple agents = multiple calls)"
      vercel_ai_sdk: "Low (just SDK, you provide models)"
      langgraph: "Medium (LangGraph Platform has fees)"
```

---

## Interview Tip

> When asked about agent frameworks: "My framework selection depends on the use case: (1) LangGraph — my default for production Python agents. Graph-based architecture handles complex workflows naturally: conditional routing, cycles (retry loops), human-in-the-loop (pause for approval), and checkpointing (resume on failure). I use it for any workflow with >3 steps or requiring state persistence. (2) CrewAI — when I need multi-agent collaboration with clear roles (researcher, analyst, writer). Faster to set up than LangGraph for team-based workflows, more intuitive role-based API. (3) Vercel AI SDK — for TypeScript/Next.js applications. Provider-agnostic, streaming-first, great React integration with hooks. My choice for customer-facing AI features in web apps. (4) Raw function calling — for simple agents (1-3 tool calls) where framework overhead isn't justified. Direct API calls with a simple loop. (5) OpenAI Assistants — for rapid prototyping when I need code execution or file search without building infrastructure. Not for production (too vendor-locked). Key insight: I often combine frameworks — LangGraph orchestrates the overall workflow, with individual nodes calling different models via their native APIs. The framework handles state, persistence, and routing; the actual LLM calls are straightforward."

---

## Common Mistakes

1. **Framework before problem** — Choosing LangGraph because it's popular, then building a simple FAQ bot as a 15-node graph. A single API call with function calling would suffice. Solution: choose framework based on actual complexity needs. Simple agents don't need frameworks.

2. **Vendor lock-in without realizing** — Building everything on OpenAI Assistants API, then needing to switch to Claude (better quality for your use case). Migration requires rewriting everything. Solution: use provider-agnostic frameworks (LangGraph, Vercel AI SDK) or abstract the LLM call layer so model switching is a config change.

3. **Ignoring persistence requirements** — Building an agent with CrewAI for a workflow that runs for 30 minutes. Network glitch at minute 25 → restart from scratch. Solution: if workflows exceed 5 minutes or involve human-in-the-loop, choose a framework with checkpointing (LangGraph with PostgresSaver).

4. **Over-engineering with LangGraph** — Creating a graph for what should be a linear pipeline. 20 nodes, conditional edges everywhere, complex state management — for a workflow that always goes A→B→C→D. Solution: use the simplest abstraction that works. Sequential pipeline? Just call functions in order. Only use graphs when you genuinely need conditional routing or cycles.

5. **Not considering the team's language** — Choosing a Python framework (LangGraph) when the entire team writes TypeScript and the product is a Next.js app. Now you need Python infrastructure + a service boundary + deployment complexity. Solution: match framework to team skills and existing stack. Vercel AI SDK for JS teams, LangGraph for Python teams.

---

## Key Takeaways

- LangGraph: production Python agents with complex workflows, persistence, human-in-the-loop
- CrewAI: multi-agent teams with clear roles (research, analysis, writing)
- OpenAI Assistants: fastest prototype (managed state, code execution, file search) — vendor-locked
- Vercel AI SDK: TypeScript/Next.js full-stack AI, provider-agnostic, streaming-first
- Raw function calling: simple agents (1-3 tools) where frameworks add unnecessary complexity
- Match framework to: language (Python vs TS), complexity, multi-agent needs, persistence requirements
- Provider flexibility matters: avoid lock-in to single model provider
- Checkpointing critical for: long-running workflows (>5 min), human-in-the-loop, failure recovery
- Combine frameworks: LangGraph orchestration + native API calls per node
- Don't over-engineer: if a simple loop works, don't build a graph
