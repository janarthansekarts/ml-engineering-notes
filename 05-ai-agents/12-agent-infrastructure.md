# Agent Infrastructure

## The Problem / Why This Matters

Agent infrastructure is the runtime layer that makes agents operational at scale — MCP (Model Context Protocol) servers that expose tools, agent orchestration platforms that manage execution, tool hosting services that provide capabilities, and the coordination layer that connects everything. Without proper infrastructure: each agent team builds their own tool integration (duplicated effort), tools can't be shared across agents (waste), there's no standard for tool discovery (agents can't find what's available), state management is ad-hoc (each agent has its own persistence approach), and multi-agent coordination requires custom code (brittle). In 2026, MCP has emerged as the standard protocol for agent-tool communication — analogous to how HTTP standardized web communication. Understanding agent infrastructure means: knowing how to build and deploy MCP servers, how to host agent runtimes that scale, how to manage tool registries, and how to orchestrate multi-agent systems in production. This is the "DevOps for agents" layer that most teams underinvest in and later regret.

---

## The Analogy

Think of agent infrastructure like a city's infrastructure:

- **MCP servers** = Utility services (electricity, water, internet). They provide capabilities that any building (agent) can connect to. Standardized interfaces (plugs, pipes, ports) mean you don't build custom connections.
- **Agent runtime** = The building itself. Provides the space (compute), utilities connections (tool access), and management (who's in the building, what they're doing).
- **Tool registry** = Yellow pages / service directory. Agents look up what tools are available, what they do, how to use them. Without it, agents don't know what exists.
- **Orchestration platform** = City planning department. Coordinates which buildings (agents) serve which purposes, manages traffic flow (requests), ensures services don't conflict.
- **Without infrastructure** = Each building generates its own electricity, pumps its own water, runs its own fiber. Expensive, unreliable, unmaintainable at scale.

---

## Deep Dive

### MCP (Model Context Protocol)

```yaml
MCP_Protocol:
  what: "Open standard for connecting AI agents to tools and data sources"
  created_by: "Anthropic (2024), now widely adopted across the industry"
  philosophy: "Standardize how agents discover and invoke tools — like USB for AI"
  
  architecture:
    client: "The agent/LLM application that needs to use tools"
    server: "The service that exposes tools, resources, and prompts"
    transport: "Communication layer (stdio, HTTP/SSE, WebSocket)"
    
  protocol_components:
    tools:
      what: "Actions the agent can execute"
      definition:
        name: "unique_tool_name"
        description: "What the tool does (used by LLM for selection)"
        input_schema: "JSON Schema for parameters"
      example:
        name: "query_database"
        description: "Execute a read-only SQL query against the analytics database"
        input_schema:
          type: "object"
          properties:
            query:
              type: "string"
              description: "SQL SELECT query"
          required: ["query"]
          
    resources:
      what: "Data sources the agent can read (similar to GET endpoints)"
      types:
        - "File contents (read source code)"
        - "Database records (read customer data)"
        - "API responses (read external data)"
      difference_from_tools: "Resources are read-only, tools can have side effects"
      
    prompts:
      what: "Reusable prompt templates the server provides"
      use_case: "Standardized ways to ask the agent to do things"
      
  transport_options:
    stdio:
      what: "Communication via standard input/output"
      use_case: "Local tools (CLI tools, file system access)"
      pros: "Simple, no network needed"
      cons: "Same machine only"
      
    http_sse:
      what: "HTTP with Server-Sent Events for streaming"
      use_case: "Remote MCP servers (production deployments)"
      pros: "Network-accessible, standard HTTP infrastructure"
      cons: "More setup than stdio"
      
    streamable_http:
      what: "Modern HTTP-based transport (replacing SSE in MCP spec)"
      use_case: "Production remote servers (2026 standard)"
      pros: "Better connection management, bidirectional"
```

### Building MCP Servers

```python
# Building a production MCP server

"""
MCP servers expose tools that agents can discover and call.
This example shows a database MCP server.
"""

from mcp.server import Server
from mcp.types import Tool, TextContent
import asyncpg
from typing import Any


# Create MCP server
app = Server("database-mcp-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Declare available tools to agents."""
    return [
        Tool(
            name="query_analytics",
            description="Run a read-only SQL query against the analytics database. "
                       "Supports SELECT queries only. Tables: events, users, sessions, "
                       "revenue. Max 1000 rows returned.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL SELECT query (read-only)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows to return (default 100, max 1000)",
                        "default": 100
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_table_schema",
            description="Get the schema (columns, types) for a database table.",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table",
                        "enum": ["events", "users", "sessions", "revenue"]
                    }
                },
                "required": ["table_name"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool invocations from agents."""
    
    if name == "query_analytics":
        return await _handle_query(arguments)
    elif name == "get_table_schema":
        return await _handle_schema(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _handle_query(arguments: dict) -> list[TextContent]:
    """Execute a validated read-only query."""
    
    query = arguments["query"].strip()
    limit = min(arguments.get("limit", 100), 1000)
    
    # Security: validate read-only
    if not _is_read_only(query):
        return [TextContent(
            type="text",
            text="Error: Only SELECT queries are allowed. "
                 "INSERT, UPDATE, DELETE, DROP, and other write operations are blocked."
        )]
    
    # Security: prevent SQL injection via parameterization
    # (query is already from LLM, but we still validate structure)
    if _contains_dangerous_patterns(query):
        return [TextContent(type="text", text="Error: Query contains blocked patterns.")]
    
    # Execute with timeout and row limit
    try:
        async with asyncpg.create_pool(DATABASE_URL) as pool:
            async with pool.acquire() as conn:
                # Set statement timeout (prevent long-running queries)
                await conn.execute("SET statement_timeout = '10s'")
                
                # Add LIMIT if not present
                if "limit" not in query.lower():
                    query = f"{query} LIMIT {limit}"
                
                rows = await conn.fetch(query)
                
                # Format results
                if not rows:
                    return [TextContent(type="text", text="Query returned 0 rows.")]
                
                # Convert to readable format
                headers = list(rows[0].keys())
                result_text = " | ".join(headers) + "\n"
                result_text += "-" * 50 + "\n"
                for row in rows:
                    result_text += " | ".join(str(row[h]) for h in headers) + "\n"
                
                result_text += f"\n({len(rows)} rows returned)"
                
                return [TextContent(type="text", text=result_text)]
                
    except asyncpg.QueryCanceledError:
        return [TextContent(type="text", text="Error: Query timed out (10s limit). Try a simpler query.")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error executing query: {str(e)}")]


def _is_read_only(query: str) -> bool:
    """Check if query is read-only (SELECT only)."""
    # Normalize and check
    normalized = query.upper().strip()
    
    # Must start with SELECT or WITH (CTE)
    if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
        return False
    
    # Block write keywords anywhere in query
    write_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", 
                      "TRUNCATE", "GRANT", "REVOKE", "EXEC", "EXECUTE"]
    for keyword in write_keywords:
        if keyword in normalized:
            return False
    
    return True


# Run the MCP server
if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server
    
    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream)
    
    asyncio.run(main())
```

### Agent Orchestration Platforms

```yaml
Orchestration_Platforms:
  langgraph_platform:
    what: "Managed deployment for LangGraph agents"
    features:
      - "Deploy agents as services (API endpoints)"
      - "Built-in persistence (PostgreSQL)"
      - "Streaming support (SSE)"
      - "Cron jobs (scheduled agent runs)"
      - "Human-in-the-loop (pause/resume)"
      - "Multi-tenant (isolated deployments)"
    deployment_options:
      cloud: "LangGraph Cloud (fully managed)"
      self_hosted: "LangGraph Platform on your infrastructure"
    pricing: "Per-execution + storage"
    
  custom_orchestration:
    when: "Need more control than managed platforms provide"
    components:
      task_queue:
        technology: "Redis Streams, AWS SQS, or Kafka"
        purpose: "Buffer incoming tasks, handle backpressure"
        
      worker_pool:
        technology: "Kubernetes Deployments + HPA (Horizontal Pod Autoscaler)"
        purpose: "Execute agent loops with auto-scaling"
        
      state_store:
        technology: "PostgreSQL + Redis (hot cache)"
        purpose: "Persist agent state, checkpoints, conversation history"
        
      event_bus:
        technology: "Redis Pub/Sub or NATS"
        purpose: "Real-time events (streaming to clients, inter-agent communication)"
        
      tool_registry:
        technology: "Custom service or Consul/etcd"
        purpose: "Discover available MCP servers and their tools"
        
      api_gateway:
        technology: "Kong, Traefik, or custom FastAPI"
        purpose: "Auth, rate limiting, request routing"
```

### Tool Hosting and Registry

```yaml
Tool_Registry:
  what: "Service that tracks available tools, their locations, and capabilities"
  
  why_needed:
    - "Agents need to discover what tools exist"
    - "Tools may be deployed on different servers/regions"
    - "Tools have different availability (some offline for maintenance)"
    - "Different agents have access to different tool sets (permissions)"
    - "Tool versions change (need backward compatibility)"
    
  registry_schema:
    tool_entry:
      id: "unique-tool-id"
      name: "query_analytics_db"
      description: "Run read-only queries against analytics database"
      version: "2.1.0"
      mcp_server_url: "https://mcp-analytics.internal/sse"
      status: "active"  # active, deprecated, maintenance
      owner_team: "data-platform"
      permissions: ["analyst", "engineer", "admin"]
      rate_limit: 100  # calls per minute
      avg_latency_ms: 250
      input_schema: { ... }  # JSON Schema
      
  operations:
    discover: "Agent queries: 'What tools can help me analyze user behavior?'"
    register: "New MCP server registers its tools on startup"
    deregister: "MCP server going offline removes its tools"
    health_check: "Periodic verification that registered tools are responsive"
    versioning: "Support multiple versions of same tool (gradual migration)"
    
  implementation:
    simple: "Static configuration file listing all tools (small scale)"
    medium: "Database-backed registry with admin UI (medium scale)"
    advanced: "Service mesh with automatic discovery (large scale, Kubernetes)"
```

### Multi-Agent Coordination

```yaml
Multi_Agent_Coordination:
  patterns:
    hierarchical:
      what: "Manager agent delegates to specialist agents"
      architecture:
        manager:
          role: "Decompose task, delegate, aggregate results"
          model: "Most capable model (Claude 4 Opus for complex planning)"
        specialists:
          - "Research agent (web search, document analysis)"
          - "Code agent (implementation, testing)"
          - "Writing agent (documentation, reports)"
          - "Data agent (SQL queries, analysis)"
      communication: "Manager sends subtask → specialist returns result"
      state_sharing: "Via shared state store (not direct communication)"
      
    pipeline:
      what: "Agents process in sequence (output of one → input of next)"
      architecture:
        stage_1: "Data collection agent (gather information)"
        stage_2: "Analysis agent (process and analyze)"
        stage_3: "Report agent (generate output)"
      communication: "Queue-based (SQS/Redis) between stages"
      benefit: "Each agent is simple, focused, independently scalable"
      
    collaborative:
      what: "Agents work on shared state, contribute independently"
      architecture:
        shared_state: "Common document/workspace all agents can read and write"
        agents: "Multiple agents working on different aspects simultaneously"
        coordination: "Lock-based or event-based to prevent conflicts"
      use_case: "Complex research tasks requiring multiple perspectives"
      
  infrastructure_for_coordination:
    shared_state_store:
      what: "Central state that all agents read/write"
      technology: "PostgreSQL with row-level locking or Redis with WATCH"
      
    message_passing:
      what: "Agents send messages to each other"
      technology: "Redis Pub/Sub, NATS, or Kafka (for durability)"
      
    task_queue:
      what: "Manager enqueues subtasks, specialists dequeue"
      technology: "Redis Streams or SQS (with dead letter queue)"
      
    result_aggregation:
      what: "Collect results from multiple agents"
      technology: "Reducer function triggered when all subtasks complete"
```

### Infrastructure as Code for Agents

```yaml
Agent_Infrastructure_as_Code:
  kubernetes_manifests:
    mcp_server:
      deployment:
        replicas: 3
        image: "registry.internal/mcp-database:2.1.0"
        resources:
          requests: { cpu: "500m", memory: "512Mi" }
          limits: { cpu: "2000m", memory: "2Gi" }
        env:
          - "DATABASE_URL (from Secret)"
          - "MCP_TRANSPORT=http_sse"
          - "MAX_CONNECTIONS=50"
        health_check: "/health (checks DB connection)"
      service:
        type: "ClusterIP"
        port: 8080
      hpa:
        min_replicas: 3
        max_replicas: 20
        target_cpu: "70%"
        
    agent_workers:
      deployment:
        replicas: "5 (auto-scaled by queue depth)"
        image: "registry.internal/agent-worker:1.5.0"
        resources:
          requests: { cpu: "1000m", memory: "2Gi" }
          limits: { cpu: "4000m", memory: "8Gi" }
        env:
          - "REDIS_URL (queue connection)"
          - "POSTGRES_URL (state store)"
          - "MCP_REGISTRY_URL (tool discovery)"
          - "LLM_API_KEY (from Secret)"
        lifecycle:
          preStop: "Checkpoint current agent, re-queue task"
          
    tool_registry:
      deployment:
        replicas: 2
        image: "registry.internal/tool-registry:1.0.0"
        resources:
          requests: { cpu: "250m", memory: "256Mi" }
      service:
        type: "ClusterIP"
        port: 8081
        
  monitoring_stack:
    prometheus:
      scrape_targets:
        - "mcp-servers (tool call metrics)"
        - "agent-workers (execution metrics)"
        - "tool-registry (discovery metrics)"
    grafana:
      dashboards:
        - "Agent performance (completion rate, latency, cost)"
        - "MCP server health (availability, error rate)"
        - "Tool usage (calls per tool, failure rates)"
    alertmanager:
      alerts:
        - "MCP server down for >5 min"
        - "Agent error rate >10%"
        - "Tool registry unreachable"
        - "Agent cost exceeds budget"
```

---

## How It Works in Practice

### Production MCP Server Deployment

```python
# Production-ready MCP server with HTTP transport

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route
import uvicorn


# Create server with production configuration
app = Server("production-mcp-server")

# ... register tools (as shown above) ...

# HTTP/SSE transport for production
sse = SseServerTransport("/messages")

starlette_app = Starlette(
    routes=[
        Route("/sse", endpoint=sse.handle_sse),
        Route("/messages", endpoint=sse.handle_post_message, methods=["POST"]),
        Route("/health", endpoint=health_check),
    ]
)

if __name__ == "__main__":
    uvicorn.run(starlette_app, host="0.0.0.0", port=8080)
```

---

## Interview Tip

> When asked about agent infrastructure: "I think of agent infrastructure in four layers: (1) Tool layer — MCP servers that expose capabilities (database queries, API calls, file operations) via a standardized protocol. Each MCP server is a microservice that agents discover through a tool registry. I build them with proper security (read-only validation, rate limiting, timeout protection) and deploy on Kubernetes with auto-scaling. (2) Runtime layer — agent workers that execute the reasoning loop. These are stateful workers processing tasks from a queue, with checkpoint persistence (PostgreSQL) for failure recovery and graceful shutdown. Auto-scaled by queue depth, not CPU. (3) Orchestration layer — how agents coordinate. For single agents: LangGraph handles the state machine. For multi-agent: hierarchical pattern (manager delegates to specialists) with shared state store and message passing via Redis. (4) Platform layer — monitoring (OpenTelemetry traces per agent execution), cost tracking (per-tenant budgets), tool registry (discovery + health checks), and the API gateway (auth, rate limiting). MCP is the key standardization: it means I build a tool once (as an MCP server) and any agent can use it — regardless of which LLM or framework the agent uses. This is analogous to how REST standardized web services."

---

## Common Mistakes

1. **Building tools without MCP** — Every agent team builds custom tool integrations (direct API calls, custom protocols). Same tool implemented 5 different ways across 5 agents. Solution: standardize on MCP. Build each tool as an MCP server once, any agent can discover and use it. Investment pays off at 2+ agents.

2. **No tool registry** — Agents hardcode tool URLs. New tool deployed? Must update every agent's config. Tool moved to new server? Breaking change for all agents. Solution: central tool registry with service discovery. Agents query registry at startup (or periodically), get current tool locations, handle unavailability gracefully.

3. **Monolithic agent** — One giant agent with 50 tools and a 10,000-word system prompt. Confusing for the LLM, slow to start, impossible to test in isolation. Solution: compose from smaller, focused agents (or split into sub-graphs). Each specialist has 5-10 tools and a clear scope. Manager routes to appropriate specialist.

4. **No graceful degradation** — MCP server goes down → agent crashes. Database timeout → entire workflow fails. Solution: design for partial failure. Agent should handle: tools being unavailable (skip or use alternatives), slow responses (timeout and inform user), and degraded data (work with what's available, flag limitations).

5. **Ignoring infrastructure cost** — Focus only on LLM API costs, but infrastructure (Kubernetes nodes for MCP servers, PostgreSQL for state, Redis for queues) adds up. 10 MCP servers × 3 replicas × 24/7 = significant compute. Solution: right-size infrastructure, use spot instances for workers, scale down during low-traffic periods, consider serverless for low-volume tools.

---

## Key Takeaways

- MCP (Model Context Protocol): standard for agent-tool communication — build once, any agent can use
- MCP servers: microservices exposing tools via standardized protocol (stdio for local, HTTP/SSE for remote)
- Tool registry: central discovery service so agents find tools dynamically
- Agent runtime: queue-based workers with checkpoint persistence and graceful shutdown
- Multi-agent coordination: hierarchical (manager + specialists), pipeline, or collaborative patterns
- Infrastructure as Code: Kubernetes manifests for MCP servers, workers, registry, monitoring
- Security in MCP servers: input validation, read-only enforcement, rate limiting, timeout protection
- Monitoring: traces per agent execution, metrics per tool (latency, error rate, calls)
- Scale pattern: auto-scale workers by queue depth, MCP servers by CPU/request count
- Design for partial failure: tools can be unavailable, agents should degrade gracefully
