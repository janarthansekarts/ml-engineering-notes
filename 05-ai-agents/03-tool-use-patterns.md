# Tool Use Patterns

## The Problem / Why This Matters

An LLM without tools is like a brain without hands — it can reason and plan but can't act in the world. Tools give agents the ability to: search the internet, query databases, execute code, send emails, modify files, call APIs, and interact with any external system. But tool use in production isn't just "give the model some functions." It's a complex engineering challenge: How do you define tools so the LLM consistently selects the right one? How do you handle tool failures gracefully? How do you sandbox execution so a misbehaving agent can't cause damage? How do you manage tool permissions (which agents get access to which tools)? How do you scale tool registries across an organization? In 2026, the ecosystem has matured with MCP (Model Context Protocol) providing a standard protocol for tool integration, function calling built into every major model, and tool registries that let organizations manage hundreds of tools safely. Getting tool use right is the difference between a demo that works and a production agent that's reliable, safe, and maintainable.

---

## The Analogy

Think of tool use like a contractor using power tools:

- **Tool definition** = The instruction manual for each power tool. Clear enough that anyone (any LLM) can understand when and how to use it. A bad manual leads to misuse.
- **Function calling** = The contractor deciding which tool to use and operating it. "This job needs a drill (not a hammer)" → configures the drill (parameters) → drills the hole → checks the result.
- **Tool registry** = The workshop tool cabinet. Organized, labeled, access-controlled (apprentice can't use the table saw without supervision).
- **Sandboxing** = Safety equipment and workspace barriers. Even if the contractor makes a mistake, they can't damage the structural beams (critical systems) because those are behind a safety barrier.
- **MCP** = Universal tool connectors. Like USB-C — any device works with any port. Any MCP tool works with any MCP-compatible agent.

---

## Deep Dive

### Function Calling Mechanics

```yaml
Function_Calling:
  what: "The model outputs a structured request to call a specific tool with parameters"
  
  how_it_works:
    step_1: "Model receives user query + list of available tools (with descriptions)"
    step_2: "Model decides whether to call a tool (or answer directly)"
    step_3: "If calling a tool, model outputs: tool name + parameters (as JSON)"
    step_4: "Application executes the tool with provided parameters"
    step_5: "Tool result is sent back to the model"
    step_6: "Model incorporates result into its reasoning/response"
    
  provider_formats:
    openai:
      request: |
        tools=[
          {"type": "function", "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {"type": "object", "properties": {...}}
          }}
        ]
      response: |
        message.tool_calls = [
          {"id": "call_abc123", "function": {
            "name": "get_weather",
            "arguments": '{"city": "Tokyo", "units": "celsius"}'
          }}
        ]
        
    anthropic:
      request: |
        tools=[
          {"name": "get_weather", "description": "...",
           "input_schema": {"type": "object", "properties": {...}}}
        ]
      response: |
        content = [
          {"type": "tool_use", "id": "toolu_abc",
           "name": "get_weather", "input": {"city": "Tokyo"}}
        ]
        
    differences:
      - "Field names differ (parameters vs input_schema)"
      - "Response structure differs"
      - "Parallel tool calling support varies"
      - "Streaming behavior differs"
      
  parallel_tool_calls:
    what: "Model calls multiple tools simultaneously (when independent)"
    example: |
      User: "What's the weather in Tokyo and the stock price of Toyota?"
      Model calls: get_weather(city="Tokyo") AND get_stock(symbol="TM") simultaneously
    benefit: "Faster execution (parallel vs sequential)"
    support: "OpenAI ✓, Anthropic ✓, Google ✓"
```

### Tool Definition Best Practices

```yaml
Tool_Design:
  naming:
    good: "search_web, create_jira_ticket, query_database, send_slack_message"
    bad: "process, handle, do_thing, tool1"
    rule: "verb_noun format, specific and descriptive"
    
  descriptions:
    critical_importance: |
      The LLM decides which tool to use BASED ON THE DESCRIPTION.
      A vague description = wrong tool selection. A clear description = reliable selection.
      
    good_description: |
      "Search the web for current, real-time information. Use this when you need 
      facts that may have changed since your training data (news, prices, events, 
      current statistics). Returns top search results with titles, URLs, and snippets.
      Do NOT use for historical facts or general knowledge (answer those directly)."
      
    bad_description: "Searches the web"
    
    include_in_description:
      - "WHEN to use this tool (positive triggers)"
      - "When NOT to use it (prevent misuse)"
      - "What it returns (so model knows what to expect)"
      - "Limitations (rate limits, data freshness, scope)"
      
  parameters:
    best_practices:
      - "Clear type + description for every parameter"
      - "Required vs optional clearly marked"
      - "Enum values for constrained choices"
      - "Default values for optional parameters"
      - "Examples in description for complex parameters"
      
    example:
      name: "query_database"
      parameters:
        type: "object"
        properties:
          query:
            type: "string"
            description: "SQL query to execute. Must be a SELECT statement (no modifications allowed). Example: 'SELECT name, email FROM users WHERE created_at > '2026-01-01''"
          database:
            type: "string"
            enum: ["production_replica", "analytics", "staging"]
            description: "Which database to query. Use 'production_replica' for live data, 'analytics' for historical aggregations."
          limit:
            type: "integer"
            description: "Maximum rows to return"
            default: 100
        required: ["query", "database"]
        
  error_handling:
    tool_should_return:
      on_success: "Structured result (JSON) with relevant data"
      on_error: "Clear error message explaining what went wrong and how to fix it"
      never: "Stack traces, internal system details, or empty responses"
      
    error_format:
      good: '{"error": "Query timeout after 30s. Try a simpler query or add a WHERE clause to reduce results."}'
      bad: '{"error": "psycopg2.OperationalError: server closed the connection unexpectedly"}'
      
    retry_guidance: |
      Include in error response whether the agent should:
      - Retry (transient error): "Temporary failure. Safe to retry."
      - Modify and retry: "Query too broad. Add filters and try again."
      - Give up: "Permission denied. You don't have access to this resource."
```

### MCP (Model Context Protocol) Deep Dive

```python
# MCP Server implementation example

from mcp.server import Server
from mcp.types import Tool, TextContent
import json


# Create MCP server
server = Server("database-tools")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Declare available tools to MCP clients."""
    return [
        Tool(
            name="query_postgres",
            description="""Execute a read-only SQL query against the PostgreSQL database.
            
            Use this for:
            - Looking up user information
            - Checking order status
            - Aggregating metrics
            
            Limitations:
            - Read-only (SELECT only, no INSERT/UPDATE/DELETE)
            - Maximum 1000 rows returned
            - 30 second timeout
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL SELECT query to execute"
                    },
                    "params": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Parameterized query values (for safety)"
                    }
                },
                "required": ["sql"]
            }
        ),
        Tool(
            name="list_tables",
            description="List all available database tables and their columns. Use this before writing queries to understand the schema.",
            inputSchema={
                "type": "object",
                "properties": {},
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a tool call from an MCP client."""
    
    if name == "query_postgres":
        return await _execute_query(arguments["sql"], arguments.get("params"))
    elif name == "list_tables":
        return await _list_tables()
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _execute_query(sql: str, params: list = None) -> list[TextContent]:
    """Execute SQL with safety checks."""
    
    # Safety: only allow SELECT
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return [TextContent(
            type="text",
            text=json.dumps({"error": "Only SELECT queries are allowed. Cannot execute modifications."})
        )]
    
    # Safety: block dangerous patterns
    dangerous_patterns = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE"]
    for pattern in dangerous_patterns:
        if pattern in sql_upper:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Query contains forbidden keyword: {pattern}"})
            )]
    
    try:
        # Execute with parameterized query (prevents SQL injection)
        results = await db.execute(sql, params or [])
        
        # Limit results
        rows = results[:1000]
        
        return [TextContent(
            type="text",
            text=json.dumps({"rows": rows, "total_count": len(results), "truncated": len(results) > 1000})
        )]
    except Exception as e:
        return [TextContent(
            type="text", 
            text=json.dumps({"error": f"Query failed: {str(e)}. Check SQL syntax and table names."})
        )]
```

### Tool Sandboxing and Security

```yaml
Sandboxing:
  why: |
    Agents execute tools based on LLM decisions. LLMs can be wrong or manipulated.
    Without sandboxing, a misbehaving agent could:
    - Delete production data (DROP TABLE)
    - Exfiltrate sensitive data (send PII to external API)
    - Execute malicious code (if code execution tool exists)
    - Make unauthorized API calls (spend money, send emails)
    
  layers:
    permission_system:
      what: "Define what each agent/tool can and cannot do"
      implementation:
        read_only_tools: "Most tools should be read-only by default"
        write_tools: "Require explicit permission grant"
        destructive_tools: "Require human approval before execution"
      example:
        customer_service_agent:
          allowed: ["lookup_order", "check_status", "search_knowledge_base"]
          requires_approval: ["initiate_refund", "cancel_order"]
          never: ["delete_account", "modify_billing", "access_admin"]
          
    execution_sandbox:
      what: "Isolate tool execution from host system"
      options:
        docker_container: "Run code execution in ephemeral containers"
        serverless_function: "Each tool call = isolated function invocation"
        vm_sandbox: "Full VM isolation for maximum security"
        wasm: "WebAssembly sandbox for lightweight isolation"
      code_execution:
        allowed: "Python with restricted imports (no os, subprocess, socket)"
        blocked: "System calls, network access, file access outside sandbox"
        resource_limits: "CPU time limit, memory limit, no disk writes"
        
    network_isolation:
      what: "Control what external systems tools can access"
      approach:
        - "Allowlist of permitted external APIs/domains"
        - "No raw internet access (only through approved gateways)"
        - "Internal services accessed via service mesh with mTLS"
        
    audit_trail:
      what: "Log every tool invocation for accountability"
      log:
        - "Timestamp"
        - "Agent identity"
        - "Tool called"
        - "Parameters provided"
        - "Result returned"
        - "Human approval (if required)"
      retention: "90 days minimum for compliance"
      
  human_in_the_loop:
    what: "Require human approval for high-risk actions"
    implementation:
      always_approve:
        - "Financial transactions (refunds, charges)"
        - "Destructive operations (delete, cancel)"
        - "External communications (emails to customers)"
        - "System modifications (config changes, deployments)"
      auto_approve:
        - "Read operations (search, lookup)"
        - "Internal computations (calculate, analyze)"
        - "Draft creation (generate content without sending)"
    ux:
      - "Agent pauses with: 'I'd like to [action]. Approve? [Yes/No]'"
      - "Timeout: if no response in 5 minutes, skip action"
      - "Batch: show all pending approvals in dashboard"
```

### Tool Registries

```yaml
Tool_Registry:
  what: "Centralized catalog of all available tools across the organization"
  
  why:
    - "Discover tools: what's available for my agent?"
    - "Reuse tools: don't rebuild what exists"
    - "Governance: control which agents access which tools"
    - "Versioning: update tools without breaking existing agents"
    
  structure:
    per_tool:
      metadata:
        - "Name, description, version"
        - "Owner team"
        - "SLA (availability, latency)"
        - "Access control (which agents/teams can use)"
      definition:
        - "Input schema (parameters)"
        - "Output schema (return format)"
        - "Error types and codes"
      operational:
        - "Rate limits"
        - "Cost per call (if any)"
        - "Dependencies (other services)"
        - "Health status"
        
  patterns:
    dynamic_tool_loading:
      what: "Agent discovers available tools at runtime (not hard-coded)"
      benefit: "New tools automatically available without agent code changes"
      implementation: "Agent queries registry → gets tool definitions → includes in LLM context"
      
    tool_versioning:
      what: "Multiple versions of a tool coexist"
      approach: "v1 (stable, widely used) + v2 (new, being tested)"
      migration: "Gradual migration from v1 to v2 with compatibility period"
      
    context_aware_tool_selection:
      what: "Limit which tools are shown to LLM based on context"
      reasoning: "50 tools in context = poor selection. 5 relevant tools = great selection."
      implementation: "Pre-filter tools based on task category before passing to LLM"
```

---

## How It Works in Practice

### Production Tool Use Operations

```yaml
Operations:
  tool_lifecycle:
    development:
      - "Define tool interface (schema, description)"
      - "Implement with proper error handling"
      - "Add to sandbox test environment"
      - "Test with agent (does it select correctly? are parameters right?)"
      
    deployment:
      - "Register in tool registry"
      - "Set permissions (which agents can use)"
      - "Configure rate limits and monitoring"
      - "Deploy MCP server (if using MCP)"
      
    monitoring:
      - "Call success/failure rate per tool"
      - "Average latency per tool"
      - "Selection accuracy (was the right tool chosen?)"
      - "Parameter validity rate (did LLM provide correct args?)"
      
    maintenance:
      - "Update descriptions if LLM misuses tool"
      - "Add parameter validation for common errors"
      - "Version upgrades with backward compatibility"
      - "Deprecation with migration guide"
      
  common_tool_categories:
    data_access: "Database queries, file reading, API lookups"
    computation: "Code execution, math, data transformation"
    communication: "Email, Slack, notifications"
    external_services: "Web search, weather, stock prices"
    internal_systems: "CRM, ticketing, HR systems, deployment"
```

---

## Interview Tip

> When asked about tool use patterns: "I architect tool use around three pillars: (1) Tool design — clear names (verb_noun), comprehensive descriptions (WHEN to use, when NOT to, what it returns, limitations). The description is the most critical part because the LLM selects tools based solely on the description. I include negative examples ('Do NOT use for X') to prevent misuse. (2) Security — defense in depth: permission system (per-agent allowlists), execution sandbox (Docker containers for code, parameterized queries for SQL), human-in-the-loop for destructive actions, full audit trail. I never trust the LLM's judgment for irreversible operations. (3) MCP integration — I use Model Context Protocol for standardized tool integration. MCP servers expose tools via a standard protocol, any MCP client (Claude, Copilot, custom agents) can discover and use them. This means I build a tool ONCE and it works with ANY agent framework. Operationally: I maintain a tool registry (versioned, access-controlled, monitored), pre-filter tools by context (don't show 50 tools — show the 5-10 relevant ones), and monitor selection accuracy (is the LLM picking the right tool? if not, improve the description)."

---

## Common Mistakes

1. **Vague tool descriptions** — "Searches for things" as a tool description. LLM can't distinguish this from 5 other search tools and picks randomly. Solution: specific descriptions with clear use cases, examples, and counter-examples. The description is your tool's documentation for the LLM.

2. **No input validation** — Agent sends `{"age": "twenty-five"}` to a tool expecting an integer. Tool crashes with an unhandled error. Agent can't recover. Solution: validate all tool inputs against the schema. Return helpful errors: "Expected integer for 'age', got string 'twenty-five'. Please provide a numeric value."

3. **Giving agents destructive tools without safeguards** — Agent has access to `delete_user_account` tool. User says "remove my trial account" but agent deletes their production account. Solution: human-in-the-loop for all destructive operations. Agent proposes the action, human confirms before execution.

4. **Too many tools in context** — Loading 100 tools into every LLM call. Model can't distinguish between similar tools, selection accuracy drops, and context is wasted. Solution: pre-filter tools by task category (show 5-10 relevant tools), use dynamic tool loading based on conversation context.

5. **No tool error recovery** — Tool fails (network timeout, rate limit, invalid response). Agent doesn't know what to do — it either crashes or retries infinitely. Solution: tools return structured errors with retry guidance ("transient error, retry in 5s" vs "permission denied, don't retry"). Agent logic handles each error type.

---

## Key Takeaways

- Tool descriptions are the most critical element — LLM selects tools based on description quality
- Function calling: model outputs structured JSON (tool name + parameters), application executes
- MCP: universal standard for tool integration (build once, works with any agent framework)
- Security layers: permissions (allowlist), sandboxing (containers), human-in-the-loop (approvals), audit trail
- Pre-filter tools by context: 5-10 relevant tools > 50 all tools (better selection accuracy)
- Error handling: structured errors with retry guidance (transient vs permanent, fix suggestions)
- Tool registry: centralized catalog with versioning, access control, monitoring, and SLAs
- Sandboxing: code execution in containers, SQL via parameterized queries, no raw system access
- Human-in-the-loop: mandatory for destructive/financial/communication actions
- Monitor: selection accuracy, parameter validity, call success rate, latency per tool
