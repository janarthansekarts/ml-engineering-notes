# Agent Deployment

## The Problem / Why This Matters

Building an agent that works locally is straightforward — getting it to run reliably in production at scale is a completely different engineering challenge. Agents have unique deployment requirements that traditional web services don't: long-running executions (minutes to hours, not milliseconds), unpredictable resource consumption (an agent might make 3 LLM calls or 50), stateful workflows (must resume after failures), real-time streaming (users need intermediate progress), and expensive compute (each request costs $0.10-$10+ in LLM API fees). You can't just "deploy an agent" the way you deploy a REST API (REpresentational State Transfer Application Programming Interface). You need: execution infrastructure (where does the agent loop run?), state management (how do you persist agent state across failures?), scaling (how do you handle 1,000 concurrent agents without blowing your LLM budget?), observability (how do you monitor 50-step workflows?), and user experience (how do you keep users informed during 30-second-long executions?). Production agent deployment is where most teams fail — the agent works in notebooks but crashes, hangs, or bankrupts you in production.

---

## The Analogy

Think of deploying agents like running a consulting firm:

- **Traditional API deployment** = Vending machine. Customer presses button, instant response, done. Stateless, fast, predictable cost.
- **Agent deployment** = Hiring a consultant. They work for hours/days, need their workspace (state), might call other consultants (tool calls), have unpredictable costs (hourly billing), and you need progress updates (streaming). You can't just stack them in a vending machine.

Key differences:
- **Duration**: Vending machine = milliseconds. Consultant = hours/days.
- **Cost**: Vending machine = fixed cost. Consultant = variable hourly rate.
- **State**: Vending machine = stateless. Consultant = needs desk, files, context.
- **Scaling**: 1000 vending machines = linear cost. 1000 consultants = need office space, coordination, budget limits.
- **Failure handling**: Vending machine stuck? Hit it (retry). Consultant sick mid-project? Need to hand off (checkpoint + resume).

---

## Deep Dive

### Deployment Architectures

```yaml
Architecture_Options:
  synchronous_request_response:
    what: "Agent runs within an HTTP request/response cycle"
    when: "Simple agents that complete in <30 seconds"
    pros:
      - "Simple architecture (standard web server)"
      - "Easy to scale (stateless, any instance handles any request)"
      - "Standard tooling (load balancers, auto-scaling)"
    cons:
      - "Timeout limits (most load balancers: 30-60s)"
      - "No resume on failure (entire request lost)"
      - "No streaming (response comes all at once)"
    suitable_for: "Simple Q&A agents, single-tool-call agents"
    
  async_with_webhooks:
    what: "Accept request → return job ID → execute async → callback when done"
    architecture:
      1: "Client submits task → API returns job_id"
      2: "Worker picks up task from queue"
      3: "Worker runs agent loop (no timeout pressure)"
      4: "Worker updates status (DB or Redis)"
      5: "Worker calls webhook on completion (or client polls)"
    when: "Agents that take 30s-30min, no real-time streaming needed"
    pros:
      - "No timeout issues"
      - "Can retry/resume on failure"
      - "Scalable (add more workers)"
    cons:
      - "More complex infrastructure"
      - "No real-time progress to user"
      - "Client must poll or receive callbacks"
      
  streaming_execution:
    what: "SSE (Server-Sent Events) or WebSocket connection for real-time updates"
    architecture:
      1: "Client opens SSE/WebSocket connection"
      2: "Server runs agent loop, streams events in real-time"
      3: "Events: thinking, tool_call, tool_result, token, done"
      4: "Client renders progress as events arrive"
    when: "Interactive agents with human-facing UI"
    pros:
      - "Real-time feedback (users see agent thinking)"
      - "Can display intermediate results"
      - "Good UX (no waiting in the dark)"
    cons:
      - "Connection management (reconnection on drop)"
      - "Stateful connections (harder to load balance)"
      - "Long-lived connections consume server resources"
    standard: "SSE preferred over WebSocket (simpler, HTTP-native, auto-reconnect)"
    
  managed_platforms:
    what: "Deploy on LangGraph Platform, AWS Bedrock Agents, or similar"
    when: "Teams wanting to avoid infrastructure complexity"
    options:
      langgraph_platform:
        features: "Persistence, streaming, cron, human-in-the-loop"
        deployment: "LangGraph Cloud or self-hosted"
      aws_bedrock_agents:
        features: "Managed agent runtime with AWS tool integrations"
        deployment: "Fully managed by AWS"
      azure_ai_agent_service:
        features: "Managed agent runtime with Azure integrations"
        deployment: "Fully managed by Azure"
```

### State Management

```yaml
State_Management:
  why_critical: "Agents are stateful workflows. Without state persistence:"
    - "Failure at step 45 of 50 → restart from scratch ($$$)"
    - "Human-in-the-loop impossible (can't pause and resume)"
    - "No audit trail (what did the agent do?)"
    - "Server restart loses everything in-flight"
    
  what_to_persist:
    conversation_state:
      what: "Full message history (user + agent + tool results)"
      storage: "PostgreSQL (structured) or Redis (fast access)"
      size: "Can grow large (100s of messages for long workflows)"
      
    execution_state:
      what: "Where in the workflow the agent is (current step, pending decisions)"
      storage: "Same as conversation state (atomic updates)"
      includes:
        - "Current node in graph (for LangGraph)"
        - "Pending tool calls"
        - "Accumulated results"
        - "Decision history"
        
    tool_results_cache:
      what: "Results of previous tool calls (avoid re-execution on resume)"
      storage: "Keyed by (tool_name + args hash)"
      benefit: "Idempotent resume (don't re-run expensive tools)"
      
    memory:
      what: "Long-term agent memory (facts learned, user preferences)"
      storage: "Vector database (similarity search) + structured DB (facts)"
      
  checkpointing_strategy:
    when_to_checkpoint:
      - "After every tool call (most common)"
      - "After every LLM response (more granular)"
      - "After human input received"
      - "Before expensive operations (so you can resume before them)"
    implementation: "LangGraph PostgresSaver checkpoints after every node execution"
```

### Scaling Agents

```python
# Agent scaling architecture

"""
Key scaling challenges:
1. Each agent execution is expensive (LLM API calls)
2. Execution time varies wildly (1s - 30min)
3. Memory usage grows with conversation length
4. Concurrent agents compete for rate limits

Solutions:
"""

# 1. Worker pool with queue-based scaling
"""
Architecture:
  Queue (Redis/SQS) → Worker Pool → LLM APIs
  
  - Queue absorbs burst traffic
  - Workers process at sustainable rate
  - Auto-scale workers based on queue depth
  - Each worker handles one agent at a time (avoid memory pressure)
"""

# Worker configuration
WORKER_CONFIG = {
    "max_concurrent_agents_per_worker": 5,  # Limited by memory
    "max_agent_duration_seconds": 1800,      # 30 min hard limit
    "heartbeat_interval_seconds": 30,        # Health check
    "graceful_shutdown_timeout": 60,          # Finish current step
}

# 2. Rate limit management
class RateLimitManager:
    """Manage LLM API rate limits across all agents."""
    
    def __init__(self, limits: dict):
        self.limits = limits  # {"openai": 10000 TPM, "anthropic": 100000 TPM}
        self.usage = {}       # Current usage tracking
    
    async def acquire_tokens(self, provider: str, tokens_needed: int):
        """Wait until rate limit allows this request."""
        while self.usage.get(provider, 0) + tokens_needed > self.limits[provider]:
            await asyncio.sleep(0.1)  # Backpressure
        self.usage[provider] = self.usage.get(provider, 0) + tokens_needed
    
    def release_tokens(self, provider: str, tokens_used: int):
        """Return tokens to the budget (called after response)."""
        self.usage[provider] = max(0, self.usage.get(provider, 0) - tokens_used)


# 3. Cost controls
class CostController:
    """Prevent runaway agent costs."""
    
    def __init__(self, budget_per_task: float = 5.0, budget_per_user_daily: float = 50.0):
        self.budget_per_task = budget_per_task
        self.budget_per_user_daily = budget_per_user_daily
    
    async def check_budget(self, task_id: str, user_id: str, estimated_cost: float):
        """Raise if budget would be exceeded."""
        task_spent = await self._get_task_cost(task_id)
        user_spent = await self._get_user_daily_cost(user_id)
        
        if task_spent + estimated_cost > self.budget_per_task:
            raise BudgetExceeded(f"Task budget exceeded: ${task_spent:.2f} + ${estimated_cost:.2f} > ${self.budget_per_task}")
        
        if user_spent + estimated_cost > self.budget_per_user_daily:
            raise BudgetExceeded(f"Daily user budget exceeded")


# 4. Scaling triggers
AUTOSCALING_CONFIG = {
    "min_workers": 2,
    "max_workers": 50,
    "scale_up_threshold": "queue_depth > 10 for 60s",
    "scale_down_threshold": "queue_depth == 0 for 300s",
    "scale_up_step": 5,
    "scale_down_step": 2,
    "cooldown_seconds": 120,
}
```

### Deployment Configuration

```yaml
Deployment_Infrastructure:
  kubernetes_deployment:
    agent_workers:
      replicas: "2-50 (auto-scaled)"
      resources:
        requests:
          memory: "1Gi"  # Agents accumulate state in memory
          cpu: "500m"
        limits:
          memory: "4Gi"  # Prevent OOM from runaway agents
          cpu: "2000m"
      health_checks:
        liveness: "Heartbeat every 30s"
        readiness: "Worker ready to accept new tasks"
      graceful_shutdown: "Checkpoint current agent, re-queue task"
      
    api_gateway:
      purpose: "Accept requests, validate auth, enqueue tasks"
      replicas: "3 (always on)"
      timeout: "10s (just enqueue, don't wait for agent)"
      
    state_store:
      type: "PostgreSQL 16"
      purpose: "Agent checkpoints, conversation history"
      storage: "100GB+ (grows with agent usage)"
      backup: "Every 6 hours"
      
    queue:
      type: "Redis 7+ or AWS SQS"
      purpose: "Task queue, rate limiting, pub/sub for events"
      
    streaming_service:
      type: "SSE endpoint or WebSocket service"
      purpose: "Real-time agent updates to clients"
      connection: "Long-lived (timeout 30 min)"
      
  docker_compose_local:
    services:
      api: "FastAPI gateway"
      worker: "Agent execution workers"
      postgres: "State storage"
      redis: "Queue + rate limiting"
      
  serverless:
    challenges:
      - "Execution time limits (AWS Lambda: 15 min max)"
      - "Cold starts (agent initialization is heavy)"
      - "No persistent connections (streaming difficult)"
      - "Memory limits (agent state grows)"
    workaround: "Step Functions (AWS) or Durable Functions (Azure) for orchestration"
```

### Production Hardening

```yaml
Production_Hardening:
  timeouts_and_limits:
    max_execution_time: "30 minutes (configurable per task type)"
    max_llm_calls: "50 per task"
    max_tool_calls: "100 per task"
    max_tokens_per_task: "500,000"
    max_output_size: "100KB"
    
  error_handling:
    llm_errors:
      rate_limit: "Exponential backoff + jitter (2s, 4s, 8s, 16s, max 60s)"
      timeout: "Retry up to 3 times, then fail gracefully"
      invalid_response: "Re-prompt with error context"
      
    tool_errors:
      timeout: "Tool-specific timeouts (API call: 30s, code execution: 60s)"
      failure: "Agent receives error message, decides how to proceed"
      repeated_failure: "After 3 failures, skip tool and inform user"
      
    infrastructure_errors:
      worker_crash: "Checkpoint ensures task resumes from last state"
      queue_full: "Return 429 to client with retry-after header"
      state_store_down: "Graceful degradation (run without persistence, log warning)"
      
  observability:
    logging:
      - "Every LLM call (input, output, latency, tokens, cost)"
      - "Every tool call (name, args, result, success, latency)"
      - "State transitions (step N → step N+1)"
      - "Errors and retries"
      
    metrics:
      - "agent_task_duration_seconds (histogram)"
      - "agent_llm_calls_total (counter, by model)"
      - "agent_tool_calls_total (counter, by tool)"
      - "agent_cost_usd (counter, by task_type)"
      - "agent_errors_total (counter, by error_type)"
      - "agent_queue_depth (gauge)"
      - "agent_active_executions (gauge)"
      
    tracing:
      tool: "OpenTelemetry"
      spans:
        - "Task (root span)"
        - "  +-- Agent Loop Iteration"
        - "      +-- LLM Call"
        - "      +-- Tool Execution"
```

---

## How It Works in Practice

### Full Deployment Example

```python
# FastAPI-based agent deployment

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

@app.post("/tasks")
async def submit_task(request: TaskRequest):
    """Submit a task for agent execution."""
    
    # Validate and create task
    task_id = str(uuid.uuid4())
    await state_store.create_task(task_id, request)
    
    # Enqueue for worker processing
    await task_queue.enqueue(task_id)
    
    return {"task_id": task_id, "status": "queued", "stream_url": f"/tasks/{task_id}/stream"}


@app.get("/tasks/{task_id}/stream")
async def stream_task(task_id: str):
    """Stream real-time agent execution updates."""
    
    async def event_generator():
        async for event in event_bus.subscribe(task_id):
            yield f"data: {json.dumps(event)}\n\n"
            
            if event["type"] == "done":
                break
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get current task status and result."""
    task = await state_store.get_task(task_id)
    return {
        "task_id": task_id,
        "status": task.status,
        "result": task.result if task.status == "completed" else None,
        "progress": task.progress,
        "cost_usd": task.total_cost,
    }
```

---

## Interview Tip

> When asked about agent deployment: "I deploy agents as asynchronous worker services, not synchronous APIs. The architecture: API gateway accepts tasks → enqueues to Redis/SQS → worker pool processes → streams progress via SSE (Server-Sent Events). Key design decisions: (1) State persistence — I checkpoint after every tool call using PostgreSQL. This enables resume on failure (agent crashes at step 45 → resumes from step 45, not restart) and human-in-the-loop (pause, wait for approval, resume). (2) Scaling — worker auto-scaling based on queue depth, with rate limit management across all agents sharing the same LLM API keys. (3) Cost controls — hard budget per task ($5 default), per user daily ($50), with automatic termination and user notification when exceeded. (4) Timeouts — 30-minute max execution, 50 LLM calls max, exponential backoff on rate limits. (5) Observability — OpenTelemetry traces per task showing every LLM call and tool execution, with metrics on cost, latency, completion rate. I do NOT deploy agents as Lambda functions — execution time limits (15 min) and cold starts make them unsuitable. For production, Kubernetes workers with Redis queues give the best control. LangGraph Platform is a good managed alternative if you want less infrastructure overhead."

---

## Common Mistakes

1. **Synchronous HTTP for long agents** — Deploying agent behind a standard API gateway with 30-second timeout. Complex task hits timeout → client gets 504 Gateway Timeout → all work lost. Solution: async architecture (enqueue → process → callback/stream). Only use synchronous for agents guaranteed to complete in <15 seconds.

2. **No checkpointing** — Agent runs for 10 minutes, server restarts at minute 9 → entire execution lost, user re-triggers, costs double. Solution: checkpoint after every tool call. LangGraph's PostgresSaver does this automatically. Cost of checkpointing (1 DB write per step) is negligible vs cost of re-running failed agents.

3. **No cost limits** — Agent enters a loop (tool failure → retry → failure → retry) or user submits an adversarial task that causes 200 LLM calls. Bill: $50 for one task. Solution: hard limits per task (max LLM calls, max tokens, max cost), automatic termination when exceeded, alert to operators.

4. **Ignoring rate limits** — 100 agents running concurrently, all hitting OpenAI API → rate limited → cascading retries → thundering herd → even more rate limiting. Solution: centralized rate limit manager with token bucket, backpressure (slow down task acceptance when near limits), multiple providers for load distribution.

5. **No graceful shutdown** — Deploy new version → Kubernetes kills old pods → in-flight agents killed mid-execution → tasks lost. Solution: graceful shutdown with preStop hook — checkpoint current state, re-enqueue task, then terminate. New worker picks up exactly where old one stopped.

---

## Key Takeaways

- Deploy agents as async workers (not synchronous APIs) — execution time is unpredictable
- Checkpoint after every tool call: enables failure recovery, human-in-the-loop, audit trails
- Stream progress via SSE (Server-Sent Events) for real-time user feedback
- Cost controls are mandatory: budget per task, per user, per day — auto-terminate on exceed
- Rate limit management must be centralized across all concurrent agents
- Auto-scale workers based on queue depth, not CPU (agents are I/O-bound, not CPU-bound)
- Graceful shutdown: checkpoint + re-queue before pod termination
- Observability: trace every LLM call and tool execution, track cost per task
- Serverless (Lambda) doesn't work well for agents — use Kubernetes workers or managed platforms
- Timeouts at every level: overall task (30 min), LLM call (60s), tool call (30s)
