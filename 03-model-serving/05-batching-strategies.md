# Batching Strategies for Inference

## The Problem / Why This Matters

A GPU is a massively parallel processor — it has thousands of cores designed to execute the same operation on many data elements simultaneously. Serving one request at a time (batch size = 1) uses perhaps 5-10% of a GPU's compute capacity while still paying for 100% of the hardware. Batching — combining multiple requests into a single forward pass — is the single most important technique for maximizing GPU utilization and reducing cost per inference. For traditional ML models, batching is straightforward (collect N requests, run together). For LLMs (Large Language Models), batching is fundamentally harder because requests have different input lengths, generate different numbers of output tokens, and arrive at different times. This led to innovations like dynamic batching, continuous batching (also called in-flight batching or iteration-level scheduling), and most recently prefix-aware batching. In 2026, continuous batching is the standard for LLM serving — it provides 2-5x throughput improvement over static batching with the same hardware. Understanding batching strategies is critical because they determine whether your GPU runs at 10% or 90% utilization.

---

## The Analogy

Think of batching like different restaurant service models:

- **No batching (batch=1)** = A chef who cooks one dish at a time. The oven, all burners, prep space — all dedicated to one order. The kitchen runs at 5% capacity while 30 orders wait. Insanely wasteful.
- **Static batching** = The chef waits for 8 orders, then cooks them all together. Efficient use of the kitchen, but: (1) early orders wait for the batch to fill, (2) if one dish takes 20 minutes and others take 5, the fast dishes sit getting cold until the slow one finishes. The whole batch is released together.
- **Dynamic batching** = Same as static, but with a timer. Wait up to 100ms OR until 8 orders arrive (whichever comes first). Reduces latency for the first order in the batch.
- **Continuous batching** = A sushi conveyor belt. As soon as one dish is plated (request completes), a new order immediately takes its slot. The belt (GPU) never runs empty. No request waits for others to finish. The kitchen runs at near-maximum capacity continuously.

---

## Deep Dive

### Static Batching (Naive Approach)

```yaml
Static_Batching:
  what: "Collect N requests, pad to max length, process as one batch, return all results together"
  
  how_it_works:
    1: "Wait for batch to fill (N requests collected)"
    2: "Pad all inputs to the longest input in the batch"
    3: "Run forward pass for all inputs simultaneously"
    4: "For generation: run decode loop until ALL sequences hit stop token or max length"
    5: "Return all results at once"
    
  problems:
    padding_waste: |
      If inputs are [10 tokens, 50 tokens, 200 tokens], all are padded to 200.
      The 10-token input wastes 190 positions of compute. At batch size 32,
      this can mean 80%+ wasted computation.
      
    wait_for_slowest: |
      In generation: one request might finish after 20 tokens, another after 500.
      The short request occupies its batch slot (GPU memory, compute) even after it's done.
      Can't send the result back until the entire batch completes.
      
    latency_vs_throughput: |
      Larger batch = higher throughput but higher latency (wait to fill + wait for slowest).
      Small batch = lower latency but GPU underutilized.
      Impossible to optimize both simultaneously with static batching.
      
  when_acceptable:
    - "Offline batch processing (process 10K inputs, don't care about latency)"
    - "All inputs are similar length (embedding models with fixed chunk size)"
    - "Non-generative models (single forward pass, no decode loop)"
```

### Dynamic Batching

```yaml
Dynamic_Batching:
  what: "Like static batching but with timeout — don't wait indefinitely for batch to fill"
  
  parameters:
    max_batch_size: "Maximum requests per batch (e.g., 32)"
    max_queue_delay: "Maximum time to wait for batch to fill (e.g., 100ms)"
    preferred_batch_sizes: "Optimal sizes for hardware (e.g., [8, 16, 32])"
    
  how_it_works:
    1: "Request arrives → added to queue"
    2: "If queue reaches max_batch_size → immediately dispatch batch"
    3: "If max_queue_delay expires → dispatch whatever is in queue (even if < max_batch_size)"
    4: "Process batch, return results"
    
  triton_config:
    example: |
      dynamic_batching {
        preferred_batch_size: [8, 16, 32, 64]
        max_queue_delay_microseconds: 100000  # 100ms max wait
        preserve_ordering: true
        default_queue_policy {
          timeout_action: DELAY
          default_timeout_microseconds: 200000
          max_queue_size: 256
        }
      }
      
  improvement_over_static:
    - "Bounded latency (max_queue_delay caps wait time)"
    - "Better GPU utilization at variable traffic (small batches at low traffic, large at high)"
    - "Still has padding waste and wait-for-slowest problems for generation"
    
  best_for:
    - "Traditional ML models (classification, embedding, ranking)"
    - "Fixed-output models (no autoregressive generation)"
    - "NVIDIA Triton Inference Server default mode"
```

### Continuous Batching (Iteration-Level Scheduling)

```yaml
Continuous_Batching:
  what: "Don't wait for entire batch to complete — release finished requests and add new ones at each decode step"
  also_called: ["In-flight batching", "Iteration-level scheduling", "Inflight batching"]
  
  key_insight: |
    In autoregressive generation, the decode step processes ONE token per request per iteration.
    Instead of treating the batch as atomic, treat each iteration independently:
    - After each decode step, check: did any request finish (hit EOS or max tokens)?
    - If yes: remove it from the batch, return its result immediately
    - If there's a free slot: pull next waiting request from queue, start it
    - GPU is ALWAYS processing a full batch (or close to it)
    
  how_it_works:
    step_1: "Batch starts with requests [A, B, C, D] (batch size 4)"
    step_2: "After 10 iterations, request B finishes (hits end token)"
    step_3: "Request B's result is immediately returned to client"
    step_4: "Request E from queue takes B's slot in the batch"
    step_5: "Next iteration processes [A, E, C, D]"
    step_6: "Continue — GPU always has 4 active requests"
    
  benefits:
    throughput: "2-5x improvement over static batching"
    latency: "Short requests return immediately (don't wait for long ones)"
    utilization: "GPU stays saturated — minimal idle slots"
    fairness: "Every request gets consistent per-token latency"
    
  implementation:
    vllm: "Built-in (default behavior)"
    tgi: "Built-in (default behavior)"
    tensorrt_llm: "In-flight batching feature"
    triton: "Supported via TensorRT-LLM backend"
    
  comparison:
    static_batch_8:
      scenario: "8 requests, lengths vary from 50 to 500 tokens"
      total_time: "Time for longest request (500 tokens) × batch overhead"
      throughput: "~500 tokens total / 500 token time = low tokens/sec"
      problem: "7 short requests wait for the 1 long request"
      
    continuous_batch_8:
      scenario: "Same 8 requests"
      behavior: "Short requests finish and leave, new requests enter"
      throughput: "GPU always has 8 active requests → ~8× more tokens generated per second"
      latency: "50-token request returns after ~50 decode steps, not 500"
```

### Prefill and Decode Phases

```yaml
Prefill_vs_Decode:
  key_distinction: |
    LLM inference has TWO phases with VERY different compute characteristics:
    1. Prefill (prompt processing): Process all input tokens in parallel — COMPUTE BOUND
    2. Decode (token generation): Generate one token at a time — MEMORY BANDWIDTH BOUND
    
  prefill:
    what: "Process the entire input prompt to build KV cache"
    compute: "Matrix-matrix multiply (parallel over all input tokens)"
    characteristic: "Compute-bound — GPU cores are fully utilized"
    duration: "Short burst of high compute (proportional to input length)"
    
  decode:
    what: "Generate output tokens one at a time, autoregressively"
    compute: "Matrix-vector multiply (one token at a time per request)"
    characteristic: "Memory-bandwidth bound — reading model weights for single token"
    duration: "Long sequential process (proportional to output length)"
    
  batching_interaction:
    problem: |
      Prefill (compute-heavy) and decode (bandwidth-heavy) compete for GPU resources.
      If you prefill a new request while others are decoding, the prefill monopolizes
      compute and increases latency for decoding requests.
      
    solutions:
      chunked_prefill:
        what: "Split long prefills into chunks, interleave with decode steps"
        benefit: "Decode latency stays consistent even when new long prompts arrive"
        implementation: "vLLM chunked prefill (default in recent versions)"
        
      splitwise_disaggregation:
        what: "Separate prefill and decode to different GPUs"
        benefit: "Prefill GPU optimized for compute, decode GPU optimized for bandwidth"
        trade_off: "Need to transfer KV cache between GPUs (network bandwidth)"
        status: "Research/early production (2026)"
```

### Advanced Batching Techniques

```python
# Conceptual illustration of continuous batching logic

class ContinuousBatchScheduler:
    def __init__(self, max_batch_size: int, max_tokens: int):
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens  # Max total tokens in KV cache
        self.running_batch: list[Request] = []
        self.waiting_queue: list[Request] = []
        
    def schedule_step(self) -> list[Request]:
        """Called before each decode iteration."""
        
        # 1. Remove finished requests
        finished = [r for r in self.running_batch if r.is_finished()]
        for req in finished:
            self.running_batch.remove(req)
            req.send_response()  # Return result immediately
            
        # 2. Check if we can preempt (out of memory for KV cache)
        while self._kv_cache_memory() > self.max_tokens:
            # Preempt lowest priority request (swap KV cache to CPU)
            victim = self._select_preemption_victim()
            self.running_batch.remove(victim)
            self.waiting_queue.insert(0, victim)  # Re-add to front of queue
            
        # 3. Add new requests from queue (if slots available)
        while (len(self.running_batch) < self.max_batch_size 
               and self.waiting_queue
               and self._can_fit_new_request()):
            new_req = self.waiting_queue.pop(0)
            new_req.start_prefill()
            self.running_batch.append(new_req)
            
        return self.running_batch
        
    def _kv_cache_memory(self) -> int:
        """Total KV cache tokens across all running requests."""
        return sum(r.total_tokens() for r in self.running_batch)
        
    def _can_fit_new_request(self) -> bool:
        """Check if GPU memory can accommodate another request's KV cache."""
        next_req = self.waiting_queue[0]
        estimated_tokens = next_req.input_length + next_req.estimated_output
        return self._kv_cache_memory() + estimated_tokens <= self.max_tokens
```

### Prefix-Aware Batching

```yaml
Prefix_Caching:
  what: "Cache and share KV cache entries for common prefixes across requests"
  
  example:
    system_prompt: "You are a helpful assistant. You answer questions concisely..."
    observation: "100 concurrent requests all have the same system prompt (200 tokens)"
    without_caching: "Compute KV for system prompt 100 times = 100 × 200 = 20,000 redundant token computations"
    with_caching: "Compute KV for system prompt ONCE, share across all 100 requests"
    savings: "99% reduction in prefill compute for shared prefix"
    
  implementation:
    vllm_automatic_prefix_caching:
      what: "vLLM automatically detects and caches common prefixes"
      config: "--enable-prefix-caching (enabled by default in recent versions)"
      benefit: "Transparent to user — works with any prompt structure"
      
    radix_tree:
      what: "Tree structure that stores cached KV blocks keyed by token sequences"
      how: "Multiple requests sharing 'You are a helpful...' find cached KV in tree"
      eviction: "LRU (Least Recently Used) when memory pressure occurs"
      
  use_cases:
    - "Chatbots with system prompts (all users share same system prompt)"
    - "RAG (Retrieval-Augmented Generation) with shared document context"
    - "Few-shot prompting (shared examples across requests)"
    - "Multi-turn conversations (previous turns already cached)"
```

---

## How It Works in Practice

### Choosing the Right Batching Strategy

```yaml
Selection_Guide:
  traditional_ml_models:
    type: "Dynamic batching"
    reason: "Single forward pass, fixed output — simple and effective"
    tool: "Triton Inference Server dynamic_batching config"
    
  llm_serving_production:
    type: "Continuous batching"
    reason: "2-5x throughput, immediate result delivery for short requests"
    tool: "vLLM or TGI (built-in, default behavior)"
    
  llm_with_shared_prompts:
    type: "Continuous batching + prefix caching"
    reason: "Avoid recomputing KV for shared system prompts"
    tool: "vLLM --enable-prefix-caching"
    
  offline_batch_processing:
    type: "Static batching with large batch size"
    reason: "Don't care about latency, maximize throughput"
    tool: "vLLM offline batch mode or custom pipeline"
    
  mixed_prefill_decode:
    type: "Continuous batching with chunked prefill"
    reason: "Keep decode latency consistent when new long prompts arrive"
    tool: "vLLM chunked prefill (default)"
```

---

## Interview Tip

> When asked about batching for LLM inference: "The key challenge with LLM batching is that generation is iterative — each token depends on the previous one. Static batching groups N requests and waits for ALL to finish, which wastes GPU on completed requests and delays short outputs. Continuous batching (used by vLLM, TGI) schedules at the iteration level: after each decode step, finished requests leave and new requests enter. The GPU always processes a full batch, giving 2-5x throughput improvement. Two additional insights: (1) LLM inference has two phases — prefill is compute-bound (parallel token processing) and decode is memory-bandwidth-bound (one token at a time). Chunked prefill interleaves these to prevent long prompts from spiking decode latency. (2) Prefix caching shares KV computation for common system prompts across requests — if 100 users have the same system prompt, compute it once, share the KV cache. Together, continuous batching + prefix caching + chunked prefill constitute the modern LLM serving stack."

---

## Common Mistakes

1. **Static batching for LLM generation** — Using batch_size=32 and waiting for all 32 to generate max_tokens. The shortest request (50 tokens) waits for the longest (500 tokens) to finish. Results in 10x higher perceived latency for fast requests and massive GPU waste.

2. **Setting batch size too large without memory accounting** — Each request in a batch holds KV cache memory (e.g., 20GB per 8K-context request for 70B model). Batch size 8 × 20GB = 160GB KV cache needed — exceeds H100's 80GB. The system OOMs or starts swapping to CPU (catastrophic latency).

3. **Not enabling prefix caching for chatbot workloads** — A chatbot system prompt is the same for every request. Without prefix caching, every request recomputes the system prompt's KV cache (100-500 tokens × 1000 requests/minute = massive waste). One config flag eliminates this entirely.

4. **Ignoring prefill-decode interference** — Adding a new request with a 10K token prompt to a batch of decoding requests. The long prefill hogs compute for 2-3 seconds, spiking decode latency for all other requests (their tokens are delayed). Solution: chunked prefill limits prefill to small chunks interleaved with decode.

5. **Using max_queue_delay = 0 (no batching)** — Setting queue delay to 0 for "minimum latency" means every request runs alone (batch size 1). GPU utilization drops to 5-10%. Even 10-50ms queue delay dramatically improves throughput with negligible latency impact.

---

## Key Takeaways

- Static batching: groups requests, pads, waits for all — only acceptable for offline batch processing
- Dynamic batching: configurable timeout + max batch size — good for traditional ML (classification, embedding)
- Continuous batching: iteration-level scheduling — 2-5x throughput for LLM generation (vLLM/TGI default)
- LLM inference has two phases: prefill (compute-bound, parallel) and decode (bandwidth-bound, sequential)
- Chunked prefill: prevents long prompts from spiking decode latency for other requests
- Prefix caching: share KV cache for common prefixes (system prompts) — eliminates redundant computation
- Batch size is limited by KV cache memory, not just GPU compute
- Even small queue delays (10-50ms) dramatically improve throughput with minimal latency impact
- GPU utilization: batch=1 ≈ 5-10%, continuous batching ≈ 70-90%
- All modern LLM serving frameworks (vLLM, TGI, TensorRT-LLM) use continuous batching by default
