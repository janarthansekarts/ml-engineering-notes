# Model Caching and KV Cache Management

## The Problem / Why This Matters

Loading a large model from disk into GPU memory takes 30-120 seconds for a 70B parameter model. During this time, no inference can happen — it's pure downtime. Worse, in autoregressive LLM generation, every token generated requires access to the Key-Value (KV) cache — a growing data structure that stores attention computations for all previous tokens. For a 70B model with 8K context, the KV cache for a single request is ~20GB. Managing this memory efficiently is the difference between serving 2 concurrent users and serving 50 concurrent users on the same GPU. The KV cache problem has three dimensions: (1) Model loading caching — keeping warm model copies ready to serve, (2) KV cache memory management — efficiently allocating/deallocating attention state, and (3) Prefix caching — reusing computations across requests that share common prefixes. PagedAttention, introduced by vLLM, revolutionized dimension (2) by applying operating system virtual memory concepts to KV cache allocation, reducing memory waste from 60-80% to near zero. In 2026, KV cache management is the primary bottleneck and optimization target for LLM serving systems.

---

## The Analogy

Think of model and KV cache management like managing a restaurant's kitchen workspace:

- **Model loading** = Setting up the kitchen (installing ovens, prep stations, tools). Takes a long time, so you don't want to tear down and rebuild between meals. Keep the kitchen ready (warm pool).
- **KV cache** = The prep work (mise en place) for each order. Every dish needs its ingredients prepped and laid out. Each in-progress order occupies counter space.
- **Static KV allocation** = Reserving an entire counter for each order, even if the dish only needs half. The unused counter space can't be used by other orders. Massive waste.
- **PagedAttention** = Using modular counter tiles. Each order gets tiles as needed, tiles returned when done. A 3-tile order and a 7-tile order use exactly 10 tiles total — no wasted space. New orders can immediately use returned tiles.
- **Prefix caching** = Pre-prepping ingredients that multiple dishes share (garlic, sauces). Prep once, share across all orders that need them. Don't re-prep garlic for every single order.

---

## Deep Dive

### Model Loading and Warm Pools

```yaml
Model_Loading:
  problem: |
    Large model loading time:
    - 7B model (14 GB FP16): ~5-10 seconds from SSD, ~30s from S3/GCS
    - 70B model (140 GB FP16): ~30-60 seconds from NVMe SSD, ~5 min from S3/GCS
    - 405B model: ~2-5 minutes from fast storage
    During loading, the endpoint serves ZERO requests (downtime).
    
  solutions:
    warm_pools:
      what: "Keep model pre-loaded in GPU memory, ready to serve"
      implementation: "Min replicas > 0, pre-loaded model endpoints"
      cost: "Pay for idle GPU time, but zero cold start"
      use: "Production services with latency SLA (Service Level Agreement)"
      
    model_preloading:
      what: "Start loading model before traffic arrives (predictive scaling)"
      triggers: "Time-based (load before business hours), traffic-based (load at threshold)"
      
    model_sharding_across_disk:
      what: "Split model into shards, load in parallel from multiple disks"
      benefit: "4× disk read parallelism → 4× faster loading"
      implementation: "SafeTensors format with model.safetensors.index.json"
      
    memory_mapped_loading:
      what: "mmap model file — pages loaded on-demand from disk to RAM"
      benefit: "Instant 'load' (just maps file), pages fault in as needed"
      limitation: "Only works for CPU inference or RAM → GPU transfer"
      
    persistent_storage:
      what: "Keep model on local NVMe SSD attached to GPU instance (not remote S3)"
      benefit: "5-10× faster loading vs remote storage"
      implementation: "Persistent disk on GCP, EBS on AWS, mounted to inference instance"
      
  multi_model_scenarios:
    problem: "100 models, 8 GPUs — can't fit all models in GPU memory simultaneously"
    solution_lru_cache:
      what: "Keep most-recently-used models in GPU, evict least-used"
      implementation: "KServe ModelMesh, custom LRU (Least Recently Used) eviction"
      trade_off: "Evicted models have cold start on next request"
      
    solution_cpu_offload:
      what: "Keep model weights in CPU RAM, copy to GPU on-demand"
      benefit: "CPU RAM is 10-20× cheaper than GPU memory"
      latency: "1-5 seconds to transfer 7B model from CPU to GPU"
```

### KV Cache Fundamentals

```yaml
KV_Cache:
  what: |
    In transformer attention, each token attends to ALL previous tokens.
    Computing Key and Value projections for all previous tokens every step is expensive.
    KV cache stores the Key and Value matrices for all computed tokens —
    so each new token only computes its own K,V and attends to the cached history.
    
  memory_calculation:
    formula: |
      KV cache per token = 2 (K and V) × num_layers × num_kv_heads × head_dim × dtype_bytes
      
    examples:
      llama_4_8b:
        layers: 32
        kv_heads: 8  # GQA (Grouped Query Attention): 8 KV heads (not 32)
        head_dim: 128
        dtype: "FP16 (2 bytes)"
        per_token: "2 × 32 × 8 × 128 × 2 = 131 KB"
        context_8k: "131 KB × 8192 = 1.05 GB per request"
        
      llama_4_70b:
        layers: 80
        kv_heads: 8  # GQA
        head_dim: 128
        dtype: "FP16 (2 bytes)"
        per_token: "2 × 80 × 8 × 128 × 2 = 328 KB"
        context_8k: "328 KB × 8192 = 2.6 GB per request"
        context_32k: "328 KB × 32768 = 10.5 GB per request"
        
  memory_pressure:
    scenario: "Llama-70B INT4 on H100 80GB"
    model_memory: "~35 GB"
    available_for_kv: "~45 GB"
    requests_at_8k: "45 GB / 2.6 GB = 17 concurrent requests"
    requests_at_32k: "45 GB / 10.5 GB = 4 concurrent requests"
    conclusion: "KV cache, not model size, limits concurrency for long contexts"
```

### The Waste Problem (Before PagedAttention)

```yaml
KV_Cache_Waste_Problem:
  static_allocation:
    how_it_worked: |
      Before PagedAttention, KV cache was allocated as a CONTIGUOUS block per request.
      You must allocate for the MAXIMUM possible output length upfront
      (because you don't know how long the response will be).
      
    example:
      max_output_tokens: 2048
      actual_output: 150  # Most responses are short
      allocated: "2048 tokens worth of KV cache"
      used: "150 tokens worth of KV cache"
      wasted: "1898 tokens worth (93% waste for this request)"
      
    aggregate_waste: |
      Across a batch of 16 requests:
      - Average output: 200 tokens
      - Allocated per request: 2048 tokens
      - Total allocated: 16 × 2048 = 32,768 token-slots of KV memory
      - Total used: 16 × 200 = 3,200 token-slots
      - Waste: 90% of KV cache memory is allocated but unused
      
    consequences:
      - "Can only serve 16 concurrent requests instead of 50+ (memory reserved but empty)"
      - "Memory fragmentation over time (allocated blocks can't be reused)"
      - "Must pre-declare max_tokens even if most requests are short"
      
  fragmentation:
    what: "Even when memory is freed, leftover gaps between allocated blocks can't fit new allocations"
    result: "Total free memory is 30GB but largest contiguous block is 5GB — can't allocate a 10GB KV cache"
    similar_to: "Disk fragmentation in file systems"
```

### PagedAttention (vLLM's Innovation)

```yaml
PagedAttention:
  what: "Manage KV cache like an operating system manages virtual memory — using pages"
  paper: "Efficient Memory Management for Large Language Model Serving with PagedAttention (2023)"
  creator: "UC Berkeley (Kwon et al.)"
  
  core_idea: |
    Don't allocate KV cache as one contiguous block per request.
    Instead, divide KV memory into fixed-size PAGES (blocks).
    Each request gets pages on-demand as it generates tokens.
    Pages can be non-contiguous in physical memory (mapped via page table).
    
  how_it_works:
    page_size: "Typically 16 tokens per page (configurable)"
    allocation: "Request starts with 0 pages. Gets new page when current page fills up."
    deallocation: "When request completes, all its pages return to free pool immediately."
    no_fragmentation: "All pages are same size — any free page can serve any request."
    no_over_allocation: "Request only holds pages for tokens it has actually generated."
    
  memory_efficiency:
    static_allocation:
      waste: "60-80% (pre-allocate for max length)"
      fragmentation: "Severe over time"
      concurrent_requests: "Low (memory reserved but unused)"
      
    paged_attention:
      waste: "< 4% (only last page partially filled)"
      fragmentation: "Zero (all pages are same size, any page fits any slot)"
      concurrent_requests: "2-4× more (only allocate what's used)"
      
  implementation_detail:
    page_table: |
      Each request has a page table mapping:
      logical_page_0 → physical_page_47
      logical_page_1 → physical_page_12
      logical_page_2 → physical_page_93
      
      Pages don't need to be contiguous in physical memory.
      The attention kernel uses the page table to gather KV entries.
      
    block_manager: |
      Centralized manager tracks:
      - Free pages (available for allocation)
      - Allocated pages (owned by specific requests)
      - Reference counts (for copy-on-write sharing)
      
  advanced_features:
    copy_on_write:
      what: "Multiple requests can share KV pages (read-only) until one modifies"
      use_case: "Beam search — all beams share prompt KV, fork only at divergence"
      benefit: "Beam search uses 1/beam_width memory vs naive approach"
      
    preemption:
      what: "When memory is full, swap lowest-priority request's pages to CPU RAM"
      how: "Copy KV pages to CPU, free GPU pages for higher-priority request"
      resume: "When GPU memory frees up, swap pages back and continue generation"
      benefit: "Graceful degradation under memory pressure (vs OOM crash)"
```

### KV Cache Quantization

```yaml
KV_Cache_Quantization:
  what: "Quantize the KV cache itself (not model weights) to reduce memory per cached token"
  
  why: |
    For long-context models, KV cache dominates memory usage.
    A 128K context request on Llama-70B needs:
    328 KB × 128K tokens = 41 GB just for KV cache (FP16)
    Quantizing KV to INT8: 20.5 GB (2× more concurrent long-context requests)
    Quantizing KV to INT4: 10.2 GB (4× more concurrent requests)
    
  approaches:
    fp8_kv_cache:
      precision: "FP8 (E4M3)"
      memory_savings: "2× vs FP16"
      quality_impact: "< 0.5% degradation on most benchmarks"
      hardware: "H100/B200 native FP8 support"
      implementation: "vLLM --kv-cache-dtype fp8"
      recommendation: "Default choice on H100+ for memory-constrained serving"
      
    int8_kv_cache:
      precision: "INT8 with per-channel scaling"
      memory_savings: "2× vs FP16"
      quality_impact: "< 1% on most tasks"
      implementation: "Research implementations, some vLLM support"
      
    int4_kv_cache:
      precision: "INT4 with group quantization"
      memory_savings: "4× vs FP16"
      quality_impact: "1-3% (more noticeable for long contexts)"
      trade_off: "Significant memory savings but quality degradation for reasoning tasks"
      
  practical_impact:
    example: "70B model on H100 (80 GB)"
    model_int4: "35 GB"
    remaining: "45 GB for KV cache"
    
    fp16_kv: "45 GB / 2.6 GB per 8K request = 17 concurrent requests"
    fp8_kv: "45 GB / 1.3 GB per 8K request = 34 concurrent requests (2× more)"
    int4_kv: "45 GB / 0.65 GB per 8K request = 69 concurrent requests (4× more)"
```

### Prefix Caching in Practice

```python
# vLLM prefix caching example

# All these requests share the same system prompt
# With prefix caching: system prompt KV computed ONCE, shared across all

import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",  # vLLM server
    api_key="not-needed",
)

# System prompt (shared across all requests)
SYSTEM_PROMPT = """You are a helpful customer service agent for TechCo. 
You have access to the following policies:
- Refund policy: 30 days, original payment method...
- Shipping: Free over $50, 3-5 business days...
[... 500 tokens of policy context ...]"""

# Without prefix caching:
# Each request computes KV for 500-token system prompt independently
# 1000 requests/minute × 500 tokens prefill = 500,000 redundant token computations/minute

# With prefix caching (vLLM --enable-prefix-caching):
# First request computes system prompt KV and caches it
# All subsequent requests SKIP system prompt prefill (KV already in cache)
# Savings: 99.8% reduction in system prompt compute (only pay once)

# This is transparent to the API consumer:
responses = []
for user_query in user_queries:  # 1000 different user questions
    response = client.chat.completions.create(
        model="meta-llama/Llama-4-8B",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},  # Cached!
            {"role": "user", "content": user_query},        # Only this is new
        ],
        max_tokens=256,
    )
    responses.append(response)

# Multi-turn conversation caching:
# Turn 1: [system + user1]  → compute all, cache
# Turn 2: [system + user1 + assistant1 + user2]  → only compute user2 (rest cached)
# Turn 3: [system + user1 + assistant1 + user2 + assistant2 + user3]  → only compute user3
# Each turn only pays for NEW tokens, not the entire conversation history
```

---

## How It Works in Practice

### vLLM Configuration for KV Cache Optimization

```bash
# Production vLLM deployment with all caching optimizations

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-4-70B-AWQ \
    --quantization awq \
    --dtype half \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8 \
    --block-size 16 \
    --swap-space 4 \
    --max-num-seqs 256 \
    --tensor-parallel-size 2

# Key parameters explained:
# --gpu-memory-utilization 0.92: Use 92% of GPU memory (rest for overhead)
# --enable-prefix-caching: Share KV for common prefixes (system prompts)
# --kv-cache-dtype fp8: Quantize KV cache to FP8 (2× more concurrent requests)
# --block-size 16: PagedAttention page size (16 tokens per page)
# --swap-space 4: 4 GB CPU swap space for preempted requests
# --max-num-seqs 256: Maximum concurrent sequences
```

---

## Interview Tip

> When asked about KV cache and model caching: "KV cache management is THE bottleneck for LLM serving — not model size, not compute. PagedAttention (vLLM's key innovation) applies OS virtual memory concepts: allocate KV cache in fixed-size pages on-demand, use page tables for non-contiguous mapping. This eliminates 60-80% memory waste from static pre-allocation and enables 2-4x more concurrent requests. Three layers of optimization: (1) PagedAttention for zero fragmentation and on-demand allocation, (2) KV cache quantization (FP8) for 2x memory reduction with <0.5% quality loss, (3) Prefix caching to share KV computation for common system prompts across requests. Together these mean: same H100 GPU serves 4-8x more concurrent users than naive implementation. For context: Llama-70B INT4 on H100 — without these optimizations you serve ~4 requests at 8K context. With PagedAttention + FP8 KV: ~34 concurrent requests. The math: 35GB model + remaining 45GB ÷ 1.3GB per FP8 KV request = 34 users."

---

## Common Mistakes

1. **Not enabling prefix caching for chatbot/RAG workloads** — Every request recomputes KV for the shared system prompt or document context. For a 1000-token system prompt at 1000 requests/minute, that's 1 million wasted token computations per minute. One configuration flag eliminates this entirely.

2. **Setting max_model_len too high** — Configuring the server for 128K context when 95% of requests use <4K. The server pre-reserves KV cache management structures for max_model_len, reducing capacity for shorter requests. Set max_model_len to your actual P99 context length.

3. **Ignoring KV cache quantization** — Running KV cache in FP16 on H100 when FP8 is available. FP8 KV cache halves memory per cached token with <0.5% quality loss — doubling concurrent request capacity for free on H100/B200 hardware.

4. **No swap space configured** — When all GPU KV cache pages are full and a new high-priority request arrives, the system rejects it (503 error) or crashes (OOM). Configuring CPU swap space allows graceful preemption: low-priority request's KV pages are temporarily moved to CPU RAM, GPU pages freed for the new request.

5. **Not accounting for model loading time in scaling** — Auto-scaling adds a new GPU instance, but the model takes 60 seconds to load. During this time, the new instance serves zero requests while you're paying for it, and existing instances remain overloaded. Solution: pre-warm instances (keep spare loaded instances) or use persistent model storage on local SSD.

---

## Key Takeaways

- KV cache is the primary memory bottleneck for LLM serving (not model weights after quantization)
- PagedAttention: allocate KV in fixed-size pages on-demand, use page tables — eliminates 60-80% waste
- KV cache memory per request: layers × kv_heads × head_dim × 2 × dtype_bytes × seq_length
- FP8 KV cache quantization: 2× more concurrent requests, <0.5% quality loss (H100/B200)
- Prefix caching: share KV for common prefixes (system prompts) — eliminates redundant computation
- Copy-on-write: efficient memory sharing for beam search and parallel sampling
- Preemption: gracefully swap KV to CPU when memory is full (vs OOM crash)
- Model loading: keep warm pools, use local SSD storage, pre-warm during scaling
- KV cache, not compute, determines maximum concurrent users at long context lengths
- vLLM configuration: --enable-prefix-caching + --kv-cache-dtype fp8 for maximum efficiency
