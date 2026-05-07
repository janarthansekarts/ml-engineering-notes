# Serving LLMs at Scale

## The Problem / Why This Matters

Serving LLMs (Large Language Models) is fundamentally different from serving traditional ML models. A BERT classifier takes a fixed input, runs one forward pass (~10ms), and returns a fixed output. An LLM takes variable-length input, runs a compute-heavy prefill phase, then generates tokens one-at-a-time in an autoregressive decode loop that can last seconds to minutes. This creates unique challenges: KV cache memory grows with every token generated (a single 32K-context request on a 70B model needs ~10 GB just for attention state), generation is memory-bandwidth bound (reading 35 GB of weights for every single output token), and output length is unpredictable (can't pre-allocate resources efficiently). At scale — thousands of concurrent users — these challenges compound: memory fragmentation, uneven request durations causing batch imbalance, and the cold-start problem when scaling up. vLLM, TGI (Text Generation Inference), and TensorRT-LLM have emerged as the production solutions, each with distinct strengths. In 2026, the LLM serving stack has standardized around: PagedAttention for memory management, continuous batching for throughput, FP8/INT4 quantization for cost reduction, and prefix caching for multi-turn efficiency. This lesson integrates all previous serving concepts into the specific context of LLM inference at production scale.

---

## The Analogy

Think of LLM serving like running a live translation service for the United Nations:

- **Traditional ML serving** = Written translation. Receive document, translate, return. Fixed input, fixed output, predictable time. Easy to batch (translate 50 documents simultaneously).
- **LLM serving** = Live simultaneous interpretation. Interpreter (GPU) listens to ongoing speech (streaming input), maintains context of everything said so far (KV cache growing with every word), produces translation word-by-word in real-time (autoregressive generation), and different speakers talk for different durations (variable output length). Can't predict when a speaker will stop, can't easily share interpreters between booths (memory isolation).
- **PagedAttention** = Instead of reserving a dedicated notebook per speaker (static allocation), interpreters use sticky notes (pages). Each word gets a sticky note, notes can be stored anywhere, and when a speaker finishes, their sticky notes are immediately available for the next speaker.
- **Continuous batching** = An interpreter handles 8 speakers simultaneously. When speaker 3 finishes mid-sentence (short response), speaker 9 from the waiting room immediately takes that slot. The interpreter never has idle capacity.

---

## Deep Dive

### vLLM Production Deployment

```yaml
vLLM_Production:
  what: "De facto standard LLM serving engine (2026)"
  key_features:
    - "PagedAttention (zero memory waste)"
    - "Continuous batching (maximum throughput)"
    - "Prefix caching (shared system prompt KV)"
    - "Tensor parallelism (multi-GPU)"
    - "Multi-LoRA serving"
    - "Speculative decoding"
    - "FP8/INT4 quantization"
    - "OpenAI-compatible API"
    - "Structured output (JSON mode)"
    - "Vision-language model support"
    
  production_config:
    command: |
      python -m vllm.entrypoints.openai.api_server \
          --model meta-llama/Llama-4-70B-AWQ \
          --quantization awq \
          --tensor-parallel-size 2 \
          --gpu-memory-utilization 0.92 \
          --max-model-len 32768 \
          --max-num-seqs 256 \
          --enable-prefix-caching \
          --kv-cache-dtype fp8 \
          --enable-chunked-prefill \
          --max-num-batched-tokens 65536 \
          --block-size 16 \
          --swap-space 8 \
          --disable-log-requests \
          --uvicorn-log-level warning
          
    parameter_explanations:
      tensor_parallel_size: "Split model across N GPUs (N=2 for 70B INT4)"
      gpu_memory_utilization: "Use 92% of GPU memory (8% for overhead/safety)"
      max_model_len: "Maximum context length to support (32K tokens)"
      max_num_seqs: "Maximum concurrent sequences in a batch"
      enable_prefix_caching: "Cache KV for shared prefixes (system prompts)"
      kv_cache_dtype: "FP8 KV cache — 2× more concurrent requests"
      enable_chunked_prefill: "Don't let long prompts block decode"
      max_num_batched_tokens: "Maximum tokens processed per iteration"
      swap_space: "GB of CPU RAM for preempted request KV cache"
      
  docker_deployment:
    dockerfile: |
      FROM vllm/vllm-openai:latest
      
      # Download model at build time (faster startup)
      RUN python -c "from huggingface_hub import snapshot_download; \
          snapshot_download('meta-llama/Llama-4-70B-AWQ', local_dir='/models/llama-70b-awq')"
      
      ENV MODEL_PATH=/models/llama-70b-awq
      
      CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
           "--model", "/models/llama-70b-awq", \
           "--quantization", "awq", \
           "--tensor-parallel-size", "2"]
           
  kubernetes_deployment:
    yaml: |
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: vllm-llama-70b
      spec:
        replicas: 3
        selector:
          matchLabels:
            app: vllm-llama-70b
        template:
          spec:
            containers:
              - name: vllm
                image: vllm-llama-70b:latest
                resources:
                  limits:
                    nvidia.com/gpu: 2  # 2 GPUs per pod (tensor parallel)
                ports:
                  - containerPort: 8000
                readinessProbe:
                  httpGet:
                    path: /health
                    port: 8000
                  initialDelaySeconds: 120  # Model loading time
                  periodSeconds: 10
                livenessProbe:
                  httpGet:
                    path: /health
                    port: 8000
                  initialDelaySeconds: 180
                  periodSeconds: 30
```

### TGI (Text Generation Inference) Production Setup

```yaml
TGI_Production:
  what: "HuggingFace's optimized LLM serving solution"
  
  deployment:
    docker: |
      docker run --gpus all \
          -p 8080:80 \
          -v /models:/models \
          -e MODEL_ID=/models/llama-70b-awq \
          -e QUANTIZE=awq \
          -e MAX_INPUT_TOKENS=4096 \
          -e MAX_TOTAL_TOKENS=8192 \
          -e MAX_BATCH_PREFILL_TOKENS=16384 \
          -e MAX_CONCURRENT_REQUESTS=128 \
          -e NUM_SHARD=2 \
          ghcr.io/huggingface/text-generation-inference:latest
          
  key_parameters:
    MAX_INPUT_TOKENS: "Maximum input prompt length (reject longer)"
    MAX_TOTAL_TOKENS: "Maximum input + output tokens per request"
    MAX_BATCH_PREFILL_TOKENS: "Limit prefill compute per batch (prevents long-prompt stalls)"
    MAX_CONCURRENT_REQUESTS: "Maximum in-flight requests"
    NUM_SHARD: "Tensor parallelism across GPUs"
    QUANTIZE: "Quantization method (awq, gptq, eetq, bitsandbytes)"
    
  api_usage:
    streaming: |
      import requests
      
      response = requests.post(
          "http://localhost:8080/generate_stream",
          json={
              "inputs": "Explain transformer attention:",
              "parameters": {
                  "max_new_tokens": 512,
                  "temperature": 0.7,
                  "top_p": 0.9,
                  "repetition_penalty": 1.1,
                  "do_sample": True,
              }
          },
          stream=True
      )
      
      for chunk in response.iter_lines():
          # Token-by-token streaming
          print(chunk.decode())
          
  vs_vllm:
    throughput: "vLLM slightly higher in most benchmarks (PagedAttention advantage)"
    ease: "TGI simpler for HuggingFace model users"
    features: "Both comparable — TGI has watermarking, vLLM has more quantization options"
    recommendation: "vLLM for maximum throughput, TGI for simplicity and HF ecosystem"
```

### Production Architecture for LLM Serving

```yaml
Production_Architecture:
  load_balancing:
    challenge: |
      LLM requests have HIGHLY variable duration:
      - "What's 2+2?" → 1 second
      - "Write a 5000-word essay" → 30 seconds
      Standard round-robin load balancing creates uneven GPU utilization.
      
    solutions:
      least_connections:
        what: "Route to server with fewest active requests"
        advantage: "Better than round-robin for variable-duration requests"
        
      kv_cache_aware:
        what: "Route to server that has relevant KV cache (for multi-turn)"
        benefit: "Skip prefill for conversation history (already cached)"
        implementation: "Session affinity + prefix cache tracking"
        
      capacity_aware:
        what: "Route based on actual available KV cache capacity"
        how: "Each server reports available_kv_blocks → route to most available"
        benefit: "Prevents overloading servers with many long-context requests"
        
  multi_gpu_strategies:
    tensor_parallelism:
      what: "Split model layers across GPUs (each GPU has part of every layer)"
      communication: "All-reduce after each layer (high bandwidth needed — NVLink)"
      scaling: "Typically 2-8 GPUs (diminishing returns beyond 8 for inference)"
      use: "Model doesn't fit on single GPU (70B FP16 needs 2× H100)"
      
    pipeline_parallelism:
      what: "Split model layers sequentially (GPU 1 has layers 0-39, GPU 2 has layers 40-79)"
      communication: "Forward activations between stages"
      advantage: "Less communication overhead per step"
      disadvantage: "Pipeline bubbles (GPUs idle while waiting for previous stage)"
      use: "Very large models on many GPUs where NVLink isn't available"
      
    data_parallelism:
      what: "Each GPU has full model copy, serves different requests"
      communication: "None between GPUs (fully independent)"
      scaling: "Linear — 4 GPUs = 4× throughput"
      use: "Model fits on single GPU, need more throughput (horizontal scaling)"
      
  high_availability:
    redundancy: "Minimum 2 replicas (survives single node failure)"
    health_checks: |
      - Liveness: /health endpoint (is process alive?)
      - Readiness: /ready endpoint (is model loaded and accepting requests?)
      - Startup: initial 120-180s grace period for model loading
    graceful_shutdown: |
      On termination signal:
      1. Stop accepting new requests
      2. Finish in-flight requests (drain)
      3. Save any persistent state
      4. Exit after all requests complete or 30s timeout
    circuit_breaker: |
      If error rate > 10% for 30 seconds:
      1. Stop routing to unhealthy instance
      2. Alert on-call engineer
      3. Attempt restart
      4. Only resume traffic after health check passes
```

### Performance Optimization Checklist

```yaml
Performance_Optimization:
  memory:
    quantize_model: "AWQ INT4 (4× memory reduction) or FP8 (2× with better quality)"
    quantize_kv_cache: "--kv-cache-dtype fp8 (2× more concurrent requests)"
    set_appropriate_max_len: "Don't set 128K if 99% of requests use <8K"
    enable_prefix_caching: "Share KV for common prefixes"
    configure_swap_space: "CPU swap for graceful memory pressure handling"
    
  throughput:
    continuous_batching: "Default in vLLM/TGI (never disable)"
    chunked_prefill: "Prevent long prompts from blocking decode steps"
    optimal_batch_size: "max_num_seqs = available_kv_blocks / avg_sequence_length"
    tensor_parallelism: "Split across GPUs if memory-constrained"
    data_parallelism: "Add replicas for more throughput"
    
  latency:
    ttft_optimization: "Reduce prefill time with chunked prefill, prefix caching"
    tpot_optimization: "TPOT (Time Per Output Token) — optimize decode with FP8, speculative decoding"
    speculative_decoding: "2-3× decode speed with matching draft model"
    warmup_prompts: "Pre-fill CUDA graphs by running sample prompts at startup"
    
  monitoring:
    key_metrics:
      - "TTFT P50/P95/P99 (Time To First Token)"
      - "TPOT P50/P95/P99 (Time Per Output Token)"
      - "E2E latency P50/P95/P99 (total request time)"
      - "Throughput (tokens/second across all requests)"
      - "GPU memory utilization (%)"
      - "KV cache utilization (%)"
      - "Queue depth (pending requests)"
      - "Request success rate (%)"
      - "Token generation rate per user"
```

### Multi-LoRA Serving

```yaml
Multi_LoRA:
  what: "Serve multiple LoRA (Low-Rank Adaptation) fine-tuned models from one base model"
  
  why: |
    You fine-tuned 10 different LoRA adapters for 10 different customers/tasks.
    Without multi-LoRA: load 10 separate copies of the base model (10× memory).
    With multi-LoRA: load base model ONCE + all 10 LoRA adapters (tiny overhead).
    LoRA adapter: ~0.1-1% of base model size (a 70B model's LoRA: 100-700 MB vs 35 GB base)
    
  how_it_works:
    base_model: "Loaded once, shared across all requests"
    lora_adapters: "Small adapter weights swapped per-request"
    batching: "Requests with DIFFERENT LoRAs can be batched together"
    overhead: "Near-zero memory overhead per adapter (shared base dominates)"
    
  vllm_multi_lora:
    config: |
      python -m vllm.entrypoints.openai.api_server \
          --model meta-llama/Llama-4-8B \
          --enable-lora \
          --lora-modules \
              customer_a=./loras/customer_a \
              customer_b=./loras/customer_b \
              code_assistant=./loras/code_assistant \
          --max-loras 16 \
          --max-lora-rank 64
          
    api_usage: |
      # Request with specific LoRA adapter
      response = client.chat.completions.create(
          model="customer_a",  # Specifies which LoRA adapter
          messages=[{"role": "user", "content": "Hello"}],
      )
      
  use_cases:
    - "Multi-tenant platform (each customer has custom fine-tune)"
    - "Task-specific adapters (code, creative, analytical) on same base"
    - "A/B testing LoRA variants"
    - "Language-specific adapters on multilingual base"
```

### Structured Output and Constrained Generation

```yaml
Structured_Output:
  what: "Force LLM to generate valid JSON, XML, or follow a specific schema"
  
  why: |
    Raw LLM output is free-form text. Applications need structured data:
    - API responses need valid JSON
    - Form filling needs specific fields
    - Tool calls need function signatures
    
  approaches:
    json_mode:
      what: "Constrain output to valid JSON"
      vllm: "--guided-decoding-backend outlines"
      api: |
        response = client.chat.completions.create(
            model="llama-4-8b",
            messages=[...],
            response_format={"type": "json_object"}
        )
        
    schema_constrained:
      what: "Constrain output to match a specific JSON Schema"
      how: "At each token generation step, mask logits for tokens that would violate schema"
      benefit: "100% valid output (guaranteed schema compliance)"
      overhead: "10-20% slower than unconstrained generation"
      tools: "Outlines (vLLM integration), Instructor, LMQL"
      
    grammar_guided:
      what: "Constrain output to match a formal grammar (BNF/EBNF)"
      use_case: "Generate valid SQL, Python, YAML, or any structured format"
      tools: "llama.cpp grammars, Outlines, Guidance"
```

---

## How It Works in Practice

### Production Deployment Checklist

```yaml
Deployment_Checklist:
  pre_deployment:
    - "Benchmark model on target hardware (measure actual TTFT, TPOT, throughput)"
    - "Calculate KV cache capacity (how many concurrent requests at expected context length)"
    - "Test with production-like traffic patterns (variable lengths, burst patterns)"
    - "Set appropriate max_model_len (don't over-allocate for rare long contexts)"
    - "Configure rate limiting and request quotas per user"
    
  deployment:
    - "Deploy with readiness probe (don't route until model is loaded)"
    - "Start with conservative max_num_seqs (increase after observing behavior)"
    - "Enable prefix caching for chatbot/multi-turn workloads"
    - "Set up Prometheus metrics export"
    - "Configure alerts on TTFT P95, error rate, and KV utilization"
    
  post_deployment:
    - "Monitor KV cache utilization (trending toward 100% = need to scale)"
    - "Track TTFT distribution (bimodal = some requests hitting long queue)"
    - "Watch for OOM events (swap space prevents crashes but degrades latency)"
    - "A/B test against previous model version"
    - "Set up cost tracking (GPU-hours per 1M tokens)"
    
  cost_optimization:
    tokens_per_dollar: "Primary cost efficiency metric"
    strategies:
      - "Quantize: INT4 (4× less GPU memory = 4× more requests per GPU)"
      - "Prefix caching: avoid recomputing shared context"
      - "Right-size GPU: L4 for 8B, H100 for 70B"
      - "Scale-to-zero for low-traffic endpoints"
      - "Model cascade: route simple queries to small model"
      - "Batch offline requests (higher batch utilization = lower cost/token)"
```

---

## Interview Tip

> When asked about serving LLMs at scale: "My production LLM serving stack: vLLM with PagedAttention (zero KV cache waste, 2-4× more concurrent users), continuous batching (GPU never idle), and prefix caching (skip redundant system prompt computation). Hardware: 70B model on 2× H100 with tensor parallelism, AWQ INT4 quantization (fits in 35 GB), FP8 KV cache (doubles concurrent request capacity). Key architecture decisions: (1) Separate prefill and decode concerns — chunked prefill prevents long prompts from spiking other users' TPOT. (2) Load balance by available KV capacity, not round-robin — requests have wildly different memory footprints (2K context vs 32K context). (3) Multi-LoRA serving — 10 customer-specific fine-tunes from one base model with negligible memory overhead. (4) Structured output via grammar-guided generation for reliable JSON APIs. Scaling: predictive autoscaling on KV utilization + reactive HPA on queue depth. The result: serve 1000+ concurrent users on 3 pods of 2× H100 each, with P95 TTFT < 500ms and $0.50 per million output tokens (vs $15/M tokens for GPT-4o API). Cost-effective self-hosting is viable at >100M tokens/month."

---

## Common Mistakes

1. **Not setting max_model_len appropriately** — Configuring 128K context when 99% of requests use < 8K. vLLM pre-allocates KV cache management structures for max_model_len. Setting it too high wastes memory that could serve more concurrent short requests. Set to your P99 actual context length (or P99.9 with swap space for outliers).

2. **Using round-robin load balancing** — One server has 50 active 32K-context requests (KV cache nearly full). Another server has 50 active 2K-context requests (KV cache mostly empty). Round-robin sends the next 32K request to the full server → OOM or extreme queuing. Use KV-cache-aware or least-connections load balancing.

3. **No request timeout** — A user sends a prompt that triggers infinite repetition (model doesn't hit stop token). Without max_tokens or timeout, this request occupies KV cache indefinitely, eventually starving other users. Always set max_tokens AND server-side timeout.

4. **Deploying without readiness probes** — Load balancer sends traffic to new pod while model is still loading (30-120 seconds). All requests during this period fail with 503/timeout. Kubernetes readiness probe must only pass AFTER model is fully loaded and first health check succeeds.

5. **Not monitoring TTFT separately from total latency** — Total latency = TTFT + (tokens × TPOT). A 500-token response with 300ms TTFT and 30ms TPOT = 15.3 seconds total — but user perceives it as responsive (sees first token quickly, then streaming). If TTFT spikes to 5 seconds, user thinks it's broken. Monitor and alert on TTFT specifically.

---

## Key Takeaways

- vLLM: production standard for LLM serving — PagedAttention, continuous batching, prefix caching, OpenAI API
- Quantization: AWQ INT4 for model weights (4× savings), FP8 for KV cache (2× more concurrent requests)
- Continuous batching: insert/remove requests per decode step — GPU never idle
- Prefix caching: skip KV computation for shared system prompts (essential for chatbots)
- Chunked prefill: prevent long prompts from blocking other requests' decode steps
- Tensor parallelism: split model across GPUs when it doesn't fit on one (NVLink required)
- Multi-LoRA: serve many fine-tuned variants from one base model (negligible overhead per adapter)
- Load balance by KV cache capacity, not round-robin — request memory footprint varies 10-100×
- Monitor TTFT separately (user-perceived responsiveness) — not just total latency
- Structured output: grammar-guided generation for guaranteed-valid JSON/schema compliance
- Cost efficiency: quantization + prefix caching + right-sizing + cascade = 10-20× cost reduction vs naive
