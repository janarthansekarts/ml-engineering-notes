# Multi-Model Serving

## The Problem / Why This Matters

Production ML systems rarely serve a single model. A search engine might route through: query understanding model → retrieval model → ranking model → reranking model → summarization model. A chatbot system uses: router model (determine intent) → specialized model (code, creative, analytical) → safety model (check output) → format model (structured output). Serving multiple models introduces challenges that don't exist with single-model deployments: resource allocation (which models share GPUs?), routing logic (how to direct requests to the right model?), latency composition (5 models × 50ms each = 250ms total), memory management (20 models competing for 80 GB GPU memory), and version coordination (upgrading one model in a pipeline without breaking others). In 2026, multi-model architectures are the norm — from simple model ensembles to complex systems like Mixture of Experts (MoE), speculative decoding, model cascades, and AI agent orchestration where a coordinator model invokes specialist models as needed. Understanding multi-model serving patterns is essential for building production-grade ML systems that are both capable and cost-efficient.

---

## The Analogy

Think of multi-model serving like a hospital with specialist departments:

- **Model routing** = The triage desk. Assess the patient's complaint and direct them to the right specialist. A broken bone goes to orthopedics, chest pain goes to cardiology. Wrong routing wastes specialist time and delays treatment.
- **Model ensemble** = A tumor board. Multiple specialists (oncologist, radiologist, surgeon) each give their assessment, then a consensus decision is made. More accurate than any single specialist, but slower and more expensive.
- **Model cascade** = Escalation. Nurse handles simple cases (80%). If too complex, escalate to general doctor (15%). If still too complex, escalate to specialist (5%). Most patients are handled cheaply by the nurse.
- **Speculative decoding** = A resident drafts the diagnosis, the attending physician reviews and corrects. The resident is fast (but less accurate), the attending is accurate (but busy). Together: attending-level quality at resident-level speed.
- **MoE (Mixture of Experts)** = A general practice with embedded specialists. Every patient enters through the same door, but an internal router sends their case to the relevant specialist within the practice. Externally looks like one entity.

---

## Deep Dive

### Model Routing

```yaml
Model_Routing:
  what: "Direct each request to the most appropriate model based on input characteristics"
  
  patterns:
    intent_based_routing:
      what: "Classify user intent, route to specialized model"
      example: |
        User: "Write a Python function to sort a list"  → Code model (DeepSeek Coder)
        User: "Write a poem about autumn"               → Creative model (Claude Opus)
        User: "Summarize this 50-page PDF"              → Long-context model (Gemini 2.5)
        User: "What's 2 + 2?"                           → Small fast model (Phi-3-mini)
      implementation: |
        1. Lightweight classifier model (BERT-based, <100ms)
        2. Classifies intent: code | creative | analytical | simple | multi-modal
        3. Routes to appropriate model
        
    complexity_based_routing:
      what: "Route easy queries to small cheap model, hard queries to large expensive model"
      benefit: "80% of queries are easy → 80% cost reduction vs always using large model"
      implementation: |
        1. Try small model first (Phi-3-mini, Gemma-2B)
        2. If confidence < threshold OR output quality check fails
        3. Escalate to large model (Llama-70B, GPT-5)
      challenge: "Need reliable confidence/quality estimation"
      
    capability_based_routing:
      what: "Route based on model capabilities (vision, code, math, languages)"
      example: |
        - Image input → Vision-language model (GPT-5V, Gemini)
        - Code + execution → Code model with sandbox (DeepSeek, Claude)
        - Math/reasoning → Reasoning model (o3, Claude Opus)
        - Translation → Multilingual model
        
  implementation:
    router_model:
      approach: "Train a small classifier to predict best model for each query"
      training_data: "Query → best model (determined by evaluating all models on queries)"
      latency: "< 10ms (doesn't meaningfully add to total latency)"
      
    rule_based:
      approach: "Heuristic rules (contains code → code model, has image → vision model)"
      advantage: "Interpretable, no training needed"
      disadvantage: "Brittle, doesn't handle ambiguous cases"
      
    llm_as_router:
      approach: "Use a small LLM to analyze the query and decide routing"
      prompt: "Given this query, which model should handle it? Options: [code, creative, analytical]"
      latency: "50-200ms (significant overhead)"
      use_when: "Complex routing decisions that rules can't capture"
```

### Model Ensemble

```yaml
Model_Ensemble:
  what: "Combine predictions from multiple models for higher accuracy"
  
  types:
    voting_ensemble:
      what: "Multiple models predict, majority vote wins"
      example: "3 classifiers: model A says spam, model B says spam, model C says not-spam → spam"
      improvement: "Typically 2-5% accuracy improvement over best single model"
      
    weighted_ensemble:
      what: "Average predictions with learned weights"
      example: "0.4 × model_A + 0.35 × model_B + 0.25 × model_C"
      weights: "Learned on validation set (optimize for target metric)"
      
    stacking:
      what: "Train a meta-model on base model outputs"
      how: "Base models predict → their outputs become features → meta-model makes final prediction"
      advantage: "Meta-model learns when to trust which base model"
      
    llm_ensemble:
      what: "Multiple LLMs generate answers, then select or merge best"
      approaches:
        judge_selection: "LLM judge picks best response from candidates"
        majority_voting: "For factual questions, pick most common answer"
        fusion: "Merge best parts of multiple responses"
      cost: "N× inference cost — only for high-value predictions"
      use_case: "AI safety evaluation, critical decisions, competition/benchmarks"
      
  serving_considerations:
    parallel_execution:
      what: "Run all ensemble models simultaneously"
      latency: "Max of individual model latencies (not sum)"
      resource: "Need GPU capacity for all models concurrently"
      
    sequential_execution:
      what: "Run models one by one (when next model depends on previous output)"
      latency: "Sum of all model latencies"
      when: "Stacking, or when models need previous model's output as input"
```

### Model Cascade

```yaml
Model_Cascade:
  what: "Escalation pattern — try cheap model first, escalate to expensive model only if needed"
  
  design:
    tier_1_small:
      model: "Gemma-2B or Phi-3-mini (or rule-based system)"
      cost: "$0.001 per request"
      handles: "80% of requests (simple, high-confidence)"
      latency: "50ms"
      escalation_criteria: "confidence < 0.85 OR flagged as complex"
      
    tier_2_medium:
      model: "Llama-4-8B"
      cost: "$0.005 per request"
      handles: "15% of requests (moderate complexity)"
      latency: "200ms"
      escalation_criteria: "confidence < 0.7 OR detected as reasoning-heavy"
      
    tier_3_large:
      model: "Claude Opus or GPT-5"
      cost: "$0.05 per request"
      handles: "5% of requests (complex reasoning, ambiguous)"
      latency: "2000ms"
      
  cost_calculation:
    without_cascade: "100% × $0.05 = $0.05 average per request"
    with_cascade: "80% × $0.001 + 15% × $0.005 + 5% × $0.05 = $0.004 average"
    savings: "12.5× cost reduction"
    
  confidence_estimation:
    for_classification: "Softmax probability (P > 0.85 = confident)"
    for_generation: "Self-consistency (generate 3 times, check agreement)"
    for_llm: |
      Ask model: "Rate your confidence in this answer (1-10)"
      Or: generate with low temperature, check if perplexity is low
    learned_router: "Train a small model to predict when large model is needed"
    
  example_implementation:
    code: |
      async def cascade_inference(query: str) -> str:
          # Tier 1: Fast small model
          result_1 = await small_model.generate(query)
          if result_1.confidence > 0.85:
              return result_1.text  # 80% of requests stop here
              
          # Tier 2: Medium model (includes tier 1 output as context)
          result_2 = await medium_model.generate(
              f"Previous attempt (low confidence): {result_1.text}\n\nQuery: {query}"
          )
          if result_2.confidence > 0.70:
              return result_2.text  # 15% of requests stop here
              
          # Tier 3: Large model (5% of requests)
          result_3 = await large_model.generate(query)
          return result_3.text
```

### Speculative Decoding

```yaml
Speculative_Decoding:
  what: "Use a small fast model to draft tokens, large model to verify in parallel"
  key_insight: |
    Large model verification of N tokens is almost as fast as generating 1 token
    (it's a single forward pass over N tokens — like prefill).
    If the draft model guesses correctly, we get N tokens for the cost of ~1 large model step.
    
  how_it_works:
    step_1: "Draft model (small, fast) generates K tokens speculatively"
    step_2: "Target model (large, accurate) verifies all K tokens in one forward pass"
    step_3: "Accept all tokens where draft matches target (from left to right)"
    step_4: "At first mismatch, sample from target model's distribution for that position"
    step_5: "Repeat from the accepted position"
    
  performance:
    acceptance_rate: "60-80% typical (depends on draft/target model similarity)"
    speedup: "2-3× faster generation with IDENTICAL output quality"
    key_point: "Output is EXACTLY what target model would produce (mathematically proven)"
    
  requirements:
    draft_model: "Same tokenizer as target model, much smaller (e.g., 1B for 70B target)"
    target_model: "The large model you actually want to serve"
    
  examples:
    llama_70b:
      target: "Llama-4-70B"
      draft: "Llama-4-8B (same tokenizer, same family)"
      speedup: "2.2-2.8× on average"
      quality: "Identical to Llama-70B (mathematically guaranteed)"
      
    self_speculative:
      what: "Use early exit from same model as draft (skip later layers)"
      advantage: "No separate draft model needed"
      
  vllm_config: |
    # vLLM with speculative decoding
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-4-70B \
        --speculative-model meta-llama/Llama-4-8B \
        --num-speculative-tokens 5 \
        --use-v2-block-manager
        
  when_to_use:
    - "Serving large models where single-user latency matters"
    - "When a good draft model exists (same tokenizer, smaller)"
    - "NOT useful when throughput-bound (high concurrency) — overhead of running draft model"
    use_case: "Interactive chat with 70B+ models, real-time applications"
```

### Mixture of Experts (MoE)

```yaml
Mixture_of_Experts:
  what: "Architecture where only a subset of model parameters are active for each token"
  
  how_it_works:
    structure: "N expert networks + 1 gating network (router)"
    per_token: |
      1. Router examines input token
      2. Selects top-K experts (usually K=2 out of N=8 or N=16)
      3. Only the selected experts process the token
      4. Outputs are weighted sum of selected experts' outputs
      
  examples:
    mixtral_8x7b:
      total_params: "46.7B total parameters"
      active_params: "12.9B active per token (2 of 8 experts)"
      performance: "Matches or exceeds Llama-70B on many tasks"
      inference_speed: "Similar to 13B model (only 13B params active)"
      memory: "Still need to load ALL 46.7B params (all experts in memory)"
      
    grok:
      architecture: "MoE with high expert count"
      
    deepseek_v3:
      architecture: "MoE with fine-grained experts"
      params: "671B total, ~37B active"
      
  serving_implications:
    memory: |
      Must load ALL expert parameters into GPU memory (even though only K are active).
      Mixtral 8×7B in FP16: ~94 GB (needs 2× H100 or 1× H200)
      Mixtral 8×7B in INT4: ~24 GB (fits on single L4/A10G)
      
    compute: |
      Compute cost is proportional to ACTIVE parameters (much less than total).
      Mixtral: compute of ~13B model, memory of ~47B model.
      This makes MoE models bandwidth-bound even more than dense models.
      
    expert_parallelism:
      what: "Place different experts on different GPUs"
      benefit: "Each GPU only holds subset of experts — reduces per-GPU memory"
      challenge: "Requires all-to-all communication (tokens routed to correct GPU)"
      
  vs_dense_models:
    advantage: "More knowledge/capacity for same inference cost (active params)"
    disadvantage: "More memory needed (all params must be loaded)"
    when_to_prefer: "When memory is available but compute/latency budget is tight"
```

### Model Pipeline (Sequential Inference)

```yaml
Model_Pipeline:
  what: "Multiple models executed sequentially, each processing the previous model's output"
  
  examples:
    rag_pipeline:
      step_1: "Query embedding model (encode user query to vector)"
      step_2: "Vector similarity search (retrieve relevant documents)"
      step_3: "Reranking model (reorder results by relevance)"
      step_4: "Generation model (synthesize answer from top documents)"
      total_latency: "50ms + 20ms + 100ms + 2000ms = ~2.2 seconds"
      
    content_moderation:
      step_1: "Language detection model (10ms)"
      step_2: "Toxicity classification model (30ms)"
      step_3: "PII detection model (20ms)"
      step_4: "Content generation model (1000ms)"
      step_5: "Output safety model (50ms)"
      
    recommendation:
      step_1: "Candidate retrieval (lightweight model, 10K → 500 items)"
      step_2: "Ranking model (heavier model, 500 → 50 items)"
      step_3: "Reranking (cross-encoder, 50 → 10 items)"
      step_4: "Diversity/freshness filter (rule-based, 10 → 5 items)"
      
  serving_with_triton:
    what: "NVIDIA Triton ensemble models — chain models in a pipeline"
    config: |
      name: "rag_pipeline"
      platform: "ensemble"
      input [{
        name: "query"
        data_type: TYPE_STRING
        dims: [1]
      }]
      output [{
        name: "answer"
        data_type: TYPE_STRING
        dims: [1]
      }]
      ensemble_scheduling {
        step [{
          model_name: "query_encoder"
          model_version: 1
          input_map { key: "text" value: "query" }
          output_map { key: "embedding" value: "query_embedding" }
        }, {
          model_name: "reranker"
          model_version: 1
          input_map { key: "query_emb" value: "query_embedding" }
          output_map { key: "documents" value: "top_documents" }
        }, {
          model_name: "generator"
          model_version: 1
          input_map { key: "context" value: "top_documents" }
          output_map { key: "response" value: "answer" }
        }]
      }
      
  latency_optimization:
    parallel_where_possible: "If two models are independent, run simultaneously"
    pipeline_parallelism: "While model B processes request 1, model A processes request 2"
    early_termination: "If safety model rejects at step 2, skip steps 3-5"
    async_side_effects: "Logging, metrics, feedback collection — don't block the pipeline"
```

---

## How It Works in Practice

### Multi-Model Resource Management

```yaml
Resource_Management:
  shared_gpu:
    problem: "10 models, 2 GPUs — how to allocate?"
    strategies:
      time_multiplexing:
        what: "Load/unload models on demand (one at a time per GPU)"
        pro: "Many models on few GPUs"
        con: "Loading latency (30-120s for large models)"
        use: "Low-traffic models that are rarely invoked simultaneously"
        
      spatial_multiplexing:
        what: "Multiple small models loaded on same GPU simultaneously"
        pro: "No loading delay (models always ready)"
        con: "Each model gets less memory (smaller batch sizes)"
        use: "Small models (embedding, classification) serving simultaneously"
        tool: "Triton Inference Server (concurrent model execution)"
        
      kserve_modelmesh:
        what: "Kubernetes-native multi-model serving with intelligent scheduling"
        how: "Pool of GPUs, models loaded/unloaded based on demand, LRU eviction"
        benefit: "Hundreds of models on small GPU cluster"
        
  gpu_allocation:
    by_model_size:
      large_llm_70b: "Dedicated H100/H200 (takes full GPU memory)"
      medium_model_7b: "Shared GPU with 2-3 other small models"
      small_model_bert: "CPU or shared GPU with many other models"
      embedding_model: "Often CPU-optimal for moderate traffic"
```

---

## Interview Tip

> When asked about multi-model serving: "I work with four main multi-model patterns: (1) Model cascade — try cheap model first (handles 80% of requests), escalate to expensive model only when confidence is low. This gives 10-12× cost reduction vs always using the large model. (2) Model routing — classify intent and route to specialized models (code model for code, creative model for writing). (3) Speculative decoding — small draft model generates candidate tokens, large target model verifies in one forward pass. 2-3× speed with mathematically identical output quality. (4) MoE (Mixture of Experts) — architectures like Mixtral where only 2 of 8 experts activate per token, giving 47B model quality at 13B model speed. For resource management: small models share GPUs (Triton concurrent execution), large models get dedicated GPUs. For model pipelines (RAG, recommendation): optimize total latency by parallelizing independent stages, implementing early termination, and pipeline-parallelizing sequential stages. Key insight: in production, the system is always multi-model — the question is how to orchestrate efficiently."

---

## Common Mistakes

1. **Running all pipeline models sequentially when some are independent** — In a RAG pipeline: query encoding and document retrieval metadata lookup are independent — run them in parallel. Only serialize when there's a true data dependency. Parallelizing independent stages can cut total latency by 30-50%.

2. **Allocating one GPU per model regardless of utilization** — A BERT classifier getting 10 requests/minute on a dedicated H100 ($3/hour). This model uses 0.1% of the GPU's capacity. Co-locate small models on shared GPUs, or use CPU for low-traffic models.

3. **Not implementing fallback for model routing** — Router sends request to "code model" which is overloaded (queue full). Request fails. Always implement fallback: if primary model is unavailable → route to general-purpose model (slower but working) → return degraded response rather than error.

4. **Speculative decoding with mismatched tokenizers** — Using a draft model with a different tokenizer than the target model. Token boundaries don't align, verification fails on almost every token, and you get WORSE performance than no speculation. Draft and target MUST share the same tokenizer.

5. **Loading MoE models expecting dense-model memory** — "Mixtral is a 7B model" — no, it's 46.7B total parameters (only 13B active). You need memory for ALL parameters even though only a subset computes each token. Memory requirement is total params, speed is proportional to active params.

---

## Key Takeaways

- Model routing: classify intent/complexity and route to appropriate specialized model
- Model cascade: try cheap model first (80% handled), escalate to expensive only if needed (10× cost savings)
- Model ensemble: combine multiple models for better accuracy (at higher cost)
- Speculative decoding: small draft model + large verifier = 2-3× speed, identical quality
- MoE: architectures where only K of N experts activate — more knowledge at lower compute cost
- Model pipelines: chain models sequentially (RAG, recommendation) — parallelize independent stages
- Resource sharing: co-locate small models on GPUs (Triton), dedicate GPUs for large models
- KServe ModelMesh: Kubernetes-native multi-model serving with LRU eviction and dynamic loading
- Always implement fallback: if primary model unavailable → route to general-purpose → don't fail
- MoE memory: load ALL parameters (total_params) even though only active_params compute per token
