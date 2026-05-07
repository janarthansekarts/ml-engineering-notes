# Cost of ML

## The Problem / Why This Matters

ML is expensive — far more expensive than most organizations anticipate. A single large model training run can cost tens of thousands of dollars. Serving a popular model in production can cost millions per year. GPU (Graphics Processing Unit) clusters for LLM (Large Language Model) fine-tuning can burn through $100K in a week. And the hidden costs — data labeling, human evaluation, monitoring infrastructure, failed experiments — often exceed the compute costs. Teams that don't understand ML economics make catastrophic mistakes: training models that are too expensive to serve, choosing architectures that scale poorly with traffic, or running GPU instances 24/7 for workloads that run 2 hours per day. In 2026, with GPU demand exceeding supply (NVIDIA H100/H200/B200 chips have long lead times), cost optimization isn't just about saving money — it's about getting access to compute at all. Understanding ML cost structure is essential for any ML engineer making architecture, infrastructure, or deployment decisions.

---

## The Analogy

Think of ML costs like running a restaurant:

- **Training costs** = Developing the menu (recipe testing, ingredient experimentation). Expensive upfront, done occasionally.
- **Inference costs** = Serving customers (ingredients per plate, kitchen staff per shift). Ongoing, scales with demand.
- **GPU costs** = Kitchen equipment (industrial ovens, specialized tools). Expensive, high demand, long delivery times.
- **Hidden costs** = Health inspections, menu photography, staff training, food waste. Not the main expense but adds up fast.
- **The mistake** = Developing an amazing menu that requires $500 of truffle per plate, then discovering customers won't pay $800. (Training an amazing model you can't afford to serve.)

---

## Deep Dive

### ML Cost Breakdown

```yaml
ML_Cost_Categories:
  compute:
    training:
      description: "GPU/TPU hours for model training and experimentation"
      examples:
        small_model: "XGBoost on 1M rows: $0.50-$5 per training run"
        medium_model: "BERT fine-tuning: $10-$100 per run"
        large_model: "GPT-scale pre-training: $1M-$100M+ (not something most teams do)"
        llm_fine_tuning: "LoRA fine-tuning Llama-4-8B: $50-$500 per run"
        llm_full_fine_tuning: "Full fine-tuning 70B model: $5,000-$50,000 per run"
      key_insight: "Training is expensive but infrequent — amortizes over model lifetime"
      
    inference:
      description: "Compute for serving predictions to users"
      examples:
        simple_model: "XGBoost serving: $50-$500/month (CPU sufficient)"
        deep_learning: "Neural network on GPU: $500-$5,000/month"
        llm_inference: "LLM serving (Llama-4-8B on A100): $2,000-$10,000/month"
        llm_api: "OpenAI GPT-4o: $2.50-$10 per 1M tokens"
        llm_api_large: "Claude Opus 4: $15/$75 per 1M input/output tokens"
      key_insight: "Inference cost = cost_per_request × QPS × 24/7. Small per-request cost compounds massively at scale."
      
    experimentation:
      description: "Failed experiments, hyperparameter search, ablation studies"
      reality: "For every successful model, 10-50 experiments were run and discarded"
      implication: "True training cost = successful_run_cost × (1 + failure_ratio)"
      
  data:
    collection: "Data acquisition, API costs, web scraping infrastructure"
    labeling: "Human annotation ($5-50 per complex label, $0.10-1 per simple label)"
    storage: "S3/GCS storage for datasets, features, model artifacts"
    processing: "Spark/Dataflow jobs for feature engineering at scale"
    
  infrastructure:
    gpu_cluster: "Managing GPU nodes, networking, storage"
    orchestration: "Kubernetes, job schedulers, pipeline tools"
    monitoring: "Observability stack for ML (Evidently, Arize, Prometheus)"
    networking: "Data transfer between regions/services"
    
  people:
    salaries: "ML engineers: $150-400K+ total comp. ML platform engineers: $200-500K+"
    team_size: "Often the largest cost category"
    insight: "A $300K ML engineer who saves $500K in compute/year is a bargain"
```

### GPU Economics in 2026

```yaml
GPU_Economics:
  hardware_landscape:
    nvidia_h100:
      specs: "80GB HBM3, 3958 TFLOPS FP8"
      cloud_cost: "$2-4/hour (on-demand), $1-2/hour (spot/reserved)"
      use_case: "LLM fine-tuning, large model training, high-throughput inference"
      
    nvidia_h200:
      specs: "141GB HBM3e, improved memory bandwidth"
      cloud_cost: "$4-6/hour (on-demand)"
      use_case: "Larger model fine-tuning, high batch inference, research"
      
    nvidia_b200:
      specs: "192GB HBM3e, 2.5x H100 performance per watt"
      cloud_cost: "$6-10/hour (limited availability in 2026)"
      use_case: "Frontier model training, next-gen inference"
      
    nvidia_a100:
      specs: "80GB HBM2e (previous generation)"
      cloud_cost: "$1-2.50/hour (widely available, good value)"
      use_case: "Most production workloads, fine-tuning medium models"
      
    nvidia_l4:
      specs: "24GB GDDR6, inference-optimized"
      cloud_cost: "$0.50-1/hour"
      use_case: "Cost-efficient inference for smaller models"
      
    nvidia_t4:
      specs: "16GB GDDR6 (older, widely available)"
      cloud_cost: "$0.35-0.75/hour"
      use_case: "Budget inference, small model training"
      
  cost_optimization_strategies:
    spot_instances:
      savings: "60-90% vs on-demand"
      risk: "Preemption (instance can be taken away)"
      mitigation: "Checkpointing every 30-60 minutes, fault-tolerant training"
      best_for: "Training jobs (can be restarted), batch inference"
      not_for: "Real-time serving (latency SLA requirements)"
      
    reserved_instances:
      savings: "30-60% vs on-demand (1-3 year commitment)"
      best_for: "Steady-state inference serving, predictable training schedules"
      risk: "Commitment to specific GPU type — may be outdated in 2 years"
      
    right_sizing:
      principle: "Match GPU to workload — don't use H100 for XGBoost inference"
      examples:
        - "Small model inference → CPU or T4 (not A100)"
        - "Medium model inference → L4 (not H100)"
        - "LLM fine-tuning → A100/H100"
        - "LLM inference → H100/H200 with batching optimization"
      tools: "GPU utilization monitoring, cloud cost explorers"
      
    inference_optimization:
      quantization: "FP32 → FP16 → INT8 → INT4 (each halves memory, improves throughput)"
      batching: "Combine multiple requests into single GPU call (2-10x throughput)"
      model_distillation: "Train smaller model to mimic large model (90% quality, 10% cost)"
      caching: "Cache common predictions (eliminates repeated inference)"
      speculative_decoding: "Use small model for draft tokens, large model for verification"
```

### LLM API Cost Analysis

```yaml
LLM_API_Costs:
  pricing_models:
    per_token:
      description: "Pay per input/output token processed"
      examples_2026:
        gpt_4o: "$2.50 input / $10 output per 1M tokens"
        gpt_4o_mini: "$0.15 input / $0.60 output per 1M tokens"
        claude_4_sonnet: "$3 input / $15 output per 1M tokens"
        claude_4_opus: "$15 input / $75 output per 1M tokens"
        gemini_2_5_pro: "$1.25 input / $10 output per 1M tokens"
      
    self_hosted:
      description: "Run open models on your own infrastructure"
      examples:
        llama_4_8b: "~$0.10-0.30 per 1M tokens (on A100, optimized with vLLM)"
        llama_4_70b: "~$0.50-1.50 per 1M tokens (on H100 cluster)"
        mistral_large: "~$0.30-1.00 per 1M tokens"
      break_even: "Self-hosted becomes cheaper at ~$10K-50K/month API spend"
      
  cost_scenarios:
    customer_support_chatbot:
      volume: "10,000 conversations/day, ~2000 tokens each"
      tokens_per_month: "600M tokens/month"
      api_cost_gpt4o: "$6,000-8,000/month"
      api_cost_mini: "$360-500/month"
      self_hosted_llama: "$200-600/month (+ infra management)"
      
    document_processing:
      volume: "50,000 documents/day, ~5000 tokens each"
      tokens_per_month: "7.5B tokens/month"
      api_cost_gpt4o: "$75,000/month"
      optimization: "Use cheap model for classification, expensive for complex extraction"
      optimized_cost: "$15,000/month (tiered approach)"
      
  cost_reduction_strategies:
    prompt_optimization:
      description: "Shorter prompts = fewer tokens = lower cost"
      savings: "20-50% by removing verbose instructions"
      technique: "Compress prompts while maintaining quality"
      
    model_routing:
      description: "Route simple queries to cheap model, complex to expensive"
      implementation: "Classifier determines complexity, routes to appropriate model"
      savings: "50-70% (most queries are simple enough for small models)"
      
    caching:
      description: "Cache identical or semantically similar requests"
      tools: "Redis, GPTCache, semantic similarity caching"
      savings: "30-60% depending on request repetition"
      
    batching:
      description: "Process multiple requests together for throughput efficiency"
      savings: "Lower per-request infrastructure cost for batch workloads"
```

### Total Cost of Ownership (TCO)

```yaml
TCO_Framework:
  example_project: "Fraud detection model serving 50M predictions/day"
  
  year_1_costs:
    people:
      ml_engineers: "$600K (2 engineers × $300K total comp)"
      data_engineers: "$300K (1 engineer × $300K)"
      mlops: "$350K (1 engineer × $350K)"
      subtotal: "$1.25M"
      
    compute:
      training: "$24K (weekly retraining, $500/run × 48 runs)"
      experimentation: "$36K (exploration, failed experiments)"
      inference_serving: "$180K (GPU inference, $15K/month)"
      subtotal: "$240K"
      
    data:
      storage: "$12K (features, models, training data)"
      labeling: "$50K (fraud label verification)"
      third_party_data: "$100K (transaction enrichment APIs)"
      subtotal: "$162K"
      
    infrastructure:
      platform_tools: "$36K (MLflow, monitoring, orchestration hosting)"
      cloud_networking: "$18K"
      subtotal: "$54K"
      
    total_year_1: "$1.7M"
    cost_breakdown: "People 74% | Compute 14% | Data 10% | Infrastructure 3%"
    key_insight: "People cost dominates. Invest in tools that multiply engineer productivity."
    
  roi_analysis:
    fraud_prevented: "$50M/year (model catches fraud that rules missed)"
    investment: "$1.7M/year"
    roi: "29x return"
    conclusion: "Even expensive ML projects can have massive ROI when applied to high-value problems"
```

---

## How It Works in Practice

### Cost Optimization Example

```yaml
Example:
  situation: "Team spending $45K/month on LLM inference for customer support chatbot"
  
  analysis:
    current_state:
      model: "GPT-4o for all queries"
      volume: "500K conversations/month"
      avg_tokens: "3000 per conversation (input + output)"
      monthly_cost: "$45K"
      
    optimization_steps:
      step_1_model_routing:
        action: "Classify query complexity, route 70% to GPT-4o-mini"
        savings: "70% of traffic × 85% cost reduction = 60% savings"
        new_cost: "$18K"
        
      step_2_prompt_optimization:
        action: "Compress system prompts from 800 to 300 tokens"
        savings: "15% token reduction across all conversations"
        new_cost: "$15.3K"
        
      step_3_caching:
        action: "Cache responses for FAQ-type questions (25% of traffic)"
        savings: "25% of remaining compute eliminated"
        new_cost: "$11.5K"
        
      step_4_self_hosted_for_simple:
        action: "Move simple queries to self-hosted Llama-4-8B"
        savings: "Replace 40% of mini traffic with near-zero marginal cost"
        new_cost: "$8.5K (+ $2K infrastructure)"
        
    result:
      original: "$45K/month"
      optimized: "$10.5K/month"
      savings: "$34.5K/month (77% reduction)"
      quality_impact: "Minimal — monitored quality scores remained within 2% of original"
```

---

## Interview Tip

> When asked about ML costs: "ML cost has four major categories: people (typically 60-80% — salaries), compute (training and inference), data (collection, labeling, storage), and infrastructure (platform tools, networking). The key insight is that inference cost compounds at scale — a $0.01 per-prediction model serving 100M predictions/day is $1M/year. My optimization approach: (1) Right-size GPU to workload (don't use H100 for XGBoost). (2) Spot instances for training (60-90% savings with checkpointing). (3) Quantization and batching for inference (2-4x throughput). (4) Model routing for LLMs — classify query complexity and route cheap queries to small models. (5) Caching for repetitive predictions. For LLM applications, the architecture choice (API vs self-hosted, model size routing) determines whether cost is $500/month or $50K/month for the same use case."

---

## Common Mistakes

1. **Ignoring inference cost during model selection** — Choosing the biggest, most accurate model without calculating serving cost at production scale. A model that's 2% more accurate but 10x more expensive to serve is rarely the right choice.

2. **Running GPUs 24/7 for batch workloads** — Keeping GPU instances running continuously for training jobs that run a few hours per day or week. Use spot instances or auto-scaling to zero.

3. **Not monitoring GPU utilization** — GPUs sitting at 10-20% utilization because of single-request inference without batching. Proper batching can achieve 80%+ utilization and 4-8x throughput.

4. **Using expensive LLM APIs for simple tasks** — Routing every query through GPT-4o when 70% of queries could be handled by GPT-4o-mini or a fine-tuned small model at 1/20th the cost.

5. **Forgetting the experiment tax** — Budgeting only for the final successful training run, not the 10-50 failed experiments that precede it. True training cost = successful_run × experiment_ratio.

---

## Key Takeaways

- People cost dominates ML budgets (60-80%), but compute gets all the attention
- Inference cost compounds: cost_per_request × QPS × 24/7 — optimize relentlessly
- GPU right-sizing: match hardware to workload (T4 for small inference, H100 for LLM training)
- Spot instances: 60-90% savings for training (with checkpointing for fault tolerance)
- LLM cost optimization: model routing + prompt compression + caching = 50-80% savings
- Quantization (FP16/INT8/INT4) reduces memory and improves throughput with minimal quality loss
- Self-hosted LLMs break even vs APIs at ~$10K-50K/month spend
- Always calculate TCO (Total Cost of Ownership) including people, not just cloud bills
- ROI must justify the investment — even expensive ML can deliver 10-100x returns on high-value problems
- The experiment tax is real: budget for 10-50 failed runs per successful model
