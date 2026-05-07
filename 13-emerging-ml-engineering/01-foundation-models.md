# Foundation Models

## The Problem / Why This Matters

Foundation models have fundamentally changed how ML engineers build AI systems. Instead of training task-specific models from scratch (image classifier trained only on your data, NLP model trained only on your corpus), you now start with a pre-trained foundation model that has already learned general capabilities from massive datasets — then adapt it to your specific task. In 2026, foundation models (GPT-5, Claude 4, Gemini 2.5, Llama 3, Mistral Large, Qwen 2.5) power everything from chatbots to code generation to medical diagnosis to autonomous agents. For ML engineers, the paradigm shift is profound: instead of "how do I train a model?" the question becomes "how do I adapt a foundation model?" This requires understanding: adaptation techniques (prompting, RAG, fine-tuning, RLHF), when each technique applies, how to evaluate adapted models, and how to serve them efficiently. Foundation models are also becoming multi-modal (text + image + video + audio + code in a single model), enabling capabilities that were impossible with single-modality models. Understanding foundation models — their capabilities, limitations, adaptation patterns, and infrastructure requirements — is now the core competency of ML engineering.

---

## The Analogy

Think of foundation models like a university education:

- **Pre-training** = A student completing a comprehensive 4-year degree. They learn general knowledge (language, reasoning, world knowledge) from massive amounts of study material (internet-scale data). Expensive and time-consuming ($10M-$100M, months of compute), but produces a generally capable individual.
- **Prompting** = Giving the graduate a specific job description. They use their general education to handle the task. No additional training — just clear instructions. Works for many tasks immediately.
- **RAG (Retrieval-Augmented Generation)** = Giving the graduate a reference book to consult during work. They combine their general knowledge with specific up-to-date information from the reference (retrieved documents).
- **Fine-tuning** = Sending the graduate to a professional certification program. Intensive, focused training on specific skills. They become specialized (medical, legal, financial) while retaining general knowledge.
- **RLHF** = An internship with mentor feedback. The graduate learns what "good work" looks like through human evaluation and adjusts behavior accordingly.

---

## Deep Dive

### Foundation Model Landscape (2026)

```yaml
Foundation_Model_Landscape:
  frontier_models:
    gpt_5:
      provider: "OpenAI (via Azure OpenAI)"
      capabilities: "Best general reasoning, code, creative writing"
      context: "256K tokens"
      modalities: "Text, image, audio input; text, image output"
      access: "API only (closed-weight)"
      
    o3_o4:
      provider: "OpenAI (reasoning models)"
      capabilities: "Multi-step reasoning, math, science, coding challenges"
      approach: "Extended thinking time (chain-of-thought at inference)"
      use_case: "Complex problems requiring step-by-step reasoning"
      
    claude_4_opus:
      provider: "Anthropic (via Bedrock, direct API)"
      capabilities: "Strongest instruction following, analysis, coding"
      context: "200K tokens"
      strengths: "Safety, nuanced understanding, long documents"
      
    gemini_2_5_pro:
      provider: "Google (via Vertex AI)"
      capabilities: "Multi-modal (text, image, video, audio, code)"
      context: "2M tokens (largest context window)"
      strengths: "Multi-modal understanding, massive context, Google Search grounding"
      
  open_weight_models:
    llama_3:
      provider: "Meta"
      sizes: "8B, 70B, 405B parameters"
      license: "Open (community license, commercial use allowed)"
      strengths: "Strong general performance, large community, fine-tunable"
      
    mistral_large:
      provider: "Mistral AI"
      architecture: "Mixture of Experts (MoE)"
      strengths: "Efficient (activates subset of parameters), multilingual"
      
    qwen_2_5:
      provider: "Alibaba"
      sizes: "7B, 14B, 32B, 72B"
      strengths: "Strong coding and math, multilingual (CJK excellent)"
      
    gemma_2:
      provider: "Google (open-weight)"
      sizes: "2B, 9B, 27B"
      strengths: "Small but capable, optimized for on-device"
      
    phi_4:
      provider: "Microsoft"
      sizes: "3.8B, 14B"
      strengths: "Extremely efficient (small model, high capability)"
      
  specialized_models:
    code:
      - "DeepSeek Coder V2 (236B MoE, 21B active)"
      - "CodeLlama (specialized Llama for code)"
      - "StarCoder2 (BigCode, 15B parameters)"
    medical:
      - "MedPaLM 2 (Google, medical QA)"
      - "BioMistral (biomedical fine-tune)"
    vision:
      - "SAM 2 (Segment Anything Model — universal segmentation)"
      - "CLIP (image-text understanding)"
      - "Stable Diffusion 3 (image generation)"
    speech:
      - "Whisper v3 (speech-to-text)"
      - "Bark (text-to-speech)"
      - "MusicGen (music generation)"
```

### Adaptation Techniques

```yaml
Adaptation_Techniques:
  spectrum: "Prompting → RAG → Fine-tuning → RLHF → Full training"
  
  prompting:
    what: "Adapt model behavior through instructions (no weight changes)"
    types:
      zero_shot: "Direct instruction, no examples"
      few_shot: "Provide 3-10 examples in context"
      chain_of_thought: "Instruct model to reason step-by-step"
      system_prompting: "Define role, personality, constraints"
    cost: "$0 training, per-token inference cost"
    effort: "Minutes to hours"
    when_to_use:
      - "Testing if task is feasible"
      - "General-purpose tasks (summarization, QA, translation)"
      - "Low volume (not worth fine-tuning)"
      - "Need to change behavior frequently"
    limitations:
      - "Context window limits (can't fit extensive knowledge)"
      - "Inconsistent compliance with complex instructions"
      - "No deep specialization (surface-level adaptation)"
      
  rag:
    what: "Retrieve relevant documents and include in context"
    components:
      indexing: "Chunk documents → embed → store in vector database"
      retrieval: "Embed query → similarity search → top-K documents"
      generation: "Include retrieved docs in prompt → model generates answer"
    cost: "Vector database hosting + per-query retrieval + extra input tokens"
    effort: "Days to weeks (build pipeline)"
    when_to_use:
      - "Knowledge base that changes frequently"
      - "Need citations/sources for answers"
      - "Large knowledge base (can't fit in context)"
      - "Domain-specific data (company docs, product info)"
    limitations:
      - "Retrieval quality caps generation quality"
      - "Latency overhead (retrieval + longer prompts)"
      - "Doesn't change model behavior/style"
      
  fine_tuning:
    what: "Update model weights on task-specific data"
    methods:
      full_fine_tuning:
        what: "Update all model parameters"
        data_needed: "10K-100K+ examples"
        cost: "High (full model in memory + optimizer states)"
        when: "Maximum performance needed, sufficient data exists"
        
      lora:
        what: "Low-Rank Adaptation — train small adapter matrices"
        data_needed: "1K-10K examples"
        cost: "Low (only adapter weights trained, ~1-5% of model)"
        parameters_trained: "0.1-2% of total model parameters"
        when: "Most fine-tuning scenarios (best effort/quality trade-off)"
        
      qlora:
        what: "LoRA on quantized base model (4-bit)"
        benefit: "Fine-tune 70B model on single A100 80GB"
        quality: "95-99% of full fine-tuning quality"
        when: "Large models, limited GPU budget"
        
    cost: "$50-5000 (depending on model size and data volume)"
    effort: "Days (data preparation + training + evaluation)"
    when_to_use:
      - "Consistent output format/style required"
      - "Domain-specific terminology"
      - "High volume (amortize training over many requests)"
      - "Better quality than prompting achieves"
    limitations:
      - "Requires labeled training data"
      - "Risk of catastrophic forgetting (lose general capabilities)"
      - "Needs re-training as data changes"
      
  rlhf_dpo:
    what: "Align model with human preferences"
    methods:
      rlhf:
        what: "Reinforcement Learning from Human Feedback"
        process: "Collect preferences → train reward model → RL optimization"
        complexity: "High (reward model + PPO training)"
        
      dpo:
        what: "Direct Preference Optimization"
        process: "Collect preference pairs → train directly (no reward model)"
        advantage: "Simpler than RLHF, similar results"
        data: "Preference pairs: (prompt, chosen_response, rejected_response)"
        
    when_to_use:
      - "Need to align model with human preferences"
      - "Reduce harmful/unhelpful outputs"
      - "Make model outputs more natural/conversational"
    cost: "High (human annotation for preferences)"
```

### Implementation Patterns

```python
# Foundation model adaptation patterns

"""
Implementation patterns for adapting foundation models to specific tasks.
Covers: prompting, RAG, fine-tuning, and evaluation.
"""

adaptation_patterns = {
    "prompting_patterns": {
        "system_prompt_engineering": {
            "description": "Structured system prompt for consistent behavior",
            "template": """You are a {role} assistant that helps with {domain}.

Rules:
1. Always respond in {format}
2. If you don't know the answer, say "I don't have enough information"
3. Cite sources when providing factual claims
4. Keep responses under {max_words} words

Style: {tone} (e.g., professional, conversational, technical)
""",
            "best_practices": [
                "Be specific about output format (JSON, markdown, bullet points)",
                "Include explicit constraints (what NOT to do)",
                "Provide role context (who the model is pretending to be)",
                "Set boundaries (topics to refuse, length limits)",
            ],
        },
        
        "few_shot_pattern": {
            "description": "Provide examples to establish pattern",
            "template": """Classify the following support ticket into categories: billing, technical, account, other.

Example 1:
Input: "I was charged twice for my subscription"
Output: {"category": "billing", "confidence": 0.95}

Example 2:
Input: "I can't log into my account"
Output: {"category": "account", "confidence": 0.90}

Now classify:
Input: "{user_input}"
Output:""",
            "rules": [
                "Use 3-5 diverse examples (cover edge cases)",
                "Examples should match expected input distribution",
                "Include output format in examples (model will follow)",
                "Order: easy → hard (curriculum effect)",
            ],
        },
    },
    
    "rag_architecture": {
        "basic_rag": {
            "flow": [
                "1. User query arrives",
                "2. Embed query with text-embedding-3-large",
                "3. Vector search in Pinecone/Weaviate (top-5 chunks)",
                "4. Construct prompt: system + retrieved chunks + user query",
                "5. Generate response with GPT-4o/Claude",
                "6. Return response with citations",
            ],
            "chunking_strategy": {
                "chunk_size": "500-1000 tokens",
                "overlap": "100-200 tokens (preserve context at boundaries)",
                "method": "Semantic chunking (split by topic, not fixed size)",
            },
        },
        
        "advanced_rag": {
            "techniques": [
                {
                    "name": "Hybrid search",
                    "what": "Combine vector similarity + keyword matching (BM25)",
                    "benefit": "Captures both semantic and exact-match relevance",
                },
                {
                    "name": "Re-ranking",
                    "what": "Retrieve top-20, re-rank with cross-encoder to top-5",
                    "benefit": "Higher relevance in final context (better answers)",
                },
                {
                    "name": "Query expansion",
                    "what": "Generate multiple queries from user input, retrieve for each",
                    "benefit": "Broader coverage (user query may miss relevant docs)",
                },
                {
                    "name": "Contextual compression",
                    "what": "Extract only relevant sentences from retrieved chunks",
                    "benefit": "Less noise in context, lower token cost",
                },
                {
                    "name": "Agentic RAG",
                    "what": "Agent decides when/what to retrieve iteratively",
                    "benefit": "Multi-step reasoning with targeted retrieval",
                },
            ],
        },
    },
    
    "fine_tuning_pipeline": {
        "data_preparation": {
            "format": "JSONL with instruction/input/output or messages format",
            "quality_over_quantity": "1000 high-quality examples > 10000 noisy ones",
            "data_collection": [
                "Human-written examples (gold standard)",
                "LLM-generated + human-verified (scalable)",
                "Production logs (real user interactions, filtered)",
            ],
            "validation": {
                "split": "80/10/10 (train/val/test)",
                "deduplication": "Remove near-duplicate examples",
                "diversity": "Cover all expected input types",
            },
        },
        
        "training_config": {
            "lora_config": {
                "r": 16,  # Rank (higher = more capacity, more compute)
                "lora_alpha": 32,  # Scaling factor (typically 2×r)
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "lora_dropout": 0.05,
                "task_type": "CAUSAL_LM",
            },
            "training_args": {
                "learning_rate": 2e-4,
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 8,
                "warmup_ratio": 0.03,
                "lr_scheduler_type": "cosine",
                "bf16": True,
                "max_grad_norm": 0.3,
            },
        },
        
        "evaluation": {
            "automated_metrics": [
                "Task-specific accuracy (classification, extraction)",
                "ROUGE/BLEU (generation quality)",
                "Perplexity (model confidence)",
                "Format compliance (% outputs matching expected format)",
            ],
            "human_evaluation": [
                "Win rate vs. base model (A/B comparison)",
                "Quality scoring (1-5 scale on random samples)",
                "Safety check (no harmful outputs introduced)",
            ],
            "regression_testing": [
                "General capability benchmarks (ensure no catastrophic forgetting)",
                "Safety benchmarks (model doesn't become harmful)",
                "Edge case test suite (specific failure modes to check)",
            ],
        },
    },
}


# Foundation model selection decision framework
model_selection = {
    "decision_factors": {
        "quality": "Does the model produce good outputs for your task?",
        "latency": "Is response time acceptable for your use case?",
        "cost": "Can you afford per-token pricing at your volume?",
        "privacy": "Can your data go to a third-party API?",
        "customization": "Do you need fine-tuning or specific behaviors?",
        "context_length": "How much context does your task require?",
    },
    
    "decision_tree": {
        "need_reasoning": {
            True: "o3/o4 (OpenAI reasoning models)",
            False: "Continue to next question",
        },
        "need_multi_modal": {
            True: "Gemini 2.5 Pro (best multi-modal) or GPT-4o",
            False: "Continue",
        },
        "need_large_context": {
            ">200K tokens": "Gemini 2.5 Pro (2M context)",
            "32K-200K": "Claude 4 (200K) or GPT-5 (256K)",
            "<32K": "Any model works",
        },
        "need_privacy": {
            "data_cannot_leave_infra": "Self-hosted (Llama 3, Mistral, Qwen)",
            "enterprise_controls_ok": "Azure OpenAI or Bedrock (data stays in your cloud)",
        },
        "cost_sensitive": {
            True: "Open models (Llama 3 70B self-hosted) or smaller models (GPT-4o-mini)",
            False: "Frontier models (GPT-5, Claude 4 Opus, Gemini Pro)",
        },
    },
}
```

---

## How It Works in Practice

### Foundation Model Adaptation Strategy

```yaml
Foundation_Model_Strategy:
  scenario: "Build AI-powered legal document analysis system"
  
  approach:
    phase_1_prompting:
      duration: "1 week"
      what: "Test if base model can handle task with prompting alone"
      model: "Claude 4 Sonnet (strong at document analysis)"
      results:
        accuracy: "78% on test set"
        issues: "Inconsistent output format, misses domain-specific terms"
        cost: "$0/training, $2K/month inference (estimated volume)"
      decision: "Good but not production-ready — needs improvement"
      
    phase_2_rag:
      duration: "2 weeks"
      what: "Add legal knowledge base for domain-specific terms and precedents"
      additions:
        - "Index 50K legal documents in vector database"
        - "Retrieve relevant case law and definitions per query"
        - "Add citation generation"
      results:
        accuracy: "88% (improved from domain knowledge)"
        issues: "Still inconsistent format for some edge cases"
        cost: "$500/month vector DB + $3K/month inference (more tokens)"
      decision: "Better — RAG handles knowledge gaps. Fine-tune for consistency."
      
    phase_3_fine_tuning:
      duration: "1 week"
      what: "Fine-tune for consistent output format and domain terminology"
      approach:
        base_model: "Llama 3 70B (open-weight, can self-host)"
        method: "QLoRA (4-bit base, train adapters)"
        data: "2000 expert-annotated legal analyses (high quality)"
        training_cost: "$200 (8 hours on spot H100)"
      results:
        accuracy: "94% (best overall)"
        format_compliance: "99% (consistent output structure)"
        cost: "$2K/month (self-hosted on 2× H100)"
      decision: "Production-ready — fine-tuned model with RAG augmentation"
      
  final_architecture:
    model: "Fine-tuned Llama 3 70B (QLoRA) + RAG"
    serving: "vLLM on 2× H100 (self-hosted)"
    knowledge: "Vector database with 50K legal documents"
    cost: "$2.5K/month total (cheaper than API at this volume)"
    quality: "94% accuracy, 99% format compliance"
```

---

## Interview Tip

> When asked about foundation models: "I approach foundation model adaptation as a spectrum — always start simple and escalate complexity only when needed. My framework: (1) Start with prompting — test if the base model can handle the task with good instructions and few-shot examples. This takes hours, not weeks. (2) Add RAG if the model needs domain knowledge it wasn't trained on. This handles knowledge gaps without retraining. (3) Fine-tune only if prompting + RAG isn't sufficient — typically for consistent output formats, domain-specific language, or when you need to self-host for cost/privacy. I use LoRA/QLoRA for fine-tuning (trains 1-2% of parameters, fits on single GPU). For model selection: reasoning tasks → o3/o4, multi-modal → Gemini 2.5 Pro, strong instruction-following → Claude 4, privacy-sensitive → self-hosted Llama 3/Mistral. Cost drives architecture: below $5K/month API spend, managed APIs are simpler. Above $5K/month, self-hosted with vLLM becomes cost-effective. Key evaluation practice: maintain a test suite that measures quality after every change (prompt update, model swap, fine-tuning). Without systematic evaluation, you're flying blind."

---

## Common Mistakes

1. **Jumping to fine-tuning first** — Spending weeks collecting data and fine-tuning before trying prompting. Often good prompting achieves 80%+ of fine-tuning quality immediately. Solution: always start with prompting. Iterate on prompts for 2-3 days before deciding fine-tuning is needed.

2. **RAG without evaluation** — Building complex RAG pipeline without measuring retrieval quality. Garbage retrieval = garbage generation regardless of model quality. Solution: evaluate retrieval independently (precision@K, recall@K) before evaluating end-to-end generation.

3. **Fine-tuning on noisy data** — Collecting 10K training examples without quality control. Model learns bad patterns from noisy labels. Solution: 1K high-quality examples > 10K noisy ones. Human-verify a random sample. Clean data ruthlessly.

4. **Catastrophic forgetting after fine-tuning** — Fine-tune on narrow task, model loses general capabilities. Can't handle anything outside the fine-tuning distribution. Solution: use LoRA (preserves base model), include diverse general examples in training mix, and test general benchmarks after fine-tuning.

5. **Ignoring context window costs** — Stuffing 100K tokens of context into every request (because the model supports it). $5+ per request with expensive models. Solution: only include relevant context. Use RAG to select specific relevant chunks instead of dumping everything into context.

---

## Key Takeaways

- Foundation models: pre-trained on massive data, adapted to specific tasks (prompting, RAG, fine-tuning)
- Adaptation spectrum: prompting (cheapest, fastest) → RAG (knowledge) → fine-tuning (specialization) → RLHF (alignment)
- Always start with prompting — escalate only when quality is insufficient
- RAG: for knowledge the model doesn't have (changing data, domain-specific, proprietary)
- Fine-tuning: for consistent behavior (format, style, terminology) — use LoRA/QLoRA
- Model selection: reasoning (o3/o4), multi-modal (Gemini), privacy (self-hosted Llama/Mistral)
- Open-weight models: Llama 3, Mistral, Qwen, Gemma — fine-tunable, self-hostable
- Self-hosted becomes cost-effective: above ~$5K/month in API spend
- Evaluation is critical: maintain test suite, measure after every change
- Multi-modal: single models now handle text + image + video + audio (Gemini, GPT-5)
