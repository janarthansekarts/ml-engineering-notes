# Fine-Tuning Operations

## The Problem / Why This Matters

Prompt engineering has limits. When you need a model to consistently output in a specific format, follow domain-specific conventions, use proprietary terminology, or achieve quality that prompting alone can't reach, fine-tuning is the answer. Fine-tuning adapts a pre-trained model to your specific use case by training on your task-specific data. But fine-tuning is not just "run training" — it's an operational discipline: when to fine-tune (vs prompt engineering vs RAG), how to prepare data (quality matters more than quantity), which technique to use (full fine-tuning vs LoRA vs QLoRA), how to evaluate (offline metrics that predict online performance), and how to deploy and monitor the fine-tuned model. In 2026, LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) have made fine-tuning accessible — you can fine-tune a 70B model on a single H100 in hours, not weeks. The democratization of fine-tuning means the operational aspects (data quality, evaluation, deployment) are now the bottleneck, not compute. Getting fine-tuning ops right means reliably improving model quality for your use case while managing the risks of catastrophic forgetting, overfitting, and deployment complexity.

---

## The Analogy

Think of fine-tuning like training a new employee:

- **Pre-trained model** = A new hire with excellent general education (university degree). They know how to think, write, and reason — but don't know your company's specific products, processes, or style.
- **Prompt engineering** = Giving them a detailed instruction manual for each task. Works for simple tasks but they constantly need to refer to the manual (prompt consumes tokens). For complex patterns, the manual becomes impossibly long.
- **RAG** = Giving them access to a knowledge base they can search. Great for factual queries but doesn't change how they write, reason, or format outputs.
- **Fine-tuning** = Actual on-the-job training. After training, they naturally speak in your company's voice, follow your formats without being told, and handle domain-specific scenarios correctly. They've internalized the patterns.
- **LoRA** = Teaching them a specific skill (Excel, Python) without changing their personality or general knowledge. A lightweight "add-on" to their existing capabilities.

---

## Deep Dive

### When to Fine-Tune vs Alternatives

```yaml
Decision_Framework:
  use_prompt_engineering_when:
    - "Task can be described clearly in instructions"
    - "You have < 50 examples of desired behavior"
    - "Output format is simple (the model already knows how)"
    - "You need to iterate quickly (prompt changes instantly)"
    - "You want to switch models easily (prompts are somewhat portable)"
    
  use_rag_when:
    - "Model needs access to specific factual knowledge"
    - "Knowledge changes frequently (RAG updates without retraining)"
    - "Answers should be grounded in source documents"
    - "You need citations and verifiability"
    
  use_fine_tuning_when:
    - "Specific output format/style that prompting can't reliably produce"
    - "Domain-specific behavior/terminology (medical, legal, finance)"
    - "Consistent persona/voice across all interactions"
    - "Reducing prompt length (bake instructions into weights → save tokens)"
    - "Improving quality on specific task beyond what prompting achieves"
    - "Latency reduction (shorter prompts = faster TTFT)"
    - "You have 100-10,000 high-quality training examples"
    
  combine_approaches:
    fine_tune_plus_rag: "Fine-tune for style/format + RAG for factual grounding"
    fine_tune_plus_prompt: "Fine-tune for base behavior + prompt for per-request specifics"
    example: |
      Fine-tune Llama-8B on your support conversation style (consistent tone, format)
      + RAG to retrieve relevant product documentation (factual grounding)
      + Prompt to set per-customer context (account type, history)
```

### LoRA and QLoRA

```yaml
LoRA:
  full_name: "Low-Rank Adaptation"
  what: |
    Instead of updating ALL model parameters (full fine-tuning),
    add small trainable matrices (adapters) alongside frozen pre-trained weights.
    These adapters capture task-specific knowledge with <1% of total parameters.
    
  how_it_works:
    concept: |
      Original weight matrix W (4096 × 4096 = 16.7M parameters)
      LoRA decomposes the update as: ΔW = A × B
      where A is 4096 × 16 and B is 16 × 4096 (rank 16)
      Total adapter parameters: 4096 × 16 + 16 × 4096 = 131K (0.8% of original)
      
    math: |
      Forward pass: output = (W + α × A × B) × input
      Only A and B are trained (W is frozen)
      α (alpha) controls the strength of adaptation
      Rank r controls capacity (higher = more parameters, more expressive)
      
  benefits:
    memory: "90-99% less GPU memory than full fine-tuning"
    speed: "2-5× faster training"
    storage: "Adapter is 10-100 MB (vs 14-140 GB for full model)"
    no_catastrophic_forgetting: "Base model frozen — general knowledge preserved"
    multi_adapter: "Multiple LoRA adapters on one base model (multi-task, multi-tenant)"
    
  hyperparameters:
    rank_r:
      what: "Rank of adaptation matrices (capacity of the adapter)"
      typical: "8-64 (most use 16-32)"
      guidance: "Start with 16. Increase if underfitting, decrease if overfitting."
      
    alpha:
      what: "Scaling factor for LoRA update"
      typical: "16-64 (often set to 2× rank)"
      effect: "Higher alpha = stronger adaptation. Too high = unstable training."
      rule: "alpha / rank ≈ 1-2 (common heuristic)"
      
    target_modules:
      what: "Which layers to apply LoRA to"
      recommendation: "All linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)"
      minimal: "Just attention projections (q_proj, v_proj) — faster but less expressive"
      
  qlora:
    full_name: "Quantized LoRA"
    what: "Load base model in 4-bit (NF4) quantization, train LoRA adapters in FP16"
    benefit: "Fine-tune 70B model on single 24 GB GPU (vs 4× 80 GB for full fine-tuning)"
    quality: "< 1% quality loss vs full fine-tuning (empirical across many tasks)"
    implementation: "bitsandbytes library (4-bit quantization) + PEFT (LoRA training)"
```

### Fine-Tuning Data Preparation

```yaml
Data_Preparation:
  data_quality:
    key_insight: "100 high-quality examples > 10,000 low-quality examples"
    quality_criteria:
      - "Correct: answers are factually accurate and task-appropriate"
      - "Consistent: same style, format, and quality across examples"
      - "Representative: covers the range of inputs the model will see"
      - "Clean: no formatting errors, truncations, or corrupted text"
      
  data_format:
    chat_format: |
      [
        {
          "messages": [
            {"role": "system", "content": "You are a helpful medical assistant..."},
            {"role": "user", "content": "What are the symptoms of Type 2 diabetes?"},
            {"role": "assistant", "content": "The main symptoms of Type 2 diabetes include..."}
          ]
        },
        ...
      ]
      
    instruction_format: |
      [
        {
          "instruction": "Summarize this medical research paper in 3 bullet points.",
          "input": "Background: This study examined...",
          "output": "• Finding 1: ...\n• Finding 2: ...\n• Finding 3: ..."
        },
        ...
      ]
      
  data_volume_guidelines:
    minimum_viable: "50-100 examples (if high quality)"
    recommended: "500-2,000 examples (good generalization)"
    diminishing_returns: "> 10,000 examples (may overfit or waste compute)"
    per_task: "100-500 examples per distinct task/behavior you want to teach"
    
  data_generation_techniques:
    human_curation:
      what: "Domain experts write ideal input-output pairs"
      quality: "Highest (experts know correct behavior)"
      cost: "Expensive, slow"
      
    production_data:
      what: "Filter best real interactions (human-rated, successful completions)"
      quality: "High (real-world patterns)"
      volume: "Large (if you track user feedback)"
      
    synthetic_generation:
      what: "Use a stronger model (GPT-5, Claude Opus) to generate training data"
      approach: |
        1. Define the behavior you want
        2. Write 10-20 seed examples manually
        3. Ask GPT-5/Claude to generate 500+ similar examples
        4. Human review and filter for quality (critical step)
      quality: "Medium-high (needs human validation)"
      volume: "Scalable"
      
    distillation:
      what: "Run production queries through a large model, use outputs to train small model"
      approach: "Large model (teacher) generates high-quality outputs → train small model (student)"
      legal: "Check model provider TOS (some prohibit using outputs for training competitors)"
```

### Fine-Tuning Training

```python
# QLoRA fine-tuning with Hugging Face PEFT and TRL

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

# 1. Load model in 4-bit quantization (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 (optimal for QLoRA)
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in BF16
    bnb_4bit_use_double_quant=True,        # Double quantization (saves memory)
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-4-8B",
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-8B")
tokenizer.pad_token = tokenizer.eos_token

# 2. Prepare model for QLoRA training
model = prepare_model_for_kbit_training(model)

# 3. Configure LoRA
lora_config = LoraConfig(
    r=32,                    # Rank (capacity of adaptation)
    lora_alpha=64,           # Scaling factor (alpha/r = 2)
    target_modules=[         # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,       # Regularization
    bias="none",             # Don't train bias terms
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Typically < 1% of total

# 4. Load training data
dataset = load_dataset("json", data_files="training_data.jsonl")

# 5. Training configuration
training_config = SFTConfig(
    output_dir="./lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,     # Effective batch size: 16
    learning_rate=2e-4,                # Higher than full fine-tuning
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    max_seq_length=2048,
)

# 6. Train
trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
)

trainer.train()

# 7. Save LoRA adapter (small file: 50-200 MB)
model.save_pretrained("./lora-adapter")
# Deploy: load base model + merge adapter, or serve with multi-LoRA (vLLM)
```

### Evaluation and Deployment

```yaml
Evaluation:
  offline_evaluation:
    held_out_test_set:
      what: "Test data never seen during training"
      size: "10-20% of total data"
      metrics: "Same task-specific metrics used for production monitoring"
      
    comparison_matrix:
      compare_against:
        - "Base model (no fine-tuning) + prompt engineering"
        - "Base model + RAG"
        - "Previous fine-tuned version"
        - "Larger model with just prompting (is fine-tuning a small model better than prompting a large one?)"
        
    task_specific_metrics:
      classification: "F1, precision, recall per class"
      generation: "ROUGE, BLEU, BERTScore, human eval"
      instruction_following: "Format compliance rate, constraint adherence"
      domain_knowledge: "Factual accuracy on domain quiz"
      
    regression_testing:
      what: "Verify fine-tuning didn't degrade general capabilities"
      test: "Run standard benchmarks (MMLU subset, HumanEval, etc.) on fine-tuned model"
      concern: "Catastrophic forgetting — model loses general knowledge"
      
  online_evaluation:
    a_b_test:
      what: "Deploy fine-tuned model to 10% of traffic, compare vs base+prompt"
      metrics: "User satisfaction, task completion, business metrics"
      duration: "7-14 days for statistical significance"
      
  deployment:
    merge_adapter:
      what: "Merge LoRA weights into base model (single model file)"
      use: "When only serving one fine-tuned variant"
      
    multi_lora_serving:
      what: "Serve base model + multiple LoRA adapters simultaneously (vLLM)"
      use: "Multi-tenant (different customers), multi-task, A/B testing variants"
      advantage: "One base model in memory, swap adapters per-request"
      
    versioning:
      track:
        - "Training data version (which data was used)"
        - "Base model version (which pre-trained model)"
        - "Hyperparameters (rank, alpha, learning rate, epochs)"
        - "Evaluation scores (on which metrics, which test set)"
      registry: "MLflow, Weights & Biases, HuggingFace Hub"
```

---

## How It Works in Practice

### Fine-Tuning Operations Workflow

```yaml
Workflow:
  trigger: "When prompt engineering plateau is reached or specific behavior needed"
  
  step_1_data:
    actions:
      - "Collect 500-2000 high-quality examples"
      - "Human review for consistency and quality"
      - "Split: 80% train, 10% validation, 10% test"
      - "Version data in data registry"
    duration: "1-2 weeks (data quality is the bottleneck)"
    
  step_2_train:
    actions:
      - "Select base model (Llama-8B for efficiency, 70B for quality)"
      - "Configure QLoRA (rank=32, alpha=64, all linear layers)"
      - "Train for 3 epochs, monitoring eval loss"
      - "Save checkpoints every 100 steps"
    duration: "1-8 hours (depending on model size and data)"
    
  step_3_evaluate:
    actions:
      - "Run test set through fine-tuned model"
      - "Compare against base model + prompt engineering baseline"
      - "Run regression tests (general capability preserved?)"
      - "If improvement < 5%: iterate on data or hyperparameters"
    gate: "Must show measurable improvement on task-specific metrics"
    
  step_4_deploy:
    actions:
      - "Deploy as canary (10% traffic via multi-LoRA or separate endpoint)"
      - "Monitor online metrics vs control"
      - "If positive: ramp to 100%"
      - "If negative: rollback, diagnose, iterate"
    duration: "1-2 weeks (canary + ramp)"
    
  step_5_maintain:
    actions:
      - "Monitor for quality degradation over time"
      - "Collect new examples from production (feedback loop)"
      - "Periodic retraining with updated data (monthly/quarterly)"
    ongoing: "Continuous improvement cycle"
```

---

## Interview Tip

> When asked about fine-tuning operations: "My fine-tuning workflow: (1) Data is king — I spend 60% of effort on data quality. 500 high-quality examples beat 10,000 noisy ones. I use a mix of human-curated seed examples + synthetic generation (stronger model generates, humans validate) + filtered production data (user-thumbs-up interactions). (2) Training — QLoRA on all linear layers (rank=32, alpha=64) is my default. Trains 70B on single H100 in 4-6 hours. I watch eval loss for overfitting (typically 3 epochs is optimal). (3) Evaluation — compare against THREE baselines: base model + prompt, base model + RAG, and previous fine-tuned version. Must improve on task-specific metrics AND pass regression tests (no catastrophic forgetting of general capability). (4) Deployment — multi-LoRA serving with vLLM (one base model, multiple adapters, swap per-request). This enables A/B testing fine-tuned vs base, multi-tenant adapters, and instant rollback. (5) Maintenance — monthly retraining with new production data (feedback loop: user ratings → filtered into training set → retrain → deploy). Key decision: fine-tune when you need consistent behavior/format that prompting can't achieve, RAG when you need factual grounding, both when you need both."

---

## Common Mistakes

1. **Fine-tuning with low-quality data** — Feeding 10,000 examples scraped from the internet with inconsistent quality. The model learns noise and produces unpredictable outputs. Solution: 500 meticulously curated examples where every output is exactly what you want the model to produce.

2. **Not comparing against prompt engineering baseline** — Fine-tuning shows 85% accuracy. Is that good? Depends — maybe prompt engineering achieves 83% at zero training cost. Always compare fine-tuned model vs best prompt engineering on the same test set. Fine-tune only if the gap is meaningful.

3. **Overfitting (training too long)** — Running for 10 epochs on a small dataset. Eval loss stops decreasing at epoch 3 and starts increasing. The model memorizes training examples instead of generalizing. Monitor eval loss, stop early, use 3-5 epochs as default.

4. **Not testing for catastrophic forgetting** — Fine-tuned model excels at your specific task but can no longer handle basic English, math, or code. LoRA reduces this risk (base model frozen), but always verify with a regression test on general benchmarks.

5. **Training on model outputs without checking TOS (Terms of Service)** — Using GPT-5 outputs to fine-tune an open-source competitor. Many providers explicitly prohibit this. Check the Terms of Service. Anthropic, OpenAI have restrictions. Use synthetic data generation from models whose licenses permit it (open models like Llama, Mistral with permissive licenses).

---

## Key Takeaways

- Fine-tune when: consistent format/style, domain-specific behavior, prompt engineering hits ceiling
- LoRA: train <1% of parameters, 90% less memory, no catastrophic forgetting, multi-adapter serving
- QLoRA: 4-bit base model + FP16 LoRA adapters — fine-tune 70B on single GPU
- Data quality > data quantity: 500 perfect examples > 10,000 noisy examples
- Default hyperparameters: rank=32, alpha=64, all linear layers, 3 epochs, lr=2e-4
- Evaluate against baselines: base+prompt, base+RAG, previous fine-tune
- Deploy via multi-LoRA (vLLM): one base model, multiple adapters, per-request routing
- Regression test: verify general capabilities aren't lost after fine-tuning
- Maintenance: monthly retraining with new production data (user feedback → training set)
- Combine: fine-tune for style/format + RAG for factual grounding = best of both worlds
