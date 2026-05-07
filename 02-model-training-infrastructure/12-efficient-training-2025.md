# Efficient Training 2025-2026

## The Problem / Why This Matters

Full fine-tuning of large models is prohibitively expensive — updating all 70 billion parameters of a model requires hundreds of GB of GPU memory and costs thousands of dollars per training run. But in 2025-2026, most practical ML (Machine Learning) engineering doesn't require training from scratch. Instead, we adapt pre-trained foundation models to specific tasks using parameter-efficient methods. PEFT (Parameter-Efficient Fine-Tuning) methods like LoRA (Low-Rank Adaptation), QLoRA (Quantized LoRA), and adapters achieve 90-99% of full fine-tuning quality while training only 0.1-5% of parameters, reducing memory requirements by 4-16x and cost by 5-50x. These techniques have fundamentally changed the economics of ML: a team can now fine-tune a 70B model on a single consumer GPU (RTX 4090 with QLoRA) — something that previously required a multi-GPU server costing $100K+. Understanding PEFT methods is no longer optional — they are the default approach for model customization in 2026.

---

## The Analogy

Think of PEFT methods like customizing a car:

- **Full fine-tuning** = Rebuilding the entire engine from scratch for your specific needs. Incredibly expensive, requires a full workshop (massive GPU memory), and takes a long time. But you get perfect customization.
- **LoRA** = Adding a turbo kit and custom ECU (Engine Control Unit) chip. You don't rebuild the engine — you add small, targeted modifications that dramatically change performance characteristics. The original engine (frozen weights) stays intact, and your mods (LoRA adapters) are tiny and swappable.
- **QLoRA** = Same turbo kit, but you also compress the engine into a smaller space first (quantization). Now the whole setup fits in a smaller garage (less GPU memory). Slightly less precise, but practically the same performance.
- **Adapters** = Adding plug-in modules between engine components. Each module is small but transforms the output of one component before it reaches the next.

---

## Deep Dive

### LoRA (Low-Rank Adaptation)

```yaml
LoRA_Theory:
  key_insight: "Weight updates during fine-tuning have LOW RANK — they live in a much smaller subspace than the full weight matrix"
  
  how_it_works:
    original: "Pre-trained weight matrix W (shape: [d_out × d_in])"
    full_finetune: "Learn new W' = W + ΔW (ΔW is full-rank: d_out × d_in parameters)"
    lora: "Learn ΔW = B × A where B is [d_out × r] and A is [r × d_in]"
    key: "r << d_out, d_in (rank r is typically 8-64)"
    
  parameter_reduction:
    example:
      layer_size: "4096 × 4096"
      full_params: "4096 × 4096 = 16,777,216 parameters"
      lora_r8: "4096 × 8 + 8 × 4096 = 65,536 parameters (0.4% of full)"
      lora_r16: "4096 × 16 + 16 × 4096 = 131,072 parameters (0.8%)"
      lora_r64: "4096 × 64 + 64 × 4096 = 524,288 parameters (3.1%)"
      
  forward_pass:
    without_lora: "output = W × input"
    with_lora: "output = W × input + (B × A) × input × (alpha/r)"
    inference: "Can merge: W_merged = W + B × A × (alpha/r) → zero overhead"
    
  hyperparameters:
    rank_r:
      description: "Rank of the low-rank matrices (dimensionality of adaptation)"
      typical: "8, 16, 32, 64"
      higher: "More expressive (closer to full fine-tuning) but more parameters"
      lower: "More parameter-efficient but less expressive"
      recommendation: "Start with r=16, increase if task is complex"
      
    alpha:
      description: "Scaling factor (effectively learning rate multiplier for LoRA)"
      typical: "alpha = 2 × rank (so alpha=32 for r=16)"
      meaning: "Scaling applied: alpha/rank. Higher alpha → stronger LoRA signal"
      
    target_modules:
      description: "Which layers to apply LoRA to"
      transformer_options:
        - "q_proj, v_proj (attention query/value — most common minimal set)"
        - "q_proj, k_proj, v_proj, o_proj (all attention projections)"
        - "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (all linear layers)"
      recommendation: "Apply to all linear layers for best quality (small additional cost)"
```

### QLoRA (Quantized LoRA)

```python
# QLoRA - Fine-tune 70B models on consumer GPUs
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Load base model in 4-bit
    bnb_4bit_quant_type="nf4",            # NormalFloat4 (optimal for normally distributed weights)
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in BF16 (dequantize for computation)
    bnb_4bit_use_double_quant=True,        # Quantize the quantization constants (extra savings)
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-4-70B",
    quantization_config=bnb_config,
    device_map="auto",  # Auto-distribute across available GPUs
    torch_dtype=torch.bfloat16,
)

# Prepare for k-bit training (freeze base, enable gradient for LoRA)
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,                          # Rank
    lora_alpha=32,                 # Alpha (2x rank)
    target_modules=[               # Apply to all linear layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,             # Dropout for regularization
    bias="none",                   # Don't train biases
    task_type="CAUSAL_LM",
)

# Create PEFT model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: "trainable params: 83,886,080 || all params: 70,553,710,592 || trainable%: 0.1189%"
```

```yaml
QLoRA_Mechanics:
  innovation: "4-bit quantization of base model + LoRA adapters in BF16"
  
  memory_comparison:
    model: "Llama-4-70B"
    full_fp16: "140 GB (base model only, no optimizer)"
    full_fp16_with_adam: "140 + 140 (grads) + 560 (adam) = 840 GB"
    qlora_4bit: "~35 GB (4-bit base) + ~0.3 GB (LoRA FP16) + ~2.4 GB (adam on LoRA params) ≈ 38 GB"
    fits_on: "1× H100 80GB or 2× RTX 4090 48GB total"
    
  key_techniques:
    nf4_quantization:
      what: "NormalFloat 4-bit — quantization levels optimized for normally distributed weights"
      why: "Neural network weights are approximately normally distributed"
      better_than: "Standard INT4 (which assumes uniform distribution)"
      
    double_quantization:
      what: "Quantize the quantization constants themselves"
      savings: "Additional ~0.4 bits per parameter (small but helps at scale)"
      
    paged_optimizers:
      what: "Use CPU memory as overflow for optimizer states"
      when: "GPU memory pressure during gradient accumulation"
      
  quality_vs_full_finetune:
    finding: "QLoRA matches full fine-tuning quality on most benchmarks"
    caveat: "Very complex tasks or domain-specific knowledge may need higher rank"
    recommendation: "Start with QLoRA r=16, only move to full fine-tuning if quality insufficient"
```

### Other PEFT Methods

```yaml
Other_PEFT_Methods:
  dora:
    name: "Weight-Decomposed Low-Rank Adaptation"
    innovation: "Decompose ΔW into magnitude and direction components"
    advantage: "Better learning dynamics than LoRA, closer to full fine-tuning"
    overhead: "Minimal additional compute vs LoRA"
    status: "Gaining adoption in 2025-2026 as LoRA successor"
    
  ia3:
    name: "Infused Adapter by Inhibiting and Amplifying Inner Activations"
    how: "Learn scaling vectors (not matrices) applied to activations"
    params: "Even fewer than LoRA (just scaling factors)"
    quality: "Slightly below LoRA for complex tasks"
    use: "Extremely parameter-constrained scenarios"
    
  prefix_tuning:
    how: "Prepend learnable 'virtual tokens' to each layer's attention"
    params: "Very few (prefix_length × hidden_size × num_layers)"
    quality: "Good for generation tasks, less effective for classification"
    limitation: "Reduces effective context length (prefix tokens consume positions)"
    
  prompt_tuning:
    how: "Learn continuous embeddings prepended to input (only embedding layer)"
    params: "Minimal (num_virtual_tokens × embedding_dim)"
    quality: "Works for large models (>10B), less effective for small models"
    advantage: "Can serve multiple tasks with same model (just swap prompt embeddings)"
    
  adapter_layers:
    how: "Insert small bottleneck layers (down-project → activation → up-project) between transformer layers"
    params: "bottleneck_dim × hidden_size × 2 per layer"
    quality: "Good, well-studied"
    disadvantage: "Adds latency during inference (extra layers)"
    
  comparison_table:
    method_vs_quality_vs_params:
      full_finetune: "quality: best | params: 100% | memory: highest"
      lora_r16: "quality: 95-99% | params: 0.5-2% | memory: moderate"
      qlora_r16: "quality: 94-98% | params: 0.1% | memory: lowest"
      dora: "quality: 96-99% | params: 0.5-2% | memory: moderate"
      ia3: "quality: 90-95% | params: 0.01% | memory: very low"
      prefix_tuning: "quality: 88-95% | params: 0.1% | memory: low"
```

### Practical LoRA Training

```python
# Complete LoRA fine-tuning setup (2026 best practices)
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch

# Configuration
MODEL_NAME = "meta-llama/Llama-4-8B"
DATASET = "your-org/instruction-dataset"
OUTPUT_DIR = "./output/llama4-8b-lora"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load model (QLoRA: 4-bit base)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=32,                              # Higher rank for better quality
    lora_alpha=64,                     # 2x rank
    target_modules="all-linear",       # Apply to ALL linear layers (best quality)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Load and prepare dataset
dataset = load_dataset(DATASET)

def format_instruction(example):
    """Format as chat template."""
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

dataset = dataset.map(format_instruction)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,     # Effective batch: 4 × 4 = 16
    learning_rate=2e-4,                # Higher LR for LoRA (vs 2e-5 for full FT)
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",          # 8-bit Adam (saves memory)
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    max_grad_norm=1.0,
    report_to="wandb",
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

trainer.train()

# Save LoRA adapter (small: ~100-500 MB vs full model: 16 GB)
model.save_pretrained(OUTPUT_DIR)
```

### Serving Multiple LoRA Adapters

```yaml
Multi_LoRA_Serving:
  concept: "One base model + multiple LoRA adapters for different tasks/customers"
  
  architecture:
    base_model: "Single Llama-4-8B in GPU memory (16 GB)"
    adapters:
      customer_a: "Support chatbot adapter (300 MB)"
      customer_b: "Code generation adapter (300 MB)"
      customer_c: "Medical Q&A adapter (300 MB)"
    total_memory: "16 GB base + 900 MB adapters = ~17 GB"
    vs_separate_models: "3 × 16 GB = 48 GB (almost 3x more memory)"
    
  serving_frameworks:
    vllm:
      support: "Native multi-LoRA serving"
      how: "Load base model + multiple adapters, route requests by adapter name"
      advantage: "Shared KV cache for base model, efficient batching across adapters"
      
    lorax:
      what: "LoRAX (LoRA eXchange) — specialized multi-LoRA serving"
      how: "Dynamic adapter loading, heterogeneous batching"
      advantage: "Can serve hundreds of adapters from shared base"
      
    huggingface_tgi:
      support: "LoRA adapter support in Text Generation Inference"
      how: "Specify adapter in request, TGI loads and applies"
      
  dynamic_loading:
    concept: "Load/unload adapters on demand (not all in memory simultaneously)"
    when: "Hundreds of adapters, each used infrequently"
    implementation: "LRU cache of active adapters, load from storage on miss"
    latency: "First request to cold adapter: +100-500ms (adapter load)"
    subsequent: "Same latency as always-loaded adapter"
    
  use_cases:
    multi_tenant_saas: "Each customer gets custom adapter on shared base model"
    a_b_testing: "Multiple adapter versions for A/B comparison"
    task_routing: "Different adapters for different task types (summary, code, chat)"
```

### Knowledge Distillation

```yaml
Knowledge_Distillation:
  what: "Train a smaller (student) model to mimic a larger (teacher) model"
  why: "Get large model quality in a smaller, cheaper-to-serve model"
  
  approaches:
    output_distillation:
      how: "Student learns to match teacher's output probability distribution"
      loss: "KL divergence between teacher softmax and student softmax"
      advantage: "Teacher's soft labels carry more information than hard labels"
      
    feature_distillation:
      how: "Student learns to match teacher's intermediate representations"
      loss: "MSE between teacher and student hidden states (with projection)"
      advantage: "Student captures teacher's internal reasoning"
      
    data_augmentation_distillation:
      how: "Use teacher to generate training data, train student on generated data"
      advantage: "Simple, works with black-box teachers (API-only models)"
      example: "Use GPT-4o to generate responses → train smaller model on them"
      
  practical_application_2026:
    scenario: "Distill Claude 4 Opus quality into a 7B model for edge deployment"
    process:
      1: "Generate 100K high-quality responses from Claude 4 Opus"
      2: "Fine-tune Llama-4-8B on these responses (QLoRA)"
      3: "Evaluate: does 8B model match Opus quality on your specific task?"
      4: "If yes: deploy 8B model (100x cheaper to serve)"
      5: "If no: generate more data, increase model size, or accept quality trade-off"
    cost: "~$5K in API costs + $200 in fine-tuning compute = $5.2K one-time"
    savings: "Serve at $0.001/request instead of $0.03/request (30x cheaper ongoing)"
```

---

## How It Works in Practice

### Choosing the Right PEFT Method

```yaml
Decision_Framework:
  scenario_1:
    need: "Fine-tune 70B model, limited to 1-2 GPUs"
    method: "QLoRA (4-bit base + LoRA r=16-32)"
    memory: "~40-50 GB (fits on 1× H100 or 2× RTX 4090)"
    quality: "95-98% of full fine-tuning"
    
  scenario_2:
    need: "Fine-tune 8B model, have 1× A100 80GB"
    method: "LoRA r=32-64 in BF16 (no quantization needed)"
    memory: "~30 GB (fits easily)"
    quality: "97-99% of full fine-tuning"
    
  scenario_3:
    need: "Serve 100 different task-specific models"
    method: "One base model + 100 LoRA adapters"
    memory: "Base (16 GB) + adapters in LRU cache"
    advantage: "100 models in memory of <2 models"
    
  scenario_4:
    need: "Maximum quality, cost is no object"
    method: "Full fine-tuning with FSDP"
    memory: "Need 4-8× A100/H100"
    quality: "100% (ceiling)"
    when: "Only if LoRA results are measurably worse on YOUR task"
    
  scenario_5:
    need: "Quick adaptation, minimal compute"
    method: "QLoRA r=8, train 1-2 epochs"
    time: "30 minutes - 2 hours (depends on dataset)"
    cost: "$5-50 in GPU compute"
    quality: "90-95% (good enough for prototyping)"
```

---

## Interview Tip

> When asked about efficient fine-tuning: "In 2026, LoRA and QLoRA are the default fine-tuning approach — full fine-tuning is only used when LoRA quality is insufficient (rare). LoRA works by learning low-rank update matrices (ΔW = B×A where B is d×r and A is r×d, with r=16-64) added to frozen pre-trained weights. QLoRA goes further: quantize the base model to 4-bit (NF4 format), keep LoRA adapters in BF16, and use paged optimizers for memory management. This lets you fine-tune 70B models on a single GPU (40 GB). Key configuration: rank 16-32 for most tasks (higher = more expressive but more params), alpha = 2× rank, apply to ALL linear layers for best quality, learning rate 2-3e-4 (10x higher than full fine-tuning). For serving: multiple LoRA adapters share one base model (vLLM, LoRAX), saving memory linearly. The LoRA adapter can be merged into base weights at inference time for zero latency overhead. Quality-wise, LoRA achieves 95-99% of full fine-tuning on most tasks — the gap is only measurable on very complex domain-specific tasks."

---

## Common Mistakes

1. **Using too low a LoRA rank** — Starting with r=4 because "lower is more efficient." While r=4 works for simple classification, instruction tuning and complex tasks typically need r=16-64 for good quality. The memory savings between r=4 and r=32 are negligible compared to base model size.

2. **Applying LoRA only to attention** — Only adding LoRA to q_proj and v_proj (the original paper's recommendation). In practice, applying to ALL linear layers (attention + MLP) gives significantly better quality with minimal additional cost. Use `target_modules="all-linear"` in PEFT.

3. **Using full fine-tuning learning rate** — Setting learning rate to 2e-5 (typical for full fine-tuning) when using LoRA. LoRA adapters are initialized to near-zero, so they need a higher learning rate (2-3e-4) to move away from zero and actually learn something.

4. **Not merging adapters for deployment** — Serving LoRA with runtime adapter addition when single-task deployment. Merging adapters into base weights (`model.merge_and_unload()`) eliminates any inference overhead while keeping the full LoRA quality.

5. **Training too many epochs with LoRA** — Over-training LoRA adapters (10+ epochs on small datasets). LoRA tends to overfit faster than full fine-tuning because fewer parameters means less capacity for regularization. Typically 1-3 epochs is sufficient for instruction tuning.

---

## Key Takeaways

- LoRA: learn low-rank updates (ΔW = B×A) to frozen pre-trained weights — 0.1-5% of parameters
- QLoRA: 4-bit quantized base + BF16 LoRA — fine-tune 70B on 1 GPU (~40 GB)
- Default recommendation: QLoRA r=16-32, alpha=2×rank, all linear layers, lr=2e-4
- Quality: LoRA achieves 95-99% of full fine-tuning for most tasks
- Memory savings: 4-16× less memory than full fine-tuning
- Merge for inference: LoRA adapters merge into base weights for zero overhead
- Multi-LoRA serving: one base model + many adapters (vLLM, LoRAX) — memory efficient
- DoRA: emerging LoRA improvement (magnitude + direction decomposition) — better learning dynamics
- Knowledge distillation: use large model (Claude 4) to generate data, train small model on it
- PEFT is the DEFAULT in 2026 — full fine-tuning only when PEFT quality is demonstrably insufficient
