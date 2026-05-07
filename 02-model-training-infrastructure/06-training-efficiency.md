# Training Efficiency

## The Problem / Why This Matters

Training ML (Machine Learning) models is expensive — a single training run for a 70B parameter model costs $1-5M in GPU compute. Even fine-tuning smaller models (7-13B) costs hundreds to thousands of dollars per experiment. Every percentage of training efficiency improvement translates directly to cost savings and faster iteration. Training efficiency means maximizing useful computation (model FLOPs) while minimizing waste (memory overhead, communication latency, GPU idle time, redundant computation). The key techniques — mixed precision training, gradient accumulation, activation checkpointing, gradient clipping, learning rate scheduling, and compilation — can collectively deliver 2-5x speedups without changing the model or losing quality. In 2026, with GPU supply still constrained and training costs dominating ML budgets, training efficiency is the difference between a team that can iterate on 5 experiments per week vs 1 experiment. ML engineers who understand these techniques ship better models faster with the same hardware budget.

---

## The Analogy

Think of training efficiency like fuel efficiency in a long-distance road trip:

- **Mixed Precision** = Using a hybrid engine. The electric motor (FP16/BF16) handles 90% of driving efficiently, while the gas engine (FP32) kicks in only when you need maximum power (critical computations). Same destination, half the fuel.
- **Gradient Accumulation** = Carpooling. Instead of driving separately (one sample per update), you fill the car completely (accumulate gradients) before making the trip (optimizer step). Same result, fewer trips.
- **Activation Checkpointing** = Not packing every item "just in case." Instead of loading the entire house into the car (storing all activations), you bring only essentials and pick up other items when needed (recompute). Less space needed, slightly more time.
- **torch.compile** = GPS route optimization. Instead of following the old route (eager execution), the GPS analyzes the entire journey and finds shortcuts (fused operations, optimized memory access). Same destination, faster arrival.

---

## Deep Dive

### Mixed Precision Training

```yaml
Mixed_Precision:
  concept:
    what: "Use lower precision (FP16/BF16) for most computation, FP32 for critical operations"
    why: "Lower precision = less memory + faster compute + tensor core utilization"
    result: "~2x throughput, ~50% memory reduction, no quality loss (when done correctly)"
    
  precision_formats:
    fp32:
      bits: 32
      range: "±3.4 × 10^38"
      precision: "~7 decimal digits"
      use: "Master weights, loss scaling, gradient accumulation"
      
    fp16:
      bits: 16
      range: "±65,504"
      precision: "~3.3 decimal digits"
      use: "Forward pass, backward pass (with loss scaling)"
      risk: "Overflow (values > 65504) and underflow (gradients too small)"
      solution: "Dynamic loss scaling (multiply loss by scale factor)"
      
    bf16:
      full_name: "Brain Float 16 (Google)"
      bits: 16
      range: "±3.4 × 10^38 (same as FP32!)"
      precision: "~2.4 decimal digits (less than FP16)"
      use: "Forward pass, backward pass (no loss scaling needed)"
      advantage: "Same range as FP32 → no overflow, no loss scaling required"
      recommendation: "Preferred over FP16 in 2026 (simpler, equally fast)"
      
    fp8:
      bits: 8
      formats: "E4M3 (precision) and E5M2 (range)"
      use: "H100/B200 tensor cores — 2x throughput vs FP16"
      status: "Emerging (2025-2026) — requires careful implementation"
      framework: "Transformer Engine (NVIDIA), MS-AMP"
      
  implementation:
    pytorch_amp:
      description: "Automatic Mixed Precision — PyTorch native"
      code_pattern: |
        scaler = torch.amp.GradScaler()  # For FP16 only (not needed for BF16)
        
        for batch in dataloader:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch)
                loss = criterion(output)
            
            # BF16: no scaler needed
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
    what_runs_in_low_precision:
      - "Matrix multiplications (Linear layers, attention)"
      - "Convolutions"
      - "Most element-wise operations"
    what_stays_in_fp32:
      - "Loss computation"
      - "Softmax"
      - "LayerNorm"
      - "Master weight copy (for optimizer)"
```

### Gradient Accumulation

```python
# Gradient Accumulation - Simulate larger batch sizes
accumulation_steps = 8  # Effective batch = micro_batch × accumulation × num_gpus
micro_batch_size = 4    # What fits in GPU memory

# Effective batch size = 4 × 8 × (num_GPUs) = 32 per GPU (or 256 with 8 GPUs)

optimizer.zero_grad()

for step, batch in enumerate(dataloader):
    # Forward pass
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(batch).loss
        loss = loss / accumulation_steps  # Normalize loss
    
    # Backward pass (accumulates gradients)
    loss.backward()
    
    # Only update weights every N steps
    if (step + 1) % accumulation_steps == 0:
        # Gradient clipping (before optimizer step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

```yaml
Gradient_Accumulation:
  why:
    - "GPU memory limits micro-batch size (can't fit large batch)"
    - "Research shows larger effective batch sizes improve training stability and quality"
    - "Simulates larger batch without needing more memory"
    
  how_it_works:
    - "Run forward + backward on micro-batch (gradients accumulate in .grad buffers)"
    - "Repeat N times (N = accumulation steps)"
    - "After N micro-batches: run optimizer step (update weights with accumulated gradients)"
    - "Normalize loss by N to maintain correct gradient magnitude"
    
  trade_offs:
    advantage: "Larger effective batch size without memory increase"
    disadvantage: "N× more forward/backward passes per optimizer step (slower wall-clock time)"
    when_beneficial: "When larger batch is needed for convergence (LLM training, contrastive learning)"
    when_harmful: "When many optimizer updates are needed quickly (fast iteration, few data)"
    
  interaction_with_ddp:
    problem: "DDP syncs gradients every backward pass (expensive with accumulation)"
    solution: "no_sync context manager — skip gradient sync until last accumulation step"
    code: |
      for i, batch in enumerate(dataloader):
          ctx = model.no_sync() if (i + 1) % accum != 0 else nullcontext()
          with ctx:
              loss = model(batch).loss / accum
              loss.backward()
          if (i + 1) % accum == 0:
              optimizer.step()
              optimizer.zero_grad()
```

### Activation Checkpointing

```yaml
Activation_Checkpointing:
  the_problem:
    what: "During backpropagation, gradients need the intermediate activations from forward pass"
    normal: "Store ALL activations in memory (for every layer)"
    cost: "For large models: activations can use MORE memory than model weights"
    example: "Llama-4-8B, batch=4, seq=4096: activations use ~60GB (vs 16GB for model in FP16)"
    
  the_solution:
    what: "Don't store all activations — recompute them during backward pass"
    how: "Mark certain layers as 'checkpoints'. Only store activations at checkpoint boundaries."
    during_backward: "When needed, recompute activations from the nearest checkpoint"
    
  strategies:
    full_checkpointing:
      what: "Checkpoint every transformer layer"
      memory_savings: "Dramatic (activations reduced by ~N× where N = layers between checkpoints)"
      compute_overhead: "~30-33% more compute (one extra forward pass per checkpointed segment)"
      when: "Always use for training models that don't fit in memory otherwise"
      
    selective_checkpointing:
      what: "Only checkpoint expensive operations (attention, large MLPs)"
      memory_savings: "Good (most of the benefit, less overhead)"
      compute_overhead: "~15-20% more compute"
      implementation: "Megatron-LM's selective recomputation"
      
  pytorch_implementation:
    utility: "torch.utils.checkpoint.checkpoint()"
    usage: |
      from torch.utils.checkpoint import checkpoint
      
      class TransformerBlock(nn.Module):
          def forward(self, x):
              # Without checkpointing: all intermediate activations stored
              # With checkpointing: only input and output stored
              return checkpoint(self._forward_impl, x, use_reentrant=False)
          
          def _forward_impl(self, x):
              x = x + self.attention(self.norm1(x))
              x = x + self.mlp(self.norm2(x))
              return x
              
  memory_vs_compute_tradeoff:
    no_checkpointing: "Memory: HIGH, Compute: BASELINE"
    checkpointing_every_layer: "Memory: LOW, Compute: +33%"
    sweet_spot: "Checkpoint every 2-3 layers for moderate savings with less overhead"
```

### torch.compile (PyTorch 2.0+ Compilation)

```python
# torch.compile - JIT compilation for faster training and inference
import torch

model = MyModel().cuda()

# Compile the model (one-time cost, huge runtime benefit)
compiled_model = torch.compile(
    model,
    mode="reduce-overhead",  # Options: "default", "reduce-overhead", "max-autotune"
    backend="inductor",       # TorchInductor backend (default, best for most cases)
    fullgraph=False           # Allow graph breaks (more compatible)
)

# Training loop (same as before, but faster)
for batch in dataloader:
    loss = compiled_model(batch).loss  # First call compiles, subsequent calls fast
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

```yaml
torch_compile:
  what: "Compiles PyTorch model into optimized GPU kernels using TorchDynamo + TorchInductor"
  
  how_it_works:
    dynamo: "Captures Python execution into computation graph (FX graph)"
    inductor: "Compiles FX graph into optimized Triton/CUDA kernels"
    optimizations:
      - "Kernel fusion (combine multiple operations into one kernel)"
      - "Memory layout optimization (avoid unnecessary copies)"
      - "Constant folding (pre-compute static values)"
      - "Dead code elimination"
      
  speedup:
    typical: "10-30% faster training, 20-50% faster inference"
    best_case: "2-3x faster (heavily memory-bound models benefit most)"
    
  modes:
    default: "Balanced compilation (moderate compile time, good runtime)"
    reduce_overhead: "Faster first iteration (less aggressive optimization)"
    max_autotune: "Try many kernel variations (long compile, best runtime)"
    
  limitations:
    - "First call is slow (compilation time: seconds-minutes)"
    - "Dynamic shapes cause recompilation (batch size changes, variable sequence lengths)"
    - "Some Python patterns cause 'graph breaks' (less optimization)"
    - "Not all operations supported (some fall back to eager mode)"
    
  best_practices:
    - "Use fullgraph=False for compatibility (allows graph breaks)"
    - "Pad sequences to fixed lengths to avoid recompilation"
    - "Profile with and without compile to verify speedup"
    - "Use mode='max-autotune' for production/deployment (worth the compile time)"
```

### Learning Rate Scheduling

```yaml
Learning_Rate_Scheduling:
  why: "Fixed learning rate is rarely optimal. Learning rate should typically start high (fast progress) and decay (fine refinement)."
  
  common_schedules:
    warmup_cosine_decay:
      description: "Linear warmup → cosine decay to near-zero"
      use: "Most popular for transformer training (LLMs, BERT, ViT)"
      implementation: |
        # Warmup for 2000 steps, then cosine decay over remaining steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=2000, 
            num_training_steps=total_steps
        )
      why_warmup: "Adam optimizer's statistics are unreliable at start — small LR until they warm up"
      why_cosine: "Smooth decay → model settles gradually into minimum"
      
    linear_decay:
      description: "Linear warmup → linear decay to zero"
      use: "Fine-tuning (shorter training)"
      
    warmup_stable_decay:
      description: "Warmup → constant LR for most of training → decay at end"
      use: "Some LLM pre-training (Llama family uses this)"
      name_alias: "WSD (Warmup-Stable-Decay) schedule"
      
    one_cycle:
      description: "Ramp up to max LR, then ramp down (super-convergence)"
      use: "Fast training when compute-constrained"
      advantage: "Often trains to same quality in fewer steps"
      
  selecting_max_learning_rate:
    rule_of_thumb: "Start with published values for similar model/task"
    lr_finder: "Gradually increase LR, plot loss — optimal LR is where loss drops fastest"
    typical_values:
      pretraining_llm: "1e-4 to 6e-4 (depends on batch size)"
      finetuning_llm: "1e-5 to 5e-5"
      bert_finetuning: "2e-5 to 5e-5"
      adam_general: "1e-4 to 1e-3"
```

### FlashAttention and Efficient Kernels

```yaml
FlashAttention:
  what: "Memory-efficient attention algorithm that avoids materializing the full attention matrix"
  
  the_problem:
    standard_attention: "Q × K^T produces [seq_len × seq_len] matrix — O(n²) memory"
    example: "Sequence length 8192: attention matrix is 8192×8192×2 bytes = 128MB per head per batch"
    with_32_heads_batch_4: "128MB × 32 × 4 = 16GB just for attention matrices"
    
  flash_attention_solution:
    key_idea: "Compute attention in tiles — never materialize full attention matrix"
    how: "Process attention in small blocks that fit in GPU SRAM (shared memory)"
    memory: "O(n) instead of O(n²) — dramatic reduction"
    speed: "2-4x faster (fewer HBM read/writes, better use of fast SRAM)"
    
  versions:
    flash_attention_1: "Tiling algorithm, O(n) memory"
    flash_attention_2: "Better parallelism, reduced non-matmul FLOPs, 2x faster than v1"
    flash_attention_3: "H100 optimized, FP8 support, even faster"
    
  usage:
    pytorch_native: "torch.nn.functional.scaled_dot_product_attention (auto-selects FlashAttention)"
    explicit: "flash_attn library (pip install flash-attn)"
    huggingface: "attn_implementation='flash_attention_2' in model config"
    
  impact_on_training:
    memory: "Can train with 4-8x longer sequences (same GPU memory)"
    speed: "15-30% faster training overall (attention is ~40% of transformer compute)"
    enables: "Long-context models (128K+ tokens) that were previously infeasible"
```

---

## How It Works in Practice

### Combining All Efficiency Techniques

```python
# Complete efficient training setup (2026 best practices)
import torch
from transformers import (
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer
)

# Load model with efficient settings
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-4-8B",
    torch_dtype=torch.bfloat16,          # BF16 precision
    attn_implementation="flash_attention_2",  # FlashAttention
    use_cache=False,                      # Disable KV cache for training
)

# Enable activation checkpointing
model.gradient_checkpointing_enable()

# Compile for additional speedup
model = torch.compile(model, mode="default")

# Training arguments with all efficiency techniques
training_args = TrainingArguments(
    output_dir="./output",
    
    # Precision
    bf16=True,                            # BF16 mixed precision
    
    # Batch and accumulation
    per_device_train_batch_size=4,        # Limited by GPU memory
    gradient_accumulation_steps=8,        # Effective batch = 4 × 8 × num_gpus
    
    # Learning rate
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    
    # Efficiency
    gradient_checkpointing=True,          # Activation checkpointing
    torch_compile=True,                   # torch.compile
    dataloader_num_workers=4,             # Parallel data loading
    dataloader_pin_memory=True,           # Faster CPU→GPU transfer
    
    # Regularization
    weight_decay=0.01,
    max_grad_norm=1.0,                    # Gradient clipping
    
    # Training duration
    num_train_epochs=3,
    
    # Logging
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

```yaml
Combined_Impact:
  baseline: "FP32, no checkpointing, no compile, batch_size=1"
  with_bf16: "2x throughput, 50% less memory"
  with_checkpointing: "60% less activation memory (+30% compute)"
  with_gradient_accumulation: "Effective batch 32× larger (same memory)"
  with_flash_attention: "2-4x faster attention, sequence length possible"
  with_torch_compile: "Additional 10-30% speedup"
  
  total_improvement: "4-8x faster training vs naive baseline"
  cost_saving: "Same quality model in 75-87% less GPU-hours"
```

---

## Interview Tip

> When asked about training efficiency: "I apply a standard efficiency stack for every training run: (1) BF16 mixed precision — 2x throughput, 50% memory reduction, no special loss scaling needed. (2) FlashAttention — O(n) memory instead of O(n²), 2-4x faster attention. (3) Activation checkpointing — trades 30% extra compute for massive memory savings, enabling larger batches or sequences. (4) Gradient accumulation — simulates large effective batch sizes when GPU memory is limited. (5) torch.compile — 10-30% free speedup from kernel fusion and memory optimization. (6) Cosine learning rate schedule with warmup — standard for transformers, helps convergence. Combined, these typically give 4-8x improvement over a naive baseline. The key insight is that most techniques are complementary — you use ALL of them together, not just one."

---

## Common Mistakes

1. **Using FP16 instead of BF16** — FP16 requires dynamic loss scaling to handle overflow/underflow, adding complexity and occasional instability. BF16 has the same dynamic range as FP32, eliminating these issues entirely. Always prefer BF16 on hardware that supports it (A100+).

2. **Forgetting gradient clipping** — Without gradient clipping, occasional large gradients can destabilize training (exploding gradients). Always clip to max_norm=1.0 as a default. This is especially important for LLM training where loss spikes can occur.

3. **Setting batch size too large with gradient accumulation** — Larger batch isn't always better. Very large batches can hurt generalization (sharp minima) and require corresponding learning rate adjustment (linear scaling rule). There's an optimal batch size for each task.

4. **Not profiling before optimizing** — Applying every optimization blindly. Sometimes activation checkpointing isn't needed (plenty of memory), or torch.compile causes issues with dynamic shapes. Profile first (PyTorch Profiler, NVIDIA Nsight) to identify the actual bottleneck.

5. **Ignoring data loading bottleneck** — Optimizing GPU compute while data loading is the bottleneck. If GPUs wait for data, all GPU-side optimizations are wasted. Check: `dataloader_num_workers`, `pin_memory`, data format (WebDataset/streaming), and storage throughput.

---

## Key Takeaways

- Mixed precision (BF16): 2x throughput + 50% memory — use always on A100+ hardware
- Gradient accumulation: simulate large batches without memory increase (use DDP no_sync for efficiency)
- Activation checkpointing: trade ~30% extra compute for dramatic memory savings — enables larger models/batches
- torch.compile: 10-30% free speedup from kernel fusion — use for production training
- FlashAttention: O(n) memory, 2-4x faster attention — essential for long sequences
- Learning rate: warmup + cosine decay is the standard for transformer training
- Gradient clipping (max_norm=1.0): prevents training instability from gradient spikes
- Combined efficiency stack: 4-8x improvement over naive baseline
- Always profile first: identify whether you're compute-bound, memory-bound, or data-loading-bound
- These techniques are complementary — apply all of them together as the default training configuration
