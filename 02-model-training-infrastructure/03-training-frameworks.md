# Training Frameworks

## The Problem / Why This Matters

Distributed training requires sophisticated software to coordinate GPUs, manage communication, handle checkpointing, and optimize memory usage. Writing this from scratch is impractical — it takes thousands of engineering hours to implement efficient AllReduce, memory-optimal sharding, pipeline scheduling, and fault tolerance. Training frameworks solve this by providing high-level APIs that abstract the complexity of multi-GPU coordination. The framework choice directly impacts: how large a model you can train (memory efficiency), how fast training runs (communication optimization), how easy it is to implement (developer experience), and how reliable training is (fault tolerance, checkpointing). In 2026, the primary frameworks are PyTorch DDP (Distributed Data Parallel) / FSDP (Fully Sharded Data Parallel), DeepSpeed (Microsoft), Megatron-LM (NVIDIA), and for the TPU (Tensor Processing Unit) ecosystem, JAX with XLA. Each has different strengths: DDP/FSDP for simplicity, DeepSpeed for memory optimization, Megatron for maximum throughput at scale, and JAX for functional programming and TPU support.

---

## The Analogy

Think of training frameworks like construction management systems:

- **PyTorch DDP** = A reliable general contractor. Handles most jobs well, straightforward to work with, widely trusted. Not the most specialized, but solid and predictable.
- **PyTorch FSDP** = The same contractor with a new efficiency toolkit. Handles bigger projects by sharing resources across crews (sharding). Slightly more complex to configure, but handles larger buildings.
- **DeepSpeed** = A specialized large-building construction firm. Their secret: they break everything into tiny, memory-efficient pieces (ZeRO optimization). They can build skyscrapers on a budget that would normally require much more capital.
- **Megatron-LM** = The massive engineering firm that builds the world's tallest skyscrapers. Maximum efficiency, but you need their specific workflow and their equipment. Not for small projects — overkill and complex.
- **JAX/XLA** = A different construction philosophy entirely (prefab modular). Everything is designed to be compiled and assembled with maximum efficiency. Different tooling, different mindset, but potentially faster for the right project.

---

## Deep Dive

### PyTorch DDP (DistributedDataParallel)

```python
# PyTorch DDP - Standard data parallelism
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_ddp(rank, world_size):
    """Initialize DDP process group."""
    dist.init_process_group(
        backend="nccl",  # NVIDIA Collective Communication Library
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def train_with_ddp(rank, world_size, model, dataset, epochs=10):
    setup_ddp(rank, world_size)
    
    # Move model to GPU and wrap with DDP
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Distributed sampler ensures each GPU gets different data
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, sampler=sampler
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch
        for batch in dataloader:
            batch = {k: v.to(rank) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()  # DDP automatically syncs gradients here
            optimizer.step()
            optimizer.zero_grad()
    
    dist.destroy_process_group()

# Launch with torchrun (recommended)
# torchrun --nproc_per_node=8 train.py
```

```yaml
DDP_Characteristics:
  strengths:
    - "Simplest distributed training — minimal code changes from single GPU"
    - "Near-linear scaling (7.5x with 8 GPUs on fast interconnect)"
    - "Gradient communication overlapped with backward computation"
    - "Well-tested, stable, part of core PyTorch"
  limitations:
    - "Each GPU must hold full model + optimizer + gradients"
    - "Memory-bound: can't train models larger than single GPU memory"
  when_to_use:
    - "Model fits on one GPU (most common case for models <10B)"
    - "Want maximum simplicity and reliability"
    - "Up to 64 GPUs (beyond this, consider FSDP for memory)"
  communication:
    pattern: "AllReduce gradients during backward pass"
    backend: "NCCL (NVIDIA Collective Communication Library)"
    overlap: "Gradient buckets communicated while later layers compute"
```

### PyTorch FSDP (Fully Sharded Data Parallel)

```python
# PyTorch FSDP - Memory-efficient distributed training
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import LlamaForCausalLM, LlamaDecoderLayer

# Define sharding strategy
fsdp_config = {
    "sharding_strategy": ShardingStrategy.FULL_SHARD,  # ZeRO Stage 3
    "mixed_precision": MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    "auto_wrap_policy": transformer_auto_wrap_policy(
        transformer_layer_cls={LlamaDecoderLayer}  # Wrap each transformer block
    ),
    "cpu_offload": CPUOffload(offload_params=False),  # Enable for extreme memory savings
    "forward_prefetch": True,  # Prefetch next layer's params during forward
    "backward_prefetch": "backward_pre",  # Prefetch during backward
}

def train_with_fsdp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-4-8B")
    
    # Wrap with FSDP (shards model across all GPUs)
    model = FSDP(model, **fsdp_config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    for batch in dataloader:
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Save checkpoint (FSDP provides utilities for distributed saving)
    from torch.distributed.checkpoint import save
    save(model.state_dict(), checkpoint_dir)
```

```yaml
FSDP_Characteristics:
  sharding_strategies:
    full_shard: "ZeRO Stage 3 — shard params, grads, optimizer (maximum memory savings)"
    shard_grad_op: "ZeRO Stage 2 — shard grads and optimizer only (less communication)"
    no_shard: "Equivalent to DDP (no memory savings, lowest communication)"
    hybrid_shard: "Full shard within node, replicate across nodes (balance memory vs communication)"
    
  strengths:
    - "Native PyTorch (no external library dependency)"
    - "ZeRO-3 level memory efficiency"
    - "Composable with other PyTorch features (compile, activation checkpointing)"
    - "Simpler API than DeepSpeed for common cases"
    - "Good integration with HuggingFace Transformers and Trainer"
    
  limitations:
    - "No tensor parallelism (need separate solution for TP)"
    - "Less mature than DeepSpeed for very large scale (improving rapidly)"
    - "Configuration tuning needed for optimal performance"
    
  when_to_use:
    - "Model too large for DDP (need memory sharding)"
    - "Want to stay within PyTorch ecosystem (no external deps)"
    - "Fine-tuning or training models 7B-100B on 8-64 GPUs"
    - "Most common choice for 2026 fine-tuning workflows"
```

### DeepSpeed

```python
# DeepSpeed - Microsoft's distributed training library
import deepspeed
import torch

# DeepSpeed config (ds_config.json)
ds_config = {
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,  # 256 / (8 GPUs × 4 micro_batch) = 8
    
    "zero_optimization": {
        "stage": 3,  # ZeRO Stage 3 (full sharding)
        "offload_param": {
            "device": "cpu",  # Offload params to CPU (extreme memory savings)
            "pin_memory": True
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6
    },
    
    "bf16": {"enabled": True},
    
    "gradient_clipping": 1.0,
    
    "zero_allow_untested_optimizer": True,
    
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": True,
        "number_checkpoints": 4
    }
}

# Initialize DeepSpeed
model, optimizer, _, scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

# Training loop (simplified)
for batch in dataloader:
    loss = model(batch)
    model.backward(loss)   # DeepSpeed handles gradient sync
    model.step()           # DeepSpeed handles optimizer step
```

```yaml
DeepSpeed_Characteristics:
  unique_features:
    zero_infinity: "Offload to NVMe SSD — train models on minimal GPUs"
    zero_offload: "Offload optimizer states and parameters to CPU memory"
    sparse_attention: "Efficient attention for long sequences"
    progressive_layer_dropping: "Speed up training by skipping layers probabilistically"
    curriculum_learning: "Built-in curriculum learning support"
    1bit_adam: "Communication-compressed Adam optimizer"
    
  strengths:
    - "Maximum memory efficiency (ZeRO + offloading)"
    - "Can train very large models on limited hardware"
    - "Well-documented with many configuration options"
    - "HuggingFace Trainer integration (deepspeed flag)"
    - "DeepSpeed-Chat: RLHF training pipeline"
    - "Extensive profiling and monitoring built-in"
    
  limitations:
    - "External dependency (not part of core PyTorch)"
    - "Configuration complexity (many interacting options)"
    - "CPU offloading significantly slows training"
    - "Debugging can be harder (additional abstraction layer)"
    
  when_to_use:
    - "Need maximum memory efficiency (budget GPU hardware)"
    - "Want CPU/NVMe offloading for very large models on few GPUs"
    - "Training with RLHF (DeepSpeed-Chat pipeline)"
    - "Need advanced features (sparse attention, 1-bit communication)"
```

### Megatron-LM

```yaml
Megatron_LM:
  creator: "NVIDIA"
  purpose: "Maximum throughput training at scale (>100B parameters)"
  
  key_capabilities:
    tensor_parallelism: "Optimized column/row split for linear layers and attention"
    pipeline_parallelism: "Interleaved scheduling for minimal bubble"
    sequence_parallelism: "Distribute sequence dimension (reduces activation memory)"
    context_parallelism: "For very long sequences (>128K tokens)"
    expert_parallelism: "For Mixture-of-Experts (MoE) models"
    selective_activation_recomputation: "Recompute only expensive operations"
    
  architecture_support:
    - "GPT-style (decoder-only)"
    - "BERT-style (encoder-only)"
    - "T5-style (encoder-decoder)"
    - "Mixture of Experts"
    
  strengths:
    - "Highest achievable throughput (MFU >50% at scale)"
    - "Optimized for NVIDIA hardware (NVLink, NVSwitch, InfiniBand)"
    - "Proven at scale (used for training 175B-1T models)"
    - "Combined with DeepSpeed for ZeRO + Megatron parallelism"
    
  limitations:
    - "Complex setup (many configuration parameters)"
    - "NVIDIA-specific (doesn't run on AMD GPUs or TPUs)"
    - "Limited model architecture support (primarily Transformer variants)"
    - "Not for fine-tuning small models (overkill)"
    
  when_to_use:
    - "Pre-training models >50B parameters on >100 GPUs"
    - "Need maximum hardware utilization (enterprises, labs)"
    - "All-NVIDIA infrastructure with NVLink + InfiniBand"
    
  mfu:
    definition: "Model FLOPs Utilization — fraction of peak hardware FLOPS actually used"
    good: ">40% MFU"
    excellent: ">50% MFU"
    megatron_achieves: "50-60% MFU on well-configured clusters"
```

### JAX / XLA

```yaml
JAX_XLA:
  jax:
    what: "Google's framework for numerical computing (NumPy + autodiff + XLA compilation)"
    paradigm: "Functional programming (pure functions, explicit state)"
    key_features:
      jit: "Just-In-Time compilation via XLA (optimized GPU/TPU kernels)"
      vmap: "Automatic vectorization (batch operations without explicit loops)"
      pmap: "Parallel map (data parallelism across devices)"
      pjit: "Partitioned JIT (flexible model + data parallelism)"
      
  xla:
    full_name: "Accelerated Linear Algebra"
    what: "Compiler that optimizes computation graphs for hardware"
    advantages:
      - "Operator fusion (combines multiple ops into optimized kernels)"
      - "Memory layout optimization"
      - "Cross-device communication optimization"
      - "Works on GPUs and TPUs"
      
  training_frameworks_on_jax:
    maxtext: "Google's scalable LLM training (used for Gemini-class models)"
    t5x: "T5/encoder-decoder training framework"
    paxml: "Google's production training framework (PaLM, Gemini)"
    levanter: "Stanford's JAX-based LLM training"
    
  strengths:
    - "First-class TPU support (Google's custom AI chips)"
    - "XLA compilation produces highly optimized code"
    - "Functional paradigm makes parallelism composition natural"
    - "Better scaling efficiency for very large models on TPUs"
    
  limitations:
    - "Smaller ecosystem than PyTorch (fewer tutorials, libraries)"
    - "Functional paradigm has learning curve for PyTorch developers"
    - "Debugging compiled code is harder"
    - "Primarily used at Google and Google-adjacent research labs"
    
  when_to_use:
    - "Training on TPUs (JAX has best TPU support)"
    - "Google Cloud infrastructure"
    - "Research requiring custom parallelism strategies"
    - "Teams already familiar with functional programming"
```

---

## How It Works in Practice

### Framework Selection Decision Tree

```yaml
Decision_Tree:
  question_1: "How large is your model?"
  
  under_1b:
    answer: "Use PyTorch DDP"
    reason: "Simple, efficient, model fits on any modern GPU"
    
  1b_to_10b:
    question_2: "Does it fit on one GPU with mixed precision?"
    yes: "Use PyTorch DDP + AMP (Automatic Mixed Precision)"
    no: "Use PyTorch FSDP (shard_grad_op or full_shard)"
    
  10b_to_70b:
    question_2: "Are you fine-tuning or pre-training?"
    fine_tuning:
      method: "FSDP + LoRA/QLoRA"
      gpus: "4-8 GPUs sufficient with QLoRA"
    pre_training:
      method: "FSDP full_shard or DeepSpeed ZeRO-3"
      gpus: "16-64 GPUs (more for faster training)"
      
  over_70b:
    question_2: "Do you have NVLink-connected GPU clusters?"
    yes: 
      method: "Megatron-LM (TP + PP) + DeepSpeed (ZeRO)"
      gpus: "64-1000+ GPUs"
    no:
      method: "DeepSpeed ZeRO-3 with offloading"
      trade_off: "Slower but works on commodity hardware"
      
  on_tpus:
    method: "JAX + MaxText/Paxml"
    reason: "JAX has first-class TPU support, PyTorch TPU support is secondary"
```

---

## Interview Tip

> When asked about training frameworks: "My framework choice depends on model size and infrastructure. For models under 10B that fit on one GPU: PyTorch DDP — simplest, fastest, no redundant memory. For 10-70B models or memory-constrained setups: PyTorch FSDP — native ZeRO-3, shards everything, good HuggingFace integration. For maximum memory efficiency on limited hardware: DeepSpeed ZeRO-3 with CPU offloading — can train surprisingly large models on few GPUs at the cost of speed. For frontier models (100B+) requiring maximum throughput: Megatron-LM with tensor + pipeline parallelism, combined with DeepSpeed for ZeRO optimization. On TPUs: JAX with MaxText. The key trade-off is always simplicity vs efficiency. DDP is 5 lines of code but wastes memory. Megatron squeezes maximum utilization but takes days to configure correctly. In practice, FSDP hits the sweet spot for 80% of 2026 training workloads."

---

## Common Mistakes

1. **Starting with Megatron for a 7B model** — Over-engineering the training setup. Megatron's complexity is only justified for >50B parameter pre-training. For fine-tuning 7B-13B models, FSDP + LoRA is simpler and often faster (less configuration time, less debugging time).

2. **Not setting gradient accumulation correctly** — With 8 GPUs and a target global batch size of 256, you need per-GPU batch 4 with gradient accumulation 8 (4 × 8 × 8 = 256). Getting this wrong means training with wrong effective batch size, leading to convergence issues.

3. **Using ZeRO-3 with CPU offloading by default** — CPU offloading makes training 3-10x slower. It's a last resort for when GPU memory is truly insufficient, not a default configuration. Try ZeRO-2 or reduce model size first.

4. **Ignoring FSDP wrapping policy** — FSDP's performance depends heavily on WHICH modules get wrapped (sharded). Wrapping too finely → excessive communication. Wrapping too coarsely → memory spikes. The `transformer_auto_wrap_policy` wrapping each transformer layer is usually optimal.

5. **Mixing framework versions blindly** — DeepSpeed + specific PyTorch version + specific CUDA version + specific NCCL version can have subtle incompatibilities causing silent performance degradation or crashes. Always test the full stack version combination.

---

## Key Takeaways

- PyTorch DDP: simplest, most efficient for models that fit on one GPU (~5 lines of code change)
- PyTorch FSDP: native ZeRO-3, handles 7B-100B, the "default choice" for 2026 fine-tuning/training
- DeepSpeed: maximum memory efficiency with ZeRO + offloading, configurable but complex
- Megatron-LM: maximum throughput for frontier models (>100B), requires NVIDIA hardware expertise
- JAX/XLA: best for TPUs, functional paradigm, used by Google for Gemini-class models
- MFU (Model FLOPs Utilization): key efficiency metric — good is >40%, excellent is >50%
- Decision: DDP (small) → FSDP (medium) → DeepSpeed ZeRO-3 (memory-constrained) → Megatron (frontier)
- Framework choice affects development velocity: simpler framework → faster iteration → often better results
- HuggingFace Trainer supports DDP, FSDP, and DeepSpeed — simplifies switching between them
- Always benchmark your specific model + hardware before committing to a framework
