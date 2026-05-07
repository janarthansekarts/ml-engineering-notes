# Training Debugging

## The Problem / Why This Matters

ML (Machine Learning) training is notoriously difficult to debug. Unlike traditional software where bugs produce clear error messages or wrong outputs, training issues manifest as: loss that doesn't decrease (is it a bug or just slow convergence?), loss that suddenly spikes (bad data? learning rate? hardware error?), NaN (Not a Number) values that appear after thousands of steps, memory that gradually leaks until OOM (Out of Memory), or models that train perfectly on metrics but produce garbage outputs. These issues are hard to diagnose because: (1) training takes hours/days — you can't iterate quickly, (2) the systems are stochastic — same code can work one run and fail the next, (3) distributed systems add complexity — bugs that only appear at multi-GPU scale, (4) the root cause is often far from the symptom — a data preprocessing bug causes loss spikes 10,000 steps later. In 2026, with training runs costing $1K-$5M, the ability to quickly diagnose and fix training issues directly impacts budgets and timelines. ML engineers who can debug training efficiently save their organizations enormous amounts of compute waste.

---

## The Analogy

Think of debugging ML training like diagnosing a sick patient:

- **Loss not decreasing** = Patient not responding to treatment. Could be wrong diagnosis (wrong task formulation), wrong medication (bad hyperparameters), or the patient is already healthy (model already converged, you're just seeing noise).
- **Loss spikes** = Sudden fever. Could be an infection (bad data batch), allergic reaction (learning rate too high), or a false alarm from a faulty thermometer (hardware error producing wrong gradients).
- **NaN loss** = Patient flatlined. Something catastrophic happened. Check the vitals in reverse order: was there an overflow (too large numbers)? Division by zero? Infinity propagated from a single bad value?
- **Slow training** = Patient recovering too slowly. Either the treatment is working but you're impatient (check: is loss DECREASING, just slowly?), or there's a hidden obstruction (data loading bottleneck, communication overhead, GPU underutilization).

---

## Deep Dive

### Loss Curve Diagnosis

```yaml
Loss_Curve_Patterns:
  healthy_training:
    pattern: "Rapid initial decrease → gradual convergence → plateau"
    what_to_expect: "Loss drops quickly in first 10% of training, then slows"
    action: "None — this is correct behavior"
    
  loss_not_decreasing:
    pattern: "Loss stays flat from the start"
    likely_causes:
      learning_rate_too_low:
        diagnosis: "Loss barely moves, gradient norm is tiny"
        fix: "Increase learning rate by 10-100x"
        
      learning_rate_too_high:
        diagnosis: "Loss oscillates wildly, doesn't converge"
        fix: "Reduce learning rate by 10-100x"
        
      data_bug:
        diagnosis: "Loss is random (matching random baseline)"
        fix: "Check data pipeline: are labels correct? Is preprocessing working?"
        test: "Overfit on 1 batch — if model can't memorize 1 batch, something is fundamentally wrong"
        
      wrong_loss_function:
        diagnosis: "Loss has wrong scale or model learns wrong thing"
        fix: "Verify loss function matches task (CrossEntropy for classification, etc.)"
        
      frozen_parameters:
        diagnosis: "Some parameters have zero gradient"
        fix: "Check requires_grad, verify LoRA targets, check for accidental .detach()"
        
  loss_spikes:
    pattern: "Training going well, then sudden 5-100x jump in loss"
    likely_causes:
      bad_data_batch:
        diagnosis: "Spike happens once, loss recovers quickly"
        fix: "Filter extreme values in data, add data validation"
        detection: "Log batch statistics, detect outliers"
        
      learning_rate_too_high:
        diagnosis: "Spikes become more frequent over time"
        fix: "Reduce max learning rate, adjust schedule"
        
      gradient_explosion:
        diagnosis: "Gradient norm spikes before loss spike"
        fix: "Reduce gradient clipping threshold, lower learning rate"
        
      hardware_error:
        diagnosis: "Spike is completely random, inconsistent across runs"
        fix: "Check GPU ECC errors, run NCCL tests, swap suspected GPU"
        
  loss_divergence:
    pattern: "Loss increases continuously (model unlearning)"
    likely_causes:
      lr_too_high: "Learning rate overshoots optimal values"
      catastrophic_forgetting: "Fine-tuning destroys pre-trained knowledge"
      data_distribution_shift: "Training data changes dramatically"
    fixes:
      - "Reduce learning rate"
      - "Add weight decay or lower fine-tuning LR (for catastrophic forgetting)"
      - "Check data pipeline for ordering issues"
      
  nan_loss:
    pattern: "Loss becomes NaN (usually suddenly, irrecoverably)"
    debugging_steps:
      1: "Check which layer first produces NaN: add hooks to all layers"
      2: "Check inputs to that layer (any Inf or NaN values?)"
      3: "Check for division by zero (LayerNorm epsilon, attention denominator)"
      4: "Check precision (FP16 overflow: values > 65504?)"
      5: "Check loss function (log(0)? negative values in log?)"
    common_fixes:
      - "Use BF16 instead of FP16 (wider dynamic range)"
      - "Add epsilon to all denominators (1e-8)"
      - "Clamp values before log operations"
      - "Reduce learning rate (prevent values from growing too large)"
```

### Debugging Tools and Techniques

```python
# Debugging training issues - Practical tools

import torch
import torch.nn as nn

# 1. Overfit one batch test (FIRST thing to try)
def overfit_one_batch(model, batch, num_steps=100):
    """If model can't memorize one batch, something is fundamentally broken."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for step in range(num_steps):
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if step % 10 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")
    
    # Loss should be near 0 after 100 steps on 1 batch
    # If not: model architecture, loss function, or data is broken
    assert loss.item() < 0.1, "Cannot overfit one batch — fundamental issue!"

# 2. Gradient monitoring hooks
def register_gradient_hooks(model):
    """Monitor gradient statistics for each layer."""
    gradient_stats = {}
    
    def hook_fn(name):
        def hook(grad):
            gradient_stats[name] = {
                "mean": grad.mean().item(),
                "std": grad.std().item(),
                "max": grad.abs().max().item(),
                "has_nan": torch.isnan(grad).any().item(),
                "has_inf": torch.isinf(grad).any().item(),
            }
        return hook
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(hook_fn(name))
    
    return gradient_stats

# 3. NaN detection hook
def detect_nan_forward(model):
    """Register forward hooks to detect first layer that produces NaN."""
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                print(f"NaN detected in: {module.__class__.__name__}")
                print(f"  Input has NaN: {any(torch.isnan(i).any() for i in input if isinstance(i, torch.Tensor))}")
                print(f"  Output shape: {output.shape}")
                raise RuntimeError(f"NaN in {module.__class__.__name__}")
    
    for name, module in model.named_modules():
        module.register_forward_hook(hook_fn)

# 4. Learning rate finder
def lr_finder(model, dataloader, min_lr=1e-7, max_lr=10, num_steps=100):
    """Find optimal learning rate by gradually increasing it."""
    import math
    
    model_state = model.state_dict()  # Save to restore later
    optimizer = torch.optim.Adam(model.parameters(), lr=min_lr)
    
    lr_mult = (max_lr / min_lr) ** (1 / num_steps)
    lrs, losses = [], []
    
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break
            
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        lrs.append(optimizer.param_groups[0]["lr"])
        losses.append(loss.item())
        
        # Increase learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] *= lr_mult
        
        # Stop if loss is exploding
        if loss.item() > 4 * min(losses):
            break
    
    model.load_state_dict(model_state)  # Restore model
    # Plot lrs vs losses — optimal LR is where loss decreases fastest
    return lrs, losses
```

### Gradient Issues

```yaml
Gradient_Problems:
  vanishing_gradients:
    symptom: "Deep layers have near-zero gradients, model doesn't learn deep features"
    diagnosis: "Plot gradient norm per layer — decreases exponentially with depth"
    causes:
      - "Sigmoid/tanh activations saturate (gradients → 0)"
      - "Too many layers without residual connections"
      - "Initialization too small"
    fixes:
      - "Use ReLU/GELU activations"
      - "Add residual connections (skip connections)"
      - "Use proper initialization (He/Kaiming for ReLU)"
      - "Use LayerNorm/BatchNorm"
      
  exploding_gradients:
    symptom: "Loss spikes, NaN values, parameter magnitudes grow unboundedly"
    diagnosis: "Gradient norm grows exponentially, often precedes loss spike"
    causes:
      - "Learning rate too high"
      - "Deep networks without gradient clipping"
      - "Recurrent architectures (long sequences)"
    fixes:
      - "Gradient clipping (max_norm=1.0 is standard)"
      - "Reduce learning rate"
      - "Use gradient accumulation with smaller micro-batches"
      - "Check for numerical issues (Inf values in forward pass)"
      
  dead_neurons:
    symptom: "Some neurons always output 0 (ReLU killed them)"
    diagnosis: "Check activation statistics — fraction of zero activations"
    causes:
      - "Large negative bias pushes all inputs below 0 (ReLU outputs 0)"
      - "Learning rate too high caused large weight updates"
    fixes:
      - "Use LeakyReLU or GELU instead of ReLU"
      - "Reduce learning rate"
      - "Better initialization"
      
  gradient_accumulation_bug:
    symptom: "Training works with accum_steps=1 but fails with accum_steps>1"
    common_bugs:
      - "Not dividing loss by accumulation_steps (gradients too large)"
      - "Not using model.no_sync() in DDP (syncing every micro-step)"
      - "Calling optimizer.zero_grad() at wrong time"
    fix: "Review accumulation loop carefully — loss/=accum, zero_grad before loop, step after loop"
```

### Distributed Training Debugging

```yaml
Distributed_Debugging:
  common_issues:
    nccl_timeout:
      symptom: "Training hangs, then NCCL timeout error after 30 min"
      causes:
        - "One GPU is slower (data loading bottleneck on one rank)"
        - "Network issue (one node has degraded connectivity)"
        - "Deadlock (ranks calling collectives in different order)"
        - "GPU hardware failure (one GPU silently broken)"
      debugging:
        - "Add barrier + timing around each collective"
        - "Check which rank is last to reach the collective"
        - "Run NCCL test independently on each node pair"
        - "Check GPU health: nvidia-smi, DCGM metrics"
        
    gradient_desync:
      symptom: "Model diverges or performance degrades in multi-GPU vs single-GPU"
      causes:
        - "Incorrect gradient synchronization (custom ops not synced)"
        - "Different data on ranks (sampler bug)"
        - "Non-deterministic operations producing different results per rank"
      debugging:
        - "Compare model weights across ranks (should be identical after sync)"
        - "Compare loss across ranks (should be different — different data)"
        - "Compare gradients before/after AllReduce"
        
    oom_on_one_rank:
      symptom: "OOM error on rank 0 (or specific rank) only"
      causes:
        - "Rank 0 has extra responsibility (logging, checkpointing, gathering metrics)"
        - "Uneven data distribution (one rank gets longer sequences)"
        - "Memory leak in logging/tracking code"
      fix:
        - "Offload logging to separate process"
        - "Ensure balanced data distribution"
        - "Use max_length padding/truncation"
        
  debugging_tools:
    torch_distributed_debug:
      how: "TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun ... "
      provides: "Detailed logging of all collective operations"
      
    nccl_debug:
      how: "NCCL_DEBUG=INFO torchrun ..."
      provides: "NCCL library debug output (topology, algorithm selection)"
      
    pytorch_profiler:
      how: |
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, 
                       torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs")
        ) as profiler:
            for step, batch in enumerate(dataloader):
                train_step(batch)
                profiler.step()
      provides: "Detailed GPU kernel timing, memory allocation, CUDA streams"
```

### Performance Debugging

```yaml
Performance_Debugging:
  symptoms_and_causes:
    low_gpu_utilization:
      measurement: "nvidia-smi shows <80% GPU utilization"
      causes:
        - "Data loading bottleneck (CPU can't prepare data fast enough)"
        - "Communication overhead (too much time in AllReduce)"
        - "Python overhead (GIL contention, inefficient code)"
        - "Memory pressure (frequent garbage collection)"
      diagnosis: "Profile with PyTorch Profiler — look at GPU idle gaps"
      
    slow_data_loading:
      measurement: "GPU idle between batches (visible in profiler)"
      diagnosis: "DataLoader workers < GPU consumption rate"
      fixes:
        - "Increase num_workers (2-4× num GPUs)"
        - "Pre-tokenize data (avoid CPU tokenization per batch)"
        - "Use faster storage (local NVMe > network storage)"
        - "Use persistent_workers=True"
        - "Profile with NVIDIA DALI for GPU-accelerated preprocessing"
        
    communication_bottleneck:
      measurement: "Profiler shows large AllReduce/AllGather blocks"
      diagnosis: "Compute-to-communication ratio too low"
      fixes:
        - "Increase batch size (more compute per communication)"
        - "Use gradient accumulation (less frequent sync)"
        - "Overlap communication with computation (check DDP bucket sizes)"
        - "Upgrade network (InfiniBand if using Ethernet)"
        - "Use gradient compression (for slow networks)"
        
    memory_leak:
      symptom: "GPU memory usage grows over time until OOM"
      common_causes:
        - "Storing tensors in list that grows (logging raw tensors)"
        - "Not detaching tensors from computation graph (loss history with grad)"
        - "Accumulating computational graph across steps"
      detection: "torch.cuda.memory_summary() — check for growing allocations"
      fix: "Use .item() for scalars, .detach() for logged tensors, clear history"
```

---

## How It Works in Practice

### Debugging Workflow

```yaml
Example:
  scenario: "Fine-tuning 8B model, loss stuck at 2.3 for 500 steps"
  
  debugging_steps:
    step_1_sanity_check:
      action: "Overfit one batch"
      result: "Model CAN memorize one batch (loss → 0.01 in 50 steps)"
      conclusion: "Model architecture and loss function are correct"
      
    step_2_check_data:
      action: "Print 10 random training examples, verify correctness"
      result: "Data looks correct, labels match inputs"
      conclusion: "Data pipeline is working"
      
    step_3_check_gradients:
      action: "Log gradient norm per layer"
      result: "Most layers have gradient norm ~0.001, but LoRA layers have ~0.0001"
      conclusion: "Gradients are flowing but very small for LoRA parameters"
      
    step_4_check_lr:
      action: "Run LR finder (sweep from 1e-7 to 1e-1)"
      result: "Loss decreases fastest at lr=2e-4 (currently using 2e-5)"
      conclusion: "Learning rate is 10x too low"
      
    step_5_fix:
      action: "Increase learning rate from 2e-5 to 2e-4"
      result: "Loss immediately starts decreasing, reaches 1.5 within 500 steps"
      
  time_spent: "45 minutes (vs potentially days of wasted training)"
  compute_saved: "$200+ (avoided running broken training for longer)"
```

---

## Interview Tip

> When asked about debugging training issues: "My debugging follows a systematic workflow. First: can the model overfit one batch? If not, there's a fundamental issue (architecture, loss function, data format). If yes: check the loss curve pattern. Flat loss → learning rate issue (try LR finder). Loss spikes → check gradient norm before spikes, usually bad data or LR too high. NaN → add forward hooks to find which layer first produces NaN, check for precision overflow. For performance issues: I use PyTorch Profiler to identify whether I'm compute-bound, memory-bound, communication-bound, or data-loading-bound. Each has different fixes: data-loading → more workers, faster storage; communication → larger batches, gradient compression; memory → activation checkpointing, reduce batch size. Key principle: always instrument first, diagnose second, fix third. Never guess — measuring is cheaper than wasting GPU hours on the wrong fix."

---

## Common Mistakes

1. **Not trying to overfit one batch first** — Spending hours debugging hyperparameters when the actual issue is a data pipeline bug (wrong labels, wrong preprocessing). The overfit-one-batch test takes 30 seconds and catches fundamental issues immediately.

2. **Assuming loss spikes are catastrophic** — Panicking at a single loss spike and restarting training. Many spikes are benign (bad data batch) and training recovers naturally. Only intervene if spikes are recurring or loss doesn't recover within a few hundred steps.

3. **Debugging distributed issues in distributed mode** — Trying to debug a subtle bug across 8 GPUs simultaneously. First reproduce on 1 GPU if possible — eliminates communication issues and makes debugging 8x easier. Only debug in distributed mode if the bug is distribution-specific.

4. **Not logging gradient norms** — Running training without monitoring gradient statistics. Gradient norm is the earliest indicator of instability — it spikes BEFORE loss spikes. Always log `torch.nn.utils.clip_grad_norm_` return value.

5. **Changing multiple things simultaneously** — "Loss is stuck, so I'll change the learning rate, add warmup, switch optimizer, AND change the data." Now you don't know which change fixed it (or made it worse). Change one variable at a time, observe effect, then proceed.

---

## Key Takeaways

- Overfit one batch FIRST — if this fails, you have a fundamental bug (architecture, loss, or data)
- Loss patterns: flat = LR issue, spikes = bad data or LR too high, NaN = precision overflow
- Gradient norm is the earliest warning signal — log it every step, alert on >5x running average
- LR finder: sweep learning rates to find optimal range before committing to a full training run
- NaN debugging: add forward hooks to all layers, find which layer FIRST produces NaN
- Distributed debugging: NCCL timeouts → find which rank is slow; gradient desync → compare weights across ranks
- Performance profiling: PyTorch Profiler identifies whether bottleneck is compute, memory, communication, or data
- Memory leaks: storing tensors with gradients in lists — use .item() and .detach()
- Change one variable at a time — scientific method applies to ML debugging
- Instrument before fixing — measuring saves more GPU-hours than guessing
