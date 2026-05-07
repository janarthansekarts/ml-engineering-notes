# Cost Optimization for Cloud ML

## The Problem / Why This Matters

ML workloads are among the most expensive in cloud computing. A single H100 GPU instance costs $8-15/hour on-demand — a 4-node training cluster (32 GPUs) runs $3,000-4,000 per day. Production serving with auto-scaling can easily exceed $50,000/month. Without deliberate cost optimization, ML teams regularly overspend by 40-70% due to: idle GPUs (provisioned but not computing), over-provisioned instances (H100 for a workload that fits on A10G), always-on endpoints (no traffic at 3 AM but still paying full price), accumulated storage (checkpoints and experiments never cleaned up), and inefficient training (poorly optimized code wastes GPU cycles). In 2026, with H100/H200 availability still constrained and prices high, cost optimization is a core ML engineering skill — not a finance concern. The best ML teams treat GPU-hours like a precious resource: every optimization that reduces training time (better data loading, mixed precision, gradient accumulation) directly reduces cost. This file covers: purchasing strategies (spot, reserved, on-demand), compute optimization (right-sizing, scheduling, auto-scaling), training efficiency (reduce GPU-hours needed), serving cost control (scale-to-zero, batching), and organizational practices (budgets, chargebacks, FinOps).

---

## The Analogy

Think of ML cost optimization like running a fleet of taxis:

- **On-demand GPUs** = Hailing a taxi at full fare. Always available, most expensive. Use only when you need it RIGHT NOW and nothing else is available.
- **Reserved instances** = Signing a 1-year contract with a taxi company. Guaranteed car every day at 30-60% discount. Makes sense if you commute daily (steady serving workload).
- **Spot instances** = Taking cancelled/available rides at 60-90% discount. Car might become unavailable mid-trip (interruption), so only works if you can handle being dropped off and resuming later (checkpointing).
- **Auto-scaling** = Only having taxis when passengers exist. At 3 AM, zero taxis running. At rush hour, maximum fleet. Don't pay for parked taxis.
- **Right-sizing** = Don't hire a 50-seat bus for one passenger. Match vehicle (GPU type) to passenger count (model size).

---

## Deep Dive

### Cost Optimization Framework

```yaml
Cost_Optimization_Framework:
  principle: "Spend every GPU-hour on actual computation, not waste"
  
  three_pillars:
    pillar_1_reduce_price:
      what: "Pay less per GPU-hour"
      strategies:
        spot_instances:
          savings: "60-90% vs. on-demand"
          requirement: "Checkpointing every 5-15 minutes"
          best_for: "Training, hyperparameter search, batch processing"
          not_for: "Real-time serving (can't tolerate interruptions)"
          
        reserved_committed:
          savings: "30-60% vs. on-demand"
          types:
            aws: "Reserved Instances (1-3 year), Savings Plans"
            gcp: "Committed Use Discounts (1-3 year)"
            azure: "Reserved VM Instances (1-3 year)"
          best_for: "Steady-state serving, persistent training clusters"
          strategy: "Reserve baseline capacity, spot for burst"
          
        alternative_providers:
          what: "Use non-hyperscaler GPU clouds for training"
          providers: ["CoreWeave", "Lambda Labs", "RunPod", "Together AI"]
          savings: "20-50% vs. AWS/GCP/Azure for equivalent GPU"
          trade_off: "Fewer enterprise features, less ecosystem integration"
          
        negotiated_pricing:
          what: "Enterprise discount programs (EDP)"
          threshold: "$1M+/year cloud spend"
          typical_discount: "15-30% on top of public pricing"
          
    pillar_2_reduce_time:
      what: "Complete work in fewer GPU-hours"
      strategies:
        mixed_precision:
          what: "Train in FP16/BF16 instead of FP32"
          speedup: "2× on modern GPUs (Tensor Cores)"
          cost_impact: "50% reduction in training GPU-hours"
          trade_off: "Minimal quality impact with proper loss scaling"
          
        efficient_data_loading:
          what: "Ensure GPUs are never idle waiting for data"
          metrics: "GPU utilization should be >80% during training"
          fixes: ["More DataLoader workers", "NVMe cache", "Efficient formats"]
          
        gradient_accumulation:
          what: "Simulate larger batch size without more GPUs"
          effect: "Better convergence → fewer total steps → less time"
          use_when: "Can't fit optimal batch size in GPU memory"
          
        distributed_training:
          what: "Use more GPUs to finish faster"
          scaling: "Near-linear speedup with proper parallelism"
          roi: "4× GPUs for 3.5× speedup = cheaper if spot price is low"
          
        early_stopping:
          what: "Stop training when validation metric plateaus"
          savings: "Often saves 20-40% of training time"
          requirement: "Proper validation monitoring every N steps"
          
    pillar_3_reduce_waste:
      what: "Eliminate unnecessary GPU-hours"
      strategies:
        auto_shutdown:
          what: "Stop idle compute automatically"
          targets:
            dev_instances: "Shutdown after 30 min inactivity"
            training_clusters: "Scale to 0 between jobs"
            endpoints: "Scale down during off-peak hours"
          typical_savings: "40-60% on development compute"
          
        right_sizing:
          what: "Use smallest GPU that fits your workload"
          common_waste: "Team defaults to H100 ($8/hr) when A10G ($1/hr) suffices"
          process: "Profile first → select GPU → validate utilization"
          
        storage_cleanup:
          what: "Delete old checkpoints, experiments, and cached data"
          rule: "Lifecycle policies on all ML storage"
          impact: "Usually saves 20-40% on storage (small absolute $)"
          
        scheduling:
          what: "Run non-urgent work during off-peak / cheap hours"
          patterns:
            - "Hyperparameter tuning at night (lower spot prices)"
            - "Batch inference during off-peak (fewer serving instances)"
            - "Training data preprocessing on spot instances"
```

### Purchasing Strategy Optimization

```python
# Purchasing strategy calculator

"""
Cost optimization through strategic purchasing of cloud compute.
Spot, reserved, on-demand — when to use each.
"""

purchasing_strategy = {
    "workload_analysis": {
        "step_1_categorize": {
            "steady_state": {
                "description": "Consistent usage 24/7 (serving endpoints)",
                "purchasing": "Reserved/Committed (30-60% savings)",
                "examples": [
                    "Production model endpoints (always-on)",
                    "Feature store online tier (always serving)",
                    "MLflow server (always running)",
                ],
            },
            "predictable_burst": {
                "description": "Regular patterns (daily training, batch scoring)",
                "purchasing": "Spot (60-90% savings) + Reserved for minimum",
                "examples": [
                    "Daily retraining (runs 2 AM, takes 4 hours)",
                    "Weekly batch scoring (overnight processing)",
                    "Nightly data processing pipeline",
                ],
            },
            "unpredictable_burst": {
                "description": "Variable usage (development, experimentation)",
                "purchasing": "Spot (preferred) + On-demand (fallback)",
                "examples": [
                    "ML engineer experimentation (varies daily)",
                    "Hyperparameter search (varies by project)",
                    "A/B testing new models (temporary capacity)",
                ],
            },
        },
        
        "step_2_portfolio": {
            "strategy": "Mix purchasing types like an investment portfolio",
            "typical_mix": {
                "reserved": "40% (steady-state serving workloads)",
                "spot": "45% (training + batch processing)",
                "on_demand": "15% (urgent/fallback only)",
            },
            "savings_calculation": {
                "100_percent_on_demand": "$100,000/month (baseline)",
                "optimized_portfolio": "$45,000/month (55% savings)",
                "breakdown": {
                    "reserved_40pct": "$40K × 0.65 = $26K (35% savings)",
                    "spot_45pct": "$45K × 0.25 = $11.25K (75% savings)",
                    "on_demand_15pct": "$15K × 1.0 = $15K (no savings)",
                    "total": "$52,250 (~48% overall savings)",
                },
            },
        },
    },
    
    "spot_instance_strategies": {
        "diversification": {
            "what": "Use multiple instance types and AZs for spot",
            "why": "Single instance type in one AZ has high interruption rate",
            "implementation": [
                "Define 3-5 equivalent instance types (same GPU, different configs)",
                "Spread across all AZs in region",
                "Use spot fleet or managed instance groups",
            ],
            "example": {
                "primary": "p4d.24xlarge (8× A100 80GB)",
                "alternatives": ["p4de.24xlarge", "p4dn.24xlarge"],
                "azs": ["us-east-1a", "us-east-1b", "us-east-1c"],
            },
        },
        "interruption_handling": {
            "detection": "AWS: 2-minute warning via metadata. GCP: 30-second warning.",
            "response": [
                "Save checkpoint immediately (emergency save to local NVMe)",
                "Upload checkpoint to S3/GCS (async, may not complete)",
                "Signal training to stop gracefully",
            ],
            "resume": [
                "New spot instance provisioned (or on-demand fallback)",
                "Load latest checkpoint from object storage",
                "Resume from exact training step",
            ],
        },
        "cost_comparison_example": {
            "training_job": "7B model fine-tuning, 24 hours on 4× A100",
            "on_demand": "4 × $32.77/hr × 24hr = $3,146",
            "spot_avg_75_off": "4 × $8.19/hr × 28hr = $918 (extra 4hr for interruptions)",
            "savings": "$2,228 (71% savings) for one training run",
        },
    },
}


# Auto-scaling configuration for ML serving
autoscaling_config = {
    "real_time_serving": {
        "metrics": [
            {
                "metric": "InvocationsPerInstance",
                "target": 100,  # Requests per instance per minute
                "scale_up_cooldown": 60,  # seconds
                "scale_down_cooldown": 300,  # seconds (slower scale-down)
            },
            {
                "metric": "GPUUtilization",
                "target": 70,  # percent
                "use_when": "GPU-bound workloads (LLM inference)",
            },
        ],
        "capacity": {
            "min": 1,  # Never scale below 1 (or 0 for serverless)
            "max": 20,  # Budget cap
            "desired": 3,  # Starting point
        },
        "schedule_based": {
            "description": "Combine with predictive scaling",
            "rules": [
                {"time": "06:00", "min": 5, "note": "Business hours start"},
                {"time": "22:00", "min": 1, "note": "Off-peak"},
                {"time": "02:00", "min": 1, "note": "Overnight minimum"},
            ],
        },
    },
    
    "batch_processing": {
        "strategy": "Scale to workload, scale to zero when done",
        "implementation": {
            "trigger": "SQS queue depth > 0 (messages waiting)",
            "scale_up": "Add instances proportional to queue depth",
            "scale_down": "Remove instances when queue empty for 5 minutes",
            "min": 0,  # Scale to zero (no cost when no work)
        },
    },
    
    "training_cluster": {
        "strategy": "Right-size cluster per job, release after completion",
        "implementation": {
            "job_scheduler": "Ray, Slurm, or Kubernetes (allocate per job)",
            "idle_timeout": "Scale down GPU nodes after 10 min idle",
            "queue_priority": "Prioritize by deadline, cost-sensitivity",
        },
    },
}
```

### Training Efficiency

```yaml
Training_Efficiency:
  goal: "Reduce total GPU-hours for same model quality"
  
  techniques:
    mixed_precision_training:
      what: "Use FP16/BF16 for computation, FP32 for accumulation"
      speedup: "1.5-2× on modern GPUs"
      memory_savings: "~50% (fit larger batch or model)"
      implementation: "PyTorch AMP (Automatic Mixed Precision) or BF16 native"
      cost_impact: "50% reduction in GPU-hours"
      
    fp8_training:
      what: "Use FP8 precision (H100/H200/B200 only)"
      speedup: "~2× over FP16 on H100 Transformer Engine"
      frameworks: "NVIDIA Transformer Engine, MS-AMP"
      caveat: "Requires careful loss scaling, not all layers support FP8"
      
    gradient_checkpointing:
      what: "Recompute activations instead of storing (save memory)"
      memory_savings: "50-70% activation memory"
      compute_overhead: "~30% extra computation"
      net_effect: "Enables larger batch sizes → faster convergence → fewer steps"
      
    efficient_optimizers:
      what: "Use memory-efficient optimizers"
      options:
        adam_8bit: "bitsandbytes 8-bit Adam (75% less optimizer memory)"
        adafactor: "O(1) memory for second moments"
        lion: "Only stores momentum (50% less than Adam)"
      effect: "More GPU memory available → larger batch → fewer steps"
      
    learning_rate_scheduling:
      what: "Aggressive warmup + cosine decay"
      effect: "Converges in fewer steps (10-20% reduction)"
      patterns:
        - "Warmup: linear 0 → peak over first 5% of steps"
        - "Decay: cosine from peak → 0.1× peak"
        - "One-cycle: fast convergence for fine-tuning"
        
    data_efficiency:
      curriculum_learning: "Train on easy examples first, hard examples later"
      data_pruning: "Remove redundant/noisy training examples (keep 50%, same quality)"
      smart_batching: "Group similar-length sequences (reduce padding waste)"
      effect: "Same quality with 30-50% less data = 30-50% less training time"
      
  roi_calculation:
    baseline: "70B model training: 1000 GPU-hours at $4/hr (spot H100) = $4,000"
    optimizations:
      mixed_precision: "Saves 40% → 600 hours → $2,400"
      efficient_data_loading: "Saves 15% → 510 hours → $2,040"
      early_stopping: "Saves 20% → 408 hours → $1,632"
      total_with_all: "408 GPU-hours → $1,632 (59% savings vs baseline)"
```

### Organizational Cost Controls

```yaml
Organizational_Cost_Controls:
  budgeting:
    per_team_budgets:
      what: "Monthly GPU budget allocated per team"
      implementation:
        - "Create separate AWS accounts/GCP projects per team"
        - "Set AWS Budgets / GCP Budget Alerts"
        - "Alert at 75%, 90%, action at 100%"
      enforcement:
        soft: "Alert and notify team lead (most orgs)"
        hard: "Auto-disable compute at budget limit (risky for production)"
        
    chargeback_model:
      what: "Track and attribute costs to teams/projects"
      tagging:
        required_tags: ["team", "project", "environment", "owner"]
        enforcement: "SCP/policy denies resource creation without required tags"
      reporting: "Weekly cost report per team/project (automated)"
      
  guardrails:
    instance_restrictions:
      what: "Limit which instance types teams can use"
      implementation: "IAM/SCP policies denying expensive instances"
      example: "Data scientists can use ml.g5.* but not ml.p5.*"
      override: "Manager approval + short-term exception grant"
      
    auto_shutdown:
      dev_instances: "Stop after 30 min idle (SageMaker auto-stop)"
      training_max_runtime: "24-hour max per training job (prevent runaway)"
      endpoint_schedule: "Dev endpoints auto-delete after 7 days"
      
    approval_workflows:
      what: "Large spend requires approval"
      thresholds:
        - "$500+: Team lead approval"
        - "$5,000+: Director approval"
        - "$50,000+: VP approval"
      automation: "Slack bot for quick approvals, auto-deny if no response in 24hr"
      
  finops_practices:
    weekly_review:
      what: "Regular cost review meetings"
      attendees: "ML engineering leads + FinOps"
      agenda:
        - "Cost trends (up/down/flat vs budget)"
        - "Top cost drivers this week"
        - "Optimization opportunities identified"
        - "Action items from last week's review"
        
    optimization_targets:
      gpu_utilization: "Target >70% for training clusters"
      spot_percentage: "Target >50% of training on spot"
      reserved_coverage: "Target >80% of serving on reserved"
      idle_waste: "Target <5% idle GPU hours"
      
    reporting_dashboard:
      metrics:
        - "Total ML spend (daily, weekly, monthly trend)"
        - "Cost per model trained (trending down = good)"
        - "Cost per prediction (serving efficiency)"
        - "GPU utilization across fleet"
        - "Spot vs. on-demand ratio"
        - "Waste: idle hours, oversized instances"
```

---

## How It Works in Practice

### Cost Optimization Case Study

```yaml
Cost_Optimization_Case_Study:
  company: "Series B startup, 10 ML engineers, 15 models in production"
  
  before_optimization:
    monthly_spend: "$95,000"
    breakdown:
      training: "$35,000 (all on-demand, no scheduling)"
      serving: "$45,000 (always-on endpoints, no auto-scaling)"
      development: "$10,000 (dev instances running 24/7)"
      storage: "$5,000 (no lifecycle policies)"
    problems:
      - "GPU utilization: 25% average (idle most of the time)"
      - "Training: 100% on-demand (no spot)"
      - "Serving: no auto-scaling (same capacity day and night)"
      - "Dev: instances never shut down (run weekends/nights)"
      - "Storage: 3 years of checkpoints never deleted"
      
  optimization_actions:
    phase_1_quick_wins:
      duration: "1 week"
      actions:
        - "Enable auto-shutdown on dev instances (30 min idle)"
        - "Delete old checkpoints (lifecycle policy: 7 days)"
        - "Enable auto-scaling on serving endpoints (min: 1)"
      savings: "$25,000/month (26% reduction)"
      
    phase_2_purchasing:
      duration: "2 weeks"
      actions:
        - "Move training to spot instances (with checkpointing)"
        - "Reserve serving baseline (1-year commitment)"
        - "Schedule overnight training (cheaper spot at night)"
      savings: "$20,000/month additional (total 47% reduction)"
      
    phase_3_efficiency:
      duration: "1 month"
      actions:
        - "Enable mixed precision training (2× faster)"
        - "Right-size GPUs (H100 → A10G where appropriate)"
        - "Implement dynamic batching on serving (3× throughput)"
        - "Add request-based auto-scaling with scale-to-zero"
      savings: "$12,000/month additional (total 60% reduction)"
      
  after_optimization:
    monthly_spend: "$38,000"
    breakdown:
      training: "$12,000 (spot + efficient training)"
      serving: "$18,000 (auto-scaled, reserved baseline)"
      development: "$5,000 (auto-shutdown, right-sized)"
      storage: "$3,000 (lifecycle policies active)"
    metrics:
      gpu_utilization: "72% average (up from 25%)"
      spot_percentage: "80% of training"
      auto_scaling_coverage: "100% of serving"
      annual_savings: "$684,000"
```

---

## Interview Tip

> When asked about ML cost optimization: "I optimize across three pillars: reduce price, reduce time, and reduce waste. For price: spot instances for all training (60-90% savings with 15-minute checkpointing), reserved instances for serving baseline (30-60% savings), and a portfolio approach (40% reserved, 45% spot, 15% on-demand). For time: mixed precision training (2× speedup, 50% cost reduction), efficient data loading (>80% GPU utilization), early stopping (save 20-40% of training time), and FP8 on H100 (additional 2× for transformer workloads). For waste: auto-shutdown idle instances (30 min timeout), auto-scaling serving endpoints (scale down at night, scale-to-zero for low-traffic), right-sizing GPUs (profile first — don't use H100 when A10G works), and storage lifecycle policies (delete checkpoints after 7 days). Organizationally: per-team budgets with alerts, required resource tagging for cost attribution, weekly FinOps reviews, and approval workflows for large spend. In practice, these optimizations together typically reduce ML spend by 50-70%. The key metric I track is 'cost per useful GPU-hour' — combining utilization, pricing, and efficiency into one number that should trend down over time."

---

## Common Mistakes

1. **No spot instances for training** — Paying full on-demand price for all training. A team running 24/7 H100 training at $98/hr on-demand when spot is $35/hr wastes $45K/month. Solution: implement spot + checkpointing as default for ALL training. On-demand only when spot unavailable.

2. **Always-on serving with no auto-scaling** — Endpoints running 24/7 at full capacity when traffic is 10% at night. Solution: auto-scaling with minimum 1 instance, or schedule-based scaling (reduce at night). For very low traffic: serverless inference (pay per request).

3. **Wrong GPU for the workload** — Using 8× H100 ($98/hr) for fine-tuning a 7B model with LoRA (fits on single A10G at $1/hr). Solution: right-size by profiling first. If GPU memory utilization <40%, you're overpaying.

4. **No training budget limits** — Hyperparameter search launches 100 training jobs without cap. Bill arrives: $50K for one week of experiments. Solution: set max_runtime on all training jobs, budget alerts, approval for large experiments, and early stopping.

5. **Accumulating storage** — Never deleting checkpoints, intermediate datasets, or old model versions. Storage grows 10% per month forever. Solution: lifecycle policies (auto-delete checkpoints after 7 days, archive old data after 90 days, delete experiments after 30 days).

---

## Key Takeaways

- Three pillars: reduce price (spot/reserved), reduce time (efficiency), reduce waste (auto-scale)
- Spot instances: 60-90% savings — use for ALL training with 15-minute checkpointing
- Reserved instances: 30-60% savings — use for steady-state serving
- Portfolio strategy: 40% reserved + 45% spot + 15% on-demand = ~50% overall savings
- Mixed precision: 2× training speedup = 50% cost reduction (free optimization)
- Auto-scaling: never deploy without it — scale down at night, scale-to-zero for low traffic
- Right-sizing: profile GPU utilization first — most teams over-provision by 2-4×
- Auto-shutdown: dev instances stop after 30 min idle (saves 40-60% on dev compute)
- FinOps: per-team budgets, required tagging, weekly reviews, approval workflows
- Typical savings: 50-70% reduction from unoptimized to fully optimized ML spend
