# Compute Management

## The Problem / Why This Matters

GPUs are the most expensive and scarce resource in ML infrastructure. A single NVIDIA H100 costs ~$30,000, and organizations need clusters of 8-256+ GPUs for training large models. Without proper compute management, you get: GPU utilization at 20-30% (teams hoard reserved GPUs "just in case"), 2-week queues for training jobs (no prioritization), runaway costs ($500K/month bills nobody can explain), and no visibility into who's using what. Compute management encompasses: GPU scheduling (which jobs run when on which GPUs), quota management (how much compute each team gets), multi-tenancy (multiple teams sharing the same cluster safely), spot/preemptible instance optimization (using cheap compute for fault-tolerant workloads), and autoscaling (spinning resources up/down based on demand). In 2026, this has become even more critical with the GPU shortage, the rise of distributed training across H100/H200/B200 GPU clusters, and LLM fine-tuning requiring 8-64 GPUs per job. The difference between 30% and 80% GPU utilization on a 100-GPU cluster is millions of dollars per year.

---

## The Analogy

Think of compute management like managing airline gate assignments at a busy airport:

- **GPUs** = Airport gates. Expensive real estate, limited supply.
- **Training jobs** = Flights. Each needs a gate (GPU) for a specific duration.
- **Scheduling** = Gate assignment system. Match flights to gates based on size, duration, priority.
- **Quotas** = Airline gate allocations. Delta gets 20 gates, United gets 15, Southwest gets 10. Each can use their allocation, but idle gates can be temporarily reassigned.
- **Preemption** = Emergency bumping. VIP flight needs a gate → bump the cargo flight to a different slot.
- **Spot instances** = Standby passengers. Cheap but you might get bumped.

A bad airport lets gates sit empty while flights circle. A good airport maintains 85%+ gate utilization while keeping priority flights on-time.

---

## Deep Dive

### GPU Cluster Architecture

```yaml
Cluster_Architecture:
  hardware_2026:
    training_nodes:
      gpu: "NVIDIA H100 (80GB HBM3) or H200 (141GB HBM3e)"
      per_node: "8 GPUs connected via NVLink/NVSwitch (900 GB/s)"
      interconnect: "InfiniBand NDR (400 Gbps) or RoCE v2 between nodes"
      cpu: "AMD EPYC or Intel Xeon (128+ cores per node)"
      ram: "1-2 TB system RAM per node"
      storage: "NVMe SSD (local scratch) + network storage (parallel filesystem)"
      
    inference_nodes:
      gpu: "H100 or L40S (for cost-effective inference)"
      per_node: "4-8 GPUs"
      focus: "Low latency, high throughput, cost per token"
      
    emerging:
      b200: "NVIDIA Blackwell B200 (192GB HBM3e, 2× H100 performance)"
      grace_hopper: "CPU+GPU unified memory architecture"
      
  cluster_sizes:
    small: "8-32 GPUs (startup, early ML team)"
    medium: "32-256 GPUs (mid-size company, multiple teams)"
    large: "256-4096 GPUs (large enterprise, LLM training)"
    hyperscale: "4096-100K+ GPUs (Meta, Google, OpenAI — frontier model training)"
    
  software_stack:
    orchestration: "Kubernetes + NVIDIA GPU Operator"
    scheduling: "Volcano, Kueue, or custom scheduler"
    monitoring: "DCGM (Data Center GPU Manager) + Prometheus"
    networking: "NCCL (NVIDIA Collective Communications Library) for multi-GPU"
    storage: "Lustre, GPFS, or cloud parallel filesystem"
```

### Job Scheduling

```yaml
Scheduling:
  schedulers:
    kubernetes_default:
      how: "First-come-first-served with resource requests"
      limitation: "No gang scheduling, no GPU topology awareness, no preemption"
      use_for: "Simple single-GPU jobs"
      
    volcano:
      what: "Kubernetes batch scheduling framework"
      features:
        - "Gang scheduling (all-or-nothing for distributed training)"
        - "Queue management with priorities"
        - "Fair-share scheduling"
        - "Preemption support"
      use_for: "Distributed training jobs requiring multi-node allocation"
      
    kueue:
      what: "Kubernetes-native job queueing (newer, by Google)"
      features:
        - "Resource quotas per namespace/team"
        - "Priority classes"
        - "Preemption"
        - "Workload admission control"
      use_for: "Quota-based resource management in Kubernetes"
      
    ray_cluster:
      what: "Ray's built-in scheduling for Ray Train/Serve jobs"
      features:
        - "Autoscaling Ray clusters"
        - "Fractional GPU support"
        - "Placement groups (topology-aware)"
      use_for: "Ray-based training and serving workloads"
      
    slurm:
      what: "Traditional HPC job scheduler"
      features:
        - "Mature, battle-tested"
        - "GPU and topology aware"
        - "Fair-share, backfill scheduling"
      use_for: "Research clusters, HPC-style ML workloads"
      limitation: "Less cloud-native than Kubernetes approaches"
      
  scheduling_policies:
    fifo: "First-in-first-out. Simple but unfair (one large job blocks everything)"
    fair_share: "Proportional allocation based on team quota. Most common."
    priority: "Higher-priority jobs run first. Risk: starvation of low-priority."
    backfill: "Fill gaps with small jobs that fit in idle time. Improves utilization."
    preemptive: "High-priority jobs can evict low-priority. Requires checkpointing."
```

### Quota Management

```python
# GPU quota management system

"""
Manages GPU quotas for multi-tenant ML clusters.
Ensures fair resource allocation while maximizing utilization.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TeamQuota:
    """GPU quota for a team."""
    team_name: str
    guaranteed_gpus: int      # Minimum always available
    max_gpus: int             # Maximum (burst) capacity
    priority: int             # 1=highest, 5=lowest
    preemptible: bool         # Can this team's jobs be preempted?
    spot_eligible: bool       # Can use spot/preemptible instances?


class QuotaManager:
    """
    Manages GPU quota allocation across teams.
    
    Design:
    - Each team has a guaranteed minimum (always available)
    - Burst capacity available when cluster has spare resources
    - Higher-priority teams can preempt lower-priority burst usage
    - Spot instances supplement quota for fault-tolerant workloads
    """
    
    def __init__(self, total_cluster_gpus: int):
        self.total_gpus = total_cluster_gpus
        self.team_quotas: dict[str, TeamQuota] = {}
        self.active_allocations: dict[str, int] = {}  # team → current GPU count
    
    def set_quota(self, quota: TeamQuota):
        """Define quota for a team."""
        self.team_quotas[quota.team_name] = quota
        
        # Validate: sum of guaranteed doesn't exceed cluster size
        total_guaranteed = sum(q.guaranteed_gpus for q in self.team_quotas.values())
        if total_guaranteed > self.total_gpus:
            raise ValueError(
                f"Total guaranteed ({total_guaranteed}) exceeds cluster capacity ({self.total_gpus})"
            )
    
    def can_allocate(self, team: str, gpus_requested: int) -> dict:
        """
        Check if a team can allocate GPUs.
        Returns allocation decision with source breakdown.
        """
        quota = self.team_quotas[team]
        current_usage = self.active_allocations.get(team, 0)
        new_total = current_usage + gpus_requested
        
        # Case 1: Within guaranteed quota → always approve
        if new_total <= quota.guaranteed_gpus:
            return {
                "approved": True,
                "source": "guaranteed",
                "gpus_granted": gpus_requested,
                "preemptible": False,
            }
        
        # Case 2: Within max (burst) quota → check availability
        if new_total <= quota.max_gpus:
            available = self._get_available_gpus()
            if available >= gpus_requested:
                return {
                    "approved": True,
                    "source": "burst",
                    "gpus_granted": gpus_requested,
                    "preemptible": True,  # Burst allocation can be preempted
                }
            
            # Can we preempt lower-priority burst allocations?
            preemptible_gpus = self._get_preemptible_gpus(min_priority=quota.priority)
            if available + preemptible_gpus >= gpus_requested:
                return {
                    "approved": True,
                    "source": "preemption",
                    "gpus_granted": gpus_requested,
                    "preempt_from": self._select_preemption_targets(
                        gpus_needed=gpus_requested - available,
                        min_priority=quota.priority,
                    ),
                }
        
        # Case 3: Exceeds max quota → deny
        return {
            "approved": False,
            "reason": f"Exceeds max quota ({quota.max_gpus} GPUs)",
            "current_usage": current_usage,
            "queue_position": self._get_queue_position(team, gpus_requested),
        }
    
    def _get_available_gpus(self) -> int:
        """Get currently unallocated GPUs."""
        total_allocated = sum(self.active_allocations.values())
        return self.total_gpus - total_allocated
    
    def _get_preemptible_gpus(self, min_priority: int) -> int:
        """Get GPUs that can be preempted by a job with given priority."""
        preemptible = 0
        for team, usage in self.active_allocations.items():
            quota = self.team_quotas[team]
            if quota.priority > min_priority:  # Lower priority number = higher priority
                # Can preempt burst allocation (above guaranteed)
                burst_usage = max(0, usage - quota.guaranteed_gpus)
                preemptible += burst_usage
        return preemptible


# Cost optimization with spot instances
class SpotInstanceManager:
    """
    Manages spot/preemptible GPU instances for cost optimization.
    
    Spot instances: 60-80% cheaper but can be reclaimed with 30-120s notice.
    Use for: fault-tolerant training (with checkpointing), hyperparameter search,
    batch inference, experiments.
    """
    
    def __init__(self):
        self.spot_savings_rate = 0.70  # 70% cheaper than on-demand
        
    def should_use_spot(self, job_config: dict) -> dict:
        """Decide whether a job should use spot instances."""
        
        checkpointing_enabled = job_config.get("checkpointing", False)
        max_runtime_hours = job_config.get("max_runtime_hours", 24)
        is_distributed = job_config.get("num_gpus", 1) > 1
        is_hyperparameter_search = job_config.get("is_hpo", False)
        
        # Decision logic
        if not checkpointing_enabled:
            return {
                "use_spot": False,
                "reason": "No checkpointing — spot interruption would lose all progress",
                "recommendation": "Enable checkpointing first, then use spot",
            }
        
        if is_distributed and job_config.get("num_gpus", 1) > 32:
            return {
                "use_spot": False,
                "reason": "Large distributed jobs (>32 GPUs) have high interruption probability",
                "recommendation": "Use on-demand for large distributed training",
            }
        
        if is_hyperparameter_search:
            return {
                "use_spot": True,
                "reason": "HPO trials are independent and expendable",
                "savings_estimate": f"{self.spot_savings_rate:.0%} cost reduction",
            }
        
        # Default: use spot for checkpointed single/small distributed jobs
        if max_runtime_hours <= 8:
            return {
                "use_spot": True,
                "reason": "Short job with checkpointing — low interruption risk",
                "savings_estimate": f"{self.spot_savings_rate:.0%} cost reduction",
            }
        
        return {
            "use_spot": True,
            "reason": "Checkpointing enabled — can resume after interruption",
            "savings_estimate": f"{self.spot_savings_rate:.0%} cost reduction",
            "warning": "May be interrupted — ensure checkpoint frequency is adequate",
        }
```

### GPU Utilization Optimization

```yaml
Utilization_Optimization:
  problem: "Average GPU utilization in most organizations: 20-40%"
  target: "70-85% (above 85% risks queuing delays)"
  
  strategies:
    gpu_sharing:
      what: "Multiple jobs share a single GPU"
      methods:
        mps: "NVIDIA MPS (Multi-Process Service) — share GPU compute"
        time_slicing: "Kubernetes GPU time-slicing — each pod gets time slices"
        mig: "MIG (Multi-Instance GPU) on A100/H100 — hardware-level partitioning"
      use_for: "Small inference jobs, development notebooks, small training experiments"
      
    right_sizing:
      what: "Match GPU allocation to actual need"
      signals:
        - "GPU memory utilization (allocated vs. used)"
        - "GPU compute utilization (SM occupancy)"
        - "Training throughput (samples/second not improving with more GPUs)"
      actions:
        - "Recommend smaller GPU (L40S vs H100 for small models)"
        - "Reduce GPU count (4 GPUs if only 2 are utilized)"
        - "Use fractional GPU for inference"
      tool: "DCGM metrics → custom recommendation engine"
      
    bin_packing:
      what: "Efficiently pack jobs onto nodes"
      example: "Two 4-GPU jobs on one 8-GPU node (instead of two nodes at 50%)"
      scheduler_feature: "Topology-aware scheduling, NUMA awareness"
      
    queue_management:
      what: "Keep queue length healthy"
      target: "Queue wait < 15 minutes for standard priority"
      mechanisms:
        - "Backfill: run small jobs in gaps between large reservations"
        - "Preemption: priority jobs evict lower-priority burst jobs"
        - "Autoscaling: add nodes when queue grows (cloud clusters)"
        
    idle_detection:
      what: "Detect and reclaim idle GPUs"
      scenarios:
        - "Notebook with GPU attached but no computation (15+ min idle)"
        - "Training job hung (no progress for 30+ min)"
        - "Development job completed but pod not terminated"
      action: "Warning after 15 min idle → auto-terminate after 60 min"
```

---

## How It Works in Practice

### Resource Request Workflow

```yaml
Workflow:
  user_submits_job:
    request: "I need 8 H100 GPUs for 12 hours (distributed training)"
    
  system_checks:
    1. "Does team have quota for 8 GPUs? → Yes (guaranteed: 16, current usage: 4)"
    2. "Are 8 H100s available? → Check cluster state"
    3. "Topology requirement? → Need NVLink-connected (same node)"
    4. "Spot eligible? → Yes if checkpointing enabled"
    
  allocation_decision:
    if_available: "Schedule immediately on best-fit node"
    if_queued: "Estimate wait time, notify user, offer alternatives"
    alternatives: "Smaller allocation? Different GPU type? Spot instances?"
    
  during_execution:
    - "Monitor GPU utilization (alert if <30% for >15 min)"
    - "Track progress (alert if loss plateaus or training hangs)"
    - "Checkpoint regularly (for spot/preemption resilience)"
    
  after_completion:
    - "Release GPUs immediately"
    - "Report utilization metrics to team dashboard"
    - "Update cost attribution (chargeback to team)"
```

---

## Interview Tip

> When asked about compute management: "My GPU cluster management has four pillars: (1) Scheduling — I use Volcano or Kueue on Kubernetes for gang scheduling (distributed training needs all GPUs simultaneously or none). Priority queues ensure production retraining jobs run before experimental exploration. (2) Quota management — each team gets a guaranteed GPU allocation (always available) plus burst capacity (available when cluster has spare resources). Burst allocations are preemptible by higher-priority teams. This prevents hoarding while ensuring critical workloads always have resources. (3) Utilization optimization — target 70-80% cluster utilization through: idle detection (auto-terminate notebooks idle >60 min), right-sizing recommendations (DCGM metrics show actual vs. allocated GPU usage), GPU sharing for small inference jobs (MIG on H100 splits one GPU into up to 7 instances), and backfill scheduling. (4) Cost optimization — spot instances for fault-tolerant workloads (60-70% savings), autoscaling for variable demand, and chargeback to make teams cost-aware. The key metric: median queue wait time. If it's >15 minutes for standard priority, the cluster needs more capacity or better scheduling."

---

## Common Mistakes

1. **No gang scheduling for distributed training** — Default Kubernetes scheduler allocates GPUs one-by-one. A 4-GPU job gets 3 GPUs, waits for the 4th (which another job took). Both jobs partially allocated, neither can run. Solution: use Volcano or Kueue with gang scheduling — allocate all resources atomically or none.

2. **Guaranteed quotas sum to 100% of cluster** — Every team has a "guaranteed" allocation that totals the full cluster. No burst capacity, no flexibility. When one team is idle, their GPUs sit unused. Solution: guaranteed quotas should sum to 60-70% of cluster, leaving headroom for burst.

3. **No idle detection** — Notebooks with GPUs attached 24/7, developer leaves for the weekend. $2,000/day in GPU costs doing nothing. Solution: auto-hibernate after 15 min idle (warning), auto-terminate after 60 min. Users can easily restart.

4. **Spot without checkpointing** — Team uses spot instances for cost savings but training script doesn't checkpoint. Spot instance reclaimed after 8 hours of training → all progress lost. Solution: mandatory checkpointing for spot workloads (platform validates before scheduling on spot).

5. **No topology awareness** — 8-GPU distributed training scheduled across 4 different nodes (2 GPUs each). Inter-node communication over Ethernet (25 Gbps) instead of NVLink (900 Gbps). Training is 3× slower. Solution: topology-aware scheduling — prefer same node, then same rack, minimize network hops.

---

## Key Takeaways

- GPU clusters are the most expensive ML infrastructure — utilization optimization saves millions
- Scheduling: Volcano/Kueue for gang scheduling, priority queues, fair-share allocation
- Quota: guaranteed minimum (always available) + burst capacity (preemptible by higher priority)
- Spot instances: 60-70% cheaper, use for checkpointed training and hyperparameter search
- Utilization target: 70-80% (below 70% = waste, above 85% = queue delays)
- Idle detection: auto-terminate notebooks/jobs idle >60 min
- Topology awareness: schedule distributed jobs on same node for NVLink (900 Gbps vs 25 Gbps Ethernet)
- Right-sizing: DCGM metrics reveal actual GPU usage — recommend smaller allocation when appropriate
- MIG (Multi-Instance GPU): partition H100 for small inference jobs (up to 7 instances per GPU)
- Cost attribution: chargeback makes teams cost-aware, drives right behavior
