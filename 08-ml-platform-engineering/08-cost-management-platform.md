# Cost Management Platform

## The Problem / Why This Matters

ML infrastructure is extraordinarily expensive. A single H100 GPU costs $3-4/hour on cloud, and large-scale training jobs can consume 64-256 GPUs for days or weeks. Organizations routinely face $100K-$1M+ monthly ML infrastructure bills with no visibility into who's spending what or why. Without cost management: nobody knows which team is responsible for the $200K spike last month, idle notebooks with GPUs burn money 24/7, training jobs request 8 GPUs but only use 3, and there's no incentive for teams to optimize (it's "someone else's budget"). A cost management platform provides: real-time cost visibility (dashboards showing spend per team, project, and job), chargeback/showback (attribute costs to teams), optimization recommendations (right-sizing, spot instances, scheduling), quota enforcement (prevent runaway spending), and forecasting (predict next month's bill before it arrives). In 2026, with GPU costs dominating ML budgets (often 70-90% of total ML spend), cost management has become a first-class platform capability — not an afterthought. The tools: cloud cost platforms (Kubecost for Kubernetes, CloudHealth, Spot.io), custom attribution systems, and FinOps (Financial Operations) practices integrated into ML workflows.

---

## The Analogy

Think of cost management like a household electricity smart meter system:

- **Without cost management** = Old-style electricity billing. Monthly bill arrives: $500. No idea which appliances caused it. Was it the AC? The space heater? The teenager's gaming PC? You can't optimize what you can't measure.
- **With cost management** = Smart meter with per-device monitoring. Real-time dashboard shows: AC using $8/day, gaming PC $3/day, refrigerator $1/day. Alerts when usage spikes. Recommendations: "Your dryer costs $2/load — consider air drying." Budget limits: cut off gaming PC after $50/month.

You can't optimize ML costs without visibility. And you can't change team behavior without attribution.

---

## Deep Dive

### Cost Breakdown in ML Infrastructure

```yaml
Cost_Components:
  compute_gpu:
    percentage: "60-80% of total ML spend"
    items:
      training: "GPU hours for model training (often the largest single item)"
      inference: "GPU hours for model serving (scales with traffic)"
      notebooks: "GPU hours for development (often high waste — idle GPUs)"
      hpo: "GPU hours for hyperparameter search (can be massive)"
    costs_2026:
      h100_on_demand: "$3.50-4.00/hour (cloud)"
      h200_on_demand: "$5.00-6.00/hour (cloud)"
      a100_on_demand: "$2.50-3.00/hour (cloud)"
      l40s_on_demand: "$1.50-2.00/hour (cloud)"
      h100_spot: "$1.00-1.50/hour (60-70% savings)"
      
  compute_cpu:
    percentage: "10-15% of total ML spend"
    items:
      data_processing: "Spark/Dask jobs for feature engineering"
      pipeline_orchestration: "Airflow/Dagster workers"
      serving_cpu: "CPU inference for lightweight models"
      
  storage:
    percentage: "5-10% of total ML spend"
    items:
      data_lake: "S3/GCS for datasets (TB to PB scale)"
      model_artifacts: "Model checkpoints, final artifacts"
      experiment_logs: "MLflow/W&B artifacts and metadata"
      feature_store: "Online store (Redis — can be expensive at scale)"
      
  networking:
    percentage: "3-5% of total ML spend"
    items:
      data_transfer: "Cross-region, egress charges"
      gpu_interconnect: "InfiniBand/RoCE for distributed training"
      
  managed_services:
    percentage: "5-15% of total ML spend"
    items:
      llm_apis: "OpenAI, Anthropic, Google API costs"
      managed_ml: "SageMaker, Vertex AI management fees"
      monitoring_tools: "Arize, W&B, Datadog subscriptions"
      feature_platform: "Tecton, Databricks fees"
```

### Cost Attribution and Chargeback

```python
# Cost attribution system for ML platform

"""
Attributes every dollar of ML infrastructure spend to a team, project, and job.
Enables chargeback (teams pay for their usage) or showback (visibility without billing).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class CostRecord:
    """A single cost record attributed to a resource."""
    timestamp: datetime
    team: str
    project: str
    user: str
    resource_type: str  # "gpu", "cpu", "storage", "network", "api"
    resource_id: str    # Specific GPU, node, volume
    duration_hours: float
    cost_usd: float
    metadata: dict      # GPU type, instance type, etc.


class CostAttributionSystem:
    """
    Attributes infrastructure costs to teams and projects.
    
    Attribution rules:
    1. Direct: GPU attached to team's training job → attributed to that team
    2. Shared: Platform infrastructure (orchestrator, monitoring) → split across teams by usage
    3. Idle: Unattached/unused resources → attributed to team that reserved them
    """
    
    def __init__(self):
        self.cost_records = []
        self.team_budgets = {}
    
    def compute_team_costs(
        self,
        team: str,
        start_date: datetime,
        end_date: datetime,
    ) -> dict:
        """
        Compute total costs for a team in a time period.
        Breaks down by: resource type, project, and user.
        """
        team_records = [
            r for r in self.cost_records
            if r.team == team and start_date <= r.timestamp <= end_date
        ]
        
        total_cost = sum(r.cost_usd for r in team_records)
        
        breakdown = {
            "total": total_cost,
            "by_resource_type": self._group_by(team_records, "resource_type"),
            "by_project": self._group_by(team_records, "project"),
            "by_user": self._group_by(team_records, "user"),
            "top_jobs": self._top_expensive_jobs(team_records, n=10),
            "waste_estimate": self._estimate_waste(team_records),
        }
        
        return breakdown
    
    def _estimate_waste(self, records: list) -> dict:
        """
        Estimate wasted spend (resources paid for but not effectively used).
        
        Waste categories:
        - Idle GPUs (allocated but <10% utilization)
        - Over-provisioned (requested 8 GPUs, only used 4)
        - Zombie resources (jobs completed but resources not released)
        """
        idle_cost = sum(
            r.cost_usd for r in records
            if r.metadata.get("gpu_utilization", 100) < 10
        )
        
        over_provisioned_cost = sum(
            r.cost_usd * (1 - r.metadata.get("actual_usage_ratio", 1.0))
            for r in records
            if r.metadata.get("actual_usage_ratio", 1.0) < 0.5
        )
        
        return {
            "idle_resources": idle_cost,
            "over_provisioned": over_provisioned_cost,
            "total_waste": idle_cost + over_provisioned_cost,
            "waste_percentage": (idle_cost + over_provisioned_cost) / sum(r.cost_usd for r in records) * 100 if records else 0,
        }
    
    def check_budget(self, team: str) -> dict:
        """
        Check team's current spend against budget.
        Returns: current spend, budget, percentage used, projected month-end.
        """
        budget = self.team_budgets.get(team, {})
        monthly_budget = budget.get("monthly_limit", float('inf'))
        
        # Current month spend
        month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0)
        current_spend = sum(
            r.cost_usd for r in self.cost_records
            if r.team == team and r.timestamp >= month_start
        )
        
        # Project month-end (linear extrapolation)
        days_elapsed = (datetime.now() - month_start).days or 1
        days_in_month = 30
        projected_spend = current_spend * (days_in_month / days_elapsed)
        
        return {
            "monthly_budget": monthly_budget,
            "current_spend": current_spend,
            "percentage_used": (current_spend / monthly_budget) * 100,
            "projected_month_end": projected_spend,
            "over_budget": projected_spend > monthly_budget,
            "days_remaining": days_in_month - days_elapsed,
        }


class BudgetEnforcement:
    """
    Enforces cost budgets — prevents teams from overspending.
    
    Levels:
    - Soft limit: Warning + notification to team lead
    - Hard limit: Block new resource allocation (existing jobs continue)
    - Emergency: Kill non-critical jobs to prevent budget breach
    """
    
    def __init__(self, attribution_system: CostAttributionSystem):
        self.attribution = attribution_system
    
    def evaluate_job_request(
        self,
        team: str,
        estimated_cost: float,
        priority: str = "normal",
    ) -> dict:
        """
        Decide whether to allow a new job based on budget.
        """
        budget_status = self.attribution.check_budget(team)
        
        # High-priority jobs bypass budget checks
        if priority == "critical":
            return {"approved": True, "reason": "Critical priority bypasses budget"}
        
        # Check if projected spend exceeds budget
        if budget_status["over_budget"]:
            if budget_status["percentage_used"] > 100:
                return {
                    "approved": False,
                    "reason": f"Budget exceeded ({budget_status['percentage_used']:.0f}%)",
                    "suggestion": "Use spot instances, reduce GPU count, or request budget increase",
                }
            else:
                return {
                    "approved": True,
                    "warning": f"Projected to exceed budget by month end ({budget_status['projected_month_end']:.0f} vs {budget_status['monthly_budget']:.0f})",
                }
        
        return {"approved": True}
```

### Cost Optimization Strategies

```yaml
Optimization_Strategies:
  spot_instances:
    what: "Use preemptible/spot GPUs for fault-tolerant workloads"
    savings: "60-80% vs on-demand"
    applicable_to:
      - "Training with checkpointing (resume after interruption)"
      - "Hyperparameter search (trials are independent)"
      - "Batch inference (can retry failed batches)"
      - "Development notebooks (acceptable to restart)"
    not_applicable:
      - "Production serving (can't have downtime)"
      - "Large distributed training without checkpointing"
    implementation:
      - "Enable checkpointing every 30 min (or per epoch)"
      - "Use spot-tolerant job controllers (handles preemption gracefully)"
      - "Mix spot + on-demand for reliability (80% spot, 20% on-demand)"
      
  right_sizing:
    what: "Match resource allocation to actual usage"
    common_waste:
      - "8 GPUs requested, only 4 utilized (50% waste)"
      - "H100 requested for inference that runs fine on L40S (70% cost difference)"
      - "128 GB RAM requested, only 16 GB used"
    implementation:
      - "Monitor actual utilization (DCGM for GPU, cAdvisor for CPU/memory)"
      - "Weekly recommendations: 'Your job used 40% GPU — consider 4 GPUs instead of 8'"
      - "Auto-suggestions when submitting jobs based on historical similar workloads"
      
  scheduling_optimization:
    what: "Run flexible jobs during off-peak hours"
    mechanism: "Queue priority changes by time-of-day"
    example: "Large training jobs get priority at night (when cluster is 40% idle)"
    savings: "Not monetary directly, but reduces queue wait and improves utilization"
    
  reserved_instances:
    what: "Commit to long-term usage for discount"
    savings: "30-60% vs on-demand (1-3 year commitments)"
    applicable: "Steady-state workloads (always-on inference, base training capacity)"
    risk: "Overpaying if usage decreases"
    strategy: "Reserve for baseline (50th percentile usage), on-demand for peaks"
    
  model_optimization:
    what: "Make models cheaper to run"
    techniques:
      - "Quantization: INT8/INT4 inference (50-75% cheaper hardware)"
      - "Distillation: Smaller model that mimics larger one"
      - "Pruning: Remove unnecessary parameters"
      - "Model routing: Simple queries → cheap model, complex → expensive"
    savings: "50-90% inference cost reduction"
    
  llm_cost_optimization:
    what: "Reduce LLM API and inference costs"
    techniques:
      prompt_caching: "Cache repeated system prompts (Claude: 90% discount on cached)"
      semantic_caching: "Cache similar queries → return cached response"
      model_routing: "GPT-5 for hard questions, GPT-4o-mini for easy ones"
      response_length: "Limit max_tokens, use concise prompts"
      batch_api: "Use batch API for non-urgent (50% discount on OpenAI batch)"
      self_hosting: "Run Llama-4-70B on own GPUs vs API (cheaper above ~1M tokens/day)"
```

### Cost Dashboards

```yaml
Dashboards:
  executive_view:
    metrics:
      - "Total monthly spend (vs. budget)"
      - "Month-over-month trend"
      - "Spend by team (top 5)"
      - "Projected month-end spend"
      - "Waste percentage (idle + over-provisioned)"
    frequency: "Daily refresh"
    audience: "VP Engineering, Finance"
    
  team_view:
    metrics:
      - "Team monthly spend (vs. team budget)"
      - "Spend by project/model"
      - "Spend by user"
      - "Top 10 most expensive jobs"
      - "GPU utilization (actual vs. allocated)"
      - "Waste breakdown (idle, over-provisioned)"
      - "Optimization recommendations"
    frequency: "Real-time (15 min refresh)"
    audience: "Team lead, ML engineers"
    
  job_view:
    metrics:
      - "Cost of current/recent job"
      - "GPU utilization during job"
      - "Estimated cost at completion (for running jobs)"
      - "Cost comparison to previous runs (is this run more expensive?)"
    frequency: "Real-time"
    audience: "Individual ML engineer/data scientist"
    
  alerts:
    budget:
      - "Team at 80% of monthly budget → notify team lead"
      - "Team at 100% of monthly budget → notify team lead + block non-critical"
      - "Single job exceeds $1,000 → notify user"
    anomaly:
      - "Daily spend 3× higher than average → investigate"
      - "New resource type appearing (someone launched expensive instance accidentally)"
    waste:
      - "GPU idle >1 hour → notify user"
      - "Notebook with GPU unused for 4+ hours → auto-terminate warning"
```

---

## How It Works in Practice

### Monthly FinOps Cycle

```yaml
Monthly_Cycle:
  week_1:
    - "Previous month cost report generated automatically"
    - "Chargeback applied to team budgets"
    - "Anomaly review: investigate any unusual spikes"
    
  week_2:
    - "Optimization recommendations generated"
    - "Teams review their waste reports"
    - "Action items: right-size, switch to spot, terminate idle"
    
  week_3:
    - "Track implementation of optimizations"
    - "Compare current month trajectory to budget"
    - "Adjust budgets if needed (reallocation between teams)"
    
  week_4:
    - "Forecast next month (based on current trends + planned work)"
    - "Budget approval for next month"
    - "Reserved instance review (need more? can release some?)"
```

---

## Interview Tip

> When asked about ML cost management: "My cost management approach has three pillars: (1) Visibility — every dollar attributed to a team, project, and job. Real-time dashboards show current spend vs. budget, GPU utilization, and waste estimates. Teams can see exactly why their bill is what it is, down to individual training runs. (2) Optimization — three biggest levers: spot instances for training (60-80% savings with checkpointing), right-sizing (weekly recommendations based on actual GPU/memory utilization vs. requested), and LLM cost optimization (semantic caching, model routing, batch APIs). For inference, model quantization (INT8) halves GPU cost with minimal accuracy loss. Combined, these typically save 40-60% of total ML spend. (3) Governance — team budgets with soft and hard limits. At 80% budget: warning. At 100%: block new non-critical jobs (existing jobs continue). Monthly FinOps review: analyze spend, identify waste, implement optimizations, forecast next month. The metric I track: 'effective cost per model-quality-unit' — cost divided by model performance. The goal isn't minimum spend, it's maximum value per dollar. Sometimes spending more (larger model, more training data) is justified by proportional quality improvement."

---

## Common Mistakes

1. **No cost visibility** — Monthly cloud bill arrives: $300K. Nobody knows which team or job caused the spike. Investigation takes days. Solution: real-time cost attribution from day one. Tag every resource with team, project, user. Dashboard refreshes every 15 minutes.

2. **Budget without enforcement** — Teams have "budgets" on paper but no mechanism prevents overspending. When someone launches a 64-GPU job by mistake, it runs for 3 days at $7K/day. Solution: soft limits (warning at 80%) + hard limits (block at 100%). Emergency stop for runaway jobs.

3. **Optimizing by limiting usage** — Management response to high costs: "Reduce GPU allocation by 50%." Result: ML team can't train models, velocity drops to zero, no ML value delivered. Saving money by not doing ML defeats the purpose. Solution: optimize efficiency (spot instances, right-sizing, quantization), not capability. Spend the same but get more value.

4. **Ignoring inference costs** — Focus all optimization on training (which happens infrequently) while inference costs grow 10× (serving 24/7 to increasing traffic). Solution: inference cost management is equally important — autoscaling (scale down at night), model optimization (quantization, distillation), and caching.

5. **No forecasting** — Surprise at month end: "How did we spend $500K?" Linear growth wasn't tracked, plus a few large training runs nobody budgeted for. Solution: daily spend projection ("at this rate, you'll spend $X by month end"). Alert when projected > budget. Require cost estimates for large jobs before approval.

---

## Key Takeaways

- GPU compute is 60-80% of ML spend — this is where optimization matters most
- Cost attribution: every dollar → team, project, user, job (real-time, not monthly reconciliation)
- Spot instances: 60-80% savings for training with checkpointing (biggest single optimization)
- Right-sizing: match resource allocation to actual usage (typical waste: 30-50%)
- Budget enforcement: soft limits (warn) + hard limits (block) + emergency (kill)
- LLM cost optimization: semantic caching, model routing, batch APIs, prompt caching
- Dashboards: executive (total/budget), team (attribution/waste), job (per-run cost)
- FinOps cycle: monthly review, optimization recommendations, budget adjustment
- Goal: maximize value per dollar (not minimize spend — spending nothing means no ML)
- Forecasting: project month-end spend daily, alert when trending over budget
