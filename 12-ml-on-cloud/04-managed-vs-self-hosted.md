# Managed vs Self-Hosted ML Infrastructure

## The Problem / Why This Matters

Every ML team faces a fundamental infrastructure decision: use managed ML platforms (SageMaker, Vertex AI, Azure ML) or build self-hosted infrastructure on Kubernetes. This isn't a one-time choice — it's a spectrum, and most mature teams use a hybrid. Managed platforms offer faster time-to-production, less operational burden, and built-in MLOps features, but at the cost of vendor lock-in, limited customization, and potentially higher long-term costs. Self-hosted (typically Kubernetes + open-source tools like Kubeflow, MLflow, Ray, Seldon) offers maximum flexibility, portability, and cost control at scale, but requires significant infrastructure expertise and ongoing maintenance. In 2026, the landscape has matured: managed platforms have become more flexible (custom containers, bring-your-own infrastructure options), and self-hosted tooling has become more polished (Kubeflow Pipelines v2, Ray Train/Serve, KServe). The right answer depends on team size, ML maturity, compliance requirements, cost sensitivity, and multi-cloud strategy. Understanding these trade-offs deeply — not just listing pros and cons — is essential for ML engineering leadership.

---

## The Analogy

Think of it like housing:

- **Managed platform (SageMaker/Vertex AI)** = Renting a luxury apartment. Move in immediately. Maintenance included. Building manager handles plumbing, electricity, security. But you can't knock down walls, you pay the landlord's premium, and moving to a different building means packing everything up.
- **Self-hosted (Kubernetes)** = Owning and maintaining a house. Full control: remodel any room, build extensions, choose your own contractors. But YOU fix the plumbing at 3 AM, YOU pay for all maintenance, and you need construction expertise to modify anything.
- **Hybrid** = Renting an apartment but owning a workshop in the garage. Daily living is convenient (managed serving, MLOps), but for specialized work (custom training, research), you have full control.

---

## Deep Dive

### Decision Framework

```yaml
Decision_Framework:
  dimensions:
    team_size:
      small_team: # 1-5 ML engineers
        recommendation: "Managed platform (SageMaker or Vertex AI)"
        reasoning: "Can't afford 1-2 FTEs on infrastructure. ML time > infra time."
        exception: "If team has strong K8s background AND cost-sensitive"
        
      medium_team: # 5-20 ML engineers
        recommendation: "Hybrid — managed serving + self-hosted training"
        reasoning: "Enough scale to justify some infrastructure investment"
        typical: "Managed endpoints + custom training on Ray/K8s clusters"
        
      large_team: # 20+ ML engineers + dedicated platform team
        recommendation: "Self-hosted core + managed for specific use cases"
        reasoning: "Dedicated platform team amortizes infrastructure investment"
        typical: "Kubeflow + KServe + MLflow with managed GPU clusters"
        
    ml_maturity:
      early: # Proving value, first models
        recommendation: "Managed (minimize time-to-production)"
        reasoning: "Focus on model quality, not infrastructure. Ship fast."
        
      growing: # 5-20 models in production
        recommendation: "Start building platform capabilities"
        reasoning: "Patterns emerge, managed platform limits become visible"
        
      mature: # 50+ models, dedicated ML platform team
        recommendation: "Self-hosted core with managed extensions"
        reasoning: "Custom requirements, cost optimization at scale"
        
    compliance:
      highly_regulated: # Healthcare, finance, government
        recommendation: "Managed (built-in compliance controls)"
        reasoning: "Managed platforms have SOC2, HIPAA, FedRAMP certifications"
        alternative: "Self-hosted in private cloud if data sovereignty requires"
        
      standard:
        recommendation: "Either — compliance isn't the differentiator"
        
    multi_cloud:
      single_cloud:
        recommendation: "Managed platform (maximize cloud-native integration)"
        
      multi_cloud_required:
        recommendation: "Self-hosted (portable stack)"
        tools: "Kubeflow, MLflow, Ray, KServe — run on any K8s"
        
    cost_sensitivity:
      low: # Startups funded, enterprise with large budgets
        recommendation: "Managed (spend money to save time)"
        
      high: # Tight margins, massive scale
        recommendation: "Self-hosted (10-30% cheaper at scale with optimization)"
        threshold: "Self-hosted becomes cheaper at ~$50K+/month ML infrastructure spend"
```

### Cost Comparison

```yaml
Cost_Comparison:
  scenario: "Team of 10 ML engineers, 20 models in production"
  
  managed_sagemaker:
    training:
      monthly_training_hours: 2000  # GPU hours
      instance: "ml.g5.2xlarge ($1.52/hr including SageMaker markup)"
      cost: "$3,040/month"
      
    inference:
      endpoints: 20  # Always-on endpoints
      instance: "ml.g5.xlarge ($1.01/hr per endpoint)"
      avg_instances: 2  # Average with auto-scaling
      cost: "$29,088/month (20 × 2 × $1.01 × 720 hours)"
      
    platform:
      studio: "$500/month (10 developers)"
      feature_store: "$800/month"
      model_monitor: "$600/month"
      pipelines: "$200/month"
      cost: "$2,100/month"
      
    total: "$34,228/month"
    markup_over_raw_compute: "~40% (SageMaker adds ~40% over raw EC2)"
    
  self_hosted_kubernetes:
    training:
      cluster: "3× g5.2xlarge reserved instances ($0.92/hr each)"
      utilization: "70% average with job scheduler"
      cost: "$1,987/month"
      
    inference:
      cluster: "10× g5.xlarge reserved instances ($0.61/hr each)"
      utilization: "60% with bin-packing (KServe)"
      cost: "$4,392/month"
      
    platform:
      mlflow: "$0 (self-hosted on K8s)"
      kubeflow: "$0 (open-source)"
      prometheus_grafana: "$200/month (managed Prometheus)"
      k8s_management: "$500/month (EKS control plane + tools)"
      cost: "$700/month"
      
    engineering:
      platform_engineer: "0.5 FTE × $15K/month = $7,500/month"
      note: "Half an engineer dedicated to ML platform maintenance"
      
    total: "$14,579/month"
    savings_vs_managed: "57% cheaper"
    
  hidden_costs_self_hosted:
    - "Initial setup: 2-3 months engineering time ($45K-90K one-time)"
    - "Incident response: on-call for GPU cluster issues"
    - "Upgrades: K8s version bumps, driver updates, tool migrations"
    - "Knowledge risk: if platform engineer leaves, bus factor = 1"
    - "Opportunity cost: time on infra ≠ time on models"
    
  break_even_analysis:
    below_15k_monthly: "Managed wins (setup cost not amortized)"
    15k_to_50k_monthly: "Depends on team capabilities"
    above_50k_monthly: "Self-hosted likely wins (savings justify platform team)"
    
  hybrid_sweet_spot:
    training: "Self-hosted (biggest cost, most custom requirements)"
    serving: "Managed (auto-scaling, zero-downtime deploys complex to build)"
    mlops: "Mix (MLflow self-hosted + managed pipelines)"
    cost: "~$22K/month (40% savings vs. full managed)"
```

### Architecture Patterns

```yaml
Architecture_Patterns:
  pattern_1_full_managed:
    description: "All-in on cloud ML platform"
    stack:
      training: "SageMaker Training / Vertex AI Custom Training"
      serving: "SageMaker Endpoints / Vertex AI Endpoints"
      pipelines: "SageMaker Pipelines / Vertex AI Pipelines"
      features: "SageMaker Feature Store / Vertex Feature Store"
      monitoring: "SageMaker Model Monitor / Vertex Model Monitoring"
      experiments: "SageMaker Experiments / Vertex Experiments"
    pros:
      - "Fastest time to production (weeks, not months)"
      - "Minimal operational burden (vendor handles infrastructure)"
      - "Built-in security and compliance"
      - "Automatic scaling (don't design for peak)"
    cons:
      - "40-60% cost premium over raw compute"
      - "Limited customization (work within platform constraints)"
      - "Vendor lock-in (migration is expensive)"
      - "Opaque failures (limited visibility into platform internals)"
    best_for: "Small teams, rapid prototyping, regulated industries"
    
  pattern_2_full_self_hosted:
    description: "Kubernetes-native ML platform"
    stack:
      training: "Ray Train / Kubeflow Training Operator"
      serving: "KServe / Seldon Core / Ray Serve"
      pipelines: "Kubeflow Pipelines v2 / Argo Workflows"
      features: "Feast / Tecton"
      monitoring: "Prometheus + Grafana + custom metrics"
      experiments: "MLflow"
    pros:
      - "Maximum flexibility (any tool, any config)"
      - "Multi-cloud portable (run on any K8s)"
      - "Cost-optimized at scale (raw compute + spot instances)"
      - "Full observability (you control the entire stack)"
    cons:
      - "Significant upfront investment (2-3 months setup)"
      - "Ongoing maintenance (security patches, upgrades, incidents)"
      - "Requires platform engineering expertise"
      - "Slower time-to-production initially"
    best_for: "Large teams (20+), multi-cloud, cost-sensitive at scale"
    
  pattern_3_hybrid:
    description: "Best of both — self-hosted training, managed serving"
    stack:
      training: "Self-hosted Ray/K8s cluster (maximum GPU utilization)"
      serving: "Managed endpoints (auto-scaling without engineering)"
      pipelines: "Argo Workflows (flexible, open-source)"
      features: "Feast (self-hosted, portable)"
      monitoring: "Managed (cloud-native monitoring)"
      experiments: "MLflow (self-hosted, central)"
    pros:
      - "Optimized cost where it matters most (training = biggest spend)"
      - "Operational simplicity for serving (hardest to build well)"
      - "Good balance of flexibility and convenience"
      - "Easier migration path (not all-in on vendor)"
    cons:
      - "Two systems to understand (managed + self-hosted)"
      - "Integration complexity at boundaries"
      - "Still need some platform expertise"
    best_for: "Medium teams (5-20), growing organizations"
```

### Migration Strategies

```python
# Migration patterns between managed and self-hosted

"""
Strategies for migrating between managed ML platforms and self-hosted infrastructure.
"""

migration_strategies = {
    "managed_to_self_hosted": {
        "phases": [
            {
                "phase": 1,
                "name": "Containerize Everything",
                "duration": "2-4 weeks",
                "actions": [
                    "Move all training code to Docker containers",
                    "Remove platform-specific SDK calls from training scripts",
                    "Use environment variables for configuration (not hardcoded paths)",
                    "Store artifacts in cloud storage (S3/GCS) not platform-specific stores",
                ],
                "principle": "Training code should run anywhere with Docker",
            },
            {
                "phase": 2,
                "name": "Self-Host Experiment Tracking",
                "duration": "1-2 weeks",
                "actions": [
                    "Deploy MLflow on Kubernetes",
                    "Migrate experiment history (if needed)",
                    "Update training scripts to log to MLflow",
                    "Set up artifact storage (S3/GCS backend)",
                ],
                "principle": "Experiments are the easiest to self-host (low risk)",
            },
            {
                "phase": 3,
                "name": "Self-Host Training",
                "duration": "4-8 weeks",
                "actions": [
                    "Deploy Ray or Kubeflow Training Operator on K8s",
                    "Set up GPU node pools with autoscaler",
                    "Implement job scheduling and priority queues",
                    "Add spot instance support with checkpointing",
                    "Migrate training jobs one by one (start with non-critical)",
                ],
                "principle": "Training is the biggest cost — self-hosting here saves most money",
            },
            {
                "phase": 4,
                "name": "Self-Host Serving (Optional)",
                "duration": "6-12 weeks",
                "actions": [
                    "Deploy KServe or Seldon Core",
                    "Implement auto-scaling (HPA + custom metrics)",
                    "Set up canary deployment and traffic splitting",
                    "Add request batching and model optimization",
                    "Implement A/B testing framework",
                ],
                "principle": "Serving is hardest to build — only do if justified by cost/flexibility",
                "warning": "Most teams keep managed serving — it's the hardest to replicate",
            },
        ],
        "total_duration": "3-6 months for full migration",
        "recommendation": "Do phases 1-3, keep managed serving unless scale justifies phase 4",
    },
    
    "self_hosted_to_managed": {
        "phases": [
            {
                "phase": 1,
                "name": "Identify Platform Equivalents",
                "actions": [
                    "Map current tools to managed equivalents",
                    "Identify gaps (features you'll lose)",
                    "Calculate cost impact",
                ],
            },
            {
                "phase": 2,
                "name": "Migrate Serving First",
                "reasoning": "Managed serving provides immediate operational relief",
                "actions": [
                    "Deploy models to managed endpoints",
                    "Set up auto-scaling and monitoring",
                    "Gradually shift traffic from self-hosted to managed",
                ],
            },
            {
                "phase": 3,
                "name": "Migrate Pipelines",
                "actions": [
                    "Rewrite Argo/Kubeflow pipelines in platform SDK",
                    "Migrate scheduling and triggers",
                    "Verify pipeline outputs match",
                ],
            },
            {
                "phase": 4,
                "name": "Migrate Training",
                "actions": [
                    "Wrap training containers for platform compatibility",
                    "Configure distributed training settings",
                    "Verify training performance matches",
                ],
            },
        ],
        "motivation": "Usually driven by reducing operational burden or compliance needs",
    },
}


# Abstraction layer for portability
abstraction_patterns = {
    "training_abstraction": {
        "principle": "Training code should be infrastructure-agnostic",
        "pattern": "Separate model code from infrastructure code",
        "example": {
            "model_code": "Pure PyTorch — no SageMaker/Vertex SDK in training loop",
            "infrastructure_code": "Thin wrapper that handles platform-specific launching",
            "config": "YAML config for hyperparameters (not hardcoded)",
            "artifacts": "Standard paths (/opt/ml/model/ or configurable)",
        },
    },
    
    "serving_abstraction": {
        "principle": "Model serving interface should be standard",
        "pattern": "Use standard model formats and serving protocols",
        "standards": [
            "ONNX for model format (portable across frameworks)",
            "gRPC/REST with standard predict API",
            "OpenInference for tracing (portable across serving platforms)",
            "MLflow model format (supports multiple platforms)",
        ],
    },
    
    "pipeline_abstraction": {
        "principle": "Pipeline logic should be separable from orchestrator",
        "pattern": "Define pipeline steps as containers with standard I/O",
        "tools": [
            "Kubeflow Pipelines v2 (runs on any K8s, including managed)",
            "ZenML (abstraction layer over multiple orchestrators)",
            "Metaflow (Netflix — runs on K8s, AWS Step Functions, etc.)",
        ],
    },
}
```

---

## How It Works in Practice

### Hybrid Architecture Example

```yaml
Hybrid_Architecture:
  scenario: "E-commerce company, 15 ML engineers, 30 models"
  
  decision:
    training: "Self-hosted Ray cluster on EKS"
    reason: "60% of ML spend is training. Self-hosting saves $15K/month"
    
    serving: "SageMaker Endpoints (managed)"
    reason: "Auto-scaling, zero-downtime deploys, monitoring — too complex to build"
    
    experiments: "MLflow on EKS (self-hosted)"
    reason: "Free, portable, team already knows it"
    
    pipelines: "Argo Workflows on EKS (self-hosted)"
    reason: "More flexible than SageMaker Pipelines, integrates with Ray"
    
    feature_store: "Feast on EKS (self-hosted)"
    reason: "Avoid SageMaker Feature Store cost, need Redis backend for low latency"
    
  architecture:
    training_cluster:
      nodes: "5× p4d.24xlarge (reserved) + 10× spot (flexible)"
      scheduler: "Ray — bin-packs jobs across GPUs efficiently"
      utilization: "~75% GPU utilization (vs ~40% with per-job provisioning)"
      checkpointing: "S3 checkpoint store (handles spot interruptions)"
      
    serving:
      platform: "SageMaker Multi-Model Endpoints"
      models: "30 models across 10 endpoints (bin-packed)"
      scaling: "Per-model auto-scaling (SageMaker handles)"
      deployment: "CI/CD pipeline deploys to SageMaker via SDK"
      
    observability:
      metrics: "Prometheus (training) + CloudWatch (serving)"
      dashboards: "Grafana (unified view)"
      alerts: "PagerDuty integration for training failures and serving degradation"
      
  cost:
    training_self_hosted: "$8,500/month (reserved instances, 75% utilization)"
    serving_managed: "$12,000/month (SageMaker endpoints)"
    platform_tools: "$1,500/month (EKS, monitoring, storage)"
    engineering: "$7,500/month (0.5 FTE platform maintenance)"
    total: "$29,500/month"
    vs_full_managed: "$48,000/month (38% savings)"
    vs_full_self_hosted: "$22,000/month (but requires 1.5 FTE platform)"
```

---

## Interview Tip

> When asked about managed vs. self-hosted ML infrastructure: "I approach this as a spectrum, not binary. My framework considers four factors: team size, ML maturity, cost sensitivity, and multi-cloud requirements. For small teams (1-5 ML engineers), managed platforms are almost always correct — the 40% cost premium is cheaper than hiring a platform engineer. For large teams (20+), self-hosting core infrastructure (training, experiment tracking) saves 30-50% and provides the flexibility mature teams need. The sweet spot for most growing teams (5-20 engineers) is hybrid: self-host training (biggest cost, most customization needed) and use managed serving (hardest to build well — auto-scaling, zero-downtime deploys, monitoring). I always containerize training code to be infrastructure-agnostic — no platform SDK in the training loop itself. This enables migration without rewriting model code. Key metrics I track: GPU utilization (self-hosted should be >70% to justify the complexity), time-to-deployment (managed should be faster — if not, it's misconfigured), and total cost of ownership (include engineering time for self-hosted, not just compute). The break-even point is typically around $50K/month in ML infrastructure spend — below that, managed is cheaper when you account for engineering time."

---

## Common Mistakes

1. **Premature self-hosting** — Team of 3 engineers building Kubernetes ML platform instead of shipping models. Solution: use managed platforms until you hit clear limitations (cost, flexibility, scale). Build platform capabilities only when the team can afford dedicated platform engineers.

2. **Not accounting for total cost** — Comparing only compute costs (self-hosted cheaper!) while ignoring: platform engineer salary, incident response time, setup time, maintenance, security patches. Solution: calculate TCO (Total Cost of Ownership) including engineering time over 2 years.

3. **Vendor lock-in without awareness** — Using platform-specific SDK calls (SageMaker Estimator, Vertex AI custom container) deep in training code. Migration becomes a rewrite. Solution: containerize training code as infrastructure-agnostic. Platform-specific code stays in a thin orchestration layer.

4. **Underestimating serving complexity** — "We'll just deploy on K8s" — then struggling with auto-scaling, canary deployment, request batching, model versioning, A/B testing, zero-downtime updates. Solution: managed serving is the last thing to self-host. Only do it at significant scale with dedicated SRE support.

5. **All-or-nothing thinking** — "We must be fully on SageMaker" or "We must be fully on Kubernetes." Solution: use the hybrid pattern — pick the best option per component. Training might be self-hosted while serving is managed. This is the most common pattern at mature companies.

---

## Key Takeaways

- Not binary: most mature teams use hybrid (self-hosted training + managed serving)
- Team size matters: <5 engineers → managed; 5-20 → hybrid; 20+ → mostly self-hosted
- Break-even: self-hosting becomes cost-effective around $50K+/month ML spend
- Cost premium: managed platforms add 40-60% over raw compute (but save engineering time)
- Containerize everything: training code must be infrastructure-agnostic (portable)
- Serving is hardest: auto-scaling, zero-downtime deploys, A/B testing — keep managed longest
- Training saves most: biggest cost component, easiest to self-host (Ray/Kubeflow)
- Migration path: containerize → self-host experiments → self-host training → (maybe) serving
- TCO includes engineering: 0.5-2 FTE for platform maintenance + on-call
- Multi-cloud forces self-hosted: portable stack (Kubeflow, MLflow, KServe) if vendor-neutral required
