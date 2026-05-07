# Model Deployment Platform

## The Problem / Why This Matters

Deploying ML models to production is the #1 bottleneck in most organizations. Data scientists train models in notebooks, hand off artifacts to engineers, who then spend weeks building serving infrastructure, writing API code, configuring autoscaling, setting up health checks, and integrating with monitoring. A model deployment platform reduces this from weeks to minutes: one-click (or one-command) deployment with built-in canary releases, automatic rollback, multi-framework support (PyTorch, TensorFlow, scikit-learn, XGBoost, LLMs), GPU inference, autoscaling, and monitoring. In 2026, the deployment platform must also handle: vLLM-based LLM serving (with PagedAttention and continuous batching), A/B testing between model versions, shadow deployments (new model receives traffic but responses aren't used), multi-model endpoints (ensemble or routing), and edge deployment (models running on mobile/IoT). The engineering challenge: supporting diverse model types (from a 100KB logistic regression to a 70B parameter LLM) through a unified interface while providing flexibility for custom serving logic when needed. KServe, Seldon Core, BentoML, Ray Serve, and cloud-managed options (SageMaker Endpoints, Vertex AI Prediction) are the primary building blocks.

---

## The Analogy

Think of a model deployment platform like a restaurant kitchen's expediting system:

- **Without platform** = Every chef builds their own serving station, designs their own plates, figures out their own timing. Chaos. Some dishes arrive cold, some never arrive.
- **With platform** = The expediting system. Chef puts finished dish in the window → expediter handles plating presentation (containerization), timing (traffic management), quality check (health checks), delivery to the right table (routing), and sends back anything that doesn't meet standards (rollback).

The chef (data scientist) focuses on cooking (model training). The expediting system (deployment platform) handles everything from kitchen window to customer table.

---

## Deep Dive

### Deployment Platform Architecture

```yaml
Architecture:
  model_packaging:
    what: "Convert model artifact into deployable unit"
    standards:
      mlflow_model:
        format: "MLflow Model (framework-agnostic wrapper)"
        includes: "Model artifact + conda/pip env + signature + example input"
        serving: "mlflow models serve OR export to container"
        
      oci_container:
        format: "Docker container with model + serving code + dependencies"
        advantage: "Universal — any framework, any custom logic"
        disadvantage: "Larger, slower to build, requires Dockerfile knowledge"
        
      bentoml_bento:
        format: "BentoML Bento (model + service definition + env)"
        advantage: "Optimized for serving (batching, runners, GPU management)"
        
      torchserve_mar:
        format: "TorchServe Model Archive (PyTorch-specific)"
        advantage: "Optimized for PyTorch models (eager and compiled)"
        
    llm_specific:
      vllm: "Load from HuggingFace hub or local model files"
      tgi: "Load from HuggingFace format"
      format: "Safetensors + tokenizer + config (HuggingFace standard)"
      
  serving_infrastructure:
    kserve:
      what: "Kubernetes-native model serving (formerly KFServing)"
      features:
        - "Multi-framework (PyTorch, TF, XGBoost, LightGBM, custom)"
        - "Autoscaling (scale-to-zero support)"
        - "Canary and blue-green deployments"
        - "Request batching"
        - "Model explanation (integrated explainability)"
        - "Transformer/predictor/explainer pattern"
      scale: "Proven at large scale (used by Bloomberg, IBM, others)"
      
    seldon_core:
      what: "Enterprise ML deployment on Kubernetes"
      features:
        - "Multi-model graphs (pipelines, ensembles, A/B tests)"
        - "Custom inference servers"
        - "Explainability and drift detection"
        - "Rich inference graph (pre/post-processing, routing)"
      scale: "Enterprise-focused, supports complex serving topologies"
      
    ray_serve:
      what: "Ray-based model serving"
      features:
        - "Python-native (decorate function → deployment)"
        - "Composition (chain multiple models)"
        - "Dynamic batching"
        - "Multi-model serving on shared GPU"
        - "Fractional GPU support"
      advantage: "Same framework for training (Ray Train) and serving"
      
    bentoml:
      what: "Framework for building ML serving APIs"
      features:
        - "Adaptive batching (optimal batch size based on latency target)"
        - "Multi-model serving"
        - "GPU management"
        - "Built-in Prometheus metrics"
        - "Deployment to any cloud (BentoCloud, K8s, Docker)"
      advantage: "Best DX for building custom serving logic"
      
    vllm:
      what: "High-performance LLM serving"
      features:
        - "PagedAttention (efficient KV cache management)"
        - "Continuous batching (process new requests without waiting)"
        - "Tensor parallelism (split model across GPUs)"
        - "LoRA serving (multiple adapters on one base model)"
        - "Speculative decoding"
        - "OpenAI-compatible API"
      use_for: "Any LLM serving (replaces custom serving code)"
      
  traffic_management:
    canary:
      what: "Route small % of traffic to new model version"
      flow: "Deploy new version → 5% traffic → monitor metrics → promote or rollback"
      duration: "24-72 hours (depends on traffic volume and confidence)"
      
    blue_green:
      what: "Two full environments, instant switch"
      flow: "Deploy to green → test → switch traffic from blue to green"
      advantage: "Instant rollback (switch back to blue)"
      disadvantage: "2× resources during transition"
      
    shadow:
      what: "New model receives all traffic, responses discarded"
      flow: "Deploy shadow → compare predictions to primary → no user impact"
      use_for: "Validate new model behavior before any user exposure"
      
    ab_testing:
      what: "Statistical comparison between model versions"
      flow: "50/50 split → measure business metric → statistical test → winner"
      duration: "Until statistical significance achieved"
```

### One-Click Deployment Interface

```python
# Model deployment platform SDK

"""
One-command model deployment with built-in best practices.
Handles: containerization, scaling, traffic management, monitoring, rollback.
"""


class ModelDeploymentPlatform:
    """
    Simple interface for deploying any model to production.
    
    Supports:
    - Traditional ML (scikit-learn, XGBoost, LightGBM)
    - Deep Learning (PyTorch, TensorFlow)
    - LLMs (via vLLM backend)
    - Custom models (bring your own container)
    """
    
    def deploy(
        self,
        model: str,          # "model-name:version" from registry
        strategy: str = "canary",  # "canary", "blue-green", "shadow", "direct"
        replicas: int = 2,
        gpu: str = None,     # "H100", "L40S", "T4", None (CPU)
        autoscale: bool = True,
        min_replicas: int = 1,
        max_replicas: int = 10,
        target_latency_ms: int = 100,
        canary_percent: int = 5,
        canary_duration_hours: int = 24,
        rollback_on_error_rate: float = 0.01,
    ) -> "Deployment":
        """
        Deploy a model to production.
        
        Example (simple):
            platform.deploy("churn-model:v3")
            # → Canary deploy, 2 replicas, autoscale, 5% traffic for 24h
            
        Example (LLM):
            platform.deploy(
                "llama-3-70b:v1",
                gpu="H100",
                replicas=4,
                target_latency_ms=500,
                strategy="shadow",
            )
        """
        # 1. Resolve model from registry
        model_artifact = self.registry.get(model)
        
        # 2. Determine serving runtime
        runtime = self._select_runtime(model_artifact)
        # PyTorch/TF → TorchServe/TF Serving or KServe
        # LLM → vLLM
        # sklearn/xgboost → MLflow Serve or custom
        
        # 3. Build serving container (if not cached)
        container = self._build_container(model_artifact, runtime)
        
        # 4. Deploy with strategy
        deployment = self._deploy_with_strategy(
            container=container,
            strategy=strategy,
            config={
                "replicas": replicas,
                "gpu": gpu,
                "autoscale": autoscale,
                "min_replicas": min_replicas,
                "max_replicas": max_replicas,
                "target_latency_ms": target_latency_ms,
            },
            canary_config={
                "percent": canary_percent,
                "duration_hours": canary_duration_hours,
                "rollback_threshold": rollback_on_error_rate,
            },
        )
        
        # 5. Attach monitoring (automatic)
        self._setup_monitoring(deployment)
        
        return deployment
    
    def rollback(self, deployment_name: str, to_version: str = "previous"):
        """
        Instant rollback to previous version.
        
        Example:
            platform.rollback("churn-model")  # Back to previous version
        """
        deployment = self._get_deployment(deployment_name)
        
        if to_version == "previous":
            target = deployment.previous_version
        else:
            target = to_version
            
        # Traffic switch is instant (update Kubernetes service selector)
        self._switch_traffic(deployment, target_version=target)
        
        # Keep old version running for 1 hour (in case of re-rollback)
        self._schedule_cleanup(deployment.current_version, delay_hours=1)
    
    def promote(self, deployment_name: str):
        """
        Promote canary to full traffic.
        Called after canary period passes successfully.
        """
        deployment = self._get_deployment(deployment_name)
        self._switch_traffic(deployment, target_version=deployment.canary_version, percent=100)
    
    def _select_runtime(self, model_artifact) -> str:
        """Auto-detect best serving runtime for model type."""
        if model_artifact.framework == "transformers" and model_artifact.model_size > 1_000_000_000:
            return "vllm"  # Large language models → vLLM
        elif model_artifact.framework == "pytorch":
            return "torchserve"
        elif model_artifact.framework == "tensorflow":
            return "tf-serving"
        elif model_artifact.framework in ("sklearn", "xgboost", "lightgbm"):
            return "mlflow-serve"
        else:
            return "custom"  # User must provide Dockerfile
```

### Multi-Model Serving

```yaml
Multi_Model_Patterns:
  ensemble:
    what: "Multiple models contribute to final prediction"
    example: "3 models vote → majority wins (or weighted average)"
    implementation: "Inference graph (KServe/Seldon) or custom routing (Ray Serve)"
    
  model_routing:
    what: "Route requests to different models based on input"
    example: "Simple queries → small model, complex queries → large model"
    use_case: "Cost optimization (use expensive model only when needed)"
    implementation: "Router component that classifies request → forwards to appropriate model"
    
  ab_test:
    what: "Statistical comparison between model versions"
    example: "50% traffic to model A, 50% to model B → compare conversion rate"
    implementation: "Traffic splitting with consistent hashing (same user always sees same model)"
    
  cascade:
    what: "Try cheap model first, escalate to expensive model if uncertain"
    example: "Fast model (confidence > 0.9) → return. Low confidence → slow model."
    benefit: "90% of requests handled cheaply, only uncertain ones use expensive model"
    
  lora_multi_tenant:
    what: "One base LLM + multiple LoRA adapters for different use cases"
    example: "Llama-3-70B base + customer-support LoRA + legal LoRA + medical LoRA"
    implementation: "vLLM with dynamic LoRA loading (switch adapter per request)"
    benefit: "One GPU cluster serves many fine-tuned variants"
```

### Autoscaling Configuration

```yaml
Autoscaling:
  metrics:
    primary:
      - "Request rate (QPS — Queries Per Second)"
      - "GPU utilization (%)"
      - "Request queue depth"
    secondary:
      - "Response latency (p95)"
      - "Memory utilization"
      - "Batch queue length"
      
  strategies:
    target_based:
      what: "Scale to maintain target metric value"
      example: "Target GPU utilization = 70% → add replicas when above, remove when below"
      tool: "Kubernetes HPA (Horizontal Pod Autoscaler) or KEDA"
      
    predictive:
      what: "Scale based on predicted traffic (before demand arrives)"
      example: "Historical pattern: traffic spikes at 9 AM → pre-scale at 8:45 AM"
      tool: "Custom controller or cloud provider predictive scaling"
      
    scale_to_zero:
      what: "Remove all replicas when no traffic"
      cold_start: "30-120 seconds to spin up first replica"
      use_for: "Low-traffic models, development endpoints"
      provider: "KServe (native support), Knative"
      
  llm_specific:
    challenge: "LLMs have long cold start (loading 70B parameters: 30-60 seconds)"
    solutions:
      - "Keep minimum 1 replica always warm (no scale-to-zero for production LLMs)"
      - "Pre-load model weights on node (persistent volume)"
      - "Use smaller quantized model for cold-start responses, switch to full when warm"
```

---

## How It Works in Practice

### Deployment Flow

```yaml
Deployment_Flow:
  data_scientist_action:
    1: "Train model, evaluate, register in model registry"
    2: "Run: platform.deploy('my-model:v7', strategy='canary')"
    
  platform_actions_automatic:
    3: "Resolve model artifact from registry"
    4: "Detect framework → select runtime (torchserve/vllm/mlflow)"
    5: "Build serving container (cached if same image base)"
    6: "Deploy canary (5% traffic to new version)"
    7: "Monitor metrics (latency, error rate, prediction distribution)"
    8: "After 24h if healthy → auto-promote to 100%"
    9: "If error rate > threshold → auto-rollback to previous version"
    10: "Notify team of deployment result (Slack/email)"
    
  total_time: "5-15 minutes from command to serving traffic"
  rollback_time: "< 30 seconds (traffic switch only)"
```

---

## Interview Tip

> When asked about model deployment platform: "My deployment platform provides three key capabilities: (1) One-command deploy — data scientist runs `platform.deploy('model:v3')` and the platform handles: framework detection, container building, runtime selection (TorchServe for PyTorch, vLLM for LLMs, MLflow Serve for sklearn), replica management, and monitoring setup. Time to deploy: 5-15 minutes. (2) Safe rollouts — every production deployment uses canary strategy: 5% traffic to new version for 24 hours, automatic promotion if metrics are healthy, automatic rollback if error rate exceeds threshold. Rollback takes <30 seconds (just a traffic switch). For LLMs, I also support shadow deployments (new model receives traffic but responses aren't served to users — for evaluation without risk). (3) Multi-framework + multi-pattern — unified interface regardless of model type: logistic regression, XGBoost, PyTorch CNN, or 70B parameter LLM. Platform automatically selects the optimal runtime. Supports patterns like model routing (simple queries → cheap model, complex → expensive), LoRA multi-tenant (one base LLM + multiple adapters switched per request), and cascade (try fast model first, escalate if uncertain). Autoscaling via HPA (Horizontal Pod Autoscaler) targeting GPU utilization or request queue depth, with scale-to-zero for low-traffic endpoints."

---

## Common Mistakes

1. **No canary or gradual rollout** — Deploy new model to 100% traffic immediately. If model is bad, all users affected. Solution: always canary (5-10% for 24h), auto-rollback on metric degradation. Even for "minor" model updates — you never know.

2. **Manual deployment process** — Data scientist trains model, sends Slack message to eng, engineer manually builds container, deploys to Kubernetes. Takes 1-2 weeks. Solution: fully automated pipeline. From model registry to production with one command. Human approval only for regulated models.

3. **No rollback plan** — Deploy new model, it's bad, nobody knows how to roll back quickly. Previous version artifacts are gone, configuration has been overwritten. Solution: always keep previous version running (just not receiving traffic). Rollback = traffic switch (30 seconds). Previous version persists for 24-72 hours minimum.

4. **One-size-fits-all runtime** — Using the same serving infrastructure for a 100KB scikit-learn model and a 70B parameter LLM. The small model doesn't need GPU or complex batching. The LLM needs vLLM with continuous batching and tensor parallelism. Solution: platform auto-selects runtime based on model type and size.

5. **Not testing the deployment, only the model** — Model passes all offline evaluation (test set metrics look great). Deploy to production → crashes due to input schema mismatch, missing feature, or library version conflict. Solution: integration testing before deployment (send real-format requests to staging endpoint, validate response format and values).

---

## Key Takeaways

- Deployment platform: one-command deploy with canary, auto-rollback, multi-framework support
- Serving runtimes: KServe (general K8s), vLLM (LLMs), Ray Serve (composition), BentoML (custom)
- Traffic management: canary (gradual), blue-green (instant switch), shadow (no user impact), A/B test
- Auto-runtime selection: platform detects model type → picks optimal serving infrastructure
- Autoscaling: HPA on GPU utilization or queue depth, scale-to-zero for low-traffic
- Rollback: < 30 seconds (traffic switch), always keep previous version running
- Multi-model patterns: ensemble, routing, cascade, LoRA multi-tenant
- LLM deployment: vLLM with PagedAttention, continuous batching, tensor parallelism
- Container caching: rebuild only when model or dependencies change (save 5-10 min)
- Integration testing: validate full request-response cycle before promoting to production
