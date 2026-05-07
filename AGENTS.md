# ML Engineering Notes — Agent Instructions

## Project Identity
- **Repo name**: `ml-engineering-notes`
- **Theme color**: Amber (#d97706)
- **Total**: 14 sections × 12 files = 168 lessons
- **Tagline**: "168 in-depth lessons on machine learning engineering — MLOps, model serving, LLMOps, fine-tuning infrastructure, AI agents, experiment tracking, and production ML systems."
- **GitHub**: `janarthansekarts/ml-engineering-notes`
- **Cloudflare URL**: `https://ml-engineering-notes.janarthansekarts.workers.dev`

## Content Requirements
- Every file: 250-500 lines
- Format: Problem → Analogy → Deep Dive (yaml/python code blocks) → Common Mistakes → Key Takeaways
- Latest info: 2025-2026 ML stack, LLMOps, AI agents, multi-modal, reasoning models, MCP protocol
- AI coverage: THIS IS THE AI REPO — everything is AI/ML. Focus on ENGINEERING not data science theory
- Cross-references: link to data-engineering-notes, kubernetes-deep-dive, python-mastery-notes, observability-notes

## Section Structure

### 00-ml-engineering-fundamentals/ (12 files)
```
01-ml-engineering-vs-data-science.md       — role distinction, engineering concerns, production focus
02-ml-system-architecture.md               — training → evaluation → deployment → monitoring lifecycle
03-ml-development-workflow.md              — experiment → validate → deploy → iterate, reproducibility
04-ml-infrastructure-overview.md           — compute (GPU/TPU), storage, orchestration, serving
05-data-centric-ai.md                      — data quality > model quality, data flywheel, labeling
06-experiment-management.md                — MLflow, W&B, experiment tracking, artifact management
07-model-registry.md                       — versioning, staging, approval workflows, lineage
08-ml-project-lifecycle.md                 — scoping, data, modeling, deployment, maintenance phases
09-ml-team-organization.md                — ML platform teams, embedded ML engineers, responsibilities
10-cost-of-ml.md                           — GPU costs, training budgets, inference costs, optimization
11-ml-technical-debt.md                    — hidden debt, pipeline jungles, dead experimental paths
12-ai-engineering-2025.md                 — shift from ML to AI engineering, LLM-first, prompt > train
```

### 01-mlops-fundamentals/ (12 files)
```
01-what-is-mlops.md                        — MLOps definition, DevOps parallels, maturity levels
02-ml-pipelines.md                         — training pipelines, data pipelines, serving pipelines
03-ci-cd-for-ml.md                         — continuous training, model testing, deployment gates
04-feature-stores.md                       — online/offline stores, Feast, Tecton, feature serving
05-model-versioning.md                     — DVC, MLflow models, model artifacts, reproducibility
06-data-versioning.md                      — DVC, LakeFS, Delta Lake time travel, dataset versioning
07-pipeline-orchestration.md               — Kubeflow Pipelines, Airflow for ML, Vertex AI Pipelines
08-model-validation.md                     — offline evaluation, shadow deployment, champion-challenger
09-infrastructure-as-code-ml.md            — Terraform for ML, GPU clusters, auto-provisioning
10-ml-monitoring.md                        — data drift, model drift, performance degradation, alerts
11-ml-governance.md                        — model cards, fairness, explainability, audit trails
12-llmops.md                              — LLMOps vs MLOps, prompt management, LLM monitoring, evaluation
```

### 02-model-training-infrastructure/ (12 files)
```
01-gpu-computing-fundamentals.md           — GPU architecture, CUDA, memory hierarchy, tensor cores
02-distributed-training.md                 — data parallelism, model parallelism, pipeline parallelism
03-training-frameworks.md                  — PyTorch DDP, DeepSpeed, FSDP, Megatron-LM
04-training-infrastructure.md             — GPU clusters, networking (InfiniBand, RoCE), storage for training
05-hyperparameter-optimization.md          — grid search, Bayesian, Optuna, Ray Tune, population-based
06-training-efficiency.md                  — mixed precision, gradient accumulation, checkpointing, compile
07-training-data-management.md             — data loading, caching, sharding, streaming datasets
08-experiment-tracking-deep.md             — MLflow, W&B, experiment comparison, hyperparameter sweeps
09-large-model-training.md                 — training LLMs, 3D parallelism, ZeRO optimizer, trillion params
10-training-on-cloud.md                    — SageMaker, Vertex AI, Azure ML, spot/preemptible training
11-training-debugging.md                   — loss curves, gradient issues, NaN debugging, reproducibility
12-efficient-training-2025.md             — LoRA, QLoRA, PEFT methods, efficient fine-tuning, adapters
```

### 03-model-serving/ (12 files)
```
01-serving-fundamentals.md                 — online vs batch, latency vs throughput, serving patterns
02-serving-frameworks.md                   — TorchServe, TF Serving, Triton, KServe, vLLM, TGI
03-model-optimization.md                   — quantization (INT8, INT4, GPTQ, AWQ), pruning, distillation
04-inference-hardware.md                   — GPU inference, CPU optimization, Apple Silicon, inference chips
05-batching-strategies.md                  — dynamic batching, continuous batching, in-flight batching
06-model-caching.md                        — model loading, warm pools, KV cache, paged attention
07-autoscaling-inference.md                — scaling GPU inference, scale-to-zero, traffic prediction
08-a-b-testing-models.md                   — traffic splitting, statistical significance, multi-armed bandit
09-edge-inference.md                       — on-device ML, TFLite, ONNX Runtime, CoreML, WebGPU
10-multi-model-serving.md                  — model routing, ensemble, cascade, speculative decoding
11-serving-llms.md                         — vLLM, TGI, continuous batching, KV cache, PagedAttention
12-serverless-inference.md                — Lambda/Cloud Functions for ML, serverless GPU, cold start optimization
```

### 04-llm-operations/ (12 files)
```
01-llm-landscape-2025.md                   — GPT-4o, Claude 4, Gemini 2, Llama 4, open vs closed
02-prompt-engineering-ops.md               — prompt versioning, A/B testing prompts, prompt management
03-rag-operations.md                       — RAG pipeline monitoring, retrieval quality, freshness
04-fine-tuning-operations.md               — when to fine-tune, LoRA/QLoRA workflow, evaluation
05-llm-evaluation.md                       — benchmarks, LLM-as-judge, human eval, task-specific metrics
06-llm-guardrails.md                       — input/output filtering, content safety, prompt injection defense
07-llm-cost-management.md                  — token optimization, caching, routing, budget controls
08-llm-observability.md                    — tracing LLM calls, token usage, latency, quality monitoring
09-llm-gateway-operations.md              — multi-provider routing, fallbacks, rate limiting, API management
10-context-window-management.md            — long context strategies, RAG vs fine-tune vs context
11-multi-modal-operations.md               — vision, audio, video models, multi-modal pipelines
12-reasoning-models.md                    — chain-of-thought, o1/o3 patterns, structured output, agentic reasoning
```

### 05-ai-agents/ (12 files)
```
01-ai-agent-fundamentals.md                — what are agents, reasoning + action, tool use, memory
02-agent-architectures.md                  — ReAct, plan-and-execute, reflexion, multi-agent
03-tool-use-patterns.md                    — function calling, MCP protocol, tool registries, sandboxing
04-agent-memory.md                         — short-term (context), long-term (vector store), episodic memory
05-multi-agent-systems.md                  — CrewAI, AutoGen, agent orchestration, communication patterns
06-agent-frameworks.md                     — LangGraph, CrewAI, OpenAI Assistants, Anthropic tools, Vercel AI SDK
07-agent-evaluation.md                     — task completion rate, tool use efficiency, safety testing
08-agent-deployment.md                     — hosting agents, scaling, state management, persistence
09-agent-safety.md                         — sandboxing, permission systems, human-in-the-loop, guardrails
10-coding-agents.md                        — Copilot architecture, code generation, test generation, debugging
11-enterprise-agents.md                    — customer service agents, internal tooling, workflow automation
12-agent-infrastructure.md                — MCP servers, tool hosting, agent runtime, orchestration platforms
```

### 06-feature-engineering/ (12 files)
```
01-feature-engineering-principles.md       — feature design, signal vs noise, feature importance
02-feature-store-architecture.md           — online/offline split, Feast architecture, Tecton, Redis
03-feature-computation.md                  — batch features (Spark), streaming features (Flink), real-time
04-feature-serving.md                      — low-latency serving, caching, pre-computation, on-demand
05-feature-monitoring.md                   — feature drift, staleness, completeness, quality metrics
06-embedding-features.md                   — learned embeddings, pre-trained embeddings, feature crossing
07-time-series-features.md                — windowed aggregations, lag features, rolling statistics
08-text-and-nlp-features.md               — TF-IDF, word embeddings, sentence embeddings, LLM features
09-feature-pipelines.md                    — orchestration, backfilling, point-in-time correctness
10-feature-discovery.md                    — feature catalogs, reusability, team sharing, documentation
11-automated-feature-eng.md               — AutoML features, feature generation, feature selection
12-features-for-llm-apps.md              — embeddings as features, retrieval features, user behavior features for AI
```

### 07-model-monitoring/ (12 files)
```
01-monitoring-fundamentals.md              — why models degrade, concept drift, data drift, model rot
02-data-drift-detection.md                 — statistical tests, KS test, PSI, distribution comparison
03-concept-drift-detection.md              — performance monitoring, drift detection algorithms, adaptation
04-prediction-monitoring.md                — output distribution, confidence calibration, anomaly detection
05-ground-truth-collection.md             — delayed feedback, proxy labels, human evaluation loops
06-alerting-for-ml.md                     — when to alert, false positives, SLOs for ML, escalation
07-model-explainability.md                — SHAP, LIME, feature importance, explainability in production
08-fairness-monitoring.md                 — bias detection, protected attributes, fairness metrics
09-a-b-test-monitoring.md                 — experiment monitoring, significance, guardrail metrics
10-monitoring-infrastructure.md            — Evidently, Whylabs, Arize, custom monitoring pipelines
11-retraining-triggers.md                 — when to retrain, automated triggers, continuous training
12-llm-monitoring.md                      — LLM quality monitoring, hallucination detection, response quality tracking
```

### 08-ml-platform-engineering/ (12 files)
```
01-ml-platform-architecture.md             — self-serve ML platform, components, team interfaces
02-compute-management.md                   — GPU clusters, scheduling, quota, multi-tenancy
03-notebook-infrastructure.md              — JupyterHub, managed notebooks, collaboration, versioning
04-pipeline-platform.md                    — shared pipeline infrastructure, templates, reusable components
05-model-deployment-platform.md            — one-click deploy, canary, rollback, multi-framework
06-data-platform-for-ml.md               — data access, feature platform, dataset management
07-experiment-platform.md                  — shared experiment tracking, comparison, collaboration
08-cost-management-platform.md             — chargeback, quota, spot instances, cost optimization
09-governance-platform.md                  — model registry, approval workflows, audit, compliance
10-developer-experience.md                — SDKs, CLI tools, documentation, onboarding, golden paths
11-platform-observability.md              — platform metrics, user metrics, reliability, capacity
12-ai-platform-2025.md                   — LLM platform, prompt management platform, agent hosting platform
```

### 09-ml-testing/ (12 files)
```
01-ml-testing-fundamentals.md              — testing pyramid for ML, data tests, model tests, integration
02-data-testing.md                         — Great Expectations, data contracts, schema validation
03-model-unit-testing.md                   — behavioral testing, invariance, directional expectations
04-integration-testing-ml.md               — pipeline testing, end-to-end, testcontainers for ML
05-performance-testing.md                  — latency testing, throughput, load testing inference
06-regression-testing.md                   — model regression, golden datasets, comparison testing
07-evaluation-datasets.md                  — creating eval sets, versioning, contamination prevention
08-adversarial-testing.md                  — robustness testing, adversarial examples, edge cases
09-fairness-testing.md                     — bias testing, subgroup analysis, counterfactual testing
10-a-b-testing.md                          — online experiments, statistical testing, bayesian AB
11-chaos-testing-ml.md                    — data failures, model failures, infrastructure failures
12-llm-testing.md                         — prompt testing, RAG evaluation, RAGAS, hallucination testing, red teaming
```

### 10-production-ml-patterns/ (12 files)
```
01-ml-design-patterns.md                   — common patterns, when to apply, trade-offs
02-batch-prediction.md                     — batch scoring, scheduling, incremental, caching results
03-online-prediction.md                    — real-time serving, pre-computation, hybrid approaches
04-streaming-ml.md                         — online learning, mini-batch, streaming inference
05-recommendation-systems.md               — collaborative filtering, content-based, hybrid, real-time
06-search-ranking.md                       — learning to rank, two-stage (retrieve + rank), features
07-fraud-detection.md                      — real-time scoring, feature engineering, feedback loops
08-nlp-in-production.md                    — NER, classification, sentiment, embeddings at scale
09-computer-vision-production.md           — image classification, object detection, video analysis
10-time-series-ml.md                       — forecasting, anomaly detection, temporal patterns
11-ml-for-ops.md                          — AIOps patterns, log analysis, incident prediction, remediation
12-genai-in-production.md                 — LLM apps, RAG, agents, guardrails, evaluation — complete pattern
```

### 11-responsible-ai/ (12 files)
```
01-responsible-ai-principles.md            — fairness, transparency, accountability, safety, privacy
02-bias-and-fairness.md                    — types of bias, measurement, mitigation, fair ML
03-explainability.md                       — interpretable models, post-hoc explanation, SHAP, LIME
04-privacy-preserving-ml.md                — differential privacy, federated learning, secure computation
05-safety-alignment.md                     — RLHF, constitutional AI, red teaming, safety evaluation
06-ai-governance.md                        — policy frameworks, EU AI Act, model documentation
07-environmental-impact.md                 — compute carbon footprint, efficient training, green AI
08-intellectual-property.md                — training data rights, model licensing, open source AI
09-content-safety.md                       — toxicity detection, content moderation, guardrails
10-ai-risk-management.md                  — NIST AI RMF, risk assessment, continuous monitoring
11-human-in-the-loop.md                   — when to involve humans, confidence thresholds, escalation
12-future-of-responsible-ai.md            — regulation trends, industry standards, technical solutions
```

### 12-ml-on-cloud/ (12 files)
```
01-aws-sagemaker.md                        — training, endpoints, pipelines, feature store, Studio
02-gcp-vertex-ai.md                        — Vertex training, prediction, pipelines, Model Garden
03-azure-machine-learning.md               — Azure ML workspace, compute, endpoints, responsible AI
04-managed-vs-self-hosted.md               — when to use managed, cost comparison, flexibility trade-offs
05-gpu-cloud-options.md                    — A100, H100, H200, cloud GPU comparison, spot pricing
06-ml-networking-cloud.md                  — VPC for ML, GPU networking, InfiniBand alternatives
07-storage-for-ml.md                       — S3, GCS, ADLS, high-performance storage, data caching
08-ml-security-cloud.md                    — data encryption, model security, IAM for ML, VPC
09-cost-optimization-cloud.md              — spot training, committed use, right-sizing, auto-shutdown
10-multi-cloud-ml.md                       — portable pipelines, abstraction layers, vendor lock-in
11-serverless-ml-cloud.md                  — serverless inference, Lambda ML, Cloud Run GPU
12-llm-cloud-services.md                  — Azure OpenAI, Vertex AI Gemini, Bedrock, managed LLM hosting
```

### 13-emerging-ml-engineering/ (12 files)
```
01-foundation-models.md                    — what are foundation models, adaptation patterns, fine-tuning vs prompting
02-multi-modal-engineering.md              — vision + language, audio + text, multi-modal pipelines
03-small-language-models.md                — Phi, Gemma, on-device models, efficiency, when to use
04-synthetic-data.md                       — generating training data, validation, LLM for data generation
05-continual-learning.md                   — online learning, catastrophic forgetting, knowledge retention
06-retrieval-augmented-gen.md              — RAG engineering patterns, advanced RAG, evaluation
07-model-compression.md                    — quantization, pruning, distillation, speculative decoding
08-ai-compiler-optimization.md            — torch.compile, XLA, TVM, graph optimization
09-edge-ml-engineering.md                  — on-device, TFLite, ONNX, CoreML, WebGPU inference
10-ai-agent-engineering.md                — building production agents, tool use, memory, evaluation
11-autonomous-ai-systems.md               — self-improving AI, AutoML 2.0, neural architecture search
12-future-of-ml-engineering.md            — trends 2025-2027, world models, embodied AI, AGI readiness
```

## Cross-Reference Map
- Data pipelines → data-engineering-notes
- K8s deployment → kubernetes-deep-dive/09-kubernetes-for-ai-ml
- Python implementation → python-mastery-notes
- Monitoring → observability-notes
- Cloud platforms → cloud-notes
- API serving → api-design-notes
- Architecture → software-architecture-notes
- Go for ML tools → go-programming-notes
- Security → security-engineering-notes

## Validation Checklist
- [ ] Every file has Problem → Analogy → Deep Dive → Common Mistakes → Key Takeaways
- [ ] Engineering focus (not data science theory/math)
- [ ] 2025-2026: LLMOps, AI agents (MCP, function calling), reasoning models
- [ ] GPU: H100/H200/B200, not just A100
- [ ] Frameworks: PyTorch 2.x, vLLM, TGI, LangGraph, CrewAI
- [ ] Fine-tuning: LoRA/QLoRA/PEFT as standard (not full fine-tune)
- [ ] Serving: continuous batching, PagedAttention, speculative decoding
- [ ] Cross-references to other repos verified
- [ ] Each file is 250-500 lines with deep technical content
