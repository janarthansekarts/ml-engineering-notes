# Serving Frameworks

## The Problem / Why This Matters

Deploying a model to production requires more than wrapping it in a Flask API. Production serving demands: high throughput (thousands of requests per second), low latency (sub-100ms for traditional ML, sub-2s TTFT for LLMs), efficient GPU utilization (batching, memory management), model management (versioning, A/B testing, rollback), and operational features (health checks, metrics, graceful shutdown). Model serving frameworks solve these challenges with purpose-built infrastructure optimized for ML inference. In 2026, the landscape has two clear tiers: (1) LLM-specific servers — vLLM (open-source, best throughput), TGI (Text Generation Inference, HuggingFace), and SGLang (compiler-based optimization) that handle the unique challenges of autoregressive generation, and (2) general-purpose servers — Triton Inference Server (NVIDIA, multi-framework, multi-model), TorchServe (PyTorch), and KServe (Kubernetes-native) that serve any model type. Choosing the right framework depends on your model type, scale requirements, and operational maturity.

---

## The Analogy

Think of model serving frameworks like different kitchen setups:

- **Flask/FastAPI (custom)** = Cooking on a hot plate in your apartment. Works for prototyping but can't handle a dinner rush. No ventilation (GPU management), no prep stations (batching), no fire suppression (error handling).
- **TorchServe** = A professional kitchen designed for one cuisine (PyTorch). Good equipment, proper workflow, but limited to what it's designed for.
- **Triton Inference Server** = An industrial kitchen that supports every cuisine (multi-framework). Multiple cooking stations (GPU instances), intelligent queuing, batch prep area. Complex to configure but handles anything.
- **vLLM** = A specialized kitchen just for sushi (LLM generation). Every aspect is optimized for this one purpose: the conveyor belt (continuous batching), the rice station (KV cache management), the ordering system (PagedAttention). Nothing else matches it for sushi.
- **KServe** = A restaurant management system. Handles multiple kitchens, routes customers, manages reservations (auto-scaling), and provides the dining room experience. The kitchens (model servers) plug in underneath.

---

## Deep Dive

### vLLM (Virtual Large Language Model)

```yaml
vLLM:
  what: "High-throughput LLM serving engine with PagedAttention"
  creator: "UC Berkeley (open-source)"
  status: "De facto standard for LLM serving in 2026"
  
  key_innovations:
    paged_attention:
      what: "Manage KV cache like virtual memory pages (non-contiguous allocation)"
      problem_solved: "Fixed KV cache allocation wastes 60-80% of GPU memory"
      how: "Allocate KV cache in small pages, map logical positions to physical pages"
      benefit: "2-4x more concurrent requests (better memory utilization)"
      
    continuous_batching:
      what: "Don't wait for all requests in a batch to complete — add new requests as others finish"
      problem_solved: "Static batching wastes GPU on completed-but-waiting requests"
      benefit: "2-3x better throughput vs static batching"
      
    prefix_caching:
      what: "Cache KV entries for shared prefixes (system prompts, common context)"
      benefit: "Avoid recomputing KV for repeated system prompts"
      
  features:
    - "OpenAI-compatible API (drop-in replacement)"
    - "Tensor parallelism (multi-GPU serving)"
    - "Multi-LoRA serving (multiple adapters, one base model)"
    - "Speculative decoding support"
    - "Quantization support (AWQ, GPTQ, FP8)"
    - "Vision-language model support"
    - "Structured output (JSON mode, grammar-guided generation)"
    
  deployment:
    docker: "docker run --gpus all vllm/vllm-openai --model meta-llama/Llama-4-8B"
    python: |
      from vllm import LLM, SamplingParams
      
      llm = LLM(
          model="meta-llama/Llama-4-8B",
          tensor_parallel_size=2,  # 2 GPUs
          gpu_memory_utilization=0.9,
          max_model_len=8192,
          quantization="awq",  # Optional quantization
      )
      
      outputs = llm.generate(
          ["What is machine learning?"],
          SamplingParams(temperature=0.7, max_tokens=512)
      )
      
  performance:
    throughput: "2-5x higher than HuggingFace Transformers generate()"
    latency: "Competitive TTFT, superior throughput under load"
    memory: "60-80% better memory utilization (PagedAttention)"
    
  when_to_use:
    - "Any LLM serving workload (default choice in 2026)"
    - "High-throughput requirements"
    - "Multi-user serving (chatbot, API)"
    - "Need OpenAI-compatible API"
```

### TGI (Text Generation Inference)

```yaml
TGI:
  full_name: "Text Generation Inference"
  what: "HuggingFace's optimized LLM serving solution"
  
  key_features:
    - "Continuous batching"
    - "Flash Attention integration"
    - "Tensor parallelism"
    - "Quantization (GPTQ, AWQ, EETQ, bitsandbytes)"
    - "Watermarking (detect AI-generated text)"
    - "Token streaming (Server-Sent Events)"
    - "Guidance/structured output"
    - "Multi-LoRA support"
    
  deployment:
    docker: |
      docker run --gpus all \
        -e MODEL_ID=meta-llama/Llama-4-8B \
        -e QUANTIZE=awq \
        -p 8080:80 \
        ghcr.io/huggingface/text-generation-inference
        
  advantages:
    - "Excellent HuggingFace ecosystem integration"
    - "Simple deployment (single Docker command)"
    - "Good documentation and community support"
    - "Native support for most HuggingFace models"
    - "Built-in safety features (watermarking)"
    
  vs_vllm:
    throughput: "vLLM slightly higher in most benchmarks"
    ease_of_use: "TGI easier for HuggingFace model users"
    features: "Both comparable, different strengths"
    recommendation: "vLLM for maximum throughput, TGI for HF ecosystem integration"
    
  when_to_use:
    - "HuggingFace-first workflow"
    - "Need watermarking or safety features"
    - "Simple deployment requirements"
    - "HuggingFace Inference Endpoints (managed TGI)"
```

### NVIDIA Triton Inference Server

```yaml
Triton:
  what: "General-purpose model serving supporting multiple frameworks"
  creator: "NVIDIA (open-source)"
  
  key_capabilities:
    multi_framework: "PyTorch, TensorFlow, ONNX, TensorRT, OpenVINO, Python backend"
    multi_model: "Serve hundreds of models simultaneously"
    dynamic_batching: "Intelligent request batching with configurable policies"
    model_ensemble: "Chain multiple models in a pipeline (preprocessing → model → postprocessing)"
    concurrent_model_execution: "Run multiple models on same GPU"
    model_management: "Load/unload models dynamically, version management"
    
  model_repository:
    structure: |
      model_repository/
      +-- text_classifier/
      |   +-- config.pbtxt
      |   +-- 1/            # Version 1
      |   |   +-- model.onnx
      |   +-- 2/            # Version 2
      |       +-- model.onnx
      +-- embedding_model/
      |   +-- config.pbtxt
      |   +-- 1/
      |       +-- model.pt
      
    config_example: |
      name: "text_classifier"
      platform: "onnxruntime_onnx"
      max_batch_size: 64
      input [{
        name: "input_ids"
        data_type: TYPE_INT64
        dims: [512]
      }]
      output [{
        name: "logits"
        data_type: TYPE_FP32
        dims: [3]
      }]
      dynamic_batching {
        preferred_batch_size: [8, 16, 32]
        max_queue_delay_microseconds: 100000
      }
      instance_group [{
        count: 2
        kind: KIND_GPU
      }]
      
  performance_optimization:
    tensorrt_backend: "Compile models to TensorRT for maximum GPU throughput"
    shared_memory: "Zero-copy data transfer between client and server"
    cuda_streams: "Concurrent model execution on same GPU"
    response_cache: "Cache responses for repeated inputs"
    
  when_to_use:
    - "Multi-framework environment (PyTorch + TensorFlow + ONNX)"
    - "Multi-model serving (hundreds of models)"
    - "Need maximum GPU utilization with dynamic batching"
    - "Model pipelines (ensemble: tokenize → embed → classify → postprocess)"
    - "Not for LLM text generation (use vLLM/TGI instead)"
    
  when_not_to_use:
    - "Pure LLM serving (vLLM/TGI are better optimized)"
    - "Simple single-model deployment (overkill complexity)"
```

### TorchServe

```yaml
TorchServe:
  what: "PyTorch's official model serving solution"
  creator: "Meta/AWS (open-source)"
  
  features:
    - "Custom model handlers (flexible preprocessing/postprocessing)"
    - "Model versioning and A/B testing"
    - "Multi-model serving"
    - "TorchScript and eager mode support"
    - "Batch inference support"
    - "Management API (register, unregister, scale models)"
    - "Metrics export (Prometheus)"
    
  model_archive:
    what: ".mar file — packaged model with handler and dependencies"
    creation: |
      torch-model-archiver \
        --model-name text_classifier \
        --version 1.0 \
        --serialized-file model.pt \
        --handler custom_handler.py \
        --export-path model_store/
        
  custom_handler:
    code: |
      class CustomHandler(BaseHandler):
          def preprocess(self, data):
              # Transform raw request into model input
              text = data[0]["body"]["text"]
              tokens = self.tokenizer(text, return_tensors="pt")
              return tokens
              
          def inference(self, inputs):
              with torch.no_grad():
                  return self.model(**inputs)
                  
          def postprocess(self, outputs):
              probs = torch.softmax(outputs.logits, dim=-1)
              return [{"class": probs.argmax().item(), "confidence": probs.max().item()}]
              
  when_to_use:
    - "PyTorch-only shop"
    - "Need custom pre/postprocessing logic"
    - "Moderate scale (not thousands of models)"
    - "Familiar with PyTorch ecosystem"
    
  limitations:
    - "PyTorch only (no TensorFlow, ONNX)"
    - "Less optimized than Triton for multi-model scenarios"
    - "Not optimized for LLM autoregressive generation"
```

### KServe (Kubernetes-Native)

```yaml
KServe:
  what: "Kubernetes-native model serving platform (abstraction over model servers)"
  formerly: "KFServing"
  
  architecture:
    control_plane: "Manages model deployments, auto-scaling, traffic routing"
    data_plane: "Actual inference (pluggable: Triton, TorchServe, custom containers)"
    
  key_features:
    serverless: "Scale-to-zero when no traffic (Knative-based)"
    canary: "Traffic splitting between model versions (10% new, 90% old)"
    transformers: "Pre/post-processing containers in the inference pipeline"
    explainability: "Built-in model explanation (SHAP, LIME integration)"
    multi_model: "ModelMesh — efficient multi-model serving on shared resources"
    
  inference_service:
    yaml: |
      apiVersion: serving.kserve.io/v1beta1
      kind: InferenceService
      metadata:
        name: llm-service
      spec:
        predictor:
          model:
            modelFormat:
              name: huggingface
            storageUri: s3://models/llama-4-8b
            resources:
              limits:
                nvidia.com/gpu: 1
        transformer:
          containers:
            - name: tokenizer
              image: my-tokenizer:latest
              
  auto_scaling:
    metrics: "Concurrency, RPS (Requests Per Second), GPU utilization"
    scale_to_zero: "No pods running when no traffic (saves cost)"
    cold_start: "First request after scale-up takes longer (model loading)"
    mitigation: "Min replicas > 0 for critical services"
    
  when_to_use:
    - "Kubernetes-native infrastructure"
    - "Need scale-to-zero (cost savings for many models)"
    - "Multi-model serving with different frameworks"
    - "Need canary deployments and traffic splitting"
    - "Enterprise ML platform (many teams, many models)"
```

---

## How It Works in Practice

### Framework Selection Guide

```yaml
Decision_Framework:
  llm_serving:
    first_choice: "vLLM"
    reason: "Best throughput, PagedAttention, continuous batching, OpenAI-compatible"
    alternative: "TGI (if HuggingFace ecosystem preference)"
    alternative_2: "SGLang (for compiler-optimized workloads)"
    
  traditional_ml_single_model:
    first_choice: "TorchServe (PyTorch) or custom FastAPI (simple)"
    reason: "Simple setup, adequate performance for most use cases"
    scale_up: "Move to Triton when optimization needed"
    
  multi_model_multi_framework:
    first_choice: "Triton Inference Server"
    reason: "Supports all frameworks, excellent batching, model ensemble"
    
  kubernetes_platform:
    first_choice: "KServe (with Triton or vLLM as backend)"
    reason: "Scale-to-zero, canary, traffic splitting, multi-model"
    
  serverless_bursty:
    first_choice: "KServe (scale-to-zero) or cloud serverless (SageMaker Serverless)"
    reason: "Pay only when invoked, auto-scales"
    
  edge_deployment:
    first_choice: "ONNX Runtime, TFLite, or CoreML (see edge inference lesson)"
    reason: "Optimized for mobile/edge hardware"
```

---

## Interview Tip

> When asked about serving frameworks: "My default choice for LLM serving in 2026 is vLLM — it has the best throughput thanks to PagedAttention (2-4x better memory utilization than naive KV cache), continuous batching (add/remove requests dynamically), and an OpenAI-compatible API. For multi-model or multi-framework serving (PyTorch + TensorFlow + ONNX), I use NVIDIA Triton — it handles model ensemble pipelines, dynamic batching with configurable policies, and concurrent model execution. For Kubernetes-native ML platforms with many models and variable traffic, KServe provides scale-to-zero, canary deployments, and ModelMesh for efficient resource sharing. Key insight: the framework choice depends primarily on model type — LLM generation has fundamentally different requirements (KV cache management, continuous batching, speculative decoding) than traditional ML inference (simple forward pass, standard batching)."

---

## Common Mistakes

1. **Using Flask/FastAPI for production LLM serving** — Building a custom API that calls model.generate() synchronously. This processes one request at a time (no batching), doesn't manage KV cache efficiently, and achieves 5-10x lower throughput than vLLM. Use proper serving frameworks.

2. **Using Triton for LLM text generation** — While Triton supports LLMs via TensorRT-LLM backend, vLLM/TGI are significantly easier to deploy and often match or exceed performance for standard LLM serving. Triton's strength is multi-model and multi-framework, not LLM generation.

3. **Not implementing health checks** — Deploying model servers without /health endpoints. When a model crashes or GPU runs out of memory, the load balancer continues sending requests to dead instances. Always: liveness probe (is process alive?) + readiness probe (is model loaded and ready?).

4. **Over-engineering for simple use cases** — Setting up KServe + Istio + Knative for serving one model to 10 requests/minute. A simple Docker container with FastAPI or TorchServe is sufficient. Match complexity to requirements.

5. **Ignoring GPU memory fragmentation** — Loading/unloading models without considering memory fragmentation. After many model loads/unloads, GPU memory becomes fragmented and new models can't allocate contiguous blocks even though total free memory is sufficient. Restart periodically or use memory-managed frameworks.

---

## Key Takeaways

- vLLM: default for LLM serving — PagedAttention, continuous batching, OpenAI-compatible, best throughput
- TGI: HuggingFace's LLM server — simpler deployment, good for HF ecosystem
- Triton: general-purpose multi-framework server — PyTorch/TF/ONNX/TensorRT, multi-model, ensemble pipelines
- TorchServe: PyTorch-native with custom handlers — good for traditional ML models
- KServe: Kubernetes-native platform — scale-to-zero, canary, ModelMesh for many models
- LLM serving requires specialized frameworks (KV cache management, continuous batching) — don't use generic ML servers
- Always add health checks (liveness + readiness probes) and metrics export (Prometheus)
- Match framework complexity to requirements — don't over-engineer for simple use cases
- vLLM OpenAI compatibility means easy migration from OpenAI API to self-hosted
- Performance benchmarking: always benchmark YOUR model on YOUR hardware before committing to a framework
