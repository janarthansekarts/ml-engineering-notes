# Serverless Inference

## The Problem / Why This Matters

Traditional model serving requires provisioning dedicated GPU instances 24/7, even when traffic is sporadic. An internal ML service that handles 50 requests during business hours and zero requests overnight still pays for GPU time around the clock — at $2-4/hour for an H100, that's $1,440-2,880/month for a service that's idle 66% of the time. Serverless inference solves this by provisioning compute only when a request arrives, billing per-invocation (or per-second of active compute), and automatically scaling to zero when idle. The challenge: GPU serverless has historically suffered from cold start problems — loading a multi-gigabyte model from storage into GPU memory takes 30-120 seconds, creating unacceptable latency for the first request after idle. In 2026, this has been largely solved through container pre-warming, model caching on local SSDs, snapshot-and-restore techniques, and purpose-built serverless GPU platforms (Modal, Replicate, RunPod Serverless, AWS SageMaker Serverless, Google Cloud Run GPU). Serverless inference is now production-viable for a wide range of ML workloads — from traditional models on CPU/Lambda to LLMs on serverless GPUs. The key trade-off: cost efficiency for variable traffic vs latency predictability for steady traffic.

---

## The Analogy

Think of serverless inference like different car ownership models:

- **Dedicated GPU instance** = Owning a car. You pay insurance, parking, depreciation 24/7 whether you drive or not. Makes sense if you drive daily for hours. Terrible economics for occasional weekend trips.
- **Serverless inference** = Uber/Lyft. Pay only when you ride. No parking costs, no insurance, no idle depreciation. But: first ride takes 3-5 minutes to arrive (cold start), and surge pricing during peak hours (concurrency limits).
- **Pre-warmed serverless** = Premium Uber (car waiting nearby). Slightly more expensive per ride, but car arrives in 30 seconds instead of 5 minutes. The platform keeps cars strategically positioned (pre-warmed containers with loaded models).
- **Spot/preemptible GPU** = Car rental with early return risk. Cheap daily rate, but the rental company might recall the car mid-trip if someone pays full price. Good for batch processing where interruption is acceptable.

---

## Deep Dive

### Serverless Inference Platforms

```yaml
Serverless_Platforms:
  modal:
    what: "Python-native serverless GPU platform"
    cold_start: "3-10 seconds (fast container snapshots, model pre-loading)"
    gpus: "A100, H100, A10G, L4, T4"
    pricing: "Per-second billing, only pay during active inference"
    killer_feature: "Python-native API — deploy with decorators, no Docker needed"
    
    example: |
      import modal
      
      app = modal.App("llm-serving")
      
      # Define GPU-accelerated function
      @app.cls(
          gpu=modal.gpu.H100(count=1),
          image=modal.Image.debian_slim().pip_install("vllm"),
          container_idle_timeout=300,  # Keep warm for 5 min after last request
          allow_concurrent_inputs=100,
      )
      class LLMService:
          @modal.enter()  # Runs once when container starts
          def load_model(self):
              from vllm import LLM
              self.llm = LLM(
                  model="meta-llama/Llama-4-8B-AWQ",
                  quantization="awq",
                  gpu_memory_utilization=0.9,
              )
              
          @modal.method()
          def generate(self, prompt: str, max_tokens: int = 256) -> str:
              from vllm import SamplingParams
              outputs = self.llm.generate(
                  [prompt],
                  SamplingParams(temperature=0.7, max_tokens=max_tokens)
              )
              return outputs[0].outputs[0].text
              
      # Deploy: modal deploy llm_service.py
      # Call: result = LLMService().generate.remote("Hello")
      
    features:
      - "Container snapshots (checkpoint model in memory → instant restore)"
      - "Volume mounts (persistent model storage — no re-download)"
      - "Concurrent inputs (multiple requests per container)"
      - "Web endpoint generation (automatic REST API)"
      - "Scheduled tasks (cron-based batch processing)"
      
  replicate:
    what: "Model hosting platform with serverless GPU inference"
    cold_start: "5-30 seconds (depends on model size)"
    interface: "Push model → get API endpoint → call via HTTP"
    pricing: "Per-second of GPU time"
    models: "Thousands of pre-deployed community models"
    
    deployment: |
      # Define model with Cog (Replicate's model packaging format)
      # cog.yaml
      build:
        gpu: true
        python_version: "3.11"
        python_packages:
          - "torch==2.4"
          - "transformers"
          
      # predict.py
      from cog import BasePredictor, Input
      import torch
      from transformers import AutoModelForCausalLM, AutoTokenizer
      
      class Predictor(BasePredictor):
          def setup(self):
              """Load model into memory (runs once on cold start)."""
              self.tokenizer = AutoTokenizer.from_pretrained("model/")
              self.model = AutoModelForCausalLM.from_pretrained("model/", torch_dtype=torch.float16)
              self.model.cuda()
              
          def predict(self, prompt: str = Input(description="Input prompt")) -> str:
              inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
              outputs = self.model.generate(**inputs, max_new_tokens=256)
              return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
              
  runpod_serverless:
    what: "GPU worker pools that scale to zero"
    cold_start: "10-30 seconds (worker initialization)"
    gpus: "A100, H100, A40, RTX 4090"
    pricing: "Per-second billing, flexible worker configuration"
    
    architecture: |
      Request → RunPod Queue → Available Worker → Response
      - Workers scale 0 → N based on queue depth
      - Workers stay warm for configurable idle time
      - Supports async (webhook) and sync (blocking) APIs
      
    config:
      min_workers: 0  # Scale to zero when idle
      max_workers: 10
      idle_timeout: 300  # Keep worker warm for 5 minutes
      
  google_cloud_run_gpu:
    what: "Google Cloud Run with GPU support (2025+)"
    cold_start: "15-60 seconds (container + model loading)"
    gpus: "L4, A100"
    pricing: "Per-request + per-second of GPU time"
    advantage: "Native GCP integration (IAM, VPC, Cloud Storage)"
    
  aws_sagemaker_serverless:
    what: "AWS managed serverless inference"
    limitation: "6 GB memory cap — only small models (no LLMs)"
    cold_start: "30-60 seconds"
    use: "Small models (BERT, embeddings) on AWS"
    
  aws_lambda_inference:
    what: "AWS Lambda for ML (CPU only, 10 GB memory limit)"
    cold_start: "1-5 seconds (with provisioned concurrency: 0)"
    use: "Lightweight models, preprocessing, routing logic"
    not_for: "GPU models, large models, LLMs"
```

### Cold Start Optimization

```yaml
Cold_Start:
  problem: |
    Serverless cold start = container start + model loading
    - Container start: 5-15 seconds (pull image, initialize runtime)
    - Model loading: 10-120 seconds (download weights, load to GPU)
    - Total: 15-135 seconds — unacceptable for interactive use
    
  optimization_techniques:
    container_snapshots:
      what: "Checkpoint entire container state (including loaded model in GPU memory)"
      how: "CRIU (Checkpoint/Restore In Userspace) or platform-specific snapshotting"
      cold_start: "3-10 seconds (restore from snapshot, not full load)"
      platforms: "Modal (best implementation), AWS SnapStart (for Lambda/Java)"
      
    model_caching_on_disk:
      what: "Cache model files on local SSD attached to serverless worker"
      benefit: "Skip network download (S3/GCS) on subsequent cold starts"
      cold_start_reduction: "60-120s → 10-30s (SSD load vs network download)"
      implementation: "Modal volumes, RunPod network storage, persistent disks"
      
    pre_warming:
      what: "Keep minimum containers warm (don't fully scale to zero)"
      config: "min_workers: 1 (or container_idle_timeout: 600)"
      trade_off: "Pay for 1 idle container but guarantee zero cold start for initial requests"
      
    smaller_models:
      what: "Use aggressively quantized models (INT4, 2-3× smaller)"
      benefit: "Faster loading (less data to read from disk/network)"
      example: "70B FP16 (140 GB): 120s load. 70B INT4 (35 GB): 30s load."
      
    lazy_loading:
      what: "Load model layers on-demand (serve with partial model while rest loads)"
      limitation: "Complex, limited to specific architectures"
      
    image_pre_building:
      what: "Bake model weights into Docker image (download during build, not runtime)"
      benefit: "Container pull includes model (cached on node)"
      limitation: "Very large images (tens of GB) — slow to pull initially"
```

### Cost Comparison

```yaml
Cost_Analysis:
  scenario_1_low_traffic:
    description: "Internal tool, 100 requests/day, average 2 seconds GPU time per request"
    
    dedicated_gpu:
      instance: "1× H100 ($3/hour)"
      monthly: "$3 × 24 × 30 = $2,160/month"
      gpu_time_used: "100 requests × 2s = 200 seconds = 3.3 minutes/day"
      utilization: "0.23% (99.77% paying for idle GPU)"
      
    serverless:
      gpu_seconds: "100 × 2 = 200 seconds/day × 30 = 6,000 seconds/month"
      price_per_second: "$0.001 (H100, typical serverless rate)"
      monthly: "$6/month"
      savings: "360× cheaper than dedicated"
      
  scenario_2_moderate_traffic:
    description: "Customer API, 10,000 requests/day, 2s avg, with peaks"
    
    dedicated_gpu:
      instance: "2× H100 (for redundancy + peak handling)"
      monthly: "$3 × 2 × 24 × 30 = $4,320/month"
      
    serverless:
      gpu_seconds: "10,000 × 2 × 30 = 600,000 seconds/month"
      price_per_second: "$0.001"
      monthly: "$600/month"
      savings: "7.2× cheaper (traffic is still bursty with idle periods)"
      
  scenario_3_high_traffic:
    description: "Production LLM, 100,000 requests/day, sustained 24/7"
    
    dedicated_gpu:
      instance: "4× H100 (sustained load)"
      monthly: "$3 × 4 × 24 × 30 = $8,640/month"
      
    serverless:
      gpu_seconds: "100,000 × 2 × 30 = 6,000,000 seconds/month"
      price_per_second: "$0.001"
      monthly: "$6,000/month"
      comparison: "30% cheaper than dedicated — but less so as utilization increases"
      note: "At this volume, dedicated becomes competitive (reserved instances even cheaper)"
      
  crossover_point:
    rule: |
      Serverless is cheaper when: GPU utilization < 30-40%
      Dedicated is cheaper when: GPU utilization > 40-50%
      Sweet spot for serverless: bursty traffic, business-hours-only, or < 1000 requests/hour
      
  hidden_costs:
    serverless: "Cold start latency, concurrency limits, less control over optimization"
    dedicated: "Ops overhead, idle time, capacity planning complexity"
```

### Serverless for Different Model Types

```yaml
Model_Type_Guide:
  traditional_ml_cpu:
    models: "XGBoost, scikit-learn, small PyTorch (<100 MB)"
    platform: "AWS Lambda, Google Cloud Functions, Azure Functions"
    cold_start: "1-5 seconds"
    cost: "< $0.001 per request"
    max_memory: "10 GB (Lambda), 32 GB (Cloud Run)"
    implementation: |
      # AWS Lambda with ML model
      import json
      import pickle
      import numpy as np
      
      # Load model at module level (persists across invocations)
      with open("model.pkl", "rb") as f:
          model = pickle.load(f)
      
      def handler(event, context):
          features = np.array(json.loads(event["body"])["features"])
          prediction = model.predict(features.reshape(1, -1))
          return {
              "statusCode": 200,
              "body": json.dumps({"prediction": prediction[0].tolist()})
          }
          
  embedding_models:
    models: "BERT, sentence-transformers, E5, BGE (100MB-1GB)"
    platform: "CPU serverless (Lambda/Cloud Run) or GPU serverless for high throughput"
    recommendation: |
      < 100 requests/minute: CPU serverless (Lambda + ONNX Runtime)
      > 100 requests/minute: GPU serverless or dedicated GPU
    optimization: "ONNX Runtime + INT8 quantization → fast CPU inference"
    
  small_llms:
    models: "Phi-3-mini (3.8B), Gemma-2B, TinyLlama (1.1B)"
    platform: "GPU serverless (Modal, Replicate, RunPod)"
    gpu: "L4 or A10G (24 GB sufficient for INT4 small LLMs)"
    cold_start: "10-20 seconds"
    
  large_llms:
    models: "Llama-70B, Mixtral 8×7B"
    platform: "Modal (best cold start), RunPod Serverless"
    gpu: "H100 (80 GB) or 2× A100"
    cold_start: "30-120 seconds (model loading dominates)"
    mitigation: "Keep-warm with min_workers=1 for latency-sensitive"
    
  image_generation:
    models: "Stable Diffusion XL, FLUX"
    platform: "Replicate (popular), Modal, RunPod"
    gpu: "A10G or L4 (SDXL fits in 8 GB VRAM quantized)"
    cold_start: "15-30 seconds"
```

### Hybrid Architecture

```yaml
Hybrid_Serverless_Dedicated:
  pattern: |
    Use DEDICATED for: baseline steady traffic (predictable, high utilization)
    Use SERVERLESS for: burst overflow (unpredictable spikes above baseline)
    
  implementation:
    baseline:
      what: "2 dedicated H100 instances handling 80% of average traffic"
      always_on: true
      cost: "Fixed $4,320/month"
      
    overflow:
      what: "Serverless GPU workers that activate when dedicated instances are full"
      trigger: "Queue depth > 10 OR all dedicated instances at > 80% KV utilization"
      scale: "0 → 10 workers in 10-30 seconds (pre-warmed)"
      cost: "Variable, only during spikes"
      
    routing:
      logic: |
        if dedicated_instances.available_capacity() > 0:
            route_to_dedicated(request)
        else:
            route_to_serverless(request)  # Overflow to serverless
            
  benefits:
    - "Dedicated handles predictable baseline (cost-efficient at high utilization)"
    - "Serverless handles unpredictable spikes (no over-provisioning)"
    - "No dropped requests during traffic surges"
    - "Cost-optimal: fixed cost for baseline + variable cost for burst only"
```

### Function Composition for ML Pipelines

```python
# Serverless ML pipeline using Modal
import modal

app = modal.App("ml-pipeline")

# Each step is a separate serverless function with appropriate resources
@app.function(cpu=2, memory=4096)
def preprocess(text: str) -> dict:
    """CPU-only preprocessing (tokenization, cleaning)."""
    # Lightweight — runs on CPU, scales independently
    tokens = tokenize(text)
    return {"tokens": tokens, "metadata": extract_metadata(text)}

@app.function(gpu=modal.gpu.L4())
def embed(tokens: dict) -> list[float]:
    """GPU embedding — L4 sufficient for embedding model."""
    # Embedding model loaded in @enter (persists across calls)
    return embedding_model.encode(tokens)

@app.function(gpu=modal.gpu.H100())
def generate(prompt: str, context: list[float]) -> str:
    """GPU generation — H100 for large LLM."""
    # LLM loaded via vLLM in @enter
    return llm.generate(prompt, context=context)

@app.function(cpu=1)
def postprocess(raw_output: str) -> dict:
    """CPU postprocessing (formatting, safety check)."""
    return {"response": format_output(raw_output), "safe": safety_check(raw_output)}

# Pipeline orchestration
@app.function()
def pipeline(user_query: str) -> dict:
    """Orchestrate the full pipeline."""
    # Each step runs on optimal hardware, scales independently
    preprocessed = preprocess.remote(user_query)
    embedding = embed.remote(preprocessed)
    raw_response = generate.remote(user_query, embedding)
    final = postprocess.remote(raw_response)
    return final
```

---

## How It Works in Practice

### When to Use Serverless vs Dedicated

```yaml
Decision_Framework:
  use_serverless_when:
    - "Traffic is bursty (10x variation between peak and trough)"
    - "Service is idle > 60% of the time"
    - "Budget is limited (can't afford 24/7 GPU)"
    - "Many models with low individual traffic (long-tail models)"
    - "Development/staging environments (occasional testing)"
    - "Batch processing with unpredictable scheduling"
    
  use_dedicated_when:
    - "Consistent high traffic (GPU utilization > 50%)"
    - "Strict latency SLA (cannot tolerate cold starts)"
    - "Cost optimization at scale (reserved instances cheaper than serverless at high utilization)"
    - "Custom hardware requirements (multi-GPU tensor parallelism)"
    - "Complex serving configuration (custom batching, model ensemble)"
    
  use_hybrid_when:
    - "Predictable baseline + unpredictable spikes"
    - "Can't tolerate dropped requests but don't want to over-provision"
    - "Multiple workloads with different traffic patterns"
```

---

## Interview Tip

> When asked about serverless inference: "Serverless GPU is now production-viable in 2026 — platforms like Modal achieve 3-10 second cold starts through container snapshotting and model caching. My decision framework: serverless when GPU utilization would be <30-40% (bursty traffic, business-hours-only services, long-tail models); dedicated when utilization >50% (steady high traffic, strict latency SLA). The crossover: at 100 requests/day with 2s GPU time each, serverless costs $6/month vs $2,160 for a dedicated H100 — 360× cheaper. At 100K requests/day sustained, dedicated becomes competitive. Best pattern: hybrid architecture with dedicated instances handling baseline traffic (high utilization, cost-efficient) and serverless overflow for unpredictable spikes (elastic, no over-provisioning). Cold start mitigation: container snapshots (Modal), pre-warmed workers (min_workers=1), model caching on local SSD (skip network download), and aggressive quantization (35 GB INT4 loads 4× faster than 140 GB FP16). For CPU-only inference (embeddings, small classifiers): AWS Lambda + ONNX Runtime — $0.001/request, 1-3s cold start with provisioned concurrency."

---

## Common Mistakes

1. **Using serverless for steady high-traffic workloads** — Running 100K requests/day on serverless GPUs when utilization would be >70% on dedicated. At high utilization, dedicated instances (especially reserved/committed-use) are 30-50% cheaper. Serverless premium is worth it only when you're avoiding paying for idle time.

2. **Not setting container_idle_timeout** — Container scales to zero after every request (default on some platforms). Next request pays full cold start. Setting idle_timeout=300 (5 minutes) keeps the container warm between requests — eliminates cold start for traffic with gaps < 5 minutes. Small cost for massive latency improvement.

3. **Downloading model from S3/HuggingFace on every cold start** — Each cold start downloads 35 GB from the internet (2-5 minutes). Use platform-native model storage (Modal volumes, persistent disks) that cache models on local SSD. Cold start becomes 10-30s (disk read) instead of 2-5 min (network download).

4. **Not implementing request queuing/buffering** — During cold start, requests pile up and timeout. Implement a queue (SQS, Pub/Sub, or platform-native queue) that buffers requests until a worker is ready. Return async response with callback/webhook instead of making users wait for cold start.

5. **Ignoring concurrency settings** — Running one request per container when the GPU could handle 50 concurrent requests (via vLLM continuous batching). Configure allow_concurrent_inputs appropriately — it dramatically improves throughput per container and reduces cost per request.

---

## Key Takeaways

- Serverless inference: pay only during active computation (no idle GPU cost)
- Cost-effective when GPU utilization would be <30-40% on dedicated (bursty, low-traffic, business-hours-only)
- Cold start mitigation: container snapshots (3-10s), model caching on local SSD, pre-warming, quantization
- Platforms: Modal (best Python DX, fast cold start), Replicate (model marketplace), RunPod (flexible GPUs)
- AWS Lambda: CPU-only, <10 GB models, excellent for preprocessing/routing/small ML
- Hybrid architecture: dedicated baseline + serverless overflow = cost-optimal for variable traffic
- Configure idle_timeout (keep warm between requests) and concurrency (batch inside container)
- At high sustained traffic, dedicated instances become cheaper — serverless premium is for elasticity
- Function composition: different pipeline stages on different hardware (CPU preprocessing → GPU inference)
- Always calculate: (monthly_requests × gpu_seconds_per_request × $/second) vs (dedicated_instance × $/month)
