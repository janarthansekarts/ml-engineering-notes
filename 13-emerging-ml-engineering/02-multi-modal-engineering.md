# Multi-Modal ML Engineering

## The Problem / Why This Matters

Real-world intelligence is inherently multi-modal — humans process text, images, audio, video, and spatial information simultaneously to understand context. Until recently, ML systems were built as separate pipelines: one model for text, another for images, a third for audio, stitched together with brittle hand-crafted logic. In 2026, multi-modal models (Gemini 2.5, GPT-5, Claude 4) process multiple modalities natively in a single model — understanding images alongside text, transcribing audio while analyzing video frames, and generating across modalities. For ML engineers, this creates new challenges: multi-modal data pipelines (processing text + images + audio at scale), efficient serving (multi-modal inputs are much larger than text-only), evaluation (how do you measure quality for image+text outputs?), and infrastructure (GPU memory requirements for multi-modal models are significantly higher). Multi-modal ML engineering encompasses: vision-language models (VLMs) for image understanding, audio-language models for speech processing, video understanding, document AI (OCR + understanding), and generative models (text-to-image, text-to-video). The engineering challenges are unique: multi-modal data is heterogeneous (different formats, sizes, and processing requirements), latency budgets are tighter (streaming video/audio), and evaluation is inherently subjective (image quality, video coherence).

---

## The Analogy

Think of multi-modal ML like a film production:

- **Single-modal (text-only)** = A radio play. Only audio — the listener must imagine everything. Works for many scenarios but limited in expression.
- **Multi-modal model** = A full film production. Combines dialogue (text), visuals (images/video), music (audio), and editing (reasoning across modalities). Richer understanding, more complex production.
- **The engineering challenge** = A film needs: cameramen (image encoders), sound engineers (audio encoders), editors (cross-attention between modalities), and a massive production budget (GPU memory for processing all modalities simultaneously).
- **Serving multi-modal** = Streaming a film vs. reading a book. Video streams require 1000× more bandwidth than text. Similarly, multi-modal model serving needs much more GPU memory and compute than text-only.

---

## Deep Dive

### Multi-Modal Architecture

```yaml
Multi_Modal_Architecture:
  unified_models:
    gemini_2_5:
      modalities: ["text", "image", "video", "audio", "code"]
      context: "2M tokens (including image/video/audio tokens)"
      approach: "Native multi-modal (single model, joint training)"
      capabilities:
        - "Understand images in context"
        - "Analyze video (temporal reasoning)"
        - "Process audio (speech, music, sounds)"
        - "Generate text from any combination of inputs"
        
    gpt_5:
      modalities: ["text", "image", "audio"]
      approach: "Multi-modal encoder + unified decoder"
      capabilities:
        - "Image understanding and description"
        - "Audio transcription and analysis"
        - "Interleaved text+image generation"
        
    claude_4:
      modalities: ["text", "image"]
      approach: "Vision encoder + language model"
      strengths: "Document understanding, chart/diagram analysis"
      
  specialized_models:
    vision_language:
      llava:
        what: "LLaVA (Large Language and Vision Assistant)"
        architecture: "CLIP vision encoder + Llama language model"
        sizes: "7B, 13B, 34B"
        open_weight: True
        
      internvl:
        what: "InternVL — open-source multi-modal model"
        strengths: "Strong OCR, document understanding"
        
    audio_language:
      whisper_v3:
        what: "OpenAI speech-to-text (open-source)"
        languages: "99+ languages"
        modes: "Transcription, translation, timestamps"
        
      seamless:
        what: "Meta's speech translation (direct speech-to-speech)"
        capability: "Real-time translation without text intermediate"
        
    video_understanding:
      video_llava:
        what: "Process video as sequence of frames + audio"
        approach: "Sample frames → vision encoder → language model"
        
      twelve_labs:
        what: "Video understanding API"
        capabilities: "Search within video, summarize, extract moments"
        
    generation:
      stable_diffusion_3:
        what: "Text-to-image generation"
        architecture: "Diffusion transformer (DiT)"
        quality: "Photorealistic, text rendering improved"
        
      sora_veo:
        what: "Text-to-video generation (OpenAI Sora, Google Veo)"
        capability: "Generate 60+ second coherent videos"
        
      musicgen:
        what: "Text-to-music generation (Meta)"
        capability: "Generate music from text descriptions"
```

### Multi-Modal Data Engineering

```yaml
Multi_Modal_Data_Engineering:
  data_formats:
    images:
      raw: "JPEG, PNG, WebP"
      preprocessed: "Resized to model input resolution (224×224, 336×336, 768×768)"
      tokens: "~256-1024 tokens per image (after vision encoder)"
      storage: "WebDataset (tar archives for efficient streaming)"
      
    video:
      raw: "MP4, WebM (H.264/H.265 codec)"
      preprocessed: "Sampled frames (1-4 fps typically)"
      tokens: "256-1024 tokens per frame × frames"
      storage: "Chunked storage (30-60 second clips)"
      challenge: "1 minute of video at 2fps = 120 frames × 256 tokens = 30K tokens"
      
    audio:
      raw: "WAV, MP3, FLAC"
      preprocessed: "Mel spectrograms, chunked into 30-second segments"
      tokens: "~1500 tokens per 30 seconds (Whisper-style encoding)"
      storage: "Pre-processed spectrograms or raw with on-the-fly processing"
      
    documents:
      raw: "PDF, DOCX, images of documents"
      preprocessed: "OCR → structured text + layout information"
      multi_modal_approach: "Image of document page + extracted text"
      
  processing_pipelines:
    image_pipeline:
      steps:
        - "Download/load from storage"
        - "Validate format and dimensions"
        - "Resize to model input size (maintain aspect ratio, pad)"
        - "Normalize pixel values (ImageNet mean/std or model-specific)"
        - "Convert to tensor (batch processing)"
      optimization:
        - "DALI (NVIDIA data loading library) for GPU-accelerated processing"
        - "torchvision transforms with multiprocessing"
        - "Pre-resize and cache common sizes"
        
    video_pipeline:
      steps:
        - "Decode video (ffmpeg or decord)"
        - "Sample frames (uniform or keyframe-based)"
        - "Process each frame as image"
        - "Optional: extract audio track separately"
        - "Batch frames as sequence input"
      optimization:
        - "Decord library (GPU-accelerated video decoding)"
        - "Sample frames lazily (don't decode entire video)"
        - "Cache decoded frames for repeated access"
        
    audio_pipeline:
      steps:
        - "Load audio (librosa or torchaudio)"
        - "Resample to model's expected rate (16kHz typical)"
        - "Compute mel spectrogram (or raw waveform for some models)"
        - "Chunk into fixed-length segments (30 seconds)"
        - "Normalize"
      optimization:
        - "torchaudio for GPU-accelerated processing"
        - "Stream processing (don't load entire file into memory)"
```

### Implementation Patterns

```python
# Multi-modal ML engineering patterns

"""
Patterns for building multi-modal ML systems: data processing,
model serving, and evaluation.
"""

multi_modal_patterns = {
    "vision_language_serving": {
        "description": "Serve vision-language model for image understanding",
        "architecture": {
            "input_processing": {
                "text": "Tokenize with model's tokenizer",
                "image": "Resize, normalize, convert to tensor",
                "combined": "Interleave text and image tokens",
            },
            "model": "LLaVA-34B or InternVL-40B",
            "hardware": "2× H100 (model needs ~70GB for weights + KV cache)",
            "serving": "vLLM with vision model support",
            "throughput": "~5-10 images/second (including generation)",
        },
        "optimization": {
            "image_preprocessing": "Resize to smallest acceptable resolution",
            "quantization": "INT4 for model weights (fit on fewer GPUs)",
            "batching": "Batch requests with similar image sizes",
            "caching": "Cache image embeddings for repeated images",
        },
    },
    
    "document_ai_pipeline": {
        "description": "Extract and understand information from documents",
        "pipeline": [
            {
                "step": "Document ingestion",
                "input": "PDF, scanned images, DOCX",
                "output": "Page images + raw text (if available)",
            },
            {
                "step": "OCR + layout analysis",
                "tools": ["Azure Document Intelligence", "Google Document AI", "PaddleOCR"],
                "output": "Structured text with bounding boxes and layout info",
            },
            {
                "step": "Multi-modal understanding",
                "input": "Page image + OCR text (both provided to model)",
                "model": "Gemini 2.5 Pro or specialized document model",
                "tasks": ["Key-value extraction", "Table parsing", "Classification"],
            },
            {
                "step": "Post-processing",
                "output": "Structured JSON with extracted fields",
                "validation": "Schema validation, confidence thresholds",
            },
        ],
        "accuracy": "95%+ for structured documents, 85%+ for handwritten/messy",
    },
    
    "video_understanding": {
        "description": "Analyze video content for search, summarization, QA",
        "approaches": {
            "frame_sampling": {
                "method": "Sample frames at 1-2 fps, process as image sequence",
                "model": "Vision-language model (Gemini, GPT-4o)",
                "limitation": "Misses fast actions, audio context lost",
                "cost": "High token usage (30 sec video = 60 frames × 256 tokens = 15K tokens)",
            },
            "native_video": {
                "method": "Model processes video natively (temporal attention)",
                "model": "Gemini 2.5 Pro (native video support)",
                "advantage": "Temporal understanding, motion detection",
                "limitation": "Very expensive (1 min video = 50K+ tokens)",
            },
            "hybrid": {
                "method": "Keyframe detection + dense sampling around events",
                "advantage": "Efficient (skip boring parts, focus on action)",
                "implementation": "Scene change detection → sample nearby frames",
            },
        },
        "serving_considerations": {
            "latency": "Video processing is slow — async/batch preferred",
            "cost": "10-100× more expensive than text per query",
            "streaming": "For real-time video: process in sliding windows",
        },
    },
    
    "multi_modal_rag": {
        "description": "RAG that handles text, images, tables, and charts",
        "architecture": {
            "indexing": {
                "text_chunks": "Standard text chunking + embedding",
                "images": "CLIP embedding for image content",
                "tables": "Serialize to markdown + embed as text",
                "charts": "Vision model describes chart → embed description",
            },
            "retrieval": {
                "query_embedding": "Multi-modal query encoder",
                "search": "Hybrid: text similarity + image similarity",
                "reranking": "Cross-modal reranker (scores text-image relevance)",
            },
            "generation": {
                "context": "Text chunks + relevant images/tables",
                "model": "Multi-modal model (Gemini, GPT-4o)",
                "output": "Text response with references to figures/tables",
            },
        },
        "challenge": "Embedding images and text in same space for unified search",
        "solutions": [
            "CLIP (shared image-text embedding space)",
            "ColPali (document retrieval using page images directly)",
            "Separate indices with fusion at retrieval time",
        ],
    },
}


# Multi-modal evaluation
multi_modal_evaluation = {
    "image_understanding": {
        "benchmarks": [
            "MMMU (multi-discipline reasoning with images)",
            "ChartQA (chart/graph understanding)",
            "DocVQA (document question answering)",
            "TextVQA (reading text in images)",
        ],
        "automated_metrics": {
            "accuracy": "Exact match on extraction/classification tasks",
            "description_quality": "LLM-as-judge scoring image descriptions",
            "hallucination_rate": "% of claims not supported by image content",
        },
    },
    
    "generation_quality": {
        "image_generation": {
            "metrics": ["FID (Frechet Inception Distance)", "CLIP score", "Human preference"],
            "challenges": "Automated metrics don't capture aesthetic quality",
        },
        "video_generation": {
            "metrics": ["Temporal consistency", "Motion quality", "Text alignment"],
            "challenges": "No widely accepted automated metric — human eval required",
        },
    },
    
    "practical_evaluation": {
        "a_b_testing": "Show two model outputs to human evaluators, track win rate",
        "task_completion": "Does the output enable the user to complete their task?",
        "error_analysis": "Categorize failures (hallucination, wrong modality focus, format error)",
    },
}
```

---

## How It Works in Practice

### Multi-Modal Production System

```yaml
Multi_Modal_Production:
  scenario: "E-commerce product understanding (image + text + video)"
  
  system:
    product_cataloging:
      input: "Product images + seller descriptions + video demos"
      model: "Gemini 2.5 Flash (multi-modal, cost-effective)"
      output:
        - "Structured product attributes (color, size, material)"
        - "Quality assessment (professional vs. amateur photos)"
        - "Category classification from image"
        - "Video highlight extraction (key product features)"
      volume: "100K products/day (batch processing)"
      
    visual_search:
      input: "User uploads photo → find similar products"
      model: "CLIP embedding (image → vector)"
      database: "Pinecone vector DB (50M product embeddings)"
      latency: "< 100ms for similarity search"
      
    content_moderation:
      input: "Product images + descriptions"
      model: "Multi-modal safety classifier"
      checks:
        - "Prohibited items detection (weapons, drugs)"
        - "Copyright/trademark infringement"
        - "Misleading images (stock photos vs. actual product)"
      volume: "500K images/day (real-time as products are listed)"
      
  infrastructure:
    batch_processing:
      hardware: "8× L4 GPUs (cost-effective for inference)"
      engine: "vLLM with multi-modal support"
      throughput: "~50 products/minute per GPU"
      cost: "$0.84/hr × 8 = $6.72/hr → processes 100K products in ~4 hours = $27/day"
      
    real_time_search:
      hardware: "CPU (CLIP inference) + vector DB"
      cost: "$2K/month (Pinecone serverless for 50M vectors)"
      
    moderation:
      hardware: "4× L4 GPUs (lightweight classifier)"
      latency: "< 500ms per image"
      cost: "$3.36/hr continuous = $2,420/month"
```

---

## Interview Tip

> When asked about multi-modal ML engineering: "Multi-modal systems have unique engineering challenges beyond single-modality ML. Data pipeline: images are 100-1000× larger than text tokens — efficient loading with DALI or torchvision, caching pre-processed tensors on NVMe, and format optimization (WebP over PNG, frame sampling for video). Serving: a single image adds 256-1024 tokens to context — GPU memory for KV cache grows significantly. I use quantization (INT4) and batching (group similar-size inputs) to maximize throughput. Architecture patterns: for document AI, I combine OCR + page image in a multi-modal model (both text and visual layout information). For video, I use keyframe sampling (1-2 fps) rather than processing every frame — 10× cheaper with minimal quality loss. For multi-modal RAG: CLIP embeddings for images in same vector space as text, or ColPali for direct page-image retrieval. Evaluation is harder: no single automated metric captures multi-modal quality — I use LLM-as-judge for descriptions, task completion rate for extraction, and human A/B tests for generation quality. Cost consideration: multi-modal inference is 5-50× more expensive than text-only (more tokens per query). I use Gemini Flash or GPT-4o-mini for high-volume processing and reserve expensive models for complex cases."

---

## Common Mistakes

1. **Processing full-resolution images** — Sending 4K images (4096×4096) to vision models when the model internally resizes to 336×336. Wasting bandwidth and processing time. Solution: resize to model's native resolution before sending. Most VLMs use 224-768px input.

2. **Processing every video frame** — Extracting all 30fps frames from a 5-minute video (9000 frames, 2M+ tokens). Costs $10+ per video query. Solution: sample at 1-2 fps (300-600 frames for 5 min) or use keyframe detection. Quality difference is minimal for most tasks.

3. **No multi-modal caching** — Processing the same product image every time a different question is asked about it. Solution: cache image embeddings/tokens. Once processed, the visual representation is reusable across queries.

4. **Text-only RAG for documents** — Using OCR text for document retrieval, losing layout information (tables, figures, spatial relationships). Solution: multi-modal RAG — index page images alongside text, retrieve both, and provide to multi-modal model.

5. **Using expensive model for simple visual tasks** — Running GPT-4o ($10/1M output tokens) for basic image classification that CLIP or a fine-tuned ViT can do. Solution: match model to task complexity. Image classification → CLIP/ViT ($0.001/image). Complex visual reasoning → GPT-4o/Gemini.

---

## Key Takeaways

- Multi-modal models: single models processing text + image + video + audio natively (Gemini, GPT-5)
- Engineering challenges: data heterogeneity, larger inputs, higher serving costs, harder evaluation
- Image tokens: 256-1024 tokens per image — significantly increases context and cost
- Video processing: sample frames (1-2 fps), don't process every frame — 10× cheaper
- Document AI: OCR + page image together (layout information matters)
- Multi-modal RAG: CLIP embeddings for images + text in shared space, or ColPali
- Serving optimization: quantize (INT4), batch similar-size inputs, cache embeddings
- Cost: multi-modal inference 5-50× more expensive than text-only per query
- Evaluation: harder — use LLM-as-judge, task completion rate, human A/B tests
- Vision-language models (VLMs): LLaVA, InternVL (open-source), Gemini/GPT-5 (frontier)
