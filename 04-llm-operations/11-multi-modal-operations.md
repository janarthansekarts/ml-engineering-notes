# Multi-Modal Operations

## The Problem / Why This Matters

LLMs are no longer text-only. In 2026, frontier models natively understand and generate multiple modalities: text, images, audio, video, and code. GPT-5 can analyze medical images, Claude 4 can describe charts in detail, Gemini 2.5 can process 1-hour videos, and specialized models generate photorealistic images (DALL-E 4, Midjourney v7) and natural speech (ElevenLabs, OpenAI TTS). This creates enormous engineering challenges: how do you build production systems that handle image+text inputs? How do you process video at scale (a 1-hour video is millions of tokens)? How do you orchestrate pipelines that combine vision, language, and audio? Multi-modal operations is the engineering discipline of building reliable, scalable, cost-effective systems that process and generate content across multiple modalities. The complexity isn't in calling an API — it's in: token economics (images cost 1000-10,000× more tokens than equivalent text), latency management (video processing takes minutes not milliseconds), quality assurance (how do you evaluate if an image description is correct?), pipeline orchestration (text→image→text→speech chains), and infrastructure (GPU memory, storage, preprocessing). Companies that master multi-modal operations build richer, more capable AI products — document understanding, visual search, video analysis, accessible content generation, and creative tools that were impossible with text-only models.

---

## The Analogy

Think of multi-modal operations like a multimedia production studio:

- **Text-only LLM** = A writer. Can create scripts, stories, articles — but only words. Needs to describe a sunset in paragraphs because it can't show you.
- **Multi-modal LLM** = A full production crew: writer + photographer + videographer + audio engineer + editor. Can create, understand, and combine any format. But coordinating the crew is complex.
- **Vision input** = The photographer reviewing existing images ("what's in this photo?"). Uses specialized equipment (GPU memory for image processing), takes longer than reading text.
- **Image generation** = The art department creating visuals from descriptions. High quality takes time and resources. Different artists (DALL-E, Midjourney, Stable Diffusion) have different styles and capabilities.
- **Pipeline orchestration** = The director coordinating: "First, transcribe this audio → then analyze the transcript → then generate a summary image → then add voiceover." Each step depends on the previous.

---

## Deep Dive

### Multi-Modal Capabilities (2026)

```yaml
Multi_Modal_Landscape:
  vision_understanding:
    what: "Models that can see and reason about images"
    models:
      gpt_5: "Native vision, strong at OCR, charts, diagrams, photos"
      claude_4: "Excellent image understanding, strong at documents and UI"
      gemini_2_5: "Best multi-image reasoning, video understanding"
    use_cases:
      - "Document OCR (Optical Character Recognition) and extraction"
      - "Chart/graph analysis and data extraction"
      - "UI screenshot understanding (accessibility, testing)"
      - "Medical image analysis (radiology, dermatology)"
      - "Product image categorization"
      - "Visual search and similarity"
      
  video_understanding:
    what: "Models that process video content"
    models:
      gemini_2_5_pro: "Native video (up to 1 hour), frame-by-frame analysis"
      gpt_5: "Video through frame sampling"
      twelve_labs: "Specialized video understanding API"
    use_cases:
      - "Video content moderation"
      - "Meeting recording analysis and summarization"
      - "Surveillance and security monitoring"
      - "Sports analytics"
      - "Educational video indexing"
      
  audio_understanding:
    what: "Models that process speech and audio"
    models:
      whisper_v4: "OpenAI's speech-to-text (99%+ accuracy)"
      gemini_2_5: "Native audio understanding"
      assembly_ai: "Transcription + speaker diarization + summarization"
    use_cases:
      - "Call center transcription and analysis"
      - "Podcast/meeting summarization"
      - "Voice command processing"
      - "Audio content moderation"
      
  image_generation:
    what: "Models that create images from text descriptions"
    models:
      dall_e_4: "OpenAI's latest (photorealistic, fast)"
      midjourney_v7: "Best aesthetic quality"
      stable_diffusion_4: "Open-source, self-hostable, customizable"
      flux: "High quality open model from Black Forest Labs"
    use_cases:
      - "Marketing content creation"
      - "Product visualization"
      - "UI/UX mockup generation"
      - "Personalized content"
      
  speech_generation:
    what: "Models that generate natural speech from text (TTS - Text-to-Speech)"
    models:
      openai_tts: "Natural-sounding, multiple voices"
      elevenlabs: "Voice cloning, emotional control"
      cartesia: "Ultra-low latency for real-time applications"
    use_cases:
      - "Voice assistants and IVR (Interactive Voice Response)"
      - "Audiobook generation"
      - "Accessibility (screen readers)"
      - "Customer service voice bots"
```

### Token Economics for Multi-Modal

```yaml
Token_Economics:
  image_input_costs:
    how_it_works: |
      Images are converted to tokens for processing.
      A typical image (1024×1024) = 1,000-5,000 tokens
      High-resolution image = 5,000-20,000 tokens
      
    pricing_examples:
      low_res_image:
        tokens: "~1,000 tokens"
        cost_gpt5: "$0.01"
        cost_claude: "$0.015"
        
      high_res_image:
        tokens: "~5,000 tokens"
        cost_gpt5: "$0.05"
        cost_claude: "$0.075"
        
      document_page_scan:
        tokens: "~3,000 tokens (standard page)"
        cost_per_page: "$0.03-0.045"
        cost_100_page_doc: "$3-4.50"
        
  video_input_costs:
    how_it_works: |
      Video sampled as frames + audio transcription
      1 minute of video ≈ 10-60 frames = 10,000-300,000 tokens
      
    pricing_examples:
      1_min_video:
        tokens: "~50,000 tokens (30 frames + audio)"
        cost_gemini: "$0.35"
        
      1_hour_video:
        tokens: "~3,000,000 tokens"
        cost_gemini: "$21"
        note: "Only Gemini 2.5 Pro handles this natively (2M context)"
        
  image_generation_costs:
    dall_e_4:
      standard: "$0.04-0.08 per image"
      hd: "$0.08-0.12 per image"
      
    midjourney_v7:
      subscription: "$30-120/month (limited generations)"
      per_image: "~$0.01-0.05 (depends on plan)"
      
    self_hosted_sd:
      per_image: "$0.001-0.005 (H100 inference)"
      upfront: "GPU infrastructure cost"
      
  audio_costs:
    whisper_transcription: "$0.006/minute"
    tts_generation: "$0.015/1000 characters"
    real_time_audio: "$0.06/minute (full-duplex)"
    
  optimization:
    image_resolution: "Use lowest resolution that meets quality needs"
    frame_sampling: "Don't process every frame — sample 1 per second or less"
    batch_processing: "Batch image analysis for 50% discount"
    caching: "Cache results for repeated images (product catalog)"
```

### Multi-Modal Pipeline Architecture

```python
# Multi-modal processing pipeline

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
import asyncio


class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MultiModalInput:
    modality: Modality
    content: Union[str, bytes]  # Text string or binary data
    metadata: dict              # MIME type, dimensions, duration, etc.


@dataclass
class ProcessingResult:
    output: str
    tokens_used: int
    cost_usd: float
    latency_ms: float
    modality_processed: Modality


class MultiModalPipeline:
    """Orchestrate multi-modal AI processing."""
    
    def __init__(self, config: dict):
        self.config = config
        self.vision_model = config["vision_model"]  # e.g., "gpt-5"
        self.audio_model = config["audio_model"]    # e.g., "whisper-v4"
        self.generation_model = config["generation_model"]  # e.g., "dall-e-4"
        
    async def process_document(self, pages: list[bytes]) -> dict:
        """Process a multi-page document with vision."""
        
        # Optimization: process pages in parallel
        tasks = [
            self._analyze_page(page, page_num) 
            for page_num, page in enumerate(pages)
        ]
        results = await asyncio.gather(*tasks)
        
        # Combine page results
        full_text = "\n\n".join(r["extracted_text"] for r in results)
        total_cost = sum(r["cost"] for r in results)
        
        return {
            "pages_processed": len(pages),
            "full_text": full_text,
            "per_page_analysis": results,
            "total_cost": total_cost,
        }
    
    async def _analyze_page(self, page_image: bytes, page_num: int) -> dict:
        """Analyze a single document page with vision model."""
        
        # Optimize image: resize if too large (save tokens)
        optimized = self._optimize_image(page_image, max_dimension=2048)
        
        response = await self._call_vision_model(
            image=optimized,
            prompt="Extract all text from this document page. Preserve formatting, tables, and structure.",
            detail="high",  # High detail for document OCR
        )
        
        return {
            "page": page_num,
            "extracted_text": response.text,
            "cost": response.cost,
            "tokens": response.tokens,
        }
    
    async def process_video(
        self, 
        video_url: str, 
        task: str = "summarize",
        sample_rate: int = 1,  # Frames per second to analyze
    ) -> dict:
        """Process video with frame sampling strategy."""
        
        # Strategy: sample frames intelligently (not every frame)
        frames = await self._extract_key_frames(video_url, sample_rate)
        audio_transcript = await self._transcribe_audio(video_url)
        
        # Analyze frames in batches (cost optimization)
        frame_descriptions = await self._batch_analyze_frames(frames)
        
        # Combine visual + audio understanding
        combined_context = self._build_video_context(
            frame_descriptions=frame_descriptions,
            transcript=audio_transcript,
        )
        
        # Final synthesis with text model (cheaper than processing everything with vision)
        summary = await self._synthesize(
            context=combined_context,
            task=task,
        )
        
        return {
            "frames_analyzed": len(frames),
            "transcript_length": len(audio_transcript),
            "summary": summary,
        }
    
    async def _extract_key_frames(self, video_url: str, fps: int) -> list[bytes]:
        """Extract key frames from video (scene changes, not every frame)."""
        # Use ffmpeg or similar to extract frames at specified rate
        # Advanced: detect scene changes and extract only transition frames
        pass
    
    async def _transcribe_audio(self, video_url: str) -> str:
        """Transcribe audio track using Whisper."""
        # Extract audio → send to Whisper API → return transcript
        pass
    
    def _optimize_image(self, image_bytes: bytes, max_dimension: int) -> bytes:
        """Resize image to reduce token usage while preserving quality."""
        # Resize to max_dimension while maintaining aspect ratio
        # Convert to efficient format (WebP)
        # Reduce quality if not needed (e.g., detail="low" for thumbnails)
        pass
    
    async def image_to_text_to_image(
        self,
        input_image: bytes,
        transformation: str,
    ) -> bytes:
        """Pipeline: understand image → transform description → generate new image."""
        
        # Step 1: Understand input image
        description = await self._call_vision_model(
            image=input_image,
            prompt=f"Describe this image in detail for reproduction. Then apply this transformation: {transformation}",
        )
        
        # Step 2: Generate new image from modified description
        new_image = await self._generate_image(
            prompt=description.text,
            style=self.config.get("generation_style", "photorealistic"),
        )
        
        return new_image
```

### Multi-Modal Quality Assurance

```yaml
Quality_Assurance:
  vision_qa:
    challenges:
      - "How do you verify an image description is correct?"
      - "How do you detect hallucinated details (objects not in image)?"
      - "How do you measure OCR accuracy?"
      
    evaluation_methods:
      ground_truth_comparison:
        what: "Compare model output against human-labeled ground truth"
        metrics: "Accuracy, F1 for detection, BLEU for descriptions"
        
      cross_model_verification:
        what: "Ask two different models to analyze same image, compare"
        benefit: "Disagreements flag potential errors"
        
      structured_extraction_validation:
        what: "For data extraction tasks, validate against schema"
        example: "Invoice extraction: validate amounts add up, dates are valid"
        
  generation_qa:
    image_generation:
      challenges:
        - "Does the image match the prompt? (text-image alignment)"
        - "Is quality acceptable? (artifacts, distortions)"
        - "Is content safe? (no NSFW, no copyrighted elements)"
        
      evaluation:
        clip_score: "Measures text-image alignment (automated)"
        fid_score: "Fréchet Inception Distance — measures image quality/diversity"
        human_eval: "Human raters assess quality and prompt adherence"
        safety_filter: "NSFW classifier on generated images"
        
    audio_generation:
      metrics:
        - "MOS (Mean Opinion Score): human-rated naturalness (1-5)"
        - "WER (Word Error Rate): if transcribing generated speech back"
        - "Speaker similarity: does voice match target?"
        - "Latency: time to first audio byte (for real-time applications)"
        
  pipeline_testing:
    end_to_end_tests:
      - "Input image → extract text → verify extracted text matches known content"
      - "Input document → OCR → structured data → validate against source"
      - "Input video → summarize → verify key events captured"
    regression_tests:
      - "Golden set of 100 images with known correct analyses"
      - "Run after model updates or pipeline changes"
      - "Alert if accuracy drops below threshold"
```

### Infrastructure Considerations

```yaml
Infrastructure:
  gpu_memory:
    vision_models:
      concern: "Processing high-res images requires significant GPU memory"
      strategy: "Batch smaller images together, process large images individually"
      
    video_processing:
      concern: "Video frames accumulate tokens rapidly"
      strategy: "Frame sampling (1 fps vs 30 fps), scene-change detection"
      
  storage:
    raw_inputs: "Images/video/audio take significant storage"
    caching: "Cache analysis results (avoid re-processing same image)"
    retention: "Define retention policies (delete raw after processing?)"
    
  preprocessing:
    image:
      - "Resize to optimal resolution for model (save tokens)"
      - "Convert to supported format (PNG, JPEG, WebP)"
      - "Strip EXIF metadata (privacy)"
      - "Validate file type (security — prevent malicious files)"
      
    audio:
      - "Convert to supported format (WAV, MP3, FLAC)"
      - "Split long audio into chunks (API limits)"
      - "Normalize volume levels"
      - "Detect and handle silence/noise"
      
    video:
      - "Extract frames at optimal sampling rate"
      - "Extract and transcribe audio track separately"
      - "Handle multiple video formats/codecs"
      - "Optimize for parallel processing"
      
  scaling:
    challenge: "Multi-modal processing is 10-100× more compute-intensive than text"
    strategies:
      - "Async processing for non-real-time (upload → process → notify)"
      - "Queue-based architecture (process images from queue, scale workers)"
      - "Tiered processing (fast low-res first, detailed on demand)"
      - "Caching aggressively (product images don't change often)"
```

---

## How It Works in Practice

### Production Multi-Modal System

```yaml
Production_Architecture:
  document_processing_system:
    flow:
      1: "User uploads document (PDF, image, scan)"
      2: "Preprocessing: convert to images, optimize resolution"
      3: "Vision model: OCR + structure extraction"
      4: "Post-processing: validate extracted data, format"
      5: "Return structured output (JSON with extracted fields)"
    scaling: "Queue-based, auto-scaling workers (0 to 50 based on queue depth)"
    latency: "2-10 seconds per page (async, user polls for result)"
    cost: "$0.03-0.05 per page"
    
  video_analysis_system:
    flow:
      1: "Video uploaded or streaming ingested"
      2: "Frame extraction (1 fps for analysis, key frames for search)"
      3: "Audio transcription (Whisper, parallel with frame extraction)"
      4: "Frame analysis (batch of 10 frames per vision API call)"
      5: "Synthesis (combine visual + audio into structured timeline)"
    scaling: "GPU workers for frame extraction, API calls for analysis"
    latency: "5-30 minutes per hour of video (async processing)"
    cost: "$5-25 per hour of video (depends on analysis depth)"
    
  real_time_multi_modal:
    flow:
      1: "User sends image + text query via chat"
      2: "Image preprocessed (resize, validate)"
      3: "Vision model processes image + text together"
      4: "Response generated (text or text + generated image)"
    latency: "1-3 seconds (synchronous, user waiting)"
    optimization: "Use detail=low for quick analysis, detail=high only when needed"
```

---

## Interview Tip

> When asked about multi-modal operations: "I architect multi-modal systems with three key considerations: (1) Token economics — images are expensive (1 image = 1,000-20,000 tokens). I optimize resolution (use lowest that meets quality needs), cache results (product images analyzed once, result cached), and batch process when real-time isn't needed (50% batch discount). Video is extreme: 1 hour = 3M tokens at $21. I use frame sampling (1 fps or scene-change detection, not every frame) + separate audio transcription (much cheaper). (2) Pipeline architecture — for document processing: queue-based async (upload → queue → worker → result). Workers auto-scale 0-50 based on queue depth. For real-time (chat with images): synchronous path optimized for 1-3 second response. For video: fully async with progress tracking. (3) Quality assurance — multi-modal is harder to evaluate than text. For OCR: validate against structured schemas (do amounts add up?). For descriptions: cross-model verification (two models analyze same image, flag disagreements). For generation: CLIP score (text-image alignment) + safety classifiers. Key infrastructure: preprocessing pipeline (resize, format conversion, metadata stripping), GPU-aware scheduling, aggressive caching (same image → same analysis), and tiered processing (fast low-res preview, then detailed on demand)."

---

## Common Mistakes

1. **Not optimizing image resolution** — Sending full 4K (4000×3000 pixel) product images to GPT-5 when 1024×1024 gives identical analysis quality. 4K = 15,000 tokens vs 1024 = 3,000 tokens per image. At 10,000 images/day, that's 120M wasted tokens ($1,200/day). Solution: resize to minimum resolution needed for the task. Document OCR needs high-res, thumbnail classification doesn't.

2. **Synchronous processing for large jobs** — Making users wait while 100-page PDF is processed (2-5 minutes). Bad UX and timeout risks. Solution: async processing (upload → return immediately with job ID → poll for results → webhook notification when done).

3. **Processing every video frame** — Analyzing 30 frames/second for a 1-hour video = 108,000 frames = $1,000+ and hours of processing time. Most frames are identical or near-identical. Solution: scene-change detection (analyze only when visual content changes), or fixed low rate (1 frame per 5 seconds = 720 frames total).

4. **No safety filters on generated content** — Generating images for users without checking for NSFW content, copyrighted elements, or harmful imagery. One inappropriate generated image = brand crisis. Solution: run NSFW classifier on all generated images before displaying, content policy filter on generation prompts.

5. **Ignoring preprocessing and validation** — Accepting any uploaded file without validation. Malicious files (embedded scripts in images, oversized files causing OOM), corrupt files, unsupported formats all crash the pipeline. Solution: validate file type (magic bytes, not just extension), enforce size limits, sanitize metadata, convert to standard format before processing.

---

## Key Takeaways

- Multi-modal processing is 10-100× more expensive than text — optimize resolution, sampling, caching
- Images: 1,000-20,000 tokens each. Use lowest resolution that meets quality needs
- Video: sample frames intelligently (scene changes, not every frame) + separate audio transcription
- Pipeline architecture: async for batch (queue-based), sync for interactive (optimized latency)
- Token economics: video hour = $5-25, document page = $0.03-0.05, single image = $0.01-0.05
- Preprocessing is critical: resize, format convert, validate, sanitize before model processing
- Quality assurance: cross-model verification, schema validation, CLIP scores for generation
- Safety: NSFW classifiers on all generated content, input validation on all uploads
- Caching: aggressive for static content (product images, known documents)
- Scale with queues: auto-scaling workers for variable multi-modal workloads
