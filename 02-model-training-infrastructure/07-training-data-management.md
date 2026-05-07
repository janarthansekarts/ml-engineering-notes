# Training Data Management

## The Problem / Why This Matters

Model quality is bounded by data quality — no amount of compute or architecture innovation compensates for poor training data. But managing training data at ML (Machine Learning) scale introduces challenges far beyond a CSV file: datasets are terabytes to petabytes in size, must be loaded efficiently to keep GPUs fed, require preprocessing that can itself be a bottleneck, and need versioning to reproduce experiments. Data loading that's 10% slower than GPU consumption means GPUs idle 10% of the time — at $3/hour per H100, that's $0.30/hour wasted PER GPU. Across a 32-GPU training run over two weeks, that's over $2,000 wasted just from slow data loading. In 2026, with datasets like RedPajama (30TB), The Stack (6TB), and custom enterprise datasets growing to hundreds of terabytes, efficient data management is a first-class engineering concern. Understanding data formats, streaming strategies, sharding, caching, preprocessing pipelines, and data quality tooling is essential for ML engineers building production training systems.

---

## The Analogy

Think of training data management like running a restaurant kitchen:

- **Dataset** = The pantry full of ingredients. Raw, unprocessed, needs preparation before cooking (training).
- **Data loading pipeline** = The prep cooks. They chop, dice, measure ingredients and have them ready exactly when the chef (GPU) needs them. If prep is slow, the chef stands idle.
- **Streaming** = Getting fresh ingredients delivered continuously from the farm instead of buying everything upfront. You never run out, but you depend on the delivery truck arriving on time.
- **Sharding** = Splitting the pantry across multiple storage rooms. Each prep cook (data loader worker) gets their own section — no one blocks anyone else.
- **Caching** = Keeping commonly used ingredients pre-chopped in the fridge. First time takes effort (preprocessing), subsequent uses are instant.
- **Data versioning** = Recipe books with dated entries. When a dish tastes wrong (model quality drops), you can trace back to exactly which ingredients (data version) were used.

---

## Deep Dive

### Data Formats for ML Training

```yaml
Data_Formats:
  raw_formats:
    csv_json:
      use: "Small datasets (< 10GB), prototyping"
      pros: "Human-readable, universal, simple"
      cons: "Slow to parse, no compression, no random access, schema-less"
      
    jsonl:
      use: "Text datasets (NLP, LLM training data)"
      pros: "One record per line (streamable), flexible schema"
      cons: "No compression (unless gzipped), slow for numerical data"
      
  columnar_formats:
    parquet:
      use: "Tabular data, feature stores, structured datasets"
      pros: "Compressed, columnar (read only needed columns), fast"
      cons: "Not ideal for unstructured data (images, audio)"
      ecosystem: "Apache Arrow, Pandas, Spark, DuckDB"
      
    arrow:
      use: "In-memory data interchange, zero-copy reads"
      pros: "Zero deserialization cost, language-agnostic, columnar"
      cons: "Not compressed on disk (use Parquet for storage)"
      library: "PyArrow, HuggingFace datasets (backed by Arrow)"
      
  ml_optimized_formats:
    tfrecord:
      use: "TensorFlow training pipelines"
      pros: "Sequential access optimized, protobuf-based"
      cons: "TensorFlow-specific, hard to inspect, no random access"
      
    webdataset:
      use: "Large-scale training with sharded data (tar archives)"
      pros: "Sequential streaming, works with remote storage (S3), simple"
      cons: "No random access (sequential only)"
      pattern: "Files packed into .tar shards, streamed during training"
      
    mosaic_mds:
      name: "MosaicML Streaming (MDS format)"
      use: "Efficient streaming from cloud storage for distributed training"
      pros: "Deterministic shuffling, resumable, elastic (worker count changes)"
      cons: "Requires conversion from raw format"
      
    huggingface_datasets:
      use: "Most common for NLP/LLM training in 2026"
      backend: "Apache Arrow memory-mapped files"
      pros: "Streaming mode, automatic caching, huge dataset hub"
      cons: "Memory-mapping can be slow for first load"
      
  choosing_format:
    text_data_llm: "HuggingFace Datasets (Arrow) or WebDataset (.tar shards)"
    image_data: "WebDataset or TFRecord"
    tabular_data: "Parquet files"
    multimodal: "WebDataset (flexible, supports any file type in tar)"
    very_large_scale: "MosaicML Streaming (MDS) or WebDataset"
```

### Data Loading Architecture

```python
# Efficient data loading with PyTorch DataLoader
import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset

# Standard DataLoader with optimized settings
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,         # Parallel data loading processes
    pin_memory=True,       # Pre-copy to GPU-accessible memory (faster transfer)
    prefetch_factor=2,     # Each worker prefetches 2 batches ahead
    persistent_workers=True,  # Don't recreate workers each epoch (faster)
    drop_last=True,        # Drop incomplete last batch (avoids shape issues)
)

# For distributed training: DistributedSampler
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(
    dataset, 
    num_replicas=world_size, 
    rank=rank,
    shuffle=True
)

dist_dataloader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,       # Each GPU gets unique subset
    num_workers=4,
    pin_memory=True,
)
```

```yaml
DataLoader_Optimization:
  num_workers:
    purpose: "Parallel data loading (separate processes prepare next batches)"
    rule: "Set to 2-4× number of GPUs on the node (but not more than CPU cores)"
    too_few: "GPU starved for data (idle time between batches)"
    too_many: "CPU/memory overhead, diminishing returns, potential OOM"
    
  pin_memory:
    purpose: "Allocate data in page-locked memory (faster CPU→GPU transfer)"
    always_use: "True when training on GPU (minimal overhead, significant speedup)"
    
  prefetch_factor:
    purpose: "Each worker prepares N batches ahead of time"
    default: 2
    increase_if: "Data preprocessing is expensive (augmentation, tokenization)"
    
  persistent_workers:
    purpose: "Keep worker processes alive between epochs (avoid process spawn overhead)"
    always_use: "True (especially with many workers)"
    
  collate_function:
    purpose: "Custom batching logic (padding sequences, creating attention masks)"
    common_for_llm: "Dynamic padding to longest sequence in batch (saves compute vs fixed padding)"
```

### Streaming Large Datasets

```yaml
Streaming:
  why: "Datasets too large to download (RedPajama: 30TB, The Pile: 800GB, custom enterprise: variable)"
  
  approaches:
    huggingface_streaming:
      how: "load_dataset('name', streaming=True) → IterableDataset"
      source: "HuggingFace Hub, S3, GCS, local files"
      features:
        - "No disk space needed (stream directly from remote)"
        - "Automatic shuffling (shuffle buffer)"
        - "Interleave multiple datasets"
        - "Skip and take operations"
      code: |
        from datasets import load_dataset
        
        dataset = load_dataset("allenai/c4", streaming=True, split="train")
        shuffled = dataset.shuffle(seed=42, buffer_size=10_000)
        
        for example in shuffled:
            tokens = tokenizer(example["text"], truncation=True, max_length=2048)
            yield tokens
            
    webdataset_streaming:
      how: "Data stored as .tar shards, streamed sequentially"
      pattern: |
        import webdataset as wds
        
        urls = "s3://bucket/dataset/shard-{000000..001023}.tar"
        dataset = (
            wds.WebDataset(urls)
            .shuffle(1000)
            .decode("pil")          # Decode images
            .to_tuple("jpg", "json")  # Extract fields
            .map(preprocess)
        )
      advantage: "Works with any storage (S3, GCS, HTTP, local)"
      
    mosaic_streaming:
      how: "Convert data to MDS format, stream with deterministic shuffling"
      features:
        - "Deterministic: same data order regardless of worker count"
        - "Resumable: restart from exact position after interruption"
        - "Elastic: handles changing number of workers mid-training"
      use_case: "Production LLM training where reproducibility matters"
      
  shuffle_strategies:
    buffer_shuffle:
      how: "Maintain buffer of N examples, randomly sample from buffer"
      trade_off: "Larger buffer = better randomization but more memory"
      typical: "Buffer size 10K-100K examples"
      
    shard_shuffle:
      how: "Shuffle the ORDER of shards, then stream sequentially within each shard"
      advantage: "No memory overhead (just shuffles URLs)"
      limitation: "Not truly random within shards"
      
    full_shuffle:
      how: "Download everything, shuffle in memory, train"
      when: "Small enough datasets that fit in memory/disk"
```

### Data Preprocessing Pipelines

```yaml
Preprocessing:
  offline_preprocessing:
    what: "Process data BEFORE training (tokenization, filtering, deduplication)"
    when: "Processing is expensive and data is reused across many training runs"
    tools:
      datatrove: "HuggingFace's large-scale data processing library"
      spark: "Apache Spark for distributed processing"
      ray_data: "Ray's data processing for ML pipelines"
      dask: "Parallel computing library for larger-than-memory datasets"
    
    common_operations:
      tokenization: "Convert text to token IDs (once, not per-batch)"
      deduplication: "Remove duplicate/near-duplicate examples (MinHash, exact match)"
      filtering: "Remove low-quality, toxic, or irrelevant examples"
      packing: "Concatenate short examples to fill sequence length (avoid padding waste)"
      
  online_preprocessing:
    what: "Process data DURING training (data loading pipeline)"
    when: "Processing is cheap, data augmentation needs randomness per epoch"
    operations:
      text: "Dynamic tokenization, random truncation, masking"
      image: "Random crop, flip, color jitter, mixup"
      multimodal: "Alignment, format conversion"
      
  tokenization_optimization:
    pre_tokenize:
      description: "Tokenize entire dataset once, save token IDs"
      advantage: "No tokenization overhead during training (significant for large vocab)"
      storage: "Larger on disk (int32 token IDs vs compressed text)"
    on_the_fly:
      description: "Tokenize each batch during data loading"
      advantage: "Flexible (change tokenizer without reprocessing)"
      disadvantage: "CPU overhead per batch (can bottleneck with fast GPUs)"
      
  sequence_packing:
    what: "Concatenate multiple short documents to fill the target sequence length"
    why: "Avoid wasting compute on padding tokens"
    example:
      without_packing: "Doc1 (500 tokens) + PADDING (1548 tokens) = 2048 tokens (75% wasted)"
      with_packing: "Doc1 (500) + Doc2 (800) + Doc3 (700) + SEP = 2048 tokens (near 100% utilized)"
    savings: "30-50% less compute for datasets with variable-length documents"
    implementation: "HuggingFace ConstantLengthDataset, custom packing logic"
```

### Data Versioning and Lineage

```yaml
Data_Versioning:
  why:
    - "Reproduce experiments (which exact data produced which model?)"
    - "Debug regressions (model quality dropped — what changed in the data?)"
    - "Compliance (audit trail for regulated industries)"
    - "Collaboration (multiple team members working with same data)"
    
  tools:
    dvc:
      full_name: "Data Version Control"
      what: "Git-like versioning for large files and datasets"
      how: "Tracks metadata in Git, stores actual data in remote storage (S3, GCS)"
      use: "Most popular open-source data versioning tool"
      
    lakefs:
      what: "Git-like operations on data lakes (S3-compatible API)"
      how: "Branch, commit, merge operations on object storage"
      use: "When data lives in S3/data lake and you want Git semantics"
      
    delta_lake:
      what: "ACID transactions on data lakes (Parquet + transaction log)"
      how: "Adds versioning and time-travel to Parquet files"
      use: "Data engineering teams using Spark/data warehouses"
      
    huggingface_hub:
      what: "Dataset hosting with Git-LFS versioning"
      how: "Push datasets to hub, version automatically, access via API"
      use: "Sharing/consuming open datasets"
      
  data_lineage:
    what: "Track where data came from and how it was transformed"
    captures:
      - "Source (raw data origin)"
      - "Transformations (filtering, dedup, augmentation)"
      - "Splits (train/val/test creation)"
      - "Which model was trained on which data version"
    tools: "MLflow (data tracking), DVC pipelines, custom metadata stores"
```

---

## How It Works in Practice

### Example: LLM Pre-training Data Pipeline

```yaml
Example:
  scenario: "Pre-training a 7B language model on 2 trillion tokens"
  
  data_sources:
    - "Common Crawl (web text) — 80% of data"
    - "GitHub code — 10%"
    - "Books/papers — 5%"
    - "Curated Q&A/instruction data — 5%"
    
  preprocessing_pipeline:
    step_1_download:
      tool: "aria2c (parallel downloads from S3/HTTP)"
      output: "Raw text files (50TB compressed)"
      
    step_2_deduplication:
      tool: "datatrove MinHashDeduplication"
      method: "MinHash with 128 hash functions, Jaccard threshold 0.8"
      removes: "~30% of Common Crawl (duplicates/near-duplicates)"
      
    step_3_quality_filtering:
      filters:
        - "Language detection (keep English, or target languages)"
        - "Perplexity filter (remove very high perplexity = gibberish)"
        - "Repetition filter (remove pages with excessive repetition)"
        - "Content filter (remove toxic/adult content)"
      removes: "~40% of remaining data"
      
    step_4_tokenization:
      tokenizer: "Llama tokenizer (128K vocab, BPE)"
      output: "Token ID arrays (uint16), packed into shards"
      shard_size: "1GB per shard, ~1000 shards total"
      format: "MosaicML MDS or NumPy memory-mapped arrays"
      
    step_5_shuffle:
      method: "Shard-level shuffle + buffer shuffle during training"
      seed: "Fixed for reproducibility"
      
  training_data_loading:
    format: "Memory-mapped token arrays (zero-copy, fast random access)"
    workers: "4 per GPU (32 total on 8-GPU node)"
    throughput: "Must sustain >5 GB/s to keep 8 H100 GPUs fed"
    prefetch: "2 batches ahead per worker"
    sequence_length: 4096
    packing: "Concatenate documents with BOS/EOS separators"
```

---

## Interview Tip

> When asked about training data management: "I think about the data pipeline in three phases: (1) Offline preprocessing — deduplication (MinHash), quality filtering (perplexity, toxicity), tokenization (done once, saves CPU during training). (2) Efficient storage — sharded formats like WebDataset or MDS that enable parallel streaming from cloud storage without downloading the full dataset. (3) High-throughput loading — multi-worker DataLoader with pin_memory, prefetching, and persistent workers to keep GPUs fed. For LLM training, sequence packing is critical — concatenating short documents to fill the sequence length avoids 30-50% compute waste on padding. I always verify the data pipeline isn't the bottleneck: profile GPU utilization — if it dips between batches, data loading is too slow. Data versioning (DVC or LakeFS) ensures reproducibility: you can always trace which data version produced which model."

---

## Common Mistakes

1. **Tokenizing during training** — Running the tokenizer on-the-fly for each batch when training on the same data multiple times. Pre-tokenize once, store as token IDs, save CPU cycles during every subsequent training step. The tokenizer adds 10-30% overhead per batch on fast GPUs.

2. **Not enough DataLoader workers** — Setting num_workers=0 or 1 on a multi-GPU node. With 8 GPUs consuming data at high throughput, you need 2-4 workers per GPU (16-32 total) to prevent data starvation. Monitor GPU utilization for drops between batches.

3. **Ignoring sequence packing** — Training with variable-length sequences padded to max_length. If average document length is 500 tokens and max_length is 2048, you're wasting 75% of compute on padding tokens. Packing saves 30-50%.

4. **No data deduplication** — Training on duplicated data leads to memorization and poor generalization. Near-duplicates in web-scraped data can constitute 30-40% of the dataset. MinHash deduplication is computationally expensive but essential.

5. **Random access on remote storage** — Using datasets that require random access (shuffle all indices) when data is on S3/GCS. Remote random access is 100-1000x slower than sequential. Use streaming with buffer shuffle, or download and cache locally.

---

## Key Takeaways

- Data loading must keep pace with GPU consumption — if GPUs idle waiting for data, money is wasted
- Pre-tokenize datasets for reuse — tokenization overhead is significant at scale
- Sequence packing: concatenate short documents to avoid 30-50% compute waste on padding
- Use streaming (HuggingFace streaming, WebDataset, MDS) for datasets larger than disk
- Optimal DataLoader: 2-4 workers per GPU, pin_memory=True, persistent_workers=True, prefetch_factor=2
- Data deduplication (MinHash): removes 30-40% duplicates from web-scraped data, improves model quality
- Data formats: Parquet (tabular), Arrow/HF Datasets (text), WebDataset (multimodal), MDS (production streaming)
- Version data with DVC or LakeFS for reproducibility and debugging model quality regressions
- Shard data (1GB per shard) for parallel loading — multiple workers read different shards simultaneously
- Quality filtering pipeline: language detection → dedup → perplexity filter → content filter → tokenize
