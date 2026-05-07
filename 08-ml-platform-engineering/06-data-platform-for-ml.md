# Data Platform for ML

## The Problem / Why This Matters

ML models are only as good as their data. Yet accessing, preparing, and serving data for ML is one of the most time-consuming activities — data scientists spend 60-80% of their time on data work. Without a proper data platform for ML, teams face: scattered datasets (some in S3, some in BigQuery, some on team NFS shares), no dataset versioning (can't reproduce last month's training run), inconsistent feature computation (training vs. serving skew), no data quality guarantees (bad data → bad models), and manual data discovery (asking Slack "where's the user activity data?"). A data platform for ML provides: centralized data access (single interface to all data sources), feature store (consistent feature computation and serving), dataset management (versioning, lineage, discovery), data quality monitoring (automated checks on all data flowing to models), and efficient data loading (optimized formats and caching for GPU training). In 2026, this includes: lakehouse architectures (Delta Lake, Apache Iceberg, Apache Hudi) unifying batch and streaming, feature platforms (Tecton, Feast 0.40+) with real-time feature serving, and vector databases (for embedding features in LLM/RAG applications).

---

## The Analogy

Think of a data platform for ML like a well-organized commercial kitchen's supply chain:

- **Without platform** = Each chef sources their own ingredients. One goes to the farmer's market (raw SQL), another to a wholesaler (data warehouse), another grows their own (custom ETL). No quality control, inconsistent ingredients, no inventory management.
- **With platform** = Professional supply chain. Central procurement (data catalog), quality inspection (data validation), consistent prep (feature store), inventory tracking (dataset versioning), and just-in-time delivery (feature serving at prediction time).

Chefs focus on cooking (modeling), not sourcing and prepping ingredients from scratch every time.

---

## Deep Dive

### Data Platform Architecture for ML

```yaml
Architecture:
  data_sources:
    operational_databases: "PostgreSQL, MySQL, DynamoDB (transactional data)"
    data_warehouse: "BigQuery, Snowflake, Redshift (analytical data)"
    data_lake: "S3, GCS, ADLS (raw and processed files)"
    streaming: "Kafka, Kinesis, Pub/Sub (real-time events)"
    third_party: "APIs, vendor data feeds, external datasets"
    
  lakehouse_layer:
    what: "Unified storage with ACID transactions, schema evolution, time travel"
    technologies:
      delta_lake:
        what: "Open-source lakehouse by Databricks"
        features: "ACID on object storage, time travel, Z-ordering"
        format: "Parquet + transaction log"
        
      apache_iceberg:
        what: "Open table format (Netflix origin)"
        features: "Hidden partitioning, schema evolution, time travel"
        adoption: "AWS (default for Athena/Glue), Snowflake, Spark"
        
      apache_hudi:
        what: "Incremental data processing"
        features: "Upserts, incremental reads, CDC ingestion"
        best_for: "CDC (Change Data Capture) workloads"
        
    benefit_for_ml:
      - "Time travel → reproduce exact training dataset from any point in time"
      - "Schema evolution → features can change without breaking pipelines"
      - "ACID → no partial reads during data updates"
      - "Partitioning → efficient reads for time-windowed training data"
      
  feature_store:
    what: "Central repository for computed ML features"
    components:
      offline_store:
        what: "Historical feature values for training"
        storage: "BigQuery, Redshift, or data lake (Parquet)"
        access: "Point-in-time correct joins for training data"
        
      online_store:
        what: "Latest feature values for real-time inference"
        storage: "Redis, DynamoDB, Bigtable (low-latency key-value)"
        access: "Feature vector lookup by entity ID (< 10ms)"
        
      feature_registry:
        what: "Metadata about all features (definitions, owners, lineage)"
        includes: "Feature name, description, data type, source, transformation logic"
        
    tools:
      feast:
        what: "Open-source feature store"
        strengths: "Lightweight, Python-native, multiple backends"
        version: "0.40+ (2026 — improved streaming, push sources)"
        
      tecton:
        what: "Managed feature platform"
        strengths: "Real-time features, streaming compute, enterprise features"
        pricing: "Based on feature read/write volume"
        
      databricks_feature_store:
        what: "Feature store integrated with Databricks/Unity Catalog"
        strengths: "Tight integration with Databricks ML workflow"
        
  dataset_management:
    versioning:
      what: "Track exact datasets used for each training run"
      tools:
        dvc: "Data Version Control — Git for data (tracks large files in S3/GCS)"
        lakefs: "Git-like branching for data lakes"
        delta_time_travel: "Use Delta Lake timestamps to recreate datasets"
      benefit: "Reproduce any training run exactly (same data + same code)"
      
    catalog:
      what: "Discover and understand available datasets"
      tools:
        datahub: "Open-source data catalog (LinkedIn origin)"
        unity_catalog: "Databricks unified governance"
        google_data_catalog: "GCP data discovery"
      features: "Search, schema, lineage, quality metrics, ownership"
      
    lineage:
      what: "Track where data came from and where it flows"
      benefit_for_ml: "When source data changes, know which features and models are affected"
      tools: "OpenLineage, Marquez, Datahub lineage"
```

### Feature Store Deep Dive

```python
# Feature store usage patterns for ML

"""
Feature store provides consistent feature computation for both 
training (historical/offline) and serving (real-time/online).
Prevents training-serving skew — the #1 source of model bugs.
"""

from datetime import datetime
import pandas as pd


class FeaturePlatform:
    """
    Platform's feature store interface.
    
    Key guarantee: Features used in training are computed IDENTICALLY
    to features served at prediction time. Same code, same logic.
    """
    
    def __init__(self):
        self.offline_store = None  # BigQuery/Redshift
        self.online_store = None   # Redis/DynamoDB
        
    # --- For Training (Offline) ---
    def get_training_data(
        self,
        feature_sets: list,      # ["user-engagement", "user-demographics"]
        entity_df: pd.DataFrame,  # Entity IDs + timestamps
        label_column: str = None,
    ) -> pd.DataFrame:
        """
        Point-in-time correct feature retrieval for training.
        
        Critical: Uses timestamps to prevent feature leakage.
        For each entity at time T, only features available BEFORE T are joined.
        
        Example:
            entity_df = pd.DataFrame({
                "user_id": ["u1", "u2", "u3"],
                "event_timestamp": ["2024-01-15", "2024-02-20", "2024-03-10"],
                "churned": [1, 0, 1],  # label
            })
            
            training_data = platform.features.get_training_data(
                feature_sets=["user-engagement-7d", "user-demographics"],
                entity_df=entity_df,
            )
            # Returns: user_id | event_timestamp | churned | login_count_7d | avg_session_min | age | ...
        """
        return self.offline_store.get_historical_features(
            feature_refs=self._resolve_feature_refs(feature_sets),
            entity_df=entity_df,
        ).to_df()
    
    # --- For Serving (Online) ---
    def get_online_features(
        self,
        feature_set: str,
        entity_ids: list,
    ) -> dict:
        """
        Low-latency feature retrieval for real-time inference.
        
        Returns latest feature values for given entities.
        Latency target: < 10ms for single entity, < 50ms for batch.
        
        Example:
            features = platform.features.get_online_features(
                feature_set="user-engagement-7d",
                entity_ids=["user_123"],
            )
            # Returns: {"user_123": {"login_count_7d": 5, "avg_session_min": 12.3, ...}}
        """
        return self.online_store.get_features(
            feature_set=feature_set,
            entity_ids=entity_ids,
        )
    
    # --- Feature Definition ---
    def define_feature(
        self,
        name: str,
        description: str,
        entity: str,
        value_type: str,
        source: str,
        transformation: str,
        owner: str,
        tags: dict = None,
    ):
        """
        Register a new feature definition.
        
        The same transformation is used for both offline (training)
        and online (serving) computation — preventing training-serving skew.
        
        Example:
            platform.features.define_feature(
                name="user_login_count_7d",
                description="Number of user logins in the last 7 days",
                entity="user_id",
                value_type="int64",
                source="events.user_logins",
                transformation="COUNT(*) WHERE timestamp > NOW() - INTERVAL 7 DAY",
                owner="growth-team",
                tags={"category": "engagement", "freshness": "hourly"},
            )
        """
        self.registry.register(
            name=name,
            description=description,
            entity=entity,
            value_type=value_type,
            source=source,
            transformation=transformation,
            owner=owner,
            tags=tags or {},
        )


# Dataset versioning and management
class DatasetManager:
    """
    Manages ML datasets with versioning, lineage, and quality.
    """
    
    def create_dataset(
        self,
        name: str,
        query: str,       # SQL or DataFrame operation
        version_tag: str = None,
        quality_checks: list = None,
    ) -> "VersionedDataset":
        """
        Create a versioned dataset for training.
        
        Example:
            dataset = platform.data.create_dataset(
                name="churn-training-v3",
                query="SELECT * FROM features.churn_features WHERE date > '2024-01-01'",
                quality_checks=[
                    {"check": "row_count", "min": 10000},
                    {"check": "null_rate", "column": "target", "max": 0.0},
                    {"check": "freshness", "max_age_hours": 24},
                ],
            )
            # Dataset is immutable once created (reproducible)
            print(dataset.version)  # "v3"
            print(dataset.row_count)  # 150,000
            print(dataset.location)  # "s3://datasets/churn-training/v3/"
        """
        # Execute query and materialize
        data = self._execute_query(query)
        
        # Run quality checks
        if quality_checks:
            self._validate(data, quality_checks)
        
        # Store immutably with version
        version = version_tag or self._auto_version(name)
        location = self._store_immutable(data, name, version)
        
        return VersionedDataset(
            name=name,
            version=version,
            location=location,
            row_count=len(data),
            schema=data.dtypes.to_dict(),
            created_at=datetime.utcnow(),
            query=query,  # Lineage: how was this dataset created?
        )
```

### Vector Database for ML

```yaml
Vector_Databases:
  what: "Specialized storage for embedding vectors (used in RAG, similarity search, recommendations)"
  
  use_cases_in_ml:
    rag: "Store document embeddings, retrieve relevant context for LLM"
    recommendations: "Store item embeddings, find similar items"
    anomaly_detection: "Store normal embeddings, detect outliers"
    semantic_search: "Store content embeddings, search by meaning"
    
  options_2026:
    pgvector:
      what: "Vector extension for PostgreSQL"
      strengths: "Use existing PostgreSQL infrastructure, ACID transactions"
      scale: "Millions of vectors (with proper indexing)"
      best_for: "Teams already on PostgreSQL, moderate scale"
      
    pinecone:
      what: "Managed vector database"
      strengths: "Fully managed, fast, auto-scaling"
      scale: "Billions of vectors"
      best_for: "Production RAG at scale, minimal ops"
      
    weaviate:
      what: "Open-source vector database"
      strengths: "Hybrid search (vector + keyword), modules ecosystem"
      scale: "Hundreds of millions of vectors"
      best_for: "Teams wanting control and hybrid search"
      
    qdrant:
      what: "Open-source vector database (Rust-based)"
      strengths: "Fast, efficient, flexible filtering"
      scale: "Billions of vectors"
      best_for: "Performance-sensitive production workloads"
      
    chromadb:
      what: "Developer-friendly embedding database"
      strengths: "Simple API, great for prototyping"
      scale: "Millions of vectors"
      best_for: "Prototyping and small-medium production"
```

### Efficient Data Loading for Training

```yaml
Data_Loading:
  challenge: "GPUs idle waiting for data — data loading is often the bottleneck"
  
  strategies:
    webdataset:
      what: "Tar-based sequential format for large datasets"
      benefit: "Sequential I/O (no random seeks), shuffling at shard level"
      use_for: "Image/video datasets, distributed training"
      
    mosaic_streaming:
      what: "MosaicML streaming dataset library"
      benefit: "Stream from cloud storage, no local download needed"
      use_for: "Large datasets that don't fit locally"
      
    data_caching:
      what: "Cache frequently-used datasets on fast local storage"
      implementation: "NVMe SSD on GPU nodes, LRU cache of recent datasets"
      benefit: "Avoid re-downloading from S3/GCS for every training run"
      
    prefetching:
      what: "Load next batch while GPU processes current batch"
      implementation: "PyTorch DataLoader with num_workers > 0, prefetch_factor > 1"
      benefit: "GPU never waits for data"
      
    format_optimization:
      parquet: "Columnar, compressed — good for tabular ML"
      arrow: "Zero-copy reads, fast serialization — good for DataFrame operations"
      tfrecord: "TensorFlow's binary format — sequential, efficient"
      safetensors: "Fast, safe model weight loading (HuggingFace standard)"
```

---

## How It Works in Practice

### End-to-End Data Flow

```yaml
Data_Flow:
  raw_data:
    sources: "Databases, event streams, APIs"
    ingestion: "CDC (Debezium) → Kafka → Delta Lake/Iceberg"
    
  processed_data:
    transformation: "dbt/Spark transforms raw → cleaned/enriched tables"
    storage: "Data warehouse (BigQuery) + data lake (S3 as Iceberg tables)"
    
  ml_features:
    computation: "Feature pipelines (Dagster/Spark) compute features from processed data"
    storage: "Feature store (offline: BigQuery, online: Redis)"
    serving: "Feature platform serves features for training and inference"
    
  training_datasets:
    creation: "Point-in-time join of features + labels"
    versioning: "Immutable, versioned snapshots (DVC or Delta time travel)"
    access: "Efficient loading via optimized formats + caching"
    
  inference:
    online: "Feature store online lookup → model → prediction"
    batch: "Feature store offline query → batch scoring → output table"
```

---

## Interview Tip

> When asked about data platform for ML: "My data platform for ML has four layers: (1) Lakehouse — all data stored in Apache Iceberg tables on object storage (S3/GCS). Iceberg gives time travel (reproduce any training dataset from any point in time), schema evolution (features can change without breaking), and ACID transactions (no partial reads). (2) Feature store — Feast or Tecton for consistent feature computation. The critical guarantee: same transformation logic for training (offline, historical) and serving (online, real-time). This prevents training-serving skew, which is the #1 source of model bugs in production. Online store (Redis) serves features in <10ms. (3) Dataset management — versioned, immutable training datasets with lineage. Every training run records exactly which dataset version was used. I can reproduce any model from 6 months ago with the exact same data. (4) Efficient data loading — WebDataset/Mosaic Streaming for large-scale training, NVMe caching on GPU nodes for frequently-used datasets, PyTorch DataLoader with prefetching so GPUs never wait for data. For LLM/RAG applications, I add a vector database layer (pgvector or Qdrant) for embedding storage and similarity search. The whole platform is discoverable via a data catalog (DataHub) where any data scientist can search for available datasets and features."

---

## Common Mistakes

1. **Training-serving skew** — Features computed differently in training (complex Spark job) vs. serving (simplified Python approximation). Model performs well offline but poorly in production because it's seeing different feature values. Solution: feature store guarantees same computation logic for both. Define feature once, compute it identically everywhere.

2. **No dataset versioning** — Training run uses "latest data." Next month, data pipeline changes upstream. Can't reproduce the model or compare fairly to new experiments. Solution: immutable versioned datasets. Every training run references a specific dataset version. Time travel in lakehouse provides this for free.

3. **Downloading data for every training run** — Each experiment re-downloads 100GB from S3 (takes 30 min before training starts). GPU idle during download. Solution: data caching on GPU nodes (NVMe SSD), streaming datasets (Mosaic Streaming — no full download needed), and persistent storage for commonly-used datasets.

4. **No data quality monitoring** — Bad data (null values, schema changes, stale data) flows into model training without anyone noticing. Model trains on garbage, performs terribly. Solution: automated data quality checks at every pipeline boundary. Great Expectations or similar — validate before any ML consumption.

5. **No catalog or discovery** — Data scientists don't know what data is available. They ask on Slack, get outdated answers, or build features that already exist elsewhere. Solution: data catalog (DataHub, Unity Catalog) with automatic discovery, schema documentation, and feature registry search.

---

## Key Takeaways

- Data platform for ML = lakehouse + feature store + dataset management + efficient loading
- Lakehouse (Iceberg/Delta): time travel for reproducibility, schema evolution, ACID transactions
- Feature store: same computation for training (offline) and serving (online) — prevents skew
- Online feature serving: < 10ms latency via Redis/DynamoDB (critical for real-time inference)
- Dataset versioning: immutable snapshots, every training run references specific version
- Data quality: automated validation at every boundary (don't train on bad data)
- Efficient loading: WebDataset/Mosaic Streaming + NVMe caching + prefetching
- Vector databases: pgvector, Pinecone, Qdrant — for RAG, similarity search, recommendations
- Catalog: DataHub or Unity Catalog for discovery (data scientists can find what exists)
- Format: Parquet (tabular), Safetensors (models), WebDataset (large-scale media)
