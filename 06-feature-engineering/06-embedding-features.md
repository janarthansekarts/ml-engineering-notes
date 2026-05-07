# Embedding Features

## The Problem / Why This Matters

Traditional features (counts, averages, one-hot encodings) can't capture the rich semantic relationships between entities. A one-hot encoding of "user_1001" tells the model nothing about that user's similarity to other users. A one-hot encoding of "product_5678" gives no signal about what category, price range, or style it belongs to. Embeddings solve this by representing entities as dense vectors in a learned space where similar entities are close together. User embeddings capture behavioral patterns (users who buy similar things are nearby in embedding space). Product embeddings capture semantic properties (red dresses are near other dresses, not near power tools). Sentence embeddings capture meaning (semantically similar text maps to nearby vectors). In production ML systems, embeddings are often the most powerful features — they encode complex relationships that simple aggregations miss. The engineering challenge: how do you learn good embeddings? How do you serve them at low latency? How do you keep them fresh as behavior changes? How do you combine them with traditional features?

---

## The Analogy

Think of embeddings like GPS coordinates for concepts:

- **One-hot encoding** = Mailing addresses. "123 Main St" and "125 Main St" have no explicit relationship in the encoding — they're just different strings. You can't tell they're next door without external knowledge.
- **Embeddings** = GPS coordinates. (37.7749, -122.4194) and (37.7751, -122.4192) are clearly very close. You can instantly compute distance, find neighbors, and cluster locations. The encoding itself captures spatial relationships.
- **For products**: Embedding space = a map where similar products are geographically close. Red sneakers are in the "footwear district" near blue sneakers, far from the "kitchen appliance zone."
- **For users**: Embedding space = a behavioral map. Users who shop similarly live in the same "neighborhood." The model can recommend products popular in the user's neighborhood.
- **For text**: Embedding space = a meaning map. "Machine learning engineer" and "ML infrastructure developer" are neighbors. "Banana bread recipe" is on the other side of the map.

---

## Deep Dive

### Types of Embeddings

```yaml
Embedding_Types:
  learned_embeddings:
    what: "Trained as part of your model from interaction data"
    how: "Embedding layer in neural network, trained end-to-end"
    examples:
      user_embeddings: "Learned from user behavior (clicks, purchases, ratings)"
      item_embeddings: "Learned from item interactions and metadata"
      query_embeddings: "Learned from search queries and click-through"
    training_approaches:
      collaborative_filtering:
        what: "Learn embeddings from user-item interaction matrix"
        models: "Matrix Factorization, Neural Collaborative Filtering"
        signal: "Users who interacted with similar items → similar embeddings"
        
      two_tower:
        what: "Separate encoders for query/user and item, trained to maximize similarity for positive pairs"
        architecture: "Query tower + Item tower → dot product similarity"
        training: "Contrastive learning (positive pairs close, negative pairs far)"
        use_case: "Recommendation, search ranking"
        
      graph_neural_networks:
        what: "Learn embeddings from relationship graphs"
        models: "Node2Vec, GraphSAGE, GNN"
        signal: "Connected entities → similar embeddings"
        use_case: "Social networks, knowledge graphs"
        
  pre_trained_embeddings:
    what: "Embeddings from models trained on large datasets (transfer learning)"
    when: "Don't have enough domain data to train your own"
    
    text_embeddings:
      models:
        sentence_transformers: "all-MiniLM-L6-v2, all-mpnet-base-v2 (open-source)"
        openai: "text-embedding-3-small, text-embedding-3-large"
        cohere: "embed-english-v3.0"
        voyage: "voyage-3 (code-focused embeddings)"
      dimensions: "384 (small) to 3072 (large)"
      use_case: "Semantic search, document similarity, text features"
      
    image_embeddings:
      models:
        clip: "CLIP (OpenAI) — joint text-image embeddings"
        dinov2: "DINOv2 (Meta) — image-only, self-supervised"
        resnet: "ResNet features (penultimate layer)"
      dimensions: "512 to 2048"
      use_case: "Visual search, image similarity, multimodal features"
      
    code_embeddings:
      models:
        voyage_code: "Voyage Code 3"
        openai: "text-embedding-3-large (reasonable for code)"
        codebert: "CodeBERT (Microsoft)"
      use_case: "Code search, similar code detection, code understanding"
      
  hybrid_embeddings:
    what: "Combine learned embeddings with pre-trained embeddings"
    approach: "Concatenate or learn a projection layer"
    example: "User embedding (learned from behavior) + product text embedding (pre-trained)"
```

### Learning Embeddings

```python
# Learning user and item embeddings with Two-Tower model

import torch
import torch.nn as nn


class TwoTowerModel(nn.Module):
    """Two-tower model for learning user and item embeddings.
    
    User tower: maps user features → user embedding
    Item tower: maps item features → item embedding
    Training: dot product of (user, positive_item) > dot product of (user, negative_item)
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_feature_dim: int,
        item_feature_dim: int,
        embedding_dim: int = 128,
    ):
        super().__init__()
        
        # User tower
        self.user_id_embedding = nn.Embedding(num_users, 64)
        self.user_tower = nn.Sequential(
            nn.Linear(64 + user_feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.functional.normalize,  # L2 normalize output
        )
        
        # Item tower
        self.item_id_embedding = nn.Embedding(num_items, 64)
        self.item_tower = nn.Sequential(
            nn.Linear(64 + item_feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )
        
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def encode_user(self, user_ids: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        """Encode user into embedding space."""
        id_emb = self.user_id_embedding(user_ids)
        x = torch.cat([id_emb, user_features], dim=-1)
        return nn.functional.normalize(self.user_tower(x), dim=-1)
    
    def encode_item(self, item_ids: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """Encode item into embedding space."""
        id_emb = self.item_id_embedding(item_ids)
        x = torch.cat([id_emb, item_features], dim=-1)
        return nn.functional.normalize(self.item_tower(x), dim=-1)
    
    def forward(self, user_ids, user_features, item_ids, item_features):
        """Compute similarity scores (for in-batch negatives training)."""
        user_emb = self.encode_user(user_ids, user_features)  # [batch, dim]
        item_emb = self.encode_item(item_ids, item_features)  # [batch, dim]
        
        # In-batch negatives: every user paired with every item in batch
        logits = torch.matmul(user_emb, item_emb.T) / self.temperature
        
        # Labels: diagonal (user_i interacted with item_i)
        labels = torch.arange(len(user_ids), device=logits.device)
        
        loss = nn.functional.cross_entropy(logits, labels)
        return loss


# After training, extract embeddings for all users and items
def extract_embeddings(model, dataloader):
    """Extract embeddings for feature store."""
    user_embeddings = {}
    item_embeddings = {}
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            user_emb = model.encode_user(batch["user_id"], batch["user_features"])
            item_emb = model.encode_item(batch["item_id"], batch["item_features"])
            
            for uid, emb in zip(batch["user_id"], user_emb):
                user_embeddings[uid.item()] = emb.cpu().numpy()
            for iid, emb in zip(batch["item_id"], item_emb):
                item_embeddings[iid.item()] = emb.cpu().numpy()
    
    return user_embeddings, item_embeddings
```

### Feature Crossing with Embeddings

```yaml
Feature_Crossing:
  what: "Combining embedding features with other features to capture interactions"
  
  methods:
    concatenation:
      what: "Concatenate embedding with dense features"
      example: "[user_embedding(128d)] + [user_age, user_tenure, purchase_count] → 131d vector"
      when: "Let the model learn interactions"
      
    dot_product:
      what: "Similarity between two embeddings"
      example: "dot(user_embedding, item_embedding) → scalar similarity score"
      use_case: "Recommendation scoring"
      
    element_wise:
      what: "Element-wise operations between embeddings"
      operations:
        hadamard: "user_emb * item_emb (element-wise multiply)"
        difference: "user_emb - item_emb (captures asymmetric relationships)"
        
    attention:
      what: "Learn which dimensions of embeddings matter for the task"
      implementation: "Attention layer over embedding dimensions"
      
  practical_embedding_features:
    user_item_similarity:
      feature: "dot(user_embedding, item_embedding)"
      use_case: "Recommendation score"
      
    cluster_membership:
      feature: "Which cluster does this embedding belong to?"
      implementation: "K-means on embeddings → cluster ID as categorical feature"
      use_case: "User segmentation, item categorization"
      
    nearest_neighbors:
      feature: "Average distance to K nearest neighbors"
      use_case: "Anomaly detection (far from neighbors = unusual)"
      
    embedding_aggregation:
      feature: "Average embedding of user's recent interactions"
      example: "mean(last_10_item_embeddings) → user interest vector"
      use_case: "Capture recent interests without re-training"
```

### Serving Embeddings in Production

```yaml
Serving_Embeddings:
  challenges:
    size: "128-dimension float32 embedding = 512 bytes per entity"
    scale: "10M users × 512 bytes = 5GB in Redis"
    freshness: "Embeddings go stale as behavior changes"
    
  serving_patterns:
    redis_storage:
      format: "Binary serialized float array"
      key: "emb:user:{user_id}"
      value: "packed float32 array (512 bytes for 128d)"
      read_latency: "~1ms"
      
    pre_computed_features:
      what: "Convert embeddings to scalar features before serving"
      examples:
        - "user_cluster_id (from K-means on embeddings)"
        - "user_item_similarity (pre-computed for candidate items)"
        - "user_embedding_norm (magnitude as engagement signal)"
      benefit: "Smaller storage, faster retrieval, interpretable"
      
    quantized_embeddings:
      what: "Reduce embedding precision to save storage"
      methods:
        float16: "Half precision — 50% size reduction, minimal quality loss"
        int8: "Quantize to 8-bit — 75% size reduction"
        binary: "1-bit per dimension — 32× size reduction (for approximate similarity)"
      
  freshness_management:
    retraining_schedule: "Retrain embedding model weekly/monthly"
    incremental_update: "Fine-tune on recent interactions (not full retrain)"
    warm_start: "Initialize new entity embeddings from similar known entities"
    cold_start_problem: "New users/items have no embedding — use content features instead"
```

---

## How It Works in Practice

### Embedding Pipeline

```yaml
Production_Pipeline:
  training:
    frequency: "Weekly full retrain, daily incremental update"
    data: "Last 90 days of user-item interactions"
    infrastructure: "GPU cluster (A100), 4-8 hours for full retrain"
    output: "User embeddings (10M × 128d) + Item embeddings (1M × 128d)"
    
  materialization:
    process: "After training, export embeddings to feature store"
    storage: "Redis (online) + Parquet (offline)"
    indexing: "Also write to vector DB (FAISS/Pinecone) for ANN search"
    
  serving:
    retrieval: "Fetch user embedding from Redis (<1ms)"
    usage: "Feed to downstream model as features or use for ANN retrieval"
    
  monitoring:
    drift: "Monitor embedding distribution stability (average norm, cluster sizes)"
    quality: "Track downstream model performance (did new embeddings improve predictions?)"
    coverage: "% of entities with embeddings (cold start gap)"
```

---

## Interview Tip

> When asked about embedding features: "I use embeddings as features in three ways: (1) Learned embeddings — trained via two-tower models (user tower + item tower) on interaction data. Users who behave similarly get similar embeddings. These become features in downstream models (ranking, recommendation, fraud). I retrain weekly and serve from Redis (<1ms retrieval). (2) Pre-trained embeddings — text embeddings (OpenAI text-embedding-3-small or sentence-transformers) for text features, CLIP for image features. These provide semantic understanding without needing domain-specific training data. (3) Derived features from embeddings — I don't always serve raw embeddings. Instead I derive scalar features: user-item similarity (dot product), cluster membership (K-means on embeddings), and distance to category centroids. These are more interpretable and cheaper to serve. Key engineering considerations: embedding freshness (retrain weekly, incremental updates between), cold start (new entities use content-based features until enough interactions), and storage optimization (float16 quantization halves storage with minimal quality loss). The two-tower architecture is particularly powerful because after training, you can independently encode users and items — enabling approximate nearest neighbor (ANN) retrieval for candidate generation at massive scale."

---

## Common Mistakes

1. **Using outdated embeddings** — Training embeddings once and serving them forever. User behavior changes, product catalog evolves, embeddings become stale. Solution: retrain regularly (weekly minimum for behavioral embeddings), monitor embedding quality via downstream metrics, use incremental learning between full retrains.

2. **Ignoring cold start** — New user signs up → no interaction history → no embedding → model fails or produces garbage prediction. Solution: cold start strategy — use content-based features (demographics, signup info) for new entities, transition to behavioral embeddings after sufficient interactions (e.g., 10+ events).

3. **Serving raw high-dimensional embeddings unnecessarily** — Storing 128-dimensional float32 vectors in Redis for every entity when the downstream model only needs a similarity score. Solution: if you only need user-item similarity, pre-compute and store the scalar. If you only need cluster membership, store the cluster ID. Only serve raw embeddings when the downstream model actually uses all dimensions.

4. **Not normalizing embeddings** — Embeddings with varying magnitudes. One user has norm 100, another has norm 0.1. Dot product is dominated by magnitude, not direction. Solution: L2-normalize embeddings before serving. Similarity should be based on direction (cosine similarity), not magnitude, unless magnitude carries meaningful information.

5. **Training on biased interaction data** — Learning embeddings only from existing interactions. Popular items get strong embeddings, niche items get poor/random embeddings (feedback loop). Solution: incorporate item content features (text, images) into the item tower, so even items with few interactions get reasonable embeddings from their metadata.

---

## Key Takeaways

- Embeddings: dense vectors capturing semantic relationships (similar entities are close in embedding space)
- Learned embeddings: two-tower models trained on interaction data (best for behavioral signals)
- Pre-trained embeddings: transfer learning from large models (text, image, code)
- Feature crossing: dot product (similarity), concatenation (combine with other features), clustering
- Serving: Redis storage, <1ms retrieval, float16 quantization for size reduction
- Freshness: retrain weekly, incremental updates between, monitor downstream metric quality
- Cold start: content-based features for new entities, transition to behavioral after sufficient data
- Derived features: sometimes scalars (similarity, cluster ID) are better than raw embeddings
- Normalization: L2-normalize for cosine similarity based comparisons
- Scale: 10M users × 128d × 4 bytes = ~5GB in Redis (manageable with quantization)
