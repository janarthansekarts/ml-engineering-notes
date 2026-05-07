# Feature Discovery and Sharing

## The Problem / Why This Matters

In any organization with more than a few ML models, teams independently create features that already exist elsewhere. Team A builds "user_purchase_count_30d" for churn prediction, Team B builds "user_orders_last_month" for recommendation — same feature, different names, slightly different logic, duplicated compute cost. Feature discovery solves this: a centralized system where teams can find, understand, and reuse features others have already built. Without it: wasted engineering time (rebuilding features that exist), inconsistent logic (same concept computed differently across models), higher infrastructure costs (duplicate pipelines computing the same thing), and knowledge silos (when the person who built a feature leaves, nobody knows what it does or how to maintain it). A mature feature catalog is like a searchable library where every feature has documentation, lineage, quality metrics, and usage history — making it trivial to find "do we already have a user activity score?" before spending a week building one.

---

## The Analogy

Think of feature discovery like a company's internal tool library:

**Without a catalog** (most companies): Every carpenter (ML engineer) builds their own hammer. Some are great, some break. Nobody knows what tools others have. When a carpenter leaves, their custom tools are abandoned because nobody knows how they work.

**With a catalog** (feature discovery): A shared tool room with labels on every tool, instructions for use, maintenance history, and a sign-out sheet showing who's using what. Before building a new tool, you check the catalog: "Do we have a Phillips-head screwdriver rated for precision work?" If yes, grab it. If no, build it and add it to the catalog for others.

The catalog doesn't just save time — it ensures consistency (everyone uses the same tested screwdriver) and enables maintenance (you know which projects break if you modify a tool).

---

## Deep Dive

### Feature Catalog Architecture

```yaml
Feature_Catalog:
  what: "Central registry of all features with metadata, documentation, and usage tracking"
  
  catalog_entry:
    identity:
      feature_name: "user_purchase_count_30d"
      namespace: "user_activity"
      version: "v3"
      entity: "user_id"
      
    description:
      human_readable: "Number of completed purchases by this user in the last 30 days"
      business_context: "Key signal for user engagement and churn prediction"
      created_by: "ml-team@company.com"
      created_date: "2024-06-15"
      last_modified: "2025-01-20"
      
    technical_metadata:
      data_type: "INT64"
      value_range: "[0, 1000]"
      null_rate: "0.2%"
      update_frequency: "daily"
      computation_latency: "15min"
      source_tables: ["orders", "order_items"]
      pipeline_id: "dagster:user_activity_daily"
      
    quality_metrics:
      freshness: "Updated daily at 03:45 UTC (SLA: before 05:00 UTC)"
      completeness: "99.8% of active users have this feature"
      stability: "Distribution stable (KS test p-value > 0.05 vs 7-day-ago)"
      
    lineage:
      upstream_sources: ["postgres.orders", "postgres.order_items"]
      downstream_models: ["churn_prediction_v4", "recommendation_v2", "ltv_model"]
      downstream_features: ["user_engagement_score (depends on this)"]
      
    usage:
      models_using: 7
      teams_using: ["growth", "recommendations", "risk"]
      monthly_reads: 14_000_000
      last_read: "2025-03-20T14:30:00Z"
```

### Feature Search and Discovery

```yaml
Search_Capabilities:
  text_search:
    what: "Search by feature name, description, or tags"
    examples:
      - "search: 'purchase count' → finds user_purchase_count_30d, user_purchase_count_7d"
      - "search: 'churn' → finds features used in churn models"
      - "search: 'user engagement' → finds engagement-related features"
      
  semantic_search:
    what: "Find features by meaning (not just exact keywords)"
    implementation: "Embed feature descriptions → vector search"
    examples:
      - "query: 'how active is this user' → finds purchase_count, login_frequency, page_views"
      - "query: 'is this user likely to leave' → finds churn-related features"
      
  filter_search:
    what: "Filter by metadata attributes"
    filters:
      - "entity=user_id (all user-level features)"
      - "update_frequency=real_time (streaming features only)"
      - "team=growth (features owned by growth team)"
      - "data_type=FLOAT64 (numeric features only)"
      - "freshness < 1h (very fresh features)"
      
  similarity_search:
    what: "Find features similar to one you've found"
    approach: "Based on: same entity, same sources, correlated values, similar descriptions"
    example: "Similar to user_purchase_count_30d → shows user_purchase_count_7d, user_order_value_avg, user_active_days"
    
  recommendation:
    what: "Suggest features for a given modeling task"
    approach: "Look at what features other models for similar tasks use"
    example: "Task: churn prediction → suggests: purchase_count, days_since_last_login, support_ticket_count, feature_usage_decline"
```

### Feature Documentation Standards

```yaml
Documentation_Standards:
  required_fields:
    name: "Unique, descriptive, follows naming convention (entity_metric_window)"
    description: "One-sentence plain English explanation"
    entity: "Primary entity (user_id, product_id, session_id)"
    data_type: "INT64, FLOAT64, STRING, ARRAY, EMBEDDING"
    update_frequency: "real_time, hourly, daily, weekly"
    owner: "Team or individual responsible"
    
  recommended_fields:
    business_context: "Why this feature exists, what business question it answers"
    computation_logic: "SQL or pseudocode showing how it's computed"
    edge_cases: "What happens for new users? Users with no activity?"
    known_issues: "Any known limitations or quirks"
    related_features: "Other features in the same family"
    
  naming_conventions:
    pattern: "{entity}_{metric}_{window}_{aggregation}"
    examples:
      - "user_purchase_count_30d" — entity: user, metric: purchase_count, window: 30d
      - "product_view_count_7d" — entity: product, metric: view_count, window: 7d
      - "user_session_duration_avg_30d" — entity: user, metric: session_duration, window: 30d, agg: avg
      - "user_last_login_days_since" — entity: user, metric: last_login, window: N/A, agg: days_since
      
  anti_patterns:
    - "feat_1, feat_2, feature_A (opaque names)"
    - "data (too generic)"
    - "john_feature_v3 (personal, unversioned)"
    - "temp_user_thing (temporary that became permanent)"
```

### Feature Reusability Patterns

```python
# Feature sharing implementation patterns

"""
Patterns for making features reusable across teams and models.
"""

# Pattern 1: Feature Groups (related features bundled together)
class UserActivityFeatureGroup:
    """
    Reusable feature group: user activity metrics.
    
    Contains all activity-related features for a user entity.
    Any team can import this group to get all user activity signals.
    """
    
    entity = "user_id"
    namespace = "user_activity"
    update_frequency = "daily"
    
    features = {
        "purchase_count_7d": FeatureSpec(
            dtype="int64",
            description="Purchases in last 7 days",
            computation="COUNT(orders) WHERE date > today - 7",
        ),
        "purchase_count_30d": FeatureSpec(
            dtype="int64", 
            description="Purchases in last 30 days",
            computation="COUNT(orders) WHERE date > today - 30",
        ),
        "total_spend_30d": FeatureSpec(
            dtype="float64",
            description="Total spend in last 30 days (USD)",
            computation="SUM(order_total) WHERE date > today - 30",
        ),
        "days_since_last_purchase": FeatureSpec(
            dtype="int64",
            description="Days since most recent purchase",
            computation="DATEDIFF(today, MAX(order_date))",
        ),
        "avg_order_value_90d": FeatureSpec(
            dtype="float64",
            description="Average order value over 90 days",
            computation="AVG(order_total) WHERE date > today - 90",
        ),
    }


# Pattern 2: Feature composition (building on shared features)
class ChurnPredictionFeatures:
    """
    Model-specific feature set that REUSES shared feature groups.
    
    Instead of computing purchase_count ourselves, we reference
    the shared UserActivityFeatureGroup — same pipeline, same values.
    """
    
    # Reuse shared feature groups (no duplication)
    shared_features = [
        UserActivityFeatureGroup.features["purchase_count_30d"],
        UserActivityFeatureGroup.features["days_since_last_purchase"],
        UserActivityFeatureGroup.features["total_spend_30d"],
        UserEngagementFeatureGroup.features["login_frequency_7d"],
        UserEngagementFeatureGroup.features["feature_usage_score"],
    ]
    
    # Model-specific features (only compute these ourselves)
    custom_features = {
        "support_ticket_count_14d": FeatureSpec(
            dtype="int64",
            description="Support tickets opened in last 14 days",
        ),
        "subscription_days_remaining": FeatureSpec(
            dtype="int64",
            description="Days until subscription expires",
        ),
    }


# Pattern 3: Feature versioning (safe iteration)
class FeatureVersioning:
    """
    Version features when computation logic changes.
    Old version remains available (existing models don't break).
    New version available for new models.
    """
    
    def register_feature_version(
        self, 
        name: str, 
        version: int,
        computation: callable,
        migration_notes: str = None,
    ):
        """Register a new version of an existing feature."""
        
        # Old version continues to be computed (for existing models)
        # New version computed in parallel
        # Models can opt-in to new version at their own pace
        
        self.catalog.register(
            name=f"{name}_v{version}",
            computation=computation,
            previous_version=f"{name}_v{version-1}",
            migration_notes=migration_notes,
            # Mark old version as "deprecated" but still active
        )
```

### Feature Lineage and Impact Analysis

```yaml
Lineage:
  what: "Track where features come from and what depends on them"
  
  upstream_lineage:
    purpose: "Know what data sources feed into a feature"
    tracks:
      - "Source tables (postgres.orders, kafka.events)"
      - "Intermediate transformations (dbt models, Spark jobs)"
      - "Other features this depends on"
    use_case: "Source system changing → which features are affected?"
    
  downstream_lineage:
    purpose: "Know what models/systems consume a feature"
    tracks:
      - "Models using this feature (model_id, version)"
      - "Other features depending on this one"
      - "Dashboards or reports using this feature"
    use_case: "Feature breaking → which models are impacted? Who to alert?"
    
  impact_analysis:
    scenario: "Upstream table schema change"
    process:
      1: "Source team announces: orders.total → orders.amount_cents"
      2: "Lineage query: What features read from orders.total?"
      3: "Result: user_purchase_total_30d, user_avg_order_value_90d"
      4: "Downstream query: What models use these features?"
      5: "Result: churn_model_v4, ltv_model_v2, pricing_model"
      6: "Action: Update pipeline + notify model owners"
      
  deprecation_flow:
    scenario: "Deprecating a feature"
    process:
      1: "Check downstream models using this feature"
      2: "Notify all consuming teams (with migration timeline)"
      3: "Provide replacement feature (if applicable)"
      4: "Add deprecation warning to catalog"
      5: "After migration period: stop computing, remove from online store"
```

### Feature Store Platforms

```yaml
Platforms:
  feast:
    type: "Open-source feature store"
    features: "Feature registry, online/offline serving, SDK"
    catalog: "YAML-based feature definitions in Git"
    discovery: "Registry API + metadata search"
    best_for: "Teams wanting open-source, Kubernetes-native"
    
  tecton:
    type: "Managed feature platform"
    features: "Real-time features, feature monitoring, workspace collaboration"
    catalog: "Web UI with search, lineage, usage metrics"
    discovery: "Full-text search + similarity + recommendation"
    best_for: "Enterprise teams wanting managed service"
    
  databricks_feature_store:
    type: "Integrated with Databricks/Unity Catalog"
    features: "Built into notebook workflow, MLflow integration"
    catalog: "Unity Catalog (data + features + models in one catalog)"
    discovery: "Search via Unity Catalog, automatic lineage"
    best_for: "Teams already on Databricks"
    
  vertex_ai_feature_store:
    type: "GCP managed feature store"
    features: "BigQuery integration, online serving, monitoring"
    catalog: "GCP console + API"
    discovery: "Metadata search, BigQuery lineage"
    best_for: "Teams on GCP wanting managed infrastructure"
    
  sagemaker_feature_store:
    type: "AWS managed feature store"
    features: "Online/offline, Glue integration"
    catalog: "Feature groups with metadata"
    discovery: "Search API, SageMaker Studio UI"
    best_for: "Teams on AWS with SageMaker workflows"
```

---

## How It Works in Practice

### Feature Discovery Workflow

```yaml
Workflow:
  new_model_development:
    1_define_task: "I'm building a churn prediction model"
    2_search_catalog: 
      - "Search: 'user activity' → find existing activity features"
      - "Search: 'engagement' → find engagement features"
      - "Search: 'churn' → see what features previous churn models used"
    3_evaluate_features:
      - "Check quality metrics (freshness, completeness)"
      - "Check if feature is still maintained (active pipeline)"
      - "Check if feature matches my entity (user_id)"
    4_select_shared: "Reuse 15 features from existing groups"
    5_identify_gaps: "Need 3 custom features (no existing ones fit)"
    6_build_custom: "Build only the 3 new features"
    7_register: "Add 3 new features to catalog for others to reuse"
    
  time_saved: "15 reused features × 4 hours each = 60 hours saved"
  consistency: "Same features across models = consistent user representation"
```

---

## Interview Tip

> When asked about feature reuse or team collaboration: "I advocate for a feature catalog as the foundation of team-scale ML. Without it, you get duplicated effort (multiple teams building the same feature with slightly different logic), inconsistency (same concept computed differently across models), and wasted compute (3 separate pipelines for essentially the same feature). My approach: centralized feature registry with documentation standards (every feature has: description, computation logic, entity, freshness SLA, quality metrics, downstream dependencies). Search enables discovery (text search + semantic search + 'features similar to X'). Feature groups bundle related features for easy reuse — a team building churn prediction imports UserActivityFeatureGroup instead of building purchase_count from scratch. Lineage tracking shows upstream (which sources feed this feature) and downstream (which models break if this feature fails) — critical for safe schema changes and deprecation. Versioning lets feature logic evolve without breaking existing models: v1 continues computing while models migrate to v2. The key metric: feature reuse rate. Target >60% of features in any new model being reused from existing catalog."

---

## Common Mistakes

1. **No naming convention** — Features named `feat1`, `johns_feature`, `temp_v3`, `user_thing`. Nobody can find or understand them. 6 months later, the author left and the feature is a black box. Solution: enforce naming convention (`entity_metric_window_aggregation`) from day one. Reject registrations that don't follow the pattern.

2. **Documentation written once, never updated** — Feature description says "uses orders table" but pipeline was migrated to events stream 6 months ago. Documentation lies. Solution: auto-generate documentation from pipeline code where possible. Link catalog entries to pipeline definitions (single source of truth). Alert when metadata diverges from reality.

3. **No deprecation process** — Team removes a feature they think nobody uses → 3 production models start failing. Solution: downstream lineage tracking. Before removing any feature: check consumers, notify them, provide migration path, wait for migration, then remove.

4. **Feature catalog is optional** — Catalog exists but nobody's required to register features there. Teams keep features in their own notebooks/pipelines. Catalog becomes stale and untrustworthy. Solution: make the catalog the ONLY way to access features in production. Models can only read from registered features → 100% coverage.

5. **Over-investing in discovery without quality** — Beautiful UI with search and recommendations, but features have no freshness SLAs, no quality metrics, no monitoring. Teams find features but can't trust them. Solution: quality metrics first (freshness, completeness, stability), discovery UI second. A trustworthy catalog with basic search beats a flashy catalog with unreliable features.

---

## Key Takeaways

- Feature catalog: centralized registry with documentation, lineage, quality metrics, and usage tracking
- Discovery: text search, semantic search, similarity search, task-based recommendations
- Naming conventions: `entity_metric_window_aggregation` — consistent, descriptive, searchable
- Feature groups: bundle related features for easy reuse across teams and models
- Versioning: evolve feature logic without breaking existing consumers (v1 and v2 coexist)
- Lineage: upstream (source tracking) + downstream (impact analysis, safe deprecation)
- Feature reuse rate: target >60% of features in new models from existing catalog
- Quality metrics in catalog: freshness, completeness, stability, coverage
- Integration: catalog must be required (not optional) for production model features
- Platforms: Feast (OSS), Tecton (managed), Databricks Feature Store, Vertex AI, SageMaker
