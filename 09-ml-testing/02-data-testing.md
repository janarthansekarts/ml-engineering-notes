# Data Testing

## The Problem / Why This Matters

In ML systems, "garbage in, garbage out" is not just a cliché — it's the primary cause of production failures. Studies consistently show that 60-80% of ML production incidents trace back to data issues, not model issues. A model trained on correct data deployed with corrupted input features will produce confident but wrong predictions. Unlike traditional software where bad input typically causes a crash (obvious failure), ML models happily process bad data and return plausible-looking predictions (silent failure). Data testing validates the quality, schema, distribution, and freshness of data at every stage: ingestion, transformation, training, and serving. In 2026, data testing is formalized through tools like Great Expectations (now GX), Pandera, Deequ (Apache Spark), and data contracts. These provide declarative "expectations" about data that run automatically in pipelines, catch issues before they corrupt models, and serve as documentation for data consumers.

---

## The Analogy

Think of data testing like food safety inspection in a restaurant kitchen:

- **No data testing** = No food inspection. You hope ingredients are fresh, hope nothing is contaminated, hope prep is hygienic. Sometimes customers get food poisoning but you never know which ingredient caused it.
- **Data testing** = Systematic food safety. Check temperature of incoming deliveries (schema validation). Inspect expiration dates (freshness). Verify no cross-contamination (consistency). Sample test for bacteria (statistical validation). Every stage has checks before food reaches the customer.

Both catch problems BEFORE they reach the end user, and both are easier/cheaper to fix at the source than after serving.

---

## Deep Dive

### Data Test Categories

```yaml
Data_Test_Categories:
  schema_tests:
    what: "Verify data structure matches expectations"
    checks:
      - "All expected columns present"
      - "Column data types correct (int, float, string, datetime)"
      - "No unexpected new columns (schema evolution detection)"
      - "Column order (if order-dependent processing)"
    why: "Schema changes from upstream break feature pipelines silently"
    
  completeness_tests:
    what: "Verify data is sufficiently complete"
    checks:
      - "Null rate per column < threshold (e.g., < 5%)"
      - "No entirely empty columns"
      - "Required fields are never null"
      - "Total row count within expected range"
    why: "Nulls propagate through feature computation → NaN predictions"
    
  freshness_tests:
    what: "Verify data is recent enough"
    checks:
      - "Most recent timestamp < X hours ago"
      - "No data gaps > Y hours"
      - "Partition date matches expected date"
    why: "Stale data means model predicts based on outdated reality"
    
  distribution_tests:
    what: "Verify feature distributions are stable"
    checks:
      - "Mean/median within expected range"
      - "Standard deviation within bounds"
      - "Quantiles stable (not shifted)"
      - "Category proportions stable (no sudden disappearance)"
      - "KL divergence from reference < threshold"
    why: "Distribution shifts indicate upstream changes or data drift"
    
  consistency_tests:
    what: "Verify logical relationships hold"
    checks:
      - "start_date <= end_date"
      - "If status='completed', completion_date is not null"
      - "age >= 0 and age <= 150"
      - "Referential integrity (foreign keys exist)"
    why: "Inconsistent data indicates upstream bugs or corruption"
    
  uniqueness_tests:
    what: "Verify no unexpected duplicates"
    checks:
      - "Primary key columns are unique"
      - "No exact duplicate rows (or within tolerance)"
      - "ID columns have expected cardinality"
    why: "Duplicates bias model training and inflate metrics"
```

### Great Expectations (GX)

```python
# Data testing with Great Expectations (GX) — the industry standard

"""
Great Expectations provides a declarative framework for
data validation with rich documentation and CI/CD integration.
"""

import great_expectations as gx
from great_expectations.core import ExpectationSuite


def create_training_data_expectations() -> ExpectationSuite:
    """
    Define expectations for training data.
    These run before model training to catch data issues early.
    """
    
    context = gx.get_context()
    
    # Create or get expectation suite
    suite = context.add_or_update_expectation_suite(
        expectation_suite_name="training_data_v2"
    )
    
    # Schema expectations
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchOrderedList(
            column_list=[
                "user_id", "feature_1", "feature_2", "feature_3",
                "category", "timestamp", "label"
            ]
        )
    )
    
    # Completeness expectations
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="user_id"
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="label"
        )
    )
    # Allow up to 2% nulls in feature_1
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="feature_1",
            mostly=0.98  # At least 98% non-null
        )
    )
    
    # Distribution expectations
    suite.add_expectation(
        gx.expectations.ExpectColumnMeanToBeBetween(
            column="feature_1",
            min_value=0.4,
            max_value=0.6
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnStdevToBeBetween(
            column="feature_1",
            min_value=0.05,
            max_value=0.20
        )
    )
    
    # Range expectations
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="feature_2",
            min_value=0,
            max_value=1000
        )
    )
    
    # Category expectations
    suite.add_expectation(
        gx.expectations.ExpectColumnDistinctValuesToBeInSet(
            column="category",
            value_set=["A", "B", "C", "D"]
        )
    )
    
    # Volume expectations
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=10000,
            max_value=1000000
        )
    )
    
    # Uniqueness expectations
    suite.add_expectation(
        gx.expectations.ExpectCompoundColumnsToBeUnique(
            column_list=["user_id", "timestamp"]
        )
    )
    
    return suite


def create_serving_data_expectations() -> ExpectationSuite:
    """
    Expectations for real-time serving data.
    These run on each prediction request (or batch of requests).
    More lenient than training expectations (production is messy).
    """
    context = gx.get_context()
    suite = context.add_or_update_expectation_suite(
        expectation_suite_name="serving_data_v1"
    )
    
    # Schema (must match exactly — serving pipeline is rigid)
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=["feature_1", "feature_2", "feature_3", "category"]
        )
    )
    
    # No nulls (serving features must be complete — imputation happened upstream)
    for col in ["feature_1", "feature_2", "feature_3", "category"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
        )
    
    # Range (reject obviously wrong values)
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="feature_2",
            min_value=-10,  # Slightly more lenient than training
            max_value=1500
        )
    )
    
    return suite


def run_validation_in_pipeline(data_path: str, suite_name: str) -> bool:
    """
    Run data validation as a pipeline step.
    Returns True if data passes, False if it fails.
    """
    context = gx.get_context()
    
    # Create data source
    datasource = context.sources.add_pandas(name="pipeline_data")
    data_asset = datasource.add_csv_asset(
        name="training_batch",
        filepath_or_buffer=data_path,
    )
    
    # Run validation
    batch_request = data_asset.build_batch_request()
    
    checkpoint = context.add_or_update_checkpoint(
        name="pipeline_checkpoint",
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": suite_name,
            }
        ],
    )
    
    result = checkpoint.run()
    
    # Generate data docs (HTML report)
    context.build_data_docs()
    
    return result.success
```

### Pandera (Pandas Schema Validation)

```python
# Pandera — lightweight schema validation for pandas DataFrames

"""
Pandera provides type annotations and runtime validation for DataFrames.
Lighter weight than Great Expectations, ideal for in-code validation.
"""

import pandera as pa
from pandera import Column, Check, DataFrameSchema
import pandas as pd


# Define schema as a class (recommended approach)
class TrainingDataSchema(pa.DataFrameModel):
    """
    Schema for training data.
    Validates on creation with @pa.check_types decorator.
    """
    user_id: pa.typing.Series[str] = pa.Field(nullable=False, unique=True)
    feature_1: pa.typing.Series[float] = pa.Field(
        ge=0.0, le=1.0,  # Between 0 and 1
        nullable=False
    )
    feature_2: pa.typing.Series[float] = pa.Field(
        ge=0, le=1000,
        nullable=True,
        coerce=True  # Coerce to float if needed
    )
    feature_3: pa.typing.Series[float] = pa.Field(
        nullable=True
    )
    category: pa.typing.Series[str] = pa.Field(
        isin=["A", "B", "C", "D"],
        nullable=False
    )
    label: pa.typing.Series[int] = pa.Field(
        isin=[0, 1],
        nullable=False
    )
    
    class Config:
        strict = True  # No extra columns allowed
        coerce = True  # Coerce types before validation
    
    # Custom cross-column checks
    @pa.check("feature_1")
    def feature_1_distribution(cls, series):
        """Feature 1 mean should be roughly 0.5."""
        return 0.3 <= series.mean() <= 0.7
    
    @pa.dataframe_check
    def no_duplicate_rows(cls, df):
        """No exact duplicate rows."""
        return ~df.duplicated().any()


# Use in function signatures with @pa.check_types
@pa.check_types
def train_model(data: pa.typing.DataFrame[TrainingDataSchema]):
    """
    Train model — input data is automatically validated.
    If data doesn't match schema, SchemaError is raised BEFORE training starts.
    """
    # Training logic here — data is guaranteed valid
    features = data[["feature_1", "feature_2", "feature_3"]]
    labels = data["label"]
    # model.fit(features, labels)
    pass


# Alternative: imperative schema definition
serving_schema = DataFrameSchema(
    columns={
        "feature_1": Column(float, Check.between(0, 1), nullable=False),
        "feature_2": Column(float, Check.between(0, 1000), nullable=False),
        "feature_3": Column(float, nullable=True),
        "category": Column(str, Check.isin(["A", "B", "C", "D"]), nullable=False),
    },
    strict=True,
    coerce=True,
)


def validate_serving_request(features: pd.DataFrame) -> bool:
    """Validate a serving request before prediction."""
    try:
        serving_schema.validate(features)
        return True
    except pa.errors.SchemaError as e:
        # Log validation error, return fallback prediction or error
        print(f"Validation failed: {e}")
        return False
```

### Data Contracts

```yaml
Data_Contracts:
  what: "Formal agreement between data producer and consumer on data shape and quality"
  why: "Without contracts, upstream schema changes break downstream ML pipelines silently"
  
  contract_elements:
    schema:
      columns: "Name, type, nullable, description"
      primary_key: "Which columns uniquely identify a row"
      
    quality_sla:
      freshness: "Data available within X hours of event"
      completeness: "< Y% null rate for required columns"
      volume: "Expected row count range per day/hour"
      
    distribution:
      stability: "Column means/stdevs don't shift > Z% week-over-week"
      categories: "Set of valid category values"
      
    ownership:
      producer: "Team responsible for generating this data"
      consumers: "Teams that depend on this data (notify on changes)"
      breaking_change_process: "How to communicate schema changes"
      
  enforcement:
    producer_side: "Data contract tests run in producer's CI/CD"
    consumer_side: "Consumer validates data matches contract on receipt"
    alerting: "Both sides alerted if contract is violated"
    
  tools:
    soda_core: "Data quality testing and data contracts"
    datahub: "Data catalog with data contracts"
    great_expectations: "Expectation suites as contracts"
    custom: "YAML contract definitions + automated test generation"
```

### Training vs Serving Data Tests

```yaml
Training_vs_Serving:
  training_data_tests:
    purpose: "Ensure training data quality before model training"
    timing: "Before training starts (gate on training pipeline)"
    strictness: "Moderate (some issues can be handled by preprocessing)"
    unique_checks:
      - "Label quality (no label noise above threshold)"
      - "Class balance (imbalanced but within acceptable range)"
      - "Temporal correctness (no future data leaking into training)"
      - "No data leakage (features don't contain label information)"
      
  serving_data_tests:
    purpose: "Ensure real-time input quality before prediction"
    timing: "At prediction time (< 5ms overhead)"
    strictness: "Strict (bad input → reject or fallback, never bad prediction)"
    unique_checks:
      - "All required features present and non-null"
      - "Feature values within trained range (out-of-range → flag)"
      - "Feature freshness (features computed recently enough)"
      - "No impossible combinations (logical consistency)"
      
  training_serving_skew_tests:
    purpose: "Detect drift between training data and serving data"
    timing: "Continuous (hourly/daily comparison)"
    checks:
      - "Feature distributions in serving match training"
      - "Category values in serving are subset of training"
      - "Feature engineering logic is identical (same transformations)"
```

---

## How It Works in Practice

### Integration in ML Pipeline

```yaml
Pipeline_Integration:
  data_ingestion:
    action: "Load raw data from source"
    test: "Schema validation + freshness check"
    on_failure: "Alert data engineering team, skip pipeline run"
    
  feature_engineering:
    action: "Transform raw data into features"
    test: "Output schema + range checks + null checks"
    on_failure: "Block training, investigate transformation logic"
    
  train_test_split:
    action: "Split data for training/evaluation"
    test: "No temporal leakage, distribution similarity across splits"
    on_failure: "Fix split logic"
    
  pre_training:
    action: "Final validation before training starts"
    test: "Full expectation suite (schema + quality + distribution + consistency)"
    on_failure: "Block training, generate data quality report"
    
  serving:
    action: "Receive prediction request"
    test: "Lightweight schema + range validation (< 5ms)"
    on_failure: "Return error response or fallback prediction"
```

---

## Interview Tip

> When asked about data testing: "I validate data at every stage of the ML pipeline because 60-80% of production ML issues trace to data problems, not model problems. My approach: (1) Schema tests — verify column names, types, nullable constraints. Catches upstream schema changes instantly. (2) Completeness — null rates below thresholds, required fields never null, sufficient volume. (3) Distribution stability — means, standard deviations, quantiles within expected range. I use KL (Kullback-Leibler) divergence or KS (Kolmogorov-Smirnov) tests to detect shifts. (4) Consistency — logical rules like start_date < end_date, mutually exclusive categories. (5) Freshness — most recent timestamp within expected window, no data gaps. I use Great Expectations for comprehensive pipeline validation (declarative expectations, auto-generated docs, CI/CD integration) and Pandera for lightweight in-code schema validation. Key insight: training data tests and serving data tests are different. Training tests can be lenient (preprocessing handles issues). Serving tests must be strict and fast (< 5ms overhead — reject bad input immediately). I also implement data contracts between producers and consumers so schema changes are communicated before they break pipelines."

---

## Common Mistakes

1. **Only testing schema, not distributions** — Schema passes (all columns present, types correct) but values are wrong (negative ages, future dates, feature means shifted 10x). Solution: test both structure AND statistical properties of values.

2. **Same tests for training and serving** — Heavy statistical tests add 500ms latency to serving requests. Solution: serving tests are lightweight (schema + range only, < 5ms). Full statistical validation happens in offline pipeline.

3. **No freshness testing** — Pipeline runs successfully on stale data (yesterday's features, not today's). Model predicts based on outdated reality. Solution: always check data freshness (most recent timestamp within expected window).

4. **Testing only the final output** — Validate the training DataFrame but not intermediate stages. Corruption introduced in step 2 of 5 is discovered at step 5 with no clue where it happened. Solution: validate at each pipeline stage (test early, test often).

5. **Hardcoded thresholds that never update** — Set "mean must be between 0.4 and 0.6" once, never update. Data legitimately evolves over time, tests start failing or become too lenient. Solution: periodically update baselines from recent production data (monthly/quarterly).

---

## Key Takeaways

- 60-80% of ML production issues are data issues — data testing is the highest-ROI testing
- Test dimensions: schema, completeness, freshness, distribution, consistency, uniqueness
- Great Expectations: declarative expectations, CI/CD integration, auto-generated docs
- Pandera: lightweight schema validation, integrates with type annotations
- Data contracts: formal agreements between data producers and consumers
- Training tests can be moderate; serving tests must be strict and fast (< 5ms)
- Test at every pipeline stage, not just the final output
- Distribution tests catch drift that schema tests miss
- Training-serving skew: verify feature distributions match between training and production
- Update test baselines periodically as data legitimately evolves
