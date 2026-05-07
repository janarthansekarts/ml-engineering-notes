# ML Testing Fundamentals

## The Problem / Why This Matters

Traditional software testing verifies deterministic logic: given input X, expect output Y. ML systems break this paradigm — models are probabilistic, trained on data, and their behavior changes when retrained. You can't write a unit test that says "given this image, the model MUST predict cat" because model behavior depends on training data, randomness, and optimization. Yet ML systems still need testing — arguably more than traditional software because failures are subtle (slightly wrong predictions, gradually degrading accuracy, bias emerging over time) rather than obvious crashes. ML testing encompasses: data testing (is input data valid and unbiased?), model testing (does the model behave correctly across known scenarios?), integration testing (does the full pipeline work end-to-end?), performance testing (is inference fast enough?), and fairness testing (is the model equitable?). In 2026, with ML systems making consequential decisions (credit, hiring, healthcare, autonomous driving), rigorous testing is not optional — it's a regulatory requirement under the EU AI Act, NIST AI RMF (Risk Management Framework), and industry-specific standards. The testing pyramid for ML adds data and model layers that don't exist in traditional software.

---

## The Analogy

Think of ML testing like quality control in a pharmaceutical factory:

- **Traditional software testing** = Testing a calculator. Input 2+2, verify output is 4. Deterministic, repeatable, binary (pass/fail).
- **ML testing** = Testing a new drug. Does it work on average (efficacy)? Does it work for all patient groups (fairness)? Are there dangerous side effects (safety)? Does it interact badly with other drugs (integration)? Does it still work after 6 months on the shelf (stability)?

Drug testing requires: clinical trials on diverse populations (evaluation datasets), monitoring for side effects (bias testing), verifying manufacturing consistency (reproducibility), and post-market surveillance (production monitoring). ML testing needs all of these too.

---

## Deep Dive

### The ML Testing Pyramid

```yaml
ML_Testing_Pyramid:
  # Bottom of pyramid (most tests, cheapest, fastest)
  level_1_data_tests:
    what: "Validate input data quality and schema"
    examples:
      - "No null values in required columns"
      - "Feature values within expected ranges"
      - "Schema matches expected (column names, types)"
      - "Row count within expected range"
      - "Distribution not drastically different from training"
    tools: "Great Expectations, Pandera, Deequ"
    speed: "Seconds"
    quantity: "Many (dozens per pipeline)"
    
  level_2_model_unit_tests:
    what: "Test model behavior on specific known scenarios"
    examples:
      - "Invariance: changing capitalization doesn't change sentiment"
      - "Directional: increasing income → higher credit score"
      - "Minimum functionality: model handles empty input gracefully"
      - "Known edge cases: extremely long text, rare categories"
    tools: "pytest, CheckList, custom assertions"
    speed: "Seconds to minutes"
    quantity: "Moderate (10-50 per model)"
    
  level_3_model_evaluation:
    what: "Statistical evaluation on held-out test data"
    examples:
      - "Accuracy >= threshold on test set"
      - "F1 >= threshold per class"
      - "No subgroup has >5% worse performance than overall"
      - "New model >= current production model"
    tools: "scikit-learn metrics, custom evaluation pipelines"
    speed: "Minutes"
    quantity: "Few (5-10 evaluation metrics)"
    
  level_4_integration_tests:
    what: "Test full pipeline end-to-end"
    examples:
      - "Feature pipeline → training → serving → prediction (correct output)"
      - "Request → preprocess → inference → postprocess → response"
      - "Model + feature store + monitoring work together"
    tools: "pytest, Testcontainers, staging environment"
    speed: "Minutes to hours"
    quantity: "Few (3-5 critical paths)"
    
  level_5_system_tests:
    what: "Test in production-like environment"
    examples:
      - "Load testing (handles expected traffic)"
      - "A/B testing (new model vs current)"
      - "Shadow deployment (new model doesn't crash under real traffic)"
      - "Chaos testing (system degrades gracefully)"
    tools: "Locust, custom A/B framework, shadow deploy"
    speed: "Hours to days"
    quantity: "Very few (1-2 per deployment cycle)"
```

### What to Test in ML

```yaml
Testing_Dimensions:
  data_quality:
    schema: "Column names, types, nullable constraints"
    completeness: "Null rates within acceptable bounds"
    freshness: "Data is recent enough (not stale)"
    volume: "Expected number of rows (not empty, not duplicated)"
    distribution: "Feature distributions within expected range"
    consistency: "Cross-column rules (start_date < end_date)"
    
  model_behavior:
    invariance: "Output shouldn't change for irrelevant input changes"
    directional: "Known relationships hold (more X → more Y)"
    minimum_functionality: "Model handles basic cases correctly"
    negation: "Model handles negation (not good ≠ good)"
    edge_cases: "Empty input, very long input, special characters"
    robustness: "Small perturbations don't cause large output changes"
    
  model_performance:
    overall_metrics: "Accuracy, precision, recall, F1, AUC-ROC"
    per_class_metrics: "Performance for each output class"
    per_segment_metrics: "Performance for each demographic group"
    calibration: "Predicted probabilities match actual frequencies"
    threshold_sensitivity: "How much does performance change with threshold?"
    
  operational:
    latency: "p50, p95, p99 inference time within SLA"
    throughput: "Handles expected QPS without degradation"
    memory: "Model fits within memory budget"
    startup: "Model loads within acceptable time"
    graceful_degradation: "Handles errors without crashing"
    
  fairness:
    demographic_parity: "Same acceptance rate across groups"
    equal_opportunity: "Same true positive rate across groups"
    predictive_equality: "Same false positive rate across groups"
    individual_fairness: "Similar individuals get similar predictions"
```

### Testing Infrastructure

```python
# ML testing framework — standardized test structure

"""
Standard testing infrastructure for ML models.
Run in CI/CD before model deployment.
"""

import pytest
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable


@dataclass
class MLTestSuite:
    """
    Defines a complete test suite for an ML model.
    Run before promoting model from staging → production.
    """
    model_name: str
    model_version: str
    tests: list  # List of test functions


class MLTestRunner:
    """
    Runs ML test suites and generates reports.
    
    Test categories:
    - MUST PASS (blocking): Model cannot deploy if these fail
    - SHOULD PASS (warning): Alert team but don't block deployment
    - INFORMATIONAL: Track metrics over time, no pass/fail
    """
    
    def __init__(self, model, test_data: pd.DataFrame):
        self.model = model
        self.test_data = test_data
        self.results = []
    
    def run_data_tests(self, data: pd.DataFrame) -> list:
        """Run all data quality tests."""
        results = []
        
        # Schema test
        results.append(self._test(
            name="schema_valid",
            blocking=True,
            passed=self._check_schema(data),
            details="All expected columns present with correct types",
        ))
        
        # Null rate test
        null_rates = data.isnull().mean()
        max_null = null_rates.max()
        results.append(self._test(
            name="null_rate_acceptable",
            blocking=True,
            passed=max_null < 0.05,  # Less than 5% nulls
            details=f"Max null rate: {max_null:.3f} (threshold: 0.05)",
        ))
        
        # Row count test
        results.append(self._test(
            name="sufficient_rows",
            blocking=True,
            passed=len(data) >= 1000,
            details=f"Row count: {len(data)} (minimum: 1000)",
        ))
        
        # Distribution test (no drastic shifts from training)
        results.append(self._test(
            name="distribution_stable",
            blocking=False,  # Warning, not blocking
            passed=self._check_distribution_stability(data),
            details="Feature distributions within 2σ of training distribution",
        ))
        
        return results
    
    def run_behavioral_tests(self) -> list:
        """Run behavioral model tests (invariance, directional, etc.)."""
        results = []
        
        # Invariance tests
        results.extend(self._invariance_tests())
        
        # Directional tests
        results.extend(self._directional_tests())
        
        # Minimum functionality tests
        results.extend(self._minimum_functionality_tests())
        
        return results
    
    def run_performance_tests(self) -> list:
        """Run statistical performance tests."""
        results = []
        
        predictions = self.model.predict(self.test_data.drop("label", axis=1))
        labels = self.test_data["label"]
        
        # Overall accuracy
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(labels, predictions)
        results.append(self._test(
            name="accuracy_threshold",
            blocking=True,
            passed=accuracy >= 0.90,
            details=f"Accuracy: {accuracy:.4f} (threshold: 0.90)",
        ))
        
        # F1 score
        f1 = f1_score(labels, predictions, average="weighted")
        results.append(self._test(
            name="f1_threshold",
            blocking=True,
            passed=f1 >= 0.85,
            details=f"F1 score: {f1:.4f} (threshold: 0.85)",
        ))
        
        # No class has F1 < 0.70 (minimum per-class performance)
        per_class_f1 = f1_score(labels, predictions, average=None)
        min_class_f1 = per_class_f1.min()
        results.append(self._test(
            name="min_class_f1",
            blocking=True,
            passed=min_class_f1 >= 0.70,
            details=f"Minimum per-class F1: {min_class_f1:.4f} (threshold: 0.70)",
        ))
        
        return results
    
    def run_fairness_tests(self, protected_attribute: str) -> list:
        """Run fairness tests across protected groups."""
        results = []
        
        groups = self.test_data[protected_attribute].unique()
        group_metrics = {}
        
        for group in groups:
            group_data = self.test_data[self.test_data[protected_attribute] == group]
            group_preds = self.model.predict(group_data.drop(["label", protected_attribute], axis=1))
            group_labels = group_data["label"]
            
            from sklearn.metrics import accuracy_score
            group_metrics[group] = accuracy_score(group_labels, group_preds)
        
        # Four-fifths rule
        max_rate = max(group_metrics.values())
        min_rate = min(group_metrics.values())
        ratio = min_rate / max_rate if max_rate > 0 else 0
        
        results.append(self._test(
            name=f"fairness_four_fifths_{protected_attribute}",
            blocking=True,
            passed=ratio >= 0.8,
            details=f"Adverse impact ratio: {ratio:.3f} (threshold: 0.8). Groups: {group_metrics}",
        ))
        
        return results
    
    def _test(self, name: str, blocking: bool, passed: bool, details: str) -> dict:
        """Create a test result."""
        return {
            "name": name,
            "blocking": blocking,
            "passed": passed,
            "details": details,
        }
    
    def generate_report(self) -> dict:
        """Generate comprehensive test report."""
        all_results = (
            self.run_data_tests(self.test_data) +
            self.run_behavioral_tests() +
            self.run_performance_tests()
        )
        
        blocking_failures = [r for r in all_results if r["blocking"] and not r["passed"]]
        warnings = [r for r in all_results if not r["blocking"] and not r["passed"]]
        
        return {
            "model": self.model_name,
            "version": self.model_version,
            "total_tests": len(all_results),
            "passed": sum(1 for r in all_results if r["passed"]),
            "failed": sum(1 for r in all_results if not r["passed"]),
            "blocking_failures": blocking_failures,
            "warnings": warnings,
            "can_deploy": len(blocking_failures) == 0,
            "details": all_results,
        }
```

### Testing in CI/CD

```yaml
CI_CD_Integration:
  when_to_run:
    on_code_change: "Unit tests, behavioral tests (fast)"
    on_data_change: "Data validation, distribution tests"
    on_model_retrain: "Full test suite (performance, fairness, integration)"
    pre_deployment: "Performance + fairness + integration (gate deployment)"
    
  pipeline:
    steps:
      1_data_validation:
        what: "Validate training/test data quality"
        blocks_if_fails: true
        duration: "< 1 minute"
        
      2_model_unit_tests:
        what: "Behavioral tests (invariance, directional)"
        blocks_if_fails: true
        duration: "< 5 minutes"
        
      3_performance_evaluation:
        what: "Accuracy, F1, AUC on test set"
        blocks_if_fails: true
        duration: "5-30 minutes (depends on test set size)"
        
      4_fairness_evaluation:
        what: "Subgroup performance, adverse impact"
        blocks_if_fails: true  # For high-risk models
        duration: "5-15 minutes"
        
      5_integration_test:
        what: "Full pipeline end-to-end"
        blocks_if_fails: true
        duration: "10-30 minutes"
        
      6_performance_test:
        what: "Latency, throughput under load"
        blocks_if_fails: true
        duration: "5-15 minutes"
        
  test_data_management:
    golden_dataset: "Curated test set that never changes (baseline comparison)"
    versioned: "Test set versioned alongside model (reproducibility)"
    representative: "Test set must represent production distribution"
    edge_cases: "Include known difficult cases (adversarial, boundary)"
```

---

## How It Works in Practice

### Testing Workflow

```yaml
Workflow:
  development:
    - "Data scientist trains model in notebook"
    - "Runs quick behavioral tests (invariance, directional) → passes"
    - "Registers model candidate"
    
  pre_staging:
    - "CI runs: data validation + behavioral tests + performance eval"
    - "If passes → model promoted to staging"
    
  pre_production:
    - "CI runs: full test suite (behavioral + performance + fairness + integration)"
    - "Human reviews: test report, Model Card, fairness results"
    - "If passes → model deployed to production (canary)"
    
  post_production:
    - "Continuous monitoring (drift, performance, fairness)"
    - "If monitoring detects degradation → trigger re-evaluation"
```

---

## Interview Tip

> When asked about ML testing: "I use a testing pyramid for ML with five layers, from most to least frequent: (1) Data tests — validate schema, null rates, distributions, and volume. Run on every pipeline execution (Great Expectations). These catch the most common production issues (bad data causing bad predictions). (2) Behavioral tests — invariance (changing irrelevant features shouldn't change output), directional (known relationships hold), and minimum functionality (handles edge cases). Run on every code change. (3) Statistical evaluation — accuracy, F1, AUC on held-out test set. Must meet minimum thresholds AND not regress vs. current production model. Run on every model retrain. (4) Fairness tests — four-fifths rule across protected groups, equal opportunity, per-segment performance. Blocking for high-risk models (credit, hiring). Run before production promotion. (5) Integration and performance tests — end-to-end pipeline, latency under load, throughput. Run before deployment. The key insight: ML testing is mostly NON-deterministic. I can't assert 'prediction must be exactly 0.73.' Instead I assert 'accuracy must be > 0.90 on this test set' or 'changing name shouldn't change credit score by more than 0.01.' Tests verify statistical properties and behavioral expectations, not exact outputs."

---

## Common Mistakes

1. **Only testing accuracy on one test set** — Model passes with 95% accuracy on the overall test set, but has 60% accuracy on a critical minority subgroup. Solution: always evaluate per-subgroup (demographic, segment, edge case). A model that's 95% overall but 60% for some users is not acceptable.

2. **No behavioral tests** — Relying only on aggregate metrics. Model has high accuracy but behaves unexpectedly on specific patterns (e.g., "not great" classified as positive sentiment). Solution: write explicit behavioral tests for known scenarios — invariance, directional expectations, negation handling.

3. **Test set contamination** — Test data leaks into training data (or features computed using test period data). Model looks great in testing, fails in production. Solution: strict temporal splits (train on past, test on future), never shuffle time-series data, validate no data leakage.

4. **Testing only at deployment** — All tests run right before production push. Failure at this stage means weeks of work wasted. Solution: progressive testing — quick tests after every code change, full tests on retrain, comprehensive tests before deployment. Catch issues early.

5. **Not testing data, only models** — Extensive model tests but no data validation. Bad data (nulls, schema changes, stale features) enters pipeline, model trains on garbage, produces garbage predictions. Tests on the model pass (because test data is clean). Solution: data tests are the FIRST layer — validate all input data before any model interaction.

---

## Key Takeaways

- ML testing pyramid: data tests → behavioral tests → performance evaluation → fairness → integration
- Data tests: schema, nulls, distributions, freshness — catch 80% of production issues
- Behavioral tests: invariance, directional, minimum functionality — verify model logic
- Performance evaluation: statistical metrics on held-out data (must exceed thresholds)
- Fairness tests: per-subgroup evaluation, four-fifths rule, equal opportunity
- Non-deterministic: test statistical properties and behavioral expectations, not exact outputs
- CI/CD integration: quick tests on every change, full suite before deployment
- Test data management: versioned, representative, includes edge cases, never contaminated
- Blocking vs. warning: some tests block deployment, others just alert the team
- Per-subgroup evaluation: overall accuracy is insufficient — evaluate every important segment
