# Integration Testing for ML

## The Problem / Why This Matters

ML systems are not just models — they're complex pipelines with many components: data ingestion, feature engineering, preprocessing, model inference, postprocessing, monitoring, and serving. Each component may pass its own unit test individually but fail when connected together. A feature engineering step outputs floats but the model expects integers. The preprocessing code applies transformations in a different order than training. The serving container uses a different library version than training. These integration failures are among the hardest to debug because each component works correctly in isolation — the bug exists only at the boundary between components. Integration testing for ML verifies that the full pipeline works end-to-end: raw input → preprocessing → feature computation → model inference → postprocessing → final output. In 2026, with ML systems spanning multiple services (feature store, model registry, inference server, monitoring), integration testing is essential for catching boundary failures before production deployment.

---

## The Analogy

Think of integration testing like testing a relay race team:

- **Unit testing** = Testing each runner's individual speed. Runner A is fast, Runner B is fast, Runner C is fast.
- **Integration testing** = Testing the actual baton handoff. Can Runner A pass to Runner B smoothly? Does Runner B receive the baton correctly and start running without dropping it? Do all runners run on the correct lanes in the correct order?

A team of individually fast runners can still lose if the baton handoff fails. Similarly, individually correct ML components can still produce wrong results if they don't integrate correctly.

---

## Deep Dive

### What Can Go Wrong at Boundaries

```yaml
Common_Integration_Failures:
  data_format_mismatch:
    example: "Feature pipeline outputs a column named 'user_age'. Model expects 'age'."
    symptom: "KeyError in production, model receives wrong/missing features"
    root_cause: "Training notebook used 'age', feature pipeline uses 'user_age'"
    
  type_mismatch:
    example: "Feature store returns int64. Model expects float32."
    symptom: "Silent incorrect predictions (integer division instead of float)"
    root_cause: "Feature materialization changed type, model not updated"
    
  ordering_mismatch:
    example: "Training features in order [A, B, C]. Serving sends [C, A, B]."
    symptom: "Model runs without error but predictions are meaningless"
    root_cause: "Dictionary/JSON doesn't preserve order, array serialization differs"
    
  preprocessing_skew:
    example: "Training normalizes with mean=5.2. Serving normalizes with mean=5.8 (stale)."
    symptom: "Slightly wrong predictions that slowly get worse"
    root_cause: "Normalization statistics computed once, never updated"
    
  version_mismatch:
    example: "Training: scikit-learn 1.4. Serving: scikit-learn 1.5."
    symptom: "Different behavior for edge cases, occasional crashes"
    root_cause: "No pinned dependency versions between training and serving"
    
  serialization_issues:
    example: "Model saved with pickle in Python 3.11, loaded in Python 3.12."
    symptom: "Deserialization error or silent behavior change"
    root_cause: "Pickle is Python-version dependent, no ONNX/standard format"
    
  network_issues:
    example: "Feature store times out under load, returns partial features."
    symptom: "Model receives nulls for some features, predictions degrade"
    root_cause: "No timeout handling, no retry logic, no fallback"
```

### End-to-End Pipeline Test

```python
# Integration test for ML inference pipeline

"""
End-to-end integration tests for ML pipelines.
Test the full path: raw input → final prediction.
"""

import pytest
import requests
import time
from typing import Dict, Any
import pandas as pd
import numpy as np


class TestMLPipelineIntegration:
    """
    Integration tests for the full ML inference pipeline.
    
    Tests verify that all components work together correctly:
    - Feature store → Feature retrieval
    - Preprocessing → Feature transformation
    - Model loading → Inference
    - Postprocessing → Final output formatting
    - Monitoring → Metrics emitted
    """
    
    @pytest.fixture(scope="class")
    def pipeline_endpoint(self):
        """
        Start the inference pipeline (or connect to staging).
        Uses testcontainers or a staging environment.
        """
        # Option 1: Use staging endpoint
        return "http://staging-ml-api:8080/predict"
        
        # Option 2: Start local with testcontainers (see below)
    
    @pytest.fixture
    def known_good_request(self) -> Dict[str, Any]:
        """A request with known expected output (golden test case)."""
        return {
            "user_id": "test_user_123",
            "features": {
                "age": 35,
                "income": 75000,
                "employment_years": 8,
                "credit_history_months": 120,
                "num_accounts": 3,
            },
            "context": {
                "request_time": "2026-01-15T10:30:00Z",
                "channel": "web",
            }
        }
    
    def test_end_to_end_prediction(self, pipeline_endpoint, known_good_request):
        """
        Test full pipeline produces a valid prediction.
        
        Verifies:
        1. Request is accepted (200 status)
        2. Response has expected schema
        3. Prediction is within valid range
        4. Latency is acceptable
        """
        start = time.time()
        response = requests.post(
            pipeline_endpoint,
            json=known_good_request,
            timeout=5.0
        )
        latency_ms = (time.time() - start) * 1000
        
        # Should succeed
        assert response.status_code == 200, f"Got {response.status_code}: {response.text}"
        
        # Response schema
        result = response.json()
        assert "prediction" in result
        assert "confidence" in result
        assert "model_version" in result
        assert "features_used" in result
        
        # Prediction validity
        assert result["prediction"] in ["approved", "denied", "review"]
        assert 0.0 <= result["confidence"] <= 1.0
        
        # Latency SLA
        assert latency_ms < 200, f"Latency {latency_ms:.0f}ms exceeds 200ms SLA"
    
    def test_feature_store_integration(self, pipeline_endpoint):
        """
        Test that pipeline correctly retrieves features from feature store.
        
        Send request with user_id → pipeline should fetch stored features
        and combine with request features for prediction.
        """
        # User with known features in feature store
        request = {
            "user_id": "known_feature_user",
            "features": {"age": 30},  # Partial — rest should come from feature store
            "context": {"request_time": "2026-01-15T10:30:00Z"}
        }
        
        response = requests.post(pipeline_endpoint, json=request, timeout=5.0)
        result = response.json()
        
        # Verify feature store features were included
        features_used = result["features_used"]
        assert "credit_score" in features_used  # Should come from feature store
        assert "transaction_count_30d" in features_used  # Should come from feature store
    
    def test_preprocessing_consistency(self, pipeline_endpoint, known_good_request):
        """
        Test that preprocessing matches training preprocessing.
        
        Known input → should produce same preprocessed features as training.
        """
        response = requests.post(
            f"{pipeline_endpoint}?debug=true",  # Debug mode returns preprocessed features
            json=known_good_request,
            timeout=5.0
        )
        result = response.json()
        
        # Verify normalized features match expected values
        preprocessed = result.get("debug", {}).get("preprocessed_features", {})
        
        # Income should be normalized (training mean ~60000, std ~30000)
        if "income_normalized" in preprocessed:
            # 75000 normalized with mean=60000, std=30000 → (75000-60000)/30000 = 0.5
            assert abs(preprocessed["income_normalized"] - 0.5) < 0.1
    
    def test_model_version_consistency(self, pipeline_endpoint):
        """
        Test that the deployed model version matches expected version.
        
        Prevents accidentally serving an old model.
        """
        response = requests.get(f"{pipeline_endpoint}/health")
        health = response.json()
        
        assert health["model_version"] == "v2.3.1"  # Expected production version
        assert health["model_loaded"] is True
        assert health["feature_store_connected"] is True
    
    def test_error_handling(self, pipeline_endpoint):
        """Test pipeline handles errors gracefully."""
        
        # Missing required field
        bad_request = {"user_id": "test"}  # Missing features
        response = requests.post(pipeline_endpoint, json=bad_request, timeout=5.0)
        assert response.status_code == 400
        assert "error" in response.json()
        
        # Invalid feature value
        bad_features = {
            "user_id": "test",
            "features": {"age": -5, "income": "not_a_number"},
            "context": {"request_time": "2026-01-15T10:30:00Z"}
        }
        response = requests.post(pipeline_endpoint, json=bad_features, timeout=5.0)
        assert response.status_code in [400, 422]  # Validation error
    
    def test_concurrent_requests(self, pipeline_endpoint, known_good_request):
        """Test pipeline handles concurrent requests correctly."""
        import concurrent.futures
        
        num_requests = 50
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    requests.post,
                    pipeline_endpoint,
                    json=known_good_request,
                    timeout=10.0
                )
                for _ in range(num_requests)
            ]
            
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        success_count = sum(1 for r in responses if r.status_code == 200)
        assert success_count == num_requests, f"Only {success_count}/{num_requests} succeeded"
        
        # All should return same prediction (same input)
        predictions = [r.json()["prediction"] for r in responses if r.status_code == 200]
        assert len(set(predictions)) == 1, "Different predictions for same input!"


class TestTrainingServingConsistency:
    """
    Test that training pipeline and serving pipeline produce the same results.
    
    Training-serving skew is one of the most common ML integration bugs.
    """
    
    def test_feature_computation_matches(self):
        """
        Same raw input should produce same features in training and serving.
        
        Run training feature pipeline + serving feature pipeline on same input.
        Compare outputs.
        """
        raw_input = {
            "user_id": "test_123",
            "transactions": [100, 200, 50, 75, 300],
            "registration_date": "2020-06-15",
        }
        
        # Training feature computation
        from training.features import compute_features as train_compute
        train_features = train_compute(raw_input)
        
        # Serving feature computation
        from serving.features import compute_features as serve_compute
        serve_features = serve_compute(raw_input)
        
        # Should be identical
        for key in train_features:
            assert key in serve_features, f"Feature '{key}' missing in serving"
            np.testing.assert_allclose(
                train_features[key],
                serve_features[key],
                rtol=1e-5,
                err_msg=f"Feature '{key}' differs: train={train_features[key]}, serve={serve_features[key]}"
            )
    
    def test_prediction_matches_training(self):
        """
        Serving prediction should match training prediction for same input.
        
        Load model in both environments, predict same input, compare.
        """
        test_features = np.array([[0.5, 0.3, 0.8, 0.2, 0.6]])
        
        # Training prediction (direct model load)
        import joblib
        training_model = joblib.load("artifacts/model_v2.3.1.pkl")
        training_pred = training_model.predict_proba(test_features)
        
        # Serving prediction (via API)
        response = requests.post(
            "http://staging-ml-api:8080/predict-raw",
            json={"features": test_features.tolist()}
        )
        serving_pred = np.array(response.json()["probabilities"])
        
        # Should match within floating point tolerance
        np.testing.assert_allclose(
            training_pred, serving_pred, rtol=1e-4,
            err_msg="Training-serving prediction mismatch!"
        )
```

### Testcontainers for ML

```python
# Using Testcontainers to spin up ML infrastructure for testing

"""
Testcontainers provides Docker containers for integration testing.
Spin up real dependencies (Redis, PostgreSQL, model server) in tests.
"""

import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer


class MLInfraContainers:
    """
    Manage ML infrastructure containers for integration tests.
    
    Spins up:
    - Feature store (Redis)
    - Metadata store (PostgreSQL)
    - Model server (custom container)
    """
    
    @pytest.fixture(scope="session")
    def feature_store(self):
        """Spin up Redis for feature store."""
        with RedisContainer("redis:7-alpine") as redis:
            # Pre-populate with test features
            import redis as redis_client
            client = redis_client.Redis(
                host=redis.get_container_host_ip(),
                port=redis.get_exposed_port(6379),
            )
            
            # Seed test features
            client.hset("user:test_123", mapping={
                "credit_score": "720",
                "transaction_count_30d": "45",
                "avg_transaction_amount": "250.50",
            })
            
            yield client
    
    @pytest.fixture(scope="session")
    def metadata_db(self):
        """Spin up PostgreSQL for model metadata."""
        with PostgresContainer("postgres:16-alpine") as pg:
            # Run migrations
            import sqlalchemy
            engine = sqlalchemy.create_engine(pg.get_connection_url())
            
            # Create tables
            with engine.begin() as conn:
                conn.execute(sqlalchemy.text("""
                    CREATE TABLE models (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255),
                        version VARCHAR(50),
                        artifact_path VARCHAR(500),
                        status VARCHAR(20)
                    )
                """))
                conn.execute(sqlalchemy.text("""
                    INSERT INTO models (name, version, artifact_path, status)
                    VALUES ('credit_model', 'v2.3.1', '/artifacts/credit_v2.3.1', 'production')
                """))
            
            yield engine
    
    @pytest.fixture(scope="session")
    def model_server(self):
        """Spin up model inference server."""
        with DockerContainer("ml-inference:latest") \
            .with_exposed_ports(8080) \
            .with_env("MODEL_PATH", "/models/credit_v2.3.1") \
            .with_env("LOG_LEVEL", "debug") as server:
            
            # Wait for server to be ready
            import time
            import requests
            
            host = server.get_container_host_ip()
            port = server.get_exposed_port(8080)
            url = f"http://{host}:{port}"
            
            for _ in range(30):  # Wait up to 30 seconds
                try:
                    r = requests.get(f"{url}/health")
                    if r.status_code == 200:
                        break
                except:
                    pass
                time.sleep(1)
            
            yield url
```

### Pipeline Contract Tests

```yaml
Pipeline_Contract_Tests:
  what: "Verify contracts between pipeline stages hold"
  
  contracts:
    ingestion_to_feature_engineering:
      producer: "Data ingestion (reads from source)"
      consumer: "Feature engineering (transforms raw data)"
      contract:
        - "Output contains columns: user_id, event_type, amount, timestamp"
        - "user_id is non-null string"
        - "amount is non-negative float"
        - "timestamp is ISO 8601 format"
        
    feature_engineering_to_model:
      producer: "Feature engineering (computes features)"
      consumer: "Model inference (uses features for prediction)"
      contract:
        - "Output contains exactly: feature_1..feature_N (no extra, no missing)"
        - "All values are float32"
        - "No NaN or infinity values"
        - "Feature names match model's expected input"
        
    model_to_postprocessing:
      producer: "Model inference (raw prediction)"
      consumer: "Postprocessing (formats output)"
      contract:
        - "Output is numpy array of shape (batch_size, num_classes)"
        - "Values are probabilities (sum to 1.0 per row)"
        - "No NaN values"
        
  testing_approach:
    consumer_driven: "Consumer defines the contract (what it expects)"
    producer_validates: "Producer runs contract tests in its CI/CD"
    breaking_change: "Failing contract test blocks producer deployment"
```

---

## How It Works in Practice

### Integration Test Strategy

```yaml
Strategy:
  scope_levels:
    narrow_integration:
      what: "Two adjacent components together"
      example: "Feature computation + model inference"
      speed: "Seconds"
      frequency: "Every commit"
      
    broad_integration:
      what: "Full pipeline end-to-end"
      example: "Raw request → final prediction (all components)"
      speed: "Minutes"
      frequency: "Every PR, pre-deployment"
      
    system_integration:
      what: "ML system + dependent services"
      example: "ML API + downstream consumer + monitoring + alerting"
      speed: "Minutes to hours"
      frequency: "Pre-production, weekly"
```

---

## Interview Tip

> When asked about integration testing for ML: "Integration testing for ML verifies components work together correctly — the most common production failures happen at boundaries, not within individual components. My approach: (1) Training-serving consistency tests — verify that the same raw input produces identical features and predictions in both training and serving code paths. Training-serving skew is the #1 integration bug in ML (different preprocessing, different library versions, different feature computation). (2) End-to-end pipeline tests — send a request to the full pipeline (raw input → preprocessing → feature retrieval → model → postprocessing → response) and verify schema, range, and latency of the output. Use known golden test cases where expected output is pre-computed. (3) Contract tests between stages — each pipeline stage has a contract (expected input/output schema). Producer validates it produces correct output; consumer validates it handles the input. Breaking a contract blocks deployment. (4) Testcontainers — spin up real dependencies in Docker (Redis for feature store, PostgreSQL for metadata, model server) for realistic integration tests without needing a staging environment. Key insight: unit tests passing doesn't mean the system works. A feature pipeline can correctly compute features, and a model can correctly predict, but if the feature names don't match or types are different, the system fails."

---

## Common Mistakes

1. **No training-serving consistency test** — Feature computation code is duplicated (Python notebook for training, Java for serving). They drift apart over time. Solution: test that same raw input → same features in both environments. Better yet: use a shared feature store.

2. **Testing against mocks only** — Mock the feature store, mock the model, mock postprocessing. All tests pass but the real system fails because mocks don't match reality. Solution: testcontainers or staging environment for integration tests with real services.

3. **Ignoring feature ordering** — Model trained with features [A, B, C] but served with [B, A, C]. Model runs without error (it's just a numpy array) but predictions are wrong. Solution: use named features (DataFrames) or explicitly test feature ordering.

4. **No latency testing in integration** — Pipeline works correctly but takes 5 seconds (SLA is 200ms). Solution: integration tests include latency assertions. If a component adds unexpected latency, catch it before production.

5. **One-time integration test, never updated** — Write integration tests at launch, never update as pipeline evolves. Tests pass but don't test current pipeline logic. Solution: integration tests are maintained alongside pipeline code. New features = new integration tests.

---

## Key Takeaways

- Integration failures happen at boundaries between components, not within them
- Training-serving skew: #1 integration bug — same input must produce same output in both paths
- End-to-end tests: raw input → final prediction, verify schema + range + latency
- Contract tests: formalize expectations between pipeline stages, test both sides
- Testcontainers: spin up real Docker dependencies for realistic testing
- Golden test cases: pre-computed expected outputs for known inputs
- Test concurrent requests: ML systems must handle parallelism correctly
- Include latency assertions: correct but slow is still a failure
- Test error handling: bad inputs should produce informative errors, not crashes
- Integration tests complement unit tests — both are necessary
