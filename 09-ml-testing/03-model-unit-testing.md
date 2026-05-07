# Model Unit Testing

## The Problem / Why This Matters

Traditional unit testing verifies exact outputs: `assertEqual(add(2, 3), 5)`. ML models can't be tested this way — they're probabilistic, their outputs depend on training data, and small changes in training can shift predictions. Yet you still need to verify that models behave sensibly. Model unit testing solves this through behavioral testing: instead of testing exact outputs, you test behavioral expectations. "Does the model respond the same way when irrelevant features change?" (invariance). "Does the model respond correctly to known directional relationships?" (directional expectations). "Can the model handle basic known-good cases?" (minimum functionality). These tests are inspired by the CheckList framework from Microsoft Research (2020), which brought systematic behavioral testing to NLP (Natural Language Processing) models. In 2026, behavioral testing is standard practice for any production ML model — especially LLMs (Large Language Models) where outputs are non-deterministic and evaluation is complex.

---

## The Analogy

Think of model unit testing like a driving test for a new driver:

- **Traditional unit test** = Written exam. "What does a red octagonal sign mean?" Exact answer: "Stop." Deterministic, binary.
- **Behavioral model test** = Road test. "Does the driver stop at stop signs?" (invariance — regardless of weather, time of day, or car color). "Does the driver slow down in school zones?" (directional — more caution where children are present). "Can the driver parallel park?" (minimum functionality — basic capabilities work).

You don't test "at this exact intersection at this exact time, the driver turns the wheel exactly 43 degrees." You test behavioral expectations that should hold in all situations.

---

## Deep Dive

### Behavioral Testing Framework (CheckList)

```yaml
CheckList_Framework:
  origin: "Microsoft Research, 2020 (ACL Best Paper)"
  philosophy: "Test model capabilities systematically, not just aggregate accuracy"
  
  test_types:
    MFT_minimum_functionality:
      what: "Test basic capabilities that MUST work"
      examples:
        sentiment: "Clearly positive text → positive sentiment"
        ner: "Common name patterns → identified as PERSON"
        classification: "Obvious category members → correct class"
      expectation: "Correct prediction (exact output check possible)"
      
    INV_invariance:
      what: "Irrelevant changes shouldn't change prediction"
      examples:
        sentiment: "Changing person name shouldn't change sentiment"
        credit: "Changing gender shouldn't change credit decision"
        classification: "Adding typos shouldn't flip category (robustness)"
      expectation: "Output unchanged (or changes < epsilon)"
      
    DIR_directional:
      what: "Known relationships should hold"
      examples:
        sentiment: "Adding 'amazing' should increase positive score"
        pricing: "Increasing square footage should increase price"
        risk: "Adding prior fraud should increase risk score"
      expectation: "Output moves in expected direction"

  template_based_testing:
    what: "Generate many test cases from templates"
    example: "I {VERB_POS} the {PRODUCT} → should be positive"
    fill: "VERB_POS = [loved, enjoyed, adored, cherished]"
    generate: "4 test cases from 1 template"
    scale: "Hundreds of tests from dozens of templates"
```

### Implementation

```python
# Behavioral model testing with pytest

"""
Model behavioral tests — verify model meets behavioral expectations.
Run these in CI/CD before model promotion.
"""

import pytest
import numpy as np
from typing import Callable, List, Tuple


class BehavioralTest:
    """Base class for behavioral model tests."""
    
    def __init__(self, model_predict: Callable):
        """
        Args:
            model_predict: Function that takes input and returns prediction.
                          For classifiers: returns class label or probability.
                          For regressors: returns numeric value.
        """
        self.predict = model_predict


class InvarianceTest(BehavioralTest):
    """
    Test that irrelevant changes don't change model output.
    
    For classification: same class predicted.
    For regression: output within tolerance.
    """
    
    def test_name_change(self, base_input: dict, names: List[str]) -> bool:
        """Changing the name field shouldn't change prediction."""
        base_prediction = self.predict(base_input)
        
        for name in names:
            modified_input = {**base_input, "name": name}
            modified_prediction = self.predict(modified_input)
            
            if modified_prediction != base_prediction:
                return False
        return True
    
    def test_perturbation(
        self, 
        inputs: List[dict], 
        perturb_fn: Callable,
        tolerance: float = 0.0
    ) -> Tuple[int, int]:
        """
        Test that perturbation doesn't change output beyond tolerance.
        
        Returns: (passed, total) counts
        """
        passed = 0
        total = len(inputs)
        
        for inp in inputs:
            original = self.predict(inp)
            perturbed_input = perturb_fn(inp)
            perturbed = self.predict(perturbed_input)
            
            if abs(original - perturbed) <= tolerance:
                passed += 1
        
        return passed, total


class DirectionalTest(BehavioralTest):
    """
    Test that known relationships hold.
    When X increases, Y should increase (or decrease).
    """
    
    def test_monotonic(
        self,
        base_input: dict,
        feature: str,
        values: List[float],
        direction: str = "increasing"  # "increasing" or "decreasing"
    ) -> bool:
        """
        Test that predictions are monotonic with respect to a feature.
        
        Example: as income increases, credit score should increase.
        """
        predictions = []
        for value in sorted(values):
            modified = {**base_input, feature: value}
            predictions.append(self.predict(modified))
        
        if direction == "increasing":
            # Each prediction >= previous (allow small tolerance)
            return all(
                predictions[i+1] >= predictions[i] - 0.01
                for i in range(len(predictions) - 1)
            )
        else:
            return all(
                predictions[i+1] <= predictions[i] + 0.01
                for i in range(len(predictions) - 1)
            )
    
    def test_directional_change(
        self,
        base_input: dict,
        modification: dict,
        expected_direction: str  # "increase" or "decrease"
    ) -> bool:
        """
        Test that a specific modification moves prediction in expected direction.
        
        Example: adding "terrible" to review should decrease sentiment.
        """
        base_pred = self.predict(base_input)
        modified_input = {**base_input, **modification}
        modified_pred = self.predict(modified_input)
        
        if expected_direction == "increase":
            return modified_pred > base_pred
        else:
            return modified_pred < base_pred


class MinimumFunctionalityTest(BehavioralTest):
    """
    Test that model handles basic cases correctly.
    These are "sanity checks" — if these fail, something is fundamentally wrong.
    """
    
    def test_known_examples(
        self,
        examples: List[Tuple[dict, any]]  # (input, expected_output)
    ) -> Tuple[int, int]:
        """
        Test model on curated known-good examples.
        
        Examples should be OBVIOUS cases where the correct answer is clear.
        If the model gets these wrong, it's broken.
        """
        passed = 0
        for input_data, expected in examples:
            prediction = self.predict(input_data)
            if prediction == expected:
                passed += 1
        
        return passed, len(examples)
    
    def test_edge_cases(self, edge_cases: List[dict]) -> List[dict]:
        """
        Test model handles edge cases without crashing.
        
        Edge cases: empty input, very long input, special characters,
        all zeros, all nulls (after imputation), extreme values.
        """
        results = []
        for case in edge_cases:
            try:
                prediction = self.predict(case)
                results.append({"input": case, "prediction": prediction, "error": None})
            except Exception as e:
                results.append({"input": case, "prediction": None, "error": str(e)})
        
        return results


# ============ Concrete test examples (pytest) ============

class TestSentimentModel:
    """Behavioral tests for a sentiment classification model."""
    
    @pytest.fixture
    def model(self):
        """Load the model under test."""
        # In practice: load from model registry
        from models import SentimentModel
        return SentimentModel.load("production_v2")
    
    # --- Minimum Functionality Tests ---
    
    @pytest.mark.parametrize("text,expected", [
        ("I love this product, it's amazing!", "positive"),
        ("This is terrible, worst purchase ever.", "negative"),
        ("The product arrived on Tuesday.", "neutral"),
        ("Absolutely fantastic experience!", "positive"),
        ("Horrible quality, broke immediately.", "negative"),
    ])
    def test_obvious_examples(self, model, text, expected):
        """Model should correctly classify obvious sentiment."""
        prediction = model.predict(text)
        assert prediction == expected, f"Expected {expected}, got {prediction} for: {text}"
    
    # --- Invariance Tests ---
    
    @pytest.mark.parametrize("name", ["John", "María", "张伟", "Aisha", "Sven"])
    def test_name_invariance(self, model, name):
        """Changing the person's name shouldn't change sentiment."""
        template = f"{name} said the product was excellent"
        prediction = model.predict(template)
        assert prediction == "positive"
    
    @pytest.mark.parametrize("location", ["New York", "Mumbai", "Lagos", "Tokyo", "Berlin"])
    def test_location_invariance(self, model, location):
        """Location shouldn't affect product sentiment."""
        template = f"Bought this in {location}. Terrible quality, fell apart in a day."
        prediction = model.predict(template)
        assert prediction == "negative"
    
    def test_capitalization_invariance(self, model):
        """Capitalization shouldn't change sentiment."""
        texts = [
            "great product works perfectly",
            "GREAT PRODUCT WORKS PERFECTLY",
            "Great Product Works Perfectly",
            "gReAt PrOdUcT wOrKs PeRfEcTlY",
        ]
        predictions = [model.predict(t) for t in texts]
        assert all(p == predictions[0] for p in predictions)
    
    # --- Directional Tests ---
    
    def test_adding_positive_word_increases_score(self, model):
        """Adding positive words should increase positive score."""
        base = "The product works."
        positive = "The product works amazingly well."
        
        base_score = model.predict_proba(base)["positive"]
        positive_score = model.predict_proba(positive)["positive"]
        
        assert positive_score > base_score
    
    def test_adding_negative_word_decreases_score(self, model):
        """Adding negative words should decrease positive score."""
        base = "The product works."
        negative = "The product barely works and is disappointing."
        
        base_score = model.predict_proba(base)["positive"]
        negative_score = model.predict_proba(negative)["positive"]
        
        assert negative_score < base_score
    
    def test_negation_flips_sentiment(self, model):
        """Negation should change sentiment direction."""
        positive = "This is a good product."
        negated = "This is not a good product."
        
        pos_pred = model.predict(positive)
        neg_pred = model.predict(negated)
        
        # At minimum: negated shouldn't be positive
        assert neg_pred != "positive" or pos_pred != "positive"
    
    # --- Edge Case Tests ---
    
    def test_empty_input(self, model):
        """Model should handle empty input gracefully."""
        prediction = model.predict("")
        assert prediction in ["positive", "negative", "neutral"]  # Doesn't crash
    
    def test_very_long_input(self, model):
        """Model should handle very long text."""
        long_text = "This product is great. " * 1000
        prediction = model.predict(long_text)
        assert prediction == "positive"
    
    def test_special_characters(self, model):
        """Model should handle special characters."""
        text = "Product: 5★★★★★! Amazing <script>alert('xss')</script> 🎉"
        prediction = model.predict(text)
        # Shouldn't crash — prediction can be anything
        assert prediction in ["positive", "negative", "neutral"]
```

### Template-Based Test Generation

```python
# Generate hundreds of tests from templates

"""
Template-based test generation creates comprehensive behavioral tests
from parameterized templates. This ensures broad coverage efficiently.
"""

from itertools import product
from typing import List, Dict


class TemplateTestGenerator:
    """Generate behavioral tests from templates with slot filling."""
    
    def generate_invariance_tests(
        self,
        template: str,
        slots: Dict[str, List[str]],
        expected: str
    ) -> List[Dict]:
        """
        Generate invariance tests by filling template slots.
        All generated examples should have the same prediction.
        
        Example:
            template: "{NAME} said the {PRODUCT} was {POSITIVE_ADJ}"
            slots: {
                "NAME": ["John", "Maria", "Wei"],
                "PRODUCT": ["phone", "laptop", "camera"],
                "POSITIVE_ADJ": ["great", "excellent", "wonderful"]
            }
            expected: "positive"
            
        Generates: 3 × 3 × 3 = 27 test cases, all should be "positive"
        """
        tests = []
        slot_names = list(slots.keys())
        slot_values = list(slots.values())
        
        for combo in product(*slot_values):
            filled = template
            for name, value in zip(slot_names, combo):
                filled = filled.replace(f"{{{name}}}", value)
            
            tests.append({
                "input": filled,
                "expected": expected,
                "template": template,
                "slots": dict(zip(slot_names, combo)),
            })
        
        return tests
    
    def generate_directional_tests(
        self,
        base_template: str,
        modifier_template: str,
        base_slots: Dict[str, List[str]],
        direction: str
    ) -> List[Dict]:
        """
        Generate directional tests: modifier should move prediction in direction.
        
        Example:
            base: "The {PRODUCT} works"
            modifier: "The {PRODUCT} works {POSITIVE_ADV}"
            direction: "increase"
            
        For each combo: predict(base) < predict(modifier)
        """
        tests = []
        slot_names = list(base_slots.keys())
        slot_values = list(base_slots.values())
        
        for combo in product(*slot_values):
            base = base_template
            modified = modifier_template
            for name, value in zip(slot_names, combo):
                base = base.replace(f"{{{name}}}", value)
                modified = modified.replace(f"{{{name}}}", value)
            
            tests.append({
                "base_input": base,
                "modified_input": modified,
                "direction": direction,
            })
        
        return tests


# Example usage
generator = TemplateTestGenerator()

# Generate invariance tests — name shouldn't affect sentiment
name_invariance = generator.generate_invariance_tests(
    template="{NAME} said the restaurant was {ADJ}",
    slots={
        "NAME": ["John", "María", "张伟", "Aisha", "Sven", "Oluwaseun"],
        "ADJ": ["terrible", "awful", "horrible", "dreadful"],
    },
    expected="negative"
)
# Generates 6 × 4 = 24 test cases
```

---

## How It Works in Practice

### Test Suite Organization

```yaml
Test_Organization:
  by_capability:
    vocabulary: "Does model handle domain vocabulary correctly?"
    robustness: "Does model handle noise, typos, variations?"
    fairness: "Is model invariant to protected attributes?"
    negation: "Does model handle negation correctly?"
    temporal: "Does model handle tense correctly?"
    
  by_severity:
    critical: "Failures here mean model is fundamentally broken (MFT tests)"
    important: "Failures indicate significant quality issues (directional)"
    nice_to_have: "Failures indicate minor robustness gaps (some invariance)"
    
  running:
    every_commit: "Critical + important tests (fast, < 5 minutes)"
    pre_deployment: "Full suite (all tests, may take 30+ minutes)"
    nightly: "Extended suite (stress tests, large template expansions)"
```

---

## Interview Tip

> When asked about model unit testing: "I use behavioral testing inspired by the CheckList framework — three types of tests that verify model behavior without testing exact outputs: (1) Minimum Functionality Tests (MFT) — obvious cases where the correct answer is unambiguous. 'I love this product' must be positive sentiment. If these fail, the model is fundamentally broken. (2) Invariance Tests (INV) — changes to irrelevant features shouldn't change the output. Changing a person's name shouldn't change sentiment. Changing gender shouldn't change credit score. These catch bias and spurious correlations. (3) Directional Tests (DIR) — known relationships must hold. Adding 'terrible' should decrease positive sentiment. Increasing income should increase credit score. These verify the model learned meaningful patterns, not just noise. I generate tests at scale using templates: 'The {PRODUCT} is {NEGATIVE_ADJ}' with 10 products and 10 adjectives gives 100 tests from one template. Key insight: behavioral tests catch issues that aggregate accuracy metrics miss. A model with 95% accuracy might fail every negation test ('not good' classified as positive), which a single accuracy number would hide."

---

## Common Mistakes

1. **Only testing accuracy on a test set** — Model has 95% accuracy but gets negation wrong 100% of the time, or is biased by name/gender. Behavioral tests expose specific failure modes that aggregate metrics hide.

2. **Writing exact-output assertions** — `assert model.predict(input) == 0.7342`. This breaks on every retrain. Solution: test behavioral properties (invariance, direction, minimum functionality) that should hold across model versions.

3. **Not using templates for coverage** — Writing 5 manual test cases and calling it done. Template-based generation gives 100+ test cases from a few templates with minimal effort.

4. **Ignoring negation** — Models notoriously struggle with negation ("not good" classified as positive). Always include negation tests explicitly.

5. **No protected-attribute invariance tests** — Not testing that changing gender, race, or name doesn't change predictions. This is both a fairness and legal requirement.

---

## Key Takeaways

- Behavioral testing: test properties (invariance, direction, minimum functionality) not exact outputs
- MFT: obvious cases that must be correct — sanity checks
- Invariance: irrelevant changes shouldn't change predictions (catches bias, spurious correlations)
- Directional: known relationships must hold (verifies model learned real patterns)
- Template-based generation: create hundreds of tests from parameterized templates
- CheckList framework: systematic capability-based testing (originated at Microsoft Research)
- Catches issues that aggregate accuracy hides (negation failures, bias, robustness gaps)
- Run critical behavioral tests in CI/CD on every code change (fast, < 5 minutes)
- Always test negation handling and protected-attribute invariance
- Tests should hold across model versions — they verify behavior, not specific weights
