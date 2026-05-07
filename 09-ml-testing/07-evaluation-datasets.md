# Evaluation Datasets

## The Problem / Why This Matters

The quality of ML evaluation is only as good as the quality of evaluation data. A model that scores 98% on a flawed evaluation dataset might perform at 70% in production. Evaluation dataset engineering is a discipline: how you construct, curate, version, and maintain evaluation data directly determines whether you can trust your model quality metrics. Key challenges include: contamination (evaluation data leaking into training — artificially inflates scores), representation (evaluation data not matching production distribution — misleading metrics), staleness (evaluation data reflects the world 2 years ago, not today), and coverage (evaluation data missing critical edge cases, adversarial inputs, or minority subgroups). In 2026, with LLMs (Large Language Models) trained on massive internet corpora, contamination is especially dangerous — if your evaluation examples appeared anywhere on the internet, the model may have memorized them. Building proper evaluation datasets requires intentional design, strict isolation from training data, continuous refreshing, and careful decontamination procedures.

---

## The Analogy

Think of evaluation datasets like a standardized exam for students:

- **Bad evaluation dataset** = An exam where students had access to the exact questions beforehand (contamination), all questions are about one topic (no coverage), and the questions are from 10 years ago (staleness). Everyone scores 100% but no one actually learned anything.
- **Good evaluation dataset** = An exam that tests a diverse range of skills (coverage), uses fresh questions never seen before (decontamination), reflects current knowledge requirements (freshness), and includes questions of varying difficulty (calibration). Scores actually predict real-world performance.

The exam (evaluation dataset) must be a reliable proxy for real-world performance. If it's not, you're optimizing for the wrong thing.

---

## Deep Dive

### Evaluation Dataset Design Principles

```yaml
Design_Principles:
  representativeness:
    what: "Eval data distribution should match production distribution"
    approach:
      - "Sample from actual production traffic (with labels)"
      - "Ensure demographic/segment proportions match production"
      - "Include both common cases AND edge cases"
    anti_pattern: "Eval data all from one source/time period/segment"
    
  difficulty_calibration:
    what: "Eval set should include easy, medium, and hard examples"
    distribution:
      easy: "30% — model should almost always get right (sanity check)"
      medium: "50% — requires real model capability"
      hard: "20% — edge cases, adversarial, ambiguous"
    why: "All-easy eval inflates metrics. All-hard eval is demoralizing/uninformative."
    
  decontamination:
    what: "Ensure eval data never appears in training data"
    methods:
      - "Temporal split: eval data from after training data cutoff"
      - "N-gram deduplication: remove eval examples with high n-gram overlap to training"
      - "Embedding dedup: remove eval examples too similar to training (cosine > 0.95)"
      - "Canary strings: plant unique strings in eval data, check training data for them"
    critical_for_llms: "LLMs trained on internet data may have seen eval benchmarks"
    
  freshness:
    what: "Eval data reflects current world state"
    strategy: "Rotate 20-30% of eval set quarterly"
    danger: "Eval set from 2023 may not test 2026 patterns (new fraud types, new language)"
    
  labeling_quality:
    what: "Labels must be high-quality and consistent"
    approach:
      - "Multiple annotators per example (majority vote or adjudication)"
      - "Inter-annotator agreement measurement (Cohen's Kappa > 0.8)"
      - "Clear labeling guidelines (reduce ambiguity)"
      - "Expert review for high-stakes labels"
    danger: "Noisy labels → model ceiling limited by label quality"
```

### Building Evaluation Datasets

```python
# Evaluation dataset construction and management

"""
Systematic approach to building and maintaining evaluation datasets.
Covers: creation, versioning, contamination prevention, and refresh.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional
from datetime import datetime, timedelta
import hashlib
import json
import numpy as np


@dataclass
class EvalExample:
    """Single evaluation example with metadata."""
    id: str
    input_data: Any  # Text, features, image path, etc.
    label: Any  # Ground truth
    difficulty: str  # "easy", "medium", "hard"
    category: str  # Type of example (edge_case, normal, adversarial)
    source: str  # Where this example came from
    created_date: str
    annotator_count: int = 1
    agreement_score: float = 1.0  # Inter-annotator agreement
    
    # Decontamination metadata
    content_hash: str = ""  # Hash for dedup
    n_gram_fingerprint: Set[str] = field(default_factory=set)


@dataclass
class EvalDataset:
    """
    Versioned evaluation dataset with quality controls.
    
    Key properties:
    - Immutable versions (once created, a version never changes)
    - Strict contamination prevention
    - Coverage tracking (what is/isn't tested)
    - Freshness metadata
    """
    name: str
    version: str
    model_name: str
    examples: List[EvalExample] = field(default_factory=list)
    
    # Metadata
    created_date: str = ""
    description: str = ""
    target_size: int = 500  # Recommended minimum
    
    # Quality metrics
    difficulty_distribution: Dict[str, float] = field(default_factory=dict)
    category_distribution: Dict[str, float] = field(default_factory=dict)
    avg_agreement: float = 0.0
    
    def add_example(self, example: EvalExample):
        """Add example with decontamination hash."""
        # Compute content hash for dedup
        content = json.dumps(example.input_data, sort_keys=True)
        example.content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Check for duplicates within eval set
        existing_hashes = {e.content_hash for e in self.examples}
        if example.content_hash in existing_hashes:
            raise ValueError(f"Duplicate example detected: {example.id}")
        
        self.examples.append(example)
        self._update_distributions()
    
    def check_contamination(self, training_hashes: Set[str]) -> List[str]:
        """
        Check if any eval examples appear in training data.
        
        Args:
            training_hashes: Set of content hashes from training data
            
        Returns:
            List of contaminated example IDs
        """
        contaminated = []
        for example in self.examples:
            if example.content_hash in training_hashes:
                contaminated.append(example.id)
        
        if contaminated:
            print(f"WARNING: {len(contaminated)} eval examples found in training data!")
        
        return contaminated
    
    def check_coverage(self, required_categories: List[str]) -> Dict:
        """
        Verify eval set covers all required categories.
        
        Returns coverage report with gaps identified.
        """
        existing_categories = {e.category for e in self.examples}
        missing = set(required_categories) - existing_categories
        
        # Per-category count
        category_counts = {}
        for example in self.examples:
            category_counts[example.category] = category_counts.get(example.category, 0) + 1
        
        # Check minimum per category (at least 20 examples each)
        undercovered = {
            cat: count for cat, count in category_counts.items()
            if count < 20
        }
        
        return {
            "total_examples": len(self.examples),
            "categories_covered": len(existing_categories),
            "categories_missing": list(missing),
            "undercovered": undercovered,
            "sufficient": len(missing) == 0 and len(undercovered) == 0,
        }
    
    def freshness_report(self) -> Dict:
        """Check how fresh the eval examples are."""
        now = datetime.now()
        
        ages = []
        for example in self.examples:
            created = datetime.fromisoformat(example.created_date)
            age_days = (now - created).days
            ages.append(age_days)
        
        return {
            "avg_age_days": np.mean(ages),
            "max_age_days": max(ages),
            "pct_older_than_90_days": sum(1 for a in ages if a > 90) / len(ages),
            "pct_older_than_365_days": sum(1 for a in ages if a > 365) / len(ages),
            "needs_refresh": np.mean(ages) > 180,  # Refresh if avg > 6 months
        }
    
    def _update_distributions(self):
        """Update internal distribution metrics."""
        total = len(self.examples)
        if total == 0:
            return
        
        # Difficulty distribution
        diff_counts = {}
        for e in self.examples:
            diff_counts[e.difficulty] = diff_counts.get(e.difficulty, 0) + 1
        self.difficulty_distribution = {k: v/total for k, v in diff_counts.items()}
        
        # Category distribution
        cat_counts = {}
        for e in self.examples:
            cat_counts[e.category] = cat_counts.get(e.category, 0) + 1
        self.category_distribution = {k: v/total for k, v in cat_counts.items()}
        
        # Average agreement
        self.avg_agreement = np.mean([e.agreement_score for e in self.examples])


class EvalDatasetBuilder:
    """
    Build evaluation datasets from various sources.
    
    Sources:
    1. Production traffic (sampled and labeled)
    2. Manual curation (domain experts)
    3. Adversarial generation (attack patterns)
    4. Synthetic generation (template-based)
    """
    
    def from_production_traffic(
        self,
        traffic_samples: List[Dict],
        label_fn,
        sample_size: int = 500,
    ) -> List[EvalExample]:
        """
        Create eval examples from production traffic.
        
        Steps:
        1. Sample diverse traffic (stratified by segment/difficulty)
        2. Get human labels
        3. Filter for high-agreement labels
        """
        # Stratified sampling
        sampled = self._stratified_sample(traffic_samples, sample_size)
        
        # Get labels (human annotation or high-confidence model + human verification)
        examples = []
        for i, sample in enumerate(sampled):
            label, agreement = label_fn(sample)  # Returns (label, annotator_agreement)
            
            if agreement >= 0.8:  # Only include high-agreement examples
                examples.append(EvalExample(
                    id=f"prod_{i:04d}",
                    input_data=sample["input"],
                    label=label,
                    difficulty=self._estimate_difficulty(sample),
                    category="production_traffic",
                    source="production_sample",
                    created_date=datetime.now().isoformat(),
                    annotator_count=3,
                    agreement_score=agreement,
                ))
        
        return examples
    
    def adversarial_examples(
        self,
        attack_patterns: List[Dict]
    ) -> List[EvalExample]:
        """
        Create adversarial eval examples.
        
        These test robustness: typos, Unicode tricks, injection attempts,
        boundary values, format variations.
        """
        examples = []
        for i, pattern in enumerate(attack_patterns):
            examples.append(EvalExample(
                id=f"adv_{i:04d}",
                input_data=pattern["input"],
                label=pattern["expected_output"],
                difficulty="hard",
                category="adversarial",
                source="adversarial_generation",
                created_date=datetime.now().isoformat(),
            ))
        return examples
    
    def _stratified_sample(self, data: List[Dict], n: int) -> List[Dict]:
        """Sample to maintain distribution of categories."""
        # Group by category
        groups = {}
        for item in data:
            cat = item.get("category", "default")
            groups.setdefault(cat, []).append(item)
        
        # Sample proportionally from each group
        sampled = []
        for cat, items in groups.items():
            cat_n = max(1, int(n * len(items) / len(data)))
            indices = np.random.choice(len(items), min(cat_n, len(items)), replace=False)
            sampled.extend([items[i] for i in indices])
        
        return sampled[:n]
    
    def _estimate_difficulty(self, sample: Dict) -> str:
        """Estimate example difficulty based on characteristics."""
        # Heuristic: longer inputs, rare categories, edge values → harder
        # In practice: use model confidence as proxy for difficulty
        return "medium"  # Placeholder
```

### LLM Evaluation Datasets

```yaml
LLM_Evaluation_Specifics:
  challenges:
    contamination_risk: "LLMs trained on internet data may have seen common benchmarks"
    non_deterministic: "Same prompt → different answers each time"
    subjective: "Many answers are valid — hard to define 'correct'"
    multi_dimensional: "Must evaluate: correctness, helpfulness, safety, coherence, relevance"
    
  benchmark_contamination:
    problem: "GPT-5/Claude 4 likely trained on MMLU, HumanEval, GSM8K answers"
    solution:
      - "Create PRIVATE evaluation sets (never published online)"
      - "Use temporal isolation (eval data created after training cutoff)"
      - "Paraphrase existing benchmarks (same concept, different wording)"
      - "Dynamic eval: generate new questions each time (parametric)"
      
  evaluation_dimensions:
    factuality: "Are claims accurate and verifiable?"
    relevance: "Does the answer address the question?"
    completeness: "Are important aspects covered?"
    coherence: "Is the answer well-structured and logical?"
    safety: "Is the content safe and appropriate?"
    helpfulness: "Would a user find this answer useful?"
    
  eval_methods:
    human_eval:
      what: "Human annotators rate responses"
      scale: "1-5 Likert or pairwise preference (A vs B)"
      cost: "$1-5 per rating"
      reliability: "High (when calibrated), but expensive and slow"
      
    llm_as_judge:
      what: "Use another LLM to evaluate responses"
      implementation: "Claude 4 Opus evaluates GPT-5 responses (or vice versa)"
      cost: "~$0.01-0.10 per evaluation"
      reliability: "Good for relative comparison, biased for absolute scoring"
      tools: "RAGAS, DeepEval, custom judge prompts"
      
    reference_based:
      what: "Compare against reference answer"
      metrics: "ROUGE, BLEU, BERTScore, semantic similarity"
      limitation: "Multiple valid answers exist — reference isn't the only correct answer"
      
    task_specific:
      what: "Measure task completion"
      examples:
        code: "Does generated code pass test cases?"
        math: "Is the final numerical answer correct?"
        extraction: "Are extracted entities correct?"
      reliability: "High — objective ground truth exists"
```

### Versioning and Lifecycle

```yaml
Dataset_Lifecycle:
  creation:
    - "Define scope and categories"
    - "Collect examples from diverse sources"
    - "Label with multiple annotators"
    - "Filter low-agreement examples"
    - "Run decontamination checks"
    - "Validate coverage and difficulty distribution"
    - "Freeze as immutable version (v1.0)"
    
  maintenance:
    quarterly_refresh:
      - "Add 20-30% new examples (from recent production traffic)"
      - "Remove stale examples (patterns that no longer exist)"
      - "Re-run decontamination (against latest training data)"
      - "Validate coverage still adequate"
      - "Release as new version (v1.1, v1.2, etc.)"
      
    incident_driven:
      - "Production incident reveals failure mode → add as eval example"
      - "New category of requests discovered → add coverage"
      - "Bias discovered → add fairness eval examples"
      
  versioning_rules:
    immutable: "Once released, a version NEVER changes"
    additive: "New versions add/remove examples (tracked in changelog)"
    comparable: "Results on different versions are NOT directly comparable"
    retention: "Keep old versions for historical comparison"
    
  storage:
    format: "JSON/JSONL with metadata per example"
    location: "Dedicated eval data store (separate from training)"
    access: "Read-only for model evaluation pipelines"
    backup: "Versioned in git or artifact store (never deleted)"
```

---

## How It Works in Practice

### Evaluation Pipeline

```yaml
Pipeline:
  trigger: "Model retrain OR prompt change OR eval set refresh"
  
  steps:
    1_load_eval_set:
      what: "Load latest eval dataset version"
      validation: "Verify integrity (hash check), no contamination"
      
    2_run_inference:
      what: "Generate predictions for all eval examples"
      config: "Fixed seed/temperature for reproducibility (where possible)"
      
    3_compute_metrics:
      what: "Calculate overall and per-category metrics"
      output: "Accuracy, F1, custom metrics per dimension"
      
    4_compare_baseline:
      what: "Compare against current production model's scores"
      output: "Delta per metric, statistical significance"
      
    5_generate_report:
      what: "Visual report with pass/fail and details"
      includes: "Confusion matrix, error analysis, segment breakdowns"
```

---

## Interview Tip

> When asked about evaluation datasets: "I treat evaluation data as a first-class engineering artifact — it's versioned, immutable, decontaminated, and maintained like code. Key principles: (1) Decontamination — I verify eval examples never appear in training data using content hashing, n-gram overlap detection, and embedding similarity. For LLMs, this is critical because models trained on internet data may have memorized common benchmarks. I use private eval sets and temporal isolation (eval data created after training cutoff). (2) Coverage — I ensure the eval set covers easy/medium/hard difficulty levels (30/50/20 split), all critical categories (normal, edge case, adversarial), and all relevant segments/demographics. I run coverage checks before any evaluation. (3) Freshness — I refresh 20-30% of the eval set quarterly with new production examples. A 2-year-old eval set doesn't reflect current patterns. (4) Labeling quality — multiple annotators per example, filter for inter-annotator agreement > 0.8 (Cohen's Kappa). Bad labels create a ceiling on model quality. (5) Immutable versioning — once a version is released, it never changes. Results on v1.0 are always comparable to other results on v1.0. New examples create v1.1. For LLM evaluation specifically, I use multi-dimensional assessment (factuality, relevance, safety, helpfulness) with LLM-as-judge for scale and human evaluation for calibration."

---

## Common Mistakes

1. **Using public benchmarks without decontamination** — Evaluating on MMLU (Massive Multitask Language Understanding) or HumanEval that the model likely saw during training. Score is inflated. Solution: create private evaluation sets OR use benchmark decontamination (check for n-gram overlap with training data, paraphrase questions).

2. **Eval dataset too small** — 50 examples for a critical model. Random noise dominates metrics — a 2% accuracy change could be 1 example. Solution: minimum 200-500 examples. Use power analysis to determine sample size needed for detecting meaningful differences.

3. **Never refreshing the eval set** — Same eval set used for 3 years. World changed, new patterns emerged, eval set no longer tests what matters. Solution: quarterly refresh — add new examples from production, remove obsolete ones, release new version.

4. **All examples are easy** — Eval set full of obvious cases. Model scores 99% on eval but 85% in production (production has harder cases). Solution: calibrate difficulty — 30% easy, 50% medium, 20% hard. Include adversarial examples.

5. **Single annotator labels** — One person labels all examples. Their biases become the "ground truth." Model optimizes for one person's judgment. Solution: multiple annotators (3-5), measure agreement, only include high-agreement examples. Adjudicate disagreements with domain experts.

---

## Key Takeaways

- Evaluation dataset quality directly determines trust in model metrics
- Decontamination: verify eval examples never appear in training data (hashing, n-gram overlap)
- Coverage: easy/medium/hard difficulty (30/50/20), all categories, all segments represented
- Freshness: refresh 20-30% quarterly with new production traffic examples
- Labeling quality: multiple annotators, filter for agreement > 0.8, clear guidelines
- Immutable versioning: once released, never modified — new examples = new version
- LLM specifics: private eval sets, temporal isolation, multi-dimensional assessment
- Minimum size: 200-500 examples to have statistical power for meaningful comparison
- Sources: production traffic, manual curation, adversarial generation, incident-driven addition
- Storage: separate from training data, read-only access, version-controlled, never deleted
