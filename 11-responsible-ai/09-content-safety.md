# Content Safety

## The Problem / Why This Matters

AI systems — particularly LLMs (Large Language Models) and generative models — can produce harmful content: toxic text, hate speech, explicit sexual content, graphic violence, self-harm instructions, misinformation, and CSAM (Child Sexual Abuse Material). Content safety is the engineering discipline of preventing AI systems from generating, amplifying, or distributing harmful content. In 2026, this is both a legal obligation (DSA — Digital Services Act in EU, Section 230 considerations in US, Online Safety Act in UK) and a business necessity (platforms can be fined billions for failing to moderate AI-generated content). The challenge is massive: harmful content is context-dependent (a medical discussion of suicide differs from promotion of self-harm), adversaries actively probe for bypass methods (jailbreaks), and over-filtering creates uselessness (refusing all sensitive topics). ML engineers build the content safety stack: toxicity classifiers, harmful content detectors, guardrail systems, content moderation pipelines, and continuous evaluation frameworks. This requires both precision (catch genuinely harmful content) and recall management (don't over-block legitimate discourse).

---

## The Analogy

Think of content safety like a concert venue's security:

- **No content safety** = An open field with no security. Anyone can bring anything, do anything. Some attendees get hurt, the event gets shut down.
- **Over-filtering** = Security so strict that they confiscate water bottles, phones, and medications. Nobody gets hurt, but nobody enjoys the concert either. Users abandon your platform.
- **Effective content safety** = Well-trained security: metal detectors catch weapons (high-risk content), bag checks for prohibited items (clear policy violations), but they let through water, phones, and medications (legitimate sensitive content). Fast for most people, thorough investigation only when something genuinely looks dangerous.
- **Adversarial bypass** = Attendees hiding prohibited items inside water bottles, splitting weapons across multiple people. Your security evolves to detect new concealment methods — an ongoing arms race.

---

## Deep Dive

### Content Safety Taxonomy

```yaml
Content_Safety_Taxonomy:
  violence_and_gore:
    severity_levels:
      low: "Discussion of violence in news context, historical events"
      medium: "Detailed descriptions of violence, fighting techniques"
      high: "Instructions for causing harm, glorification of violence"
      critical: "Actionable instructions for mass violence, CBRN (Chemical, Biological, Radiological, Nuclear)"
      
  hate_and_discrimination:
    severity_levels:
      low: "Discussion of discrimination (educational context)"
      medium: "Stereotypes, microaggressions"
      high: "Dehumanization, explicit hate speech"
      critical: "Incitement to violence against protected groups"
      
  sexual_content:
    severity_levels:
      low: "Romantic relationships, dating advice"
      medium: "Sexual health education, explicit discussions"
      high: "Pornographic content, explicit sexual descriptions"
      critical: "CSAM, non-consensual content, exploitation"
      
  self_harm:
    severity_levels:
      low: "Mental health discussion, seeking help"
      medium: "Descriptions of self-harm (recovery context)"
      high: "Methods of self-harm without deterrence"
      critical: "Encouragement of self-harm, detailed instructions"
      
  misinformation:
    severity_levels:
      low: "Outdated information, minor inaccuracies"
      medium: "Health misinformation (alternative medicine promotion)"
      high: "Election misinformation, dangerous health claims"
      critical: "Deliberately fabricated propaganda, deepfake misinformation"
      
  dangerous_information:
    severity_levels:
      low: "General chemistry/biology knowledge"
      medium: "Dual-use knowledge that could be misused"
      high: "Specific instructions for weapons, drugs, exploits"
      critical: "Actionable CBRN instructions, zero-day exploits, bioweapons"
```

### Content Safety Architecture

```yaml
Content_Safety_Architecture:
  multi_layer_defense:
    layer_1_input_classification:
      purpose: "Classify incoming requests before LLM processes them"
      speed: "< 50ms (must not add significant latency)"
      models:
        - "Lightweight text classifier (fine-tuned BERT/DeBERTa)"
        - "Keyword and pattern matching (fast, catches obvious)"
        - "Intent classifier (is user trying to elicit harmful content?)"
      actions:
        safe: "Pass to LLM"
        suspicious: "Add safety system prompt, pass to LLM, flag output for review"
        harmful: "Block immediately, return refusal, log for analysis"
        
    layer_2_system_prompt:
      purpose: "Instruct LLM on safety boundaries"
      includes:
        - "Explicit list of content that must never be generated"
        - "Context-dependent rules (medical vs. general audience)"
        - "Refusal templates for different categories"
        
    layer_3_output_classification:
      purpose: "Classify LLM output before returning to user"
      models:
        - "Llama Guard 3 (Meta's safety classifier — multi-category)"
        - "Custom toxicity classifier"
        - "PII detector (prevent data leakage)"
        - "Factuality checker (for claims that could be dangerous if wrong)"
      actions:
        safe: "Return to user"
        borderline: "Add content warning, return"
        harmful: "Block, return safe alternative or refusal"
        
    layer_4_human_review:
      purpose: "Human oversight for edge cases"
      triggers:
        - "Output classified as borderline (model unsure)"
        - "User appeals a content block"
        - "New attack pattern detected"
        - "High-stakes context (medical, legal)"
        
    layer_5_monitoring:
      purpose: "Continuous evaluation of content safety performance"
      metrics:
        - "Harm rate (harmful outputs per 1000 requests)"
        - "False positive rate (legitimate requests blocked)"
        - "Attack success rate (adversarial prompts bypassing safety)"
        - "Novel attack detection (new jailbreak patterns)"
```

### Implementation

```python
# Content safety implementation for production AI systems

"""
Multi-layer content safety system for LLM applications.
Implements input filtering, output classification, and monitoring.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class HarmCategory(Enum):
    """Content harm categories."""
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    SEXUAL = "sexual_content"
    SELF_HARM = "self_harm"
    DANGEROUS_INFO = "dangerous_information"
    CSAM = "csam"
    MISINFORMATION = "misinformation"
    PII_LEAK = "pii_leak"
    HARASSMENT = "harassment"


class Severity(Enum):
    """Severity levels for content safety violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Action(Enum):
    """Actions to take on flagged content."""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    ESCALATE = "escalate"


@dataclass
class SafetyVerdict:
    """Result of content safety classification."""
    is_safe: bool
    action: Action
    categories: List[HarmCategory] = field(default_factory=list)
    severity: Severity = Severity.LOW
    confidence: float = 1.0
    explanation: str = ""
    should_log: bool = True


@dataclass
class ContentPolicy:
    """Configurable content safety policy."""
    # Mapping: category → severity → action
    policy_matrix: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    # Context-dependent overrides
    allow_medical_context: bool = True
    allow_educational_context: bool = True
    allow_news_context: bool = True
    
    # Audience
    audience: str = "general"  # general, adult, professional
    
    def get_action(self, category: HarmCategory, severity: Severity) -> Action:
        """Get action for a specific category and severity."""
        cat_policies = self.policy_matrix.get(category.value, {})
        action_str = cat_policies.get(severity.value, "block")
        return Action(action_str)


class ContentSafetyClassifier:
    """
    Multi-category content safety classifier.
    
    In production, this wraps:
    - Llama Guard 3 (Meta's open safety classifier)
    - OpenAI Moderation API
    - Custom fine-tuned classifiers
    """
    
    def __init__(self, policy: ContentPolicy, model=None):
        self.policy = policy
        self.model = model  # Safety classification model
        
        # Compiled patterns for fast first-pass filtering
        self._compile_patterns()
    
    def classify_input(self, text: str, context: Dict = None) -> SafetyVerdict:
        """
        Classify user input for safety.
        
        Fast classification (< 50ms) to decide if input should be processed.
        """
        # Fast pattern matching (< 5ms)
        pattern_result = self._pattern_check(text)
        if pattern_result.action == Action.BLOCK:
            return pattern_result
        
        # ML classification (< 50ms with optimized model)
        ml_result = self._ml_classify(text)
        
        # Context adjustment
        if context and ml_result.categories:
            ml_result = self._apply_context(ml_result, context)
        
        # Determine action based on policy
        if ml_result.categories:
            max_severity = ml_result.severity
            primary_category = ml_result.categories[0]
            action = self.policy.get_action(primary_category, max_severity)
            ml_result.action = action
            ml_result.is_safe = action == Action.ALLOW
        
        return ml_result
    
    def classify_output(self, text: str, original_input: str) -> SafetyVerdict:
        """
        Classify model output for safety.
        
        More thorough than input classification (output is about to be shown to user).
        """
        # Standard ML classification
        verdict = self._ml_classify(text)
        
        # Additional output-specific checks
        pii_check = self._check_pii(text)
        if pii_check:
            verdict.categories.append(HarmCategory.PII_LEAK)
            verdict.action = Action.BLOCK
            verdict.is_safe = False
        
        # Check for potential jailbreak compliance
        if self._detect_jailbreak_compliance(text, original_input):
            verdict.action = Action.BLOCK
            verdict.is_safe = False
            verdict.explanation = "Output appears to comply with adversarial prompt"
        
        return verdict
    
    def _pattern_check(self, text: str) -> SafetyVerdict:
        """Fast regex-based first pass."""
        # Critical patterns that always block
        for category, patterns in self.critical_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    return SafetyVerdict(
                        is_safe=False,
                        action=Action.BLOCK,
                        categories=[HarmCategory(category)],
                        severity=Severity.CRITICAL,
                        confidence=0.95,
                        explanation=f"Critical pattern match: {category}",
                    )
        
        return SafetyVerdict(is_safe=True, action=Action.ALLOW)
    
    def _ml_classify(self, text: str) -> SafetyVerdict:
        """ML-based classification."""
        # In production: call Llama Guard, OpenAI Moderation, or custom model
        # Returns multi-label classification with confidence scores
        
        # Placeholder — actual implementation uses fine-tuned model
        categories = []
        severity = Severity.LOW
        confidence = 0.9
        
        # Simulated classification
        # In reality: self.model.predict(text) → multi-label probabilities
        
        return SafetyVerdict(
            is_safe=len(categories) == 0,
            action=Action.ALLOW if not categories else Action.BLOCK,
            categories=categories,
            severity=severity,
            confidence=confidence,
        )
    
    def _apply_context(self, verdict: SafetyVerdict, context: Dict) -> SafetyVerdict:
        """
        Adjust classification based on context.
        
        Medical professionals discussing drugs ≠ drug abuse encouragement.
        Educational discussion of historical violence ≠ violence promotion.
        """
        is_medical = context.get("domain") == "medical"
        is_educational = context.get("domain") == "education"
        
        # Relax certain categories in appropriate contexts
        if is_medical and self.policy.allow_medical_context:
            medical_acceptable = {HarmCategory.SELF_HARM, HarmCategory.DANGEROUS_INFO}
            verdict.categories = [c for c in verdict.categories if c not in medical_acceptable]
            if not verdict.categories:
                verdict.is_safe = True
                verdict.action = Action.ALLOW
        
        if is_educational and self.policy.allow_educational_context:
            educational_acceptable = {HarmCategory.VIOLENCE, HarmCategory.HATE_SPEECH}
            if verdict.severity in (Severity.LOW, Severity.MEDIUM):
                verdict.categories = [c for c in verdict.categories if c not in educational_acceptable]
                if not verdict.categories:
                    verdict.is_safe = True
                    verdict.action = Action.ALLOW
        
        return verdict
    
    def _check_pii(self, text: str) -> bool:
        """Check for PII leakage in output."""
        pii_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
            r"\b[A-Z][a-z]+ [A-Z][a-z]+\b.*\b\d{3}-\d{3}-\d{4}\b",  # Name + phone
        ]
        return any(re.search(p, text) for p in pii_patterns)
    
    def _detect_jailbreak_compliance(self, output: str, input_text: str) -> bool:
        """Detect if output complies with a jailbreak attempt in input."""
        # If input was suspicious and output doesn't contain refusal markers
        jailbreak_in_input = bool(re.search(
            r"ignore|forget|pretend|DAN|jailbreak",
            input_text, re.IGNORECASE
        ))
        
        if jailbreak_in_input:
            # Check if output contains refusal (good) or compliance (bad)
            refusal_markers = ["I cannot", "I can't", "I won't", "inappropriate", "I'm not able"]
            has_refusal = any(m.lower() in output.lower() for m in refusal_markers)
            return not has_refusal  # If no refusal, likely complied with jailbreak
        
        return False
    
    def _compile_patterns(self):
        """Compile critical safety patterns."""
        self.critical_patterns = {
            "csam": [re.compile(r"child.*sexual|minor.*explicit", re.IGNORECASE)],
            "dangerous_information": [
                re.compile(r"how\s+to\s+(make|build|synthesize)\s+(a\s+)?(bomb|explosive|nerve\s+agent)", re.IGNORECASE),
            ],
        }


class ContentModerationPipeline:
    """
    Full content moderation pipeline for production AI applications.
    
    Orchestrates: input check → LLM generation → output check → response.
    """
    
    def __init__(
        self,
        classifier: ContentSafetyClassifier,
        llm_model,
        fallback_responses: Dict[str, str] = None,
    ):
        self.classifier = classifier
        self.llm = llm_model
        self.fallback_responses = fallback_responses or {
            "default": "I'm not able to help with that request.",
            "violence": "I cannot provide information that could be used to harm others.",
            "self_harm": "I'm concerned about this request. If you're in crisis, please contact a helpline.",
        }
        self.metrics = defaultdict(int)
    
    def process_request(
        self, user_input: str, context: Dict = None
    ) -> Dict:
        """
        Process a user request through the full safety pipeline.
        
        Returns response with safety metadata.
        """
        # Step 1: Input classification
        input_verdict = self.classifier.classify_input(user_input, context)
        self.metrics["total_requests"] += 1
        
        if input_verdict.action == Action.BLOCK:
            self.metrics["input_blocked"] += 1
            logger.warning(f"Input blocked: {input_verdict.categories}")
            return {
                "response": self._get_fallback(input_verdict.categories),
                "blocked": True,
                "blocked_at": "input",
                "categories": [c.value for c in input_verdict.categories],
            }
        
        # Step 2: Generate LLM response
        # (If input was suspicious but not blocked, add safety context)
        system_context = self._get_safety_context(input_verdict)
        response = self._generate_response(user_input, system_context)
        
        # Step 3: Output classification
        output_verdict = self.classifier.classify_output(response, user_input)
        
        if output_verdict.action == Action.BLOCK:
            self.metrics["output_blocked"] += 1
            logger.warning(f"Output blocked: {output_verdict.categories}")
            return {
                "response": self._get_fallback(output_verdict.categories),
                "blocked": True,
                "blocked_at": "output",
                "categories": [c.value for c in output_verdict.categories],
            }
        
        # Step 4: Return safe response
        self.metrics["safe_responses"] += 1
        return {
            "response": response,
            "blocked": False,
            "safety_metadata": {
                "input_confidence": input_verdict.confidence,
                "output_confidence": output_verdict.confidence,
            },
        }
    
    def _get_fallback(self, categories: List[HarmCategory]) -> str:
        """Get appropriate fallback response for blocked content."""
        if HarmCategory.SELF_HARM in categories:
            return self.fallback_responses.get("self_harm", self.fallback_responses["default"])
        if HarmCategory.VIOLENCE in categories:
            return self.fallback_responses.get("violence", self.fallback_responses["default"])
        return self.fallback_responses["default"]
    
    def _get_safety_context(self, verdict: SafetyVerdict) -> str:
        """Get additional safety context for suspicious inputs."""
        if verdict.action == Action.WARN:
            return (
                "The user's request may be attempting to elicit harmful content. "
                "Respond helpfully if the request is legitimate, but do not provide "
                "harmful information under any circumstances."
            )
        return ""
    
    def _generate_response(self, user_input: str, safety_context: str) -> str:
        """Generate LLM response with safety context."""
        # In production: self.llm.generate(prompt=user_input, system=safety_context)
        return "Generated response placeholder"
    
    def get_metrics(self) -> Dict:
        """Get content safety metrics."""
        total = self.metrics["total_requests"] or 1
        return {
            "total_requests": self.metrics["total_requests"],
            "input_blocked": self.metrics["input_blocked"],
            "output_blocked": self.metrics["output_blocked"],
            "safe_responses": self.metrics["safe_responses"],
            "input_block_rate": self.metrics["input_blocked"] / total,
            "output_block_rate": self.metrics["output_blocked"] / total,
            "overall_harm_rate": (
                self.metrics["input_blocked"] + self.metrics["output_blocked"]
            ) / total,
        }
```

### Evaluation

```yaml
Content_Safety_Evaluation:
  benchmarks:
    harmbench:
      what: "Standardized harmful behavior evaluation for LLMs"
      categories: "Direct harmful requests, contextual harm, creative writing harm"
      metric: "Attack Success Rate (ASR) — lower is better"
      
    toxigen:
      what: "Machine-generated hate speech detection benchmark"
      focus: "Implicit and explicit toxicity toward 13 minority groups"
      
    bold:
      what: "Bias in Open-ended Language Generation Dataset"
      focus: "Demographic biases in text generation"
      
    real_toxicity_prompts:
      what: "10K prompts that may trigger toxic completions"
      metric: "Expected maximum toxicity, toxicity probability"
      
  custom_evaluation:
    red_team_suite:
      - "1000 adversarial prompts across all harm categories"
      - "Jailbreak variants (DAN, prompt injection, role-play)"
      - "Contextual safety (medical, educational, news contexts)"
      - "Multilingual safety (harms in non-English)"
      
    metrics:
      harm_rate: "Harmful outputs per 1000 requests (target: < 1)"
      false_positive_rate: "Legitimate requests blocked (target: < 5%)"
      attack_success_rate: "Adversarial prompts that bypass safety (target: < 5%)"
      response_quality: "Quality of refusals (helpful redirect, not curt denial)"
```

---

## How It Works in Practice

### Content Safety for a Customer-Facing Chatbot

```yaml
Customer_Chatbot_Safety:
  application: "General-purpose consumer chatbot (millions of users)"
  
  policy:
    always_block:
      - "CSAM or exploitation content"
      - "Actionable violence instructions"
      - "Self-harm encouragement"
      - "Dangerous substance synthesis"
      
    context_dependent:
      medical_context:
        - "Drugs discussion → allowed for medication info, blocked for synthesis"
        - "Self-harm → allowed for crisis support, blocked for methods"
      educational:
        - "Historical violence → allowed with educational framing"
        - "Hate speech examples → allowed for awareness, blocked for promotion"
        
    user_age_policy:
      general: "Most restrictive (assume minor)"
      verified_adult: "Allow mature themes, still block critical categories"
      
  architecture:
    input_layer:
      model: "DeBERTa-v3 fine-tuned on safety data (< 20ms latency)"
      fallback: "Regex patterns for critical categories (< 1ms)"
      
    llm_layer:
      model: "Llama 3 70B with DPO safety alignment"
      system_prompt: "Detailed safety instructions with examples"
      
    output_layer:
      model: "Llama Guard 3 (6 harm categories, 8B parameters)"
      latency: "< 100ms on GPU"
      
    monitoring:
      dashboards: "Real-time harm rate, false positive rate, attack patterns"
      alerts: "Spike in any category triggers immediate investigation"
      review_queue: "Borderline cases sent to human moderators"
      
  results:
    harm_rate: "0.3 per 1000 requests (99.97% safe)"
    false_positive_rate: "3.2% (legitimate requests blocked)"
    attack_success_rate: "4.1% of adversarial probes bypass safety"
    user_satisfaction: "92% (down from 95% without safety — acceptable trade-off)"
```

---

## Interview Tip

> When asked about content safety: "I implement content safety as defense-in-depth with multiple layers. Input layer: fast classifier (DeBERTa fine-tuned on safety data, <20ms) catches obviously harmful requests before they reach the LLM — saves compute and prevents the model from even processing harmful prompts. LLM layer: safety-aligned model (RLHF/DPO trained) with detailed system prompt that specifies what content is never appropriate. Output layer: dedicated safety classifier (Llama Guard 3 or similar) evaluates the response before it reaches the user — catches cases where the model generates harmful content despite alignment. Key metrics I track: harm rate (harmful outputs per 1000 requests — target <1), false positive rate (legitimate requests blocked — target <5%), and attack success rate (jailbreaks bypassing all layers — target <5%). Context matters enormously — I implement context-aware policies: a medical professional asking about drug interactions gets a helpful response; the same question without medical context gets a safety redirect. For evaluation: I maintain a red team suite of 1000+ adversarial prompts across all harm categories, run it on every model update, and track attack success rate over time. New jailbreak patterns emerge weekly, so I continuously update the test suite and retrain classifiers. The trade-off is always safety vs. utility — over-filtering makes the system useless. I measure user satisfaction alongside safety metrics and calibrate thresholds to minimize harm without excessive false positives."

---

## Common Mistakes

1. **Single-layer safety** — Only relying on the LLM's safety training without input/output classifiers. Models can be jailbroken. Solution: defense-in-depth with input classification, aligned model, output classification, and monitoring.

2. **No context awareness** — Blocking all mentions of sensitive topics regardless of context. Medical professionals, educators, and researchers need to discuss sensitive topics legitimately. Solution: context-aware policies that adjust based on user role, domain, and framing.

3. **English-only safety** — Safety classifiers trained only on English. Users exploit non-English prompts to bypass safety. Solution: multilingual safety classifiers, or translation-based approaches for languages without training data.

4. **Static safety evaluation** — Testing safety once before launch. New jailbreak techniques emerge weekly. Solution: continuous red teaming (automated + manual), regular re-evaluation, and rapid response to new attack patterns.

5. **Ignoring false positives** — Focusing only on harm rate without measuring over-blocking. Users abandon platforms that refuse legitimate requests. Solution: track false positive rate, measure user satisfaction, A/B test safety threshold changes.

---

## Key Takeaways

- Content safety: preventing AI from generating harmful content (violence, hate, CSAM, self-harm)
- Defense-in-depth: input filtering → aligned model → output classification → monitoring
- Tools: Llama Guard 3, OpenAI Moderation, NeMo Guardrails, custom fine-tuned classifiers
- Metrics: harm rate (<1/1000), false positive rate (<5%), attack success rate (<5%)
- Context matters: medical/educational contexts need different policies than general audience
- Multilingual: safety must work in all languages (attacks exploit non-English gaps)
- Continuous: new jailbreaks emerge weekly — ongoing red teaming and classifier updates
- Trade-off: safety vs. utility — over-filtering kills usability, under-filtering causes harm
- Regulatory: DSA (EU), Online Safety Act (UK), platform liability for AI-generated content
- Human-in-loop: borderline cases escalated to human moderators for review
