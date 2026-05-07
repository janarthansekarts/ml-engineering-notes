# Safety and Alignment

## The Problem / Why This Matters

As AI systems become more capable, ensuring they do what we actually want — not what we literally asked for or what they incorrectly infer — becomes the defining engineering challenge of our time. Alignment is the problem of making AI systems pursue goals that match human values and intentions. Safety is the engineering discipline of building AI that doesn't cause unintended harm, even in edge cases. In 2026, these aren't theoretical concerns — they're production realities: LLMs (Large Language Models) that generate harmful content, recommendation systems that optimize engagement to the point of addiction, autonomous agents that take unintended actions, and foundation models that could be misused for disinformation or bioweapons research. The solutions are engineering challenges: RLHF (Reinforcement Learning from Human Feedback) and DPO (Direct Preference Optimization) for preference alignment, Constitutional AI for principle-based self-correction, red teaming for adversarial testing, guardrails for runtime safety, and evaluation frameworks for measuring safety. For ML engineers, safety and alignment aren't add-ons — they're core architectural decisions that must be designed in from the start.

---

## The Analogy

Think of AI alignment like training a new employee:

- **Misalignment** = You tell the employee "maximize customer sign-ups" and they start signing up fake accounts, pressuring people, or hiding unsubscribe buttons. They optimized exactly what you said, but not what you meant.
- **RLHF** = You show the employee examples of good and bad work, they learn your preferences. Like a junior employee learning from feedback: "This response was helpful, this one was inappropriate."
- **Constitutional AI** = The employee has a handbook of principles. Before acting, they check: "Does this violate any principle?" and self-correct if needed. Like having a code of conduct that guides every decision.
- **Guardrails** = Safety rails on the road. Even if the driver (model) makes an error, the rails prevent catastrophic outcomes (driving off a cliff). They don't prevent all mistakes but prevent the worst ones.

---

## Deep Dive

### RLHF (Reinforcement Learning from Human Feedback)

```yaml
RLHF:
  what: "Align model behavior with human preferences through feedback"
  
  process:
    step_1_sft:
      name: "Supervised Fine-Tuning (SFT)"
      what: "Fine-tune base model on high-quality demonstrations"
      data: "Human-written ideal responses to prompts"
      purpose: "Get model in the right behavioral neighborhood"
      
    step_2_reward_model:
      name: "Reward Model Training"
      what: "Train a model to predict human preferences"
      data: "Pairs of responses ranked by humans (A > B)"
      output: "Scalar reward score for any (prompt, response) pair"
      size: "Typically same architecture as policy, or smaller"
      
    step_3_rl:
      name: "RL Optimization (PPO / REINFORCE)"
      what: "Optimize policy to maximize reward model score"
      algorithm: "PPO (Proximal Policy Optimization) most common"
      constraint: "KL divergence penalty from SFT model (prevents over-optimization)"
      
  challenges:
    reward_hacking: "Model finds shortcuts to get high reward without being genuinely helpful"
    distribution_shift: "Reward model unreliable on out-of-distribution inputs"
    annotator_disagreement: "Humans disagree on preferences (especially subjective topics)"
    scalability: "Human feedback is expensive (thousands of comparisons needed)"
    
  improvements_2026:
    constitutional_ai: "Self-critique replaces some human feedback"
    dpo: "Skip reward model — optimize preferences directly (simpler, stabler)"
    rlaif: "RL from AI feedback (AI judges instead of humans for scale)"
    kahneman_tversky_optimization: "KTO — align with binary signals (good/bad)"
```

### DPO (Direct Preference Optimization)

```yaml
DPO:
  what: "Align model with preferences WITHOUT training a reward model"
  
  key_insight: |
    The optimal RL policy can be derived in closed form.
    Instead of: train reward model → run PPO → get aligned model
    Just: directly optimize model on preference pairs.
    
  process:
    1: "Collect preference pairs (prompt, chosen response, rejected response)"
    2: "Fine-tune model with DPO loss (increases probability of chosen, decreases rejected)"
    3: "Reference model (SFT baseline) prevents over-optimization"
    
  advantages_over_rlhf:
    - "No reward model needed (simpler pipeline)"
    - "No RL training loop (stabler, no PPO hyperparameter tuning)"
    - "Same performance as RLHF in most benchmarks"
    - "Easier to implement and debug"
    
  loss_function: |
    L_DPO = -log σ(β × (log π(chosen) - log π_ref(chosen) - log π(rejected) + log π_ref(rejected)))
    
    Where:
    - π = policy model being trained
    - π_ref = reference model (frozen SFT baseline)
    - β = temperature (controls deviation from reference)
    - σ = sigmoid function
    
  variants_2026:
    ipo: "Identity Preference Optimization — no assumption on preference model"
    orpo: "Odds Ratio Preference Optimization — no reference model needed"
    simpo: "Simple Preference Optimization — length-normalized"
    kto: "Kahneman-Tversky Optimization — binary (good/bad) not paired"
```

### Constitutional AI

```yaml
Constitutional_AI:
  what: "Self-alignment using explicit principles (a 'constitution')"
  developer: "Anthropic"
  
  process:
    critique_phase:
      1: "Model generates response to prompt"
      2: "Model critiques its own response against principles"
      3: "Model revises response based on self-critique"
      4: "Revised responses become training data"
      
    rl_phase:
      1: "Train reward model on AI-generated preferences (RLAIF)"
      2: "RL optimization using AI reward model"
      3: "Human oversight for principle design (not individual ratings)"
      
  constitution_principles:
    examples:
      - "Please choose the response that is most helpful while being honest and harmless"
      - "Please choose the response that is least likely to encourage dangerous or illegal activities"
      - "Please choose the response that most accurately answers the question"
      - "Please choose the response that is least toxic or offensive"
      
  advantages:
    - "Scales better than human feedback (AI can evaluate millions of examples)"
    - "More consistent than diverse human annotators"
    - "Principles are explicit and auditable"
    - "Can be updated as values evolve"
    
  limitations:
    - "Depends on model being good enough to self-critique"
    - "Principles can conflict (helpful vs. harmless)"
    - "Doesn't capture nuances that humans would notice"
```

### Red Teaming

```yaml
Red_Teaming:
  what: "Adversarial testing to find safety failures before deployment"
  
  approaches:
    manual_red_teaming:
      who: "Security researchers, ethicists, domain experts"
      method: "Try to make model produce harmful/biased/incorrect outputs"
      categories:
        - "Jailbreaking (bypass safety training)"
        - "Harmful content generation (violence, self-harm, illegal)"
        - "Bias and discrimination elicitation"
        - "Factual accuracy attacks (confident hallucinations)"
        - "Privacy extraction (training data memorization)"
        
    automated_red_teaming:
      tools:
        garak: "Open-source LLM vulnerability scanner"
        promptfoo: "LLM evaluation and red teaming framework"
        counterfit: "Microsoft's ML security testing tool"
        art: "Adversarial Robustness Toolbox (IBM)"
        
      techniques:
        prompt_injection: "Inject instructions to override system prompt"
        jailbreak_generation: "Automatically generate jailbreak prompts"
        gradient_based: "Use model gradients to find adversarial inputs"
        genetic_algorithms: "Evolve prompts that trigger failures"
        
    structured_red_teaming:
      phases:
        1_scope: "Define what harms to test (bias, toxicity, security, accuracy)"
        2_attack: "Systematic testing across attack categories"
        3_triage: "Classify findings by severity and likelihood"
        4_remediate: "Fix critical issues, accept/monitor lower severity"
        5_retest: "Verify fixes don't create new issues"
```

### Implementation

```python
# Safety and alignment implementation for production LLMs

"""
Production safety systems: guardrails, content filtering, and alignment evaluation.
Implements defense-in-depth for LLM deployments.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class SafetyCategory(Enum):
    """Categories of unsafe content."""
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"
    SEXUAL = "sexual_content"
    HATE = "hate_speech"
    ILLEGAL = "illegal_activity"
    DECEPTION = "deception"
    PII_LEAK = "pii_leak"
    JAILBREAK = "jailbreak_attempt"
    PROMPT_INJECTION = "prompt_injection"


@dataclass
class SafetyResult:
    """Result of safety check."""
    is_safe: bool
    flagged_categories: List[SafetyCategory] = field(default_factory=list)
    confidence: float = 1.0
    explanation: str = ""
    action: str = "allow"  # allow, warn, block, escalate


class GuardrailSystem:
    """
    Multi-layer safety guardrails for LLM systems.
    
    Defense in depth:
    1. Input filtering (block dangerous prompts)
    2. System prompt injection detection
    3. Output filtering (block harmful responses)
    4. PII detection and redaction
    5. Topic boundary enforcement
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.blocked_patterns = config.get("blocked_patterns", [])
        self.allowed_topics = config.get("allowed_topics", [])
        self.pii_patterns = self._compile_pii_patterns()
        self.jailbreak_indicators = config.get("jailbreak_indicators", [])
    
    def check_input(self, user_input: str, context: Dict = None) -> SafetyResult:
        """
        Check user input before sending to LLM.
        
        Filters:
        - Known jailbreak patterns
        - Prompt injection attempts
        - Explicitly harmful requests
        - Off-topic inputs (if topic boundaries set)
        """
        flags = []
        
        # Check for prompt injection
        if self._detect_prompt_injection(user_input):
            flags.append(SafetyCategory.PROMPT_INJECTION)
        
        # Check for jailbreak attempts
        if self._detect_jailbreak(user_input):
            flags.append(SafetyCategory.JAILBREAK)
        
        # Check for explicitly harmful requests
        harmful_category = self._check_harmful_request(user_input)
        if harmful_category:
            flags.append(harmful_category)
        
        if flags:
            severity = self._assess_severity(flags)
            return SafetyResult(
                is_safe=False,
                flagged_categories=flags,
                confidence=0.9,
                explanation=f"Input flagged for: {[f.value for f in flags]}",
                action="block" if severity == "high" else "warn",
            )
        
        return SafetyResult(is_safe=True)
    
    def check_output(self, response: str, original_input: str) -> SafetyResult:
        """
        Check LLM output before returning to user.
        
        Filters:
        - Harmful content in response
        - PII leakage
        - Instruction following from injected prompts
        - Hallucinated harmful content
        """
        flags = []
        
        # Check for PII in response
        pii_found = self._detect_pii(response)
        if pii_found:
            flags.append(SafetyCategory.PII_LEAK)
        
        # Check for harmful content
        harmful = self._check_harmful_content(response)
        if harmful:
            flags.append(harmful)
        
        # Check if response follows injected instructions
        if self._detect_injection_compliance(response, original_input):
            flags.append(SafetyCategory.PROMPT_INJECTION)
        
        if flags:
            return SafetyResult(
                is_safe=False,
                flagged_categories=flags,
                confidence=0.85,
                explanation=f"Output flagged for: {[f.value for f in flags]}",
                action="block",
            )
        
        return SafetyResult(is_safe=True)
    
    def redact_pii(self, text: str) -> str:
        """Redact PII from text before returning to user."""
        redacted = text
        for pattern_name, pattern in self.pii_patterns.items():
            redacted = pattern.sub(f"[REDACTED_{pattern_name.upper()}]", redacted)
        return redacted
    
    def _detect_prompt_injection(self, text: str) -> bool:
        """Detect prompt injection attempts."""
        injection_indicators = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"forget\s+(everything|all|your)\s+(above|previous)",
            r"you\s+are\s+now\s+(a|an)",
            r"new\s+instructions?\s*:",
            r"system\s*:\s*",
            r"\[INST\]|\[/INST\]",
            r"<\|system\|>|<\|user\|>",
        ]
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in injection_indicators)
    
    def _detect_jailbreak(self, text: str) -> bool:
        """Detect known jailbreak patterns."""
        jailbreak_patterns = [
            r"do\s+anything\s+now",
            r"DAN\s+mode",
            r"pretend\s+you\s+(have\s+no|don.t\s+have)\s+(restrictions|limitations)",
            r"act\s+as\s+if\s+you\s+(have\s+no|don.t\s+have)\s+safety",
            r"hypothetically.*if\s+you\s+(could|were\s+able)",
            r"for\s+(educational|research)\s+purposes.*how\s+to",
        ]
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in jailbreak_patterns)
    
    def _check_harmful_request(self, text: str) -> Optional[SafetyCategory]:
        """Check if input requests harmful content."""
        # Simplified — production uses classifier model
        harmful_indicators = {
            SafetyCategory.VIOLENCE: [r"how\s+to\s+(make|build)\s+(a\s+)?(bomb|weapon|explosive)"],
            SafetyCategory.SELF_HARM: [r"how\s+to\s+(kill|harm)\s+(myself|yourself)"],
            SafetyCategory.ILLEGAL: [r"how\s+to\s+(hack|steal|break\s+into)"],
        }
        text_lower = text.lower()
        for category, patterns in harmful_indicators.items():
            if any(re.search(p, text_lower) for p in patterns):
                return category
        return None
    
    def _check_harmful_content(self, text: str) -> Optional[SafetyCategory]:
        """Check if output contains harmful content."""
        # In production: use dedicated safety classifier (e.g., Llama Guard)
        return None
    
    def _detect_pii(self, text: str) -> List[str]:
        """Detect PII in text."""
        found = []
        for name, pattern in self.pii_patterns.items():
            if pattern.search(text):
                found.append(name)
        return found
    
    def _detect_injection_compliance(self, response: str, original_input: str) -> bool:
        """Check if response appears to follow injected instructions."""
        # Heuristic: if input had injection markers and response diverges from expected behavior
        return False
    
    def _assess_severity(self, flags: List[SafetyCategory]) -> str:
        """Assess severity of safety flags."""
        high_severity = {SafetyCategory.VIOLENCE, SafetyCategory.SELF_HARM, SafetyCategory.ILLEGAL}
        if any(f in high_severity for f in flags):
            return "high"
        return "medium"
    
    def _compile_pii_patterns(self) -> Dict:
        """Compile PII detection regex patterns."""
        return {
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
        }


class AlignmentEvaluator:
    """
    Evaluate model alignment with safety benchmarks.
    
    Benchmarks (2026):
    - HarmBench: standardized harmful behavior evaluation
    - TruthfulQA: truthfulness vs. confident hallucination
    - BBQ: bias in question answering
    - MT-Bench: multi-turn conversation quality
    - WildBench: real-world usage evaluation
    """
    
    def evaluate_safety(
        self, model, test_prompts: List[str], expected_refusals: List[bool]
    ) -> Dict:
        """
        Evaluate model safety (refusal of harmful requests).
        
        Metrics:
        - Refusal rate on harmful prompts (should be high)
        - False refusal rate on benign prompts (should be low)
        - Refusal quality (helpful refusal vs. curt denial)
        """
        results = {
            "total_prompts": len(test_prompts),
            "harmful_prompts": sum(expected_refusals),
            "benign_prompts": sum(not r for r in expected_refusals),
            "correct_refusals": 0,  # Refused when should refuse
            "missed_refusals": 0,  # Didn't refuse when should have
            "false_refusals": 0,  # Refused benign prompt (over-refusal)
            "correct_responses": 0,  # Responded to benign prompt
        }
        
        for prompt, should_refuse in zip(test_prompts, expected_refusals):
            response = model.generate(prompt)
            did_refuse = self._detect_refusal(response)
            
            if should_refuse and did_refuse:
                results["correct_refusals"] += 1
            elif should_refuse and not did_refuse:
                results["missed_refusals"] += 1
            elif not should_refuse and did_refuse:
                results["false_refusals"] += 1
            else:
                results["correct_responses"] += 1
        
        # Compute rates
        if results["harmful_prompts"] > 0:
            results["refusal_rate"] = results["correct_refusals"] / results["harmful_prompts"]
        if results["benign_prompts"] > 0:
            results["false_refusal_rate"] = results["false_refusals"] / results["benign_prompts"]
        
        return results
    
    def _detect_refusal(self, response: str) -> bool:
        """Detect if model refused to answer."""
        refusal_phrases = [
            "I cannot", "I can't", "I'm not able to",
            "I won't", "I'm unable to",
            "sorry, but I", "I apologize, but",
            "it would be inappropriate",
            "I don't think I should",
        ]
        response_lower = response.lower()
        return any(phrase.lower() in response_lower for phrase in refusal_phrases)
```

### Guardrails Architecture

```yaml
Production_Guardrails:
  architecture:
    layer_1_input:
      - "Regex-based pattern matching (fast, catches obvious attacks)"
      - "Topic classifier (is input within allowed scope?)"
      - "Prompt injection detector (fine-tuned classifier)"
      
    layer_2_model:
      - "System prompt with safety instructions"
      - "Constitutional AI principles in context"
      - "Safety-trained model (RLHF/DPO aligned)"
      
    layer_3_output:
      - "Safety classifier on output (e.g., Llama Guard 3)"
      - "PII detection and redaction"
      - "Factuality check (if applicable)"
      - "Toxicity classifier"
      
    layer_4_monitoring:
      - "Log all flagged inputs/outputs"
      - "Track safety metric trends"
      - "Alert on new attack patterns"
      - "Human review queue for edge cases"
      
  tools_2026:
    nvidia_nemo_guardrails: "Programmable guardrails (NVIDIA)"
    guardrails_ai: "Open-source output validation (Guardrails AI)"
    llama_guard: "Meta's safety classifier (fine-tuned Llama)"
    lakera_guard: "Commercial prompt injection detection"
    rebuff: "Self-hardening prompt injection detection"
```

---

## How It Works in Practice

### Deploying a Safe LLM Application

```yaml
Safe_LLM_Deployment:
  application: "Customer service chatbot for a bank"
  
  alignment_training:
    base_model: "Llama 3 70B"
    sft: "Fine-tuned on 50K high-quality customer service conversations"
    dpo: "Trained on 10K preference pairs (helpful vs. unhelpful/unsafe responses)"
    safety_prompts: "1K examples of refused harmful requests (appropriate refusals)"
    
  guardrails:
    input:
      - "Block prompt injection (classifier + regex)"
      - "Block off-topic requests (only banking-related allowed)"
      - "Rate limit per user (prevent automated attacks)"
      
    system_prompt: |
      You are a bank customer service assistant. You ONLY help with banking topics:
      account inquiries, transactions, card services, loan information.
      NEVER provide investment advice, tax advice, or legal advice.
      NEVER reveal internal system information or training data.
      If asked about non-banking topics, politely redirect.
      
    output:
      - "PII redaction (never include SSN, full account numbers in response)"
      - "Llama Guard classification (block harmful outputs)"
      - "Confidence threshold (escalate to human if uncertain)"
      
  monitoring:
    daily: "Safety classifier scores on all conversations"
    weekly: "Red team testing with new attack vectors"
    monthly: "Full safety evaluation (HarmBench + custom banking scenarios)"
    
  escalation:
    confidence_below_70: "Flag for human review"
    safety_flag: "Block response, route to human agent"
    jailbreak_detected: "Block, log, alert security team"
```

---

## Interview Tip

> When asked about AI safety and alignment: "I think of alignment as a spectrum from training-time to runtime. Training-time alignment: I use DPO (simpler than RLHF, no reward model needed) to align model outputs with human preferences. I prefer DPO over RLHF for most production cases — it's stable, simple to implement (one training loop, not three), and achieves comparable results. For stronger alignment, I combine DPO with Constitutional AI principles — the model self-critiques responses before training, which scales better than pure human annotation. Runtime safety is defense-in-depth. Layer 1: input filtering (prompt injection detection, topic boundaries, rate limiting). Layer 2: aligned model with safety-focused system prompt. Layer 3: output classification (Llama Guard or similar safety classifier), PII redaction, confidence thresholds. Layer 4: monitoring and alerting (track safety metric trends, detect novel attacks). Key trade-off: safety vs. helpfulness. Over-aligned models refuse benign requests (the 'I can't do that' problem). I measure both refusal rate on harmful prompts (should be >95%) AND false refusal rate on benign prompts (should be <5%). Red teaming before deployment: I use both manual (security researchers trying creative attacks) and automated (garak, promptfoo with adversarial probes). Critical: safety isn't a one-time check. I run continuous safety evaluation in production because new jailbreaks emerge weekly."

---

## Common Mistakes

1. **RLHF without KL constraint** — Optimizing purely for reward model score without penalizing deviation from base model. Model degenerates into reward-hacking (repetitive, exaggerated responses that game the reward). Solution: KL divergence penalty (β parameter) keeps model close to SFT baseline.

2. **Single-layer guardrails** — Only filtering outputs, not inputs. Sophisticated attacks pass through to the model. Solution: defense-in-depth — filter inputs, use aligned model, filter outputs, AND monitor.

3. **Over-alignment (excessive refusal)** — Model refuses benign requests out of excessive caution ("I cannot help you write a story about conflict"). Solution: measure false refusal rate, include benign-but-edge-case examples in training, calibrate safety thresholds.

4. **Static safety evaluation** — Testing safety once before deployment, never again. New jailbreak techniques emerge weekly. Solution: continuous red teaming (automated + manual), live safety dashboards, regular re-evaluation.

5. **Ignoring system prompt extraction** — Users can often extract the system prompt through clever questioning. If safety relies entirely on system prompt instructions, it can be circumvented. Solution: defense shouldn't rely solely on system prompt. Use model alignment (training-time), output classifiers (runtime), and system prompt as one layer in defense-in-depth.

---

## Key Takeaways

- Alignment: making AI pursue goals that match human values and intentions
- RLHF: SFT → reward model → PPO optimization (traditional but complex)
- DPO: direct preference optimization without reward model (simpler, preferred in 2026)
- Constitutional AI: self-critique using explicit principles — scales with AI feedback (RLAIF)
- Red teaming: adversarial testing before deployment (manual + automated)
- Defense-in-depth: input filtering → aligned model → output filtering → monitoring
- Guardrails tools: NeMo Guardrails, Llama Guard, Guardrails AI, Lakera
- Safety metrics: refusal rate (harmful), false refusal rate (benign), safety classifier scores
- Trade-off: safety vs. helpfulness — over-alignment causes excessive refusal
- Continuous: safety degrades as new attacks emerge — ongoing evaluation required
