# Explainability

## The Problem / Why This Matters

When an ML model denies someone a loan, rejects a job application, or recommends a medical treatment, people have a right to understand why. Explainability (also called interpretability) is the ability to understand and communicate how a model makes decisions. This isn't just an ethical nicety — it's a legal requirement under GDPR (General Data Protection Regulation) Article 22 (right to explanation for automated decisions), a regulatory requirement for financial models (SR 11-7 model risk management), and a practical necessity for debugging, trust-building, and model improvement. In 2026, explainability spans two worlds: (1) traditional ML models where techniques like SHAP (SHapley Additive exPlanations), LIME (Local Interpretable Model-agnostic Explanations), and feature importance provide clear answers, and (2) LLMs (Large Language Models) where explainability is fundamentally harder (billions of parameters, emergent behaviors, no clear feature-to-output mapping). For ML engineers, explainability means choosing between inherently interpretable models (linear regression, decision trees, GAMs) and post-hoc explanation methods (SHAP, LIME, attention visualization), implementing explanations in production systems, and communicating model behavior to non-technical stakeholders.

---

## The Analogy

Think of model explainability like explaining a doctor's diagnosis:

- **Inherently interpretable model** = A doctor who shows their reasoning step by step: "Your blood pressure is high (feature 1), your cholesterol is elevated (feature 2), and you have family history (feature 3). Each factor contributes this much to your risk score." Clear, auditable, trustworthy.
- **Black-box model with post-hoc explanation** = A doctor who uses advanced imaging and AI to diagnose but then must explain to the patient: "Based on many factors I analyzed, you have condition X." The explanation is a simplification of a complex process — useful but imperfect.
- **No explanation** = "You have condition X. Trust me." Legally problematic, ethically questionable, and practically useless for getting a second opinion or understanding what to change.

---

## Deep Dive

### Explainability Spectrum

```yaml
Explainability_Spectrum:
  inherently_interpretable:
    models:
      linear_regression: "Each feature has a coefficient (weight). Prediction = sum of (feature × weight)."
      logistic_regression: "Same as linear, with sigmoid. Coefficient = log-odds contribution."
      decision_tree: "If-then rules. Follow the tree from root to leaf."
      rule_lists: "Ordered list of if-then rules."
      gam: "Generalized Additive Models — individual feature effects visualized as curves."
      ebm: "Explainable Boosting Machine (InterpretML) — GAM with interactions."
    
    advantages:
      - "Explanation IS the model (no approximation)"
      - "Globally interpretable (understand entire model behavior)"
      - "Legally defensible (can audit every decision)"
      - "Easy to debug (identify problematic features immediately)"
    
    disadvantages:
      - "Often less accurate than complex models"
      - "Can't capture complex interactions (linear models)"
      - "Decision trees can be too deep to interpret"
      
  post_hoc_explanation:
    methods:
      shap: "Shapley values — mathematically rigorous feature attribution"
      lime: "Local surrogate — approximate complex model with simple one locally"
      integrated_gradients: "Gradient-based attribution for neural networks"
      attention: "Attention weights in transformers (controversial — not always faithful)"
      counterfactual: "What would need to change for a different outcome?"
      
    advantages:
      - "Can explain any model (model-agnostic for SHAP/LIME)"
      - "Provides local explanations (per-prediction)"
      - "Works with state-of-the-art complex models"
      
    disadvantages:
      - "Approximation — may not perfectly represent model's true reasoning"
      - "Can be manipulated (adversarial explanations)"
      - "Computationally expensive (SHAP for large models)"
      - "LIME can be unstable (different runs, different explanations)"
```

### SHAP (SHapley Additive exPlanations)

```yaml
SHAP:
  what: "Game-theory based approach to feature attribution"
  principle: "Each feature's contribution = average marginal contribution across all possible feature subsets"
  
  properties:
    local_accuracy: "Feature contributions sum to prediction (complete decomposition)"
    consistency: "If a feature's contribution increases in model, its SHAP value increases"
    missingness: "Features not in model get zero attribution"
    
  variants:
    tree_shap: "Exact SHAP for tree models (XGBoost, LightGBM) — O(TLD) fast"
    kernel_shap: "Model-agnostic SHAP (any model) — slower, approximation"
    deep_shap: "SHAP for neural networks (DeepLIFT + Shapley)"
    
  interpretations:
    positive_shap: "Feature pushes prediction UP (toward positive class / higher value)"
    negative_shap: "Feature pushes prediction DOWN (toward negative class / lower value)"
    magnitude: "Absolute value = importance of feature for THIS prediction"
    
  global_explanations:
    feature_importance: "Mean absolute SHAP values across dataset"
    summary_plot: "SHAP values for all features, all samples (beeswarm plot)"
    dependence_plot: "How one feature's SHAP value changes with feature value"
```

### Implementation

```python
# Explainability implementation for production ML

"""
Production explainability system using SHAP, LIME, and counterfactuals.
Provides both local (per-prediction) and global (model-level) explanations.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Explanation for a single prediction."""
    prediction: float
    feature_contributions: Dict[str, float]  # feature → SHAP value
    top_positive_features: List[Tuple[str, float]]  # Features pushing prediction up
    top_negative_features: List[Tuple[str, float]]  # Features pushing prediction down
    base_value: float  # Expected prediction (average)
    explanation_method: str
    confidence: float = 1.0
    counterfactual: Optional[Dict] = None  # What would change the outcome


class ProductionExplainer:
    """
    Production-ready explainability system.
    
    Provides explanations for individual predictions and global model behavior.
    Designed for serving-time use (fast enough for real-time).
    """
    
    def __init__(
        self,
        model,
        explainer_type: str = "tree_shap",
        background_data=None,
        feature_names: List[str] = None,
    ):
        self.model = model
        self.explainer_type = explainer_type
        self.feature_names = feature_names or []
        
        # Initialize SHAP explainer
        self._init_explainer(background_data)
    
    def _init_explainer(self, background_data):
        """Initialize appropriate SHAP explainer based on model type."""
        # In production: use shap library
        # import shap
        # if self.explainer_type == "tree_shap":
        #     self.explainer = shap.TreeExplainer(self.model)
        # elif self.explainer_type == "kernel_shap":
        #     self.explainer = shap.KernelExplainer(self.model.predict, background_data)
        self.background_data = background_data
        logger.info(f"Initialized {self.explainer_type} explainer")
    
    def explain_prediction(
        self, features: Dict[str, float], prediction: float
    ) -> Explanation:
        """
        Explain a single prediction.
        
        Returns:
        - Which features contributed most (positive and negative)
        - How much each feature shifted the prediction from baseline
        - What the prediction would be without each feature
        """
        # Compute SHAP values
        feature_array = np.array([features[f] for f in self.feature_names])
        shap_values = self._compute_shap(feature_array)
        
        # Map to feature names
        contributions = {}
        for i, name in enumerate(self.feature_names):
            contributions[name] = float(shap_values[i])
        
        # Sort by absolute contribution
        sorted_features = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        top_positive = [(f, v) for f, v in sorted_features if v > 0][:5]
        top_negative = [(f, v) for f, v in sorted_features if v < 0][:5]
        
        return Explanation(
            prediction=prediction,
            feature_contributions=contributions,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            base_value=self._get_base_value(),
            explanation_method=self.explainer_type,
        )
    
    def explain_for_user(
        self, features: Dict[str, float], prediction: float
    ) -> str:
        """
        Generate human-readable explanation.
        
        For non-technical users (customers, business stakeholders).
        """
        explanation = self.explain_prediction(features, prediction)
        
        # Generate natural language
        lines = []
        lines.append(f"Prediction: {prediction:.2f}")
        lines.append(f"(Baseline average: {explanation.base_value:.2f})")
        lines.append("")
        
        if explanation.top_positive_features:
            lines.append("Factors that INCREASED the score:")
            for feature, value in explanation.top_positive_features[:3]:
                feature_val = features.get(feature, "N/A")
                lines.append(f"  • {self._human_name(feature)} = {feature_val} (+{value:.3f})")
        
        if explanation.top_negative_features:
            lines.append("Factors that DECREASED the score:")
            for feature, value in explanation.top_negative_features[:3]:
                feature_val = features.get(feature, "N/A")
                lines.append(f"  • {self._human_name(feature)} = {feature_val} ({value:.3f})")
        
        return "\n".join(lines)
    
    def generate_counterfactual(
        self,
        features: Dict[str, float],
        target_prediction: float,
        immutable_features: List[str] = None,
    ) -> Optional[Dict]:
        """
        Generate counterfactual explanation.
        
        "What would need to change for a different outcome?"
        
        Example: Loan denied. Counterfactual shows:
        "If your income were $5K higher OR credit score 50 points higher,
        the loan would be approved."
        
        Respects immutable features (can't change age, race).
        """
        immutable = set(immutable_features or [])
        mutable_features = [f for f in self.feature_names if f not in immutable]
        
        # Find minimal changes that flip the prediction
        # (Simplified greedy approach — production uses DiCE or other libraries)
        current_pred = self.model.predict(features)
        changes = {}
        
        # Sort features by SHAP value (change most impactful first)
        explanation = self.explain_prediction(features, current_pred)
        sorted_by_impact = sorted(
            explanation.feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        modified_features = dict(features)
        for feature, shap_value in sorted_by_impact:
            if feature in immutable:
                continue
            
            # Try adjusting this feature
            # Direction: if we want higher prediction, increase features with positive SHAP correlation
            adjustment_direction = 1 if target_prediction > current_pred else -1
            step = float(np.std([features.get(feature, 0)])) * 0.1 or 1.0
            
            for _ in range(10):
                modified_features[feature] += step * adjustment_direction
                new_pred = self.model.predict(modified_features)
                
                if (adjustment_direction > 0 and new_pred >= target_prediction) or \
                   (adjustment_direction < 0 and new_pred <= target_prediction):
                    changes[feature] = {
                        "from": features[feature],
                        "to": modified_features[feature],
                        "change": modified_features[feature] - features[feature],
                    }
                    break
            
            if changes:
                break  # Found minimal change
        
        return {
            "original_prediction": current_pred,
            "target_prediction": target_prediction,
            "changes_needed": changes,
            "immutable_features_respected": list(immutable),
        }
    
    def _compute_shap(self, features: np.ndarray) -> np.ndarray:
        """Compute SHAP values for a single instance."""
        # In production: use self.explainer.shap_values(features)
        # Simplified placeholder — returns random values for illustration
        n_features = len(features)
        # Actual implementation would use shap library
        return np.random.randn(n_features) * 0.1
    
    def _get_base_value(self) -> float:
        """Get base value (expected prediction over background data)."""
        # In production: self.explainer.expected_value
        return 0.5  # Placeholder
    
    def _human_name(self, feature: str) -> str:
        """Convert feature name to human-readable label."""
        replacements = {
            "annual_income": "Annual Income",
            "credit_score": "Credit Score",
            "debt_to_income": "Debt-to-Income Ratio",
            "employment_years": "Years Employed",
            "loan_amount": "Loan Amount",
        }
        return replacements.get(feature, feature.replace("_", " ").title())


class GlobalExplanation:
    """
    Global model explanations (understand model behavior overall).
    
    Not per-prediction, but across the entire model:
    - Which features matter most globally?
    - How does each feature affect predictions?
    - Are there surprising interactions?
    """
    
    def __init__(self, explainer: ProductionExplainer, sample_data: np.ndarray):
        self.explainer = explainer
        self.sample_data = sample_data
    
    def feature_importance(self) -> Dict[str, float]:
        """
        Global feature importance (mean absolute SHAP).
        
        Most reliable importance metric — accounts for interactions
        and non-linear effects (unlike permutation importance).
        """
        all_shap_values = []
        
        for sample in self.sample_data:
            shap_vals = self.explainer._compute_shap(sample)
            all_shap_values.append(shap_vals)
        
        shap_matrix = np.array(all_shap_values)
        
        # Mean absolute SHAP per feature
        importance = np.mean(np.abs(shap_matrix), axis=0)
        
        return {
            name: float(imp)
            for name, imp in zip(self.explainer.feature_names, importance)
        }
    
    def feature_dependence(self, feature_idx: int) -> Dict:
        """
        How does one feature's effect change with its value?
        
        Shows non-linear relationships:
        e.g., income has diminishing effect above $200K
        """
        feature_values = self.sample_data[:, feature_idx]
        shap_values_for_feature = []
        
        for sample in self.sample_data:
            shap_vals = self.explainer._compute_shap(sample)
            shap_values_for_feature.append(shap_vals[feature_idx])
        
        return {
            "feature_name": self.explainer.feature_names[feature_idx],
            "feature_values": feature_values.tolist(),
            "shap_values": shap_values_for_feature,
            "correlation": float(np.corrcoef(feature_values, shap_values_for_feature)[0, 1]),
        }
```

### LLM Explainability

```yaml
LLM_Explainability:
  challenges:
    - "Billions of parameters — can't point to individual feature contributions"
    - "Emergent behaviors — model does things training didn't explicitly teach"
    - "Context-dependent — same input can produce different outputs"
    - "Attention ≠ explanation — attention weights don't reliably indicate reasoning"
    
  approaches:
    chain_of_thought:
      what: "Ask model to show reasoning step by step"
      limitation: "Post-hoc rationalization — model may reason differently than it explains"
      value: "Useful for users even if not perfectly faithful to internal process"
      
    attribution:
      what: "Which input tokens influenced which output tokens?"
      methods: "Integrated gradients, attention rollout, probing classifiers"
      tools: "inseq library, TransformerLens (mechanistic interpretability)"
      
    probing:
      what: "What knowledge is stored where in the model?"
      approach: "Train simple classifiers on hidden states to find information location"
      
    mechanistic_interpretability:
      what: "Reverse-engineer model circuits to understand computations"
      state_2026: "Active research area (Anthropic, DeepMind). Scaling to large models."
      tools: "TransformerLens, SAE (Sparse Autoencoder) features, circuit analysis"
      
    constitutional_explanations:
      what: "Model explains decisions referencing explicit principles"
      implementation: "System prompt includes principles, model cites which principle applies"
```

---

## How It Works in Practice

### Explainability in a Lending System

```yaml
Lending_Explanations:
  scenario: "Customer denied a $50K personal loan"
  
  model_explanation:
    prediction: "Deny (probability of default: 0.68)"
    base_rate: "Average default probability: 0.12"
    
    top_factors_increasing_risk:
      - "Debt-to-income ratio: 0.65 (high) → +0.25 to risk score"
      - "Credit utilization: 92% (high) → +0.18 to risk score"
      - "Recent hard inquiries: 5 in last 6 months → +0.08 to risk score"
      
    top_factors_decreasing_risk:
      - "Employment tenure: 8 years → -0.05 from risk score"
      - "Income: $95,000 → -0.03 from risk score"
      
  customer_facing_explanation:
    message: |
      Your loan application was not approved because your current debt level
      relative to your income (debt-to-income ratio of 65%) exceeds our 
      lending criteria. Additionally, your credit utilization (92%) suggests
      existing financial strain.
      
      To improve your chances in the future, consider:
      • Reducing outstanding debt (target debt-to-income below 40%)
      • Lowering credit card utilization (target below 30%)
      • Avoiding new credit applications for 6 months
      
  regulatory_compliance:
    requirement: "ECOA (Equal Credit Opportunity Act) requires specific adverse action reasons"
    format: "Up to 4 principal reasons for denial, ordered by significance"
    delivery: "Written notice within 30 days of decision"
```

---

## Interview Tip

> When asked about explainability: "I approach explainability at three levels: model choice, explanation method, and audience. First, model choice — for high-stakes decisions (lending, hiring), I prefer inherently interpretable models like EBMs (Explainable Boosting Machines) or GAMs (Generalized Additive Models) when accuracy permits. These don't need post-hoc explanation — the model IS the explanation. When I need complex models (deep learning, large ensembles), I use post-hoc methods. SHAP is my default for feature attribution — it has mathematical guarantees (consistency, local accuracy) and works for any model. TreeSHAP for tree models is exact and fast (milliseconds per prediction). KernelSHAP for model-agnostic (but slower). For actionable explanations, I add counterfactuals: 'If your income were $5K higher, the decision would change.' This respects immutable features (can't suggest changing age or race). Implementation in production: SHAP values computed at serving time (fast enough for TreeSHAP), stored alongside predictions for audit trail. For customer-facing: translate SHAP values into natural language (top 3 factors, human-readable feature names). For model debugging: global SHAP summary plots reveal unexpected feature relationships. Key distinction: explanation ≠ justification. A model might explain 'zip code was the top factor' — that's transparent but might reveal an unfair proxy. Explainability helps DETECT problems, not hide them."

---

## Common Mistakes

1. **Using attention as explanation** — Showing attention weights as "what the model focused on." Research shows attention doesn't reliably indicate causal reasoning. Solution: use gradient-based attribution or SHAP, not attention weights, for faithful explanations.

2. **Explaining black box when interpretable model suffices** — Using a deep neural network + SHAP when a GAM/EBM achieves similar accuracy and IS interpretable. Solution: always try interpretable models first. Only use complex models + post-hoc explanation when accuracy gap is significant.

3. **Global explanation only** — Providing average feature importance but no per-prediction explanations. Customers want to know why THEIR specific application was denied. Solution: implement local explanations (per-prediction SHAP) alongside global understanding.

4. **Explanations that don't match the model** — Using LIME (which can be unstable) and getting different explanations for similar inputs. Undermines trust. Solution: prefer SHAP (mathematically grounded, consistent), validate explanations are stable for similar inputs.

5. **No actionable recommendations** — Explaining "your credit score was the top factor" but not saying what to do about it. Solution: counterfactual explanations that show what needs to change, respecting immutable attributes (don't suggest changing age/race).

---

## Key Takeaways

- Explainability: understanding and communicating how models make decisions
- Inherently interpretable: linear models, decision trees, GAMs, EBMs (explanation IS the model)
- Post-hoc methods: SHAP (gold standard), LIME (simpler but less stable), counterfactuals (actionable)
- SHAP properties: local accuracy, consistency, mathematically grounded feature attribution
- Counterfactuals: "what would need to change?" — respects immutable features
- Audience matters: technical (SHAP plots) vs. customer-facing (natural language) vs. regulatory (structured reasons)
- LLM explainability: chain-of-thought (useful but not faithful), mechanistic interpretability (research frontier)
- Legal requirements: GDPR right to explanation, ECOA adverse action notices, SR 11-7
- Production: compute explanations at serving time, store for audit trail
- Explainability detects problems: revealing proxies, unexpected features, unfair patterns
