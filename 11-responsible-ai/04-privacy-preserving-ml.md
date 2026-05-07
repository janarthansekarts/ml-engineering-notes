# Privacy-Preserving ML

## The Problem / Why This Matters

ML models are data-hungry — the more data, the better the model. But data often contains sensitive personal information: medical records, financial transactions, location history, communication patterns. Training a model on this data creates multiple privacy risks: (1) the training data can be extracted from models (membership inference attacks reveal if someone's data was used), (2) model outputs can leak private information (LLMs memorize and regurgitate training data), (3) centralized data collection creates honeypot targets for breaches. Privacy-preserving ML solves this tension: how to build effective models WITHOUT exposing individual data. In 2026, this field has matured with three major approaches: differential privacy (mathematical guarantees that no individual's data is identifiable in model outputs), federated learning (train models across devices without centralizing data), and secure computation (encrypted data processing where even the model trainer can't see the data). With GDPR (General Data Protection Regulation) fines reaching billions of euros, HIPAA (Health Insurance Portability and Accountability Act) violations in healthcare AI, and increasing regulation of LLM training data, privacy-preserving ML is no longer optional — it's a core engineering requirement for any production ML system handling personal data.

---

## The Analogy

Think of privacy-preserving ML like taking a census:

- **No privacy (current default)** = Government collects everyone's detailed personal records in one database. Great for analysis, terrible for privacy. One breach exposes everything.
- **Differential privacy** = Each person's response is randomly perturbed before submission. The aggregate statistics are accurate, but no individual answer can be recovered. Even if the database leaks, individual data is protected by mathematical noise.
- **Federated learning** = Census workers visit each household, compute local statistics, and only report aggregates to headquarters. The actual personal data never leaves the home.
- **Secure computation** = Census responses are encrypted before submission. The analysis runs on encrypted data. Even the analysts never see raw responses — only the encrypted final results.

---

## Deep Dive

### Differential Privacy

```yaml
Differential_Privacy:
  definition: |
    A mechanism M satisfies ε-differential privacy if for any two datasets
    D and D' that differ by one record, and any output set S:
    P(M(D) ∈ S) ≤ e^ε × P(M(D') ∈ S)
    
  meaning: |
    Whether or not any one person's data is in the dataset,
    the output looks approximately the same.
    An adversary can't tell if you were in the training data.
    
  parameters:
    epsilon:
      what: "Privacy budget — lower = more privacy"
      typical_values:
        strong: "ε = 0.1 to 1.0 (high privacy, some utility loss)"
        moderate: "ε = 1.0 to 10.0 (balanced)"
        weak: "ε > 10.0 (minimal privacy guarantee)"
      apple_use: "ε = 2 to 8 for emoji usage statistics"
      google_use: "ε = 1 to 10 for Chrome usage analytics (RAPPOR)"
      
    delta:
      what: "Probability of privacy failure"
      typical: "δ < 1/n (less than probability of any one person)"
      
  mechanisms:
    laplace: "Add Laplace noise (for numeric queries)"
    gaussian: "Add Gaussian noise (for (ε,δ)-DP)"
    exponential: "For non-numeric outputs (select from options)"
    
  composition:
    sequential: "Each query uses budget. Total ε = sum of per-query ε"
    parallel: "Queries on disjoint data don't compound"
    advanced: "Rényi DP, concentrated DP give tighter composition"
    
  dp_sgd:
    what: "Differentially Private Stochastic Gradient Descent"
    how: |
      1. Clip per-sample gradients (bound sensitivity)
      2. Aggregate clipped gradients
      3. Add calibrated Gaussian noise to aggregate
      4. Update model parameters
    trade_off: "Privacy ↔ accuracy ↔ training time"
    tools: "Opacus (PyTorch), TensorFlow Privacy, JAX DP"
```

### Federated Learning

```yaml
Federated_Learning:
  what: "Train models across distributed devices/institutions WITHOUT centralizing data"
  
  architecture:
    participants: "Phones, hospitals, banks — each with local data"
    coordinator: "Central server orchestrates training (doesn't see data)"
    
  process:
    1: "Server sends current model to participants"
    2: "Each participant trains locally on their data"
    3: "Participants send model updates (gradients) — NOT data"
    4: "Server aggregates updates (e.g., FedAvg)"
    5: "Repeat until convergence"
    
  variants:
    cross_device:
      what: "Millions of mobile devices (Google Keyboard, Apple Siri)"
      challenges: "Device heterogeneity, unreliable connectivity, non-IID data"
      
    cross_silo:
      what: "Handful of institutions (hospitals, banks)"
      challenges: "Regulatory compliance, data heterogeneity, institution incentives"
      
  aggregation_algorithms:
    fedavg: "Average model weights (simple, works well in practice)"
    fedprox: "FedAvg with proximal term (handles heterogeneous data)"
    scaffold: "Corrects client drift for non-IID data"
    fedopt: "Server-side optimizer (Adam, Adagrad) for better convergence"
    
  privacy_attacks:
    gradient_inversion: "Reconstruct training data from gradients"
    membership_inference: "Determine if specific data was used by specific client"
    model_poisoning: "Malicious client sends bad updates"
    
  defenses:
    secure_aggregation: "Server can't see individual updates (only aggregate)"
    dp_federated: "Combine DP with federated learning (add noise to updates)"
    robust_aggregation: "Detect and exclude malicious/outlier updates"
    
  real_world:
    google_keyboard: "Next-word prediction trained on-device (federated)"
    apple_siri: "Voice model improvements without centralizing audio"
    healthcare: "Multi-hospital model training (without sharing patient data)"
    finance: "Fraud detection across banks (without sharing transactions)"
```

### Secure Computation

```yaml
Secure_Computation:
  types:
    homomorphic_encryption:
      what: "Compute on encrypted data without decrypting"
      levels:
        partial: "PHE — supports one operation (addition OR multiplication)"
        somewhat: "SHE — limited number of both operations"
        fully: "FHE (Fully Homomorphic Encryption) — arbitrary computation on encrypted data"
      state_2026:
        - "FHE still ~1000x slower than plaintext (improving)"
        - "Practical for specific use cases (private inference, simple analytics)"
        - "Libraries: Microsoft SEAL, Zama Concrete-ML, OpenFHE"
        
    secure_multiparty_computation:
      what: "Multiple parties jointly compute function without revealing individual inputs"
      example: "3 hospitals compute average patient outcome without sharing records"
      protocols: "Secret sharing, garbled circuits, oblivious transfer"
      tools: "MP-SPDZ, CrypTen (PyTorch), PySyft"
      
    trusted_execution_environments:
      what: "Hardware-isolated secure enclaves (data decrypted only inside TEE)"
      hardware: "Intel SGX, AMD SEV, ARM TrustZone, AWS Nitro Enclaves"
      approach: "Data encrypted at rest and in transit. Decrypted only in TEE for computation."
      limitation: "Side-channel attacks possible. Trust hardware vendor."
```

### Implementation

```python
# Privacy-preserving ML implementation patterns

"""
Privacy-preserving ML training with differential privacy and federated learning.
Production patterns for handling sensitive data in ML pipelines.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudget:
    """Track differential privacy budget consumption."""
    total_epsilon: float  # Total privacy budget
    total_delta: float
    consumed_epsilon: float = 0.0
    consumed_delta: float = 0.0
    
    @property
    def remaining_epsilon(self) -> float:
        return self.total_epsilon - self.consumed_epsilon
    
    @property
    def is_exhausted(self) -> bool:
        return self.consumed_epsilon >= self.total_epsilon
    
    def consume(self, epsilon: float, delta: float = 0.0):
        """Consume privacy budget for a query/training step."""
        if self.consumed_epsilon + epsilon > self.total_epsilon:
            raise PrivacyBudgetExhausted(
                f"Would exceed budget: {self.consumed_epsilon + epsilon} > {self.total_epsilon}"
            )
        self.consumed_epsilon += epsilon
        self.consumed_delta += delta
        logger.info(
            f"Privacy budget: {self.consumed_epsilon:.4f}/{self.total_epsilon} ε consumed "
            f"({self.remaining_epsilon:.4f} remaining)"
        )


class PrivacyBudgetExhausted(Exception):
    """Raised when privacy budget is fully consumed."""
    pass


class DPTrainer:
    """
    Differentially Private model training using DP-SGD.
    
    DP-SGD (Differentially Private Stochastic Gradient Descent):
    1. Clip per-sample gradients (bound sensitivity)
    2. Add calibrated noise to aggregate gradient
    3. Track privacy budget consumption
    """
    
    def __init__(
        self,
        model,
        privacy_budget: PrivacyBudget,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.1,
        batch_size: int = 256,
        dataset_size: int = 10000,
    ):
        self.model = model
        self.budget = privacy_budget
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        
        # Compute per-step privacy cost
        # Using Rényi DP accounting for tight bounds
        self.sampling_rate = batch_size / dataset_size
    
    def train_step(self, batch_features: np.ndarray, batch_labels: np.ndarray) -> Dict:
        """
        One training step with differential privacy.
        
        1. Compute per-sample gradients
        2. Clip each gradient to max_grad_norm (bound sensitivity)
        3. Average clipped gradients
        4. Add Gaussian noise (calibrated to sensitivity and noise_multiplier)
        5. Update model
        """
        # Step 1: Per-sample gradients (requires per-sample gradient support)
        per_sample_grads = self._compute_per_sample_gradients(batch_features, batch_labels)
        
        # Step 2: Clip gradients (bound sensitivity)
        clipped_grads = self._clip_gradients(per_sample_grads)
        
        # Step 3: Average
        avg_grad = np.mean(clipped_grads, axis=0)
        
        # Step 4: Add noise
        noise_std = self.noise_multiplier * self.max_grad_norm / self.batch_size
        noise = np.random.normal(0, noise_std, size=avg_grad.shape)
        noisy_grad = avg_grad + noise
        
        # Step 5: Update model
        # self.model.update(noisy_grad)
        
        # Track privacy budget
        step_epsilon = self._compute_step_epsilon()
        self.budget.consume(step_epsilon)
        
        return {
            "grad_norm_before_clip": float(np.mean(np.linalg.norm(per_sample_grads, axis=1))),
            "fraction_clipped": float(np.mean(
                np.linalg.norm(per_sample_grads, axis=1) > self.max_grad_norm
            )),
            "noise_std": noise_std,
            "epsilon_consumed": step_epsilon,
            "total_epsilon": self.budget.consumed_epsilon,
        }
    
    def _clip_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """Clip per-sample gradients to max_grad_norm."""
        norms = np.linalg.norm(gradients, axis=1, keepdims=True)
        # Clip: if norm > max, scale down
        clip_factor = np.minimum(1.0, self.max_grad_norm / (norms + 1e-8))
        return gradients * clip_factor
    
    def _compute_per_sample_gradients(
        self, features: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Compute gradients for each sample individually."""
        # In practice: use Opacus (PyTorch) or TF Privacy
        # These libraries handle per-sample gradient computation efficiently
        n_samples = features.shape[0]
        n_params = 100  # placeholder
        return np.random.randn(n_samples, n_params)
    
    def _compute_step_epsilon(self) -> float:
        """Compute privacy cost of one training step using RDP accounting."""
        # Simplified — in practice use privacy accounting libraries
        # (autodp, tensorflow_privacy.compute_dp_sgd_privacy)
        # RDP gives much tighter bounds than naive composition
        return self.sampling_rate * self.noise_multiplier * 0.01  # Simplified


class FederatedTrainer:
    """
    Federated learning coordinator.
    
    Orchestrates training across distributed participants
    without accessing their data.
    """
    
    def __init__(
        self,
        global_model,
        num_rounds: int = 100,
        min_participants_per_round: int = 10,
        fraction_fit: float = 0.1,  # Fraction of clients to train per round
    ):
        self.global_model = global_model
        self.num_rounds = num_rounds
        self.min_participants = min_participants_per_round
        self.fraction_fit = fraction_fit
        self.history: List[Dict] = []
    
    def run_round(self, available_clients: List["FederatedClient"]) -> Dict:
        """
        Execute one federated training round.
        
        1. Select clients
        2. Send model to clients
        3. Clients train locally
        4. Collect and aggregate updates
        5. Update global model
        """
        # Select clients for this round
        n_selected = max(
            self.min_participants,
            int(len(available_clients) * self.fraction_fit)
        )
        selected = np.random.choice(
            available_clients, size=n_selected, replace=False
        )
        
        # Send current model to selected clients
        global_weights = self._get_model_weights()
        
        # Clients train locally and return updates
        client_updates = []
        client_data_sizes = []
        for client in selected:
            update = client.train_local(global_weights)
            client_updates.append(update["weight_delta"])
            client_data_sizes.append(update["data_size"])
        
        # Aggregate updates (FedAvg — weighted by data size)
        aggregated_update = self._fedavg(client_updates, client_data_sizes)
        
        # Update global model
        self._apply_update(aggregated_update)
        
        result = {
            "round": len(self.history) + 1,
            "clients_selected": n_selected,
            "clients_available": len(available_clients),
            "avg_local_loss": np.mean([u.get("loss", 0) for u in client_updates]) if client_updates else 0,
        }
        self.history.append(result)
        return result
    
    def _fedavg(
        self, updates: List[np.ndarray], data_sizes: List[int]
    ) -> np.ndarray:
        """
        Federated Averaging — weight updates by client data size.
        
        Clients with more data have proportionally more influence
        (they've seen more of the data distribution).
        """
        total_data = sum(data_sizes)
        weighted_sum = sum(
            update * (size / total_data)
            for update, size in zip(updates, data_sizes)
        )
        return weighted_sum
    
    def _get_model_weights(self) -> np.ndarray:
        """Get current global model weights."""
        # In practice: model.state_dict() (PyTorch) or model.get_weights() (TF)
        return np.zeros(100)  # Placeholder
    
    def _apply_update(self, update: np.ndarray):
        """Apply aggregated update to global model."""
        # global_weights += update
        pass


@dataclass
class FederatedClient:
    """
    Federated learning participant (device/institution).
    
    Trains locally on private data, shares only model updates.
    """
    client_id: str
    local_data_size: int
    local_epochs: int = 5
    learning_rate: float = 0.01
    
    def train_local(self, global_weights: np.ndarray) -> Dict:
        """
        Train on local data using current global model as starting point.
        
        Returns model update (delta from global weights), NOT the data.
        """
        # 1. Initialize local model with global weights
        local_weights = global_weights.copy()
        
        # 2. Train for local_epochs on local data
        # In practice: standard training loop on local dataset
        for epoch in range(self.local_epochs):
            # local_weights = train_one_epoch(local_weights, self.local_data)
            pass
        
        # 3. Compute update (delta)
        weight_delta = local_weights - global_weights
        
        return {
            "weight_delta": weight_delta,
            "data_size": self.local_data_size,
            "loss": 0.5,  # Placeholder
        }


class PrivacyAuditor:
    """
    Audit ML models for privacy risks.
    
    Tests:
    - Membership inference: Can we tell if a specific sample was in training data?
    - Model inversion: Can we reconstruct training samples from model?
    - Attribute inference: Can we infer sensitive attributes from model outputs?
    """
    
    def membership_inference_attack(
        self,
        model,
        member_data: np.ndarray,
        non_member_data: np.ndarray,
    ) -> Dict:
        """
        Test if model leaks membership information.
        
        Attack: train a classifier to distinguish members vs non-members
        based on model's confidence/loss on each sample.
        
        If attack accuracy > 50%, model leaks membership.
        Higher accuracy = more privacy leakage.
        """
        # Get model's behavior on members and non-members
        member_signals = self._extract_signals(model, member_data)
        non_member_signals = self._extract_signals(model, non_member_data)
        
        # Simple threshold attack (more sophisticated: train meta-classifier)
        # Members typically have lower loss / higher confidence
        member_losses = member_signals["losses"]
        non_member_losses = non_member_signals["losses"]
        
        # Find optimal threshold
        all_losses = np.concatenate([member_losses, non_member_losses])
        labels = np.concatenate([np.ones(len(member_losses)), np.zeros(len(non_member_losses))])
        
        best_acc = 0.5
        for threshold in np.percentile(all_losses, range(1, 100)):
            predictions = (all_losses < threshold).astype(int)  # Lower loss → predicted member
            acc = np.mean(predictions == labels)
            best_acc = max(best_acc, acc)
        
        return {
            "attack_accuracy": float(best_acc),
            "privacy_risk": "HIGH" if best_acc > 0.7 else "MEDIUM" if best_acc > 0.6 else "LOW",
            "recommendation": (
                "Apply DP training" if best_acc > 0.6 
                else "Acceptable privacy level"
            ),
        }
    
    def _extract_signals(self, model, data: np.ndarray) -> Dict:
        """Extract signals used by membership inference attack."""
        # In practice: compute loss, confidence, gradient norm per sample
        n = len(data)
        return {
            "losses": np.random.exponential(1.0, size=n),
            "confidences": np.random.uniform(0.5, 1.0, size=n),
        }
```

---

## How It Works in Practice

### Healthcare Federated Learning

```yaml
Healthcare_Federated_Learning:
  scenario: "Train cancer detection model across 5 hospitals"
  
  constraints:
    - "HIPAA: patient data cannot leave hospital"
    - "Each hospital has different demographics, scanners, protocols"
    - "No hospital has enough data alone for robust model"
    
  solution:
    architecture: "Cross-silo federated learning with secure aggregation"
    
    setup:
      coordinator: "Cloud server (no access to patient data)"
      participants: "5 hospital data centers (behind firewall)"
      communication: "Encrypted model updates only (no patient data transmitted)"
      
    training_protocol:
      1: "Coordinator sends global model to all hospitals"
      2: "Each hospital trains on local patient scans (3 local epochs)"
      3: "Hospitals send encrypted model weight updates"
      4: "Secure aggregation: coordinator sees ONLY the sum of updates"
      5: "Global model updated, sent back for next round"
      6: "Repeat for 50 rounds"
      
    privacy_enhancements:
      dp_noise: "Each hospital adds DP noise to updates (ε=2 per round)"
      secure_aggregation: "Coordinator can't see individual hospital updates"
      minimum_update_norm: "Reject updates that could indicate single-patient influence"
      
    results:
      accuracy_centralized: "94.2% (if all data were pooled — impossible due to HIPAA)"
      accuracy_federated: "92.8% (federated — only 1.4% lower)"
      accuracy_single_hospital: "85.1% (best single hospital alone)"
      privacy_guarantee: "ε = 8 total (strong protection for individual patients)"
```

---

## Interview Tip

> When asked about privacy in ML: "I think of privacy-preserving ML along three dimensions: what guarantees do we need, where does computation happen, and what's the accuracy trade-off. For formal guarantees, differential privacy (DP) is the gold standard — it's mathematical, not just policy-based. I've used DP-SGD with Opacus for model training with ε=1-8 depending on sensitivity. Key parameters: max gradient norm (clip sensitivity), noise multiplier (calibrate noise), and privacy accountant (track budget using Rényi DP for tight bounds). For data locality constraints (HIPAA, GDPR), federated learning keeps data on-premises while training a shared model. Cross-silo (hospitals, banks) is more practical than cross-device (millions of phones) for enterprise. I combine FL with secure aggregation so the coordinator never sees individual updates, only the encrypted sum. The trade-off is always accuracy: DP training typically costs 2-5% accuracy, federated 1-3% compared to centralized. But the alternative — centralized sensitive data — creates breach risk, regulatory violations, and trust erosion. For LLMs in 2026, the key risk is training data memorization. I implement: (1) deduplication of training data (reduces memorization), (2) membership inference testing to quantify leakage, (3) output filtering to catch regurgitated PII, and (4) DP fine-tuning for sensitive domain adaptation."

---

## Common Mistakes

1. **Claiming federated learning = private** — FL keeps data local but model updates can leak information (gradient inversion attacks reconstruct training data from gradients). Solution: combine FL with secure aggregation AND differential privacy for actual privacy guarantees.

2. **Setting ε too high** — Using ε=100 "for privacy" but this provides essentially zero protection. Solution: ε should be single digits (ideally 1-10). If accuracy requires ε>50, the DP guarantee is meaningless — reconsider the approach.

3. **Ignoring composition** — Each query/epoch consumes privacy budget. 100 training epochs with ε=0.1 each = ε=10 total (sequential composition). Not tracking composition means overspending budget. Solution: use privacy accountant (Rényi DP, GDP) for tight composition tracking.

4. **Privacy as afterthought** — Training a model normally, then trying to add privacy. DP must be integrated into training from the start (DP-SGD replaces normal SGD). Solution: design for privacy from the beginning — it affects hyperparameters, architecture, and data pipeline.

5. **Thinking anonymization = privacy** — Removing names/IDs but keeping other features. Research shows 87% of Americans can be re-identified from zip code + birth date + gender alone. Solution: differential privacy provides formal guarantees against re-identification, unlike naive anonymization.

---

## Key Takeaways

- Differential privacy: mathematical guarantee — model output independent of any one individual's data
- ε (epsilon): privacy budget (lower = more privacy). Practical range: 1-10 for strong guarantees
- DP-SGD: clip per-sample gradients + add noise — standard for private model training
- Federated learning: train across distributed data without centralizing. FL ≠ private without DP
- Secure aggregation: coordinator can't see individual updates (only encrypted sum)
- Homomorphic encryption: compute on encrypted data (slow but improving — 2026: ~100x overhead)
- Privacy budget: finite resource — each query/epoch consumes it. Track with RDP accountant
- Membership inference: test if model leaks training data (>60% attack accuracy = privacy risk)
- LLM memorization: large models memorize training data — deduplicate, filter outputs, DP fine-tune
- Accuracy trade-off: DP training costs 2-5% accuracy — but breach/violation cost is far higher
