# AI Governance

## The Problem / Why This Matters

AI governance is the framework of policies, processes, and controls that ensures AI systems are developed and deployed responsibly within an organization and in compliance with regulations. Without governance, teams ship models with undocumented risks, no audit trail, inconsistent ethical standards, and unknown regulatory exposure. In 2026, governance isn't optional — the EU AI Act (effective August 2025, enforcement from 2026) requires risk classification, conformity assessments, and documentation for high-risk AI systems. US Executive Order 14110 mandates safety evaluations for frontier models. Industry-specific regulations (Basel III/IV for banking AI, FDA guidelines for clinical AI) add domain requirements. ML engineers must understand governance because it directly shapes what they build and how: model documentation (model cards), risk assessment before deployment, approval workflows, lineage tracking, and ongoing monitoring obligations. The alternative — governance theater (paper compliance without engineering integration) — creates false security while exposing the organization to regulatory fines (up to 7% of global revenue under EU AI Act), reputational damage, and actual harm from ungoverned AI.

---

## The Analogy

Think of AI governance like air traffic control:

- **No governance** = Every pilot flies wherever they want, at whatever altitude, no communication. Planes eventually collide. In AI: teams ship models independently with no coordination, standards, or oversight.
- **Governance framework** = Air traffic control system: defined altitudes (risk tiers), required flight plans (model documentation), clearance before takeoff (deployment approval), continuous radar monitoring (production monitoring), and incident investigation (post-deployment audit).
- **Over-governance** = Every flight needs 6 months of committee review. Planes sit on the tarmac. In AI: excessive bureaucracy where every model change requires legal review, killing innovation. Good governance is fast for low-risk, thorough for high-risk.

---

## Deep Dive

### Regulatory Landscape (2026)

```yaml
Regulatory_Landscape_2026:
  eu_ai_act:
    status: "In force. Full enforcement from August 2026."
    scope: "Any AI system deployed in or affecting EU citizens"
    
    risk_classification:
      unacceptable_risk:
        examples:
          - "Social scoring by governments"
          - "Real-time biometric identification in public (with exceptions)"
          - "Manipulation exploiting vulnerabilities"
          - "Emotion recognition in workplace/education"
        action: "BANNED"
        
      high_risk:
        examples:
          - "Biometric identification and categorization"
          - "Safety components in critical infrastructure"
          - "Education (admissions, grading)"
          - "Employment (recruiting, HR decisions)"
          - "Essential services (credit scoring, insurance)"
          - "Law enforcement (predictive policing)"
          - "Border control and immigration"
          - "Administration of justice"
        requirements:
          - "Risk management system"
          - "Data governance and quality requirements"
          - "Technical documentation"
          - "Record-keeping (logging)"
          - "Transparency to users"
          - "Human oversight mechanisms"
          - "Accuracy, robustness, cybersecurity"
        action: "Conformity assessment required before deployment"
        
      limited_risk:
        examples: "Chatbots, emotion detection, deepfakes"
        action: "Transparency obligations (inform user they're interacting with AI)"
        
      minimal_risk:
        examples: "Spam filters, AI in games, inventory management"
        action: "No specific requirements (voluntary codes of practice)"
        
    gpai_rules:
      what: "General-Purpose AI models (foundation models, LLMs)"
      all_gpai:
        - "Technical documentation"
        - "EU copyright law compliance"
        - "Summary of training data"
      systemic_risk_gpai:
        threshold: "Training compute > 10^25 FLOP (or Commission designation)"
        additional:
          - "Model evaluation (red teaming, safety testing)"
          - "Incident monitoring and reporting"
          - "Adequate cybersecurity"
          - "Energy consumption reporting"
          
    penalties:
      unacceptable_risk: "Up to €35M or 7% global annual turnover"
      high_risk_violations: "Up to €15M or 3% global annual turnover"
      incorrect_information: "Up to €7.5M or 1% global annual turnover"
      
  us_regulation:
    executive_order_14110:
      what: "Safe, Secure, and Trustworthy Development and Use of AI"
      requirements:
        - "Safety testing for dual-use foundation models"
        - "Red teaming before deployment"
        - "Reporting to government for models trained with >10^26 FLOP"
        - "Watermarking AI-generated content"
        
    nist_ai_rmf:
      what: "AI Risk Management Framework (voluntary but widely adopted)"
      functions: "Govern, Map, Measure, Manage"
      
    state_laws:
      colorado: "AI Act — governance requirements for high-risk AI in consumer decisions"
      california: "Proposed AI transparency and accountability legislation"
      
  sector_specific:
    finance: "SR 11-7 (model risk management), Fair Lending (ECOA), BSA/AML"
    healthcare: "FDA AI/ML guidance, HIPAA for health data in AI"
    automotive: "UNECE regulations for autonomous driving AI"
```

### Model Documentation (Model Cards)

```yaml
Model_Cards:
  what: "Standardized documentation for ML models"
  purpose: "Transparency, accountability, informed decision-making about model use"
  
  standard_sections:
    model_details:
      - "Model name and version"
      - "Developer/organization"
      - "Model type (architecture)"
      - "Training date and data cutoff"
      - "License"
      
    intended_use:
      - "Primary intended uses"
      - "Primary intended users"
      - "Out-of-scope uses (explicitly state what it should NOT be used for)"
      
    training_data:
      - "Data sources and composition"
      - "Preprocessing steps"
      - "Data limitations and biases"
      
    evaluation:
      - "Metrics used"
      - "Overall performance"
      - "Disaggregated performance (by demographic group, use case)"
      - "Intersectional performance"
      
    fairness_analysis:
      - "Fairness metrics computed"
      - "Known biases"
      - "Mitigation steps taken"
      
    limitations:
      - "Known limitations"
      - "Failure modes"
      - "Domains where model should not be trusted"
      
    ethical_considerations:
      - "Potential harms"
      - "Sensitive uses"
      - "Recommendations for responsible use"
      
  automation:
    tools:
      - "Google Model Cards Toolkit"
      - "Hugging Face model card metadata"
      - "Custom CI/CD integration (auto-generate from training pipeline)"
    integration: "Generate model card as artifact in training pipeline"
```

### Implementation

```python
# AI Governance implementation for ML platforms

"""
AI Governance framework: model registration, risk assessment, approval workflows,
and compliance tracking for enterprise ML deployments.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RiskTier(Enum):
    """EU AI Act inspired risk classification."""
    MINIMAL = "minimal"
    LIMITED = "limited"
    HIGH = "high"
    UNACCEPTABLE = "unacceptable"


class ApprovalStatus(Enum):
    """Model approval workflow status."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    RISK_REVIEW = "risk_review"
    TECHNICAL_REVIEW = "technical_review"
    ETHICS_REVIEW = "ethics_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED = "deployed"
    RETIRED = "retired"


@dataclass
class ModelCard:
    """Standardized model documentation."""
    model_name: str
    model_version: str
    developer: str
    description: str
    
    # Intended use
    intended_use: str
    intended_users: List[str]
    out_of_scope_uses: List[str]
    
    # Training
    training_data_description: str
    training_data_size: str
    training_date: str
    
    # Performance
    metrics: Dict[str, float]
    disaggregated_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Fairness
    fairness_metrics: Dict[str, float] = field(default_factory=dict)
    known_biases: List[str] = field(default_factory=list)
    bias_mitigations: List[str] = field(default_factory=list)
    
    # Limitations
    limitations: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    
    # Ethical considerations
    potential_harms: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    risk_tier: Optional[RiskTier] = None


@dataclass
class RiskAssessment:
    """AI risk assessment for a model."""
    model_name: str
    assessor: str
    date: str
    
    # Risk dimensions
    risk_tier: RiskTier
    impact_if_wrong: str  # What happens if model makes errors?
    affected_population: str  # Who is affected by model decisions?
    reversibility: str  # Can decisions be reversed?
    human_oversight: str  # Is there human-in-the-loop?
    
    # Specific risks identified
    identified_risks: List[Dict] = field(default_factory=list)
    # Each: {"risk": str, "likelihood": str, "impact": str, "mitigation": str}
    
    # Compliance
    applicable_regulations: List[str] = field(default_factory=list)
    compliance_requirements: List[str] = field(default_factory=list)
    
    # Decision
    recommendation: str = ""  # approve, reject, conditional_approve
    conditions: List[str] = field(default_factory=list)


class GovernanceRegistry:
    """
    Central registry for all AI models in an organization.
    
    Every model must be registered before deployment.
    Tracks: documentation, risk assessment, approvals, lineage, monitoring status.
    """
    
    def __init__(self):
        self.models: Dict[str, Dict] = {}
        self.approval_workflows: Dict[str, List[Dict]] = {}
    
    def register_model(
        self,
        model_card: ModelCard,
        risk_assessment: RiskAssessment,
        training_artifacts: Dict = None,
    ) -> str:
        """
        Register a model in the governance registry.
        
        Required before deployment. Creates audit trail.
        """
        model_id = f"{model_card.model_name}:{model_card.model_version}"
        
        self.models[model_id] = {
            "model_card": model_card,
            "risk_assessment": risk_assessment,
            "training_artifacts": training_artifacts or {},
            "status": ApprovalStatus.DRAFT,
            "registered_at": datetime.now().isoformat(),
            "audit_trail": [
                {
                    "action": "registered",
                    "timestamp": datetime.now().isoformat(),
                    "actor": model_card.developer,
                }
            ],
        }
        
        logger.info(f"Model registered: {model_id} (Risk: {risk_assessment.risk_tier.value})")
        return model_id
    
    def submit_for_approval(self, model_id: str) -> Dict:
        """
        Submit model for approval workflow.
        
        Workflow depends on risk tier:
        - Minimal: auto-approve with logging
        - Limited: technical review only
        - High: technical + ethics + risk review
        - Unacceptable: reject (not allowed to deploy)
        """
        if model_id not in self.models:
            raise ValueError(f"Model not registered: {model_id}")
        
        model = self.models[model_id]
        risk_tier = model["risk_assessment"].risk_tier
        
        if risk_tier == RiskTier.UNACCEPTABLE:
            model["status"] = ApprovalStatus.REJECTED
            return {
                "status": "rejected",
                "reason": "Unacceptable risk tier — deployment prohibited by policy",
            }
        
        # Determine required reviews
        required_reviews = self._get_required_reviews(risk_tier)
        
        model["status"] = ApprovalStatus.SUBMITTED
        self.approval_workflows[model_id] = [
            {"review_type": r, "status": "pending", "reviewer": None}
            for r in required_reviews
        ]
        
        self._add_audit_entry(model_id, "submitted_for_approval")
        
        return {
            "status": "submitted",
            "required_reviews": required_reviews,
            "estimated_timeline": self._estimate_timeline(risk_tier),
        }
    
    def approve_review(
        self,
        model_id: str,
        review_type: str,
        reviewer: str,
        decision: str,
        comments: str = "",
    ) -> Dict:
        """Record a review decision."""
        if model_id not in self.approval_workflows:
            raise ValueError(f"No pending workflow for: {model_id}")
        
        workflow = self.approval_workflows[model_id]
        for review in workflow:
            if review["review_type"] == review_type and review["status"] == "pending":
                review["status"] = decision  # "approved" or "rejected"
                review["reviewer"] = reviewer
                review["comments"] = comments
                review["timestamp"] = datetime.now().isoformat()
                break
        
        # Check if all reviews complete
        all_complete = all(r["status"] != "pending" for r in workflow)
        all_approved = all(r["status"] == "approved" for r in workflow)
        
        if all_complete:
            if all_approved:
                self.models[model_id]["status"] = ApprovalStatus.APPROVED
                self._add_audit_entry(model_id, "approved", reviewer)
            else:
                self.models[model_id]["status"] = ApprovalStatus.REJECTED
                self._add_audit_entry(model_id, "rejected", reviewer)
        
        return {
            "model_id": model_id,
            "review_type": review_type,
            "decision": decision,
            "all_reviews_complete": all_complete,
            "final_status": self.models[model_id]["status"].value,
        }
    
    def get_compliance_report(self) -> Dict:
        """
        Generate organization-wide compliance report.
        
        Shows: total models, risk distribution, compliance gaps, overdue reviews.
        """
        report = {
            "total_models": len(self.models),
            "by_risk_tier": {},
            "by_status": {},
            "compliance_gaps": [],
            "report_date": datetime.now().isoformat(),
        }
        
        for model_id, model_data in self.models.items():
            tier = model_data["risk_assessment"].risk_tier.value
            status = model_data["status"].value
            
            report["by_risk_tier"][tier] = report["by_risk_tier"].get(tier, 0) + 1
            report["by_status"][status] = report["by_status"].get(status, 0) + 1
            
            # Check for compliance gaps
            if model_data["status"] == ApprovalStatus.DEPLOYED:
                if not model_data.get("model_card"):
                    report["compliance_gaps"].append({
                        "model": model_id,
                        "gap": "Missing model card documentation",
                    })
                if not model_data.get("risk_assessment"):
                    report["compliance_gaps"].append({
                        "model": model_id,
                        "gap": "Missing risk assessment",
                    })
        
        return report
    
    def _get_required_reviews(self, risk_tier: RiskTier) -> List[str]:
        """Determine required reviews based on risk tier."""
        reviews = {
            RiskTier.MINIMAL: ["auto_review"],
            RiskTier.LIMITED: ["technical_review"],
            RiskTier.HIGH: ["technical_review", "ethics_review", "risk_review", "legal_review"],
        }
        return reviews.get(risk_tier, ["technical_review"])
    
    def _estimate_timeline(self, risk_tier: RiskTier) -> str:
        """Estimate approval timeline."""
        timelines = {
            RiskTier.MINIMAL: "Immediate (auto-approved)",
            RiskTier.LIMITED: "1-3 business days",
            RiskTier.HIGH: "2-4 weeks",
        }
        return timelines.get(risk_tier, "Unknown")
    
    def _add_audit_entry(self, model_id: str, action: str, actor: str = "system"):
        """Add entry to model's audit trail."""
        self.models[model_id]["audit_trail"].append({
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "actor": actor,
        })
```

### Governance Operating Model

```yaml
Governance_Operating_Model:
  roles:
    ai_ethics_board:
      responsibility: "Set organizational AI principles and policies"
      composition: "C-suite, legal, ethics, engineering, customer representatives"
      frequency: "Quarterly reviews, ad-hoc for high-risk decisions"
      
    model_risk_management:
      responsibility: "Review and approve high-risk models"
      activities: "Risk assessment, independent validation, ongoing monitoring"
      who: "Risk team, not the model developers (independent)"
      
    ml_engineering:
      responsibility: "Build compliant models with proper documentation"
      activities: "Model cards, fairness testing, lineage tracking, monitoring"
      
    legal_compliance:
      responsibility: "Map regulations to engineering requirements"
      activities: "Regulatory interpretation, compliance gap analysis, audit preparation"
      
  processes:
    model_lifecycle:
      1_ideation: "Use case review, initial risk classification"
      2_development: "Responsible development practices, bias testing"
      3_validation: "Independent evaluation, fairness assessment"
      4_approval: "Governance review (tier-appropriate)"
      5_deployment: "Monitoring setup, incident response plan"
      6_monitoring: "Ongoing performance, fairness, drift monitoring"
      7_retirement: "Decommission plan, successor model, data retention"
      
    incident_response:
      severity_1: "Model causing active harm → immediate takedown (<1 hour)"
      severity_2: "Bias/fairness failure detected → mitigation within 24 hours"
      severity_3: "Performance degradation → investigation within 1 week"
      
    periodic_reviews:
      quarterly: "All high-risk models re-evaluated"
      annual: "Full governance framework review and update"
      triggered: "Any incident, regulatory change, or significant model update"
```

---

## How It Works in Practice

### Enterprise AI Governance Implementation

```yaml
Enterprise_Governance:
  scenario: "Large bank implementing AI governance for EU AI Act compliance"
  
  inventory:
    total_models: 47
    high_risk: 12 (credit scoring, fraud, collections, hiring)
    limited_risk: 20 (chatbot, document processing, recommendations)
    minimal_risk: 15 (spam filters, internal analytics)
    
  implementation_phases:
    phase_1_inventory:
      duration: "2 months"
      activities:
        - "Catalog all AI/ML models across organization"
        - "Classify each by EU AI Act risk tier"
        - "Identify gaps (undocumented models, no owner)"
        - "Assign responsible owners"
        
    phase_2_documentation:
      duration: "3 months"
      activities:
        - "Create model cards for all high-risk models"
        - "Document training data lineage"
        - "Record fairness and performance metrics"
        - "Document intended use and limitations"
        
    phase_3_process:
      duration: "2 months"
      activities:
        - "Implement approval workflow (risk-tier based)"
        - "Establish AI Ethics Board"
        - "Create incident response procedures"
        - "Set up continuous monitoring dashboards"
        
    phase_4_automation:
      duration: "3 months"
      activities:
        - "Automate model card generation in CI/CD"
        - "Integrate fairness testing in training pipeline"
        - "Build governance registry (searchable, auditable)"
        - "Automate compliance reporting"
```

---

## Interview Tip

> When asked about AI governance: "I see governance as engineering infrastructure, not bureaucracy. Concretely: First, risk classification — not all models need the same oversight. I follow EU AI Act tiers: minimal (spam filters — auto-approve), limited (chatbots — transparency required), high (credit scoring — full conformity assessment). This prevents governance from being a bottleneck. Second, model documentation is automated: model cards generated in CI/CD from training pipeline metadata (metrics, data lineage, fairness results). Third, approval workflow is tier-appropriate: minimal-risk deploys with logging, high-risk requires independent validation plus ethics review. Fourth, ongoing monitoring — governance doesn't end at deployment. Continuous fairness monitoring, drift detection, and periodic re-evaluation. Fifth, audit trail — every decision (training data choice, hyperparameter, deployment approval) is logged with who, when, why. For EU AI Act compliance specifically: I ensure technical documentation covers training data description, evaluation results disaggregated by group, limitations, and human oversight mechanisms. For high-risk: conformity assessment, risk management system, and transparency to affected users. The key engineering principle: governance should be a paved road, not a roadblock. If doing the right thing is easy (automated documentation, integrated fairness testing, one-click audit reports), teams will actually do it."

---

## Common Mistakes

1. **Governance theater** — Creating policy documents that don't connect to engineering. Beautiful AI ethics principles on the website, but no actual enforcement in development pipeline. Solution: governance must be automated and integrated into CI/CD — model registration, documentation, and testing enforced by tooling.

2. **One-size-fits-all governance** — Requiring 4-week ethics board review for a spam filter. Teams route around governance because it's too slow. Solution: risk-tiered governance — fast approval for low-risk, thorough review for high-risk.

3. **Governance at deployment only** — No oversight during development, then a single approval gate before production. By then, addressing issues is expensive. Solution: governance checkpoints throughout the lifecycle (use case review, data audit, validation, deployment, monitoring).

4. **Ignoring existing models** — Building governance for new models while 50 legacy models run in production undocumented. Solution: inventory first — catalog all existing models, retroactively classify and document.

5. **No incident response plan** — Having governance but no plan for what happens when it fails (biased model discovered, harmful outputs in production). Solution: defined severity levels, response timelines, escalation paths, and post-incident review process.

---

## Key Takeaways

- AI governance: policies, processes, controls for responsible AI development and deployment
- EU AI Act: risk-based regulation (minimal/limited/high/unacceptable), enforcement from 2026
- Penalties: up to 7% global revenue for unacceptable-risk violations
- Risk classification: determines governance intensity (fast for low-risk, thorough for high-risk)
- Model cards: standardized documentation (intended use, performance, fairness, limitations)
- Approval workflows: risk-tiered (auto-approve minimal, multi-party review for high-risk)
- Governance registry: central catalog of all AI models (searchable, auditable, tracked)
- Automation: governance integrated into CI/CD (auto-generate docs, enforce testing, audit trails)
- Ongoing: governance doesn't end at deployment — continuous monitoring, periodic re-evaluation
- Anti-pattern: governance as bureaucracy — should be engineering infrastructure (paved roads, not roadblocks)
