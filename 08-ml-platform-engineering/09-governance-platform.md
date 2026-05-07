# Governance Platform

## The Problem / Why This Matters

As ML systems make consequential decisions (credit approvals, hiring recommendations, medical diagnoses, content moderation), organizations need governance: tracking what models are deployed, who approved them, what data they were trained on, whether they meet fairness criteria, and whether they comply with regulations. Without governance: models get deployed without review (one engineer pushes a biased model to production), nobody knows what's actually running (shadow models everywhere), there's no audit trail (regulator asks "how was this decision made?" and nobody can answer), and compliance violations go undetected (model violates EU AI Act but nobody checked). A governance platform provides: model registry with approval workflows (staging → review → production), model documentation (Model Cards with performance, limitations, intended use), compliance checks (automated bias testing, fairness validation), audit trails (complete history of who did what when), and access control (who can deploy to production?). In 2026, with the EU AI Act in effect, NIST AI RMF (Risk Management Framework) widely adopted, and increasing regulatory scrutiny globally, ML governance has moved from "nice to have" to "legally required" for many applications.

---

## The Analogy

Think of governance like the pharmaceutical drug approval process:

- **Without governance** = Any chemist can synthesize a compound and sell it as medicine. No testing requirements, no safety reviews, no tracking of side effects, no accountability.
- **With governance** = FDA approval process. Structured phases (pre-clinical, Phase 1/2/3 trials), documented evidence of safety and efficacy, review by independent board, post-market surveillance, and ability to recall if problems emerge.

ML governance is the "FDA process" for models. It doesn't slow down innovation — it ensures models are safe, fair, and accountable before they affect real people.

---

## Deep Dive

### Governance Framework

```yaml
Governance_Components:
  model_registry:
    what: "Central catalog of all models (development, staging, production)"
    information_tracked:
      - "Model name, version, framework"
      - "Training data version and lineage"
      - "Performance metrics (accuracy, fairness, latency)"
      - "Owner (team + individual)"
      - "Stage (development → staging → production → archived)"
      - "Approval history (who approved, when, why)"
      - "Dependencies (upstream data, feature sets, libraries)"
    implementation: "MLflow Model Registry, custom on metadata database"
    
  approval_workflows:
    what: "Required reviews before models reach production"
    stages:
      development:
        who_approves: "Self (no approval needed)"
        requirements: "None — experimentation is free"
        
      staging:
        who_approves: "Team lead"
        requirements:
          - "Basic documentation (what the model does)"
          - "Performance metrics on test set"
          - "No known critical issues"
          
      production:
        who_approves: "Team lead + ML platform team + (optional) compliance"
        requirements:
          - "Complete Model Card documentation"
          - "Fairness testing passed"
          - "Performance meets SLA thresholds"
          - "Latency within serving SLA"
          - "Security review (no data leakage)"
          - "Monitoring configured"
          - "Rollback plan documented"
          
      high_risk_production:
        who_approves: "Team lead + ML platform + compliance + legal"
        requirements:
          - "All production requirements +"
          - "Regulatory compliance check (EU AI Act, sector-specific)"
          - "Explainability report"
          - "Human-in-the-loop plan"
          - "Impact assessment"
          - "Regular audit schedule defined"
          
  model_documentation:
    model_card:
      what: "Standardized documentation for every production model"
      sections:
        - "Model details (type, framework, version, intended use)"
        - "Training data (source, size, date range, any filtering)"
        - "Evaluation metrics (overall + per-subgroup)"
        - "Limitations (known failure modes, out-of-scope use)"
        - "Ethical considerations (potential harms, mitigation)"
        - "Maintenance (retraining schedule, monitoring plan)"
      enforcement: "Required before production deployment (automated check)"
      
  audit_trail:
    what: "Complete history of all ML lifecycle events"
    events_tracked:
      - "Model trained (who, when, data version, code version)"
      - "Model evaluated (results, evaluator, date)"
      - "Approval granted/denied (who, when, reason)"
      - "Model deployed (version, environment, approver)"
      - "Model rolled back (reason, who initiated)"
      - "Monitoring alert (what triggered, response)"
      - "Model retrained (trigger, new version, comparison)"
      - "Model archived/decommissioned (reason, replacement)"
    retention: "7+ years (regulatory requirement for financial services)"
    immutable: "Audit entries cannot be modified or deleted"
```

### Automated Compliance Checks

```python
# Automated governance checks for ML models

"""
Runs automated compliance checks before model deployment.
Ensures models meet organizational and regulatory requirements.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ComplianceResult(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NEEDS_REVIEW = "needs_review"


@dataclass
class GovernanceCheckResult:
    """Result of a single governance check."""
    check_name: str
    result: ComplianceResult
    details: str
    evidence: dict  # Supporting data for the result
    remediation: Optional[str] = None  # How to fix if failed


class GovernanceChecks:
    """
    Automated governance checks run before production deployment.
    
    Check categories:
    1. Performance: Does the model meet quality thresholds?
    2. Fairness: Is the model equitable across protected groups?
    3. Documentation: Is the model properly documented?
    4. Security: Does the model handle data safely?
    5. Operational: Is monitoring and rollback configured?
    """
    
    def run_all_checks(
        self,
        model_version: str,
        risk_level: str = "standard",  # "low", "standard", "high"
    ) -> list:
        """Run all applicable governance checks for a model."""
        results = []
        
        # Performance checks (all models)
        results.extend(self.performance_checks(model_version))
        
        # Fairness checks (all models)
        results.extend(self.fairness_checks(model_version))
        
        # Documentation checks (all models)
        results.extend(self.documentation_checks(model_version))
        
        # Security checks (all models)
        results.extend(self.security_checks(model_version))
        
        # Operational checks (all models)
        results.extend(self.operational_checks(model_version))
        
        # Additional checks for high-risk models
        if risk_level == "high":
            results.extend(self.regulatory_checks(model_version))
            results.extend(self.explainability_checks(model_version))
        
        return results
    
    def fairness_checks(self, model_version: str) -> list:
        """
        Check model fairness across protected groups.
        
        Checks:
        - Four-fifths rule (adverse impact ratio)
        - Equal opportunity (TPR across groups)
        - Predictive parity (precision across groups)
        - Calibration (predicted probabilities match actual rates per group)
        """
        results = []
        model_info = self._get_model_info(model_version)
        
        # Four-fifths rule: selection rate for any group must be ≥ 80% 
        # of the group with highest selection rate
        protected_attributes = model_info.get("protected_attributes", [])
        
        for attribute in protected_attributes:
            selection_rates = self._compute_selection_rates(model_version, attribute)
            
            if selection_rates:
                max_rate = max(selection_rates.values())
                min_rate = min(selection_rates.values())
                ratio = min_rate / max_rate if max_rate > 0 else 0
                
                results.append(GovernanceCheckResult(
                    check_name=f"four_fifths_rule_{attribute}",
                    result=ComplianceResult.PASSED if ratio >= 0.8 else ComplianceResult.FAILED,
                    details=f"Adverse impact ratio for {attribute}: {ratio:.3f} (threshold: 0.8)",
                    evidence={
                        "attribute": attribute,
                        "selection_rates": selection_rates,
                        "ratio": ratio,
                    },
                    remediation="Apply bias mitigation (re-sampling, adversarial debiasing, threshold adjustment)" if ratio < 0.8 else None,
                ))
        
        return results
    
    def documentation_checks(self, model_version: str) -> list:
        """Verify model documentation is complete."""
        model_info = self._get_model_info(model_version)
        model_card = model_info.get("model_card", {})
        
        required_sections = [
            "model_details",
            "intended_use",
            "training_data",
            "evaluation_metrics",
            "limitations",
            "ethical_considerations",
            "maintenance_plan",
        ]
        
        results = []
        missing_sections = [s for s in required_sections if s not in model_card or not model_card[s]]
        
        results.append(GovernanceCheckResult(
            check_name="model_card_completeness",
            result=ComplianceResult.PASSED if not missing_sections else ComplianceResult.FAILED,
            details=f"Model Card complete: {len(required_sections) - len(missing_sections)}/{len(required_sections)} sections",
            evidence={"missing_sections": missing_sections},
            remediation=f"Complete missing Model Card sections: {', '.join(missing_sections)}" if missing_sections else None,
        ))
        
        return results
    
    def security_checks(self, model_version: str) -> list:
        """Check model security posture."""
        results = []
        model_info = self._get_model_info(model_version)
        
        # Check: model doesn't memorize training data (membership inference risk)
        # Check: input validation configured (adversarial input protection)
        # Check: model doesn't expose sensitive features directly
        
        sensitive_features = model_info.get("features_used", [])
        pii_features = [f for f in sensitive_features if self._is_pii(f)]
        
        results.append(GovernanceCheckResult(
            check_name="pii_feature_check",
            result=ComplianceResult.FAILED if pii_features else ComplianceResult.PASSED,
            details=f"PII features in model: {pii_features}" if pii_features else "No PII features detected",
            evidence={"pii_features": pii_features},
            remediation="Remove PII features or apply anonymization/aggregation" if pii_features else None,
        ))
        
        return results
    
    def operational_checks(self, model_version: str) -> list:
        """Check operational readiness."""
        results = []
        model_info = self._get_model_info(model_version)
        
        checks = {
            "monitoring_configured": model_info.get("monitoring_enabled", False),
            "alerting_configured": model_info.get("alerting_configured", False),
            "rollback_plan": bool(model_info.get("rollback_version")),
            "sla_defined": bool(model_info.get("latency_sla_ms")),
            "owner_assigned": bool(model_info.get("owner")),
        }
        
        for check_name, passed in checks.items():
            results.append(GovernanceCheckResult(
                check_name=check_name,
                result=ComplianceResult.PASSED if passed else ComplianceResult.FAILED,
                details=f"{check_name}: {'configured' if passed else 'NOT configured'}",
                evidence={check_name: passed},
                remediation=f"Configure {check_name} before production deployment" if not passed else None,
            ))
        
        return results
```

### Risk Classification

```yaml
Risk_Classification:
  eu_ai_act_levels:
    unacceptable:
      examples: "Social scoring, real-time biometric surveillance"
      action: "Prohibited — cannot deploy"
      
    high_risk:
      examples: "Credit scoring, hiring, medical diagnosis, law enforcement"
      requirements:
        - "Conformity assessment (third-party audit)"
        - "Quality management system"
        - "Technical documentation (detailed)"
        - "Data governance (training data quality)"
        - "Human oversight provisions"
        - "Accuracy, robustness, cybersecurity measures"
        - "Registration in EU database"
      platform_support: "Automated checks + human review + audit trail"
      
    limited_risk:
      examples: "Chatbots, emotion detection, deepfakes"
      requirements:
        - "Transparency: Users must be informed they're interacting with AI"
        - "Disclosure of AI-generated content"
      platform_support: "Automated transparency checks"
      
    minimal_risk:
      examples: "Spam filters, recommendation systems, game AI"
      requirements: "No specific requirements (voluntary codes of practice)"
      platform_support: "Standard governance (Model Card, basic fairness checks)"
      
  organizational_risk_levels:
    low:
      criteria: "Internal tools, non-consequential decisions"
      review: "Team lead approval only"
      checks: "Performance + documentation"
      
    medium:
      criteria: "Customer-facing, moderate impact decisions"
      review: "Team lead + platform team"
      checks: "Performance + fairness + documentation + security"
      
    high:
      criteria: "Consequential decisions (financial, health, safety)"
      review: "Team lead + platform + compliance + legal"
      checks: "All checks + explainability + regulatory + audit schedule"
```

### Access Control

```yaml
Access_Control:
  rbac:
    what: "RBAC (Role-Based Access Control) for ML platform"
    roles:
      ml_engineer:
        can: "Train models, run experiments, deploy to development"
        cannot: "Deploy to production, approve models, change governance rules"
        
      senior_ml_engineer:
        can: "All of ml_engineer + deploy to staging + review models"
        cannot: "Deploy to production without approval, change governance rules"
        
      ml_lead:
        can: "All of senior + approve models for staging + request production deploy"
        cannot: "Final production approval (needs platform team), change governance rules"
        
      platform_admin:
        can: "Approve production deployments, configure governance rules, manage access"
        cannot: "Train models (shouldn't — separation of duties)"
        
      compliance_officer:
        can: "Review model documentation, approve high-risk models, audit access"
        cannot: "Train or deploy models directly"
        
  data_access:
    principle: "Least privilege — access only data needed for current task"
    implementation:
      - "Feature store access scoped to team's features"
      - "Training data access audited and logged"
      - "PII data requires additional approval"
      - "Production data isolated from development"
```

---

## How It Works in Practice

### Model Lifecycle with Governance

```yaml
Lifecycle:
  development:
    governance: "Minimal — don't slow down experimentation"
    auto_checks: "None (or just code quality)"
    
  staging_promotion:
    trigger: "Engineer requests staging promotion"
    auto_checks:
      - "Performance meets minimum threshold ✓"
      - "Model Card started (at least model_details + intended_use) ✓"
      - "No known security issues ✓"
    human_review: "Team lead approves (reviews metrics, documentation)"
    
  production_promotion:
    trigger: "Team lead requests production promotion"
    auto_checks:
      - "Performance meets SLA ✓"
      - "Fairness checks pass (four-fifths rule) ✓"
      - "Model Card complete ✓"
      - "Monitoring configured ✓"
      - "Rollback plan documented ✓"
      - "Latency within SLA ✓"
      - "Security check passed ✓"
    human_review: "Platform team + (for high-risk) compliance officer"
    
  post_production:
    ongoing: "Quarterly model review (performance, fairness, documentation freshness)"
    triggered: "Re-review if model retrained, data source changes, or incident occurs"
```

---

## Interview Tip

> When asked about ML governance: "My governance platform has four layers: (1) Model registry — every model version cataloged with: training data lineage, performance metrics, owner, and stage (development/staging/production). Nothing reaches production without being in the registry. (2) Approval workflows — staged promotion with automated and human gates. Development → staging requires: basic documentation + team lead approval. Staging → production requires: complete Model Card, fairness testing (four-fifths rule), latency validation, monitoring configured, and platform team approval. High-risk models (credit, hiring, medical) additionally require compliance review. (3) Automated compliance checks — run automatically before each promotion: performance thresholds, fairness across protected groups, PII feature detection, documentation completeness, and operational readiness (monitoring, alerting, rollback configured). If any check fails, promotion is blocked with specific remediation steps. (4) Audit trail — immutable log of every ML lifecycle event: training, evaluation, approval, deployment, rollback, retraining. Retained 7+ years for regulated industries. Enables answering 'how was this decision made?' at any point in time. For EU AI Act compliance: I classify models by risk level, apply proportional governance (minimal for spam filters, comprehensive for credit decisions), and maintain documentation that satisfies conformity assessment requirements."

---

## Common Mistakes

1. **Governance only for production** — No tracking during development. When it's time to deploy, nobody can reconstruct how the model was built (what data? what preprocessing?). Solution: lightweight tracking from day one (experiment platform captures everything automatically). Governance gates increase rigor as you approach production.

2. **All models treated equally** — Same heavyweight process for a spam filter and a credit scoring model. Data scientists frustrated by bureaucracy for low-risk models. Solution: risk-based governance. Low-risk = minimal process. High-risk = comprehensive review. Classify models by impact, apply proportional controls.

3. **Manual compliance checks** — A human manually reviews fairness metrics, reads documentation, checks monitoring setup. Inconsistent (different reviewers have different standards), slow (2-week review queue), and doesn't scale. Solution: automate everything that can be automated. Human review only for judgment calls (is the intended use appropriate? are the limitations adequately documented?).

4. **Governance as blocker, not enabler** — Governance is perceived as "the team that says no." Engineers avoid it, find workarounds, or deploy without approval. Solution: make governance a quality accelerator — fast automated checks (minutes, not weeks), clear requirements (no ambiguity), and helpful feedback (not just "failed" but "here's how to fix it").

5. **No post-deployment governance** — Model approved and deployed, then forgotten. Performance degrades over 6 months, fairness drifts, documentation becomes stale. Solution: scheduled quarterly reviews (is the model still meeting its documented performance? has the data changed?). Trigger re-review on significant changes.

---

## Key Takeaways

- Governance = model registry + approval workflows + automated compliance + audit trails
- Risk-based: proportional governance (minimal for low-risk, comprehensive for high-risk)
- Automated checks: fairness (four-fifths rule), documentation completeness, security, operational readiness
- Approval workflows: staged promotion (dev → staging → production) with increasing rigor
- Model Cards: standardized documentation required for all production models
- Audit trail: immutable, retained 7+ years, answers "how was this decision made?"
- Access control: RBAC with separation of duties (who trains ≠ who approves ≠ who deploys)
- EU AI Act: classify by risk level, apply proportional requirements, maintain conformity documentation
- Speed: automated checks run in minutes (not weeks) — governance shouldn't be a bottleneck
- Post-production: quarterly reviews, triggered re-review on significant changes
