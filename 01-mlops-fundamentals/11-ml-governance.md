# ML Governance

## The Problem / Why This Matters

ML models make decisions that affect people's lives: who gets a loan, which candidates get interviews, what content users see, whether a medical image suggests cancer. With this power comes responsibility — and regulatory scrutiny. ML governance is the framework of policies, processes, and tools that ensures models are developed, deployed, and maintained responsibly: documented (model cards), auditable (full lineage), fair (bias testing), explainable (interpretable decisions), and compliant (regulatory requirements). Without governance, organizations face: regulatory fines (EU AI Act violations can be 7% of global revenue), reputational damage (biased AI in the news), legal liability (discriminatory decisions), and operational risk (nobody knows what models are running or why they make certain predictions). In 2026, with the EU AI Act (Artificial Intelligence Act) fully in force, US state-level AI regulations proliferating, and industry-specific requirements (banking, healthcare, insurance), ML governance has shifted from "nice to have" to "legal requirement."

---

## The Analogy

Think of ML governance like pharmaceutical regulation:

- **Clinical trials** = Model evaluation (thorough testing before release)
- **FDA approval** = Model approval workflow (expert review, sign-off)
- **Package insert** = Model card (capabilities, side effects, contraindications)
- **Adverse event reporting** = Incident management (model failures tracked and reported)
- **Manufacturing standards** = MLOps practices (reproducible, validated pipelines)
- **Post-market surveillance** = Production monitoring (ongoing safety tracking)
- **Recall procedure** = Rollback and decommissioning (remove harmful model quickly)

Without pharmaceutical regulation, dangerous drugs reach patients. Without ML governance, harmful models reach users.

---

## Deep Dive

### Governance Framework

```yaml
Governance_Framework:
  pillars:
    documentation:
      what: "Complete documentation of every model's purpose, behavior, and limitations"
      artifacts:
        model_card: "Structured documentation (Google's Model Cards framework)"
        data_card: "Documentation of training data (sources, biases, limitations)"
        system_card: "Documentation of the full AI system (multiple models + rules)"
      requirement: "Every production model must have an up-to-date model card"
      
    accountability:
      what: "Clear ownership and responsibility for every model"
      elements:
        model_owner: "Person/team responsible for model quality and maintenance"
        approval_chain: "Who must approve before production deployment"
        incident_response: "Who to contact when model fails or causes harm"
        decommissioning: "Process for retiring models safely"
      requirement: "Every production model has a named owner and escalation path"
      
    auditability:
      what: "Complete lineage and decision trail for every model"
      elements:
        data_lineage: "What data was used, where it came from, how it was processed"
        model_lineage: "How the model was trained, evaluated, and deployed"
        decision_lineage: "Why specific model decisions were made (explainability)"
        change_history: "All modifications to model, data, and config over time"
      requirement: "Any model decision can be traced back to its inputs and reasoning"
      
    fairness:
      what: "Models don't discriminate against protected groups"
      elements:
        bias_testing: "Regular evaluation across demographics"
        fairness_metrics: "Quantitative measures of equitable treatment"
        mitigation: "Procedures for addressing identified bias"
        protected_attributes: "Definition of which attributes require fairness"
      requirement: "Every model undergoes bias testing before and during production"
      
    safety:
      what: "Models don't cause harm to users or society"
      elements:
        risk_assessment: "Pre-deployment risk analysis (severity × probability)"
        safety_testing: "Red teaming, adversarial testing, edge case evaluation"
        guardrails: "Technical safeguards against harmful outputs"
        human_oversight: "When and how humans override model decisions"
      requirement: "Risk-appropriate safety measures proportional to impact"
```

### Model Cards

```yaml
Model_Cards:
  description: "Structured documentation framework for ML model transparency"
  origin: "Google Research (2018), now industry standard, required by EU AI Act"
  
  sections:
    model_details:
      contents:
        - "Model name, version, date"
        - "Developed by (team, organization)"
        - "Model architecture and type"
        - "Training framework and parameters"
        - "License and usage restrictions"
      example: "Fraud Detector v3.2, XGBoost, developed by Risk ML Team, trained 2026-03"
      
    intended_use:
      contents:
        - "Primary intended use cases"
        - "Primary intended users"
        - "Out-of-scope use cases (explicitly stated)"
      example: "Intended for real-time transaction fraud scoring. NOT intended for account-level risk assessment or credit decisions."
      
    training_data:
      contents:
        - "Dataset description (source, size, date range)"
        - "Data collection methodology"
        - "Known biases or limitations in data"
        - "Preprocessing steps applied"
        - "Label methodology (how ground truth was determined)"
      example: "12M transactions from 2024-2026, labeled by confirmed fraud (chargeback) with 30-day confirmation window."
      
    evaluation:
      contents:
        - "Evaluation datasets and methodology"
        - "Aggregate metrics with confidence intervals"
        - "Per-subgroup metrics (fairness evaluation)"
        - "Comparison to previous version and baseline"
      example: "AUC: 0.94 ± 0.01. Per-age-group: 18-25: 0.91, 26-45: 0.95, 46+: 0.93."
      
    ethical_considerations:
      contents:
        - "Potential harms and mitigations"
        - "Bias analysis results"
        - "Environmental impact (training compute)"
        - "Privacy considerations"
      example: "Model may over-flag international transactions (2% higher FPR for non-domestic). Mitigation: separate threshold for international."
      
    limitations:
      contents:
        - "Known limitations and failure modes"
        - "Scenarios where model should NOT be trusted"
        - "Degradation conditions"
        - "Recommended human oversight level"
      example: "Performance degrades for new merchants (<30 days old) due to limited behavioral history. Human review required for new merchant transactions >$5000."
      
  automation:
    principle: "Auto-generate from experiment tracking data, human-fill subjective sections"
    auto_generated: "Metrics, data stats, architecture, training config, version history"
    human_required: "Intended use, ethical considerations, limitations, out-of-scope uses"
    tools: "Hugging Face Model Cards, Google Model Card Toolkit, custom templates"
```

### Regulatory Landscape (2026)

```yaml
Regulatory_Landscape:
  eu_ai_act:
    status: "Fully enforceable 2026"
    risk_categories:
      unacceptable_risk:
        examples: "Social scoring, real-time biometric identification in public"
        requirement: "BANNED"
      high_risk:
        examples: "Credit scoring, hiring, medical devices, education, law enforcement"
        requirements:
          - "Risk management system"
          - "Data governance (quality, representativeness)"
          - "Technical documentation (model cards)"
          - "Record-keeping (logging, auditability)"
          - "Transparency (inform affected persons)"
          - "Human oversight (meaningful human control)"
          - "Accuracy and robustness"
          - "Conformity assessment before market"
        penalties: "Up to 7% of global annual revenue"
      limited_risk:
        examples: "Chatbots, deepfakes, emotion recognition"
        requirements: "Transparency obligations (disclose AI use)"
      minimal_risk:
        examples: "Spam filters, recommendation systems (most consumer AI)"
        requirements: "Voluntary codes of practice"
        
  us_regulations:
    federal: "Executive Order on AI Safety (2023), NIST AI RMF (Risk Management Framework)"
    state_level: "Colorado, Illinois, California with AI-specific legislation"
    sector_specific:
      banking: "SR 11-7 (OCC/Fed model risk management), fair lending laws"
      healthcare: "FDA regulation of AI/ML medical devices"
      insurance: "State insurance regulators requiring explainability"
      
  industry_standards:
    nist_ai_rmf:
      full_name: "NIST Artificial Intelligence Risk Management Framework"
      functions: "Govern, Map, Measure, Manage"
      adoption: "Widely adopted as best practice framework in US"
    iso_42001: "AI Management System standard (certification available)"
    ieee_standards: "Various AI ethics and transparency standards"
```

### Governance Tooling

```yaml
Governance_Tools:
  model_registry_governance:
    tool: "MLflow Model Registry with approval workflows"
    capabilities:
      - "Stage gates (model can't reach production without approval)"
      - "Approval tracking (who approved, when, with what evidence)"
      - "Lineage (link to training data, code, experiments)"
    enhancement: "Custom webhooks for compliance checks before stage transition"
    
  fairness_testing:
    tools:
      fairlearn: "Microsoft's open-source fairness toolkit (Python)"
      aif360: "IBM's AI Fairness 360 toolkit"
      what_if_tool: "Google's interactive fairness exploration (TensorBoard)"
    integration: "Run fairness tests as validation gate in CI/CD pipeline"
    
  explainability:
    tools:
      shap: "SHapley Additive exPlanations — feature contribution to predictions"
      lime: "Local Interpretable Model-agnostic Explanations"
      eli5: "Explain Like I'm 5 — simple explanations for ML models"
    requirement: "For high-risk models, provide explanation for any individual prediction"
    
  audit_platforms:
    tools:
      - "Credo AI (AI governance platform)"
      - "Holistic AI (risk management and compliance)"
      - "Arthur AI (monitoring with governance focus)"
    capability: "Centralized governance dashboard, compliance reporting, risk tracking"
```

---

## How It Works in Practice

### Governance Workflow Example

```yaml
Example:
  model: "Loan approval recommendation model (HIGH RISK — EU AI Act)"
  organization: "Digital bank, EU operations"
  
  pre_development:
    risk_assessment:
      category: "High-risk (credit decisions affecting financial access)"
      requirements: "Full EU AI Act compliance required"
      approval_to_proceed: "Risk Committee + Legal Team sign-off"
    data_governance:
      data_sources: "Documented and approved"
      protected_attributes: "Identified (race, gender, age, disability)"
      data_quality: "Validation plan defined"
      
  development:
    model_card_draft: "Created at project start, updated throughout"
    fairness_testing:
      during_development: "Every training run evaluated for bias"
      metrics: "Demographic parity, equalized odds across protected groups"
      threshold: "Max 15% disparity between groups"
    explainability: "SHAP values computed for all predictions"
    
  pre_deployment_review:
    technical_review:
      reviewer: "ML Lead"
      checks: "Performance metrics, code review, pipeline quality"
    fairness_review:
      reviewer: "Responsible AI Team"
      checks: "Bias testing results, mitigation effectiveness"
    legal_review:
      reviewer: "Legal/Compliance Team"
      checks: "EU AI Act compliance, documentation completeness"
    business_review:
      reviewer: "Product Owner"
      checks: "Business impact assessment, user communication plan"
    final_approval: "All four reviewers must sign off"
    
  production:
    monitoring:
      - "Continuous fairness monitoring (weekly bias reports)"
      - "Performance monitoring (daily evaluation)"
      - "Drift detection (hourly)"
    audit_trail:
      - "All predictions logged with SHAP explanations"
      - "All model changes tracked with justification"
      - "Regular compliance reports generated"
    incident_management:
      - "Bias incident detected → immediate investigation"
      - "If confirmed bias → model modified or paused within 24 hours"
      - "Regulatory notification if required"
    periodic_review:
      - "Quarterly model review (performance, fairness, relevance)"
      - "Annual full audit (external auditor for EU AI Act)"
```

---

## Interview Tip

> When asked about ML governance: "I implement governance across four dimensions: (1) Documentation — model cards for every production model (intended use, training data, limitations, fairness analysis). (2) Accountability — clear model ownership, approval workflows before production deployment, incident response procedures. (3) Auditability — full lineage from data through training to deployment, every prediction explainable (SHAP), all changes tracked. (4) Fairness — bias testing as a CI/CD gate (Fairlearn), continuous fairness monitoring in production, defined thresholds and remediation procedures. For regulated industries (banking, healthcare), I ensure EU AI Act compliance: risk categorization, conformity assessment for high-risk systems, human oversight mechanisms, and transparency obligations. The tooling: MLflow Model Registry for lifecycle governance, Fairlearn/AIF360 for bias testing, SHAP for explainability, and custom dashboards for compliance reporting."

---

## Common Mistakes

1. **Governance as afterthought** — Building and deploying the model first, then trying to "add governance." By then, lineage is lost, documentation doesn't exist, and bias hasn't been tested. Start governance from project inception.

2. **Model cards as checkbox** — Creating minimal model cards to satisfy a requirement without useful content. A model card that says "this model predicts things" is useless. It must contain specific limitations, intended uses, and known failure modes.

3. **Fairness testing only at deployment** — Testing for bias once, declaring "it's fair," and never checking again. Fairness can degrade over time as data distributions shift. Monitor fairness continuously in production.

4. **No explainability for individual predictions** — Having aggregate feature importance but unable to explain WHY a specific user was denied a loan. Regulators and affected users want individual explanations, not just model-level statistics.

5. **Ignoring regulatory requirements until audit** — Not tracking EU AI Act requirements until an audit notice arrives. By then, compliance gaps are extensive and expensive to remediate. Build compliance into the development process from day one.

---

## Key Takeaways

- ML governance: documentation (model cards), accountability (ownership), auditability (lineage), fairness (bias testing), safety (risk assessment)
- Model cards are mandatory for production models — document capabilities, limitations, fairness, and intended use
- EU AI Act (2026): high-risk AI systems require risk management, documentation, human oversight, fairness, and conformity assessment
- Approval workflows: technical + fairness + legal + business review before production deployment
- Fairness as CI/CD gate: every model version tested for bias before deployment
- Explainability: SHAP/LIME for individual prediction explanations (regulatory requirement for high-risk)
- Audit trail: full lineage from data → model → predictions, all changes tracked with justification
- Continuous compliance: periodic reviews, fairness monitoring, updated documentation
- Start governance at project inception — not after deployment
- Penalties for non-compliance can be 7% of global revenue (EU AI Act) — governance is a business imperative
