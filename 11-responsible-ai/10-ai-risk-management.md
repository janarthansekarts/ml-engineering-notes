# AI Risk Management

## The Problem / Why This Matters

AI systems fail in ways traditional software doesn't — they degrade silently, amplify biases, hallucinate confidently, and exhibit emergent behaviors that weren't anticipated during development. AI risk management is the systematic process of identifying, assessing, mitigating, and monitoring risks specific to AI/ML systems throughout their lifecycle. In 2026, the NIST AI RMF (National Institute of Standards and Technology AI Risk Management Framework) has become the de facto standard for AI risk management in the US, while ISO/IEC 42001 provides the international standard for AI management systems. The EU AI Act requires mandatory risk management systems for high-risk AI. Unlike traditional software risk management (focused on known failure modes), AI risk management must account for: distributional shift (model performance degrades as world changes), emergent capabilities (models exhibit unexpected behaviors), adversarial threats (deliberate attacks on model integrity), and sociotechnical risks (interaction between model and social context). For ML engineers, risk management means building systems that continuously assess and respond to risks — not just checking a box before deployment.

---

## The Analogy

Think of AI risk management like aviation safety:

- **Pre-flight checklist** = Model validation before deployment. Check known risks, verify performance, ensure monitoring is configured. Catches known problems.
- **In-flight monitoring** = Production monitoring. Continuous telemetry on model health, drift detection, fairness metrics, anomaly detection. Catches problems as they emerge.
- **Incident investigation** = Post-mortem analysis when a model fails. Root cause analysis, prevention measures, fleet-wide inspection. Learn from failures.
- **Safety culture** = Organization-wide commitment to responsible AI. Everyone (not just safety team) is empowered to flag and escalate concerns. Incidents reported without blame.
- **The difference** = In aviation, a failure is immediately obvious (plane crashes). In AI, failures are often silent — a biased model operates for months before anyone notices the harm.

---

## Deep Dive

### NIST AI RMF (Risk Management Framework)

```yaml
NIST_AI_RMF:
  overview:
    purpose: "Voluntary framework for managing AI risks throughout the AI lifecycle"
    scope: "All AI systems (not just high-risk)"
    structure: "Four core functions: Govern, Map, Measure, Manage"
    
  govern:
    purpose: "Establish and maintain organizational AI risk governance"
    activities:
      - "Define organizational AI risk tolerance"
      - "Establish roles and responsibilities for AI risk"
      - "Create policies for AI development and deployment"
      - "Ensure accountability mechanisms exist"
      - "Foster a culture of responsible AI"
      - "Allocate resources for risk management"
      
    outputs:
      - "AI governance policy document"
      - "RACI matrix for AI risk"
      - "Risk tolerance thresholds per use case"
      - "Escalation procedures"
      
  map:
    purpose: "Identify and contextualize AI risks for specific systems"
    activities:
      - "Identify intended use and context"
      - "Map stakeholders (who benefits, who's affected, who decides)"
      - "Identify potential harms (individual, group, societal)"
      - "Assess likelihood and severity of harms"
      - "Consider interdependencies and cascading failures"
      - "Document assumptions and limitations"
      
    risk_categories:
      technical: "Accuracy, robustness, reliability, security"
      fairness: "Bias, discrimination, equity"
      privacy: "Data protection, consent, surveillance"
      safety: "Physical safety, harmful content"
      transparency: "Explainability, accountability"
      societal: "Environmental, economic, democratic"
      
  measure:
    purpose: "Quantify and track identified risks"
    activities:
      - "Define metrics for each identified risk"
      - "Establish measurement methodologies"
      - "Conduct quantitative and qualitative assessments"
      - "Track metrics over time (trends and drift)"
      - "Benchmark against standards and baselines"
      
    metric_types:
      performance: "Accuracy, precision, recall, F1 — disaggregated by group"
      fairness: "Demographic parity, equal opportunity, calibration per group"
      robustness: "Performance under distribution shift, adversarial inputs"
      privacy: "Memorization rate, membership inference attack accuracy"
      safety: "Harm rate, refusal rate, attack success rate"
      
  manage:
    purpose: "Prioritize and address identified risks"
    activities:
      - "Prioritize risks (severity × likelihood)"
      - "Select mitigation strategies"
      - "Implement mitigations"
      - "Monitor residual risk"
      - "Accept, transfer, or avoid risks as appropriate"
      - "Respond to incidents"
      - "Document decisions and rationale"
      
    response_options:
      mitigate: "Reduce risk through technical/process controls"
      accept: "Risk is below tolerance, monitor without action"
      transfer: "Insurance, contractual transfer, partner responsibility"
      avoid: "Don't deploy system for this use case"
```

### Risk Assessment Methodology

```yaml
Risk_Assessment:
  framework:
    step_1_identify:
      what: "List all potential risks for this specific AI system"
      inputs: "Use case description, model architecture, data sources, deployment context"
      techniques:
        - "Threat modeling (STRIDE adapted for AI)"
        - "Failure mode analysis (FMEA adapted for ML)"
        - "Stakeholder impact assessment"
        - "Historical incident analysis (learn from others' failures)"
        
    step_2_assess:
      what: "Rate each risk by likelihood and impact"
      dimensions:
        likelihood:
          rare: "< 1% probability in deployment period"
          unlikely: "1-10% probability"
          possible: "10-50% probability"
          likely: "50-90% probability"
          almost_certain: "> 90% probability"
          
        impact:
          negligible: "No measurable harm, minor inconvenience"
          minor: "Limited harm to individuals, easily reversed"
          moderate: "Significant harm to individuals or groups"
          major: "Severe harm, difficult to reverse, legal liability"
          catastrophic: "Widespread harm, irreversible, existential to organization"
          
    step_3_prioritize:
      what: "Risk score = likelihood × impact"
      risk_matrix: |
        High risk: likely/possible + major/catastrophic → MUST mitigate before deployment
        Medium risk: possible + moderate → Should mitigate, can accept with monitoring
        Low risk: unlikely/rare + minor/negligible → Accept with monitoring
        
    step_4_mitigate:
      what: "Select and implement appropriate controls"
      control_types:
        preventive: "Stop the risk from materializing (input validation, fairness constraints)"
        detective: "Detect when risk materializes (monitoring, anomaly detection)"
        corrective: "Respond after risk materializes (rollback, human override)"
        
    step_5_monitor:
      what: "Continuously track residual risk"
      activities:
        - "Real-time risk metric dashboards"
        - "Automated alerting on threshold breaches"
        - "Periodic risk re-assessment (quarterly)"
        - "Incident tracking and trend analysis"
```

### Implementation

```python
# AI Risk Management implementation

"""
AI Risk Management framework: identify, assess, mitigate, and monitor
risks for ML systems in production.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class Likelihood(Enum):
    RARE = 1
    UNLIKELY = 2
    POSSIBLE = 3
    LIKELY = 4
    ALMOST_CERTAIN = 5


class Impact(Enum):
    NEGLIGIBLE = 1
    MINOR = 2
    MODERATE = 3
    MAJOR = 4
    CATASTROPHIC = 5


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ControlType(Enum):
    PREVENTIVE = "preventive"
    DETECTIVE = "detective"
    CORRECTIVE = "corrective"


@dataclass
class Risk:
    """An identified AI risk."""
    risk_id: str
    title: str
    description: str
    category: str  # technical, fairness, privacy, safety, societal
    likelihood: Likelihood
    impact: Impact
    affected_stakeholders: List[str]
    
    # Calculated
    risk_score: int = 0
    risk_level: RiskLevel = RiskLevel.LOW
    
    # Mitigation
    mitigations: List[Dict] = field(default_factory=list)
    residual_likelihood: Optional[Likelihood] = None
    residual_impact: Optional[Impact] = None
    
    # Status
    status: str = "identified"  # identified, assessed, mitigating, monitored, closed
    owner: str = ""
    
    def __post_init__(self):
        self.risk_score = self.likelihood.value * self.impact.value
        self.risk_level = self._compute_level()
    
    def _compute_level(self) -> RiskLevel:
        if self.risk_score >= 16:
            return RiskLevel.CRITICAL
        elif self.risk_score >= 9:
            return RiskLevel.HIGH
        elif self.risk_score >= 4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


@dataclass
class RiskMitigation:
    """A risk mitigation control."""
    control_id: str
    risk_id: str
    description: str
    control_type: ControlType
    implementation_status: str  # planned, in_progress, implemented, verified
    effectiveness: float = 0.0  # 0-1, how much it reduces risk
    owner: str = ""
    deadline: Optional[str] = None


class AIRiskAssessment:
    """
    Conduct AI risk assessment for a model/system.
    
    Follows NIST AI RMF methodology.
    """
    
    def __init__(self, system_name: str, system_description: str):
        self.system_name = system_name
        self.system_description = system_description
        self.risks: List[Risk] = []
        self.mitigations: List[RiskMitigation] = []
        self.assessment_date = datetime.now().isoformat()
    
    def identify_risks(self, use_case: Dict) -> List[Risk]:
        """
        Systematically identify risks based on use case context.
        
        Uses pre-defined risk taxonomy adapted from NIST AI RMF.
        """
        identified = []
        
        # Technical risks
        identified.extend(self._assess_technical_risks(use_case))
        
        # Fairness risks
        identified.extend(self._assess_fairness_risks(use_case))
        
        # Privacy risks
        identified.extend(self._assess_privacy_risks(use_case))
        
        # Safety risks
        identified.extend(self._assess_safety_risks(use_case))
        
        # Societal risks
        identified.extend(self._assess_societal_risks(use_case))
        
        self.risks = identified
        logger.info(f"Identified {len(identified)} risks for {self.system_name}")
        return identified
    
    def _assess_technical_risks(self, use_case: Dict) -> List[Risk]:
        """Identify technical risks."""
        risks = []
        
        # Distribution shift
        if use_case.get("deployed_to_production"):
            risks.append(Risk(
                risk_id="TECH-001",
                title="Distribution Shift",
                description=(
                    "Model performance degrades as production data distribution "
                    "diverges from training data over time."
                ),
                category="technical",
                likelihood=Likelihood.LIKELY,
                impact=Impact(min(use_case.get("impact_if_wrong", 3), 5)),
                affected_stakeholders=use_case.get("affected_users", ["end_users"]),
            ))
        
        # Adversarial attacks
        if use_case.get("public_facing"):
            risks.append(Risk(
                risk_id="TECH-002",
                title="Adversarial Attacks",
                description=(
                    "Malicious users craft inputs to cause model misbehavior, "
                    "bypass safety controls, or extract sensitive information."
                ),
                category="technical",
                likelihood=Likelihood.POSSIBLE,
                impact=Impact.MAJOR,
                affected_stakeholders=["end_users", "organization"],
            ))
        
        # Hallucination / confabulation
        if use_case.get("model_type") in ["llm", "generative"]:
            risks.append(Risk(
                risk_id="TECH-003",
                title="Hallucination",
                description=(
                    "Model generates confident but incorrect information. "
                    "Users may act on false information."
                ),
                category="technical",
                likelihood=Likelihood.ALMOST_CERTAIN,
                impact=Impact(use_case.get("impact_if_wrong", 3)),
                affected_stakeholders=["end_users"],
            ))
        
        return risks
    
    def _assess_fairness_risks(self, use_case: Dict) -> List[Risk]:
        """Identify fairness risks."""
        risks = []
        
        if use_case.get("affects_individuals"):
            risks.append(Risk(
                risk_id="FAIR-001",
                title="Demographic Bias",
                description=(
                    "Model performs differently across demographic groups, "
                    "potentially discriminating against protected classes."
                ),
                category="fairness",
                likelihood=Likelihood.LIKELY,
                impact=Impact.MAJOR if use_case.get("high_stakes") else Impact.MODERATE,
                affected_stakeholders=["underrepresented_groups"],
            ))
        
        return risks
    
    def _assess_privacy_risks(self, use_case: Dict) -> List[Risk]:
        """Identify privacy risks."""
        risks = []
        
        if use_case.get("trained_on_personal_data"):
            risks.append(Risk(
                risk_id="PRIV-001",
                title="Training Data Memorization",
                description=(
                    "Model memorizes and can regurgitate personal information "
                    "from training data (PII leakage)."
                ),
                category="privacy",
                likelihood=Likelihood.POSSIBLE,
                impact=Impact.MAJOR,
                affected_stakeholders=["data_subjects"],
            ))
        
        return risks
    
    def _assess_safety_risks(self, use_case: Dict) -> List[Risk]:
        """Identify safety risks."""
        risks = []
        
        if use_case.get("model_type") in ["llm", "generative"]:
            risks.append(Risk(
                risk_id="SAFE-001",
                title="Harmful Content Generation",
                description=(
                    "Model generates content that could cause harm: "
                    "misinformation, toxic content, dangerous instructions."
                ),
                category="safety",
                likelihood=Likelihood.POSSIBLE,
                impact=Impact.MAJOR,
                affected_stakeholders=["end_users", "society"],
            ))
        
        return risks
    
    def _assess_societal_risks(self, use_case: Dict) -> List[Risk]:
        """Identify societal risks."""
        risks = []
        
        if use_case.get("scale") == "large":
            risks.append(Risk(
                risk_id="SOC-001",
                title="Feedback Loop Amplification",
                description=(
                    "At scale, model decisions influence future data, "
                    "creating self-reinforcing patterns (filter bubbles, bias amplification)."
                ),
                category="societal",
                likelihood=Likelihood.LIKELY,
                impact=Impact.MODERATE,
                affected_stakeholders=["society", "end_users"],
            ))
        
        return risks
    
    def add_mitigation(self, mitigation: RiskMitigation):
        """Add a mitigation control for an identified risk."""
        self.mitigations.append(mitigation)
        
        # Update risk with mitigation info
        for risk in self.risks:
            if risk.risk_id == mitigation.risk_id:
                risk.mitigations.append({
                    "control_id": mitigation.control_id,
                    "description": mitigation.description,
                    "type": mitigation.control_type.value,
                })
                risk.status = "mitigating"
    
    def generate_report(self) -> Dict:
        """Generate comprehensive risk assessment report."""
        risk_summary = {
            "critical": sum(1 for r in self.risks if r.risk_level == RiskLevel.CRITICAL),
            "high": sum(1 for r in self.risks if r.risk_level == RiskLevel.HIGH),
            "medium": sum(1 for r in self.risks if r.risk_level == RiskLevel.MEDIUM),
            "low": sum(1 for r in self.risks if r.risk_level == RiskLevel.LOW),
        }
        
        return {
            "system_name": self.system_name,
            "assessment_date": self.assessment_date,
            "total_risks": len(self.risks),
            "risk_summary": risk_summary,
            "risks": [
                {
                    "id": r.risk_id,
                    "title": r.title,
                    "level": r.risk_level.value,
                    "score": r.risk_score,
                    "category": r.category,
                    "mitigations": len(r.mitigations),
                    "status": r.status,
                }
                for r in sorted(self.risks, key=lambda r: r.risk_score, reverse=True)
            ],
            "recommendation": self._overall_recommendation(risk_summary),
        }
    
    def _overall_recommendation(self, summary: Dict) -> str:
        """Generate overall deployment recommendation."""
        if summary["critical"] > 0:
            return "DO NOT DEPLOY — critical risks require mitigation first"
        elif summary["high"] > 2:
            return "CONDITIONAL — deploy only after high-risk mitigations are verified"
        elif summary["high"] > 0:
            return "CONDITIONAL — deploy with enhanced monitoring and mitigation plan"
        else:
            return "APPROVE — acceptable risk level with standard monitoring"


class ContinuousRiskMonitor:
    """
    Continuous risk monitoring in production.
    
    Tracks risk indicators and alerts when thresholds are breached.
    """
    
    def __init__(self, risk_assessment: AIRiskAssessment):
        self.assessment = risk_assessment
        self.risk_indicators: Dict[str, List[Dict]] = {}
        self.alerts: List[Dict] = []
        self.thresholds: Dict[str, float] = {}
    
    def set_threshold(self, indicator: str, threshold: float, direction: str = "above"):
        """Set alerting threshold for a risk indicator."""
        self.thresholds[indicator] = {"value": threshold, "direction": direction}
    
    def record_indicator(self, indicator: str, value: float, metadata: Dict = None):
        """Record a risk indicator measurement."""
        if indicator not in self.risk_indicators:
            self.risk_indicators[indicator] = []
        
        measurement = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.risk_indicators[indicator].append(measurement)
        
        # Check threshold
        threshold_config = self.thresholds.get(indicator)
        if threshold_config:
            breached = (
                (threshold_config["direction"] == "above" and value > threshold_config["value"]) or
                (threshold_config["direction"] == "below" and value < threshold_config["value"])
            )
            if breached:
                alert = {
                    "indicator": indicator,
                    "value": value,
                    "threshold": threshold_config["value"],
                    "timestamp": datetime.now().isoformat(),
                    "severity": "critical" if value > threshold_config["value"] * 1.5 else "warning",
                }
                self.alerts.append(alert)
                logger.warning(f"Risk threshold breached: {indicator} = {value} (threshold: {threshold_config['value']})")
    
    def get_risk_status(self) -> Dict:
        """Get current risk status across all indicators."""
        status = {}
        for indicator, measurements in self.risk_indicators.items():
            if measurements:
                recent = measurements[-1]["value"]
                trend = self._compute_trend(measurements[-7:]) if len(measurements) >= 7 else 0
                status[indicator] = {
                    "current": recent,
                    "trend": trend,
                    "threshold": self.thresholds.get(indicator, {}).get("value"),
                    "status": "OK" if not any(
                        a["indicator"] == indicator for a in self.alerts[-10:]
                    ) else "ALERT",
                }
        return status
    
    def _compute_trend(self, measurements: List[Dict]) -> float:
        """Compute trend direction (positive = increasing risk)."""
        if len(measurements) < 2:
            return 0.0
        values = [m["value"] for m in measurements]
        # Simple linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        return numerator / denominator if denominator > 0 else 0.0
```

---

## How It Works in Practice

### Risk Assessment for a Healthcare AI System

```yaml
Healthcare_Risk_Assessment:
  system: "Clinical decision support — predicts patient deterioration risk"
  deployment: "Hospital ICU, 500-bed facility"
  
  identified_risks:
    - risk_id: "TECH-001"
      title: "Distribution Shift"
      description: "Patient population changes seasonally (flu season, pandemics)"
      likelihood: "Likely"
      impact: "Major (missed deterioration → patient harm)"
      risk_score: 20
      level: "CRITICAL"
      mitigation:
        - "Weekly model performance monitoring (AUROC, calibration)"
        - "Automated alert if performance drops >5%"
        - "Seasonal retraining protocol"
        - "Fallback to clinical scoring (NEWS2) if model flagged"
        
    - risk_id: "FAIR-001"
      title: "Age/Race Bias"
      description: "Model may perform differently across demographics"
      likelihood: "Likely"
      impact: "Major (missed deterioration for specific groups)"
      risk_score: 20
      level: "CRITICAL"
      mitigation:
        - "Monthly fairness audit (TPR by age group, race, sex)"
        - "Alert if any group TPR drops below 85%"
        - "Training data balanced across demographics"
        - "Quarterly clinical review of missed deterioration events by group"
        
    - risk_id: "SAFE-001"
      title: "Over-reliance"
      description: "Clinicians stop independent assessment, rely solely on model"
      likelihood: "Possible"
      impact: "Major (model failure undetected)"
      risk_score: 12
      level: "HIGH"
      mitigation:
        - "Model presented as decision SUPPORT (not decision maker)"
        - "Confidence display — low confidence triggers manual assessment"
        - "Training: clinicians must document independent assessment"
        - "Audit: track cases where clinician overrides model"
        
  overall_recommendation: "CONDITIONAL — deploy with enhanced monitoring and weekly performance review"
  
  monitoring_dashboard:
    indicators:
      - name: "Model AUROC"
        threshold: "< 0.85 = alert"
        frequency: "Daily"
      - name: "Fairness disparity (max group TPR gap)"
        threshold: "> 0.10 = alert"
        frequency: "Weekly"
      - name: "Calibration error"
        threshold: "> 0.05 = alert"
        frequency: "Daily"
      - name: "Clinician override rate"
        threshold: "> 30% = investigate (model may be degraded)"
        frequency: "Weekly"
```

---

## Interview Tip

> When asked about AI risk management: "I follow the NIST AI RMF framework with four functions: Govern, Map, Measure, Manage. Govern: establish organizational risk tolerance and accountability — who owns AI risk for each system? Map: systematically identify risks using adapted FMEA (Failure Mode and Effects Analysis) for ML — covering technical risks (drift, adversarial), fairness risks (bias amplification), privacy risks (memorization), safety risks (harmful outputs), and societal risks (feedback loops). Measure: quantify each risk with likelihood × impact scoring, and define metrics to track over time. A healthcare prediction model might track: AUROC (technical), TPR gap between demographic groups (fairness), memorization rate (privacy), and clinician override rate (safety). Manage: prioritize by risk score, implement controls (preventive, detective, corrective), and monitor residual risk continuously. My key insight: risk management isn't a pre-deployment gate — it's continuous. I implement automated risk indicator dashboards that track metrics in production and alert when thresholds are breached. For critical risks, I define escalation paths: threshold breach → automated alert → human investigation → mitigation within SLA. The output of risk assessment directly determines deployment strategy: critical risks unmitigated → don't deploy; high risks → conditional with enhanced monitoring; medium/low → standard monitoring."

---

## Common Mistakes

1. **Risk assessment as a one-time activity** — Doing risk assessment before deployment, filing it away, never updating. Risks change as population shifts, model drifts, and new threats emerge. Solution: continuous risk monitoring with quarterly re-assessment and incident-triggered updates.

2. **Only assessing technical risks** — Focusing on model accuracy while ignoring fairness, privacy, safety, and societal risks. Solution: use NIST AI RMF categories as a checklist — technical, fairness, privacy, safety, transparency, societal.

3. **No risk ownership** — Identified risks with no assigned owner. Nobody is responsible for tracking or mitigating. Solution: every risk gets an owner (person, not team), with clear SLAs for response.

4. **Risk tolerance not defined** — Teams don't know what level of risk is acceptable. Every risk becomes a debate. Solution: organizational risk tolerance defined upfront — "we accept <1% performance degradation, <10% fairness disparity, zero memorization of PII."

5. **Mitigations without verification** — Implementing controls (monitoring, fairness constraints) without verifying they actually reduce risk. Solution: after implementing mitigation, verify effectiveness (did the risk indicator actually improve?), and measure residual risk.

---

## Key Takeaways

- NIST AI RMF: four functions — Govern (policy), Map (identify risks), Measure (quantify), Manage (mitigate)
- Risk scoring: likelihood × impact matrix → prioritize (critical/high/medium/low)
- Risk categories: technical, fairness, privacy, safety, societal — check ALL for every system
- Control types: preventive (stop risk), detective (detect occurrence), corrective (respond after)
- Continuous monitoring: risk indicators tracked in production with automated alerting
- EU AI Act: mandatory risk management system for high-risk AI systems
- ISO 42001: international standard for AI management systems (certifiable)
- Deployment gating: risk assessment determines deployment strategy (deploy/conditional/block)
- Risk ownership: every risk has an assigned owner with response SLA
- Not one-time: quarterly re-assessment, incident-triggered updates, continuous monitoring
