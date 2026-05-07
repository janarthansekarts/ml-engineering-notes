# Intellectual Property

## The Problem / Why This Matters

AI models are trained on massive datasets that often include copyrighted content — books, code, images, music, articles. This creates a web of IP (Intellectual Property) questions with no settled answers in 2026: Does training on copyrighted data constitute fair use? Who owns the output of a model trained on copyrighted inputs? Can generated code inherit the license of training data? If a model memorizes and regurgitates copyrighted text verbatim, is that infringement? These questions have massive practical implications for ML engineers: the datasets you choose, the models you deploy, and the outputs you generate all carry legal risk. Major lawsuits are active — The New York Times v. OpenAI, Getty Images v. Stability AI, Authors Guild v. Meta, and dozens more. Meanwhile, open-source AI has its own complexities: Llama's license restricts commercial use above 700M users, "open-weight" doesn't mean "open-source," and model licenses vary wildly. For ML engineers in 2026, understanding IP means knowing which training data is safe to use, which model licenses permit your use case, how to document data provenance for compliance, and how to implement technical safeguards against memorization and regurgitation of copyrighted content.

---

## The Analogy

Think of AI IP like a musician's creative process:

- **Training on data** = A musician who listens to thousands of songs and develops a style. They've absorbed patterns, structures, and conventions. Is their new music a derivative work of everything they heard? Legally: no (for humans). For AI: contested.
- **Memorization** = A musician who accidentally reproduces an exact melody from a song they heard. Even if unintentional, this IS infringement. For AI: models that memorize and regurgitate training data are generating infringing content.
- **Style transfer** = A musician who writes "in the style of" another artist without copying specific melodies. For humans: generally legal. For AI: still being debated (style isn't copyrightable, but the training process might be).
- **Open source** = A musician who releases their music under Creative Commons. Others can use, remix, and build upon it — but must follow the license terms.

---

## Deep Dive

### Training Data Rights

```yaml
Training_Data_Rights:
  key_questions:
    is_training_fair_use:
      us_law: "Four-factor test: purpose, nature, amount, market effect"
      arguments_for: "Transformative use (learning patterns, not copying), nonexpressive use"
      arguments_against: "Commercial purpose, uses entire works, substitutes for original market"
      status_2026: "Mixed rulings. No Supreme Court clarity. Case-by-case."
      
    eu_law:
      tdm_exception: "Text and Data Mining (TDM) exception in EU Copyright Directive"
      commercial_tdm: "Allowed UNLESS rightsholder explicitly opts out (robots.txt, meta tags)"
      opt_out: "Rightsholders can reserve rights with machine-readable opt-out"
      ai_act: "GPAI providers must describe training data sources, respect copyright"
      
  major_lawsuits_2026:
    nyt_v_openai:
      claim: "GPT-4 memorizes and reproduces NYT articles verbatim"
      status: "Ongoing. Key for memorization/regurgitation liability."
      
    getty_v_stability:
      claim: "Stable Diffusion trained on Getty's copyrighted images without license"
      status: "Ongoing. Key for image generation IP."
      
    authors_guild_v_meta:
      claim: "Llama trained on copyrighted books (LibGen, Z-Library)"
      status: "Ongoing. Key for large-scale book training."
      
    github_copilot_litigation:
      claim: "Copilot reproduces licensed code without attribution"
      status: "Partially dismissed, partially proceeding. Key for code generation."
      
  safe_data_sources:
    public_domain: "Works with expired copyright (pre-1928 in US)"
    creative_commons: "CC-BY, CC-BY-SA (attribution required), CC0 (no restriction)"
    open_data: "Government data, openly licensed datasets"
    licensed_data: "Data specifically licensed for AI training (stock photo AI licenses)"
    synthetic_data: "AI-generated data (but check if generator was trained on copyrighted data)"
    opt_in_data: "Data where creators explicitly consented to AI training"
```

### Model Licensing

```yaml
Model_Licensing:
  types:
    fully_open_source:
      definition: "Open weights + open data + open code + permissive license"
      examples: "OLMo (AI2), Pythia (EleutherAI), BLOOM (BigScience — RAIL license)"
      can_do: "Anything — commercial, modification, redistribution"
      
    open_weights:
      definition: "Model weights available, but restricted license"
      examples:
        llama_3: "Meta Llama License — free for <700M monthly users, no use in competing LLMs"
        gemma: "Google — restricted to specific uses, no generating training data for competing models"
        mistral: "Apache 2.0 — truly permissive (rare for large models)"
        
    api_only:
      definition: "No weights access. Use through API only."
      examples: "GPT-4, Claude, Gemini Pro"
      limitations: "Can't fine-tune (beyond what API allows), can't self-host, can't inspect"
      
  license_comparison:
    apache_2_0:
      permissions: "Commercial use, modification, distribution, patent grant"
      conditions: "Include license, state changes, include NOTICE"
      examples: "Mistral, Falcon, StableLM"
      
    mit:
      permissions: "Everything (most permissive)"
      conditions: "Include copyright notice"
      examples: "Some smaller models"
      
    llama_license:
      permissions: "Commercial use (under 700M MAU), modification, fine-tuning"
      restrictions: "No >700M MAU without Meta agreement, no training competing LLMs"
      
    rail_license:
      what: "Responsible AI License — permissive with use restrictions"
      restrictions: "Cannot use for: surveillance, military, deception, harm"
      examples: "BLOOM, Stable Diffusion (modified RAIL)"
      
    cc_by_nc:
      permissions: "Modification, redistribution"
      restrictions: "Non-commercial only"
      examples: "Some research models"
```

### Generated Content Ownership

```yaml
Generated_Content_Ownership:
  who_owns_ai_output:
    us_law:
      copyright_office_position: "Works generated solely by AI are not copyrightable"
      human_authorship_required: "Must have 'sufficient human authorship' for copyright"
      implication: "Pure AI output = public domain. AI-assisted (human direction) = copyrightable."
      
    eu_law:
      position: "Similar — originality requires human intellectual creation"
      implication: "AI outputs not automatically copyrightable by the user"
      
    practical_implication:
      for_companies: "AI-generated code/content may not be protectable as trade secret or copyright"
      for_users: "Your AI-generated images/text may not be exclusively yours"
      mitigation: "Add sufficient human creative input to claim authorship"
      
  code_generation:
    risk: "Generated code may match training data (copyrighted code with licenses)"
    copylot_example: "GitHub Copilot can generate code matching GPL-licensed training data"
    licenses_that_propagate: "GPL, LGPL, AGPL — copyleft licenses require derivative works to use same license"
    safe_licenses: "MIT, Apache 2.0, BSD — permissive, no copyleft obligation"
    
    safeguards:
      - "Filter outputs that match training data verbatim"
      - "Provide attribution when possible"
      - "Scan generated code for license compatibility"
      - "Use models trained only on permissively-licensed code"
```

### Implementation

```python
# IP compliance implementation for ML systems

"""
Intellectual property compliance for ML: training data provenance,
memorization detection, and license compliance tracking.
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
import re

logger = logging.getLogger(__name__)


class LicenseType(Enum):
    """Common license types for training data and models."""
    PUBLIC_DOMAIN = "public_domain"
    CC0 = "cc0"
    CC_BY = "cc_by"
    CC_BY_SA = "cc_by_sa"
    CC_BY_NC = "cc_by_nc"
    MIT = "mit"
    APACHE_2 = "apache_2"
    GPL = "gpl"
    PROPRIETARY = "proprietary"
    UNKNOWN = "unknown"
    OPT_OUT = "opt_out"  # Rightsholder opted out of AI training


@dataclass
class DataSource:
    """Track provenance of training data source."""
    name: str
    url: str
    license: LicenseType
    license_url: str
    size_gb: float
    description: str
    acquisition_date: str
    opt_out_checked: bool = False
    opt_out_respected: bool = True
    attribution_required: bool = False
    commercial_use_allowed: bool = True
    ai_training_explicitly_allowed: bool = False


@dataclass
class DataProvenanceRegistry:
    """
    Central registry of all training data sources and their licenses.
    
    Required for:
    - EU AI Act compliance (GPAI must describe training data)
    - Copyright compliance (respect opt-outs, track licenses)
    - Audit trail (prove training data was legally obtained)
    """
    
    sources: List[DataSource] = field(default_factory=list)
    
    def add_source(self, source: DataSource) -> bool:
        """
        Add a data source after compliance check.
        
        Returns False if source is not compliant for use.
        """
        # Check if license allows AI training
        if source.license == LicenseType.OPT_OUT:
            logger.warning(f"Source {source.name} has opted out of AI training. CANNOT use.")
            return False
        
        if source.license == LicenseType.PROPRIETARY and not source.ai_training_explicitly_allowed:
            logger.warning(f"Source {source.name} is proprietary without AI training license. RISKY.")
            return False
        
        if source.license == LicenseType.GPL:
            logger.warning(
                f"Source {source.name} is GPL. Model outputs may need to be GPL-licensed. "
                "Consult legal before using for commercial model."
            )
        
        self.sources.append(source)
        logger.info(f"Added source: {source.name} (license: {source.license.value})")
        return True
    
    def compliance_report(self) -> Dict:
        """Generate compliance report for all training data."""
        report = {
            "total_sources": len(self.sources),
            "total_size_gb": sum(s.size_gb for s in self.sources),
            "by_license": {},
            "risks": [],
            "attribution_required": [],
            "commercial_allowed": True,
        }
        
        for source in self.sources:
            license_type = source.license.value
            report["by_license"][license_type] = report["by_license"].get(license_type, 0) + 1
            
            if source.attribution_required:
                report["attribution_required"].append(source.name)
            
            if not source.commercial_use_allowed:
                report["commercial_allowed"] = False
                report["risks"].append(
                    f"{source.name}: Non-commercial license — cannot use for commercial model"
                )
            
            if source.license == LicenseType.UNKNOWN:
                report["risks"].append(
                    f"{source.name}: Unknown license — legal risk"
                )
        
        return report
    
    def eu_ai_act_summary(self) -> str:
        """
        Generate training data summary for EU AI Act GPAI compliance.
        
        Required: sufficiently detailed summary of training data.
        """
        lines = ["# Training Data Summary (EU AI Act GPAI Compliance)\n"]
        lines.append(f"Total sources: {len(self.sources)}")
        lines.append(f"Total data size: {sum(s.size_gb for s in self.sources):.1f} GB\n")
        
        lines.append("## Sources\n")
        for source in self.sources:
            lines.append(f"- **{source.name}**")
            lines.append(f"  - License: {source.license.value}")
            lines.append(f"  - Size: {source.size_gb:.1f} GB")
            lines.append(f"  - Description: {source.description}")
            lines.append(f"  - Opt-out respected: {'Yes' if source.opt_out_respected else 'NO'}")
            lines.append("")
        
        return "\n".join(lines)


class MemorizationDetector:
    """
    Detect if model memorizes and regurgitates training data.
    
    Critical for copyright compliance:
    - If model outputs training data verbatim, that's likely infringement
    - Must detect and filter memorized content before serving to users
    """
    
    def __init__(self, training_data_hashes: Set[str] = None):
        """
        Args:
            training_data_hashes: Set of n-gram hashes from training data
                                 for efficient similarity detection.
        """
        self.training_hashes = training_data_hashes or set()
        self.n_gram_size = 50  # Number of tokens to match
    
    def check_output(self, generated_text: str) -> Dict:
        """
        Check if generated text contains memorized training data.
        
        Returns:
        - is_memorized: bool (likely contains verbatim training content)
        - overlap_score: float (what fraction matches training data)
        - flagged_segments: list of potentially memorized text spans
        """
        # Generate n-grams from output
        words = generated_text.split()
        flagged_segments = []
        
        for i in range(len(words) - self.n_gram_size + 1):
            ngram = " ".join(words[i:i + self.n_gram_size])
            ngram_hash = hashlib.md5(ngram.encode()).hexdigest()
            
            if ngram_hash in self.training_hashes:
                flagged_segments.append({
                    "start": i,
                    "end": i + self.n_gram_size,
                    "text": ngram[:100] + "...",  # Truncate for logging
                })
        
        overlap_score = len(flagged_segments) / max(
            len(words) - self.n_gram_size + 1, 1
        )
        
        return {
            "is_memorized": overlap_score > 0.1 or len(flagged_segments) > 0,
            "overlap_score": overlap_score,
            "flagged_segments": flagged_segments,
            "recommendation": (
                "BLOCK — likely verbatim training data" if overlap_score > 0.3
                else "WARNING — possible memorization" if flagged_segments
                else "OK — no memorization detected"
            ),
        }
    
    def check_code_output(self, generated_code: str) -> Dict:
        """
        Check generated code for license-incompatible training data.
        
        Specifically checks for:
        - Verbatim code from GPL/copyleft repositories
        - Code with copyright headers from training data
        - Known snippets from proprietary codebases
        """
        issues = []
        
        # Check for copyright headers
        copyright_pattern = r"Copyright\s+\(c\)\s+\d{4}.*?\n"
        matches = re.finditer(copyright_pattern, generated_code, re.IGNORECASE)
        for match in matches:
            issues.append({
                "type": "copyright_header",
                "text": match.group(),
                "action": "Remove or verify attribution is appropriate",
            })
        
        # Check for license headers
        license_patterns = [
            (r"GNU General Public License", "GPL — copyleft, may propagate"),
            (r"AGPL", "AGPL — strongest copyleft, propagates to network use"),
            (r"Licensed under the Apache License", "Apache 2.0 — permissive, OK"),
            (r"MIT License", "MIT — permissive, OK"),
        ]
        for pattern, description in license_patterns:
            if re.search(pattern, generated_code, re.IGNORECASE):
                issues.append({
                    "type": "license_header",
                    "license": description,
                    "action": "Verify license compatibility with your project",
                })
        
        return {
            "issues_found": len(issues),
            "issues": issues,
            "recommendation": (
                "Review code for license compliance" if issues
                else "No obvious license issues detected"
            ),
        }


class ModelLicenseChecker:
    """
    Verify model license compatibility with intended use.
    
    Different models have different restrictions:
    - Commercial use limits (Llama: >700M MAU needs Meta agreement)
    - Derivative work restrictions
    - Use case restrictions (RAIL licenses)
    """
    
    def __init__(self):
        self.license_db = {
            "llama-3": {
                "commercial": True,
                "max_mau": 700_000_000,
                "restrictions": ["No training competing LLMs", "No >700M MAU without Meta approval"],
                "fine_tuning_allowed": True,
                "redistribution": True,
            },
            "mistral": {
                "commercial": True,
                "max_mau": None,  # No limit
                "restrictions": [],
                "fine_tuning_allowed": True,
                "redistribution": True,
                "license": "Apache 2.0",
            },
            "gemma": {
                "commercial": True,
                "restrictions": [
                    "No generating training data for competing models",
                    "Must follow Google's Prohibited Use Policy",
                ],
                "fine_tuning_allowed": True,
            },
            "gpt-4": {
                "commercial": True,
                "restrictions": [
                    "API-only (no weights access)",
                    "Subject to OpenAI usage policies",
                    "No competing with OpenAI services",
                ],
                "fine_tuning_allowed": True,  # Via API only
                "redistribution": False,
            },
        }
    
    def check_compatibility(self, model_name: str, use_case: Dict) -> Dict:
        """
        Check if model license is compatible with intended use.
        
        Args:
            model_name: Name of the model
            use_case: Dict with "commercial", "mau", "fine_tune", "redistribute"
        """
        if model_name not in self.license_db:
            return {"compatible": "unknown", "reason": "Model not in license database"}
        
        license_info = self.license_db[model_name]
        issues = []
        
        # Commercial use
        if use_case.get("commercial") and not license_info.get("commercial"):
            issues.append("Commercial use not allowed by license")
        
        # MAU limit
        max_mau = license_info.get("max_mau")
        if max_mau and use_case.get("mau", 0) > max_mau:
            issues.append(f"Expected MAU ({use_case['mau']}) exceeds limit ({max_mau})")
        
        # Fine-tuning
        if use_case.get("fine_tune") and not license_info.get("fine_tuning_allowed"):
            issues.append("Fine-tuning not allowed by license")
        
        # Redistribution
        if use_case.get("redistribute") and not license_info.get("redistribution"):
            issues.append("Redistribution of model weights not allowed")
        
        return {
            "compatible": len(issues) == 0,
            "issues": issues,
            "restrictions": license_info.get("restrictions", []),
            "recommendation": (
                "License compatible with use case" if not issues
                else f"License INCOMPATIBLE: {'; '.join(issues)}"
            ),
        }
```

---

## How It Works in Practice

### IP-Compliant Model Training Pipeline

```yaml
IP_Compliant_Pipeline:
  scenario: "Enterprise training a domain-specific LLM for commercial use"
  
  data_sourcing:
    step_1_inventory:
      - "Catalog all potential data sources"
      - "Check license of each source"
      - "Verify opt-out status (robots.txt, AI training opt-outs)"
      - "Reject: proprietary without license, GPL (copyleft risk), opt-out sources"
      
    step_2_approved_sources:
      - "Common Crawl (license: varies per page — filter to permissive)"
      - "Wikipedia (CC-BY-SA — attribution required)"
      - "Licensed data providers (contractual AI training rights)"
      - "Internal company data (owned, no external IP issues)"
      - "Synthetic data generated from own models (verify generator's training data)"
      
    step_3_documentation:
      - "Record every source in Data Provenance Registry"
      - "Generate EU AI Act training data summary"
      - "Maintain audit trail of all data decisions"
      
  training_safeguards:
    deduplication: "Remove duplicate documents (reduces memorization)"
    pii_removal: "Strip personal information before training"
    opt_out_filtering: "Respect robots.txt AI directives and rightsholder opt-outs"
    provenance_tracking: "Link training samples to source and license"
    
  deployment_safeguards:
    memorization_detection: "Filter outputs matching training data verbatim"
    attribution: "Cite sources when possible (for RAG systems)"
    code_scanning: "Scan generated code for license headers/copyleft indicators"
    user_agreements: "Clear terms about ownership of AI-generated content"
```

---

## Interview Tip

> When asked about AI and intellectual property: "I approach AI IP from three angles: training data rights, model licensing, and output ownership. Training data: I maintain a Data Provenance Registry documenting every data source, its license, and whether the rightsholder opted out of AI training. For EU AI Act compliance, GPAI providers must publish a sufficiently detailed summary of training data — I automate this from the registry. I only use data that's public domain, permissively licensed (CC-BY, Apache, MIT), or explicitly licensed for AI training. I reject: proprietary content without license, GPL (copyleft propagation risk), and opted-out sources. Model licensing: critical for commercial use. Llama 3 is free under 700M MAU but restricts training competing LLMs. Mistral (Apache 2.0) is truly permissive. I verify license compatibility with the specific use case before building on any model. Memorization: the biggest practical risk. If a model regurgitates copyrighted text verbatim, that's likely infringement regardless of fair use arguments about training. I implement: training deduplication (reduces memorization), output filtering (detect verbatim matches), and n-gram comparison against training data. For code generation: I scan outputs for license headers and copyleft indicators. Generated content ownership: in most jurisdictions, pure AI output isn't copyrightable — it requires sufficient human authorship. I document human creative contribution for any AI-assisted content that needs IP protection."

---

## Common Mistakes

1. **Assuming all public web data is usable** — "It's on the internet so it's free to use" is legally incorrect. Web content has copyright. Solution: check licenses, respect opt-outs, prefer explicitly licensed data sources.

2. **Ignoring copyleft licenses in training data** — Training on GPL-licensed code without understanding propagation. If model outputs are derivative works of GPL code, your entire system may need to be GPL. Solution: filter training data to permissive licenses (MIT, Apache, BSD) for commercial models.

3. **No memorization detection** — Deploying models that can regurgitate training data verbatim when prompted. Solution: implement deduplication during training (reduces memorization), and output filtering that compares against training data at serving time.

4. **Conflating open-weights with open-source** — Assuming Llama is "open source" and has no restrictions. It has commercial use limits and restricts training competing models. Solution: read the actual license. "Open weights" means you can see/use the model, not that all uses are permitted.

5. **Not tracking data provenance** — Using datasets without recording where they came from. When audited or sued, can't demonstrate compliance. Solution: maintain a Data Provenance Registry from day one — document every source, license, and acquisition decision.

---

## Key Takeaways

- Training on copyrighted data: legally contested (US fair use unclear, EU has TDM exception with opt-out)
- Major lawsuits active: NYT v. OpenAI, Getty v. Stability AI, Authors Guild v. Meta
- Data provenance: maintain registry of all training sources with licenses — required for EU AI Act
- Model licensing: open-weights ≠ open-source. Llama (restricted), Mistral (Apache 2.0), Gemma (restricted)
- Generated content: pure AI output generally not copyrightable — needs human authorship
- Memorization: biggest practical risk — models regurgitating training data = likely infringement
- Safeguards: deduplication, output filtering, license scanning, provenance tracking
- Code generation: scan for copyleft license indicators (GPL propagation risk)
- EU AI Act: GPAI must publish training data summary, respect opt-outs
- Safe sources: public domain, CC0, CC-BY, Apache 2.0, MIT, explicitly licensed for AI
