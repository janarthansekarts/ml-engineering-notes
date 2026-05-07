# ML Security in the Cloud

## The Problem / Why This Matters

ML systems are uniquely vulnerable because they combine multiple attack surfaces: sensitive training data (PII, proprietary information), expensive compute resources (GPU clusters worth thousands per hour), valuable intellectual property (trained models worth millions in development cost), and inference endpoints that process user data in real-time. In 2026, ML security has expanded beyond traditional cloud security to include: model theft (extracting model weights through API probing), training data extraction (recovering private data from model outputs), adversarial attacks (crafted inputs that cause misclassification), prompt injection (manipulating LLM behavior through user input), supply chain attacks (poisoned models or datasets from public hubs), and model backdoors (hidden behaviors triggered by specific inputs). Cloud ML security requires defense-in-depth across multiple layers: IAM (Identity and Access Management — who can access what), network isolation (VPC, private endpoints), data encryption (at-rest and in-transit), model security (access controls, serving protections), and compliance (audit trails, data governance). Getting security wrong means: data breaches (regulatory fines, reputation damage), model theft (competitive advantage lost), compute hijacking (crypto-mining on your GPUs), and adversarial attacks (model produces harmful outputs in production).

---

## The Analogy

Think of ML security like securing a pharmaceutical research facility:

- **Network isolation (VPC)** = The building's walls, fences, and locked entrances. Only authorized people can enter, and the building has separate wings (subnets) for different sensitivity levels.
- **IAM** = Badge access system. Researchers can enter labs but not the vault. Janitors can enter hallways but not labs. The CEO has a master badge but still can't override safety protocols.
- **Data encryption** = Locked filing cabinets AND coded language. Even if someone breaks into the building (network breach), they can't read the research notes (encrypted data) without the specific key.
- **Model security** = Protecting the drug formula. The formula (model weights) is the IP. Someone might try to reverse-engineer it by ordering lots of samples (API probing) or steal it from the lab (unauthorized model download).
- **Prompt injection** = Social engineering the receptionist. Someone calls pretending to be maintenance staff and tricks the receptionist into unlocking doors. LLMs can be "tricked" by inputs that override their instructions.

---

## Deep Dive

### ML Security Layers

```yaml
ML_Security_Layers:
  layer_1_identity:
    what: "Control WHO can access ML resources"
    services:
      aws: "IAM, STS, Organizations"
      gcp: "Cloud IAM, Workload Identity"
      azure: "Entra ID, Managed Identity"
    principles:
      least_privilege:
        description: "Grant minimum permissions needed for each role"
        ml_roles:
          data_scientist:
            can: "Read data, submit training jobs, view experiments"
            cannot: "Deploy to production, modify IAM, access other team's data"
          ml_engineer:
            can: "Deploy models, manage pipelines, view production logs"
            cannot: "Access raw PII data, modify network config"
          platform_admin:
            can: "Manage infrastructure, networking, IAM policies"
            cannot: "Read training data, view model predictions"
      service_accounts:
        rule: "Never use personal credentials in code or pipelines"
        pattern: "Each service gets dedicated service account with minimal permissions"
        rotation: "Rotate keys every 90 days (or use short-lived tokens)"
      mfa:
        rule: "MFA required for all human access to ML infrastructure"
        
  layer_2_network:
    what: "Control WHERE traffic can flow"
    components:
      vpc_isolation:
        training: "Private subnet, no internet access"
        serving: "Private subnet with load balancer (no direct public access)"
        data: "Private subnet, accessible only from training/serving"
      private_endpoints:
        what: "Access cloud services without traversing public internet"
        examples:
          - "S3/GCS via VPC endpoint (training data stays in VPC)"
          - "Container registry via private link (pull images privately)"
          - "Model registry via private endpoint"
      firewall_rules:
        inbound: "Only load balancer can reach serving endpoints"
        outbound: "Only approved destinations (package registries, monitoring)"
        inter_service: "Explicit allow-lists between ML services"
      no_public_ips:
        rule: "GPU instances NEVER have public IP addresses"
        access: "Use bastion host, VPN, or SSM/IAP for admin access"
        
  layer_3_data:
    what: "Protect data at-rest and in-transit"
    encryption:
      at_rest:
        training_data: "AES-256, customer-managed keys (CMK)"
        model_artifacts: "Encrypted in object storage (SSE-KMS)"
        checkpoints: "Encrypted at storage layer"
        feature_store: "Encrypted (both online and offline)"
      in_transit:
        all_traffic: "TLS 1.3 minimum"
        inter_node_training: "Encrypted NCCL traffic (for sensitive workloads)"
        api_calls: "HTTPS only, certificate validation"
      key_management:
        service: "AWS KMS / GCP Cloud KMS / Azure Key Vault"
        rotation: "Automatic annual rotation"
        access: "Separate key access from data access"
        
    data_governance:
      classification:
        public: "Open datasets, public model outputs"
        internal: "Proprietary training data, model weights"
        confidential: "PII, healthcare data, financial records"
        restricted: "Government classified, biometric data"
      access_controls:
        column_level: "Mask PII columns in training data"
        row_level: "Filter data by team jurisdiction"
        time_limited: "Temporary access grants (expire after task)"
        
  layer_4_model:
    what: "Protect models from theft, tampering, and misuse"
    threats:
      model_theft:
        description: "Attacker copies model weights"
        vectors: ["Unauthorized API access", "Insider downloading weights", "Supply chain compromise"]
        mitigations:
          - "Access control on model artifacts (IAM)"
          - "Model registry with audit logs (who downloaded what)"
          - "Watermarking (embed identifiable signal in model outputs)"
          
      model_extraction:
        description: "Reconstruct model by querying API extensively"
        vectors: ["Repeated API calls with crafted inputs", "Distillation attacks"]
        mitigations:
          - "Rate limiting on inference endpoints"
          - "Query logging and anomaly detection"
          - "Output perturbation (add small noise to logits)"
          - "Monitor for systematic querying patterns"
          
      adversarial_attacks:
        description: "Crafted inputs that cause model to misclassify"
        types: ["Evasion (test-time)", "Poisoning (training-time)", "Backdoor"]
        mitigations:
          - "Input validation and sanitization"
          - "Adversarial training (train on adversarial examples)"
          - "Ensemble methods (harder to fool multiple models)"
          - "Input anomaly detection (reject out-of-distribution inputs)"
          
      prompt_injection:
        description: "User input overrides LLM system instructions"
        types: ["Direct (user prompt)", "Indirect (injected in retrieved content)"]
        mitigations:
          - "Input/output filtering (content safety systems)"
          - "Prompt sandboxing (separate system/user instruction processing)"
          - "Output validation (check response doesn't leak system prompt)"
          - "Defense in depth (never rely on LLM alone for security decisions)"
```

### Cloud Security Architecture

```yaml
Cloud_ML_Security_Architecture:
  aws_security:
    iam:
      sagemaker_execution_role:
        purpose: "Permissions for training jobs and endpoints"
        permissions:
          - "s3:GetObject on training data bucket"
          - "s3:PutObject on model artifact bucket"
          - "ecr:GetDownloadUrlForLayer (pull training containers)"
          - "cloudwatch:PutMetricData (emit metrics)"
        denied:
          - "s3:* on other buckets"
          - "ec2:* (no direct EC2 access)"
          - "iam:* (no IAM modification)"
          
      data_scientist_role:
        purpose: "Human user permissions for development"
        permissions:
          - "sagemaker:CreateTrainingJob"
          - "sagemaker:CreateEndpoint (dev only)"
          - "s3:GetObject on approved datasets"
        conditions:
          - "aws:RequestedRegion: us-east-1 (restrict to approved region)"
          - "sagemaker:InstanceType: ml.g5.* (limit expensive instances)"
          
    network:
      vpc_config:
        subnets: "Private only (no internet gateway)"
        endpoints: ["s3", "ecr.api", "ecr.dkr", "sagemaker.api", "sts", "kms"]
        security_groups: "Least-privilege (only allow necessary ports)"
        
    encryption:
      s3: "SSE-KMS with customer-managed key"
      ebs: "Encrypted with KMS key (required by SCP)"
      sagemaker: "Inter-container encryption for training"
      
  gcp_security:
    iam:
      service_accounts:
        training_sa: "roles/aiplatform.user, roles/storage.objectViewer"
        serving_sa: "roles/aiplatform.predictor, roles/storage.objectViewer"
        pipeline_sa: "roles/aiplatform.admin (limited to project)"
      vpc_service_controls:
        what: "Define security perimeters around GCP resources"
        effect: "Data cannot leave the perimeter even with valid credentials"
        use_case: "Prevent training data exfiltration"
        
    network:
      private_service_connect: "Access Vertex AI without public endpoint"
      authorized_networks: "Restrict Vertex AI access to specific VPC"
      
  azure_security:
    identity:
      managed_identity:
        what: "Azure-managed credentials (no secrets to rotate)"
        types: ["System-assigned (per resource)", "User-assigned (shared)"]
        use: "Azure ML workspace uses managed identity for all access"
      conditional_access:
        what: "Policy-based access control (location, device, risk level)"
        example: "Block ML access from untrusted locations"
        
    network:
      managed_vnet:
        what: "Azure ML managed virtual network"
        effect: "All training/serving traffic stays in VNet"
        private_endpoints: "Storage, ACR, Key Vault, OpenAI — all private"
```

### LLM-Specific Security

```python
# LLM security patterns

"""
Security patterns specific to LLM (Large Language Model) deployment:
prompt injection defense, output filtering, and data leakage prevention.
"""

llm_security_patterns = {
    "prompt_injection_defense": {
        "description": "Prevent user inputs from overriding system instructions",
        "layers": [
            {
                "layer": "Input filtering",
                "what": "Detect and reject obviously malicious inputs",
                "patterns_to_detect": [
                    "Ignore previous instructions",
                    "You are now in developer mode",
                    "Repeat your system prompt",
                    "Encoded instructions (base64, rot13)",
                    "Markdown/HTML injection attempting to hide instructions",
                ],
                "implementation": "Regex + classifier (fine-tuned model for injection detection)",
            },
            {
                "layer": "Prompt architecture",
                "what": "Structure prompts to resist injection",
                "techniques": [
                    "XML/delimiter separation between system and user content",
                    "Role-based prefixes (SYSTEM: vs USER: clearly marked)",
                    "Sandwich defense (repeat critical instructions after user input)",
                    "Never include secrets in system prompt (assume it can be extracted)",
                ],
            },
            {
                "layer": "Output validation",
                "what": "Check model output before returning to user",
                "checks": [
                    "Does output contain system prompt fragments? (block)",
                    "Does output contain PII not in the input? (block)",
                    "Does output match expected format? (validate)",
                    "Content safety check (harmful, violent, illegal content)",
                ],
            },
            {
                "layer": "Indirect injection defense",
                "what": "Prevent injections in retrieved/external content",
                "scenario": "RAG system retrieves document with hidden injection",
                "defenses": [
                    "Sanitize retrieved content (strip suspicious patterns)",
                    "Separate retrieval context from user input in prompt",
                    "Use citations to trace which source influenced output",
                    "Monitor for outputs that contradict system instructions",
                ],
            },
        ],
    },
    
    "data_leakage_prevention": {
        "description": "Prevent model from exposing training data or PII",
        "risks": [
            "Model memorizes and outputs PII from training data",
            "System prompt leaked through adversarial queries",
            "RAG system exposes documents user shouldn't access",
        ],
        "mitigations": [
            {
                "technique": "Output filtering",
                "what": "Scan outputs for PII patterns (SSN, email, phone, credit card)",
                "action": "Redact or block response if PII detected",
            },
            {
                "technique": "Differential privacy in training",
                "what": "Add noise during training to prevent memorization",
                "trade_off": "Slightly lower model quality for privacy guarantees",
            },
            {
                "technique": "Access-controlled RAG",
                "what": "Filter retrieved documents by user's access level",
                "implementation": "Check user permissions before including document in context",
            },
            {
                "technique": "Rate limiting and monitoring",
                "what": "Detect systematic extraction attempts",
                "signals": [
                    "Repeated similar queries (probing for memorized data)",
                    "Requests for lists of names/emails/numbers",
                    "High query volume from single user",
                ],
            },
        ],
    },
    
    "model_supply_chain_security": {
        "description": "Ensure downloaded models and datasets are safe",
        "risks": [
            "Poisoned model from HuggingFace Hub (backdoor embedded)",
            "Malicious code in model pickle files (arbitrary code execution)",
            "Tampered datasets (label poisoning for specific behaviors)",
        ],
        "mitigations": [
            {
                "technique": "Model verification",
                "what": "Verify model integrity before deployment",
                "actions": [
                    "Check SHA-256 hash matches expected value",
                    "Prefer safetensors format (no pickle, no arbitrary code execution)",
                    "Scan model files for embedded code/exploits",
                    "Download from verified sources only (official model pages)",
                ],
            },
            {
                "technique": "Sandbox execution",
                "what": "Load and test models in isolated environment",
                "implementation": "Container with no network access, limited filesystem",
            },
            {
                "technique": "Model scanning",
                "what": "Automated tools to detect backdoors/anomalies",
                "tools": ["ModelScan", "Fickling (pickle security)", "Custom behavior tests"],
            },
        ],
    },
}


# Security monitoring and audit
security_monitoring = {
    "audit_logging": {
        "what_to_log": [
            "All model artifact downloads (who, when, which model)",
            "Training data access (who accessed what data, when)",
            "Model deployments (who deployed, which version, where)",
            "Inference requests (for sensitive models — rate, patterns)",
            "IAM changes (permission grants, role modifications)",
            "Network changes (security group modifications, endpoint creation)",
        ],
        "retention": "7 years minimum for regulated industries",
        "alerting": [
            "Unusual model download patterns (potential theft)",
            "Training data access outside business hours",
            "Failed authentication attempts on ML endpoints",
            "Inference spike from single source (potential extraction)",
        ],
    },
    
    "compliance_frameworks": {
        "hipaa": {
            "applies_to": "Healthcare ML models processing PHI",
            "requirements": [
                "Encryption at-rest and in-transit",
                "Access logging and audit trail",
                "BAA (Business Associate Agreement) with cloud provider",
                "Minimum necessary access (least privilege)",
            ],
        },
        "gdpr": {
            "applies_to": "ML models processing EU personal data",
            "requirements": [
                "Right to erasure (can you remove data from trained model?)",
                "Data minimization (don't train on unnecessary PII)",
                "Consent management (track training data consent)",
                "Model explainability (explain predictions involving personal data)",
            ],
            "challenge": "ML models can memorize training data — full erasure may require retraining",
        },
        "sox_pci": {
            "applies_to": "Financial ML models",
            "requirements": [
                "Change management (model updates tracked and approved)",
                "Access control (separation of duties)",
                "Monitoring and alerting",
                "Regular security assessments",
            ],
        },
    },
}
```

---

## How It Works in Practice

### Secure ML Platform Architecture

```yaml
Secure_ML_Platform:
  scenario: "Healthcare company deploying ML models processing patient data (PHI)"
  
  architecture:
    network:
      vpc: "Dedicated VPC for ML workloads"
      subnets:
        training: "Private, no internet (VPC endpoints for all services)"
        serving: "Private, ALB fronting endpoints"
        data: "Private, accessible only from training subnet"
      connectivity:
        admin: "AWS SSM Session Manager (no SSH keys, full audit)"
        internet: "NAT gateway for package installs only (allowlisted domains)"
        cross_account: "PrivateLink for shared services"
        
    identity:
      authentication: "SSO via Okta → AWS IAM Identity Center"
      authorization: "Attribute-based access control (department, project, sensitivity)"
      machine_identity: "IAM roles for all services (no long-lived keys)"
      
    data_protection:
      classification: "PHI tagged with 'sensitivity:high' resource tags"
      encryption_key: "AWS KMS CMK (customer-managed, auto-rotation)"
      access_logging: "CloudTrail + S3 access logs (7 year retention)"
      dlp: "Macie scans training data for unexpected PII"
      
    model_security:
      registry: "Private ECR + SageMaker Model Registry"
      signing: "All containers signed (Sigstore/Cosign)"
      scanning: "ECR image scanning before deployment"
      serving: "Private endpoints only (no public internet access to models)"
      
    monitoring:
      siem: "CloudTrail → Security Hub → Splunk"
      anomaly_detection: "GuardDuty for infrastructure, custom for ML-specific"
      alerts:
        - "Model artifact downloaded outside business hours"
        - "Training job accessing unauthorized data bucket"
        - "Inference endpoint receiving >10× normal traffic"
        - "New IAM role with broad ML permissions created"
        
  compliance:
    hipaa:
      baa: "Signed with AWS"
      encryption: "PHI encrypted at-rest (KMS) and in-transit (TLS 1.3)"
      access: "MFA required, access reviewed quarterly"
      audit: "Full audit trail, 7-year retention"
    training_data:
      de_identification: "De-identify PHI before model training where possible"
      access_log: "Every access to patient data logged and auditable"
      retention: "Training data deleted after model is trained and validated"
```

---

## Interview Tip

> When asked about ML security in the cloud: "I design ML security in layers. Network: all training infrastructure in private subnets with no public IPs — VPC endpoints for service access (S3, container registry, model registry). Access is via Session Manager or IAP, never SSH with keys. Identity: least-privilege IAM roles — data scientists can submit training jobs but can't deploy to production. Service accounts for all automation (no human credentials in pipelines), with short-lived tokens. Data: customer-managed encryption keys (CMK) for all data at-rest, TLS 1.3 in-transit. Column-level access controls on training data (mask PII fields not needed for model). Model security: private model registry with download audit logs, container image signing, inference endpoints behind private load balancers. For LLMs specifically: prompt injection defense (input filtering + output validation + structured prompt architecture), content safety filtering, rate limiting to prevent model extraction, and output scanning for PII leakage. Supply chain: prefer safetensors format (no arbitrary code execution like pickle), verify model hashes, scan images before deployment. Monitoring: alert on unusual patterns — bulk model downloads, inference spikes from single IP, data access outside hours. For compliance (HIPAA, GDPR): encryption everywhere, full audit trail (7-year retention), access reviews, and data minimization (don't train on PII you don't need)."

---

## Common Mistakes

1. **Training instances with public IPs** — GPU instances ($8-15/hr) exposed to internet for "easy access." Crypto miners find them within hours. Solution: private subnets only, access via SSM/IAP/bastion. No public IPs ever on training infrastructure.

2. **Overly broad IAM roles** — Service account with `s3:*` and `sagemaker:*` because "it's easier." Compromised service account means full data access. Solution: scope to specific buckets, actions, and resource ARNs. Use IAM Access Analyzer to identify unused permissions.

3. **Pickle model files from untrusted sources** — Loading model from HuggingFace Hub that uses pickle format — arbitrary code execution on model load. Solution: prefer safetensors format (safe, no code execution). If pickle required, scan with Fickling and load in sandboxed container.

4. **No prompt injection defense** — Deploying LLM-powered application without input/output filtering. Users extract system prompts, bypass content filters, and cause harmful outputs. Solution: multi-layer defense (input sanitization, structured prompts, output validation, content safety API).

5. **Secrets in training code** — Hardcoded API keys, database passwords, or model registry tokens in training scripts (committed to Git). Solution: use secrets manager (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault). Inject at runtime via environment variables from secure store.

---

## Key Takeaways

- Defense in depth: network isolation + IAM + encryption + model security + monitoring
- Network: private subnets, no public IPs, VPC endpoints, cluster placement groups
- IAM: least privilege per role, service accounts (no human credentials), short-lived tokens
- Encryption: CMK for at-rest, TLS 1.3 for in-transit, separate key access from data access
- LLM security: prompt injection defense (input filter + output validation + structured prompts)
- Supply chain: safetensors format, hash verification, container signing, image scanning
- Model theft prevention: access controls, download audit logs, rate limiting, watermarking
- Data governance: classification, column-level access, PII masking, access logging
- Compliance: HIPAA/GDPR/SOX require encryption, audit trails, access reviews, data minimization
- Monitoring: alert on anomalies (bulk downloads, inference spikes, off-hours access)
