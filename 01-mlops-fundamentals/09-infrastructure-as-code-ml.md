# Infrastructure as Code for ML

## The Problem / Why This Matters

ML infrastructure is complex: GPU clusters, training job configurations, model serving endpoints, feature store backends, monitoring stacks, networking for distributed training, storage for datasets and artifacts, and IAM (Identity and Access Management) policies for data access. Manually provisioning and configuring this infrastructure through cloud consoles leads to: undocumented environments that can't be reproduced, "snowflake" clusters that drift from intended configuration, inability to spin up identical environments for testing, and security misconfigurations from clicking through UIs. IaC (Infrastructure as Code) applies software engineering practices to infrastructure — define infrastructure in version-controlled code, review changes via pull requests, test before applying, and maintain consistent environments across development, staging, and production. In 2026, IaC for ML has matured with GPU-aware modules, ML-specific Terraform providers, and Pulumi components that encapsulate common ML infrastructure patterns.

---

## The Analogy

Think of IaC for ML like architectural blueprints for a data center:

- **Without IaC** = Building a data center by telling construction workers "put a server rack there, run cables here, add cooling somehow." Each worker interprets differently, nothing is documented, and building an identical second data center is impossible.
- **With IaC** = Detailed blueprints that specify every component: rack positions, cable routing, cooling capacity, power distribution. Anyone can read the blueprint and build an identical facility. Changes go through an approval process (PR review). You can simulate the build before executing (plan/preview).
- **Terraform/Pulumi** = The architectural software that generates and manages the blueprints. Declare what you want (10 GPU nodes, a Redis cluster, an S3 bucket), and it figures out how to build it and tracks state.

---

## Deep Dive

### IaC Tools for ML Infrastructure

```yaml
IaC_Tools:
  terraform:
    description: "Most widely used IaC tool — declarative, provider-based, HCL language"
    strengths:
      - "Massive provider ecosystem (AWS, GCP, Azure, Kubernetes, Datadog, etc.)"
      - "State management (tracks what exists vs what's declared)"
      - "Plan before apply (preview changes safely)"
      - "Modules for reusability"
      - "Mature, battle-tested, large community"
    ml_relevant_providers:
      - "aws (SageMaker, EKS, EC2 GPU, S3)"
      - "google (Vertex AI, GKE, Cloud Storage, TPU)"
      - "azurerm (Azure ML, AKS, Blob Storage)"
      - "kubernetes (K8s resources for model serving)"
      - "helm (deploy KServe, Kubeflow, MLflow)"
    version: "1.12+ (2026)"
    
  pulumi:
    description: "IaC using real programming languages (Python, TypeScript, Go)"
    strengths:
      - "Full programming language (loops, conditions, functions)"
      - "Native testing (unit test infrastructure code)"
      - "Strong typing (IDE autocomplete, compile-time errors)"
      - "Reusable components as packages"
    advantage_for_ml: "ML engineers already know Python — no new language (vs HCL)"
    
  crossplane:
    description: "Kubernetes-native IaC — define cloud resources as K8s custom resources"
    strengths: "GitOps workflow, Kubernetes-native teams"
    use_case: "Teams that want to manage everything through Kubernetes"
```

### Terraform for ML Infrastructure

```yaml
Terraform_ML_Examples:
  gpu_training_cluster:
    description: "Provision GPU cluster for model training on AWS"
    code: |
      # GPU Training Cluster
      resource "aws_eks_cluster" "ml_training" {
        name     = "ml-training-cluster"
        role_arn = aws_iam_role.eks_cluster.arn
        version  = "1.30"
        
        vpc_config {
          subnet_ids = var.private_subnet_ids
        }
      }
      
      # GPU Node Group (H100 instances for LLM training)
      resource "aws_eks_node_group" "gpu_h100" {
        cluster_name    = aws_eks_cluster.ml_training.name
        node_group_name = "gpu-h100-training"
        node_role_arn   = aws_iam_role.node_group.arn
        subnet_ids      = var.private_subnet_ids
        
        instance_types = ["p5.48xlarge"]  # 8x H100 GPUs
        
        scaling_config {
          min_size     = 0   # Scale to zero when not training
          max_size     = 8   # Up to 8 nodes (64 H100s)
          desired_size = 0
        }
        
        labels = {
          "node-type"    = "gpu-training"
          "gpu-type"     = "h100"
          "accelerator"  = "nvidia"
        }
        
        taint {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
      
      # GPU Node Group (L4 instances for inference)
      resource "aws_eks_node_group" "gpu_l4_inference" {
        cluster_name    = aws_eks_cluster.ml_training.name
        node_group_name = "gpu-l4-inference"
        node_role_arn   = aws_iam_role.node_group.arn
        subnet_ids      = var.private_subnet_ids
        
        instance_types = ["g6.xlarge"]  # 1x L4 GPU
        
        scaling_config {
          min_size     = 2   # Always-on for serving
          max_size     = 20  # Scale with traffic
          desired_size = 4
        }
        
        labels = {
          "node-type"   = "gpu-inference"
          "gpu-type"    = "l4"
        }
      }
      
  model_serving_infrastructure:
    description: "Provision model serving stack"
    code: |
      # Redis for feature store online serving
      resource "aws_elasticache_replication_group" "feature_store" {
        replication_group_id = "ml-feature-store"
        description          = "Online feature store for real-time serving"
        node_type            = "cache.r7g.xlarge"
        num_cache_clusters   = 3
        
        automatic_failover_enabled = true
        multi_az_enabled           = true
        
        at_rest_encryption_enabled = true
        transit_encryption_enabled = true
      }
      
      # S3 bucket for model artifacts
      resource "aws_s3_bucket" "model_artifacts" {
        bucket = "ml-model-artifacts-${var.environment}"
        
        versioning {
          enabled = true
        }
        
        lifecycle_rule {
          id      = "archive-old-models"
          enabled = true
          
          transition {
            days          = 90
            storage_class = "GLACIER"
          }
        }
      }
      
      # SageMaker endpoint for model serving
      resource "aws_sagemaker_endpoint" "fraud_model" {
        name                 = "fraud-detection-${var.environment}"
        endpoint_config_name = aws_sagemaker_endpoint_configuration.fraud.name
      }
      
  mlflow_server:
    description: "Provision MLflow tracking server"
    code: |
      # RDS for MLflow metadata
      resource "aws_db_instance" "mlflow_db" {
        identifier     = "mlflow-metadata"
        engine         = "postgres"
        engine_version = "16.4"
        instance_class = "db.t4g.medium"
        
        db_name  = "mlflow"
        username = var.mlflow_db_username
        password = var.mlflow_db_password
        
        storage_encrypted = true
        multi_az          = var.environment == "production"
      }
      
      # S3 for MLflow artifacts
      resource "aws_s3_bucket" "mlflow_artifacts" {
        bucket = "mlflow-artifacts-${var.environment}"
      }
      
      # ECS service for MLflow server
      resource "aws_ecs_service" "mlflow" {
        name            = "mlflow-server"
        cluster         = aws_ecs_cluster.ml_platform.id
        task_definition = aws_ecs_task_definition.mlflow.arn
        desired_count   = 2
        
        load_balancer {
          target_group_arn = aws_lb_target_group.mlflow.arn
          container_name   = "mlflow"
          container_port   = 5000
        }
      }
```

### ML Infrastructure Modules

```yaml
Reusable_Modules:
  principle: "Package common ML infrastructure patterns as reusable modules"
  
  examples:
    gpu_cluster_module:
      inputs: "gpu_type, min_nodes, max_nodes, spot_enabled"
      provisions: "EKS node group + NVIDIA device plugin + cluster autoscaler config"
      
    model_serving_module:
      inputs: "model_name, instance_type, min_replicas, max_replicas, endpoint_type"
      provisions: "SageMaker endpoint OR KServe InferenceService (based on endpoint_type)"
      
    feature_store_module:
      inputs: "store_type (feast/tecton), online_backend (redis/dynamodb), offline_backend"
      provisions: "Online store + offline store + IAM roles + networking"
      
    mlflow_module:
      inputs: "environment, high_availability, artifact_storage_size"
      provisions: "RDS + S3 + ECS service + load balancer + IAM"
      
    monitoring_module:
      inputs: "model_names, alert_channels, drift_threshold"
      provisions: "Prometheus + Grafana + alerting rules + Evidently integration"
      
  usage: |
    # Use GPU cluster module
    module "training_cluster" {
      source = "./modules/gpu-cluster"
      
      gpu_type     = "h100"
      min_nodes    = 0
      max_nodes    = 8
      spot_enabled = true
      environment  = "production"
    }
    
    # Use model serving module
    module "fraud_serving" {
      source = "./modules/model-serving"
      
      model_name   = "fraud-detector"
      instance_type = "ml.g5.xlarge"
      min_replicas  = 2
      max_replicas  = 10
      endpoint_type = "sagemaker"
      environment   = "production"
    }
```

### Auto-Provisioning Patterns

```yaml
Auto_Provisioning:
  scale_to_zero:
    description: "GPU nodes scale to 0 when not in use (massive cost savings)"
    implementation:
      - "Training cluster min_size = 0"
      - "Job submission triggers scale-up (Karpenter or Cluster Autoscaler)"
      - "After job completes, nodes drain and scale down"
      - "Savings: 80%+ for sporadic training workloads"
    tools: "Karpenter (AWS), GKE Autopilot, AKS cluster autoscaler"
    
  spot_instance_management:
    description: "Use preemptible/spot instances for training (60-90% savings)"
    implementation:
      - "Training jobs request spot instances"
      - "Checkpointing every 30 min (survive preemption)"
      - "Automatic failover to on-demand if spot unavailable"
      - "Mixed instance groups (some spot, some on-demand for reliability)"
    terraform: |
      resource "aws_eks_node_group" "gpu_spot" {
        capacity_type = "SPOT"
        
        instance_types = [
          "p4d.24xlarge",  # Primary preference
          "p4de.24xlarge", # Fallback
        ]
      }
      
  environment_parity:
    description: "Identical infrastructure for dev/staging/prod (different scale)"
    implementation:
      - "Same Terraform modules for all environments"
      - "Variables file per environment (dev.tfvars, prod.tfvars)"
      - "Dev: min_nodes=0, max_nodes=2, spot_only"
      - "Prod: min_nodes=2, max_nodes=20, mixed spot/on-demand"
    benefit: "Bugs caught in dev/staging, not production"
```

---

## How It Works in Practice

### IaC-Managed ML Platform

```yaml
Example:
  organization: "ML platform team managing infrastructure for 5 ML teams"
  tool: "Terraform + Kubernetes"
  
  infrastructure_layout:
    shared_platform:
      mlflow: "ECS service + RDS + S3 (module: mlflow)"
      feature_store: "Redis cluster + BigQuery (module: feature-store)"
      monitoring: "Prometheus + Grafana + Evidently (module: monitoring)"
      networking: "VPC, subnets, security groups, VPN"
      
    per_team_resources:
      fraud_team:
        training: "GPU node group (A100, spot, 0-4 nodes)"
        serving: "KServe deployment (L4, 2-10 replicas)"
        storage: "S3 bucket for model artifacts"
        
      recommendation_team:
        training: "GPU node group (H100, spot, 0-8 nodes)"
        serving: "KServe deployment (A100, 4-20 replicas)"
        storage: "S3 bucket + vector DB (Qdrant)"
        
  workflow:
    change_process:
      - "Engineer modifies Terraform code in feature branch"
      - "PR opened → terraform plan runs automatically (CI)"
      - "Team reviews plan output (what will change)"
      - "Merge → terraform apply runs (CD)"
      - "Infrastructure updated, state tracked"
      
    new_model_deployment:
      - "ML engineer requests new endpoint (fills template)"
      - "PR adds new module instance to team's infrastructure"
      - "Review → merge → endpoint provisioned automatically"
      - "DNS, monitoring, IAM all configured by module"
      
  benefits_realized:
    - "New environment spin-up: 15 minutes (was 2 days manual)"
    - "Infrastructure changes reviewed like code (catch misconfigurations)"
    - "Complete disaster recovery possible (recreate from code)"
    - "Cost visibility (infrastructure tied to teams via tags)"
    - "Security compliance (IAM policies defined in code, auditable)"
```

---

## Interview Tip

> When asked about ML infrastructure management: "I use Infrastructure as Code (Terraform or Pulumi) for all ML infrastructure — GPU clusters, model serving endpoints, feature store backends, experiment tracking servers, and monitoring. Key practices: (1) Reusable modules for common patterns (GPU cluster module, model serving module, feature store module). (2) Environment parity — same modules for dev/staging/prod, different scale via variables. (3) Scale-to-zero for GPU training nodes (cost savings: 80%+ for sporadic workloads). (4) Spot instance management with checkpointing for fault tolerance. (5) GitOps workflow — infrastructure changes via PR review, automated plan/apply. The biggest win is reproducibility: if a production cluster has issues, we can recreate it identically from code. For ML specifically, I manage GPU node groups with taints/labels for workload isolation, and use Karpenter for right-size provisioning (matches GPU type to job requirements)."

---

## Common Mistakes

1. **Console clicking for ML infrastructure** — Provisioning GPU clusters, SageMaker endpoints, or Redis caches through the cloud console. Undocumented, unreproducible, and dangerous for security (forgotten open ports, overly permissive IAM).

2. **Not using modules** — Copy-pasting Terraform code for each team/model/environment instead of abstracting into reusable modules. When a security patch is needed, you update 15 copies instead of one module.

3. **Forgetting scale-to-zero** — GPU instances running 24/7 for training jobs that run 4 hours per week. Proper autoscaling with min_size=0 saves thousands per month per cluster.

4. **No state management** — Running terraform apply without remote state, or with state file on a developer's laptop. State drift, conflicts, and potential infrastructure corruption.

5. **Infrastructure without monitoring integration** — Provisioning GPU nodes without corresponding monitoring, alerting, and cost tracking. IaC should provision the observability stack alongside the compute.

---

## Key Takeaways

- IaC (Terraform/Pulumi) for ML: version-controlled, reviewed, reproducible infrastructure
- Key ML resources to manage: GPU clusters, model endpoints, feature stores, MLflow, monitoring
- Reusable modules: package common patterns (GPU cluster, model serving, feature store) as modules
- Scale-to-zero: GPU training nodes should scale to 0 when idle (80%+ cost savings)
- Spot instances: 60-90% savings for training with checkpointing for fault tolerance
- Environment parity: same code for dev/staging/prod, different scale via variables
- GitOps workflow: infrastructure changes via PR → review → plan → apply
- Auto-provisioning: Karpenter/autoscaler matches GPU type to job requirements automatically
- Security: IAM policies, encryption, networking defined in code — auditable and consistent
- Disaster recovery: entire ML platform reproducible from IaC code in hours
