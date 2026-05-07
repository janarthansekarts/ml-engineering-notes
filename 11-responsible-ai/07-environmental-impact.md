# Environmental Impact

## The Problem / Why This Matters

Training a single large language model can emit as much carbon as five cars over their entire lifetimes. GPT-4 training is estimated at 50+ GWh of energy. A single H100 GPU consumes 700W under load — a cluster of 10,000 GPUs for a training run consumes 7MW continuously for weeks or months. In 2026, global AI compute demand is doubling every 6-9 months while the world tries to reduce emissions. This isn't just an ethical concern — it's a cost problem (energy is the largest operational cost for training), a regulatory problem (EU AI Act requires energy consumption reporting for GPAI models), and a competitive problem (efficient training is faster and cheaper). ML engineers must understand and minimize the environmental footprint of their work: choosing efficient architectures, optimizing training runs to avoid waste, selecting low-carbon compute regions, leveraging techniques like mixed-precision training, gradient checkpointing, model distillation, and progressive training. The goal isn't to stop training models — it's to get the same (or better) results with dramatically less energy through engineering discipline.

---

## The Analogy

Think of ML environmental impact like water usage in agriculture:

- **Naive training** = Flood irrigation. Dump water (compute) everywhere. Most is wasted, but crops (model) eventually grow. Traditional full-precision training on oversized clusters with poor utilization.
- **Efficient training** = Drip irrigation. Deliver exactly the right amount of water (compute) to exactly where it's needed. Mixed precision, efficient data loading, optimal batch sizes, and right-sized infrastructure.
- **Green AI** = Growing drought-resistant crops. Choose architectures that inherently need less water (compute): smaller models, sparse architectures, distilled models. Change what you grow, not just how you irrigate.

---

## Deep Dive

### Carbon Footprint of AI

```yaml
AI_Carbon_Footprint:
  training_examples:
    gpt_3_175b:
      energy: "1,287 MWh"
      co2: "552 tonnes CO2e (using US average grid)"
      equivalent: "~120 cars driving for a year"
      
    gpt_4:
      energy: "~50,000 MWh (estimated)"
      co2: "~12,000 tonnes CO2e (estimated)"
      equivalent: "~2,500 cars driving for a year"
      
    llama_3_405b:
      energy: "~30,000 MWh (estimated from Meta's reporting)"
      co2: "Significantly lower (Meta uses renewable energy)"
      
    stable_diffusion:
      energy: "~150 MWh"
      co2: "~15 tonnes CO2e"
      
  inference_vs_training:
    insight: "Inference total emissions often EXCEED training (deployed at scale)"
    example: "ChatGPT serving millions of users: inference energy > training energy within weeks"
    per_query: "~0.001-0.01 kWh per LLM query (varies by model size and response length)"
    
  breakdown:
    computation: "60-70% of energy (GPU/TPU operations)"
    memory_access: "20-30% (DRAM reads/writes — often overlooked)"
    networking: "5-10% (distributed training communication)"
    cooling: "30-50% overhead (PUE — Power Usage Effectiveness)"
    
  pue:
    what: "PUE (Power Usage Effectiveness) = Total facility energy / IT equipment energy"
    ideal: "1.0 (impossible — all energy goes to compute)"
    average_datacenter: "1.57"
    google_average: "1.10"
    hyperscaler_best: "1.05-1.08"
    implication: "PUE 1.5 means 50% energy wasted on cooling/overhead"
```

### Measuring ML Carbon Footprint

```yaml
Measuring_Carbon:
  formula: |
    CO2e = Energy (kWh) × Carbon Intensity (kgCO2/kWh) × PUE
    
    Energy = GPU_Power (kW) × GPU_Count × Training_Hours × Utilization
    
  variables:
    gpu_power:
      h100_tdp: "700W per GPU"
      a100_tdp: "400W per GPU"
      h200_tdp: "700W per GPU"
      b200_tdp: "1000W per GPU"
      tpu_v5e: "~200W per chip"
      
    carbon_intensity:
      varies_by_region: true
      examples:
        iceland: "0.028 kgCO2/kWh (geothermal/hydro)"
        france: "0.052 kgCO2/kWh (nuclear)"
        norway: "0.029 kgCO2/kWh (hydro)"
        canada_quebec: "0.002 kgCO2/kWh (hydro)"
        us_average: "0.370 kgCO2/kWh"
        us_west: "0.230 kgCO2/kWh"
        germany: "0.350 kgCO2/kWh"
        india: "0.720 kgCO2/kWh"
        
    implication: "Same training job: 100 tonnes CO2 in India vs. 4 tonnes in Quebec"
    
  tools:
    codecarbon: "Python library — measure energy/carbon of code execution"
    mlco2: "ML CO2 Impact Calculator (web tool)"
    carbontracker: "Real-time carbon tracking during training (PyTorch)"
    cloud_carbon_footprint: "Cloud provider carbon reporting (AWS, GCP, Azure)"
    experiment_impact_tracker: "Academic research carbon tracking"
```

### Efficiency Techniques

```yaml
Efficiency_Techniques:
  training_efficiency:
    mixed_precision:
      what: "Use FP16/BF16 for most operations, FP32 for critical accumulations"
      savings: "~50% memory, ~2x throughput, negligible quality loss"
      implementation: "torch.cuda.amp, BF16 on H100/TPUv4+"
      energy_savings: "40-50% per training step"
      
    gradient_checkpointing:
      what: "Recompute activations during backward pass instead of storing"
      savings: "~60-70% memory reduction"
      cost: "~30% more compute (recomputation)"
      when: "When memory-bound (allows larger batch, compensates compute cost)"
      
    efficient_data_loading:
      what: "Avoid GPU idle time waiting for data"
      techniques: "Pre-fetching, memory-mapped datasets, WebDataset, Mosaic StreamingDataset"
      impact: "5-20% training speedup (prevents GPU starvation)"
      
    optimal_batch_size:
      what: "Find batch size that maximizes GPU utilization"
      approach: "Use largest batch that fits in memory with gradient accumulation"
      sweet_spot: "Batch size where GPU compute utilization > 90%"
      
    learning_rate_scheduling:
      what: "Cosine schedule, warmup, reduces wasted early/late steps"
      impact: "10-30% fewer total steps to converge"
      
    early_stopping:
      what: "Stop training when validation metrics plateau"
      impact: "Prevents 10-40% wasted compute on over-training"
      
  architecture_efficiency:
    sparse_models:
      moe: "Mixture of Experts — only activate subset of parameters per token"
      savings: "Use 8x parameters but only compute through 1x per token"
      examples: "Mixtral 8x7B, GPT-4 (rumored MoE), Switch Transformer"
      
    efficient_attention:
      flash_attention: "IO-aware exact attention — 2-4x faster than standard"
      sliding_window: "Mistral's approach — local attention for most layers"
      linear_attention: "Approximate attention in O(n) instead of O(n²)"
      
    pruning:
      what: "Remove unnecessary weights/neurons (sparse model)"
      structured: "Remove entire neurons/layers (hardware-friendly sparsity)"
      unstructured: "Remove individual weights (higher compression, harder to accelerate)"
      savings: "50-90% parameter reduction with <5% quality loss"
      
    distillation:
      what: "Train small model to mimic large model"
      savings: "10-100x smaller model, 10-100x less inference energy"
      examples: "DistilBERT (60% size, 97% performance), Llama 3.2 (distilled from 405B)"
      
  infrastructure_efficiency:
    region_selection:
      strategy: "Train in low-carbon regions (Quebec, Iceland, Norway, France)"
      cloud_tools: "GCP carbon-free energy score, AWS sustainability region data"
      savings: "50-95% carbon reduction just by choosing the right region"
      
    time_shifting:
      what: "Schedule training during low-carbon-intensity hours"
      approach: "Grid carbon intensity varies by time (more solar midday, more wind at night)"
      tools: "Electricity Maps API, WattTime, carbon-aware schedulers"
      savings: "20-40% carbon reduction in many regions"
      
    right_sizing:
      what: "Don't use more GPUs than needed"
      problem: "Teams request 64 GPUs, use 16 effectively"
      solution: "Auto-scaling, utilization monitoring, cluster scheduling policies"
      
    spot_instances:
      what: "Use preemptible/spot GPUs (cheaper, same carbon)"
      savings: "60-90% cost reduction"
      requirement: "Checkpointing to handle preemption"
```

### Implementation

```python
# Environmental impact measurement and optimization

"""
Carbon footprint tracking and efficiency optimization for ML training.
Measures energy consumption, estimates CO2 emissions, and suggests optimizations.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class CarbonConfig:
    """Configuration for carbon tracking."""
    region: str = "us-east-1"
    carbon_intensity_kgco2_per_kwh: float = 0.370  # US average
    pue: float = 1.1  # Power Usage Effectiveness
    gpu_type: str = "H100"
    gpu_tdp_watts: float = 700.0
    gpu_count: int = 8
    include_embodied: bool = True  # Include manufacturing emissions


@dataclass
class EnergyReport:
    """Energy and carbon report for a training run."""
    run_id: str
    duration_hours: float
    energy_kwh: float
    co2_kg: float
    co2_equivalent: str  # Human-readable equivalent
    gpu_utilization_avg: float
    efficiency_score: float  # 0-1, how efficient vs. theoretical max
    recommendations: List[str] = field(default_factory=list)


class CarbonTracker:
    """
    Track energy consumption and carbon emissions of ML training.
    
    Integrates with training loop to measure actual GPU power draw
    and estimate CO2 emissions based on region and grid carbon intensity.
    """
    
    def __init__(self, config: CarbonConfig):
        self.config = config
        self.start_time: Optional[datetime] = None
        self.power_measurements: List[float] = []  # Watts
        self.utilization_measurements: List[float] = []  # 0-1
        
        # Carbon intensity by region (kgCO2/kWh)
        self.carbon_intensities = {
            "us-east-1": 0.370,
            "us-west-2": 0.230,
            "eu-west-1": 0.310,  # Ireland
            "eu-north-1": 0.029,  # Sweden
            "ca-central-1": 0.002,  # Quebec (hydro)
            "eu-west-3": 0.052,  # France (nuclear)
            "ap-south-1": 0.720,  # India
        }
    
    def start(self):
        """Start tracking energy consumption."""
        self.start_time = datetime.now()
        self.power_measurements = []
        self.utilization_measurements = []
        logger.info(f"Carbon tracking started (region: {self.config.region})")
    
    def record_step(self, gpu_power_watts: float, gpu_utilization: float):
        """Record power and utilization for one training step."""
        self.power_measurements.append(gpu_power_watts)
        self.utilization_measurements.append(gpu_utilization)
    
    def report(self, run_id: str = "default") -> EnergyReport:
        """Generate energy and carbon report."""
        if not self.start_time:
            raise ValueError("Tracking not started. Call start() first.")
        
        duration = datetime.now() - self.start_time
        duration_hours = duration.total_seconds() / 3600
        
        # Calculate energy
        avg_power = sum(self.power_measurements) / max(len(self.power_measurements), 1)
        total_power_kw = (avg_power * self.config.gpu_count) / 1000
        energy_kwh = total_power_kw * duration_hours * self.config.pue
        
        # Calculate CO2
        carbon_intensity = self.carbon_intensities.get(
            self.config.region, self.config.carbon_intensity_kgco2_per_kwh
        )
        co2_kg = energy_kwh * carbon_intensity
        
        # Add embodied carbon (manufacturing of hardware)
        if self.config.include_embodied:
            # H100: ~150kg CO2 embodied, 5-year lifespan = ~0.0034 kg/hour per GPU
            embodied_per_hour = 0.0034 * self.config.gpu_count
            co2_kg += embodied_per_hour * duration_hours
        
        # GPU utilization average
        avg_utilization = sum(self.utilization_measurements) / max(
            len(self.utilization_measurements), 1
        )
        
        # Efficiency score (actual useful compute / theoretical max)
        efficiency = avg_utilization * 0.8  # 80% at full utilization is excellent
        
        # Human-readable equivalent
        equivalent = self._co2_equivalent(co2_kg)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            avg_utilization, energy_kwh, duration_hours
        )
        
        return EnergyReport(
            run_id=run_id,
            duration_hours=duration_hours,
            energy_kwh=energy_kwh,
            co2_kg=co2_kg,
            co2_equivalent=equivalent,
            gpu_utilization_avg=avg_utilization,
            efficiency_score=efficiency,
            recommendations=recommendations,
        )
    
    def _co2_equivalent(self, co2_kg: float) -> str:
        """Convert CO2 kg to human-readable equivalent."""
        if co2_kg < 1:
            return f"{co2_kg*1000:.0f}g CO2 (≈ charging a phone {co2_kg*1000/0.008:.0f} times)"
        elif co2_kg < 100:
            return f"{co2_kg:.1f}kg CO2 (≈ {co2_kg/0.21:.0f} km driving a car)"
        elif co2_kg < 10000:
            return f"{co2_kg:.0f}kg CO2 (≈ {co2_kg/4600:.1f} transatlantic flights)"
        else:
            return f"{co2_kg/1000:.1f} tonnes CO2 (≈ {co2_kg/4600:.0f} cars for a year)"
    
    def _generate_recommendations(
        self, utilization: float, energy: float, hours: float
    ) -> List[str]:
        """Generate efficiency recommendations based on tracking data."""
        recommendations = []
        
        if utilization < 0.5:
            recommendations.append(
                f"GPU utilization is {utilization*100:.0f}%. "
                "Consider: reducing GPU count, increasing batch size, "
                "or improving data loading pipeline."
            )
        
        if self.config.region in ["us-east-1", "ap-south-1"]:
            low_carbon = "ca-central-1" if "us" in self.config.region else "eu-north-1"
            savings = 1 - (
                self.carbon_intensities[low_carbon] / 
                self.carbon_intensities[self.config.region]
            )
            recommendations.append(
                f"Region {self.config.region} has high carbon intensity. "
                f"Moving to {low_carbon} would reduce emissions by {savings*100:.0f}%."
            )
        
        if energy > 100:  # >100 kWh
            recommendations.append(
                "Consider mixed-precision training (BF16) if not already used. "
                "Typically reduces energy by 40-50%."
            )
        
        return recommendations


class GreenTrainingOptimizer:
    """
    Optimize training for minimal environmental impact.
    
    Strategies:
    1. Region selection (lowest carbon intensity)
    2. Time shifting (train during low-carbon hours)
    3. Architecture efficiency (smaller, sparse, distilled)
    4. Training efficiency (mixed precision, early stopping)
    """
    
    def recommend_region(
        self,
        required_gpu_type: str = "H100",
        max_latency_ms: int = 100,
        budget_per_hour: float = None,
    ) -> List[Dict]:
        """
        Recommend training region for lowest carbon footprint.
        
        Considers: carbon intensity, GPU availability, cost, latency.
        """
        regions = [
            {"region": "ca-central-1", "carbon": 0.002, "cost_multiplier": 1.0},
            {"region": "eu-north-1", "carbon": 0.029, "cost_multiplier": 1.1},
            {"region": "eu-west-3", "carbon": 0.052, "cost_multiplier": 1.05},
            {"region": "us-west-2", "carbon": 0.230, "cost_multiplier": 1.0},
            {"region": "us-east-1", "carbon": 0.370, "cost_multiplier": 0.95},
        ]
        
        # Sort by carbon intensity
        return sorted(regions, key=lambda r: r["carbon"])
    
    def estimate_savings_mixed_precision(
        self, current_precision: str, model_size_params: int
    ) -> Dict:
        """Estimate savings from switching to mixed precision."""
        if current_precision == "fp32":
            return {
                "recommended": "bf16 (mixed precision)",
                "memory_savings": "~50%",
                "throughput_improvement": "~2x",
                "energy_savings": "~40-50%",
                "quality_impact": "Negligible for most models (< 0.1% metric change)",
            }
        elif current_precision == "fp16":
            return {
                "recommended": "Already using FP16. Consider BF16 for stability.",
                "memory_savings": "0%",
                "energy_savings": "0%",
            }
        return {"recommended": "Already optimized"}
    
    def estimate_distillation_savings(
        self, teacher_params: int, student_params: int
    ) -> Dict:
        """Estimate inference energy savings from distillation."""
        compression_ratio = teacher_params / student_params
        return {
            "teacher_params": f"{teacher_params/1e9:.1f}B",
            "student_params": f"{student_params/1e9:.1f}B",
            "compression_ratio": f"{compression_ratio:.1f}x",
            "inference_energy_savings": f"~{(1 - 1/compression_ratio)*100:.0f}%",
            "inference_latency_improvement": f"~{compression_ratio:.1f}x faster",
            "quality_retention": "Typically 90-97% of teacher performance",
        }
```

---

## How It Works in Practice

### Making a Training Run Carbon-Efficient

```yaml
Carbon_Efficient_Training:
  scenario: "Training a 7B parameter model for specialized domain"
  baseline: "Standard training on us-east-1, FP32, 64 H100s, 2 weeks"
  
  optimizations_applied:
    1_region_selection:
      change: "us-east-1 → ca-central-1 (Quebec, 99.5% hydro)"
      carbon_reduction: "99.5% (0.370 → 0.002 kgCO2/kWh)"
      cost_impact: "Neutral (similar pricing)"
      
    2_mixed_precision:
      change: "FP32 → BF16 mixed precision"
      energy_reduction: "45%"
      training_time_reduction: "50% (2x throughput)"
      quality_impact: "<0.1% metric change"
      
    3_flash_attention:
      change: "Standard attention → FlashAttention-2"
      energy_reduction: "20% (fewer memory operations)"
      training_time_reduction: "25%"
      
    4_efficient_data_loading:
      change: "Standard DataLoader → Mosaic StreamingDataset"
      gpu_utilization_improvement: "75% → 92%"
      waste_reduction: "17% less idle GPU time"
      
    5_optimal_batch_size:
      change: "Batch 32 → Batch 2048 (with gradient accumulation)"
      gpu_utilization_improvement: "Better hardware saturation"
      
    6_right_sizing:
      change: "64 GPUs → 32 GPUs (better utilized)"
      waste_reduction: "50% fewer GPUs, same wall-clock time"
      
  results:
    baseline:
      energy: "12,000 kWh"
      co2: "4,440 kg CO2"
      training_time: "336 hours (2 weeks)"
      cost: "$150,000"
      
    optimized:
      energy: "3,200 kWh (73% reduction)"
      co2: "6.4 kg CO2 (99.8% reduction)"
      training_time: "168 hours (1 week)"
      cost: "$45,000 (70% reduction)"
      
  key_insight: "Region selection alone cut 99.5% of carbon. Efficiency techniques cut cost and time by 70%."
```

---

## Interview Tip

> When asked about environmental impact of AI: "I approach ML sustainability from three angles: measure, reduce, report. Measurement: I use CodeCarbon or similar libraries to track energy consumption per training run. The formula is straightforward: Energy = GPU_Power × GPU_Count × Hours × PUE. CO2 = Energy × Carbon_Intensity (varies 100x between regions: 0.002 in Quebec to 0.72 in India). Reduction strategies in priority order: (1) Region selection — training in Quebec vs. US East reduces carbon 99%+ at similar cost. Biggest single lever. (2) Mixed precision (BF16) — 40-50% energy reduction, 2x throughput, negligible quality impact. (3) Efficient architectures — FlashAttention for 20% speedup, Mixture of Experts for sparse computation. (4) Right-sizing — don't use 64 GPUs when 32 achieve similar throughput (better utilized). (5) Distillation for inference — deploy a 7B distilled model instead of 70B for 10x energy savings at serving time. Reporting: EU AI Act requires energy reporting for GPAI models. I track per-run metrics (kWh, CO2, GPU utilization) and aggregate organizational carbon from AI. The key insight: efficiency and sustainability are aligned. Mixed precision saves energy AND money AND time. Choosing the right region saves carbon AND often reduces latency. There's no accuracy-sustainability trade-off in most cases — it's pure engineering optimization."

---

## Common Mistakes

1. **Ignoring inference emissions** — Focusing only on training carbon while deploying models at scale. ChatGPT's inference emissions likely exceed training within weeks. Solution: track inference energy per query, optimize inference (quantization, distillation, caching), right-size serving infrastructure.

2. **Training in high-carbon regions by default** — Using us-east-1 because it's the default without considering alternatives. Quebec (99.5% hydro) has similar costs. Solution: default to low-carbon regions. Use cloud provider carbon dashboards to make informed choices.

3. **FP32 training when BF16 suffices** — Full-precision training wasting 50% of compute capacity. BF16 achieves identical results for most model types and sizes. Solution: always start with BF16 mixed precision. Only fall back to FP32 if you observe specific numerical stability issues.

4. **Low GPU utilization** — Running training with 40% GPU utilization (60% of energy wasted on idle GPUs). Solution: profile training to find bottlenecks (data loading, communication, memory). Increase batch size, improve data pipeline, or reduce GPU count.

5. **Not reporting carbon** — Training large models without tracking or reporting emissions. Creates inability to optimize what you don't measure, and non-compliance with emerging regulations. Solution: integrate CodeCarbon/CarbonTracker into training scripts, report per-experiment emissions, set organizational carbon budgets.

---

## Key Takeaways

- AI training carbon: large models emit hundreds of tonnes of CO2 (region-dependent)
- Formula: CO2 = Energy × Carbon_Intensity × PUE. Energy = Power × GPUs × Hours
- Biggest lever: region selection (0.002 vs. 0.72 kgCO2/kWh = 360x difference)
- Mixed precision (BF16): 40-50% energy savings, 2x throughput, no quality loss
- Inference > training: serving at scale may emit more than training — optimize both
- Efficiency = sustainability: faster training, lower cost, AND lower carbon are aligned
- Tools: CodeCarbon, CarbonTracker, cloud provider carbon dashboards
- Regulation: EU AI Act requires energy reporting for general-purpose AI models
- Distillation: 10-100x inference efficiency (deploy small models for serving)
- PUE matters: data center efficiency adds 10-50% overhead — choose efficient providers
