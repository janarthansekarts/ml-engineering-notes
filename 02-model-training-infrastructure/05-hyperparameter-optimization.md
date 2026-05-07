# Hyperparameter Optimization

## The Problem / Why This Matters

Every ML (Machine Learning) model has hyperparameters — settings that aren't learned from data but must be chosen by the engineer: learning rate, batch size, number of layers, dropout rate, weight decay, warmup steps, and dozens more. The right hyperparameters can mean the difference between a model that converges in hours with excellent performance and one that diverges, trains for days, or plateaus at mediocre accuracy. Manual hyperparameter tuning is slow, biased (humans repeat what worked before), and doesn't scale. HPO (Hyperparameter Optimization) automates this search: systematically exploring the hyperparameter space to find configurations that maximize model performance while minimizing wasted compute. In 2026, with GPU time costing $2-6/hour per GPU, and training runs taking hours to days, efficient HPO directly saves thousands to millions of dollars. Understanding when to use grid search vs Bayesian optimization vs population-based training — and how to configure trials efficiently — is essential for ML engineers who want to squeeze maximum performance from their training budget.

---

## The Analogy

Think of hyperparameter optimization like finding the best recipe for a cake:

- **Grid Search** = Try every combination from a recipe book: 5 sugar amounts × 5 butter amounts × 5 temperatures = 125 cakes baked. Thorough but incredibly wasteful (most combinations are obviously bad).
- **Random Search** = Randomly pick combinations and bake. Surprisingly effective — often finds good recipes faster than grid search because you sample the full range of each ingredient.
- **Bayesian Optimization** = A smart chef who tastes each cake, builds a mental model of "what makes a good cake," and uses that to decide the next recipe to try. Each experiment is informed by all previous results.
- **Population-Based Training** = A cooking competition. Start 20 chefs with random recipes. Every hour, check who's making the best cake. The worst chefs copy the best chef's recipe and slightly modify it. Over time, the group converges on excellence.

---

## Deep Dive

### Hyperparameter Search Strategies

```yaml
Search_Strategies:
  grid_search:
    how: "Evaluate ALL combinations of predefined hyperparameter values"
    example:
      learning_rate: [0.001, 0.01, 0.1]
      batch_size: [16, 32, 64]
      total_trials: "3 × 3 = 9 trials"
      
    pros:
      - "Complete coverage of defined grid"
      - "Simple to implement and understand"
      - "Reproducible (deterministic)"
    cons:
      - "Exponential explosion: 5 params × 5 values each = 3,125 trials"
      - "Wastes compute on unimportant dimensions"
      - "Assumes you know good value ranges upfront"
      - "Misses good values between grid points"
    when_to_use:
      - "Very few hyperparameters (1-2)"
      - "Small search space"
      - "Need complete coverage for publication/reporting"
      
  random_search:
    how: "Sample hyperparameters randomly from defined distributions"
    example:
      learning_rate: "log_uniform(1e-5, 1e-1)"
      batch_size: "choice([16, 32, 64, 128])"
      weight_decay: "uniform(0.0, 0.3)"
      total_trials: "50 random samples"
      
    pros:
      - "Explores full range of each parameter"
      - "More efficient than grid (finds good regions faster)"
      - "Easy to parallelize (trials are independent)"
      - "Works well in high dimensions"
    cons:
      - "No learning from previous trials (each trial is independent)"
      - "May waste trials in poor regions"
    when_to_use:
      - "Initial exploration (don't know good ranges)"
      - "High-dimensional search space"
      - "Embarrassingly parallel compute available"
    key_insight: "Random search is often 2-10x more efficient than grid search because most hyperparameters have a few that matter and many that don't"
    
  bayesian_optimization:
    how: "Build a probabilistic model of objective → hyperparameters, use it to choose next trial"
    algorithms:
      tpe:
        name: "Tree-structured Parzen Estimators"
        how: "Model P(hyperparameters|good) and P(hyperparameters|bad) separately"
        tool: "Optuna (default algorithm)"
      gaussian_process:
        name: "Gaussian Process (GP)"
        how: "Fit GP surrogate to observed (hyperparameters, objective) pairs"
        tool: "BoTorch, scikit-optimize"
      random_forest:
        name: "SMAC (Sequential Model-based Algorithm Configuration)"
        how: "Use random forest as surrogate model"
        tool: "SMAC3"
        
    pros:
      - "Sample-efficient (finds good configs in fewer trials)"
      - "Learns from previous trials (intelligent exploration)"
      - "Handles noisy objectives well"
      - "Can model parameter interactions"
    cons:
      - "Sequential by nature (each trial depends on previous)"
      - "Parallelization is possible but less straightforward"
      - "Surrogate model has overhead"
      - "Can get stuck in local optima"
    when_to_use:
      - "Expensive trials (hours-days per evaluation)"
      - "Limited budget (can only run 20-100 trials)"
      - "Medium-dimensional space (5-20 hyperparameters)"
```

### Optuna (State of the Art HPO)

```python
# Optuna - Modern hyperparameter optimization
import optuna
from optuna.integration import PyTorchLightningPruningCallback

def objective(trial):
    """Define one training trial with Optuna-suggested hyperparameters."""
    
    # Suggest hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    num_layers = trial.suggest_int("num_hidden_layers", 2, 6)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    
    # Build and train model with these hyperparameters
    model = build_model(
        num_layers=num_layers,
        dropout=dropout
    )
    
    trainer_args = {
        "learning_rate": lr,
        "per_device_train_batch_size": batch_size,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "num_train_epochs": 3,
    }
    
    # Train and evaluate
    metrics = train_and_evaluate(model, trainer_args)
    
    # Report intermediate values for pruning
    for epoch, val_loss in enumerate(metrics["val_losses"]):
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return metrics["final_eval_score"]

# Create study with Bayesian optimization (TPE)
study = optuna.create_study(
    direction="maximize",  # maximize eval score
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,     # No pruning for first 5 trials
        n_warmup_steps=1,       # No pruning before epoch 1
        interval_steps=1        # Check for pruning every epoch
    ),
    storage="sqlite:///hpo_study.db",  # Persistent storage
    study_name="llm_finetuning_v1"
)

# Run optimization
study.optimize(
    objective,
    n_trials=100,
    n_jobs=4,  # 4 parallel trials
    timeout=3600 * 24  # 24 hour budget
)

# Results
print(f"Best trial: {study.best_trial.params}")
print(f"Best value: {study.best_value}")

# Analyze parameter importance
importance = optuna.importance.get_param_importances(study)
print(f"Parameter importance: {importance}")
```

```yaml
Optuna_Features:
  key_capabilities:
    define_by_run: "Search space defined inside objective function (dynamic, conditional)"
    pruning: "Stop unpromising trials early (MedianPruner, HyperbandPruner)"
    multi_objective: "Optimize multiple objectives simultaneously (Pareto front)"
    distributed: "Multiple workers across machines (shared database)"
    visualization: "Built-in plots (parameter importance, optimization history, parallel coordinates)"
    
  advanced_features:
    conditional_params: "Parameters that depend on other parameters"
    integration: "PyTorch Lightning, HuggingFace, XGBoost, LightGBM callbacks"
    resumable: "Studies stored in database, can resume after interruption"
    custom_samplers: "Implement custom search algorithms"
```

### Ray Tune (Distributed HPO)

```python
# Ray Tune - Scalable distributed hyperparameter optimization
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search.optuna import OptunaSearch

# Define search space
search_space = {
    "learning_rate": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "weight_decay": tune.uniform(0.0, 0.3),
    "warmup_steps": tune.randint(0, 1000),
    "num_layers": tune.randint(2, 8),
}

# ASHA Scheduler (Asynchronous Successive Halving)
# Aggressively prunes poor trials, allocating budget to promising ones
scheduler = ASHAScheduler(
    metric="eval_accuracy",
    mode="max",
    max_t=10,           # Maximum epochs
    grace_period=2,     # Minimum epochs before pruning
    reduction_factor=3  # Keep top 1/3 at each rung
)

# Run distributed HPO
analysis = tune.run(
    train_function,       # Your training function
    config=search_space,
    num_samples=100,      # Total trials
    scheduler=scheduler,
    search_alg=OptunaSearch(),  # Use Optuna's TPE as the search algorithm
    resources_per_trial={"gpu": 1, "cpu": 4},
    max_concurrent_trials=8,
    local_dir="/results/ray_tune",
    name="llm_hpo_experiment"
)

print(f"Best config: {analysis.best_config}")
print(f"Best result: {analysis.best_result}")
```

### Population-Based Training (PBT)

```yaml
Population_Based_Training:
  concept:
    what: "Evolutionary approach — maintain a population of training runs, periodically exploit+explore"
    how:
      - "Start N training runs (population) with different hyperparameters"
      - "Train all in parallel for some interval"
      - "At each interval: rank by performance"
      - "Bottom performers EXPLOIT: copy weights+hyperparameters from a top performer"
      - "Then EXPLORE: randomly perturb the copied hyperparameters"
      - "Continue training with new hyperparameters (from copied weights)"
    key_insight: "Hyperparameter SCHEDULE is optimized, not just static values"
    
  advantages:
    - "Discovers hyperparameter schedules (not just fixed values)"
    - "Example: high LR initially, then decay — PBT discovers this automatically"
    - "Efficient — poor runs are recycled (not wasted)"
    - "Naturally parallel (all population members train simultaneously)"
    
  when_to_use:
    - "When hyperparameter schedules matter (learning rate, data augmentation strength)"
    - "Long training runs where the optimal value changes over time"
    - "Enough parallel compute to run population (8-64 members)"
    
  ray_tune_pbt:
    scheduler: "PopulationBasedTraining"
    config:
      time_attr: "training_iteration"
      perturbation_interval: 5  # Every 5 epochs
      hyperparam_mutations:
        learning_rate: [1e-5, 1e-4, 1e-3, 1e-2]
        weight_decay: "lambda: random.uniform(0.0, 0.3)"
```

### Early Stopping and Trial Pruning

```yaml
Early_Stopping:
  why: "80% of trials are clearly bad within the first few epochs — stop them early, save compute"
  
  strategies:
    median_pruning:
      how: "Prune trial if its intermediate result is below median of all previous trials at same step"
      aggressive: "Cuts ~50% of trials early"
      
    successive_halving:
      name: "ASHA (Asynchronous Successive Halving Algorithm)"
      how:
        - "Start many trials with small budget (few epochs)"
        - "After grace period, keep only top 1/N (N = reduction factor)"
        - "Give survivors more budget, repeat"
      example:
        start: "81 trials, 1 epoch each"
        round_1: "Keep top 27 (run 3 more epochs)"
        round_2: "Keep top 9 (run 9 more epochs)"
        round_3: "Keep top 3 (run 27 more epochs)"
        round_4: "Keep top 1 (run 81 epochs)"
        total_budget: "Equivalent to ~5 full trials but explored 81 configurations"
        
    hyperband:
      how: "Run multiple brackets of successive halving with different starting budgets"
      advantage: "Robust to the trade-off between many cheap trials vs few expensive ones"
      
  compute_savings:
    without_pruning: "100 trials × 10 epochs = 1000 epochs of compute"
    with_asha: "Same exploration but ~200 epochs total (5x reduction)"
    principle: "Explore broadly (many configs) then exploit deeply (best configs)"
```

---

## How It Works in Practice

### HPO Strategy for Different Scenarios

```yaml
Practical_Strategies:
  fine_tuning_llm:
    critical_params:
      - "learning_rate (most impactful — usually 1e-5 to 5e-4)"
      - "num_epochs (2-5 for fine-tuning)"
      - "lora_rank (if using LoRA: 8, 16, 32, 64)"
      - "lora_alpha (usually 2× rank)"
    less_critical:
      - "weight_decay (0.0-0.1)"
      - "warmup_ratio (0.03-0.1)"
    strategy: "20-30 trials with Optuna TPE, ASHA pruning"
    budget: "~$200-500 in GPU cost"
    
  training_from_scratch:
    critical_params:
      - "learning_rate + schedule (warmup + cosine decay)"
      - "batch_size (larger often better, up to memory limit)"
      - "weight_decay"
      - "model architecture (layers, hidden size, heads)"
    strategy: "Start with random search (20 trials), then Bayesian (30 trials focused)"
    budget: "Model-dependent ($500-50K depending on size)"
    
  xgboost_tabular:
    critical_params:
      - "max_depth, learning_rate, n_estimators"
      - "subsample, colsample_bytree"
      - "min_child_weight, reg_alpha, reg_lambda"
    strategy: "100-200 trials with Optuna (cheap per trial)"
    budget: "Minutes-hours on CPU ($0-10)"
    
  quick_and_dirty:
    approach: "Use published hyperparameters from similar paper/tutorial"
    when: "Prototyping, time pressure, model isn't the focus"
    then: "Only tune learning rate (5 values: 1e-5, 5e-5, 1e-4, 5e-4, 1e-3)"
```

---

## Interview Tip

> When asked about hyperparameter optimization: "My approach depends on compute budget and trial cost. For cheap trials (XGBoost, small models): I use Optuna with TPE and aggressive pruning — 100-200 trials, each taking minutes. For expensive trials (LLM fine-tuning): I first narrow the search space based on known good ranges from the literature, then run 20-30 trials with Bayesian optimization (Optuna TPE) and ASHA scheduling to prune bad runs early. For very expensive training (full model training): I use Population-Based Training to discover hyperparameter schedules — it's especially valuable because optimal learning rate changes during training. Key insight: learning rate is almost always the most important hyperparameter — if I can only tune one thing, it's learning rate. I also measure parameter importance after HPO to understand which parameters actually mattered, informing future experiments. Tools: Optuna for single-node, Ray Tune for distributed multi-GPU HPO."

---

## Common Mistakes

1. **Grid search in high dimensions** — Using grid search with 7+ hyperparameters. 5 values each = 78,125 trials. Random search with 50-100 trials typically finds equivalent or better configurations because it covers more of each parameter's range.

2. **Not setting a compute budget** — Running HPO indefinitely "until it converges." Set a hard budget (number of trials or GPU-hours) upfront. The law of diminishing returns applies heavily — 90% of the gain comes from the first 30% of trials.

3. **Tuning everything at once** — Including 20+ hyperparameters in the search space. Most hyperparameters have minimal impact. Start by identifying the 3-5 most impactful parameters (sensitivity analysis), then optimize only those.

4. **No early stopping** — Running every trial to full completion. With ASHA or median pruning, you can explore 5-10x more configurations in the same budget by cutting bad trials early.

5. **Forgetting to fix the seed** — HPO results aren't reproducible because training has randomness (data shuffling, dropout). Either fix seeds during HPO (for reproducibility) or run the best config multiple times (for statistical confidence).

---

## Key Takeaways

- HPO automates finding optimal hyperparameters — learning rate is almost always the most important one
- Grid search: exhaustive but exponentially expensive — use only for 1-2 parameters
- Random search: surprisingly effective — covers parameter ranges better than grid in high dimensions
- Bayesian optimization (TPE): sample-efficient — learns from previous trials, best for expensive experiments
- Population-Based Training: discovers hyperparameter SCHEDULES — optimal for long training runs
- ASHA pruning: stop bad trials early — explore 5-10x more configurations in the same budget
- Optuna: best single-machine HPO (TPE, pruning, multi-objective, visualizations)
- Ray Tune: distributed HPO across GPU cluster (integrates Optuna, ASHA, PBT)
- Always set a compute budget — diminishing returns after 20-50 trials for most problems
- Focus on important parameters: learning rate > batch size > regularization > architecture
