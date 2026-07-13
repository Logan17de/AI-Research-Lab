# Research Roadmap

The roadmap is organized around evidence gates. A later phase should not be treated as validated until the earlier phase produces reproducible results.

## Phase 1 — Reproducible Baselines

### Build

- Freeze dataset versions and evaluation splits
- Record exact tokenizer and control-token state
- Save all CLI arguments and model configurations
- Standardize random seeds and decoding settings
- Export machine-readable metrics

### Compare

- Full fine-tuning
- Frozen pretrained base
- LoRA at multiple ranks
- Individual MOD families
- Combined MOD configurations

### Exit criteria

- Runs reproduce within an acceptable variance
- Parameter counts and active parameters are verified
- Training/evaluation leakage is ruled out
- Wall-clock, throughput, and VRAM are reported

## Phase 2 — Capacity and Scaling

### Questions

- How should MOD dimension scale with dataset size?
- Does added capacity delay or eliminate plateaus?
- When does a larger MOD merely memorize?
- How does MOD size compare with a parameter-matched LoRA rank?
- Does sharing across layers help efficiency but limit specialization?

### Experiments

- Capacity sweeps under a fixed token budget
- Parameter-matched LoRA comparisons
- Shared versus layer-specific adaptation
- Learning curves rather than final perplexity alone
- Multiple random seeds

## Phase 3 — Modern Backbone Validation

### Target

Move beyond GPT-2 to a modern pretrained base model while retaining a small enough scale for controlled experimentation.

### Work

- Validate the Qwen/Tulu training pipeline
- Complete approximately one meaningful training pass
- Compare full fine-tuning and MOD under identical supervision
- Test general conversation, instruction following, arithmetic, and technical explanation
- Audit behavior on architecture components not covered by a modifier family

### Exit criteria

- Stable training
- Correct multi-turn stopping behavior
- No tokenizer/control-token mismatch
- Repeated evaluation on fixed probes and held-out data
- No architectural advantage claimed from an incomplete epoch

## Phase 4 — Continual-Learning Retention

### Protocol

1. Train or adapt on domain A
2. Evaluate domain A and a general benchmark
3. Adapt on domain B
4. Re-evaluate domain A, domain B, and the general benchmark
5. Compare forgetting across full fine-tuning, LoRA, and MOD

### Metrics

- New-domain learning
- Old-domain retention
- General-capability drift
- Recovery after re-exposure
- Update locality
- Trainable and active parameters
- Time and memory cost

### Exit criteria

The method must show a useful learning-retention tradeoff, not merely fast fitting.

## Phase 5 — Multi-MOD Routing

### Build

- Independent domain modules
- Frozen completed modules during later domain training
- Router capable of selecting one or more domains
- General module for cross-domain composition
- Explicit fallback when routing confidence is low

### Evaluate

- Single-domain accuracy
- Mixed-domain prompts
- Cross-domain composition
- Router confusion
- Old-module stability after adding a new module
- Compute cost of combining multiple modules

## Phase 6 — Governance and Reasoning Integration

### Integrate

- Update ownership and audit logs
- Drift detection
- Rollback and recovery
- Bounded working memory
- Evidence-grounded relational reasoning
- Confidence calibration and abstention
- Out-of-distribution evaluation

### Principle

Reasoning capability should increase only when memory updates remain stable, attributable, and recoverable.

## Phase 7 — Research Communication

### Deliverables

- Architecture description at a confidentiality-safe level
- Reproducible baseline code and configurations where appropriate
- Experiment tables with negative results included
- Ablation study
- Retention benchmark
- Technical report or preprint
- Compute-sponsorship summary with a concrete experiment budget

## Decision Rules

Continue investing in the approach only if it demonstrates at least one durable advantage:

- better retention at matched new-task quality;
- faster convergence at matched parameter and compute budgets;
- lower active compute for comparable performance;
- easier domain isolation and composition;
- safer rollback or removal of learned behavior.

If none of these survives controlled testing, the honest conclusion is that the modifier is extra capacity without a meaningful practical advantage. That outcome would still be scientifically useful.
