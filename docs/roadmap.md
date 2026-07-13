# Research Roadmap

The roadmap is organized around evidence gates. A later phase should not be treated as validated until the earlier phase produces reproducible results.

## Phase 1 — Correct and Reproducible Baselines

### Build

- Freeze dataset versions and evaluation splits
- Record exact tokenizer and control-token state
- Save all CLI arguments and model configurations
- Standardize random seeds and decoding settings
- Export machine-readable metrics
- Verify iterable-dataset worker sharding
- Use optimizer steps and supervised-token counts as primary budgets

### Current reference

The corrected GPT-2 Small UltraChat comparison at ~8,000 steps is:

| Variant | Eval PPL |
|---|---:|
| LoRA | 8.313 |
| LoRA + FFN-oriented MOD | 8.063 |
| Complete | 8.033 |

### Exit criteria

- Results reproduce across at least three seeds
- Parameter counts and active parameters are verified
- Training/evaluation leakage is ruled out
- Wall-clock, throughput, and VRAM are reported
- Epoch accounting agrees with actual data traversal

## Phase 2 — Critical LoRA Rank Test

This is the next decisive experiment.

### Compare

- LoRA r=8
- LoRA r=16
- LoRA r=32
- Optional higher rank if parameter matching requires it
- LoRA + FFN-oriented MOD
- Complete MOD system

### Hold fixed

- GPT-2 Small backbone
- UltraChat training subset and held-out set
- ~8,000 optimizer steps
- sequence length
- effective batch size
- optimizer and learning-rate schedule
- seed
- evaluation procedure
- decoding settings

### Measure

- Eval PPL
- Trainable and active parameters
- Tokens/s
- VRAM
- Wall-clock time
- Instruction-following score
- Long-range coherence
- EOS success
- Hallucination and topic drift

### Decision

If increased LoRA rank reaches the same quality with fewer parameters or lower cost, LoRA remains the better method. If the FFN-oriented MOD preserves an advantage under parameter and compute matching, the result becomes substantially stronger.

## Phase 3 — Capacity and Scaling

### Questions

- How should MOD dimension scale with dataset size?
- Does added capacity delay plateaus or merely memorize?
- Does sharing across layers help efficiency but limit specialization?
- How does vocabulary size affect storage cost?
- Does relative overhead improve as model depth grows?

### Experiments

- MOD-capacity sweeps under fixed token budgets
- Parameter-matched LoRA comparisons
- Shared versus layer-specific adaptation
- Learning curves, not only final PPL
- Multiple datasets and seeds

## Phase 4 — Behavioral Evaluation

### Automated prompt suites

- Exact item-count compliance
- Bullet/table/JSON formatting
- Professional rewriting
- Topic retention across long answers
- Short planning tasks
- Technical explanation
- Basic arithmetic
- EOS and continuation confidence

### Metrics

- Constraint adherence
- Semantic correctness
- Topic drift
- Long-range coherence
- Hallucination rate
- EOS success
- Human preference or blinded rating

### Principle

Perplexity remains a training-fit metric, not a complete assistant-quality metric.

## Phase 5 — Modern Backbone Validation

### Target

Move beyond GPT-2 to a modern pretrained base model while retaining a scale suitable for controlled experiments.

### Work

- Validate the Qwen/Tulu training pipeline
- Complete a meaningful training pass
- Compare full fine-tuning, LoRA, and MOD under identical supervision
- Test general conversation, instruction following, arithmetic, and technical explanation
- Audit unsupported architecture components

### Exit criteria

- Stable training
- Correct multi-turn stopping behavior
- No tokenizer/control-token mismatch
- Repeated evaluation on fixed probes and held-out data
- No advantage claimed from an incomplete epoch

## Phase 6 — Continual-Learning Retention

### Protocol

1. Adapt on domain A
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

## Phase 7 — Multi-MOD Routing

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
- Compute cost of combining modules

## Phase 8 — Governance and Reasoning Integration

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

## Phase 9 — Research Communication

### Deliverables

- Confidentiality-safe architecture description
- Reproducible baseline configurations where appropriate
- Experiment tables including negative results
- LoRA-rank and MOD ablations
- Retention benchmark
- Technical report or preprint
- Compute-sponsorship summary with a concrete experiment budget

## Decision Rules

Continue investing only if the method demonstrates at least one durable advantage:

- better retention at matched new-task quality;
- faster convergence at matched parameters and compute;
- lower active compute for comparable performance;
- easier domain isolation and composition;
- safer rollback or removal of learned behavior.

If none survives controlled testing, the honest conclusion is that the modifier is extra capacity without a meaningful practical advantage. That outcome would still be scientifically useful.
