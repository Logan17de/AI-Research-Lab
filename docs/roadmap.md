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

## Phase 5 — Modern Backbone and Capability-Gap Validation

### Problem discovered

The tested Qwen “Base” checkpoint already showed assistant-like behavior and reasoning-shaped generation. It is not a perfectly clean raw-pretraining baseline.

### Preferred comparison

Use a model family with explicit checkpoint separation:

```text
Small Base
Official Small SFT
Small Base + MOD
Larger reference model
```

Candidate families include OLMo 2 and SmolLM2.

### Capability Gap Recovery

```text
CGR = (MOD score - small-base score)
      / (larger-reference score - small-base score)
```

Report CGR separately for:

- instruction following;
- factual completion;
- reasoning;
- code;
- multilingual behavior;
- general language modeling.

### Exit criteria

- Behaviorally clean starting checkpoint
- Official full-SFT reference
- Step-zero evaluation
- Stable single-turn and multi-turn training
- Repeated held-out and behavioral evaluation
- No advantage claimed from an incomplete epoch

## Phase 6 — Continual-Learning Retention

### Protocol

1. Adapt on domain A
2. Evaluate A using both completion and instruction formats
3. Add domain B while replaying accumulated A+B data
4. Re-evaluate A, B, general language modeling, and instruction behavior
5. Repeat with domain C
6. Compare full fine-tuning, LoRA, reusable MOD, and progressively frozen MODs

### Separate knowledge from chat behavior

Knowledge retention should include completion-style prompts and token metrics:

- correct-token probability;
- correct-token rank;
- top-1 and top-5 accuracy;
- original-corpus perplexity;
- change relative to the untouched base.

This prevents weak chat-format understanding from being misdiagnosed as factual forgetting.

### Metrics

- New-domain plasticity
- Old-domain stability
- Mixed unseen validation
- General-capability drift
- Recovery after re-exposure
- Trainable and active parameters
- Time and memory cost

### Exit criteria

The method must show a better stability–plasticity trade-off, not merely fast fitting.

## Phase 7 — Progressive Frozen Capacity

### Equal-capacity comparison

```text
Single reusable MOD, dim 32
versus
MOD 1/2/3/4, dim 8 each, trained and frozen progressively
```

All stages use accumulated-data replay.

### Architecture controls

Test separately:

1. Reusable MOD
2. Progressive frozen MODs
3. Width expansion
4. Depth expansion
5. Width + depth only after individual evidence

### Decision

If progressive 4×8 retains earlier tasks better than reusable 32 at similar cost, capacity isolation is contributing. If not, freezing modules mainly adds complexity.

## Phase 8 — Multi-MOD Routing

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

## Phase 9 — Governance and Reasoning Integration

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

## Phase 10 — Research Communication

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
