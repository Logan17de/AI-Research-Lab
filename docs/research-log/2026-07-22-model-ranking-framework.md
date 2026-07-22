# 2026-07-22 — Model Ranking Framework

## Evidence Status

- **Locked UltraChat validation metrics:** VALID for the recorded checkpoints
- **Matched-token ranking:** VALID at step 200 / 22,094,998 processed tokens
- **ATE ranking:** PROVISIONAL because training is only recorded through step 200
- **Parameter-efficiency track:** CROSS-BENCHMARK INTERPRETATION until matched UltraChat MOD and LoRA baselines exist
- **Architectural-novelty track:** RESEARCH INTERPRETATION, not a numerical benchmark
- **Final model ranking:** PENDING untouched-test, free-generation, retention, identical-hardware, and multi-seed evaluation

This framework prevents results from different datasets and evaluation protocols from being collapsed into one misleading leaderboard. The COT, direct-answer V3, locked UltraChat, GPT-2, Qwen/Tülu, and Inverted Transformer result series remain separate.

## Track A — Absolute Quality on Locked UltraChat

All four runs use the same locked dataset, token order, sequence length 1,024, effective batch size 160, and evaluation schedule.

### Best recorded validation checkpoints

| Rank | Model | Best assistant PPL | Step | Status |
|---:|---|---:|---:|---|
| 1 | Pythia-2.8B full FT | **3.2402** | 1,450 | current benchmark |
| 2 | ATE h1/l1 | **3.5475** | 200 | provisional early result |
| 3 | MOD + plastic-24 | 3.5511 | 500 | active nearly full-backbone hybrid |
| 4 | MOD + plastic-4 | 3.5608 | 950 | completed limited-plasticity run |

ATE, plastic-24, and plastic-4 are close at their reported best checkpoints. Their training maturity differs, so this table is a current snapshot rather than a final convergence comparison.

### Matched step-200 comparison

Each model had processed 22,094,998 tokens in the same order.

| Rank | Model | Assistant PPL | Combined PPL | Top-1 |
|---:|---|---:|---:|---:|
| 1 | Pythia-2.8B full FT | **3.3908** | **3.3799** | **67.33%** |
| 2 | ATE h1/l1 | **3.5475** | **3.5328** | 66.60% |
| 3 | MOD + plastic-24 | 3.6520 | 3.6382 | 66.09% |
| 4 | MOD + plastic-4 | 3.6970 | 3.6828 | 65.87% |

At this matched budget, ATE is approximately 2.9% lower in assistant PPL than plastic-24 and 4.0% lower than plastic-4. The 2.8B dense control remains approximately 4.6% lower than ATE.

The scientific questions are different:

- **Full FT:** What quality is achievable by adapting the mature 2.8B dense model?
- **ATE:** Can a pretrained 1.4B model be expanded in width and depth to recover part of the larger model's capacity and quality?

Therefore, Pythia-2.8B is the absolute-quality control, while ATE is the architecture result.

## Track B — Parameter-Efficient Adaptation

This is currently a research-track ranking, not a matched locked-UltraChat leaderboard.

| Rank | Family | Current interpretation |
|---:|---|---|
| 1 | Shared MOD | leading parameter-efficient research family on the earlier matched experiments |
| 2 | MOD + plastic-4 | strongest limited-plasticity hybrid trade-off |
| 3 | LoRA | required matched locked-UltraChat baseline; current placement is provisional |
| 4 | MOD + plastic-24 | strong validation quality but weak parameter-efficiency claim |
| 5 | Full FT | quality benchmark, not parameter-efficient adaptation |

Shared MOD's strongest defensible claim is scoped to the earlier matched GPT-2 and Pythia experiments: it demonstrated rapid adaptation, competitive generations, and strong learning capacity in the FFN component. A universal claim that MOD is faster or better than LoRA requires both methods to be rerun on the locked UltraChat benchmark under identical conditions.

Plastic-4 selected 201,437,184 original base parameters plus MOD. Plastic-24 selected 1,311,625,216 original base parameters—approximately 92.7% of Pythia-1.4B—before MOD is counted.

Plastic-24 improves over plastic-4 by only 0.0097 assistant PPL at their best recorded checkpoints. Plastic-4 is therefore the clearer efficiency result even though plastic-24 is marginally ahead numerically.

## Track C — Architectural Novelty

| Rank | Direction | Research contribution |
|---:|---|---|
| 1 | Adaptive Transformer Expansion | grows pretrained width and depth with neutral initialization and controlled plasticity |
| 2 | Shared MOD | adds shared external adaptation capacity while preserving the pretrained backbone |
| 3 | Controlled Plasticity | tests how much of the original backbone must move |
| 4 | LoRA | established low-rank adaptation control |
| 5 | Full FT | essential quality control without architectural novelty |

This track describes research contribution; it is not a claim of empirical superiority.

## Canonical Winners

| Category | Current winner | Meaning |
|---|---|---|
| Benchmark champion | Pythia-2.8B full FT | lowest locked-UltraChat validation PPL |
| Architecture winner | ATE h1/l1 | strongest Pythia-1.4B-derived system at the matched 22.1M-token budget |
| Parameter-efficiency winner | Shared MOD | leading parameter-efficient family from the earlier matched evidence |
| Hybrid trade-off winner | MOD + plastic-4 | near plastic-24 quality with far less backbone movement |
| Useful but inefficient ablation | MOD + plastic-24 | demonstrates the limited gain from broad plasticity |

## Important ATE Qualification

The current ATE h1/l1 model expands Pythia-1.4B from 1,414,647,808 to 1,640,127,360 total parameters by adding 225,479,552 parameters. Under the current quadratic-plasticity configuration, all 1.64B parameters are trainable.

ATE is therefore **not PEFT in this run**. Its contribution is architecture expansion, neutral initialization, optimizer separation, and the reuse of a smaller pretrained starting point rather than minimal trainable parameter count.

The canonical claim is:

> ATE is currently the strongest Pythia-1.4B-derived system under the locked UltraChat benchmark at the matched 22,094,998-token budget.

## Three Research Hypotheses

1. **Shared MOD:** Can useful adaptation be stored in compact shared external modules while preserving the backbone?
2. **Controlled Plasticity:** What is the minimum portion of a pretrained backbone that must move?
3. **Adaptive Transformer Expansion:** Can new width and depth increase capacity while preserving useful pretrained behavior?

These are now separate research directions, not incremental variants competing for one universal title.

## Required Evidence Before Final Publication

1. Continue ATE beyond step 200 and record its convergence or plateau.
2. Run pure frozen Shared MOD and a parameter-matched LoRA baseline on the locked manifest.
3. Evaluate selected checkpoints on the untouched test split only after model selection.
4. Run the scored free-generation suite.
5. Measure pretrained-knowledge retention and catastrophic forgetting.
6. Repeat experiments across multiple seeds.
7. Measure systems performance on identical hardware.
8. Keep every dataset generation in a separate result series.
