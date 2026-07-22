# 2026-07-22 — Model Ranking Framework

## Evidence Status

- **Locked UltraChat validation metrics:** VALID for the recorded checkpoints
- **Matched-token ranking:** VALID at step 200 / 22,094,998 processed tokens
- **Fresh ATE h1/l1 ranking:** VALID through its latest completed evaluation at step 400; model-quality interpretation remains provisional
- **Sequential ATE ranking:** EXPLORATORY because h1/l2 inherits prior h1/l1 training and h2/l2 has only one early record
- **Parameter-efficiency track:** CROSS-BENCHMARK INTERPRETATION until matched UltraChat MOD and LoRA baselines exist
- **Architectural-novelty track:** RESEARCH INTERPRETATION, not a numerical benchmark
- **Final model ranking:** PENDING untouched-test, free-generation, retention, identical-hardware, and multi-seed evaluation

This framework prevents results from different datasets and evaluation protocols from being collapsed into one misleading leaderboard. The COT, direct-answer V3, locked UltraChat, GPT-2, Qwen/Tülu, and Inverted Transformer result series remain separate.

## Track A — Absolute Quality on Locked UltraChat

All four fresh-run comparisons use the same locked dataset, token order, sequence length 1,024, and effective batch size 160. Evaluation cadence differs for some later ATE runs and is reported explicitly in the expansion-sweep entry.

### Best recorded validation checkpoints

| Rank | Model | Best assistant PPL | Step | Status |
|---:|---|---:|---:|---|
| 1 | Pythia-2.8B full FT | **3.2402** | 1,450 | current benchmark |
| 2 | ATE h1/l1 | **3.4602** | 400 | strongest fresh Pythia-1.4B-derived result |
| 3 | MOD + plastic-24 | 3.5511 | 500 | active nearly full-backbone hybrid |
| 4 | MOD + plastic-4 | 3.5608 | 950 | completed limited-plasticity run |

Fresh ATE h1/l1 continues improving through its latest completed evaluation. It is 2.56% lower in assistant PPL than plastic-24 and 2.83% lower than plastic-4 at their respective best checkpoints. Training maturity still differs, so this is a current validation snapshot rather than a final convergence comparison.

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

## Track A2 — Sequential Architecture Expansion

This track is separated from fresh-run ranking because later architectures inherit trained ATE checkpoints and restart their local optimizer/data schedule.

| Rank | Model | Inherited baseline | Best assistant PPL | Local step | Status |
|---:|---|---:|---:|---:|---|
| 1 | ATE h1/l2 | 3.4602 from h1/l1 | **3.4358** | 450 | valid checkpoint; causal depth gain unproven |
| — | ATE h2/l2 | not fully verified | 3.4922 | 10 | insufficient evidence; no checkpoint |

ATE h1/l2 improves over its inherited h1/l1 value by approximately 0.71%, but the best checkpoint represents about 94.0M cumulative source-plus-local training tokens. A no-expansion h1/l1 continuation with the same reset and additional-token budget is required to isolate the contribution of the second added layer.

After local step 450, h1/l2 validation PPL regresses 7.60% to 3.6970 at step 600 while training PPL falls from 3.4007 to 2.7604. This is severe generalization collapse or overfitting, not a recorded numerical failure.

See the [ATE Sequential Expansion Sweep](2026-07-22-ate-expansion-sweep.md).

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
| Architecture winner | ATE family | h1/l1 leads the matched fresh track; h1/l2 is the best observed sequential checkpoint |
| Parameter-efficiency winner | Shared MOD | leading parameter-efficient family from the earlier matched evidence |
| Hybrid trade-off winner | MOD + plastic-4 | near plastic-24 quality with far less backbone movement |
| Useful but inefficient ablation | MOD + plastic-24 | demonstrates the limited gain from broad plasticity |

## Important ATE Qualification

The current ATE h1/l1 model expands Pythia-1.4B from 1,414,647,808 to 1,640,127,360 total parameters by adding 225,479,552 parameters. Under the current quadratic-plasticity configuration, all 1.64B parameters are trainable.

ATE is therefore **not PEFT in this run**. Its contribution is architecture expansion, neutral initialization, optimizer separation, and the reuse of a smaller pretrained starting point rather than minimal trainable parameter count.

The canonical claims are:

> ATE h1/l1 is the strongest Pythia-1.4B-derived fresh-run system under the locked UltraChat benchmark at the matched 22,094,998-token budget.

> ATE h1/l2 is the best observed Pythia-1.4B-derived sequential checkpoint, but its 3.4358 result cannot yet be attributed specifically to added depth.

## Three Research Hypotheses

1. **Shared MOD:** Can useful adaptation be stored in compact shared external modules while preserving the backbone?
2. **Controlled Plasticity:** What is the minimum portion of a pretrained backbone that must move?
3. **Adaptive Transformer Expansion:** Can new width and depth increase capacity while preserving useful pretrained behavior?

These are now separate research directions, not incremental variants competing for one universal title.

## Required Evidence Before Final Publication

1. Evaluate fresh h1/l1 at its next scheduled checkpoint and preserve its full curve.
2. Run a no-expansion h1/l1 continuation from the same source, with the same reset and additional-token budget as h1/l2.
3. Persist source checkpoint identity and pre-update step-zero equivalence for every incremental expansion.
4. Run pure frozen Shared MOD and a parameter-matched LoRA baseline on the locked manifest.
5. Evaluate selected checkpoints on the untouched test split only after model selection.
6. Run the scored free-generation suite.
7. Measure pretrained-knowledge retention and catastrophic forgetting.
8. Repeat experiments across multiple seeds.
9. Measure systems performance on identical hardware.
10. Keep every dataset generation in a separate result series.
