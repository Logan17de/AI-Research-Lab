# 2026-07-22 — Locked UltraChat Plasticity Benchmark and ATE Launch

> **Later same-date update:** This launch snapshot preserves the evidence available before ATE produced metrics. ATE subsequently reached assistant PPL 3.5475 at step 200, and plastic-24 reached 3.5511 at step 500. See the [dated Model Ranking Framework](2026-07-22-model-ranking-framework.md) for the current comparison. Historical statements below are intentionally retained.

## Evidence Status

- **Dataset contract:** VALID
- **Pythia-1.4B MOD + plastic-4 metrics:** VALID through step 1,350
- **Pythia-1.4B MOD + plastic-24 metrics:** VALID snapshot through step 240; run still active
- **Pythia-2.8B full-FT metrics:** VALID through step 1,450
- **Cross-model interpretation:** EXPLORATORY until qualitative, retention, hardware-controlled, and multi-seed evaluation is complete
- **Adaptive Transformer Expansion:** IMPLEMENTED / INITIALIZED; no training result yet

This entry summarizes the Drive artifacts available on 2026-07-22. It does not overwrite the earlier Q&A, COT, or direct-answer V3 result series.

## Locked Dataset

The new comparison uses a deterministic filtered subset of HuggingFaceH4/ultrachat_200k.

The actual manifest—not the default script description—is authoritative:

| Split | Examples | Tokens |
|---|---:|---:|
| Train | 72,000 | 49,777,171 |
| Validation | 1,500 | 1,019,539 |
| Test | 1,500 | 1,033,646 |
| **Total** | **75,000** | **51,830,356** |

Filtering and locking include:

- English probability at least 0.90;
- 2–6 complete role messages;
- 128–1,024 Pythia tokens;
- coding-heavy and visibly broken sample removal;
- normalized first-user-prompt deduplication;
- conversation-level train/validation/test splitting;
- fixed category quotas;
- seed 42;
- strict sample order;
- zero shuffle buffer;
- zero data-loader workers;
- file, sample-order, token-order, and tokenizer hashes.

Category allocation:

| Category | Total examples |
|---|---:|
| Explanation / knowledge | 22,500 |
| Practical reasoning | 15,000 |
| Advice / planning | 11,250 |
| Summarization / rewriting | 11,250 |
| Creative generation | 7,500 |
| Multi-turn follow-up | 7,500 |

The training configurations reference only train and validation. The manifest's test split remains available for final comparison.

## Compared Runs

All measured variants use sequence length 1,024, effective batch size 160, evaluation every 50 optimizer steps, and the same locked training order.

| Run | Base | Adaptation scope | Status |
|---|---|---|---|
| MOD + plastic-4 | Pythia-1.4B | MOD plus final 4 transformer blocks and final LayerNorm | stopped at step 1,350 |
| MOD + plastic-24 | Pythia-1.4B | MOD plus broad plasticity across all 24 blocks, norms, biases, outputs, and LM head | active; snapshot at step 240 |
| Full FT | Pythia-2.8B | all dense parameters | stopped at step 1,450 |
| ATE h1/l1 | Pythia-1.4B | added width and depth | initialized; no metrics |

The MOD runs use 512-wide Input, Output, Attention, and FFN families with globally shared internal token-memory tables.

Exact proprietary modifier placement remains excluded.

## Main Metrics

| Variant | Best step | Effective epochs | Best assistant PPL | Combined PPL | Top-1 | First-answer | EOS |
|---|---:|---:|---:|---:|---:|---:|---:|
| MOD + plastic-4 | 950 | 2.111 | **3.5608** | 3.5476 | 66.61% | 50.37% | 85.93% |
| MOD + plastic-24 | 200 | 0.444 | **3.6520** | 3.6382 | 66.09% | 45.68% | 86.32% |
| Pythia-2.8B full FT | 1,450 | 3.222 | **3.2402** | 3.2287 | 68.24% | 50.79% | 87.37% |

The plastic-24 row is an early snapshot, not a completed-run best.

## Matched Step-200 Comparison

At step 200, all three runs had processed 22,094,998 tokens in the same order.

| Variant | Assistant PPL | Combined PPL | Top-1 | First-answer | EOS |
|---|---:|---:|---:|---:|---:|
| MOD + plastic-4 | 3.6970 | 3.6828 | 65.87% | 45.27% | 86.32% |
| MOD + plastic-24 | 3.6520 | 3.6382 | 66.09% | 45.68% | 86.32% |
| Pythia-2.8B full FT | **3.3908** | **3.3799** | **67.33%** | 45.10% | 84.46% |

Broad 24-layer plasticity improves slightly over plastic-4 early, but does not close the dense 2.8B gap.

## MOD + Plastic-4 Analysis

The selected plastic base tensors contain 201,437,184 parameters across layers 20–23 and the final LayerNorm, in addition to MOD parameters.

The run improved steadily to step 950:

| Step | Assistant PPL | Plastic drift | Output-MOD/base logit ratio |
|---:|---:|---:|---:|
| 50 | 3.9714 | 0.000148 | 7.17% |
| 500 | 3.5947 | 0.001094 | 13.14% |
| 950 | **3.5608** | 0.001594 | 17.49% |
| 1,350 | 3.5717 | 0.001846 | 19.93% |

After the step-950 optimum, training PPL continued from 3.3137 to 3.1976 while validation PPL rose slightly. This is a mild but clear generalization gap.

The increasing output-MOD ratio shows that the MOD output contribution continued growing even after validation stopped improving.

## MOD + Plastic-24 Analysis

This configuration makes 1,311,625,216 original base parameters plastic—approximately 92.7% of the Pythia-1.4B backbone—before counting MOD parameters.

Therefore, it is not a clean parameter-efficient baseline. It is better interpreted as a low-learning-rate, nearly full-backbone hybrid with added MOD capacity.

At step 200:

- assistant PPL: 3.6520;
- relative plastic drift: 0.001207;
- output-MOD/base logit ratio: 9.74%;
- selected base tensors: 291;
- selected base parameter count: 1,311,625,216.

Plastic drift is non-uniform. At step 200, the largest recorded group drift was around layer 8 (0.002325), while layer 23 was approximately 0.000660 and the LM head approximately 0.001927.

This suggests that broad plastic training is not simply concentrating change in the final layers.

The lower output-MOD ratio than plastic-4 at the same step is consistent with more adaptation being carried by the plastic base, but this remains an inference until per-family gradient and residual telemetry is compared directly.

## Pythia-2.8B Full-FT Analysis

The dense model's best recorded checkpoint is also its latest evaluation:

- step 1,450;
- 160,384,108 processed tokens;
- 3.222 effective epochs;
- assistant PPL 3.2402;
- target top-1 68.24%.

Unlike plastic-4, it had not formed a clear validation plateau by the final snapshot.

The dense model reached assistant PPL 3.5014 by step 100, already below the plastic-4 run's eventual best of 3.5608 at step 950.

This is the clearest current result:

> Pythia-2.8B full fine-tuning is more sample-efficient on the locked UltraChat objective, despite the smaller hybrid processing tokens faster in its recorded environment.

At each run's best checkpoint, plastic-4 remains approximately 9.9% higher in perplexity than the 2.8B dense baseline. The first-answer accuracies are close, but the dense model retains an approximately 1.64 percentage-point top-1 advantage.

## Systems Measurements

Recorded median training throughput:

| Variant | Median tokens/s | Reported GPU memory |
|---|---:|---:|
| MOD + plastic-4 | 13,391 | 13.4 / 93.5 GB |
| MOD + plastic-24 | 8,452 | 13.4 / 56.1 GB |
| Pythia-2.8B full FT | 3,355 | 19.4 / 67.3 GB |

These are valid run measurements but **not a controlled architecture-speed comparison**. The reported GPU capacities differ, and plastic-4 also used a different micro-batch/accumulation split while preserving the same effective batch size.

A same-device benchmark is required before claiming a throughput multiplier.

## Adaptive Transformer Expansion

The new ATE branch is separate from MOD and LoRA. At a high level, it adds trainable width and depth while retaining pretrained tensors as independent parameters and supports controlled base plasticity.

The current run directory is ultrachat_pythia_1.4b_ate_h1_l1.

Available evidence:

- the locked dataset was resolved successfully;
- the step cache reports 450 optimizer steps per epoch;
- the run directory contains no metrics, config, drift log, or checkpoint yet.

**Status: PROPOSED / INITIALIZED, not an experimental result.**

No quality, convergence, efficiency, or retention claim should be attached to ATE until metrics and free generations exist.

## What Is Established

- The 75,000-conversation UltraChat benchmark is reproducibly locked.
- Plastic-4 reaches assistant PPL 3.5608 with limited base drift.
- Plastic-4 shows a mild validation plateau after step 950.
- Broad plastic-24 improves early PPL slightly over plastic-4.
- Plastic-24 changes nearly the whole 1.4B backbone and should not be described as strongly parameter-efficient.
- Pythia-2.8B full FT is substantially more sample-efficient and currently stronger on validation PPL.
- ATE is ready to run but has no measured result.

## What Remains Unresolved

- Whether plastic-4 preserves pretrained knowledge better than dense full FT.
- Whether its free generations are competitive at matched checkpoints.
- Whether broad plasticity eventually closes the quality gap.
- Whether a pure frozen-MOD baseline benefits from adding four plastic layers.
- Whether the smaller hybrid is faster on identical hardware.
- Whether the result reproduces across seeds.
- Whether ATE width/depth expansion is superior to MOD, plasticity, LoRA, or full FT.

## Required Next Controls

1. Finish or deliberately stop plastic-24 and record its final valid checkpoint.
2. Run pure frozen MOD on the same locked manifest.
3. Add Pythia-1.4B full FT and a parameter-matched LoRA baseline.
4. Evaluate all selected checkpoints on the untouched test split.
5. Run the scored free-generation suite.
6. Measure pretrained retention.
7. Repeat systems measurements on identical hardware.
8. Run multiple seeds.
9. Begin ATE only with its exact parameter report and neutral-initialization checks recorded.
10. Keep result series separated from the earlier COT and V3 datasets.

Machine-readable snapshot: [pythia-ultrachat-plasticity-2026-07-22.csv](../../results/pythia-ultrachat-plasticity-2026-07-22.csv).
