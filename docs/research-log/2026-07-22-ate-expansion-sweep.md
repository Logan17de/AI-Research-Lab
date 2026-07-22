# 2026-07-22 — ATE Sequential Expansion Sweep

## Evidence Status

- **Locked UltraChat dataset contract:** VALID
- **ATE h1/l1 metrics:** VALID through local step 490; latest completed evaluation at step 400
- **ATE h1/l2 metrics:** VALID through local step 600; SEQUENTIAL / EXPLORATORY architecture comparison
- **ATE h2/l2 metrics:** INSUFFICIENT — one early record at step 10 and no checkpoint
- **Cross-model ranking:** VALID only where initialization and token budgets are matched
- **Model-quality conclusion:** PROVISIONAL until scored generation, retention, untouched-test, and multi-seed evaluation

This entry records new Drive artifacts inspected on 2026-07-22. It extends, but does not overwrite, the earlier [ATE ranking framework](2026-07-22-model-ranking-framework.md).

## Shared Protocol

All runs use:

- the locked 75,000-conversation filtered UltraChat manifest;
- 72,000 training, 1,500 validation, and 1,500 untouched test conversations;
- sequence length 1,024;
- effective batch size 160;
- seed 42;
- strict sample order and zero shuffle buffer;
- new-parameter learning rate (3\times10^{-4});
- quadratic base plasticity with maximum base learning rate (1\times10^{-5});
- seven configured epochs.

The untouched test split remains unused.

## Expansion Configurations

| Run | Added heads | Added layers | Starting interpretation | Eval interval |
|---|---:|---:|---|---:|
| ATE h1/l1 | 1 | 1 | fresh expansion from Pythia-1.4B | 100 |
| ATE h1/l2 | 1 | 2 | sequential expansion from the trained h1/l1 state | 50 |
| ATE h2/l2 | 2 | 2 | later expansion stage; provenance requires stronger metadata | 50 |

ATE h1/l1 expands Pythia-1.4B from 1,414,647,808 to 1,640,127,360 parameters by adding 225,479,552 parameters. All original and added parameters are trainable under quadratic plasticity, so this is not PEFT.

Exact parameter reports for h1/l2 and h2/l2 were not present in the inspected run folders and are not estimated here.

## ATE h1/l1 — Fresh Expansion

| Step | Processed tokens | Effective epochs | Assistant PPL | Combined PPL | Top-1 | First-answer | EOS | Base drift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial record | 1,124,355 | 0.022 | 5.2234 | 5.2616 | 60.13% | 1.31% | 0.21% | 0 |
| 100 | 11,052,595 | 0.222 | 3.6398 | 3.6270 | 66.12% | 46.98% | 84.45% | 0.000175 |
| 200 | 22,094,998 | 0.444 | 3.5475 | 3.5328 | 66.60% | 49.82% | 87.39% | 0.000361 |
| 300 | 33,218,806 | 0.667 | 3.4981 | 3.4849 | 66.93% | 51.60% | 85.71% | 0.000479 |
| 400 | 44,243,903 | 0.889 | **3.4602** | **3.4481** | **67.05%** | **53.23%** | 84.71% | 0.000565 |

The run continues in the file through step 490, but step 400 is the latest completed evaluation because this configuration evaluates every 100 steps.

This is a stronger result than the earlier step-200 snapshot:

- assistant PPL improves from 3.5475 to 3.4602;
- first-answer accuracy improves from 49.82% to 53.23%;
- top-1 improves from 66.60% to 67.05%;
- drift remains low at 0.000565.

At its current best, fresh h1/l1 is 2.56% lower in assistant PPL than plastic-24's best 3.5511 and 2.83% lower than plastic-4's best 3.5608. Pythia-2.8B full FT remains 6.79% lower in PPL.

No validation plateau is established yet because the next scheduled evaluation at step 500 is absent.

## ATE h1/l2 — Sequential Depth Expansion

The first stored h1/l2 evaluation value is exactly 3.460216, matching the h1/l1 step-400 best. This is consistent with successful checkpoint inheritance and output-neutral initialization of the added layer.

| Local step | New-run tokens | Assistant PPL | Train PPL | Top-1 | Base drift |
|---:|---:|---:|---:|---:|---:|
| inherited baseline | — | 3.4602 | — | 67.05% | 0 |
| 50 | 5,533,790 | 3.4564 | 3.2309 | 67.12% | 0.000039 |
| 100 | 11,052,595 | 3.5695 | 3.1356 | 66.48% | 0.000127 |
| 250 | 27,646,764 | 3.5482 | 3.1115 | 66.70% | 0.000322 |
| 350 | 38,722,201 | 3.5295 | 3.0983 | 66.76% | 0.000403 |
| 400 | 44,243,903 | 3.4780 | 3.3372 | 66.97% | 0.000439 |
| 450 | 49,777,171 | **3.4358** | 3.4007 | **67.24%** | 0.000481 |
| 500 | 55,310,961 | 3.5226 | 2.9884 | 66.84% | 0.000518 |
| 550 | 60,829,766 | 3.6717 | 2.8157 | 66.33% | 0.000551 |
| 600 | 66,357,075 | 3.6970 | 2.7604 | 66.18% | 0.000583 |

The best sequential checkpoint improves over its inherited h1/l1 starting point by 0.0244 PPL, or approximately 0.71%.

However, this is not a fresh-run architecture win. If the source is the h1/l1 step-400 checkpoint, the best h1/l2 result represents approximately:

- 44,243,903 source-training tokens;
- 49,777,171 additional h1/l2 tokens;
- **94,021,074 cumulative tokens**;
- approximately 1.889 cumulative passes over the training-set token count.

A matched control must restart h1/l1 from the same source checkpoint, reset optimizer and data position identically, and train for the same additional 49,777,171 tokens without adding the second layer.

### Late generalization collapse

After the step-450 best:

- assistant validation PPL worsens from 3.4358 to 3.6970, a 7.60% regression;
- training PPL improves from 3.4007 to 2.7604;
- drift rises smoothly rather than exploding;
- no NaN or numerical failure is recorded.

This is evidence of severe post-one-epoch generalization collapse or overfitting, not numerical instability. Step 450 should remain the selected checkpoint.

## ATE h2/l2 — Early Launch Only

The h2/l2 folder contains one metric row:

| Step | Processed tokens | Assistant PPL | Combined PPL | Top-1 | First-answer | EOS |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1,124,355 | 3.4922 | 3.4801 | 66.95% | 51.97% | 82.66% |

The run has no checkpoint and no later evaluation. It is not ranked.

The first h2/l2 value does not exactly reproduce the h1/l2 best, so neutral-transfer quality cannot be confirmed from the available artifacts. Record the exact source checkpoint identity and an explicit pre-update step-zero evaluation before interpreting this branch.

## Current Ranking Structure

### Fresh-run locked UltraChat track

| Rank | Model | Best assistant PPL | Step |
|---:|---|---:|---:|
| 1 | Pythia-2.8B full FT | **3.2402** | 1,450 |
| 2 | ATE h1/l1 | **3.4602** | 400 |
| 3 | MOD + plastic-24 | 3.5511 | 500 |
| 4 | MOD + plastic-4 | 3.5608 | 950 |

### Sequential expansion track

| Rank | Model | Best assistant PPL | Local step | Status |
|---:|---|---:|---:|---|
| 1 | ATE h1/l2 | **3.4358** | 450 | valid checkpoint; causal architecture gain unproven |
| — | ATE h2/l2 | 3.4922 | 10 | insufficient evidence |

The best observed Pythia-1.4B-derived checkpoint is h1/l2, but the strongest matched fresh-run result remains h1/l1.

## What Is Established

- Fresh ATE h1/l1 continues improving through its latest evaluation at step 400.
- h1/l1 clearly outperforms both current MOD-plasticity hybrids on validation PPL.
- A sequential second-layer expansion can inherit h1/l1 quality and temporarily improve it.
- h1/l2 reaches its best at one local epoch and then overfits sharply.
- h2/l2 is too early to interpret.
- Pythia-2.8B full FT remains the absolute-quality benchmark.

## What Is Not Established

- The h1/l2 improvement is caused by added depth rather than its extra training and schedule reset.
- ATE preserves pretrained knowledge better.
- ATE produces better free generations.
- ATE is compute-, memory-, or parameter-efficient.
- ATE h2/l2 preserves the source checkpoint neutrally.
- Any result reproduces across seeds.

## Required Next Controls

1. Preserve the exact source checkpoint path or hash in every incremental ATE config and checkpoint.
2. Evaluate the expanded model at local step zero before any optimizer update.
3. Run a no-expansion h1/l1 continuation control with the same optimizer reset, data restart, and additional-token budget as h1/l2.
4. Stop h1/l2 at step 450 and retain that checkpoint.
5. Continue h2/l2 only after verifying neutral initialization against its source.
6. Run scored free-generation and pretrained-retention evaluations.
7. Keep the untouched test split sealed until model selection.
8. Repeat the selected comparison across multiple seeds.

Machine-readable metrics: [pythia-ate-expansion-2026-07-22.csv](../../results/pythia-ate-expansion-2026-07-22.csv).
