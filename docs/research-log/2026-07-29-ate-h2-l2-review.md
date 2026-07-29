# 2026-07-29 — Stage-Aware ATE H2/L2 Review

## Evidence Status

- **Locked UltraChat protocol:** VALID and directly comparable to the existing Pythia UltraChat runs.
- **H2/L2 metrics:** VALID through local step 650.
- **Best checkpoint:** PRESENT (`checkpoints/best.pt`), selected at local step 450.
- **Expansion safety configuration:** `ate_output_init=exact` and `allow_nonpreserving_expansion=false`.
- **Source lineage:** UNRESOLVED from the saved sidecar artifacts. The serialized `config.json` does not retain the incremental source path, and no explicit full-validation step-zero row is present.
- **Architecture interpretation:** PROVISIONAL until the exact source checkpoint identity and pre-update validation equivalence are recorded.

This review inspects the Drive run folder:

```text
Pythia_run/runs/ultrachat_pythia_1.4b_ate_h2_l2
```

The folder contains `metrics.csv`, `plasticity_drift.jsonl`, `config.json`, a step-500 checkpoint, and a best checkpoint.

## Run Configuration

| Setting | Value |
|---|---:|
| Final added attention heads | 2 |
| Final added transformer layers | 2 |
| Sequence length | 1,024 |
| Micro batch | 40 |
| Gradient accumulation | 4 |
| Effective batch | 160 |
| Evaluation interval | 50 steps |
| Base/new-stage learning rate | 3e-4 |
| Plasticity base learning rate | 1e-5 |
| Previous ATE stages trainable | yes |
| Output initialization | exact |
| Unsafe non-preserving expansion allowed | no |

The training code intentionally clears `incremental_ate_checkpoint` before serializing the dataclass config, so the `null` value in `config.json` does not prove that the run was fresh. However, it also means the sidecar config cannot independently prove which checkpoint was used as the source.

## H2/L2 Validation Curve

| Local step | Local processed tokens | Train PPL | Assistant PPL | Combined PPL | Top-1 | First-answer | EOS | Base drift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1,124,355 | 3.3637 | 3.4599 | 3.4477 | 67.06% | 53.07% | 84.39% | 0 |
| 50 | 5,533,790 | 3.2442 | 3.4636 | 3.4516 | 67.07% | 52.50% | 84.97% | 0.000038 |
| 100 | 11,052,595 | 3.1538 | 3.5870 | 3.5691 | 66.49% | 49.13% | 92.01% | 0.000123 |
| 200 | 22,094,998 | 3.0845 | 3.5763 | 3.5654 | 66.46% | 52.29% | 80.40% | 0.000268 |
| 350 | 38,722,201 | 3.1220 | 3.5474 | 3.5309 | 66.61% | 52.81% | 90.49% | 0.000397 |
| 400 | 44,243,903 | 3.3475 | 3.5111 | 3.4994 | 66.81% | 52.76% | 84.29% | 0.000432 |
| 450 | 49,777,171 | 3.4244 | **3.4548** | **3.4410** | **67.11%** | 53.18% | 87.28% | 0.000473 |
| 500 | 55,310,961 | 3.0153 | 3.5446 | 3.5309 | 66.67% | 52.86% | 85.86% | 0.000508 |
| 550 | 60,829,766 | 2.8416 | 3.6567 | 3.6380 | 66.15% | **53.76%** | 92.59% | 0.000537 |
| 600 | 66,357,075 | 2.7437 | 3.7113 | 3.6958 | 66.13% | 51.97% | 86.60% | 0.000566 |
| 650 | 71,872,169 | 2.6916 | 3.7254 | 3.7062 | 66.10% | 53.02% | 91.59% | 0.000595 |

## Selected Checkpoint

The best H2/L2 assistant PPL is:

```text
3.4548079 at local step 450
```

Associated metrics:

- combined PPL: 3.4409744;
- target Top-1: 67.1077%;
- first-answer accuracy: 53.1792%;
- EOS accuracy: 87.2832%;
- original-backbone relative drift: 0.0004730.

The best is only 0.147% lower than the first stored evaluation value of 3.4599. After step 450, validation PPL regresses to 3.7254 at step 650 while training PPL falls to 2.6916. This is a 7.83% validation regression from the best and is consistent with overfitting/generalization collapse rather than numerical instability.

## Updated Locked UltraChat Best-Checkpoint Ranking

The following table combines the best recorded validation checkpoints on the same locked dataset. Sequential ATE checkpoints have larger inherited training budgets, so this is a quality leaderboard rather than a matched-compute leaderboard.

| Rank | Model | Best assistant PPL | Best step | Evidence status |
|---:|---|---:|---:|---|
| 1 | Pythia-2.8B Full FT | **3.2402** | 1,450 | absolute-quality control |
| 2 | ATE H1/L2 | **3.4358** | 450 | valid sequential checkpoint; causal depth gain unproven |
| 3 | ATE H2/L2 | **3.4548** | 450 | valid metric/checkpoint; source lineage unresolved |
| 4 | ATE H1/L1 | **3.4602** | 400 | strongest matched fresh ATE result |
| 5 | MOD + plastic-24 | 3.5511 | 500 | strong PPL; nearly full-backbone hybrid |
| 6 | MOD + plastic-4 | 3.5608 | 950 | limited-plasticity trade-off winner |

Relative comparisons:

- H1/L2 is 0.55% lower in PPL than H2/L2.
- H2/L2 is 0.16% lower in PPL than H1/L1.
- H1/L2 is 0.70% lower in PPL than H1/L1.
- Pythia-2.8B Full FT is 6.21% lower in PPL than H2/L2.

## Provenance Problem

The intended next experiment was a corrected stage-aware continuation whose step-zero validation score exactly reproduced the H1/L2 source score of 3.4358. The inspected artifacts do not establish that condition:

1. no explicit pre-update full-validation row is stored;
2. the first stored value occurs after ten optimizer steps and is 3.4599, not 3.4358;
3. the saved config cannot recover the incremental source path;
4. the current Drive `runs` folder does not contain an H1/L2 run folder or checkpoint sidecar that independently establishes the lineage.

The code-level preservation guard is still meaningful: unsafe non-preserving expansion was disabled, and exact output initialization was requested. But the startup probe and source identity must be persisted as an artifact before this run can be called a verified H1/L2 → H2/L2 continuation.

## Interpretation

Established:

- the stage-aware H2/L2 architecture trained stably through step 650;
- a best checkpoint exists at step 450;
- H2/L2 reaches 3.4548 assistant PPL, slightly improving over H1/L1 but remaining behind H1/L2;
- the run shows the same post-one-epoch validation collapse observed in H1/L2;
- increasing both width and depth does not automatically improve the current validation floor.

Not established:

- that H2/L2 inherited H1/L2 exactly;
- that the extra attention head improves quality;
- that H2/L2 is more sample-efficient than H1/L2;
- that any ATE model preserves pretrained knowledge better;
- that the result generalizes across seeds or the untouched test split.

## Required Next Control

Run H1/L2 → H2/L2 again with all of the following persisted outside the checkpoint:

1. source checkpoint path and SHA-256;
2. source architecture stage history;
3. source best metric and source step;
4. full-validation step-zero metrics before any update;
5. startup preservation probe values;
6. requested stage increment separately from final total heads/layers;
7. a no-expansion H1/L2 continuation with the same optimizer reset, data restart, and local-token budget.

Until then, H2/L2 should remain on the leaderboard with the label **valid checkpoint, unresolved lineage**, not as a confirmed continuation of H1/L2.
