# 2026-07-19 — Pythia Token MOD Variant Sweep

## Status

**VALID for the recorded causal-pipeline metrics.**

**EXPLORATORY for architecture comparison.**

This sweep measures how shared versus layer-unique Token MOD tables, MOD width, and training schedule affect a frozen Pythia-1.4B model. It also retains fully fine-tuned Pythia-2.8B as a stronger target-performance baseline.

The comparison does not yet establish general superiority. It uses one seed and one chain-of-thought Q&A train/validation split, and the 1.4B MOD systems do not share the 2.8B baseline's starting model quality.

## Protocol

Common settings:

- training data: `data/chain_of_thought/split/train.txt`;
- validation data: `data/chain_of_thought/split/validation.txt`;
- sequence length: 2,048;
- effective batch size: 20;
- 321 optimizer steps per epoch;
- evaluation every 50 steps;
- seed: 42;
- assistant-only and combined validation perplexity;
- frozen Pythia-1.4B backbone for every MOD variant;
- fully fine-tuned Pythia-2.8B for the dense reference.

The comparison uses each run's minimum assistant-validation perplexity rather than its final checkpoint because several MOD runs entered visible validation overfitting.

## Variant Results

Dimensions are written as Input / Output / Attention / FFN.

| Variant | Dimensions | Internal tables | Approx. trainable parameters | Best assistant PPL | Best combined PPL | Best step | Top-1 at best | Last/current assistant PPL |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| v1_1 | 128 / 128 / 128 / 256 | 1 shared table per family | 51,593,216 | 4.0188 | 4.0099 | 1,150 | 66.34% | 4.0673 |
| v1_2 | 256 / 256 / 256 / 512 | 1 shared table per family | 103,186,432 | 3.9859 | 3.9770 | 950 | 66.48% | 4.1664 |
| v1_3 | 256 / 256 / 256 / 512 | 1 shared table per family | 103,186,432 | **3.9724** | **3.9636** | 950 | **66.60%** | 4.0289 |
| v1_4 | 256 / 256 / 512 / 1,024 | 1 shared table per family | 179,568,640 | **3.9724** | **3.9637** | 600 | 66.48% | 3.9968 |
| v2 | 128 / 128 / 64 / 128 | 24 layer-unique tables per family | 254,640,128 | 4.0429 | 4.0339 | 950 | 66.15% | 4.0732 |
| v2_1 | 128 / 128 / 128 / 64 | 24 layer-unique tables per family | 254,640,128 | 4.1028 | 4.0936 | 600 | 65.87% | 4.1028 at step 640 |
| Pythia-2.8B full FT | — | — | 2,775,208,960 | **3.5190** | **3.5119** | 2,250 | **68.60%** | 3.5202 |

v2_1 was still in progress when this record was written. Its row is a step-640 snapshot, not a completed minimum.

The machine-readable summary is stored in [results/pythia-variant-sweep-2026-07-19.csv](../../results/pythia-variant-sweep-2026-07-19.csv).

## Findings

### 1. Shared MOD remains the strongest current MOD design

v1_3 and v1_4 reached the same minimum assistant PPL to four decimal places: approximately 3.9724.

v1_3 is the parameter-efficient winner of this sweep. v1_4 used approximately 76.4 million additional trainable parameters and reached the same validation floor earlier, but it did not improve the final minimum.

This supports a narrow conclusion:

> Increasing shared internal MOD width accelerated early adaptation, but did not lower the validation floor on this dataset.

### 2. Layer-unique tables were parameter-inefficient on this dataset

v2 used approximately 254.6 million trainable parameters—about 2.47 times v1_3—yet its best assistant PPL was 4.0429 rather than 3.9724.

The result does not prove that layer uniqueness is generally harmful. It suggests that replicating token-memory tables across 24 layers sacrifices the gradient sharing and regularization of the shared design, making the unique design more data-hungry.

The present data therefore supports shared cross-layer token memory as the stronger inductive bias for this benchmark.

### 3. Attention/FFN width allocation produced almost no early difference

v2 uses Attention 64 / FFN 128. v2_1 reverses the allocation to Attention 128 / FFN 64 while preserving the same total parameter count.

Through step 400, the two assistant-PPL curves differed by no more than approximately 0.0053:

| Step | v2 | v2_1 |
|---:|---:|---:|
| 100 | 4.8123 | 4.8141 |
| 250 | 4.3457 | 4.3474 |
| 350 | 4.2389 | 4.2389 |
| 400 | 4.2038 | 4.2036 |

This is evidence that width allocation between these two token-residual locations did not materially affect early learning.

It does not yet distinguish between two explanations:

1. Attention and FFN token residuals are functionally interchangeable under this implementation; or
2. the identical Input and Output MOD paths dominate the observed learning.

Module-specific gradient and residual-norm telemetry is required to decide between them.

### 4. The approximately 4.05 unique-table minimum is not a hard learning-capacity wall

v2 reached its best assistant PPL of 4.0429 at step 950. Training PPL continued falling afterward while validation PPL rose to 4.0732.

The same pattern was clearer in v1_2:

- best assistant PPL: 3.9859;
- last assistant PPL: 4.1664;
- training batch PPL continued decreasing.

This is validation overfitting or a generalization limit under the current data and schedule—not proof that the architecture cannot fit additional information.

Best-checkpoint selection is therefore mandatory.

### 5. The shorter schedule was healthier

v1_2 and v1_3 use the same architecture. Their principal difference is the planned 50-epoch versus 7-epoch schedule.

v1_3 learned faster and reached a slightly better minimum. The logged learning rates show that the warmup duration changed with the planned total step count despite both configurations recording 100 warmup steps. The 50-epoch plan remained near peak learning rate while validation was already worsening.

Future sweeps should report the resolved warmup and decay steps, not only the user-facing warmup configuration.

### 6. Full fine-tuning remains the quality leader

Pythia-2.8B full fine-tuning reached:

- assistant PPL 3.5190;
- combined PPL 3.5119;
- target top-1 68.60%.

The best MOD result reached assistant PPL 3.9724 and top-1 66.60%.

The dense baseline was approximately 2.3 times slower per optimization step at the same effective batch size, but this is not a method-isolated comparison. Pythia-2.8B began with substantially better validation PPL than Pythia-1.4B.

The valid conclusion is that the larger dense model remains the target-performance leader, while the MOD systems offer much lower trainable-parameter and wall-clock cost.

### 7. Assistant and combined PPL tell the same story

After the initial region, combined PPL remained approximately 0.009 below assistant PPL for most variants and preserved the same ranking. It is useful as a consistency check but does not currently provide an independent architecture signal.

## Evidence Status

| Claim | Status |
|---|---|
| Shared v1_3/v1_4 reached approximately 3.972 assistant PPL | **VALID** |
| v1_3 is the most parameter-efficient best-performing MOD in this sweep | **VALID within this protocol** |
| v1_4 learns faster but reaches the same minimum as v1_3 | **VALID within this protocol** |
| v2 reached approximately 4.043 and then overfit | **VALID** |
| v2 and v2_1 are nearly identical through step 400 | **VALID** |
| Layer-unique tables are generally inferior | **NOT ESTABLISHED** |
| Attention and FFN MODs are functionally interchangeable | **NOT ESTABLISHED** |
| Input/Output MOD dominates the internal MODs | **HYPOTHESIS** |
| MOD matches fully fine-tuned Pythia-2.8B | **NOT SUPPORTED** |
| Current results establish continual-learning superiority | **NOT SUPPORTED** |

## Next Controlled Experiment

1. Fix dimensions, seed, data, and the seven-epoch schedule.
2. Compare unique-table counts 1, 4, 8, and 24.
3. Run a separate parameter-matched count sweep.
4. Test Input+Output only, +Attention, +FFN, and all-family ablations.
5. Log gradient norms and residual-to-base norms independently for Input, Output, Attention, and FFN.
6. Use automatic early stopping and restore the best checkpoint.
7. Add Pythia-1.4B full fine-tuning and a parameter-matched LoRA baseline.
8. Repeat decisive configurations over multiple seeds.
9. Evaluate untouched-pretraining retention and free generation before making continual-learning claims.

## Current Conclusion

The sweep produced a useful negative result: adding layer-unique token-memory tables increased parameter count without improving validation quality on the current dataset.

The strongest present interpretation is that cross-layer sharing is doing useful work—not merely reducing parameters. It improves sample efficiency and acts as a regularizing inductive bias. The next count-controlled and module-ablation experiments must test that explanation directly.
