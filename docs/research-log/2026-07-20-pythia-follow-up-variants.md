# 2026-07-20 — Pythia Follow-up Variants

## Status

**VALID for recorded metrics, configurations, and dataset reports.**

**EXPLORATORY for architecture interpretation and cross-run conclusions.**

This follow-up closes the v2_1 record, adds v2_2, and starts a new direct-answer dataset series with v3. The V3 step-600 free-generation test is documented separately in [Pythia V3 Chat Evaluation](2026-07-20-pythia-v3-chat-evaluation.md).

## Results

| Variant | Dataset | Dimensions (I/O/A/F) | Internal tables | Trainable parameters | Best assistant PPL | Best step | Last step | Status |
|---|---|---:|---|---:|---:|---:|---:|---|
| v2_1 | original COT-answer | 128 / 128 / 128 / 64 | 24 / 24 unique | 254,640,128 | **4.0420** | 950 | 1,060 | complete |
| v2_2 | original COT-answer | 512 / 512 / 64 / 64 | 24 / 24 unique | 214,433,792 | **4.0589** | 900 | 1,020 | stopped after instability |
| v3 | rebuilt direct-answer | 128 / 128 / 128 / 512 | 1 / 1 shared | 77,053,952 | **6.5713** | 600 | 760 | stopped |

The v3 number is **not comparable** to the July 19 leaderboard because the train/validation corpus was rebuilt.

## v2_1 — Attention/FFN width swap

v2_1 reverses the internal width allocation used by v2 while holding the total unique internal width and parameter count constant:

| Variant | Attention | FFN | Best assistant PPL | Best step |
|---|---:|---:|---:|---:|
| v2 | 64 | 128 | 4.0429226086 | 950 |
| v2_1 | 128 | 64 | 4.0420111890 | 950 |

The difference is approximately 0.0009 PPL. The curves are also nearly identical:

| Step | v2 | v2_1 |
|---:|---:|---:|
| 100 | 4.8123 | 4.8141 |
| 400 | 4.2038 | 4.2036 |
| 700 | 4.0907 | 4.0890 |
| 900 | 4.0568 | 4.0518 |
| 950 | 4.0429 | 4.0420 |
| 1,000 | 4.0534 | 4.0525 |

This is a true tie within the resolution of the run. It supports the narrower observation that moving a fixed unique-table width budget between Attention and FFN did not materially affect this protocol. It does not prove that the two families are generally interchangeable. Per-family residual and gradient norms are required to explain the tie.

## v2_2 — Faster early learning, then instability

v2_2 increases Input and Output width to 512 while using 64-wide, 24-table Attention and FFN families. It learned faster early than v2/v2_1:

| Step | v2 | v2_1 | v2_2 |
|---:|---:|---:|---:|
| 50 | 5.8606 | 5.8590 | **5.2752** |
| 100 | 4.8123 | 4.8141 | **4.6600** |
| 300 | 4.2903 | 4.2851 | **4.2227** |
| 500 | 4.1449 | 4.1487 | **4.1027** |
| 600 | 4.1014 | 4.1028 | **4.0695** |
| 800 | approximately 4.069 | approximately 4.069 | approximately 4.069 |
| 900 | 4.0568 | **4.0518** | 4.0589 |

The larger Input/Output paths accelerated early optimization but did not lower the stable validation floor.

Between steps 900 and 950, assistant PPL abruptly increased from 4.0589 to 5.9601 and target top-1 fell from 66.17% to 57.85%. Training PPL at step 950 was still 2.9858, then rose above 5 around steps 970–1,000. Evaluation partially recovered to 5.4197 by the final record.

This looks like training destabilization rather than gradual overfitting. The output/base norm ratio stayed near 0.34, so the Output family is not established as the sole cause. Checkpoints 950 and later should not be used for comparison. The step-900 checkpoint remains the valid best record.

Required diagnostics:

1. compare checkpoints 900, 950, and 1,000;
2. log per-family residual and gradient norms;
3. check gradients and optimizer state for non-finite values;
4. record learning-rate and loss-scaler state around the transition.

## v3 — Rebuilt direct-answer dataset

v3 begins a separate dataset generation intended to remove templated chain-of-thought, reasoning tags, and source-analysis scaffolding.

### Conversion

| Item | Count |
|---|---:|
| Source records | 7,143 |
| Converted direct answers | 7,139 |
| Skipped | 4 |
| Raw converted records | 7,139 |
| Unique records | 7,103 |
| Exact duplicates removed | 36 |
| Connected groups | 6,390 |
| Train | 6,393 |
| Validation | 710 |

Direct answers were extracted from the source `<output>...</output>` field and the visible thinking content was discarded. Four missing, incomplete, or empty outputs were skipped.

### Validation report

| Check | Train | Validation |
|---|---:|---:|
| Examples | 6,393 | 710 |
| Input tokens | 1,650,954 | 196,945 |
| Supervised target tokens | 1,453,316 | 173,847 |
| Duplicate questions | 696 | 13 |
| Duplicate answers | 28 | 0 |
| Malformed records | 0 | 0 |
| Truncated records | 0 | 0 |
| Target tokens lost | 0 | 0 |

Train/validation exact overlap, question overlap, and answer overlap were all zero. The validation report passed.

### Comparability boundary

The earlier v2-family runs used a 21,791,514-byte training file and 321 steps per epoch. v3 used an 8,743,518-byte training file and 320 steps per epoch. Therefore, its PPL must not be placed in the July 19 COT leaderboard or interpreted as a regression against those values.

### Training curve

| Step | Assistant PPL | Top-1 |
|---:|---:|---:|
| 10 | 8.9998 | 49.93% |
| 100 | 7.1557 | 54.59% |
| 300 | 6.7048 | 55.71% |
| 500 | 6.6019 | 56.09% |
| 600 | **6.5713** | **56.23%** |
| 700 | 6.6688 | — |
| 760 | 6.7076 | 56.01% |

Training PPL continued to fall to 5.0068 by step 760 while validation worsened after step 600. The run shows another early-fit/late-generalization gap.

## V3 free-generation result at step 600

The owner-confirmed V3 transcript contains nine prompts. The printed `[V2]` label is stale test-harness output; run identity is V3.

Key observations:

- EOS stopping succeeded on all 9/9 prompts;
- only 2 of 6 arithmetic/consistency probes were answered correctly;
- the model failed to reject the false statement `2 × 3 = 1`;
- the ten-word constraint was ignored;
- the procrastination answer was largely off-topic;
- fluent self-help prose frequently masked weak grounding and semantic control.

The direct-answer rebuild removed some prior tag leakage, but did not establish reliable direct-answer behavior. See the [full evaluation](2026-07-20-pythia-v3-chat-evaluation.md) and [owner-supplied transcript](../../results/pythia-v3-chat-evaluation-2026-07-20.txt).

## Evidence status

| Claim | Status |
|---|---|
| v2 and v2_1 finish in an effective tie | **VALID within this protocol** |
| fixed total unique internal width matters more than its Attention/FFN location | **HYPOTHESIS** |
| wider Input/Output accelerates early v2_2 optimization | **VALID within this run** |
| v2_2 destabilized between steps 900 and 950 | **VALID observation; cause unknown** |
| v3 dataset passed the reported overlap and format checks | **VALID report** |
| v3 underperforms the July 19 variants | **NOT COMPARABLE** |
| direct-answer conversion fixed chat reliability | **NOT SUPPORTED** |
| minimum validation PPL identifies the best chat checkpoint | **NOT SUPPORTED** |

## Next actions

1. Run v3-matched Pythia-1.4B full fine-tuning and LoRA baselines on the rebuilt dataset.
2. Score V3 on a fixed, confirmed-unseen evaluation set for arithmetic, false-premise correction, brevity, relevance, hallucination, and paraphrase consistency.
3. Inspect v2_2 checkpoints around the failure boundary.
4. Add automatic best-checkpoint restoration and stop after sustained validation degradation.
5. Log per-family residual, gradient, and update norms.
6. Repeat decisive configurations over multiple seeds.
