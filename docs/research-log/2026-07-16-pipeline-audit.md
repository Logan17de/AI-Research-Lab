# 2026-07-16 — GPT-2 Pipeline Audit and Evidence Reset

## Summary

A contradiction between very low teacher-forced PPL and catastrophic free generation triggered a complete repository audit.

The audit found future-token leakage in the attention-mask path and invalidated all earlier GPT-2 chat PPL comparisons produced by that pipeline.

## Trigger

Old checkpoints reported low PPL but generated empty answers, prompt copies, repeated phrases, premature stops, and unstable continuations.

## Root Cause

When an attention mask was supplied, implicit causal behavior was disabled. The supplied mask handled padding but lacked a triangular future mask.

| Path | Broken-versus-causal maximum difference |
|---|---:|
| Base | 54.49 |
| Full | 52.70 |
| MOD | 19.28 |

## Audit Fixes

- causal future-token leakage;
- Full-FT accidental freezing;
- token-incorrect gradient accumulation;
- resume and RNG/scaler/data-position state;
- worker epoch accounting;
- nondeterministic packed pretraining;
- evaluation line cap;
- cache-compatible generation;
- optimizer grouping;
- BF16 scaler handling;
- metric categorization;
- legacy-data acceptance.

## Post-Fix Numerical Validation

| Check | Maximum difference |
|---|---:|
| HF GPT-2 equivalence, eager and SDPA | 0.0 |
| Future-token influence | 0.0 |
| Explicit versus implicit causal mask | 0.0 |
| Right-padding invariance | 0.0 |
| Batched versus individual | 1.19e-7 |
| Cached versus full-context | 8.94e-8 |
| Direct versus chat first-step | 0.0 |

Test suite: **54 passing tests**.

## Dedicated-EOT Quality Gate

| Variant | Content PPL | First PPL | EOT PPL | EOT probability | Stops | Repeat-3 |
|---|---:|---:|---:|---:|---:|---:|
| Full | 5.84 | 33.97 | 936,959 | 0.000007 | 0/20 | 42.8% |
| MOD | 6.47 | 38.88 | 440 | 0.0867 | 7/20 | 38.3% |
| LoRA | 6.06 | 23.17 | 51.1 | 0.1753 | 0/20 | 48.5% |

Every variant failed the required stop and repetition gates. The new EOT row also created an uneven burden for frozen-embedding baselines.

## Protocol Decision

Return to GPT-2's pretrained EOS token as the end of every assistant turn.

## First Repaired EOS Result

| Metric | Step 0 | Step 100 |
|---|---:|---:|
| Assistant-content PPL | 21.96 | 20.76 |
| Assistant top-1 | 41.2% | 43.2% |
| EOS PPL | 285.44 | 1.12 |
| EOS top-1 | 0% | 100% |
| Combined PPL | 25.41 | 17.59 |

The newest valid signal is rapid EOS acquisition. Model-comparison claims cannot yet be made.

## Remaining Issue

The evaluator reports first_answer n=0. This must be fixed before Full/MOD/LoRA comparison.

## Evidence Status

- old GPT-2 UltraChat PPL tables: **invalidated**;
- old long-run low-PPL curves: **invalidated**;
- dedicated-EOT gate: **valid but superseded**;
- repaired EOS step-100 result: **valid early signal**;
- MOD versus Full versus LoRA superiority: **unresolved**.

## Next Action

Fix first-answer classification, finish the EOS smoke gate, run fresh three-way quality tests, then compare pretrained-memory retention at matched assistant-content PPL.
