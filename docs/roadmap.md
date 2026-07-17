# Research Roadmap

**Last updated:** 2026-07-17

The active benchmark is now **frozen Pythia-1.4B + Token MOD versus fully fine-tuned Pythia-2.8B**.

The repaired GPT-2 pipeline remains the causal-validation foundation. Its detailed gates and evidence reset are preserved in the [2026-07-16 audit](research-log/2026-07-16-pipeline-audit.md).

## Gate 0 — Architecture Integrity

**Status: passed**

- four independent MOD families verified;
- layer-specific Attention and FFN interpretation verified;
- frozen Pythia backbone unchanged;
- base gradients absent;
- original control-token rows unchanged;
- old incompatible checkpoints rejected;
- 18 focused Pythia tests passed;
- 78 full repository tests passed.

## Gate 1 — Small-Data Diagnostic

**Status: completed**

The initial Q&A run established:

- stable MOD optimization;
- taught-association learning;
- some semantic reframing;
- limited lexical stability;
- weak unseen arithmetic and procedural generalization;
- late overfitting in both MOD and full-FT variants;
- insufficient dataset coverage for the main hypothesis.

This run is diagnostic evidence, not a winner-selection benchmark.

## Gate 2 — Dataset v2

**Status: active**

Create a substantially larger, cleaner dataset with:

- separate train and held-out evaluation files;
- broad token-frequency coverage;
- multiple contexts per concept;
- deduplicated prompts;
- precise factual targets;
- controlled short and long answer styles;
- balanced factual, procedural, compositional, and reasoning categories;
- no train/eval concept leakage in the true held-out subsets.

## Gate 3 — Evaluation Suite

**Status: active**

Build category-specific evaluation for:

1. taught facts;
2. rephrased taught facts;
3. true held-out facts;
4. semantic composition;
5. concrete procedures;
6. arithmetic and rule extrapolation;
7. multi-turn context;
8. free generation and stopping;
9. untouched-pretraining retention;
10. lexical stability under prompt variation.

Exact-token PPL must be accompanied by semantic and generation evaluation.

## Gate 4 — Controlled Baselines

Run from clean starting checkpoints:

1. Pythia-1.4B untouched
2. Pythia-1.4B full fine-tuning
3. Pythia-1.4B + Token MOD
4. Pythia-2.8B untouched
5. Pythia-2.8B full fine-tuning
6. parameter-matched LoRA or adapter baseline

Hold rendering, data, target tokens, schedules, evaluation, and decoding fixed where scientifically appropriate.

## Gate 5 — Matched-Quality Comparison

Compare variants at similar held-out task quality rather than only at the same step.

Measure:

- taught-task quality;
- rephrasing and composition;
- algorithmic extrapolation;
- pretrained knowledge retention;
- language-model retention;
- stopping and repetition;
- trainable parameters;
- resident parameters.

> Can 1.466B frozen Pythia + Token MOD approach the useful behavior of fully fine-tuned 2.775B Pythia without sacrificing pretrained capability?

## Gate 6 — Systems Benchmark

Measure, rather than estimate:

- tokens per second;
- time to target quality;
- peak training VRAM;
- inference VRAM;
- first-token latency;
- cached decoding latency;
- wall-clock cost.

Compare measured results with the analytical estimate of approximately 25.84M additional MACs per generated token.

## Gate 7 — Ablations and Reproducibility

- Input-family ablation
- Output-family ablation
- Attention-family ablation
- FFN-family ablation
- shared versus layer-specific interpretation
- fixed versus learned scales
- MOD dimension sweep
- multiple seeds
- checkpoint round-trip equivalence

## Gate 8 — Continual Expansion

Only after Gates 2–7:

- sequential domain training;
- progressively frozen MODs;
- reusable versus isolated token memory;
- replay versus no replay;
- width expansion;
- depth expansion;
- interference and forgetting measurements.

## Gate 9 — Multi-MOD Routing

Introduce routing only after direct module interference is measured.

## Gate 10 — Governance and Reasoning

Integrate ownership, drift detection, rollback, bounded memory, calibration, abstention, and OOD evaluation.

## Publication Gate

Requires:

- reproducible multi-seed results;
- clean train/eval separation;
- matched budgets and quality;
- free-generation evidence;
- pretrained-retention evidence;
- measured systems cost;
- negative results and failure cases;
- confidentiality-safe reporting.

No claim of superiority should be made from the current small-data minimum PPL.
