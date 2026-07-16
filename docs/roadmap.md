# Research Roadmap

**Last updated:** 2026-07-16

Long runs must not begin until the current evidence gate passes.

## Gate 0 — Pipeline Integrity

**Status: passed**

- future-token influence = 0.0;
- HF GPT-2 equivalence = 0.0;
- cached versus full-context difference = 8.94e-8;
- direct versus chat first-step difference = 0.0;
- 54 tests passing;
- legacy data and checkpoints rejected.

## Gate 1 — Evaluator Completion

**Status: active**

Fix first_answer n=0 / PPL=NaN.

Verify category totals, token weighting, and step-zero persistence.

## Gate 2 — Single-Variant EOS Smoke Run

**Status: active**

Pass conditions:

- EOS stop rate ≥80%;
- repeated-trigram rate ≤25%;
- low premature-stop rate;
- improving assistant-content PPL;
- healthy free greedy generations;
- reliable first-answer metric.

## Gate 3 — Matched Three-Way Quality Test

Run from fresh checkpoints:

1. Full fine-tuning
2. Complete MOD
3. LoRA

Hold data, causal rendering, masking, sequence length, batch size, milestones, evaluation, and decoding fixed.

No long training if any variant fails generation gates.

## Gate 4 — Matched-SFT-Quality Retention

Compare at similar assistant-content PPL.

Measure factual completion, original-corpus PPL, instruction following, prompt conditioning, multi-turn context, EOS behavior, repetition, and forgetting.

> Does MOD preserve and use pretrained knowledge better than Full fine-tuning at comparable SFT quality?

## Gate 5 — Long Tülu Comparison

Begin only after Gates 1–4 pass, using fresh versioned exports and checkpoints.

## Gate 6 — LoRA Capacity and Component Ablations

- LoRA rank sweep
- individual MOD-family ablations
- parameter-matched comparison
- compute-matched comparison
- multiple seeds
- throughput, VRAM, and wall-clock reporting

## Gate 7 — Modern Backbone Validation

Use clear Base and official SFT checkpoint separation. Candidates include OLMo 2 and SmolLM2.

## Gate 8 — Continual Expansion

Compare full sequential fine-tuning, reusable MOD, progressively frozen MODs, width expansion, depth expansion, and replay.

## Gate 9 — Multi-MOD Routing

Introduce routing only after direct module interference is measured.

## Gate 10 — Governance and Reasoning

Integrate ownership, drift detection, rollback, bounded memory, calibration, abstention, and OOD evaluation.

## Publication Gate

Requires a repaired causal pipeline, reproducible multi-seed runs, matched budgets, successful free generation, knowledge-retention evidence, negative results, and confidentiality-safe reporting.
