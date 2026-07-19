# Research Roadmap

**Last updated:** 2026-07-20

The active benchmark is **frozen Pythia-1.4B + Token MOD versus fully fine-tuned Pythia-2.8B**, followed by controlled shared-versus-layer-unique table ablations.

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

**Status: active — first direct-answer rebuild completed; behavioral quality gate failed**

The first rebuild converted 7,139 direct answers, retained 7,103 unique records, and reported zero train/validation overlap. V3 trained on this new split, so its PPL is a separate series. The dataset still needs stronger coverage of elementary arithmetic, direct instruction following, and topic-specific answers.

Continue toward a substantially larger, cleaner dataset with:

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

The first six-checkpoint chat comparison found:

- v1_2 was the strongest practical MOD on two general-advice prompts;
- v1_3 failed to emit a proper user-facing answer twice despite leading MOD PPL;
- V2 showed role confusion and internal meta-tag leakage;
- all variants hallucinated nonexistent source context;
- EOS stopping was reliable but insufficient as a generation-quality gate.

A V3 step-600 follow-up stopped correctly on all nine prompts but answered only two of six arithmetic/consistency probes correctly, accepted a false arithmetic premise, ignored a ten-word constraint, and drifted off-topic. The evaluation suite must therefore score answer presence, tag validity, reasoning/meta leakage, role correctness, unsupported attribution, relevance, factuality, arithmetic, false-premise correction, exact constraints, and paraphrase consistency independently.

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
5. Pythia-2.8B full fine-tuning — **completed for the current chain-of-thought split**
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

**Status: active; first width/table sweep completed**

The 2026-07-19 sweep established:

- shared v1_3 and v1_4 both reached approximately 3.972 assistant PPL;
- v1_3 is the most parameter-efficient best-performing current MOD;
- extra shared width accelerated early learning without lowering the validation minimum;
- the 24-table unique v2 reached approximately 4.043 despite using more trainable parameters;
- completed v2 and v2_1 were effectively tied at step 950 after swapping Attention and FFN widths;
- v2_2 learned faster early with wider Input/Output paths but destabilized abruptly after step 900;
- all mature MOD curves showed validation overfitting after their best checkpoint.

Required next:

- fixed-dimension unique-count sweep: 1, 4, 8, and 24 tables;
- separate parameter-matched count sweep;
- Input+Output-only baseline;
- Attention-only and FFN-only additions;
- complete-family ablation;
- per-family gradient norms;
- per-family residual-to-base norms;
- fixed versus learned scales;
- multiple seeds;
- automatic best-checkpoint restoration;
- checkpoint round-trip equivalence.

See [2026-07-19 Pythia Variant Sweep](research-log/2026-07-19-pythia-variant-sweep.md) and [2026-07-20 Follow-up Variants](research-log/2026-07-20-pythia-follow-up-variants.md).

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
