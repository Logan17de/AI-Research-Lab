# Research Evolution

**Last updated:** 2026-07-17

This file records major research milestones in chronological order. Detailed experimental events belong in the [dated research log](research-log/README.md).

## 2025 — FusionFormer and FusionGrammar

The first architecture branch explored partially separated grammar and meaning streams, controlled cross-stream communication, and specialized parameter roles.

**Contribution:** established the long-running themes of modularity, controlled information flow, and separate structural versus semantic evaluation.

**Limitation:** training a new architecture from scratch was expensive and difficult to benchmark with limited compute.

## 2026-06 — Dynamic Transformer

Dynamic Transformer explored token-local neuron ownership, split/merge behavior, copy-on-write specialization, and adaptive capacity growth.

**Contribution:** made continual capacity expansion concrete.

**Limitation:** dynamic routing, materialization, writeback, and Python-level operations created serious performance costs.

**Pivot:** add localized capacity to an existing pretrained transformer instead of rebuilding the whole architecture.

## 2026-06 to 2026-07 — Inverted Transformer

This branch investigated target-side credit assignment, context influence, and localized updates in a causal language model.

A Wikipedia experiment reached approximately 2.83 evaluation PPL around 8.2k steps, but the architecture did not demonstrate a clear advantage over conventional causal language modeling.

**Outcome:** deprioritized for general generation; useful as a credit-assignment study.

## 2026-06 to 2026-07 — GPT_MOD Exploratory Phase

GPT_MOD introduced high-level embedding-, attention-, and FFN-oriented trainable capacity around GPT-2 and compared it with LoRA and Full fine-tuning.

Small-data experiments suggested fast fitting, and early UltraChat runs appeared to show modifier gains.

**Important status:** these early GPT-2 SFT PPL comparisons are no longer current evidence. The 2026-07-16 audit found future-token leakage in the training attention path.

The old results remain historically useful because they motivated component ablations, LoRA-rank comparisons, generation-quality evaluation, stopping diagnostics, and retention-at-matched-SFT-quality experiments.

## 2026-07-12 to 2026-07-14 — Qwen/Tülu Exploration

A small Qwen-family model was used to test modern architecture integration and broad Tülu SFT.

The run was stable and produced assistant-like text, but exact constraints and numerical reasoning remained weak.

The untouched “Base” checkpoint already displayed assistant behavior and reasoning-shaped traces.

**Lesson:** a clean scientific baseline requires clear Base/SFT checkpoint separation.

## 2026-07-13 to 2026-07-15 — Multi-Turn SFT Design

The GPT-2 pipeline gained registered role-control tokens, assistant-only supervision, complete multi-turn retention, turn-safe truncation, train/chat sequence equivalence, and strict versioning.

A dedicated EOT token was initially used to separate assistant-turn ending from conversation ending.

## 2026-07-16 — Full Pipeline Audit and Evidence Reset

Free generation contradicted the extremely low teacher-forced PPL. A full audit found the root cause:

> Supplying a padding mask disabled implicit causal attention without adding a triangular causal mask.

Teacher-forced tokens could see future gold answers.

The audit also fixed Full-FT accidental freezing, incorrect gradient accumulation, resume state, worker accounting, nondeterministic packing, limited evaluation, generation caching, optimizer grouping, BF16 scaling, ambiguous metrics, and legacy-data handling.

After repair:

- all causal and equivalence checks were near exact;
- 54 tests passed;
- old chat checkpoints and exports were rejected;
- the evidence baseline was reset.

See [2026-07-16 Pipeline Audit](research-log/2026-07-16-pipeline-audit.md).

## 2026-07-16 — Return to Pretrained EOS

A 500-step EOT quality gate showed that the newly added stop token created an unfair problem, especially for frozen-embedding baselines.

The experiment returned to GPT-2's pretrained EOS token as the end of every assistant turn.

In the repaired Complete-MOD smoke run, EOS improved from PPL 285.44 and 0% top-1 at step 0 to PPL 1.12 and 100% top-1 at step 100.

**Outcome:** the repaired GPT-2 track established a reliable causal and evaluation foundation, but GPT-2 Small remained too limited for the main capacity comparison.

## 2026-07-17 — Pythia Scale-Controlled Token MOD

The research moved to a same-family comparison:

- frozen Pythia-1.4B plus Token MOD;
- fully fine-tuned Pythia-2.8B.

The architecture was revised from three global families into four independent high-level families: Input, Output, Attention, and FFN. Shared token memory is interpreted differently at each transformer layer.

The current configuration adds **51,593,216** trainable parameters to the 1,414,647,808-parameter backbone, producing a 1,466,241,024-parameter resident model—approximately **47.17% fewer resident parameters** than Pythia-2.8B.

Implementation validation reached:

- 18 focused Pythia tests passed;
- 78 full repository tests passed;
- frozen-base parameters and gradients remained unchanged;
- checkpoint and ablation behavior matched the revised specification.

Small-data training proved that the architecture trains and learns associations, but both the MOD model and the 2.8B dense baseline overfit the small mixed Q&A dataset.

Generation suggests an early trade-off:

- MOD is strong at taught associations and some semantic reframing;
- the larger dense model is generally stronger at practical composition and limited rule extrapolation;
- neither model is a clean winner.

**Current position:** implementation validity is established; competitive model quality is unresolved.

See [2026-07-17 Pythia Token MOD Findings](research-log/2026-07-17-pythia-token-mod.md).

## Future Branches

These remain proposed, not validated:

- larger, cleaner Pythia training and held-out evaluation;
- matched-quality Pythia retention tests;
- parameter- and compute-matched baselines;
- progressive frozen MODs;
- equal-capacity reusable versus isolated modules;
- width expansion;
- depth expansion;
- multi-domain routing;
- Capability Gap Recovery on clean modern checkpoints;
- governed memory and reasoning integration.
