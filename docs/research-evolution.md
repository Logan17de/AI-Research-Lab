# Research Evolution

**Last updated:** 2026-07-22

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

## 2026-07-18 to 2026-07-19 — Shared and Layer-Unique MOD Sweep

The Pythia benchmark moved from one 51.6M-parameter MOD configuration to a controlled family of shared-width and layer-unique-table variants.

The architecture gained independently configurable Attention and FFN table counts, contiguous layer grouping, strict divisibility validation, exact layer-map checkpoint persistence, and startup reporting.

The first sweep produced three important signals:

1. v1_3 and v1_4 both reached approximately 3.972 assistant validation PPL;
2. increasing shared internal width improved early learning speed but did not lower the final validation minimum;
3. the 24-table layer-unique v2 reached approximately 4.043 despite using approximately 254.6M trainable parameters.

A follow-up v2_1 run reversed Attention and FFN widths while preserving the same total parameter count. Its curve was nearly identical to v2 through step 400.

All mature MOD runs showed training improvement after validation had begun worsening. The approximately 4.05 unique-table result is therefore treated as a generalization optimum under the current protocol, not a hard storage-capacity wall.

**Current position:** cross-layer sharing appears to improve sample efficiency and regularization. A fixed-dimension table-count sweep, parameter-matched count sweep, and module-specific gradient/residual telemetry are required before attributing the result to uniqueness itself.

See [2026-07-19 Pythia Variant Sweep](research-log/2026-07-19-pythia-variant-sweep.md).

## 2026-07-20 — Direct-Answer Dataset Pivot and Reliability Check

The layer-unique follow-up closed with v2 and v2_1 effectively tied at approximately 4.042 assistant PPL. v2_2 improved early learning speed with wider Input/Output paths, but abruptly destabilized after step 900.

The research then created a new dataset generation by extracting direct answers and discarding visible thinking. The rebuild produced 7,103 unique records with reported zero train/validation overlap. V3 reached its best validation point at step 600.

The owner-confirmed V3 chat test exposed a new boundary: reasoning tags were less prominent, but direct answers remained unreliable. V3 stopped correctly on all nine prompts yet failed elementary arithmetic, false-premise correction, exact length control, and topic alignment.

**Current position:** direct-answer formatting alone does not solve semantic grounding. V3 requires matched baselines on its own data rather than comparison with the earlier COT leaderboard.

See [2026-07-20 Follow-up Variants](research-log/2026-07-20-pythia-follow-up-variants.md) and [V3 Chat Evaluation](research-log/2026-07-20-pythia-v3-chat-evaluation.md).

## 2026-07-21 to 2026-07-22 — Locked UltraChat, Controlled Plasticity, and ATE

The benchmark moved to a locked English UltraChat subset with 75,000 conversations: 72,000 train, 1,500 validation, and 1,500 untouched test examples. The manifest fixes sample order, tokenizer, split hashes, category quotas, and data-loader behavior.

The first fresh-run comparison produced:

- Pythia-1.4B + MOD + final-four-layer plasticity: assistant PPL **3.5608** at step 950;
- Pythia-1.4B + MOD + broad 24-layer plasticity: **3.5511** at step 500;
- fresh ATE h1/l1: **3.4602** at step 400;
- Pythia-2.8B full fine-tuning: **3.2402** at step 1,450.

At matched step 200, assistant PPL was 3.6970 for plastic-4, 3.6520 for plastic-24, 3.5475 for ATE h1/l1, and 3.3908 for the 2.8B full-FT control.

Plastic-24 made approximately 92.7% of the original Pythia-1.4B backbone plastic yet improved over plastic-4 by only 0.0097 at their best checkpoints. This strengthened plastic-4's role as the cleaner hybrid efficiency result.

Fresh ATE h1/l1 expanded width from 2,048 to 2,176, heads from 16 to 17, and depth from 24 to 25 layers. It added 225,479,552 parameters for 1,640,127,360 total and reached the strongest fresh Pythia-1.4B-derived validation result. Its current quadratic-plasticity configuration trains the entire expanded model, so it is architecture expansion rather than PEFT.

## 2026-07-22 — Incremental ATE Depth and Width Expansion

The research then moved from one-shot expansion to sequential growth.

ATE h1/l2 inherited the h1/l1 step-400 validation state of 3.4602 and added a second new layer. It reached 3.4358 at local step 450, an approximately 0.71% improvement over the inherited value.

This result created a new experimental distinction:

- **fresh architecture comparison** asks whether h1/l1 beats other models from a common pretrained start;
- **sequential expansion comparison** asks whether a trained expanded model can gain additional capacity without losing its learned state.

H1/l2 is promising evidence for the second question, but not yet proof that the extra layer caused the improvement. Its best represents approximately 94.0M cumulative source-plus-local training tokens. A matched h1/l1 continuation with the same optimizer reset, data restart, and additional-token budget is required.

After local step 450, h1/l2 validation PPL worsened to 3.6970 by step 600 while training PPL fell from 3.4007 to 2.7604. The branch therefore established a sharp one-epoch generalization boundary. Step 450 remains the valid selected checkpoint.

ATE h2/l2 began with two added heads and two added layers, but only one step-10 metric row exists and no checkpoint was saved. It remains unranked.

A new reproducibility requirement emerged: incremental checkpoints must persist the exact source path or hash and record a step-zero equivalence evaluation before any optimizer update.

**Current position:** Pythia-2.8B full FT remains the absolute-quality leader. ATE h1/l1 leads the matched fresh Pythia-1.4B-derived track. ATE h1/l2 is the best observed sequential checkpoint, but causal attribution to added depth remains unresolved. Retention, scored generation, same-hardware systems tests, matched no-expansion continuation, LoRA, pure frozen MOD, and multiple seeds remain required.

See the [ATE Sequential Expansion Sweep](research-log/2026-07-22-ate-expansion-sweep.md) and [Model Ranking Framework](research-log/2026-07-22-model-ranking-framework.md).

## Future Branches

These remain proposed, not validated:

- larger, cleaner Pythia training and held-out evaluation;
- matched-quality Pythia retention tests;
- parameter- and compute-matched baselines;
- progressive frozen MODs;
- equal-capacity reusable versus isolated modules;
- ATE width/depth evaluation;
- expansion-versus-MOD and expansion-versus-LoRA controls;
- multi-domain routing;
- Capability Gap Recovery on clean modern checkpoints;
- governed memory and reasoning integration.
