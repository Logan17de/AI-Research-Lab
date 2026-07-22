# AI Research Lab

**Independent AI research by Logeshkumar Duraisamy (Logan), based in Japan.**

**Last updated:** 2026-07-22

This repository tracks research on **continual learning**, **localized adaptation**, **parameter-efficient training**, **capacity expansion**, and **governed machine reasoning**.

> **Central question:** Can a smaller frozen pretrained model gain useful new capacity while preserving access to its original knowledge—and compete with a much larger densely trained model?

## Current Experiment

The research now tracks three distinct benchmark generations: the original chain-of-thought Q&A split, the rebuilt direct-answer V3 split, and the new locked filtered-UltraChat benchmark. Metrics must not be compared across dataset generations.

The MOD architecture contains four independent high-level families:

1. Input
2. Output
3. Attention
4. FFN

Attention and FFN token-memory tables may be shared across layers or grouped into layer-specific tables. Transformer layers retain distinct projections. Pythia's frozen language-model head and native residual structure remain intact.

Exact modifier placement and proprietary implementation mechanics remain intentionally excluded.

## Latest Variant Sweep — 2026-07-19

Dimensions are Input / Output / Attention / FFN.

| Variant | Dimensions | Internal tables | Trainable parameters | Best assistant PPL | Best step |
|---|---:|---|---:|---:|---:|
| v1_1 | 128 / 128 / 128 / 256 | shared | 51.6M | 4.0188 | 1,150 |
| v1_2 | 256 / 256 / 256 / 512 | shared | 103.2M | 3.9859 | 950 |
| v1_3 | 256 / 256 / 256 / 512 | shared | 103.2M | **3.9724** | 950 |
| v1_4 | 256 / 256 / 512 / 1,024 | shared | 179.6M | **3.9724** | 600 |
| v2 | 128 / 128 / 64 / 128 | 24 layer-unique | 254.6M | 4.0429 | 950 |
| v2_1 | 128 / 128 / 128 / 64 | 24 layer-unique | 254.6M | **4.0420** | 950 |
| v2_2 | 512 / 512 / 64 / 64 | 24 layer-unique | 214.4M | 4.0589 | 900 |
| Pythia-2.8B full FT | — | dense | 2.775B | **3.5190** | 2,250 |

The principal findings are:

- **v1_3 is the most parameter-efficient best-performing MOD** in the sweep.
- v1_4 reaches the same validation floor earlier, but additional shared width does not lower that floor.
- v2 uses approximately 2.47 times v1_3's trainable parameters and performs worse on the current dataset.
- v2 and completed v2_1 are effectively tied at step 950 despite reversing Attention and FFN widths.
- v2_2 learned faster early with wider Input/Output paths, but abruptly destabilized between steps 900 and 950; only step 900 is retained as its valid best checkpoint.
- training PPL continues falling after validation PPL turns upward, so the approximately 4.05 unique-table minimum is a generalization/overfitting boundary under this setup—not a demonstrated hard capacity wall.
- Pythia-2.8B full fine-tuning remains the quality leader, while running approximately 2.3 times slower per optimization step at the same effective batch size.

The current evidence suggests that cross-layer table sharing improves sample efficiency and acts as a useful regularizing inductive bias. Layer-unique tables may require more data or a different width/count allocation.

See the full [2026-07-19 Pythia Variant Sweep](docs/research-log/2026-07-19-pythia-variant-sweep.md) and its [machine-readable summary](results/pythia-variant-sweep-2026-07-19.csv).

## Follow-up Variants and Direct-Answer V3 — 2026-07-20

The completed follow-up adds three results:

- **v2_1** finished at assistant PPL 4.0420, effectively tying v2 and showing no measurable benefit from swapping a fixed unique-table width budget between Attention and FFN;
- **v2_2** accelerated early optimization through 512-wide Input/Output paths, then suffered an abrupt training destabilization after step 900;
- **v3** started a separate direct-answer dataset generation with 7,103 unique records, zero reported train/validation overlap, and a best assistant PPL of 6.5713 at step 600.

V3's PPL is not comparable to the July 19 leaderboard. The owner-confirmed V3 step-600 chat test produced 9/9 EOS stops but only 2/6 correct arithmetic/consistency responses, accepted the false premise `2 × 3 = 1`, ignored the ten-word constraint, and drifted off-topic on procrastination. V3 is fluent but not reliable enough for user-facing chat.

See the [2026-07-20 Follow-up Variants](docs/research-log/2026-07-20-pythia-follow-up-variants.md), [V3 Chat Evaluation](docs/research-log/2026-07-20-pythia-v3-chat-evaluation.md), and [machine-readable metrics](results/pythia-follow-up-variants-2026-07-20.csv).

## Locked UltraChat Plasticity Benchmark — 2026-07-22

A deterministic English UltraChat manifest now provides the cleanest large-data comparison:

| Split | Examples | Tokens |
|---|---:|---:|
| Train | 72,000 | 49,777,171 |
| Validation | 1,500 | 1,019,539 |
| Test | 1,500 | 1,033,646 |

The manifest fixes filtering, category quotas, tokenizer identity, file hashes, token order, sample order, seed, and loader behavior.

Current results:

| Variant | Snapshot | Best assistant PPL | Best step | Interpretation |
|---|---|---:|---:|---|
| Pythia-1.4B MOD + plastic-4 | completed to 1,350 | **3.5608** | 950 | limited tail plasticity; mild late plateau |
| Pythia-1.4B MOD + plastic-24 | active through 240 | **3.6520** | 200 | early snapshot; nearly full-backbone hybrid |
| Pythia-2.8B full FT | completed to 1,450 | **3.2402** | 1,450 | current validation leader |

At matched step 200, assistant PPL was 3.6970, 3.6520, and 3.3908 respectively.

The 2.8B dense model reached 3.5014 by step 100, already below the plastic-4 model's eventual best. This means full fine-tuning is currently more sample-efficient on the locked objective.

Plastic-4 selected 201.4M original base parameters. Plastic-24 selected 1.312B—approximately 92.7% of Pythia-1.4B—so plastic-24 should not be presented as a strongly parameter-efficient baseline.

Recorded throughput favored the smaller hybrids, but the runs used different reported GPU environments. No controlled speed multiplier is claimed.

The new Adaptive Transformer Expansion branch has also been implemented and initialized. Its first run has no metrics yet, so ATE remains **Proposed / Initialized**, not a model-quality result.

See the full [2026-07-22 Locked UltraChat Plasticity and ATE Analysis](docs/research-log/2026-07-22-ultrachat-plasticity-ate.md) and [machine-readable snapshot](results/pythia-ultrachat-plasticity-2026-07-22.csv).

## Qualitative Chat Test — 2026-07-19

A two-prompt free-generation comparison exposed a major mismatch between validation perplexity and deployed behavior:

- **Pythia-2.8B full FT** was the most consistently polished and aligned model.
- **v1_2 / 256_MOD** produced the strongest practical MOD answers despite not owning the best MOD PPL.
- **v1_3 / 256_MOD_epoch_7** failed to produce a proper user-facing output on both prompts, even though it shares the best MOD validation PPL.
- **v2 layer-unique** showed role confusion, irrelevant recommendations, and internal meta-tag leakage on the teacher/student prompt.
- every variant referred to nonexistent source material, articles, books, images, standards, or experts.
- EOS stopping was reliable, but answer presence, tag validity, grounding, and role consistency were not.

This test is exploratory because it contains only two prompts. It nevertheless proves that minimum teacher-forced PPL cannot be the sole checkpoint-selection criterion.

See the full [Pythia Chat Comparison](docs/research-log/2026-07-19-pythia-chat-comparison.md) and [unedited transcript](results/pythia-chat-comparison-2026-07-19.txt).

## Implementation Status

The architecture verifies:

- a strictly frozen Pythia-1.4B backbone;
- independent input and output capacity for Pythia's untied weights;
- grouped Attention and FFN token-memory tables;
- contiguous layer grouping with strict divisibility;
- layer-specific Attention and FFN projections;
- one combined-logit vocabulary softmax;
- zero-effect, gradient-safe initialization;
- sparse MOD checkpoints and runtime ablations;
- exact group-map checkpoint persistence;
- rejection of incompatible older checkpoints.

Earlier validation reached 18 focused Pythia tests and 78 full repository tests. The grouped-table revision requires its own updated test-count record before the repository claims a newer total.

## Evidence Foundation

On 2026-07-16, a full GPT-2 repository audit discovered future-token leakage in the old attention-mask path. Earlier GPT-2 SFT perplexity comparisons—including 8.313 / 8.063 / 8.033—were invalidated as causal language-model evidence.

The repaired causal pipeline reached near-exact numerical equivalence checks and 54 passing tests. That audit remains the foundation for all newer experiments.

Historical results remain documented with explicit **Valid**, **Exploratory**, **Superseded**, **Invalidated**, or **Proposed** status.

## Current Conclusion

Established:

- the grouped Token MOD architecture is operational;
- frozen Pythia-1.4B can adapt through MOD-only training;
- the best current shared MOD reaches approximately 3.972 assistant validation PPL;
- shared tables outperform the current 24-table unique configuration;
- additional shared width improves early learning speed more than the final validation minimum;
- all evaluated MOD variants eventually show a train/eval generalization gap;
- the 2.8B dense baseline remains stronger on held-out perplexity;
- on locked UltraChat, plastic-4 reaches 3.5608 while 2.8B full FT reaches 3.2402;
- limited plasticity narrows but does not close the quality gap;
- ATE is implemented but has no measured result.

Not established:

- shared tables are universally better than layer-unique tables;
- Attention and FFN MOD capacity are functionally interchangeable;
- Input/Output MOD dominates internal MODs;
- MOD matches or beats fully fine-tuned Pythia-2.8B;
- MOD or controlled plasticity preserves pretrained knowledge better;
- broad plasticity remains parameter-efficient;
- ATE improves quality, efficiency, or retention;
- current results prove continual-learning superiority.

## Next Experiment

The next controlled work requires:

- pure frozen MOD, Pythia-1.4B Full, LoRA, plastic-4, plastic-24, and Pythia-2.8B Full baselines on the locked UltraChat manifest;
- a fixed scored chat suite covering arithmetic, false-premise correction, brevity, relevance, hallucination, and paraphrase consistency;
- v2_2 checkpoint and optimizer diagnostics around steps 900–1,000;
- identical dimensions and schedules with unique-table counts 1, 4, 8, and 24;
- a separate parameter-matched count sweep;
- Input+Output-only, Attention, FFN, and complete-family ablations;
- per-family gradient norms and residual-to-base ratios;
- automatic best-checkpoint restoration;
- Pythia-1.4B Full and parameter-matched LoRA baselines;
- multiple seeds;
- untouched-test, pretrained-retention, and free-generation evaluation;
- identical-hardware systems measurements;
- ATE parameter accounting, neutral-initialization validation, and first controlled run.

## Documentation

- [Research Evolution](docs/research-evolution.md)
- [Experimental Evidence Ledger](docs/experimental-evidence.md)
- [Current Research Roadmap](docs/roadmap.md)
- [Dated Research Log](docs/research-log/README.md)
- [2026-07-22 Locked UltraChat Plasticity and ATE Analysis](docs/research-log/2026-07-22-ultrachat-plasticity-ate.md)
- [2026-07-20 Pythia Follow-up Variants](docs/research-log/2026-07-20-pythia-follow-up-variants.md)
- [2026-07-20 Pythia V3 Chat Evaluation](docs/research-log/2026-07-20-pythia-v3-chat-evaluation.md)
- [2026-07-19 Pythia Variant Sweep](docs/research-log/2026-07-19-pythia-variant-sweep.md)
- [2026-07-17 Pythia Token MOD Findings](docs/research-log/2026-07-17-pythia-token-mod.md)
- [2026-07-16 GPT-2 Pipeline Audit](docs/research-log/2026-07-16-pipeline-audit.md)
- [Current SFT Pipeline Validation](docs/sft-pipeline-validation.md)
- [Continual Expansion Proposal](docs/continual-expansion.md)
- [Superseded UltraChat Ablation](docs/gpt2-ultrachat-ablation.md)

## Research Principles

1. **Causal validity before perplexity**
2. **Free generation before claims**
3. **Evidence status must be explicit**
4. **Matched quality matters more than matched steps**
5. **Retention must be separated from task fitting**
6. **Measured systems cost must be separated from analytical estimates**
7. **Negative, superseded, and invalidated results remain documented**
8. **Proprietary implementation details remain confidential**

## Contact

For research discussion, benchmarking, collaboration, or compute sponsorship, open an issue in this repository or contact me through my GitHub profile.
