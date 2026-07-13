# AI Research Lab

**Independent AI research by Logeshkumar Duraisamy (Logan), based in Japan.**

This repository tracks an evolving research program on **continual learning**, **localized credit assignment**, **parameter-efficient adaptation**, and **governed machine reasoning**.

> **Central question:** Can a language model learn new knowledge quickly while limiting interference with what it already knows?

## Current Focus

- Token-conditioned and token-local adaptation
- Continual learning with reduced catastrophic forgetting
- Controlled comparisons with full fine-tuning and LoRA
- Modular domain adaptation and routing
- Memory governance, recovery, and calibrated reasoning
- Compute-aware experiments using GPT-2 and Qwen-family models

## Research Program

| Branch | Purpose | Current status |
|---|---|---|
| **GPT_MOD** | Test token-conditioned modifier modules against LoRA and full fine-tuning | Active |
| **Multi-MOD Router** | Compose frozen domain modules through learned routing | Planned experiments |
| **Qwen + Tulu SFT** | Validate the approach on a modern pretrained architecture and general instruction data | Active |
| **Governed Memory & Reasoning** | Build stable, recoverable memory before increasing reasoning autonomy | Active |
| **Inverted Transformer** | Explore localized target-side credit assignment | Prototype evaluated; deprioritized |
| **Dynamic Transformer** | Explore split/merge neurons and token-local adaptive capacity | Early exploration; paused |
| **FusionFormer / FusionGrammar** | Separate grammar and meaning into interacting streams | Foundational earlier work |

## Strongest Current Evidence

A corrected, fixed-budget UltraChat SFT ablation on GPT-2 Small produced:

| Model | Training budget | Best eval PPL |
|---|---:|---:|
| LoRA | ~8,000 optimizer steps | 8.313 |
| LoRA + FFN-oriented MOD | ~8,000 optimizer steps | 8.063 |
| Complete: LoRA + FFN MOD + Embedding MOD | ~8,000 optimizer steps | 8.033 |

At the same step budget:

- Adding the FFN-oriented MOD improved evaluation perplexity by **0.250** over LoRA.
- Adding the embedding-oriented MOD after that improved it by another **0.030**.
- The FFN-oriented component therefore accounted for most of the measured modifier gain in this experiment.
- The Complete model achieved approximately **3.4% lower eval PPL** than LoRA alone.

This establishes a measurable ablation result, but not yet parameter efficiency or broad generalization. A higher-rank LoRA may recover the same gain with fewer parameters.

GPT-2 Medium + LoRA later reached approximately **6.142 eval PPL** at ~12,000 steps, confirming that the modifier did not turn GPT-2 Small into GPT-2 Medium. Medium was also much slower in the recorded setup: approximately **31k tokens/s** versus **70k tokens/s** for the Small models.

Qualitative testing showed that lower PPL did not reliably produce stronger assistant behavior. Every model still struggled with strict formatting, long-range coherence, instruction grounding, and arithmetic.

These results remain **preliminary, dataset-dependent, single-run evidence and are not peer-reviewed**.

Read the full [GPT-2 UltraChat Ablation](docs/gpt2-ultrachat-ablation.md) and [Experimental Evidence](docs/experimental-evidence.md).

## Research Evolution

The work has progressed through a sequence of connected questions:

1. Can grammar and meaning be represented in specialized streams?
2. Can model capacity grow dynamically through local neuron ownership?
3. Can credit assignment be localized to reduce interference?
4. Can a frozen pretrained model gain token-conditioned adaptive capacity?
5. Does this capacity outperform simply increasing LoRA rank?
6. Can multiple learned domain modules be routed and composed?
7. Can memory and reasoning remain stable, auditable, and recoverable during continual updates?

See [Research Evolution](docs/research-evolution.md) for the detailed lineage.

## Evaluation Standard

Experiments aim to control:

- pretrained backbone;
- dataset and held-out split;
- sequence length and token budget;
- optimizer steps and learning-rate schedule;
- trainable parameter count;
- random seeds;
- decoding configuration;
- data-loader sharding;
- VRAM, throughput, and wall-clock time.

Primary outcomes include:

- held-out loss and perplexity;
- convergence speed;
- generation and instruction-following quality;
- EOS behavior;
- sequential-learning retention;
- catastrophic forgetting;
- parameter and compute efficiency;
- behavioral drift.

## Current Roadmap

- Run LoRA rank sweeps under the same ~8,000-step UltraChat budget
- Match trainable parameters and compute, not only training steps
- Repeat the ablation across multiple seeds
- Add automated instruction-following, coherence, format, and EOS evaluation
- Extend controlled tests to Qwen-family base models
- Test old-versus-new knowledge retention after sequential training
- Train isolated domain MODs and evaluate learned routing/composition
- Separate memorization, adaptation, and genuine generalization

See [Research Roadmap](docs/roadmap.md) for the staged plan.

## Research Principles

1. **Evidence before claims** — a promising loss curve is not a conclusion.
2. **Controlled comparisons** — architecture changes need matched conditions.
3. **Retention matters** — new learning is incomplete if old capabilities silently disappear.
4. **Perplexity is not assistant quality** — prediction fit and instruction following are different outcomes.
5. **Governance before autonomy** — memory and reasoning should be inspectable and recoverable.
6. **Compute-aware research** — useful ideas should be testable at small scale before expensive scaling.
7. **Confidentiality where necessary** — findings can be documented without exposing proprietary implementation details.

## Confidentiality

This repository documents research goals, experimental structure, observations, and high-level architecture families. **Exact modifier placement, implementation mechanics, and proprietary usage details are intentionally excluded.**

## Status

This is active independent research. Architectures, terminology, results, and conclusions will change as stronger evidence becomes available.

The present question is no longer merely whether the modifier lowers loss. It is:

> **Does token-conditioned FFN adaptation provide a distinct or more efficient learning path than increasing conventional low-rank adaptation capacity?**

## Contact

For research discussion, benchmarking, collaboration, or compute sponsorship, open an issue in this repository or contact me through my GitHub profile.
