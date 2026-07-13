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

## Current Evidence

Preliminary controlled experiments have produced several useful signals:

- Modifier variants involving the transformation path learned the small 100-QA SFT dataset faster than LoRA r=32 in the tested setup.
- On that narrow dataset, some modifier variants reached approximately **1–2 perplexity** after extended training, while LoRA r=32 stabilized around **5–6**.
- In an early one-epoch UltraChat comparison using GPT-2 Medium, the FFN-oriented modifier reached roughly **8.0 perplexity**, compared with roughly **8.3** for LoRA.
- Modifier capacity was dataset-dependent: a medium embedding-modifier dimension worked better than both smaller and larger settings on the narrow experiment.
- The Inverted Transformer achieved approximately **2.83 evaluation perplexity** in one Wikipedia experiment, but it did not demonstrate a clear advantage over a standard causal language model.
- Early Qwen/Tulu results are promising enough to continue testing, but are far too immature for strong conclusions.

These results are **preliminary, dataset-dependent, and not peer-reviewed**. Small datasets and many epochs can reward memorization, so held-out evaluation, multiple seeds, retention tests, and broader benchmarks remain mandatory.

## Research Evolution

The work has progressed through a sequence of connected questions:

1. Can grammar and meaning be represented in specialized streams?
2. Can model capacity grow dynamically through local neuron ownership?
3. Can credit assignment be localized to reduce interference?
4. Can a frozen pretrained model gain token-conditioned adaptive capacity?
5. Can multiple learned domain modules be routed and composed?
6. Can memory and reasoning remain stable, auditable, and recoverable during continual updates?

See [Research Evolution](docs/research-evolution.md) for the detailed lineage.

## Evaluation Standard

Experiments aim to control:

- pretrained backbone;
- dataset and held-out split;
- sequence length and token budget;
- training steps and optimizer schedule;
- trainable parameter count;
- random seeds;
- decoding configuration;
- VRAM, throughput, and wall-clock time.

Primary outcomes include:

- held-out loss and perplexity;
- convergence speed;
- generation quality;
- sequential-learning retention;
- catastrophic forgetting;
- parameter and compute efficiency;
- behavioral drift.

See [Experimental Evidence](docs/experimental-evidence.md) for recorded results and limitations.

## Current Roadmap

- Complete matched GPT-2 full fine-tuning versus MOD comparisons
- Run broader and multi-seed LoRA rank comparisons
- Extend tests to Qwen-family base models
- Evaluate individual and combined modifier families
- Test old-versus-new knowledge retention after sequential training
- Train isolated domain MODs and evaluate learned routing/composition
- Separate memorization, adaptation, and genuine generalization
- Prepare reproducible reports suitable for research collaboration

See [Research Roadmap](docs/roadmap.md) for the staged plan.

## Research Principles

1. **Evidence before claims** — a promising loss curve is not a conclusion.
2. **Controlled comparisons** — architecture changes need matched conditions.
3. **Retention matters** — new learning is incomplete if old capabilities silently disappear.
4. **Governance before autonomy** — memory and reasoning should be inspectable and recoverable.
5. **Compute-aware research** — useful ideas should be testable at small scale before expensive scaling.
6. **Confidentiality where necessary** — findings can be documented without exposing proprietary implementation details.

## Confidentiality

This repository documents research goals, experimental structure, observations, and high-level architecture families. **Exact modifier placement, implementation mechanics, and proprietary usage details are intentionally excluded.**

## Status

This is active independent research. Architectures, terminology, results, and conclusions will change as stronger evidence becomes available.

The objective is not to prematurely declare a replacement for established fine-tuning methods. It is to determine precisely **when localized token-conditioned adaptation helps, when it does not, and why**.

## Contact

For research discussion, benchmarking, collaboration, or compute sponsorship, open an issue in this repository or contact me through my GitHub profile.
