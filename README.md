# AI Research Lab

**Independent AI research by Logeshkumar Duraisamy (Logan), based in Japan.**

**Last updated:** 2026-07-16

This repository tracks research on **continual learning**, **localized adaptation**, **parameter-efficient training**, **capacity expansion**, and **governed machine reasoning**.

> **Central question:** Can a model learn new instructions and knowledge while preserving effective access to what its frozen pretrained backbone already knows?

## Current Experiment

| Variant | Trainable path |
|---|---|
| Full fine-tuning | Original GPT-2 parameters |
| Complete MOD | Added token-conditioned capacity with frozen GPT-2 |
| LoRA | Low-rank adaptation with frozen GPT-2 |

All variants must use the same starting checkpoint, chat renderer, masking, dataset split, supervised-target budget, evaluation samples, and generation settings.

## Evidence Reset — 2026-07-16

A full repository audit discovered **future-token leakage in the old attention-mask path**. Teacher-forced training could see future answer tokens, while free generation could not.

Therefore, earlier GPT-2 SFT perplexity comparisons—including the former 8.313 / 8.063 / 8.033 UltraChat table—are **invalid as causal language-model evidence**.

They remain documented only as superseded research history.

The repaired implementation now verifies:

| Numerical check | Maximum difference |
|---|---:|
| Hugging Face GPT-2 equivalence, eager and SDPA | 0.0 |
| Future-token influence | 0.0 |
| Explicit versus implicit causal mask | 0.0 |
| Right-padding invariance | 0.0 |
| Batched versus individual inference | 1.19e-7 |
| Cached versus full-context decoding | 8.94e-8 |
| Direct versus chat first-step logits | 0.0 |

The test suite now has **54 passing tests**.

## Current Valid Signal

The current format reuses GPT-2's pretrained EOS token as the end of each assistant turn.

In the first repaired Complete-MOD smoke run:

| Metric | Step 0 | Step 100 |
|---|---:|---:|
| Assistant-content PPL | 21.96 | 20.76 |
| Assistant-content top-1 | 41.2% | 43.2% |
| EOS PPL | 285.44 | 1.12 |
| EOS top-1 | 0% | 100% |
| Combined PPL | 25.41 | 17.59 |

This is the newest valid result. It shows that the pretrained EOS representation learned turn termination rapidly, while assistant-content quality improved more gradually.

It does **not** establish that MOD is superior to Full fine-tuning or LoRA.

## Current Quality Gates

Before any long Tülu run, Full, MOD, and LoRA must all pass:

- EOS stop rate ≥80%;
- repeated-trigram rate ≤25%;
- improving assistant-content PPL;
- reliable first-answer-token measurement;
- low premature-stop rate;
- healthy free greedy generation.

The next immediate fix is the evaluator's first_answer n=0 classification bug.

## Documentation

- [Research Evolution](docs/research-evolution.md) — dated architecture and experiment milestones
- [Experimental Evidence](docs/experimental-evidence.md) — valid, superseded, and invalidated claims
- [Current Research Roadmap](docs/roadmap.md)
- [Current SFT Pipeline Validation](docs/sft-pipeline-validation.md)
- [Dated Research Log](docs/research-log/README.md)
- [2026-07-16 Pipeline Audit](docs/research-log/2026-07-16-pipeline-audit.md)
- [Continual Expansion Proposal](docs/continual-expansion.md)
- [Superseded UltraChat Ablation](docs/gpt2-ultrachat-ablation.md)

## Research Principles

1. **Causal validity before perplexity**
2. **Free generation before long training**
3. **Evidence status must be explicit**
4. **Perplexity is not assistant quality**
5. **Retention must be separated from chat-format skill**
6. **Matched quality matters more than matched wall-clock progress**
7. **Negative results and invalidated runs remain documented**
8. **Proprietary implementation details remain confidential**

## Confidentiality

This repository documents research questions, evaluation protocols, audit findings, and high-level architecture families. **Exact modifier placement and proprietary implementation mechanics are intentionally excluded.**

## Status

The clean Full-versus-MOD-versus-LoRA experiment is beginning again from fresh data and checkpoints after the 2026-07-16 audit.

> **At comparable SFT quality, can Complete MOD preserve and use GPT-2's pretrained knowledge better than Full fine-tuning while remaining competitive with LoRA?**

## Contact

For research discussion, benchmarking, collaboration, or compute sponsorship, open an issue in this repository or contact me through my GitHub profile.
