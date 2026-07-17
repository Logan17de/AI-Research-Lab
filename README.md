# AI Research Lab

**Independent AI research by Logeshkumar Duraisamy (Logan), based in Japan.**

**Last updated:** 2026-07-17

This repository tracks research on **continual learning**, **localized adaptation**, **parameter-efficient training**, **capacity expansion**, and **governed machine reasoning**.

> **Central question:** Can a smaller frozen pretrained model gain useful new capacity while preserving access to its original knowledge—and compete with a much larger densely trained model?

## Current Experiment

| Variant | Resident parameters | Training path |
|---|---:|---|
| Pythia-1.4B + Token MOD | 1,466,241,024 | Frozen 1.4B backbone + 51,593,216 trainable MOD parameters |
| Pythia-2.8B | 2,775,208,960 | Full fine-tuning |

The MOD system has approximately **47.17% fewer resident parameters**.

The revised architecture uses four independent high-level capacity families:

1. Input
2. Output
3. Attention
4. FFN

Token memory is shared where appropriate, while transformer layers learn distinct interpretations. Pythia's frozen language-model head and native residual structure remain intact.

Exact modifier placement and proprietary implementation mechanics are intentionally excluded.

## Implementation Status — 2026-07-17

The architecture now verifies:

- a strictly frozen Pythia-1.4B backbone;
- independent input and output capacity for Pythia's untied weights;
- layer-specific Attention and FFN interpretation;
- one combined-logit vocabulary softmax;
- zero-effect, gradient-safe initialization;
- sparse MOD checkpoints and runtime ablations;
- rejection of incompatible older checkpoints.

| Test scope | Result |
|---|---:|
| Focused Pythia tests | 18 passed |
| Full repository tests | 78 passed |

## Current Experimental Signal

The first small-data run shows that Token MOD trains stably:

```text
Train PPL: approximately 11.57 → 1.17
Assistant validation PPL: 10.55 → 7.00 → 7.81 → 10.05 → 12.88
```

Pythia-2.8B full fine-tuning showed the same broad pattern: early validation improvement followed by degradation while training loss continued falling.

This indicates that the tiny mixed Q&A dataset and validation design dominate the late result. Selecting a winner from minimum PPL would be scientific karaoke—confident, loud, and not necessarily accurate.

Generation suggests an early trade-off:

- MOD is strong at taught associations and some semantic reframing;
- MOD is currently weaker at unseen arithmetic, procedural specificity, and rarely trained token combinations;
- Pythia-2.8B is generally stronger at practical composition, but still hallucinates and fails harder extrapolation;
- neither model is a clean winner.

## Evidence Foundation

On 2026-07-16, a full GPT-2 repository audit discovered future-token leakage in the old attention-mask path. Earlier GPT-2 SFT perplexity comparisons—including 8.313 / 8.063 / 8.033—were invalidated as causal language-model evidence.

The repaired causal pipeline reached near-exact numerical equivalence checks and 54 passing tests. That audit remains the foundation for all newer experiments.

Historical results remain documented with explicit **Valid**, **Exploratory**, **Superseded**, **Invalidated**, or **Proposed** status.

## Current Conclusion

Established:

- the revised Token MOD architecture is operational;
- 51.6M trainable parameters can adapt a frozen Pythia-1.4B backbone;
- the base remains frozen;
- the model learns taught associations and some transferable semantic behavior;
- the present dataset is too small and noisy for the main comparison.

Not established:

- MOD matches or beats fully fine-tuned Pythia-2.8B;
- MOD preserves pretrained knowledge better;
- MOD generalizes algorithmic rules as well as dense training;
- analytical compute estimates predict measured latency;
- current results prove continual-learning superiority.

## Next Experiment

The decisive benchmark requires:

- a much larger and cleaner training set;
- a separate true held-out evaluation set;
- broad token and context coverage;
- rephrasing, compositional, procedural, and arithmetic evaluation;
- untouched-pretraining retention tests;
- matched-quality comparison;
- measured latency, throughput, VRAM, and cost;
- multiple seeds and additional Pythia-1.4B baselines.

## Documentation

- [Research Evolution](docs/research-evolution.md)
- [Experimental Evidence Ledger](docs/experimental-evidence.md)
- [Current Research Roadmap](docs/roadmap.md)
- [Dated Research Log](docs/research-log/README.md)
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
