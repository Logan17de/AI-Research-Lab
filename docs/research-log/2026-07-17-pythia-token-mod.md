# 2026-07-17 — Pythia Token MOD Architecture and Small-Data Findings

> **Follow-up:** The larger chain-of-thought variant sweep and grouped layer-table experiments are documented in [2026-07-19 — Pythia Token MOD Variant Sweep](2026-07-19-pythia-variant-sweep.md). This page remains the historical record of the initial small-data architecture experiment.

## Status

**VALID for implementation integrity and small-data behavior.**

**EXPLORATORY for model-quality comparison.**

This experiment establishes that the revised Token MOD architecture is operational. It does **not** establish that the smaller MOD model matches or beats the larger dense model.

## Research Question

Can a frozen Pythia-1.4B backbone with compact token-specific trainable capacity compete with a fully fine-tuned Pythia-2.8B model while using substantially fewer resident and trainable parameters?

| Variant | Base parameters | Added MOD parameters | Resident parameters | Training path |
|---|---:|---:|---:|---|
| Pythia-1.4B + Token MOD | 1,414,647,808 | 51,593,216 | 1,466,241,024 | Frozen backbone; MOD only |
| Pythia-2.8B | 2,775,208,960 | 0 | 2,775,208,960 | Full fine-tuning |

The MOD system has approximately **47.17% fewer resident parameters** than Pythia-2.8B.

Optional learned scales add 50 parameters, giving 51,593,266 trainable MOD parameters and 1,466,241,074 total resident parameters.

## Why Pythia

GPT-2 Small could demonstrate prototype behavior, but its failures were ambiguous: limited capacity, narrow pretraining, dataset quality, and MOD limitations could not be separated.

Pythia provides a cleaner controlled comparison because the 1.4B and 2.8B checkpoints belong to the same GPT-NeoX model family.

The earlier Qwen checkpoint was not retained as the principal baseline because its starting behavior already showed instruction-tuned bias.

## Architecture Revision

The earlier three-family prototype reused global contextual projections across layers. That implementation behaved more like repeated token-conditioned residual memory than layer-specific contextual capacity.

The revised design uses four independent high-level MOD families:

1. Input
2. Output
3. Attention
4. FFN

Input and output capacity remain independent because Pythia uses untied input and output weights. Attention and FFN token memory is shared across layers, while each layer learns its own interpretation.

The frozen LM head remains intact, and the output branch contributes to the same vocabulary logits before one standard softmax.

Exact modifier placement and proprietary implementation mechanics are intentionally excluded from this repository.

## Parameter Accounting

Configuration used for the current specification:

| Family | MOD dimension | Parameters |
|---|---:|---:|
| Input | 128 | 6,701,056 |
| Output | 128 | 6,701,056 |
| Attention | 128 | 12,730,368 |
| FFN | 256 | 25,460,736 |
| **Total** | — | **51,593,216** |

The analytical additional MOD work is approximately 25.84 million MACs per generated token, about 1.97% of reported Pythia-1.4B static linear MACs. This is an estimate; measured latency, throughput, and VRAM remain required.

## Implementation Validation

The implementation verifies:

- four independent MOD families;
- layer-specific attention and FFN interpretation;
- one combined-logit vocabulary softmax;
- optional learned scales;
- sparse MOD checkpoints;
- incompatible old checkpoint rejection;
- zero initial MOD contribution with a gradient-safe initialization;
- all base parameters remain frozen;
- base gradients remain absent;
- original control-token input and output rows remain unchanged;
- native GPT-NeoX parallel residual behavior remains intact;
- runtime ablations for all four MOD families.

Validation status:

| Test scope | Result |
|---|---:|
| Focused Pythia tests | 18 passed |
| Full repository tests | 78 passed |

## Small-Data Training Signal

### Frozen Pythia-1.4B + MOD

Training PPL decreased smoothly from approximately **11.57 to 1.17**.

Assistant validation PPL initially improved and then rose:

```text
10.55 → 7.00 → 7.81 → 10.05 → 12.88
```

The best validation region was around step 100.

### Fully Fine-Tuned Pythia-2.8B

The larger dense baseline showed the same broad pattern: early evaluation improvement followed by rising validation PPL while training loss continued to fall.

Because both models turned upward, the late validation degradation is best interpreted as small-data overfitting and evaluation instability—not as proof that either architecture wins.

## Generation Findings

### Shared strengths

Both models learned several taught facts, including:

- 7 × 8 = 56;
- the heart has four chambers;
- the Mona Lisa was painted by Leonardo da Vinci;
- humans breathe out carbon dioxide.

### MOD strengths observed

The MOD model sometimes produced cleaner taught facts and meaningful semantic reframing. For example, it connected losing with focusing on process rather than perfection.

This suggests that the architecture can store associations and transfer some semantic behavior beyond exact memorization.

### MOD weaknesses observed

The MOD model was less reliable on:

- unseen arithmetic combinations;
- stable lexical realization;
- concrete multi-step procedures;
- rarely trained token combinations.

It could reach the correct semantic neighborhood while selecting an unstable or vague expression.

### Dense-model strengths and limits

Pythia-2.8B was generally stronger at practical phrasing and limited rule extrapolation, but it also hallucinated, corrupted phrases, and failed harder unseen arithmetic.

Neither model is a clean winner on the current evidence.

## Dataset Diagnosis

The small Q&A dataset mixes factual questions, arithmetic, science, advice, encouragement, and different answer styles.

Known limitations include:

- repeated templates;
- duplicate or conflicting prompts;
- vague factual targets;
- narrow token and context coverage;
- exact-token PPL penalizing valid alternative wording;
- too few examples to test broad compositional generalization.

For Input, Attention, and FFN MOD, only token rows appearing in the sequence receive direct lookup updates. Output rows receive softmax gradients globally, but rare target tokens receive little useful positive signal.

This supports the latest hypothesis: the dataset activates too little of the available token-specific memory to fairly test the full architecture.

## Evidence Status

| Claim | Status |
|---|---|
| The revised architecture trains stably | **VALID** |
| The frozen backbone remains unchanged during MOD training | **VALID** |
| The architecture can learn taught associations and some semantic reframing | **VALID within this dataset** |
| The small dataset produces late overfitting in both variants | **VALID** |
| MOD matches or beats Pythia-2.8B | **UNRESOLVED** |
| MOD generalizes algorithmic rules as well as dense fine-tuning | **NOT SUPPORTED** |
| Estimated compute overhead matches real latency | **UNMEASURED** |
| Current results establish continual-learning superiority | **NOT SUPPORTED** |

## Next Experiment

The decisive experiment requires:

1. a substantially larger and cleaner training set;
2. a separate held-out evaluation set;
3. broad token-frequency and context coverage;
4. true held-out facts and rephrased questions;
5. compositional, procedural, and arithmetic tests;
6. untouched-pretraining retention evaluation;
7. fixed checkpoints, data, target budgets, and decoding;
8. measured latency, throughput, VRAM, and wall-clock cost;
9. multiple seeds;
10. comparison at matched task quality, not merely matched steps.

## Current Conclusion

The GPT-2 work demonstrated the prototype behavior. The revised Pythia implementation demonstrates that the architecture is technically operational. The current small-data run exposes important failure modes.

The next larger-data controlled experiment—not the current PPL minimum—will test whether a 1.466B frozen model with Token MOD can compete with a fully fine-tuned 2.775B dense model.
