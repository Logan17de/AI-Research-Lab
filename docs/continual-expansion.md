# Continual Expansion of a Frozen Transformer

## Status

This is a proposed research direction. It is not yet validated experimentally.

## Objective

Learn new instructions and knowledge by adding capacity around a frozen model while preserving previously acquired behavior.

The central trade-off is:

```text
Plasticity = ability to learn new behavior
Stability  = ability to retain old behavior
```

The method does not need to beat full fine-tuning on raw adaptation if it achieves a better stability–plasticity balance.

## Accumulated-Data Replay

Each new stage trains on old and new data together:

```text
Stage 1: train base on A
Stage 2: freeze base; train MOD 1 on A + B
Stage 3: freeze base and MOD 1; train MOD 2 on A + B + C
```

Evaluation must remain separated into:

- old-domain validation;
- new-domain validation;
- mixed unseen validation;
- instruction following;
- original-language-model evaluation;
- completion-style factual retention.

Training loss alone is not sufficient.

## Progressive Frozen MOD Benchmark

### Single reusable capacity

```text
one MOD, dimension 32
trained across all stages
```

### Progressive isolated capacity

```text
MOD 1, dimension 8 → train → freeze
MOD 2, dimension 8 → train → freeze
MOD 3, dimension 8 → train → freeze
MOD 4, dimension 8 → train → freeze
```

Both have approximately the same total added dimension.

### Interpretation

- If progressive 4×8 retains earlier tasks better, isolation matters.
- If both perform similarly, total capacity matters more than freezing.
- If progressive capacity hurts composition or compute, isolation has a cost.

## Composition Designs

### Stacked

```text
Base → MOD 1 → MOD 2 → MOD 3
```

Potential advantage: forward transfer through earlier modules.

Risk: later modules can disturb earlier behavior.

### Parallel additive

```text
output = base + MOD 1 + MOD 2 + MOD 3
```

Potential advantage: each module sees the same base representation.

Risk: outputs can still conflict when added.

The initial benchmark should avoid a router. Routing should be introduced only if measured interference justifies the additional machinery.

## Width Expansion

Freeze the original model dimensions and add a small trainable width throughout the network.

Conceptual example:

```text
frozen width: 768
new width:      8
total width:  776
```

This creates new representational capacity rather than repeatedly rewriting the original dimensions.

Research questions:

- Can new width learn useful behavior?
- Do cross-connections disturb frozen behavior?
- How should the new FFN width scale?
- Is width expansion more efficient than LoRA or MOD?

## Depth Expansion

Add trainable computation around a frozen backbone.

Options:

- full new transformer layers;
- small residual depth blocks;
- bottleneck residual blocks.

Depth expansion should be evaluated separately from width expansion before combining them.

## Baselines

1. Untouched base
2. Full sequential fine-tuning
3. LoRA sequential fine-tuning
4. Single reusable MOD
5. Progressive frozen MODs
6. Width expansion
7. Depth expansion
8. Width + depth only after separate ablations

## Evaluation

At every stage measure:

- new-task learning;
- old-task retention;
- original-corpus perplexity;
- completion-token rank and probability;
- instruction following;
- general language modeling;
- parameter count;
- active parameters;
- throughput;
- VRAM;
- wall-clock time.

## Strongest Possible Claim

A meaningful result would be:

> Progressively added frozen capacity approaches continued full fine-tuning on new tasks while producing substantially less backward forgetting at a competitive parameter and compute cost.

If the retention advantage disappears under equal-capacity and replay controls, progressive freezing is not justified.
