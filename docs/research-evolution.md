# Research Evolution

This document records how the current research direction emerged. The projects are not isolated ideas; each branch exposed a limitation that motivated the next one.

## 1. FusionFormer and FusionGrammar

The earliest architecture work explored whether grammar and meaning could be learned in partially separated streams.

### Main ideas

- Dedicated grammar and meaning representations
- Controlled cross-stream communication
- Orthogonality objectives to encourage specialization
- Sparse or gated information exchange
- Staged grammar-only, lexical, and joint training

### What this contributed

The experiments established the recurring themes that still guide the work:

- modularity;
- specialized parameter roles;
- controlled information flow;
- separate evaluation of structural and semantic learning.

The main limitation was the cost and uncertainty of training a new architecture from scratch with limited compute.

## 2. Dynamic Transformer

The next branch explored adaptive capacity through token-local neuron ownership.

### Main ideas

- Separate embedding and transformation neuron pools
- Shared neurons with occurrence-based updates
- Local splitting and merging
- Copy-on-write specialization
- Dynamic growth instead of a completely fixed parameter budget

### What was learned

The design was conceptually aligned with continual learning, but the prototype was slowed by dynamic routing, neuron materialization, writeback, and Python-level control flow. The hardware and systems challenge became as important as the learning theory.

This led to a more practical question: could token-local capacity be added to an existing pretrained transformer without rebuilding the entire architecture?

## 3. Inverted Transformer / Target-Side Learning

This branch directly investigated causal-language-model credit assignment.

### Main ideas

- Give target tokens a more explicit role in learning
- Separate target semantics from contextual influence
- Localize selected updates
- Measure gradient flow and token ownership
- Test whether local learning reduces interference

### Outcome

The causal prototype learned successfully and produced useful gradient diagnostics. One Wikipedia experiment reached approximately 2.83 evaluation perplexity around 8.2k steps.

However, it did not demonstrate a clear advantage over a standard causal language model. The experiment clarified the problem but did not yet provide the strongest solution, so the main effort returned to pretrained-model adaptation.

## 4. GPT_MOD

GPT_MOD applies token-conditioned adaptive capacity to a pretrained GPT backbone while retaining the backbone's existing language knowledge.

### Experimental families

- Full fine-tuning
- LoRA at multiple ranks
- Embedding-oriented MOD
- Transformation-oriented MOD
- Combined MOD systems
- Attention-oriented MOD research
- Frozen-base and trainable-base configurations

### Main hypothesis

Localized token-conditioned parameters may add learnable capacity while disturbing fewer global weights than full fine-tuning.

### Current interpretation

Early experiments suggest that transformation-oriented modifier variants can learn narrow SFT datasets quickly. Embedding-oriented adaptation can add useful capacity but may also produce behavioral drift. Modifier size is not universally “bigger is better”; the useful capacity depends on dataset size, training duration, and optimization.

The implementation details that determine exact modifier placement and operation remain confidential.

## 5. Multi-MOD Domain Routing

The next extension treats learned domain adaptations as composable modules.

### Proposed system

- Frozen pretrained base
- Independent domain MODs, such as general, coding, medical, or legal
- A learned router that selects one or more modules
- A general module capable of combining or explaining domain knowledge
- Incremental addition of new domain modules without retraining every existing module

### Research question

Can modular routing improve continual learning by isolating updates while still allowing cross-domain composition?

The critical evaluation is not only new-domain accuracy, but also retention of old-domain behavior and the compute cost of combining modules.

## 6. Qwen and General SFT

GPT-2 is useful for controlled experiments, but it is not a strong modern assistant model. The research therefore moved toward Qwen-family models and Tulu-style general instruction data.

### Current work

- Assistant-only supervision
- Explicit conversation control tokens
- Correct handling of single-turn and multi-turn examples
- Conversation isolation rather than cross-example packing
- Full fine-tuning versus MOD comparison
- Careful support for hybrid attention architectures

Early runs show that the pipeline is functioning, but the experiments are not mature enough to establish an architectural advantage.

## 7. Governed Memory and Reasoning

A parallel research track studies how an adaptive model should govern memory and reasoning.

### Main principles

- Updates need ownership and lifecycle controls
- Drift should be attributable
- Failures need rollback and recovery
- Working memory should be bounded
- Reasoning should abstain when evidence is insufficient
- Confidence should remain calibrated under ambiguity and distribution shift

This track connects continual learning to a larger objective: an intelligent system should not only learn, but also understand when and how its internal state changed.

## Present Direction

The active research program combines three themes:

1. **Localized adaptation** through token-conditioned capacity
2. **Modular composition** through domain-specific modules and routing
3. **Governed continual learning** through retention, recovery, and calibrated reasoning

The immediate goal is controlled evidence—not a grand declaration. The research must identify where the method works, where it fails, and whether any advantage survives broader datasets, multiple seeds, and stronger baselines.
