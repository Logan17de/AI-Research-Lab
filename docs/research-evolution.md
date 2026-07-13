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

### Contribution

These experiments established themes that still guide the work: modularity, specialized parameter roles, controlled information flow, and separate evaluation of structural and semantic learning.

The main limitation was the cost and uncertainty of training a new architecture from scratch with limited compute.

## 2. Dynamic Transformer

The next branch explored adaptive capacity through token-local neuron ownership.

### Main ideas

- Separate embedding and transformation neuron pools
- Shared neurons with occurrence-based updates
- Local splitting and merging
- Copy-on-write specialization
- Dynamic growth rather than a fully fixed parameter budget

### What was learned

The design aligned conceptually with continual learning, but the prototype was slowed by dynamic routing, neuron materialization, writeback, and Python-level control flow. The hardware and systems challenge became as important as the learning theory.

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
- FFN-oriented MOD
- Combined MOD systems
- Attention-oriented MOD research
- Frozen-base and trainable-base configurations

### First finding: fast narrow-data fitting

Small SFT experiments suggested that FFN-oriented MOD variants learned narrow datasets faster than selected LoRA configurations. These experiments were useful for finding signal, but many epochs on tiny data made memorization a major confound.

### Corrected UltraChat benchmark

A later UltraChat experiment exposed a data-loader worker duplication problem. Epoch percentages were misleading because each worker iterated the full dataset. Comparisons therefore moved from reported epochs to a fixed budget of approximately 8,000 optimizer steps.

Under that corrected budget:

| GPT-2 Small variant | Eval PPL |
|---|---:|
| LoRA | 8.313 |
| LoRA + FFN-oriented MOD | 8.063 |
| Complete | 8.033 |

This isolated the FFN-oriented MOD as the main contributor to the measurable gain. The embedding-oriented component added a much smaller incremental improvement.

### Shift in the research question

The question is no longer:

> Does adding a modifier improve this LoRA baseline?

The current runs say yes, under this setup.

The serious question is now:

> Does the FFN-oriented MOD provide a more parameter-efficient or behaviorally distinct adaptation path than simply increasing LoRA rank?

That question can invalidate the current advantage or make the method substantially more interesting.

### Current interpretation

- FFN-oriented modification has a measurable validation-PPL effect.
- Embedding-oriented modification adds a smaller incremental gain.
- Embedding capacity may influence initial topic alignment, but a causal link to generation drift has not been established.
- Bigger modifier dimensions are not automatically better; useful capacity depends on data and optimization.
- Lower perplexity does not guarantee stronger instruction following.
- Exact placement and proprietary mechanics remain confidential.

## 5. Multi-MOD Domain Routing

The next extension treats learned domain adaptations as composable modules.

### Proposed system

- Frozen pretrained base
- Independent domain MODs, such as general, coding, medical, or legal
- A learned router that selects one or more modules
- A general module capable of combining or explaining domain knowledge
- Incremental addition of new modules without retraining every existing module

### Research question

Can modular routing improve continual learning by isolating updates while still allowing cross-domain composition?

The critical evaluation is not only new-domain accuracy, but also old-domain retention, router reliability, and the compute cost of combining modules.

## 6. Qwen and General SFT

GPT-2 is valuable for controlled experiments, but it is not a strong modern assistant model. The research therefore moved toward Qwen-family models and Tulu-style general instruction data.

### Current work

- Assistant-only supervision
- Explicit conversation control tokens
- Correct handling of single-turn and multi-turn examples
- End-of-turn separation within conversations
- Conversation isolation rather than cross-example packing
- Full fine-tuning versus MOD comparison
- Careful support for hybrid attention architectures

Early runs show that the pipeline is functioning, but the experiments are not mature enough to establish an architectural advantage.

### Baseline caveat

The tested Qwen “Base” checkpoint already displayed assistant-like behavior and could generate reasoning-shaped traces. That makes it unsuitable as a perfectly clean measure of how much SFT capability MOD creates from a raw pretrained model.

The next modern-model comparison should prefer checkpoints with clearly separated Base and SFT releases, such as:

- OLMo 2 Base versus its official SFT reference;
- SmolLM2 Base versus Instruct;
- Qwen only when pre-existing assistant bias is explicitly measured.

A proposed **Capability Gap Recovery** metric measures how much of the gap from a small base model to a larger reference model is recovered by MOD:

```text
CGR = (MOD score - small-base score)
      / (larger-model score - small-base score)
```

### GPT-2 Tülu pipeline validation

The GPT-2 pipeline was rebuilt around correct assistant-turn and conversation-end semantics. Assistant-only causal masking, multi-turn rendering, truncation, train/chat equivalence, neutral initialization, and gradient routing passed 25 regression tests.

This is engineering validation, not evidence that MOD is better. It creates the reliable measurement instrument needed for the next Full versus MOD versus LoRA experiment.

See [SFT Pipeline Validation](sft-pipeline-validation.md).

## 7. Progressive Frozen Capacity

The continual-learning direction now includes progressively adding and freezing small units of capacity.

### Core experiment

Compare equal total added capacity:

- one reusable MOD of dimension 32 trained across stages;
- four MODs of dimension 8, with each completed MOD frozen before adding the next.

All stages train on accumulated old and new data to provide replay.

If progressive capacity retains earlier tasks better than the single reusable module, the improvement would support **capacity isolation** rather than parameter count alone.

### Expansion beyond modifiers

Two larger controls are proposed:

- **Width expansion:** freeze the original representation and add a small trainable width throughout the model.
- **Depth expansion:** add trainable residual blocks or transformer layers around a frozen backbone.

Width and depth must be tested separately before combination.

See [Continual Expansion](continual-expansion.md).

## 8. Governed Memory and Reasoning

A parallel research track studies how an adaptive model should govern memory and reasoning.

### Main principles

- Updates need ownership and lifecycle controls
- Drift should be attributable
- Failures need rollback and recovery
- Working memory should be bounded
- Reasoning should abstain when evidence is insufficient
- Confidence should remain calibrated under ambiguity and distribution shift

This connects continual learning to a larger objective: an intelligent system should not only learn, but also understand when and how its internal state changed.

## Present Direction

The active research program combines:

1. **Localized adaptation** through token-conditioned capacity
2. **Baseline skepticism** through parameter-matched LoRA and full-SFT comparisons
3. **Progressive expansion** through successively frozen capacity, width, and depth
4. **Modular composition** through domain-specific modules and routing
5. **Governed continual learning** through retention, recovery, and calibrated reasoning

The immediate goal is controlled evidence—not a grand declaration. The research must identify where the method works, where it fails, and whether any advantage survives broader datasets, multiple seeds, higher-rank LoRA, and stronger backbones.
