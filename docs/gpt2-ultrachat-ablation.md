# GPT-2 UltraChat Ablation

## Research Question

Does adding token-conditioned modifier capacity improve GPT-2 Small instruction tuning relative to LoRA under a fixed training budget?

## Models

| Model | Purpose |
|---|---|
| GPT-2 Small + LoRA | Primary baseline |
| GPT-2 Small + LoRA + FFN-oriented MOD | Isolate the FFN-oriented contribution |
| GPT-2 Small Complete | Measure FFN-oriented plus embedding-oriented adaptation |
| GPT-2 Medium + LoRA | Larger-backbone capacity reference |

Exact modifier placement and proprietary mechanics are intentionally excluded.

## Training-Budget Correction

The original iterable data loader used two workers without correctly separating their data. Each worker traversed the full dataset, approximately doubling the true data-loader epoch.

This created a mismatch:

- progress percentage assumed one dataset pass;
- epoch rollover waited for both duplicated worker passes.

Consequently, epoch labels were not reliable enough for comparison. The GPT-2 Small benchmark was realigned around approximately **8,000 optimizer steps**.

## Quantitative Results

### Matched GPT-2 Small comparison

| Model | Steps | Best eval PPL | Relative to LoRA |
|---|---:|---:|---:|
| LoRA | ~8,000 | 8.313 | baseline |
| LoRA + FFN-oriented MOD | ~8,000 | 8.063 | -0.250 |
| Complete | ~8,000 | 8.033 | -0.280 |

The Complete model's PPL was approximately 3.4% lower than LoRA's.

The key ablation is:

- FFN-oriented contribution: 8.313 → 8.063
- Additional embedding-oriented contribution: 8.063 → 8.033

Most of the measurable improvement came from the FFN-oriented component.

### Larger-backbone reference

GPT-2 Medium + LoRA reached approximately **6.142 eval PPL** at ~12,000 steps. This was quantitatively stronger than every GPT-2 Small variant, but it was not a matched comparison because both backbone capacity and training budget differed.

Recorded throughput:

| Model class | Approximate throughput |
|---|---:|
| GPT-2 Small variants | 70k tokens/s |
| GPT-2 Medium + LoRA | 31k tokens/s |

Medium was roughly 2.25 times slower in this environment.

## Qualitative Evaluation

Fixed generation settings:

```text
max_new_tokens = 128
temperature = 0.8
top_k = 50
top_p = 0.95
repetition_penalty = 1.15
no_repeat_ngram_size = 3
```

Prompt categories included:

- explanations;
- exact-count lists;
- professional rewriting;
- study planning;
- technical comparison;
- storytelling;
- coding;
- advice and emotional support.

### Observed behavior

**LoRA**

- Often ignored the prompt
- Produced generic web-style text
- Drifted heavily
- Sometimes stopped cleanly

**LoRA + FFN-oriented MOD**

- Often produced a stronger opening
- Sometimes stayed closer to the topic
- Still generated fake references and semantic drift
- Failed strict formatting constraints

**Complete**

- Often recognized the prompt's semantic domain quickly
- Produced structured beginnings
- Frequently drifted later
- Did not consistently translate its lower PPL into better instruction following

**GPT-2 Medium**

- Strongest quantitative model
- Still weak on strict instructions and long-range coherence
- Its 12k checkpoint was not consistently better in conversation than its 8k checkpoint

## EOS Observation

EOS rank and probability varied widely by prompt. Some generations assigned EOS rank 1 with high confidence; others ranked it in the hundreds or thousands.

Because top-k and top-p filtering can remove a low-ranked EOS token, isolated examples are insufficient. EOS should be reported across a fixed prompt suite using:

- EOS rank;
- EOS probability;
- filter survival;
- generated length;
- natural-stop success rate.

## What the Experiment Establishes

- Adding the FFN-oriented MOD improved eval PPL over LoRA at the same ~8k-step budget.
- The FFN-oriented component accounted for most of the Complete model's PPL gain.
- The embedding-oriented component added a small additional gain.
- GPT-2 Medium remained substantially stronger quantitatively.
- Lower PPL did not guarantee stronger assistant behavior.

## What It Does Not Establish

- Superiority over higher-rank LoRA
- Parameter efficiency
- Compute efficiency
- Better broad generalization
- Better continual-learning retention
- A causal relationship between embedding-oriented adaptation and drift
- GPT-2 Small equivalence to GPT-2 Medium

## Next Decisive Experiment

Run a LoRA rank sweep at the same training budget and report both quality and cost:

| Variant | Eval PPL | Trainable params | Tokens/s | VRAM | Wall time | Behavioral score |
|---|---:|---:|---:|---:|---:|---:|
| LoRA r=8 |  |  |  |  |  |  |
| LoRA r=16 |  |  |  |  |  |  |
| LoRA r=32 |  |  |  |  |  |  |
| LoRA + FFN-oriented MOD | 8.063 |  |  |  |  |  |
| Complete | 8.033 |  |  |  |  |  |

If higher-rank LoRA matches the modifier with fewer parameters or less compute, the current modifier advantage disappears. If the modifier retains an advantage after parameter and compute matching, the result becomes scientifically meaningful.
