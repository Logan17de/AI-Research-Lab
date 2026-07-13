# Experimental Evidence

This document records the current evidence and its limitations. Values are approximate where experiments were logged across evolving code versions.

> These results are preliminary and not peer-reviewed. Comparisons are meaningful only when backbone, dataset, split, optimizer-step budget, preprocessing, and evaluation procedure are matched.

## 1. Data-Loader Correction

An early UltraChat run displayed inconsistent epoch progress. Investigation found that an iterable dataset was running with two workers, and each worker traversed the full dataset instead of a separate shard.

Consequences:

- the effective data-loader epoch was approximately doubled;
- reported progress percentages and epoch rollover disagreed;
- what appeared close to two epochs was roughly one true dataset pass.

The benchmark was corrected by using **optimizer steps**, rather than displayed epochs, as the primary training-budget unit. GPT-2 Small comparisons were aligned at approximately **8,000 optimizer steps**.

This correction matters because uncorrected epoch labels would have overstated training progress and weakened the comparison.

## 2. GPT-2 Small UltraChat Ablation

All three models used the same GPT-2 Small backbone and were compared at approximately the same 8,000-step budget.

| Model | Eval PPL | Change from LoRA |
|---|---:|---:|
| LoRA | 8.313 | — |
| LoRA + FFN-oriented MOD | 8.063 | -0.250 |
| Complete: LoRA + FFN MOD + Embedding MOD | 8.033 | -0.280 |

### Quantitative interpretation

- The Complete model achieved approximately **3.4% lower eval PPL** than LoRA.
- FFN-oriented modification recovered **0.250 of the total 0.280 PPL improvement**.
- The embedding-oriented component added the remaining **0.030**.
- Roughly 89% of the measured improvement over LoRA was already present without the embedding-oriented component.

The strongest current ablation conclusion is:

> In this UltraChat SFT setup, the FFN-oriented MOD was responsible for most of the measurable modifier gain.

This does **not** establish superiority over higher-rank or parameter-matched LoRA.

## 3. GPT-2 Medium Capacity Baseline

GPT-2 Medium + LoRA was trained to approximately 12,000 steps and reached:

| Metric | Approximate value |
|---|---:|
| Best eval PPL | 6.142 |
| Training loss | 1.878 |
| Training PPL | 6.54 |
| Throughput | 31k tokens/s |

Recorded GPT-2 Small throughput was approximately 70k tokens/s, making Medium about 2.25 times slower in this environment.

Medium remained much stronger quantitatively. However, it used a larger backbone and a longer training budget, so this is a capacity reference—not a clean fixed-budget ablation.

The modifier improved GPT-2 Small; it did not turn Small into Medium.

## 4. Perplexity Versus Assistant Quality

Manual tests included explanation, rewriting, structured lists, planning, technical comparison, storytelling, and coding prompts.

Common failures across all models included:

- locally fluent but globally incoherent responses;
- topic drift;
- failure to follow exact item counts;
- failure to return requested structures;
- fabricated citations or terminology;
- weak conversational recall;
- weak arithmetic;
- failure to stop consistently.

The FFN-oriented model often produced better openings and sometimes stayed closer to the requested topic than LoRA. The Complete model frequently recognized the semantic domain of a prompt quickly, but sometimes drifted later.

GPT-2 Medium at 12k steps was not consistently better in conversation than its 8k checkpoint despite its lower PPL.

Therefore:

> Lower evaluation perplexity measured better next-token fit, but did not reliably measure instruction following, planning, long-range coherence, or formatting compliance.

## 5. Embedding-Oriented MOD Hypotheses

### Supported

- Adding the embedding-oriented component improved the fixed-budget eval PPL from 8.063 to 8.033.
- The gain was much smaller than the FFN-oriented gain.

### Plausible but unproven

- It may improve early topic recognition or local semantic flexibility.
- It may affect long-range generation stability.

### Not established

- It definitely causes drift.
- It directly injects factual knowledge.
- Its small PPL gain translates into better assistant behavior.

All tested models drifted, including LoRA and GPT-2 Medium. Drift cannot currently be attributed uniquely to the embedding-oriented component.

## 6. EOS and Generation Behavior

Generation settings were held fixed:

- maximum new tokens: 128
- temperature: 0.8
- top-k: 50
- top-p: 0.95
- repetition penalty: 1.15
- no-repeat n-gram size: 3

EOS behavior varied substantially by prompt and model. Some runs produced EOS at rank 1 with high probability; others ranked EOS in the hundreds or thousands, where sampling filters removed it.

The inconsistency suggests that EOS behavior should be measured statistically across a fixed prompt suite rather than inferred from isolated examples.

## 7. Narrow 100-QA SFT Experiments

Earlier experiments used a small mixture of general questions, motivation, health, and mathematics. Many variants were trained for numerous epochs.

| Variant family | Observed result | Limitation |
|---|---:|---|
| LoRA r=32 | PPL approximately 5–6 | Narrow, heavily repeated data |
| Transformation-oriented MOD variants | PPL approximately 1–2 | Fast fit may reflect memorization |
| Small embedding MOD | Underfit | Insufficient capacity for this setup |
| Medium embedding MOD | Best tested balance | Dataset-specific |
| Large embedding MOD | Noisy or slow to justify capacity | Needed more data or training |

The dataset strongly emphasized short positive-advice answers. This caused category confusion: technical prompts could be answered like life advice. Dataset composition, not only architecture, was limiting generalization.

## 8. Inverted Transformer: Wikipedia Experiment

Example configuration:

- model dimension: 256
- layers: 6
- attention heads: 8
- sequence length: 256
- batch size: 64
- training data: approximately 1 GB of Wikipedia text

Recorded around 8.2k steps:

| Metric | Approximate value |
|---|---:|
| Evaluation PPL | 2.83 |
| Embedding cosine diagnostic | 0.69 |
| Modifier cosine diagnostic | 0.68 |

The system learned effectively but did not demonstrate a clear advantage over conventional causal language modeling. This branch was deprioritized rather than presented as a success.

## 9. Qwen + Tulu SFT: Early Run

Early diagnostics around 3k steps included:

| Metric | Observed range/value |
|---|---:|
| Training loss | 0.72–1.03 |
| Training PPL | 2.06–2.80 |
| Evaluation PPL | approximately 2.42 |
| Throughput | approximately 1.8k–2.4k tokens/s |
| Seen supervised target tokens | approximately 10 million |

The pretrained model started with relatively low PPL, so the raw value alone does not prove that the adaptation method is better. Qualitative probes produced structured responses, but at least one arithmetic probe was incorrect.

### Pipeline safeguards

- Explicit user, assistant, system, and end-of-turn control tokens
- Assistant-only supervision
- User/system content and padding masked from loss
- End-of-turn used between turns
- End-of-sequence reserved for complete-conversation termination
- Single-turn and multi-turn examples handled distinctly
- Separate conversations not packed together

## 10. Established, Hypothesized, and Unsupported

### Established by current runs

- The FFN-oriented MOD improved GPT-2 Small + LoRA eval PPL at the same ~8k-step budget.
- It accounted for most of the Complete model's measured PPL gain.
- The embedding-oriented component added a smaller incremental gain.
- GPT-2 Medium remained substantially stronger quantitatively.
- Lower PPL did not guarantee better instruction-following behavior.
- Modifier variants changed generation behavior relative to LoRA.

### Hypotheses requiring targeted tests

- Embedding-oriented adaptation improves initial topic alignment.
- Embedding-oriented adaptation contributes to long-range drift.
- FFN-oriented adaptation provides a distinct learning path from LoRA.
- The modifier is more parameter-efficient than increasing LoRA rank.
- Localized adaptation improves sequential-learning retention.

### Not supported

- The Complete model replaces GPT-2 Medium.
- Embedding-oriented adaptation definitely causes drift.
- Embedding-oriented adaptation directly adds factual knowledge.
- Lower PPL alone establishes a better assistant.
- Current results establish continual-learning superiority.

## 11. Required Evidence Before Strong Claims

1. Identical backbone, tokenizer, data, split, and preprocessing
2. Matched optimizer-step and token budgets
3. Correct worker sharding
4. LoRA rank sweep under the same conditions
5. Parameter-matched and compute-matched comparisons
6. At least three random seeds
7. Automated instruction-following and coherence evaluation
8. EOS statistics across a fixed prompt suite
9. Sequential-domain retention tests
10. Wall-clock, VRAM, throughput, and active-parameter reporting

## Current Conclusion

The strongest defensible statement is:

> At a matched ~8,000-step UltraChat SFT budget, adding an FFN-oriented token-conditioned modifier to GPT-2 Small + LoRA reduced eval PPL from 8.313 to 8.063. Adding the embedding-oriented component reduced it further to 8.033. Most of the measured gain came from the FFN-oriented component, but it is not yet known whether this gain survives higher-rank LoRA, parameter matching, multiple seeds, or broader behavioral evaluation.
