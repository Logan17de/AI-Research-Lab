# Experimental Evidence

This document records the current evidence and its limitations. Values are approximate where experiments were logged across evolving code versions.

> These results are preliminary and not peer-reviewed. Comparisons are meaningful only when backbone, dataset, split, token budget, optimizer, and evaluation procedure are matched.

## GPT-2 Small: Narrow 100-QA SFT

The dataset contained a small mixture of general, motivational, health, and mathematics examples. Most variants were trained for many epochs, making memorization a serious risk.

| Variant family | Observed result | Interpretation |
|---|---:|---|
| LoRA r=32 | PPL approximately 5–6 | Learned the task, but plateaued above several MOD variants |
| MOD variants involving the transformation path | PPL approximately 1–2 after extended training | Fast fit to the narrow dataset; generalization not established |
| Small embedding MOD | Underfit relative to the medium setting | Insufficient capacity for this setup |
| Medium embedding MOD | Best of the tested embedding sizes on the narrow set | Capacity matched this experiment better |
| Large embedding MOD | Needed more data or training to justify its capacity | Bigger was not automatically better |

### Main lesson

The result supports a **capacity-and-optimization hypothesis**, not yet a general superiority claim. The modifiers may provide extra storage that fits a small dataset quickly, but that same behavior can also mean memorization.

## GPT-2 Medium: Early UltraChat Comparison

An early one-epoch comparison produced:

| Variant | Approximate evaluation PPL |
|---|---:|
| FFN-oriented MOD | 8.0 |
| LoRA | 8.3 |

The MOD variant reached the lower value first, but the margin is small and based on limited training. Multiple seeds, matched parameter budgets, and longer runs are required.

## Inverted Transformer: Wikipedia Experiment

Example configuration:

- model dimension: 256
- layers: 6
- attention heads: 8
- sequence length: 256
- batch size: 64
- training data: approximately 1 GB of Wikipedia text

Recorded result around 8.2k steps:

| Metric | Approximate value |
|---|---:|
| Evaluation perplexity | 2.83 |
| Embedding cosine diagnostic | 0.69 |
| Modifier cosine diagnostic | 0.68 |

The system learned effectively, but did not demonstrate a clear advantage over conventional causal language modeling. This branch was therefore deprioritized rather than presented as a success.

## Qwen + Tulu SFT: Early Run

The modern-model experiment uses a small Qwen-family model and general Tulu-style SFT data.

Early diagnostics around 3k steps included:

| Metric | Observed range/value |
|---|---:|
| Training loss | 0.72–1.03 |
| Training perplexity | 2.06–2.80 |
| Evaluation perplexity | approximately 2.42 |
| Throughput | approximately 1.8k–2.4k tokens/s |
| Seen supervised target tokens | approximately 10 million |

The pretrained model started with relatively low perplexity, so the raw value alone does not prove that the adaptation method is working better. Qualitative probes produced structured responses, but at least one arithmetic probe was incorrect.

### Training pipeline safeguards

- Explicit user, assistant, system, and end-of-turn control tokens
- Assistant-only supervision
- User/system content and padding masked from the loss
- Single-turn and multi-turn conversations handled distinctly
- End-of-turn tokens used between turns
- End-of-sequence reserved for the end of a complete conversation
- Separate conversations not packed together

## Known Confounds

Current results may be affected by:

- tiny datasets;
- excessive epochs;
- train/evaluation similarity;
- single-seed runs;
- unequal parameter budgets;
- different checkpoints or training durations;
- decoding sensitivity;
- pretrained-base strength;
- data-loader and worker behavior;
- insufficient retention evaluation.

## Required Evidence Before Strong Claims

A credible comparison needs:

1. Identical backbones and tokenizer state
2. Identical training and held-out evaluation data
3. Matched token budgets and optimizer schedules
4. Parameter-matched and compute-matched comparisons
5. At least three random seeds
6. LoRA rank sweeps
7. Full fine-tuning and frozen-base controls
8. Sequential-domain retention tests
9. Generation evaluation with fixed decoding
10. Wall-clock, VRAM, and throughput reporting

## Current Conclusion

The strongest defensible statement is:

> Token-conditioned modifier variants have shown faster fitting than selected LoRA configurations in some small experiments, especially when the modifier adds transformation-side capacity. It is not yet known whether this advantage survives broader data, stronger baselines, matched budgets, or continual-learning retention tests.
