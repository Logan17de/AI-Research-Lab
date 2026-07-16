# Superseded GPT-2 UltraChat Ablation

**Original experiment:** early July 2026  
**Invalidated:** 2026-07-16  
**Status:** historical record only

> Do not cite the PPL values on this page as valid causal language-model evidence.

## Former Claim

| Variant | Former reported PPL |
|---|---:|
| LoRA | 8.313 |
| LoRA + FFN-oriented MOD | 8.063 |
| Complete | 8.033 |
| GPT-2 Medium + LoRA | 6.142 |

It appeared that the FFN-oriented component accounted for most of the modifier improvement.

## Why the Claim Was Invalidated

The attention implementation became non-causal whenever an attention mask was supplied.

The supplied mask handled padding but did not add a triangular future-token mask. During teacher forcing, answer positions could inspect future gold-answer tokens.

At free generation time, those future tokens did not exist.

This explained the contradiction: very low teacher-forced PPL alongside prompt copying, empty responses, repeated phrases, loops, premature stopping, and unstable generation.

| Path | Broken-versus-causal maximum difference |
|---|---:|
| Base | 54.49 |
| Full | 52.70 |
| MOD | 19.28 |

A correct causal implementation should produce approximately zero.

## Historically Useful Lessons

The experiment motivated:

- fixed-step rather than misleading epoch comparisons;
- worker-sharding audits;
- component ablations;
- LoRA-rank skepticism;
- assistant-content versus stop-token metrics;
- free-generation gates;
- EOS-rank diagnostics;
- matched-SFT-quality retention testing.

## Current Replacement

The clean experiment uses repaired triangular causal attention, GPT-2 pretrained EOS after each assistant turn, fresh data and checkpoints, token-weighted evaluation, train/chat logit equivalence, and deterministic data handling.

See:

- [Experimental Evidence Ledger](experimental-evidence.md)
- [Current SFT Pipeline](sft-pipeline-validation.md)
- [2026-07-16 Pipeline Audit](research-log/2026-07-16-pipeline-audit.md)
