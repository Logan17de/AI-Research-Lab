# Current GPT-2 SFT Pipeline Validation

**Effective date:** 2026-07-16  
**Status:** current

This document describes the repaired causal SFT pipeline. Earlier EOT-based documentation is superseded.

## Current Conversation Format

Role markers remain registered single tokens:

| Role | Token ID |
|---|---:|
| User | 50257 |
| Assistant | 50258 |
| System | 50259 |

GPT-2's pretrained EOS token remains 50256.

### Single turn

```text
<USER> question
<ASSISTANT> answer<EOS>
```

### Multi-turn

```text
<SYSTEM> optional instruction
<USER> question 1
<ASSISTANT> answer 1<EOS>
<USER> question 2
<ASSISTANT> answer 2<EOS>
```

EOS ends one complete assistant turn. It does not erase previous context.

During training:

- assistant content is supervised;
- EOS after every assistant turn is supervised;
- user, system, headers, separators, and padding are masked;
- the next user turn after EOS is masked;
- later assistant responses can attend to prior turns.

During chat:

- generation stops on EOS;
- EOS is removed from displayed text;
- EOS remains in internal conversation history.

## Why Dedicated EOT Was Retired

The dedicated EOT token was newly initialized and created an unfair comparison. Frozen-embedding baselines could not directly learn a pretrained-quality output row.

A 500-step quality gate showed poor stopping and high repetition across every variant.

GPT-2 EOS already has a meaningful pretrained representation and learned turn ending rapidly in the repaired smoke run.

## Causal Attention Repair

The critical old bug was:

```text
attention_mask supplied
→ implicit is_causal disabled
→ padding mask present
→ triangular future mask missing
```

The repaired pipeline always preserves triangular causal attention while also handling padding.

## Numerical Validation

| Check | Maximum difference |
|---|---:|
| Hugging Face GPT-2 equivalence, eager and SDPA | 0.0 |
| Future-token influence | 0.0 |
| Explicit versus implicit causal mask | 0.0 |
| Right-padding invariance | 0.0 |
| Batched versus individual inference | 1.19e-7 |
| Cached versus full-context decoding | 8.94e-8 |
| Direct versus chat first-step logits | 0.0 |

## Additional Audit Fixes

- Full-FT launcher no longer freezes GPT-2
- Gradient accumulation uses correct token weighting
- Resume state includes step, RNG, scaler, and data position
- Multi-worker epoch accounting is correct
- Packed pretraining is deterministic
- Evaluation is not silently limited
- MOD generation is cache-compatible
- Bias and normalization parameters avoid inappropriate weight decay
- BF16 scaler behavior is correct
- Dashboard metrics are token-weighted and separated by category
- Legacy/unversioned chat data is rejected

## Test Status

- **54 tests passing**
- synthetic Full/MOD/LoRA overfit tests passing
- save/resume smoke tests passing
- old chat checkpoints rejected
- old chat JSONL requires re-export

## Current Evaluation Categories

Report separately:

- assistant-content PPL and top-1;
- first-answer-token PPL and top-1;
- EOS PPL, top-1, and probability;
- combined target PPL;
- free-generation stop rate;
- premature-stop rate;
- repeated-trigram rate;
- prompt conditioning;
- multi-turn context use.

## Remaining Known Issue

The current evaluator reports first_answer n=0 and PPL=NaN.

This classification bug must be fixed before comparative training.

## Fresh-Run Requirement

All pre-audit GPT-2 chat checkpoints and old exported chat data are obsolete.

The clean experiment must begin with fresh exports, fresh checkpoints, step-zero evaluation, fixed smoke-test milestones, and free generation checked before long training.

Exact proprietary modifier placement and mechanics remain excluded.
