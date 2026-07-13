# SFT Pipeline Validation

## Purpose

Before comparing Full fine-tuning, MOD, and LoRA, the training data, loss mask, stopping semantics, initialization, and interactive chat path must be identical and correct.

This document records the validated behavior of the corrected GPT-2 Tülu SFT pipeline. It establishes experimental readiness—not architectural superiority.

## Control Tokens

The tokenizer registers four single-token conversation controls:

| Role | Token ID |
|---|---:|
| User | 50257 |
| Assistant | 50258 |
| System | 50259 |
| End of assistant turn | 50260 |

GPT-2 EOS remains token 50256. The resulting vocabulary size is 50,261.

## Single-Turn Semantics

```text
<USER> question
<ASSISTANT> answer<EOT>
<EOS>
```

## Multi-Turn Semantics

```text
<SYSTEM> optional instruction
<USER> question 1
<ASSISTANT> answer 1<EOT>
<USER> question 2
<ASSISTANT> answer 2<EOT>
<EOS>
```

The meanings are deliberately separated:

- EOT: one assistant response is complete;
- EOS: the full conversation/document is complete.

No EOS is inserted between assistant turns.

## Supervision

Direct loss is applied to:

- assistant answer tokens;
- every EOT;
- one final EOS.

Direct loss is masked for:

- system headers and content;
- user headers and content;
- assistant headers;
- separators;
- padding.

Context still influences assistant predictions through the causal network even when it has no direct target loss.

## Shifted Causal Targets

The verified behavior is:

```text
assistant header   → first answer token
last answer token  → EOT
intermediate EOT   → following user tokens masked
final EOT          → separator
separator          → final EOS
```

The model is not supervised to generate the next user turn.

## Chat Equivalence

Interactive chat:

- preserves earlier assistant responses ending in EOT;
- does not insert EOS between turns;
- stops assistant generation on EOT;
- reconstructs the same token prefix used during training.

Training and chat token sequences were verified as identical for equivalent conversations.

## Truncation

When a conversation exceeds sequence length:

1. remove the oldest complete exchange;
2. preserve the latest exchange;
3. preserve the system message when possible;
4. remove the system message only as a complete unit;
5. truncate the final answer only as a last resort;
6. reconstruct a valid EOT/EOS ending;
7. drop samples containing no supervised answer tokens.

Separate conversations are never packed together.

## Initialization and Gradients

The corrected MOD setup begins from predictions identical to the untouched base model. This enables a true step-zero comparison.

Validation confirmed:

- frozen base parameters receive no gradients in the MOD run;
- intended trainable components receive gradients;
- input/output representation tying remains consistent;
- special conversation controls receive trainable adaptation where intended.

Exact proprietary modifier placement and mechanics are excluded from this document.

## Validation Matrix

The corrected implementation passed 25 regression tests covering:

- control-token registration;
- single-turn formatting;
- multi-turn formatting;
- loss masking;
- causal shifting;
- truncation;
- train/chat equivalence;
- EOT generation stopping;
- legacy-format rejection;
- Full and MOD gradient behavior;
- neutral initialization.

Python compilation also passed.

## Compatibility

Previous data exported with EOS after every assistant turn is obsolete. Old checkpoints trained with that format should not be reused for the corrected experiment.

Fresh training is required.

## Controlled Experiment

Primary variants:

1. Full fine-tuning
2. Complete MOD
3. LoRA

Hold constant:

- dataset split;
- renderer and truncation;
- sequence length;
- supervised assistant-token budget;
- evaluation samples;
- checkpoint milestones;
- generation settings.

Evaluate at step zero and supervised-target milestones such as 1M, 5M, 10M, 25M, 50M, and one complete epoch.

## Questions

- Which method learns instruction behavior fastest?
- Which reaches the best held-out SFT loss?
- Which preserves pretrained completion knowledge?
- Which performs best on instruction and reasoning benchmarks?
- Which component contributes most?
- Does MOD improve sample efficiency, final capacity, retention, or only training fit?
