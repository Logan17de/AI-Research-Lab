# 2026-07-20 — Pythia V3 Chat Evaluation

## Status

**VALID for the owner-confirmed V3 generations.**

**EXPLORATORY for general model-quality claims.**

The complete owner-supplied transcript is stored in [results/pythia-v3-chat-evaluation-2026-07-20.txt](../../results/pythia-v3-chat-evaluation-2026-07-20.txt).

Although the runner printed `[V2] (EleutherAI/pythia-1.4b, step 600)`, the experiment owner confirmed that every response in this record is from **V3 at step 600**. The printed label is retained in the raw transcript as a test-harness provenance issue and should be fixed before the next run.

## Test scope

Nine prompts tested:

- open-ended kindness advice;
- five procrastination tips;
- short-answer arithmetic;
- an exact ten-word constraint;
- repeated and paraphrased multiplication;
- a false arithmetic premise;
- a trivial arithmetic question.

V3 step 600 is the best recorded validation checkpoint on the rebuilt direct-answer dataset: assistant PPL 6.5713, combined PPL 6.5347, and target top-1 56.23%.

## Scorecard

| Capability | Result | Observation |
|---|---:|---|
| Response emitted | 9/9 | no empty answers |
| EOS stopping | 9/9 | every answer stopped by EOS |
| Arithmetic/consistency probes | 2/6 correct | only `5 × 10 = 50` and `4 × 10 = 40` were correct |
| False-premise rejection | 0/1 | accepted `2 × 3 = 1` and invented an induction argument |
| Exact ten-word compliance | 0/1 | produced a long paragraph |
| Short-form compliance | 1/1 structurally | answer was short, but mathematically wrong |
| Topic relevance: advice prompts | 1/2 | kindness was relevant; procrastination largely was not |

The arithmetic/consistency group counts the six prompts beginning with “What is 5 x 10?” through “1 x 1 =?”.

## Per-prompt findings

| Prompt | Assessment |
|---|---|
| How can I be kind to others? | Relevant and fluent, but extremely long and includes questionable advice such as giving unsolicited advice. |
| Give me 5 tips to overcome procastination. | Five items were produced, but most discuss defensiveness, emotions, and difficult conversations instead of procrastination. |
| What is 5 x 10? answer in short | Short but incorrect: `500,000 (5×10^6)`. |
| How can you help me? answer in 10 words. | Ignores the exact word-count constraint and does not directly enumerate capabilities. |
| what is 5 times 10? | Correct: 50. |
| 4 times 10? | Correct: 40, followed by an irrelevant and mathematically confused sentence. |
| 2 times 3 times 5 | Incorrect: 10 rather than 30. |
| 2 x 3 = 1 | Fails contradiction handling; confidently validates a false premise with unrelated induction language. |
| 1 x 1 =? | Severe hallucination: invents an egg puzzle and invalid calculations instead of answering 1. |

## Interpretation

V3 produces answer-shaped, grammatically fluent text and stops reliably, but surface fluency is masking weak semantic grounding.

The direct-answer dataset appears to have reduced the explicit reasoning/meta-tag failures seen in the earlier COT-trained chat comparison. It did not solve:

- elementary factual reliability;
- arithmetic composition;
- resistance to false user premises;
- exact constraint following;
- topic alignment;
- consistency across paraphrases.

The paired `5 × 10` prompts are especially informative: one response gives 500,000 and another gives 50. This is not a single missing fact; it shows unstable retrieval or continuation behavior under minor prompt variation.

Likewise, the model can emit generic self-help structure without mapping the requested topic correctly. The procrastination answer is coherent sentence-by-sentence but semantically off-target.

## Relationship to validation metrics

Step 600 is the best recorded validation checkpoint, yet the qualitative test shows severe deployment failures. The result reinforces the earlier finding that teacher-forced PPL and target top-1 cannot serve as the only checkpoint-selection criteria.

This does not prove that step 600 is worse than every other V3 checkpoint because no matched chat evaluation has been supplied for the other steps. It proves that the best validation checkpoint is not sufficiently reliable for user-facing chat.

## Evidence status

| Claim | Status |
|---|---|
| The transcript belongs to V3 step 600 | **OWNER CONFIRMED** |
| V3 stopped with EOS on all nine prompts | **VALID** |
| V3 answered only two of six arithmetic/consistency probes correctly | **VALID within this transcript** |
| V3 ignored the ten-word constraint | **VALID** |
| V3 is fluent but unreliable in this sample | **VALID qualitative summary** |
| Direct-answer conversion fully fixed prior formatting behavior | **NOT ESTABLISHED** |
| V3 is generally worse than every earlier checkpoint | **NOT ESTABLISHED** |
| The architecture is the cause of these failures | **NOT ESTABLISHED** |

## Required next evaluation

Use a fixed, versioned, confirmed-unseen prompt suite with deterministic decoding and at least these metrics:

1. exact arithmetic accuracy;
2. false-premise correction;
3. word-count and brevity compliance;
4. topic relevance;
5. unsupported-claim rate;
6. answer consistency across paraphrases;
7. response-length distribution;
8. EOS and repetition behavior.

Record the model identity independently from the display label so a stale V2/V3 tag cannot recur.
