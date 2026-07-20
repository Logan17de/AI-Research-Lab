# 2026-07-19 — Pythia Variant Chat Comparison

## Status

**VALID for the recorded generations.**

**EXPLORATORY for model-quality ranking.**

This qualitative test compares six checkpoints on two general-advice prompts. It is intended to check whether teacher-forced validation perplexity predicts actual answer quality, format reliability, task alignment, and hallucination behavior.

The complete unedited transcript is stored in [results/pythia-chat-comparison-2026-07-19.txt](../../results/pythia-chat-comparison-2026-07-19.txt).

## Checkpoints

| Test label | Architecture record | Checkpoint step | Best recorded assistant PPL |
|---|---|---:|---:|
| 128_MOD | shared v1_1 | 1,450 | 4.0188 |
| 256_MOD | shared v1_2 | 1,409 | 3.9859 |
| 256_MOD_epoch_7 | shared v1_3 | 1,100 | 3.9724 |
| Custom_MOD_epoch_7 | shared v1_4 | 896 | 3.9724 |
| V2 | 24-table layer-unique v2 | 950 | 4.0429 |
| 2.8B | Pythia-2.8B full fine-tuning | 2,250 | 3.5190 |

The generation checkpoints are close to, but not always exactly identical to, the best-evaluation step. This record evaluates the named generation checkpoints rather than silently substituting another checkpoint.

## Prompts

1. “Give me some tips to maintain my attitude.”
2. “How can a teacher help a student to achieve good marks in exam?”

The record does not yet establish whether these exact prompts or close paraphrases occur in the training data. If they are confirmed unseen, the generalization interpretation becomes stronger.

## Qualitative Summary

| Variant | Task alignment | Output-format reliability | Practical usefulness | Hallucination or grounding risk | Two-prompt assessment |
|---|---|---|---|---|---|
| 2.8B full FT | strongest and most consistent | clean answers on both prompts | strong | invented source context and occasional questionable advice | best overall |
| 256_MOD / v1_2 | strong | clean answers on both prompts | strongest MOD | fabricated “Holding Hands” method and “Dr. Michael Merz” attribution | best MOD |
| 128_MOD / v1_1 | generally strong | clean answers on both prompts | useful but verbose | assumes nonexistent source text; occasional irrelevant framing | reliable third |
| Custom_MOD / v1_4 | relevant but shallower | clean answers on both prompts | moderate | unsupported source framing and awkward recommendations | clean but limited |
| V2 layer-unique | inconsistent | answer emitted, but meta-tags leaked | weak on second prompt | teacher/student role changed to parent/child; irrelevant examples | below shared MODs |
| 256_MOD_epoch_7 / v1_3 | useful ideas inside reasoning | **failed to produce a proper user-facing output twice** | unusable without post-processing | malformed reasoning/output tags | worst deployment reliability |

## Prompt 1 Findings

### 128_MOD

Strengths:

- comprehensive;
- actionable;
- covers habits, gratitude, feedback, and resilience;
- stops cleanly.

Weaknesses:

- excessively long for the request;
- drifts from personal attitude into organizational culture;
- several suggestions are generic rather than tightly targeted.

### 256_MOD

Strengths:

- concise and practical;
- strong advice on setbacks, boundaries, reflection, and consistency;
- clean user-facing output;
- arguably competitive with the 2.8B answer on usefulness.

Weaknesses:

- “focus on body parts you care about” is awkward and poorly justified;
- the hidden reasoning still assumes a source that was never supplied.

### 256_MOD_epoch_7

The model generated a long structured analysis but never delivered a proper `<output>` answer. It closed the reasoning with `</output>` and stopped.

This is a severe format-reliability failure. A low teacher-forced PPL did not predict whether the model would complete the required response structure.

### Custom_MOD_epoch_7

Strengths:

- clean, concise answer;
- relevant emphasis on responsibility, collaboration, motivation, and perseverance.

Weaknesses:

- some recommendations are weakly connected to maintaining attitude;
- several causal claims are unsupported;
- less practical than v1_2 or the dense baseline.

### V2

Strengths:

- short;
- readable;
- clean final answer.

Weaknesses:

- “Work is serious business” is rigid and may undermine the requested positive attitude;
- advice is narrower and less empathetic;
- reasoning contains spelling and template artifacts.

### 2.8B full FT

Strengths:

- polished;
- coherent;
- directly organized around optimism and resilience;
- strongest overall fluency.

Weaknesses:

- claims the advice comes from *The Optimistic Child* without grounding;
- “law of averages” is poorly suited to the task;
- “Stop viewing challenges as obstacles to overcome” is logically awkward.

## Prompt 2 Findings

### 128_MOD

The answer correctly focuses on praise, feedback, testing, and varied practice. However, parts of the explanation are confused: test preparation is positioned against classroom resources, “multiple answers” is unclear, and “solve problems before grading” is not a meaningful teaching procedure as written.

### 256_MOD

This is the strongest MOD answer in the second test. It includes goals, active recall, targeted feedback, foundational learning, deep understanding, learning from mistakes, positive reinforcement, tutorials, preparation, and multiple study methods.

Its major flaw is fabricated authority. The model invents a “Holding Hands” technique, attributes it to “Dr. Michael Merz,” and refers to something “shown on the right” even though no source or image exists.

### 256_MOD_epoch_7

The content contains several useful teaching strategies, but it is emitted inside malformed `<execution>`, `<self_improvement>`, and closing tags rather than as a clean user answer.

This repeats the first prompt's failure and makes the checkpoint operationally unreliable despite its leading MOD perplexity.

### Custom_MOD_epoch_7

The response is clean and teacher-focused, but shallow. The unexplained Bloom’s Taxonomy reference and the phrase “they’re working hard but not achieving much themselves” reduce clarity and empathy.

### V2

This is the weakest semantically aligned complete answer:

- it changes the teacher/student relationship into parent/child;
- recommends new subjects and extracurricular activities that may distract from exam preparation;
- introduces an unnecessarily complex differential-equation example;
- leaks `<meta_learning>`, `<self_improvement>`, and `<conclusion>` structures;
- reports unsupported “Confidence: 95%.”

The unique-table checkpoint learned answer-shaped text but showed weaker role stability and instruction alignment.

### 2.8B full FT

The answer is the most consistently teacher-centered. It covers communication, reinforcement, study methods, progress monitoring, clarification, and a supportive environment.

Weaknesses remain:

- Common Core and article-derived framing appears in the reasoning without a source;
- weekly progress reports may be excessive;
- “fair consideration ... regardless of quantity” is ambiguous.

## Cross-Model Findings

### 1. Perplexity did not predict deployment reliability

v1_3 and v1_4 share the best MOD validation PPL, approximately 3.9724.

Their generation behavior differs sharply:

- v1_4 produced two clean answers;
- v1_3 failed to produce a proper user-facing answer twice.

The evaluation pipeline therefore needs explicit answer-presence and format-validity metrics. PPL alone cannot detect this failure.

### 2. v1_2 is the strongest practical MOD in this snapshot

v1_2 has slightly worse minimum PPL than v1_3/v1_4, yet produced the most useful and consistently structured MOD answers across the two prompts.

This does not overturn the quantitative sweep. It shows that selecting a checkpoint solely by minimum PPL can select a worse deployed model.

### 3. All variants display source hallucination

The reasoning repeatedly refers to nonexistent material:

- “given text”;
- “the article”;
- an author or book;
- Common Core Standards;
- “as shown on the right”;
- fabricated named techniques and experts.

The chain-of-thought data appears to have taught a source-analysis template that activates even when no source is present.

### 4. Much of the visible reasoning is templated rather than grounded

Common patterns include:

- generic task restatement;
- long numbered plans;
- self-certifying reflections;
- claims that the answer is complete or accurate without verification;
- irrelevant meta-learning and self-improvement sections;
- unsupported confidence scores.

These traces resemble reasoning-shaped scaffolding. They are not reliable evidence that the model performed grounded deliberation.

### 5. EOS reliability is solved, but response-structure reliability is not

Every variant stopped through EOS with very high reported probability.

However, EOS success did not guarantee:

- a user-facing answer;
- correctly nested tags;
- absence of internal meta-sections;
- role consistency;
- grounded attribution.

Stopping and formatting must be scored separately.

## Provisional Ranking

For these two prompts only:

1. Pythia-2.8B full fine-tuning
2. 256_MOD / v1_2
3. 128_MOD / v1_1
4. Custom_MOD / v1_4
5. V2 layer-unique
6. 256_MOD_epoch_7 / v1_3, because of repeated answer-format failure

This ranking is exploratory and must not be generalized from two prompts.

## Required Evaluation Expansion

Run at least 20–30 confirmed-unseen prompts and score each checkpoint blindly on:

1. answer presence;
2. valid tag structure;
3. absence of reasoning/meta leakage;
4. correct user/teacher/parent role;
5. direct relevance;
6. actionable usefulness;
7. factual accuracy;
8. invented source attribution;
9. repetition;
10. response length;
11. semantic quality;
12. EOS stopping.

The evaluation should include:

- short direct questions;
- factual questions;
- procedures;
- arithmetic;
- rephrased training concepts;
- genuinely held-out concepts;
- ambiguous prompts;
- prompts with and without supplied source text.

Multiple human or model judges should evaluate randomized answers without seeing variant names.

## Evidence Status

| Claim | Status |
|---|---|
| 2.8B was the most consistent model on these two prompts | **VALID within this test** |
| v1_2 was the strongest practical MOD on these two prompts | **VALID within this test** |
| v1_3 failed to emit a proper output twice | **VALID** |
| all tested variants showed unsupported source framing | **VALID** |
| EOS stopping was reliable across the transcript | **VALID** |
| v1_2 is generally the best MOD | **NOT ESTABLISHED** |
| v1_3 is generally worse than the other variants | **NOT ESTABLISHED** |
| visible reasoning traces demonstrate genuine reasoning | **NOT SUPPORTED** |
| the unique-table architecture is inherently worse at instruction following | **NOT ESTABLISHED** |

## 2026-07-20 direct-answer follow-up

A new V3 checkpoint was trained on a rebuilt direct-answer dataset that discards visible thinking. The owner-confirmed step-600 chat test is recorded separately in [Pythia V3 Chat Evaluation](2026-07-20-pythia-v3-chat-evaluation.md).

The new transcript shows fewer explicit reasoning/meta tags, but severe reliability failures remain: only two of six arithmetic/consistency probes were correct, a false arithmetic premise was accepted, an exact word-count instruction was ignored, and a procrastination answer drifted off-topic. Direct-answer formatting did not by itself solve semantic grounding.

## Current Conclusion

The chat test changes the interpretation of the metric sweep.

The best held-out perplexity still belongs to shared v1_3/v1_4, but v1_3 is not currently deployable without output-format repair. v1_2 is the strongest practical MOD checkpoint in this small qualitative sample.

The result strengthens the repository's existing principle:

> Teacher-forced perplexity is necessary but insufficient. Free generation, format reliability, grounding, and task alignment must be evaluated before selecting a winner.
