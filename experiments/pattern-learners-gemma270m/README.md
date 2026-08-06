# Pattern Learners on Gemma 3 270M

A controlled prototype for adding **named, isolated residual learners** to a pretrained Transformer while keeping general computation shared.

The first experiment compares two approximately parameter-matched designs on `google/gemma-3-270m`:

| Variant | Placement | Bottleneck | Learner parameters |
|---|---|---:|---:|
| Single-layer | After the final decoder layer | 640 | 819,201 |
| All-layers | After every one of 18 decoder layers | 36 per layer | 829,458 |

The learner output projection is zero-initialized, so attaching a fresh learner has exactly zero effect before training.

## Independent freezing and learning rates

The code separates parameters into three non-overlapping groups:

- **Embedding:** input embedding plus tied/untied output embedding or LM head
- **Backbone:** every original model parameter not in the embedding group
- **Learner:** the newly attached pattern learner modules

Each group has its own learning rate and can be frozen independently:

```bash
--embedding-lr 1e-5 \
--backbone-lr 1e-5 \
--learner-lr 3e-4 \
--freeze-embeddings \
--freeze-backbone \
--unfreeze-learner
```

The opposite flags are also available: `--unfreeze-embeddings`, `--unfreeze-backbone`, and `--freeze-learner`.

> Gemma ties its LM head to the token embeddings. They are intentionally treated as one parameter group, so the same tensor is never assigned to two optimizers or two learning rates.

## Setup

Gemma weights require accepting Google's Gemma license on Hugging Face and logging in locally.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
huggingface-cli login
```

## Generate the first dataset

```bash
python generate_data.py \
  --output-dir data/addition \
  --operation addition \
  --train-examples 4000 \
  --validation-examples 500
```

Custom data uses JSONL:

```json
{"prompt": "What is 17 + 8?", "answer": "25"}
```

Only answer tokens contribute to the language-model loss.

## Model A: one full-width learner at one layer

```bash
python train.py \
  --config configs/single_layer.json \
  --train-file data/addition/train.jsonl \
  --validation-file data/addition/validation.jsonl \
  --output-dir runs/single-layer \
  --pattern-name addition
```

Equivalent architecture arguments:

```bash
--learner-mode single --learner-dim 640 --learner-layer -1
```

## Model B: small learner at every layer

```bash
python train.py \
  --config configs/all_layers.json \
  --train-file data/addition/train.jsonl \
  --validation-file data/addition/validation.jsonl \
  --output-dir runs/all-layers \
  --pattern-name addition
```

Equivalent architecture arguments:

```bash
--learner-mode all --learner-dim 36
```

You can derive the all-layer width automatically from a single-layer width:

```bash
--learner-mode all --match-single-dim 640
```

## Train other model parts

Learner only, the default controlled experiment:

```bash
--freeze-embeddings --freeze-backbone --unfreeze-learner
```

Learner plus embeddings:

```bash
--unfreeze-embeddings --freeze-backbone --unfreeze-learner \
--embedding-lr 1e-5 --learner-lr 3e-4
```

Full model plus learner, with three independent learning rates:

```bash
--unfreeze-embeddings --unfreeze-backbone --unfreeze-learner \
--embedding-lr 5e-6 --backbone-lr 1e-5 --learner-lr 3e-4
```

Freeze the learner and train only the original model as a control:

```bash
--unfreeze-embeddings --unfreeze-backbone --freeze-learner
```

At startup, the script prints exact total/trainable counts and learning rates for all three groups. It aborts if every group is frozen.

## Evaluate a checkpoint

```bash
python evaluate.py \
  --checkpoint runs/single-layer/best \
  --prompt "What is 31 + 47?"
```

Checkpoints store learner weights separately. Any trainable original-model parameters are stored as a named delta file, so frozen runs do not duplicate the 270M base checkpoint.

## Named learners

`PatternLearnerSystem` already supports adding independent names:

```python
system.add_pattern("addition")
system.add_pattern("multiplication")
system.set_active_pattern("addition")
```

Existing learners are not modified when a new learner is registered. Automatic decomposition, learner identification, and creation are intentionally left for the next experiment; this project first tests whether the learner placement itself can absorb a pattern without disturbing frozen knowledge.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests verify zero-effect initialization, correct layer placement, parameter matching, tied-embedding deduplication, independent freezing, and separate optimizer learning rates.
