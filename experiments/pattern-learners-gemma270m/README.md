# Pattern Learners on Gemma 3 270M

A controlled prototype for adding **named, isolated residual learners** to a pretrained Transformer while keeping general computation shared.

The first experiment compares two approximately parameter-matched designs on `google/gemma-3-270m`:

| Variant | Placement | Bottleneck | Learner parameters |
|---|---|---:|---:|
| Single-layer | After the final decoder layer | 640 | 819,201 |
| All-layers | After every one of 18 decoder layers | 36 per layer | 829,458 |

A new learner is zero-effect at initialization. Its output projection starts at zero, so attaching it does not alter the pretrained model before training.

## Included datasets

The repository datasets are used directly:

- `add.txt` — 200 addition questions in varied formats
- `multiply.txt` — 200 multiplication questions in varied formats

Both use numbered Q/A blocks:

```text
1. Q: What is 4 + 1?
   A: 5
```

Pass either file through `--data-file`. The trainer parses it and creates a deterministic shuffled split. The default `--validation-ratio 0.2` produces 160 training and 40 validation examples.

JSONL remains supported:

```json
{"prompt": "What is 17 + 8?", "answer": "25"}
```

Only answer tokens contribute to the language-model loss.

## Independent freezing and learning rates

Parameters are separated into three non-overlapping groups:

- **Embedding:** input embedding plus tied/untied output embedding or LM head
- **Backbone:** every original model parameter outside the embedding group
- **Learner:** named residual pattern learner modules

Each group has an independent learning rate:

```bash
--embedding-lr 1e-5 \
--backbone-lr 1e-5 \
--learner-lr 3e-4
```

Each group can be frozen independently:

```bash
--freeze-embeddings --freeze-backbone --unfreeze-learner
```

The opposite flags are available:

```bash
--unfreeze-embeddings --unfreeze-backbone --freeze-learner
```

Gemma ties its LM head to token embeddings. The shared tensor is assigned only to the embedding group.

## Setup

Accept the Gemma license on Hugging Face, then authenticate:

```bash
pip install -r requirements.txt
huggingface-cli login
```

## Train addition from the pretrained model

### Model A — single final-layer learner

```bash
python train.py \
  --config configs/single_layer.json \
  --data-file add.txt \
  --output-dir runs/single/addition \
  --pattern-name addition \
  --epochs 20 \
  --batch-size 8 \
  --gradient-accumulation 2 \
  --eval-every 10 \
  --freeze-embeddings \
  --freeze-backbone \
  --unfreeze-learner
```

### Model B — small learner at every layer

```bash
python train.py \
  --config configs/all_layers.json \
  --data-file add.txt \
  --output-dir runs/all/addition \
  --pattern-name addition \
  --epochs 20 \
  --batch-size 8 \
  --gradient-accumulation 2 \
  --eval-every 10 \
  --freeze-embeddings \
  --freeze-backbone \
  --unfreeze-learner
```

## Add multiplication without modifying addition

Load the addition checkpoint, create a new `multiplication` learner, and train only that learner. Existing learners are explicitly frozen.

### Model A

```bash
python train.py \
  --config configs/single_layer.json \
  --source-checkpoint runs/single/addition/best \
  --data-file multiply.txt \
  --output-dir runs/single/addition-then-multiplication \
  --pattern-name multiplication \
  --epochs 20 \
  --batch-size 8 \
  --gradient-accumulation 2 \
  --eval-every 10 \
  --freeze-embeddings \
  --freeze-backbone \
  --unfreeze-learner
```

### Model B

```bash
python train.py \
  --config configs/all_layers.json \
  --source-checkpoint runs/all/addition/best \
  --data-file multiply.txt \
  --output-dir runs/all/addition-then-multiplication \
  --pattern-name multiplication \
  --epochs 20 \
  --batch-size 8 \
  --gradient-accumulation 2 \
  --eval-every 10 \
  --freeze-embeddings \
  --freeze-backbone \
  --unfreeze-learner
```

The resulting checkpoint contains both `addition` and `multiplication`. Only the learner named by `--pattern-name` is trainable during each run.

## Train other model parts

Learner plus embeddings:

```bash
--unfreeze-embeddings --freeze-backbone --unfreeze-learner \
--embedding-lr 1e-5 --learner-lr 3e-4
```

Full model plus learner:

```bash
--unfreeze-embeddings --unfreeze-backbone --unfreeze-learner \
--embedding-lr 5e-6 --backbone-lr 1e-5 --learner-lr 3e-4
```

Original model only control:

```bash
--unfreeze-embeddings --unfreeze-backbone --freeze-learner
```

At startup, the script prints exact parameter counts, trainable counts, active pattern, and learning rate for every group. It aborts when every group is frozen.

## Evaluate either stored learner

```bash
python evaluate.py \
  --checkpoint runs/single/addition-then-multiplication/best \
  --pattern-name addition \
  --prompt "What is 31 + 47?"
```

```bash
python evaluate.py \
  --checkpoint runs/single/addition-then-multiplication/best \
  --pattern-name multiplication \
  --prompt "What is 13 times 8?"
```

## Checkpoint behavior

Checkpoints store:

- every named learner;
- the tokenizer;
- current trainable base parameters;
- inherited base deltas from a source checkpoint;
- the dataset split metadata;
- active pattern and optimizer-group configuration.

Frozen runs do not duplicate the complete 270M base model.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover zero-effect initialization, layer placement, parameter matching, tied embeddings, independent freezing/LRs, active-pattern-only training, Q/A text parsing, and deterministic splitting.
