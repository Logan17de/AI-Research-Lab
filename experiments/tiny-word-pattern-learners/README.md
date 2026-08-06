# Tiny Word-Level Pattern Learners

A clean experiment for testing whether isolated residual learners can acquire arithmetic patterns without relying on pretrained-model knowledge.

## Fixed architecture

- 4 decoder layers
- hidden size 128
- 4 attention heads
- head dimension 32
- SwiGLU FFN size 512
- RoPE positional encoding
- pre-RMSNorm
- untied token embedding and LM head
- no linear biases
- context length 256
- randomly initialized shared base

## Learner comparison

| Model | Placement | Shape | Parameters |
|---|---|---|---:|
| A | after final layer | `128 → 128 → 128` | 32,769 |
| B | after all 4 layers | `128 → 32 → 128` per layer | 32,772 |

Both learners begin with exactly zero effect because their output projections are initialized to zero.

## Live word tokenizer

The tokenizer is built locally from files supplied to `init_base.py` and performs exactly:

```python
text.strip().split()
```

Therefore:

```text
What is 12 + 5? -> What | is | 12 | + | 5?
```

It is Unicode-safe and can store words from many languages, but it only separates text where spaces exist. Japanese, Chinese, and Thai text without spaces becomes one large token. This tokenizer is deliberately simple for this experiment, not a production multilingual tokenizer.

The vocabulary is built once and saved with the shared base. It must not expand independently between Model A and Model B, because changing vocabulary size would change the embedding and LM-head parameters.

## Important experimental limitation

Whole numbers are individual word tokens. A result that was not included while building the tokenizer becomes `<unk>` and cannot be generated. Build the tokenizer using every dataset that may later be evaluated, even though the model itself remains randomly initialized and receives no training examples during base creation.

## Setup

```bash
pip install -r requirements.txt
```

The existing arithmetic datasets live in the Gemma experiment directory.

## 1. Initialize one shared random base

Use both addition and multiplication files to establish one stable vocabulary. This command does **not** train the model.

```bash
python init_base.py \
  --tokenizer-files \
    ../pattern-learners-gemma270m/add.txt \
    ../pattern-learners-gemma270m/multiply.txt \
  --output-dir runs/base \
  --layers 4 \
  --hidden-size 128 \
  --heads 4 \
  --ffn-hidden-size 512 \
  --max-seq-len 256 \
  --seed 42
```

Both A and B must start from this same `runs/base` checkpoint.

## 2. Train only the addition learner

All original parameters are frozen. The embedding LR, backbone LR, and LM-head LR are printed but inactive.

### Model A — final-layer learner

```bash
python train.py \
  --config configs/single_layer.json \
  --base-checkpoint runs/base \
  --data-file ../pattern-learners-gemma270m/add.txt \
  --output-dir runs/single/addition \
  --pattern-name addition \
  --epochs 100 \
  --batch-size 16 \
  --gradient-accumulation 1 \
  --eval-every 10 \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 0.0 \
  --freeze-embeddings \
  --freeze-backbone \
  --freeze-lm-head \
  --unfreeze-learner \
  --embedding-lr 1e-6 \
  --backbone-lr 1e-6 \
  --lm-head-lr 1e-6 \
  --learner-lr 1e-3 \
  --weight-decay 0.01 \
  --seed 42
```

### Model B — learner at all layers

```bash
python train.py \
  --config configs/all_layers.json \
  --base-checkpoint runs/base \
  --data-file ../pattern-learners-gemma270m/add.txt \
  --output-dir runs/all/addition \
  --pattern-name addition \
  --epochs 100 \
  --batch-size 16 \
  --gradient-accumulation 1 \
  --eval-every 10 \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 0.0 \
  --freeze-embeddings \
  --freeze-backbone \
  --freeze-lm-head \
  --unfreeze-learner \
  --embedding-lr 1e-6 \
  --backbone-lr 1e-6 \
  --lm-head-lr 1e-6 \
  --learner-lr 1e-3 \
  --weight-decay 0.01 \
  --seed 42
```

Exact-answer accuracy, validation loss, and perplexity are measured at every evaluation. Early stopping primarily follows exact-answer accuracy, using validation loss to break ties.

## Optional tiny backbone plasticity later

The current learner-only experiment freezes the whole base. A later controlled run can enable extremely small backbone updates:

```bash
--freeze-embeddings \
--unfreeze-backbone \
--freeze-lm-head \
--unfreeze-learner \
--backbone-lr 1e-6 \
--learner-lr 1e-3
```

Do not mix this with the first comparison; it answers a different question.

## Evaluate

```bash
python evaluate.py \
  --checkpoint runs/single/addition/best \
  --data-file ../pattern-learners-gemma270m/add.txt \
  --pattern-name addition \
  --validation-only \
  --split-seed 42
```

```bash
python evaluate.py \
  --checkpoint runs/all/addition/best \
  --data-file ../pattern-learners-gemma270m/add.txt \
  --pattern-name addition \
  --validation-only \
  --split-seed 42
```

## Interactive test

```bash
python chat.py \
  --checkpoint runs/all/addition/best \
  --pattern-name addition
```

## Add multiplication later

Use the best addition checkpoint as the source. The addition learner remains frozen; only the new multiplication learner trains.

```bash
python train.py \
  --config configs/all_layers.json \
  --source-checkpoint runs/all/addition/best \
  --data-file ../pattern-learners-gemma270m/multiply.txt \
  --output-dir runs/all/addition-then-multiplication \
  --pattern-name multiplication \
  --epochs 100 \
  --batch-size 16 \
  --eval-every 10 \
  --early-stopping-patience 8 \
  --freeze-embeddings \
  --freeze-backbone \
  --freeze-lm-head \
  --unfreeze-learner \
  --learner-lr 1e-3
```

## Tests

```bash
python -m unittest discover -s tests -v
```
