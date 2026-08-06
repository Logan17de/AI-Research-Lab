# Tiny Word-Level Pattern Learners

A clean from-scratch experiment for testing whether isolated residual learners can acquire arithmetic patterns without relying on a pretrained model's existing knowledge.

## Architecture

- 4 decoder layers
- hidden size 128
- 4 attention heads (`head_dim = 32`)
- SwiGLU FFN size 512
- RoPE positional encoding
- pre-RMSNorm
- untied token embedding and LM head
- no linear biases
- context length 256
- randomly initialized shared base

## Parameter-matched learners

| Model | Placement | Shape | Learner parameters |
|---|---|---|---:|
| A | after final layer | `128 → 128 → 128` | 32,769 |
| B | after all 4 layers | `128 → 32 → 128` per layer | 32,772 |

Fresh learners have zero initial effect because each output projection starts at zero.

## Live word tokenizer

The tokenizer is created locally and performs exactly:

```python
text.strip().split()
```

Example:

```text
What is 12 + 5? -> What | is | 12 | + | 5?
```

It is Unicode-safe and accepts words from any language. Languages without spaces, such as ordinary Japanese or Chinese text, become large phrase-level tokens unless spaces are inserted.

The vocabulary is built once and stored with the shared base. Model A and Model B therefore use identical token IDs, random embeddings, random LM-head weights, and Transformer weights.

Whole numbers are individual tokens. `init_base.py` adds tokens from `0` through `10000` by default without training the model on arithmetic. This lets an unseen answer exist in the output vocabulary instead of becoming `<unk>`.

## Colab setup

```python
%cd /content/AI-Research-Lab
!git pull origin main
%cd experiments/tiny-word-pattern-learners
!pip install -q -r requirements.txt
```

## 1. Initialize one shared random base

This builds the whitespace vocabulary and initializes the base. It does not train any model parameters.

```python
!python init_base.py \
  --tokenizer-files \
    ../pattern-learners-gemma270m/add.txt \
    ../pattern-learners-gemma270m/multiply.txt \
  --output-dir runs/base-word-seed42 \
  --numeric-vocab-min 0 \
  --numeric-vocab-max 10000 \
  --layers 4 \
  --hidden-size 128 \
  --heads 4 \
  --ffn-hidden-size 512 \
  --max-seq-len 256 \
  --seed 42
```

Both learner variants must start from this exact checkpoint.

## 2. Model A — final-layer learner only

```python
!python train.py \
  --config configs/single_layer.json \
  --base-checkpoint runs/base-word-seed42 \
  --data-file ../pattern-learners-gemma270m/add.txt \
  --output-dir runs/single/addition \
  --pattern-name addition \
  --epochs 200 \
  --batch-size 16 \
  --gradient-accumulation 1 \
  --eval-every 50 \
  --early-stopping-patience 6 \
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

## 3. Model B — all-layer learner only

```python
!python train.py \
  --config configs/all_layers.json \
  --base-checkpoint runs/base-word-seed42 \
  --data-file ../pattern-learners-gemma270m/add.txt \
  --output-dir runs/all/addition \
  --pattern-name addition \
  --epochs 200 \
  --batch-size 16 \
  --gradient-accumulation 1 \
  --eval-every 50 \
  --early-stopping-patience 6 \
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

For these first runs, only the selected learner receives gradients. Embeddings, all attention/FFN/RMSNorm parameters, and the untied LM head remain frozen.

Validation prints exact-answer accuracy, loss, and perplexity. Early stopping follows exact accuracy first and validation loss as a tie-breaker.

## Compare learner against the unchanged base

```python
!python evaluate.py \
  --checkpoint runs/all/addition/best \
  --data-file ../pattern-learners-gemma270m/add.txt \
  --pattern-name addition \
  --validation-only \
  --split-seed 42
```

```python
!python evaluate.py \
  --checkpoint runs/all/addition/best \
  --data-file ../pattern-learners-gemma270m/add.txt \
  --pattern-name base \
  --validation-only \
  --split-seed 42
```

## Interactive test

```python
%run chat.py \
  --checkpoint runs/all/addition/best \
  --pattern-name addition
```

Commands inside chat:

```text
/patterns
/use addition
/base
/exit
```

## Optional tiny base plasticity later

Do not mix this with the learner-only comparison. A separate controlled run can use:

```text
--unfreeze-embeddings --embedding-lr 1e-6
--unfreeze-backbone --backbone-lr 1e-6
--unfreeze-lm-head --lm-head-lr 1e-6
--unfreeze-learner --learner-lr 1e-3
```

## Add multiplication later

```python
!python train.py \
  --config configs/all_layers.json \
  --source-checkpoint runs/all/addition/best \
  --data-file ../pattern-learners-gemma270m/multiply.txt \
  --output-dir runs/all/addition-then-multiplication \
  --pattern-name multiplication \
  --epochs 200 \
  --batch-size 16 \
  --eval-every 50 \
  --early-stopping-patience 6 \
  --freeze-embeddings \
  --freeze-backbone \
  --freeze-lm-head \
  --unfreeze-learner \
  --learner-lr 1e-3 \
  --seed 42
```

The existing addition learner remains frozen; only the new multiplication learner trains.

## Tests

```bash
python -m unittest discover -s tests -v
```
