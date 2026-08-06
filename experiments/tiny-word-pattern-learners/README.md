# Tiny Word-Level Pattern Learners

A from-scratch experiment comparing three places where a named pattern learner can own or modify computation without relying on pretrained arithmetic knowledge.

## Shared base

- 4 decoder layers
- hidden size 128
- 4 attention heads (`head_dim = 32`)
- SwiGLU FFN size 512
- RoPE
- pre-RMSNorm
- untied token embedding and shared LM head
- bias-free linear layers
- context length 256
- random initialization

The tokenizer is live, Unicode-safe, and splits exactly on whitespace:

```python
text.strip().split()
```

For example:

```text
what is 12 + 5 ? -> what | is | 12 | + | 5 | ?
```

## Three learner models

All three receive the same frozen word embeddings and use their own pattern-specific RMSNorm and LM head. The pattern head is counted as learner capacity and is trainable. The random shared LM head remains frozen.

| Model | Computation | Core learner parameters |
|---|---|---:|
| **1. Standalone path** | shared embedding → one learner Transformer block → learner head | 78,080 |
| **2. Pre-activation** | `gate = W_gate(h) + L_pre(h)` before SwiGLU at all 4 FFNs | 76,804 |
| **3. Post-activation** | `z = z + L_post(z)` after SwiGLU at all 4 FFNs | 77,828 |

The common pattern head has the same size in every model, so the three complete learners remain closely parameter-matched.

### Model 1 — standalone path

The learner bypasses the frozen random backbone:

```text
shared word embedding
→ learner-owned causal Transformer block
→ learner-owned RMSNorm and LM head
```

One small block is used so its core capacity stays matched to the two FFN-placement learners.

### Model 2 — pre-activation learner

```python
gate = W_gate(h) + L_pre(h)
up = W_up(h)
z = silu(gate) * up
out = W_down(z)
```

The adjustment output is initialized to zero. The learner controls which frozen FFN neurons activate.

### Model 3 — post-activation learner

```python
z = silu(W_gate(h)) * W_up(h)
z = z + L_post(z)
out = W_down(z)
```

The adjustment output is initialized to zero. The learner modifies selected FFN features before the frozen down projection.

## Controlled arithmetic data

The previous natural-language file had many unique whole-number answers. With a whitespace tokenizer, an unseen whole number is an independent token, so that split can confuse token coverage with arithmetic learning.

The included generator creates a controlled split where:

- commuted pairs such as `2 + 7` and `7 + 2` stay together;
- no reversed validation pair leaks into training;
- every validation answer token occurs somewhere in training;
- spaces explicitly separate operators and punctuation.

## Colab setup

```python
%cd /content/AI-Research-Lab
!git pull origin main
%cd experiments/tiny-word-pattern-learners
!pip install -q -r requirements.txt
```

## 1. Generate fixed addition and multiplication splits

```python
!python generate_arithmetic_data.py \
  --operation addition \
  --min-operand 0 \
  --max-operand 19 \
  --validation-ratio 0.2 \
  --output-dir data/addition \
  --seed 42
```

```python
!python generate_arithmetic_data.py \
  --operation multiplication \
  --min-operand 0 \
  --max-operand 19 \
  --validation-ratio 0.2 \
  --output-dir data/multiplication \
  --seed 42
```

## 2. Initialize one shared random base

This creates the vocabulary and random weights only. It performs no training.

```python
!python init_base.py \
  --tokenizer-files \
    data/addition/train.jsonl \
    data/addition/validation.jsonl \
    data/multiplication/train.jsonl \
    data/multiplication/validation.jsonl \
  --output-dir runs/base-word-seed42 \
  --numeric-vocab-min 0 \
  --numeric-vocab-max 400 \
  --layers 4 \
  --hidden-size 128 \
  --heads 4 \
  --ffn-hidden-size 512 \
  --max-seq-len 256 \
  --seed 42
```

Every model below must start from this exact checkpoint.

## 3. Train Model 1 — standalone path

```python
!python train.py \
  --config configs/standalone.json \
  --base-checkpoint runs/base-word-seed42 \
  --data-file data/addition/train.jsonl \
  --validation-file data/addition/validation.jsonl \
  --output-dir runs/three/standalone/addition \
  --pattern-name addition \
  --epochs 200 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --eval-every 20 \
  --max-answer-tokens 2 \
  --early-stopping-patience 10 \
  --freeze-embeddings \
  --freeze-backbone \
  --freeze-lm-head \
  --unfreeze-learner \
  --learner-lr 1e-3 \
  --weight-decay 0.01 \
  --seed 42
```

## 4. Train Model 2 — pre-activation

```python
!python train.py \
  --config configs/pre_activation.json \
  --base-checkpoint runs/base-word-seed42 \
  --data-file data/addition/train.jsonl \
  --validation-file data/addition/validation.jsonl \
  --output-dir runs/three/pre-activation/addition \
  --pattern-name addition \
  --epochs 200 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --eval-every 20 \
  --max-answer-tokens 2 \
  --early-stopping-patience 10 \
  --freeze-embeddings \
  --freeze-backbone \
  --freeze-lm-head \
  --unfreeze-learner \
  --learner-lr 1e-3 \
  --weight-decay 0.01 \
  --seed 42
```

## 5. Train Model 3 — post-activation

```python
!python train.py \
  --config configs/post_activation.json \
  --base-checkpoint runs/base-word-seed42 \
  --data-file data/addition/train.jsonl \
  --validation-file data/addition/validation.jsonl \
  --output-dir runs/three/post-activation/addition \
  --pattern-name addition \
  --epochs 200 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --eval-every 20 \
  --max-answer-tokens 2 \
  --early-stopping-patience 10 \
  --freeze-embeddings \
  --freeze-backbone \
  --freeze-lm-head \
  --unfreeze-learner \
  --learner-lr 1e-3 \
  --weight-decay 0.01 \
  --seed 42
```

Only the selected learner—including its pattern-specific output head—receives gradients. Shared embeddings, attention, FFNs, RMSNorms, and the shared LM head remain unchanged.

## Evaluate the three best checkpoints

```python
!python evaluate.py \
  --checkpoint runs/three/standalone/addition/best \
  --data-file data/addition/validation.jsonl \
  --pattern-name addition
```

```python
!python evaluate.py \
  --checkpoint runs/three/pre-activation/addition/best \
  --data-file data/addition/validation.jsonl \
  --pattern-name addition
```

```python
!python evaluate.py \
  --checkpoint runs/three/post-activation/addition/best \
  --data-file data/addition/validation.jsonl \
  --pattern-name addition
```

## Interactive test

```python
%run chat.py \
  --checkpoint runs/three/post-activation/addition/best \
  --pattern-name addition \
  --max-answer-tokens 2
```

## Add multiplication later

Use the matching addition checkpoint as the source. For example, post-activation:

```python
!python train.py \
  --config configs/post_activation.json \
  --source-checkpoint runs/three/post-activation/addition/best \
  --data-file data/multiplication/train.jsonl \
  --validation-file data/multiplication/validation.jsonl \
  --output-dir runs/three/post-activation/addition-then-multiplication \
  --pattern-name multiplication \
  --epochs 200 \
  --batch-size 32 \
  --eval-every 20 \
  --max-answer-tokens 2 \
  --early-stopping-patience 10 \
  --freeze-embeddings \
  --freeze-backbone \
  --freeze-lm-head \
  --unfreeze-learner \
  --learner-lr 1e-3 \
  --seed 42
```

The existing addition learner and its head stay frozen while the new multiplication learner trains.

## Tests

```bash
python -m unittest discover -s tests -v
```

The original after-block residual configs remain available only for reproducing the earlier negative experiment.
