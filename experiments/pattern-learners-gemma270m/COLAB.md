# Colab commands

Use a GPU runtime, then run these cells in order.

## 1. Clone and install

```python
!git clone https://github.com/Logan17de/AI-Research-Lab.git
%cd /content/AI-Research-Lab/experiments/pattern-learners-gemma270m
!pip install -q -r requirements.txt
```

## 2. Hugging Face login

Accept the Gemma license on Hugging Face first.

```python
from huggingface_hub import notebook_login
notebook_login()
```

## 3. Train Model A on addition

```python
!python train.py \
  --config configs/single_layer.json \
  --data-file add.txt \
  --output-dir runs/single/addition \
  --pattern-name addition \
  --epochs 20 \
  --batch-size 8 \
  --gradient-accumulation 2 \
  --eval-every 10 \
  --dtype bfloat16 \
  --freeze-embeddings \
  --freeze-backbone \
  --unfreeze-learner
```

For a GPU without BF16 support, replace `--dtype bfloat16` with `--dtype float16`.

## 4. Add multiplication to Model A

```python
!python train.py \
  --config configs/single_layer.json \
  --source-checkpoint runs/single/addition/best \
  --data-file multiply.txt \
  --output-dir runs/single/addition-then-multiplication \
  --pattern-name multiplication \
  --epochs 20 \
  --batch-size 8 \
  --gradient-accumulation 2 \
  --eval-every 10 \
  --dtype bfloat16 \
  --freeze-embeddings \
  --freeze-backbone \
  --unfreeze-learner
```

## 5. Train Model B on addition

```python
!python train.py \
  --config configs/all_layers.json \
  --data-file add.txt \
  --output-dir runs/all/addition \
  --pattern-name addition \
  --epochs 20 \
  --batch-size 8 \
  --gradient-accumulation 2 \
  --eval-every 10 \
  --dtype bfloat16 \
  --gradient-checkpointing \
  --freeze-embeddings \
  --freeze-backbone \
  --unfreeze-learner
```

## 6. Add multiplication to Model B

```python
!python train.py \
  --config configs/all_layers.json \
  --source-checkpoint runs/all/addition/best \
  --data-file multiply.txt \
  --output-dir runs/all/addition-then-multiplication \
  --pattern-name multiplication \
  --epochs 20 \
  --batch-size 8 \
  --gradient-accumulation 2 \
  --eval-every 10 \
  --dtype bfloat16 \
  --gradient-checkpointing \
  --freeze-embeddings \
  --freeze-backbone \
  --unfreeze-learner
```

## 7. Check both learners

```python
!python evaluate.py \
  --checkpoint runs/single/addition-then-multiplication/best \
  --pattern-name addition \
  --prompt "What is 31 + 47?"
```

```python
!python evaluate.py \
  --checkpoint runs/single/addition-then-multiplication/best \
  --pattern-name multiplication \
  --prompt "What is 13 times 8?"
```
