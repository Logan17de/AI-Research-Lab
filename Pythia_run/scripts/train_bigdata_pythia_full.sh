#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python train.py \
  --experiment-type full_finetune \
  --model-name EleutherAI/pythia-2.8b \
  --tokenizer-name EleutherAI/pythia-1.4b \
  --dataset-path data/bigdata/split/train.txt \
  --validation-dataset-path data/bigdata/split/validation.txt \
  --text_data_format qa_sft \
  --run-dir runs/bigdata_pythia_2.8b_full_ft \
  --device cuda \
  --seed 42 \
  --epochs 50 \
  --batch_size 16 \
  --grad_accum_steps 2 \
  --seq_len 512 \
  --shuffle_buffer 1024 \
  --num_workers 2 \
  --emb-mod-dim 0 \
  --out-mod-dim 0 \
  --attn-mod-dim 0 \
  --ffn-mod-dim 0 \
  --lr 1e-5 \
  --modifier-lr 0 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  --warmup-ratio 0.03 \
  --min_lr_ratio 0.03 \
  --eval_every 50 \
  --eval_batches 0 \
  --early_stop_patience 10 \
  --early_stop_min_delta 0.001 \
  --save_every 250 \
  --log_every 10 \
  --plot_every 50 \
  --bf16 \
  --resume 0
