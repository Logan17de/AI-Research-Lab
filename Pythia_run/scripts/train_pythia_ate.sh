#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python train.py \
  --experiment-type ate \
  --model-name EleutherAI/pythia-1.4b \
  --tokenizer-name EleutherAI/pythia-1.4b \
  --text_data_path data/ultrachat_filtered/train.jsonl \
  --eval_text_data_path data/ultrachat_filtered/validation.jsonl \
  --dataset-manifest data/ultrachat_filtered/manifest.json \
  --strict-sample-order 1 \
  --text_data_format chat_jsonl \
  --run-dir runs/ultrachat_pythia_1.4b_ate \
  --device cuda \
  --seed 42 \
  --epochs 7 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --seq_len 1024 \
  --shuffle_buffer 0 \
  --num_workers 0 \
  --new-attn-heads 1 \
  --new-ffn-layers 1 \
  --plasticity-mode quadratic \
  --plasticity-base-lr 1e-5 \
  --lr 3e-4 \
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
  --gradient-checkpointing \
  --bf16 \
  --resume 0
