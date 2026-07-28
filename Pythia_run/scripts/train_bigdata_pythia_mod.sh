#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ATTN_UNIQUE_MOD_COUNT="${ATTN_UNIQUE_MOD_COUNT:-1}"
FFN_UNIQUE_MOD_COUNT="${FFN_UNIQUE_MOD_COUNT:-1}"

python train.py \
  --experiment-type frozen_mod \
  --model-name EleutherAI/pythia-1.4b \
  --tokenizer-name EleutherAI/pythia-1.4b \
  --dataset-path data/bigdata/split/train.txt \
  --validation-dataset-path data/bigdata/split/validation.txt \
  --text_data_format qa_sft \
  --run-dir runs/bigdata_pythia_1.4b_token_mod_v1_1 \
  --device cuda \
  --seed 42 \
  --epochs 50 \
  --batch_size 32 \
  --grad_accum_steps 1 \
  --seq_len 512 \
  --shuffle_buffer 1024 \
  --num_workers 2 \
  --emb-mod-dim 128 \
  --out-mod-dim 128 \
  --attn-mod-dim 128 \
  --ffn-mod-dim 256 \
  --attn-unique-mod-count "$ATTN_UNIQUE_MOD_COUNT" \
  --ffn-unique-mod-count "$FFN_UNIQUE_MOD_COUNT" \
  --modifier-lr 3e-4 \
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
