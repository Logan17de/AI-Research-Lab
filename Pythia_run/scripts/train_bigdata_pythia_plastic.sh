#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PLASTIC_LAST_N_LAYERS="${PLASTIC_LAST_N_LAYERS:-4}"
PLASTIC_LR="${PLASTIC_LR:-1e-5}"
PLASTIC_LAYERNORM_LR="${PLASTIC_LAYERNORM_LR:-3e-5}"
ATTN_UNIQUE_MOD_COUNT="${ATTN_UNIQUE_MOD_COUNT:-1}"
FFN_UNIQUE_MOD_COUNT="${FFN_UNIQUE_MOD_COUNT:-1}"

python train.py \
  --experiment-type frozen_mod \
  --model-name EleutherAI/pythia-1.4b \
  --tokenizer-name EleutherAI/pythia-1.4b \
  --dataset-path data/bigdata/split/train.txt \
  --validation-dataset-path data/bigdata/split/validation.txt \
  --text_data_format qa_sft \
  --run-dir runs/bigdata_pythia_1.4b_token_mod_plastic_tail \
  --device cuda \
  --seed 42 \
  --epochs 50 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --seq_len 512 \
  --shuffle_buffer 1024 \
  --num_workers 2 \
  --emb-mod-dim 128 \
  --out-mod-dim 128 \
  --attn-mod-dim 128 \
  --ffn-mod-dim 512 \
  --attn-unique-mod-count "$ATTN_UNIQUE_MOD_COUNT" \
  --ffn-unique-mod-count "$FFN_UNIQUE_MOD_COUNT" \
  --modifier-lr 3e-4 \
  --plastic-last-n-layers "$PLASTIC_LAST_N_LAYERS" \
  --plastic-final-layer-norm 1 \
  --plastic-lm-head 0 \
  --plastic-lr "$PLASTIC_LR" \
  --plastic-layernorm-lr "$PLASTIC_LAYERNORM_LR" \
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
