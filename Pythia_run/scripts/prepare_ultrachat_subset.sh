#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TARGET_SAMPLES="${TARGET_SAMPLES:-50000}"
OUTPUT_DIR="${OUTPUT_DIR:-data/ultrachat_filtered}"

python prepare_ultrachat_subset.py \
  --dataset HuggingFaceH4/ultrachat_200k \
  --source-splits train_sft test_sft \
  --output-dir "$OUTPUT_DIR" \
  --tokenizer-name EleutherAI/pythia-1.4b \
  --target-samples "$TARGET_SAMPLES" \
  --min-turns 2 \
  --max-turns 6 \
  --min-tokens 128 \
  --max-tokens 1024 \
  --validation-fraction 0.02 \
  --test-fraction 0.02 \
  --seed 42 \
  --streaming
