# Pythia Grouped Token MOD

This folder is the self-contained Pythia training and comparison project moved
out of `GPT_MOD`.

It compares:

- `EleutherAI/pythia-1.4b` with a strictly frozen backbone and grouped Token MOD.
- `EleutherAI/pythia-2.8b` with full fine-tuning and no MOD modules.

The architecture specification is included as both:

- `Token_MOD_Architecture_Specification_v1.1.pdf`

Attention and FFN token-memory tables can be independently shared across
contiguous layer groups. Projections and scales remain unique per layer. For a
24-layer model, this gives eight Attention tables with three layers per table:

```bash
--attn-unique-mod-count 8 \
--ffn-unique-mod-count 1
```

Both counts default to `1`, which preserves the original globally shared table
behavior. The count must evenly divide the model's decoder-layer count; using
the full layer count creates one token-memory table per layer.
- `Token_MOD_Architecture_Specification_v1.1.docx`

## Install

```bash
pip install -r requirements.txt
```

## Validate

```bash
python -m pytest tests -q
python report_pythia_parameters.py --output outputs/token_mod_v1_1_parameter_report.json
```

## Prepare data

```bash
python prepare_pythia_split.py \
  --source-path data/pythia/all_chat.jsonl \
  --output-dir data/pythia/split \
  --data-format chat_jsonl \
  --eval-fraction 0.02 \
  --seed 42
```

## Run comparison

```bash
python run_pythia_comparison.py \
  --experiments frozen_mod plastic_mod full_finetune \
  --train-path data/pythia/split/train.jsonl \
  --validation-path data/pythia/split/validation.jsonl \
  --data-format chat_jsonl \
  --run-root runs/pythia_comparison \
  --target-training-tokens 100000000 \
  --eval-every-tokens 1000000 \
  --save-every-tokens 5000000 \
  --bf16
```

## Train Token MOD on bigdata

The checked-in `data/bigdata/split` contains a deterministic 90/10 split of
`bigdata.txt`. Run the frozen Pythia-1.4B Token MOD v1.1 configuration with:

```bash
bash scripts/train_bigdata_pythia_mod.sh
```

Override the grouped table counts without editing the script:

```bash
ATTN_UNIQUE_MOD_COUNT=8 FFN_UNIQUE_MOD_COUNT=1 \
  bash scripts/train_bigdata_pythia_mod.sh
```

Train the Pythia-2.8B full-fine-tuning comparison with:

```bash
bash scripts/train_bigdata_pythia_full.sh
```

## Controlled base plasticity

Plasticity keeps MOD trainable, freezes the rest of Pythia, and unfreezes only
the explicitly selected base tensors. The recommended first comparison is MOD
plus the final four transformer layers and final LayerNorm:

```bash
bash scripts/train_bigdata_pythia_plastic.sh
```

The script uses MOD LR `3e-4`, plastic base LR `1e-5`, and plastic LayerNorm LR
`3e-5`. Override the tail size or rates through environment variables:

```bash
PLASTIC_LAST_N_LAYERS=1 PLASTIC_LR=1e-5 PLASTIC_LAYERNORM_LR=3e-5 \
  bash scripts/train_bigdata_pythia_plastic.sh
```

Available independent selectors are:

```text
--plastic-last-n-layers N
--plastic-layer-norms 1
--plastic-biases 1
--plastic-attn-output 1
--plastic-ffn-output 1
--plastic-final-layer-norm 1
--plastic-lm-head 1
--plastic-lr 1e-5
--plastic-layernorm-lr 3e-5
```

Suggested ladder:

```text
A  MOD only
B  MOD + --plastic-layer-norms 1 --plastic-final-layer-norm 1
C  MOD + --plastic-last-n-layers 1
D  MOD + --plastic-last-n-layers 4 --plastic-final-layer-norm 1
E  MOD + --plastic-ffn-output 1 --plastic-final-layer-norm 1
```

The dashboard and `metrics.csv` report the plastic learning rates and aggregate
relative drift. `plasticity_drift.jsonl` records drift separately for every
selected transformer layer, final LayerNorm, and LM head at each evaluation.
Sparse checkpoints contain MOD tensors plus only the selected plastic base
tensors; selection mismatches are rejected during resume or inference loading.

## Filtered UltraChat comparison subset

Build the deterministic 50,000-conversation subset with:

```bash
pip install -r requirements.txt
bash scripts/prepare_ultrachat_subset.sh
```

Override its size with `TARGET_SAMPLES`, for example:

```bash
TARGET_SAMPLES=100000 bash scripts/prepare_ultrachat_subset.sh
```

The builder enforces:

- English detection probability of at least 0.90.
- 2-6 complete role messages (1-3 user/assistant exchanges).
- 128-1,024 tokens under the shared Pythia tokenizer.
- Removal of coding-heavy and visibly broken samples.
- Deduplication by normalized first-user prompt.
- Conversation-level train/validation/test splitting.
- Category quotas of 30% explanation/knowledge, 20% practical reasoning,
  15% advice/planning, 15% summarization/rewriting, 10% creative generation,
  and 10% multi-turn follow-up.

Outputs are `train.jsonl`, `validation.jsonl`, `test.jsonl`, and
`manifest.json`. The manifest records ordered sample/token hashes, file hashes,
tokenizer identity, rejection counts, token totals, and category counts.

Run all comparison models against the exact locked train/validation files:

```bash
python run_pythia_comparison.py \
  --experiments frozen_mod plastic_mod full_finetune \
  --train-path data/ultrachat_filtered/train.jsonl \
  --validation-path data/ultrachat_filtered/validation.jsonl \
  --dataset-manifest data/ultrachat_filtered/manifest.json \
  --data-format chat_jsonl \
  --run-root runs/ultrachat_comparison \
  --seq-len 1024 \
  --target-training-tokens 100000000 \
  --eval-every-tokens 1000000 \
  --save-every-tokens 5000000 \
  --num-workers 0 \
  --bf16
```

Supplying the manifest forces `shuffle_buffer=0` and `num_workers=0`. Training
aborts if either JSONL file or the tokenizer differs, ensuring both models use
the same tokenized conversations in the same order. Keep `test.jsonl` untouched
until final comparison.

## Adaptive Transformer Expansion (ATE)

ATE preserves pretrained GPT-NeoX tensors as independent parameters while
adding trainable width/depth tensors. It is not MOD or LoRA. Extra heads keep
the pretrained head dimension fixed, so one head on Pythia-1.4B expands hidden
size from 2048 to 2176 and preserves the original FFN ratio. Added blocks are
appended after the pretrained stack.

```bash
bash scripts/train_pythia_ate.sh
```

The important controls are:

```text
--experiment-type ate
--new-attn-heads N
--new-ffn-layers N
--plasticity-mode off|linear|quadratic|custom
--plasticity-base-lr 1e-5
--lr 3e-4
```

`--lr` applies only to newly added ATE parameters. Original transformer layer
parameters use independent optimizer groups with `plasticity_base_lr` scaled
by `(layer/L)` or `(layer/L)^2`. Layers are numbered from 1 to L; the final
LayerNorm and LM head use multiplier 1. Input embeddings use the first-layer
multiplier. `off` freezes every pretrained parameter.

Custom mode accepts one multiplier per original layer:

```bash
--plasticity-mode custom \
--plasticity-custom-scales 0,0,0,0.1,0.2,0.5
```

The number of custom values must equal the pretrained layer count. At startup,
ATE prints exact original/added/final parameter counts and waits for the exact
input `YES` before creating the optimizer or loading checkpoint state. Automated
runs may pass `--ate-confirm YES`; every other value exits safely.

Width initialization is output-neutral: old-to-old tensors are untouched,
new-to-old contributions and new LM-head columns start at zero, while new
feature-producing tensors use the model initializer. Added transformer blocks
start as identity residual blocks. Segmented expanded LayerNorms retain the
pretrained normalization exactly while normalizing added channels separately.
Every width increment remains a nested generation (`[base | H1 | H2 | ...]`),
including an independent normalization segment. This **Stage-Segmented
LayerNorm** is intentionally not mathematically equivalent to a standard global
widened GPT-NeoX LayerNorm.

Stage controls:

```text
--ate-output-init exact|small
--ate-output-init-scale 0.0
--previous-ate-stages-trainable 0|1
--ate-current-stage-lr 3e-4
--ate-previous-stage-lr 1e-5
--allow-nonpreserving-expansion
```

`exact` is the default and enforces max/mean source-logit differences below
`1e-6`/`1e-7`. `small` is explicitly non-preserving. The unsafe preservation
override is never enabled by default.

ATE checkpoints contain strict architecture metadata for original/expanded
hidden size, heads, and layers. Resume or inference aborts when requested and
saved expansion layouts differ.

### Incremental ATE expansion

Use an existing ATE checkpoint as a model-only source for a larger architecture:

```bash
python train.py \
  --experiment-type ate \
  --incremental-ate-checkpoint runs/ate_h1_l1/checkpoints/best.pt \
  --run-dir runs/ate_h1_l2 \
  --new-attn-heads 0 \
  --new-ffn-layers 1 \
  --resume 0 \
  ...
```

In incremental mode the two expansion counts are deltas. The example preserves
the existing added head and adds a second depth stage.
The destination must be a different run directory with no checkpoints. Source
architecture and stage boundaries are reconstructed first, source tensors are
strict-loaded, and only then is the new stage wrapped around that exact source
in place. Prefix-copy widening is not used. A startup report verifies logits,
existing hidden states, and first-backward gradients. Optimizer, scheduler,
scaler, RNG, step, and epoch restart because the topology changed.

The incremental launcher also looks for `steps_per_epoch_cache.json` beside the
source run's `checkpoints` directory. It copies that cache into the destination
only when dataset identity, tokenizer, sequence length, micro-batch size,
gradient accumulation, worker count, data format, and system-prompt behavior all
match. Any mismatch is printed and falls back to normal step inference.

After the expanded run saves a checkpoint, normal exact resume uses total counts
and omits `--incremental-ate-checkpoint`:

```text
--new-attn-heads 1 --new-ffn-layers 2 --resume 1
```

Legacy `pythia_ate_v1` checkpoints remain directly loadable. To split their
control-token rows and record an explicit inferred legacy stage before future
width expansion:

```bash
python migrate_ate_checkpoint.py \
  --input runs/old/checkpoints/best.pt \
  --output runs/migrated/checkpoints/best.pt
```

Migration intentionally drops optimizer/scheduler/scaler/RNG state.

### Standalone step inference

Precompute epoch length without loading the model. Use the same run directory
and cardinality-related arguments as the subsequent training command:

```bash
python infer_steps.py \
  --text_data_path data/ultrachat_filtered/train.jsonl \
  --text_data_format chat_jsonl \
  --dataset-manifest data/ultrachat_filtered/manifest.json \
  --tokenizer-name EleutherAI/pythia-1.4b \
  --run-dir runs/ultrachat_pythia_1.4b_mod_plastic4 \
  --batch-size 10 \
  --grad-accum-steps 2 \
  --seq-len 1024 \
  --num-workers 0 \
  --epochs 7
```

With a validated UltraChat manifest this is a constant-time count operation.
Without `--dataset-manifest`, the tool uses Parquet row metadata where possible
or performs the same exact dataset scan as `train.py`. It writes
`steps_per_epoch_cache.json`; `train.py` reuses that file only when the dataset,
tokenizer, sequence length, batch size, accumulation, worker count, format, and
system-prompt setting all match.

## Direct-answer ShareGPT data

Convert the single-turn ShareGPT source while discarding the complete thinking
section. Only the text inside `<output>...</output>` becomes the supervised
direct answer; the output tags themselves are removed:

```bash
python convert_sharegpt_to_qa.py \
  --input data/chain_of_thought/chain_of_thought_sharegpt.json \
  --output data/chain_of_thought/chain_of_thought_qa.txt \
  --report data/chain_of_thought/conversion_report.json
```

Ready-to-train leakage-safe files are under `data/chain_of_thought/split`. Use
`--text_data_format qa_sft` for training.

After training, run teacher-forced and greedy evaluation with:

```bash
python evaluate.py \
  --run-dir runs/your_run \
  --eval-config configs/chain_of_thought_eval.json \
  --device cuda \
  --max-new-tokens 1024 \
  --max-examples 100
```

The regular assistant-content and first-answer metrics apply to these direct
answers. Reasoning/output delimiter metrics are not expected because those tags
are intentionally absent from the converted dataset.

## Native chat Parquet data

Hugging Face-style Parquet files with a `messages` column can be streamed
directly without conversion. Each row must contain valid system/user/assistant
role-content objects. Use the included files as follows:

```bash
python train.py \
  --experiment-type frozen_mod \
  --model-name EleutherAI/pythia-1.4b \
  --tokenizer-name EleutherAI/pythia-1.4b \
  --text_data_path parquet/train-00000-of-00001.parquet \
  --eval_text_data_path parquet/test-00000-of-00001.parquet \
  --text_data_format chat_parquet \
  --ignore_system_prompt 1 \
  --run_dir runs/pythia_parquet_mod \
  --seq_len 2048 \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --num_workers 2 \
  --epochs 5 \
  --emb-mod-dim 128 \
  --out-mod-dim 128 \
  --attn-mod-dim 128 \
  --ffn-mod-dim 256 \
  --attn-unique-mod-count 1 \
  --ffn-unique-mod-count 1 \
  --modifier-lr 3e-4 \
  --bf16
```

The format is also auto-detected from the `.parquet` suffix. With
`--ignore_system_prompt 1`, every system message is removed before tokenization,
so training and evaluation contain only `<USER>` and `<ASSISTANT>` turns. The
setting is saved in checkpoints and is also enforced by `chat.py`; any supplied
`--system_prompt` is ignored for those checkpoints.

To inspect the dataset as readable text and canonical JSONL shards without
changing the source Parquet files:

```bash
python export_parquet_readable.py \
  --input-dir parquet \
  --output-dir parquet/readable \
  --rows-per-shard 5000 \
  --preview-rows 3
```

Open `parquet/readable/train_preview.txt` or
`parquet/readable/test_preview.txt` for formatted examples. The manifest lists
every exported shard and its source row count.

To combine selected Parquet files into one human-readable TXT file:

```bash
python export_parquet_to_txt.py \
  --input-dir parquet \
  --pattern "*-00000-of-00002.parquet" \
  --output parquet/new_dataset_all_conversations.txt \
  --skip-invalid
```

`--skip-invalid` excludes empty or malformed conversations and records the
skipped count in the TXT footer. Training remains strict and rejects such rows.

## Export for chat

Resumable checkpoints include optimizer state. Export a smaller model-only copy
before interactive comparison:

```bash
python export_inference_checkpoint.py runs/bigdata_pythia_2.8b_full_ft/checkpoints/best.pt
python export_inference_checkpoint.py runs/bigdata_pythia_1.4b_token_mod_v1_1/checkpoints/best.pt
```

Use each generated `checkpoints/inference.pt` with `chat.py`. The trainer keeps
the lowest assistant-content validation-PPL checkpoint as `best.pt`; numbered
`step_*.pt` files provide periodic recovery points. It no longer writes the
duplicated `latest.pt` file. `--resume 1` selects the most recently written
resumable `best.pt` or numbered checkpoint.

If exact resume is not required, add `--no_optimizer` (or `--no-optimizer`) to
training. Then every `step_*.pt` and `best.pt` is already model-only and can
be loaded directly by `chat.py`; no export step is needed.

Run commands from this directory so its local trainer, model, data pipeline,
checkpointing, evaluation, chat, and dashboard modules are used.
