from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


@dataclass
class TrainConfig:
    variant: str
    model_name: str
    tokenizer_name: str
    text_data_path: str
    eval_text_data_path: str | None
    text_data_format: str
    run_dir: str
    device: str
    seed: int
    epochs: int
    max_steps: int
    steps_per_epoch: int | None
    batch_size: int
    grad_accum_steps: int
    seq_len: int
    mix: int
    shuffle_buffer: int
    num_workers: int
    eval_every: int
    eval_batches: int
    early_stop_patience: int
    early_stop_min_delta: float
    save_every: int
    log_every: int
    plot_every: int
    resume: bool
    checkpoint_path: str | None
    freeze_base: bool
    mod_dim: int
    attn_mod_dim: int | None
    ffn_mod_dim: int | None
    lr: float
    modifier_lr: float
    lora_lr: float | None
    weight_decay: float
    grad_clip: float
    warmup_steps: int
    min_lr_ratio: float
    mixed_precision: bool
    lora_r: int
    lora_alpha: float
    lora_dropout: float
    out_mod_dim: int | None = None
    dropout: float | None = None
    stop_eval_ppl: float = 0.0
    stop_eval_combined_ppl: float = 0.0
    experiment_type: str = "legacy"
    revision: str | None = None
    emb_mod_scale: float = 1.0
    out_mod_scale: float = 1.0
    attn_mod_scale: float = 1.0
    ffn_mod_scale: float = 1.0
    learnable_mod_scales: bool = False
    disable_emb_mod: bool = False
    disable_out_mod: bool = False
    disable_attn_mod: bool = False
    disable_ffn_mod: bool = False
    target_training_tokens: int = 0
    eval_every_tokens: int = 0
    save_every_tokens: int = 0
    warmup_ratio: float = 0.0
    gradient_checkpointing: bool = False
    bf16: bool = False
    fp16: bool = False
    optimizer_state_offload: bool = False
    no_optimizer: bool = False
    attn_unique_mod_count: int = 1
    ffn_unique_mod_count: int = 1
    ignore_system_prompt: bool = False
    plastic_last_n_layers: int = 0
    plastic_layer_norms: bool = False
    plastic_biases: bool = False
    plastic_attn_output: bool = False
    plastic_ffn_output: bool = False
    plastic_final_layer_norm: bool = False
    plastic_lm_head: bool = False
    plastic_lr: float = 1e-5
    plastic_layernorm_lr: float = 3e-5
    dataset_manifest: str | None = None
    strict_sample_order: bool = False
    new_attn_heads: int = 0
    new_ffn_layers: int = 0
    plasticity_mode: str = "quadratic"
    plasticity_base_lr: float = 1e-5
    plasticity_custom_scales: str = ""
    ate_confirm: str | None = None
    incremental_ate_checkpoint: str | None = None
    ate_output_init: str = "exact"
    ate_output_init_scale: float = 0.0
    allow_nonpreserving_expansion: bool = False
    previous_ate_stages_trainable: bool = True
    ate_current_stage_lr: float | None = None
    ate_previous_stage_lr: float | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GPT-2 or controlled Pythia full/MOD experiments.")
    parser.add_argument(
        "--experiment_type", "--experiment-type",
        choices=["legacy", "frozen_mod", "full_finetune", "ate"],
        default="legacy",
    )
    parser.add_argument(
        "--variant",
        choices=[
            "custom", "full", "emb_mod", "ffn_mod", "attn_mod", "lora",
            "emb_ffn_mod", "emb_attn_mod", "ffn_attn_mod", "emb_ffn_attn_mod",
            "emb_mod_lora", "ffn_mod_lora", "attn_mod_lora",
            "emb_ffn_mod_lora", "emb_attn_mod_lora", "ffn_attn_mod_lora",
            "emb_ffn_attn_mod_lora",
        ],
        default="custom",
        help="Select a controlled GPT-2 SFT variant; custom uses the explicit dimension flags.",
    )
    parser.add_argument("--model_name", "--model-name", type=str, default="gpt2")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--tokenizer_name", "--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--text_data_path", "--dataset-path", type=str, default=None)
    parser.add_argument("--eval_text_data_path", "--validation-dataset-path", type=str, default=None)
    parser.add_argument(
        "--dataset_manifest", "--dataset-manifest", type=str, default=None,
        help="Validate train/eval file and tokenizer hashes against a prepared dataset manifest.",
    )
    parser.add_argument(
        "--strict_sample_order", "--strict-sample-order", type=str2bool, default=False,
        help="Require shuffle_buffer=0 and num_workers=0 for identical cross-model sample order.",
    )
    parser.add_argument(
        "--text_data_format",
        choices=["auto", "plain", "qa_sft", "chat_jsonl", "chat_parquet"],
        default="auto",
    )
    parser.add_argument("--run_dir", "--run-dir", "--run-name", type=str, default="runs/gpt_mod")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--seq_len", "--max-seq-len", type=int, default=256)
    parser.add_argument("--mix", type=int, default=0)
    parser.add_argument(
        "--shuffle_buffer", "--shuffle-buffer",
        type=int,
        default=10_000,
        help="Bounded per-epoch shuffle buffer for SFT examples; 0 preserves file order.",
    )
    parser.add_argument(
        "--ignore_system_prompt", "--ignore-system-prompt",
        nargs="?",
        const=True,
        type=str2bool,
        default=False,
        help="Remove system messages entirely before chat SFT tokenization and generation.",
    )
    parser.add_argument("--num_workers", "--num-workers", type=int, default=0)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--plot_every", type=int, default=500)
    parser.add_argument("--resume", type=str2bool, default=False)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--freeze_base", type=str2bool, default=False)
    parser.add_argument(
        "--plastic_last_n_layers", "--plastic-last-n-layers", type=int, default=0,
        help="Unfreeze the final N complete Pythia transformer layers while keeping earlier layers frozen.",
    )
    parser.add_argument(
        "--plastic_layer_norms", "--plastic-layer-norms", type=str2bool, default=False,
        help="Unfreeze input/post-attention LayerNorm parameters in every transformer layer.",
    )
    parser.add_argument(
        "--plastic_biases", "--plastic-biases", type=str2bool, default=False,
        help="Unfreeze base-model bias parameters in all transformer layers.",
    )
    parser.add_argument(
        "--plastic_attn_output", "--plastic-attn-output", type=str2bool, default=False,
        help="Unfreeze every attention dense output projection.",
    )
    parser.add_argument(
        "--plastic_ffn_output", "--plastic-ffn-output", type=str2bool, default=False,
        help="Unfreeze every FFN dense_4h_to_h output projection.",
    )
    parser.add_argument(
        "--plastic_final_layer_norm", "--plastic-final-layer-norm", type=str2bool, default=False,
        help="Unfreeze the final GPT-NeoX LayerNorm.",
    )
    parser.add_argument(
        "--plastic_lm_head", "--plastic-lm-head", type=str2bool, default=False,
        help="Unfreeze the base LM output head; disabled by default.",
    )
    parser.add_argument("--plastic_lr", "--plastic-lr", type=float, default=1e-5)
    parser.add_argument(
        "--plastic_layernorm_lr", "--plastic-layernorm-lr", type=float, default=3e-5,
    )
    parser.add_argument(
        "--new_attn_heads", "--new-attn-heads", type=int, default=0,
        help="ATE: append this many fixed-head-dimension attention heads and hidden-width slices.",
    )
    parser.add_argument(
        "--new_ffn_layers", "--new-ffn-layers", type=int, default=0,
        help="ATE: append this many complete transformer blocks after the pretrained stack.",
    )
    parser.add_argument(
        "--plasticity_mode", "--plasticity-mode",
        choices=["off", "linear", "quadratic", "custom"], default="quadratic",
    )
    parser.add_argument("--plasticity_base_lr", "--plasticity-base-lr", type=float, default=1e-5)
    parser.add_argument(
        "--plasticity_custom_scales", "--plasticity-custom-scales", default="",
        help="ATE custom mode: comma-separated multiplier for every original transformer layer.",
    )
    parser.add_argument(
        "--ate_confirm", "--ate-confirm", default=None,
        help="Non-interactive ATE confirmation. Only the exact value YES is accepted.",
    )
    parser.add_argument(
        "--incremental_ate_checkpoint", "--incremental-ate-checkpoint", default=None,
        help=(
            "Expand an existing ATE checkpoint into a new run. In this mode --new-attn-heads and "
            "--new-ffn-layers are additional counts, and optimizer/scheduler state restarts."
        ),
    )
    parser.add_argument(
        "--ate_output_init", "--ate-output-init",
        choices=["exact", "small"], default="exact",
        help="Initialize newest-stage output paths to exact zero or a small random value.",
    )
    parser.add_argument(
        "--ate_output_init_scale", "--ate-output-init-scale", type=float, default=0.0,
        help="Standard deviation for --ate-output-init small (for example 1e-3).",
    )
    parser.add_argument(
        "--allow_nonpreserving_expansion", "--allow-nonpreserving-expansion",
        action="store_true",
        help="Continue after an incremental ATE preservation failure (unsafe experiment only).",
    )
    parser.add_argument(
        "--previous_ate_stages_trainable", "--previous-ate-stages-trainable",
        type=str2bool, default=True,
        help="Keep checkpoint-resident ATE stages trainable after adding a new stage.",
    )
    parser.add_argument(
        "--ate_current_stage_lr", "--ate-current-stage-lr", type=float, default=None,
    )
    parser.add_argument(
        "--ate_previous_stage_lr", "--ate-previous-stage-lr", type=float, default=None,
    )
    parser.add_argument("--mod_dim", "--emb-mod-dim", type=int, default=32)
    parser.add_argument("--out_mod_dim", "--out-mod-dim", type=int, default=None)
    parser.add_argument("--attn_mod_dim", "--attn-mod-dim", type=int, default=None)
    parser.add_argument("--ffn_mod_dim", "--ffn-mod-dim", type=int, default=None)
    parser.add_argument(
        "--attn_unique_mod_count", "--attn-unique-mod-count", type=int, default=1,
        help="Number of contiguous layer groups with distinct Attention MOD token tables.",
    )
    parser.add_argument(
        "--ffn_unique_mod_count", "--ffn-unique-mod-count", type=int, default=1,
        help="Number of contiguous layer groups with distinct FFN MOD token tables.",
    )
    parser.add_argument("--emb_mod_scale", "--emb-mod-scale", type=float, default=1.0)
    parser.add_argument("--out_mod_scale", "--out-mod-scale", type=float, default=1.0)
    parser.add_argument("--attn_mod_scale", "--attn-mod-scale", type=float, default=1.0)
    parser.add_argument("--ffn_mod_scale", "--ffn-mod-scale", type=float, default=1.0)
    parser.add_argument("--learnable_mod_scales", "--learnable-mod-scales", action="store_true")
    parser.add_argument("--disable_emb_mod", "--disable-emb-mod", action="store_true")
    parser.add_argument("--disable_out_mod", "--disable-out-mod", action="store_true")
    parser.add_argument("--disable_attn_mod", "--disable-attn-mod", action="store_true")
    parser.add_argument("--disable_ffn_mod", "--disable-ffn-mod", action="store_true")
    parser.add_argument("--lr", "--learning-rate", type=float, default=3e-4)
    parser.add_argument("--modifier_lr", "--modifier-lr", type=float, default=1e-3)
    parser.add_argument("--lora_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--warmup_ratio", "--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.03)
    parser.add_argument("--mixed_precision", type=str2bool, default=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", "--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--optimizer_state_offload", "--optimizer-state-offload",
        action="store_true",
        help="Explicitly keep AdamW state on CPU between CUDA optimizer steps (slow, lower VRAM).",
    )
    parser.add_argument(
        "--no_optimizer", "--no-optimizer",
        action="store_true",
        help="Save model-only checkpoints without optimizer/scheduler/scaler/RNG state; exact resume is unavailable.",
    )
    parser.add_argument("--target_training_tokens", "--target-training-tokens", type=int, default=0)
    parser.add_argument("--eval_every_tokens", "--eval-every-tokens", type=int, default=0)
    parser.add_argument("--save_every_tokens", "--save-every-tokens", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Override every GPT-2/LoRA dropout probability; use 0 for deterministic overfit tests.",
    )
    parser.add_argument(
        "--stop_eval_ppl",
        type=float,
        default=0.0,
        help="Stop after an evaluation when assistant-content PPL is at or below this value; 0 disables.",
    )
    parser.add_argument(
        "--stop_eval_combined_ppl",
        type=float,
        default=0.0,
        help="Stop when combined assistant-content/EOS eval PPL reaches this value; 0 disables.",
    )
    return parser


def parse_args() -> TrainConfig:
    parser = build_arg_parser()
    values = vars(parser.parse_args())
    experiment_type = values["experiment_type"]
    if values["bf16"] and values["fp16"]:
        parser.error("--bf16 and --fp16 are mutually exclusive")
    if values["attn_unique_mod_count"] < 1 or values["ffn_unique_mod_count"] < 1:
        parser.error("Attention and FFN unique MOD counts must be at least 1")
    if values["plastic_last_n_layers"] < 0:
        parser.error("--plastic-last-n-layers must be non-negative")
    if values["new_attn_heads"] < 0 or values["new_ffn_layers"] < 0:
        parser.error("ATE expansion counts must be non-negative")
    if values["ate_output_init"] == "exact" and values["ate_output_init_scale"] != 0.0:
        parser.error("--ate-output-init exact requires --ate-output-init-scale 0.0")
    if values["ate_output_init"] == "small" and values["ate_output_init_scale"] <= 0.0:
        parser.error("--ate-output-init small requires --ate-output-init-scale > 0")
    for name in ("ate_current_stage_lr", "ate_previous_stage_lr"):
        if values[name] is not None and values[name] <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if experiment_type != "ate" and (values["new_attn_heads"] or values["new_ffn_layers"]):
        parser.error("--new-attn-heads and --new-ffn-layers require --experiment-type ate")
    if values["incremental_ate_checkpoint"] and experiment_type != "ate":
        parser.error("--incremental-ate-checkpoint requires --experiment-type ate")
    if values["incremental_ate_checkpoint"] and values["resume"]:
        parser.error("Incremental ATE starts a new optimization phase; use --resume 0")
    if values["incremental_ate_checkpoint"] and values["checkpoint_path"]:
        parser.error("Use only --incremental-ate-checkpoint, not --checkpoint_path")
    if values["incremental_ate_checkpoint"] and not (
        values["new_attn_heads"] or values["new_ffn_layers"]
    ):
        parser.error("Incremental ATE requires at least one newly requested head or layer")
    plasticity_enabled = values["plastic_last_n_layers"] > 0 or any(
        values[name]
        for name in (
            "plastic_layer_norms", "plastic_biases", "plastic_attn_output",
            "plastic_ffn_output", "plastic_final_layer_norm", "plastic_lm_head",
        )
    )
    if plasticity_enabled and experiment_type != "frozen_mod":
        parser.error("Controlled base plasticity requires --experiment-type frozen_mod")
    if plasticity_enabled and (values["plastic_lr"] <= 0 or values["plastic_layernorm_lr"] <= 0):
        parser.error("Plasticity learning rates must be positive when plasticity is enabled")
    if values["dataset_manifest"] and not values["strict_sample_order"]:
        parser.error("--dataset-manifest requires --strict-sample-order 1")
    if values["strict_sample_order"] and (values["shuffle_buffer"] != 0 or values["num_workers"] != 0):
        parser.error("--strict-sample-order 1 requires --shuffle-buffer 0 and --num-workers 0")
    if not 0.0 <= values["warmup_ratio"] < 1.0:
        parser.error("--warmup-ratio must be in [0, 1)")
    if experiment_type == "frozen_mod":
        modifier_lr_explicit = any(
            argument == "--modifier_lr" or argument == "--modifier-lr"
            or argument.startswith("--modifier_lr=") or argument.startswith("--modifier-lr=")
            for argument in sys.argv[1:]
        )
        if not modifier_lr_explicit:
            values["modifier_lr"] = values["lr"]
        values.update(variant="custom", freeze_base=True, lora_r=0, lr=0.0)
        if values["out_mod_dim"] is None:
            values["out_mod_dim"] = values["mod_dim"]
        if values["attn_mod_dim"] is None:
            values["attn_mod_dim"] = 32
        if values["ffn_mod_dim"] is None:
            values["ffn_mod_dim"] = 64
    elif experiment_type == "full_finetune":
        values.update(
            variant="full", freeze_base=False, mod_dim=0, out_mod_dim=0, attn_mod_dim=0,
            ffn_mod_dim=0, lora_r=0, attn_unique_mod_count=1, ffn_unique_mod_count=1,
        )
    elif experiment_type == "ate":
        if values["lr"] <= 0:
            parser.error("ATE newly added parameters require a positive --lr")
        if values["plasticity_mode"] != "off" and values["plasticity_base_lr"] <= 0:
            parser.error("ATE plasticity requires a positive --plasticity-base-lr")
        if values["plasticity_mode"] == "custom" and not values["plasticity_custom_scales"].strip():
            parser.error("ATE custom plasticity requires --plasticity-custom-scales")
        values.update(
            variant="custom", freeze_base=False, mod_dim=0, out_mod_dim=0,
            attn_mod_dim=0, ffn_mod_dim=0, lora_r=0,
            attn_unique_mod_count=1, ffn_unique_mod_count=1,
        )
    variant = values["variant"]
    if variant != "custom":
        if variant == "full":
            values.update(
                freeze_base=False, mod_dim=0, out_mod_dim=0,
                attn_mod_dim=0, ffn_mod_dim=0, lora_r=0,
                attn_unique_mod_count=1, ffn_unique_mod_count=1,
            )
        else:
            values["freeze_base"] = True
            values["mod_dim"] = values["mod_dim"] if "emb" in variant else 0
            values["attn_mod_dim"] = (values["attn_mod_dim"] or values["mod_dim"] or 64) if "attn" in variant else 0
            values["ffn_mod_dim"] = (values["ffn_mod_dim"] or values["mod_dim"] or 64) if "ffn" in variant else 0
            values["lora_r"] = values["lora_r"] if "lora" in variant else 0
    return TrainConfig(**values)


def default_config_dict() -> dict[str, Any]:
    parser = build_arg_parser()
    return vars(parser.parse_args([]))


def config_from_dict(data: dict[str, Any]) -> TrainConfig:
    merged = default_config_dict()
    merged.update(data)
    return TrainConfig(**merged)
