from __future__ import annotations

import math
import json
import random
import re
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from checkpointing import (
    CHAT_FORMAT_VERSION,
    checkpoint_paths,
    load_checkpoint,
    newest_resume_checkpoint,
    save_checkpoint,
    validate_tokenizer_metadata,
)
from config import TrainConfig, parse_args
from dashboard import build_dashboard_metrics, clear_live_dashboard, render_dashboard
from data import (
    CHAT_DATA_FORMATS,
    build_eval_loader,
    build_train_loader,
    infer_steps_per_epoch,
    resolve_text_data_format,
    save_steps_per_epoch_cache,
    steps_per_epoch_cache_key,
)
from dataset_lock import validate_locked_dataset
from logger import MetricLogger
from model import GPTModModel, build_tokenizer


@dataclass
class _CategoryTotals:
    count: int = 0
    nll_sum: float = 0.0
    top1_correct: int = 0
    top5_correct: int = 0
    rank_sum: float = 0.0
    probability_sum: float = 0.0

    def add(self, nll, correct, top5, ranks) -> None:
        if nll.numel() == 0:
            return
        self.count += int(nll.numel())
        self.nll_sum += float(nll.double().sum().item())
        self.top1_correct += int(correct.sum().item())
        self.top5_correct += int(top5.sum().item())
        self.rank_sum += float(ranks.double().sum().item())
        self.probability_sum += float(torch.exp(-nll.double()).sum().item())

    def metrics(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0, "nll_sum": 0.0, "nll": float("nan"), "ppl": float("nan"),
                "top1": float("nan"), "top5": float("nan"), "rank": float("nan"),
                "prob": float("nan"),
            }
        mean_nll = self.nll_sum / self.count
        return {
            "count": self.count,
            "nll_sum": self.nll_sum,
            "nll": mean_nll,
            "ppl": math.exp(min(mean_nll, 20.0)),
            "top1": self.top1_correct / self.count,
            "top5": self.top5_correct / self.count,
            "rank": self.rank_sum / self.count,
            "prob": self.probability_sum / self.count,
        }


@dataclass
class EvalTotals:
    reasoning_open_ids: tuple[tuple[int, ...], ...] = ()
    reasoning_close_ids: tuple[tuple[int, ...], ...] = ()
    final_open_ids: tuple[tuple[int, ...], ...] = ()
    final_close_ids: tuple[tuple[int, ...], ...] = ()
    categories: dict[str, _CategoryTotals] = field(
        default_factory=lambda: {
            name: _CategoryTotals()
            for name in (
                "combined", "assistant_content", "first_answer", "remaining_content",
                "reasoning", "final_answer", "final_answer_first", "eos",
            )
        }
    )
    first_eos_probability_sum: float = 0.0
    first_eos_top1: int = 0
    first_count: int = 0

    @staticmethod
    def _find_any(sequence: list[int], variants: tuple[tuple[int, ...], ...], start: int = 0):
        for position in range(max(start, 0), len(sequence)):
            for variant in variants:
                end = position + len(variant)
                if variant and tuple(sequence[position:end]) == variant:
                    return position, end
        return None

    def _reasoning_final_masks(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reasoning = torch.zeros_like(input_ids, dtype=torch.bool, device="cpu")
        final = torch.zeros_like(input_ids, dtype=torch.bool, device="cpu")
        if not all((self.reasoning_open_ids, self.reasoning_close_ids, self.final_open_ids, self.final_close_ids)):
            return reasoning.to(input_ids.device), final.to(input_ids.device)
        for row, sequence in enumerate(input_ids.detach().cpu().tolist()):
            reasoning_open = self._find_any(sequence, self.reasoning_open_ids)
            if reasoning_open is None:
                continue
            reasoning_close = self._find_any(sequence, self.reasoning_close_ids, reasoning_open[1])
            if reasoning_close is None:
                continue
            final_open = self._find_any(sequence, self.final_open_ids, reasoning_close[1])
            if final_open is None:
                continue
            final_close = self._find_any(sequence, self.final_close_ids, final_open[1])
            if final_close is None:
                continue
            reasoning[row, reasoning_open[1] : reasoning_close[0]] = True
            final[row, final_open[1] : final_close[0]] = True
        return reasoning.to(input_ids.device), final.to(input_ids.device)

    def update(self, logits, input_ids, labels, assistant_id, eos_id) -> None:
        del assistant_id
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        flat_nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)
        predictions = shift_logits.argmax(dim=-1)
        correct = predictions.eq(shift_labels)
        safe_labels = shift_labels.clamp_min(0)
        gold_scores = shift_logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        ranks = shift_logits.gt(gold_scores.unsqueeze(-1)).sum(dim=-1).add(1)
        top5 = shift_logits.topk(min(5, shift_logits.size(-1)), dim=-1).indices.eq(
            safe_labels.unsqueeze(-1)
        ).any(dim=-1)
        supervised = shift_labels.ne(-100)
        eos = shift_labels.eq(eos_id)
        content = supervised & ~eos
        previous_supervised = torch.cat(
            [torch.zeros_like(supervised[:, :1]), supervised[:, :-1]], dim=1
        )
        first = content & ~previous_supervised
        reasoning_positions, final_positions = self._reasoning_final_masks(input_ids)
        reasoning = reasoning_positions[:, 1:] & supervised
        final = final_positions[:, 1:] & supervised
        previous_final = torch.cat([torch.zeros_like(final[:, :1]), final[:, :-1]], dim=1)
        final_first = final & ~previous_final
        if first.any():
            first_logits = shift_logits[first]
            self.first_eos_probability_sum += float(
                first_logits.softmax(dim=-1)[:, eos_id].double().sum().item()
            )
            self.first_eos_top1 += int(first_logits.argmax(dim=-1).eq(eos_id).sum().item())
            self.first_count += int(first.sum().item())
        masks = {
            "combined": supervised,
            "assistant_content": content,
            "first_answer": first,
            "remaining_content": content & ~first,
            "reasoning": reasoning,
            "final_answer": final,
            "final_answer_first": final_first,
            "eos": eos,
        }
        for name, mask in masks.items():
            self.categories[name].add(
                flat_nll[mask].detach().cpu(), correct[mask].detach().cpu(),
                top5[mask].detach().cpu(), ranks[mask].detach().cpu(),
            )

    def metrics(self) -> dict[str, dict[str, float | int]]:
        result = {name: totals.metrics() for name, totals in self.categories.items()}
        result["first_answer"]["eos_prob"] = (
            self.first_eos_probability_sum / self.first_count if self.first_count else float("nan")
        )
        result["first_answer"]["eos_top1"] = (
            self.first_eos_top1 / self.first_count if self.first_count else float("nan")
        )
        return result


class PlasticityDriftTracker:
    """Measure selected base-weight movement relative to pretrained initialization."""

    def __init__(self, model) -> None:
        plastic_ids = {
            id(parameter)
            for parameter in (
                model.plastic_base_parameters() if hasattr(model, "plastic_base_parameters") else ()
            )
        }
        self.parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if id(parameter) in plastic_ids
        }
        self.reference = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.parameters.items()
        }

    @staticmethod
    def _group_name(name: str) -> str:
        match = re.search(r"\.layers\.(\d+)\.", name)
        if match:
            return f"layer_{int(match.group(1)):02d}"
        if "final_layer_norm" in name:
            return "final_layer_norm"
        if "embed_out" in name or "lm_head" in name:
            return "lm_head"
        return "other"

    def measure(self) -> dict[str, object]:
        totals: dict[str, list[float]] = {"all": [0.0, 0.0]}
        with torch.no_grad():
            for name, parameter in self.parameters.items():
                current = parameter.detach().cpu().float()
                reference = self.reference[name].float()
                group = self._group_name(name)
                totals.setdefault(group, [0.0, 0.0])
                delta_sq = float(torch.sum((current - reference) ** 2).item())
                reference_sq = float(torch.sum(reference ** 2).item())
                totals["all"][0] += delta_sq
                totals["all"][1] += reference_sq
                totals[group][0] += delta_sq
                totals[group][1] += reference_sq
        relative = {
            name: math.sqrt(delta_sq) / max(math.sqrt(reference_sq), 1e-12)
            for name, (delta_sq, reference_sq) in totals.items()
        }
        return {
            "relative_drift": relative.get("all", float("nan")),
            "by_group": {name: value for name, value in relative.items() if name != "all"},
            "tensor_count": len(self.parameters),
            "parameter_count": sum(parameter.numel() for parameter in self.parameters.values()),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def prepare_incremental_ate_config(config: TrainConfig) -> None:
    source_value = getattr(config, "incremental_ate_checkpoint", None)
    if not source_value:
        return
    source_path = Path(source_value).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    target_run = Path(config.run_dir).expanduser().resolve()
    source_run = source_path.parent.parent if source_path.parent.name == "checkpoints" else source_path.parent
    if target_run == source_run:
        raise RuntimeError(
            "Incremental ATE requires a new --run-dir; do not mix different architectures in one checkpoint folder."
        )
    existing_targets = list((target_run / "checkpoints").glob("*.pt"))
    if existing_targets:
        raise RuntimeError(
            f"Incremental ATE destination already contains checkpoints: {existing_targets[:3]}"
        )
    try:
        payload = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
    metadata = payload.get("architecture_metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict) or metadata.get("architecture_version") not in {
        "pythia_ate_v1", "pythia_ate_v2_staged"
    }:
        raise RuntimeError(f"Incremental source is not a compatible ATE checkpoint: {metadata}")
    if payload.get("model_state_type", "full") != "full":
        raise RuntimeError("Incremental ATE requires a full ATE model checkpoint.")
    for key in ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "rng_state"):
        payload.pop(key, None)
    requested_heads = int(config.new_attn_heads)
    requested_layers = int(config.new_ffn_layers)
    config.incremental_ate_checkpoint = None
    config._ate_incremental_source_path = str(source_path)
    config._ate_incremental_payload = payload
    config._ate_incremental_requested_heads = requested_heads
    config._ate_incremental_requested_layers = requested_layers
    print(
        "incremental ATE source prepared | "
        f"source_heads={metadata['new_attention_heads']} + requested={requested_heads} "
        f"-> total={int(metadata['new_attention_heads']) + requested_heads} | "
        f"source_layers={metadata['new_transformer_layers']} + requested={requested_layers} "
        f"-> total={int(metadata['new_transformer_layers']) + requested_layers}",
        flush=True,
    )


def prepare_ate_resume_architecture(config: TrainConfig) -> None:
    """Peek only at metadata so the exact staged graph exists before strict load."""
    if getattr(config, "_ate_incremental_payload", None) is not None:
        return
    checkpoint_path = Path(config.checkpoint_path).expanduser() if config.checkpoint_path else None
    if checkpoint_path is None and config.resume:
        checkpoint_path = newest_resume_checkpoint(config.run_dir)
    if checkpoint_path is None or not checkpoint_path.exists():
        return
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = payload.get("architecture_metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        raise RuntimeError(f"ATE checkpoint {checkpoint_path} has no architecture metadata.")
    saved_heads = int(metadata["new_attention_heads"])
    saved_layers = int(metadata["new_transformer_layers"])
    if config.new_attn_heads not in {0, saved_heads} or config.new_ffn_layers not in {0, saved_layers}:
        raise RuntimeError(
            "ATE resume CLI counts disagree with checkpoint metadata: "
            f"cli=({config.new_attn_heads} heads, {config.new_ffn_layers} layers) "
            f"saved=({saved_heads} heads, {saved_layers} layers)."
        )
    config.new_attn_heads = saved_heads
    config.new_ffn_layers = saved_layers
    config._ate_resume_metadata = metadata
    del payload


def apply_incremental_ate_checkpoint(config: TrainConfig, model, tokenizer) -> None:
    payload = getattr(config, "_ate_incremental_payload", None)
    if payload is None:
        return
    # The source is strictly loaded and expanded in build_model_and_tokenizer.
    result = getattr(config, "_ate_incremental_load_result", {"exact_tensors": 0, "expanded_tensors": 0})
    print(
        f"incremental ATE weights loaded | source={config._ate_incremental_source_path} | "
        f"exact_tensors={result['exact_tensors']} expanded_tensors={result['expanded_tensors']} | "
        "optimizer=scheduler=scaler reset",
        flush=True,
    )
    del config._ate_incremental_payload


def reuse_incremental_steps_cache(config: TrainConfig, tokenizer) -> bool:
    source_value = getattr(config, "_ate_incremental_source_path", None)
    if not source_value:
        return False
    source_checkpoint = Path(source_value)
    candidates = [
        source_checkpoint.parent.parent / "steps_per_epoch_cache.json",
        source_checkpoint.parent / "steps_per_epoch_cache.json",
    ]
    source_cache = next((path for path in candidates if path.exists()), None)
    if source_cache is None:
        print(
            "incremental ATE steps cache not found in the source run; normal inference will be used.",
            flush=True,
        )
        return False
    try:
        payload = json.loads(source_cache.read_text(encoding="utf-8"))
        saved_key = payload["cache_key"]
        saved_steps = int(payload["steps_per_epoch"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"incremental ATE steps cache is invalid ({exc}); normal inference will be used.", flush=True)
        return False
    expected_key = steps_per_epoch_cache_key(config, tokenizer)
    if saved_key != expected_key or saved_steps <= 0:
        mismatches = {
            key: (saved_key.get(key), expected_key.get(key))
            for key in sorted(set(saved_key) | set(expected_key))
            if saved_key.get(key) != expected_key.get(key)
        }
        print(
            f"incremental ATE steps cache does not match this run; normal inference will be used. "
            f"mismatches={mismatches}",
            flush=True,
        )
        return False
    target_cache = save_steps_per_epoch_cache(config, tokenizer, saved_steps)
    print(
        f"incremental ATE steps cache reused | source={source_cache} | "
        f"steps_per_epoch={saved_steps} | target={target_cache}",
        flush=True,
    )
    return True


def build_model_and_tokenizer(config: TrainConfig, device: torch.device):
    if config.experiment_type != "legacy" and config.tokenizer_name == "gpt2":
        config.tokenizer_name = config.model_name
    tokenizer = build_tokenizer(config.tokenizer_name)
    if config.experiment_type in {"frozen_mod", "full_finetune", "ate"}:
        try:
            from pythia_model import PythiaModModel
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Pythia training was moved to the sibling Pythia_run folder. "
                "Run this command from D:\\inverted_model\\Pythia_run."
            ) from error
        dtype = None
        if device.type == "cuda" and config.bf16:
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("--bf16 was requested but this CUDA device does not support bfloat16.")
            dtype = torch.bfloat16
        elif device.type == "cuda" and config.fp16:
            dtype = torch.float16
        common = {
            "model_name": config.model_name,
            "revision": config.revision,
            "vocab_size": len(tokenizer),
            "gradient_checkpointing": config.gradient_checkpointing,
            "torch_dtype": dtype,
            "control_token_ids": tuple(
                tokenizer.convert_tokens_to_ids(token)
                for token in ("<USER>", "<ASSISTANT>", "<SYSTEM>")
            ),
        }
        if config.experiment_type == "ate":
            from ate_model import PythiaATEModel
            incremental_payload = getattr(config, "_ate_incremental_payload", None)
            architecture_metadata = getattr(config, "_ate_resume_metadata", None)
            if incremental_payload is not None:
                architecture_metadata = incremental_payload.get("architecture_metadata")
                # Preservation is an FP32 architectural invariant. Running this
                # probe in BF16/FP16 can change attention-kernel rounding when
                # the head count changes even though every new output path is
                # exactly zero.
                common["torch_dtype"] = torch.float32
                print(
                    "incremental ATE preservation build uses FP32; "
                    f"verified model will be cast to {dtype} before training",
                    flush=True,
                )
            model = PythiaATEModel.from_pretrained(
                **common,
                new_attention_heads=(0 if architecture_metadata else config.new_attn_heads),
                new_transformer_layers=(0 if architecture_metadata else config.new_ffn_layers),
                architecture_metadata=architecture_metadata,
                ate_output_init=config.ate_output_init,
                ate_output_init_scale=config.ate_output_init_scale,
            )
            # Keep incremental verification on CPU FP32. GPU attention kernels
            # may choose a different reduction schedule after the head count
            # changes, which measures kernel rounding rather than architectural
            # preservation.
            if incremental_payload is None:
                model.to(device)
            if incremental_payload is not None:
                model.validate_incremental_source_metadata(architecture_metadata)
                validate_tokenizer_metadata(incremental_payload, tokenizer)
                incompatible = model.load_state_dict(incremental_payload["model_state_dict"], strict=True)
                if incompatible.missing_keys or incompatible.unexpected_keys:
                    raise RuntimeError(f"Strict incremental source load failed: {incompatible}")
                model.eval()
                probe_ids = torch.tensor(
                    [[1, 2, 3, 4]], dtype=torch.long, device="cpu"
                ).clamp_max(max(model.original_vocab_size - 1, 0))
                with torch.no_grad():
                    source_output = model(input_ids=probe_ids, output_hidden_states=True)
                    source_logits = source_output["logits"].float().clone()
                    source_hidden = tuple(item.float().clone() for item in source_output["hidden_states"])
                if architecture_metadata.get("architecture_version") == "pythia_ate_v1":
                    model.migrate_legacy_control_rows()
                model.add_expansion_stage(
                    added_attention_heads=config._ate_incremental_requested_heads,
                    added_transformer_layers=config._ate_incremental_requested_layers,
                    source_checkpoint=config._ate_incremental_source_path,
                    output_init=config.ate_output_init,
                    output_init_scale=config.ate_output_init_scale,
                )
                preservation = model.preservation_probe(source_logits, probe_ids)
                with torch.no_grad():
                    final_hidden = model(input_ids=probe_ids, output_hidden_states=True)["hidden_states"]
                hidden_differences = []
                source_width = source_hidden[0].shape[-1]
                for index, expected_hidden in enumerate(source_hidden):
                    target_index = -1 if index == len(source_hidden) - 1 else index
                    actual_hidden = final_hidden[target_index][..., :source_width].float()
                    hidden_differences.append(float((actual_hidden - expected_hidden).abs().max().item()))
                preservation["existing_hidden_state_max_differences"] = hidden_differences
                preservation["hidden_states_passed"] = all(value < 1e-6 for value in hidden_differences)
                preservation["passed"] = bool(
                    preservation["passed"] and preservation["hidden_states_passed"]
                )
                preservation["verification_dtype"] = "float32"
                preservation["verification_device"] = "cpu"
                config._ate_preservation_report = preservation
                config._ate_incremental_load_result = {
                    "exact_tensors": len(incremental_payload["model_state_dict"]),
                    "expanded_tensors": 0,
                }
                if not preservation["passed"] and not config.allow_nonpreserving_expansion:
                    raise RuntimeError(
                        "Incremental ATE preservation failed: "
                        f"max={preservation['max_abs_logit_difference']:.3e} "
                        f"mean={preservation['mean_abs_logit_difference']:.3e}. "
                        "Use --allow-nonpreserving-expansion only for an intentional unsafe experiment."
                    )
                config.new_attn_heads = model.new_attention_heads
                config.new_ffn_layers = model.new_transformer_layers
                if dtype is None:
                    model.to(device=device)
                else:
                    model.to(device=device, dtype=dtype)
                print(
                    f"incremental ATE model moved to {device} with training dtype "
                    f"{dtype or next(model.parameters()).dtype}",
                    flush=True,
                )
            custom_scales = tuple(
                float(item.strip())
                for item in config.plasticity_custom_scales.split(",")
                if item.strip()
            )
            model.configure_ate_plasticity(
                config.plasticity_mode,
                custom_scales,
                previous_stages_trainable=config.previous_ate_stages_trainable,
            )
        else:
            model = PythiaModModel.from_pretrained(
                **common,
                emb_mod_dim=config.mod_dim,
                out_mod_dim=config.out_mod_dim or 0,
                attn_mod_dim=config.attn_mod_dim or 0,
                ffn_mod_dim=config.ffn_mod_dim or 0,
                attn_unique_mod_count=config.attn_unique_mod_count,
                ffn_unique_mod_count=config.ffn_unique_mod_count,
                emb_mod_scale=config.emb_mod_scale,
                out_mod_scale=config.out_mod_scale,
                attn_mod_scale=config.attn_mod_scale,
                ffn_mod_scale=config.ffn_mod_scale,
                learnable_mod_scales=config.learnable_mod_scales,
            )
        if config.experiment_type != "ate":
            model.set_mod_ablation(
                disable_emb=config.disable_emb_mod,
                disable_out=config.disable_out_mod,
                disable_attn=config.disable_attn_mod,
                disable_ffn=config.disable_ffn_mod,
            )
        if config.experiment_type == "frozen_mod":
            model.configure_base_plasticity(
                last_n_layers=config.plastic_last_n_layers,
                layer_norms=config.plastic_layer_norms,
                biases=config.plastic_biases,
                attn_output=config.plastic_attn_output,
                ffn_output=config.plastic_ffn_output,
                final_layer_norm=config.plastic_final_layer_norm,
                lm_head=config.plastic_lm_head,
            )
        elif config.experiment_type == "full_finetune":
            model.enable_full_finetuning()
        if config.experiment_type != "ate":
            for summary_line in model.mod_group_summary_lines():
                print(summary_line, flush=True)
    else:
        model = GPTModModel.from_pretrained(
            model_name=config.model_name,
            mod_dim=config.mod_dim,
            attn_mod_dim=config.attn_mod_dim,
            ffn_mod_dim=config.ffn_mod_dim,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            vocab_size=len(tokenizer),
        )
    context_limit = int(getattr(model.config, "n_positions", getattr(model.config, "max_position_embeddings", 0)))
    if context_limit and config.seq_len > context_limit:
        raise ValueError(
            f"--seq_len {config.seq_len} exceeds {config.model_name} context limit {context_limit}."
        )
    if config.dropout is not None:
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("--dropout must be in [0, 1).")
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = float(config.dropout)
        model.config.resid_pdrop = float(config.dropout)
        model.config.embd_pdrop = float(config.dropout)
        model.config.attn_pdrop = float(config.dropout)
    if config.freeze_base and config.experiment_type == "legacy":
        model.freeze_base_model()
    return model.to(device), tokenizer


def build_optimizer(config: TrainConfig, model) -> AdamW:
    if getattr(config, "experiment_type", "legacy") == "ate":
        named = list(model.named_parameters())
        names_by_id = {id(parameter): name for name, parameter in named}
        groups: list[dict] = []
        included: set[int] = set()
        for group_name, learning_rate, parameters in model.ate_optimizer_specs(
            config.lr,
            config.plasticity_base_lr,
            config.ate_current_stage_lr,
            config.ate_previous_stage_lr,
        ):
            for decay in (True, False):
                selected = []
                for parameter in parameters:
                    name = names_by_id[id(parameter)]
                    lowered = name.lower()
                    no_decay = (
                        parameter.ndim < 2 or name.endswith(".bias")
                        or "layernorm" in lowered or "layer_norm" in lowered
                    )
                    if decay == no_decay:
                        continue
                    if id(parameter) in included:
                        raise RuntimeError(f"ATE optimizer parameter appears more than once: {name}")
                    included.add(id(parameter))
                    selected.append(parameter)
                if selected:
                    groups.append(
                        {
                            "name": f"{group_name}_{'decay' if decay else 'no_decay'}",
                            "params": selected,
                            "lr": learning_rate,
                            "weight_decay": config.weight_decay if decay else 0.0,
                        }
                    )
        expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        if included != expected:
            missing = [name for name, parameter in named if parameter.requires_grad and id(parameter) not in included]
            raise RuntimeError(f"ATE trainable parameters missing from optimizer: {missing}")
        if not groups:
            raise ValueError("ATE selected no trainable parameters; add capacity or enable plasticity.")
        return AdamW(groups)

    lora_lr = config.lora_lr if config.lora_lr is not None else config.lr
    plastic_ids = {
        id(param) for param in (
            model.plastic_base_parameters() if hasattr(model, "plastic_base_parameters") else ()
        )
    }
    plastic_norm_ids = {
        id(param) for param in (
            model.plastic_norm_parameters() if hasattr(model, "plastic_norm_parameters") else ()
        )
    }
    category_ids = {
        "base": {id(param) for param in model.base_parameters()} - plastic_ids,
        "plastic": plastic_ids - plastic_norm_ids,
        "plastic_norm": plastic_norm_ids,
        "modifier": {id(param) for param in model.modifier_parameters()},
        "lora": {id(param) for param in model.lora_parameters()},
    }
    learning_rates = {
        "base": config.lr,
        "plastic": config.plastic_lr,
        "plastic_norm": config.plastic_layernorm_lr,
        "modifier": config.modifier_lr,
        "lora": lora_lr,
    }
    named = list(model.named_parameters())
    groups: list[dict] = []
    included: set[int] = set()
    for category in ("base", "plastic", "plastic_norm", "modifier", "lora"):
        for decay in (True, False):
            selected = []
            for name, param in named:
                if not param.requires_grad or id(param) not in category_ids[category]:
                    continue
                lowered = name.lower()
                no_decay = (
                    param.ndim < 2 or name.endswith(".bias") or ".ln_" in name
                    or "layernorm" in lowered or name.endswith(".ln_f.weight")
                )
                if decay == no_decay:
                    continue
                if id(param) in included:
                    raise RuntimeError(f"Optimizer parameter appears more than once: {name}")
                included.add(id(param))
                selected.append(param)
            if selected:
                groups.append(
                    {
                        "name": f"{category}_{'decay' if decay else 'no_decay'}",
                        "params": selected,
                        "lr": learning_rates[category],
                        "weight_decay": config.weight_decay if decay else 0.0,
                    }
                )
    expected = {id(param) for param in model.parameters() if param.requires_grad}
    if included != expected:
        missing = [name for name, param in named if param.requires_grad and id(param) not in included]
        raise RuntimeError(f"Trainable parameters missing from optimizer: {missing}")
    if not groups:
        raise ValueError("No trainable parameters were selected for the optimizer.")
    return AdamW(groups)


def build_scheduler(config: TrainConfig, optimizer: AdamW, total_steps: int) -> LambdaLR:
    min_lr_ratio = min(max(config.min_lr_ratio, 0.0), 1.0)
    warmup_steps = round(total_steps * config.warmup_ratio) if config.warmup_ratio > 0 else config.warmup_steps

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        decay_steps = max(total_steps - warmup_steps, 1)
        decay_step = min(max(current_step - warmup_steps, 0), decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * float(decay_step) / float(decay_steps)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def maybe_resume(config: TrainConfig, model, optimizer, scheduler, scaler, tokenizer, device: torch.device) -> dict:
    checkpoint_path = None
    if config.checkpoint_path is not None:
        checkpoint_path = Path(config.checkpoint_path)
    elif config.resume:
        checkpoint_path = newest_resume_checkpoint(config.run_dir)

    if checkpoint_path is None or not checkpoint_path.exists():
        return {"step": 0, "epoch": 1, "microbatches_in_epoch": 0, "best_eval_loss": float("inf"), "evals_without_improvement": 0}

    payload = load_checkpoint(
        str(checkpoint_path),
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        map_location=device,
        expected_config=asdict(config),
        expected_format_version=(
            CHAT_FORMAT_VERSION if config.text_data_format in CHAT_DATA_FORMATS else None
        ),
        tokenizer=tokenizer,
        restore_rng=True,
    )
    if config.text_data_format in CHAT_DATA_FORMATS:
        format_version = payload.get("format_version")
        if format_version != CHAT_FORMAT_VERSION:
            raise RuntimeError(
                "Refusing to resume a legacy/incomplete SFT checkpoint. Controlled resume requires causal masking, "
                "tokenizer metadata, RNG, scaler, and data-position state. Expected format_version="
                f"{CHAT_FORMAT_VERSION}, "
                f"got {format_version!r}."
            )
    state = dict(payload.get("training_state") or {})
    state.setdefault("microbatches_in_epoch", 0)
    state.setdefault("best_eval_loss", float("inf"))
    state.setdefault("evals_without_improvement", 0)
    state.setdefault("samples_processed", 0)
    state.setdefault("tokens_processed", 0)
    state.setdefault("target_tokens_processed", 0)
    state.setdefault("elapsed_seconds", 0.0)
    state["step"] = int(payload.get("step", 0))
    state["epoch"] = int(payload.get("epoch", 1))
    return state


def current_group_lr(optimizer: AdamW, group_name: str, fallback: float) -> float:
    for group in optimizer.param_groups:
        if str(group.get("name", "")).startswith(group_name + "_"):
            return float(group["lr"])
    return fallback


def maximum_group_lr(optimizer: AdamW, group_prefix: str, fallback: float) -> float:
    values = [
        float(group["lr"])
        for group in optimizer.param_groups
        if str(group.get("name", "")).startswith(group_prefix + "_")
    ]
    return max(values) if values else fallback


def confirm_ate_training(config: TrainConfig, model) -> None:
    report = model.ate_parameter_report()
    metadata = model.ate_metadata()
    incremental_payload = getattr(config, "_ate_incremental_payload", None)
    is_incremental = incremental_payload is not None
    is_resume = getattr(config, "_ate_resume_metadata", None) is not None and not is_incremental
    existing_stage_count = (
        len(metadata["expansion_stages"]) - 1 if is_incremental
        else len(metadata["expansion_stages"]) if is_resume
        else 0
    )
    source_step = int(incremental_payload.get("step", 0)) if is_incremental else 0
    source_metric = (
        incremental_payload.get("training_state", {}).get("best_eval_loss", incremental_payload.get("best_eval_loss"))
        if is_incremental else None
    )
    requested_heads = int(getattr(config, "_ate_incremental_requested_heads", 0 if is_resume else config.new_attn_heads))
    requested_layers = int(getattr(config, "_ate_incremental_requested_layers", 0 if is_resume else config.new_ffn_layers))
    requested_hidden = requested_heads * (model.original_hidden // model.original_heads)
    print("=" * 78)
    print("Adaptive Transformer Expansion Startup Verification")
    print("=" * 78)
    print("SOURCE ARCHITECTURE")
    print(f"Base hidden / heads / layers : {model.original_hidden} / {model.original_heads} / {model.original_layers}")
    print(f"Existing expansion stages    : {existing_stage_count}")
    print(f"Original pretrained params   : {report.original_parameters:,}")
    if is_incremental:
        print(f"Source checkpoint / step     : {config._ate_incremental_source_path} / {source_step}")
        print(f"Source checkpoint metric     : {source_metric if source_metric is not None else 'n/a'}")
    print("REQUESTED STAGE")
    print(f"Added heads / hidden / layers: +{requested_heads} / +{requested_hidden} / +{requested_layers}")
    print("FINAL ARCHITECTURE")
    print(f"Hidden / heads / layers      : {metadata['expanded_hidden']} / {metadata['expanded_heads']} / {metadata['expanded_layers']}")
    print(f"Existing ATE parameters      : {report.previous_stage_parameters:,}")
    print(f"Parameters added this stage  : {report.current_stage_parameters:,}")
    print(f"Trainable original params    : {report.plastic_parameters:,}")
    print(f"Trainable previous ATE params: {sum(p.numel() for p in model.previous_stage_parameters() if p.requires_grad):,}")
    print(f"Trainable current ATE params : {sum(p.numel() for p in model.current_stage_parameters() if p.requires_grad):,}")
    print(f"Frozen / total / trainable   : {report.frozen_parameters:,} / {report.total_parameters:,} / {report.total_trainable_parameters:,}")
    print(f"Normalization               : {metadata['normalization_strategy']}")
    for index, stage in enumerate(metadata["expansion_stages"]):
        stage_parameters = sum(
            parameter.numel()
            for parameter in model._parameters_from_ids(model._stage_parameter_ids[index])
        )
        print(
            f"Stage {stage['stage_id']} parameters     : {stage_parameters:,} | "
            f"+H{stage['added_attention_heads']} +L{stage['added_transformer_layers']} "
            f"-> {stage['result_hidden_size']}/{stage['result_layers']}"
        )
    preservation = getattr(config, "_ate_preservation_report", None)
    if preservation:
        print("PRESERVATION TEST")
        print(
            f"Verification device / dtype  : "
            f"{preservation.get('verification_device', 'unknown')} / "
            f"{preservation.get('verification_dtype', 'unknown')}"
        )
        print(f"Max / mean logit difference : {preservation['max_abs_logit_difference']:.3e} / {preservation['mean_abs_logit_difference']:.3e}")
        print(f"Existing hidden differences : {preservation.get('existing_hidden_state_max_differences', [])}")
        print(f"Result                      : {'PASS' if preservation['passed'] else 'FAIL'}")
    print("=" * 78)
    if config.ate_confirm is not None:
        answer = config.ate_confirm
    else:
        try:
            answer = input("Continue?\nType YES to continue:\n> ")
        except EOFError:
            answer = ""
    if answer != "YES":
        print("ATE training cancelled safely; no optimizer or checkpoint state was modified.")
        raise SystemExit(0)


def move_optimizer_state(optimizer: AdamW, device: torch.device | str) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device, non_blocking=False)


def parameter_category(name: str) -> str:
    lowered = name.lower()
    if "input_modifier" in lowered or "embedding_modifier" in lowered:
        return "input_mod"
    if "output_modifier" in lowered:
        return "output_mod"
    if "attention_modifier" in lowered:
        return "attention_mod"
    if "ffn_modifier" in lowered:
        return "ffn_mod"
    if "embed_in" in lowered or "embedding" in lowered:
        return "embedding"
    if "attention" in lowered or "query_key_value" in lowered:
        return "attention"
    if ".mlp" in lowered or "dense_h_to_4h" in lowered or "dense_4h_to_h" in lowered:
        return "ffn"
    if "layernorm" in lowered or "final_layer_norm" in lowered:
        return "normalization"
    if "embed_out" in lowered or "lm_head" in lowered:
        return "lm_head"
    return "other"


def print_trainable_parameter_summary(model) -> None:
    categories: dict[str, dict[str, object]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        item = categories.setdefault(parameter_category(name), {"tensors": 0, "parameters": 0, "names": []})
        item["tensors"] += 1
        item["parameters"] += parameter.numel()
        item["names"].append(name)
    print("trainable parameter categories:")
    for category, item in sorted(categories.items()):
        print(f"  {category}: tensors={item['tensors']} parameters={item['parameters']:,}")
        if category.endswith("_mod"):
            for name in item["names"]:
                print(f"    {name}")


def limit_supervised_targets(labels: torch.Tensor, maximum_targets: int) -> torch.Tensor:
    """Keep exactly the first N supervised targets in deterministic row-major order."""
    if maximum_targets < 0:
        raise ValueError("maximum_targets must be non-negative")
    supervised = labels.ne(-100)
    count = int(supervised.sum().item())
    if count <= maximum_targets:
        return labels
    limited = labels.clone()
    positions = supervised.reshape(-1).nonzero(as_tuple=False).flatten()
    flat = limited.reshape(-1)
    flat[positions[maximum_targets:]] = -100
    return limited


def trim_batch_to_target_budget(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    maximum_targets: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trim the terminal batch to the exact ordered target/data prefix."""
    if maximum_targets <= 0:
        empty_mask = torch.zeros_like(input_ids[:0])
        return input_ids[:0], labels[:0], empty_mask
    supervised_positions = labels.ne(-100).nonzero(as_tuple=False)
    if supervised_positions.shape[0] <= maximum_targets:
        mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        return input_ids, labels, mask
    final_row, final_column = supervised_positions[maximum_targets - 1].tolist()
    input_ids = input_ids[: final_row + 1]
    labels = limit_supervised_targets(labels[: final_row + 1], maximum_targets)
    mask = (
        attention_mask[: final_row + 1].clone()
        if attention_mask is not None
        else torch.ones_like(input_ids)
    )
    mask[-1, final_column + 1 :] = 0
    labels[-1, final_column + 1 :] = -100
    return input_ids, labels, mask


def autocast_context(device: torch.device, enabled: bool, *, bf16: bool = False, fp16: bool = False):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if bf16 or (not fp16 and torch.cuda.is_bf16_supported()) else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def evaluate(config: TrainConfig, model: GPTModModel, tokenizer, device: torch.device) -> dict[str, dict[str, float | int]]:
    was_training = model.training
    model.eval()
    eval_loader = build_eval_loader(config, tokenizer)
    def tag_variants(tag: str) -> tuple[tuple[int, ...], ...]:
        variants = {
            tuple(tokenizer.encode(tag, add_special_tokens=False)),
            tuple(tokenizer.encode(" " + tag, add_special_tokens=False)),
        }
        return tuple(variant for variant in variants if variant)

    totals = EvalTotals(
        reasoning_open_ids=tag_variants("<thinking>"),
        reasoning_close_ids=tag_variants("</thinking>"),
        final_open_ids=tag_variants("<output>"),
        final_close_ids=tag_variants("</output>"),
    )
    with torch.inference_mode():
        for idx, batch in enumerate(eval_loader):
            if config.eval_batches > 0 and idx >= config.eval_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            with autocast_context(device, config.mixed_precision, bf16=config.bf16, fp16=config.fp16):
                output = model(input_ids=input_ids, attention_mask=attention_mask)
            totals.update(
                output["logits"],
                input_ids,
                labels,
                tokenizer.convert_tokens_to_ids("<ASSISTANT>"),
                tokenizer.eos_token_id,
            )
    model.train(was_training)
    return totals.metrics()


def print_eval_metrics(prefix: str, metrics: dict[str, dict[str, float | int]]) -> None:
    fields = []
    for name in ("assistant_content", "first_answer", "eos", "combined"):
        item = metrics[name]
        fields.append(f"{name} ppl={item['ppl']:.2f} top1={100.0 * item['top1']:.1f}% n={item['count']}")
    print(prefix + " | " + " | ".join(fields))


def next_train_batch(config: TrainConfig, tokenizer, current_epoch: int, train_loader, train_iterator):
    try:
        batch = next(train_iterator)
    except StopIteration:
        current_epoch += 1
        train_loader = build_train_loader(config, tokenizer, epoch=current_epoch)
        train_iterator = iter(train_loader)
        batch = next(train_iterator)
    return batch, current_epoch, train_loader, train_iterator


def main() -> None:
    config = parse_args()
    if not config.text_data_path:
        raise ValueError("--text_data_path is required for training.")
    config.text_data_format = resolve_text_data_format(config)
    set_seed(config.seed)
    device = resolve_device(config.device)
    if config.experiment_type == "ate":
        prepare_incremental_ate_config(config)
        prepare_ate_resume_architecture(config)
    Path(config.run_dir).mkdir(parents=True, exist_ok=True)

    model, tokenizer = build_model_and_tokenizer(config, device)
    if config.experiment_type == "ate":
        confirm_ate_training(config, model)
        if getattr(config, "_ate_incremental_payload", None) is not None or not config.resume:
            probe_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device).clamp_max(
                max(model.original_vocab_size - 1, 0)
            )
            if model.control_token_ids and model.expansion_stages:
                newest = model.expansion_stages[-1]
                if not newest.get("added_attention_heads") and not newest.get("added_transformer_layers"):
                    probe_ids[0, 1] = int(model.control_token_ids[0])
            gradient_report = model.gradient_diagnostics(probe_ids)
            config._ate_gradient_report = gradient_report
            print("GRADIENT TEST")
            for category, values in gradient_report.items():
                print(
                    f"{category:18s} total={values['total_parameter_tensors']} "
                    f"grad={values['tensors_with_gradients']} "
                    f"nonzero={values['tensors_with_nonzero_gradients']} "
                    f"norm={values['gradient_norm']:.3e}"
                )
            newest_stage = model.expansion_stages[-1] if model.expansion_stages else {}
            if int(newest_stage.get("added_attention_heads", 0)):
                required_categories = ("Attention output", "FFN output", "LM head")
            elif int(newest_stage.get("added_transformer_layers", 0)):
                required_categories = ("Depth block",)
            else:
                required_categories = ("Embedding", "LM head")
            missing_output_gradients = [
                category for category in required_categories
                if gradient_report[category]["tensors_with_nonzero_gradients"] == 0
            ]
            internal_nonzero = sum(
                gradient_report[category]["tensors_with_nonzero_gradients"]
                for category in ("Attention input", "FFN input")
            )
            gradient_status = "FAIL" if missing_output_gradients else "PASS" if internal_nonzero else "WARN"
            print(
                f"GRADIENT RESULT {gradient_status} | output_gates_missing={missing_output_gradients or 'none'} "
                f"| new_internal_nonzero={internal_nonzero}"
            )
            if missing_output_gradients:
                raise RuntimeError(
                    "ATE newest-stage output paths did not receive gradients: "
                    f"{missing_output_gradients}"
                )
        apply_incremental_ate_checkpoint(config, model, tokenizer)
    if config.dataset_manifest:
        validate_locked_dataset(
            config.dataset_manifest,
            [
                config.text_data_path,
                *( [config.eval_text_data_path] if config.eval_text_data_path else [] ),
            ],
            tokenizer,
        )
    if config.experiment_type == "ate":
        reuse_incremental_steps_cache(config, tokenizer)
    if config.steps_per_epoch is None:
        if config.max_steps > 0:
            # A step-limited run does not need an exact epoch cardinality.
            # Avoid tokenizing the entire dataset before the first update.
            config.steps_per_epoch = config.max_steps
            print(
                f"Step-limited run: using steps_per_epoch={config.steps_per_epoch} for display; "
                "skipping full-dataset epoch inference."
            )
        else:
            config.steps_per_epoch = infer_steps_per_epoch(config, tokenizer)
    planned_steps = max(config.max_steps, 1) if config.max_steps > 0 else max(config.epochs * config.steps_per_epoch, 1)
    scheduler_units = config.target_training_tokens if config.target_training_tokens > 0 else planned_steps
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, total_steps=max(scheduler_units, 1))
    use_fp16_scaler = config.mixed_precision and device.type == "cuda" and (
        config.fp16 or (not config.bf16 and not torch.cuda.is_bf16_supported())
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    drift_tracker = PlasticityDriftTracker(model)
    resume_state = maybe_resume(config, model, optimizer, scheduler, scaler, tokenizer, device)
    if config.optimizer_state_offload:
        if device.type != "cuda":
            raise ValueError("--optimizer-state-offload requires CUDA training.")
        move_optimizer_state(optimizer, "cpu")
        print("optimizer state offload enabled: AdamW state stays on CPU between steps; expect PCIe overhead.")
    global_step = int(resume_state["step"])
    current_epoch = int(resume_state["epoch"])
    resume_microbatches = int(resume_state["microbatches_in_epoch"])
    metric_logger = MetricLogger(config.run_dir, resume_step=global_step if global_step > 0 else None)
    drift_path = Path(config.run_dir) / "plasticity_drift.jsonl"
    if global_step == 0:
        drift_path.write_text("", encoding="utf-8")

    Path(config.run_dir, "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )

    stats = model.model_stats()
    print(
        f"params total={stats.total_params:,} trainable={stats.trainable_params:,} "
        f"base={stats.base_params:,} modifier={stats.modifier_params:,} lora={stats.lora_params:,}"
    )
    print(
        f"batch micro={config.batch_size} grad_accum={max(config.grad_accum_steps, 1)} "
        f"effective={config.batch_size * max(config.grad_accum_steps, 1)}"
    )
    if config.text_data_format in CHAT_DATA_FORMATS:
        print(f"system prompts: {'ignored' if config.ignore_system_prompt else 'included'}")
    if config.experiment_type in {"frozen_mod", "full_finetune"}:
        report = model.parameter_report()
        print(
            f"pythia mode={config.experiment_type} frozen_base={report.frozen_base_parameters:,} "
            f"trainable_mod={report.trainable_mod_parameters:,} "
            f"trainable_plastic={report.trainable_plastic_parameters:,} "
            f"active/token={report.active_parameters_per_token:,} "
            f"parameter_memory={report.parameter_memory_bytes / 2**30:.2f}GiB "
            f"trainable={report.trainable_percentage:.4f}%"
        )
        print_trainable_parameter_summary(model)
        if report.trainable_plastic_parameters:
            print(
                f"plasticity lr={config.plastic_lr:.2e} layernorm_lr={config.plastic_layernorm_lr:.2e} "
                f"tensors={len(model.plastic_parameter_names())}"
            )

    window_start = time.time()
    window_input_tokens = 0
    window_target_tokens = 0
    window_nll_sum = 0.0
    window_base_logit_norm_sum = 0.0
    window_output_mod_logit_norm_sum = 0.0
    window_logit_norm_batches = 0
    window_target_tokens_lost = 0
    window_truncated_samples = 0
    last_eval_loss = float("nan")
    last_eval_combined_loss = float("nan")
    last_eos_accuracy = float("nan")
    last_first_answer_accuracy = float("nan")
    last_target_top1_accuracy = float("nan")
    startup_drift_report = drift_tracker.measure() if drift_tracker.parameters else None
    last_plastic_drift = (
        float(startup_drift_report["relative_drift"])
        if startup_drift_report is not None else float("nan")
    )
    if startup_drift_report is not None:
        print(f"plastic drift at load step {global_step}: {last_plastic_drift:.3e}", flush=True)
        if global_step == 0:
            with open(drift_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": 0, **startup_drift_report}, sort_keys=True) + "\n")
    best_eval_loss = float(resume_state["best_eval_loss"])
    evals_without_improvement = int(resume_state["evals_without_improvement"])
    samples_processed = int(resume_state.get("samples_processed", 0))
    tokens_processed = int(resume_state.get("tokens_processed", 0))
    target_tokens_processed = int(resume_state.get("target_tokens_processed", 0))
    training_start_time = time.time() - float(resume_state.get("elapsed_seconds", 0.0))
    last_checkpoint_path = ""
    should_stop = False
    stop_reason = ""
    grad_accum_steps = max(config.grad_accum_steps, 1)

    if global_step == 0 and (config.eval_every > 0 or config.eval_every_tokens > 0):
        initial_eval = evaluate(config, model, tokenizer, device)
        last_eval_loss = float(initial_eval["assistant_content"]["nll"])
        last_eval_combined_loss = float(initial_eval["combined"]["nll"])
        last_eos_accuracy = float(initial_eval["eos"]["top1"])
        last_first_answer_accuracy = float(initial_eval["first_answer"]["top1"])
        last_target_top1_accuracy = float(initial_eval["combined"]["top1"])
        if not math.isnan(last_eval_loss):
            print_eval_metrics("base reference", initial_eval)
            print_eval_metrics(f"step 0 | variant {config.variant}", initial_eval)

    displayed_total_epochs = config.epochs if config.max_steps <= 0 else max(
        config.epochs, math.ceil(planned_steps / max(config.steps_per_epoch, 1))
    )
    resume_position = resume_microbatches

    def budget_reached() -> bool:
        if config.target_training_tokens > 0:
            return target_tokens_processed >= config.target_training_tokens
        return global_step >= planned_steps

    def checkpoint(
        epoch: int,
        microbatches_in_epoch: int,
        *,
        save_step: bool = True,
        save_best: bool = False,
    ) -> None:
        nonlocal last_checkpoint_path
        saved_path = save_checkpoint(
            run_dir=config.run_dir,
            step=global_step,
            epoch=epoch,
            model=model,
            optimizer=None if config.no_optimizer else optimizer,
            scheduler=None if config.no_optimizer else scheduler,
            config_dict=asdict(config),
            tokenizer=tokenizer,
            scaler=None if config.no_optimizer else scaler,
            training_state={
                "microbatches_in_epoch": microbatches_in_epoch,
                "best_eval_loss": best_eval_loss,
                "best_eval_ppl": math.exp(min(best_eval_loss, 20.0)) if math.isfinite(best_eval_loss) else float("nan"),
                "evals_without_improvement": evals_without_improvement,
                "samples_processed": samples_processed,
                "tokens_processed": tokens_processed,
                "target_tokens_processed": target_tokens_processed,
                "elapsed_seconds": time.time() - training_start_time,
            },
            save_step=save_step,
            save_best=save_best,
        )
        last_checkpoint_path = str(saved_path)

    def complete_update(
        epoch: int,
        microbatches_in_epoch: int,
        accumulated_nll: float,
        accumulated_targets: int,
        accumulated_input_tokens: int,
        accumulated_samples: int,
    ) -> None:
        nonlocal global_step, last_eval_loss, last_eval_combined_loss, best_eval_loss, evals_without_improvement, should_stop, stop_reason
        nonlocal last_eos_accuracy, last_first_answer_accuracy, last_target_top1_accuracy
        nonlocal last_plastic_drift
        nonlocal samples_processed, tokens_processed, target_tokens_processed
        nonlocal window_start, window_input_tokens, window_target_tokens, window_nll_sum
        nonlocal window_base_logit_norm_sum, window_output_mod_logit_norm_sum, window_logit_norm_batches
        nonlocal window_target_tokens_lost, window_truncated_samples, resume_position
        if accumulated_targets <= 0:
            raise RuntimeError("An optimizer update cannot have zero supervised targets.")
        scaler.unscale_(optimizer)
        for param in model.parameters():
            if param.grad is not None:
                param.grad.div_(float(accumulated_targets))
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
        previous_scale = scaler.get_scale()
        if config.optimizer_state_offload:
            move_optimizer_state(optimizer, device)
        scaler.step(optimizer)
        if config.optimizer_state_offload:
            move_optimizer_state(optimizer, "cpu")
        scaler.update()
        if scaler.is_enabled() and scaler.get_scale() < previous_scale:
            raise FloatingPointError("FP16 gradient overflow skipped an optimizer update; aborting controlled run.")
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        previous_target_tokens = target_tokens_processed
        samples_processed += accumulated_samples
        tokens_processed += accumulated_input_tokens
        target_tokens_processed += accumulated_targets
        if config.target_training_tokens > 0:
            scheduler.step(target_tokens_processed)
        else:
            scheduler.step()
        resume_position = microbatches_in_epoch
        window_nll_sum += accumulated_nll
        window_target_tokens += accumulated_targets
        window_input_tokens += accumulated_input_tokens

        token_eval_due = config.eval_every_tokens > 0 and (
            previous_target_tokens // config.eval_every_tokens
            < target_tokens_processed // config.eval_every_tokens
        )
        step_eval_due = (
            config.eval_every_tokens <= 0
            and config.eval_every > 0
            and global_step % config.eval_every == 0
        )
        best_checkpoint_due = False
        if token_eval_due or step_eval_due or budget_reached():
            clear_live_dashboard()
            eval_metrics = evaluate(config, model, tokenizer, device)
            last_eval_loss = float(eval_metrics["assistant_content"]["nll"])
            last_eval_combined_loss = float(eval_metrics["combined"]["nll"])
            last_eos_accuracy = float(eval_metrics["eos"]["top1"])
            last_first_answer_accuracy = float(eval_metrics["first_answer"]["top1"])
            last_target_top1_accuracy = float(eval_metrics["combined"]["top1"])
            print_eval_metrics(f"eval step {global_step}", eval_metrics)
            if drift_tracker.parameters:
                drift_report = drift_tracker.measure()
                last_plastic_drift = float(drift_report["relative_drift"])
                with open(drift_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": global_step, **drift_report}, sort_keys=True) + "\n")
                group_text = " ".join(
                    f"{name}={value:.3e}"
                    for name, value in drift_report["by_group"].items()
                )
                print(f"plastic drift total={last_plastic_drift:.3e} {group_text}", flush=True)
            assistant_ppl = float(eval_metrics["assistant_content"]["ppl"])
            if config.stop_eval_ppl > 0 and assistant_ppl <= config.stop_eval_ppl:
                should_stop = True
                stop_reason = f"assistant-content eval PPL reached {assistant_ppl:.6f} <= {config.stop_eval_ppl:.6f}"
            combined_ppl = float(eval_metrics["combined"]["ppl"])
            if config.stop_eval_combined_ppl > 0 and combined_ppl <= config.stop_eval_combined_ppl:
                should_stop = True
                stop_reason = (
                    f"combined assistant/EOS eval PPL reached {combined_ppl:.6f} "
                    f"<= {config.stop_eval_combined_ppl:.6f}"
                )
            if not math.isnan(last_eval_loss):
                previous_best = best_eval_loss
                if last_eval_loss < previous_best:
                    best_eval_loss = last_eval_loss
                    best_checkpoint_due = True
                if last_eval_loss < previous_best - config.early_stop_min_delta:
                    evals_without_improvement = 0
                else:
                    evals_without_improvement += 1
                    if config.early_stop_patience > 0 and evals_without_improvement >= config.early_stop_patience:
                        should_stop = True
                        stop_reason = f"assistant-content eval PPL did not improve for {evals_without_improvement} evals"

        token_save_due = config.save_every_tokens > 0 and (
            previous_target_tokens // config.save_every_tokens
            < target_tokens_processed // config.save_every_tokens
        )
        step_save_due = (
            config.save_every_tokens <= 0
            and config.save_every > 0
            and global_step % config.save_every == 0
        )
        if token_save_due or step_save_due or budget_reached() or should_stop:
            if not best_checkpoint_due:
                checkpoint(epoch, microbatches_in_epoch, save_step=True)
        if best_checkpoint_due:
            checkpoint(epoch, microbatches_in_epoch, save_step=False, save_best=True)
            print(
                f"best checkpoint updated | step={global_step} | "
                f"assistant-content eval ppl={math.exp(min(best_eval_loss, 20.0)):.6f} | "
                f"path={last_checkpoint_path}",
                flush=True,
            )

        if config.log_every > 0:
            now = time.time()
            elapsed = max(now - window_start, 1e-6)
            train_loss = window_nll_sum / max(window_target_tokens, 1)
            dashboard = build_dashboard_metrics(
                step=global_step,
                epoch=epoch,
                total_epochs=displayed_total_epochs,
                loss=train_loss,
                eval_assistant_loss=last_eval_loss,
                eval_combined_loss=last_eval_combined_loss,
                lr=current_group_lr(
                    optimizer, "ate_new" if config.experiment_type == "ate" else "base", 0.0
                ),
                modifier_lr=current_group_lr(optimizer, "modifier", 0.0),
                lora_lr=current_group_lr(optimizer, "lora", 0.0),
                plastic_lr=(
                    maximum_group_lr(optimizer, "ate_plastic", 0.0)
                    if config.experiment_type == "ate"
                    else current_group_lr(optimizer, "plastic", 0.0)
                ),
                plastic_layernorm_lr=current_group_lr(optimizer, "plastic_norm", 0.0),
                plastic_drift=last_plastic_drift,
                tokens_per_sec=window_input_tokens / elapsed,
                target_tokens_per_sec=window_target_tokens / elapsed,
                target_tokens=window_target_tokens,
                progress_pct=(
                    100.0 * target_tokens_processed / config.target_training_tokens
                    if config.target_training_tokens > 0
                    else 100.0 * float(global_step) / float(planned_steps)
                ),
                device=device.type,
                samples_processed=samples_processed,
                tokens_processed=tokens_processed,
                target_tokens_processed=target_tokens_processed,
                effective_epochs=global_step / max(config.steps_per_epoch, 1),
                checkpoint_path=last_checkpoint_path,
            )
            dashboard.update(
                elapsed_seconds=time.time() - training_start_time,
                eos_accuracy=last_eos_accuracy,
                first_answer_accuracy=last_first_answer_accuracy,
                target_top1_accuracy=last_target_top1_accuracy,
                base_logit_norm=(
                    window_base_logit_norm_sum / window_logit_norm_batches
                    if window_logit_norm_batches else float("nan")
                ),
                output_mod_logit_norm=(
                    window_output_mod_logit_norm_sum / window_logit_norm_batches
                    if window_logit_norm_batches else float("nan")
                ),
                output_mod_logit_ratio=(
                    window_output_mod_logit_norm_sum / max(window_base_logit_norm_sum, 1e-12)
                    if window_logit_norm_batches else float("nan")
                ),
                micro_batch_size=config.batch_size,
                grad_accum_steps=grad_accum_steps,
                effective_batch_size=config.batch_size * grad_accum_steps,
            )
            log_boundary = global_step % config.log_every == 0 or budget_reached() or should_stop
            render_dashboard(dashboard, live=not log_boundary)
            if log_boundary:
                if window_truncated_samples:
                    print(f"  truncation samples={window_truncated_samples} target_tokens_lost={window_target_tokens_lost}")
                metric_logger.log(dashboard)
                if config.plot_every > 0 and (global_step % config.plot_every == 0 or budget_reached()):
                    metric_logger.plot()
                window_start = now
                window_input_tokens = 0
                window_target_tokens = 0
                window_nll_sum = 0.0
                window_base_logit_norm_sum = 0.0
                window_output_mod_logit_norm_sum = 0.0
                window_logit_norm_batches = 0
                window_target_tokens_lost = 0
                window_truncated_samples = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        while not budget_reached() and not should_stop:
            train_loader = build_train_loader(config, tokenizer, epoch=current_epoch)
            train_iterator = iter(train_loader)
            skipped = 0
            while skipped < resume_position:
                try:
                    next(train_iterator)
                except StopIteration:
                    break
                skipped += 1
            if skipped < resume_position:
                current_epoch += 1
                resume_position = 0
                continue

            accumulation_count = 0
            accumulation_nll = 0.0
            accumulation_targets = 0
            accumulation_input_tokens = 0
            accumulation_samples = 0
            consumed = resume_position
            for batch in train_iterator:
                consumed += 1
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                if config.target_training_tokens > 0:
                    remaining_targets = config.target_training_tokens - (
                        target_tokens_processed + accumulation_targets
                    )
                    input_ids, labels, attention_mask = trim_batch_to_target_budget(
                        input_ids, labels, attention_mask, max(remaining_targets, 0)
                    )
                    if input_ids.shape[0] == 0 or int(labels.ne(-100).sum().item()) == 0:
                        break

                with autocast_context(device, config.mixed_precision, bf16=config.bf16, fp16=config.fp16):
                    output = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                if output.get("base_logit_norm") is not None:
                    window_base_logit_norm_sum += float(output["base_logit_norm"].detach().item())
                    window_output_mod_logit_norm_sum += float(
                        output["output_mod_logit_norm"].detach().item()
                    )
                    window_logit_norm_batches += 1
                scaler.scale(output["loss_sum"]).backward()
                if config.experiment_type == "frozen_mod":
                    model.assert_base_gradients_absent()
                batch_targets = int(output["target_count"].detach().item())
                accumulation_nll += float(output["loss_sum"].detach().item())
                accumulation_targets += batch_targets
                accumulation_input_tokens += int(attention_mask.sum().item()) if attention_mask is not None else input_ids.numel()
                accumulation_samples += int(input_ids.shape[0])
                accumulation_count += 1
                lost = int(batch.get("target_tokens_lost", torch.tensor(0)).sum().item())
                truncated = int(batch.get("was_truncated", torch.tensor(0)).sum().item())
                if truncated < 0 or truncated > input_ids.size(0):
                    raise RuntimeError(f"Invalid truncation metadata: {truncated} for batch size {input_ids.size(0)}")
                window_target_tokens_lost += lost
                window_truncated_samples += truncated
                if accumulation_count == grad_accum_steps:
                    complete_update(
                        current_epoch, consumed, accumulation_nll, accumulation_targets,
                        accumulation_input_tokens, accumulation_samples,
                    )
                    accumulation_count = 0
                    accumulation_nll = 0.0
                    accumulation_targets = 0
                    accumulation_input_tokens = 0
                    accumulation_samples = 0
                    if budget_reached() or should_stop:
                        break

            if budget_reached() or should_stop:
                break
            if accumulation_count:
                complete_update(
                    current_epoch, consumed, accumulation_nll, accumulation_targets,
                    accumulation_input_tokens, accumulation_samples,
                )
            current_epoch += 1
            resume_position = 0
    except KeyboardInterrupt:
        clear_live_dashboard()
        optimizer.zero_grad(set_to_none=True)
        checkpoint(current_epoch, resume_position)
        print(f"Interrupted safely; checkpoint saved at optimizer step {global_step}.")
        return

    if not last_checkpoint_path:
        checkpoint(current_epoch, resume_position)
    if should_stop:
        clear_live_dashboard()
        metric_logger.plot()
        eval_ppl = math.exp(min(last_eval_loss, 20.0)) if not math.isnan(last_eval_loss) else float("nan")
        print(f"stopping at step {global_step}: {stop_reason or f'assistant-content eval PPL {eval_ppl:.4f}'}")


if __name__ == "__main__":
    main()
