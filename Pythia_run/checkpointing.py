from __future__ import annotations

import os
import random
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHAT_FORMAT_VERSION = "gpt_mod_eos_turn_causal_v4"
PLAIN_FORMAT_VERSION = "gpt_mod_v2"
ARCHITECTURE_CONFIG_KEYS = (
    "variant",
    "model_name",
    "tokenizer_name",
    "text_data_format",
    "ignore_system_prompt",
    "mod_dim",
    "out_mod_dim",
    "attn_mod_dim",
    "ffn_mod_dim",
    "attn_unique_mod_count",
    "ffn_unique_mod_count",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "freeze_base",
    "dropout",
    "experiment_type",
    "revision",
    "emb_mod_scale",
    "out_mod_scale",
    "attn_mod_scale",
    "ffn_mod_scale",
    "learnable_mod_scales",
    "plastic_last_n_layers",
    "plastic_layer_norms",
    "plastic_biases",
    "plastic_attn_output",
    "plastic_ffn_output",
    "plastic_final_layer_norm",
    "plastic_lm_head",
    "strict_sample_order",
    "dataset_manifest",
    "new_attn_heads",
    "new_ffn_layers",
    "plasticity_mode",
    "plasticity_base_lr",
    "plasticity_custom_scales",
    "ate_output_init",
    "ate_output_init_scale",
    "previous_ate_stages_trainable",
)
ARCHITECTURE_CONFIG_DEFAULTS = {
    "experiment_type": "legacy",
    "revision": None,
    "out_mod_dim": None,
    "emb_mod_scale": 1.0,
    "out_mod_scale": 1.0,
    "attn_mod_scale": 1.0,
    "ffn_mod_scale": 1.0,
    "learnable_mod_scales": False,
    "attn_unique_mod_count": 1,
    "ffn_unique_mod_count": 1,
    "ignore_system_prompt": False,
    "plastic_last_n_layers": 0,
    "plastic_layer_norms": False,
    "plastic_biases": False,
    "plastic_attn_output": False,
    "plastic_ffn_output": False,
    "plastic_final_layer_norm": False,
    "plastic_lm_head": False,
    "strict_sample_order": False,
    "dataset_manifest": None,
    "new_attn_heads": 0,
    "new_ffn_layers": 0,
    "plasticity_mode": "quadratic",
    "plasticity_base_lr": 1e-5,
    "plasticity_custom_scales": "",
    "ate_output_init": "exact",
    "ate_output_init_scale": 0.0,
    "previous_ate_stages_trainable": True,
}


def checkpoint_paths(run_dir: str) -> tuple[Path, Path]:
    checkpoint_dir = Path(run_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, checkpoint_dir / "best.pt"


def newest_resume_checkpoint(run_dir: str) -> Path | None:
    checkpoint_dir, best_path = checkpoint_paths(run_dir)
    candidates = list(checkpoint_dir.glob("step_*.pt"))
    if best_path.exists():
        candidates.append(best_path)
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        raise RuntimeError("Checkpoint does not contain RNG state.")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = state["torch"]
    if not isinstance(torch_state, torch.Tensor) or torch_state.dtype != torch.uint8:
        raise TypeError("Checkpoint torch RNG state must be a torch.uint8 tensor.")
    torch.set_rng_state(torch_state.cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def tokenizer_metadata(tokenizer) -> dict[str, Any]:
    control_tokens = ("<USER>", "<ASSISTANT>", "<SYSTEM>")
    vocabulary = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
    vocab_sha256 = None
    if vocabulary is not None:
        vocab_sha256 = hashlib.sha256(
            json.dumps(sorted(vocabulary.items()), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    return {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "control_token_ids": {token: tokenizer.convert_tokens_to_ids(token) for token in control_tokens},
        "vocab_sha256": vocab_sha256,
    }


def validate_checkpoint_config(payload: dict[str, Any], expected_config: dict[str, Any] | None) -> None:
    if expected_config is None:
        return
    saved = payload.get("config")
    if not isinstance(saved, dict):
        raise RuntimeError("Checkpoint is missing its training config.")
    mismatches = {
        key: (
            saved.get(key, ARCHITECTURE_CONFIG_DEFAULTS.get(key)),
            expected_config.get(key, ARCHITECTURE_CONFIG_DEFAULTS.get(key)),
        )
        for key in ARCHITECTURE_CONFIG_KEYS
        if saved.get(key, ARCHITECTURE_CONFIG_DEFAULTS.get(key))
        != expected_config.get(key, ARCHITECTURE_CONFIG_DEFAULTS.get(key))
    }
    if mismatches:
        raise RuntimeError(f"Checkpoint architecture/config mismatch: {mismatches}")


def validate_tokenizer_metadata(payload: dict[str, Any], tokenizer) -> None:
    saved = payload.get("tokenizer_metadata")
    if not isinstance(saved, dict):
        raise RuntimeError("Checkpoint is missing tokenizer metadata.")
    current = tokenizer_metadata(tokenizer)
    for key in ("vocab_size", "eos_token_id", "pad_token_id", "control_token_ids", "vocab_sha256"):
        if saved.get(key) != current.get(key):
            raise RuntimeError(
                f"Checkpoint tokenizer mismatch for {key}: saved={saved.get(key)!r} current={current.get(key)!r}"
            )


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def validate_finite_model(model, context: str) -> None:
    for name, param in model.named_parameters():
        if not bool(torch.isfinite(param.detach()).all()):
            raise FloatingPointError(f"Non-finite model parameter {name!r} detected {context}.")


def save_checkpoint(
    run_dir: str,
    step: int,
    epoch: int,
    model,
    optimizer,
    scheduler,
    config_dict: dict,
    *,
    tokenizer=None,
    scaler=None,
    training_state: dict[str, Any] | None = None,
    save_step: bool = True,
    save_best: bool = False,
) -> Path:
    if not save_step and not save_best:
        raise ValueError("Checkpoint save requires save_step and/or save_best.")
    checkpoint_dir, best_path = checkpoint_paths(run_dir)
    validate_finite_model(model, "before checkpoint save")
    checkpoint_path = checkpoint_dir / f"step_{step:07d}.pt"
    sparse_mod = config_dict.get("experiment_type") == "frozen_mod" and hasattr(model, "sparse_checkpoint_state")
    payload = {
        "format_version": (
            CHAT_FORMAT_VERSION
            if config_dict.get("text_data_format") in {"chat_jsonl", "chat_parquet"}
            else PLAIN_FORMAT_VERSION
        ),
        "step": int(step),
        "epoch": int(epoch),
        "model_state_type": "pythia_sparse_mod" if sparse_mod else "full",
        "model_state_dict": model.sparse_checkpoint_state() if sparse_mod else model.state_dict(),
        "config": config_dict,
        "tokenizer_metadata": tokenizer_metadata(tokenizer) if tokenizer is not None else None,
        "training_state": dict(training_state or {}),
        "checkpoint_kind": "training" if optimizer is not None else "inference",
        "checkpoint_role": "best" if save_best and not save_step else "step",
    }
    if hasattr(model, "ate_metadata"):
        payload["architecture_metadata"] = model.ate_metadata()
    if optimizer is not None:
        payload.update(
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
            scaler_state_dict=scaler.state_dict() if scaler is not None else None,
            rng_state=capture_rng_state(),
        )
    saved_paths: list[Path] = []
    if save_step:
        _atomic_torch_save(payload, checkpoint_path)
        saved_paths.append(checkpoint_path)
    if save_best:
        best_payload = dict(payload)
        best_payload["checkpoint_role"] = "best"
        _atomic_torch_save(best_payload, best_path)
        saved_paths.append(best_path)
    legacy_latest = checkpoint_dir / "latest.pt"
    if legacy_latest.exists():
        legacy_latest.unlink()
    return best_path if save_best else saved_paths[0]


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location: str | torch.device = "cpu",
    *,
    expected_config: dict[str, Any] | None = None,
    expected_format_version: str | None = None,
    tokenizer=None,
    restore_rng: bool = False,
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Unable to load checkpoint {path}: {exc}") from exc
    return load_checkpoint_payload(
        payload,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_config=expected_config,
        expected_format_version=expected_format_version,
        tokenizer=tokenizer,
        restore_rng=restore_rng,
        source=path,
    )


def load_checkpoint_payload(
    payload: dict[str, Any],
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    *,
    expected_config: dict[str, Any] | None = None,
    expected_format_version: str | None = None,
    tokenizer=None,
    restore_rng: bool = False,
    source: str = "in-memory checkpoint",
) -> dict[str, Any]:
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError(f"Checkpoint {source} is not a GPT_MOD training checkpoint.")
    if expected_format_version is not None and payload.get("format_version") != expected_format_version:
        raise RuntimeError(
            f"Checkpoint format mismatch: saved={payload.get('format_version')!r} "
            f"expected={expected_format_version!r}. EOT and EOS-per-turn checkpoints are not interchangeable."
        )
    validate_checkpoint_config(payload, expected_config)
    if hasattr(model, "validate_ate_metadata"):
        model.validate_ate_metadata(payload.get("architecture_metadata"))
    if tokenizer is not None:
        validate_tokenizer_metadata(payload, tokenizer)
    if payload.get("model_state_type", "full") == "pythia_sparse_mod":
        if not hasattr(model, "load_sparse_checkpoint_state"):
            raise RuntimeError("Checkpoint contains sparse Pythia MOD state but the model does not support it.")
        model.load_sparse_checkpoint_state(payload["model_state_dict"])
    else:
        incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Strict checkpoint load failed: missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
            )
    validate_finite_model(model, "after checkpoint load")
    if optimizer is not None:
        state = payload.get("optimizer_state_dict")
        if state is None:
            raise RuntimeError("Checkpoint is missing optimizer state.")
        optimizer.load_state_dict(state)
    if scheduler is not None:
        state = payload.get("scheduler_state_dict")
        if state is None:
            raise RuntimeError("Checkpoint is missing scheduler state.")
        scheduler.load_state_dict(state)
    if scaler is not None:
        state = payload.get("scaler_state_dict")
        if state is None:
            raise RuntimeError("Checkpoint is missing GradScaler state.")
        scaler.load_state_dict(state)
    if restore_rng:
        restore_rng_state(payload.get("rng_state"))
    return payload
