from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc
import random
import time

import numpy as np
import torch

from checkpointing import load_checkpoint_payload, newest_resume_checkpoint
from config import TrainConfig, config_from_dict
from data import render_chat_prompt, without_system_messages
from model import build_tokenizer
from train import build_model_and_tokenizer, resolve_device


@dataclass
class LoadedExperiment:
    model: object
    tokenizer: object
    config: TrainConfig
    payload: dict
    checkpoint_path: str
    device: torch.device


def resolve_checkpoint(run_dir_or_checkpoint: str) -> Path:
    path = Path(run_dir_or_checkpoint).expanduser()
    if path.is_dir():
        best_path = path / "checkpoints" / "best.pt"
        path = best_path if best_path.exists() else newest_resume_checkpoint(str(path))
        if path is None:
            raise FileNotFoundError(f"No best.pt or step_*.pt checkpoint found under {run_dir_or_checkpoint}")
    if not path.exists():
        raise FileNotFoundError(path)
    return path.resolve()


def load_inference_payload(checkpoint_path: Path) -> dict:
    size_gib = checkpoint_path.stat().st_size / (1024 ** 3)
    print(
        f"reading checkpoint | path={checkpoint_path} | size={size_gib:.2f} GiB | mmap=attempting",
        flush=True,
    )
    started = time.time()
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        load_mode = "memory-mapped"
    except (TypeError, ValueError, RuntimeError) as exc:
        print(f"memory-mapped load unavailable ({exc}); falling back to normal CPU load", flush=True)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        load_mode = "normal"
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError(f"Checkpoint {checkpoint_path} is not a GPT_MOD training checkpoint.")
    removed = []
    for key in ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "rng_state"):
        if payload.pop(key, None) is not None:
            removed.append(key)
    gc.collect()
    print(
        f"checkpoint index ready | mode={load_mode} | elapsed={time.time() - started:.1f}s | "
        f"discarded={','.join(removed) or 'none'}",
        flush=True,
    )
    return payload


def load_experiment(
    run_dir_or_checkpoint: str,
    device_name: str,
    *,
    disable_emb_mod: bool = False,
    disable_out_mod: bool = False,
    disable_attn_mod: bool = False,
    disable_ffn_mod: bool = False,
    disable_all_mods: bool = False,
    payload: dict | None = None,
) -> LoadedExperiment:
    checkpoint_path = resolve_checkpoint(run_dir_or_checkpoint)
    if payload is None:
        payload = load_inference_payload(checkpoint_path)
    config = config_from_dict(payload["config"])
    if config.experiment_type == "ate":
        metadata = payload.get("architecture_metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("ATE inference checkpoint is missing architecture metadata.")
        config._ate_resume_metadata = metadata
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(device_name)
    print(
        f"constructing model | model={config.model_name} | experiment={config.experiment_type} | device={device}",
        flush=True,
    )
    model, tokenizer = build_model_and_tokenizer(config, device)
    print("applying checkpoint model weights", flush=True)
    load_checkpoint_payload(
        payload,
        model,
        expected_config=payload["config"],
        tokenizer=tokenizer,
        restore_rng=False,
        source=str(checkpoint_path),
    )
    if hasattr(model, "set_mod_ablation"):
        model.set_mod_ablation(
            disable_emb=disable_emb_mod,
            disable_out=disable_out_mod,
            disable_attn=disable_attn_mod,
            disable_ffn=disable_ffn_mod,
            disable_all=disable_all_mods,
        )
    model.eval()
    metadata = {
        key: value
        for key, value in payload.items()
        if key != "model_state_dict"
    }
    payload.pop("model_state_dict", None)
    gc.collect()
    print(f"model ready | model={config.model_name} | device={device}", flush=True)
    return LoadedExperiment(model, tokenizer, config, metadata, str(checkpoint_path), device)


def greedy_generate_messages(
    loaded: LoadedExperiment,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    *,
    data_format: str | None = None,
) -> tuple[str, str, int]:
    active_format = data_format or loaded.config.text_data_format
    if active_format == "qa_sft":
        user_messages = [
            str(message.get("content", "")).strip()
            for message in messages
            if str(message.get("role", "")).strip().lower() == "user"
        ]
        if not user_messages or not user_messages[-1]:
            raise ValueError("QA generation requires a non-empty user question.")
        prompt = f"Q: {user_messages[-1]}\nA:"
    else:
        if getattr(loaded.config, "ignore_system_prompt", False):
            messages = without_system_messages(messages)
        prompt = render_chat_prompt(messages, open_assistant=True, eos_token=loaded.tokenizer.eos_token)
    prompt_ids = loaded.tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Generation prompt encoded to zero tokens.")
    max_context = loaded.config.seq_len
    generated: list[int] = []
    stop_reason = "max_new_tokens"
    cache = None
    attention_mask = None
    with torch.inference_mode():
        for index in range(max_new_tokens):
            if cache is None:
                context = prompt_ids[-max_context:]
                input_ids = torch.tensor([context], dtype=torch.long, device=loaded.device)
                attention_mask = torch.ones_like(input_ids)
            else:
                input_ids = torch.tensor([[generated[-1]]], dtype=torch.long, device=loaded.device)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=loaded.device)], dim=1
                )
            output = loaded.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            cache = output["past_key_values"]
            next_scores = output["logits"][0, -1].float().clone()
            if next_scores.numel() > len(loaded.tokenizer):
                next_scores[len(loaded.tokenizer) :] = float("-inf")
            next_id = int(next_scores.argmax().item())
            if next_id == loaded.tokenizer.eos_token_id:
                stop_reason = "eos"
                break
            generated.append(next_id)
            if len(prompt_ids) + len(generated) >= max_context:
                stop_reason = "context_limit"
                break
    text = loaded.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, stop_reason, len(generated)
