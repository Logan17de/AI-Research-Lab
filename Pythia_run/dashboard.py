from __future__ import annotations

import sys
from typing import Any

import torch


_live_line_width = 0


def _gpu_string(device: str) -> str:
    if device != "cuda" or not torch.cuda.is_available():
        return "cpu"
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    return f"{allocated:.1f}/{reserved:.1f}G"


def format_dashboard(metrics: dict[str, Any]) -> str:
    fields = [
            f"step {metrics['step']}",
            f"ep {metrics['epoch']}/{metrics['total_epochs']}",
            f"{metrics['progress_pct']:.1f}%",
            f"train target loss {metrics['loss']:.4f}",
            f"train target ppl {metrics['ppl']:.2f}",
            f"eval assistant ppl {metrics['eval_assistant_ppl']:.2f}",
            f"eval combined ppl {metrics['eval_combined_ppl']:.2f}",
            f"lr {metrics['lr']:.2e}",
            f"mod {metrics['modifier_lr']:.2e}",
            f"plastic {metrics['plastic_lr']:.2e}",
            f"norm {metrics['plastic_layernorm_lr']:.2e}",
            f"lora {metrics['lora_lr']:.2e}",
            f"tok/s {metrics['tokens_per_sec']:.0f}",
            f"target/s {metrics['target_tokens_per_sec']:.0f}",
            f"targets {metrics.get('target_tokens_processed', metrics.get('target_tokens', 0))}",
            f"gpu {metrics['gpu']}",
    ]
    ratio = metrics.get("output_mod_logit_ratio")
    if isinstance(ratio, (int, float)) and ratio == ratio:
        fields.append(f"out/base {ratio:.3f}")
    drift = metrics.get("plastic_drift")
    if isinstance(drift, (int, float)) and drift == drift:
        fields.append(f"drift {drift:.2e}")
    return " | ".join(fields)


def render_dashboard(metrics: dict[str, Any], *, live: bool = False) -> None:
    global _live_line_width
    line = format_dashboard(metrics)
    padding = " " * max(_live_line_width - len(line), 0)
    if live:
        sys.stdout.write("\r" + line + padding)
        _live_line_width = len(line)
    else:
        sys.stdout.write("\r" + line + padding + "\n")
        _live_line_width = 0
    sys.stdout.flush()


def clear_live_dashboard() -> None:
    global _live_line_width
    if _live_line_width:
        sys.stdout.write("\r" + (" " * _live_line_width) + "\r")
        sys.stdout.flush()
        _live_line_width = 0


def build_dashboard_metrics(
    *,
    step: int,
    epoch: int,
    total_epochs: int,
    loss: float,
    eval_assistant_loss: float,
    eval_combined_loss: float,
    lr: float,
    modifier_lr: float,
    lora_lr: float,
    plastic_lr: float = 0.0,
    plastic_layernorm_lr: float = 0.0,
    plastic_drift: float = float("nan"),
    tokens_per_sec: float,
    target_tokens_per_sec: float,
    target_tokens: int,
    progress_pct: float,
    device: str,
    samples_processed: int = 0,
    tokens_processed: int = 0,
    target_tokens_processed: int = 0,
    effective_epochs: float = 0.0,
    checkpoint_path: str = "",
) -> dict[str, Any]:
    return {
        "step": step,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "progress_pct": progress_pct,
        "loss": loss,
        "ppl": min(float("inf"), pow(2.718281828459045, min(loss, 20.0))),
        "eval_assistant_ppl": min(float("inf"), pow(2.718281828459045, min(eval_assistant_loss, 20.0))),
        "eval_combined_ppl": min(float("inf"), pow(2.718281828459045, min(eval_combined_loss, 20.0))),
        "lr": lr,
        "modifier_lr": modifier_lr,
        "lora_lr": lora_lr,
        "plastic_lr": plastic_lr,
        "plastic_layernorm_lr": plastic_layernorm_lr,
        "plastic_drift": plastic_drift,
        "tokens_per_sec": tokens_per_sec,
        "target_tokens_per_sec": target_tokens_per_sec,
        "target_tokens": target_tokens,
        "samples_processed": samples_processed,
        "tokens_processed": tokens_processed,
        "target_tokens_processed": target_tokens_processed,
        "effective_epochs": effective_epochs,
        "checkpoint_path": checkpoint_path,
        "gpu": _gpu_string(device),
    }
