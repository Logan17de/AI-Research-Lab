from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .model import LearnerLayout, TinyConfig, TinyPatternLM
from .tokenizer import LiveWordTokenizer


def save_base(
    output_dir: str | Path,
    model: TinyPatternLM,
    tokenizer: LiveWordTokenizer,
    *,
    metadata: dict[str, Any],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output / "tokenizer.json")
    torch.save(model.state_dict(), output / "base_model.pt")
    (output / "config.json").write_text(
        json.dumps(model.config.to_dict(), indent=2), encoding="utf-8"
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_base(path: str | Path) -> tuple[TinyPatternLM, LiveWordTokenizer]:
    root = Path(path)
    config = TinyConfig(**json.loads((root / "config.json").read_text(encoding="utf-8")))
    tokenizer = LiveWordTokenizer.load(root / "tokenizer.json")
    model = TinyPatternLM(config)
    model.load_state_dict(torch.load(root / "base_model.pt", map_location="cpu"))
    return model, tokenizer


def save_run(
    output_dir: str | Path,
    model: TinyPatternLM,
    tokenizer: LiveWordTokenizer,
    *,
    summary: dict[str, Any],
) -> None:
    if model.learner_layout is None:
        raise RuntimeError("Cannot save a learner run without learner layout")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output / "tokenizer.json")
    torch.save(model.state_dict(), output / "model.pt")
    payload = {
        "config": model.config.to_dict(),
        "learner_layout": model.learner_layout.to_dict(),
        "patterns": list(model.patterns.keys()),
        "active_pattern": model.active_pattern,
        "summary": summary,
    }
    (output / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_run(path: str | Path) -> tuple[TinyPatternLM, LiveWordTokenizer, dict[str, Any]]:
    root = Path(path)
    payload = json.loads((root / "run.json").read_text(encoding="utf-8"))
    config = TinyConfig(**payload["config"])
    layout = LearnerLayout(**payload["learner_layout"])
    model = TinyPatternLM(config, layout)
    for name in payload["patterns"]:
        model.add_pattern(name, activate=False)
    model.load_state_dict(torch.load(root / "model.pt", map_location="cpu"))
    model.set_active_pattern(payload.get("active_pattern"))
    tokenizer = LiveWordTokenizer.load(root / "tokenizer.json")
    return model, tokenizer, payload
