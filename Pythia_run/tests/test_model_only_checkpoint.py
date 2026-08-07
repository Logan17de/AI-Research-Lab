from __future__ import annotations

import sys

import pytest
import torch

from checkpointing import checkpoint_paths, load_checkpoint, newest_resume_checkpoint, save_checkpoint
from config import parse_args


def test_no_optimizer_cli_aliases(monkeypatch) -> None:
    for flag in ("--no_optimizer", "--no-optimizer"):
        monkeypatch.setattr(sys, "argv", ["train.py", flag])
        assert parse_args().no_optimizer


def test_model_only_checkpoint_loads_for_inference_but_not_resume(tmp_path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_checkpoint(
        run_dir=str(tmp_path),
        step=4,
        epoch=2,
        model=model,
        optimizer=None,
        scheduler=None,
        config_dict={"text_data_format": "qa_sft"},
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["checkpoint_kind"] == "inference"
    assert "model_state_dict" in payload
    for key in ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "rng_state"):
        assert key not in payload
    checkpoint_dir, best_path = checkpoint_paths(str(tmp_path))
    assert not (checkpoint_dir / "latest.pt").exists()
    assert not best_path.exists()

    restored = torch.nn.Linear(3, 2)
    load_checkpoint(str(path), restored)
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected)

    with pytest.raises(RuntimeError, match="missing optimizer state"):
        load_checkpoint(str(path), restored, optimizer=optimizer)


def test_best_checkpoint_replaces_latest_and_is_resolvable(tmp_path) -> None:
    model = torch.nn.Linear(3, 2)
    legacy_latest = tmp_path / "checkpoints" / "latest.pt"
    legacy_latest.parent.mkdir(parents=True)
    legacy_latest.write_bytes(b"obsolete")

    best = save_checkpoint(
        run_dir=str(tmp_path),
        step=10,
        epoch=3,
        model=model,
        optimizer=None,
        scheduler=None,
        config_dict={"text_data_format": "qa_sft"},
        training_state={"best_eval_loss": 0.25, "best_eval_ppl": 1.284025},
        save_step=False,
        save_best=True,
    )
    assert best.name == "best.pt"
    assert not legacy_latest.exists()
    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert payload["checkpoint_role"] == "best"
    assert payload["training_state"]["best_eval_loss"] == 0.25
    assert newest_resume_checkpoint(str(tmp_path)) == best
