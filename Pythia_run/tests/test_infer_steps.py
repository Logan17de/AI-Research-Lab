from __future__ import annotations

import pytest
from types import SimpleNamespace

from data import isolated_sample_steps_per_epoch, save_steps_per_epoch_cache
from infer_steps import manifest_example_count
from train import reuse_incremental_steps_cache


def test_isolated_sample_step_count_includes_accumulation() -> None:
    assert isolated_sample_steps_per_epoch(100, batch_size=10, grad_accum_steps=2, num_workers=0) == 5
    assert isolated_sample_steps_per_epoch(101, batch_size=10, grad_accum_steps=2, num_workers=2) == 6


def test_manifest_example_count_finds_requested_split() -> None:
    manifest = {
        "splits": {
            "train": {"file": "train.jsonl", "examples": 48_000, "tokens": 12_345_678},
            "validation": {"file": "validation.jsonl", "examples": 1_000, "tokens": 250_000},
        }
    }
    assert manifest_example_count(manifest, "/data/train.jsonl") == (48_000, 12_345_678)
    with pytest.raises(RuntimeError, match="absent"):
        manifest_example_count(manifest, "/data/test.jsonl")


def test_incremental_run_reuses_matching_source_step_cache(tmp_path) -> None:
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"messages":[]}\n', encoding="utf-8")
    source_run = tmp_path / "source"
    source_checkpoint = source_run / "checkpoints" / "best.pt"
    source_checkpoint.parent.mkdir(parents=True)
    source_checkpoint.touch()

    def config(run_dir):
        return SimpleNamespace(
            text_data_path=str(data_path),
            text_data_format="chat_jsonl",
            run_dir=str(run_dir),
            seq_len=1024,
            batch_size=8,
            grad_accum_steps=4,
            num_workers=0,
            ignore_system_prompt=False,
        )

    tokenizer = SimpleNamespace(name_or_path="fake-tokenizer")
    save_steps_per_epoch_cache(config(source_run), tokenizer, 123)
    target = config(tmp_path / "target")
    target._ate_incremental_source_path = str(source_checkpoint)
    assert reuse_incremental_steps_cache(target, tokenizer)
    payload = __import__("json").loads(
        (tmp_path / "target" / "steps_per_epoch_cache.json").read_text(encoding="utf-8")
    )
    assert payload["steps_per_epoch"] == 123

    target.batch_size = 16
    assert not reuse_incremental_steps_cache(target, tokenizer)
