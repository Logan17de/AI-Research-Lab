from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from config import str2bool
from data import (
    infer_steps_per_epoch,
    isolated_sample_steps_per_epoch,
    resolve_text_data_format,
    save_steps_per_epoch_cache,
)
from dataset_lock import validate_locked_dataset
from model import build_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer optimizer steps per epoch without loading a language model or starting training."
    )
    parser.add_argument("--text_data_path", "--dataset-path", required=True)
    parser.add_argument(
        "--text_data_format",
        choices=["auto", "plain", "qa_sft", "chat_jsonl", "chat_parquet"],
        default="auto",
    )
    parser.add_argument("--tokenizer_name", "--tokenizer-name", default="EleutherAI/pythia-1.4b")
    parser.add_argument(
        "--run_dir", "--run-dir", required=True,
        help="Use the exact training run directory so train.py can reuse the generated cache.",
    )
    parser.add_argument("--batch_size", "--batch-size", type=int, required=True)
    parser.add_argument("--grad_accum_steps", "--grad-accum-steps", type=int, required=True)
    parser.add_argument("--seq_len", "--seq-len", type=int, required=True)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--ignore_system_prompt", "--ignore-system-prompt",
        nargs="?", const=True, type=str2bool, default=False,
    )
    parser.add_argument(
        "--dataset_manifest", "--dataset-manifest",
        default=None,
        help="Use the locked UltraChat manifest count for instant inference after hash validation.",
    )
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.grad_accum_steps <= 0 or args.seq_len <= 0:
        parser.error("batch size, gradient accumulation, and sequence length must be positive")
    if args.num_workers < 0 or args.epochs <= 0:
        parser.error("num workers must be non-negative and epochs must be positive")
    return args


def manifest_example_count(manifest: dict, data_path: str) -> tuple[int, int | None]:
    filename = Path(data_path).name
    for entry in manifest.get("splits", {}).values():
        if entry.get("file") == filename:
            count = int(entry.get("examples", 0))
            if count <= 0:
                raise RuntimeError(f"Manifest entry for {filename!r} has no positive example count.")
            tokens = entry.get("tokens")
            return count, int(tokens) if tokens is not None else None
    raise RuntimeError(f"Dataset file {filename!r} is absent from the manifest splits.")


def main() -> None:
    args = parse_args()
    data_path = Path(args.text_data_path).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    config = SimpleNamespace(
        text_data_path=str(data_path),
        text_data_format=args.text_data_format,
        run_dir=args.run_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        ignore_system_prompt=args.ignore_system_prompt,
    )
    config.text_data_format = resolve_text_data_format(config)
    cache_path = Path(args.run_dir) / "steps_per_epoch_cache.json"
    if args.force_recompute and cache_path.exists():
        cache_path.unlink()

    print(f"Loading tokenizer only: {args.tokenizer_name}", flush=True)
    tokenizer = build_tokenizer(args.tokenizer_name)
    sample_count: int | None = None
    actual_tokens: int | None = None

    if args.dataset_manifest:
        manifest = validate_locked_dataset(args.dataset_manifest, [str(data_path)], tokenizer)
        if config.text_data_format != "chat_jsonl":
            raise ValueError("Manifest-count inference currently requires --text_data_format chat_jsonl.")
        sample_count, actual_tokens = manifest_example_count(manifest, str(data_path))
        steps_per_epoch = isolated_sample_steps_per_epoch(
            sample_count,
            args.batch_size,
            args.grad_accum_steps,
            args.num_workers,
        )
        save_steps_per_epoch_cache(config, tokenizer, steps_per_epoch)
        inference_method = "validated_manifest_count"
    else:
        steps_per_epoch = infer_steps_per_epoch(config, tokenizer)
        inference_method = "dataset_scan_or_parquet_metadata"

    effective_batch = args.batch_size * args.grad_accum_steps
    total_steps = steps_per_epoch * args.epochs
    max_tokens_per_step = effective_batch * args.seq_len
    report = {
        "dataset": str(data_path.resolve()),
        "format": config.text_data_format,
        "method": inference_method,
        "examples": sample_count,
        "actual_dataset_tokens": actual_tokens,
        "micro_batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": effective_batch,
        "seq_len": args.seq_len,
        "num_workers": args.num_workers,
        "steps_per_epoch": steps_per_epoch,
        "epochs": args.epochs,
        "total_optimizer_steps": total_steps,
        "max_tokens_per_optimizer_step": max_tokens_per_step,
        "max_padded_tokens_per_epoch": steps_per_epoch * max_tokens_per_step,
        "cache_path": str(cache_path.resolve()),
    }
    print("\nSTEP INFERENCE RESULT")
    for key, value in report.items():
        if isinstance(value, int):
            value = f"{value:,}"
        print(f"{key}: {value}")
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"report_json: {output_path.resolve()}")


if __name__ == "__main__":
    main()
