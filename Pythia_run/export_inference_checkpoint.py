from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import torch


TRAINING_ONLY_KEYS = (
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scaler_state_dict",
    "rng_state",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strip optimizer state from a resumable checkpoint for faster chat loading."
    )
    parser.add_argument("checkpoint", help="Input best.pt or numbered step checkpoint.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path; defaults to inference.pt beside the input checkpoint.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = (
        Path(args.output).expanduser().resolve()
        if args.output
        else source.with_name("inference.pt")
    )
    if source == destination:
        raise ValueError("Inference checkpoint output must differ from the training checkpoint.")
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"{destination} already exists; pass --overwrite to replace it.")

    source_gib = source.stat().st_size / (1024 ** 3)
    print(f"reading {source} ({source_gib:.2f} GiB) with memory mapping", flush=True)
    started = time.time()
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, ValueError, RuntimeError) as exc:
        print(f"memory mapping unavailable ({exc}); using normal CPU loading", flush=True)
        payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload or "config" not in payload:
        raise RuntimeError(f"{source} is not a supported Pythia_run checkpoint.")

    removed = [key for key in TRAINING_ONLY_KEYS if payload.pop(key, None) is not None]
    payload["checkpoint_kind"] = "inference"
    payload["source_checkpoint"] = str(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    print(f"writing model-only checkpoint to {destination}", flush=True)
    torch.save(payload, temporary)
    os.replace(temporary, destination)

    output_gib = destination.stat().st_size / (1024 ** 3)
    print(
        f"export complete | removed={','.join(removed) or 'none'} | "
        f"size={output_gib:.2f} GiB | elapsed={time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
