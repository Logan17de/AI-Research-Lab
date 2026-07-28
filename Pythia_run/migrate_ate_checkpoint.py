from __future__ import annotations

import argparse
from pathlib import Path

import torch

from checkpointing import validate_tokenizer_metadata
from config import config_from_dict
from model import build_tokenizer
from ate_model import LEGACY_ATE_ARCHITECTURE_VERSION, PythiaATEModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate a legacy pythia_ate_v1 checkpoint to staged v2.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    metadata = payload.get("architecture_metadata")
    if not isinstance(metadata, dict) or metadata.get("architecture_version") != LEGACY_ATE_ARCHITECTURE_VERSION:
        raise RuntimeError(f"Input is not a legacy ATE v1 checkpoint: {metadata}")
    config = config_from_dict(payload["config"])
    tokenizer = build_tokenizer(config.tokenizer_name)
    validate_tokenizer_metadata(payload, tokenizer)
    model = PythiaATEModel.from_pretrained(
        config.model_name,
        revision=config.revision,
        vocab_size=len(tokenizer),
        control_token_ids=tuple(
            tokenizer.convert_tokens_to_ids(token) for token in ("<USER>", "<ASSISTANT>", "<SYSTEM>")
        ),
        architecture_metadata=metadata,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.migrate_legacy_control_rows()
    payload["model_state_dict"] = model.state_dict()
    payload["architecture_metadata"] = model.ate_metadata()
    payload["checkpoint_kind"] = "inference"
    for key in ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "rng_state"):
        payload.pop(key, None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    print(f"migrated ATE checkpoint | source={source} | output={destination}")
    print(f"architecture={model.ate_metadata()['architecture_version']} stages={len(model.expansion_stages)}")


if __name__ == "__main__":
    main()
