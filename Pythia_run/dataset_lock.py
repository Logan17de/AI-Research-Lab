from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data import CHAT_DATA_FORMAT_VERSION


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_locked_dataset(manifest_path: str, data_paths: list[str], tokenizer) -> dict:
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != "pythia_ultrachat_filtered_v1":
        raise RuntimeError(f"Unsupported locked dataset manifest: {manifest.get('format_version')!r}")
    if manifest.get("chat_format_version") != CHAT_DATA_FORMAT_VERSION:
        raise RuntimeError(
            f"Dataset chat serialization mismatch: saved={manifest.get('chat_format_version')!r} "
            f"current={CHAT_DATA_FORMAT_VERSION!r}"
        )
    vocabulary = tokenizer.get_vocab()
    vocab_sha256 = hashlib.sha256(
        json.dumps(sorted(vocabulary.items()), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    saved_tokenizer = manifest.get("tokenizer", {})
    if saved_tokenizer.get("vocab_size") != len(tokenizer) or saved_tokenizer.get("vocab_sha256") != vocab_sha256:
        raise RuntimeError("Dataset lock tokenizer vocabulary does not match the active training tokenizer.")
    if saved_tokenizer.get("eos_token_id") != tokenizer.eos_token_id:
        raise RuntimeError("Dataset lock EOS token does not match the active training tokenizer.")
    current_control_ids = {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in ("<USER>", "<ASSISTANT>", "<SYSTEM>")
    }
    if saved_tokenizer.get("control_token_ids") != current_control_ids:
        raise RuntimeError("Dataset lock control-token IDs do not match the active training tokenizer.")
    files = {item["file"]: item for item in manifest.get("splits", {}).values()}
    validated = []
    for raw_path in data_paths:
        data_path = Path(raw_path).expanduser().resolve()
        entry = files.get(data_path.name)
        if entry is None:
            raise RuntimeError(f"Dataset file {data_path.name!r} is not present in {path}.")
        actual_sha = _sha256_file(data_path)
        if actual_sha != entry.get("sha256"):
            raise RuntimeError(
                f"Dataset lock hash mismatch for {data_path}: expected={entry.get('sha256')} actual={actual_sha}"
            )
        validated.append(data_path.name)
    print(
        f"dataset lock PASS | manifest={path} | files={validated} | "
        f"tokenizer_vocab_sha256={vocab_sha256}",
        flush=True,
    )
    return manifest
