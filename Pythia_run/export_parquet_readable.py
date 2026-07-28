from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from data import CHAT_DATA_FORMAT_VERSION, chat_parquet_examples, parquet_row_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export chat Parquet files as readable canonical JSONL shards and text previews."
    )
    parser.add_argument("--input-dir", default="parquet")
    parser.add_argument("--output-dir", default="parquet/readable")
    parser.add_argument("--rows-per-shard", type=int, default=5000)
    parser.add_argument("--preview-rows", type=int, default=3)
    return parser.parse_args()


def preview_text(index: int, messages: list[dict[str, str]]) -> str:
    sections = [f"{'=' * 80}\nCONVERSATION {index}\n{'=' * 80}\n"]
    for message in messages:
        sections.append(f"[{message['role'].upper()}]\n{message['content']}\n\n")
    return "".join(sections)


def export_file(
    source: Path,
    output_dir: Path,
    rows_per_shard: int,
    preview_rows: int,
) -> dict[str, object]:
    split = source.name.split("-", 1)[0]
    row_count = parquet_row_count(source)
    shard_count = math.ceil(row_count / rows_per_shard)
    shard_paths: list[Path] = []
    shard_handle = None
    preview_parts: list[str] = []
    written = 0
    try:
        for index, messages in enumerate(chat_parquet_examples(source)):
            shard_index = index // rows_per_shard
            if index % rows_per_shard == 0:
                if shard_handle is not None:
                    shard_handle.close()
                shard_path = output_dir / (
                    f"{split}-{shard_index:05d}-of-{shard_count:05d}.jsonl"
                )
                shard_paths.append(shard_path)
                shard_handle = shard_path.open("w", encoding="utf-8", newline="\n")
            payload = {
                "format_version": CHAT_DATA_FORMAT_VERSION,
                "messages": messages,
            }
            shard_handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            if index < preview_rows:
                preview_parts.append(preview_text(index + 1, messages))
            written += 1
    finally:
        if shard_handle is not None:
            shard_handle.close()

    if written != row_count:
        raise RuntimeError(f"{source}: metadata reports {row_count} rows but exported {written}.")
    preview_path = output_dir / f"{split}_preview.txt"
    preview_path.write_text("".join(preview_parts), encoding="utf-8")
    return {
        "split": split,
        "source": str(source.resolve()),
        "rows": written,
        "rows_per_shard": rows_per_shard,
        "shards": [str(path) for path in shard_paths],
        "preview": str(preview_path),
    }


def main() -> None:
    args = parse_args()
    if args.rows_per_shard < 1 or args.preview_rows < 0:
        raise ValueError("--rows-per-shard must be positive and --preview-rows cannot be negative.")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    sources = sorted(input_dir.glob("*.parquet"))
    if not sources:
        raise FileNotFoundError(f"No Parquet files found under {input_dir.resolve()}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.jsonl")) + list(output_dir.glob("*_preview.txt"))
    if existing:
        raise FileExistsError(
            f"Output directory already contains generated files: {[str(path) for path in existing[:5]]}"
        )

    reports = [
        export_file(source, output_dir, args.rows_per_shard, args.preview_rows)
        for source in sources
    ]
    manifest = {
        "format_version": CHAT_DATA_FORMAT_VERSION,
        "source_format": "parquet messages column",
        "outputs": reports,
        "total_rows": sum(int(report["rows"]) for report in reports),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
