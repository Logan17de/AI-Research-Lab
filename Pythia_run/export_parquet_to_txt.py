from __future__ import annotations

import argparse
import os
from pathlib import Path

from data import chat_parquet_examples, parquet_row_count


SEPARATOR = "=" * 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine chat Parquet splits into one readable text file."
    )
    parser.add_argument("--input-dir", default="parquet")
    parser.add_argument("--output", default="parquet/all_conversations.txt")
    parser.add_argument("--pattern", default="*.parquet")
    parser.add_argument("--skip-invalid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    sources = sorted(
        input_dir.glob(args.pattern), key=lambda path: (path.name.startswith("test"), path.name)
    )
    if not sources:
        raise FileNotFoundError(f"No Parquet files found under {input_dir.resolve()}.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path.resolve()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    total_expected = sum(parquet_row_count(source) for source in sources)
    total_written = 0
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                f"PARQUET CHAT DATASET\nSOURCE CONVERSATIONS: {total_expected}\n"
                f"SOURCE FILES: {', '.join(source.name for source in sources)}\n\n"
            )
            for source in sources:
                split = source.name.split("-", 1)[0].upper()
                split_count = parquet_row_count(source)
                handle.write(f"{SEPARATOR}\nSPLIT: {split} | SOURCE CONVERSATIONS: {split_count}\n{SEPARATOR}\n\n")
                for split_index, messages in enumerate(
                    chat_parquet_examples(source, skip_invalid=args.skip_invalid), start=1
                ):
                    total_written += 1
                    handle.write(
                        f"{SEPARATOR}\n"
                        f"CONVERSATION {total_written} | SPLIT {split} | VALID SPLIT INDEX {split_index}\n"
                        f"{SEPARATOR}\n"
                    )
                    for message in messages:
                        handle.write(f"[{message['role'].upper()}]\n{message['content']}\n\n")
            skipped = total_expected - total_written
            handle.write(
                f"{SEPARATOR}\nEXPORT SUMMARY | WRITTEN: {total_written} | SKIPPED INVALID: {skipped}\n"
                f"{SEPARATOR}\n"
            )
        if total_written != total_expected and not args.skip_invalid:
            raise RuntimeError(
                f"Expected {total_expected} conversations but wrote {total_written}."
            )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(
        f"Wrote {total_written} conversations (skipped {total_expected - total_written}) "
        f"to {output_path.resolve()} "
        f"({output_path.stat().st_size / 2**20:.2f} MiB)."
    )


if __name__ == "__main__":
    main()
