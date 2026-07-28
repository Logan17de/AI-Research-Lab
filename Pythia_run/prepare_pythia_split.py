from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

from data import CHAT_DATA_FORMAT_VERSION, iter_training_examples
from validate_dataset import normalized_example, question_answer_texts


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic question/answer leakage-free split.")
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-format", choices=["qa_sft", "chat_jsonl"], default="chat_jsonl")
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def render(example: Any, data_format: str) -> str:
    if data_format == "qa_sft":
        question, answer = example
        return f"Q: {question}\nA: {answer}\n\n"
    return json.dumps(
        {"format_version": CHAT_DATA_FORMAT_VERSION, "messages": example},
        ensure_ascii=False,
    ) + "\n"


def grouped_split(examples: list[Any], eval_fraction: float, seed: int) -> tuple[list[Any], list[Any], int]:
    union_find = UnionFind(len(examples))
    owners: dict[tuple[str, str], int] = {}
    for index, example in enumerate(examples):
        questions, answers = question_answer_texts(example)
        for kind, texts in (("q", questions), ("a", answers)):
            for text in texts:
                key = (kind, text)
                if key in owners:
                    union_find.union(index, owners[key])
                else:
                    owners[key] = index
    components: dict[int, list[int]] = {}
    for index in range(len(examples)):
        components.setdefault(union_find.find(index), []).append(index)
    groups = list(components.values())
    random.Random(seed).shuffle(groups)
    target_eval = max(1, round(len(examples) * eval_fraction))
    eval_indices: set[int] = set()
    for group in groups:
        if len(eval_indices) >= target_eval and eval_indices:
            break
        if len(eval_indices) + len(group) >= len(examples):
            continue
        eval_indices.update(group)
    if not eval_indices:
        eval_indices.update(groups[0])
    train = [example for index, example in enumerate(examples) if index not in eval_indices]
    validation = [example for index, example in enumerate(examples) if index in eval_indices]
    return train, validation, len(groups)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.eval_fraction < 1.0:
        raise ValueError("--eval-fraction must be between 0 and 1")
    with open(args.source_path, "r", encoding="utf-8") as handle:
        raw_examples = list(iter_training_examples(handle, args.data_format))
    unique: list[Any] = []
    seen_samples: set[str] = set()
    for example in raw_examples:
        key = normalized_example(example)
        if key not in seen_samples:
            seen_samples.add(key)
            unique.append(example)
    if len(unique) < 2:
        raise ValueError("At least two unique examples are required.")

    train, validation, group_count = grouped_split(unique, args.eval_fraction, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "jsonl" if args.data_format == "chat_jsonl" else "txt"
    train_path = output_dir / f"train.{suffix}"
    validation_path = output_dir / f"validation.{suffix}"
    train_path.write_text("".join(render(example, args.data_format) for example in train), encoding="utf-8")
    validation_path.write_text("".join(render(example, args.data_format) for example in validation), encoding="utf-8")
    metadata = {
        "source": str(Path(args.source_path).resolve()),
        "data_format": args.data_format,
        "seed": args.seed,
        "eval_fraction": args.eval_fraction,
        "raw_examples": len(raw_examples),
        "unique_examples": len(unique),
        "duplicate_examples_removed": len(raw_examples) - len(unique),
        "connected_groups": group_count,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
