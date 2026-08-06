from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate controlled word-token arithmetic train/validation splits"
    )
    parser.add_argument("--operation", choices=("addition", "multiplication"), required=True)
    parser.add_argument("--min-operand", type=int, default=0)
    parser.add_argument("--max-operand", type=int, default=19)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def operation_spec(name: str) -> tuple[str, Callable[[int, int], int]]:
    if name == "addition":
        return "+", lambda left, right: left + right
    if name == "multiplication":
        return "*", lambda left, right: left * right
    raise ValueError(f"Unknown operation: {name}")


def build_groups(
    operation: str,
    minimum: int,
    maximum: int,
) -> list[list[dict[str, str]]]:
    if minimum > maximum:
        raise ValueError("min-operand cannot exceed max-operand")
    symbol, function = operation_spec(operation)
    groups: list[list[dict[str, str]]] = []
    for left in range(minimum, maximum + 1):
        for right in range(left, maximum + 1):
            orientations = [(left, right)] if left == right else [(left, right), (right, left)]
            groups.append(
                [
                    {
                        "prompt": f"what is {a} {symbol} {b} ?",
                        "answer": str(function(a, b)),
                    }
                    for a, b in orientations
                ]
            )
    return groups


def split_groups(
    groups: list[list[dict[str, str]]],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation-ratio must be between zero and one")
    if len(groups) < 2:
        raise ValueError("At least two commutative groups are required")

    answer_counts = Counter(row["answer"] for group in groups for row in group)
    operand_counts: Counter[str] = Counter()
    for group in groups:
        for row in group:
            tokens = row["prompt"].split()
            operand_counts[tokens[2]] += 1
            operand_counts[tokens[4]] += 1

    target_examples = round(sum(len(group) for group in groups) * validation_ratio)
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    validation_groups: list[list[dict[str, str]]] = []
    validation_examples = 0

    for group in shuffled:
        if validation_examples >= target_examples:
            break
        group_answers = Counter(row["answer"] for row in group)
        group_operands: Counter[str] = Counter()
        for row in group:
            tokens = row["prompt"].split()
            group_operands[tokens[2]] += 1
            group_operands[tokens[4]] += 1

        if any(answer_counts[token] - count < 1 for token, count in group_answers.items()):
            continue
        if any(operand_counts[token] - count < 1 for token, count in group_operands.items()):
            continue

        validation_groups.append(group)
        validation_examples += len(group)
        answer_counts.subtract(group_answers)
        operand_counts.subtract(group_operands)

    validation_ids = {id(group) for group in validation_groups}
    train = [row for group in groups if id(group) not in validation_ids for row in group]
    validation = [row for group in validation_groups for row in group]
    if not validation:
        raise RuntimeError("Could not create a validation split with vocabulary coverage")
    return train, validation


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    groups = build_groups(args.operation, args.min_operand, args.max_operand)
    train, validation = split_groups(
        groups,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    output = Path(args.output_dir)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "validation.jsonl", validation)
    train_answers = {row["answer"] for row in train}
    metadata = {
        "operation": args.operation,
        "min_operand": args.min_operand,
        "max_operand": args.max_operand,
        "validation_ratio_requested": args.validation_ratio,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "validation_ratio_actual": len(validation) / (len(train) + len(validation)),
        "commuted_pairs_kept_together": True,
        "validation_answers_present_in_train": all(
            row["answer"] in train_answers for row in validation
        ),
        "seed": args.seed,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
