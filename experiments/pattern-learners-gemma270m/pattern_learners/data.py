from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset


_Q_PATTERN = re.compile(r"^\s*(?:\d+\.\s*)?Q:\s*(.+?)\s*$")
_A_PATTERN = re.compile(r"^\s*A:\s*(.+?)\s*$")


def load_prompt_answer_rows(path: str | Path) -> list[dict[str, str]]:
    """Load either JSONL rows or the repository's numbered Q/A text format."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    if source.suffix.lower() == ".jsonl":
        rows: list[dict[str, str]] = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = row.get("prompt")
                answer = row.get("answer")
                if not isinstance(prompt, str) or not isinstance(answer, str):
                    raise ValueError(
                        f"Invalid JSONL row at {source}:{line_number}; "
                        "prompt and answer strings are required"
                    )
                rows.append({"prompt": prompt.strip(), "answer": answer.strip()})
    else:
        rows = parse_numbered_qa_text(source.read_text(encoding="utf-8"), source=str(source))

    if not rows:
        raise ValueError(f"No training examples found in {source}")
    return rows


def parse_numbered_qa_text(text: str, *, source: str = "<text>") -> list[dict[str, str]]:
    """Parse blocks such as `1. Q: ...` followed by `A: ...`."""
    rows: list[dict[str, str]] = []
    pending_prompt: str | None = None

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue

        question_match = _Q_PATTERN.match(line)
        if question_match:
            if pending_prompt is not None:
                raise ValueError(
                    f"Question at {source}:{line_number} appeared before the previous answer"
                )
            pending_prompt = question_match.group(1).strip()
            continue

        answer_match = _A_PATTERN.match(line)
        if answer_match:
            if pending_prompt is None:
                raise ValueError(f"Answer at {source}:{line_number} has no question")
            rows.append(
                {
                    "prompt": pending_prompt,
                    "answer": answer_match.group(1).strip(),
                }
            )
            pending_prompt = None
            continue

        raise ValueError(
            f"Unrecognized non-empty line at {source}:{line_number}: {line.strip()!r}"
        )

    if pending_prompt is not None:
        raise ValueError(f"Final question in {source} has no answer")
    return rows


def split_prompt_answer_rows(
    rows: Sequence[dict[str, str]],
    *,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Create a deterministic shuffled train/validation split."""
    if len(rows) < 2:
        raise ValueError("At least two examples are required for a train/validation split")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, round(len(rows) * validation_ratio))
    validation_count = min(validation_count, len(rows) - 1)
    validation_indices = set(indices[:validation_count])

    train_rows = [dict(row) for index, row in enumerate(rows) if index not in validation_indices]
    validation_rows = [dict(row) for index, row in enumerate(rows) if index in validation_indices]
    return train_rows, validation_rows


class PromptAnswerDataset(Dataset[dict[str, torch.Tensor]]):
    """Causal-LM dataset with answer-only loss."""

    def __init__(
        self,
        path: str | Path | None,
        tokenizer: Any,
        *,
        max_length: int = 256,
        rows: Sequence[dict[str, str]] | None = None,
    ) -> None:
        if (path is None) == (rows is None):
            raise ValueError("Provide exactly one of path or rows")
        self.tokenizer = tokenizer
        self.max_length = max_length
        loaded_rows = load_prompt_answer_rows(path) if path is not None else list(rows or [])
        self.rows = [
            {"prompt": row["prompt"].strip(), "answer": row["answer"].strip()}
            for row in loaded_rows
        ]
        if not self.rows:
            raise ValueError("No training examples were provided")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        prefix = f"Question: {row['prompt']}\nAnswer:"
        eos = self.tokenizer.eos_token or ""
        full_text = f"{prefix} {row['answer']}{eos}"

        full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        prefix_tokens = self.tokenizer(
            f"{prefix} ",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

        input_ids = torch.tensor(full["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(full["attention_mask"], dtype=torch.long)
        labels = input_ids.clone()
        prefix_length = min(len(prefix_tokens["input_ids"]), labels.numel())
        labels[:prefix_length] = -100
        if torch.all(labels == -100):
            raise ValueError(
                "An example was truncated before its answer. Increase max_length or shorten the prompt."
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class CausalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_length = max(row["input_ids"].numel() for row in rows)
        batch_size = len(rows)
        input_ids = torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long)

        for index, row in enumerate(rows):
            length = row["input_ids"].numel()
            input_ids[index, :length] = row["input_ids"]
            attention_mask[index, :length] = row["attention_mask"]
            labels[index, :length] = row["labels"]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def write_arithmetic_dataset(
    output_directory: str | Path,
    *,
    seed: int = 42,
    train_examples: int = 4000,
    validation_examples: int = 500,
    operation: str = "addition",
    minimum: int = 0,
    maximum: int = 99,
) -> None:
    """Create deterministic arithmetic data for optional larger experiments."""
    if operation not in {"addition", "multiplication", "star"}:
        raise ValueError("operation must be addition, multiplication, or star")
    if minimum > maximum:
        raise ValueError("minimum cannot exceed maximum")

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    def make_row() -> dict[str, str]:
        left = rng.randint(minimum, maximum)
        right = rng.randint(minimum, maximum)
        if operation == "addition":
            prompt = f"What is {left} + {right}?"
            answer = str(left + right)
        elif operation == "multiplication":
            prompt = f"What is {left} × {right}?"
            answer = str(left * right)
        else:
            prompt = f"In this task, a ★ b means a × b + 2. What is {left} ★ {right}?"
            answer = str(left * right + 2)
        return {"prompt": prompt, "answer": answer}

    for split, count in (("train", train_examples), ("validation", validation_examples)):
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for _ in range(count):
                handle.write(json.dumps(make_row(), ensure_ascii=False) + "\n")
