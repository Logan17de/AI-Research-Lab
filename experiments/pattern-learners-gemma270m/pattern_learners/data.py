from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class PromptAnswerDataset(Dataset[dict[str, torch.Tensor]]):
    """JSONL causal-LM dataset with answer-only loss."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        *,
        max_length: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rows: list[dict[str, str]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row.get("prompt"), str) or not isinstance(row.get("answer"), str):
                    raise ValueError(f"Invalid row at line {line_number}: prompt and answer are required")
                self.rows.append({"prompt": row["prompt"], "answer": row["answer"]})
        if not self.rows:
            raise ValueError(f"No training examples found in {path}")

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
    """Create deterministic arithmetic data for the first learner experiment."""
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
