from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .tokenizer import LiveWordTokenizer


QUESTION_RE = re.compile(r"^\s*(?:\d+\.\s*)?Q:\s*(.*?)\s*$")
ANSWER_RE = re.compile(r"^\s*A:\s*(.*?)\s*$")


def parse_qa_text(text: str, *, source: str = "<text>") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pending: str | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        question = QUESTION_RE.match(line)
        if question:
            if pending is not None:
                raise ValueError(f"Missing answer before {source}:{line_number}")
            pending = question.group(1).strip()
            continue
        answer = ANSWER_RE.match(line)
        if answer:
            if pending is None:
                raise ValueError(f"Answer without question at {source}:{line_number}")
            rows.append({"prompt": pending, "answer": answer.group(1).strip()})
            pending = None
            continue
        raise ValueError(f"Unrecognized line at {source}:{line_number}: {line!r}")
    if pending is not None:
        raise ValueError(f"Final question in {source} has no answer")
    return rows


def load_rows(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        rows: list[dict[str, str]] = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("prompt"), str) or not isinstance(row.get("answer"), str):
                raise ValueError(f"Invalid JSONL row at {source}:{line_number}")
            rows.append({"prompt": row["prompt"].strip(), "answer": row["answer"].strip()})
    else:
        rows = parse_qa_text(source.read_text(encoding="utf-8"), source=str(source))
    if not rows:
        raise ValueError(f"No examples found in {source}")
    return rows


def formatted_example(row: dict[str, str]) -> str:
    return f"Question: {row['prompt']} Answer: {row['answer']}"


def tokenizer_texts(paths: Sequence[str | Path]) -> list[str]:
    texts: list[str] = []
    for path in paths:
        source = Path(path)
        try:
            rows = load_rows(source)
        except ValueError:
            texts.append(source.read_text(encoding="utf-8"))
        else:
            texts.extend(formatted_example(row) for row in rows)
    return texts


def split_rows(
    rows: Sequence[dict[str, str]],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if len(rows) < 2:
        raise ValueError("At least two rows are required")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between zero and one")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, min(len(rows) - 1, round(len(rows) * validation_ratio)))
    validation_ids = set(indices[:validation_count])
    train = [dict(row) for index, row in enumerate(rows) if index not in validation_ids]
    validation = [dict(row) for index, row in enumerate(rows) if index in validation_ids]
    return train, validation


class AnswerOnlyDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        rows: Sequence[dict[str, str]],
        tokenizer: LiveWordTokenizer,
        *,
        max_length: int,
    ) -> None:
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        prefix = f"Question: {row['prompt']} Answer:"
        prefix_ids = self.tokenizer.encode(prefix, add_bos=True)
        answer_ids = self.tokenizer.encode(row["answer"], add_eos=True)
        ids = prefix_ids + answer_ids
        if len(ids) > self.max_length:
            raise ValueError(f"Example {index} has {len(ids)} tokens; max_length={self.max_length}")
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[: len(prefix_ids)] = -100
        return {"input_ids": input_ids, "labels": labels}


class Collator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        length = max(row["input_ids"].numel() for row in rows)
        input_ids = torch.full((len(rows), length), self.pad_id, dtype=torch.long)
        labels = torch.full((len(rows), length), -100, dtype=torch.long)
        for index, row in enumerate(rows):
            count = row["input_ids"].numel()
            input_ids[index, :count] = row["input_ids"]
            labels[index, :count] = row["labels"]
        return {"input_ids": input_ids, "labels": labels}
