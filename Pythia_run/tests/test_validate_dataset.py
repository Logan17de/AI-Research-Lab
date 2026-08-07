from __future__ import annotations

from validate_dataset import validate_file
from prepare_pythia_split import grouped_split


class TinyTokenizer:
    eos_token_id = 0
    eos_token = "<EOS>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1 + (ord(character) % 31) for character in text]


def test_dataset_validator_detects_duplicates_empty_fields_and_overlap(tmp_path) -> None:
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    train.write_text(
        "Q: Shared question\nA: Shared answer\n\n"
        "Q: Shared question\nA: Shared answer\n\n"
        "Q:\nA: Missing question\n\n",
        encoding="utf-8",
    )
    validation.write_text("Q: Shared question\nA: Shared answer\n\n", encoding="utf-8")
    train_report, train_ids = validate_file(str(train), "qa_sft", TinyTokenizer(), 256)
    validation_report, validation_ids = validate_file(str(validation), "qa_sft", TinyTokenizer(), 256)

    assert train_report["duplicates"] == 1
    assert train_report["empty_questions"] == 1
    assert validation_report["errors"] == []
    assert len(train_ids["samples"] & validation_ids["samples"]) == 1
    assert len(train_ids["questions"] & validation_ids["questions"]) == 1
    assert len(train_ids["answers"] & validation_ids["answers"]) == 1


def test_grouped_split_keeps_shared_questions_and_answers_together() -> None:
    examples = [
        ("shared question", "answer one"),
        ("shared question", "answer two"),
        ("different question", "answer two"),
        ("independent", "independent answer"),
    ]
    train, validation, groups = grouped_split(examples, 0.25, 42)
    assert groups == 2
    train_set = set(train)
    validation_set = set(validation)
    connected = set(examples[:3])
    assert connected <= train_set or connected <= validation_set
