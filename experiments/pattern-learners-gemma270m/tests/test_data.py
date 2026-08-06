from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pattern_learners.data import (
    load_prompt_answer_rows,
    parse_numbered_qa_text,
    split_prompt_answer_rows,
)


class DataLoadingTests(unittest.TestCase):
    def test_numbered_qa_text_parser(self) -> None:
        rows = parse_numbered_qa_text(
            "1. Q: What is 2 + 3?\n"
            "   A: 5\n\n"
            "2. Q: What's 4 times 6?\n"
            "   A: 24\n"
        )
        self.assertEqual(
            rows,
            [
                {"prompt": "What is 2 + 3?", "answer": "5"},
                {"prompt": "What's 4 times 6?", "answer": "24"},
            ],
        )

    def test_loads_txt_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text_path = root / "add.txt"
            text_path.write_text("1. Q: 1 + 2 = ?\n   A: 3\n", encoding="utf-8")
            jsonl_path = root / "add.jsonl"
            jsonl_path.write_text(
                json.dumps({"prompt": "2 + 2 = ?", "answer": "4"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_prompt_answer_rows(text_path)[0]["answer"], "3")
            self.assertEqual(load_prompt_answer_rows(jsonl_path)[0]["answer"], "4")

    def test_split_is_deterministic(self) -> None:
        rows = [
            {"prompt": f"question-{index}", "answer": str(index)}
            for index in range(10)
        ]
        first_train, first_validation = split_prompt_answer_rows(
            rows,
            validation_ratio=0.2,
            seed=7,
        )
        second_train, second_validation = split_prompt_answer_rows(
            rows,
            validation_ratio=0.2,
            seed=7,
        )
        self.assertEqual(first_train, second_train)
        self.assertEqual(first_validation, second_validation)
        self.assertEqual(len(first_train), 8)
        self.assertEqual(len(first_validation), 2)


if __name__ == "__main__":
    unittest.main()
