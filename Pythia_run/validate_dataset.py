from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from typing import Any

from data import (
    CHAT_DATA_FORMAT_VERSION,
    CHAT_DATA_FORMATS,
    chat_parquet_examples,
    encode_chat_with_turn_safe_truncation,
    encode_training_example,
    iter_training_examples,
)
from model import CONTROL_TOKENS, build_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GPT_MOD supervised train/validation data.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--validation-dataset-path", required=True)
    parser.add_argument(
        "--data-format", choices=["qa_sft", "chat_jsonl", "chat_parquet"], default="chat_jsonl"
    )
    parser.add_argument("--tokenizer-name", default="EleutherAI/pythia-1.4b")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def normalized_example(example: Any) -> str:
    if isinstance(example, tuple):
        payload = {"question": example[0].strip(), "answer": example[1].strip()}
    else:
        payload = [
            {"role": str(item.get("role", "")).strip().lower(), "content": str(item.get("content", "")).strip()}
            for item in example
        ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def question_answer_texts(example: Any) -> tuple[list[str], list[str]]:
    if isinstance(example, tuple):
        return [example[0].strip().lower()], [example[1].strip().lower()]
    questions = [
        str(item.get("content", "")).strip().lower()
        for item in example if str(item.get("role", "")).strip().lower() == "user"
    ]
    answers = [
        str(item.get("content", "")).strip().lower()
        for item in example if str(item.get("role", "")).strip().lower() == "assistant"
    ]
    return questions, answers


def validate_file(path: str, data_format: str, tokenizer, seq_len: int) -> tuple[dict[str, Any], dict[str, set[str]]]:
    report: dict[str, Any] = {
        "path": str(Path(path).resolve()),
        "examples": 0,
        "duplicates": 0,
        "duplicate_questions": 0,
        "duplicate_answers": 0,
        "malformed": 0,
        "empty_questions": 0,
        "empty_answers": 0,
        "truncated_examples": 0,
        "target_tokens_lost": 0,
        "input_tokens": 0,
        "supervised_target_tokens": 0,
        "errors": [],
    }
    fingerprints = {"samples": set(), "questions": set(), "answers": set()}
    seen = {"samples": set(), "questions": set(), "answers": set()}
    try:
        context = nullcontext(None) if data_format == "chat_parquet" else open(path, "r", encoding="utf-8")
        with context as handle:
            examples = (
                chat_parquet_examples(path)
                if data_format == "chat_parquet"
                else iter_training_examples(handle, data_format)
            )
            for index, example in enumerate(examples):
                report["examples"] += 1
                normalized = normalized_example(example)
                fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                if fingerprint in seen["samples"]:
                    report["duplicates"] += 1
                seen["samples"].add(fingerprint)
                fingerprints["samples"].add(fingerprint)
                questions, answers = question_answer_texts(example)
                for question in questions:
                    question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
                    if question_hash in seen["questions"]:
                        report["duplicate_questions"] += 1
                    seen["questions"].add(question_hash)
                    fingerprints["questions"].add(question_hash)
                for answer in answers:
                    answer_hash = hashlib.sha256(answer.encode("utf-8")).hexdigest()
                    if answer_hash in seen["answers"]:
                        report["duplicate_answers"] += 1
                    seen["answers"].add(answer_hash)
                    fingerprints["answers"].add(answer_hash)

                if isinstance(example, tuple):
                    if not example[0].strip():
                        report["empty_questions"] += 1
                    if not example[1].strip():
                        report["empty_answers"] += 1
                    encoded = encode_training_example(tokenizer, data_format, example)
                    if encoded is None:
                        report["errors"].append(f"example {index}: produced no tokens")
                        continue
                    ids, labels = encoded
                    was_truncated = len(ids) > seq_len
                    if was_truncated:
                        original_targets = sum(label != -100 for label in labels)
                        ids = ids[:seq_len]
                        labels = labels[:seq_len]
                        if any(label != -100 for label in labels):
                            ids[-1] = tokenizer.eos_token_id
                            labels[-1] = tokenizer.eos_token_id
                        report["target_tokens_lost"] += original_targets - sum(label != -100 for label in labels)
                else:
                    for message in example:
                        role = str(message.get("role", "")).strip().lower()
                        content = str(message.get("content", "")).strip()
                        if role == "user" and not content:
                            report["empty_questions"] += 1
                        if role == "assistant" and not content:
                            report["empty_answers"] += 1
                    result = encode_chat_with_turn_safe_truncation(tokenizer, example, seq_len)
                    if result is None:
                        report["errors"].append(f"example {index}: cannot retain a supervised answer after truncation")
                        continue
                    ids, labels, lost, was_truncated = result
                    report["target_tokens_lost"] += lost

                report["truncated_examples"] += int(was_truncated)
                report["input_tokens"] += len(ids)
                report["supervised_target_tokens"] += sum(label != -100 for label in labels[1:])
                if labels[0] != -100:
                    report["errors"].append(f"example {index}: first label cannot be causally supervised")
                eos_targets = sum(label == tokenizer.eos_token_id for label in labels)
                expected_eos = 1 if isinstance(example, tuple) else sum(
                    str(message.get("role", "")).strip().lower() == "assistant" for message in example
                )
                if not was_truncated and eos_targets != expected_eos:
                    report["errors"].append(
                        f"example {index}: supervised EOS count={eos_targets}, expected={expected_eos}"
                    )
    except Exception as exc:
        report["malformed"] += 1
        report["errors"].append(str(exc))
    return report, fingerprints


def main() -> None:
    args = parse_args()
    tokenizer = build_tokenizer(args.tokenizer_name)
    special_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in CONTROL_TOKENS}
    if len(set(special_ids.values())) != len(CONTROL_TOKENS):
        raise RuntimeError(f"Control-token IDs are not unique: {special_ids}")
    train, train_fingerprints = validate_file(
        args.dataset_path, args.data_format, tokenizer, args.max_seq_len
    )
    validation, validation_fingerprints = validate_file(
        args.validation_dataset_path, args.data_format, tokenizer, args.max_seq_len
    )
    overlap = train_fingerprints["samples"] & validation_fingerprints["samples"]
    question_overlap = train_fingerprints["questions"] & validation_fingerprints["questions"]
    answer_overlap = train_fingerprints["answers"] & validation_fingerprints["answers"]
    result = {
        "format_version": CHAT_DATA_FORMAT_VERSION if args.data_format in CHAT_DATA_FORMATS else "qa_sft",
        "tokenizer": args.tokenizer_name,
        "tokenizer_size": len(tokenizer),
        "special_token_ids": special_ids,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "train": train,
        "validation": validation,
        "train_validation_overlap": len(overlap),
        "train_validation_question_overlap": len(question_overlap),
        "train_validation_answer_overlap": len(answer_overlap),
        "passed": not overlap and not question_overlap and not answer_overlap
        and not train["errors"] and not validation["errors"]
        and train["malformed"] == 0 and validation["malformed"] == 0,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
