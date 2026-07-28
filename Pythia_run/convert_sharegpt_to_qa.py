from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


OUTPUT_OPEN = "<output>"
OUTPUT_CLOSE = "</output>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert single-turn ShareGPT data to direct-answer Pythia QA SFT format."
    )
    parser.add_argument("--input", required=True, help="ShareGPT JSON input file.")
    parser.add_argument("--output", required=True, help="Q:/A: text output file.")
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON conversion report path.",
    )
    return parser.parse_args()


def normalize_field(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def extract_direct_answer(answer: str) -> str | None:
    """Return only the final output section and discard all reasoning text."""
    output_open = answer.find(OUTPUT_OPEN)
    output_close = answer.find(OUTPUT_CLOSE, output_open + len(OUTPUT_OPEN))
    if output_open < 0 or output_close <= output_open:
        return None
    direct_answer = normalize_field(
        answer[output_open + len(OUTPUT_OPEN) : output_close]
    )
    return direct_answer or None


def convert_records(records: object) -> tuple[list[tuple[str, str]], list[dict[str, object]]]:
    if not isinstance(records, list):
        raise ValueError("ShareGPT input must be a JSON list.")
    converted: list[tuple[str, str]] = []
    skipped: list[dict[str, object]] = []
    for index, record in enumerate(records):
        conversations = record.get("conversations") if isinstance(record, dict) else None
        if not isinstance(conversations, list) or len(conversations) != 2:
            skipped.append({"index": index, "reason": "expected exactly two conversation turns"})
            continue
        human, assistant = conversations
        if not isinstance(human, dict) or not isinstance(assistant, dict):
            skipped.append({"index": index, "reason": "conversation turns must be objects"})
            continue
        if human.get("from") != "human" or assistant.get("from") != "gpt":
            skipped.append({"index": index, "reason": "expected human then gpt roles"})
            continue
        question = normalize_field(human.get("value"))
        raw_answer = str(assistant.get("value") or "")
        if not question:
            skipped.append({"index": index, "reason": "empty question"})
            continue
        answer = extract_direct_answer(raw_answer)
        if answer is None:
            skipped.append(
                {
                    "index": index,
                    "reason": "missing, incomplete, or empty output section",
                    "question": question,
                }
            )
            continue
        converted.append((question, answer))
    return converted, skipped


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else output_path.with_suffix(".report.json")
    )
    records = json.loads(input_path.read_text(encoding="utf-8"))
    converted, skipped = convert_records(records)
    if not converted:
        raise RuntimeError("Conversion produced no valid QA records.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = "".join(f"Q: {question}\nA: {answer}\n\n" for question, answer in converted)
    output_path.write_text(output_text, encoding="utf-8")
    report = {
        "source": str(input_path),
        "output": str(output_path),
        "source_records": len(records),
        "converted_records": len(converted),
        "skipped_records": len(skipped),
        "answer_format": "direct answer extracted from <output>...</output>; thinking discarded",
        "skipped": skipped,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
