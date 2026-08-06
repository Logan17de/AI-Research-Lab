from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tiny_pl import load_rows, load_run, split_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exact answers from a tiny pattern learner")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--pattern-name")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-answer-tokens", type=int, default=4)
    parser.add_argument("--show-mistakes", type=int, default=20)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, payload = load_run(args.checkpoint)
    if args.pattern_name is not None:
        model.set_active_pattern(args.pattern_name)
    model.to(device).eval()

    rows = load_rows(args.data_file)
    if args.validation_only:
        _, rows = split_rows(
            rows,
            validation_ratio=args.validation_ratio,
            seed=args.split_seed,
        )

    correct = 0
    mistakes: list[dict[str, str]] = []
    unknown_prompt_tokens = 0

    for row in rows:
        prompt = f"Question: {row['prompt']} Answer:"
        prompt_tokens = tokenizer.split(prompt)
        unknown_prompt_tokens += sum(token not in tokenizer.token_to_id for token in prompt_tokens)
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(
            inputs,
            eos_id=tokenizer.eos_id,
            max_new_tokens=args.max_answer_tokens,
        )[0, len(prompt_ids) :].tolist()
        if tokenizer.eos_id in generated:
            generated = generated[: generated.index(tokenizer.eos_id)]
        predicted = tokenizer.decode(generated).strip()
        expected = row["answer"].strip()
        if predicted == expected:
            correct += 1
        elif len(mistakes) < args.show_mistakes:
            mistakes.append(
                {
                    "question": row["prompt"],
                    "expected": expected,
                    "predicted": predicted,
                }
            )

    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "active_pattern": model.active_pattern,
        "examples": len(rows),
        "correct": correct,
        "exact_accuracy": correct / len(rows),
        "unknown_prompt_tokens": unknown_prompt_tokens,
        "patterns": list(model.patterns.keys()),
        "mistakes": mistakes,
        "training_summary": payload.get("summary", {}),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
