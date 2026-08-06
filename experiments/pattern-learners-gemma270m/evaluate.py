from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pattern_learners.checkpoint import load_trained_base_parameters
from pattern_learners.learner import PatternLearnerSystem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate with a trained pattern learner")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pattern-name")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    summary = json.loads((checkpoint / "training_summary.json").read_text(encoding="utf-8"))
    model_name = summary["model_name"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    learner_system = PatternLearnerSystem.load(checkpoint / "learner", model)
    load_trained_base_parameters(model, checkpoint / "trained_base_parameters.pt")
    if args.pattern_name is not None:
        learner_system.set_active_pattern(args.pattern_name)

    # Loaded learner modules are created in float32; convert the complete model so
    # learner weights match BF16/FP16 hidden states during generation.
    model.to(device=device, dtype=dtype).eval()

    text = f"Question: {args.prompt}\nAnswer:"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature

    with torch.no_grad():
        output = model.generate(**inputs, **generation_kwargs)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print(f"active_pattern={learner_system.active_pattern}")
    print(f"available_patterns={list(learner_system.patterns.keys())}")


if __name__ == "__main__":
    main()
