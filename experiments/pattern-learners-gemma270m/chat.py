from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pattern_learners.checkpoint import load_trained_base_parameters
from pattern_learners.learner import PatternLearnerSystem


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively chat with a trained pattern-learner checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--pattern-name",
        help="Initial learner to activate. Defaults to the checkpoint's active learner.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    return parser.parse_args()


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def clean_generated_answer(text: str) -> str:
    """Keep the first generated answer line and remove prompt-like spillover."""
    answer = text.strip()
    if not answer:
        return "<empty response>"

    lines = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines:
                break
            continue
        lowered = stripped.lower()
        if lines and (lowered.startswith("question:") or lowered.startswith("user:")):
            break
        lines.append(stripped)

    return " ".join(lines).strip() or "<empty response>"


def parse_chat_command(text: str) -> tuple[str, str | None] | None:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    command_text = stripped[1:]
    command, separator, argument = command_text.partition(" ")
    normalized = command.lower()
    aliases = {
        "q": "exit",
        "quit": "exit",
        "patterns": "patterns",
        "ls": "patterns",
        "use": "use",
        "base": "base",
        "none": "base",
        "help": "help",
        "h": "help",
        "settings": "settings",
    }
    return aliases.get(normalized, normalized), argument.strip() if separator else None


def generation_kwargs(
    config: GenerationConfig,
    tokenizer: Any,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": config.repetition_penalty,
    }
    if config.temperature > 0:
        kwargs.update(
            {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
            }
        )
    return kwargs


@torch.inference_mode()
def answer_question(
    model: torch.nn.Module,
    tokenizer: Any,
    question: str,
    device: torch.device,
    config: GenerationConfig,
) -> str:
    prompt = f"Question: {question.strip()}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        **generation_kwargs(config, tokenizer),
    )
    generated_ids = output[0, inputs["input_ids"].shape[1] :]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return clean_generated_answer(decoded)


def print_help() -> None:
    print(
        "\nCommands:\n"
        "  /patterns          list available learners\n"
        "  /use NAME          activate a learner\n"
        "  /base              disable all learners\n"
        "  /settings          show generation settings\n"
        "  /help              show this help\n"
        "  /exit              quit\n"
    )


def print_patterns(learner_system: PatternLearnerSystem) -> None:
    active = learner_system.active_pattern
    print("Available learners:")
    for name in learner_system.patterns.keys():
        marker = "*" if name == active else " "
        print(f"  {marker} {name}")
    print("  * base" if active is None else "    base")


def run_chat() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    summary_path = checkpoint / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing checkpoint summary: {summary_path}")

    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    config.validate()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_name = summary["model_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, device)

    print(f"Loading {model_name} on {device} with {dtype}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    learner_system = PatternLearnerSystem.load(checkpoint / "learner", model)
    load_trained_base_parameters(model, checkpoint / "trained_base_parameters.pt")

    if args.pattern_name is not None:
        learner_system.set_active_pattern(args.pattern_name)

    # Checkpoint-loaded learner modules start as float32. Explicit dtype conversion
    # keeps them aligned with the BF16/FP16 hidden states used by the base model.
    model.to(device=device, dtype=dtype).eval()
    model.config.use_cache = True

    print("\nInteractive pattern-learner chat")
    print("Each turn uses the same Question/Answer format used during training.")
    print_patterns(learner_system)
    print_help()

    while True:
        active = learner_system.active_pattern or "base"
        try:
            user_text = input(f"You [{active}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_text:
            continue

        parsed = parse_chat_command(user_text)
        if parsed is not None:
            command, argument = parsed
            if command == "exit":
                print("Bye.")
                break
            if command == "help":
                print_help()
                continue
            if command == "patterns":
                print_patterns(learner_system)
                continue
            if command == "settings":
                print(config)
                continue
            if command == "base":
                learner_system.set_active_pattern(None)
                print("Active learner: base")
                continue
            if command == "use":
                if not argument:
                    print("Usage: /use NAME")
                    continue
                try:
                    learner_system.set_active_pattern(argument)
                except KeyError:
                    print(f"Unknown learner: {argument!r}")
                    print_patterns(learner_system)
                else:
                    print(f"Active learner: {argument}")
                continue

            print(f"Unknown command: /{command}. Use /help.")
            continue

        try:
            response = answer_question(model, tokenizer, user_text, device, config)
        except RuntimeError as error:
            if "out of memory" in str(error).lower() and device.type == "cuda":
                torch.cuda.empty_cache()
                print("Model> CUDA ran out of memory. Reduce --max-new-tokens.")
                continue
            raise
        print(f"Model [{active}]> {response}\n")


if __name__ == "__main__":
    run_chat()
