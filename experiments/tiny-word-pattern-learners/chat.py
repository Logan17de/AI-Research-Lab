from __future__ import annotations

import argparse

import torch

from tiny_pl import load_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a tiny word-level pattern learner")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pattern-name")
    parser.add_argument("--max-answer-tokens", type=int, default=4)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = load_run(args.checkpoint)
    if args.pattern_name is not None:
        model.set_active_pattern(args.pattern_name)
    model.to(device).eval()

    print(f"device={device}")
    print(f"patterns={list(model.patterns.keys())}")
    print(f"active_pattern={model.active_pattern}")
    print("Commands: /patterns, /use NAME, /base, /exit")

    while True:
        active = model.active_pattern or "base"
        try:
            text = input(f"You [{active}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not text:
            continue
        if text in {"/exit", "/quit", "/q"}:
            print("Bye.")
            break
        if text == "/patterns":
            print(f"patterns={list(model.patterns.keys())} active={model.active_pattern}")
            continue
        if text == "/base":
            model.set_active_pattern(None)
            print("Active pattern: base")
            continue
        if text.startswith("/use "):
            name = text[5:].strip()
            try:
                model.set_active_pattern(name)
            except KeyError as error:
                print(error)
            else:
                print(f"Active pattern: {name}")
            continue

        prompt = f"Question: {text} Answer:"
        prompt_tokens = tokenizer.split(prompt)
        unknown = [token for token in prompt_tokens if token not in tokenizer.token_to_id]
        if unknown:
            print(f"warning: unknown word tokens -> {unknown}")
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output = model.generate(
            inputs,
            eos_id=tokenizer.eos_id,
            max_new_tokens=args.max_answer_tokens,
        )[0, len(prompt_ids) :].tolist()
        if tokenizer.eos_id in output:
            output = output[: output.index(tokenizer.eos_id)]
        print(f"Model [{active}]> {tokenizer.decode(output).strip() or '<empty>'}\n")


if __name__ == "__main__":
    main()
