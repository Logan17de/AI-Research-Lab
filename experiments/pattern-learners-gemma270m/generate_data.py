from __future__ import annotations

import argparse

from pattern_learners.data import write_arithmetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic arithmetic JSONL data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--operation", choices=("addition", "multiplication", "star"), default="addition")
    parser.add_argument("--train-examples", type=int, default=4000)
    parser.add_argument("--validation-examples", type=int, default=500)
    parser.add_argument("--minimum", type=int, default=0)
    parser.add_argument("--maximum", type=int, default=99)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write_arithmetic_dataset(
        args.output_dir,
        seed=args.seed,
        train_examples=args.train_examples,
        validation_examples=args.validation_examples,
        operation=args.operation,
        minimum=args.minimum,
        maximum=args.maximum,
    )


if __name__ == "__main__":
    main()
