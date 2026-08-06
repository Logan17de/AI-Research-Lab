from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from tiny_pl import LiveWordTokenizer, TinyConfig, TinyPatternLM, save_base, tokenizer_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the shared random tiny Transformer base")
    parser.add_argument("--tokenizer-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=None)
    parser.add_argument(
        "--numeric-vocab-min",
        type=int,
        default=0,
        help="First whole-number token added without training exposure.",
    )
    parser.add_argument(
        "--numeric-vocab-max",
        type=int,
        default=10_000,
        help="Last whole-number token added without training exposure.",
    )
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-hidden-size", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def numeric_vocabulary_text(minimum: int, maximum: int) -> str:
    if minimum > maximum:
        raise ValueError("numeric-vocab-min cannot exceed numeric-vocab-max")
    return " ".join(str(value) for value in range(minimum, maximum + 1))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    texts = tokenizer_texts(args.tokenizer_files)
    # Whole numbers are word tokens in this experiment. Add a fixed numeric
    # vocabulary without showing the model any arithmetic examples, so an unseen
    # correct result can still be generated instead of becoming <unk>.
    texts.append(numeric_vocabulary_text(args.numeric_vocab_min, args.numeric_vocab_max))
    tokenizer = LiveWordTokenizer.build(
        texts,
        min_frequency=args.min_frequency,
        max_vocab_size=args.max_vocab_size,
    )
    config = TinyConfig(
        vocab_size=tokenizer.vocab_size,
        num_layers=args.layers,
        hidden_size=args.hidden_size,
        num_heads=args.heads,
        ffn_hidden_size=args.ffn_hidden_size,
        max_seq_len=args.max_seq_len,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.rms_norm_eps,
        dropout=args.dropout,
    )
    model = TinyPatternLM(config)
    save_base(
        args.output_dir,
        model,
        tokenizer,
        metadata={
            "seed": args.seed,
            "tokenizer_files": args.tokenizer_files,
            "tokenizer_rule": "text.strip().split()",
            "numeric_vocabulary": {
                "minimum": args.numeric_vocab_min,
                "maximum": args.numeric_vocab_max,
                "training_examples_seen": 0,
            },
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir)),
                "vocab_size": tokenizer.vocab_size,
                "numeric_vocabulary": [args.numeric_vocab_min, args.numeric_vocab_max],
                "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "config": config.to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
