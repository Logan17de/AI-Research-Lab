from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


EXPERIMENTS = ("frozen_mod", "plastic_mod", "full_finetune", "ate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the controlled Pythia-1.4B MOD vs Pythia-2.8B full-FT runs.")
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--validation-path", required=True)
    parser.add_argument("--dataset-manifest", default=None)
    parser.add_argument(
        "--data-format", choices=["qa_sft", "chat_jsonl", "chat_parquet"], default="chat_jsonl"
    )
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--target-training-tokens", type=int, required=True)
    parser.add_argument("--eval-every-tokens", type=int, required=True)
    parser.add_argument("--save-every-tokens", type=int, required=True)
    parser.add_argument("--mod-batch-size", type=int, default=2)
    parser.add_argument("--mod-grad-accum", type=int, default=16)
    parser.add_argument("--full-batch-size", type=int, default=1)
    parser.add_argument("--full-grad-accum", type=int, default=32)
    parser.add_argument("--emb-mod-dim", type=int, default=128)
    parser.add_argument("--out-mod-dim", type=int, default=128)
    parser.add_argument("--attn-mod-dim", type=int, default=128)
    parser.add_argument("--ffn-mod-dim", type=int, default=256)
    parser.add_argument("--attn-unique-mod-count", type=int, default=1)
    parser.add_argument("--ffn-unique-mod-count", type=int, default=1)
    parser.add_argument("--mod-lr", type=float, default=3e-4)
    parser.add_argument("--full-lr", type=float, default=1e-5)
    parser.add_argument("--plastic-last-n-layers", type=int, default=4)
    parser.add_argument("--plastic-lr", type=float, default=1e-5)
    parser.add_argument("--plastic-layernorm-lr", type=float, default=3e-5)
    parser.add_argument("--new-attn-heads", type=int, default=1)
    parser.add_argument("--new-ffn-layers", type=int, default=1)
    parser.add_argument(
        "--plasticity-mode", choices=["off", "linear", "quadratic"], default="quadratic"
    )
    parser.add_argument("--plasticity-base-lr", type=float, default=1e-5)
    parser.add_argument("--ate-lr", type=float, default=3e-4)
    parser.add_argument("--ate-confirm", default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--min-lr-ratio", type=float, default=0.03)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--optimizer-state-offload", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command(args: argparse.Namespace, experiment: str) -> list[str]:
    uses_mod = experiment in {"frozen_mod", "plastic_mod"}
    uses_ate = experiment == "ate"
    model_name = "EleutherAI/pythia-1.4b" if uses_mod or uses_ate else "EleutherAI/pythia-2.8b"
    run_name = {
        "frozen_mod": "pythia_1.4b_frozen_mod",
        "plastic_mod": "pythia_1.4b_plastic_mod",
        "full_finetune": "pythia_2.8b_full_ft",
        "ate": "pythia_1.4b_ate",
    }[experiment]
    batch_size = args.mod_batch_size if uses_mod or uses_ate else args.full_batch_size
    grad_accum = args.mod_grad_accum if uses_mod or uses_ate else args.full_grad_accum
    values = [
        sys.executable, "train.py",
        "--experiment-type", "frozen_mod" if uses_mod else ("ate" if uses_ate else "full_finetune"),
        "--model-name", model_name,
        "--tokenizer-name", "EleutherAI/pythia-1.4b",
        "--dataset-path", args.train_path,
        "--validation-dataset-path", args.validation_path,
        "--text_data_format", args.data_format,
        "--run-dir", str(Path(args.run_root) / run_name),
        "--device", args.device,
        "--seed", str(args.seed),
        "--max-seq-len", str(args.seq_len),
        "--target-training-tokens", str(args.target_training_tokens),
        "--eval-every-tokens", str(args.eval_every_tokens),
        "--save-every-tokens", str(args.save_every_tokens),
        "--batch_size", str(batch_size),
        "--grad_accum_steps", str(grad_accum),
        "--num_workers", str(args.num_workers),
        "--emb-mod-dim", str(args.emb_mod_dim if uses_mod else 0),
        "--out-mod-dim", str(args.out_mod_dim if uses_mod else 0),
        "--attn-mod-dim", str(args.attn_mod_dim if uses_mod else 0),
        "--ffn-mod-dim", str(args.ffn_mod_dim if uses_mod else 0),
        "--attn-unique-mod-count", str(args.attn_unique_mod_count if uses_mod else 1),
        "--ffn-unique-mod-count", str(args.ffn_unique_mod_count if uses_mod else 1),
        "--learning-rate", str(args.ate_lr if uses_ate else (args.full_lr if not uses_mod else 0.0)),
        "--modifier_lr", str(args.mod_lr),
        "--weight_decay", str(args.weight_decay),
        "--warmup-ratio", str(args.warmup_ratio),
        "--min_lr_ratio", str(args.min_lr_ratio),
        "--log_every", str(args.log_every),
        "--resume", "1" if args.resume else "0",
        "--mixed_precision", "1",
    ]
    if args.dataset_manifest:
        if args.num_workers != 0:
            raise ValueError("A locked comparison dataset requires --num-workers 0.")
        values.extend(
            [
                "--dataset-manifest", args.dataset_manifest,
                "--strict-sample-order", "1",
                "--shuffle-buffer", "0",
            ]
        )
    if experiment == "plastic_mod":
        values.extend(
            [
                "--plastic-last-n-layers", str(args.plastic_last_n_layers),
                "--plastic-final-layer-norm", "1",
                "--plastic-lr", str(args.plastic_lr),
                "--plastic-layernorm-lr", str(args.plastic_layernorm_lr),
            ]
        )
    if uses_ate:
        values.extend(
            [
                "--new-attn-heads", str(args.new_attn_heads),
                "--new-ffn-layers", str(args.new_ffn_layers),
                "--plasticity-mode", args.plasticity_mode,
                "--plasticity-base-lr", str(args.plasticity_base_lr),
            ]
        )
        if args.ate_confirm is not None:
            values.extend(["--ate-confirm", args.ate_confirm])
    if args.bf16:
        values.append("--bf16")
    if args.gradient_checkpointing:
        values.append("--gradient-checkpointing")
    if args.optimizer_state_offload:
        values.append("--optimizer-state-offload")
    return values


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    for experiment in args.experiments:
        values = command(args, experiment)
        print("\n" + " ".join(values), flush=True)
        if not args.dry_run:
            subprocess.run(values, cwd=root, check=True)


if __name__ == "__main__":
    main()
