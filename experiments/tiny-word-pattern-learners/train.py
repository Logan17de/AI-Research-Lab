from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tiny_pl import (
    AnswerOnlyDataset,
    Collator,
    LearnerLayout,
    TinyPatternLM,
    load_base,
    load_rows,
    load_run,
    save_run,
    split_rows,
)


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args()
    defaults: dict[str, Any] = {}
    if known.config:
        defaults = json.loads(Path(known.config).read_text(encoding="utf-8"))

    parser = argparse.ArgumentParser(
        description="Train a named learner on a frozen tiny random Transformer",
        parents=[pre],
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-checkpoint")
    source.add_argument("--source-checkpoint")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pattern-name", default="addition")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)

    parser.add_argument("--learner-mode", choices=("single", "all"), default="single")
    parser.add_argument("--learner-dim", type=int, default=128)
    parser.add_argument("--learner-layer", type=int, default=-1)
    parser.add_argument("--learner-dropout", type=float, default=0.0)

    for group in ("embeddings", "backbone", "lm-head", "learner"):
        destination = f"freeze_{group.replace('-', '_')}"
        parser.add_argument(f"--freeze-{group}", action="store_true", dest=destination)
        parser.add_argument(f"--unfreeze-{group}", action="store_false", dest=destination)
    parser.set_defaults(
        freeze_embeddings=True,
        freeze_backbone=True,
        freeze_lm_head=True,
        freeze_learner=False,
    )

    parser.add_argument("--embedding-lr", type=float, default=1e-6)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--lm-head-lr", type=float, default=1e-6)
    parser.add_argument("--learner-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--max-answer-tokens", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(**defaults)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_groups(model: TinyPatternLM) -> dict[str, list[torch.nn.Parameter]]:
    learner_ids = {id(parameter) for parameter in model.patterns.parameters()}
    embedding_ids = {id(parameter) for parameter in model.token_embedding.parameters()}
    head_ids = {id(parameter) for parameter in model.lm_head.parameters()}
    groups = {"embedding": [], "backbone": [], "lm_head": [], "learner": []}
    for parameter in model.parameters():
        identifier = id(parameter)
        if identifier in learner_ids:
            groups["learner"].append(parameter)
        elif identifier in embedding_ids:
            groups["embedding"].append(parameter)
        elif identifier in head_ids:
            groups["lm_head"].append(parameter)
        else:
            groups["backbone"].append(parameter)
    return groups


def configure_optimizer(
    model: TinyPatternLM,
    args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, dict[str, dict[str, Any]]]:
    groups = parameter_groups(model)
    freeze = {
        "embedding": args.freeze_embeddings,
        "backbone": args.freeze_backbone,
        "lm_head": args.freeze_lm_head,
        "learner": args.freeze_learner,
    }
    lrs = {
        "embedding": args.embedding_lr,
        "backbone": args.backbone_lr,
        "lm_head": args.lm_head_lr,
        "learner": args.learner_lr,
    }

    active_learner_ids = {
        id(parameter) for parameter in model.patterns[args.pattern_name].parameters()
    }
    optimizer_groups: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}

    for name, parameters in groups.items():
        for parameter in parameters:
            parameter.requires_grad = not freeze[name]
        if name == "learner" and not freeze[name]:
            for parameter in parameters:
                parameter.requires_grad = id(parameter) in active_learner_ids

        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        summary[name] = {
            "total_parameters": sum(parameter.numel() for parameter in parameters),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "frozen": not trainable,
            "learning_rate": lrs[name],
        }
        if trainable:
            optimizer_groups.append(
                {
                    "params": trainable,
                    "lr": lrs[name],
                    "weight_decay": args.weight_decay,
                    "group_name": name,
                }
            )

    if not optimizer_groups:
        raise ValueError("Every parameter group is frozen")
    return torch.optim.AdamW(optimizer_groups, betas=(0.9, 0.95)), summary


def load_model(args: argparse.Namespace) -> tuple[TinyPatternLM, Any]:
    requested_layout = LearnerLayout(
        mode=args.learner_mode,
        bottleneck_dim=args.learner_dim,
        single_layer=args.learner_layer,
        dropout=args.learner_dropout,
    )

    if args.base_checkpoint:
        base_model, tokenizer = load_base(args.base_checkpoint)
        model = TinyPatternLM(base_model.config, requested_layout)
        model.load_state_dict(base_model.state_dict(), strict=True)
        model.add_pattern(args.pattern_name)
        return model, tokenizer

    source_model, tokenizer, payload = load_run(args.source_checkpoint)
    if source_model.learner_layout != requested_layout:
        raise ValueError(
            f"Requested layout {requested_layout} does not match source layout "
            f"{source_model.learner_layout}"
        )
    if args.pattern_name in source_model.patterns:
        source_model.set_active_pattern(args.pattern_name)
    else:
        source_model.add_pattern(args.pattern_name)
    return source_model, tokenizer


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}


@torch.inference_mode()
def evaluate(
    model: TinyPatternLM,
    loader: DataLoader,
    rows: list[dict[str, str]],
    tokenizer: Any,
    device: torch.device,
    max_answer_tokens: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        output = model(**move_batch(batch, device))
        total_loss += float(output["loss"].detach())
        batches += 1

    correct = 0
    for row in rows:
        prompt = f"Question: {row['prompt']} Answer:"
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output_ids = model.generate(
            inputs,
            eos_id=tokenizer.eos_id,
            max_new_tokens=max_answer_tokens,
        )[0, len(prompt_ids) :].tolist()
        if tokenizer.eos_id in output_ids:
            output_ids = output_ids[: output_ids.index(tokenizer.eos_id)]
        predicted = tokenizer.decode(output_ids).strip()
        correct += int(predicted == row["answer"].strip())

    mean_loss = total_loss / max(batches, 1)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "exact_accuracy": correct / len(rows),
        "correct": float(correct),
        "total": float(len(rows)),
    }


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.batch_size, args.gradient_accumulation) <= 0:
        raise ValueError("epochs, batch-size, and gradient-accumulation must be positive")
    if args.eval_every <= 0:
        raise ValueError("eval-every must be positive")
    if args.early_stopping_patience < 0:
        raise ValueError("early-stopping-patience cannot be negative")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args)
    model.to(device)
    optimizer, group_summary = configure_optimizer(model, args)

    rows = load_rows(args.data_file)
    train_rows, validation_rows = split_rows(
        rows,
        validation_ratio=args.validation_ratio,
        seed=args.split_seed,
    )
    train_dataset = AnswerOnlyDataset(
        train_rows,
        tokenizer,
        max_length=model.config.max_seq_len,
    )
    validation_dataset = AnswerOnlyDataset(
        validation_rows,
        tokenizer,
        max_length=model.config.max_seq_len,
    )
    collator = Collator(tokenizer.pad_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )

    print(
        json.dumps(
            {
                "device": str(device),
                "model_config": model.config.to_dict(),
                "learner_layout": model.learner_layout.to_dict(),
                "patterns": list(model.patterns.keys()),
                "active_pattern": model.active_pattern,
                "active_learner_parameters": model.learner_parameter_count(args.pattern_name),
                "optimizer_groups": group_summary,
                "dataset": {
                    "total": len(rows),
                    "train": len(train_rows),
                    "validation": len(validation_rows),
                },
            },
            indent=2,
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    best_loss = float("inf")
    evaluations_without_improvement = 0
    global_step = 0
    stopped_early = False
    optimizer.zero_grad(set_to_none=True)

    def save_checkpoint(name: str, metrics: dict[str, float], epoch: int) -> None:
        save_run(
            output_dir / name,
            model,
            tokenizer,
            summary={
                "epoch": epoch,
                "global_step": global_step,
                "metrics": metrics,
                "data_file": args.data_file,
                "split_seed": args.split_seed,
                "validation_ratio": args.validation_ratio,
                "optimizer_groups": group_summary,
                "learner_only": (
                    args.freeze_embeddings
                    and args.freeze_backbone
                    and args.freeze_lm_head
                    and not args.freeze_learner
                ),
            },
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for micro_step, batch in enumerate(progress, start=1):
            output = model(**move_batch(batch, device))
            loss = output["loss"] / args.gradient_accumulation
            loss.backward()

            should_step = (
                micro_step % args.gradient_accumulation == 0
                or micro_step == len(train_loader)
            )
            if not should_step:
                continue

            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.set_postfix(loss=f"{float(loss.detach()) * args.gradient_accumulation:.4f}")

            if global_step % args.eval_every != 0:
                continue

            metrics = evaluate(
                model,
                validation_loader,
                validation_rows,
                tokenizer,
                device,
                args.max_answer_tokens,
            )
            print(f"step={global_step} validation={metrics}")
            accuracy_improved = (
                metrics["exact_accuracy"] > best_accuracy + args.early_stopping_min_delta
            )
            tie_but_loss_improved = (
                abs(metrics["exact_accuracy"] - best_accuracy)
                <= args.early_stopping_min_delta
                and metrics["loss"] < best_loss
            )
            if accuracy_improved or tie_but_loss_improved:
                best_accuracy = metrics["exact_accuracy"]
                best_loss = metrics["loss"]
                evaluations_without_improvement = 0
                save_checkpoint("best", metrics, epoch)
            else:
                evaluations_without_improvement += 1

            if (
                args.early_stopping_patience > 0
                and evaluations_without_improvement >= args.early_stopping_patience
            ):
                print(
                    f"Early stopping after {evaluations_without_improvement} "
                    "evaluations without improvement."
                )
                stopped_early = True
                break
            model.train()

        if stopped_early:
            break

    final_metrics = evaluate(
        model,
        validation_loader,
        validation_rows,
        tokenizer,
        device,
        args.max_answer_tokens,
    )
    save_checkpoint("final", final_metrics, epoch)
    print(
        json.dumps(
            {
                "finished": True,
                "stopped_early": stopped_early,
                "best_exact_accuracy": best_accuracy,
                "final": final_metrics,
                "best_checkpoint": str(output_dir / "best"),
                "final_checkpoint": str(output_dir / "final"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
