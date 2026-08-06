from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from pattern_learners.checkpoint import save_experiment_checkpoint
from pattern_learners.data import CausalCollator, PromptAnswerDataset
from pattern_learners.learner import (
    LearnerLayout,
    PatternLearnerSystem,
    infer_model_shape,
    matched_all_layer_dim,
)
from pattern_learners.optimization import LearningRates, TrainabilityConfig, build_optimizer


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    known, _ = pre_parser.parse_known_args()
    config_defaults: dict[str, Any] = {}
    if known.config is not None:
        config_defaults = json.loads(Path(known.config).read_text(encoding="utf-8"))

    parser = argparse.ArgumentParser(
        description="Train named residual pattern learners on Gemma 3 270M",
        parents=[pre_parser],
    )
    parser.add_argument("--model-name", default="google/gemma-3-270m")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pattern-name", default="math")

    parser.add_argument("--learner-mode", choices=("single", "all"), default="single")
    parser.add_argument("--learner-dim", type=int, default=None)
    parser.add_argument(
        "--match-single-dim",
        type=int,
        default=None,
        help="For all-layer mode, derive N ~= single_dim / num_layers.",
    )
    parser.add_argument("--learner-layer", type=int, default=-1)
    parser.add_argument("--learner-dropout", type=float, default=0.0)

    freeze_group = parser.add_argument_group("independent freezing")
    freeze_group.add_argument("--freeze-embeddings", action="store_true", dest="freeze_embeddings")
    freeze_group.add_argument("--unfreeze-embeddings", action="store_false", dest="freeze_embeddings")
    freeze_group.add_argument("--freeze-backbone", action="store_true", dest="freeze_backbone")
    freeze_group.add_argument("--unfreeze-backbone", action="store_false", dest="freeze_backbone")
    freeze_group.add_argument("--freeze-learner", action="store_true", dest="freeze_learner")
    freeze_group.add_argument("--unfreeze-learner", action="store_false", dest="freeze_learner")
    parser.set_defaults(freeze_embeddings=True, freeze_backbone=True, freeze_learner=False)

    parser.add_argument("--embedding-lr", type=float, default=1e-5)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--learner-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-best-only", action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(**config_defaults)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_context(device, dtype):
            loss = model(**batch, use_cache=False).loss
        total_loss += float(loss.detach())
        total_batches += 1
    mean_loss = total_loss / max(total_batches, 1)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
    }


def train() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("epochs, batch-size, and gradient-accumulation must be positive")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, device)
    print(f"device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    hidden_size, num_layers = infer_model_shape(model)

    if args.learner_dim is not None and args.match_single_dim is not None:
        raise ValueError("Use either --learner-dim or --match-single-dim, not both")
    if args.match_single_dim is not None:
        if args.learner_mode != "all":
            raise ValueError("--match-single-dim is only valid with --learner-mode all")
        learner_dim = matched_all_layer_dim(args.match_single_dim, num_layers)
    elif args.learner_dim is not None:
        learner_dim = args.learner_dim
    else:
        learner_dim = hidden_size if args.learner_mode == "single" else matched_all_layer_dim(hidden_size, num_layers)

    layout = LearnerLayout(
        mode=args.learner_mode,
        bottleneck_dim=learner_dim,
        single_layer=args.learner_layer,
        dropout=args.learner_dropout,
    )
    learner_system = PatternLearnerSystem(hidden_size, num_layers, layout)
    learner_system.add_pattern(args.pattern_name)
    learner_system.attach(model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to(device)

    trainability = TrainabilityConfig(
        freeze_embeddings=args.freeze_embeddings,
        freeze_backbone=args.freeze_backbone,
        freeze_learner=args.freeze_learner,
    )
    learning_rates = LearningRates(
        embedding=args.embedding_lr,
        backbone=args.backbone_lr,
        learner=args.learner_lr,
    )
    optimizer, group_summary = build_optimizer(
        model,
        learner_system,
        trainability,
        learning_rates,
        weight_decay=args.weight_decay,
    )

    print(json.dumps({"layout": asdict(layout), "optimizer_groups": group_summary}, indent=2))
    print(f"learner_parameters={learner_system.learner_parameter_count(args.pattern_name):,}")

    train_dataset = PromptAnswerDataset(args.train_file, tokenizer, max_length=args.max_length)
    validation_dataset = PromptAnswerDataset(args.validation_file, tokenizer, max_length=args.max_length)
    collator = CausalCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    best_perplexity = float("inf")
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for micro_step, batch in enumerate(progress, start=1):
            batch = move_batch(batch, device)
            with autocast_context(device, dtype):
                loss = model(**batch, use_cache=False).loss / args.gradient_accumulation

            scaler.scale(loss).backward()
            should_update = micro_step % args.gradient_accumulation == 0 or micro_step == len(train_loader)
            if not should_update:
                continue

            scaler.unscale_(optimizer)
            trainable_parameters = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            progress.set_postfix(loss=f"{float(loss) * args.gradient_accumulation:.4f}")
            if args.eval_every > 0 and global_step % args.eval_every == 0:
                metrics = evaluate(model, validation_loader, device, dtype)
                print(f"step={global_step} validation={metrics}")
                is_best = metrics["perplexity"] < best_perplexity
                if is_best:
                    best_perplexity = metrics["perplexity"]
                if is_best or not args.save_best_only:
                    save_experiment_checkpoint(
                        output / ("best" if is_best else f"step-{global_step}"),
                        model,
                        learner_system,
                        tokenizer,
                        training_summary={
                            "model_name": args.model_name,
                            "global_step": global_step,
                            "epoch": epoch + 1,
                            "validation": metrics,
                            "layout": asdict(layout),
                            "trainability": asdict(trainability),
                            "learning_rates": asdict(learning_rates),
                            "optimizer_groups": group_summary,
                        },
                    )
                model.train()

    final_metrics = evaluate(model, validation_loader, device, dtype)
    is_best = final_metrics["perplexity"] < best_perplexity
    checkpoint_name = "best" if is_best else "final"
    save_experiment_checkpoint(
        output / checkpoint_name,
        model,
        learner_system,
        tokenizer,
        training_summary={
            "model_name": args.model_name,
            "global_step": global_step,
            "epoch": args.epochs,
            "validation": final_metrics,
            "layout": asdict(layout),
            "trainability": asdict(trainability),
            "learning_rates": asdict(learning_rates),
            "optimizer_groups": group_summary,
        },
    )
    print(f"finished validation={final_metrics} checkpoint={output / checkpoint_name}")


if __name__ == "__main__":
    train()
