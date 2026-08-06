from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .learner import PatternLearnerSystem


@dataclass(frozen=True)
class TrainabilityConfig:
    freeze_embeddings: bool = True
    freeze_backbone: bool = True
    freeze_learner: bool = False


@dataclass(frozen=True)
class LearningRates:
    embedding: float = 1e-5
    backbone: float = 1e-5
    learner: float = 3e-4

    def validate(self) -> None:
        for name, value in (
            ("embedding", self.embedding),
            ("backbone", self.backbone),
            ("learner", self.learner),
        ):
            if value <= 0:
                raise ValueError(f"{name} learning rate must be positive")


def parameter_partitions(
    model: nn.Module,
    learner_system: PatternLearnerSystem,
) -> dict[str, list[nn.Parameter]]:
    """Split unique parameters into embeddings, original backbone, and learners.

    Input embeddings and output embeddings/lm_head are treated as one embedding
    group. Gemma ties them, so parameter IDs are deduplicated automatically.
    """
    embedding_ids: set[int] = set()
    for accessor_name in ("get_input_embeddings", "get_output_embeddings"):
        accessor = getattr(model, accessor_name, None)
        module = accessor() if callable(accessor) else None
        if module is not None:
            embedding_ids.update(id(parameter) for parameter in module.parameters())

    learner_ids = {id(parameter) for parameter in learner_system.parameters()}
    partitions: dict[str, list[nn.Parameter]] = {
        "embedding": [],
        "backbone": [],
        "learner": [],
    }
    seen: set[int] = set()

    for parameter in model.parameters():
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        if parameter_id in learner_ids:
            partitions["learner"].append(parameter)
        elif parameter_id in embedding_ids:
            partitions["embedding"].append(parameter)
        else:
            partitions["backbone"].append(parameter)

    missing_learner_ids = learner_ids - seen
    if missing_learner_ids:
        raise RuntimeError("Learner system must be attached to the model before partitioning")
    return partitions


def configure_trainability(
    model: nn.Module,
    learner_system: PatternLearnerSystem,
    config: TrainabilityConfig,
) -> dict[str, list[nn.Parameter]]:
    partitions = parameter_partitions(model, learner_system)
    frozen = {
        "embedding": config.freeze_embeddings,
        "backbone": config.freeze_backbone,
        "learner": config.freeze_learner,
    }
    for group_name, parameters in partitions.items():
        requires_grad = not frozen[group_name]
        for parameter in parameters:
            parameter.requires_grad = requires_grad
    return partitions


def build_optimizer(
    model: nn.Module,
    learner_system: PatternLearnerSystem,
    trainability: TrainabilityConfig,
    learning_rates: LearningRates,
    *,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
) -> tuple[torch.optim.Optimizer, dict[str, dict[str, Any]]]:
    learning_rates.validate()
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")

    partitions = configure_trainability(model, learner_system, trainability)
    lr_by_group = {
        "embedding": learning_rates.embedding,
        "backbone": learning_rates.backbone,
        "learner": learning_rates.learner,
    }

    optimizer_groups: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    for name, parameters in partitions.items():
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        total_count = sum(parameter.numel() for parameter in parameters)
        trainable_count = sum(parameter.numel() for parameter in trainable)
        summary[name] = {
            "total_parameters": total_count,
            "trainable_parameters": trainable_count,
            "frozen": trainable_count == 0,
            "learning_rate": lr_by_group[name],
        }
        if trainable:
            optimizer_groups.append(
                {
                    "params": trainable,
                    "lr": lr_by_group[name],
                    "weight_decay": weight_decay,
                    "group_name": name,
                }
            )

    if not optimizer_groups:
        raise ValueError("Every parameter group is frozen; there is nothing to train")

    optimizer = torch.optim.AdamW(optimizer_groups, betas=betas)
    return optimizer, summary
