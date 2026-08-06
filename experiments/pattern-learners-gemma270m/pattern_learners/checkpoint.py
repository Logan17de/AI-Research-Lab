from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from .learner import PatternLearnerSystem
from .optimization import parameter_partitions


def save_experiment_checkpoint(
    output_directory: str | Path,
    model: nn.Module,
    learner_system: PatternLearnerSystem,
    tokenizer: Any,
    *,
    training_summary: dict[str, Any],
    inherited_base_parameter_names: Iterable[str] = (),
) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    learner_system.save(output / "learner")
    tokenizer.save_pretrained(output / "tokenizer")

    partitions = parameter_partitions(model, learner_system)
    learner_ids = {id(parameter) for parameter in partitions["learner"]}
    inherited_names = set(inherited_base_parameter_names)
    base_parameters = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if id(parameter) not in learner_ids
        and (parameter.requires_grad or name in inherited_names)
    }
    torch.save(base_parameters, output / "trained_base_parameters.pt")
    (output / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2), encoding="utf-8"
    )


def load_trained_base_parameters(
    model: nn.Module,
    checkpoint_path: str | Path,
) -> set[str]:
    state = torch.load(checkpoint_path, map_location="cpu")
    named_parameters = dict(model.named_parameters())
    missing = [name for name in state if name not in named_parameters]
    if missing:
        raise KeyError(f"Checkpoint parameters are absent from the model: {missing[:5]}")
    with torch.no_grad():
        for name, tensor in state.items():
            parameter = named_parameters[name]
            if parameter.shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: model={tuple(parameter.shape)} "
                    f"checkpoint={tuple(tensor.shape)}"
                )
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    return set(state)
