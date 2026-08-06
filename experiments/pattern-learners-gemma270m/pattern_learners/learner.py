from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class LearnerLayout:
    """Shape and placement of one named pattern learner."""

    mode: str = "single"  # single | all
    bottleneck_dim: int = 640
    single_layer: int = -1
    dropout: float = 0.0

    def validate(self) -> None:
        if self.mode not in {"single", "all"}:
            raise ValueError(f"Unsupported learner mode: {self.mode!r}")
        if self.bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class ResidualPatternLearner(nn.Module):
    """A zero-effect bottleneck residual module."""

    def __init__(self, hidden_size: int, bottleneck_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_size <= 0 or bottleneck_dim <= 0:
            raise ValueError("hidden_size and bottleneck_dim must be positive")

        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.down = nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.ones(()))

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        update = self.up(self.dropout(self.activation(self.down(hidden_states))))
        return hidden_states + self.gate.to(update.dtype) * update

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class _PatternModules(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        layout: LearnerLayout,
    ) -> None:
        super().__init__()
        layout.validate()
        self.layout = layout
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if layout.mode == "single":
            layer_index = normalize_layer_index(layout.single_layer, num_layers)
            self.layer_indices = (layer_index,)
        else:
            self.layer_indices = tuple(range(num_layers))

        self.modules_by_layer = nn.ModuleDict(
            {
                str(layer_index): ResidualPatternLearner(
                    hidden_size=hidden_size,
                    bottleneck_dim=layout.bottleneck_dim,
                    dropout=layout.dropout,
                )
                for layer_index in self.layer_indices
            }
        )

    def apply(self, layer_index: int, output: Any) -> Any:
        key = str(layer_index)
        if key not in self.modules_by_layer:
            return output
        learner = self.modules_by_layer[key]

        if torch.is_tensor(output):
            return learner(output)
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            return (learner(output[0]), *output[1:])
        raise TypeError(
            "Pattern learner expected a decoder layer output tensor or a tuple "
            f"whose first item is a tensor, received {type(output)!r}."
        )


class PatternLearnerSystem(nn.Module):
    """Named residual learners attached after Transformer decoder layers."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        layout: LearnerLayout,
    ) -> None:
        super().__init__()
        layout.validate()
        if hidden_size <= 0 or num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layout = layout
        self.patterns = nn.ModuleDict()
        self.active_pattern: str | None = None
        self._hook_handles: list[Any] = []
        self._attached_model_id: int | None = None

    def add_pattern(self, name: str, *, activate: bool = True) -> None:
        cleaned = validate_pattern_name(name)
        if cleaned in self.patterns:
            raise ValueError(f"Pattern learner {cleaned!r} already exists")
        self.patterns[cleaned] = _PatternModules(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            layout=self.layout,
        )
        if activate:
            self.active_pattern = cleaned

    def set_active_pattern(self, name: str | None) -> None:
        if name is None:
            self.active_pattern = None
            return
        cleaned = validate_pattern_name(name)
        if cleaned not in self.patterns:
            raise KeyError(f"Unknown pattern learner: {cleaned!r}")
        self.active_pattern = cleaned

    def attach(self, model: nn.Module) -> None:
        """Attach forward hooks after decoder layers and register on the model."""
        if self._hook_handles:
            if self._attached_model_id == id(model):
                return
            raise RuntimeError("PatternLearnerSystem is already attached to another model")

        layers = find_decoder_layers(model)
        if len(layers) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} decoder layers, found {len(layers)}"
            )

        setattr(model, "pattern_learner_system", self)
        self._attached_model_id = id(model)

        for layer_index, layer in enumerate(layers):
            handle = layer.register_forward_hook(self._make_hook(layer_index))
            self._hook_handles.append(handle)

    def detach(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._attached_model_id = None

    def _make_hook(self, layer_index: int) -> Callable[[nn.Module, tuple[Any, ...], Any], Any]:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            if self.active_pattern is None:
                return output
            return self.patterns[self.active_pattern].apply(layer_index, output)

        return hook

    def learner_parameter_count(self, pattern_name: str | None = None) -> int:
        if pattern_name is None:
            modules: Iterable[nn.Module] = self.patterns.values()
        else:
            modules = (self.patterns[pattern_name],)
        return sum(parameter.numel() for module in modules for parameter in module.parameters())

    def metadata(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "layout": asdict(self.layout),
            "patterns": list(self.patterns.keys()),
            "active_pattern": self.active_pattern,
        }

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "pattern_learners.pt")
        (path / "pattern_learners.json").write_text(
            json.dumps(self.metadata(), indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path, model: nn.Module) -> "PatternLearnerSystem":
        path = Path(directory)
        metadata = json.loads((path / "pattern_learners.json").read_text(encoding="utf-8"))
        system = cls(
            hidden_size=int(metadata["hidden_size"]),
            num_layers=int(metadata["num_layers"]),
            layout=LearnerLayout(**metadata["layout"]),
        )
        for pattern_name in metadata["patterns"]:
            system.add_pattern(pattern_name, activate=False)
        system.load_state_dict(torch.load(path / "pattern_learners.pt", map_location="cpu"))
        system.set_active_pattern(metadata.get("active_pattern"))
        system.attach(model)
        return system


def validate_pattern_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Pattern name cannot be empty")
    if "." in cleaned:
        raise ValueError("Pattern names cannot contain '.' because ModuleDict reserves it")
    return cleaned


def normalize_layer_index(index: int, num_layers: int) -> int:
    normalized = index if index >= 0 else num_layers + index
    if normalized < 0 or normalized >= num_layers:
        raise IndexError(f"Layer index {index} is outside a {num_layers}-layer model")
    return normalized


def find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Find common Hugging Face decoder-layer locations, including Gemma 3."""
    candidates = (
        ("model", "layers"),
        ("model", "model", "layers"),
        ("model", "text_model", "layers"),
        ("transformer", "h"),
    )
    for path in candidates:
        current: Any = model
        try:
            for name in path:
                current = getattr(current, name)
        except AttributeError:
            continue
        if isinstance(current, nn.ModuleList):
            return current
    raise AttributeError(
        "Could not locate decoder layers. Expected model.model.layers for Gemma 3 "
        "or another supported Hugging Face layout."
    )


def infer_model_shape(model: nn.Module) -> tuple[int, int]:
    config = getattr(model, "config", None)
    if config is None:
        raise AttributeError("Model has no config")
    hidden_size = getattr(config, "hidden_size", None)
    num_layers = getattr(config, "num_hidden_layers", None)
    if hidden_size is None or num_layers is None:
        text_config = getattr(config, "text_config", None)
        hidden_size = hidden_size or getattr(text_config, "hidden_size", None)
        num_layers = num_layers or getattr(text_config, "num_hidden_layers", None)
    if hidden_size is None or num_layers is None:
        raise AttributeError("Could not infer hidden_size and num_hidden_layers")
    return int(hidden_size), int(num_layers)


def matched_all_layer_dim(single_dim: int, num_layers: int) -> int:
    """Nearest all-layer bottleneck width with approximately matched parameters."""
    if single_dim <= 0 or num_layers <= 0:
        raise ValueError("single_dim and num_layers must be positive")
    return max(1, round(single_dim / num_layers))
