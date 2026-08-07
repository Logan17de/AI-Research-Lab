from __future__ import annotations

from dataclasses import dataclass
import math
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GPTNeoXConfig, GPTNeoXForCausalLM

from model import ModelStats


TOKEN_MOD_ARCHITECTURE_VERSION = "token_mod_v1_2_grouped"
PLASTIC_TOKEN_MOD_ARCHITECTURE_VERSION = "token_mod_v1_3_plastic"
LEGACY_TOKEN_MOD_ARCHITECTURE_VERSION = "token_mod_v1_1"
_MOD_STATE_PREFIXES = (
    "input_modifier.",
    "output_modifier.",
    "attention_modifiers.",
    "attention_modifier_projections.",
    "ffn_modifiers.",
    "ffn_modifier_projections.",
)


def _make_scale(value: float, learnable: bool) -> nn.Parameter | torch.Tensor:
    if not math.isfinite(float(value)):
        raise ValueError(f"MOD scale must be finite, got {value}.")
    if learnable and float(value) == 0.0:
        raise ValueError(
            "Learnable MOD scales must start nonzero because MOD projections start at zero."
        )
    scale = torch.tensor(float(value), dtype=torch.float32)
    return nn.Parameter(scale) if learnable else scale


class _ModifierRuntime:
    def __init__(self) -> None:
        self.input_ids: torch.Tensor | None = None


class TokenResidualModifier(nn.Module):
    """Input-side token memory with one shared projection into hidden space."""

    def __init__(
        self,
        vocab_size: int,
        mod_dim: int,
        hidden_size: int,
        scale: float,
        learnable_scale: bool,
    ) -> None:
        super().__init__()
        self.mod_dim = int(mod_dim)
        self.table = nn.Embedding(vocab_size, mod_dim)
        self.proj = nn.Linear(mod_dim, hidden_size, bias=False)
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.weight)
        scale_value = _make_scale(scale, learnable_scale)
        if isinstance(scale_value, nn.Parameter):
            self.scale = scale_value
        else:
            self.register_buffer("scale", scale_value, persistent=True)
        self.enabled = True

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        delta = self.proj(self.table(input_ids))
        return delta * self.scale.to(dtype=delta.dtype)


class OutputTokenModifier(nn.Module):
    """Independent low-dimensional residual vocabulary scorer."""

    def __init__(
        self,
        vocab_size: int,
        mod_dim: int,
        hidden_size: int,
        scale: float,
        learnable_scale: bool,
    ) -> None:
        super().__init__()
        self.mod_dim = int(mod_dim)
        self.table = nn.Embedding(vocab_size, mod_dim)
        self.proj = nn.Linear(hidden_size, mod_dim, bias=False)
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.weight)
        scale_value = _make_scale(scale, learnable_scale)
        if isinstance(scale_value, nn.Parameter):
            self.scale = scale_value
        else:
            self.register_buffer("scale", scale_value, persistent=True)
        self.enabled = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = self.proj(hidden_states.to(dtype=self.proj.weight.dtype))
        logits = F.linear(projected, self.table.weight)
        return logits * self.scale.to(dtype=logits.dtype)


class SharedTokenMemory(nn.Module):
    """One vocabulary table shared by a contiguous group of transformer layers."""

    def __init__(self, vocab_size: int, mod_dim: int) -> None:
        super().__init__()
        self.mod_dim = int(mod_dim)
        self.table = nn.Embedding(vocab_size, mod_dim)
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)
        self.enabled = True

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.table(input_ids)


class LayerTokenProjection(nn.Module):
    """Layer-specific reader for its group's shared token-memory table."""

    def __init__(
        self,
        memory: SharedTokenMemory,
        hidden_size: int,
        scale: float,
        learnable_scale: bool,
    ) -> None:
        super().__init__()
        self._memory_ref = weakref.ref(memory)
        self.proj = nn.Linear(memory.mod_dim, hidden_size, bias=False)
        nn.init.zeros_(self.proj.weight)
        scale_value = _make_scale(scale, learnable_scale)
        if isinstance(scale_value, nn.Parameter):
            self.scale = scale_value
        else:
            self.register_buffer("scale", scale_value, persistent=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        memory = self._memory_ref()
        if memory is None:
            raise RuntimeError("The shared token-memory table is no longer available.")
        delta = self.proj(memory(input_ids))
        return delta * self.scale.to(dtype=delta.dtype)


class _AttentionModifierWrapper(nn.Module):
    """Adds a layer-specific Attention MOD after the frozen attention output."""

    def __init__(
        self,
        base_module: nn.Module,
        memory: SharedTokenMemory,
        reader: LayerTokenProjection,
        runtime: _ModifierRuntime,
    ) -> None:
        super().__init__()
        self.base_module = base_module
        self._memory_ref = weakref.ref(memory)
        self._reader_ref = weakref.ref(reader)
        self._runtime = runtime

    def forward(self, *args, **kwargs):
        output = self.base_module(*args, **kwargs)
        memory = self._memory_ref()
        reader = self._reader_ref()
        input_ids = self._runtime.input_ids
        if memory is None or reader is None or not memory.enabled or input_ids is None:
            return output
        delta = reader(input_ids).to(dtype=output[0].dtype)
        if delta.shape != output[0].shape:
            raise RuntimeError(f"Attention MOD shape {tuple(delta.shape)} != attention output {tuple(output[0].shape)}")
        return (output[0] + delta, *output[1:])


class _FFNModifierWrapper(nn.Module):
    """Adds a layer-specific FFN MOD after the frozen dense_4h_to_h output."""

    def __init__(
        self,
        base_module: nn.Module,
        memory: SharedTokenMemory,
        reader: LayerTokenProjection,
        runtime: _ModifierRuntime,
    ) -> None:
        super().__init__()
        self.base_module = base_module
        self._memory_ref = weakref.ref(memory)
        self._reader_ref = weakref.ref(reader)
        self._runtime = runtime

    def forward(self, *args, **kwargs):
        output = self.base_module(*args, **kwargs)
        memory = self._memory_ref()
        reader = self._reader_ref()
        input_ids = self._runtime.input_ids
        if memory is None or reader is None or not memory.enabled or input_ids is None:
            return output
        delta = reader(input_ids).to(dtype=output.dtype)
        if delta.shape != output.shape:
            raise RuntimeError(f"FFN MOD shape {tuple(delta.shape)} != MLP output {tuple(output.shape)}")
        return output + delta


class _OutputModifierWrapper(nn.Module):
    """Adds Output MOD residual logits before the model's single softmax."""

    def __init__(self, base_module: nn.Module, modifier: OutputTokenModifier) -> None:
        super().__init__()
        self.base_module = base_module
        self._modifier_ref = weakref.ref(modifier)
        self.last_base_logit_norm: torch.Tensor | None = None
        self.last_mod_logit_norm: torch.Tensor | None = None

    @property
    def weight(self) -> nn.Parameter:
        return self.base_module.weight

    @property
    def bias(self):
        return getattr(self.base_module, "bias", None)

    @property
    def in_features(self) -> int:
        return int(self.base_module.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base_module.out_features)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_logits = self.base_module(hidden_states)
        modifier = self._modifier_ref()
        if modifier is None or not modifier.enabled:
            self.last_base_logit_norm = base_logits.detach().float().norm()
            self.last_mod_logit_norm = base_logits.new_zeros(())
            return base_logits
        mod_logits = modifier(hidden_states).to(dtype=base_logits.dtype)
        if mod_logits.shape != base_logits.shape:
            raise RuntimeError(f"Output MOD shape {tuple(mod_logits.shape)} != base logits {tuple(base_logits.shape)}")
        self.last_base_logit_norm = base_logits.detach().float().norm()
        self.last_mod_logit_norm = mod_logits.detach().float().norm()
        return base_logits + mod_logits


@dataclass(frozen=True)
class PythiaParameterReport:
    total_parameters: int
    frozen_base_parameters: int
    trainable_parameters: int
    trainable_mod_parameters: int
    trainable_plastic_parameters: int
    active_parameters_per_token: int
    physically_accessed_parameters_per_token: int
    parameter_memory_bytes: int
    trainable_percentage: float
    trainable_names: tuple[str, ...]


class PythiaModModel(nn.Module):
    """Grouped Token MOD architecture for GPT-NeoX/Pythia."""

    architecture_version = TOKEN_MOD_ARCHITECTURE_VERSION

    def __init__(
        self,
        base_model: GPTNeoXForCausalLM,
        *,
        emb_mod_dim: int = 0,
        out_mod_dim: int = 0,
        attn_mod_dim: int = 0,
        ffn_mod_dim: int = 0,
        attn_unique_mod_count: int = 1,
        ffn_unique_mod_count: int = 1,
        emb_mod_scale: float = 1.0,
        out_mod_scale: float = 1.0,
        attn_mod_scale: float = 1.0,
        ffn_mod_scale: float = 1.0,
        learnable_mod_scales: bool = False,
        base_model_name: str = "",
        base_revision: str | None = None,
        original_vocab_size: int | None = None,
        control_token_ids: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        if getattr(base_model.config, "model_type", None) != "gpt_neox":
            raise TypeError(f"Pythia MOD requires GPT-NeoX, got model_type={base_model.config.model_type!r}")
        self.base_model = base_model
        self.config = base_model.config
        self.base_model_name = base_model_name
        self.base_revision = base_revision
        self.hidden_size = int(self.config.hidden_size)
        self.vocab_size = int(base_model.get_input_embeddings().num_embeddings)
        self.num_layers = len(self.base_model.gpt_neox.layers)
        self.original_vocab_size = int(original_vocab_size or self.vocab_size)
        self.control_token_ids = tuple(int(token_id) for token_id in control_token_ids)
        self._plastic_parameter_ids: set[int] = set()
        self._plastic_norm_parameter_ids: set[int] = set()
        self._runtime = _ModifierRuntime()
        self.attn_unique_mod_count = self._validate_unique_mod_count(
            "attention", attn_mod_dim, attn_unique_mod_count
        )
        self.ffn_unique_mod_count = self._validate_unique_mod_count(
            "ffn", ffn_mod_dim, ffn_unique_mod_count
        )

        self.input_modifier = self._build_input_modifier(
            emb_mod_dim, emb_mod_scale, learnable_mod_scales
        )
        self.output_modifier = self._build_output_modifier(
            out_mod_dim, out_mod_scale, learnable_mod_scales
        )
        (
            self.attention_modifiers,
            self.attention_modifier_projections,
            self.attention_modifier_layer_map,
        ) = self._build_layer_family(
            attn_mod_dim,
            attn_mod_scale,
            learnable_mod_scales,
            self.attn_unique_mod_count,
        )
        (
            self.ffn_modifiers,
            self.ffn_modifier_projections,
            self.ffn_modifier_layer_map,
        ) = self._build_layer_family(
            ffn_mod_dim,
            ffn_mod_scale,
            learnable_mod_scales,
            self.ffn_unique_mod_count,
        )
        self._attach_modifiers()
        self.assert_architecture_invariants()

    @property
    def embedding_modifier(self) -> TokenResidualModifier | None:
        return self.input_modifier

    @property
    def attention_modifier(self) -> SharedTokenMemory | None:
        """Compatibility accessor for the first Attention MOD table."""
        return self.attention_modifiers[0] if self.attention_modifiers else None

    @property
    def ffn_modifier(self) -> SharedTokenMemory | None:
        """Compatibility accessor for the first FFN MOD table."""
        return self.ffn_modifiers[0] if self.ffn_modifiers else None

    def _validate_unique_mod_count(self, name: str, mod_dim: int, count: int) -> int:
        count = int(count)
        if count < 1:
            raise ValueError(f"{name} unique MOD count must be at least 1, got {count}.")
        if int(mod_dim) <= 0:
            if count != 1:
                raise ValueError(
                    f"{name} unique MOD count {count} requires an enabled {name} MOD dimension."
                )
            return count
        if count > self.num_layers:
            raise ValueError(
                f"{name} unique MOD count {count} exceeds decoder layer count {self.num_layers}."
            )
        if self.num_layers % count != 0:
            raise ValueError(
                f"{name} unique MOD count {count} cannot evenly split {self.num_layers} layers."
            )
        return count

    def _build_input_modifier(
        self, mod_dim: int, scale: float, learnable_scale: bool
    ) -> TokenResidualModifier | None:
        if int(mod_dim) <= 0:
            return None
        return TokenResidualModifier(
            self.vocab_size, int(mod_dim), self.hidden_size, scale, learnable_scale
        )

    def _build_output_modifier(
        self, mod_dim: int, scale: float, learnable_scale: bool
    ) -> OutputTokenModifier | None:
        if int(mod_dim) <= 0:
            return None
        return OutputTokenModifier(
            self.vocab_size, int(mod_dim), self.hidden_size, scale, learnable_scale
        )

    def _build_layer_family(
        self,
        mod_dim: int,
        scale: float,
        learnable_scale: bool,
        unique_count: int,
    ) -> tuple[nn.ModuleList, nn.ModuleList, tuple[int, ...]]:
        if int(mod_dim) <= 0:
            return nn.ModuleList(), nn.ModuleList(), ()
        memories = nn.ModuleList(
            SharedTokenMemory(self.vocab_size, int(mod_dim)) for _ in range(unique_count)
        )
        layers_per_group = self.num_layers // unique_count
        layer_map = tuple(layer_index // layers_per_group for layer_index in range(self.num_layers))
        readers = nn.ModuleList(
            LayerTokenProjection(
                memories[layer_map[layer_index]], self.hidden_size, scale, learnable_scale
            )
            for layer_index in range(self.num_layers)
        )
        return memories, readers, layer_map

    def _attach_modifiers(self) -> None:
        for layer_index, layer in enumerate(self.base_model.gpt_neox.layers):
            if self.attention_modifiers:
                memory = self.attention_modifiers[self.attention_modifier_layer_map[layer_index]]
                layer.attention = _AttentionModifierWrapper(
                    layer.attention,
                    memory,
                    self.attention_modifier_projections[layer_index],
                    self._runtime,
                )
            if self.ffn_modifiers:
                memory = self.ffn_modifiers[self.ffn_modifier_layer_map[layer_index]]
                layer.mlp = _FFNModifierWrapper(
                    layer.mlp,
                    memory,
                    self.ffn_modifier_projections[layer_index],
                    self._runtime,
                )
        if self.output_modifier is not None:
            self.base_model.embed_out = _OutputModifierWrapper(
                self.base_model.embed_out, self.output_modifier
            )

    def assert_architecture_invariants(self) -> None:
        if self.input_modifier is not None and self.output_modifier is not None:
            if self.input_modifier.table.weight is self.output_modifier.table.weight:
                raise RuntimeError("Input and Output MOD token tables must be independent.")
            if self.input_modifier.proj.weight is self.output_modifier.proj.weight:
                raise RuntimeError("Input and Output MOD projections must be independent.")
        for memories, readers, layer_map, count, name in (
            (
                self.attention_modifiers,
                self.attention_modifier_projections,
                self.attention_modifier_layer_map,
                self.attn_unique_mod_count,
                "attention",
            ),
            (
                self.ffn_modifiers,
                self.ffn_modifier_projections,
                self.ffn_modifier_layer_map,
                self.ffn_unique_mod_count,
                "ffn",
            ),
        ):
            expected = self.num_layers if memories else 0
            if len(readers) != expected:
                raise RuntimeError(f"{name} MOD requires {expected} layer readers, got {len(readers)}.")
            if not memories:
                if layer_map:
                    raise RuntimeError(f"Disabled {name} MOD must not have a layer map.")
                continue
            if len(memories) != count:
                raise RuntimeError(f"{name} MOD requires {count} tables, got {len(memories)}.")
            layers_per_group = self.num_layers // count
            expected_map = tuple(index // layers_per_group for index in range(self.num_layers))
            if layer_map != expected_map:
                raise RuntimeError(f"{name} MOD layer map is not the required contiguous mapping.")
            if len({id(memory.table.weight) for memory in memories}) != count:
                raise RuntimeError(f"{name} MOD tables must be distinct between groups.")
            for layer_index, reader in enumerate(readers):
                if reader._memory_ref() is not memories[layer_map[layer_index]]:
                    raise RuntimeError(f"{name} MOD reader {layer_index} references the wrong table.")
            projection_ids = {id(reader.proj.weight) for reader in readers}
            if len(projection_ids) != len(readers):
                raise RuntimeError(f"{name} MOD projections must be distinct for every layer.")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        vocab_size: int | None = None,
        revision: str | None = None,
        torch_dtype: torch.dtype | None = None,
        gradient_checkpointing: bool = False,
        control_token_ids: tuple[int, ...] = (),
        **modifier_kwargs,
    ) -> "PythiaModModel":
        load_kwargs = {"revision": revision} if revision else {}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if not isinstance(base_model, GPTNeoXForCausalLM):
            raise TypeError(f"Expected GPTNeoXForCausalLM for {model_name}, got {type(base_model).__name__}")
        original_vocab_size = int(base_model.get_input_embeddings().num_embeddings)
        if vocab_size is not None and vocab_size > original_vocab_size:
            try:
                base_model.resize_token_embeddings(vocab_size, mean_resizing=False)
            except TypeError:
                base_model.resize_token_embeddings(vocab_size)
        if gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
            base_model.config.use_cache = False
        return cls(
            base_model,
            base_model_name=model_name,
            base_revision=revision,
            original_vocab_size=original_vocab_size,
            control_token_ids=control_token_ids,
            **modifier_kwargs,
        )

    @classmethod
    def from_config(cls, config: GPTNeoXConfig, **modifier_kwargs) -> "PythiaModModel":
        return cls(GPTNeoXForCausalLM(config), **modifier_kwargs)

    def modifier_modules(self) -> tuple[nn.Module, ...]:
        modules: list[nn.Module] = []
        for module in (
            self.input_modifier,
            self.output_modifier,
            self.attention_modifiers,
            self.attention_modifier_projections,
            self.ffn_modifiers,
            self.ffn_modifier_projections,
        ):
            if module is not None:
                modules.append(module)
        return tuple(modules)

    def modifier_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for module in self.modifier_modules():
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter)
        return parameters

    def lora_parameters(self) -> list[nn.Parameter]:
        return []

    def base_parameters(self) -> list[nn.Parameter]:
        modifier_ids = {id(parameter) for parameter in self.modifier_parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in modifier_ids]

    def freeze_base_model(self) -> None:
        self._plastic_parameter_ids.clear()
        self._plastic_norm_parameter_ids.clear()
        for parameter in self.base_parameters():
            parameter.requires_grad_(False)
        for parameter in self.modifier_parameters():
            parameter.requires_grad_(True)
        self.assert_frozen_mod_contract()
        if not all(not parameter.requires_grad for parameter in self.base_model.parameters()):
            raise RuntimeError("Strict Token MOD requires every base_model parameter to be frozen.")

    @staticmethod
    def _unwrap_base_module(module: nn.Module) -> nn.Module:
        return getattr(module, "base_module", module)

    def configure_base_plasticity(
        self,
        *,
        last_n_layers: int = 0,
        layer_norms: bool = False,
        biases: bool = False,
        attn_output: bool = False,
        ffn_output: bool = False,
        final_layer_norm: bool = False,
        lm_head: bool = False,
    ) -> tuple[str, ...]:
        """Unfreeze only explicitly selected base parameters after strict MOD freezing."""
        self.freeze_base_model()
        last_n_layers = int(last_n_layers)
        if last_n_layers < 0 or last_n_layers > self.num_layers:
            raise ValueError(
                f"plastic last_n_layers must be in [0, {self.num_layers}], got {last_n_layers}."
            )

        selected: set[int] = set()
        norm_ids: set[int] = set()

        def select(module: nn.Module, *, normalization: bool = False) -> None:
            for parameter in module.parameters():
                selected.add(id(parameter))
                if normalization:
                    norm_ids.add(id(parameter))

        layers = self.base_model.gpt_neox.layers
        if last_n_layers:
            for layer in layers[self.num_layers - last_n_layers :]:
                select(layer)
        if layer_norms:
            for layer in layers:
                select(layer.input_layernorm, normalization=True)
                select(layer.post_attention_layernorm, normalization=True)
        if biases:
            for name, parameter in self.base_model.named_parameters():
                if name.endswith(".bias"):
                    selected.add(id(parameter))
        if attn_output:
            for layer in layers:
                attention = self._unwrap_base_module(layer.attention)
                select(attention.dense)
        if ffn_output:
            for layer in layers:
                mlp = self._unwrap_base_module(layer.mlp)
                select(mlp.dense_4h_to_h)
        if final_layer_norm:
            select(self.base_model.gpt_neox.final_layer_norm, normalization=True)
        if lm_head:
            select(self._unwrap_base_module(self.base_model.embed_out))

        # LayerNorm parameters selected through a complete plastic tail still use
        # the dedicated normalization learning-rate group.
        for layer in layers:
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                norm_ids.update(id(parameter) for parameter in norm.parameters() if id(parameter) in selected)
        norm_ids.update(
            id(parameter)
            for parameter in self.base_model.gpt_neox.final_layer_norm.parameters()
            if id(parameter) in selected
        )

        base_ids = {id(parameter) for parameter in self.base_parameters()}
        unknown = selected - base_ids
        if unknown:
            raise RuntimeError("Plasticity selector included non-base parameters.")
        self._plastic_parameter_ids = selected
        self._plastic_norm_parameter_ids = norm_ids
        for parameter in self.base_parameters():
            parameter.requires_grad_(id(parameter) in selected)
        for parameter in self.modifier_parameters():
            parameter.requires_grad_(True)
        self.assert_plastic_mod_contract()
        return self.plastic_parameter_names()

    def plastic_base_parameters(self) -> list[nn.Parameter]:
        return [
            parameter for parameter in self.base_parameters()
            if id(parameter) in self._plastic_parameter_ids
        ]

    def plastic_norm_parameters(self) -> list[nn.Parameter]:
        return [
            parameter for parameter in self.base_parameters()
            if id(parameter) in self._plastic_norm_parameter_ids
        ]

    def plastic_non_norm_parameters(self) -> list[nn.Parameter]:
        return [
            parameter for parameter in self.base_parameters()
            if id(parameter) in self._plastic_parameter_ids
            and id(parameter) not in self._plastic_norm_parameter_ids
        ]

    def plastic_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name for name, parameter in self.named_parameters()
            if id(parameter) in self._plastic_parameter_ids
        )

    def assert_plastic_mod_contract(self) -> None:
        modifier_ids = {id(parameter) for parameter in self.modifier_parameters()}
        unexpected_trainable = [
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
            and id(parameter) not in modifier_ids
            and id(parameter) not in self._plastic_parameter_ids
        ]
        missing_plastic = [
            name for name, parameter in self.named_parameters()
            if id(parameter) in self._plastic_parameter_ids and not parameter.requires_grad
        ]
        if unexpected_trainable or missing_plastic:
            raise RuntimeError(
                "Plastic-MOD contract failed: "
                f"unexpected_trainable={unexpected_trainable} missing_plastic={missing_plastic}"
            )

    def enable_full_finetuning(self) -> None:
        if self.modifier_parameters():
            raise RuntimeError("Full Pythia fine-tuning must not have MOD modules attached.")
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        frozen = [name for name, parameter in self.named_parameters() if not parameter.requires_grad]
        if frozen:
            raise RuntimeError(f"Full-finetune contract failed; frozen parameters remain: {frozen}")

    def assert_frozen_mod_contract(self) -> None:
        modifier_ids = {id(parameter) for parameter in self.modifier_parameters()}
        trainable_base = [
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad and id(parameter) not in modifier_ids
        ]
        frozen_mod = [
            name for name, parameter in self.named_parameters()
            if not parameter.requires_grad and id(parameter) in modifier_ids
        ]
        if trainable_base or frozen_mod:
            raise RuntimeError(f"Frozen-MOD contract failed: trainable_base={trainable_base} frozen_mod={frozen_mod}")

    def assert_base_gradients_absent(self) -> None:
        base_gradients = [
            name for name, parameter in self.base_model.named_parameters()
            if parameter.grad is not None and id(parameter) not in self._plastic_parameter_ids
        ]
        if base_gradients:
            raise RuntimeError(
                f"Frozen base parameters received gradients, found: {base_gradients}"
            )

    def set_mod_ablation(
        self,
        *,
        disable_emb: bool = False,
        disable_out: bool = False,
        disable_attn: bool = False,
        disable_ffn: bool = False,
        disable_all: bool = False,
    ) -> None:
        for module, disabled in (
            (self.input_modifier, disable_all or disable_emb),
            (self.output_modifier, disable_all or disable_out),
        ):
            if module is not None:
                module.enabled = not disabled
        for memory in self.attention_modifiers:
            memory.enabled = not (disable_all or disable_attn)
        for memory in self.ffn_modifiers:
            memory.enabled = not (disable_all or disable_ffn)

    def model_stats(self) -> ModelStats:
        total = sum(parameter.numel() for parameter in self.parameters())
        modifiers = sum(parameter.numel() for parameter in self.modifier_parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return ModelStats(total, trainable, modifiers, 0, total - modifiers)

    def _physically_accessed_mod_parameters_per_token(self) -> int:
        total = 0
        if self.input_modifier is not None and self.input_modifier.enabled:
            total += self.input_modifier.mod_dim + self.input_modifier.proj.weight.numel()
            total += int(isinstance(self.input_modifier.scale, nn.Parameter))
        if self.output_modifier is not None and self.output_modifier.enabled:
            total += self.output_modifier.table.weight.numel() + self.output_modifier.proj.weight.numel()
            total += int(isinstance(self.output_modifier.scale, nn.Parameter))
        for memories, readers in (
            (self.attention_modifiers, self.attention_modifier_projections),
            (self.ffn_modifiers, self.ffn_modifier_projections),
        ):
            if memories and memories[0].enabled:
                total += sum(memory.mod_dim for memory in memories)
                total += sum(reader.proj.weight.numel() for reader in readers)
                total += sum(int(isinstance(reader.scale, nn.Parameter)) for reader in readers)
        return total

    def parameter_report(self) -> PythiaParameterReport:
        stats = self.model_stats()
        enabled_mod = sum(
            parameter.numel()
            for module in (
                self.input_modifier if self.input_modifier is not None and self.input_modifier.enabled else None,
                self.output_modifier if self.output_modifier is not None and self.output_modifier.enabled else None,
                self.attention_modifiers
                if self.attention_modifiers and self.attention_modifiers[0].enabled else None,
                self.attention_modifier_projections
                if self.attention_modifiers and self.attention_modifiers[0].enabled else None,
                self.ffn_modifiers if self.ffn_modifiers and self.ffn_modifiers[0].enabled else None,
                self.ffn_modifier_projections
                if self.ffn_modifiers and self.ffn_modifiers[0].enabled else None,
            )
            if module is not None
            for parameter in module.parameters()
        )
        input_weight = self.base_model.get_input_embeddings().weight
        output_weight = self.base_model.get_output_embeddings().weight
        base_physical = stats.base_params - input_weight.numel() + input_weight.shape[1]
        if output_weight is not input_weight:
            base_physical = stats.base_params - input_weight.numel() + input_weight.shape[1]
        memory = sum(parameter.numel() * parameter.element_size() for parameter in self.parameters())
        trainable_names = tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)
        return PythiaParameterReport(
            total_parameters=stats.total_params,
            frozen_base_parameters=sum(p.numel() for p in self.base_parameters() if not p.requires_grad),
            trainable_parameters=stats.trainable_params,
            trainable_mod_parameters=sum(p.numel() for p in self.modifier_parameters() if p.requires_grad),
            trainable_plastic_parameters=sum(p.numel() for p in self.plastic_base_parameters()),
            active_parameters_per_token=stats.base_params + enabled_mod,
            physically_accessed_parameters_per_token=(
                base_physical + self._physically_accessed_mod_parameters_per_token()
            ),
            parameter_memory_bytes=memory,
            trainable_percentage=100.0 * stats.trainable_params / max(stats.total_params, 1),
            trainable_names=trainable_names,
        )

    @staticmethod
    def _is_modifier_state_name(name: str) -> bool:
        return name.startswith(_MOD_STATE_PREFIXES)

    def modifier_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if self._is_modifier_state_name(name)
        }

    def load_modifier_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        current = self.state_dict()
        expected = {name for name in current if self._is_modifier_state_name(name)}
        supplied = set(state_dict)
        if expected != supplied:
            raise RuntimeError(
                f"Grouped Token MOD checkpoint mismatch: missing={sorted(expected - supplied)} "
                f"unexpected={sorted(supplied - expected)}"
            )
        current.update(state_dict)
        self.load_state_dict(current, strict=True)

    def sparse_checkpoint_state(self) -> dict[str, object]:
        input_weight = self.base_model.get_input_embeddings().weight
        output_weight = self.base_model.get_output_embeddings().weight
        control_ids = torch.tensor(self.control_token_ids, dtype=torch.long, device=input_weight.device)
        named_plastic = {
            name: parameter.detach().cpu()
            for name, parameter in self.named_parameters()
            if id(parameter) in self._plastic_parameter_ids
        }
        return {
            "architecture_version": (
                PLASTIC_TOKEN_MOD_ARCHITECTURE_VERSION if named_plastic else self.architecture_version
            ),
            "attn_unique_mod_count": self.attn_unique_mod_count,
            "ffn_unique_mod_count": self.ffn_unique_mod_count,
            "attention_modifier_layer_map": self.attention_modifier_layer_map,
            "ffn_modifier_layer_map": self.ffn_modifier_layer_map,
            "modifier_state_dict": self.modifier_state_dict(),
            "plastic_parameter_names": tuple(named_plastic),
            "plastic_state_dict": named_plastic,
            "original_vocab_size": self.original_vocab_size,
            "control_token_ids": self.control_token_ids,
            "control_input_embeddings": input_weight.index_select(0, control_ids).detach().cpu(),
            "control_output_embeddings": output_weight.index_select(0, control_ids).detach().cpu(),
        }

    def load_sparse_checkpoint_state(self, state: dict[str, object]) -> None:
        saved_version = state.get("architecture_version")
        if saved_version == LEGACY_TOKEN_MOD_ARCHITECTURE_VERSION:
            if self.attn_unique_mod_count != 1 or self.ffn_unique_mod_count != 1:
                raise RuntimeError(
                    "token_mod_v1_1 checkpoints are compatible only when Attention and FFN "
                    "unique MOD counts are both 1."
                )
            modifier_state = self._migrate_v11_modifier_state_dict(state["modifier_state_dict"])
        elif saved_version in {self.architecture_version, PLASTIC_TOKEN_MOD_ARCHITECTURE_VERSION}:
            saved_attn_count = int(state.get("attn_unique_mod_count", -1))
            saved_ffn_count = int(state.get("ffn_unique_mod_count", -1))
            saved_attn_map = tuple(int(item) for item in state.get("attention_modifier_layer_map", ()))
            saved_ffn_map = tuple(int(item) for item in state.get("ffn_modifier_layer_map", ()))
            if (
                saved_attn_count != self.attn_unique_mod_count
                or saved_ffn_count != self.ffn_unique_mod_count
                or saved_attn_map != self.attention_modifier_layer_map
                or saved_ffn_map != self.ffn_modifier_layer_map
            ):
                raise RuntimeError(
                    "Grouped Token MOD checkpoint mapping mismatch: "
                    f"saved_attn_count={saved_attn_count} current_attn_count={self.attn_unique_mod_count} "
                    f"saved_ffn_count={saved_ffn_count} current_ffn_count={self.ffn_unique_mod_count} "
                    f"saved_attn_map={saved_attn_map} current_attn_map={self.attention_modifier_layer_map} "
                    f"saved_ffn_map={saved_ffn_map} current_ffn_map={self.ffn_modifier_layer_map}"
                )
            modifier_state = state["modifier_state_dict"]
        else:
            raise RuntimeError(
                f"Pythia MOD architecture mismatch: saved={saved_version!r} "
                f"expected={self.architecture_version!r}."
            )
        original_vocab_size = int(state.get("original_vocab_size", -1))
        if original_vocab_size != self.original_vocab_size:
            raise RuntimeError(
                f"Sparse checkpoint base vocabulary mismatch: saved={original_vocab_size} current={self.original_vocab_size}"
            )
        self.load_modifier_state_dict(modifier_state)
        saved_plastic_names = tuple(state.get("plastic_parameter_names", ()))
        current_plastic_names = self.plastic_parameter_names()
        if saved_version == PLASTIC_TOKEN_MOD_ARCHITECTURE_VERSION:
            if saved_plastic_names != current_plastic_names:
                raise RuntimeError(
                    "Plastic checkpoint parameter selection mismatch: "
                    f"saved={saved_plastic_names} current={current_plastic_names}"
                )
            plastic_state = state.get("plastic_state_dict")
            if not isinstance(plastic_state, dict) or set(plastic_state) != set(current_plastic_names):
                raise RuntimeError("Plastic checkpoint state is missing or does not match its parameter names.")
            current_named = dict(self.named_parameters())
            with torch.no_grad():
                for name in current_plastic_names:
                    source = plastic_state[name]
                    target = current_named[name]
                    if tuple(source.shape) != tuple(target.shape):
                        raise RuntimeError(
                            f"Plastic checkpoint shape mismatch for {name}: "
                            f"saved={tuple(source.shape)} current={tuple(target.shape)}"
                        )
                    target.copy_(source.to(target))
        elif current_plastic_names:
            raise RuntimeError(
                f"Checkpoint {saved_version!r} contains no plastic base state, but the current run enables "
                f"{len(current_plastic_names)} plastic tensors."
            )
        saved_control_ids = tuple(int(token_id) for token_id in state.get("control_token_ids", ()))
        if saved_control_ids != self.control_token_ids:
            raise RuntimeError(
                f"Sparse checkpoint control IDs mismatch: saved={saved_control_ids} current={self.control_token_ids}"
            )
        input_rows = state["control_input_embeddings"]
        output_rows = state["control_output_embeddings"]
        if input_rows.shape[0] != len(self.control_token_ids) or output_rows.shape[0] != len(self.control_token_ids):
            raise RuntimeError("Sparse checkpoint control-token row count does not match the active tokenizer.")
        control_ids = torch.tensor(
            self.control_token_ids,
            dtype=torch.long,
            device=self.base_model.get_input_embeddings().weight.device,
        )
        with torch.no_grad():
            self.base_model.get_input_embeddings().weight.index_copy_(
                0, control_ids, input_rows.to(self.base_model.get_input_embeddings().weight)
            )
            self.base_model.get_output_embeddings().weight.index_copy_(
                0, control_ids, output_rows.to(self.base_model.get_output_embeddings().weight)
            )

    @staticmethod
    def _migrate_v11_modifier_state_dict(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        migrated: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            if name.startswith("attention_modifier."):
                name = "attention_modifiers.0." + name.removeprefix("attention_modifier.")
            elif name.startswith("ffn_modifier."):
                name = "ffn_modifiers.0." + name.removeprefix("ffn_modifier.")
            migrated[name] = tensor
        return migrated

    def mod_group_summary_lines(self) -> tuple[str, str]:
        def render(name: str, memories: nn.ModuleList, layer_map: tuple[int, ...]) -> str:
            if not memories:
                return f"{name} MOD: disabled"
            layers_per_group = self.num_layers // len(memories)
            return (
                f"{name} MOD: groups={len(memories)} layers/group={layers_per_group} "
                f"map={list(layer_map)}"
            )

        return (
            render("attention", self.attention_modifiers, self.attention_modifier_layer_map),
            render("ffn", self.ffn_modifiers, self.ffn_modifier_layer_map),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError(f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}")
        if labels is not None and (past_key_values is not None or use_cache):
            raise ValueError("Training labels cannot be combined with cached decoding.")
        self._runtime.input_ids = input_ids
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        if self.input_modifier is not None and self.input_modifier.enabled:
            inputs_embeds = inputs_embeds + self.input_modifier(input_ids).to(dtype=inputs_embeds.dtype)
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        result = {"logits": outputs.logits}
        if isinstance(self.base_model.embed_out, _OutputModifierWrapper):
            result["base_logit_norm"] = self.base_model.embed_out.last_base_logit_norm
            result["output_mod_logit_norm"] = self.base_model.embed_out.last_mod_logit_norm
        if use_cache:
            result["past_key_values"] = outputs.past_key_values
        if output_hidden_states:
            result["hidden_states"] = outputs.hidden_states
        if labels is not None:
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_sum = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            target_count = shift_labels.ne(-100).sum()
            if int(target_count.item()) == 0:
                raise ValueError("A training batch must contain at least one supervised next-token target.")
            result.update(loss=loss_sum / target_count, loss_sum=loss_sum, target_count=target_count)
        return result
