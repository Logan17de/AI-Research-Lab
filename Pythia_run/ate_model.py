from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import re
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GPTNeoXForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import (
    ALL_ATTENTION_FUNCTIONS,
    GPTNeoXLayer,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from pythia_model import PythiaModModel


LEGACY_ATE_ARCHITECTURE_VERSION = "pythia_ate_v1"
ATE_ARCHITECTURE_VERSION = "pythia_ate_v2_staged"
NORMALIZATION_STRATEGY = "stage_segmented_layernorm"


def _normal_parameter(reference: torch.Tensor, shape: tuple[int, ...], std: float) -> nn.Parameter:
    value = torch.empty(shape, device=reference.device, dtype=reference.dtype)
    nn.init.normal_(value, mean=0.0, std=std)
    return nn.Parameter(value)


def _output_parameter(
    reference: torch.Tensor,
    shape: tuple[int, ...],
    mode: str,
    scale: float,
) -> nn.Parameter:
    if mode == "exact":
        return nn.Parameter(torch.zeros(shape, device=reference.device, dtype=reference.dtype))
    if mode != "small":
        raise ValueError(f"Unknown ATE output initialization {mode!r}.")
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("--ate-output-init small requires a finite positive scale.")
    return _normal_parameter(reference, shape, scale)


class ExpandedLinear(nn.Module):
    """One stage of a block-factorized linear operation.

    ``base`` contains every earlier stage unchanged. The outer blocks add the
    newest source/destination segment. This nesting is the stage boundary.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        new_in_features: int,
        new_out_features: int,
        initializer_range: float,
        output_init: str = "exact",
        output_init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.old_in_features = int(base.in_features)
        self.old_out_features = int(base.out_features)
        self.new_in_features = int(new_in_features)
        self.new_out_features = int(new_out_features)
        self.in_features = self.old_in_features + self.new_in_features
        self.out_features = self.old_out_features + self.new_out_features
        reference = base.weight
        # Newest-stage channels cannot alter any earlier destination at exact init.
        self.new_to_old = _output_parameter(
            reference,
            (self.old_out_features, self.new_in_features),
            output_init,
            output_init_scale,
        )
        # Existing channels can create useful newest-stage features immediately.
        self.old_to_new = _normal_parameter(
            reference, (self.new_out_features, self.old_in_features), initializer_range
        )
        self.new_to_new = _normal_parameter(
            reference, (self.new_out_features, self.new_in_features), initializer_range
        )
        self.new_bias = (
            nn.Parameter(torch.zeros(self.new_out_features, device=reference.device, dtype=reference.dtype))
            if base.bias is not None
            else None
        )

    @property
    def weight(self) -> torch.Tensor:
        top = torch.cat((self.base.weight, self.new_to_old), dim=1)
        bottom = torch.cat((self.old_to_new, self.new_to_new), dim=1)
        return torch.cat((top, bottom), dim=0)

    @property
    def bias(self) -> torch.Tensor | None:
        if self.base.bias is None:
            return None
        return torch.cat((self.base.bias, self.new_bias), dim=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        old_inputs, new_inputs = inputs.split(
            (self.old_in_features, self.new_in_features), dim=-1
        )
        old_output = self.base(old_inputs) + F.linear(new_inputs, self.new_to_old)
        new_output = F.linear(old_inputs, self.old_to_new)
        new_output = new_output + F.linear(new_inputs, self.new_to_new, self.new_bias)
        return torch.cat((old_output, new_output), dim=-1)


class StageQKVProjection(nn.Module):
    """QKV projection for only the newest attention-head group."""

    def __init__(
        self,
        *,
        old_features: int,
        new_features: int,
        new_heads: int,
        head_size: int,
        reference: torch.Tensor,
        initializer_range: float,
        bias: bool,
    ) -> None:
        super().__init__()
        self.old_features = int(old_features)
        self.new_features = int(new_features)
        self.new_heads = int(new_heads)
        self.head_size = int(head_size)
        self.in_features = self.old_features + self.new_features
        self.out_features = 3 * self.new_heads * self.head_size
        self.old_to_new = _normal_parameter(
            reference, (self.out_features, self.old_features), initializer_range
        )
        self.new_to_new = _normal_parameter(
            reference, (self.out_features, self.new_features), initializer_range
        )
        self.bias = (
            nn.Parameter(torch.zeros(self.out_features, device=reference.device, dtype=reference.dtype))
            if bias
            else None
        )

    @property
    def weight(self) -> torch.Tensor:
        return torch.cat((self.old_to_new, self.new_to_new), dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        old, new = hidden_states.split((self.old_features, self.new_features), dim=-1)
        result = F.linear(old, self.old_to_new)
        return result + F.linear(new, self.new_to_new, self.bias)


class StageExpandedAttention(nn.Module):
    """Adds a head group without changing earlier groups' execution shape."""

    def __init__(
        self,
        base_attention: nn.Module,
        *,
        new_hidden: int,
        new_heads: int,
        head_size: int,
        initializer_range: float,
        output_init: str,
        output_init_scale: float,
    ) -> None:
        super().__init__()
        if new_hidden != new_heads * head_size:
            raise ValueError(
                f"Attention stage width {new_hidden} does not match "
                f"{new_heads} heads x {head_size}."
            )
        # Later global config updates must not change the checkpointed group.
        base_attention.config = copy.deepcopy(base_attention.config)
        self.base_attention = base_attention
        self.old_hidden = self._attention_hidden_size(base_attention)
        self.new_hidden = int(new_hidden)
        self.total_hidden = self.old_hidden + self.new_hidden
        self.new_heads = int(new_heads)
        self.head_size = int(head_size)
        self.scaling = self.head_size**-0.5
        self.rotary_ndims = int(getattr(base_attention, "rotary_ndims"))
        self.attention_dropout = float(getattr(base_attention, "attention_dropout"))
        self.is_causal = True
        self.layer_idx = int(getattr(base_attention, "layer_idx", 0) or 0)
        self.config = copy.deepcopy(base_attention.config)

        reference = next(base_attention.parameters())
        qkv = getattr(base_attention, "query_key_value", None)
        has_qkv_bias = bool(qkv is not None and qkv.bias is not None)
        self.query_key_value = StageQKVProjection(
            old_features=self.old_hidden,
            new_features=self.new_hidden,
            new_heads=self.new_heads,
            head_size=self.head_size,
            reference=reference,
            initializer_range=initializer_range,
            bias=has_qkv_bias,
        )
        self.new_to_old = _output_parameter(
            reference,
            (self.old_hidden, self.new_hidden),
            output_init,
            output_init_scale,
        )
        self.old_to_new = _normal_parameter(
            reference, (self.new_hidden, self.old_hidden), initializer_range
        )
        self.new_to_new = _normal_parameter(
            reference, (self.new_hidden, self.new_hidden), initializer_range
        )
        leaf_dense = self._leaf_dense(base_attention)
        self.new_output_bias = (
            nn.Parameter(torch.zeros(self.new_hidden, device=reference.device, dtype=reference.dtype))
            if leaf_dense.bias is not None
            else None
        )

    @staticmethod
    def _attention_hidden_size(attention: nn.Module) -> int:
        if isinstance(attention, StageExpandedAttention):
            return attention.total_hidden
        return int(attention.query_key_value.in_features)

    @staticmethod
    def _leaf_dense(attention: nn.Module) -> nn.Module:
        while isinstance(attention, StageExpandedAttention):
            attention = attention.base_attention
        return attention.dense

    @property
    def total_heads(self) -> int:
        return self._attention_head_count(self.base_attention) + self.new_heads

    @classmethod
    def _attention_head_count(cls, attention: nn.Module) -> int:
        if isinstance(attention, StageExpandedAttention):
            return attention.total_heads
        return int(attention.query_key_value.out_features // (3 * attention.head_size))

    @staticmethod
    def _attention_interface(module: nn.Module, output_attentions: bool, head_mask):
        attention_type = module.config._attn_implementation
        if (output_attentions or head_mask is not None) and attention_type in {
            "sdpa",
            "flash_attention_2",
        }:
            attention_type = "eager"
        elif (
            module.training
            and module.attention_dropout > 0
            and attention_type == "flex_attention"
        ):
            attention_type = "eager"
        if attention_type == "eager":
            return eager_attention_forward
        return ALL_ATTENTION_FUNCTIONS[attention_type]

    @staticmethod
    def _split_head_mask(head_mask: torch.Tensor | None, sizes: list[int]):
        if head_mask is None:
            return [None] * len(sizes)
        total = sum(sizes)
        dimensions = [index for index, size in enumerate(head_mask.shape) if size == total]
        if not dimensions:
            raise RuntimeError(
                f"Cannot split head mask shape {tuple(head_mask.shape)} across {sizes}."
            )
        return list(head_mask.split(sizes, dim=dimensions[0]))

    @classmethod
    def _collect_from_attention(
        cls,
        attention: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> list[dict]:
        if isinstance(attention, StageExpandedAttention):
            return attention._collect_groups(hidden_states, position_embeddings)
        input_shape = hidden_states.shape[:-1]
        head_size = int(attention.head_size)
        heads = cls._attention_head_count(attention)
        qkv = attention.query_key_value(hidden_states)
        qkv = qkv.view(*input_shape, heads, 3 * head_size).transpose(1, 2)
        query, key, value = qkv.chunk(3, dim=-1)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        return [{
            "module": attention,
            "query": query,
            "key": key,
            "value": value,
            "heads": heads,
        }]

    def _collect_groups(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> list[dict]:
        old_hidden, _ = hidden_states.split((self.old_hidden, self.new_hidden), dim=-1)
        groups = self._collect_from_attention(
            self.base_attention, old_hidden, position_embeddings
        )
        input_shape = hidden_states.shape[:-1]
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(*input_shape, self.new_heads, 3 * self.head_size).transpose(1, 2)
        query, key, value = qkv.chunk(3, dim=-1)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        groups.append({
            "module": self,
            "query": query,
            "key": key,
            "value": value,
            "heads": self.new_heads,
        })
        return groups

    def _project_attention(self, attention_output: torch.Tensor) -> torch.Tensor:
        old, new = attention_output.split((self.old_hidden, self.new_hidden), dim=-1)
        if isinstance(self.base_attention, StageExpandedAttention):
            old_output = self.base_attention._project_attention(old)
        else:
            old_output = self.base_attention.dense(old)
        old_output = old_output + F.linear(new, self.new_to_old)
        new_output = F.linear(old, self.old_to_new)
        new_output = new_output + F.linear(new, self.new_to_new, self.new_output_bias)
        return torch.cat((old_output, new_output), dim=-1)

    def zero_output_paths(self, mode: str, scale: float) -> None:
        if isinstance(self.base_attention, StageExpandedAttention):
            self.base_attention.zero_output_paths(mode, scale)
        else:
            for parameter in self.base_attention.dense.parameters():
                if mode == "exact":
                    nn.init.zeros_(parameter)
                else:
                    nn.init.normal_(parameter, mean=0.0, std=scale)
        for parameter in (
            self.new_to_old,
            self.old_to_new,
            self.new_to_new,
            self.new_output_bias,
        ):
            if parameter is None:
                continue
            if mode == "exact":
                nn.init.zeros_(parameter)
            else:
                nn.init.normal_(parameter, mean=0.0, std=scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        layer_past=None,
        output_attentions: bool = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ):
        if position_embeddings is None:
            raise RuntimeError("StageExpandedAttention requires position embeddings.")
        input_shape = hidden_states.shape[:-1]
        groups = self._collect_groups(hidden_states, position_embeddings)
        sizes = [int(group["heads"]) for group in groups]

        if layer_past is not None:
            cos, sin = position_embeddings
            all_keys = torch.cat([group["key"] for group in groups], dim=1)
            all_values = torch.cat([group["value"] for group in groups], dim=1)
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "partial_rotation_size": self.rotary_ndims,
                "cache_position": cache_position,
            }
            all_keys, all_values = layer_past.update(
                all_keys, all_values, self.layer_idx, cache_kwargs
            )
            split_keys = all_keys.split(sizes, dim=1)
            split_values = all_values.split(sizes, dim=1)
            for group, key, value in zip(groups, split_keys, split_values):
                group["key"] = key
                group["value"] = value

        masks = self._split_head_mask(head_mask, sizes)
        raw_outputs = []
        attention_weights = []
        for group, group_mask in zip(groups, masks):
            module = group["module"]
            interface = self._attention_interface(module, output_attentions, group_mask)
            output, weights = interface(
                module,
                group["query"],
                group["key"],
                group["value"],
                attention_mask,
                scaling=module.scaling,
                dropout=0.0 if not self.training else module.attention_dropout,
                head_mask=group_mask,
                **kwargs,
            )
            raw_outputs.append(output.reshape(*input_shape, -1).contiguous())
            if weights is not None:
                attention_weights.append(weights)

        projected = self._project_attention(torch.cat(raw_outputs, dim=-1))
        weights = (
            torch.cat(attention_weights, dim=1)
            if attention_weights and len(attention_weights) == len(groups)
            else None
        )
        return projected, weights


class ExpandedEmbedding(nn.Module):
    def __init__(self, base: nn.Module, new_features: int, initializer_range: float) -> None:
        super().__init__()
        self.base = base
        self.num_embeddings = int(base.num_embeddings)
        self.old_embedding_dim = int(base.embedding_dim)
        self.new_embedding_dim = int(new_features)
        self.embedding_dim = self.old_embedding_dim + self.new_embedding_dim
        self.padding_idx = getattr(base, "padding_idx", None)
        self.new_weight = _normal_parameter(
            base.weight, (self.num_embeddings, self.new_embedding_dim), initializer_range
        )

    @property
    def weight(self) -> torch.Tensor:
        return torch.cat((self.base.weight, self.new_weight), dim=1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        old = self.base(input_ids)
        new = F.embedding(input_ids, self.new_weight, padding_idx=self.padding_idx)
        return torch.cat((old, new), dim=-1)


class ExpandedLMHead(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        new_features: int,
        output_init: str = "exact",
        output_init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.in_features = int(base.in_features) + int(new_features)
        self.out_features = int(base.out_features)
        self.new_features = int(new_features)
        self.new_weight = _output_parameter(
            base.weight,
            (self.out_features, self.new_features),
            output_init,
            output_init_scale,
        )

    @property
    def weight(self) -> torch.Tensor:
        return torch.cat((self.base.weight, self.new_weight), dim=1)

    @property
    def bias(self):
        return self.base.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        old, new = hidden_states.split((self.base.in_features, self.new_features), dim=-1)
        return self.base(old) + F.linear(new, self.new_weight)


class ExpandedLayerNorm(nn.Module):
    """Stage-Segmented LayerNorm; it is intentionally not a global widened norm."""

    def __init__(self, base: nn.Module, new_features: int) -> None:
        super().__init__()
        self.base = base
        self.old_features = int(base.normalized_shape[0])
        self.new_features = int(new_features)
        self.normalized_shape = (self.old_features + self.new_features,)
        self.eps = float(base.eps)
        self.new_weight = nn.Parameter(
            torch.ones(self.new_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.new_bias = nn.Parameter(
            torch.zeros(self.new_features, device=base.weight.device, dtype=base.weight.dtype)
        )

    @property
    def weight(self) -> torch.Tensor:
        return torch.cat((self.base.weight, self.new_weight), dim=0)

    @property
    def bias(self) -> torch.Tensor:
        return torch.cat((self.base.bias, self.new_bias), dim=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        old, new = hidden_states.split((self.old_features, self.new_features), dim=-1)
        old = self.base(old)
        new = F.layer_norm(new, (self.new_features,), self.new_weight, self.new_bias, self.eps)
        return torch.cat((old, new), dim=-1)


class ControlTokenEmbedding(nn.Module):
    """Overrides arbitrary control-token rows without modifying the frozen table."""

    def __init__(
        self,
        base: nn.Module,
        control_token_ids: tuple[int, ...],
        total_vocab_size: int,
        initializer_range: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.base_vocab_size = int(base.num_embeddings)
        self.control_count = len(control_token_ids)
        self.num_embeddings = int(total_vocab_size)
        self.embedding_dim = int(base.embedding_dim)
        self.padding_idx = getattr(base, "padding_idx", None)
        self.register_buffer(
            "control_ids", torch.tensor(control_token_ids, dtype=torch.long), persistent=False
        )
        self.control_weight = _normal_parameter(
            base.weight, (self.control_count, self.embedding_dim), initializer_range
        )

    @property
    def weight(self) -> torch.Tensor:
        weight = self.base.weight
        if self.num_embeddings > self.base_vocab_size:
            weight = F.pad(weight, (0, 0, 0, self.num_embeddings - self.base_vocab_size))
        return torch.index_copy(
            weight, 0, self.control_ids.to(weight.device), self.control_weight
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if bool((input_ids < 0).any()) or bool((input_ids >= self.num_embeddings).any()):
            raise IndexError("Token id is outside the split ATE vocabulary.")
        base_ids = input_ids.clamp(max=max(self.base_vocab_size - 1, 0))
        output = self.base(base_ids)
        for row, token_id in enumerate(self.control_ids.tolist()):
            mask = input_ids.eq(token_id).unsqueeze(-1)
            output = torch.where(mask, self.control_weight[row].to(output.dtype), output)
        return output


class ControlTokenLMHead(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        control_token_ids: tuple[int, ...],
        total_vocab_size: int,
        initializer_range: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.in_features = int(base.in_features)
        self.base_vocab_size = int(base.out_features)
        self.control_count = len(control_token_ids)
        self.out_features = int(total_vocab_size)
        self.register_buffer(
            "control_ids", torch.tensor(control_token_ids, dtype=torch.long), persistent=False
        )
        self.control_weight = _normal_parameter(
            base.weight, (self.control_count, self.in_features), initializer_range
        )
        self.control_bias = (
            nn.Parameter(torch.zeros(self.control_count, device=base.weight.device, dtype=base.weight.dtype))
            if base.bias is not None
            else None
        )

    @property
    def weight(self) -> torch.Tensor:
        weight = self.base.weight
        if self.out_features > self.base_vocab_size:
            weight = F.pad(weight, (0, 0, 0, self.out_features - self.base_vocab_size))
        return torch.index_copy(
            weight, 0, self.control_ids.to(weight.device), self.control_weight
        )

    @property
    def bias(self) -> torch.Tensor | None:
        if self.base.bias is None:
            return None
        bias = self.base.bias
        if self.out_features > self.base_vocab_size:
            bias = F.pad(bias, (0, self.out_features - self.base_vocab_size))
        return torch.index_copy(
            bias, 0, self.control_ids.to(bias.device), self.control_bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.base(hidden_states)
        if self.out_features > self.base_vocab_size:
            logits = F.pad(logits, (0, self.out_features - self.base_vocab_size))
        control_logits = []
        for row, token_id in enumerate(self.control_ids.tolist()):
            if token_id < self.base_vocab_size:
                # Preserve the full LM-head reduction exactly at migration.
                # The zero-initialized delta remains independently trainable.
                weight_delta = self.control_weight[row] - self.base.weight[token_id]
                bias_delta = None
                if self.control_bias is not None:
                    bias_delta = self.control_bias[row] - self.base.bias[token_id]
                value = logits[..., token_id : token_id + 1]
                value = value + F.linear(
                    hidden_states,
                    weight_delta.unsqueeze(0),
                    None if bias_delta is None else bias_delta.unsqueeze(0),
                )
            else:
                bias = None
                if self.control_bias is not None:
                    bias = self.control_bias[row : row + 1]
                value = F.linear(
                    hidden_states,
                    self.control_weight[row : row + 1],
                    bias,
                )
            control_logits.append(value)
        return torch.index_copy(
            logits,
            -1,
            self.control_ids.to(logits.device),
            torch.cat(control_logits, dim=-1),
        )


@dataclass(frozen=True)
class ATEParameterReport:
    original_parameters: int
    embedding_expansion: int
    attention_expansion: int
    ffn_expansion: int
    depth_expansion: int
    plastic_parameters: int
    newly_trainable_parameters: int
    total_trainable_parameters: int
    total_parameters: int
    previous_stage_parameters: int = 0
    current_stage_parameters: int = 0
    frozen_parameters: int = 0


class PythiaATEModel(PythiaModModel):
    architecture_version = ATE_ARCHITECTURE_VERSION

    def __init__(
        self,
        base_model: GPTNeoXForCausalLM,
        *,
        original_hidden: int,
        original_heads: int,
        original_layers: int,
        original_intermediate: int,
        original_parameter_ids: set[int],
        original_layer_parameter_ids: tuple[set[int], ...],
        original_embedding_parameter_ids: set[int],
        original_final_norm_parameter_ids: set[int],
        original_lm_head_parameter_ids: set[int],
        base_model_name: str,
        base_revision: str | None,
        original_vocab_size: int,
        control_token_ids: tuple[int, ...],
        expansion_stages: list[dict] | None = None,
        stage_parameter_ids: list[set[int]] | None = None,
        legacy_mode: bool = False,
    ) -> None:
        super().__init__(
            base_model,
            emb_mod_dim=0,
            out_mod_dim=0,
            attn_mod_dim=0,
            ffn_mod_dim=0,
            base_model_name=base_model_name,
            base_revision=base_revision,
            original_vocab_size=original_vocab_size,
            control_token_ids=control_token_ids,
        )
        self.original_hidden = int(original_hidden)
        self.original_heads = int(original_heads)
        self.original_layers = int(original_layers)
        self.original_intermediate = int(original_intermediate)
        self._original_parameter_ids = set(original_parameter_ids)
        self._original_layer_parameter_ids = tuple(set(items) for items in original_layer_parameter_ids)
        self._original_embedding_parameter_ids = set(original_embedding_parameter_ids)
        self._original_final_norm_parameter_ids = set(original_final_norm_parameter_ids)
        self._original_lm_head_parameter_ids = set(original_lm_head_parameter_ids)
        self.expansion_stages = copy.deepcopy(expansion_stages or [])
        self._stage_parameter_ids = [set(items) for items in (stage_parameter_ids or [])]
        self._legacy_mode = bool(legacy_mode)
        self._ate_plastic_groups: list[tuple[str, float, tuple[nn.Parameter, ...]]] = []
        self._previous_stages_trainable = True
        self._refresh_parameter_ownership()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        new_attention_heads: int = 0,
        new_transformer_layers: int = 0,
        vocab_size: int | None = None,
        revision: str | None = None,
        torch_dtype: torch.dtype | None = None,
        gradient_checkpointing: bool = False,
        control_token_ids: tuple[int, ...] = (),
        architecture_metadata: dict | None = None,
        ate_output_init: str = "exact",
        ate_output_init_scale: float = 0.0,
    ) -> "PythiaATEModel":
        load_kwargs = {"revision": revision} if revision else {}
        if torch_dtype is not None:
            # dtype is accepted by current Transformers and avoids the deprecated alias.
            load_kwargs["dtype"] = torch_dtype
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if not isinstance(base_model, GPTNeoXForCausalLM):
            raise TypeError(f"ATE requires GPTNeoXForCausalLM, got {type(base_model).__name__}")
        if architecture_metadata and architecture_metadata.get("architecture_version") == LEGACY_ATE_ARCHITECTURE_VERSION:
            return cls._build_legacy(
                base_model,
                model_name=model_name,
                revision=revision,
                original_vocab_size=int(base_model.get_input_embeddings().num_embeddings),
                vocab_size=vocab_size,
                control_token_ids=control_token_ids,
                new_attention_heads=int(architecture_metadata["new_attention_heads"]),
                new_transformer_layers=int(architecture_metadata["new_transformer_layers"]),
                gradient_checkpointing=gradient_checkpointing,
            )
        stages = architecture_metadata.get("expansion_stages", []) if architecture_metadata else []
        if stages and stages[0].get("legacy_single_stage"):
            model = cls._build_legacy(
                base_model,
                model_name=model_name,
                revision=revision,
                original_vocab_size=int(base_model.get_input_embeddings().num_embeddings),
                vocab_size=vocab_size,
                control_token_ids=control_token_ids,
                new_attention_heads=int(stages[0]["added_attention_heads"]),
                new_transformer_layers=int(stages[0]["added_transformer_layers"]),
                gradient_checkpointing=False,
            )
            model.migrate_legacy_control_rows()
            for stage in stages[1:]:
                model.add_expansion_stage(
                    added_attention_heads=int(stage["added_attention_heads"]),
                    added_transformer_layers=int(stage["added_transformer_layers"]),
                    source_checkpoint=stage.get("source_checkpoint"),
                    output_init=str(stage.get("output_init", "exact")),
                    output_init_scale=float(stage.get("output_init_scale", 0.0)),
                )
            if gradient_checkpointing:
                model.base_model.gradient_checkpointing_enable()
                model.base_model.config.use_cache = False
            return model
        model = cls._from_unexpanded_base(
            base_model,
            model_name=model_name,
            revision=revision,
            vocab_size=vocab_size,
            control_token_ids=control_token_ids,
        )
        if stages:
            for stage in stages:
                model.add_expansion_stage(
                    added_attention_heads=int(stage["added_attention_heads"]),
                    added_transformer_layers=int(stage["added_transformer_layers"]),
                    source_checkpoint=stage.get("source_checkpoint"),
                    output_init=str(stage.get("output_init", "exact")),
                    output_init_scale=float(stage.get("output_init_scale", 0.0)),
                )
        elif new_attention_heads or new_transformer_layers or model._pending_control_ids:
            model.add_expansion_stage(
                added_attention_heads=new_attention_heads,
                added_transformer_layers=new_transformer_layers,
                output_init=ate_output_init,
                output_init_scale=ate_output_init_scale,
            )
        if gradient_checkpointing:
            model.base_model.gradient_checkpointing_enable()
            model.base_model.config.use_cache = False
        return model

    @classmethod
    def from_base_model(
        cls,
        base_model: GPTNeoXForCausalLM,
        *,
        new_attention_heads: int = 0,
        new_transformer_layers: int = 0,
        architecture_metadata: dict | None = None,
        ate_output_init: str = "exact",
        ate_output_init_scale: float = 0.0,
    ) -> "PythiaATEModel":
        model = cls._from_unexpanded_base(
            base_model,
            model_name="test",
            revision=None,
            vocab_size=base_model.get_input_embeddings().num_embeddings,
            control_token_ids=(),
        )
        stages = architecture_metadata.get("expansion_stages", []) if architecture_metadata else []
        if stages:
            for stage in stages:
                model.add_expansion_stage(
                    added_attention_heads=int(stage["added_attention_heads"]),
                    added_transformer_layers=int(stage["added_transformer_layers"]),
                    source_checkpoint=stage.get("source_checkpoint"),
                    output_init=str(stage.get("output_init", "exact")),
                    output_init_scale=float(stage.get("output_init_scale", 0.0)),
                )
        elif new_attention_heads or new_transformer_layers:
            model.add_expansion_stage(
                added_attention_heads=new_attention_heads,
                added_transformer_layers=new_transformer_layers,
                output_init=ate_output_init,
                output_init_scale=ate_output_init_scale,
            )
        return model

    @classmethod
    def _from_unexpanded_base(
        cls,
        base_model: GPTNeoXForCausalLM,
        *,
        model_name: str,
        revision: str | None,
        vocab_size: int | None,
        control_token_ids: tuple[int, ...],
    ) -> "PythiaATEModel":
        config = base_model.config
        original_vocab_size = int(base_model.get_input_embeddings().num_embeddings)
        original_parameter_ids = {id(parameter) for parameter in base_model.parameters()}
        original_layer_parameter_ids = tuple(
            {id(parameter) for parameter in layer.parameters()} for layer in base_model.gpt_neox.layers
        )
        original_embedding_parameter_ids = {id(p) for p in base_model.gpt_neox.embed_in.parameters()}
        original_final_norm_parameter_ids = {
            id(p) for p in base_model.gpt_neox.final_layer_norm.parameters()
        }
        original_lm_head_parameter_ids = {id(p) for p in base_model.embed_out.parameters()}
        requested_vocab = max(int(vocab_size or original_vocab_size), original_vocab_size)
        if control_token_ids:
            if len(set(control_token_ids)) != len(control_token_ids):
                raise ValueError(f"ATE control token IDs must be unique, got {control_token_ids}.")
            if min(control_token_ids) < 0 or max(control_token_ids) >= requested_vocab:
                raise ValueError(
                    f"ATE control token IDs {control_token_ids} are outside vocabulary size {requested_vocab}."
                )
            appended_ids = set(range(original_vocab_size, requested_vocab))
            if not appended_ids.issubset(set(control_token_ids)):
                raise ValueError(
                    "Every vocabulary row beyond the pretrained table must be a registered control token."
                )
            initializer = float(config.initializer_range)
            input_base = base_model.gpt_neox.embed_in
            output_base = base_model.embed_out
            input_wrapper = ControlTokenEmbedding(
                base_model.gpt_neox.embed_in, control_token_ids, requested_vocab, initializer
            )
            output_wrapper = ControlTokenLMHead(
                base_model.embed_out, control_token_ids, requested_vocab, initializer
            )
            with torch.no_grad():
                for row, token_id in enumerate(control_token_ids):
                    if token_id < original_vocab_size:
                        input_wrapper.control_weight[row].copy_(input_base.weight[token_id])
                        output_wrapper.control_weight[row].copy_(output_base.weight[token_id])
                        if output_wrapper.control_bias is not None:
                            output_wrapper.control_bias[row].copy_(output_base.bias[token_id])
            base_model.gpt_neox.embed_in = input_wrapper
            base_model.embed_out = output_wrapper
            config.vocab_size = requested_vocab
        result = cls(
            base_model,
            original_hidden=int(config.hidden_size),
            original_heads=int(config.num_attention_heads),
            original_layers=len(base_model.gpt_neox.layers),
            original_intermediate=int(config.intermediate_size),
            original_parameter_ids=original_parameter_ids,
            original_layer_parameter_ids=original_layer_parameter_ids,
            original_embedding_parameter_ids=original_embedding_parameter_ids,
            original_final_norm_parameter_ids=original_final_norm_parameter_ids,
            original_lm_head_parameter_ids=original_lm_head_parameter_ids,
            base_model_name=model_name,
            base_revision=revision,
            original_vocab_size=original_vocab_size,
            control_token_ids=control_token_ids,
        )
        result._pending_control_ids = {
            id(p) for p in result.parameters() if id(p) not in original_parameter_ids
        }
        return result

    @classmethod
    def _build_legacy(
        cls,
        base_model: GPTNeoXForCausalLM,
        *,
        model_name: str,
        revision: str | None,
        original_vocab_size: int,
        vocab_size: int | None,
        control_token_ids: tuple[int, ...],
        new_attention_heads: int,
        new_transformer_layers: int,
        gradient_checkpointing: bool,
    ) -> "PythiaATEModel":
        # Exact v1 graph reconstruction is retained solely for strict legacy load/migration.
        if vocab_size is not None and vocab_size > original_vocab_size:
            try:
                base_model.resize_token_embeddings(vocab_size, mean_resizing=False)
            except TypeError:
                base_model.resize_token_embeddings(vocab_size)
        config = base_model.config
        original_hidden = int(config.hidden_size)
        original_heads = int(config.num_attention_heads)
        original_layers = len(base_model.gpt_neox.layers)
        original_intermediate = int(config.intermediate_size)
        original_ids = {id(p) for p in base_model.parameters()}
        layer_ids = tuple({id(p) for p in layer.parameters()} for layer in base_model.gpt_neox.layers)
        embedding_ids = {id(p) for p in base_model.gpt_neox.embed_in.parameters()}
        norm_ids = {id(p) for p in base_model.gpt_neox.final_layer_norm.parameters()}
        head_ids = {id(p) for p in base_model.embed_out.parameters()}
        head_dim = original_hidden // original_heads
        added_hidden = int(new_attention_heads) * head_dim
        added_intermediate = int(round((original_hidden + added_hidden) * original_intermediate / original_hidden)) - original_intermediate
        if added_hidden:
            cls._legacy_widen_existing_modules(
                base_model,
                added_hidden,
                added_intermediate,
                float(config.initializer_range),
                "exact",
                0.0,
            )
        config.hidden_size = original_hidden + added_hidden
        config.num_attention_heads = original_heads + int(new_attention_heads)
        config.intermediate_size = original_intermediate + added_intermediate
        config.num_hidden_layers = original_layers + int(new_transformer_layers)
        cls._refresh_attention_configs(base_model, head_dim)
        for layer_index in range(original_layers, config.num_hidden_layers):
            layer = GPTNeoXLayer(config, layer_index)
            reference = next(base_model.parameters())
            layer.to(device=reference.device, dtype=reference.dtype)
            cls._initialize_depth_outputs(layer, "exact", 0.0)
            base_model.gpt_neox.layers.append(layer)
        stage = {
            "stage_id": 1,
            "source_checkpoint": None,
            "added_attention_heads": int(new_attention_heads),
            "added_hidden_size": added_hidden,
            "added_transformer_layers": int(new_transformer_layers),
            "result_hidden_size": int(config.hidden_size),
            "result_attention_heads": int(config.num_attention_heads),
            "result_layers": len(base_model.gpt_neox.layers),
            "legacy_single_stage": True,
            "output_init": "exact",
            "output_init_scale": 0.0,
        }
        model = cls(
            base_model,
            original_hidden=original_hidden,
            original_heads=original_heads,
            original_layers=original_layers,
            original_intermediate=original_intermediate,
            original_parameter_ids=original_ids,
            original_layer_parameter_ids=layer_ids,
            original_embedding_parameter_ids=embedding_ids,
            original_final_norm_parameter_ids=norm_ids,
            original_lm_head_parameter_ids=head_ids,
            base_model_name=model_name,
            base_revision=revision,
            original_vocab_size=original_vocab_size,
            control_token_ids=control_token_ids,
            expansion_stages=[stage],
            stage_parameter_ids=[{id(p) for p in base_model.parameters()} - original_ids],
            legacy_mode=True,
        )
        model._pending_control_ids = set()
        if gradient_checkpointing:
            model.base_model.gradient_checkpointing_enable()
            model.base_model.config.use_cache = False
        return model

    @staticmethod
    def _legacy_widen_layer(
        layer: nn.Module,
        added_hidden: int,
        added_intermediate: int,
        initializer: float,
        output_init: str,
        output_scale: float,
    ) -> None:
        layer.input_layernorm = ExpandedLayerNorm(layer.input_layernorm, added_hidden)
        layer.post_attention_layernorm = ExpandedLayerNorm(layer.post_attention_layernorm, added_hidden)
        layer.attention.query_key_value = ExpandedLinear(
            layer.attention.query_key_value,
            new_in_features=added_hidden,
            new_out_features=3 * added_hidden,
            initializer_range=initializer,
            output_init=output_init,
            output_init_scale=output_scale,
        )
        layer.attention.dense = ExpandedLinear(
            layer.attention.dense,
            new_in_features=added_hidden,
            new_out_features=added_hidden,
            initializer_range=initializer,
            output_init=output_init,
            output_init_scale=output_scale,
        )
        layer.mlp.dense_h_to_4h = ExpandedLinear(
            layer.mlp.dense_h_to_4h,
            new_in_features=added_hidden,
            new_out_features=added_intermediate,
            initializer_range=initializer,
            output_init=output_init,
            output_init_scale=output_scale,
        )
        layer.mlp.dense_4h_to_h = ExpandedLinear(
            layer.mlp.dense_4h_to_h,
            new_in_features=added_intermediate,
            new_out_features=added_hidden,
            initializer_range=initializer,
            output_init=output_init,
            output_init_scale=output_scale,
        )

    @classmethod
    def _legacy_widen_existing_modules(
        cls,
        base_model: GPTNeoXForCausalLM,
        added_hidden: int,
        added_intermediate: int,
        initializer: float,
        output_init: str,
        output_scale: float,
    ) -> None:
        base_model.gpt_neox.embed_in = ExpandedEmbedding(
            base_model.gpt_neox.embed_in, added_hidden, initializer
        )
        base_model.embed_out = ExpandedLMHead(
            base_model.embed_out, added_hidden, output_init, output_scale
        )
        base_model.gpt_neox.final_layer_norm = ExpandedLayerNorm(
            base_model.gpt_neox.final_layer_norm, added_hidden
        )
        for layer in base_model.gpt_neox.layers:
            cls._legacy_widen_layer(
                layer, added_hidden, added_intermediate, initializer, output_init, output_scale
            )

    @staticmethod
    def _widen_layer(
        layer: nn.Module,
        added_hidden: int,
        added_intermediate: int,
        initializer: float,
        output_init: str,
        output_scale: float,
    ) -> None:
        head_size = int(getattr(layer.attention, "head_size"))
        if added_hidden % head_size:
            raise ValueError(
                f"Added attention width {added_hidden} is not divisible by head size {head_size}."
            )
        layer.input_layernorm = ExpandedLayerNorm(layer.input_layernorm, added_hidden)
        layer.post_attention_layernorm = ExpandedLayerNorm(
            layer.post_attention_layernorm, added_hidden
        )
        layer.attention = StageExpandedAttention(
            layer.attention,
            new_hidden=added_hidden,
            new_heads=added_hidden // head_size,
            head_size=head_size,
            initializer_range=initializer,
            output_init=output_init,
            output_init_scale=output_scale,
        )
        layer.mlp.dense_h_to_4h = ExpandedLinear(
            layer.mlp.dense_h_to_4h,
            new_in_features=added_hidden,
            new_out_features=added_intermediate,
            initializer_range=initializer,
            output_init=output_init,
            output_init_scale=output_scale,
        )
        layer.mlp.dense_4h_to_h = ExpandedLinear(
            layer.mlp.dense_4h_to_h,
            new_in_features=added_intermediate,
            new_out_features=added_hidden,
            initializer_range=initializer,
            output_init=output_init,
            output_init_scale=output_scale,
        )

    @classmethod
    def _widen_existing_modules(
        cls,
        base_model: GPTNeoXForCausalLM,
        added_hidden: int,
        added_intermediate: int,
        initializer: float,
        output_init: str,
        output_scale: float,
    ) -> None:
        base_model.gpt_neox.embed_in = ExpandedEmbedding(
            base_model.gpt_neox.embed_in, added_hidden, initializer
        )
        base_model.embed_out = ExpandedLMHead(
            base_model.embed_out, added_hidden, output_init, output_scale
        )
        base_model.gpt_neox.final_layer_norm = ExpandedLayerNorm(
            base_model.gpt_neox.final_layer_norm, added_hidden
        )
        for layer in base_model.gpt_neox.layers:
            cls._widen_layer(
                layer, added_hidden, added_intermediate, initializer, output_init, output_scale
            )

    @staticmethod
    def _refresh_attention_configs(base_model: GPTNeoXForCausalLM, head_dim: int) -> None:
        for index, layer in enumerate(base_model.gpt_neox.layers):
            layer.attention.config = base_model.config
            layer.attention.layer_idx = index
            active = int(getattr(layer.attention, "head_size", getattr(layer.attention, "head_dim", head_dim)))
            if active != head_dim:
                raise RuntimeError(f"ATE changed head_dim from {head_dim} to {active} at layer {index}.")

    @staticmethod
    def _initialize_depth_outputs(layer: nn.Module, output_init: str, output_scale: float) -> None:
        if isinstance(layer.attention, StageExpandedAttention):
            layer.attention.zero_output_paths(output_init, output_scale)
        else:
            for parameter in layer.attention.dense.parameters():
                if output_init == "exact":
                    nn.init.zeros_(parameter)
                else:
                    nn.init.normal_(parameter, mean=0.0, std=output_scale)
        for parameter in layer.mlp.dense_4h_to_h.parameters():
            if output_init == "exact":
                nn.init.zeros_(parameter)
            else:
                nn.init.normal_(parameter, mean=0.0, std=output_scale)

    def _new_base_layer(self, layer_index: int) -> nn.Module:
        config = copy.deepcopy(self.config)
        config.hidden_size = self.original_hidden
        config.num_attention_heads = self.original_heads
        config.intermediate_size = self.original_intermediate
        config.num_hidden_layers = self.original_layers
        layer = GPTNeoXLayer(config, layer_index)
        reference = next(self.parameters())
        layer.to(device=reference.device, dtype=reference.dtype)
        for stage in self.expansion_stages:
            added_hidden = int(stage["added_hidden_size"])
            if not added_hidden:
                continue
            added_intermediate = int(stage.get("added_intermediate_size", round(
                added_hidden * self.original_intermediate / self.original_hidden
            )))
            self._widen_layer(
                layer,
                added_hidden,
                added_intermediate,
                float(self.config.initializer_range),
                str(stage.get("output_init", "exact")),
                float(stage.get("output_init_scale", 0.0)),
            )
        return layer

    def add_expansion_stage(
        self,
        *,
        added_attention_heads: int,
        added_transformer_layers: int,
        source_checkpoint: str | None = None,
        output_init: str = "exact",
        output_init_scale: float = 0.0,
    ) -> dict:
        if added_attention_heads < 0 or added_transformer_layers < 0:
            raise ValueError("ATE stage additions must be non-negative.")
        if not added_attention_heads and not added_transformer_layers and not getattr(self, "_pending_control_ids", set()):
            raise ValueError("An ATE stage must add width, depth, or pending control-token rows.")
        before_ids = {id(p) for p in self.parameters()}
        previous_hidden = int(self.config.hidden_size)
        previous_heads = int(self.config.num_attention_heads)
        previous_intermediate = int(self.config.intermediate_size)
        previous_layers = len(self.base_model.gpt_neox.layers)
        head_dim = self.original_hidden // self.original_heads
        added_hidden = int(added_attention_heads) * head_dim
        ratio = self.original_intermediate / self.original_hidden
        added_intermediate = int(round(added_hidden * ratio))
        if added_hidden:
            self._widen_existing_modules(
                self.base_model,
                added_hidden,
                added_intermediate,
                float(self.config.initializer_range),
                output_init,
                output_init_scale,
            )
            self.config.hidden_size = previous_hidden + added_hidden
            self.config.num_attention_heads = previous_heads + int(added_attention_heads)
            self.config.intermediate_size = previous_intermediate + added_intermediate
        stage = {
            "stage_id": len(self.expansion_stages) + 1,
            "source_checkpoint": source_checkpoint,
            "added_attention_heads": int(added_attention_heads),
            "added_hidden_size": added_hidden,
            "added_intermediate_size": added_intermediate,
            "added_transformer_layers": int(added_transformer_layers),
            "result_hidden_size": int(self.config.hidden_size),
            "result_attention_heads": int(self.config.num_attention_heads),
            "result_layers": previous_layers + int(added_transformer_layers),
            "output_init": output_init,
            "output_init_scale": float(output_init_scale),
        }
        # Make the stage visible while depth blocks replay all width generations.
        self.expansion_stages.append(stage)
        for index in range(previous_layers, stage["result_layers"]):
            layer = self._new_base_layer(index)
            self._initialize_depth_outputs(layer, output_init, output_init_scale)
            self.base_model.gpt_neox.layers.append(layer)
        self.config.num_hidden_layers = len(self.base_model.gpt_neox.layers)
        self._refresh_attention_configs(self.base_model, head_dim)
        after_ids = {id(p) for p in self.parameters()}
        stage_ids = (after_ids - before_ids) | set(getattr(self, "_pending_control_ids", set()))
        self._pending_control_ids = set()
        self._stage_parameter_ids.append(stage_ids)
        self._legacy_mode = False
        self._refresh_parameter_ownership()
        return stage

    def migrate_legacy_control_rows(self) -> None:
        """Split legacy resized rows after strict v1 loading without changing values."""
        if not self._legacy_mode:
            return
        if not self.control_token_ids:
            self._legacy_mode = False
            self._refresh_parameter_ownership()
            return
        count = len(self.control_token_ids)
        control_ids = tuple(int(item) for item in self.control_token_ids)
        if len(set(control_ids)) != count or min(control_ids) < 0:
            raise RuntimeError(f"Legacy ATE has invalid control token IDs: {control_ids}.")

        def split_embedding(module: nn.Module) -> None:
            if isinstance(module, ExpandedEmbedding):
                split_embedding(module.base)
                return
            if not isinstance(module, nn.Embedding):
                raise RuntimeError(f"Cannot migrate legacy embedding type {type(module).__name__}.")
            old = module.weight.detach().clone()
            wrapper = ControlTokenEmbedding(
                module, control_ids, int(module.num_embeddings), float(self.config.initializer_range)
            )
            wrapper.control_weight.data.copy_(old[list(control_ids)])
            self._replace_module_reference(self.base_model.gpt_neox.embed_in, module, wrapper, "embedding")

        def split_head(module: nn.Module) -> None:
            if isinstance(module, ExpandedLMHead):
                split_head(module.base)
                return
            if not isinstance(module, nn.Linear):
                raise RuntimeError(f"Cannot migrate legacy LM head type {type(module).__name__}.")
            old_w = module.weight.detach().clone()
            old_b = module.bias.detach().clone() if module.bias is not None else None
            wrapper = ControlTokenLMHead(
                module, control_ids, int(module.out_features), float(self.config.initializer_range)
            )
            wrapper.control_weight.data.copy_(old_w[list(control_ids)])
            if old_b is not None:
                wrapper.control_bias.data.copy_(old_b[list(control_ids)])
            self._replace_module_reference(self.base_model.embed_out, module, wrapper, "head")

        split_embedding(self.base_model.gpt_neox.embed_in)
        split_head(self.base_model.embed_out)
        # Migration changes parameter objects but not the function. Treat all legacy
        # ATE rows and wrappers as stage 1; original IDs are recomputed by location.
        self._rebuild_original_parameter_ids()
        all_non_original = {id(p) for p in self.parameters()} - self._original_parameter_ids
        self._stage_parameter_ids = [all_non_original]
        self.expansion_stages[0]["legacy_single_stage"] = True
        self._legacy_mode = False
        self._refresh_parameter_ownership()

    def _replace_module_reference(self, root: nn.Module, old: nn.Module, new: nn.Module, kind: str) -> None:
        if root is old:
            if kind == "embedding":
                self.base_model.gpt_neox.embed_in = new
            else:
                self.base_model.embed_out = new
            return
        for child in root.modules():
            if getattr(child, "base", None) is old:
                child.base = new
                return
        raise RuntimeError(f"Unable to replace legacy {kind} module.")

    def _rebuild_original_parameter_ids(self) -> None:
        def deepest(module: nn.Module) -> nn.Module:
            while hasattr(module, "base"):
                module = module.base
            return module
        original: set[int] = set()
        embed = deepest(self.base_model.gpt_neox.embed_in)
        head = deepest(self.base_model.embed_out)
        original.update(id(p) for p in embed.parameters())
        original.update(id(p) for p in head.parameters())
        layer_sets = []
        for layer in self.base_model.gpt_neox.layers[: self.original_layers]:
            ids = set()
            for module in (
                layer.input_layernorm, layer.post_attention_layernorm,
                layer.attention.query_key_value, layer.attention.dense,
                layer.mlp.dense_h_to_4h, layer.mlp.dense_4h_to_h,
            ):
                leaf = deepest(module)
                ids.update(id(p) for p in leaf.parameters())
            layer_sets.append(ids)
            original.update(ids)
        final_norm = deepest(self.base_model.gpt_neox.final_layer_norm)
        original.update(id(p) for p in final_norm.parameters())
        self._original_parameter_ids = original
        self._original_embedding_parameter_ids = {id(p) for p in embed.parameters()}
        self._original_lm_head_parameter_ids = {id(p) for p in head.parameters()}
        self._original_final_norm_parameter_ids = {id(p) for p in final_norm.parameters()}
        self._original_layer_parameter_ids = tuple(layer_sets)

    def _refresh_parameter_ownership(self) -> None:
        all_ids = {id(p) for p in self.parameters()}
        owned = set().union(*self._stage_parameter_ids) if self._stage_parameter_ids else set()
        self._new_parameter_ids = owned & all_ids
        self._new_parameter_groups = self._classify_parameters(self._new_parameter_ids)

    def _classify_parameters(self, ids: set[int]) -> dict[str, tuple[nn.Parameter, ...]]:
        groups: dict[str, list[nn.Parameter]] = {
            "embedding": [], "attention": [], "ffn": [], "depth": []
        }
        for name, parameter in self.named_parameters():
            if id(parameter) not in ids:
                continue
            match = re.search(r"\.layers\.(\d+)\.", name)
            if match and int(match.group(1)) >= self.original_layers:
                groups["depth"].append(parameter)
            elif ".attention." in name:
                groups["attention"].append(parameter)
            elif ".mlp." in name:
                groups["ffn"].append(parameter)
            else:
                groups["embedding"].append(parameter)
        return {key: tuple(value) for key, value in groups.items()}

    def _parameters_from_ids(self, parameter_ids: Iterable[int]) -> tuple[nn.Parameter, ...]:
        wanted = set(parameter_ids)
        return tuple(p for p in self.parameters() if id(p) in wanted)

    def configure_ate_plasticity(
        self,
        mode: str,
        custom_scales: tuple[float, ...] = (),
        previous_stages_trainable: bool = True,
    ) -> None:
        if mode not in {"off", "linear", "quadratic", "custom"}:
            raise ValueError(f"Unknown ATE plasticity mode {mode!r}.")
        if mode == "custom" and len(custom_scales) != self.original_layers:
            raise ValueError(
                f"ATE custom plasticity requires {self.original_layers} comma-separated layer scales."
            )
        self._previous_stages_trainable = bool(previous_stages_trainable)
        current = self._stage_parameter_ids[-1] if self._stage_parameter_ids else set()
        previous = set().union(*self._stage_parameter_ids[:-1]) if len(self._stage_parameter_ids) > 1 else set()
        for parameter in self.parameters():
            train = id(parameter) in current or (self._previous_stages_trainable and id(parameter) in previous)
            parameter.requires_grad_(train)

        def schedule(layer_number: int) -> float:
            ratio = layer_number / max(self.original_layers, 1)
            if mode == "off":
                return 0.0
            if mode == "linear":
                return ratio
            if mode == "quadratic":
                return ratio * ratio
            return float(custom_scales[layer_number - 1])

        groups: list[tuple[str, float, tuple[nn.Parameter, ...]]] = []
        if schedule(1) > 0:
            groups.append(("ate_plastic_embedding", schedule(1), self._parameters_from_ids(self._original_embedding_parameter_ids)))
        for index, ids in enumerate(self._original_layer_parameter_ids):
            scale = schedule(index + 1)
            if scale > 0:
                groups.append((f"ate_plastic_layer_{index:02d}", scale, self._parameters_from_ids(ids)))
        if mode != "off":
            final_ids = self._original_final_norm_parameter_ids | self._original_lm_head_parameter_ids
            groups.append(("ate_plastic_output", 1.0, self._parameters_from_ids(final_ids)))
        for _, _, parameters in groups:
            for parameter in parameters:
                parameter.requires_grad_(True)
        self._ate_plastic_groups = groups

    def current_stage_parameters(self) -> tuple[nn.Parameter, ...]:
        return self._parameters_from_ids(self._stage_parameter_ids[-1] if self._stage_parameter_ids else set())

    def previous_stage_parameters(self) -> tuple[nn.Parameter, ...]:
        ids = set().union(*self._stage_parameter_ids[:-1]) if len(self._stage_parameter_ids) > 1 else set()
        return self._parameters_from_ids(ids)

    def new_parameters(self) -> tuple[nn.Parameter, ...]:
        return self._parameters_from_ids(self._new_parameter_ids)

    def plastic_base_parameters(self) -> list[nn.Parameter]:
        return [p for _, _, parameters in self._ate_plastic_groups for p in parameters]

    def ate_optimizer_specs(
        self,
        training_lr: float,
        plastic_base_lr: float,
        current_stage_lr: float | None = None,
        previous_stage_lr: float | None = None,
    ) -> list[tuple[str, float, tuple[nn.Parameter, ...]]]:
        specs = []
        current = tuple(p for p in self.current_stage_parameters() if p.requires_grad)
        previous = tuple(p for p in self.previous_stage_parameters() if p.requires_grad)
        if current:
            specs.append(("ate_new", float(current_stage_lr or training_lr), current))
        if previous:
            specs.append(("ate_previous", float(previous_stage_lr or training_lr), previous))
        specs.extend(
            (name, float(plastic_base_lr) * multiplier, parameters)
            for name, multiplier, parameters in self._ate_plastic_groups
        )
        return specs

    @property
    def new_attention_heads(self) -> int:
        return int(self.config.num_attention_heads) - self.original_heads

    @property
    def new_transformer_layers(self) -> int:
        return len(self.base_model.gpt_neox.layers) - self.original_layers

    def ate_metadata(self) -> dict:
        if self._legacy_mode:
            return {
                "architecture_version": LEGACY_ATE_ARCHITECTURE_VERSION,
                "original_hidden": self.original_hidden,
                "expanded_hidden": self.hidden_size,
                "original_layers": self.original_layers,
                "expanded_layers": self.num_layers,
                "original_heads": self.original_heads,
                "expanded_heads": int(self.config.num_attention_heads),
                "new_attention_heads": self.new_attention_heads,
                "new_transformer_layers": self.new_transformer_layers,
            }
        return {
            "architecture_version": ATE_ARCHITECTURE_VERSION,
            "original_architecture": {
                "hidden_size": self.original_hidden,
                "attention_heads": self.original_heads,
                "layers": self.original_layers,
                "head_dim": self.original_hidden // self.original_heads,
                "intermediate_size": self.original_intermediate,
                "vocab_size": self.original_vocab_size,
            },
            "normalization_strategy": NORMALIZATION_STRATEGY,
            "control_token_strategy": "split_rows",
            "expansion_stages": copy.deepcopy(self.expansion_stages),
            # Compatibility fields used by existing reports and CLI validation.
            "original_hidden": self.original_hidden,
            "expanded_hidden": self.hidden_size,
            "original_layers": self.original_layers,
            "expanded_layers": self.num_layers,
            "original_heads": self.original_heads,
            "expanded_heads": int(self.config.num_attention_heads),
            "new_attention_heads": self.new_attention_heads,
            "new_transformer_layers": self.new_transformer_layers,
        }

    def validate_ate_metadata(self, metadata: dict | None) -> None:
        if not isinstance(metadata, dict):
            raise RuntimeError("ATE checkpoint is missing architecture metadata.")
        if metadata.get("architecture_version") == LEGACY_ATE_ARCHITECTURE_VERSION and self._legacy_mode:
            expected = {
                "architecture_version": LEGACY_ATE_ARCHITECTURE_VERSION,
                "original_hidden": self.original_hidden,
                "expanded_hidden": self.hidden_size,
                "original_layers": self.original_layers,
                "expanded_layers": self.num_layers,
                "original_heads": self.original_heads,
                "expanded_heads": int(self.config.num_attention_heads),
                "new_attention_heads": self.new_attention_heads,
                "new_transformer_layers": self.new_transformer_layers,
            }
            if metadata != expected:
                raise RuntimeError(f"ATE checkpoint architecture mismatch: saved={metadata} current={expected}")
            return
        if metadata != self.ate_metadata():
            raise RuntimeError(f"ATE checkpoint architecture mismatch: saved={metadata} current={self.ate_metadata()}")

    def validate_incremental_source_metadata(self, metadata: dict | None) -> None:
        if not isinstance(metadata, dict) or metadata.get("architecture_version") not in {
            LEGACY_ATE_ARCHITECTURE_VERSION, ATE_ARCHITECTURE_VERSION
        }:
            raise RuntimeError(f"Incremental source is not a compatible ATE checkpoint: {metadata}")

    def preservation_probe(self, source_logits: torch.Tensor, input_ids: torch.Tensor) -> dict:
        training = self.training
        self.eval()
        with torch.no_grad():
            expanded = self(input_ids=input_ids)["logits"].float()
        if training:
            self.train()
        source = source_logits.float().to(expanded.device)
        difference = (source - expanded).abs()
        return {
            "max_abs_logit_difference": float(difference.max().item()),
            "mean_abs_logit_difference": float(difference.mean().item()),
            "passed": bool(difference.max().item() < 1e-6 and difference.mean().item() < 1e-7),
        }

    def gradient_diagnostics(self, input_ids: torch.Tensor) -> dict[str, dict[str, float | int]]:
        current_ids = self._stage_parameter_ids[-1] if self._stage_parameter_ids else set()
        original_requires = {id(p): p.requires_grad for p in self.parameters()}
        was_training = self.training
        self.train()
        self.zero_grad(set_to_none=True)
        for p in self.parameters():
            p.requires_grad_(id(p) in current_ids)
        labels = input_ids.clone()
        output = self(input_ids=input_ids, labels=labels)
        output["loss"].backward()
        categories = {
            "Embedding": [], "Attention input": [], "Attention output": [],
            "FFN input": [], "FFN output": [], "LayerNorm": [],
            "LM head": [], "Depth block": [],
        }
        for name, p in self.named_parameters():
            if id(p) not in current_ids:
                continue
            layer_match = re.search(r"\.layers\.(\d+)\.", name)
            if layer_match and int(layer_match.group(1)) >= self.original_layers:
                category = "Depth block"
            elif "query_key_value" in name:
                category = "Attention input"
            elif (
                ".attention.dense" in name
                or re.search(
                    r"\.attention\.(?:base_attention\.)*(?:new_to_old|old_to_new|new_to_new|new_output_bias)$",
                    name,
                )
            ):
                category = "Attention output"
            elif "dense_h_to_4h" in name:
                category = "FFN input"
            elif "dense_4h_to_h" in name:
                category = "FFN output"
            elif "layernorm" in name or "layer_norm" in name:
                category = "LayerNorm"
            elif "embed_out" in name:
                category = "LM head"
            else:
                category = "Embedding"
            categories[category].append(p)
        report = {}
        for name, parameters in categories.items():
            grads = [p.grad for p in parameters if p.grad is not None]
            norm = math.sqrt(sum(float(g.float().pow(2).sum().item()) for g in grads)) if grads else 0.0
            report[name] = {
                "total_parameter_tensors": len(parameters),
                "tensors_with_gradients": len(grads),
                "tensors_with_nonzero_gradients": sum(bool(torch.count_nonzero(g)) for g in grads),
                "gradient_norm": norm,
            }
        self.zero_grad(set_to_none=True)
        for p in self.parameters():
            p.requires_grad_(original_requires[id(p)])
        if not was_training:
            self.eval()
        return report

    def incremental_source_parameter_count(self, source_state: dict[str, torch.Tensor]) -> int:
        return sum(int(t.numel()) for t in source_state.values() if isinstance(t, torch.Tensor))

    def incremental_source_expansion_counts(self, source_state: dict[str, torch.Tensor]) -> dict[str, int]:
        del source_state
        return {
            name: sum(p.numel() for p in parameters)
            for name, parameters in self._new_parameter_groups.items()
        }

    def load_incremental_checkpoint_state(self, source_state: dict[str, torch.Tensor]) -> dict[str, int]:
        incompatible = self.load_state_dict(source_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Strict source load failed: {incompatible}")
        return {"exact_tensors": len(source_state), "expanded_tensors": 0}

    def ate_parameter_report(self) -> ATEParameterReport:
        counts = {
            name: sum(p.numel() for p in parameters)
            for name, parameters in self._new_parameter_groups.items()
        }
        plastic = sum(p.numel() for p in self.plastic_base_parameters())
        previous = sum(p.numel() for p in self.previous_stage_parameters())
        current = sum(p.numel() for p in self.current_stage_parameters())
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return ATEParameterReport(
            original_parameters=sum(p.numel() for p in self.parameters() if id(p) in self._original_parameter_ids),
            embedding_expansion=counts["embedding"],
            attention_expansion=counts["attention"],
            ffn_expansion=counts["ffn"],
            depth_expansion=counts["depth"],
            plastic_parameters=plastic,
            newly_trainable_parameters=sum(p.numel() for p in self.new_parameters()),
            total_trainable_parameters=trainable,
            total_parameters=total,
            previous_stage_parameters=previous,
            current_stage_parameters=current,
            frozen_parameters=total - trainable,
        )
