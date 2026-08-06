from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TinyConfig:
    vocab_size: int
    num_layers: int = 4
    hidden_size: int = 128
    num_heads: int = 4
    ffn_hidden_size: int = 512
    max_seq_len: int = 256
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0

    def validate(self) -> None:
        if self.vocab_size <= 4:
            raise ValueError("vocab_size must exceed the special-token count")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        head_dim = self.hidden_size // self.num_heads
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        if min(self.num_layers, self.hidden_size, self.num_heads, self.ffn_hidden_size) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.max_seq_len <= 1:
            raise ValueError("max_seq_len must be greater than one")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LearnerLayout:
    # The first four fields preserve compatibility with the original checkpoints.
    mode: str = "single"
    bottleneck_dim: int = 128
    single_layer: int = -1
    dropout: float = 0.0
    kind: str = "residual"
    standalone_layers: int = 1
    standalone_ffn_size: int = 32

    def validate(self, num_layers: int) -> None:
        if self.kind not in {"residual", "standalone", "pre_activation", "post_activation"}:
            raise ValueError(
                "learner kind must be residual, standalone, pre_activation, or post_activation"
            )
        if self.mode not in {"single", "all"}:
            raise ValueError("learner mode must be single or all")
        if self.bottleneck_dim <= 0:
            raise ValueError("learner bottleneck_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("learner dropout must be in [0, 1)")
        if self.mode == "single" and self.kind != "standalone":
            normalize_layer_index(self.single_layer, num_layers)
        if self.standalone_layers <= 0:
            raise ValueError("standalone_layers must be positive")
        if self.standalone_ffn_size <= 0:
            raise ValueError("standalone_ffn_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_layer_index(index: int, num_layers: int) -> int:
    normalized = index if index >= 0 else num_layers + index
    if normalized < 0 or normalized >= num_layers:
        raise IndexError(f"Layer {index} is outside a {num_layers}-layer model")
    return normalized


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        values = x.float()
        normalized = values * torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(dtype) * self.weight.to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(q: torch.Tensor, k: torch.Tensor, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_length = q.shape[-2]
    head_dim = q.shape[-1]
    positions = torch.arange(sequence_length, device=q.device, dtype=torch.float32)
    inverse_frequency = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim)
    )
    angles = torch.outer(positions, inverse_frequency)
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1)[None, None, :, :].to(q.dtype)
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1)[None, None, :, :].to(q.dtype)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.rope_theta = config.rope_theta
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, hidden = x.shape
        shape = (batch, sequence, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q, k = apply_rope(q, k, self.rope_theta)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        output = output.transpose(1, 2).contiguous().view(batch, sequence, hidden)
        return self.o_proj(output)


class FFNAdjustment(nn.Module):
    """A zero-initialized low-rank learner adjustment."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bottleneck_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.down = nn.Linear(input_size, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, output_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.ones(()))
        self.restore_zero_output()

    def restore_zero_output(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        with torch.no_grad():
            self.gate.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.up(self.dropout(F.silu(self.down(x))))
        return self.gate.to(update.dtype) * update


class SwiGLU(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        pre_adjustment: FFNAdjustment | None = None,
        post_adjustment: FFNAdjustment | None = None,
    ) -> torch.Tensor:
        gate = self.gate_proj(x)
        if pre_adjustment is not None:
            gate = gate + pre_adjustment(x)
        up = self.up_proj(x)
        activated = F.silu(gate) * up
        if post_adjustment is not None:
            activated = activated + post_adjustment(activated)
        return self.down_proj(self.dropout(activated))


class DecoderBlock(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        pre_adjustment: FFNAdjustment | None = None,
        post_adjustment: FFNAdjustment | None = None,
    ) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x))
        x = x + self.ffn(
            self.ffn_norm(x),
            pre_adjustment=pre_adjustment,
            post_adjustment=post_adjustment,
        )
        return x


class ResidualLearner(nn.Module):
    """Legacy after-block learner kept for old checkpoint compatibility."""

    def __init__(self, hidden_size: int, bottleneck_dim: int, dropout: float) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.ones(()))
        self.restore_zero_output()

    def restore_zero_output(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        with torch.no_grad():
            self.gate.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.up(self.dropout(F.silu(self.down(x))))
        return x + self.gate.to(update.dtype) * update


class PatternOutputHead(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm(x))


class PatternModules(nn.Module):
    def __init__(self, config: TinyConfig, layout: LearnerLayout) -> None:
        super().__init__()
        layout.validate(config.num_layers)
        self.kind = layout.kind

        if layout.kind == "standalone":
            standalone_config = replace(
                config,
                num_layers=layout.standalone_layers,
                ffn_hidden_size=layout.standalone_ffn_size,
                dropout=layout.dropout,
            )
            self.standalone_blocks = nn.ModuleList(
                [DecoderBlock(standalone_config) for _ in range(layout.standalone_layers)]
            )
            self.output_head = PatternOutputHead(config)
            return

        if layout.mode == "single":
            indices = (normalize_layer_index(layout.single_layer, config.num_layers),)
        else:
            indices = tuple(range(config.num_layers))

        if layout.kind == "residual":
            self.adapters = nn.ModuleDict(
                {
                    str(index): ResidualLearner(
                        config.hidden_size,
                        layout.bottleneck_dim,
                        layout.dropout,
                    )
                    for index in indices
                }
            )
        elif layout.kind == "pre_activation":
            self.pre_adjustments = nn.ModuleDict(
                {
                    str(index): FFNAdjustment(
                        config.hidden_size,
                        config.ffn_hidden_size,
                        layout.bottleneck_dim,
                        layout.dropout,
                    )
                    for index in indices
                }
            )
            self.output_head = PatternOutputHead(config)
        elif layout.kind == "post_activation":
            self.post_adjustments = nn.ModuleDict(
                {
                    str(index): FFNAdjustment(
                        config.ffn_hidden_size,
                        config.ffn_hidden_size,
                        layout.bottleneck_dim,
                        layout.dropout,
                    )
                    for index in indices
                }
            )
            self.output_head = PatternOutputHead(config)

    def restore_zero_effect(self) -> None:
        if self.kind == "residual":
            for module in self.adapters.values():
                module.restore_zero_output()
        elif self.kind == "pre_activation":
            for module in self.pre_adjustments.values():
                module.restore_zero_output()
        elif self.kind == "post_activation":
            for module in self.post_adjustments.values():
                module.restore_zero_output()

    def apply_residual(self, layer_index: int, x: torch.Tensor) -> torch.Tensor:
        if self.kind != "residual":
            return x
        key = str(layer_index)
        return self.adapters[key](x) if key in self.adapters else x

    def pre_adjustment(self, layer_index: int) -> FFNAdjustment | None:
        if self.kind != "pre_activation":
            return None
        key = str(layer_index)
        return self.pre_adjustments[key] if key in self.pre_adjustments else None

    def post_adjustment(self, layer_index: int) -> FFNAdjustment | None:
        if self.kind != "post_activation":
            return None
        key = str(layer_index)
        return self.post_adjustments[key] if key in self.post_adjustments else None

    def forward_standalone(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind != "standalone":
            raise RuntimeError("forward_standalone is only valid for standalone learners")
        for block in self.standalone_blocks:
            x = block(x)
        return x

    def project(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "residual":
            raise RuntimeError("Legacy residual learners use the shared LM head")
        return self.output_head(x)

    def parameter_breakdown(self) -> dict[str, int]:
        head = 0
        if self.kind != "residual":
            head = sum(parameter.numel() for parameter in self.output_head.parameters())
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"core": total - head, "head": head, "total": total}


class TinyPatternLM(nn.Module):
    def __init__(self, config: TinyConfig, learner_layout: LearnerLayout | None = None) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.learner_layout = learner_layout
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.patterns = nn.ModuleDict()
        self.active_pattern: str | None = None
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def add_pattern(self, name: str, *, activate: bool = True) -> None:
        cleaned = name.strip()
        if not cleaned or "." in cleaned:
            raise ValueError("Pattern names must be non-empty and cannot contain '.'")
        if self.learner_layout is None:
            raise RuntimeError("Model was created without a learner layout")
        if cleaned in self.patterns:
            raise ValueError(f"Pattern {cleaned!r} already exists")
        pattern = PatternModules(self.config, self.learner_layout)
        pattern.apply(self._init_weights)
        pattern.restore_zero_effect()
        self.patterns[cleaned] = pattern
        if activate:
            self.active_pattern = cleaned

    def set_active_pattern(self, name: str | None) -> None:
        if name is None:
            self.active_pattern = None
            return
        if name not in self.patterns:
            raise KeyError(f"Unknown pattern: {name!r}")
        self.active_pattern = name

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("Input exceeds max_seq_len")

        x = self.token_embedding(input_ids)
        active = self.patterns[self.active_pattern] if self.active_pattern is not None else None

        if active is not None and active.kind == "standalone":
            x = active.forward_standalone(x)
            logits = active.project(x)
        else:
            for layer_index, block in enumerate(self.blocks):
                x = block(
                    x,
                    pre_adjustment=(active.pre_adjustment(layer_index) if active is not None else None),
                    post_adjustment=(active.post_adjustment(layer_index) if active is not None else None),
                )
                if active is not None:
                    x = active.apply_residual(layer_index, x)

            if active is not None and active.kind in {"pre_activation", "post_activation"}:
                logits = active.project(x)
            else:
                logits = self.lm_head(self.final_norm(x))

        output = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            output["loss"] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return output

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        eos_id: int,
        max_new_tokens: int = 4,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        generated = input_ids
        for _ in range(max_new_tokens):
            if generated.shape[1] >= self.config.max_seq_len:
                break
            logits = self(generated)["logits"][:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_id), dim=1)
            if torch.all(next_id.squeeze(-1) == eos_id):
                break
        return generated

    def learner_parameter_count(self, pattern_name: str | None = None) -> int:
        if pattern_name is None:
            modules = self.patterns.values()
        else:
            modules = (self.patterns[pattern_name],)
        return sum(parameter.numel() for module in modules for parameter in module.parameters())

    def pattern_parameter_breakdown(self, pattern_name: str) -> dict[str, int]:
        return self.patterns[pattern_name].parameter_breakdown()
