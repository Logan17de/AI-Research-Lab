from __future__ import annotations

import math
from dataclasses import asdict, dataclass
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
    mode: str = "single"
    bottleneck_dim: int = 128
    single_layer: int = -1
    dropout: float = 0.0

    def validate(self, num_layers: int) -> None:
        if self.mode not in {"single", "all"}:
            raise ValueError("learner mode must be single or all")
        if self.bottleneck_dim <= 0:
            raise ValueError("learner bottleneck_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("learner dropout must be in [0, 1)")
        if self.mode == "single":
            normalize_layer_index(self.single_layer, num_layers)

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


class SwiGLU(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class DecoderBlock(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ResidualLearner(nn.Module):
    def __init__(self, hidden_size: int, bottleneck_dim: int, dropout: float) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.ones(()))
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.up(self.dropout(F.silu(self.down(x))))
        return x + self.gate.to(update.dtype) * update


class PatternModules(nn.Module):
    def __init__(self, config: TinyConfig, layout: LearnerLayout) -> None:
        super().__init__()
        layout.validate(config.num_layers)
        if layout.mode == "single":
            indices = (normalize_layer_index(layout.single_layer, config.num_layers),)
        else:
            indices = tuple(range(config.num_layers))
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

    def apply_at_layer(self, layer_index: int, x: torch.Tensor) -> torch.Tensor:
        key = str(layer_index)
        return self.adapters[key](x) if key in self.adapters else x


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
        self.patterns[cleaned] = PatternModules(self.config, self.learner_layout)
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
        for layer_index, block in enumerate(self.blocks):
            x = block(x)
            if self.active_pattern is not None:
                x = self.patterns[self.active_pattern].apply_at_layer(layer_index, x)
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
