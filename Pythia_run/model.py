from __future__ import annotations

from dataclasses import dataclass
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from transformers.cache_utils import Cache, DynamicCache


CONTROL_TOKENS = (
    "<USER>",
    "<ASSISTANT>",
    "<SYSTEM>",
)


def build_tokenizer(tokenizer_name: str, add_control_tokens: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if add_control_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": list(CONTROL_TOKENS)})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if add_control_tokens:
        ids = []
        for token in CONTROL_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if encoded != [token_id]:
                raise RuntimeError(f"Control token {token!r} is not registered as one token: {encoded}")
            ids.append(token_id)
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"Control token IDs must be unique, got {ids}")
    return tokenizer


class LoRALinearAdapter(nn.Module):
    def __init__(self, base_module: nn.Module, r: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base_module = base_module
        self.r = int(r)
        self.scaling = float(alpha) / float(max(self.r, 1))
        self.dropout = nn.Dropout(dropout)

        in_features, out_features = self._infer_dims(base_module)
        if self.r > 0:
            self.lora_a = nn.Parameter(torch.empty(in_features, self.r))
            self.lora_b = nn.Parameter(torch.zeros(self.r, out_features))
            nn.init.normal_(self.lora_a, mean=0.0, std=0.02)
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    @staticmethod
    def _infer_dims(module: nn.Module) -> tuple[int, int]:
        if hasattr(module, "weight"):
            weight = module.weight
            if isinstance(module, nn.Linear):
                return module.in_features, module.out_features
            if weight.ndim == 2:
                return weight.shape[0], weight.shape[1]
        raise TypeError(f"Unsupported LoRA base module: {type(module).__name__}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_module(x)
        if self.r <= 0:
            return output
        delta = torch.matmul(self.dropout(x), self.lora_a)
        delta = torch.matmul(delta, self.lora_b) * self.scaling
        return output + delta

    def freeze_base(self) -> None:
        for param in self.base_module.parameters():
            param.requires_grad_(False)

    def lora_parameters(self) -> list[nn.Parameter]:
        if self.r <= 0:
            return []
        return [self.lora_a, self.lora_b]


@dataclass
class ModelStats:
    total_params: int
    trainable_params: int
    modifier_params: int
    lora_params: int
    base_params: int


class GPTModModel(nn.Module):
    def __init__(
        self,
        base_model: GPT2LMHeadModel,
        mod_dim: int,
        attn_mod_dim: int | None,
        ffn_mod_dim: int | None,
        lora_r: int,
        lora_alpha: float,
        lora_dropout: float,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.mod_dim = int(mod_dim)
        self.attn_mod_dim = int(attn_mod_dim) if attn_mod_dim is not None else int(mod_dim)
        self.ffn_mod_dim = int(ffn_mod_dim) if ffn_mod_dim is not None else int(mod_dim)
        self.hidden_size = self.config.n_embd
        self.ffn_hidden_size = base_model.transformer.h[0].mlp.c_fc.weight.shape[1]
        self.vocab_size = base_model.transformer.wte.num_embeddings
        attention_parameters = inspect.signature(base_model.transformer.h[0].attn.forward).parameters
        self._attention_cache_argument = (
            "past_key_values" if "past_key_values" in attention_parameters else "past_key_value"
        )

        self.embedding_modifier = self._build_optional_embedding(self.mod_dim)
        self.embedding_modifier_proj = self._build_optional_projection(self.mod_dim, self.hidden_size)
        self.attention_modifier = self._build_optional_embedding(self.attn_mod_dim)
        self.attention_modifier_proj = self._build_optional_projection(self.attn_mod_dim, self.hidden_size)
        self.post_activation_modifier = self._build_optional_embedding(self.ffn_mod_dim)
        self.post_activation_modifier_proj = self._build_optional_projection(self.ffn_mod_dim, self.ffn_hidden_size)
        adapter_without_embedding = self.mod_dim <= 0 and any(
            value > 0 for value in (self.attn_mod_dim, self.ffn_mod_dim, lora_r)
        )
        self.control_token_count = min(len(CONTROL_TOKENS), self.vocab_size)
        if adapter_without_embedding:
            # Only the three appended controls receive this tiny tied delta.
            # GPT-2 EOS remains its pretrained row for frozen-base comparisons.
            self.control_token_modifier = nn.Parameter(
                torch.zeros(self.control_token_count, self.hidden_size)
            )
        else:
            self.register_parameter("control_token_modifier", None)

        self._init_modifiers()
        self._inject_lora(lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        mod_dim: int,
        attn_mod_dim: int | None,
        ffn_mod_dim: int | None,
        lora_r: int,
        lora_alpha: float,
        lora_dropout: float,
        vocab_size: int | None = None,
    ) -> "GPTModModel":
        base_model = GPT2LMHeadModel.from_pretrained(model_name)
        if vocab_size is not None and vocab_size != base_model.config.vocab_size:
            base_model.resize_token_embeddings(vocab_size)
            base_model.tie_weights()
        return cls(
            base_model,
            mod_dim=mod_dim,
            attn_mod_dim=attn_mod_dim,
            ffn_mod_dim=ffn_mod_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

    @classmethod
    def from_config(
        cls,
        config: GPT2Config,
        mod_dim: int,
        attn_mod_dim: int | None,
        ffn_mod_dim: int | None,
        lora_r: int,
        lora_alpha: float,
        lora_dropout: float,
    ) -> "GPTModModel":
        base_model = GPT2LMHeadModel(config)
        return cls(
            base_model,
            mod_dim=mod_dim,
            attn_mod_dim=attn_mod_dim,
            ffn_mod_dim=ffn_mod_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

    def _build_optional_embedding(self, dim: int) -> nn.Embedding | None:
        if dim <= 0:
            return None
        return nn.Embedding(self.vocab_size, dim)

    def _build_optional_projection(self, input_dim: int, output_dim: int) -> nn.Linear | None:
        if input_dim <= 0:
            return None
        return nn.Linear(input_dim, output_dim, bias=False)

    def _init_modifiers(self) -> None:
        if self.embedding_modifier is not None:
            nn.init.zeros_(self.embedding_modifier.weight)
        if self.attention_modifier is not None:
            nn.init.normal_(self.attention_modifier.weight, mean=0.0, std=0.02)
        if self.post_activation_modifier is not None:
            nn.init.normal_(self.post_activation_modifier.weight, mean=0.0, std=0.02)
        if self.embedding_modifier_proj is not None:
            nn.init.normal_(self.embedding_modifier_proj.weight, mean=0.0, std=0.02)
        if self.attention_modifier_proj is not None:
            nn.init.zeros_(self.attention_modifier_proj.weight)
        if self.post_activation_modifier_proj is not None:
            nn.init.zeros_(self.post_activation_modifier_proj.weight)

    def _inject_lora(self, lora_r: int, lora_alpha: float, lora_dropout: float) -> None:
        for block in self.base_model.transformer.h:
            block.attn.c_attn = LoRALinearAdapter(block.attn.c_attn, lora_r, lora_alpha, lora_dropout)
            block.attn.c_proj = LoRALinearAdapter(block.attn.c_proj, lora_r, lora_alpha, lora_dropout)
            block.mlp.c_fc = LoRALinearAdapter(block.mlp.c_fc, lora_r, lora_alpha, lora_dropout)
            block.mlp.c_proj = LoRALinearAdapter(block.mlp.c_proj, lora_r, lora_alpha, lora_dropout)

    def freeze_base_model(self) -> None:
        for name, param in self.base_model.named_parameters():
            if ".lora_a" in name or ".lora_b" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        for block in self.base_model.transformer.h:
            block.attn.c_attn.freeze_base()
            block.attn.c_proj.freeze_base()
            block.mlp.c_fc.freeze_base()
            block.mlp.c_proj.freeze_base()
        for param in self.modifier_parameters():
            param.requires_grad_(True)

    def modifier_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        if self.control_token_modifier is not None:
            params.append(self.control_token_modifier)
        for module in (
            self.embedding_modifier,
            self.embedding_modifier_proj,
            self.attention_modifier,
            self.attention_modifier_proj,
            self.post_activation_modifier,
            self.post_activation_modifier_proj,
        ):
            if module is not None:
                params.extend(module.parameters())
        return params

    def _modifier_delta(
        self,
        input_ids: torch.Tensor,
        table: nn.Embedding | None,
        projection: nn.Linear | None,
    ) -> torch.Tensor | None:
        if table is None or projection is None:
            return None
        return projection(table(input_ids))

    def lora_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for block in self.base_model.transformer.h:
            params.extend(block.attn.c_attn.lora_parameters())
            params.extend(block.attn.c_proj.lora_parameters())
            params.extend(block.mlp.c_fc.lora_parameters())
            params.extend(block.mlp.c_proj.lora_parameters())
        return params

    def base_parameters(self) -> list[nn.Parameter]:
        modifier_ids = {id(param) for param in self.modifier_parameters()}
        lora_ids = {id(param) for param in self.lora_parameters()}
        return [param for param in self.parameters() if id(param) not in modifier_ids and id(param) not in lora_ids]

    def model_stats(self) -> ModelStats:
        total_params = sum(param.numel() for param in self.parameters())
        trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
        modifier_params = sum(param.numel() for param in self.modifier_parameters())
        lora_params = sum(param.numel() for param in self.lora_parameters())
        base_params = sum(param.numel() for param in self.base_parameters())
        return ModelStats(
            total_params=total_params,
            trainable_params=trainable_params,
            modifier_params=modifier_params,
            lora_params=lora_params,
            base_params=base_params,
        )

    def effective_embedding_weight(self) -> torch.Tensor:
        """Return the tied input/output token matrix, including Emb MOD."""
        weight = self.base_model.transformer.wte.weight
        if self.embedding_modifier is None or self.embedding_modifier_proj is None:
            return weight
        return weight + self.embedding_modifier_proj(self.embedding_modifier.weight)

    def _embed_input(self, input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        hidden_states = F.embedding(input_ids, weight)
        if self.control_token_modifier is None:
            return hidden_states
        first_control_id = self.vocab_size - self.control_token_count
        control_indices = (input_ids - first_control_id).clamp(0, self.control_token_count - 1)
        active = input_ids.ge(first_control_id).unsqueeze(-1)
        return hidden_states + F.embedding(control_indices, self.control_token_modifier) * active

    def _output_logits(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden_states, weight)
        if self.control_token_modifier is None:
            return logits
        first_control_id = self.vocab_size - self.control_token_count
        control_delta = F.linear(hidden_states, self.control_token_modifier)
        return torch.cat(
            [logits[..., :first_control_id], logits[..., first_control_id:] + control_delta],
            dim=-1,
        )

    @staticmethod
    def _causal_attention_mask(
        attention_mask: torch.Tensor | None,
        hidden_states: torch.Tensor,
        cache_position: torch.Tensor,
        key_length: int,
    ) -> torch.Tensor | None:
        """Build the additive 4D mask expected by GPT-2 attention backends.

        Transformers 5 removed GPT2Model._update_causal_mask. Keeping this
        small implementation local also avoids depending on another private HF
        API while preserving right-padding and cached-decoding semantics.
        """
        query_length = hidden_states.shape[1]
        if attention_mask is None:
            all_keys_valid = True
        else:
            all_keys_valid = bool(attention_mask.all())

        # GPT-2 attention can use its native causal path in these common cases.
        # A one-token cached query may attend to every key currently in cache.
        if all_keys_valid and (cache_position[0].item() == 0 or query_length == 1):
            return None

        dtype = hidden_states.dtype
        min_value = torch.finfo(dtype).min
        key_positions = torch.arange(key_length, device=hidden_states.device)
        future_keys = key_positions.unsqueeze(0) > cache_position.unsqueeze(1)
        blocked = future_keys.unsqueeze(0).unsqueeze(0).expand(hidden_states.shape[0], 1, -1, -1)
        if attention_mask is not None:
            blocked = blocked | ~attention_mask[:, None, None, :].bool()
        return torch.zeros(
            blocked.shape,
            dtype=dtype,
            device=hidden_states.device,
        ).masked_fill(blocked, min_value)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        if input_ids.ndim != 2 or seq_len == 0:
            raise ValueError(f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}")
        if labels is not None and (past_key_values is not None or use_cache):
            raise ValueError("Training labels cannot be combined with cached decoding.")

        if use_cache and past_key_values is None:
            try:
                past_key_values = DynamicCache(config=self.config)
            except TypeError:
                past_key_values = DynamicCache()
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(past_length, past_length + seq_len, device=device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        effective_embedding_weight = self.effective_embedding_weight()
        hidden_states = self._embed_input(input_ids, effective_embedding_weight)
        hidden_states = hidden_states + self.base_model.transformer.wpe(position_ids)
        hidden_states = self.base_model.transformer.drop(hidden_states)

        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape != (batch_size, past_length + seq_len):
                raise ValueError(
                    "attention_mask must be a 2D full-key mask with shape "
                    f"{(batch_size, past_length + seq_len)}, got {tuple(attention_mask.shape)}"
                )
            if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
                raise ValueError("attention_mask must contain only 0 and 1.")
            # This project intentionally supports right padding only. GPT-2's
            # absolute positions require separate position-id handling for left padding.
            if bool(((attention_mask[:, 1:] > attention_mask[:, :-1])).any()):
                raise ValueError("Left padding is unsupported; use right-padded batches.")
            attention_mask = attention_mask.to(device)

        causal_mask = self._causal_attention_mask(
            attention_mask=attention_mask,
            hidden_states=hidden_states,
            cache_position=cache_position,
            key_length=past_length + seq_len,
        )

        all_hidden_states: list[torch.Tensor] | None = [] if output_hidden_states else None

        for block in self.base_model.transformer.h:
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            residual = hidden_states
            attn_input = block.ln_1(hidden_states)
            attention_kwargs = {
                self._attention_cache_argument: past_key_values,
                "cache_position": cache_position,
                "attention_mask": causal_mask,
                "output_attentions": False,
            }
            attn_output = block.attn(attn_input, **attention_kwargs)[0]
            attn_delta = self._modifier_delta(input_ids, self.attention_modifier, self.attention_modifier_proj)
            if attn_delta is not None:
                attn_output = attn_output + attn_delta
            hidden_states = residual + attn_output

            residual = hidden_states
            mlp_input = block.ln_2(hidden_states)
            mlp_hidden = block.mlp.act(block.mlp.c_fc(mlp_input))
            ffn_delta = self._modifier_delta(input_ids, self.post_activation_modifier, self.post_activation_modifier_proj)
            if ffn_delta is not None:
                mlp_hidden = mlp_hidden + ffn_delta
            mlp_hidden = block.mlp.c_proj(mlp_hidden)
            mlp_hidden = block.mlp.dropout(mlp_hidden)
            hidden_states = residual + mlp_hidden

        hidden_states = self.base_model.transformer.ln_f(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
        logits = self._output_logits(hidden_states, effective_embedding_weight)

        output = {"logits": logits}
        if use_cache:
            output["past_key_values"] = past_key_values
        if all_hidden_states is not None:
            output["hidden_states"] = tuple(all_hidden_states)
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
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
            loss = loss_sum / target_count
            output["loss"] = loss
            output["loss_sum"] = loss_sum
            output["target_count"] = target_count
        return output
