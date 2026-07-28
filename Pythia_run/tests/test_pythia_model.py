from __future__ import annotations

import copy

import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

from pythia_model import (
    LEGACY_TOKEN_MOD_ARCHITECTURE_VERSION,
    PLASTIC_TOKEN_MOD_ARCHITECTURE_VERSION,
    PythiaModModel,
)
from train import build_optimizer, limit_supervised_targets, trim_batch_to_target_budget


def tiny_config() -> GPTNeoXConfig:
    return GPTNeoXConfig(
        vocab_size=64,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cache=True,
    )


def grouped_config() -> GPTNeoXConfig:
    return GPTNeoXConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=24,
        num_attention_heads=2,
        max_position_embeddings=32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cache=False,
    )


def build_mod() -> PythiaModModel:
    return PythiaModModel.from_config(
        tiny_config(),
        emb_mod_dim=4,
        out_mod_dim=4,
        attn_mod_dim=4,
        ffn_mod_dim=8,
        emb_mod_scale=0.5,
        out_mod_scale=0.5,
        attn_mod_scale=0.5,
        ffn_mod_scale=0.5,
        learnable_mod_scales=True,
    )


def test_frozen_pythia_only_modifiers_train_and_receive_gradients() -> None:
    model = build_mod()
    model.freeze_base_model()
    ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, 2, 3, 4]])
    model(input_ids=ids, labels=labels)["loss"].backward()

    for name, parameter in model.named_parameters():
        if "modifier" in name:
            assert parameter.requires_grad
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        else:
            assert not parameter.requires_grad
            assert parameter.grad is None


def test_full_pythia_has_no_mods_and_all_parameters_trainable() -> None:
    model = PythiaModModel.from_config(
        tiny_config(), emb_mod_dim=0, out_mod_dim=0, attn_mod_dim=0, ffn_mod_dim=0
    )
    model.enable_full_finetuning()
    assert not model.modifier_parameters()
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_controlled_plasticity_selects_only_requested_base_paths() -> None:
    model = build_mod()
    names = model.configure_base_plasticity(
        last_n_layers=1,
        layer_norms=True,
        ffn_output=True,
        final_layer_norm=True,
    )
    assert any("layers.1.attention.base_module.query_key_value.weight" in name for name in names)
    assert not any("layers.0.attention.base_module.query_key_value.weight" in name for name in names)
    assert any("layers.0.mlp.base_module.dense_4h_to_h.weight" in name for name in names)
    assert any("layers.0.input_layernorm.weight" in name for name in names)
    assert any("final_layer_norm.weight" in name for name in names)
    assert not any("embed_in.weight" in name or "embed_out" in name for name in names)

    ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, 2, 3, 4]])
    model(input_ids=ids, labels=labels)["loss"].backward()
    model.assert_base_gradients_absent()
    plastic_ids = {id(parameter) for parameter in model.plastic_base_parameters()}
    assert all(parameter.grad is not None for parameter in model.plastic_base_parameters())
    assert all(
        parameter.grad is None
        for parameter in model.base_parameters()
        if id(parameter) not in plastic_ids
    )


def test_plastic_optimizer_uses_separate_learning_rates() -> None:
    from types import SimpleNamespace

    model = build_mod()
    model.configure_base_plasticity(last_n_layers=1, layer_norms=True)
    config = SimpleNamespace(
        lr=0.0,
        modifier_lr=3e-4,
        lora_lr=None,
        plastic_lr=1e-5,
        plastic_layernorm_lr=3e-5,
        weight_decay=0.01,
    )
    optimizer = build_optimizer(config, model)
    assert all(group["lr"] == 3e-4 for group in optimizer.param_groups if group["name"].startswith("modifier_"))
    assert all(
        group["lr"] == 1e-5
        for group in optimizer.param_groups
        if group["name"].startswith("plastic_") and not group["name"].startswith("plastic_norm_")
    )
    assert all(
        group["lr"] == 3e-5
        for group in optimizer.param_groups
        if group["name"].startswith("plastic_norm_")
    )


def test_plastic_sparse_checkpoint_round_trip_and_selection_mismatch() -> None:
    torch.manual_seed(101)
    model = build_mod().eval()
    model.configure_base_plasticity(last_n_layers=1, final_layer_norm=True)
    with torch.no_grad():
        for parameter in model.plastic_base_parameters():
            parameter.add_(0.01)
        expected = model(input_ids=torch.tensor([[1, 2, 3]]))["logits"]
    state = model.sparse_checkpoint_state()
    assert state["architecture_version"] == PLASTIC_TOKEN_MOD_ARCHITECTURE_VERSION

    torch.manual_seed(101)
    restored = build_mod().eval()
    restored.configure_base_plasticity(last_n_layers=1, final_layer_norm=True)
    restored.load_sparse_checkpoint_state(state)
    with torch.no_grad():
        actual = restored(input_ids=torch.tensor([[1, 2, 3]]))["logits"]
    torch.testing.assert_close(actual, expected)

    mismatched = build_mod()
    mismatched.configure_base_plasticity(layer_norms=True)
    with pytest.raises(RuntimeError, match="selection mismatch"):
        mismatched.load_sparse_checkpoint_state(state)


def test_zero_effect_mod_matches_untouched_gpt_neox() -> None:
    torch.manual_seed(7)
    untouched = GPTNeoXForCausalLM(tiny_config()).eval()
    wrapped_base = GPTNeoXForCausalLM(tiny_config()).eval()
    wrapped_base.load_state_dict(copy.deepcopy(untouched.state_dict()))
    wrapped = PythiaModModel(
        wrapped_base,
        emb_mod_dim=4,
        out_mod_dim=4,
        attn_mod_dim=4,
        ffn_mod_dim=8,
    ).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        reference = untouched(input_ids=ids, attention_mask=mask).logits
        actual = wrapped(input_ids=ids, attention_mask=mask)["logits"]
    torch.testing.assert_close(actual, reference, rtol=0.0, atol=1e-6)


def test_modifier_shapes_ablation_and_round_trip() -> None:
    torch.manual_seed(19)
    model = build_mod().eval()
    ids = torch.tensor([[1, 2, 3]])
    assert model.input_modifier(ids).shape == (1, 3, tiny_config().hidden_size)
    assert model.output_modifier.table(ids).shape == (1, 3, 4)
    assert model.attention_modifier(ids).shape == (1, 3, 4)
    assert model.ffn_modifier(ids).shape == (1, 3, 8)
    torch.nn.init.normal_(model.input_modifier.proj.weight, std=0.01)
    torch.nn.init.normal_(model.output_modifier.proj.weight, std=0.01)
    for reader in (*model.attention_modifier_projections, *model.ffn_modifier_projections):
        torch.nn.init.normal_(reader.proj.weight, std=0.01)

    with torch.no_grad():
        enabled = model(input_ids=ids)["logits"]
        model.set_mod_ablation(disable_all=True)
        disabled = model(input_ids=ids)["logits"]
    assert not torch.allclose(enabled, disabled)

    state = model.modifier_state_dict()
    torch.manual_seed(19)
    restored = build_mod().eval()
    restored.load_modifier_state_dict(state)
    with torch.no_grad():
        restored_enabled = restored(input_ids=ids)["logits"]
    torch.testing.assert_close(restored_enabled, enabled)
    restored.set_mod_ablation(disable_all=True)
    with torch.no_grad():
        restored_logits = restored(input_ids=ids)["logits"]
    torch.testing.assert_close(restored_logits, disabled)


def test_sparse_checkpoint_restores_mods_and_added_token_rows() -> None:
    torch.manual_seed(23)
    config = tiny_config()
    config.vocab_size = 68
    model = PythiaModModel(
        GPTNeoXForCausalLM(config),
        emb_mod_dim=4,
        out_mod_dim=4,
        attn_mod_dim=4,
        ffn_mod_dim=8,
        original_vocab_size=64,
        control_token_ids=(64, 65, 66, 67),
    )
    with torch.no_grad():
        model.embedding_modifier.proj.weight.normal_(std=0.01)
        model.output_modifier.proj.weight.normal_(std=0.01)
        for reader in (*model.attention_modifier_projections, *model.ffn_modifier_projections):
            reader.proj.weight.normal_(std=0.01)
        model.base_model.get_input_embeddings().weight[64:].fill_(0.25)
        model.base_model.get_output_embeddings().weight[64:].fill_(-0.5)
        expected_logits = model(input_ids=torch.tensor([[1, 2, 64]]))["logits"]
    state = model.sparse_checkpoint_state()

    torch.manual_seed(23)
    restored = PythiaModModel(
        GPTNeoXForCausalLM(config),
        emb_mod_dim=4,
        out_mod_dim=4,
        attn_mod_dim=4,
        ffn_mod_dim=8,
        original_vocab_size=64,
        control_token_ids=(64, 65, 66, 67),
    )
    restored.load_sparse_checkpoint_state(state)
    torch.testing.assert_close(
        restored.base_model.get_input_embeddings().weight[64:],
        torch.full_like(restored.base_model.get_input_embeddings().weight[64:], 0.25),
    )
    torch.testing.assert_close(
        restored.base_model.get_output_embeddings().weight[64:],
        torch.full_like(restored.base_model.get_output_embeddings().weight[64:], -0.5),
    )
    torch.testing.assert_close(
        restored.embedding_modifier.proj.weight, model.embedding_modifier.proj.weight
    )
    with torch.no_grad():
        restored_logits = restored(input_ids=torch.tensor([[1, 2, 64]]))["logits"]
    torch.testing.assert_close(restored_logits, expected_logits)


def test_v11_single_table_sparse_checkpoint_migrates() -> None:
    torch.manual_seed(29)
    model = build_mod().eval()
    state = model.sparse_checkpoint_state()
    legacy_state = dict(state)
    legacy_state["architecture_version"] = LEGACY_TOKEN_MOD_ARCHITECTURE_VERSION
    legacy_state.pop("attn_unique_mod_count")
    legacy_state.pop("ffn_unique_mod_count")
    legacy_state.pop("attention_modifier_layer_map")
    legacy_state.pop("ffn_modifier_layer_map")
    legacy_modifiers = {}
    for name, tensor in state["modifier_state_dict"].items():
        name = name.replace("attention_modifiers.0.", "attention_modifier.")
        name = name.replace("ffn_modifiers.0.", "ffn_modifier.")
        legacy_modifiers[name] = tensor
    legacy_state["modifier_state_dict"] = legacy_modifiers

    torch.manual_seed(29)
    restored = build_mod().eval()
    restored.load_sparse_checkpoint_state(legacy_state)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(restored(input_ids=ids)["logits"], model(input_ids=ids)["logits"])


def test_v11_checkpoint_is_rejected_for_grouped_tables() -> None:
    state = build_mod().sparse_checkpoint_state()
    state["architecture_version"] = LEGACY_TOKEN_MOD_ARCHITECTURE_VERSION
    grouped = PythiaModModel.from_config(
        tiny_config(), attn_mod_dim=4, ffn_mod_dim=8,
        attn_unique_mod_count=2, ffn_unique_mod_count=1,
    )
    with pytest.raises(RuntimeError, match="compatible only"):
        grouped.load_sparse_checkpoint_state(state)


def test_grouped_sparse_checkpoint_rejects_different_mapping() -> None:
    source = PythiaModModel.from_config(
        grouped_config(), attn_mod_dim=2, ffn_mod_dim=2,
        attn_unique_mod_count=8, ffn_unique_mod_count=6,
    )
    state = source.sparse_checkpoint_state()
    target = PythiaModModel.from_config(
        grouped_config(), attn_mod_dim=2, ffn_mod_dim=2,
        attn_unique_mod_count=6, ffn_unique_mod_count=6,
    )
    with pytest.raises(RuntimeError, match="mapping mismatch"):
        target.load_sparse_checkpoint_state(state)


def test_target_budget_masks_to_exact_count() -> None:
    labels = torch.tensor([[-100, 1, 2, 3], [-100, 4, 5, 6]])
    limited = limit_supervised_targets(labels, 4)
    assert int(limited.ne(-100).sum().item()) == 4
    assert limited.tolist() == [[-100, 1, 2, 3], [-100, 4, -100, -100]]
    ids = torch.arange(8).reshape(2, 4)
    trimmed_ids, trimmed_labels, trimmed_mask = trim_batch_to_target_budget(
        ids, labels, torch.ones_like(ids), 4
    )
    assert trimmed_ids.shape == (2, 4)
    assert int(trimmed_labels.ne(-100).sum()) == 4
    assert trimmed_mask.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]


def test_pythia_mod_cached_generation_matches_full_forward() -> None:
    torch.manual_seed(31)
    model = build_mod().eval()
    torch.nn.init.normal_(model.input_modifier.proj.weight, std=0.01)
    torch.nn.init.normal_(model.output_modifier.proj.weight, std=0.01)
    for reader in (*model.attention_modifier_projections, *model.ffn_modifier_projections):
        torch.nn.init.normal_(reader.proj.weight, std=0.01)
    prefix = torch.tensor([[1, 2, 3]])
    full = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = model(input_ids=full)["logits"][:, -1]
        first = model(input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=True)
        next_mask = torch.ones((1, 4), dtype=torch.long)
        cached = model(
            input_ids=torch.tensor([[4]]),
            attention_mask=next_mask,
            past_key_values=first["past_key_values"],
            use_cache=True,
        )["logits"][:, -1]
    torch.testing.assert_close(cached, expected, rtol=1e-5, atol=1e-5)


def test_v11_has_four_independent_families_and_layer_specific_readers() -> None:
    model = build_mod()
    assert model.input_modifier is not None
    assert model.output_modifier is not None
    assert model.attention_modifier is not None
    assert model.ffn_modifier is not None
    assert model.input_modifier.table.weight is not model.output_modifier.table.weight
    assert model.input_modifier.proj.weight is not model.output_modifier.proj.weight
    assert len(model.attention_modifier_projections) == tiny_config().num_hidden_layers
    assert len(model.ffn_modifier_projections) == tiny_config().num_hidden_layers
    assert len({id(reader.proj.weight) for reader in model.attention_modifier_projections}) == 2
    assert len({id(reader.proj.weight) for reader in model.ffn_modifier_projections}) == 2


def test_grouped_tables_use_independent_contiguous_attention_and_ffn_maps() -> None:
    model = PythiaModModel.from_config(
        grouped_config(),
        attn_mod_dim=4,
        ffn_mod_dim=6,
        attn_unique_mod_count=8,
        ffn_unique_mod_count=6,
    )
    assert len(model.attention_modifiers) == 8
    assert len(model.ffn_modifiers) == 6
    assert model.attention_modifier_layer_map == tuple(group for group in range(8) for _ in range(3))
    assert model.ffn_modifier_layer_map == tuple(group for group in range(6) for _ in range(4))
    assert len({id(memory.table.weight) for memory in model.attention_modifiers}) == 8
    assert len({id(memory.table.weight) for memory in model.ffn_modifiers}) == 6
    assert len({id(reader.proj.weight) for reader in model.attention_modifier_projections}) == 24
    assert len({id(reader.proj.weight) for reader in model.ffn_modifier_projections}) == 24


def test_one_unique_mod_per_layer_is_supported() -> None:
    model = PythiaModModel.from_config(
        grouped_config(), attn_mod_dim=2, ffn_mod_dim=2,
        attn_unique_mod_count=24, ffn_unique_mod_count=24,
    )
    assert model.attention_modifier_layer_map == tuple(range(24))
    assert model.ffn_modifier_layer_map == tuple(range(24))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"attn_mod_dim": 4, "attn_unique_mod_count": 0}, "at least 1"),
        ({"attn_mod_dim": 4, "attn_unique_mod_count": 25}, "exceeds decoder layer count"),
        ({"attn_mod_dim": 4, "attn_unique_mod_count": 7}, "cannot evenly split"),
        ({"attn_mod_dim": 0, "attn_unique_mod_count": 2}, "requires an enabled"),
        ({"ffn_mod_dim": 4, "ffn_unique_mod_count": 5}, "cannot evenly split"),
    ],
)
def test_invalid_unique_mod_counts_are_rejected(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PythiaModModel.from_config(grouped_config(), **kwargs)


def test_grouped_tables_receive_gradients_and_remain_distinct() -> None:
    torch.manual_seed(47)
    model = PythiaModModel.from_config(
        grouped_config(),
        attn_mod_dim=2,
        ffn_mod_dim=2,
        attn_unique_mod_count=8,
        ffn_unique_mod_count=6,
    )
    model.freeze_base_model()
    for reader in (*model.attention_modifier_projections, *model.ffn_modifier_projections):
        torch.nn.init.normal_(reader.proj.weight, std=0.02)
    ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, 2, 3, 4]])
    model(input_ids=ids, labels=labels)["loss"].backward()
    for memory in (*model.attention_modifiers, *model.ffn_modifiers):
        assert memory.table.weight.grad is not None
        assert memory.table.weight.grad.abs().sum() > 0


def test_output_mod_adds_residual_logits_before_loss() -> None:
    torch.manual_seed(37)
    model = build_mod().eval()
    model.set_mod_ablation(disable_emb=True, disable_attn=True, disable_ffn=True)
    torch.nn.init.normal_(model.output_modifier.proj.weight, std=0.03)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        enabled = model(input_ids=ids, output_hidden_states=True)
        hidden = enabled["hidden_states"][-1]
        expected_delta = model.output_modifier(hidden).to(enabled["logits"].dtype)
        model.set_mod_ablation(disable_out=True)
        base = model(input_ids=ids)["logits"]
    torch.testing.assert_close(enabled["logits"], base + expected_delta, rtol=1e-5, atol=1e-5)


def test_exact_v11_parameter_formula_and_scale_count() -> None:
    fixed = PythiaModModel.from_config(
        tiny_config(), emb_mod_dim=4, out_mod_dim=4, attn_mod_dim=4, ffn_mod_dim=8
    )
    learned = build_mod()
    expected_fixed = (
        64 * 4 + 24 * 4
        + 64 * 4 + 24 * 4
        + 64 * 4 + 2 * 24 * 4
        + 64 * 8 + 2 * 24 * 8
    )
    assert sum(parameter.numel() for parameter in fixed.modifier_parameters()) == expected_fixed
    assert sum(parameter.numel() for parameter in learned.modifier_parameters()) == expected_fixed + 6


def test_grouped_parameter_formula_multiplies_only_token_tables() -> None:
    model = PythiaModModel.from_config(
        grouped_config(),
        attn_mod_dim=4,
        ffn_mod_dim=6,
        attn_unique_mod_count=8,
        ffn_unique_mod_count=6,
    )
    expected = (
        8 * 32 * 4 + 24 * 8 * 4
        + 6 * 32 * 6 + 24 * 8 * 6
    )
    assert sum(parameter.numel() for parameter in model.modifier_parameters()) == expected


def test_private_table_gradient_coverage_matches_v11() -> None:
    torch.manual_seed(41)
    model = build_mod()
    model.freeze_base_model()
    torch.nn.init.normal_(model.input_modifier.proj.weight, std=0.02)
    torch.nn.init.normal_(model.output_modifier.proj.weight, std=0.02)
    for reader in (*model.attention_modifier_projections, *model.ffn_modifier_projections):
        torch.nn.init.normal_(reader.proj.weight, std=0.02)
    ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, 2, 3, 4]])
    model(input_ids=ids, labels=labels)["loss"].backward()

    selected = set(ids[:, :-1].flatten().tolist())
    for table in (
        model.input_modifier.table.weight,
        model.attention_modifier.table.weight,
        model.ffn_modifier.table.weight,
    ):
        row_norms = table.grad.abs().sum(dim=1)
        assert all(row_norms[token_id] > 0 for token_id in selected)
        assert all(row_norms[token_id] == 0 for token_id in set(range(64)) - selected)
    output_row_norms = model.output_modifier.table.weight.grad.abs().sum(dim=1)
    assert torch.all(output_row_norms > 0)


def test_old_sparse_checkpoint_is_rejected() -> None:
    model = build_mod()
    state = model.sparse_checkpoint_state()
    del state["architecture_version"]
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        build_mod().load_sparse_checkpoint_state(state)


def test_control_rows_and_entire_base_remain_strictly_frozen() -> None:
    torch.manual_seed(43)
    config = tiny_config()
    config.vocab_size = 68
    model = PythiaModModel(
        GPTNeoXForCausalLM(config),
        emb_mod_dim=4,
        out_mod_dim=4,
        attn_mod_dim=4,
        ffn_mod_dim=8,
        original_vocab_size=64,
        control_token_ids=(64, 65, 66, 67),
    )
    model.freeze_base_model()
    assert all(not parameter.requires_grad for parameter in model.base_model.parameters())
    input_before = model.base_model.get_input_embeddings().weight[64:].detach().clone()
    output_before = model.base_model.get_output_embeddings().weight[64:].detach().clone()
    optimizer = torch.optim.AdamW(model.modifier_parameters(), lr=1e-2)
    ids = torch.tensor([[64, 1, 65, 2]])
    labels = torch.tensor([[-100, 1, 65, 2]])
    model(input_ids=ids, labels=labels)["loss"].backward()
    model.assert_base_gradients_absent()
    assert all(parameter.grad is None for parameter in model.base_model.parameters())
    optimizer.step()
    torch.testing.assert_close(model.base_model.get_input_embeddings().weight[64:], input_before)
    torch.testing.assert_close(model.base_model.get_output_embeddings().weight[64:], output_before)


def test_learnable_scales_start_nonzero_and_zero_is_rejected() -> None:
    model = build_mod()
    scales = [
        model.input_modifier.scale,
        model.output_modifier.scale,
        *(reader.scale for reader in model.attention_modifier_projections),
        *(reader.scale for reader in model.ffn_modifier_projections),
    ]
    assert len(scales) == 6
    assert all(isinstance(scale, torch.nn.Parameter) for scale in scales)
    assert all(float(scale.detach()) == 0.5 for scale in scales)
    with pytest.raises(ValueError, match="must start nonzero"):
        PythiaModModel.from_config(
            tiny_config(), emb_mod_dim=4, emb_mod_scale=0.0, learnable_mod_scales=True
        )


def test_mod_wrappers_preserve_native_parallel_residual_layer_topology() -> None:
    model = build_mod()
    for layer in model.base_model.gpt_neox.layers:
        assert isinstance(layer, GPTNeoXLayer)
        assert layer.use_parallel_residual is True
        assert layer.forward.__func__ is GPTNeoXLayer.forward
        assert type(layer.attention).__name__ == "_AttentionModifierWrapper"
        assert type(layer.mlp).__name__ == "_FFNModifierWrapper"
