from __future__ import annotations

import copy

import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from ate_model import PythiaATEModel, StageExpandedAttention
from checkpointing import load_checkpoint, save_checkpoint
from config import config_from_dict
from train import build_optimizer, confirm_ate_training


def tiny_base() -> GPTNeoXForCausalLM:
    config = GPTNeoXConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_parallel_residual=True,
    )
    return GPTNeoXForCausalLM(config)


def test_width_and_depth_expansion_preserve_initial_logits_and_cache() -> None:
    base = tiny_base().eval()
    reference = copy.deepcopy(base).eval()
    model = PythiaATEModel.from_base_model(
        base, new_attention_heads=1, new_transformer_layers=1
    ).eval()
    model.configure_ate_plasticity("off")
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = reference(input_ids).logits
        actual = model(input_ids)["logits"]
        cached = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=True,
        )
        next_output = model(
            input_ids=torch.tensor([[5]]),
            attention_mask=torch.ones((1, 5), dtype=torch.long),
            past_key_values=cached["past_key_values"],
            use_cache=True,
        )
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
    assert next_output["logits"].shape == (1, 1, 64)
    assert model.config.hidden_size == 20
    assert model.config.num_attention_heads == 5
    assert model.config.intermediate_size == 80
    assert len(model.base_model.gpt_neox.layers) == 3


def test_off_mode_freezes_original_parameters_but_trains_added_capacity() -> None:
    model = PythiaATEModel.from_base_model(
        tiny_base(), new_attention_heads=1, new_transformer_layers=1
    )
    model.configure_ate_plasticity("off")
    original_ids = model._original_parameter_ids
    new_ids = model._new_parameter_ids
    assert all(not parameter.requires_grad for parameter in model.parameters() if id(parameter) in original_ids)
    assert all(parameter.requires_grad for parameter in model.parameters() if id(parameter) in new_ids)
    labels = torch.tensor([[1, 2, 3, 4]])
    model(input_ids=labels, labels=labels)["loss"].backward()
    assert all(parameter.grad is None for parameter in model.parameters() if id(parameter) in original_ids)
    assert any(parameter.grad is not None for parameter in model.parameters() if id(parameter) in new_ids)


def test_quadratic_plasticity_builds_exact_layerwise_optimizer_rates() -> None:
    model = PythiaATEModel.from_base_model(tiny_base(), new_attention_heads=1)
    model.configure_ate_plasticity("quadratic")
    config = config_from_dict(
        {
            "experiment_type": "ate",
            "lr": 3e-4,
            "plasticity_base_lr": 1e-5,
            "weight_decay": 0.01,
        }
    )
    optimizer = build_optimizer(config, model)
    rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert rates["ate_new_decay"] == pytest.approx(3e-4)
    assert rates["ate_plastic_layer_00_decay"] == pytest.approx(2.5e-6)
    assert rates["ate_plastic_layer_01_decay"] == pytest.approx(1e-5)
    assert rates["ate_plastic_output_decay"] == pytest.approx(1e-5)


def test_ate_checkpoint_round_trip_and_metadata_rejection(tmp_path) -> None:
    model = PythiaATEModel.from_base_model(tiny_base(), new_attention_heads=1)
    model.configure_ate_plasticity("off")
    path = save_checkpoint(
        str(tmp_path), 1, 1, model, None, None,
        {"text_data_format": "plain", "experiment_type": "ate"},
    )
    restored = PythiaATEModel.from_base_model(tiny_base(), new_attention_heads=1)
    restored.configure_ate_plasticity("off")
    load_checkpoint(str(path), restored)
    for left, right in zip(model.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(left, right)

    mismatched = PythiaATEModel.from_base_model(tiny_base(), new_attention_heads=2)
    mismatched.configure_ate_plasticity("off")
    with pytest.raises(RuntimeError, match="ATE checkpoint architecture mismatch"):
        load_checkpoint(str(path), mismatched)


def test_ate_confirmation_requires_exact_yes(tmp_path) -> None:
    model = PythiaATEModel.from_base_model(tiny_base(), new_attention_heads=1)
    model.configure_ate_plasticity("off")
    config = config_from_dict(
        {
            "experiment_type": "ate",
            "run_dir": str(tmp_path),
            "resume": False,
            "checkpoint_path": None,
            "ate_confirm": "NO",
        }
    )
    with pytest.raises(SystemExit) as error:
        confirm_ate_training(config, model)
    assert error.value.code == 0
    config.ate_confirm = "YES"
    confirm_ate_training(config, model)


def test_incremental_width_and_depth_load_preserves_source_logits() -> None:
    source = PythiaATEModel.from_base_model(
        tiny_base(), new_attention_heads=1, new_transformer_layers=1
    )
    source.configure_ate_plasticity("off")
    input_ids = torch.tensor([[1, 2, 3, 4]])
    optimizer = torch.optim.AdamW(source.current_stage_parameters(), lr=1e-3)
    optimizer.zero_grad()
    source(input_ids=input_ids, labels=input_ids)["loss"].backward()
    optimizer.step()
    source.eval()
    with torch.no_grad():
        source_logits = source(input_ids)["logits"]
    source_parameter_values = {id(parameter): parameter.detach().clone() for parameter in source.parameters()}
    source.add_expansion_stage(added_attention_heads=1, added_transformer_layers=1)
    with torch.no_grad():
        target_logits = source(input_ids)["logits"]
    assert torch.equal(source_logits, target_logits)
    assert all(
        torch.equal(value, parameter.detach())
        for parameter in source.parameters()
        if (value := source_parameter_values.get(id(parameter))) is not None
    )
    assert len(source.ate_metadata()["expansion_stages"]) == 2


def test_incremental_confirmation_separates_checkpoint_and_new_additions(capsys) -> None:
    source = PythiaATEModel.from_base_model(
        tiny_base(), new_attention_heads=1, new_transformer_layers=1
    )
    source.configure_ate_plasticity("off")
    target = PythiaATEModel.from_base_model(
        tiny_base(), new_attention_heads=1, new_transformer_layers=2
    )
    target.configure_ate_plasticity("off")
    config = config_from_dict(
        {
            "experiment_type": "ate",
            "run_dir": "unused",
            "resume": False,
            "checkpoint_path": None,
            "new_attn_heads": 1,
            "new_ffn_layers": 2,
            "ate_confirm": "YES",
        }
    )
    config._ate_incremental_payload = {
        "architecture_metadata": source.ate_metadata(),
        "model_state_dict": source.state_dict(),
    }
    config._ate_incremental_source_path = "source/checkpoints/latest.pt"
    config._ate_incremental_requested_heads = 0
    config._ate_incremental_requested_layers = 1
    confirm_ate_training(config, target)
    output = capsys.readouterr().out
    assert "SOURCE ARCHITECTURE" in output
    assert "REQUESTED STAGE" in output
    assert "FINAL ARCHITECTURE" in output
    assert "Added heads / hidden / layers" in output


def test_multiple_trained_width_stages_keep_segmented_norms_and_exact_logits() -> None:
    model = PythiaATEModel.from_base_model(tiny_base(), new_attention_heads=1)
    model.configure_ate_plasticity("off")
    input_ids = torch.tensor([[4, 3, 2, 1]])
    optimizer = torch.optim.AdamW(model.current_stage_parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        model(input_ids=input_ids, labels=input_ids)["loss"].backward()
        optimizer.step()
    model.eval()
    prior_norm = model.base_model.gpt_neox.layers[0].input_layernorm
    prior_weight = prior_norm.new_weight.detach().clone()
    with torch.no_grad():
        expected = model(input_ids)["logits"]
    model.add_expansion_stage(added_attention_heads=1, added_transformer_layers=0)
    with torch.no_grad():
        actual = model(input_ids)["logits"]
    newest_norm = model.base_model.gpt_neox.layers[0].input_layernorm
    assert isinstance(newest_norm.base, type(prior_norm))
    assert newest_norm.base is prior_norm
    assert torch.equal(prior_norm.new_weight, prior_weight)
    assert torch.equal(expected, actual)


def test_small_output_init_reports_nonzero_preservation_difference() -> None:
    model = PythiaATEModel.from_base_model(tiny_base(), new_attention_heads=1)
    input_ids = torch.tensor([[1, 2, 3]])
    model.eval()
    with torch.no_grad():
        source = model(input_ids)["logits"].clone()
    model.add_expansion_stage(
        added_attention_heads=1,
        added_transformer_layers=0,
        output_init="small",
        output_init_scale=1e-3,
    )
    report = model.preservation_probe(source, input_ids)
    assert report["max_abs_logit_difference"] > 0
    assert not report["passed"]


def test_multistage_checkpoint_reconstructs_boundaries_and_logits(tmp_path) -> None:
    model = PythiaATEModel.from_base_model(tiny_base(), new_attention_heads=1, new_transformer_layers=1)
    model.add_expansion_stage(added_attention_heads=0, added_transformer_layers=1)
    model.add_expansion_stage(added_attention_heads=1, added_transformer_layers=0)
    model.configure_ate_plasticity("off", previous_stages_trainable=True)
    input_ids = torch.tensor([[1, 5, 9, 2]])
    with torch.no_grad():
        expected = model(input_ids)["logits"]
    path = save_checkpoint(
        str(tmp_path), 7, 1, model, None, None,
        {"text_data_format": "plain", "experiment_type": "ate"},
    )
    restored = PythiaATEModel.from_base_model(
        tiny_base(), architecture_metadata=model.ate_metadata()
    )
    restored.configure_ate_plasticity("off", previous_stages_trainable=True)
    load_checkpoint(str(path), restored)
    with torch.no_grad():
        actual = restored(input_ids)["logits"]
    assert torch.equal(expected, actual)
    assert restored.ate_metadata() == model.ate_metadata()
    assert [stage["added_attention_heads"] for stage in restored.expansion_stages] == [1, 0, 1]


def test_control_rows_are_new_and_trainable_with_plasticity_off() -> None:
    base = tiny_base()
    original_vocab = base.config.vocab_size
    model = PythiaATEModel._from_unexpanded_base(
        base,
        model_name="test",
        revision=None,
        vocab_size=original_vocab + 3,
        control_token_ids=(original_vocab, original_vocab + 1, original_vocab + 2),
    )
    model.add_expansion_stage(added_attention_heads=1, added_transformer_layers=0)
    model.configure_ate_plasticity("off")
    input_control = model.base_model.gpt_neox.embed_in.base.control_weight
    output_control = model.base_model.embed_out.base.control_weight
    assert id(input_control) in model._new_parameter_ids
    assert id(output_control) in model._new_parameter_ids
    assert input_control.requires_grad and output_control.requires_grad
    assert all(
        not parameter.requires_grad
        for parameter in model.parameters()
        if id(parameter) in model._original_parameter_ids
    )


def test_exact_stage_output_paths_open_on_first_backward() -> None:
    model = PythiaATEModel.from_base_model(tiny_base(), new_attention_heads=1)
    model.configure_ate_plasticity("off")
    report = model.gradient_diagnostics(torch.tensor([[1, 2, 3, 4]]))
    assert report["LM head"]["tensors_with_nonzero_gradients"] > 0
    assert report["Attention output"]["tensors_with_nonzero_gradients"] > 0
    assert report["FFN output"]["tensors_with_nonzero_gradients"] > 0


def test_merged_normalization_is_detected_by_preservation_probe() -> None:
    model = PythiaATEModel.from_base_model(tiny_base(), new_attention_heads=1)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    model.eval()
    with torch.no_grad():
        source = model(input_ids)["logits"].clone()
    model.add_expansion_stage(added_attention_heads=1, added_transformer_layers=0)
    # Deliberately replace the final segmented norm with one global norm.
    old = model.base_model.gpt_neox.final_layer_norm
    merged = torch.nn.LayerNorm(old.normalized_shape[0], eps=old.eps)
    merged.to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        merged.weight.copy_(old.weight)
        merged.bias.copy_(old.bias)
    model.base_model.gpt_neox.final_layer_norm = merged
    assert not model.preservation_probe(source, input_ids)["passed"]


def test_previous_stage_freeze_and_stage_specific_learning_rates() -> None:
    model = PythiaATEModel.from_base_model(tiny_base(), new_attention_heads=1)
    model.add_expansion_stage(added_attention_heads=0, added_transformer_layers=1)
    model.configure_ate_plasticity("off", previous_stages_trainable=False)
    assert all(not parameter.requires_grad for parameter in model.previous_stage_parameters())
    assert all(parameter.requires_grad for parameter in model.current_stage_parameters())
    specs = model.ate_optimizer_specs(
        training_lr=3e-4,
        plastic_base_lr=1e-5,
        current_stage_lr=2e-4,
        previous_stage_lr=7e-6,
    )
    assert {name: lr for name, lr, _ in specs} == {"ate_new": pytest.approx(2e-4)}
    model.configure_ate_plasticity("off", previous_stages_trainable=True)
    specs = model.ate_optimizer_specs(
        training_lr=3e-4,
        plastic_base_lr=1e-5,
        current_stage_lr=2e-4,
        previous_stage_lr=7e-6,
    )
    rates = {name: lr for name, lr, _ in specs}
    assert rates["ate_new"] == pytest.approx(2e-4)
    assert rates["ate_previous"] == pytest.approx(7e-6)


def test_legacy_padded_control_rows_migrate_without_suffix_assumption() -> None:
    model = PythiaATEModel._build_legacy(
        tiny_base(),
        model_name="test",
        revision=None,
        original_vocab_size=64,
        vocab_size=64,
        control_token_ids=(61, 62, 63),
        new_attention_heads=1,
        new_transformer_layers=1,
        gradient_checkpointing=False,
    ).eval()
    input_ids = torch.tensor([[61, 2, 62, 3]])
    with torch.no_grad():
        expected = model(input_ids)["logits"].clone()
    model.migrate_legacy_control_rows()
    with torch.no_grad():
        actual = model(input_ids)["logits"]
    assert torch.equal(expected, actual)
    assert model.ate_metadata()["architecture_version"] == "pythia_ate_v2_staged"


def test_legacy_trained_stage_gains_width_and_depth_without_numeric_drift() -> None:
    model = PythiaATEModel._build_legacy(
        tiny_base(),
        model_name="test",
        revision=None,
        original_vocab_size=64,
        vocab_size=64,
        control_token_ids=(61, 62, 63),
        new_attention_heads=1,
        new_transformer_layers=1,
        gradient_checkpointing=False,
    )
    model.configure_ate_plasticity("off")
    input_ids = torch.tensor([[3, 1, 4, 1]])
    optimizer = torch.optim.AdamW(model.current_stage_parameters(), lr=1e-3)
    optimizer.zero_grad()
    model(input_ids=input_ids, labels=input_ids)["loss"].backward()
    optimizer.step()
    model.eval()
    model.migrate_legacy_control_rows()
    source_attention = model.base_model.gpt_neox.layers[0].attention
    with torch.no_grad():
        expected = model(input_ids)["logits"]

    model.add_expansion_stage(added_attention_heads=1, added_transformer_layers=1)
    with torch.no_grad():
        actual = model(input_ids)["logits"]

    expanded_attention = model.base_model.gpt_neox.layers[0].attention
    assert isinstance(expanded_attention, StageExpandedAttention)
    assert expanded_attention.base_attention is source_attention
    assert expanded_attention._attention_head_count(source_attention) == 5
    assert expanded_attention.new_heads == 1
    assert torch.equal(actual, expected)


def test_segmented_attention_cached_decoding_matches_full_forward() -> None:
    model = PythiaATEModel.from_base_model(
        tiny_base(), new_attention_heads=1, new_transformer_layers=1
    ).eval()
    model.add_expansion_stage(added_attention_heads=1, added_transformer_layers=1)
    input_ids = torch.tensor([[2, 7, 1, 8]])
    with torch.no_grad():
        full = model(input_ids=input_ids, use_cache=False)["logits"]
        cached = None
        pieces = []
        for index in range(input_ids.shape[1]):
            output = model(
                input_ids=input_ids[:, index : index + 1],
                attention_mask=torch.ones((1, index + 1), dtype=torch.long),
                past_key_values=cached,
                use_cache=True,
            )
            cached = output["past_key_values"]
            pieces.append(output["logits"])
    incremental = torch.cat(pieces, dim=1)
    assert torch.allclose(incremental, full, atol=1e-6, rtol=1e-6)
