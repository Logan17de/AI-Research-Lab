from __future__ import annotations

import sys

import pytest

from config import parse_args


def test_frozen_mod_learning_rate_targets_modifiers(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--experiment-type", "frozen_mod", "--learning-rate", "0.0003"],
    )
    config = parse_args()
    assert config.lr == 0.0
    assert config.modifier_lr == 3e-4
    assert config.freeze_base
    assert (config.mod_dim, config.out_mod_dim, config.attn_mod_dim, config.ffn_mod_dim) == (32, 32, 32, 64)
    assert (config.attn_unique_mod_count, config.ffn_unique_mod_count) == (1, 1)


def test_frozen_mod_explicit_modifier_lr_wins(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py", "--experiment-type", "frozen_mod",
            "--learning-rate", "0.0003", "--modifier-lr", "0.0007",
        ],
    )
    config = parse_args()
    assert config.lr == 0.0
    assert config.modifier_lr == 7e-4


def test_full_finetune_disables_all_adapter_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py", "--experiment-type", "full_finetune",
            "--learning-rate", "0.00001", "--optimizer-state-offload",
            "--attn-unique-mod-count", "8", "--ffn-unique-mod-count", "4",
        ],
    )
    config = parse_args()
    assert config.lr == 1e-5
    assert not config.freeze_base
    assert (config.mod_dim, config.out_mod_dim, config.attn_mod_dim, config.ffn_mod_dim, config.lora_r) == (0, 0, 0, 0, 0)
    assert (config.attn_unique_mod_count, config.ffn_unique_mod_count) == (1, 1)
    assert config.optimizer_state_offload


def test_frozen_mod_accepts_independent_unique_counts(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py", "--experiment-type", "frozen_mod",
            "--attn_unique_mod_count", "8", "--ffn-unique-mod-count", "6",
        ],
    )
    config = parse_args()
    assert (config.attn_unique_mod_count, config.ffn_unique_mod_count) == (8, 6)


def test_ignore_system_prompt_accepts_flag_and_numeric_forms(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--ignore-system-prompt"])
    assert parse_args().ignore_system_prompt

    monkeypatch.setattr(sys, "argv", ["train.py", "--ignore_system_prompt", "0"])
    assert not parse_args().ignore_system_prompt


def test_frozen_mod_accepts_controlled_plasticity(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py", "--experiment-type", "frozen_mod",
            "--plastic-last-n-layers", "4",
            "--plastic-layer-norms", "1",
            "--plastic-ffn-output", "1",
            "--plastic-final-layer-norm", "1",
            "--plastic-lr", "1e-5",
            "--plastic-layernorm-lr", "3e-5",
        ],
    )
    config = parse_args()
    assert config.plastic_last_n_layers == 4
    assert config.plastic_layer_norms
    assert config.plastic_ffn_output
    assert config.plastic_final_layer_norm
    assert config.plastic_lr == 1e-5
    assert config.plastic_layernorm_lr == 3e-5


def test_plasticity_is_rejected_outside_frozen_mod(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--plastic-last-n-layers", "1"])
    with pytest.raises(SystemExit):
        parse_args()


def test_locked_dataset_requires_strict_single_worker_order(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py", "--dataset-manifest", "manifest.json",
            "--strict-sample-order", "1", "--shuffle-buffer", "0", "--num-workers", "0",
        ],
    )
    config = parse_args()
    assert config.strict_sample_order
    assert config.dataset_manifest == "manifest.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py", "--dataset-manifest", "manifest.json",
            "--strict-sample-order", "1", "--shuffle-buffer", "1",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_ate_cli_configures_expansion_and_plasticity(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py", "--experiment-type", "ate",
            "--new-attn-heads", "1", "--new-ffn-layers", "2",
            "--plasticity-mode", "linear", "--plasticity-base-lr", "2e-5",
            "--learning-rate", "3e-4", "--ate-confirm", "YES",
        ],
    )
    config = parse_args()
    assert config.experiment_type == "ate"
    assert config.new_attn_heads == 1
    assert config.new_ffn_layers == 2
    assert config.plasticity_mode == "linear"
    assert config.plasticity_base_lr == 2e-5
    assert config.mod_dim == config.out_mod_dim == 0


def test_ate_flags_are_rejected_outside_ate(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--new-attn-heads", "1"])
    with pytest.raises(SystemExit):
        parse_args()
