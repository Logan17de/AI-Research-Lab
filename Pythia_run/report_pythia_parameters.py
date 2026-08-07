from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, GPTNeoXConfig, GPTNeoXForCausalLM

from pythia_model import PythiaModModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact grouped Token MOD parameter and active-capacity report.")
    parser.add_argument("--model-name", default="EleutherAI/pythia-1.4b")
    parser.add_argument("--comparison-model", default="EleutherAI/pythia-2.8b")
    parser.add_argument("--emb-mod-dim", type=int, default=128)
    parser.add_argument("--out-mod-dim", type=int, default=128)
    parser.add_argument("--attn-mod-dim", type=int, default=128)
    parser.add_argument("--ffn-mod-dim", type=int, default=256)
    parser.add_argument("--attn-unique-mod-count", type=int, default=1)
    parser.add_argument("--ffn-unique-mod-count", type=int, default=1)
    parser.add_argument("--learnable-mod-scales", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def named_parameter_count(model, *, remove_duplicate: bool) -> int:
    return sum(
        parameter.numel()
        for _, parameter in model.named_parameters(remove_duplicate=remove_duplicate)
    )


def unique_storage_parameter_count(model) -> int:
    seen: set[tuple[Any, ...]] = set()
    total = 0
    for parameter in model.parameters():
        if parameter.device.type == "meta":
            key = ("meta_parameter", id(parameter))
        else:
            storage = parameter.untyped_storage()
            key = (
                storage.data_ptr(), storage.nbytes(), parameter.storage_offset(),
                tuple(parameter.shape), tuple(parameter.stride()),
            )
        if key not in seen:
            seen.add(key)
            total += parameter.numel()
    return total


def scale_parameter_count(module) -> int:
    return int(module.scale.numel()) if isinstance(module.scale, torch.nn.Parameter) else 0


def simple_component(module) -> dict[str, int]:
    if module is None:
        return {"table": 0, "projections": 0, "scales": 0, "total": 0}
    table = module.table.weight.numel()
    projections = module.proj.weight.numel()
    scales = scale_parameter_count(module)
    return {
        "table": table,
        "projections": projections,
        "scales": scales,
        "total": table + projections + scales,
    }


def layer_component(memories, readers) -> dict[str, int]:
    if not memories:
        return {"table": 0, "table_count": 0, "projections": 0, "scales": 0, "readers": 0, "total": 0}
    tables = sum(memory.table.weight.numel() for memory in memories)
    projections = sum(reader.proj.weight.numel() for reader in readers)
    scales = sum(scale_parameter_count(reader) for reader in readers)
    return {
        "table": tables,
        "table_count": len(memories),
        "projections": projections,
        "scales": scales,
        "readers": len(readers),
        "total": tables + projections + scales,
    }


def config_parameter_count(config) -> int:
    vocab = int(config.vocab_size)
    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    layers = int(config.num_hidden_layers)
    embedding_and_head = 2 * vocab * hidden if not config.tie_word_embeddings else vocab * hidden
    per_layer = 4 * hidden * hidden + 2 * hidden * intermediate + 9 * hidden + intermediate
    return embedding_and_head + layers * per_layer + 2 * hidden


def main() -> None:
    args = parse_args()
    base_config = AutoConfig.from_pretrained(args.model_name)
    comparison_config = AutoConfig.from_pretrained(args.comparison_model)
    with torch.device("meta"):
        plain = GPTNeoXForCausalLM(base_config)
        comparison_plain = GPTNeoXForCausalLM(comparison_config)
        mod = PythiaModModel(
            GPTNeoXForCausalLM(base_config),
            emb_mod_dim=args.emb_mod_dim,
            out_mod_dim=args.out_mod_dim,
            attn_mod_dim=args.attn_mod_dim,
            ffn_mod_dim=args.ffn_mod_dim,
            attn_unique_mod_count=args.attn_unique_mod_count,
            ffn_unique_mod_count=args.ffn_unique_mod_count,
            learnable_mod_scales=args.learnable_mod_scales,
        )
    mod.freeze_base_model()

    base_count = unique_storage_parameter_count(plain)
    comparison_count = unique_storage_parameter_count(comparison_plain)
    if base_count != config_parameter_count(base_config):
        raise RuntimeError("Pythia-1.4B instantiated and formula parameter counts differ.")
    if comparison_count != config_parameter_count(comparison_config):
        raise RuntimeError("Pythia-2.8B instantiated and formula parameter counts differ.")

    components = {
        "input_mod": simple_component(mod.input_modifier),
        "output_mod": simple_component(mod.output_modifier),
        "attention_mod": layer_component(
            mod.attention_modifiers, mod.attention_modifier_projections
        ),
        "ffn_mod": layer_component(mod.ffn_modifiers, mod.ffn_modifier_projections),
    }
    mod_total = sum(component["total"] for component in components.values())
    trainable = sum(parameter.numel() for parameter in mod.parameters() if parameter.requires_grad)
    total = unique_storage_parameter_count(mod)
    if trainable != mod_total or total != base_count + mod_total:
        raise RuntimeError(
            f"Count invariant failed: trainable={trainable} mod={mod_total} "
            f"total={total} base+mod={base_count + mod_total}"
        )

    hidden = int(base_config.hidden_size)
    vocab = int(base_config.vocab_size)
    input_embedding = vocab * hidden
    base_physical = base_count - input_embedding + hidden
    physical_mod = {
        "input_mod": args.emb_mod_dim + components["input_mod"]["projections"] + components["input_mod"]["scales"],
        "output_mod": components["output_mod"]["total"],
        "attention_mod": (
            args.attn_unique_mod_count * args.attn_mod_dim
            + components["attention_mod"]["projections"]
            + components["attention_mod"]["scales"]
        ),
        "ffn_mod": (
            args.ffn_unique_mod_count * args.ffn_mod_dim
            + components["ffn_mod"]["projections"]
            + components["ffn_mod"]["scales"]
        ),
    }
    ablation_sets = {
        "all_enabled": tuple(components),
        "input_only": ("input_mod",),
        "output_only": ("output_mod",),
        "attention_only": ("attention_mod",),
        "ffn_only": ("ffn_mod",),
        "all_disabled": (),
    }
    ablations = {}
    for name, enabled in ablation_sets.items():
        enabled_stored = sum(components[item]["total"] for item in enabled)
        enabled_accessed = sum(physical_mod[item] for item in enabled)
        ablations[name] = {
            "conventional_active_parameters": base_count + enabled_stored,
            "physically_accessed_unique_parameters_per_token": base_physical + enabled_accessed,
            "enabled_mod_parameters": enabled_stored,
        }

    smoke_config = GPTNeoXConfig(
        vocab_size=vocab,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=16,
        rotary_pct=0.25,
        tie_word_embeddings=False,
        use_cache=False,
    )
    smoke = PythiaModModel(
        GPTNeoXForCausalLM(smoke_config),
        emb_mod_dim=args.emb_mod_dim,
        out_mod_dim=args.out_mod_dim,
        attn_mod_dim=args.attn_mod_dim,
        ffn_mod_dim=args.ffn_mod_dim,
        attn_unique_mod_count=1,
        ffn_unique_mod_count=1,
        learnable_mod_scales=args.learnable_mod_scales,
    ).eval()
    dummy_ids = torch.ones((1, 2), dtype=torch.long)
    with torch.no_grad():
        dummy_output = smoke(input_ids=dummy_ids)

    input_weight = mod.base_model.get_input_embeddings().weight
    output_weight = mod.base_model.get_output_embeddings().weight
    named_tensors = [
        {"name": name, "shape": list(parameter.shape), "parameters": parameter.numel()}
        for name, parameter in mod.named_parameters()
        if parameter.requires_grad
    ]
    result = {
        "architecture_version": mod.architecture_version,
        "architecture": {
            "model_name": args.model_name,
            "layers": base_config.num_hidden_layers,
            "hidden_size": base_config.hidden_size,
            "intermediate_size": base_config.intermediate_size,
            "vocab_size": base_config.vocab_size,
            "attention_heads": base_config.num_attention_heads,
            "tie_word_embeddings": bool(base_config.tie_word_embeddings),
        },
        "comparison_architecture": {
            "model_name": args.comparison_model,
            "layers": comparison_config.num_hidden_layers,
            "hidden_size": comparison_config.hidden_size,
            "intermediate_size": comparison_config.intermediate_size,
            "vocab_size": comparison_config.vocab_size,
            "attention_heads": comparison_config.num_attention_heads,
            "tie_word_embeddings": bool(comparison_config.tie_word_embeddings),
        },
        "counts": {
            "plain_base": base_count,
            "components": components,
            "total_mod": mod_total,
            "total_model": total,
            "trainable": trainable,
            "comparison_model": comparison_count,
            "named_parameters_remove_duplicate_false": named_parameter_count(mod, remove_duplicate=False),
            "named_parameters_remove_duplicate_true": named_parameter_count(mod, remove_duplicate=True),
            "unique_storage_parameters": total,
        },
        "sharing": {
            "input_output_base_same_parameter": input_weight is output_weight,
            "input_output_mod_same_table": (
                mod.input_modifier.table.weight is mod.output_modifier.table.weight
            ),
            "input_output_mod_same_projection": (
                mod.input_modifier.proj.weight is mod.output_modifier.proj.weight
            ),
            "attention_unique_table_count": len(mod.attention_modifiers),
            "attention_layers_per_table": (
                mod.num_layers // len(mod.attention_modifiers) if mod.attention_modifiers else 0
            ),
            "attention_layer_map": list(mod.attention_modifier_layer_map),
            "attention_projection_count": len(mod.attention_modifier_projections),
            "ffn_unique_table_count": len(mod.ffn_modifiers),
            "ffn_layers_per_table": mod.num_layers // len(mod.ffn_modifiers) if mod.ffn_modifiers else 0,
            "ffn_layer_map": list(mod.ffn_modifier_layer_map),
            "ffn_projection_count": len(mod.ffn_modifier_projections),
        },
        "logit_path": (
            "z_base = h @ W_out.T; z_mod = (h @ P_out) @ M_out.T; "
            "final_logits = z_base + alpha_out * z_mod; one vocabulary softmax"
        ),
        "dummy_forward": {
            "method": "same grouped PythiaModModel path with one GPT-NeoX layer and real vocabulary size",
            "output_shape": list(dummy_output["logits"].shape),
            "reports_base_logit_norm": dummy_output["base_logit_norm"] is not None,
            "reports_output_mod_logit_norm": dummy_output["output_mod_logit_norm"] is not None,
        },
        "ablations": ablations,
        "trainable_tensors": named_tensors,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
