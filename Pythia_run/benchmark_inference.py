from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from data import render_chat_prompt
from experiment_utils import load_experiment
from model import build_tokenizer
from pythia_model import PythiaModModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Pythia full/MOD inference cost under identical settings.")
    parser.add_argument("--runs", nargs=2, required=True, metavar=("FROZEN_MOD_RUN", "FULL_FT_RUN"))
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--output", default="inference_benchmark.json")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def read_prompts(path: str) -> list[list[dict[str, str]]]:
    prompts: list[list[dict[str, str]]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if "messages" in payload:
                messages = payload["messages"]
            elif "prompt" in payload:
                messages = [{"role": "user", "content": str(payload["prompt"])}]
            else:
                raise ValueError(f"Prompt line {line_number} needs 'prompt' or 'messages'.")
            prompts.append(messages)
    if not prompts:
        raise ValueError("Prompt file contains no prompts.")
    return prompts


def benchmark_once(model, tokenizer, config, messages, batch_size: int, max_new_tokens: int, device):
    prompt = render_chat_prompt(messages, open_assistant=True, eos_token=tokenizer.eos_token)
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded.input_ids.to(device).repeat(batch_size, 1)
    attention_mask = torch.ones_like(input_ids)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    synchronize(device)
    first_token_end = time.perf_counter()
    cache = output["past_key_values"]
    first_scores = output["logits"][:, -1].float().clone()
    if first_scores.shape[-1] > len(tokenizer):
        first_scores[..., len(tokenizer) :] = float("-inf")
    next_ids = first_scores.argmax(dim=-1, keepdim=True)
    generated = 1
    decode_start = time.perf_counter()
    with torch.inference_mode():
        while generated < max_new_tokens:
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)], dim=1
            )
            output = model(
                input_ids=next_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            cache = output["past_key_values"]
            next_scores = output["logits"][:, -1].float().clone()
            if next_scores.shape[-1] > len(tokenizer):
                next_scores[..., len(tokenizer) :] = float("-inf")
            next_ids = next_scores.argmax(dim=-1, keepdim=True)
            generated += 1
    synchronize(device)
    end = time.perf_counter()
    decode_time = end - decode_start
    return {
        "prompt_tokens": int(input_ids.numel()),
        "generated_tokens": generated * batch_size,
        "prompt_latency_seconds": first_token_end - start,
        "time_to_first_token_seconds": first_token_end - start,
        "decode_latency_per_token_seconds": decode_time / max((generated - 1) * batch_size, 1),
        "tokens_per_second": (generated * batch_size) / max(end - start, 1e-9),
        "total_generation_seconds": end - start,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
    }


def summarize(label: str, loaded, prompts, args) -> dict[str, Any]:
    measurements = []
    total_iterations = args.warmup_iterations + args.repetitions
    for iteration in range(total_iterations):
        messages = prompts[iteration % len(prompts)]
        measurement = benchmark_once(
            loaded.model, loaded.tokenizer, loaded.config, messages,
            args.batch_size, args.max_new_tokens, loaded.device,
        )
        if iteration >= args.warmup_iterations:
            measurements.append(measurement)
    stats = loaded.model.model_stats()
    active_parameters = stats.total_params
    if hasattr(loaded.model, "parameter_report"):
        active_parameters = loaded.model.parameter_report().active_parameters_per_token
    aggregate: dict[str, Any] = {
        "label": label,
        "model_name": loaded.config.model_name,
        "precision": "bf16" if args.bf16 else str(next(loaded.model.parameters()).dtype),
        "batch_size": args.batch_size,
        "total_parameters": stats.total_params,
        "trainable_parameters": stats.trainable_params,
        "active_parameters_per_forward": active_parameters,
        "estimated_flops_per_generated_token": 2 * active_parameters,
        "flops_note": "Approximation: 2 x active parameters; measured latency/throughput are authoritative.",
    }
    for key in measurements[0]:
        aggregate[key] = statistics.mean(float(item[key]) for item in measurements)
    return aggregate


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable.")
    prompts = read_prompts(args.prompt_file)
    frozen_run, full_run = args.runs
    specifications = [
        ("pythia_1.4b_frozen_mod_all", frozen_run, {}),
        ("pythia_1.4b_without_emb_mod", frozen_run, {"disable_emb_mod": True}),
        ("pythia_1.4b_without_out_mod", frozen_run, {"disable_out_mod": True}),
        ("pythia_1.4b_without_attn_mod", frozen_run, {"disable_attn_mod": True}),
        ("pythia_1.4b_without_ffn_mod", frozen_run, {"disable_ffn_mod": True}),
        ("pythia_1.4b_all_mods_disabled", frozen_run, {"disable_all_mods": True}),
        ("pythia_2.8b_full_finetune", full_run, {}),
    ]
    results = []
    for label, run, ablation in specifications:
        loaded = load_experiment(run, args.device, **ablation)
        if args.bf16:
            loaded.model.to(dtype=torch.bfloat16)
        results.append(summarize(label, loaded, prompts, args))
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for label, model_name in (
        ("pythia_1.4b_untouched", "EleutherAI/pythia-1.4b"),
        ("pythia_2.8b_untouched", "EleutherAI/pythia-2.8b"),
    ):
        tokenizer = build_tokenizer(model_name)
        dtype = torch.bfloat16 if args.bf16 else None
        model = PythiaModModel.from_pretrained(
            model_name,
            vocab_size=len(tokenizer),
            torch_dtype=dtype,
            emb_mod_dim=0,
            out_mod_dim=0,
            attn_mod_dim=0,
            ffn_mod_dim=0,
            control_token_ids=tuple(tokenizer.convert_tokens_to_ids(token) for token in ("<USER>", "<ASSISTANT>", "<SYSTEM>")),
        ).to(args.device).eval()
        config = type("Config", (), {"model_name": model_name, "seq_len": model.config.max_position_embeddings})()
        loaded = type("Loaded", (), {"model": model, "tokenizer": tokenizer, "config": config, "device": torch.device(args.device)})()
        results.append(summarize(label, loaded, prompts, args))
        del loaded, model
        gc.collect()
        torch.cuda.empty_cache()

    output = {"conditions": vars(args), "results": results}
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
