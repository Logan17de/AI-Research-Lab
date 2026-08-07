from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import re
from typing import Any, Iterator

from data import chat_jsonl_examples, chat_parquet_examples, qa_sft_blocks
from experiment_utils import greedy_generate_messages, load_experiment
from train import evaluate as teacher_forced_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped teacher-forced and greedy GPT_MOD/Pythia evaluation.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--disable-emb-mod", action="store_true")
    parser.add_argument("--disable-out-mod", action="store_true")
    parser.add_argument("--disable-attn-mod", action="store_true")
    parser.add_argument("--disable-ffn-mod", action="store_true")
    parser.add_argument("--disable-all-mods", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def repeated_trigram(text: str) -> bool:
    words = normalize(text).split()
    trigrams = [tuple(words[index : index + 3]) for index in range(max(len(words) - 2, 0))]
    return len(trigrams) != len(set(trigrams))


def tagged_section(text: str, tag: str) -> str | None:
    match = re.search(
        rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else None


def generation_examples(path: str, data_format: str) -> Iterator[tuple[list[dict[str, str]], str]]:
    context = nullcontext(None) if data_format == "chat_parquet" else open(path, "r", encoding="utf-8")
    with context as handle:
        if data_format == "qa_sft":
            for question, answer in qa_sft_blocks(handle):
                yield [{"role": "user", "content": question}], answer
        else:
            examples = (
                chat_parquet_examples(path)
                if data_format == "chat_parquet"
                else chat_jsonl_examples(handle)
            )
            for messages in examples:
                assistant_positions = [
                    index for index, message in enumerate(messages)
                    if str(message.get("role", "")).strip().lower() == "assistant"
                ]
                if not assistant_positions:
                    continue
                target = assistant_positions[-1]
                yield [dict(message) for message in messages[:target]], str(messages[target]["content"]).strip()


def main() -> None:
    args = parse_args()
    loaded = load_experiment(
        args.run_dir,
        args.device,
        disable_emb_mod=args.disable_emb_mod,
        disable_out_mod=args.disable_out_mod,
        disable_attn_mod=args.disable_attn_mod,
        disable_ffn_mod=args.disable_ffn_mod,
        disable_all_mods=args.disable_all_mods,
    )
    spec = json.loads(Path(args.eval_config).read_text(encoding="utf-8"))
    groups = spec.get("groups", spec)
    if not isinstance(groups, dict) or not groups:
        raise ValueError("eval config must contain a non-empty 'groups' object")
    output_path = Path(args.output or (Path(args.run_dir) / "evaluation.json"))
    generations_path = Path(args.run_dir) / "generations.jsonl"
    generation_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}

    for group_name, group in groups.items():
        path = str(group["dataset_path"])
        data_format = str(group.get("data_format", "chat_jsonl"))
        eval_config = replace(
            loaded.config,
            text_data_path=path,
            eval_text_data_path=path,
            text_data_format=data_format,
            num_workers=0,
            eval_batches=0,
        )
        teacher = teacher_forced_evaluate(eval_config, loaded.model, loaded.tokenizer, loaded.device)
        exact = normalized_exact = success = malformed = premature = repeated = count = 0
        final_exact = final_normalized_exact = final_valid = reasoning_valid = 0
        for messages, reference in generation_examples(path, data_format):
            if args.max_examples > 0 and count >= args.max_examples:
                break
            generated, stop_reason, generated_tokens = greedy_generate_messages(
                loaded, messages, args.max_new_tokens, data_format=data_format
            )
            count += 1
            exact += int(generated.strip() == reference.strip())
            normalized_exact += int(normalize(generated) == normalize(reference))
            success += int(bool(generated.strip()))
            malformed += int(not generated.strip())
            minimum_reasonable_tokens = min(3, max(len(loaded.tokenizer.encode(reference, add_special_tokens=False)), 1))
            premature += int(stop_reason == "eos" and generated_tokens < minimum_reasonable_tokens)
            repeated += int(repeated_trigram(generated))
            reference_final = tagged_section(reference, "output")
            generated_final = tagged_section(generated, "output")
            generated_reasoning = tagged_section(generated, "thinking")
            final_valid += int(generated_final is not None)
            reasoning_valid += int(generated_reasoning is not None)
            if reference_final is not None and generated_final is not None:
                final_exact += int(generated_final == reference_final)
                final_normalized_exact += int(normalize(generated_final) == normalize(reference_final))
            generation_rows.append(
                {
                    "group": group_name,
                    "prompt": messages,
                    "reference_answer": reference,
                    "generated_answer": generated,
                    "reference_final_answer": reference_final,
                    "generated_final_answer": generated_final,
                    "generated_reasoning": generated_reasoning,
                    "checkpoint_step": int(loaded.payload.get("step", 0)),
                    "processed_token_count": int((loaded.payload.get("training_state") or {}).get("tokens_processed", 0)),
                    "model_type": loaded.config.experiment_type,
                    "enabled_mod_components": {
                        "embedding": not args.disable_all_mods and not args.disable_emb_mod and loaded.config.mod_dim > 0,
                        "output": not args.disable_all_mods and not args.disable_out_mod and (loaded.config.out_mod_dim or 0) > 0,
                        "attention": not args.disable_all_mods and not args.disable_attn_mod and (loaded.config.attn_mod_dim or 0) > 0,
                        "ffn": not args.disable_all_mods and not args.disable_ffn_mod and (loaded.config.ffn_mod_dim or 0) > 0,
                    },
                    "decoding": {"mode": "greedy", "max_new_tokens": args.max_new_tokens},
                    "stop_reason": stop_reason,
                    "generated_tokens": generated_tokens,
                }
            )
        denominator = max(count, 1)
        results[group_name] = {
            "teacher_forced": teacher,
            "generation": {
                "examples": count,
                "exact_match": exact / denominator,
                "normalized_exact_match": normalized_exact / denominator,
                "semantic_similarity": None,
                "success_rate": success / denominator,
                "malformed_answer_rate": malformed / denominator,
                "premature_eos_rate": premature / denominator,
                "repeated_trigram_rate": repeated / denominator,
                "reasoning_format_rate": reasoning_valid / denominator,
                "final_answer_format_rate": final_valid / denominator,
                "final_answer_exact_match": final_exact / denominator,
                "final_answer_normalized_exact_match": final_normalized_exact / denominator,
            },
        }

    output = {
        "checkpoint": loaded.checkpoint_path,
        "step": int(loaded.payload.get("step", 0)),
        "groups": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    with generations_path.open("w", encoding="utf-8") as handle:
        for row in generation_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
