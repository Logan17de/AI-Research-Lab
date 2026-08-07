from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from config import TrainConfig, config_from_dict
from checkpointing import CHAT_FORMAT_VERSION, validate_tokenizer_metadata
from model import GPTModModel, build_tokenizer
from data import (
    ASSISTANT_TAG,
    CHAT_DATA_FORMATS,
    SYSTEM_TAG,
    USER_TAG,
    render_chat_prompt,
    truncate_chat_messages_for_prompt,
    without_system_messages,
)


@dataclass
class LoadedChatModel:
    alias: str
    checkpoint_path: str
    config: TrainConfig | None
    tokenizer: object
    model: object
    step: int
    history: list[tuple[str, str]]
    generation_kind: str
    format_version: str = "legacy_unspecified"


@dataclass
class GenerationResult:
    text: str
    stop_reason: str
    stop_token: str
    stop_rank: int | None
    stop_prob: float | None
    stop_raw_rank: int | None
    stop_raw_prob: float | None
    stop_filtered_out: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive chat for GPT_MOD checkpoints.")
    parser.add_argument("checkpoints", nargs="*", help="Alias=/path/to/checkpoint.pt entries.")
    parser.add_argument("--gpt2", action="store_true", help="Include plain GPT-2 small.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", choices=["separate", "continuous"], default="separate")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Strict audit mode: argmax with no sampling or repetition filters.",
    )
    parser.add_argument("--no_cache", action="store_true", help="Recompute the full context instead of using KV cache.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def parse_checkpoint_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Invalid checkpoint spec '{spec}'. Expected Alias=/path/to/checkpoint.pt")
    alias, path = spec.split("=", 1)
    alias = alias.strip()
    path = path.strip()
    if not alias or not path:
        raise ValueError(f"Invalid checkpoint spec '{spec}'. Expected Alias=/path/to/checkpoint.pt")
    return alias, path


def load_gpt_mod_checkpoint(alias: str, checkpoint_path: str, device: torch.device) -> LoadedChatModel:
    resolved_path = str(Path(checkpoint_path).expanduser().resolve())
    checkpoint_size = Path(resolved_path).stat().st_size / (1024 ** 3)
    print(
        f"[{alias}] starting load | checkpoint={resolved_path} | size={checkpoint_size:.2f} GiB",
        flush=True,
    )
    from experiment_utils import load_inference_payload

    payload = load_inference_payload(Path(resolved_path))
    config = config_from_dict(payload["config"])
    if config.experiment_type in {"frozen_mod", "full_finetune", "ate"}:
        from experiment_utils import load_experiment

        loaded = load_experiment(resolved_path, str(device), payload=payload)
        format_version = str(payload.get("format_version") or "legacy_unspecified")
        control_ids = {
            token: loaded.tokenizer.convert_tokens_to_ids(token)
            for token in (USER_TAG, ASSISTANT_TAG, SYSTEM_TAG)
        }
        print(
            f"checkpoint loaded | requested={checkpoint_path} | resolved={resolved_path} | "
            f"step={int(payload.get('step', 0))} | experiment={config.experiment_type} | "
            f"model={config.model_name} | sparse={payload.get('model_state_type') == 'pythia_sparse_mod'} | "
            f"vocab={len(loaded.tokenizer)} | control_ids={control_ids} | eos={loaded.tokenizer.eos_token_id}"
        )
        return LoadedChatModel(
            alias=alias,
            checkpoint_path=resolved_path,
            config=config,
            tokenizer=loaded.tokenizer,
            model=loaded.model,
            step=int(payload.get("step", 0)),
            history=[],
            generation_kind="gpt_mod",
            format_version=format_version,
        )
    tokenizer = build_tokenizer(config.tokenizer_name)
    format_version = str(payload.get("format_version") or "legacy_unspecified")
    if config.text_data_format in CHAT_DATA_FORMATS and format_version != CHAT_FORMAT_VERSION:
        raise RuntimeError(
            f"Incompatible chat checkpoint format {format_version!r}; expected {CHAT_FORMAT_VERSION!r}. "
            "EOT-format checkpoints cannot be loaded by the EOS-per-turn pipeline."
        )
    if payload.get("tokenizer_metadata") is not None:
        validate_tokenizer_metadata(payload, tokenizer)
    model = GPTModModel.from_pretrained(
        model_name=config.model_name,
        mod_dim=config.mod_dim,
        attn_mod_dim=config.attn_mod_dim,
        ffn_mod_dim=config.ffn_mod_dim,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        vocab_size=len(tokenizer),
    )
    if config.freeze_base:
        model.freeze_base_model()
    expected = model.state_dict()
    supplied = payload["model_state_dict"]
    missing = sorted(set(expected) - set(supplied))
    unexpected = sorted(set(supplied) - set(expected))
    shape_mismatches = sorted(
        key for key in set(expected) & set(supplied) if tuple(expected[key].shape) != tuple(supplied[key].shape)
    )
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            f"Checkpoint architecture mismatch: missing={missing} unexpected={unexpected} "
            f"shape_mismatches={shape_mismatches}"
        )
    incompatible = model.load_state_dict(supplied, strict=True)
    model = model.to(device)
    model.eval()
    control_ids = {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in (USER_TAG, ASSISTANT_TAG, SYSTEM_TAG)
    }
    print(
        f"checkpoint loaded | requested={checkpoint_path} | resolved={resolved_path} | "
        f"step={int(payload.get('step', 0))} | variant={config.variant} | format={format_version} | strict=True | "
        f"model={config.model_name} seq_len={config.seq_len} mod={config.mod_dim} "
        f"out_mod={config.out_mod_dim} attn_mod={config.attn_mod_dim} "
        f"ffn_mod={config.ffn_mod_dim} attn_groups={config.attn_unique_mod_count} "
        f"ffn_groups={config.ffn_unique_mod_count} lora_r={config.lora_r} | "
        f"vocab={len(tokenizer)} | control_ids={control_ids} | eos={tokenizer.eos_token_id} | "
        f"missing={list(incompatible.missing_keys)} | unexpected={list(incompatible.unexpected_keys)} | "
        "shape_mismatches=[]"
    )
    return LoadedChatModel(
        alias=alias,
        checkpoint_path=resolved_path,
        config=config,
        tokenizer=tokenizer,
        model=model,
        step=int(payload.get("step", 0)),
        history=[],
        generation_kind="gpt_mod",
        format_version=format_version,
    )


def load_plain_gpt2(device: torch.device) -> LoadedChatModel:
    tokenizer = build_tokenizer("gpt2", add_control_tokens=False)
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    return LoadedChatModel(
        alias="gpt2",
        checkpoint_path="huggingface:gpt2",
        config=None,
        tokenizer=tokenizer,
        model=model,
        step=0,
        history=[],
        generation_kind="gpt2",
        format_version="plain_gpt2",
    )


def uses_qa_sft_prompt(loaded: LoadedChatModel) -> bool:
    return loaded.config is not None and loaded.config.text_data_format == "qa_sft"


def uses_chat_jsonl_prompt(loaded: LoadedChatModel) -> bool:
    return loaded.config is not None and loaded.config.text_data_format in CHAT_DATA_FORMATS


def render_qa_prompt(history: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    pending_question: str | None = None
    for role, text in history:
        content = text.strip()
        if role == "user":
            if pending_question is not None:
                lines.append(f"Q: {pending_question}")
                lines.append("A:")
                lines.append("")
            pending_question = content
            continue
        if pending_question is None:
            continue
        lines.append(f"Q: {pending_question}")
        lines.append(f"A: {content}")
        lines.append("")
        pending_question = None
    if pending_question is not None:
        lines.append(f"Q: {pending_question}")
        lines.append("A:")
    return "\n".join(lines).rstrip()


def render_chat_jsonl_prompt(loaded: LoadedChatModel, history: list[tuple[str, str]], mode: str) -> str:
    if mode == "separate":
        selected = [{"role": "system", "content": text} for role, text in history if role == "system"]
        for role, text in reversed(history):
            if role == "user":
                selected.append({"role": "user", "content": text})
                if loaded.config is not None and loaded.config.ignore_system_prompt:
                    selected = without_system_messages(selected)
                return render_chat_prompt(
                    selected, open_assistant=True, eos_token=loaded.tokenizer.eos_token,
                )
        raise ValueError("Separate chat requires a user turn.")

    messages = [{"role": role, "content": text} for role, text in history]
    if loaded.config is not None and loaded.config.ignore_system_prompt:
        messages = without_system_messages(messages)
    return render_chat_prompt(messages, open_assistant=True, eos_token=loaded.tokenizer.eos_token)


def build_chat_jsonl_prompt(loaded: LoadedChatModel, history: list[tuple[str, str]], mode: str) -> str:
    if mode == "separate":
        selected = [{"role": "system", "content": text} for role, text in history if role == "system"]
        selected.extend({"role": role, "content": text} for role, text in history[-1:] if role == "user")
    else:
        selected = [{"role": role, "content": text} for role, text in history]
    if loaded.config is not None and loaded.config.ignore_system_prompt:
        selected = without_system_messages(selected)
    max_context = 1024 if loaded.config is None else loaded.config.seq_len
    selected, dropped = truncate_chat_messages_for_prompt(loaded.tokenizer, selected, max_context)
    if dropped:
        print(f"[{loaded.alias}] dropped {dropped} complete old chat turn(s) to fit context")
    return render_chat_prompt(selected, open_assistant=True, eos_token=loaded.tokenizer.eos_token)


def render_plain_prompt(history: list[tuple[str, str]], mode: str) -> str:
    if mode == "separate":
        for role, text in reversed(history):
            if role == "user":
                return text
        return ""
    lines: list[str] = []
    for role, text in history:
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {text.strip()}")
    lines.append("Assistant:")
    return "\n".join(lines)


def banned_ngram_tokens(generated_ids: list[int], ngram_size: int) -> set[int]:
    if ngram_size <= 0 or len(generated_ids) + 1 < ngram_size:
        return set()
    prefix = tuple(generated_ids[-(ngram_size - 1) :]) if ngram_size > 1 else tuple()
    banned: set[int] = set()
    limit = len(generated_ids) - ngram_size + 1
    for start in range(max(limit, 0)):
        ngram = generated_ids[start : start + ngram_size]
        if tuple(ngram[:-1]) == prefix:
            banned.add(int(ngram[-1]))
    return banned


def apply_repetition_penalty(scores: torch.Tensor, generated_ids: list[int], penalty: float) -> torch.Tensor:
    if penalty <= 1.0 or not generated_ids:
        return scores
    adjusted = scores.clone()
    for token_id in set(generated_ids):
        value = adjusted[token_id]
        adjusted[token_id] = value / penalty if value.item() > 0 else value * penalty
    return adjusted


def apply_no_repeat_ngram(scores: torch.Tensor, generated_ids: list[int], ngram_size: int) -> torch.Tensor:
    banned = banned_ngram_tokens(generated_ids, ngram_size)
    if not banned:
        return scores
    adjusted = scores.clone()
    for token_id in banned:
        adjusted[token_id] = float("-inf")
    return adjusted


def apply_top_k_top_p(scores: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    filtered = scores.clone()
    if top_k > 0 and top_k < filtered.numel():
        kth_value = torch.topk(filtered, top_k).values[-1]
        filtered[filtered < kth_value] = float("-inf")
    if 0.0 < top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[1:] = sorted_mask[:-1].clone()
        sorted_mask[0] = False
        filtered[sorted_indices[sorted_mask]] = float("-inf")
    return filtered


def prepare_sampling_scores(
    scores: torch.Tensor,
    generated_ids: list[int],
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> torch.Tensor:
    adjusted = apply_repetition_penalty(scores.float(), generated_ids, repetition_penalty)
    adjusted = apply_no_repeat_ngram(adjusted, generated_ids, no_repeat_ngram_size)
    adjusted = apply_top_k_top_p(adjusted, top_k=top_k, top_p=top_p)
    if not torch.isfinite(adjusted).any():
        adjusted = scores.float()
    return adjusted


def sample_next_id(
    adjusted_scores: torch.Tensor,
    temperature: float,
) -> int:
    if temperature <= 0:
        return int(adjusted_scores.argmax(dim=-1).item())
    probs = torch.softmax(adjusted_scores / max(temperature, 1e-5), dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def eos_rank_and_probability(
    scores: torch.Tensor,
    eos_token_id: int,
    temperature: float,
) -> tuple[int | None, float | None]:
    if eos_token_id < 0 or eos_token_id >= scores.numel():
        return None, None
    if not torch.isfinite(scores[eos_token_id]):
        return None, 0.0
    rank = int((scores > scores[eos_token_id]).sum().item()) + 1
    if temperature <= 0:
        is_argmax = int(scores.argmax(dim=-1).item()) == eos_token_id
        return rank, (1.0 if is_argmax else 0.0)
    probs = torch.softmax(scores / max(temperature, 1e-5), dim=-1)
    return rank, float(probs[eos_token_id].item())


def trim_generated_text(text: str, qa_mode: bool) -> str:
    stop_markers = ("\nUser:", "\nAssistant:", f"\n{USER_TAG}", f"\n{ASSISTANT_TAG}", f"\n{SYSTEM_TAG}")
    if qa_mode:
        stop_markers = stop_markers + ("\nQ:", "\n\nQ:")
    cut_points = [text.find(marker) for marker in stop_markers if marker in text]
    if cut_points:
        text = text[: min(cut_points)]
    return text.strip()


def first_step_logits(loaded: LoadedChatModel, prompt_ids: list[int]) -> torch.Tensor:
    """Return the unfiltered final-position logits used by chat generation."""
    if not prompt_ids:
        raise ValueError("A generation prompt must contain at least one token.")
    device = next(loaded.model.parameters()).device
    max_context = 1024 if loaded.config is None else loaded.config.seq_len
    input_ids = torch.tensor([prompt_ids[-max_context:]], dtype=torch.long, device=device)
    loaded.model.eval()
    with torch.inference_mode():
        if loaded.generation_kind == "gpt2":
            output = loaded.model(input_ids=input_ids)
            return output.logits[0, -1].float()
        output = loaded.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        return output["logits"][0, -1].float()


def generate_reply(
    loaded: LoadedChatModel,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    use_cache: bool = False,
) -> GenerationResult:
    tokenizer = loaded.tokenizer
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    generated_ids = list(prompt_ids)
    device = next(loaded.model.parameters()).device
    max_context = 1024 if loaded.config is None else loaded.config.seq_len
    stop_reason = "max_new_tokens"
    stop_token_id = tokenizer.eos_token_id
    stop_token = tokenizer.eos_token or "<EOS>"
    final_stop_rank: int | None = None
    final_stop_prob: float | None = None
    final_stop_raw_rank: int | None = None
    final_stop_raw_prob: float | None = None
    final_stop_filtered_out = False

    if not generated_ids:
        generated_ids = [tokenizer.eos_token_id]

    loaded.model.eval()
    with torch.inference_mode():
        cache = None
        cached_scores = None
        cached_attention_mask = None
        if use_cache and loaded.generation_kind == "gpt_mod":
            context_ids = generated_ids[-max_context:]
            input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
            cached_attention_mask = torch.ones_like(input_ids)
            cached_output = loaded.model(input_ids=input_ids, attention_mask=cached_attention_mask, use_cache=True)
            cache = cached_output["past_key_values"]
            cached_scores = cached_output["logits"][0, -1].float()
        for _ in range(max_new_tokens):
            if cached_scores is not None:
                next_scores = cached_scores
            else:
                context_ids = generated_ids[-max_context:]
                next_scores = first_step_logits(loaded, context_ids)
            next_scores = next_scores.clone()
            tokenizer_size = len(tokenizer) if hasattr(tokenizer, "__len__") else next_scores.numel()
            if next_scores.numel() > tokenizer_size:
                next_scores[tokenizer_size:] = float("-inf")
            if (
                tokenizer.pad_token_id is not None
                and tokenizer.pad_token_id != tokenizer.eos_token_id
                and 0 <= tokenizer.pad_token_id < next_scores.numel()
            ):
                next_scores[tokenizer.pad_token_id] = float("-inf")
            final_stop_raw_rank, final_stop_raw_prob = eos_rank_and_probability(
                next_scores,
                eos_token_id=stop_token_id,
                temperature=temperature,
            )
            adjusted_scores = prepare_sampling_scores(
                next_scores,
                generated_ids=generated_ids,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            final_stop_rank, final_stop_prob = eos_rank_and_probability(
                adjusted_scores,
                eos_token_id=stop_token_id,
                temperature=temperature,
            )
            final_stop_filtered_out = final_stop_rank is None
            next_id = sample_next_id(
                adjusted_scores,
                temperature=temperature,
            )
            generated_ids.append(next_id)
            if next_id == stop_token_id:
                stop_reason = "eos"
                break
            if cache is not None:
                if cache.get_seq_length() >= max_context:
                    stop_reason = "context_limit"
                    break
                next_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
                cached_attention_mask = torch.cat(
                    [cached_attention_mask, torch.ones((1, 1), dtype=cached_attention_mask.dtype, device=device)], dim=1
                )
                cached_output = loaded.model(
                    input_ids=next_input,
                    attention_mask=cached_attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = cached_output["past_key_values"]
                cached_scores = cached_output["logits"][0, -1].float()

    generated_text = tokenizer.decode(generated_ids[len(prompt_ids) :], skip_special_tokens=True)
    return GenerationResult(
        text=trim_generated_text(generated_text, qa_mode=uses_qa_sft_prompt(loaded) or uses_chat_jsonl_prompt(loaded)),
        stop_reason=stop_reason,
        stop_token=stop_token,
        stop_rank=final_stop_rank,
        stop_prob=final_stop_prob,
        stop_raw_rank=final_stop_raw_rank,
        stop_raw_prob=final_stop_raw_prob,
        stop_filtered_out=final_stop_filtered_out,
    )


def run_turn(loaded_models: list[LoadedChatModel], user_text: str, args: argparse.Namespace) -> None:
    for loaded in loaded_models:
        if args.mode == "continuous":
            pending_history = loaded.history + [("user", user_text)]
        else:
            system_history = [item for item in loaded.history if item[0] == "system"]
            pending_history = system_history + [("user", user_text)]

        if uses_qa_sft_prompt(loaded):
            prompt_text = render_qa_prompt(pending_history)
        elif uses_chat_jsonl_prompt(loaded):
            prompt_text = build_chat_jsonl_prompt(loaded, pending_history, mode=args.mode)
        else:
            prompt_text = render_plain_prompt(pending_history, mode=args.mode)

        generation = generate_reply(
            loaded,
            prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            use_cache=not args.no_cache,
        )
        if args.mode == "continuous":
            loaded.history = pending_history + [("assistant", generation.text)]

        model_type = "gpt2" if loaded.config is None else loaded.config.model_name
        print(f"\n[{loaded.alias}] ({model_type}, step {loaded.step})")
        print(generation.text if generation.text else "<empty response>")
        raw_rank = "n/a" if generation.stop_raw_rank is None else str(generation.stop_raw_rank)
        raw_prob = "n/a" if generation.stop_raw_prob is None else f"{generation.stop_raw_prob:.4f}"
        if generation.stop_filtered_out:
            print(f"{generation.stop_token} raw rank/prob at stop position: rank {raw_rank} | prob {raw_prob}")
            print(f"{generation.stop_token} sampled rank/prob: filtered out by top-k/top-p/no-repeat")
        else:
            sampled_rank = "n/a" if generation.stop_rank is None else str(generation.stop_rank)
            sampled_prob = "n/a" if generation.stop_prob is None else f"{generation.stop_prob:.4f}"
            print(f"{generation.stop_token} raw rank/prob at stop position: rank {raw_rank} | prob {raw_prob}")
            print(f"{generation.stop_token} sampled rank/prob: rank {sampled_rank} | prob {sampled_prob}")
        if generation.stop_reason == "eos":
            print("stopped by EOS")


def interactive_loop(loaded_models: list[LoadedChatModel], args: argparse.Namespace) -> None:
    if args.prompt is not None:
        run_turn(loaded_models, args.prompt, args)
        return

    print("Enter prompts. Commands: /reset, /exit")
    while True:
        try:
            user_text = input("\nuser> ").strip()
        except EOFError:
            print()
            return
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            return
        if user_text == "/reset":
            for loaded in loaded_models:
                loaded.history.clear()
            print("Histories cleared.")
            continue
        run_turn(loaded_models, user_text, args)


def main() -> None:
    args = parse_args()
    if args.greedy:
        args.temperature = 0.0
        args.top_k = 0
        args.top_p = 1.0
        args.repetition_penalty = 1.0
        args.no_repeat_ngram_size = 0
    device = resolve_device(args.device)
    if not args.checkpoints and not args.gpt2:
        raise ValueError("Provide at least one checkpoint spec or use --gpt2.")

    parsed_checkpoints: list[tuple[str, str]] = []
    for spec in args.checkpoints:
        try:
            parsed_checkpoints.append(parse_checkpoint_spec(spec))
        except ValueError as error:
            raise ValueError(
                f"{error}\n"
                "Each model must be one shell argument with no spaces around '='. "
                "Example: ATE=/content/checkpoints/latest.pt. "
                "If the alias or path contains spaces, quote the complete argument: "
                "\"ATE Model=/content/My Drive/checkpoints/latest.pt\"."
            ) from error

    loaded_models: list[LoadedChatModel] = []
    for alias, checkpoint_path in parsed_checkpoints:
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        loaded_models.append(load_gpt_mod_checkpoint(alias, checkpoint_path, device))
    if args.gpt2:
        loaded_models.append(load_plain_gpt2(device))

    print(
        "Generation settings: "
        f"greedy={args.greedy or args.temperature <= 0} temperature={args.temperature} "
        f"top_k={args.top_k} top_p={args.top_p} repetition_penalty={args.repetition_penalty} "
        f"no_repeat_ngram_size={args.no_repeat_ngram_size}"
    )
    print("Loaded models:")
    for loaded in loaded_models:
        if args.system_prompt:
            if loaded.config is not None and loaded.config.ignore_system_prompt:
                print(f"  {loaded.alias}: --system_prompt ignored by checkpoint configuration")
            else:
                loaded.history = [("system", args.system_prompt)]
        seq_len = 1024 if loaded.config is None else loaded.config.seq_len
        print(
            f"  {loaded.alias}: type={loaded.generation_kind} step={loaded.step} "
            f"seq_len={seq_len} checkpoint={loaded.checkpoint_path}"
        )
    interactive_loop(loaded_models, args)


if __name__ == "__main__":
    main()
