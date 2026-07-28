from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import re
import struct
import unicodedata
from typing import Any, Callable, Iterable

from data import CHAT_DATA_FORMAT_VERSION, canonical_chat_segments
from model import CONTROL_TOKENS, build_tokenizer


SUBSET_FORMAT_VERSION = "pythia_ultrachat_filtered_v1"
CATEGORY_SHARES = {
    "explanation_knowledge": 0.30,
    "practical_reasoning": 0.20,
    "advice_planning": 0.15,
    "summarization_rewriting": 0.15,
    "creative_generation": 0.10,
    "multi_turn_follow_up": 0.10,
}
CONTENT_CATEGORIES = tuple(name for name in CATEGORY_SHARES if name != "multi_turn_follow_up")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic filtered UltraChat comparison subset.")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--source-splits", nargs="+", default=["train_sft", "test_sft"])
    parser.add_argument("--output-dir", default="data/ultrachat_filtered")
    parser.add_argument("--tokenizer-name", default="EleutherAI/pythia-1.4b")
    parser.add_argument("--target-samples", type=int, default=50_000)
    parser.add_argument("--min-turns", type=int, default=2, help="Minimum role messages per conversation.")
    parser.add_argument("--max-turns", type=int, default=6, help="Maximum role messages per conversation.")
    parser.add_argument("--min-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--test-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-samples", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def normalize_prompt(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def normalize_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    raw = row.get("messages") or row.get("conversation") or row.get("conversations")
    if not isinstance(raw, list):
        return None
    aliases = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}
    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        role = aliases.get(str(item.get("role", item.get("from", ""))).strip().lower())
        content = str(item.get("content", item.get("value", ""))).strip()
        if role is None:
            # System prompts are intentionally excluded from this experiment.
            if str(item.get("role", item.get("from", ""))).strip().lower() == "system":
                continue
            return None
        if not content:
            return None
        messages.append({"role": role, "content": content})
    expected = "user"
    for message in messages:
        if message["role"] != expected:
            return None
        expected = "assistant" if expected == "user" else "user"
    if not messages or messages[-1]["role"] != "assistant":
        return None
    return messages


def has_broken_formatting(messages: list[dict[str, str]]) -> bool:
    for message in messages:
        text = message["content"]
        if any(marker in text for marker in ("<USER>", "<ASSISTANT>", "<SYSTEM>", "<EOT>", "<|endoftext|>")):
            return True
        if "\ufffd" in text or "\x00" in text or re.search(r"[\ud800-\udfff]", text):
            return True
        if re.search(r"([^\w\s])\1{15,}", text):
            return True
        if any(len(token) > 500 for token in text.split()):
            return True
        if text.count("```") % 2:
            return True
        control = sum(ord(char) < 32 and char not in "\n\r\t" for char in text)
        if control:
            return True
    return False


def is_coding_heavy(messages: list[dict[str, str]]) -> bool:
    text = "\n".join(message["content"] for message in messages)
    lowered = text.lower()
    prompt = messages[0]["content"].lower()
    explicit = re.search(
        r"\b(write|implement|debug|refactor|compile|code)\b.{0,40}"
        r"\b(python|javascript|typescript|java|c\+\+|rust|sql|function|program|script|algorithm)\b",
        prompt,
    )
    if explicit:
        return True
    fenced_chars = sum(len(block) for block in re.findall(r"```.*?```", text, flags=re.DOTALL))
    code_lines = sum(
        bool(re.search(r"(^\s*(def|class|import|from|function|const|let|var)\b|[{};]\s*$)", line))
        for line in text.splitlines()
    )
    line_count = max(len(text.splitlines()), 1)
    code_terms = len(re.findall(
        r"\b(api|class|compiler|database|debug|function|github|html|javascript|python|sql|variable)\b",
        lowered,
    ))
    return fenced_chars / max(len(text), 1) > 0.08 or code_lines / line_count > 0.25 or code_terms >= 8


def content_category(messages: list[dict[str, str]]) -> str:
    prompt = messages[0]["content"].casefold()
    patterns = {
        "summarization_rewriting": (
            r"\b(summarize|summary|rewrite|rephrase|paraphrase|proofread|edit|translate|shorten|"
            r"condense|revise)\b|\b(following|provided|given) (text|paragraph|passage|article)\b"
        ),
        "creative_generation": (
            r"\b(write|create|compose|invent|imagine|generate|story|poem|dialogue|screenplay|"
            r"song|fiction|roleplay|character)\b"
        ),
        "advice_planning": (
            r"\b(advice|recommend|suggest|tips|plan|schedule|itinerary|strategy|prepare|choose|"
            r"improve|manage|handle|avoid)\b|\b(should|could|can) i\b|\bhow (should|can|could) i\b|"
            r"\bbest way to\b|\bwhat (should|would)\b"
        ),
        "practical_reasoning": (
            r"\b(calculate|solve|compare|estimate|decide|reason|analyze|analyse|evaluate|"
            r"troubleshoot|budget|cost|percentage|probability)\b|\bstep by step\b|\bhow many\b|"
            r"\bpros and cons\b|\bwhat is the difference\b"
        ),
    }
    for category in (
        "summarization_rewriting", "creative_generation", "advice_planning", "practical_reasoning"
    ):
        if re.search(patterns[category], prompt):
            return category
    return "explanation_knowledge"


def category_affinity(messages: list[dict[str, str]], category: str) -> int:
    """Rank deterministic quota backfills by weak semantic evidence."""
    prompt = messages[0]["content"].casefold()
    signals = {
        "explanation_knowledge": (
            r"\b(explain|describe|define|why|what is|how does|history|meaning|concept)\b",
        ),
        "practical_reasoning": (
            r"\b(calculate|solve|compare|estimate|decide|reason|analy[sz]e|evaluate|"
            r"troubleshoot|budget|cost|number|amount|difference|result)\b",
            r"\b(how many|how much|step by step|pros and cons)\b",
        ),
        "advice_planning": (
            r"\b(advice|recommend|suggest|tips|plan|schedule|strategy|prepare|choose|"
            r"improve|manage|handle|avoid|goal)\b",
            r"\b(should|could|can|would) i\b|\bbest way\b",
        ),
        "summarization_rewriting": (
            r"\b(summarize|summary|rewrite|rephrase|paraphrase|proofread|edit|translate|"
            r"shorten|condense|revise|text|paragraph|passage|article)\b",
        ),
        "creative_generation": (
            r"\b(write|create|compose|invent|imagine|generate|story|poem|dialogue|"
            r"screenplay|song|fiction|roleplay|character)\b",
        ),
    }
    return sum(len(re.findall(pattern, prompt)) for pattern in signals.get(category, ()))


def category_quotas(total: int) -> dict[str, int]:
    raw = {name: total * share for name, share in CATEGORY_SHARES.items()}
    quotas = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(quotas.values())
    order = sorted(raw, key=lambda name: (raw[name] - quotas[name], name), reverse=True)
    for name in order[:remaining]:
        quotas[name] += 1
    return quotas


def stable_priority(seed: int, namespace: str, sample_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def reservoir_add(heap: list[tuple[int, str, dict[str, Any]]], capacity: int, priority: int, item: dict[str, Any]) -> None:
    entry = (-priority, item["sample_id"], item)
    if len(heap) < capacity:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def select_stratified_items(
    pools: dict[str, list[tuple[int, str, dict[str, Any]]]],
    all_pool: list[tuple[int, str, dict[str, Any]]],
    quotas: dict[str, int],
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    """Select unique records, using deterministic semantic backfills for sparse categories."""
    selected: dict[str, list[dict[str, Any]]] = {}
    selected_ids: set[str] = set()
    reassigned: Counter[str] = Counter()

    multi_items = sorted(
        (entry[2] for entry in pools["multi_turn_follow_up"]),
        key=lambda item: stable_priority(seed, "multi_turn_follow_up", item["sample_id"]),
    )[: quotas["multi_turn_follow_up"]]
    selected["multi_turn_follow_up"] = multi_items
    selected_ids.update(item["sample_id"] for item in multi_items)

    for category in CONTENT_CATEGORIES:
        candidates = sorted(
            (entry[2] for entry in pools[category]),
            key=lambda item: stable_priority(seed, category, item["sample_id"]),
        )
        chosen = [item for item in candidates if item["sample_id"] not in selected_ids][: quotas[category]]
        selected[category] = chosen
        selected_ids.update(item["sample_id"] for item in chosen)

    # The categories are heuristic and may not naturally match the requested
    # distribution. Backfill from the retained eligible universe, preferring
    # prompts with the strongest secondary evidence for each sparse category.
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for entry in all_pool:
        candidate_by_id[entry[2]["sample_id"]] = entry[2]
    for category in CONTENT_CATEGORIES:
        for entry in pools[category]:
            candidate_by_id[entry[2]["sample_id"]] = entry[2]

    for category in CONTENT_CATEGORIES:
        needed = quotas[category] - len(selected[category])
        if needed <= 0:
            continue
        available = [item for sample_id, item in candidate_by_id.items() if sample_id not in selected_ids]
        available.sort(
            key=lambda item: (
                -category_affinity(item["messages"], category),
                stable_priority(seed, f"backfill:{category}", item["sample_id"]),
            )
        )
        chosen = available[:needed]
        selected[category].extend(chosen)
        selected_ids.update(item["sample_id"] for item in chosen)
        for item in chosen:
            reassigned[f"{item['content_category']}->{category}"] += 1

    return selected, reassigned


def detect_english(text: str, detector: Callable[[str], bool]) -> bool:
    try:
        return bool(detector(text))
    except Exception:
        return False


def split_category(items: list[dict[str, Any]], validation_fraction: float, test_fraction: float, seed: int):
    ordered = sorted(items, key=lambda item: stable_priority(seed, "split", item["sample_id"]))
    test_count = round(len(ordered) * test_fraction)
    validation_count = round(len(ordered) * validation_fraction)
    return {
        "test": ordered[:test_count],
        "validation": ordered[test_count : test_count + validation_count],
        "train": ordered[test_count + validation_count :],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_token_hash(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        for token_id in record["token_ids"]:
            digest.update(struct.pack("<I", int(token_id)))
        digest.update(struct.pack("<I", 0xFFFFFFFF))
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.target_samples <= 0:
        raise ValueError("--target-samples must be positive")
    if args.min_turns < 2 or args.max_turns < args.min_turns:
        raise ValueError("Turn bounds must satisfy 2 <= min <= max")
    if args.min_tokens <= 0 or args.max_tokens < args.min_tokens:
        raise ValueError("Token bounds must satisfy 0 < min <= max")
    if args.validation_fraction < 0 or args.test_fraction < 0 or args.validation_fraction + args.test_fraction >= 1:
        raise ValueError("Validation/test fractions must be non-negative and sum to less than 1")

    try:
        from datasets import load_dataset
        from langdetect import DetectorFactory, detect_langs
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt; UltraChat preparation requires datasets and langdetect.") from exc

    DetectorFactory.seed = args.seed

    def english_detector(text: str) -> bool:
        predictions = detect_langs(text)
        return bool(predictions and predictions[0].lang == "en" and predictions[0].prob >= 0.90)

    tokenizer = build_tokenizer(args.tokenizer_name)
    quotas = category_quotas(args.target_samples)
    multi_quota = quotas["multi_turn_follow_up"]
    pools: dict[str, list[tuple[int, str, dict[str, Any]]]] = {name: [] for name in CATEGORY_SHARES}
    capacities = {
        name: quota if name == "multi_turn_follow_up" else quota + multi_quota
        for name, quota in quotas.items()
    }
    # Keep enough globally sampled eligible records to fill category shortfalls
    # after reserving the true multi-turn quota.
    all_pool: list[tuple[int, str, dict[str, Any]]] = []
    all_pool_capacity = args.target_samples + multi_quota
    seen_prompts: set[str] = set()
    rejected: Counter[str] = Counter()
    source_seen = 0

    stop = False
    for source_split in args.source_splits:
        dataset = load_dataset(
            args.dataset,
            args.dataset_config,
            split=source_split,
            revision=args.revision,
            streaming=args.streaming,
        )
        for source_index, row in enumerate(dataset):
            source_seen += 1
            if args.progress_every > 0 and source_seen % args.progress_every == 0:
                retained = {name: len(pool) for name, pool in pools.items()}
                print(
                    f"UltraChat scan | rows={source_seen:,} eligible={len(seen_prompts):,} "
                    f"retained={retained} rejected={dict(rejected)}",
                    flush=True,
                )
            if args.max_source_samples and source_seen > args.max_source_samples:
                stop = True
                break
            messages = normalize_messages(row)
            if messages is None:
                rejected["invalid_role_sequence"] += 1
                continue
            if not args.min_turns <= len(messages) <= args.max_turns:
                rejected["turn_count"] += 1
                continue
            if has_broken_formatting(messages):
                rejected["broken_formatting"] += 1
                continue
            if is_coding_heavy(messages):
                rejected["coding_heavy"] += 1
                continue
            all_text = "\n".join(message["content"] for message in messages)
            if not detect_english(all_text, english_detector):
                rejected["not_english"] += 1
                continue
            prompt_key = normalize_prompt(messages[0]["content"])
            if not prompt_key:
                rejected["empty_normalized_prompt"] += 1
                continue
            prompt_hash = hashlib.sha256(prompt_key.encode("utf-8")).hexdigest()
            if prompt_hash in seen_prompts:
                rejected["duplicate_user_prompt"] += 1
                continue
            try:
                serialized = "".join(
                    text for text, _ in canonical_chat_segments(
                        messages, open_assistant=False, eos_token=tokenizer.eos_token,
                    )
                )
            except ValueError:
                rejected["invalid_serialization"] += 1
                continue
            token_ids = tokenizer.encode(serialized, add_special_tokens=False)
            if not args.min_tokens <= len(token_ids) <= args.max_tokens:
                rejected["token_count"] += 1
                continue
            seen_prompts.add(prompt_hash)
            normalized_conversation = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            sample_id = hashlib.sha256(normalized_conversation.encode("utf-8")).hexdigest()
            category = content_category(messages)
            item = {
                "sample_id": sample_id,
                "source_split": source_split,
                "source_index": source_index,
                "content_category": category,
                "token_count": len(token_ids),
                "messages": messages,
                "token_ids": token_ids,
            }
            reservoir_add(
                pools[category], capacities[category], stable_priority(args.seed, category, sample_id), item
            )
            reservoir_add(
                all_pool, all_pool_capacity, stable_priority(args.seed, "all_eligible", sample_id), item
            )
            if len(messages) > 2:
                reservoir_add(
                    pools["multi_turn_follow_up"], multi_quota,
                    stable_priority(args.seed, "multi_turn_follow_up", sample_id), item,
                )
        if stop:
            break

    eligible_count = len(seen_prompts)
    if eligible_count < args.target_samples:
        raise RuntimeError(
            f"Requested {args.target_samples:,} samples, but only {eligible_count:,} unique conversations "
            f"survived the required filters after scanning {source_seen:,} source rows. The requested size "
            f"is mathematically impossible with these constraints. Rerun with --target-samples "
            f"{eligible_count:,} or lower (50,000 recommended), or explicitly relax the filters."
        )

    selected, category_reassignments = select_stratified_items(pools, all_pool, quotas, args.seed)

    shortfalls = {name: quotas[name] - len(selected[name]) for name in CATEGORY_SHARES if len(selected[name]) < quotas[name]}
    if shortfalls:
        availability = {name: len(pools[name]) for name in pools}
        raise RuntimeError(
            f"Unable to satisfy category quotas; shortfalls={shortfalls} retained_pool_sizes={availability}. "
            f"eligible={eligible_count:,} global_candidates={len(all_pool):,}. "
            "Lower --target-samples or inspect rejection counts."
        )

    split_records: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for category, items in selected.items():
        categorized = [dict(item, category=category) for item in items]
        parts = split_category(categorized, args.validation_fraction, args.test_fraction, args.seed)
        for split_name, records in parts.items():
            split_records[split_name].extend(records)
    for split_name in split_records:
        split_records[split_name].sort(
            key=lambda item: stable_priority(args.seed, f"order:{split_name}", item["sample_id"])
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_metadata = {}
    for split_name, records in split_records.items():
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                payload = {
                    "format_version": CHAT_DATA_FORMAT_VERSION,
                    "subset_format_version": SUBSET_FORMAT_VERSION,
                    "sample_id": record["sample_id"],
                    "category": record["category"],
                    "token_count": record["token_count"],
                    "messages": record["messages"],
                }
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        categories = Counter(record["category"] for record in records)
        split_metadata[split_name] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "examples": len(records),
            "tokens": sum(record["token_count"] for record in records),
            "min_tokens": min(record["token_count"] for record in records),
            "max_tokens": max(record["token_count"] for record in records),
            "category_counts": dict(sorted(categories.items())),
            "ordered_sample_sha256": hashlib.sha256(
                "\n".join(record["sample_id"] for record in records).encode("ascii")
            ).hexdigest(),
            "ordered_token_sha256": ordered_token_hash(records),
        }

    vocabulary = tokenizer.get_vocab()
    vocab_sha256 = hashlib.sha256(
        json.dumps(sorted(vocabulary.items()), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    actual_categories = Counter(item["category"] for records in split_records.values() for item in records)
    manifest = {
        "format_version": SUBSET_FORMAT_VERSION,
        "chat_format_version": CHAT_DATA_FORMAT_VERSION,
        "source": {"dataset": args.dataset, "config": args.dataset_config, "revision": args.revision, "splits": args.source_splits},
        "seed": args.seed,
        "filters": {
            "english_probability_min": 0.90,
            "min_turns": args.min_turns,
            "max_turns": args.max_turns,
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
            "coding_heavy_removed": True,
            "broken_formatting_removed": True,
            "deduplicate_key": "normalized_first_user_prompt",
        },
        "tokenizer": {
            "name": args.tokenizer_name,
            "vocab_size": len(tokenizer),
            "vocab_sha256": vocab_sha256,
            "eos_token_id": tokenizer.eos_token_id,
            "control_token_ids": {token: tokenizer.convert_tokens_to_ids(token) for token in CONTROL_TOKENS},
        },
        "target_samples": args.target_samples,
        "source_samples_scanned": source_seen,
        "category_shares_requested": CATEGORY_SHARES,
        "category_quotas": quotas,
        "category_counts": dict(sorted(actual_categories.items())),
        "category_reassignments": dict(sorted(category_reassignments.items())),
        "category_assignment": (
            "primary keyword classification followed by deterministic semantic-affinity backfill"
        ),
        "rejected": dict(sorted(rejected.items())),
        "splits": split_metadata,
        "comparison_contract": {
            "same_files_for_all_models": True,
            "strict_sample_order_required": True,
            "training_shuffle_buffer": 0,
            "training_num_workers": 0,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
