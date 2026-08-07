from __future__ import annotations

import hashlib
import json

import pytest

from data import CHAT_DATA_FORMAT_VERSION
from dataset_lock import validate_locked_dataset
from prepare_ultrachat_subset import (
    CATEGORY_SHARES,
    category_quotas,
    content_category,
    has_broken_formatting,
    is_coding_heavy,
    normalize_messages,
    normalize_prompt,
    reservoir_add,
    select_stratified_items,
)


class FakeTokenizer:
    eos_token_id = 3

    def __init__(self) -> None:
        self.vocab = {"a": 0, "b": 1, "<USER>": 2, "<ASSISTANT>": 4, "<SYSTEM>": 5}

    def __len__(self) -> int:
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]


def test_normalization_removes_system_and_requires_complete_alternating_turns() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Follow up"},
            {"role": "assistant", "content": "Answer"},
        ]
    }
    messages = normalize_messages(row)
    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
    assert normalize_messages({"messages": row["messages"][:-1]}) is None
    assert normalize_prompt("  What's   NEW?! ") == "what s new"


def test_quality_and_coding_filters_are_conservative_but_explicit() -> None:
    normal = [{"role": "user", "content": "Explain photosynthesis."}, {"role": "assistant", "content": "Plants use light."}]
    coding = [{"role": "user", "content": "Write a Python function and debug this script."}, {"role": "assistant", "content": "Use def f(): pass"}]
    broken = [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "broken \ufffd text"}]
    assert not is_coding_heavy(normal)
    assert is_coding_heavy(coding)
    assert has_broken_formatting(broken)
    assert not has_broken_formatting(normal)


def test_categories_and_quotas_match_requested_distribution() -> None:
    assert content_category([{"role": "user", "content": "Rewrite this paragraph."}]) == "summarization_rewriting"
    assert content_category([{"role": "user", "content": "Plan a weekend itinerary."}]) == "advice_planning"
    quotas = category_quotas(10_003)
    assert sum(quotas.values()) == 10_003
    for name, share in CATEGORY_SHARES.items():
        assert abs(quotas[name] / 10_003 - share) < 0.001


def test_sparse_category_pools_are_deterministically_backfilled_without_duplicates() -> None:
    quotas = {name: 1 for name in CATEGORY_SHARES}
    pools = {name: [] for name in CATEGORY_SHARES}
    all_pool = []
    for index in range(6):
        messages = [
            {"role": "user", "content": f"General question number {index}"},
            {"role": "assistant", "content": "General answer"},
        ]
        if index == 0:
            messages.extend([
                {"role": "user", "content": "A follow-up"},
                {"role": "assistant", "content": "A follow-up answer"},
            ])
        item = {
            "sample_id": str(index),
            "content_category": "explanation_knowledge",
            "messages": messages,
        }
        reservoir_add(pools["explanation_knowledge"], 6, index, item)
        reservoir_add(all_pool, 6, index, item)
        if index == 0:
            reservoir_add(pools["multi_turn_follow_up"], 1, index, item)

    selected, reassigned = select_stratified_items(pools, all_pool, quotas, seed=42)
    selected_ids = [item["sample_id"] for items in selected.values() for item in items]
    assert all(len(selected[name]) == 1 for name in CATEGORY_SHARES)
    assert len(selected_ids) == len(set(selected_ids)) == 6
    assert sum(reassigned.values()) == 4


def test_dataset_lock_validates_file_and_tokenizer_hashes(tmp_path) -> None:
    tokenizer = FakeTokenizer()
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"messages":[]}\n', encoding="utf-8")
    vocab_sha = hashlib.sha256(
        json.dumps(sorted(tokenizer.get_vocab().items()), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    file_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    manifest = {
        "format_version": "pythia_ultrachat_filtered_v1",
        "chat_format_version": CHAT_DATA_FORMAT_VERSION,
        "tokenizer": {
            "vocab_size": len(tokenizer),
            "vocab_sha256": vocab_sha,
            "eos_token_id": tokenizer.eos_token_id,
            "control_token_ids": {
                token: tokenizer.convert_tokens_to_ids(token)
                for token in ("<USER>", "<ASSISTANT>", "<SYSTEM>")
            },
        },
        "splits": {"train": {"file": data_path.name, "sha256": file_sha}},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validate_locked_dataset(str(manifest_path), [str(data_path)], tokenizer)
    data_path.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_locked_dataset(str(manifest_path), [str(data_path)], tokenizer)
