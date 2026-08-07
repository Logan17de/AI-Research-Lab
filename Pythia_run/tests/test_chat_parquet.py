from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data import (
    canonical_chat_segments,
    chat_parquet_examples,
    infer_steps_per_epoch,
    parquet_row_count,
    resolve_text_data_format,
    without_system_messages,
)


def write_messages(path, count: int = 5) -> None:
    rows = [
        {
            "messages": [
                {"role": "system", "content": "Use the available tool."},
                {"role": "user", "content": f"Question {index}"},
                {"role": "assistant", "content": f"<tool_call>{index}</tool_call>"},
            ]
        }
        for index in range(count)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_chat_parquet_streams_messages_and_auto_detects_format(tmp_path) -> None:
    path = tmp_path / "train.parquet"
    write_messages(path, count=3)
    examples = list(chat_parquet_examples(path))
    assert parquet_row_count(path) == 3
    assert [message[1]["content"] for message in examples] == [
        "Question 0", "Question 1", "Question 2"
    ]
    config = SimpleNamespace(text_data_format="auto", text_data_path=str(path))
    assert resolve_text_data_format(config) == "chat_parquet"


def test_chat_parquet_worker_shards_are_disjoint_and_complete(tmp_path) -> None:
    path = tmp_path / "train.parquet"
    write_messages(path, count=7)
    shards = [
        list(chat_parquet_examples(path, worker_id=worker_id, num_workers=3))
        for worker_id in range(3)
    ]
    questions = [message[1]["content"] for shard in shards for message in shard]
    assert len(questions) == len(set(questions)) == 7
    assert set(questions) == {f"Question {index}" for index in range(7)}


def test_chat_parquet_rejects_missing_messages_column(tmp_path) -> None:
    path = tmp_path / "invalid.parquet"
    pq.write_table(pa.table({"text": ["not a chat row"]}), path)
    with pytest.raises(ValueError, match="missing the required 'messages' column"):
        list(chat_parquet_examples(path))


def test_chat_parquet_can_explicitly_skip_empty_assistant_rows(tmp_path) -> None:
    path = tmp_path / "mixed.parquet"
    rows = [
        {"messages": [{"role": "user", "content": "valid"}, {"role": "assistant", "content": "answer"}]},
        {"messages": [{"role": "user", "content": "invalid"}, {"role": "assistant", "content": ""}]},
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="empty content"):
        list(chat_parquet_examples(path))
    examples = list(chat_parquet_examples(path, skip_invalid=True))
    assert len(examples) == 1
    assert examples[0][0]["content"] == "valid"


def test_chat_parquet_step_count_uses_metadata_and_worker_batching(tmp_path) -> None:
    path = tmp_path / "train.parquet"
    write_messages(path, count=7)
    config = SimpleNamespace(
        text_data_format="chat_parquet",
        text_data_path=str(path),
        run_dir=str(tmp_path / "run"),
        seq_len=512,
        batch_size=2,
        grad_accum_steps=2,
        num_workers=3,
    )
    tokenizer = SimpleNamespace(name_or_path="test-tokenizer")
    # Worker shards contain 3, 2, and 2 rows: 2 + 1 + 1 microbatches,
    # followed by gradient accumulation of two microbatches per update.
    assert infer_steps_per_epoch(config, tokenizer) == 2


def test_system_messages_can_be_removed_from_parquet_conversations(tmp_path) -> None:
    path = tmp_path / "train.parquet"
    write_messages(path, count=1)
    messages = list(chat_parquet_examples(path, ignore_system_prompt=True))[0]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert without_system_messages(messages) == messages


def test_ignored_system_text_is_absent_from_canonical_training_sequence() -> None:
    messages = [
        {"role": "system", "content": "secret system instruction"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    filtered = without_system_messages(messages)
    segments = canonical_chat_segments(filtered, open_assistant=False)
    serialized = "".join(text for text, _ in segments)
    assert serialized == "<USER> question\n<ASSISTANT> answer<|endoftext|>\n"
    assert "secret system instruction" not in serialized
    assert all("<SYSTEM>" not in text for text, _ in segments)
