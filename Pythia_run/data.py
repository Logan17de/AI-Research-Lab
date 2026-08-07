from __future__ import annotations

import itertools
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from config import TrainConfig


USER_TAG = "<USER>"
ASSISTANT_TAG = "<ASSISTANT>"
SYSTEM_TAG = "<SYSTEM>"
LEGACY_EOT_TAG = "<EOT>"
GPT2_EOS_TOKEN = "<|endoftext|>"
CHAT_DATA_FORMAT_VERSION = "gpt_mod_raw_chat_eos_turn_v3"
CHAT_DATA_FORMATS = frozenset({"chat_jsonl", "chat_parquet"})


class LegacyChatFormatError(ValueError):
    pass


def without_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of a conversation containing only user/assistant turns."""
    return [
        dict(message)
        for message in messages
        if str(message.get("role", "")).strip().lower() != "system"
    ]


def canonical_chat_segments(
    messages: list[dict[str, Any]],
    *,
    open_assistant: bool,
    eos_token: str = GPT2_EOS_TOKEN,
) -> list[tuple[str, bool]]:
    """Return the one canonical chat serialization as (text, supervised) segments."""
    segments: list[tuple[str, bool]] = []
    awaiting_assistant = False
    saw_turn = False
    saw_assistant = False
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            raise ValueError(f"Chat messages must have non-empty content; role={role!r}.")
        if _contains_legacy_chat_serialization(content, eos_token):
            raise LegacyChatFormatError(
                "Raw chat content contains a registered control/EOS token. "
                "Provide role/content text without serialized boundaries."
            )
        if role == "system":
            if saw_turn:
                raise ValueError("System messages are only valid before the first user turn.")
            segments.append((f"{SYSTEM_TAG} {content}\n", False))
        elif role == "user":
            if awaiting_assistant:
                raise ValueError("A user turn cannot follow an unanswered user turn.")
            segments.append((f"{USER_TAG} {content}\n", False))
            awaiting_assistant = True
            saw_turn = True
        elif role == "assistant":
            if not awaiting_assistant:
                raise ValueError("An assistant turn must follow a user turn.")
            segments.extend(
                [
                    (ASSISTANT_TAG, False),
                    (f" {content}", True),
                    (eos_token, True),
                    ("\n", False),
                ]
            )
            awaiting_assistant = False
            saw_assistant = True
        else:
            raise ValueError(f"Unsupported chat role: {role!r}")
    if open_assistant:
        if not awaiting_assistant:
            raise ValueError("An open assistant prefix requires a final user turn.")
        segments.append((ASSISTANT_TAG, False))
    elif awaiting_assistant or not saw_assistant:
        raise ValueError("A training conversation must end with a non-empty assistant turn.")
    return segments


def render_chat_prompt(
    messages: list[dict[str, Any]],
    open_assistant: bool = True,
    eos_token: str = GPT2_EOS_TOKEN,
) -> str:
    """Render the canonical live-chat prefix used by EOS-per-turn SFT."""
    return "".join(
        text for text, _ in canonical_chat_segments(
            messages, open_assistant=open_assistant, eos_token=eos_token,
        )
    ).rstrip("\n")


def truncate_chat_messages_for_prompt(
    tokenizer,
    messages: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[list[dict[str, Any]], int]:
    """Drop complete oldest exchanges until an open-assistant prompt fits."""
    working = [dict(message) for message in messages]
    dropped_turns = 0

    def token_count() -> int:
        return len(tokenizer.encode(
            render_chat_prompt(
                working, open_assistant=True, eos_token=tokenizer.eos_token,
            ),
            add_special_tokens=False,
        ))

    while token_count() > max_tokens:
        user_positions = [
            index for index, message in enumerate(working)
            if str(message.get("role", "")).strip().lower() == "user"
        ]
        if len(user_positions) <= 1:
            break
        del working[user_positions[0] : user_positions[1]]
        dropped_turns += 1
    if token_count() > max_tokens and working and str(working[0].get("role", "")).strip().lower() == "system":
        del working[0]
        dropped_turns += 1
    if token_count() > max_tokens:
        raise ValueError(
            f"The final user turn needs {token_count()} tokens but the checkpoint context limit is {max_tokens}."
        )
    return working, dropped_turns


def _contains_legacy_chat_serialization(content: str, eos_token: str | None) -> bool:
    markers = [USER_TAG, ASSISTANT_TAG, SYSTEM_TAG, LEGACY_EOT_TAG]
    if eos_token:
        markers.append(eos_token)
    return any(marker in content for marker in markers)


def resolve_text_data_format(config: TrainConfig) -> str:
    if config.text_data_format != "auto":
        return config.text_data_format

    data_path = Path(config.text_data_path)
    if data_path.suffix.lower() == ".parquet":
        return "chat_parquet"
    try:
        with open(data_path, "r", encoding="utf-8", errors="ignore") as handle:
            sample_lines = [handle.readline().strip() for _ in range(8)]
    except OSError:
        return "plain"

    has_chat_json = any(line.startswith("{") and '"messages"' in line for line in sample_lines if line)
    if has_chat_json:
        return "chat_jsonl"

    has_question = any(line.startswith("Q:") for line in sample_lines if line)
    has_answer = any(line.startswith("A:") for line in sample_lines if line)
    return "qa_sft" if has_question and has_answer else "plain"


def qa_sft_blocks(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
    question: str | None = None
    answer_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if question is not None and answer_lines:
                yield question, " ".join(answer_lines).strip()
            question = None
            answer_lines = []
            continue
        if line.startswith("Q:"):
            if question is not None and answer_lines:
                yield question, " ".join(answer_lines).strip()
            question = line[2:].strip()
            answer_lines = []
            continue
        if line.startswith("A:"):
            answer_lines.append(line[2:].strip())
            continue
        if answer_lines:
            answer_lines.append(line)
        elif question is not None:
            question = f"{question} {line}".strip()

    if question is not None and answer_lines:
        yield question, " ".join(answer_lines).strip()


def plain_text_blocks(lines: Iterable[str]) -> Iterator[str]:
    for raw_line in lines:
        text = raw_line.strip()
        if text:
            yield text


def chat_jsonl_examples(lines: Iterable[str]) -> Iterator[list[dict[str, Any]]]:
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed chat JSONL at line {line_number}: {exc}") from exc
        if payload.get("format_version") != CHAT_DATA_FORMAT_VERSION:
            raise LegacyChatFormatError(
                f"chat_jsonl line {line_number} has format_version={payload.get('format_version')!r}; "
                f"expected {CHAT_DATA_FORMAT_VERSION!r}. Re-export with prepare_tulu.py."
            )
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"chat_jsonl line {line_number} is missing a messages list.")
        yield messages


def chat_parquet_examples(
    path: str | Path,
    *,
    worker_id: int = 0,
    num_workers: int = 1,
    skip_invalid: bool = False,
    ignore_system_prompt: bool = False,
) -> Iterator[list[dict[str, Any]]]:
    """Stream Hugging Face-style Parquet rows containing a messages column."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "chat_parquet requires pyarrow. Install project requirements or run `pip install pyarrow`."
        ) from exc

    parquet_path = Path(path)
    parquet_file = pq.ParquetFile(parquet_path)
    if "messages" not in parquet_file.schema_arrow.names:
        raise ValueError(
            f"Parquet dataset {parquet_path} is missing the required 'messages' column; "
            f"columns={parquet_file.schema_arrow.names}."
        )
    row_index = 0
    for batch in parquet_file.iter_batches(batch_size=1024, columns=["messages"]):
        for payload in batch.to_pylist():
            current_index = row_index
            row_index += 1
            if num_workers > 1 and current_index % num_workers != worker_id:
                continue
            try:
                messages = payload.get("messages") if isinstance(payload, dict) else None
                if not isinstance(messages, list):
                    raise ValueError("missing a messages list")
                normalized = [
                    {
                        "role": str(message.get("role") or "").strip().lower(),
                        "content": str(message.get("content") or "").strip(),
                    }
                    for message in messages
                    if isinstance(message, dict)
                ]
                if len(normalized) != len(messages):
                    raise ValueError("contains a non-object message")
                if ignore_system_prompt:
                    normalized = without_system_messages(normalized)
                # Validate role ordering and content before the row reaches a worker batch.
                canonical_chat_segments(normalized, open_assistant=False)
            except ValueError as exc:
                if skip_invalid:
                    continue
                raise ValueError(
                    f"Invalid Parquet row {current_index} in {parquet_path}: {exc}"
                ) from exc
            yield normalized


def parquet_row_count(path: str | Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("chat_parquet requires pyarrow.") from exc
    return int(pq.ParquetFile(path).metadata.num_rows)


def iter_training_examples(lines: Iterable[str], text_data_format: str) -> Iterator[str | tuple[str, str] | list[dict[str, Any]]]:
    if text_data_format == "plain":
        yield from plain_text_blocks(lines)
        return
    if text_data_format == "qa_sft":
        yield from qa_sft_blocks(lines)
        return
    if text_data_format == "chat_jsonl":
        yield from chat_jsonl_examples(lines)
        return
    raise ValueError(f"Unsupported text_data_format: {text_data_format}")


def block_mix_examples(
    examples: Iterable[str | tuple[str, str] | list[dict[str, Any]]],
    mix: int,
    seed: int,
) -> Iterator[str | tuple[str, str] | list[dict[str, Any]]]:
    if mix <= 0:
        yield from examples
        return

    example_list = list(examples)
    if len(example_list) <= mix:
        yield from example_list
        return

    blocks = [example_list[idx : idx + mix] for idx in range(0, len(example_list), mix)]
    rng = random.Random(seed)
    rng.shuffle(blocks)
    for block in blocks:
        yield from block


def buffered_shuffle_examples(
    examples: Iterable[str | tuple[str, str] | list[dict[str, Any]]],
    buffer_size: int,
    seed: int,
) -> Iterator[str | tuple[str, str] | list[dict[str, Any]]]:
    if buffer_size <= 1:
        yield from examples
        return
    rng = random.Random(seed)
    iterator = iter(examples)
    buffer = list(itertools.islice(iterator, buffer_size))
    while buffer:
        index = rng.randrange(len(buffer))
        yield buffer[index]
        try:
            buffer[index] = next(iterator)
        except StopIteration:
            buffer.pop(index)


def shard_examples(
    examples: Iterable[str | tuple[str, str] | list[dict[str, Any]]],
    worker_id: int,
    num_workers: int,
) -> Iterator[str | tuple[str, str] | list[dict[str, Any]]]:
    if num_workers <= 1:
        yield from examples
        return
    yield from itertools.islice(examples, worker_id, None, num_workers)


def encode_training_example(
    tokenizer,
    text_data_format: str,
    example: str | tuple[str, str] | list[dict[str, Any]],
) -> tuple[list[int], list[int]] | None:
    eos_id = tokenizer.eos_token_id
    if text_data_format == "plain":
        token_ids = tokenizer.encode(example, add_special_tokens=False)
        if not token_ids:
            return None
        sequence = token_ids + [eos_id]
        return sequence, sequence

    if text_data_format == "qa_sft":
        question, answer = example
        prompt_text = f"Q: {question}\nA:"
        answer_text = f" {answer}"
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
        if not prompt_ids or not answer_ids:
            return None
        sequence = prompt_ids + answer_ids + [eos_id]
        labels = ([-100] * len(prompt_ids)) + answer_ids + [eos_id]
        return sequence, labels

    if text_data_format == "chat_jsonl":
        token_ids: list[int] = []
        labels: list[int] = []
        segments = canonical_chat_segments(
            example,
            open_assistant=False,
            eos_token=getattr(tokenizer, "eos_token", None),
        )
        for text, supervised in segments:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                raise ValueError(f"Canonical chat segment encoded to no tokens: {text!r}")
            token_ids.extend(ids)
            labels.extend(ids if supervised else [-100] * len(ids))
        return token_ids, labels

    raise ValueError(f"Unsupported text_data_format: {text_data_format}")


def encode_chat_with_turn_safe_truncation(
    tokenizer,
    messages: list[dict[str, Any]],
    seq_len: int,
) -> tuple[list[int], list[int], int, bool] | None:
    """Drop complete oldest exchanges before applying a final right truncation."""
    encoded = encode_training_example(tokenizer, "chat_jsonl", messages)
    if encoded is None:
        return None
    original_ids, original_labels = encoded
    original_targets = sum(label != -100 for label in original_labels)
    if len(original_ids) <= seq_len:
        return original_ids, original_labels, 0, False

    working = [dict(message) for message in messages]
    truncated = True

    def user_count() -> int:
        return sum(str(message.get("role", "")).strip().lower() == "user" for message in working)

    # Preserve the latest exchange and remove older user+assistant exchanges atomically.
    while user_count() > 1:
        first_user = next(
            (idx for idx, message in enumerate(working) if str(message.get("role", "")).strip().lower() == "user"),
            None,
        )
        if first_user is None:
            break
        next_user = next(
            (
                idx
                for idx in range(first_user + 1, len(working))
                if str(working[idx].get("role", "")).strip().lower() == "user"
            ),
            None,
        )
        if next_user is None:
            break
        del working[first_user:next_user]
        current = encode_training_example(tokenizer, "chat_jsonl", working)
        if current is not None and len(current[0]) <= seq_len:
            ids, labels = current
            kept_targets = sum(label != -100 for label in labels)
            return ids, labels, original_targets - kept_targets, truncated

    current = encode_training_example(tokenizer, "chat_jsonl", working)
    if current is None:
        return None
    ids, labels = current

    # If the system prompt prevents the final exchange from fitting, remove it as a whole.
    if len(ids) > seq_len:
        working = [
            message
            for message in working
            if str(message.get("role", "")).strip().lower() != "system"
        ]
        current = encode_training_example(tokenizer, "chat_jsonl", working)
        if current is None:
            return None
        ids, labels = current

    if len(ids) > seq_len:
        separator_ids = tokenizer.encode("\n", add_special_tokens=False)
        tail_ids = [tokenizer.eos_token_id] + separator_ids
        tail_labels = [tokenizer.eos_token_id] + ([-100] * len(separator_ids))
        prefix_len = seq_len - len(tail_ids)
        if prefix_len <= 0:
            return None
        prefix_ids = list(ids[:prefix_len])
        prefix_labels = list(labels[:prefix_len])
        assistant_id = tokenizer.convert_tokens_to_ids(ASSISTANT_TAG)
        if assistant_id not in prefix_ids or not any(label != -100 for label in prefix_labels):
            return None
        # A cut answer remains a valid final assistant turn with EOS restored.
        ids = prefix_ids + tail_ids
        labels = prefix_labels + tail_labels

    kept_targets = sum(label != -100 for label in labels)
    return ids, labels, max(original_targets - kept_targets, 0), truncated


def token_label_stream(
    tokenizer,
    lines: Iterable[str],
    text_data_format: str,
    mix: int = 0,
    mix_seed: int = 0,
    worker_id: int = 0,
    num_workers: int = 1,
) -> Iterator[tuple[list[int], list[int]]]:
    examples = iter_training_examples(lines, text_data_format)
    examples = shard_examples(examples, worker_id=worker_id, num_workers=num_workers)
    mixed_examples = block_mix_examples(examples, mix=mix, seed=mix_seed)
    for example in mixed_examples:
        encoded = encode_training_example(tokenizer, text_data_format, example)
        if encoded is not None:
            yield encoded


class LocalTextDataset(IterableDataset):
    def __init__(self, config: TrainConfig, tokenizer, path: str, training: bool, epoch: int = 1) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.path = path
        self.training = training
        self.epoch = epoch

    def _line_iterator(self) -> Iterable[str]:
        with open(self.path, "r", encoding="utf-8", errors="ignore") as handle:
            yield from handle

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        needed = self.config.seq_len
        token_buffer: list[int] = []
        label_buffer: list[int] = []
        text_data_format = resolve_text_data_format(self.config)
        mix = self.config.mix if self.training else 0
        mix_seed = self.config.seed + max(self.epoch - 1, 0)
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        if text_data_format != "plain":
            if text_data_format == "chat_parquet":
                examples = chat_parquet_examples(
                    self.path,
                    worker_id=worker_id,
                    num_workers=num_workers,
                    ignore_system_prompt=getattr(self.config, "ignore_system_prompt", False),
                )
            else:
                examples = iter_training_examples(self._line_iterator(), text_data_format)
                examples = shard_examples(examples, worker_id=worker_id, num_workers=num_workers)
            if self.training:
                examples = buffered_shuffle_examples(
                    examples,
                    buffer_size=self.config.shuffle_buffer,
                    seed=mix_seed + worker_id * 1_000_003,
                )
            examples = block_mix_examples(examples, mix=mix, seed=mix_seed)
            for example in examples:
                if text_data_format in CHAT_DATA_FORMATS:
                    if getattr(self.config, "ignore_system_prompt", False):
                        example = without_system_messages(example)
                    result = encode_chat_with_turn_safe_truncation(self.tokenizer, example, needed)
                    if result is None:
                        continue
                    token_ids, label_ids, targets_lost, was_truncated = result
                else:
                    encoded = encode_training_example(self.tokenizer, text_data_format, example)
                    if encoded is None:
                        continue
                    token_ids, label_ids = self._truncate_sft(*encoded, needed)
                    targets_lost = 0
                    was_truncated = len(encoded[0]) > needed
                if not any(label != -100 for label in label_ids[1:]):
                    continue
                attention_mask = [1] * len(token_ids)
                pad_len = needed - len(token_ids)
                if pad_len > 0:
                    token_ids += [self.tokenizer.pad_token_id] * pad_len
                    label_ids += [-100] * pad_len
                    attention_mask += [0] * pad_len
                if len(token_ids) != needed or len(label_ids) != needed or len(attention_mask) != needed:
                    raise RuntimeError("SFT collation produced inconsistent sequence lengths.")
                if any(label != -100 for label, mask in zip(label_ids, attention_mask) if mask == 0):
                    raise RuntimeError("Padding positions must use label -100.")
                yield {
                    "input_ids": torch.tensor(token_ids, dtype=torch.long),
                    "labels": torch.tensor(label_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "target_tokens_lost": torch.tensor(targets_lost, dtype=torch.long),
                    "was_truncated": torch.tensor(int(was_truncated), dtype=torch.long),
                }
            return

        for token_ids, label_ids in token_label_stream(
            self.tokenizer,
            self._line_iterator(),
            text_data_format,
            mix=mix,
            mix_seed=mix_seed,
            worker_id=worker_id,
            num_workers=num_workers,
        ):
            token_buffer.extend(token_ids)
            label_buffer.extend(label_ids)

            while len(token_buffer) >= needed:
                chunk = token_buffer[:needed]
                chunk_labels = label_buffer[:needed]
                token_buffer = token_buffer[needed:]
                label_buffer = label_buffer[needed:]
                labels = torch.tensor(chunk_labels, dtype=torch.long)
                if bool((labels != -100).sum().item() == 0):
                    continue
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "labels": labels,
                    "attention_mask": torch.ones(needed, dtype=torch.long),
                }

    def _truncate_sft(
        self,
        token_ids: list[int],
        label_ids: list[int],
        seq_len: int,
    ) -> tuple[list[int], list[int]]:
        if len(token_ids) <= seq_len:
            return list(token_ids), list(label_ids)
        token_ids = list(token_ids[:seq_len])
        label_ids = list(label_ids[:seq_len])
        # Preserve a supervised stop target when truncation cuts an answer.
        last_target = max((idx for idx, value in enumerate(label_ids) if value != -100), default=-1)
        if last_target >= 0:
            token_ids[-1] = self.tokenizer.eos_token_id
            label_ids[-1] = self.tokenizer.eos_token_id
        return token_ids, label_ids


def steps_per_epoch_cache_key(config: TrainConfig, tokenizer) -> dict[str, object]:
    data_path = Path(config.text_data_path)
    file_stat = data_path.stat()
    return {
        "path": str(data_path.resolve()),
        "size": file_stat.st_size,
        "mtime_ns": file_stat.st_mtime_ns,
        "tokenizer": getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
        "seq_len": config.seq_len,
        "batch_size": config.batch_size,
        "grad_accum_steps": config.grad_accum_steps,
        "num_workers": config.num_workers,
        "text_data_format": resolve_text_data_format(config),
        "ignore_system_prompt": getattr(config, "ignore_system_prompt", False),
    }


def save_steps_per_epoch_cache(config: TrainConfig, tokenizer, steps_per_epoch: int) -> Path:
    cache_path = Path(config.run_dir) / "steps_per_epoch_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {"cache_key": steps_per_epoch_cache_key(config, tokenizer), "steps_per_epoch": steps_per_epoch},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return cache_path


def isolated_sample_steps_per_epoch(
    sample_count: int,
    batch_size: int,
    grad_accum_steps: int,
    num_workers: int,
) -> int:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    worker_count = max(num_workers, 1)
    worker_sample_counts = [
        ((sample_count - 1 - worker_id) // worker_count) + 1
        for worker_id in range(min(worker_count, sample_count))
    ]
    microbatches = sum(math.ceil(count / max(batch_size, 1)) for count in worker_sample_counts)
    return max(1, math.ceil(microbatches / max(grad_accum_steps, 1)))


def infer_steps_per_epoch(config: TrainConfig, tokenizer) -> int:
    data_path = Path(config.text_data_path)
    cache_path = Path(config.run_dir) / "steps_per_epoch_cache.json"
    file_stat = data_path.stat()
    cache_key = steps_per_epoch_cache_key(config, tokenizer)

    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("cache_key") == cache_key:
                cached_value = int(payload["steps_per_epoch"])
                if cached_value > 0:
                    print(f"Loaded cached steps_per_epoch={cached_value} from {cache_path}")
                    return cached_value
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            print(f"Ignoring invalid steps_per_epoch cache {cache_path}: {exc}")

    needed = config.seq_len
    chunk_count = 0
    worker_sample_counts = [0] * max(config.num_workers, 1)
    buffered_tokens = 0
    processed_bytes = 0
    last_log_time = time.time()
    text_data_format = resolve_text_data_format(config)

    print(f"Inferring steps_per_epoch from {data_path} ...")
    if text_data_format == "chat_parquet":
        total_rows = parquet_row_count(data_path)
        if total_rows <= 0:
            raise ValueError(f"Parquet dataset {data_path} contains no rows.")
        # One chat row produces one isolated padded SFT sample. Parquet metadata
        # provides cardinality without tokenizing the whole dataset before step 1.
        chunk_count = total_rows
        for worker_id in range(len(worker_sample_counts)):
            if worker_id < total_rows:
                worker_sample_counts[worker_id] = (
                    (total_rows - 1 - worker_id) // len(worker_sample_counts)
                ) + 1
        print(f"Read {total_rows} chat examples from Parquet metadata.")
    else:
        with open(data_path, "r", encoding="utf-8", errors="ignore") as handle:
            def counted_lines() -> Iterator[str]:
                nonlocal processed_bytes
                for line in handle:
                    processed_bytes += len(line.encode("utf-8", errors="ignore"))
                    yield line

            if text_data_format != "plain":
                for example_index, example in enumerate(iter_training_examples(counted_lines(), text_data_format)):
                    if text_data_format == "chat_jsonl":
                        if getattr(config, "ignore_system_prompt", False):
                            example = without_system_messages(example)
                        encoded = encode_chat_with_turn_safe_truncation(tokenizer, example, needed)
                    else:
                        encoded = encode_training_example(tokenizer, text_data_format, example)
                        if encoded is not None:
                            encoded = LocalTextDataset(config, tokenizer, str(data_path), True)._truncate_sft(*encoded, needed)
                    if encoded is None:
                        continue
                    labels = encoded[1]
                    if any(label != -100 for label in labels[1:]):
                        chunk_count += 1
                        worker_sample_counts[example_index % len(worker_sample_counts)] += 1
                    if time.time() - last_log_time >= 10.0:
                        progress = 100.0 * processed_bytes / max(file_stat.st_size, 1)
                        print(f"Inferring steps_per_epoch ... {progress:.1f}%")
                        last_log_time = time.time()
            else:
                for token_ids, _ in token_label_stream(tokenizer, counted_lines(), text_data_format):
                    buffered_tokens += len(token_ids)
                    new_chunks, buffered_tokens = divmod(buffered_tokens, needed)
                    chunk_count += new_chunks
                    if time.time() - last_log_time >= 10.0:
                        progress = 100.0 * processed_bytes / max(file_stat.st_size, 1)
                        print(f"Inferring steps_per_epoch ... {progress:.1f}%")
                        last_log_time = time.time()

    if chunk_count <= 0:
        raise ValueError("No trainable chunks produced from the text file.")

    if text_data_format == "plain":
        if config.num_workers > 0:
            raise ValueError(
                "Packed plain-text pretraining requires --num_workers 0; worker sharding changes token packing boundaries."
            )
        microbatches = math.ceil(chunk_count / max(config.batch_size, 1))
    else:
        # IterableDataset workers batch independently, including one partial batch per worker.
        microbatches = sum(math.ceil(count / max(config.batch_size, 1)) for count in worker_sample_counts if count)
    steps_per_epoch = max(1, math.ceil(microbatches / max(config.grad_accum_steps, 1)))
    save_steps_per_epoch_cache(config, tokenizer, steps_per_epoch)
    print(f"Inferred steps_per_epoch={steps_per_epoch} and cached it at {cache_path}")
    return steps_per_epoch


def build_train_loader(config: TrainConfig, tokenizer, epoch: int = 1) -> DataLoader:
    if resolve_text_data_format(config) == "plain" and config.num_workers > 0:
        raise ValueError("Packed plain-text pretraining requires --num_workers 0 for deterministic token packing.")
    dataset = LocalTextDataset(config, tokenizer, config.text_data_path, training=True, epoch=epoch)
    generator = torch.Generator().manual_seed(config.seed + epoch * 10_000_019)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        generator=generator,
    )


def build_eval_loader(config: TrainConfig, tokenizer) -> DataLoader:
    path = config.eval_text_data_path or config.text_data_path
    dataset = LocalTextDataset(config, tokenizer, path, training=False)
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, generator=generator)
