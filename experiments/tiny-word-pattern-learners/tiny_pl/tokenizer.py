from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


class LiveWordTokenizer:
    """A Unicode-safe tokenizer that splits text only on whitespace.

    Punctuation remains attached to the surrounding word by design. The
    vocabulary is built from supplied corpus files and then saved so token IDs
    remain stable across learner experiments.
    """

    SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")

    def __init__(self, token_to_id: dict[str, int]) -> None:
        if tuple(token_to_id)[: len(self.SPECIAL_TOKENS)] != self.SPECIAL_TOKENS:
            raise ValueError("Special tokens must occupy the first tokenizer IDs")
        ids = sorted(token_to_id.values())
        if ids != list(range(len(ids))):
            raise ValueError("Tokenizer IDs must be contiguous from zero")
        self.token_to_id = dict(token_to_id)
        self.id_to_token = [""] * len(token_to_id)
        for token, index in token_to_id.items():
            self.id_to_token[index] = token

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        *,
        min_frequency: int = 1,
        max_vocab_size: int | None = None,
    ) -> "LiveWordTokenizer":
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(cls.split(text))

        candidates = [
            (token, frequency)
            for token, frequency in counts.items()
            if frequency >= min_frequency and token not in cls.SPECIAL_TOKENS
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))

        if max_vocab_size is not None:
            if max_vocab_size < len(cls.SPECIAL_TOKENS):
                raise ValueError("max_vocab_size is smaller than the special-token set")
            candidates = candidates[: max_vocab_size - len(cls.SPECIAL_TOKENS)]

        ordered = list(cls.SPECIAL_TOKENS) + [token for token, _ in candidates]
        return cls({token: index for index, token in enumerate(ordered)})

    @staticmethod
    def split(text: str) -> list[str]:
        return text.strip().split()

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = [self.token_to_id.get(token, self.unk_id) for token in self.split(text)]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        specials = set(self.SPECIAL_TOKENS)
        for index in ids:
            if index < 0 or index >= self.vocab_size:
                raise IndexError(f"Token ID {index} is outside vocabulary")
            token = self.id_to_token[index]
            if skip_special_tokens and token in specials:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"token_to_id": self.token_to_id}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "LiveWordTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({str(token): int(index) for token, index in payload["token_to_id"].items()})
