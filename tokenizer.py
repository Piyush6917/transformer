"""A small word-level tokenizer for the educational Transformer project."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>")
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class Tokenizer:
    """Convert text to token IDs and token IDs back to text.

    This intentionally uses a simple word-and-punctuation vocabulary instead
    of a pretrained tokenizer so that every step is visible and explainable.
    """

    def __init__(self, tokens: Iterable[str] = ()) -> None:
        self.token_to_id: dict[str, int] = {
            token: index for index, token in enumerate(SPECIAL_TOKENS)
        }
        for token in tokens:
            self._add_token(token)

    @property
    def id_to_token(self) -> dict[int, str]:
        """Return the reverse vocabulary mapping."""
        return {index: token for token, index in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<UNK>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.token_to_id)
        return self.token_to_id[token]

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Split words and punctuation while preserving punctuation tokens."""
        return TOKEN_PATTERN.findall(text.lower())

    @classmethod
    def from_text(cls, text: str) -> "Tokenizer":
        """Build a vocabulary from text in first-seen order."""
        return cls(cls.tokenize(text))

    @classmethod
    def from_file(cls, path: str | Path) -> "Tokenizer":
        """Build a vocabulary from a UTF-8 text file."""
        return cls.from_text(Path(path).read_text(encoding="utf-8"))

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode text as integer IDs, mapping unknown tokens to ``<UNK>``."""
        token_ids = [self.token_to_id.get(token, self.unk_id) for token in self.tokenize(text)]
        if add_special_tokens:
            return [self.bos_id, *token_ids, self.eos_id]
        return token_ids

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        """Decode IDs into readable text."""
        tokens = []
        reverse_vocabulary = self.id_to_token
        for token_id in token_ids:
            token = reverse_vocabulary.get(token_id, "<UNK>")
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)

        text = " ".join(tokens)
        return re.sub(r"\s+([,.!?;:])", r"\1", text)

    def __len__(self) -> int:
        return len(self.token_to_id)


if __name__ == "__main__":
    tokenizer = Tokenizer.from_file(Path(__file__).with_name("data.txt"))
    sample = "Transformers use attention."
    encoded = tokenizer.encode(sample)
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")
