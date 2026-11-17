"""The tokens of the odor decoder: special tokens, SELFIES symbols and odor words, read from ``vocabulary.json``.

``ds-build`` writes the file once, before stage 1, so the two training stages and every saved model share the
same token numbers. A token's group follows from its number: the special tokens come first, then the SELFIES
symbols, then the odor words.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import selfies as sf

from scripts.preprocessing.odor_tables import SPECIAL_TOKENS, VOCABULARY_PATH

PAD, BOS, EOS, SEP, ODOR, MOL = SPECIAL_TOKENS


@dataclass(frozen=True)
class Vocabulary:
    tokens: tuple[str, ...]
    first_symbol: int
    first_word: int

    @classmethod
    def load(cls, path: Path = VOCABULARY_PATH) -> Vocabulary:
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Run ds-build.")
        content = json.loads(path.read_text(encoding="utf-8"))
        if tuple(content["special"]) != SPECIAL_TOKENS:
            raise ValueError(
                "vocabulary.json has other special tokens than the code. Run ds-build."
            )
        tokens = (*content["special"], *content["symbols"], *content["words"])
        return cls(
            tokens, len(content["special"]), len(content["special"]) + len(content["symbols"])
        )

    def __len__(self) -> int:
        return len(self.tokens)

    @cached_property
    def index(self) -> dict[str, int]:
        return {token: number for number, token in enumerate(self.tokens)}

    def special(self, name: str) -> int:
        return SPECIAL_TOKENS.index(name)

    def symbol_ids(self, selfies: str) -> list[int]:
        """The token numbers of a SELFIES string."""
        index = self.index
        return [index[symbol] for symbol in sf.split_selfies(selfies)]

    def word_ids(self, words: list[str]) -> list[int]:
        index = self.index
        return [index[word] for word in words]
