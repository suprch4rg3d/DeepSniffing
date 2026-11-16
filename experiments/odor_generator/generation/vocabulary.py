"""The tokens of the decoder.

A sequence is ``<bos> <word> <word> ... <sep> SELFIES symbols ... <eos>``. The vocabulary is, in order:
the special tokens, one token for each odor word, and the SELFIES symbols of the training molecules.
Symbols that no training molecule has are not in it, so a molecule that needs one cannot be used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .chemistry import selfies_tokens

PAD, BOS, SEP, EOS, NONE = "<pad>", "<bos>", "<sep>", "<eos>", "<none>"
SPECIAL = (PAD, BOS, SEP, EOS, NONE)
MAX_WORDS_IN_PROMPT = 6
MAX_SYMBOLS = 80  # the longest molecule, in SELFIES symbols, that the decoder reads or writes


class Vocabulary:
    def __init__(self, words: list[str], symbols: list[str]):
        self.words = list(words)
        self.tokens = list(SPECIAL) + [f"<{word}>" for word in words] + list(symbols)
        self.index = {token: i for i, token in enumerate(self.tokens)}
        self.first_symbol = len(SPECIAL) + len(words)

    def word_id(self, word: str) -> int:
        return self.index[f"<{word}>"]

    def can_write(self, symbols: list[str] | None) -> bool:
        """Whether a molecule fits the decoder: it has symbols, not too many, and all are known."""
        return (
            bool(symbols) and len(symbols) <= MAX_SYMBOLS and all(s in self.index for s in symbols)
        )

    def __len__(self) -> int:
        return len(self.tokens)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"words": self.words, "tokens": self.tokens}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        saved = json.loads(path.read_text(encoding="utf-8"))
        vocabulary = cls(saved["words"], [])
        vocabulary.tokens = saved["tokens"]
        vocabulary.index = {token: i for i, token in enumerate(vocabulary.tokens)}
        return vocabulary


def build_vocabulary(frame: pd.DataFrame, words: list[str]) -> Vocabulary:
    """The vocabulary of the training molecules and the given odor words."""
    symbols = sorted(
        {
            s
            for smiles in frame.loc[frame["split"] == "train", "smiles"]
            for s in selfies_tokens(smiles) or []
        }
    )
    return Vocabulary(words, symbols)
