"""A step of ``ds-build``: the pretraining molecules as token numbers, ready to be trained on.

    stage1_tokens.npy      every molecule's molecule tokens, as token numbers, one after another
    stage1_offsets.npy     where each molecule starts and ends in that array
    stage1_split.npy       0 for a training molecule, 1 for a validation one

Why not read ``structures.csv`` and look the molecule tokens up every epoch, as the training used to? Because a
token number under 2,000 costs 28 bytes as a Python ``int`` and 2 bytes as an ``int16``, and the lists
were built again on every pass:

    3.4 million molecules as Python lists   about 5.4 GB, rebuilt every epoch
    the same as one int16 array             about 230 MB, written once

That is the difference between a pool of molecules that fits in memory and one that does not, and the
GPU stops waiting on Python between batches. The array is written once here and read with ``mmap``, so
the pages are shared and nothing is loaded that is not touched.

The vocabulary is fixed before this step, so the numbers stay valid for as long as ``vocabulary.json``
does; the saved tokens are stamped with the vocabulary they were made with, and the training refuses
tokens made with another one.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.core.vocabulary import Vocabulary
from scripts.preprocessing.odor_tables import STRUCTURES_PATH
from scripts.preprocessing.odor_tokens import TABLES_DIR

TOKENS_PATH = TABLES_DIR / "stage1_tokens.npy"
OFFSETS_PATH = TABLES_DIR / "stage1_offsets.npy"
SPLIT_PATH = TABLES_DIR / "stage1_split.npy"
STAMP_PATH = TABLES_DIR / "stage1_tokens.json"  # which vocabulary the numbers belong to
FILES = ("stage1_tokens.npy", "stage1_offsets.npy", "stage1_split.npy", "stage1_tokens.json")
ROWS_AT_A_TIME = 200_000
SPLITS = ("train", "validation")


def pack(vocabulary: Vocabulary, path=STRUCTURES_PATH, chunk: int = ROWS_AT_A_TIME):
    """The molecules of ``path`` as one array of token numbers, with where each begins and its split."""
    pieces: list[np.ndarray] = []
    lengths: list[int] = []
    splits: list[int] = []
    unknown = 0
    for block in pd.read_csv(path, usecols=["selfies", "split"], chunksize=chunk):
        for selfies, split in zip(block["selfies"], block["split"], strict=True):
            try:
                numbers = vocabulary.molecule_ids(selfies)
            except KeyError:
                unknown += 1  # a molecule token the vocabulary does not have
                continue
            pieces.append(np.array(numbers, dtype=np.int16))
            lengths.append(len(numbers))
            splits.append(SPLITS.index(split) if split in SPLITS else 0)
    if unknown:
        print(
            f"{unknown:,} molecules use a molecule token the vocabulary does not have and are left out"
        )
    tokens = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int16)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return tokens, offsets, np.array(splits, dtype=np.uint8)


def write(vocabulary: Vocabulary, structures_path, directory) -> tuple[np.ndarray, np.ndarray]:
    """Pack ``structures_path`` and save the four files into ``directory``; return the tokens and splits."""
    tokens, offsets, splits = pack(vocabulary, structures_path)
    np.save(directory / FILES[0], tokens)
    np.save(directory / FILES[1], offsets)
    np.save(directory / FILES[2], splits)
    (directory / FILES[3]).write_text(
        json.dumps({"tokens": list(vocabulary.tokens), "molecules": len(offsets) - 1}),
        encoding="utf-8",
    )
    return tokens, splits


def main() -> None:
    vocabulary = Vocabulary.load()
    print(
        f"Packing {STRUCTURES_PATH.name} with a vocabulary of {len(vocabulary)} tokens", flush=True
    )
    tokens, splits = write(vocabulary, STRUCTURES_PATH, TOKENS_PATH.parent)
    molecules = len(splits)
    print(f"Saved {molecules:,} molecules, {len(tokens):,} tokens to {TOKENS_PATH}")
    print(
        f"  {tokens.nbytes / 1e6:.0f} MB, against about {28 * len(tokens) / 1e9:.1f} GB as Python ints"
    )
    for number, name in enumerate(SPLITS):
        print(f"  {name}: {int((splits == number).sum()):,} molecules")


if __name__ == "__main__":
    main()
