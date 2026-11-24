"""The training texts of the odor decoder, and the batches made of them.

Two kinds of text, each a list of token numbers with the place where the loss starts:

    molecule             <bos> molecule tokens <eos>                         pretraining: learn the molecule
    odor tokens -> molecule    <bos> <odor> odor token odor token ... <sep> molecule tokens <eos>      generator: write a molecule for the odor tokens

The loss is taken on what is to be predicted, token by token, and not on the instruction. The odor tokens of a
request are a set, so the model has no position for them and they cannot see each other; which of a
molecule's odor tokens are asked for, and in which order, is drawn again every time. (The classifier does not
read texts: it reads the molecules through the frozen decoder, see ``classifier``.)
"""

from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd
import torch

from scripts.core.vocabulary import Vocabulary
from scripts.preprocessing.molecule_tokens import (
    OFFSETS_PATH,
    SPLIT_PATH,
    SPLITS,
    STAMP_PATH,
    TOKENS_PATH,
)
from scripts.preprocessing.odor_tables import DATASET_PATH

MAX_ODOR_TOKENS_IN_REQUEST = 6  # a request names 1 to 6 of a molecule's odor tokens
NO_ODOR_TOKENS_PROBABILITY = (
    0.1  # and a tenth of the time none, so the model can write for no request
)
BATCH_SIZE = 256  # the most texts in a batch
# The attention of a batch takes memory in proportion to texts x length x length, so a batch of long texts has
# fewer texts: texts x length x length stays below this. 64 texts of 160 tokens fit, and one text of 837 does.
ATTENTION_BUDGET = 64 * 160 * 160
BUCKET_BATCHES = 50  # batches are made of texts of about the same length, within this many batches

# token numbers, and the index the loss starts at
Example = tuple[list[int], int]
Item = tuple[str, list[str]]  # a molecule's SELFIES and its odor tokens


def molecule_text(
    vocabulary: Vocabulary, selfies: str, odor_tokens: list[str], rng: random.Random
) -> Example:
    ids = [
        vocabulary.special("<bos>"),
        *vocabulary.molecule_ids(selfies),
        vocabulary.special("<eos>"),
    ]
    return ids, 1


def request_text(vocabulary: Vocabulary, selfies: str, asked: list[str]) -> Example:
    """A request for the odor tokens in ``asked`` and the molecule that answers it; the loss is on the molecule."""
    prompt = [
        vocabulary.special("<bos>"),
        vocabulary.special("<odor>"),
        *vocabulary.odor_ids(asked),
        vocabulary.special("<sep>"),
    ]
    return (
        [*prompt, *vocabulary.molecule_ids(selfies), vocabulary.special("<eos>")],
        len(prompt),
    )


def odor_to_molecule(
    vocabulary: Vocabulary, selfies: str, odor_tokens: list[str], rng: random.Random
) -> Example:
    chosen: list[str] = []
    if odor_tokens and rng.random() >= NO_ODOR_TOKENS_PROBABILITY:
        chosen = rng.sample(
            odor_tokens, rng.randint(1, min(MAX_ODOR_TOKENS_IN_REQUEST, len(odor_tokens)))
        )
    return request_text(vocabulary, selfies, chosen)


# Two fixed requests for the same molecule, to measure whether the odor tokens are used at all: the loss of
# writing a molecule with its own odor tokens and with none. If the model reads its request, the first is lower,
# and the difference (``request_gain`` in the history) is how much the odor tokens are worth. A loss that mixes
# the two, like the training one, cannot show it.
def asked_molecule(
    vocabulary: Vocabulary, selfies: str, odor_tokens: list[str], rng: random.Random
) -> Example:
    chosen = (
        rng.sample(odor_tokens, min(MAX_ODOR_TOKENS_IN_REQUEST, len(odor_tokens)))
        if odor_tokens
        else []
    )
    return request_text(vocabulary, selfies, chosen)


def unasked_molecule(
    vocabulary: Vocabulary, selfies: str, odor_tokens: list[str], rng: random.Random
) -> Example:
    return request_text(vocabulary, selfies, [])


def load_packed(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stage 1 from ``molecule_tokens``: the token array, where each molecule sits in it, and this split's rows.

    The tokens are mapped, not read, so a pool of any size costs no memory until it is touched, and the
    numbers are never built again as Python objects.
    """
    if not TOKENS_PATH.is_file():
        raise FileNotFoundError(f"{TOKENS_PATH} not found. Run ds-build (or molecule_tokens).")
    stamp = json.loads(STAMP_PATH.read_text(encoding="utf-8"))
    if tuple(stamp["tokens"]) != Vocabulary.load().tokens:
        raise ValueError(
            f"{TOKENS_PATH} was packed with another vocabulary. Run molecule_tokens again."
        )
    tokens = np.load(TOKENS_PATH, mmap_mode="r")
    offsets = np.load(OFFSETS_PATH)
    splits = np.load(SPLIT_PATH)
    return tokens, offsets, np.flatnonzero(splits == SPLITS.index(split))


def packed_batches(
    vocabulary: Vocabulary,
    packed: tuple[np.ndarray, np.ndarray, np.ndarray],
    rng: random.Random,
    shuffle: bool,
    batch_size: int = BATCH_SIZE,
    budget: int = ATTENTION_BUDGET,
    request: bool = False,
):
    """Stage 1 batches cut straight out of the packed tokens: ``<bos> molecule tokens <eos>``, loss from the first.

    Yields the same two things as ``batches``: the token numbers and the mask of what the loss counts.

    With ``request`` the molecule is written after an empty request, ``<bos> <odor> <sep> molecule tokens <eos>``,
    which is how the model writes when it is asked for nothing. Mixed into the fine-tuning, these
    molecules keep that prediction in practice (and it is the one that guidance pushes away from), with
    molecules the model has no odor tokens for.
    """
    tokens, offsets, rows = packed
    bos, eos, pad = (vocabulary.special(name) for name in ("<bos>", "<eos>", "<pad>"))
    front = (
        [bos, vocabulary.special("<odor>"), vocabulary.special("<sep>")] if request else [bos]
    )  # what comes before the molecule tokens; the loss starts after it
    lead = len(front)
    order = rows.copy()
    if shuffle:
        np.random.default_rng(rng.randrange(1 << 32)).shuffle(order)
    lengths = (offsets[order + 1] - offsets[order]).astype(np.int64)
    chunk = batch_size * BUCKET_BATCHES
    grouped: list[np.ndarray] = []
    for start in range(0, len(order), chunk):
        part = order[start : start + chunk]
        part = part[np.argsort(lengths[start : start + chunk], kind="stable")]
        current: list[int] = []
        for row in part.tolist():
            length = int(offsets[row + 1] - offsets[row]) + lead + 1
            if current and (len(current) == batch_size or (len(current) + 1) * length**2 > budget):
                grouped.append(np.array(current))
                current = []
            current.append(row)
        if current:
            grouped.append(np.array(current))
    if shuffle:
        rng.shuffle(grouped)
    for batch in grouped:
        sizes = (offsets[batch + 1] - offsets[batch]).astype(np.int64)
        longest = int(sizes.max()) + lead + 1  # what comes before, the molecule tokens and <eos>
        ids = torch.full((len(batch), longest), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.bool)
        ids[:, :lead] = torch.tensor(front)
        for row, (molecule, size) in enumerate(zip(batch.tolist(), sizes.tolist(), strict=True)):
            start = int(offsets[molecule])
            ids[row, lead : lead + size] = torch.from_numpy(
                np.asarray(tokens[start : start + size], dtype=np.int64)
            )
            ids[row, lead + size] = eos
            mask[row, lead : lead + size + 1] = (
                True  # the molecule tokens and the end token are predicted
            )
        yield ids, mask


def with_replay(
    texts,
    vocabulary: Vocabulary,
    packed: tuple[np.ndarray, np.ndarray, np.ndarray],
    count: int,
    rng: random.Random,
):
    """The batches of ``texts`` and of ``count`` molecules of the pretraining written after an empty request,
    in one shuffled order.

    The pretraining molecules are drawn again every epoch, from the packed tokens, so nothing but the
    batches of the moment is held in memory. A batch is made of one kind of text and not of both, so the
    shuffle is of batches.
    """
    tokens, offsets, rows = packed
    chosen = np.random.default_rng(rng.randrange(1 << 32)).choice(
        rows, size=min(count, len(rows)), replace=False
    )
    everything = [
        *texts,
        *packed_batches(vocabulary, (tokens, offsets, chosen), rng, True, request=True),
    ]
    rng.shuffle(everything)
    return iter(everything)


def load_odor(split: str) -> list[Item]:
    """Stage 2: the molecules of the split with their odor tokens."""
    table = pd.read_csv(DATASET_PATH).fillna("")
    rows = table[table["split"] == split]
    return [
        (selfies, text.split(";") if text else [])
        for selfies, text in zip(rows["selfies"], rows["odor_tokens"], strict=True)
    ]


def batches(
    vocabulary: Vocabulary,
    items: list[Item],
    kinds: list,
    rng: random.Random,
    shuffle: bool,
    batch_size: int = BATCH_SIZE,
    budget: int = ATTENTION_BUDGET,
):
    """Padded token numbers and the mask of the tokens the loss counts, per batch."""
    examples = [
        kind(vocabulary, selfies, odor_tokens, rng)
        for selfies, odor_tokens in items
        for kind in kinds
    ]
    order = list(range(len(examples)))
    if shuffle:
        rng.shuffle(order)
    # Texts of about the same length share a batch, so that little is wasted on padding. A batch is full at
    # ``batch_size`` texts, or earlier when the attention budget is used up by long texts.
    chunk = batch_size * BUCKET_BATCHES
    grouped: list[list[int]] = []
    for start in range(0, len(order), chunk):
        part = sorted(order[start : start + chunk], key=lambda i: len(examples[i][0]))
        current: list[int] = []
        for i in part:
            length = len(examples[i][0])
            if current and (len(current) == batch_size or (len(current) + 1) * length**2 > budget):
                grouped.append(current)
                current = []
            current.append(i)
        if current:
            grouped.append(current)
    if shuffle:
        rng.shuffle(grouped)
    for batch in grouped:
        longest = max(len(examples[i][0]) for i in batch)
        ids = torch.full((len(batch), longest), vocabulary.special("<pad>"), dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.bool)
        for row, i in enumerate(batch):
            tokens, start = examples[i]
            ids[row, : len(tokens)] = torch.tensor(tokens)
            mask[row, start : len(tokens)] = True
        yield ids, mask
