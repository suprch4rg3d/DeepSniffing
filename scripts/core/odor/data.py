"""The training texts of the odor decoder, and the batches made of them.

Three kinds of text, each a list of token numbers with the place where the loss starts:

    molecule             <bos> SELFIES symbols <eos>                         stage 1 and 2: learn the molecule
    words -> molecule    <bos> <odor> word word ... <sep> symbols <eos>      stage 2: write a molecule for the words
    molecule -> words    <bos> <mol> symbols <sep> word word ... <eos>       stage 2: name the words of a molecule

The loss is taken on what is to be predicted and not on the instruction: the symbols and the end token
for the first two, the words and the end token for the last. The words are a set, so their order is random
every time, and the model has no position for them.
"""

from __future__ import annotations

import random

import pandas as pd
import torch

from scripts.core.odor.vocabulary import Vocabulary
from scripts.preprocessing.odor_tables import DATASET_PATH, STRUCTURES_PATH
from scripts.preprocessing.odor_variants import VARIANTS_PATH

MAX_WORDS_IN_REQUEST = 6  # a request names 1 to 6 of a molecule's words
NO_WORDS_PROBABILITY = 0.1  # and a tenth of the time none, so the model can write for no request
MAX_WORDS_OUT = 40  # the most words that are listed for one molecule
BATCH_SIZE = 64  # the most texts in a batch
# The attention of a batch takes memory in proportion to texts x length x length, so a batch of long texts has
# fewer texts: texts x length x length stays below this. 64 texts of 160 tokens fit, and one text of 837 does.
ATTENTION_BUDGET = 64 * 160 * 160
BUCKET_BATCHES = 50  # batches are made of texts of about the same length, within this many batches

Example = tuple[
    list[int], int
]  # token numbers, and the index of the first token that the loss counts
Item = tuple[str, list[str]]  # a molecule's SELFIES and its odor words


def molecule_text(
    vocabulary: Vocabulary, selfies: str, words: list[str], rng: random.Random
) -> Example:
    ids = [
        vocabulary.special("<bos>"),
        *vocabulary.symbol_ids(selfies),
        vocabulary.special("<eos>"),
    ]
    return ids, 1


def words_to_molecule(
    vocabulary: Vocabulary, selfies: str, words: list[str], rng: random.Random
) -> Example:
    chosen: list[str] = []
    if words and rng.random() >= NO_WORDS_PROBABILITY:
        chosen = rng.sample(words, rng.randint(1, min(MAX_WORDS_IN_REQUEST, len(words))))
    prompt = [
        vocabulary.special("<bos>"),
        vocabulary.special("<odor>"),
        *vocabulary.word_ids(chosen),
        vocabulary.special("<sep>"),
    ]
    return [*prompt, *vocabulary.symbol_ids(selfies), vocabulary.special("<eos>")], len(prompt)


def molecule_to_words(
    vocabulary: Vocabulary, selfies: str, words: list[str], rng: random.Random
) -> Example:
    shuffled = rng.sample(words, min(len(words), MAX_WORDS_OUT))
    prompt = [
        vocabulary.special("<bos>"),
        vocabulary.special("<mol>"),
        *vocabulary.symbol_ids(selfies),
        vocabulary.special("<sep>"),
    ]
    return [*prompt, *vocabulary.word_ids(shuffled), vocabulary.special("<eos>")], len(prompt)


def load_structures(split: str) -> list[Item]:
    """Stage 1: every molecule of the split, without words."""
    table = pd.read_csv(STRUCTURES_PATH)
    return [(selfies, []) for selfies in table.loc[table["split"] == split, "selfies"]]


def load_odor(split: str) -> list[Item]:
    """Stage 2: the molecules of the split with their odor words."""
    table = pd.read_csv(DATASET_PATH).fillna("")
    rows = table[table["split"] == split]
    return [
        (selfies, text.split(";") if text else [])
        for selfies, text in zip(rows["selfies"], rows["words"], strict=True)
    ]


def load_variants() -> dict[str, list[str]]:
    """The other writings of every molecule (``selfies_variants.csv``), by its canonical SELFIES."""
    canonical: dict[int, str] = {}
    for path in (STRUCTURES_PATH, DATASET_PATH):
        table = pd.read_csv(path, usecols=["cid", "selfies"])
        canonical.update(zip(table["cid"], table["selfies"], strict=True))
    found: dict[str, list[str]] = {}
    variants = pd.read_csv(VARIANTS_PATH)
    for cid, selfies in zip(variants["cid"], variants["selfies"], strict=True):
        found.setdefault(canonical[cid], []).append(selfies)
    return found


def batches(
    vocabulary: Vocabulary,
    items: list[Item],
    kinds: list,
    rng: random.Random,
    shuffle: bool,
    batch_size: int = BATCH_SIZE,
    budget: int = ATTENTION_BUDGET,
):
    """Padded token numbers and the mask of the tokens that the loss counts, for every kind of text."""
    examples = [kind(vocabulary, selfies, words, rng) for selfies, words in items for kind in kinds]
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
