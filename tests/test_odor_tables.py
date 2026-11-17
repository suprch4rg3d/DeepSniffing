"""Tests for the odor words and for the tables of the two training stages."""

from __future__ import annotations

import pandas as pd
import pytest
import selfies as sf
from rdkit import Chem

from scripts.preprocessing import odor_tables as tables
from scripts.preprocessing import odor_words as words


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Fruity aroma??", "fruity"),
        ("  FLOWERS ", "floral"),  # the lemma, then the synonym
        ("sulphurous", "sulfurous"),
        ("lemon", "lemon"),  # a word with its own molecules is not merged
        ("odorless", ""),
        ("characteristic", ""),
        ("nan", ""),
    ],
)
def test_normalize_word(raw, expected):
    assert words.normalize_word(raw) == expected


def test_canonical_smiles_keeps_stereochemistry_and_mixtures():
    left, right = (
        words.canonical_smiles("C[C@H](N)C(=O)O"),
        words.canonical_smiles("C[C@@H](N)C(=O)O"),
    )
    assert left != right and "@" in left
    assert words.canonical_smiles("CCO.Cl") is not None  # nothing is dropped for being a mixture
    assert words.canonical_smiles("not a molecule") is None


def test_selfies_round_trip_keeps_the_mirror_image():
    for smiles in ("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "C/C=C/C=O"):
        back = sf.decoder(tables.to_selfies(smiles))
        assert Chem.MolToSmiles(Chem.MolFromSmiles(back)) == Chem.MolToSmiles(
            Chem.MolFromSmiles(smiles)
        )


def odor_table(count: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cid": range(1, count + 1),
            "smiles": ["CCO"] * count,
            "words": ["fruity;green"] * count,
            "sources": ["goodscents"] * count,
        }
    )


def test_odor_dataset_split_holds_out_the_dream_molecules():
    dream = {1, 2, 3}
    dataset = tables.build_odor_dataset(odor_table(), dream)
    assert set(dataset.loc[dataset["split"] == "dream", "cid"]) == dream
    free = dataset[dataset["split"] != "dream"]
    assert len(free) == 197
    assert (free["split"] == "validation").sum() == (free["split"] == "test").sum() == 9
    again = tables.build_odor_dataset(odor_table(), dream)
    assert dataset["split"].tolist() == again["split"].tolist()  # the seed is fixed


def test_structures_leave_out_dream_validation_and_test_molecules(monkeypatch):
    monkeypatch.setattr(
        tables,
        "known_structures",
        lambda: {cid: "CCO" for cid in range(1, 401)},  # 200 more molecules than the odor table has
    )
    dream = {1, 2, 3}
    dataset = tables.build_odor_dataset(odor_table(), dream)
    structures = tables.build_structures(dataset, dream)
    held_back = set(dataset.loc[dataset["split"].isin(["dream", "validation", "test"]), "cid"])
    assert not set(structures["cid"]) & held_back
    assert set(structures["cid"]) >= set(dataset.loc[dataset["split"] == "train", "cid"])
    assert (structures.loc[structures["cid"] > 200, "source"] == "other").all()
    assert set(structures["split"]) == {"train", "validation"}


def test_vocabulary_has_special_tokens_symbols_and_words():
    structures = pd.DataFrame({"selfies": [tables.to_selfies("CCO")]})
    dataset = pd.DataFrame({"selfies": [tables.to_selfies("CC=O")], "words": ["fruity;green"]})
    vocabulary = tables.build_vocabulary(structures, dataset)
    assert vocabulary["special"] == list(tables.SPECIAL_TOKENS)
    assert {"[C]", "[O]", "[=O]"} <= set(vocabulary["symbols"])
    assert vocabulary["words"] == ["fruity", "green"]
