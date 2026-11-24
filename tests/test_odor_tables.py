"""Tests for the odor tokens and for the tables of the two training stages."""

from __future__ import annotations

import pandas as pd
import pytest
import selfies as sf
from rdkit import Chem

from scripts.preprocessing import odor_tables as tables
from scripts.preprocessing import odor_tokens as odor_tokens


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Fruity aroma??", "fruity"),
        ("  FLOWERS ", "floral"),  # the lemma, then the synonym
        ("sulphurous", "sulfurous"),
        ("lemon", "lemon"),  # an odor token with its own molecules is not merged
        ("odorless", ""),
        ("characteristic", ""),
        ("nan", ""),
    ],
)
def test_normalize_odor_token(raw, expected):
    assert odor_tokens.normalize_odor_token(raw) == expected


def test_canonical_smiles_keeps_stereochemistry_and_mixtures():
    left, right = (
        odor_tokens.canonical_smiles("C[C@H](N)C(=O)O"),
        odor_tokens.canonical_smiles("C[C@@H](N)C(=O)O"),
    )
    assert left != right and "@" in left
    assert (
        odor_tokens.canonical_smiles("CCO.Cl") is not None
    )  # nothing is dropped for being a mixture
    assert odor_tokens.canonical_smiles("not a molecule") is None


def test_selfies_round_trip_keeps_the_mirror_image():
    for smiles in ("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "C/C=C/C=O"):
        back = sf.decoder(tables.to_selfies(smiles))
        assert Chem.MolToSmiles(Chem.MolFromSmiles(back)) == Chem.MolToSmiles(
            Chem.MolFromSmiles(smiles)
        )


ENDINGS = ("O", "N", "C=O", "C(=O)O", "S", "Cl")
PREFIXES = ("", "F", "Br", "I")


def molecule(cid: int) -> str:
    """A different real molecule for every CID, so that a structure stands for one molecule only."""
    return (
        PREFIXES[(cid // 150) % len(PREFIXES)]
        + "C" * (cid % 25 + 1)
        + ENDINGS[(cid // 25) % len(ENDINGS)]
    )


def no_pubchem() -> pd.DataFrame:
    """Stands in for the PubChem table, so a test never depends on that 178 MB file being built."""
    return pd.DataFrame(columns=["cid", "smiles", "selfies", "source"])


def odor_table(count: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cid": range(1, count + 1),
            "smiles": [molecule(cid) for cid in range(1, count + 1)],
            "odor_tokens": ["fruity;green"] * count,
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
        # 200 more molecules than the odor table has
        lambda: {cid: molecule(cid) for cid in range(1, 401)},
    )
    monkeypatch.setattr(tables, "read_pubchem_structures", no_pubchem)
    monkeypatch.setattr(tables, "dream_structures", set)
    dream = {1, 2, 3}
    dataset = tables.build_odor_dataset(odor_table(), dream)
    structures = tables.build_structures(dataset, dream)
    held_back = set(dataset.loc[dataset["split"].isin(["dream", "validation", "test"]), "cid"])
    assert not set(structures["cid"]) & held_back
    assert set(structures["cid"]) >= set(dataset.loc[dataset["split"] == "train", "cid"])
    assert (structures.loc[structures["cid"] > 200, "source"] == "other").all()
    # the validation molecules are a fixed and very thin slice of the whole pool (about 5,000 of 58.85
    # million), so a table of 400 molecules is not expected to hold any of them
    assert set(structures["split"]) <= {"train", "validation"}


def test_a_pubchem_molecule_of_a_held_out_structure_is_left_out(monkeypatch):
    """PubChem gives one molecule more than one CID, so a held-out structure must go by its structure."""
    monkeypatch.setattr(tables, "known_structures", dict)
    monkeypatch.setattr(tables, "dream_structures", set)
    dream = {1, 2, 3}
    dataset = tables.build_odor_dataset(odor_table(), dream)
    held_back = dataset[dataset["split"].isin(["dream", "validation", "test"])]
    sneaked = tables.canonical_smiles(held_back["smiles"].iloc[0])
    monkeypatch.setattr(
        tables,
        "read_pubchem_structures",
        lambda: pd.DataFrame(
            {  # the same molecule under a CID of its own, and one that is new
                "cid": [9_000_001, 9_000_002],
                "smiles": [sneaked, tables.canonical_smiles("CCCCCCCCO")],
                "selfies": [tables.to_selfies(sneaked), tables.to_selfies("CCCCCCCCO")],
                "source": ["pubchem", "pubchem"],
            }
        ),
    )
    structures = tables.build_structures(dataset, dream)
    assert sneaked not in set(structures["smiles"])  # kept out by its structure
    assert 9_000_001 not in set(structures["cid"])
    assert 9_000_002 in set(structures["cid"])  # and the new one is kept


def test_vocabulary_has_special_tokens_molecule_tokens_and_odor_tokens():
    structures = pd.DataFrame({"selfies": [tables.to_selfies("CCO")]})
    dataset = pd.DataFrame(
        {"selfies": [tables.to_selfies("CC=O")], "odor_tokens": ["fruity;green"]}
    )
    vocabulary = tables.build_vocabulary(structures, dataset)
    assert vocabulary["special"] == list(tables.SPECIAL_TOKENS)
    assert {"[C]", "[O]", "[=O]"} <= set(vocabulary["molecule_tokens"])
    assert vocabulary["odor_tokens"] == ["fruity", "green"]


def test_a_dream_molecule_with_no_odor_tokens_is_kept_out_by_its_structure(monkeypatch):
    """Twelve of the 480 DREAM molecules never reach odor_dataset, so only a structure can hold them out."""
    monkeypatch.setattr(tables, "known_structures", dict)
    sneaked = tables.canonical_smiles("CCCCCCCCO")
    monkeypatch.setattr(tables, "dream_structures", lambda: {sneaked})
    monkeypatch.setattr(
        tables,
        "read_pubchem_structures",
        lambda: pd.DataFrame(
            {
                "cid": [9_000_001, 9_000_002],
                "smiles": [sneaked, tables.canonical_smiles("CCCCO")],
                "selfies": [tables.to_selfies(sneaked), tables.to_selfies("CCCCO")],
                "source": ["pubchem", "pubchem"],
            }
        ),
    )
    dataset = tables.build_odor_dataset(odor_table(), {9_000_000})
    structures = tables.build_structures(dataset, {9_000_000})
    assert sneaked not in set(structures["smiles"])  # held out by its structure
    assert 9_000_002 in set(structures["cid"])  # and a new molecule is kept


def test_the_leakage_check_finds_a_dream_molecule_in_the_pretraining(monkeypatch, tmp_path):
    from scripts.preprocessing import check_leakage

    keller = tmp_path / "keller.csv"
    structures = tmp_path / "structures.csv"
    keller.write_text(
        '"","CID","CanonicalSMILES"\n"1","1","CCO"\n"2","2","CCN"\n', encoding="utf-8"
    )
    monkeypatch.setattr(check_leakage, "DREAM_MOLECULES_PATH", keller)
    monkeypatch.setattr(check_leakage, "STRUCTURES_PATH", structures)
    structures.write_text("cid,smiles\n1,CCO\n3,CCC\n", encoding="utf-8")
    assert check_leakage.main() == 1
    structures.write_text("cid,smiles\n3,CCC\n4,CCCC\n", encoding="utf-8")
    assert check_leakage.main() == 0
