"""Tests for the other writings of a molecule: they are the same molecule, mirror images included."""

from __future__ import annotations

import json

import pandas as pd
import selfies as sf
from rdkit import Chem

from scripts.preprocessing import odor_variants
from scripts.preprocessing.odor_tables import to_selfies

R_CARVONE = "C=C(C)[C@@H]1CC=C(C)C(=O)C1"
S_CARVONE = "C=C(C)[C@H]1CC=C(C)C(=O)C1"


def canonical(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def variants(cid: int, smiles: str) -> list[str]:
    return odor_variants.variants_of(cid, canonical(smiles), to_selfies(canonical(smiles)))[0]


def test_every_variant_is_the_same_molecule():
    found = variants(439570, R_CARVONE)
    assert len(found) == odor_variants.VARIANTS
    assert {canonical(sf.decoder(text)) for text in found} == {canonical(R_CARVONE)}


def test_the_mirror_images_share_no_variant():
    assert not set(variants(439570, R_CARVONE)) & set(variants(16724, S_CARVONE))


def test_the_canonical_writing_is_not_a_variant_and_none_repeats():
    found = variants(439570, R_CARVONE)
    assert to_selfies(canonical(R_CARVONE)) not in found
    assert len(set(found)) == len(found)


def test_a_small_molecule_has_few_variants_and_the_same_ones_every_time():
    assert 0 < len(variants(702, "CCO")) < odor_variants.VARIANTS
    assert variants(439570, R_CARVONE) == variants(439570, R_CARVONE)


def test_symbols_that_only_a_variant_uses_join_the_vocabulary(tmp_path):
    path = tmp_path / "vocabulary.json"
    path.write_text(json.dumps({"special": ["<pad>"], "symbols": ["[C]"], "words": ["sweet"]}))
    new = odor_variants.add_to_vocabulary(pd.DataFrame({"selfies": ["[C][#C-1]"]}), path)
    assert new == ["[#C-1]"]
    assert json.loads(path.read_text())["symbols"] == ["[#C-1]", "[C]"]
    assert odor_variants.add_to_vocabulary(pd.DataFrame({"selfies": ["[C]"]}), path) == []


def test_a_saved_table_is_checked_and_replaces_the_file(tmp_path):
    path = tmp_path / "variants.csv"
    table = pd.DataFrame({"cid": [1, 2], "selfies": ["[C]", "[C][C]"]})
    odor_variants.save_checked(table, path)
    assert pd.read_csv(path).equals(table)
    assert not path.with_suffix(".tmp").exists()
