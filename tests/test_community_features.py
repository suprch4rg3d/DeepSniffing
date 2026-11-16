"""Tests for the community feature table, built from small synthetic source files."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.preprocessing import community_features as module

DREAM = [10, 20, 30, 40, 50, 60]
REFERENCE = [901, 902]
EVERY_CID = DREAM + REFERENCE
# The NSPDK files list the molecules in an order of their own, with references mixed in.
NSPDK_ORDER = [901, 10, 20, 902, 30, 40, 50, 60]


def nspdk_vectors() -> np.ndarray:
    """Unit length vectors, like the ones that EDeN produces."""
    generator = np.random.default_rng(5)
    vectors = generator.random((8, 6)) * (generator.random((8, 6)) < 0.6)
    vectors[:, 0] += 0.1
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    generator = np.random.default_rng(0)
    data, hulab = tmp_path / "data", tmp_path / "hulab"
    (data / "derived").mkdir(parents=True)
    hulab.mkdir()

    windows = np.array(
        [[1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]]
    )  # the third window belongs to one molecule only
    pd.DataFrame(windows, index=DREAM).to_csv(data / "name_features.txt", sep="\t", header=False)

    epi = pd.DataFrame(
        {
            "CID": DREAM,
            "SMILES": ["C"] * 6,
            "Estimated Log Kow ": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
            "Constant": [7.0] * 6,
            "Biodegrades": ["YES ", "NO ", "YES ", "NO ", "YES ", "NO "],
        }
    )
    epi.to_csv(data / "DREAM_episuite_descriptors.txt", sep="\t", index=False)

    values = generator.random((6, 8)) * 0.8
    for row in range(6):
        values[row, row] = 1.0  # a DREAM molecule is most similar to itself
    pd.DataFrame(values, index=DREAM, columns=EVERY_CID).to_csv(data / "morgan_sim.csv")

    nspdk = pd.DataFrame(
        [[1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
        index=DREAM,
        columns=range(4),
        dtype=float,
    )  # the last substructure belongs to one molecule only
    nspdk.to_csv(hulab / "nspdk2.csv")

    pd.Series(NSPDK_ORDER).to_csv(data / "derived/nspdk_cid.csv", header=False, index=False)
    lines = []
    for vector in nspdk_vectors():
        items = " ".join(f"{index}:{value}" for index, value in enumerate(vector) if value)
        lines.append(f"0 {items}")
    (data / "derived/nspdk_r3_d4_unaug.svm").write_text("\n".join(lines) + "\n", encoding="utf-8")

    monkeypatch.setattr(module, "NAME_PATH", data / "name_features.txt")
    monkeypatch.setattr(module, "EPI_PATH", data / "DREAM_episuite_descriptors.txt")
    monkeypatch.setattr(module, "MORGAN_SIMILARITY_PATH", data / "morgan_sim.csv")
    monkeypatch.setattr(module, "NSPDK_PATH", hulab / "nspdk2.csv")
    monkeypatch.setattr(module, "NSPDK_VECTORS_PATH", data / "derived/nspdk_r3_d4_unaug.svm")
    monkeypatch.setattr(module, "NSPDK_CIDS_PATH", data / "derived/nspdk_cid.csv")
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "community_features.csv")
    monkeypatch.setattr(module, "EXPECTED_MOLECULES", 6)
    monkeypatch.setattr(module, "MIN_MOLECULES", 2)
    return tmp_path


def columns_with(table: pd.DataFrame, prefix: str) -> list[str]:
    return [column for column in table if column.startswith(prefix)]


def test_every_feature_family_is_joined_by_cid(sources):
    table = module.build_community_features()

    assert table["CID"].tolist() == DREAM
    for prefix in (
        module.NAME_PREFIX,
        module.EPI_PREFIX,
        module.NSPDK_PREFIX,
        module.NSPDK_SIMILARITY_PREFIX,
        module.MORGAN_SIMILARITY_PREFIX,
    ):
        assert columns_with(table, prefix)
    assert pd.read_csv(module.OUTPUT).shape == table.shape


def test_name_windows_and_substructures_held_by_one_molecule_are_dropped(sources):
    table = module.build_community_features()

    assert len(columns_with(table, module.NAME_PREFIX)) == 3  # the window of one is out
    assert len(columns_with(table, module.NSPDK_PREFIX)) == 3  # the substructure of one is out


def test_similarity_to_the_molecules_themselves_is_dropped(sources):
    table = module.build_community_features()

    # Only the two reference odorants remain, so no molecule has a column of its own.
    for prefix in (module.NSPDK_SIMILARITY_PREFIX, module.MORGAN_SIMILARITY_PREFIX):
        assert len(columns_with(table, prefix)) == len(REFERENCE)
    assert table[columns_with(table, module.MORGAN_SIMILARITY_PREFIX)].max().max() < 1.0


def test_nspdk_similarity_is_the_dot_product_of_the_vectors(sources):
    table = module.build_community_features().set_index("CID")

    vectors = nspdk_vectors()
    position = {cid: index for index, cid in enumerate(NSPDK_ORDER)}
    for cid in DREAM:
        for reference in REFERENCE:
            column = f"{module.NSPDK_SIMILARITY_PREFIX}{position[reference]:04d}"
            expected = float(vectors[position[cid]] @ vectors[position[reference]])
            assert table.loc[cid, column] == pytest.approx(expected, abs=1e-5)


def test_nspdk_vectors_that_are_not_normalised_are_rejected(sources):
    path = sources / "data/derived/nspdk_r3_d4_unaug.svm"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace("0 ", "0 0:5 ", 1)  # the vector of a DREAM molecule, CID 10
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not normalised"):
        module.build_community_features()


def test_nspdk_vectors_and_their_molecule_list_must_agree(sources):
    path = sources / "data/derived/nspdk_cid.csv"
    path.write_text("\n".join(map(str, NSPDK_ORDER[:-1])) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="differ in length"):
        module.build_community_features()


def test_epi_suite_is_numeric_and_keeps_only_varying_properties(sources):
    table = module.build_community_features()
    epi = columns_with(table, module.EPI_PREFIX)

    assert len(epi) == 2  # the constant property is out, the YES or NO one is kept
    assert table[epi].dtypes.map(pd.api.types.is_numeric_dtype).all()
    assert table[epi].isna().sum().sum() == 1  # a missing estimate stays missing
    assert not any("smiles" in column for column in epi)


def test_a_missing_source_file_names_the_command_that_fetches_it(sources, monkeypatch):
    monkeypatch.setattr(module, "NSPDK_PATH", sources / "hulab/missing.csv")

    with pytest.raises(FileNotFoundError, match="ds-prepare"):
        module.build_community_features()


def test_molecules_that_differ_between_files_are_rejected(sources):
    nspdk = pd.read_csv(sources / "hulab/nspdk2.csv", index_col=0).iloc[:-1]
    nspdk.to_csv(sources / "hulab/nspdk2.csv")

    with pytest.raises(ValueError, match="same unique molecules"):
        module.build_community_features()


def test_name_windows_must_be_binary(sources):
    path = sources / "data/name_features.txt"
    path.write_text(path.read_text(encoding="utf-8").replace("\t0\t", "\t2\t", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="binary"):
        module.build_community_features()
