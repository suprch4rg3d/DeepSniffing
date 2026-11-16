"""Tests for the PubChem 3D structures: the download, the atom mapping, and the graph."""

from __future__ import annotations

import io
import urllib.error

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from scripts.core.data import DISTANCE_RBF_COUNT, load_conformers, molecule_graph
from scripts.data import prepare_data
from scripts.preprocessing import conformers as module


def record(cid: int) -> str:
    """A minimal PubChem style record: the identifier first, and the usual terminators."""
    return (
        f"{cid}\n  test\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
        f"M  END\n> <PUBCHEM_COMPOUND_CID>\n{cid}\n\n$$$$\n"
    )


def pubchem_record(smiles: str, cid: int) -> str:
    """A real 3D record of a molecule, whose atoms are numbered like the SMILES says."""
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(molecule, randomSeed=1)
    molecule.SetProp("_Name", str(cid))
    return Chem.MolToMolBlock(molecule) + "$$$$\n"


# ---------------------------------------------------------------------------------------
# the download
# ---------------------------------------------------------------------------------------
def test_records_are_split_by_their_compound_identifier():
    records = prepare_data.split_sdf_records(record(126) + record(702))

    assert sorted(records) == [126, 702]
    assert all(text.rstrip().endswith("$$$$") for text in records.values())


def test_an_empty_answer_has_no_records():
    assert prepare_data.split_sdf_records("") == {}


def test_a_cut_off_answer_is_an_error_and_not_a_missing_compound():
    with pytest.raises(ValueError, match="middle of a record"):
        prepare_data.split_sdf_records(record(126)[:-10])


def test_a_record_without_an_identifier_is_rejected():
    with pytest.raises(ValueError, match="Malformed"):
        prepare_data.split_sdf_records(record(126).replace("126", "name", 1))


class Answer(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def answers(monkeypatch, outcomes):
    """Make urlopen return, in turn, each text or raise each exception."""
    calls = []

    def fake(request, timeout):
        calls.append(request.full_url)
        outcome = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return Answer(outcome.encode("utf-8"))

    monkeypatch.setattr(prepare_data.urllib.request, "urlopen", fake)
    monkeypatch.setattr(prepare_data.time, "sleep", lambda seconds: None)
    return calls


def test_a_batch_is_one_request_and_a_missing_compound_is_left_out(monkeypatch):
    calls = answers(monkeypatch, [record(126) + record(176)])

    records = prepare_data.fetch_conformer_batch([126, 176, 6505])

    assert len(calls) == 1 and "126,176,6505" in calls[0] and "record_type=3d" in calls[0]
    assert sorted(records) == [126, 176]


def test_a_404_means_no_compound_of_the_batch_has_a_3d_structure(monkeypatch):
    error = urllib.error.HTTPError("url", 404, "not found", {}, None)
    answers(monkeypatch, [error])

    assert prepare_data.fetch_conformer_batch([6505]) == {}


def test_a_temporary_failure_is_retried(monkeypatch):
    calls = answers(monkeypatch, [urllib.error.URLError("down"), record(126)])

    assert sorted(prepare_data.fetch_conformer_batch([126])) == [126]
    assert len(calls) == 2


def test_a_failure_that_lasts_is_reported(monkeypatch):
    answers(monkeypatch, [urllib.error.URLError("down")])

    with pytest.raises(urllib.error.URLError):
        prepare_data.fetch_conformer_batch([126])


def test_a_compound_that_was_not_requested_is_rejected(monkeypatch):
    answers(monkeypatch, [record(999)])

    with pytest.raises(ValueError, match="not requested"):
        prepare_data.fetch_conformer_batch([126])


@pytest.fixture
def downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_data, "CONFORMERS_PATH", tmp_path / "poses.sdf")
    monkeypatch.setattr(prepare_data, "CONFORMERS_UNAVAILABLE_PATH", tmp_path / "none.txt")
    monkeypatch.setattr(prepare_data, "CONFORMER_BATCH_SIZE", 2)
    monkeypatch.setattr(prepare_data, "read_wanted_cids", lambda: {126, 176, 177, 6505})
    monkeypatch.setattr(prepare_data.time, "sleep", lambda seconds: None)
    return tmp_path


def test_the_download_keeps_what_it_has_and_asks_only_for_what_is_missing(downloads, monkeypatch):
    asked = []

    def fake(batch):
        asked.append(list(batch))
        return {cid: record(cid) for cid in batch if cid != 6505}

    monkeypatch.setattr(prepare_data, "fetch_conformer_batch", fake)

    assert prepare_data.fetch_pubchem_conformers() == 0
    assert asked == [[126, 176], [177, 6505]]
    saved = prepare_data.split_sdf_records((downloads / "poses.sdf").read_text())
    assert sorted(saved) == [126, 176, 177]
    assert (downloads / "none.txt").read_text().split() == ["6505"]

    asked.clear()
    assert prepare_data.fetch_pubchem_conformers() == 0
    assert asked == []  # everything is known, so nothing is requested again


def test_a_failed_batch_keeps_the_progress_and_reports_failure(downloads, monkeypatch):
    def fake(batch):
        if 177 in batch:
            raise urllib.error.URLError("down")
        return {cid: record(cid) for cid in batch}

    monkeypatch.setattr(prepare_data, "fetch_conformer_batch", fake)

    assert prepare_data.fetch_pubchem_conformers() == 1
    saved = prepare_data.split_sdf_records((downloads / "poses.sdf").read_text())
    assert sorted(saved) == [126, 176]
    assert not (downloads / "none.txt").read_text().strip()  # an error is not "no 3D structure"
    assert not list(downloads.glob("*.part"))  # no half written file is left


# ---------------------------------------------------------------------------------------
# the atom mapping
# ---------------------------------------------------------------------------------------
def distance(coordinates, left, right) -> float:
    return float(np.linalg.norm(coordinates[left] - coordinates[right]))


def test_the_pose_follows_the_atom_order_of_our_smiles():
    # PubChem numbers ethanol O, C, C here, and our SMILES numbers it C, C, O.
    ordered = module.pose_in_our_atom_order(pubchem_record("OCC", 702), "CCO")

    assert ordered.shape == (3, 3)
    assert distance(ordered, 0, 1) == pytest.approx(1.5, abs=0.15)  # carbon to carbon
    assert distance(ordered, 1, 2) == pytest.approx(1.43, abs=0.15)  # carbon to oxygen
    assert distance(ordered, 0, 2) > 2.2  # the oxygen is not next to the first carbon


def test_a_pose_of_another_molecule_is_rejected():
    with pytest.raises(ValueError, match="atom counts differ"):
        module.pose_in_our_atom_order(pubchem_record("CCCC", 1), "CCO")


def test_a_pose_of_the_same_size_but_other_elements_is_rejected():
    with pytest.raises(ValueError):
        module.pose_in_our_atom_order(pubchem_record("CCN", 1), "CCO")


def test_a_flat_zero_pose_is_rejected_but_a_lone_atom_is_fine():
    flat = pubchem_record("CC", 1).replace("-", " ")
    zeros = Chem.MolFromMolBlock(pubchem_record("CC", 1))
    for index in range(zeros.GetNumAtoms()):
        zeros.GetConformer().SetAtomPosition(index, (0.0, 0.0, 0.0))
    zero_block = Chem.MolToMolBlock(zeros) + "$$$$\n"

    with pytest.raises(ValueError, match="all zero"):
        module.pose_in_our_atom_order(zero_block, "CC")
    water = module.pose_in_our_atom_order(pubchem_record("O", 962).replace("-", "-"), "O")
    assert water.shape == (1, 3)
    assert flat


def test_the_table_has_one_row_per_atom_and_names_the_molecules_left_out(tmp_path, monkeypatch):
    sdf = tmp_path / "poses.sdf"
    sdf.write_text(pubchem_record("OCC", 702))
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        {"CID": [702, 6505], "rdkit_canonical_smiles": ["CCO", "CCCCCCCCCCCCCCCCCCCC"]}
    ).to_csv(labels, index=False)
    monkeypatch.setattr(module, "SDF_PATH", sdf)
    monkeypatch.setattr(module, "INPUT", labels)
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "conformers.csv")

    table = module.build_conformers()

    assert table["CID"].tolist() == [702, 702, 702]
    assert table["atom_index"].tolist() == [0, 1, 2]
    assert table["symbol"].tolist() == ["C", "C", "O"]
    assert pd.read_csv(module.OUTPUT).shape == table.shape


# ---------------------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------------------
ETHANOL = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.1, 1.2, 0.0]])
CHARGES = [0.0, 0.1, -0.2]


def test_a_graph_with_a_pose_connects_every_atom_pair_within_range():
    atoms, edges, edge_features = molecule_graph("CCO", 702, CHARGES, [0.0] * 3, ETHANOL)

    assert atoms.shape[0] == 3
    assert edges.shape[1] == 6  # every ordered pair of the three atoms
    assert edge_features[:, -DISTANCE_RBF_COUNT:].sum() > 0


def test_a_graph_without_a_pose_has_the_bonds_only_and_no_distances():
    _, edges, edge_features = molecule_graph("CCO", 702, CHARGES, [0.0] * 3)

    assert edges.shape[1] == 4  # two bonds, in both directions
    assert (edge_features[:, -DISTANCE_RBF_COUNT:] == 0).all()
    assert edge_features[:, :-DISTANCE_RBF_COUNT].sum() > 0  # the bond features are still there


def test_a_pose_that_does_not_fit_the_molecule_is_rejected():
    with pytest.raises(ValueError, match="do not fit"):
        molecule_graph("CCO", 702, CHARGES, [0.0] * 3, ETHANOL[:2])


def test_the_same_molecule_always_gets_the_same_graph():
    first = molecule_graph("CCO", 702, CHARGES, [0.0] * 3, ETHANOL)
    second = molecule_graph("CCO", 702, CHARGES, [0.0] * 3, ETHANOL)

    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_a_missing_table_names_the_commands_that_create_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="ds-build"):
        load_conformers(tmp_path / "missing.csv")


def test_the_table_is_read_into_one_pose_per_molecule(tmp_path):
    path = tmp_path / "conformers.csv"
    pd.DataFrame(
        {
            "CID": [7, 7, 7],
            "atom_index": [2, 0, 1],
            "symbol": ["O", "C", "C"],
            "x": [2.0, 0.0, 1.0],
            "y": [0.0] * 3,
            "z": [0.0] * 3,
        }
    ).to_csv(path, index=False)

    poses = load_conformers(path)

    assert list(poses) == [7]
    assert poses[7][:, 0].tolist() == [0.0, 1.0, 2.0]  # sorted by the atom index
