"""Tests for the inference entry point, using a small synthetic checkpoint."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from scripts.core.data import (
    ATOM_FEATURE_COUNT,
    INTERACTION_COLUMNS,
    RDKIT_COLUMNS,
    dragon_group_indices,
    fit_feature_scaler,
)
from scripts.core.metrics import TARGET_COUNT, TARGETS
from scripts.preprocessing import feature_selection
from scripts.training.custom_model import GraphDescriptorModel
from scripts.training.predict import build_inputs, load_model, predict

DRAGON_COLUMNS = [f"dragon_d{i}" for i in range(6)]
DRAGON_GROUPS = {"alpha": DRAGON_COLUMNS[:3], "beta": DRAGON_COLUMNS[3:]}
MORGAN_COLUMNS = [f"morgan_{i}" for i in range(4)]
NAME_VOCABULARY = ["ETHA", "THAN", "HANO", "ANOL"]
NUMERIC_COUNT = len(RDKIT_COLUMNS) + len(INTERACTION_COLUMNS) + len(MORGAN_COLUMNS) + 2


@pytest.fixture
def tables():
    row = {
        "CID": 1,
        "rdkit_canonical_smiles": "CCO",
        "OdorName": "ETHANOL",
        "rdkit_gasteiger_atom_charges": json.dumps([0.0, 0.1, -0.2]),
        "rdkit_gasteiger_implicit_hydrogen_charges": json.dumps([0.0, 0.0, 0.0]),
    }
    row.update({column: 1.0 for column in RDKIT_COLUMNS})
    row.update({column: 0.5 for column in INTERACTION_COLUMNS})
    row.update({column: 2.0 for column in DRAGON_COLUMNS})
    fingerprints = {"CID": 1, **{column: 1.0 for column in MORGAN_COLUMNS}}
    return pd.DataFrame([row]), pd.DataFrame([fingerprints])


def test_predict_returns_one_row_per_attribute(community_model_dir, tables, community_table):
    artifacts = load_model(community_model_dir, device="cpu")

    table = predict("CCO", artifacts, (*tables, community_table))

    assert list(table.columns) == ["attribute", "mean", "sigma"]
    assert table["attribute"].tolist() == TARGETS
    assert table["mean"].between(0, 100).all()
    assert table["sigma"].between(0, 100).all()


def test_predict_finds_a_molecule_by_cid_or_by_how_it_is_written(
    community_model_dir, tables, community_table
):
    artifacts = load_model(community_model_dir, device="cpu")

    by_smiles = predict("CCO", artifacts, (*tables, community_table))

    assert predict("OCC", artifacts, (*tables, community_table)).equals(by_smiles)
    assert predict("1", artifacts, (*tables, community_table)).equals(by_smiles)


def test_predict_rejects_a_molecule_outside_the_dream_set(
    community_model_dir, tables, community_table
):
    artifacts = load_model(community_model_dir, device="cpu")

    with pytest.raises(ValueError, match="not one of the DREAM molecules"):
        predict("CCCC", artifacts, (*tables, community_table))


def test_predict_rejects_invalid_smiles(community_model_dir, tables, community_table):
    artifacts = load_model(community_model_dir, device="cpu")

    with pytest.raises(ValueError, match="could not parse"):
        predict("this is not a molecule", artifacts, (*tables, community_table))


def test_load_model_reports_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train the model first"):
        load_model(tmp_path, device="cpu")


def test_load_model_rejects_a_checkpoint_from_another_model(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    torch.save({"model_name": "CustomOdorModel"}, tmp_path / "checkpoints/custom_model.pt")

    with pytest.raises(ValueError, match="Train the model again"):
        load_model(tmp_path, device="cpu")


def test_load_model_rejects_a_subchallenge_one_checkpoint(community_model_dir):
    checkpoint_path = community_model_dir / "checkpoints/custom_model.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["challenge"] = "q1"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="Subchallenge 1"):
        load_model(community_model_dir, device="cpu")


EPI_COLUMNS = ["epi_00_log_kow", "epi_01_vapor_pressure"]
STRUCTURE_COLUMNS = ["nspdk_00000", "nsim_0000", "msim_0000"]


@pytest.fixture
def community_model_dir(tmp_path):
    torch.manual_seed(1)
    numeric_count = NUMERIC_COUNT + len(EPI_COLUMNS)
    name_count = len(NAME_VOCABULARY)
    model = GraphDescriptorModel(
        dragon_group_indices(DRAGON_COLUMNS, DRAGON_GROUPS),
        numeric_count,
        name_count,
        TARGET_COUNT * 2,
        structure_count=len(STRUCTURE_COLUMNS),
    )
    generator = np.random.default_rng(1)
    training = pd.DataFrame({"CID": range(8)})
    for column in [*DRAGON_COLUMNS, *RDKIT_COLUMNS, *INTERACTION_COLUMNS, *EPI_COLUMNS]:
        training[column] = generator.lognormal(size=8)
    checkpoint = {
        "model_name": "GraphDescriptorModel",
        "state_dict": model.state_dict(),
        "dragon_columns": DRAGON_COLUMNS,
        "dragon_feature_groups": DRAGON_GROUPS,
        "morgan_columns": MORGAN_COLUMNS,
        "name_ngram_vocabulary": NAME_VOCABULARY,
        "name_count": name_count,
        "structure_count": len(STRUCTURE_COLUMNS),
        "community_columns": {"structure": STRUCTURE_COLUMNS},
        "numeric_count": numeric_count,
        "target_columns": [f"column_{i}" for i in range(TARGET_COUNT * 2)],
        "target_names": list(TARGETS),
        "high_log10_concentration": -3.0,
        "atom_feature_count": ATOM_FEATURE_COUNT,
    }
    (tmp_path / "checkpoints").mkdir()
    joblib.dump(
        fit_feature_scaler(training, DRAGON_COLUMNS), tmp_path / "checkpoints/feature_scaler.joblib"
    )
    joblib.dump(
        StandardScaler().fit(generator.normal(50, 10, size=(8, TARGET_COUNT * 2))),
        tmp_path / "checkpoints/target_scaler.joblib",
    )
    torch.save(checkpoint, tmp_path / "checkpoints/custom_model.pt")
    return tmp_path


@pytest.fixture
def community_table():
    row = {"CID": 1, "epi_00_log_kow": -0.3, "epi_01_vapor_pressure": 4.5e3}
    row.update({column: 0.25 for column in STRUCTURE_COLUMNS})
    return pd.DataFrame([row])


def test_predict_uses_the_community_features(community_model_dir, tables, community_table):
    artifacts = load_model(community_model_dir, device="cpu")

    table = predict("CCO", artifacts, (*tables, community_table))

    assert table["attribute"].tolist() == TARGETS
    assert np.isfinite(table[["mean", "sigma"]].to_numpy()).all()


def test_predict_without_the_community_table_names_the_command_to_create_it(
    community_model_dir, tables
):
    artifacts = load_model(community_model_dir, device="cpu")

    with pytest.raises(ValueError, match="ds-build"):
        predict("CCO", artifacts, tables)


@pytest.fixture(autouse=True)
def without_feature_manifest(monkeypatch, tmp_path):
    """These tests use small synthetic tables, so they must not depend on the real manifest."""
    monkeypatch.setattr(feature_selection, "MANIFEST_PATH", tmp_path / "no_manifest.csv")


def test_predict_takes_each_score_group_from_its_own_model(
    community_model_dir, tables, community_table
):
    artifacts = load_model(community_model_dir, device="cpu")
    torch.manual_seed(7)
    other = GraphDescriptorModel(
        dragon_group_indices(DRAGON_COLUMNS, DRAGON_GROUPS),
        NUMERIC_COUNT + len(EPI_COLUMNS),
        len(NAME_VOCABULARY),
        TARGET_COUNT * 2,
        structure_count=len(STRUCTURE_COLUMNS),
    ).eval()
    default = predict("CCO", artifacts, (*tables, community_table))

    artifacts["group_models"] = {"intensity_mean": other}
    artifacts["checkpoint"]["group_columns"] = {"intensity_mean": [0]}
    mixed = predict("CCO", artifacts, (*tables, community_table))

    artifacts["model"] = other
    artifacts["group_models"] = {}
    alone = predict("CCO", artifacts, (*tables, community_table))

    assert mixed["mean"].iloc[0] == pytest.approx(alone["mean"].iloc[0])
    assert mixed["mean"].iloc[0] != pytest.approx(default["mean"].iloc[0])
    assert mixed["mean"].iloc[1:].tolist() == default["mean"].iloc[1:].tolist()
    assert mixed["sigma"].tolist() == default["sigma"].tolist()


def test_predict_reads_the_organizer_name_windows_the_model_was_trained_with(
    community_model_dir, tables, community_table
):
    artifacts = load_model(community_model_dir, device="cpu")
    windows = ["orgname_00000", "orgname_00001"]
    artifacts["checkpoint"]["community_columns"]["name"] = windows
    table = community_table.assign(orgname_00000=1.0, orgname_00001=0.0)
    molecules, fingerprints = tables

    inputs = build_inputs(molecules.iloc[0], fingerprints, artifacts, table)

    assert inputs["name_features"].shape == (len(NAME_VOCABULARY) + 2,)
    assert inputs["name_features"][-2:].tolist() == [1.0, 0.0]


def test_predict_takes_the_spread_from_the_curve_of_the_mean(
    community_model_dir, tables, community_table
):
    from scripts.core.data import std_curve_values

    artifacts = load_model(community_model_dir, device="cpu")
    artifacts["checkpoint"]["std_curve"] = {
        "k0": np.full(TARGET_COUNT, 0.7),
        "k1": np.full(TARGET_COUNT, 1.1),
        "weight": 1.0,
    }

    table = predict("CCO", artifacts, (*tables, community_table))

    expected = std_curve_values(table["mean"].to_numpy(), 0.7, 1.1)
    assert np.allclose(table["sigma"].to_numpy(), expected, atol=0.05)


def test_predict_gives_intensity_at_the_standard_dilution_and_nothing_else_changes(
    community_model_dir, tables, community_table
):
    artifacts = load_model(community_model_dir, device="cpu")
    artifacts["checkpoint"]["high_log10_concentration"] = -1.0
    before = predict("CCO", artifacts, (*tables, community_table))

    artifacts["checkpoint"]["intensity_standard_log10_dilution"] = -3.0
    after = predict("CCO", artifacts, (*tables, community_table))

    assert before.loc[0, "mean"] != after.loc[0, "mean"]  # intensity moved
    assert before["mean"].iloc[1:].tolist() == after["mean"].iloc[1:].tolist()
    assert before["sigma"].iloc[1:].tolist() == after["sigma"].iloc[1:].tolist()


def test_the_command_scores_a_molecule_with_the_community_features(
    community_model_dir, tables, community_table, monkeypatch, capsys
):
    import sys

    from scripts.training import predict as predict_module

    monkeypatch.setattr(predict_module, "load_tables", lambda: tables)
    monkeypatch.setattr(predict_module, "load_community_table", lambda: community_table)
    arguments = ["ds-predict", "--smiles", "CCO", "--model-dir", str(community_model_dir)]
    monkeypatch.setattr(sys, "argv", [*arguments, "--device", "cpu"])

    predict_module.main()

    assert "sigma" in capsys.readouterr().out
