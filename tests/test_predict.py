"""Tests for the inference entry point."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.core.data import ENRICHED_FEATURE_COLUMNS, FINGERPRINT_SIZE
from scripts.core.metrics import TARGETS
from scripts.training.predict import (
    DEFAULT_MODEL_DIR,
    featurize_smiles,
    load_model,
    predict,
)


def test_featurize_returns_descriptors_and_fingerprint():
    descriptors, fingerprint = featurize_smiles("CCO", ENRICHED_FEATURE_COLUMNS)

    assert descriptors.shape == (1, len(ENRICHED_FEATURE_COLUMNS))
    assert fingerprint.shape == (FINGERPRINT_SIZE,)
    assert np.isin(fingerprint, (0.0, 1.0)).all()


def test_featurize_reads_the_chemical_formula_of_ethanol():
    descriptors, _ = featurize_smiles("CCO", ENRICHED_FEATURE_COLUMNS)
    values = dict(zip(ENRICHED_FEATURE_COLUMNS, descriptors[0], strict=True))

    assert values["rdkit_atom_count_c"] == 2
    assert values["rdkit_atom_count_h"] == 6
    assert values["rdkit_atom_count_o"] == 1
    assert values["rdkit_heavy_atom_count"] == 3


def test_featurize_rejects_invalid_smiles():
    with pytest.raises(ValueError):
        featurize_smiles("this is not a molecule", ENRICHED_FEATURE_COLUMNS)


def test_featurize_is_invariant_to_how_the_molecule_is_written():
    first, _ = featurize_smiles("CCO", ENRICHED_FEATURE_COLUMNS)
    second, _ = featurize_smiles("OCC", ENRICHED_FEATURE_COLUMNS)

    assert np.allclose(first, second)


def test_predict_returns_one_row_per_attribute():
    artifacts = load_model(DEFAULT_MODEL_DIR, device="cpu")

    table = predict("CCO", artifacts)

    assert list(table.columns) == ["attribute", "mean", "sigma"]
    assert table["attribute"].tolist() == TARGETS
    assert table["mean"].between(0, 100).all()
    assert table["sigma"].between(0, 100).all()


def test_load_model_reports_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train the model first"):
        load_model(tmp_path, device="cpu")
