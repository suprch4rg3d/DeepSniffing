"""Tests for the data contract."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.core.data import (
    ENRICHED_FEATURE_COLUMNS,
    FINGERPRINT_SIZE,
    MAX_SEQUENCE_LENGTH,
    OFFICIAL_TEST_SIZE,
)
from scripts.core.metrics import TARGET_COUNT
from scripts.core.splits import official_test_cids


def test_bundle_split_sizes(bundle):
    assert bundle.train_size == len(bundle.train_cids) == 287
    assert bundle.validation_size == len(bundle.validation_cids) == 51
    assert bundle.test_size == len(bundle.test_cids) == OFFICIAL_TEST_SIZE


def test_test_split_is_exactly_the_official_set(bundle):
    assert set(bundle.test_cids) == set(official_test_cids())


def test_no_official_test_molecule_reaches_training(bundle):
    held_out = set(official_test_cids())

    assert held_out.isdisjoint(bundle.train_cids)
    assert held_out.isdisjoint(bundle.validation_cids)
    assert set(bundle.train_cids).isdisjoint(bundle.validation_cids)


def test_train_and_validation_carry_targets(bundle):
    for split in ("train", "validation"):
        arrays = bundle.arrays[split]
        assert len(arrays) == 4
        tokens, features, fingerprints, targets = arrays
        rows = len(tokens)

        assert tokens.shape == (rows, MAX_SEQUENCE_LENGTH)
        assert features.shape == (rows, len(ENRICHED_FEATURE_COLUMNS))
        assert fingerprints.shape == (rows, FINGERPRINT_SIZE)
        assert targets.shape == (rows, TARGET_COUNT * 2)


def test_test_split_has_no_targets_attached(bundle):
    arrays = bundle.arrays["test"]

    assert len(arrays) == 3
    assert len(arrays[0]) == OFFICIAL_TEST_SIZE


def test_fingerprints_are_binary(bundle):
    fingerprints = bundle.arrays["train"][2].numpy()

    assert np.isin(fingerprints, (0.0, 1.0)).all()


def test_tokens_stay_inside_the_vocabulary(bundle):
    tokens = bundle.arrays["train"][0]

    assert int(tokens.max()) < bundle.vocab_size
    assert int(tokens.min()) >= 0


def test_feature_scaler_centres_the_training_split(bundle):
    scaled_features = bundle.arrays["train"][1].numpy()

    assert np.allclose(scaled_features.mean(axis=0), 0.0, atol=1e-5)


def test_target_scaler_centres_the_training_targets(bundle):
    targets = bundle.arrays["train"][3].numpy()

    assert np.allclose(targets.mean(axis=0), 0.0, atol=1e-5)


def test_raw_test_targets_are_finite_and_in_rating_range(bundle):
    actual = bundle.raw_targets["test"]

    assert actual.shape == (OFFICIAL_TEST_SIZE, TARGET_COUNT * 2)
    assert np.isfinite(actual).all()
    assert actual.min() >= 0.0
    assert actual.max() <= 100.0


def test_bundle_counts_match_their_columns(bundle):
    assert bundle.target_count == TARGET_COUNT * 2
    assert bundle.feature_count == len(ENRICHED_FEATURE_COLUMNS)
    assert bundle.fingerprint_count == FINGERPRINT_SIZE
    assert bundle.vocab_size > 2


def test_prepare_data_is_stable_across_calls(bundle):
    from scripts.core.data import prepare_data

    second = prepare_data()

    assert second.train_cids == bundle.train_cids
    assert second.validation_cids == bundle.validation_cids
    assert second.test_cids == bundle.test_cids
    assert np.array_equal(second.raw_targets["test"], bundle.raw_targets["test"])


def test_unknown_columns_are_reported(monkeypatch, tmp_path):
    import pandas as pd

    from scripts.core import data as data_module

    molecules = pd.read_csv(data_module.MOLECULES_PATH)
    broken = molecules.drop(columns=[ENRICHED_FEATURE_COLUMNS[0]])
    broken_path = tmp_path / "molecules_enriched.csv"
    broken.to_csv(broken_path, index=False)

    monkeypatch.setattr(data_module, "MOLECULES_PATH", broken_path)
    with pytest.raises(ValueError, match="missing"):
        data_module.load_molecules()
