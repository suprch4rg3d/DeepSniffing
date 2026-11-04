"""Tests for the official DREAM scoring."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.core.metrics import (
    OFFICIAL_NULL_SD,
    TARGET_COUNT,
    TARGETS,
    dream_challenge_metrics,
    pearson_across_molecules,
    prediction_matrix_from_submission,
)


def make_matrix(means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return np.concatenate((means, stds), axis=1)


def test_target_list_matches_official_attribute_count():
    assert TARGET_COUNT == 21
    assert len(TARGETS) == len(set(TARGETS))


def test_perfect_predictions_score_one_over_null_standard_deviation():
    rng = np.random.default_rng(0)
    actual = make_matrix(rng.random((69, 21)), rng.random((69, 21)))

    metrics = dream_challenge_metrics(actual, actual)

    for name, null_sd in OFFICIAL_NULL_SD.items():
        assert metrics["pearson_correlations"][name] == pytest.approx(1.0)
        assert metrics["z_scores"][name] == pytest.approx(1.0 / null_sd)
    assert metrics["dream_challenge_score"] == pytest.approx(
        np.mean([1.0 / null_sd for null_sd in OFFICIAL_NULL_SD.values()])
    )


def test_constant_predictions_score_zero():
    rng = np.random.default_rng(1)
    actual = make_matrix(rng.random((69, 21)), rng.random((69, 21)))
    constant = np.full_like(actual, 42.0)

    metrics = dream_challenge_metrics(actual, constant)

    assert metrics["dream_challenge_score"] == pytest.approx(0.0)
    assert all(value == pytest.approx(0.0) for value in metrics["z_scores"].values())


def test_inverted_predictions_score_negative():
    rng = np.random.default_rng(2)
    actual = make_matrix(rng.random((69, 21)), rng.random((69, 21)))
    inverted = 100.0 - actual

    metrics = dream_challenge_metrics(actual, inverted)

    assert metrics["dream_challenge_score"] < 0


def test_metrics_reject_wrong_number_of_outputs():
    actual = np.zeros((69, TARGET_COUNT))
    with pytest.raises(ValueError):
        dream_challenge_metrics(actual, actual)


def test_metrics_reject_mismatched_shapes():
    actual = np.zeros((69, TARGET_COUNT * 2))
    shorter = np.zeros((68, TARGET_COUNT * 2))
    with pytest.raises(ValueError):
        dream_challenge_metrics(actual, shorter)


def test_pearson_ignores_missing_values():
    actual = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    assert pearson_across_molecules(actual, predicted) == pytest.approx(1.0)


def test_pearson_is_zero_for_degenerate_input():
    assert pearson_across_molecules(np.array([1.0, 2.0]), np.array([5.0, 5.0])) == 0.0
    assert pearson_across_molecules(np.array([1.0]), np.array([2.0])) == 0.0


def test_pearson_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        pearson_across_molecules(np.zeros(3), np.zeros(4))


def test_submission_loader_reads_the_reference_file(bundle):
    from scripts.core.data import REFERENCE_PREDICTIONS_PATH

    matrix = prediction_matrix_from_submission(REFERENCE_PREDICTIONS_PATH, bundle.test_cids)

    assert matrix.shape == (len(bundle.test_cids), TARGET_COUNT * 2)
    assert np.isfinite(matrix).all()


def test_submission_loader_rejects_unknown_cids(bundle):
    from scripts.core.data import REFERENCE_PREDICTIONS_PATH

    with pytest.raises(ValueError):
        prediction_matrix_from_submission(REFERENCE_PREDICTIONS_PATH, bundle.test_cids[:-1])
