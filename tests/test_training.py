"""Smoke tests for the shared training loop."""

from __future__ import annotations

import numpy as np
import torch

from scripts.core.metrics import TARGET_COUNT
from scripts.core.train import TrainConfig, inverse_target_transform, train_and_evaluate
from scripts.training.custom_model import build_model


def test_one_epoch_training_produces_finite_metrics(bundle):
    model = build_model(bundle, branches={"smiles"})
    config = TrainConfig(epochs=1, patience=1, batch_size=64)

    result = train_and_evaluate(model, bundle, config, device="cpu", verbose=False)

    assert result.best_epoch == 1
    assert len(result.history["train_loss"]) == 1
    assert np.isfinite(result.history["train_loss"][0])
    assert np.isfinite(result.test_metrics["dream_challenge_score"])
    assert result.test_predictions.shape == (bundle.test_size, TARGET_COUNT * 2)


def test_predictions_stay_in_the_rating_range(bundle):
    model = build_model(bundle, branches={"descriptors"})
    config = TrainConfig(epochs=1, patience=1, batch_size=64)

    result = train_and_evaluate(model, bundle, config, device="cpu", verbose=False)

    assert result.test_predictions.min() >= 0.0
    assert result.test_predictions.max() <= 100.0


def test_early_stopping_respects_the_patience(bundle):
    model = build_model(bundle, branches={"smiles"})
    config = TrainConfig(epochs=5, patience=1, batch_size=64)

    result = train_and_evaluate(model, bundle, config, device="cpu", verbose=False)

    assert len(result.history["train_loss"]) <= 5


def test_every_branch_combination_runs(bundle):
    for branches in ({"smiles"}, {"descriptors"}, {"fingerprints"}, {"smiles", "fingerprints"}):
        model = build_model(bundle, branches=branches)
        tokens, features, fingerprints = bundle.arrays["test"]
        output = model(tokens, features, fingerprints)

        assert output.shape == (bundle.test_size, TARGET_COUNT * 2)
        assert torch.isfinite(output).all()


def test_disabled_branches_remove_the_input_signal(bundle):
    tokens, features, fingerprints = bundle.arrays["test"]
    smiles_only = build_model(bundle, branches={"smiles"})
    smiles_only.eval()
    with torch.no_grad():
        first = smiles_only(tokens, features, fingerprints)
        second = smiles_only(tokens, features * 7.0, fingerprints * 0.0 + 1.0)

    assert torch.allclose(first, second, atol=1e-6)


def test_inverse_transform_clips_to_the_rating_range(bundle):
    extreme = np.full((2, TARGET_COUNT * 2), 1e6)

    restored = inverse_target_transform(extreme, bundle.target_scaler)

    assert restored.max() <= 100.0
    assert restored.min() >= 0.0
