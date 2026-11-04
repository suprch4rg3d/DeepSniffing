"""Tests that the figures are written without a display."""

from __future__ import annotations

from scripts.core.plots import save_test_metrics_plot, save_training_plots


def test_test_metrics_plot_is_written(tmp_path, bundle):
    from scripts.core.metrics import dream_challenge_metrics
    from scripts.core.train import predict_values
    from scripts.training.custom_model import build_model

    model = build_model(bundle, branches={"descriptors"})
    predictions = predict_values(model, bundle.arrays["test"], bundle.target_scaler, "cpu")
    metrics = dream_challenge_metrics(bundle.raw_targets["test"], predictions)

    path = save_test_metrics_plot(metrics, tmp_path)

    assert path.is_file()
    assert path.stat().st_size > 0


def test_training_plots_are_written(tmp_path):
    history = {
        "train_loss": [1.0, 0.9, 0.8],
        "validation_loss": [1.1, 1.0, 0.95],
        "validation_metrics": [
            {
                "z_scores": {
                    "intensity_mean": 0.1,
                    "valence_mean": 0.2,
                    "descriptors_mean": 0.3,
                    "intensity_std": 0.1,
                    "valence_std": 0.1,
                    "descriptors_std": 0.1,
                },
                "dream_challenge_score": 0.2,
            }
        ]
        * 3,
    }

    paths = save_training_plots(history, tmp_path)

    assert len(paths) == 2
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
