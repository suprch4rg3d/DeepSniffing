"""Tests that the figures are written without a display."""

from __future__ import annotations

import numpy as np

from scripts.core.metrics import TARGET_COUNT, dream_challenge_metrics
from scripts.core.plots import save_test_metrics_plot, save_training_plots


def test_test_metrics_plot_is_written(tmp_path):
    generator = np.random.default_rng(0)
    actual = generator.uniform(0, 100, size=(30, TARGET_COUNT * 2))
    predicted = actual + generator.normal(0, 10, size=actual.shape)
    metrics = dream_challenge_metrics(actual, predicted)

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


def test_training_plot_labels_describe_the_loss(tmp_path, monkeypatch):
    import matplotlib.axes

    labels = []
    original = matplotlib.axes.Axes.set_ylabel

    def record(self, label, *args, **kwargs):
        labels.append(label)
        return original(self, label, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylabel", record)
    history = {
        "train_loss": [-1.0, -2.0],
        "validation_loss": [-0.5, -1.0],
        "validation_metrics": [
            {
                "z_scores": dict.fromkeys(
                    (
                        "intensity_mean",
                        "valence_mean",
                        "descriptors_mean",
                        "intensity_std",
                        "valence_std",
                        "descriptors_std",
                    ),
                    0.1,
                ),
                "dream_challenge_score": 0.1,
            }
        ]
        * 2,
    }

    save_training_plots(history, tmp_path, loss_label="Negative DREAM score")

    assert labels[0] == "Negative DREAM score"


def write_run(output_dir):
    """The files that ds-train leaves for one model: per-epoch scores and the test scores."""
    import json

    import pandas as pd

    metrics = output_dir / "metrics"
    metrics.mkdir(parents=True)
    pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_loss": [-0.1, -0.2, -0.3],
            "dream_challenge_score": [1.0, 2.0, 3.0],
            "pearson_intensity_mean": [0.1, 0.2, 0.3],
            "z_intensity_mean": [1.0, 2.0, 3.0],
            "z_valence_mean": [0.5, 1.5, 2.5],
        }
    ).to_csv(metrics / "validation_dream_challenge_metrics.csv", index=False)
    (metrics / "dream_challenge_test_metrics.json").write_text(
        json.dumps(
            {
                "dream_challenge_score": 2.5,
                "z_scores": {"intensity_mean": 2.0, "valence_mean": 3.0},
                "stored_reference_prediction_score": 9.6,
            }
        ),
        encoding="utf-8",
    )


def test_a_run_is_read_back_from_the_files_it_left(tmp_path):
    from scripts.core.plots import read_run

    write_run(tmp_path)

    history, test_metrics = read_run(tmp_path)

    assert history["train_loss"] == [-0.1, -0.2, -0.3]
    assert history["validation_loss"] == [-1.0, -2.0, -3.0]
    assert history["validation_metrics"][1]["z_scores"] == {
        "intensity_mean": 2.0,
        "valence_mean": 1.5,
    }
    assert test_metrics["dream_challenge_score"] == 2.5


def test_the_plot_script_draws_every_figure_from_the_saved_files(tmp_path, monkeypatch, capsys):
    from scripts.core import plots

    write_run(tmp_path / "q2")
    monkeypatch.setattr(plots, "Q2_OUTPUT_DIR", tmp_path / "q2")
    monkeypatch.setattr(plots, "Q1_OUTPUT_DIR", tmp_path / "q1")  # a model that was never trained

    plots.main()

    drawn = {path.name for path in (tmp_path / "q2/plots").iterdir()}
    assert drawn == {
        "training_loss.png",
        "dream_challenge_metrics.png",
        "test_metrics.png",
        "dream_challenge_comparison.png",
    }
    assert "No Q1 results" in capsys.readouterr().out
