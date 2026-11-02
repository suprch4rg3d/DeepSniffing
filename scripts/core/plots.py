"""Figures produced by a training run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.core.data import PUBLISHED_BENCHMARKS_PATH
from scripts.core.metrics import OFFICIAL_NULL_SD

BENCHMARK_LABELS = {
    "Published challenge winner": "Published challenge\nwinner",
    "Published all-team model aggregate": "Published all-team\nmodel aggregate",
    "Published post-challenge ensemble": "Published post-challenge\nensemble",
}


def save_training_plots(history: dict, output_dir: Path) -> list[Path]:
    """Plot training and validation loss plus the validation challenge score."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history["train_loss"], label="Training")
    axis.plot(history["validation_loss"], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Scaled MSE (42 outputs)")
    axis.set_title("Combined model training loss")
    axis.legend()
    figure.tight_layout()
    loss_path = output_dir / "training_loss.png"
    figure.savefig(loss_path, dpi=160)
    plt.close(figure)
    written.append(loss_path)

    metric_names = list(OFFICIAL_NULL_SD)
    figure, axis = plt.subplots(figsize=(9, 5))
    for name in metric_names:
        values = [entry["z_scores"][name] for entry in history["validation_metrics"]]
        axis.plot(range(1, len(values) + 1), values, label=name.replace("_", " "))
    scores = [entry["dream_challenge_score"] for entry in history["validation_metrics"]]
    axis.plot(
        range(1, len(scores) + 1),
        scores,
        color="black",
        linewidth=2.5,
        label="Mean Z-score",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Official DREAM Q2 Z-score components")
    axis.set_title("DREAM Subchallenge 2 validation metrics")
    axis.legend(fontsize=8, ncol=2)
    axis.text(
        0.01,
        -0.24,
        "Internal validation fold; the final 69-molecule test score is reported separately.",
        transform=axis.transAxes,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    metrics_path = output_dir / "dream_challenge_metrics.png"
    figure.savefig(metrics_path, dpi=160)
    plt.close(figure)
    written.append(metrics_path)
    return written


def save_test_metrics_plot(metrics: dict, output_dir: Path) -> Path:
    """Plot the official test set z scores as a bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(OFFICIAL_NULL_SD)
    values = [metrics["z_scores"][name] for name in names]
    names.append("Mean Z-score")
    values.append(metrics["dream_challenge_score"])

    figure, axis = plt.subplots(figsize=(9, 4.5))
    bars = axis.bar(names, values, color=["#4c78a8"] * 6 + ["#f58518"])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Official DREAM Q2 Z-score")
    axis.set_title("Official test-set challenge metrics")
    axis.tick_params(axis="x", rotation=25)
    axis.bar_label(bars, fmt="%.2f", padding=3)
    figure.tight_layout()
    path = output_dir / "test_metrics.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def save_challenge_comparison_plot(
    model_metrics: dict,
    output_dir: Path,
    reference_metrics: dict | None = None,
    benchmarks_path: Path = PUBLISHED_BENCHMARKS_PATH,
) -> Path:
    """Compare a model score against the stored and published DREAM scores."""
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmarks = json.loads(benchmarks_path.read_text(encoding="utf-8"))

    names = ["Custom model"]
    scores = [model_metrics["dream_challenge_score"]]
    colors = ["#f58518"]
    if reference_metrics is not None:
        names.append("Stored reference prediction\n(unattributed)")
        scores.append(reference_metrics["dream_challenge_score"])
        colors.append("#4c78a8")
    for result in benchmarks["scores"]:
        names.append(BENCHMARK_LABELS.get(result["name"], result["name"]))
        scores.append(float(result["score"]))
        colors.append("#72b7b2")

    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(names, scores, color=colors)
    axis.set_ylabel("DREAM Q2 mean Z-score (higher is better)")
    axis.set_title("Custom model compared with DREAM Subchallenge 2 scores")
    axis.bar_label(bars, fmt="%.2f", padding=3)
    axis.set_ylim(0, max(scores) * 1.18)
    axis.tick_params(axis="x", labelsize=9)
    figure.tight_layout()
    path = output_dir / "dream_challenge_comparison.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
