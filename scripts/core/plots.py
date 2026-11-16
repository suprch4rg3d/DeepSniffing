"""``ds-plot``: draw the figures of the trained models from the files they left.

``ds-train`` writes numbers only: the scores of every epoch, the test scores and the predictions. This script
reads them back from ``output/q1`` and ``output/q2``, so the figures can be redrawn at any time without
training again. Each model's ``plots/`` folder gets four figures: the training loss, the validation scores
by epoch, the test scores, and the comparison with the published DREAM scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.core.data import PUBLISHED_BENCHMARKS_PATH, PUBLISHED_Q1_BENCHMARKS_PATH
from scripts.core.train import Q1_OUTPUT_DIR, Q2_OUTPUT_DIR

BENCHMARK_LABELS = {
    "Published challenge winner": "Published challenge\nwinner",
    "Published all-team model aggregate": "Published all-team\nmodel aggregate",
    "Published post-challenge ensemble": "Published post-challenge\nensemble",
}


def save_training_plots(
    history: dict,
    output_dir: Path,
    *,
    loss_label: str = "Scaled MSE (42 outputs)",
    loss_title: str = "Combined model training loss",
    challenge: str = "Q2",
) -> list[Path]:
    """Plot training and validation loss plus the validation challenge score.

    The labels describe the loss, because models do not share one: most train on a
    scaled squared error, while the custom model trains on the negative DREAM score.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history["train_loss"], label="Training")
    axis.plot(history["validation_loss"], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(loss_label)
    axis.set_title(loss_title)
    axis.legend()
    figure.tight_layout()
    loss_path = output_dir / "training_loss.png"
    figure.savefig(loss_path, dpi=160)
    plt.close(figure)
    written.append(loss_path)

    metric_names = list(history["validation_metrics"][0]["z_scores"])
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
    axis.set_ylabel(f"Official DREAM {challenge} Z-score components")
    axis.set_title(f"DREAM Subchallenge {challenge[-1]} validation metrics")
    axis.legend(fontsize=8, ncol=2)
    axis.text(
        0.01,
        -0.24,
        "Validation on the 69 DREAM leaderboard molecules; the 69-molecule test score is reported separately.",
        transform=axis.transAxes,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    metrics_path = output_dir / "dream_challenge_metrics.png"
    figure.savefig(metrics_path, dpi=160)
    plt.close(figure)
    written.append(metrics_path)
    return written


def save_test_metrics_plot(metrics: dict, output_dir: Path, challenge: str = "Q2") -> Path:
    """Plot the official test set z scores as a bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(metrics["z_scores"])
    values = [metrics["z_scores"][name] for name in names]
    names.append("Mean Z-score")
    values.append(metrics["dream_challenge_score"])

    figure, axis = plt.subplots(figsize=(9, 4.5))
    bars = axis.bar(names, values, color=["#4c78a8"] * (len(names) - 1) + ["#f58518"])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel(f"Official DREAM {challenge} Z-score")
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
    benchmarks_path: Path | None = PUBLISHED_BENCHMARKS_PATH,
    challenge: str = "Q2",
) -> Path:
    """Compare a model score against the stored and published DREAM scores.

    Without a ``benchmarks_path`` only the model, and the stored reference when given,
    are drawn.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmarks = (
        json.loads(benchmarks_path.read_text(encoding="utf-8"))
        if benchmarks_path
        else {"scores": []}
    )

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
    axis.set_ylabel(f"DREAM {challenge} mean Z-score (higher is better)")
    axis.set_title(f"Custom model compared with DREAM Subchallenge {challenge[-1]} scores")
    axis.bar_label(bars, fmt="%.2f", padding=3)
    axis.set_ylim(0, max(scores) * 1.18)
    axis.tick_params(axis="x", labelsize=9)
    figure.tight_layout()
    path = output_dir / "dream_challenge_comparison.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def read_run(output_dir: Path) -> tuple[dict, dict]:
    """The validation history and the test metrics that ``ds-train`` saved for one model."""
    table = pd.read_csv(output_dir / "metrics/validation_dream_challenge_metrics.csv")
    z_names = [column for column in table if column.startswith("z_")]
    history = {
        "train_loss": table["train_loss"].tolist(),
        "validation_loss": (-table["dream_challenge_score"]).tolist(),
        "validation_metrics": [
            {
                "dream_challenge_score": row["dream_challenge_score"],
                "z_scores": {name.removeprefix("z_"): row[name] for name in z_names},
            }
            for _, row in table.iterrows()
        ],
    }
    test_metrics = json.loads(
        (output_dir / "metrics/dream_challenge_test_metrics.json").read_text(encoding="utf-8")
    )
    return history, test_metrics


def main() -> None:
    for challenge, output_dir, benchmarks in (
        ("Q2", Q2_OUTPUT_DIR, PUBLISHED_BENCHMARKS_PATH),
        ("Q1", Q1_OUTPUT_DIR, PUBLISHED_Q1_BENCHMARKS_PATH),
    ):
        if not (output_dir / "metrics/dream_challenge_test_metrics.json").is_file():
            print(f"No {challenge} results in {output_dir}. Run ds-train first.")
            continue
        header = (output_dir / "metrics/validation_dream_challenge_metrics.csv").read_text(
            encoding="utf-8"
        )
        if "train_loss" not in header.splitlines()[0]:
            print(
                f"The {challenge} model was trained before the loss of each epoch was saved. Run ds-train again."
            )
            continue
        history, test_metrics = read_run(output_dir)
        reference = test_metrics.get("stored_reference_prediction_score")
        plots_dir = output_dir / "plots"
        save_training_plots(
            history,
            plots_dir,
            loss_label="Negative DREAM score (lower is better)",
            loss_title="Custom model training objective",
            challenge=challenge,
        )
        save_test_metrics_plot(test_metrics, plots_dir, challenge)
        save_challenge_comparison_plot(
            test_metrics,
            plots_dir,
            None if reference is None else {"dream_challenge_score": reference},
            benchmarks,
            challenge,
        )
        print(f"Saved the {challenge} figures to {plots_dir}")


if __name__ == "__main__":
    main()
