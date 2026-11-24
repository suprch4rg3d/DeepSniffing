"""Every figure: the curves of a training run, and the figures of an evaluation.

Each function takes the numbers and a path and draws one picture; none reads a model. ``python -m
scripts.core.plots`` draws the curves of a run from its ``history.csv``; ``report`` draws the evaluation figures.

    python -m scripts.core.plots --stage 1|2|3 [--tag NAME]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.core import classifier, generator, history, pretrain  # noqa: E402

# figure name -> (title, y label, the history columns it draws). A figure is drawn from the columns that the
# history has, so a run from before a column existed still gets its other figures.
CURVES = {
    "loss": (
        "Loss",
        "step 1 and 2: nats per molecule token; step 3: nats per molecule",
        ["train", "validation", "with_odor", "without_odor"],
    ),
    "accuracy": (
        "Accuracy of the next molecule token",
        "share of molecule tokens",
        ["train_accuracy", "validation_accuracy", "with_odor_accuracy", "without_odor_accuracy"],
    ),
    "request_gain": (
        "What the odor tokens add",
        "loss without the odor tokens minus loss with them",
        ["request_gain"],
    ),
    "roc_auc": (
        "Naming the odor tokens of a molecule",
        "macro over the odor tokens (chance is 0.5 for ROC-AUC)",
        ["roc_auc", "average_precision"],
    ),
    "precision": (
        "Share of the best-scored odor tokens that are true",
        "precision",
        ["precision_at_5", "precision_at_own_count"],
    ),
}


def training_curves(rows: pd.DataFrame, folder: Path) -> list[Path]:
    """One picture for each entry of ``CURVES`` that the history has a column for, in ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (title, label, columns) in CURVES.items():
        present = [c for c in columns if c in rows and rows[c].notna().any()]
        if not present:
            continue
        figure, axis = plt.subplots(figsize=(7, 4))
        for column in present:
            axis.plot(rows["epoch"], rows[column], label=column.replace("_", " "))
        if name == "loss" and "validation" in rows:
            best = int(rows["validation"].idxmin())
            axis.axvline(rows["epoch"][best], color="grey", linestyle=":", label="best epoch")
        axis.set(title=title, xlabel="epoch", ylabel=label)
        axis.legend()
        axis.grid(alpha=0.3)
        path = folder / f"{name}.png"
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)
    return written


def gain_histogram(own: list[float], none: list[float], path: Path, title: str) -> None:
    """How much each molecule gains from its own odor tokens: the loss without them minus the loss with them."""
    gains = np.asarray(none) - np.asarray(own)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(gains, bins=40)
    axis.axvline(0, color="black", linewidth=1)
    axis.axvline(gains.mean(), color="red", linestyle="--", label=f"mean {gains.mean():+.3f}")
    axis.set(
        title=title,
        xlabel="gain from the odor tokens (nats per molecule token)",
        ylabel="molecules",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def generation_bars(generation: dict, path: Path, title: str) -> None:
    """The valid, unique and novel shares of what the model wrote for each request."""
    requests = list(generation)
    shares = ("valid", "unique", "novel")
    width = 0.8 / len(shares)
    figure, axis = plt.subplots(figsize=(8, 4))
    for offset, share in enumerate(shares):
        axis.bar(
            np.arange(len(requests)) + offset * width,
            [generation[request][share] for request in requests],
            width,
            label=share,
        )
    axis.set_xticks(np.arange(len(requests)) + width, requests, rotation=30, ha="right")
    axis.set(title=title, ylabel="share of what was written", ylim=(0, 1.05))
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw the curves of a training run.")
    parser.add_argument("--stage", choices=("1", "2", "3"), default="2")
    parser.add_argument("--tag", help="a run kept apart with --tag")
    arguments = parser.parse_args()
    folder = {
        "1": pretrain.folder,
        "2": lambda: generator.folder(arguments.tag),
        "3": lambda: classifier.folder(arguments.tag),
    }[arguments.stage]()
    for path in training_curves(history.load(folder), folder / "plots"):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
