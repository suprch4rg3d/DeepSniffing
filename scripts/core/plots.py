"""Every figure: the curves of a training run, and the figures of an evaluation.

Each function takes the numbers and a path and draws one picture; none reads a model. ``python -m
scripts.core.plots`` draws the curves of a run from its ``history.csv``; ``report`` draws the evaluation figures.

    python -m scripts.core.plots --stage 1|2|3 [--tag NAME]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Colab exports an inline backend that the environment's matplotlib does not have, and matplotlib refuses to
# import at all when MPLBACKEND names it. Every figure here is written to a file, so Agg is right everywhere.
os.environ["MPLBACKEND"] = "Agg"

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


def odor_map_figure(
    before: np.ndarray, after: np.ndarray, tokens: list[str], positions: list[int], path: Path
) -> None:
    """The same 890 odor tokens in two dimensions before and after step 2, the forty most frequent named."""
    figure, axes = plt.subplots(2, 1, figsize=(11, 12))
    for ax, coordinates, title in zip(
        axes, (before, after), ("before step 2", "after step 2"), strict=True
    ):
        span = coordinates.max(axis=0) - coordinates.min(axis=0)
        middle = coordinates.mean(axis=0)
        ax.scatter(coordinates[:, 0], coordinates[:, 1], s=16, color="#B4C7E7", alpha=0.85)
        for name, position in zip(tokens, positions, strict=True):
            point = coordinates[position]
            direction = point - middle
            length = float(np.linalg.norm(direction))
            away = direction / length if length else np.array([0.0, 1.0])
            ax.annotate(
                name,
                point,
                xytext=point + away * 0.10 * span,
                fontsize=7.5,
                color="#1F3864",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.8},
                arrowprops={"arrowstyle": "-", "color": "#8FAADC", "lw": 0.6},
            )
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    figure.suptitle("The 890 odor-token embeddings, the forty most frequent named", fontsize=13)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def odor_tsne_figure(
    before: np.ndarray, after: np.ndarray, tokens: list[str], positions: list[int], path: Path
) -> None:
    """The same tokens spread by t-SNE, which separates the neighbourhoods PCA keeps in one blob."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, coordinates, title in zip(
        axes, (before, after), ("before step 2", "after step 2"), strict=True
    ):
        span = coordinates.max(axis=0) - coordinates.min(axis=0)
        middle = coordinates.mean(axis=0)
        ax.scatter(coordinates[:, 0], coordinates[:, 1], s=40, color="#B4C7E7", alpha=0.85)
        for name, position in zip(tokens, positions, strict=True):
            point = coordinates[position]
            direction = point - middle
            length = float(np.linalg.norm(direction))
            away = direction / length if length else np.array([0.0, 1.0])
            ax.annotate(
                name,
                point,
                xytext=point + away * 0.09 * span,
                fontsize=9,
                color="#1F3864",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85},
                arrowprops={"arrowstyle": "-", "color": "#8FAADC", "lw": 0.7},
            )
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    figure.suptitle(
        "The 890 odor-token embeddings by t-SNE, the forty most frequent named", fontsize=14
    )
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def request_norms(values: dict, path: Path) -> None:
    """The mean norm of the scale and the shift at each layer, beside a typical hidden state."""
    layers = np.arange(1, len(values["scale"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(layers, values["scale"], marker="o", label="scale")
    axes[0].plot(layers, values["shift"], marker="o", label="shift")
    axes[0].plot(layers, values["hidden"], marker="o", linestyle="--", label="typical hidden state")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("norm, log scale")
    axes[0].legend(frameon=False)
    axes[1].plot(layers, values["scale_over_hidden"], marker="o", label="scale / hidden")
    axes[1].plot(layers, values["shift_over_hidden"], marker="o", label="shift / hidden")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("share of a typical hidden state")
    axes[1].legend(frameon=False)
    figure.suptitle("What the request adds at each layer")
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def request_heatmap(norms: np.ndarray, tokens: list[str], path: Path) -> None:
    """The norm of each layer's shift for the thirty most frequent odor tokens."""
    figure, ax = plt.subplots(figsize=(6, 8))
    image = ax.imshow(norms, aspect="auto", cmap="viridis")
    ax.set_xticks(range(norms.shape[1]), [str(i + 1) for i in range(norms.shape[1])])
    ax.set_yticks(range(len(tokens)), tokens, fontsize=8)
    ax.set_xlabel("layer")
    figure.colorbar(image, ax=ax, label="norm of the shift")
    ax.set_title("Where each of the thirty most frequent tokens pushes")
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def odor_overlap_figure(bins: list[dict], path: Path) -> None:
    """How similar the embeddings are as the molecules two tokens share go up, one bar per overlap band."""
    figure, ax = plt.subplots(figsize=(7, 4))
    labels = [f"{found['label']}\n{found['pairs']:,} pairs" for found in bins]
    values = [found["mean_cosine"] for found in bins]
    bars = ax.bar(range(len(bins)), values, color="#1F3864", width=0.65)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
    ax.set_xticks(range(len(bins)), labels, fontsize=8)
    ax.set_ylim(0.94, max(values) + 0.004)
    ax.set_xlabel("Jaccard overlap of the molecules the two tokens label")
    ax.set_ylabel("mean cosine between the embeddings")
    ax.set_title("Shared molecules against embedding closeness (scale starts at 0.94)")
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def odor_heatmap(similarity: np.ndarray, tokens: list[str], path: Path) -> None:
    """Cosine between the forty most frequent odor tokens, in an order that puts similar ones together."""
    figure, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(similarity, cmap="RdBu_r", vmin=0.90, vmax=1.0)
    ax.set_xticks(range(len(tokens)), tokens, rotation=90, fontsize=7)
    ax.set_yticks(range(len(tokens)), tokens, fontsize=7)
    ax.set_xticks(np.arange(-0.5, len(tokens)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tokens)), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    figure.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label="cosine")
    ax.set_title(
        "Cosine between the forty most frequent odor tokens, after step 2 (scale starts at 0.90)",
        fontsize=11,
    )
    figure.savefig(path, dpi=200, bbox_inches="tight")
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
