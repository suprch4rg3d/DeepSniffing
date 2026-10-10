from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data/processed/dream_pyrfume"
BASE_DATASET_PATH = DATASET_DIR / "molecule_labels_with_split.csv"
ENRICHED_DATASET_PATH = DATASET_DIR / "molecules_enriched_with_split.csv"
OUTPUT_ROOT = DATASET_DIR / "inspection"
DREAM_TARGET_PREFIX = "dream_train_mean_"


def inspect_dataset() -> None:
    enriched_source_exists = (DATASET_DIR / "molecules_enriched.csv").is_file()
    enriched_split_exists = ENRICHED_DATASET_PATH.is_file()
    enriched = enriched_source_exists or enriched_split_exists
    dataset_path = ENRICHED_DATASET_PATH if enriched else BASE_DATASET_PATH
    output_dir = OUTPUT_ROOT / ("enriched" if enriched else "base")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Split dataset not found: {dataset_path}")

    dataset = pd.read_csv(dataset_path)
    required_columns = {"CID", "rdkit_canonical_smiles", "split"}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}"
        )
    if dataset["CID"].duplicated().any():
        raise ValueError("Expected one row per CID in the molecule-level dataset.")
    if not {"train", "validation", "test"}.issubset(set(dataset["split"].dropna())):
        raise ValueError("Dataset must contain train, validation, and test split assignments.")

    canonical_smiles = dataset["rdkit_canonical_smiles"].fillna("").astype(str).str.strip()
    dataset = dataset.loc[canonical_smiles.ne("")].copy()
    target_columns = [
        column for column in dataset.columns if column.startswith(DREAM_TARGET_PREFIX)
    ]
    if len(target_columns) != 21:
        raise ValueError(f"Expected 21 DREAM mean targets, found {len(target_columns)}.")

    train = dataset.loc[dataset["split"] == "train"]
    rdkit_feature_columns = []
    for column in dataset.columns:
        if column.startswith("rdkit_"):
            values = pd.to_numeric(dataset[column], errors="coerce")
            if values.notna().any():
                rdkit_feature_columns.append(column)

    output_dir.mkdir(parents=True, exist_ok=True)

    split_names = ("train", "validation", "test")
    split_counts = dataset["split"].value_counts().reindex(split_names, fill_value=0)

    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(
        split_counts.index, split_counts.values, color=("#386cb0", "#7fc97f", "#fdc086")
    )
    axis.set_title("Molecules in each split")
    axis.set_ylabel("Molecules")
    axis.bar_label(bars, fmt="%d")
    figure.tight_layout()
    figure.savefig(output_dir / "split_sizes.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    target_coverage = pd.DataFrame(
        {
            split_name: dataset.loc[dataset["split"] == split_name, target_columns]
            .notna()
            .mean()
            .mul(100)
            for split_name in split_names
        }
    )
    target_names = [
        column.removeprefix(DREAM_TARGET_PREFIX).replace("_", " ").title()
        for column in target_columns
    ]
    figure, axis = plt.subplots(figsize=(7, max(6, 0.32 * len(target_columns))))
    image = axis.imshow(target_coverage.to_numpy(), cmap="viridis", vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(range(len(split_names)), labels=[name.title() for name in split_names])
    axis.set_yticks(range(len(target_names)), labels=target_names)
    axis.set_title("Observed DREAM target ratings by split")
    figure.colorbar(image, ax=axis, label="Molecules with rating (%)")
    figure.tight_layout()
    figure.savefig(output_dir / "dream_target_coverage_by_split.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    column_count = 3
    row_count = (len(target_columns) + column_count - 1) // column_count
    figure, axes = plt.subplots(
        row_count, column_count, figsize=(15, 3.2 * row_count), squeeze=False
    )
    for index, column in enumerate(target_columns):
        axis = axes.flat[index]
        values = pd.to_numeric(train[column], errors="coerce").dropna()
        axis.hist(values, bins=20, color="#386cb0", edgecolor="white")
        axis.set_title(column.removeprefix(DREAM_TARGET_PREFIX).replace("_", " ").title())
        axis.set_xlabel("Mean DREAM rating")
        axis.set_ylabel("Molecules")
    for axis in axes.flat[len(target_columns) :]:
        axis.set_visible(False)
    figure.suptitle("Training split: DREAM target distributions", fontsize=16, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    histogram_path = output_dir / "train_dream_target_distributions.png"
    figure.savefig(histogram_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    if rdkit_feature_columns:
        feature_column_count = min(3, len(rdkit_feature_columns))
        feature_row_count = (
            len(rdkit_feature_columns) + feature_column_count - 1
        ) // feature_column_count
        figure, axes = plt.subplots(
            feature_row_count,
            feature_column_count,
            figsize=(5 * feature_column_count, 3.2 * feature_row_count),
            squeeze=False,
        )
        for index, column in enumerate(rdkit_feature_columns):
            numeric_values = pd.to_numeric(train[column], errors="coerce")
            values = numeric_values.dropna()
            axes.flat[index].hist(values, bins=20, color="#4daf4a", edgecolor="white")
            coverage = 100 * len(values) / len(train) if len(train) else 0
            axes.flat[index].set_title(
                f"{column.removeprefix('rdkit_').replace('_', ' ').title()}\n"
                f"Available for {coverage:.1f}% of train"
            )
            axes.flat[index].set_xlabel("RDKit descriptor value")
            axes.flat[index].set_ylabel("Molecules")
        for axis in axes.flat[len(rdkit_feature_columns) :]:
            axis.set_visible(False)
        figure.suptitle("Training split: numeric RDKit descriptor distributions", fontsize=15)
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        feature_histogram_path = output_dir / "train_rdkit_feature_distributions.png"
        figure.savefig(feature_histogram_path, dpi=160, bbox_inches="tight")
        plt.close(figure)

    correlations = train[target_columns].apply(pd.to_numeric, errors="coerce").corr()
    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(target_columns)))
    axis.set_yticks(range(len(target_columns)))
    axis.set_xticklabels(target_names, rotation=60, ha="right")
    axis.set_yticklabels(target_names)
    axis.set_title("Training split: DREAM target correlations")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    correlation_path = output_dir / "train_dream_target_correlation.png"
    figure.savefig(correlation_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved inspection files under {output_dir}")


def main() -> None:
    inspect_dataset()


if __name__ == "__main__":
    main()
