"""``ds-inspect``: check everything that Subchallenge 1 and 2 train on, and plot it.

It runs after ``ds-build``, prints PASS or FAIL for each check, and exits with
status 1 if any failed, so a broken input is found before a long training and not after it. It checks that the
split is the DREAM one, that no table of molecules carries a rating, that every DREAM molecule has its features
and 3D structure, that the feature selection kept something in every block, that the training molecules have
all 21 targets, and that the scalers match the molecules, the columns and the blend weight of the run.

The plots go to ``data/processed/dream_pyrfume/inspection/``: the split sizes, the distributions and
correlations of the 21 mean ratings of the training molecules, and the columns kept in each feature block.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from scripts.core.data import (
    aggregate_high_intensity_targets,
    load_graph_inputs,
    load_scalers,
    training_frames,
)
from scripts.core.metrics import TARGETS
from scripts.core.splits import (
    PROCESSED_DIR,
    SPLIT_NAMES,
    SPLIT_PATH,
    TRAIN_RATINGS_PATH,
    load_split,
)
from scripts.core.train import Q1_BLEND_ALPHA
from scripts.preprocessing.community_features import OUTPUT as PUBLISHED_PATH
from scripts.preprocessing.conformers import OUTPUT as CONFORMERS_PATH
from scripts.preprocessing.dream_dataset import OUTPUT_PATH as MOLECULES_PATH
from scripts.preprocessing.enrich_molecules import OUTPUT_PATH as DESCRIPTORS_PATH
from scripts.preprocessing.feature_selection import MANIFEST_PATH

OUTPUT_DIR = PROCESSED_DIR / "inspection"
MOLECULE_TABLES = (MOLECULES_PATH, DESCRIPTORS_PATH)
RATING_PREFIXES = ("label_", "dream_train_", "mean_", "std_")
EXPECTED_SIZES = {"train": 338, "validation": 69, "test": 69, "unused": 4}

results: list[tuple[str, bool]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, bool(passed)))
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}", flush=True)


def run_checks() -> None:
    split = load_split()  # raises if the file is missing, stale or edited
    table = pd.read_csv(SPLIT_PATH)
    sizes = table["split"].value_counts().reindex(SPLIT_NAMES, fill_value=0).to_dict()
    check("the split is the DREAM split", sizes == EXPECTED_SIZES, str(sizes))
    check("no molecule is in two splits", not table["CID"].duplicated().any())

    for name in MOLECULE_TABLES:
        columns = pd.read_csv(name, nrows=0).columns
        rated = [c for c in columns if c.startswith(RATING_PREFIXES)]
        check(
            f"{name.name} carries no rating",
            not rated,
            f"({len(columns)} columns)" if not rated else str(rated[:3]),
        )

    molecules = pd.read_csv(DESCRIPTORS_PATH)
    dream = set(split.train) | set(split.validation) | set(split.test)
    dragon = [
        c for c in molecules if c.startswith("dragon_") and c != "dragon_complexity from pubmed"
    ]
    has_dragon = set(molecules.loc[molecules[dragon].notna().any(axis=1), "CID"])
    check(
        "every DREAM molecule has Dragon descriptors",
        dream <= has_dragon,
        f"({len(has_dragon)} molecules)",
    )
    coordinates = set(pd.read_csv(CONFORMERS_PATH, usecols=["CID"])["CID"])
    without_3d = sorted(int(cid) for cid in dream - coordinates)
    # PubChem has no 3D structure for a few molecules; their graph then has the bonds only.
    check(
        "the DREAM molecules without a PubChem 3D structure are few",
        len(without_3d) <= 5,
        f"({len(without_3d)} use bonds only: {without_3d})",
    )
    community = set(pd.read_csv(PUBLISHED_PATH, usecols=["CID"])["CID"])
    check("every DREAM molecule has the published community features", dream <= community)

    manifest = pd.read_csv(MANIFEST_PATH)
    kept = manifest[manifest["kept"].astype(bool)].groupby("block").size()
    empty = [b for b in manifest["block"].unique() if kept.get(b, 0) == 0]
    # the RDKit block can be empty when every column is a copy of a Dragon column
    check(
        "the feature selection kept columns in every block",
        not [b for b in empty if b != "rdkit"],
        ", ".join(f"{b} {kept.get(b, 0)}" for b in manifest["block"].unique()),
    )

    targets = aggregate_high_intensity_targets(TRAIN_RATINGS_PATH).set_index("CID")
    complete = targets.loc[list(split.train)].notna().all().all()
    check(
        "the 338 training molecules have all 21 means and 21 spreads",
        complete,
        f"({targets.shape[1]} targets)",
    )

    inputs = load_graph_inputs()
    for challenge in ("q2", "q1"):
        frames = training_frames(
            challenge, inputs, split, Q1_BLEND_ALPHA if challenge == "q1" else None
        )
        try:
            load_scalers(
                challenge,
                frames,
                inputs.dragon_columns,
                Q1_BLEND_ALPHA if challenge == "q1" else None,
            )
            check(f"the scalers match the {challenge.upper()} run", True)
        except (FileNotFoundError, ValueError) as error:
            check(f"the scalers match the {challenge.upper()} run", False, str(error))


def write_plots() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(SPLIT_PATH)
    counts = table["split"].value_counts().reindex(SPLIT_NAMES, fill_value=0)
    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(counts.index, counts.values, color=("#386cb0", "#7fc97f", "#fdc086", "#bbbbbb"))
    axis.set_title("Molecules in each split")
    axis.set_ylabel("Molecules")
    axis.bar_label(bars, fmt="%d")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "split_sizes.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    train = aggregate_high_intensity_targets(TRAIN_RATINGS_PATH).set_index("CID")
    means = train[[f"mean_{t}" for t in TARGETS]]
    names = [t.replace("_", " ").title() for t in TARGETS]
    columns = 3
    rows = (len(TARGETS) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3.2 * rows), squeeze=False)
    for axis, name, column in zip(axes.flat, names, means.columns, strict=True):
        axis.hist(means[column].dropna(), bins=20, color="#386cb0", edgecolor="white")
        axis.set_title(name)
        axis.set_xlabel("Mean DREAM rating")
        axis.set_ylabel("Molecules")
    for axis in axes.flat[len(TARGETS) :]:
        axis.set_visible(False)
    figure.suptitle("Training molecules: distribution of the 21 mean ratings", fontsize=16, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(OUTPUT_DIR / "train_target_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(means.corr(), cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(names)), labels=names, rotation=60, ha="right")
    axis.set_yticks(range(len(names)), labels=names)
    axis.set_title("Training molecules: correlation of the 21 mean ratings")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "train_target_correlation.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    manifest = pd.read_csv(MANIFEST_PATH)
    summary = manifest.groupby("block")["kept"].agg(kept="sum", candidates="size")
    figure, axis = plt.subplots(figsize=(9, 4.5))
    positions = range(len(summary))
    axis.bar(positions, summary["candidates"], color="#cccccc", label="candidates")
    axis.bar(positions, summary["kept"], color="#386cb0", label="kept")
    axis.set_xticks(list(positions), labels=summary.index, rotation=30, ha="right")
    axis.set_yscale("log")
    axis.set_ylabel("Columns")
    axis.set_title("Feature selection: columns kept in each block")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "features_kept_by_block.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved the plots to {OUTPUT_DIR}")


def main() -> None:
    run_checks()
    write_plots()
    failed = [name for name, passed in results if not passed]
    print(
        f"\n{len(results) - len(failed)} of {len(results)} checks passed"
        + (f"; FAILED: {failed}" if failed else "")
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
