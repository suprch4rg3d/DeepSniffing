"""Canonical train, validation, and test assignment for the DREAM task.

The DREAM challenge defines three molecule sets, and the split follows them: the
338 TrainSet molecules are the training split, the 69 leaderboard molecules are the
validation split, and the 69 official test molecules are the test split, which is never
sampled and never used for training or epoch selection. Molecules without ratings
that took part in none of these sets are marked as unused so they cannot leak into
training by accident.

This module is the single source of truth for the split. ``ds-build`` writes it to
``split.csv`` with ``build_split``, and ``ds-train`` reads it back with ``load_split``, which
checks the file against the DREAM lists again, so a stale or edited file is never trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data/processed/dream_pyrfume"
RATINGS_DIR = PROJECT_ROOT / "data/raw/dream-olfaction-prediction/data"
TRAIN_RATINGS_PATH = RATINGS_DIR / "TrainSet.txt"
TEST_RATINGS_PATH = RATINGS_DIR / "TestSet.txt"
LEADERBOARD_RATINGS_PATH = RATINGS_DIR / "LBs1.txt"
OFFICIAL_TEST_CIDS_PATH = RATINGS_DIR / "CID_testset.txt"
LEADERBOARD_CIDS_PATH = RATINGS_DIR / "CID_leaderboard.txt"
SPLIT_PATH = PROCESSED_DIR / "split.csv"

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
UNUSED = "unused"
SPLIT_NAMES = (TRAIN, VALIDATION, TEST, UNUSED)

RANDOM_SEED = 42


def read_cid_list(path: Path) -> list[int]:
    """Read a whitespace separated list of compound identifiers."""
    if not path.is_file():
        raise FileNotFoundError(f"Compound identifier list does not exist: {path}")
    values = {int(token) for token in path.read_text(encoding="utf-8").split()}
    if not values:
        raise ValueError(f"Compound identifier list is empty: {path}")
    return sorted(values)


def official_test_cids() -> list[int]:
    """Compound identifiers of the official DREAM test set."""
    return read_cid_list(OFFICIAL_TEST_CIDS_PATH)


def leaderboard_cids() -> list[int]:
    """Compound identifiers of the DREAM leaderboard set, used here for validation."""
    return read_cid_list(LEADERBOARD_CIDS_PATH)


def training_pool_cids() -> list[int]:
    """Compound identifiers that carry DREAM training ratings."""
    if not TRAIN_RATINGS_PATH.is_file():
        raise FileNotFoundError(f"DREAM training ratings are missing: {TRAIN_RATINGS_PATH}")
    ratings = pd.read_csv(TRAIN_RATINGS_PATH, sep="\t", usecols=["Compound Identifier"])
    cids = pd.to_numeric(ratings["Compound Identifier"], errors="raise").astype("int64")
    return sorted({int(cid) for cid in cids})


def usable_training_cids(cids: list[int] | np.ndarray) -> np.ndarray:
    """Identifiers that may be trained on, meaning rated and not held out."""
    molecules = sorted({int(cid) for cid in cids})
    test_cids = set(official_test_cids()) | set(leaderboard_cids())
    pool_cids = set(training_pool_cids())
    return np.array(
        [cid for cid in molecules if cid in pool_cids and cid not in test_cids],
        dtype="int64",
    )


def assign_splits(cids: list[int] | np.ndarray) -> dict[int, str]:
    """Assign every given compound to train, validation, test, or unused.

    The official test set is held out unconditionally. Train is every molecule with
    DREAM training ratings, and validation is the DREAM leaderboard set. The assignment
    does not depend on the order of the identifiers, so it is stable across machines.
    """
    molecules = sorted({int(cid) for cid in cids})
    if not molecules:
        raise ValueError("No compound identifiers were supplied.")

    test_cids = set(official_test_cids())
    validation_cids = set(leaderboard_cids())
    pool_cids = set(training_pool_cids())
    for name, left, right in (
        ("training pool and the official test set", pool_cids, test_cids),
        ("leaderboard set and the official test set", validation_cids, test_cids),
        ("training pool and the leaderboard set", pool_cids, validation_cids),
    ):
        leaked = left.intersection(right)
        if leaked:
            raise ValueError(
                f"The {name} overlap, which would leak held-out molecules: {sorted(leaked)[:5]}"
            )

    assignment = {cid: UNUSED for cid in molecules}
    assignment.update({cid: TRAIN for cid in pool_cids.intersection(molecules)})
    assignment.update({cid: VALIDATION for cid in validation_cids.intersection(molecules)})
    assignment.update({cid: TEST for cid in test_cids.intersection(molecules)})
    return assignment


def build_split(cids: list[int] | np.ndarray, *, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """The split table: ``CID``, ``split`` and, for training molecules, ``train_order``.

    The training order matters, because the data loader shuffles batches starting from the row
    order of the dataset. It is one fixed permutation of the training molecules, so every model
    trained from this table sees identical batches.
    """
    assignment = assign_splits(cids)
    order = np.random.RandomState(seed).permutation(usable_training_cids(list(assignment)))
    table = pd.DataFrame({"CID": sorted(assignment)})
    table["split"] = table["CID"].map(assignment)
    table["train_order"] = table["CID"].map(
        {int(cid): position for position, cid in enumerate(order)}
    )
    return table


@dataclass(frozen=True)
class Split:
    """The molecules of each split, as ``split.csv`` lists them. Training is in its fixed order."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def load_split(path: Path = SPLIT_PATH) -> Split:
    """Read ``split.csv`` and check it against the DREAM lists, so a stale file is never used."""
    if not path.is_file():
        raise FileNotFoundError(f"The split is missing: {path}. Run ds-build.")
    table = pd.read_csv(path)
    if not {"CID", "split", "train_order"}.issubset(table.columns):
        raise ValueError(f"{path.name} needs the columns CID, split and train_order. Run ds-build.")
    if table["CID"].duplicated().any():
        raise ValueError(f"{path.name} lists a molecule twice.")
    unknown = set(table["split"]) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"{path.name} has unknown split labels: {sorted(unknown)}")

    members = {name: set(table.loc[table["split"] == name, "CID"]) for name in SPLIT_NAMES}
    for name, expected in (
        (TRAIN, training_pool_cids()),
        (VALIDATION, leaderboard_cids()),
        (TEST, official_test_cids()),
    ):
        if members[name] != set(expected):
            raise ValueError(f"The {name} split in {path.name} is not the DREAM set. Run ds-build.")
    ordered = table.loc[table["split"] == TRAIN].sort_values("train_order")
    if ordered["train_order"].isna().any() or list(ordered["train_order"]) != list(
        range(len(ordered))
    ):
        raise ValueError(f"The training order in {path.name} is not 0 to {len(ordered) - 1}.")
    return Split(
        train=ordered["CID"].to_numpy(dtype="int64"),
        validation=np.array(sorted(members[VALIDATION]), dtype="int64"),
        test=np.array(sorted(members[TEST]), dtype="int64"),
    )
