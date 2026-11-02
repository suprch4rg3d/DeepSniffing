"""Canonical train, validation, and test assignment for the DREAM task.

The DREAM challenge defines its own test set, so the test split is fixed and
never sampled. Everything else that has training ratings is split into train and
validation with a fixed seed. Molecules that never took part in the challenge are
marked as unused so they cannot leak into training by accident.

This module is the single source of truth for the split. The materialised CSV
columns are a view of it, and the tests check that the two always agree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RATINGS_DIR = PROJECT_ROOT / "data/raw/dream-olfaction-prediction/data"
TRAIN_RATINGS_PATH = RATINGS_DIR / "TrainSet.txt"
OFFICIAL_TEST_CIDS_PATH = RATINGS_DIR / "CID_testset.txt"

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
UNUSED = "unused"
SPLIT_NAMES = (TRAIN, VALIDATION, TEST, UNUSED)

VALIDATION_FRACTION = 0.15
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


def training_pool_cids() -> list[int]:
    """Compound identifiers that carry DREAM training ratings."""
    if not TRAIN_RATINGS_PATH.is_file():
        raise FileNotFoundError(f"DREAM training ratings are missing: {TRAIN_RATINGS_PATH}")
    ratings = pd.read_csv(TRAIN_RATINGS_PATH, sep="\t", usecols=["Compound Identifier"])
    cids = pd.to_numeric(ratings["Compound Identifier"], errors="raise").astype("int64")
    return sorted({int(cid) for cid in cids})


def ordered_split_cids(
    cids: list[int] | np.ndarray,
    *,
    validation_fraction: float = VALIDATION_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train and validation identifiers in shuffled order.

    Training order matters, because the data loader shuffles batches starting
    from the row order of the dataset. Keeping one implementation of the split
    means models trained by different scripts see identical batches.
    """
    usable = usable_training_cids(cids)
    if usable.size == 0:
        return np.array([], dtype="int64"), np.array([], dtype="int64")
    train_cids, validation_cids = train_test_split(
        usable,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
    )
    return train_cids, validation_cids


def usable_training_cids(cids: list[int] | np.ndarray) -> np.ndarray:
    """Identifiers that may be trained on, meaning rated and not held out."""
    molecules = sorted({int(cid) for cid in cids})
    test_cids = set(official_test_cids())
    pool_cids = set(training_pool_cids())
    return np.array(
        [cid for cid in molecules if cid in pool_cids and cid not in test_cids],
        dtype="int64",
    )


def assign_splits(
    cids: list[int] | np.ndarray,
    *,
    validation_fraction: float = VALIDATION_FRACTION,
    seed: int = RANDOM_SEED,
) -> dict[int, str]:
    """Assign every given compound to train, validation, test, or unused.

    The official test set is held out unconditionally. Train and validation are
    drawn from the molecules that have DREAM training ratings, using the sorted
    identifier order so the assignment is stable across machines.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")

    molecules = sorted({int(cid) for cid in cids})
    if not molecules:
        raise ValueError("No compound identifiers were supplied.")

    test_cids = set(official_test_cids())
    pool_cids = set(training_pool_cids())
    leaked = test_cids.intersection(pool_cids)
    if leaked:
        raise ValueError(
            "The official test set overlaps the training pool, which would leak test "
            f"molecules into training: {sorted(leaked)[:5]}"
        )

    train_cids, validation_cids = ordered_split_cids(
        molecules, validation_fraction=validation_fraction, seed=seed
    )

    assignment = {cid: UNUSED for cid in molecules}
    assignment.update({int(cid): TRAIN for cid in train_cids})
    assignment.update({int(cid): VALIDATION for cid in validation_cids})
    assignment.update({cid: TEST for cid in test_cids.intersection(molecules)})
    return assignment


def assign_split_column(molecules: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the molecule table with a split column attached."""
    if "CID" not in molecules.columns:
        raise ValueError("Molecule table has no CID column.")
    if molecules["CID"].isna().any():
        raise ValueError("Molecule table contains missing CIDs.")
    if molecules["CID"].duplicated().any():
        raise ValueError("Expected one row per CID before assigning splits.")

    cids = pd.to_numeric(molecules["CID"], errors="raise").astype("int64")
    assignment = assign_splits(cids.to_numpy())
    result = molecules.copy()
    result["CID"] = cids
    result["split"] = result["CID"].map(assignment)
    if result["split"].isna().any():
        raise RuntimeError("Failed to assign a split to every molecule.")
    return result
