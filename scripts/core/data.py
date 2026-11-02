"""Canonical data loading for the DREAM olfaction task.

This module owns the data contract. Everything a training script needs comes
from :func:`prepare_data`:

* molecules and their features, taken from the processed DREAM and Pyrfume
  tables plus the Morgan fingerprints
* the 21 target attributes as a mean and a standard deviation, derived from the
  raw high intensity ratings exactly as the challenge scores them
* the canonical split, which holds the official DREAM test compounds out
* the fitted feature and target scalers, plus the SMILES tokenizer

Nothing here reads the ``split`` column of the CSV. The split is recomputed from
the identifiers, so a stale CSV can never silently change what a model trains
on, and the tests check that the two agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from scripts.core.metrics import TARGET_COLUMNS, TARGETS, normalize_target_name
from scripts.core.splits import official_test_cids, ordered_split_cids
from scripts.preprocessing.smiles_tokenizer import SmilesTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data/processed/dream_pyrfume"
MOLECULES_PATH = PROCESSED_DIR / "molecules_enriched.csv"
SPLIT_TABLE_PATH = PROCESSED_DIR / "molecule_labels_with_split.csv"
FINGERPRINTS_PATH = PROCESSED_DIR / "morgan_fingerprints.csv"
RATINGS_DIR = PROJECT_ROOT / "data/raw/dream-olfaction-prediction/data"
TRAIN_RATINGS_PATH = RATINGS_DIR / "TrainSet.txt"
TEST_RATINGS_PATH = RATINGS_DIR / "TestSet.txt"
REFERENCE_PREDICTIONS_PATH = (
    PROJECT_ROOT / "results/baseline/dream_challenge/challenge_2_testset_1446435104.txt"
)
PUBLISHED_BENCHMARKS_PATH = (
    PROJECT_ROOT / "results/baseline/dream_challenge/published_challenge_scores.json"
)

MAX_SEQUENCE_LENGTH = 128
FINGERPRINT_SIZE = 2048
FINGERPRINT_RADIUS = 2
OFFICIAL_TEST_SIZE = 69

METADATA_COLUMNS = {"CID", "Odor", "Replicate", "Intensity", "Dilution", "subject #"}

ENRICHED_FEATURE_COLUMNS = [
    "rdkit_atom_count_c",
    "rdkit_atom_count_h",
    "rdkit_atom_count_n",
    "rdkit_atom_count_o",
    "rdkit_atom_count_p",
    "rdkit_atom_count_s",
    "rdkit_atom_count_na",
    "rdkit_atom_count_cl",
    "rdkit_atom_count_i",
    "rdkit_heavy_atom_count",
    "rdkit_aromatic_ring_count",
    "rdkit_aliphatic_ring_count",
    "rdkit_rotatable_bond_count",
    "rdkit_hydrogen_bond_donor_count",
    "rdkit_hydrogen_bond_acceptor_count",
    "rdkit_topological_polar_surface_area",
    "rdkit_mol_logp",
]


@dataclass(frozen=True)
class DatasetBundle:
    """Everything needed to train and score one model."""

    arrays: dict[str, tuple[torch.Tensor, ...]]
    raw_targets: dict[str, np.ndarray]
    train_cids: list[int]
    validation_cids: list[int]
    test_cids: list[int]
    tokenizer: SmilesTokenizer
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    features: list[str]
    fingerprint_columns: list[str]
    target_columns: list[str]

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer.token_to_id)

    @property
    def feature_count(self) -> int:
        return len(self.features)

    @property
    def fingerprint_count(self) -> int:
        return len(self.fingerprint_columns)

    @property
    def target_count(self) -> int:
        return len(self.target_columns)

    @property
    def train_size(self) -> int:
        return len(self.arrays["train"][0])

    @property
    def validation_size(self) -> int:
        return len(self.arrays["validation"][0])

    @property
    def test_size(self) -> int:
        return len(self.arrays["test"][0])


def aggregate_high_intensity_targets(path: Path) -> pd.DataFrame:
    """Aggregate the official challenge targets from a raw rating file.

    The challenge predicts the mean and the spread of the ratings that each
    compound received at high intensity. Ratings are averaged per participant
    first, so a participant who rated a compound more often does not dominate
    the result, and then averaged across participants.
    """
    if not path.is_file():
        raise FileNotFoundError(f"DREAM rating file does not exist: {path}")

    ratings = pd.read_csv(path, sep="\t").rename(columns={"Compound Identifier": "CID"})
    required = {"CID", "Intensity", *TARGETS}
    normalized_columns = {
        column: normalize_target_name(column)
        for column in ratings.columns
        if column not in METADATA_COLUMNS
    }
    ratings = ratings.rename(columns=normalized_columns)
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(f"DREAM rating file is missing columns: {', '.join(sorted(missing))}")

    high = ratings.loc[ratings["Intensity"].astype(str).str.strip().str.lower() == "high"].copy()
    if high.empty:
        raise ValueError(f"No high intensity ratings found in {path}")
    high["CID"] = pd.to_numeric(high["CID"], errors="raise").astype("int64")
    for target in TARGETS:
        high[target] = pd.to_numeric(high[target], errors="coerce")

    if "subject #" not in high.columns:
        raise ValueError(f"DREAM rating file has no subject identifiers: {path}")
    subject_means = high.groupby(["CID", "subject #"])[TARGETS].mean()
    means = subject_means.groupby(level="CID").mean()
    deviations = subject_means.groupby(level="CID").std(ddof=1)
    summary = pd.concat([means.add_prefix("mean_"), deviations.add_prefix("std_")], axis=1)
    ordered_columns = [f"mean_{target}" for target in TARGETS] + [
        f"std_{target}" for target in TARGETS
    ]
    return summary[ordered_columns].reset_index()


def load_molecules() -> pd.DataFrame:
    """Load the enriched molecule table joined with Morgan fingerprints."""
    for path in (MOLECULES_PATH, FINGERPRINTS_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required processed input is missing: {path}")

    molecules = pd.read_csv(MOLECULES_PATH)
    fingerprints = pd.read_csv(FINGERPRINTS_PATH)
    features = ENRICHED_FEATURE_COLUMNS

    if "CID" not in fingerprints.columns:
        raise ValueError(f"Morgan fingerprint file has no CID column: {FINGERPRINTS_PATH}")
    fingerprint_columns = [
        column for column in fingerprints.columns if column.startswith("morgan_")
    ]
    expected_fingerprint_columns = [f"morgan_{index:04d}" for index in range(FINGERPRINT_SIZE)]
    if fingerprint_columns != expected_fingerprint_columns:
        raise ValueError(
            "Morgan fingerprint columns must be morgan_0000 through morgan_2047 in "
            f"{FINGERPRINTS_PATH}"
        )
    if fingerprints["CID"].isna().any() or fingerprints["CID"].duplicated().any():
        raise ValueError("Morgan fingerprints must have one non-empty row per CID.")
    fingerprints["CID"] = pd.to_numeric(fingerprints["CID"], errors="raise").astype("int64")
    fingerprint_values = fingerprints[fingerprint_columns].to_numpy()
    if not np.isin(fingerprint_values, (0, 1)).all():
        raise ValueError("Morgan fingerprint columns must contain only binary values.")

    required_molecules = {"CID", "rdkit_canonical_smiles", *features}
    missing = required_molecules.difference(molecules.columns)
    if missing:
        raise ValueError(f"Enriched molecule data is missing: {', '.join(sorted(missing))}")
    if molecules["CID"].duplicated().any():
        raise ValueError("Enriched molecule data must contain one row per CID.")
    molecules["CID"] = pd.to_numeric(molecules["CID"], errors="raise").astype("int64")
    if set(molecules["CID"]) != set(fingerprints["CID"]):
        raise ValueError("Morgan fingerprint CIDs do not match the enriched molecule CIDs.")
    return molecules.merge(fingerprints, on="CID", how="left", validate="one_to_one")


def prepare_data() -> DatasetBundle:
    """Assemble the training, validation, and official test tensors."""
    molecules = load_molecules()
    molecule_by_cid = molecules.set_index("CID")

    train_targets = aggregate_high_intensity_targets(TRAIN_RATINGS_PATH)
    test_targets = aggregate_high_intensity_targets(TEST_RATINGS_PATH)
    if not set(train_targets["CID"]).issubset(molecule_by_cid.index):
        raise ValueError("Some official DREAM training CIDs have no molecular features.")
    if not set(test_targets["CID"]).issubset(molecule_by_cid.index):
        raise ValueError("Some official DREAM test CIDs have no molecular features.")
    if not set(test_targets["CID"]).issubset(set(official_test_cids())):
        raise ValueError("Only official DREAM test compounds may be used for testing.")

    train_cids, validation_cids = ordered_split_cids(train_targets["CID"].to_numpy())
    train_targets = train_targets.set_index("CID")
    test_targets = test_targets.set_index("CID")

    frames = {}
    for name, cids in (("train", train_cids), ("validation", validation_cids)):
        subset = train_targets.loc[cids].reset_index()
        subset = subset.merge(molecules, on="CID", how="left", validate="one_to_one")
        subset = subset.dropna(subset=TARGET_COLUMNS)
        if subset.empty:
            raise ValueError(f"The {name} split has no labelled molecules.")
        frames[name] = subset

    test = test_targets.loc[sorted(test_targets.index)].reset_index()
    test = test.merge(molecules, on="CID", how="left", validate="one_to_one")
    test = test.dropna(subset=TARGET_COLUMNS)
    if len(test) != OFFICIAL_TEST_SIZE:
        raise ValueError(
            f"Expected the official {OFFICIAL_TEST_SIZE} molecule test set; found {len(test)}."
        )
    frames["test"] = test

    features = ENRICHED_FEATURE_COLUMNS.copy()
    train = frames["train"]
    feature_scaler = StandardScaler().fit(train[features].to_numpy(dtype=np.float64))
    target_scaler = StandardScaler().fit(train[TARGET_COLUMNS].to_numpy(dtype=np.float64))

    tokenizer = SmilesTokenizer()
    train_smiles = train["rdkit_canonical_smiles"].fillna("").astype(str).tolist()
    if not train_smiles or any(not value.strip() for value in train_smiles):
        raise ValueError("Training split has missing SMILES or no rows.")
    tokenizer.fit(train_smiles)

    fingerprint_columns = [f"morgan_{index:04d}" for index in range(FINGERPRINT_SIZE)]

    def encode_rows(rows: pd.DataFrame, with_targets: bool) -> tuple[torch.Tensor, ...]:
        smiles = rows["rdkit_canonical_smiles"].fillna("").astype(str).tolist()
        if any(not value.strip() for value in smiles):
            raise ValueError("A DREAM split contains missing canonical SMILES.")
        tokens = np.asarray(
            [tokenizer.encode(value, MAX_SEQUENCE_LENGTH) for value in smiles], dtype=np.int64
        )
        scaled_features = feature_scaler.transform(rows[features].to_numpy(dtype=np.float64))
        fingerprint_values = rows[fingerprint_columns].to_numpy(dtype=np.float32)
        tensors = (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(scaled_features, dtype=torch.float32),
            torch.tensor(fingerprint_values, dtype=torch.float32),
        )
        if with_targets:
            scaled_targets = target_scaler.transform(
                rows[TARGET_COLUMNS].to_numpy(dtype=np.float64)
            )
            tensors += (torch.tensor(scaled_targets, dtype=torch.float32),)
        return tensors

    arrays = {
        "train": encode_rows(frames["train"], True),
        "validation": encode_rows(frames["validation"], True),
        "test": encode_rows(frames["test"], False),
    }
    raw_targets = {
        "validation": frames["validation"][TARGET_COLUMNS].to_numpy(dtype=np.float64),
        "test": frames["test"][TARGET_COLUMNS].to_numpy(dtype=np.float64),
    }
    test_cids = frames["test"]["CID"].astype(int).tolist()

    return DatasetBundle(
        arrays=arrays,
        raw_targets=raw_targets,
        train_cids=frames["train"]["CID"].astype(int).tolist(),
        validation_cids=frames["validation"]["CID"].astype(int).tolist(),
        test_cids=test_cids,
        tokenizer=tokenizer,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        features=features,
        fingerprint_columns=fingerprint_columns,
        target_columns=TARGET_COLUMNS,
    )
