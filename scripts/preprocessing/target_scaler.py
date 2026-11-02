"""Standardise the molecule level DREAM target columns for inspection.

The training pipeline does not consume this artifact. It derives the official
challenge targets from the raw high intensity ratings and fits its own scaler on
the training split, because the DREAM score is defined on per subject means and
standard deviations rather than on the molecule level averages stored here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data/processed/dream_pyrfume"
BASE_DATASET_PATH = DATASET_DIR / "molecule_labels_with_split.csv"
ARTIFACT_PATH = PROJECT_ROOT / "data/processed/dream_pyrfume/scalers/target_scaler.joblib"
DREAM_TARGET_COLUMNS = [
    "dream_train_mean_intensity_strength",
    "dream_train_mean_valence_pleasantness",
    "dream_train_mean_bakery",
    "dream_train_mean_sweet",
    "dream_train_mean_fruit",
    "dream_train_mean_fish",
    "dream_train_mean_garlic",
    "dream_train_mean_spices",
    "dream_train_mean_cold",
    "dream_train_mean_sour",
    "dream_train_mean_burnt",
    "dream_train_mean_acid",
    "dream_train_mean_warm",
    "dream_train_mean_musky",
    "dream_train_mean_sweaty",
    "dream_train_mean_ammonia_urinous",
    "dream_train_mean_decayed",
    "dream_train_mean_wood",
    "dream_train_mean_grass",
    "dream_train_mean_flower",
    "dream_train_mean_chemical",
]


def fit_target_scaler(training_targets: pd.DataFrame, target_columns: list[str]) -> dict[str, Any]:
    missing = set(target_columns).difference(training_targets.columns)
    if missing:
        raise ValueError(f"Missing target columns: {', '.join(sorted(missing))}")
    if not target_columns:
        raise ValueError("At least one target column is required.")
    if len(set(target_columns)) != len(target_columns):
        raise ValueError("Target column names must be unique.")

    values = training_targets[target_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Targets contain missing or non-finite values.")

    scaler = StandardScaler()
    scaler.fit(values)
    return {"target_columns": target_columns.copy(), "scaler": scaler}


def transform_targets(targets: pd.DataFrame, artifact: dict[str, Any]) -> np.ndarray:
    target_columns = artifact["target_columns"]
    missing = set(target_columns).difference(targets.columns)
    if missing:
        raise ValueError(f"Missing target columns: {', '.join(sorted(missing))}")

    values = targets[target_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Targets contain missing or non-finite values.")
    return artifact["scaler"].transform(values)


def inverse_transform_targets(predictions: np.ndarray, artifact: dict[str, Any]) -> np.ndarray:
    return artifact["scaler"].inverse_transform(predictions)


def save_target_scaler(artifact: dict[str, Any], artifact_path: Path) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)


def load_target_scaler(artifact_path: Path) -> dict[str, Any]:
    return joblib.load(artifact_path)


def main() -> None:
    if not BASE_DATASET_PATH.is_file():
        raise FileNotFoundError(f"Base split dataset not found: {BASE_DATASET_PATH}")

    dataset = pd.read_csv(BASE_DATASET_PATH)
    if "split" not in dataset.columns:
        raise ValueError("Base dataset is missing the 'split' column.")
    missing_targets = set(DREAM_TARGET_COLUMNS).difference(dataset.columns)
    if missing_targets:
        raise ValueError(
            f"Base dataset is missing DREAM mean targets: {', '.join(sorted(missing_targets))}"
        )

    training_targets = dataset.loc[dataset["split"] == "train", DREAM_TARGET_COLUMNS].dropna(
        subset=DREAM_TARGET_COLUMNS
    )
    if training_targets.empty:
        raise ValueError("Training split has no complete DREAM target rows.")

    artifact = fit_target_scaler(training_targets, DREAM_TARGET_COLUMNS)
    save_target_scaler(artifact, ARTIFACT_PATH)
    print(f"Saved DREAM target scaler using {len(training_targets)} training molecules")
    print(f"Targets: {len(DREAM_TARGET_COLUMNS)}")
    print(f"Scaler artifact: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
