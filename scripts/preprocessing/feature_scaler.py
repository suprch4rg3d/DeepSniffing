from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data/processed/dream_pyrfume/molecules_enriched_with_split.csv"
ARTIFACT_PATH = PROJECT_ROOT / "data/processed/dream_pyrfume/scalers/feature_scaler.joblib"

FEATURE_COLUMNS = [
    "rdkit_atom_count_c",
    "rdkit_atom_count_h",
    "rdkit_atom_count_n",
    "rdkit_atom_count_o",
    "rdkit_atom_count_s",
    "rdkit_heavy_atom_count",
    "rdkit_aromatic_ring_count",
    "rdkit_aliphatic_ring_count",
    "rdkit_rotatable_bond_count",
    "rdkit_hydrogen_bond_donor_count",
    "rdkit_hydrogen_bond_acceptor_count",
    "rdkit_topological_polar_surface_area",
    "rdkit_mol_logp",
]


def fit_and_save_feature_scaler(training_data: pd.DataFrame, artifact_path: Path) -> dict[str, Any]:
    if "split" in training_data and not training_data["split"].eq("train").all():
        raise ValueError("Fit the feature scaler using training rows only.")

    missing = set(FEATURE_COLUMNS).difference(training_data.columns)
    if missing:
        raise ValueError(f"Missing molecular features: {', '.join(sorted(missing))}")

    values = training_data[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Molecular features contain missing or non-finite values.")

    scaler = StandardScaler()
    scaler.fit(values)
    artifact = {"feature_columns": FEATURE_COLUMNS.copy(), "scaler": scaler}
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    return artifact


def transform_features(data: pd.DataFrame, artifact: dict[str, Any]) -> np.ndarray:
    feature_columns = artifact["feature_columns"]
    missing = set(feature_columns).difference(data.columns)
    if missing:
        raise ValueError(f"Missing molecular features: {', '.join(sorted(missing))}")

    values = data[feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Molecular features contain missing or non-finite values.")
    return artifact["scaler"].transform(values)


def load_feature_scaler(artifact_path: Path) -> dict[str, Any]:
    artifact = joblib.load(artifact_path)
    if set(artifact) != {"feature_columns", "scaler"}:
        raise ValueError("Molecular feature scaler file has an invalid format.")
    return artifact


def main() -> None:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Enriched split dataset not found: {DATASET_PATH}")

    dataset = pd.read_csv(DATASET_PATH)
    if "split" not in dataset.columns:
        raise ValueError("Enriched dataset is missing the 'split' column.")
    training_data = dataset.loc[dataset["split"] == "train"]
    if training_data.empty:
        raise ValueError("Enriched dataset has no training rows.")

    fit_and_save_feature_scaler(training_data, ARTIFACT_PATH)
    print(f"Saved standard feature scaler using {len(training_data)} training molecules")
    print(f"Scaler artifact: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
