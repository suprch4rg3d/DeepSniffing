"""Scaling the inputs and the targets. Every scaler is fitted on training molecules only, by ``ds-build``, and
read back here."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from scripts.core.data.features import INTERACTION_COLUMNS, NAME_COLUMN_PREFIX, RDKIT_COLUMNS
from scripts.core.data.targets import TrainingFrames
from scripts.core.splits import (
    PROCESSED_DIR,
)
from scripts.preprocessing.community_features import (
    EPI_PREFIX,
    NAME_PREFIX,
    STRUCTURE_PREFIXES,
)
from scripts.preprocessing.feature_selection import keep_selected

FEATURE_SCALERS_PATH = PROCESSED_DIR / "scalers/feature_scalers.joblib"


TARGET_SCALERS_PATH = PROCESSED_DIR / "scalers/target_scalers.joblib"


# The concentration of each rating, and whether it is the high one, close the numeric block.
CONCENTRATION_COLUMNS = ("log10_concentration", "is_high_concentration")


def replace_infinite(values: np.ndarray) -> np.ndarray:
    """Treat infinite descriptor values as missing so that they are imputed."""
    return np.where(np.isinf(values), np.nan, values)


def numeric_scaler(cube_root: bool = False) -> Pipeline:
    """Impute with the median, optionally take the cube root, then standardise.

    Dragon descriptors are heavy tailed and partly undefined, so they get the cube
    root as well. Every step is fitted on training molecules only.
    """
    steps = [
        ("finite", FunctionTransformer(replace_infinite)),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if cube_root:
        steps.append(("cube_root", FunctionTransformer(np.cbrt)))
    steps.append(("standardise", StandardScaler()))
    return Pipeline(steps)


def name_feature_columns(frame: pd.DataFrame) -> list[str]:
    """The name window columns of a table: the windows built from the DREAM odor names first,
    then the organizers' windows."""
    columns = sorted(column for column in frame if column.startswith(NAME_COLUMN_PREFIX))
    return columns + sorted(column for column in frame if column.startswith(NAME_PREFIX))


def structure_feature_columns(frame: pd.DataFrame) -> list[str]:
    """The NSPDK and similarity columns of a table, family by family."""
    return [
        column
        for prefix in STRUCTURE_PREFIXES
        for column in sorted(name for name in frame if name.startswith(prefix))
    ]


def epi_columns(frame: pd.DataFrame) -> list[str]:
    """The EPI Suite property columns of a table, in a fixed order."""
    return sorted(column for column in frame if column.startswith(EPI_PREFIX))


def scaler_columns(molecules: pd.DataFrame, dragon_columns: list[str]) -> dict[str, list[str]]:
    """The columns that each scaler of the feature scaler takes: Dragon, RDKit, charges and EPI Suite."""
    columns = {
        "dragon": list(dragon_columns),
        "rdkit": keep_selected("rdkit", list(RDKIT_COLUMNS)),
        "interaction": keep_selected("charge", list(INTERACTION_COLUMNS)),
    }
    epi = epi_columns(molecules)
    if epi:
        columns["epi"] = epi
    return columns


def fit_feature_scaler(training: pd.DataFrame, dragon_columns: list[str]) -> dict[str, Pipeline]:
    """Fit the Dragon, RDKit, charge, and EPI Suite scalers on the unique training molecules."""
    molecules = training.drop_duplicates("CID")
    columns = scaler_columns(molecules, dragon_columns)
    scalers = {
        "dragon": numeric_scaler(cube_root=True),
        "rdkit": numeric_scaler(),
        "interaction": numeric_scaler(),
    }
    if "epi" in columns:
        scalers["epi"] = numeric_scaler(cube_root=True)
    for name, scaler in scalers.items():
        scaler.fit(molecules[columns[name]].to_numpy(dtype=np.float64))
        # A scaler keeps the columns it was fitted on, so that a saved model never depends
        # on the feature manifest at prediction time.
        scaler.columns_ = columns[name]

    return scalers


def transform_names(rows: pd.DataFrame) -> np.ndarray:
    """The name features of a set of rows: the windows as they are, with missing ones as 0."""
    return rows[name_feature_columns(rows)].fillna(0.0).to_numpy(dtype=np.float32)


def transform_structure(rows: pd.DataFrame) -> np.ndarray:
    """The structure features of a set of rows: NSPDK and similarity columns, missing ones as 0."""
    return rows[structure_feature_columns(rows)].fillna(0.0).to_numpy(dtype=np.float32)


def transform_features(
    rows: pd.DataFrame,
    scalers: dict[str, Pipeline],
    dragon_columns: list[str],
    morgan_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the scaled Dragon matrix and the numeric block for a set of rows.

    The numeric block holds the scaled RDKit and charge descriptors, then the scaled EPI
    Suite properties when they were fitted, then the raw Morgan bits, then the
    concentration inputs when they are requested.
    """
    dragon = scalers["dragon"].transform(rows[dragon_columns].to_numpy(dtype=np.float64))

    def scaled(name: str, default: list[str]) -> np.ndarray:
        scaler = scalers[name]
        columns = getattr(scaler, "columns_", default)
        return scaler.transform(rows[columns].to_numpy(dtype=np.float64))

    parts = [
        scaled("rdkit", list(RDKIT_COLUMNS)),
        scaled("interaction", list(INTERACTION_COLUMNS)),
    ]
    if "epi" in scalers:
        parts.append(scaled("epi", epi_columns(rows)))
    parts.append(rows[morgan_columns].to_numpy(dtype=np.float32))
    parts.append(rows[list(CONCENTRATION_COLUMNS)].to_numpy(dtype=np.float32))
    return dragon.astype(np.float32), np.concatenate(parts, axis=1).astype(np.float32)


def fit_masked_target_scaler(training: pd.DataFrame, columns: list[str]) -> StandardScaler:
    """Fit a target scaler on the ratings that exist, ignoring the missing ones.

    A column with no ratings, or no spread, is left unscaled rather than divided by zero.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        scaler = StandardScaler().fit(training[columns].to_numpy(dtype=np.float64))
    scale = scaler.scale_
    scaler.mean_ = np.nan_to_num(scaler.mean_, nan=0.0)
    scaler.scale_ = np.where(np.isfinite(scale) & (scale >= 1e-8), scale, 1.0)
    return scaler


def scale_masked_targets(
    frame: pd.DataFrame, columns: list[str], scaler: StandardScaler
) -> tuple[np.ndarray, np.ndarray]:
    """Scale targets and return the mask of ratings that exist; missing ones become 0."""
    values = frame[columns].to_numpy(dtype=np.float64)
    mask = np.isfinite(values)
    scaled = np.where(mask, scaler.transform(values), 0.0)
    return scaled.astype(np.float32), mask


def stage_cids(frames: TrainingFrames) -> dict[str, list[int]]:
    """The molecules that each stage fits its scalers on: the training molecules, and training plus validation."""
    return {
        "validation": sorted({int(cid) for cid in frames.training["CID"]}),
        "final": sorted({int(cid) for cid in frames.full_training["CID"]}),
    }


def load_scalers(
    challenge: str, frames: TrainingFrames, dragon_columns: list[str], blend_alpha: float | None
) -> tuple[dict[str, Pipeline], dict[str, Pipeline], StandardScaler, StandardScaler]:
    """The feature and target scalers of ``ds-build``, for the stage that chooses epochs and the final stage.

    A scaler fitted on other molecules, on other columns or with another blend weight than this run uses
    would silently change what the model sees, so any such file stops the run.
    """
    for path in (FEATURE_SCALERS_PATH, TARGET_SCALERS_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"The scalers are missing: {path}. Run ds-build.")
    features = joblib.load(FEATURE_SCALERS_PATH)
    targets = joblib.load(TARGET_SCALERS_PATH)[challenge]
    cids = stage_cids(frames)
    for stage in ("validation", "final"):
        if features["cids"][stage] != cids[stage] or targets["cids"][stage] != cids[stage]:
            raise ValueError(
                "The scalers were fitted on other molecules than this run uses. Run ds-build."
            )
        expected = scaler_columns(frames.training.drop_duplicates("CID"), dragon_columns)
        if {name: features[stage][name].columns_ for name in expected} != expected:
            raise ValueError(
                "The scalers were fitted on other columns than this run uses. Run ds-build."
            )
    if challenge == "q1" and targets["blend_alpha"] != blend_alpha:
        raise ValueError(
            "The Q1 target scalers were fitted with another blend weight. Run ds-build."
        )
    return features["validation"], features["final"], targets["validation"], targets["final"]
