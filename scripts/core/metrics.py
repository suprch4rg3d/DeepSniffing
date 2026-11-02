"""Official DREAM Olfaction Challenge scoring.

The challenge is scored as a pseudo R squared, Q2, computed per attribute and
then z scored against the challenge null model. The mean of those z scores is
the single number used on the leaderboard, and it is what this module reports.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = [
    "intensity_strength",
    "valence_pleasantness",
    "bakery",
    "sweet",
    "fruit",
    "fish",
    "garlic",
    "spices",
    "cold",
    "sour",
    "burnt",
    "acid",
    "warm",
    "musky",
    "sweaty",
    "ammonia_urinous",
    "decayed",
    "wood",
    "grass",
    "flower",
    "chemical",
]
TARGET_COUNT = len(TARGETS)
TARGET_COLUMNS = [f"mean_{target}" for target in TARGETS] + [f"std_{target}" for target in TARGETS]

# Standard deviations of the challenge null model, taken from the official
# scoring script. Dividing a correlation by these values produces the z score.
OFFICIAL_NULL_SD = {
    "intensity_mean": 0.119307474,
    "valence_mean": 0.126471379,
    "descriptors_mean": 0.026512975,
    "intensity_std": 0.119494682,
    "valence_std": 0.11490659,
    "descriptors_std": 0.02808499,
}


def normalize_target_name(name: str) -> str:
    """Normalise a DREAM attribute label to the internal snake case form."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def pearson_across_molecules(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Pearson correlation across molecules for one attribute."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted arrays must have the same shape.")
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if len(actual) < 2 or np.std(actual) == 0 or np.std(predicted) == 0:
        return 0.0
    return float(np.corrcoef(actual, predicted)[0, 1])


def dream_challenge_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute the official Q2 z scores and the mean z score.

    Both arrays must hold 21 means followed by 21 standard deviations, in the
    order given by ``TARGETS``.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape or actual.shape[1] != TARGET_COUNT * 2:
        raise ValueError("DREAM Q2 scoring expects 21 means followed by 21 standard deviations.")

    correlations = {
        "intensity_mean": pearson_across_molecules(actual[:, 0], predicted[:, 0]),
        "valence_mean": pearson_across_molecules(actual[:, 1], predicted[:, 1]),
        "descriptors_mean": float(
            np.mean(
                [
                    pearson_across_molecules(actual[:, index], predicted[:, index])
                    for index in range(2, TARGET_COUNT)
                ]
            )
        ),
        "intensity_std": pearson_across_molecules(
            actual[:, TARGET_COUNT], predicted[:, TARGET_COUNT]
        ),
        "valence_std": pearson_across_molecules(
            actual[:, TARGET_COUNT + 1], predicted[:, TARGET_COUNT + 1]
        ),
        "descriptors_std": float(
            np.mean(
                [
                    pearson_across_molecules(
                        actual[:, TARGET_COUNT + index], predicted[:, TARGET_COUNT + index]
                    )
                    for index in range(2, TARGET_COUNT)
                ]
            )
        ),
    }
    z_scores = {name: correlations[name] / null_sd for name, null_sd in OFFICIAL_NULL_SD.items()}
    return {
        "pearson_correlations": correlations,
        "z_scores": z_scores,
        "dream_challenge_score": float(np.mean(list(z_scores.values()))),
    }


def prediction_matrix_from_submission(path: Path, cids: list[int]) -> np.ndarray:
    """Load an official DREAM submission file as a 21 + 21 prediction matrix."""
    submission = pd.read_csv(path, sep="\t")
    submission = submission.rename(columns={"#oID": "CID"})
    if not {"CID", "descriptor", "value", "sigma"}.issubset(submission.columns):
        raise ValueError(f"Unexpected DREAM prediction file format: {path}")
    submission["CID"] = pd.to_numeric(submission["CID"], errors="raise").astype("int64")
    submission["target"] = submission["descriptor"].astype(str).map(normalize_target_name)

    value = submission.pivot(index="CID", columns="target", values="value")
    sigma = submission.pivot(index="CID", columns="target", values="sigma")
    expected_cids = set(cids)
    if set(value.index) != expected_cids or set(sigma.index) != expected_cids:
        raise ValueError(f"Reference predictions do not match the official test CIDs: {path}")
    if set(value.columns) != set(TARGETS) or set(sigma.columns) != set(TARGETS):
        raise ValueError(f"Reference predictions do not contain exactly 21 DREAM targets: {path}")

    value = value.loc[cids, TARGETS].to_numpy(dtype=np.float64)
    sigma = sigma.loc[cids, TARGETS].to_numpy(dtype=np.float64)
    return np.concatenate((value, sigma), axis=1)


def write_metrics(path: Path, metrics: dict) -> None:
    """Write a metrics dictionary as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
