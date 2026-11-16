"""The targets of the two subchallenges, and the rows that each stage of a run learns from.

Subchallenge 2: intensity at one dilution, and the curve that gives the spread of a rating from its mean.
Subchallenge 1: each participant's rating smoothed towards the people who agree with them."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from scripts.core.data.features import GraphInputs
from scripts.core.data.ratings import (
    STANDARD_LOG10_DILUTION,
    SUBJECT_IDS,
    aggregate_high_intensity_targets,
    aggregate_individual_ratings,
    read_official_leaderboard_targets,
)
from scripts.core.metrics import TARGET_COLUMNS, TARGETS
from scripts.core.splits import (
    LEADERBOARD_RATINGS_PATH,
    TEST_RATINGS_PATH,
    Split,
)

Q1_TARGET_COLUMNS = [
    f"q1_{target}_subject_{subject}" for target in TARGETS for subject in SUBJECT_IDS
]


def intensity_output_columns(challenge: str) -> list[int]:
    """The outputs of intensity: its 49 participants in Q1, its mean and spread in Q2."""
    if challenge == "q1":
        return list(range(len(SUBJECT_IDS)))
    return [0, len(TARGETS)]


def restrict_intensity_to_standard_dilution(
    mask: np.ndarray, numeric: np.ndarray, columns: list[int] | None = None
) -> np.ndarray:
    """Keep the intensity ratings that were given at the standard dilution, drop the others.

    ``mask`` marks the ratings that exist, ``columns`` are the intensity outputs (the first 49
    by default), and ``numeric`` ends with the log10 concentration and the high concentration
    flag of every row.
    """
    columns = list(range(len(SUBJECT_IDS))) if columns is None else columns
    at_standard = np.abs(numeric[:, -2] - STANDARD_LOG10_DILUTION) < 0.5
    restricted = mask.copy()
    restricted[:, columns] &= at_standard[:, None]
    return restricted


# Subchallenge 2 predicts the spread of the ratings next to their mean. On a scale of 0 to 100 the
# spread follows the mean closely, large for middle ratings and small near either end. A curve of
# the predicted mean is mixed into the predicted spread. These are the fixed weights of the curve in
# that mix, one for each attribute, in the order of the attributes: the curve explains the spread of
# intensity and of most descriptors, and it explains that of pleasantness badly, so its weight is
# small. A weight of 0 leaves the spread as it is.
STD_FROM_MEAN_WEIGHTS = (
    1.0,
    0.03,
    1.0,
    0.62,
    0.79,
    0.97,
    0.68,
    0.88,
    1.0,
    0.76,
    0.91,
    0.5,
    1.0,
    0.35,
    0.67,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.38,
)


def std_curve_values(mean: np.ndarray, k0: np.ndarray, k1: np.ndarray) -> np.ndarray:
    """The spread that a rating with this mean has, on the 0 to 100 scale.

    It is zero at both ends and largest in the middle, and ``k0`` and ``k1`` set its height and
    its skew. One curve is fitted for every attribute.
    """
    x = np.clip(np.asarray(mean, dtype=np.float64), 0.0, 100.0) / 100.0
    return 100.0 * (k0 * x ** (k1 * 0.5) - k0 * x ** (k1 * 2.0))


def fit_std_curve(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fit the curve of the spread on the mean for each of the 21 attributes.

    The fit uses the rows of the given molecules, which are the training molecules only: the high
    concentration rows for every attribute, and for intensity the rows at 1/1000, because that is
    the dilution of the intensity that is scored. Also returns the average mean of every attribute,
    ``center``, which is where predictions are shrunk to.
    """
    high = rows.loc[rows["condition"].eq("high")]
    at_standard = high
    if "log10_concentration" in rows:
        near = (rows["log10_concentration"] - STANDARD_LOG10_DILUTION).abs() < 0.5
        at_standard = rows.loc[near] if near.any() else high
    k0, k1, center = [], [], []
    for position, target in enumerate(TARGETS):
        used = at_standard if position == 0 else high
        mean = pd.to_numeric(used[f"mean_{target}"], errors="coerce").to_numpy(dtype=np.float64)
        spread = pd.to_numeric(used[f"std_{target}"], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(mean) & np.isfinite(spread)
        result = minimize(
            lambda k, m=mean[valid], s=spread[valid]: float(
                np.sum((std_curve_values(m, *k) - s) ** 2)
            ),
            [1.0, 1.0],
            method="L-BFGS-B",
            bounds=[(1e-3, 10.0), (0.05, 10.0)],
        )
        k0.append(result.x[0])
        k1.append(result.x[1])
        center.append(float(np.mean(mean[valid])))
    return {"k0": np.array(k0), "k1": np.array(k1), "center": np.array(center)}


def fit_mean_shrinkage(predicted: np.ndarray, observed: np.ndarray) -> float:
    """How much to pull predictions towards their centre so that they match what happens.

    This is the slope of the observed means on the predicted means across molecules, shared by the
    given attributes, and it lies between 0.1 and 1. A prediction with a correlation of 0.75 to the
    truth has a slope well below 1 when it is as spread as the truth.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    predicted = predicted - predicted.mean(axis=0)
    observed = observed - observed.mean(axis=0)
    variance = float(np.sum(predicted**2))
    if variance <= 0.0:
        return 1.0
    return float(np.clip(np.sum(predicted * observed) / variance, 0.1, 1.0))


def apply_std_curve(
    predictions: np.ndarray,
    curve: dict[str, np.ndarray],
    weight: float | np.ndarray | tuple[float, ...] | None = None,
) -> np.ndarray:
    """Mix the curve of the predicted mean into the predicted spread of Subchallenge 2.

    ``predictions`` hold the 21 means and then the 21 spreads of every molecule. The new spread
    is ``weight`` times the curve of the mean plus the rest of the predicted spread. The weight is
    one number, or one for each of the 21 attributes, and it is the ``weight`` of the curve when
    none is given, and otherwise ``STD_FROM_MEAN_WEIGHTS``. When the curve has a ``shrink``, the means
    are pulled that far towards their ``center`` (and moved by ``offset``, one number for each
    attribute, when the curve has one) before the curve is read, because the curve is
    steepest at the ends and a mean that is a little off there gives a spread that is far off. The
    predicted means themselves are not changed.
    """
    count = len(TARGETS)
    mixed = np.array(predictions, dtype=np.float64, copy=True)
    if weight is None:
        weight = curve.get("weight", STD_FROM_MEAN_WEIGHTS)
    weight = np.asarray(weight, dtype=np.float64)
    means = mixed[:, :count]
    if "shrink" in curve and "center" in curve:
        means = curve["center"] + curve["shrink"] * (means - curve["center"])
    if "offset" in curve:
        means = means + curve["offset"]
    from_mean = std_curve_values(means, curve["k0"], curve["k1"])
    mixed[:, count : 2 * count] = weight * from_mean + (1.0 - weight) * mixed[:, count : 2 * count]
    return mixed


# The individual ratings are smoothed towards the people who perceive odors like the participant
# does, and not towards everyone: every other participant's rating is weighted by how well the two
# participants agree across the training ratings of that attribute. The smoothing of Subchallenge 1
# uses that mean in place of the plain population mean.
def similar_participant_means(
    rows: pd.DataFrame,
    targets: tuple[str, ...] = tuple(TARGETS),
    subjects: tuple[int, ...] = tuple(SUBJECT_IDS),
) -> dict[str, np.ndarray]:
    """For every row and participant, the mean of the other participants' ratings, weighted by
    how much they agree with that participant.

    The agreement of two participants is their correlation over the rows that both rated, for
    one attribute. A negative or undefined correlation gives no weight, and a participant never
    counts for themselves. The result for a participant and row is missing when nobody with a
    weight rated that row. Each array is shaped (rows, participants).
    """
    means = {}
    for target in targets:
        columns = [f"q1_{target}_subject_{subject}" for subject in subjects]
        ratings = rows[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        known = np.isfinite(ratings)
        agreement = pd.DataFrame(ratings).corr(min_periods=10).to_numpy()
        weights = np.clip(np.nan_to_num(agreement), 0.0, None)
        np.fill_diagonal(weights, 0.0)
        total = np.where(known, ratings, 0.0) @ weights
        norm = known.astype(np.float64) @ weights
        means[target] = np.where(norm > 1e-9, total / np.where(norm > 1e-9, norm, 1.0), np.nan)
    return means


def blend_q1_training_targets(rows: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Smooth every participant's rating towards the molecule's population mean.

    Each rating becomes ``alpha * individual + (1 - alpha) * population``. Only the
    training labels are blended. Validation and test are scored against the unblended
    individual ratings, so a larger alpha cannot win by changing what it is scored on.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("The population blend alpha must lie between 0 and 1.")
    blended = rows.copy()
    similar = similar_participant_means(rows)
    for target in TARGETS:
        population = pd.to_numeric(blended[f"mean_{target}"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        for position, subject in enumerate(SUBJECT_IDS):
            column = f"q1_{target}_subject_{subject}"
            individual = pd.to_numeric(blended[column], errors="coerce").to_numpy(dtype=np.float64)
            smoothing = population
            if similar is not None:
                own_people = similar[target][:, position]
                smoothing = np.where(np.isfinite(own_people), own_people, population)
            observed = np.isfinite(individual) & np.isfinite(smoothing)
            blended[column] = np.where(
                observed, alpha * individual + (1.0 - alpha) * smoothing, np.nan
            )
    return blended


@dataclass
class TrainingFrames:
    """The rows that a subchallenge learns from and is scored on, one frame for each stage."""

    target_columns: list[str]
    training: pd.DataFrame  # the 338 training molecules: epochs are chosen on this
    validation: pd.DataFrame  # the 69 leaderboard molecules, at high concentration only
    full_training: (
        pd.DataFrame
    )  # training and validation molecules: the final model learns from this
    official_test: pd.DataFrame  # the 69 official test molecules, at high concentration only


def training_frames(
    challenge: str, inputs: GraphInputs, split: Split, blend_alpha: float | None = None
) -> TrainingFrames:
    """Build the frames of a subchallenge from the rating rows and the split.

    ``ds-build`` and ``prepare_graph_data`` both call this, so the scalers are fitted on exactly the rows
    that the models then train on.
    """
    is_q1 = challenge == "q1"
    rows = inputs.rows
    validation_source = inputs.validation_rows
    in_training = rows["CID"].isin(split.train)
    in_validation = validation_source["CID"].isin(split.validation)
    # The final model also learns from the validation molecules, which carry real DREAM ratings.
    final_source = pd.concat([rows, validation_source], ignore_index=True)
    if is_q1:
        target_columns = list(Q1_TARGET_COLUMNS)
        training_rows = rows.loc[in_training].reset_index(drop=True)
        training = blend_q1_training_targets(training_rows, blend_alpha)
        validation = validation_source.loc[
            in_validation & validation_source["condition"].eq("high")
        ].reset_index(drop=True)
        full_rows = final_source.reset_index(drop=True)
        full_training = blend_q1_training_targets(full_rows, blend_alpha)
        official_test = inputs.test_rows.loc[inputs.test_rows["condition"].eq("high")].reset_index(
            drop=True
        )
    else:
        target_columns = list(TARGET_COLUMNS)
        training = rows.loc[in_training].dropna(subset=target_columns).reset_index(drop=True)
        validation = (
            validation_source.loc[in_validation & validation_source["condition"].eq("high")]
            .dropna(subset=target_columns)
            .reset_index(drop=True)
        )
        full_training = final_source.dropna(subset=target_columns).reset_index(drop=True)
        official_test = (
            inputs.test_rows.loc[inputs.test_rows["condition"].eq("high")]
            .dropna(subset=target_columns)
            .reset_index(drop=True)
        )
    if set(full_training["CID"]) & set(official_test["CID"]):
        raise ValueError("Official test molecules cannot be part of the final model's training.")
    print(
        f"{challenge.upper()}: training on {len(training)} rows from "
        f"{training['CID'].nunique()} molecules; validation and the official test stay "
        "high concentration only."
    )
    return TrainingFrames(target_columns, training, validation, full_training, official_test)


def official_targets_q1(
    validation: pd.DataFrame, official_test: pd.DataFrame
) -> tuple[list[int], list[int], dict[str, np.ndarray]]:
    """The validation and test molecules of Subchallenge 1 with each participant's own ratings."""
    official = {}
    for name, path in (("validation", LEADERBOARD_RATINGS_PATH), ("test", TEST_RATINGS_PATH)):
        cids, values = aggregate_individual_ratings(path)
        official[name] = (cids, values)
    available = set(official["validation"][0])
    validation_cids = sorted(set(validation["CID"]) & available)
    test_cids = sorted(set(official_test["CID"]))

    def pick(name: str, wanted: list[int]) -> np.ndarray:
        cids, values = official[name]
        position = {cid: index for index, cid in enumerate(cids)}
        return values[[position[cid] for cid in wanted]]

    raw_targets = {
        "validation": pick("validation", validation_cids),
        "test": pick("test", test_cids),
    }
    return validation_cids, test_cids, raw_targets


def official_targets_q2(
    validation: pd.DataFrame, official_test: pd.DataFrame, target_columns: list[str]
) -> tuple[list[int], list[int], dict[str, np.ndarray]]:
    """The validation and test molecules of Subchallenge 2 with their official means and spreads."""
    official_frames = {
        # The leaderboard answers are the organizers' own file. The test set has no such
        # file in the repository, so its answers are aggregated from TestSet.txt.
        "validation": read_official_leaderboard_targets().set_index("CID"),
        "test": aggregate_high_intensity_targets(TEST_RATINGS_PATH).set_index("CID"),
    }
    validation_cids = sorted(
        set(validation["CID"])
        & set(official_frames["validation"].dropna(subset=target_columns).index)
    )
    test_cids = sorted(set(official_test["CID"]))
    raw_targets = {
        "validation": official_frames["validation"].loc[validation_cids, target_columns].to_numpy(),
        "test": official_frames["test"].loc[test_cids, target_columns].to_numpy(),
    }
    return validation_cids, test_cids, raw_targets
