"""The DREAM rating files, read exactly as the challenge scores them: per participant, per molecule, per
dilution. Also where the published reference scores are."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.core.metrics import SUBJECT_COUNT, TARGET_COLUMNS, TARGETS, normalize_target_name
from scripts.core.splits import (
    PROJECT_ROOT,
    RATINGS_DIR,
)

LEADERBOARD_TARGETS_PATH = RATINGS_DIR / "LBs2.txt"


LEADERBOARD_DILUTIONS_PATH = RATINGS_DIR / "dilution_leaderboard.txt"


REFERENCE_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data/reference/dream_challenge/challenge_2_testset_1446435104.txt"
)


PUBLISHED_BENCHMARKS_PATH = (
    PROJECT_ROOT / "data/reference/dream_challenge/published_challenge_scores.json"
)


PUBLISHED_Q1_BENCHMARKS_PATH = (
    PROJECT_ROOT / "data/reference/dream_challenge/published_challenge_scores_q1.json"
)


METADATA_COLUMNS = {"CID", "Odor", "Replicate", "Intensity", "Dilution", "subject #"}


SUBJECT_IDS = tuple(range(1, SUBJECT_COUNT + 1))


# Every molecule was rated at two dilutions, and 280 of the 338 training molecules have a rating
# at 1/1000. Intensity is learned from those ratings only, so that the intensity of all the
# molecules is taken at the same physical strength, and it is predicted at that dilution too.
STANDARD_LOG10_DILUTION = -3.0


def read_rating_table(path: Path) -> pd.DataFrame:
    """Read a raw DREAM rating file as one row per rating event, like ``TrainSet.txt``.

    The leaderboard ratings (``LBs1.txt``) are published in a long format with one
    row per molecule, participant, and attribute, and only for the high concentration.
    They are pivoted into the TrainSet layout, with the high concentration dilution of
    each molecule taken from ``dilution_leaderboard.txt``, so every function that
    reads ratings treats all three DREAM sets the same way.
    """
    if not path.is_file():
        raise FileNotFoundError(f"DREAM rating file does not exist: {path}")
    table = pd.read_csv(path, sep="\t")
    if "#oID" not in table.columns:
        return table

    if not LEADERBOARD_DILUTIONS_PATH.is_file():
        raise FileNotFoundError(f"Leaderboard dilutions are missing: {LEADERBOARD_DILUTIONS_PATH}")
    dilutions = pd.read_csv(LEADERBOARD_DILUTIONS_PATH, sep="\t")
    dilution_by_cid = dict(zip(dilutions.iloc[:, 0].astype(int), dilutions.iloc[:, 1], strict=True))

    wide = table.pivot(index=["#oID", "individual"], columns="descriptor", values="value")
    wide.columns.name = None
    wide = wide.reset_index().rename(
        columns={"#oID": "Compound Identifier", "individual": "subject #"}
    )
    wide["Compound Identifier"] = wide["Compound Identifier"].astype("int64")
    if not set(wide["Compound Identifier"]).issubset(dilution_by_cid):
        raise ValueError(f"{LEADERBOARD_DILUTIONS_PATH.name} does not cover every rated molecule.")
    wide["Odor"] = ""
    wide["Replicate"] = ""
    wide["Intensity"] = "high"
    wide["Dilution"] = wide["Compound Identifier"].map(dilution_by_cid)
    return wide


def read_high_intensity_ratings(path: Path) -> pd.DataFrame:
    """Read a raw rating file and keep the high intensity ratings of the 21 attributes."""
    ratings = read_rating_table(path).rename(columns={"Compound Identifier": "CID"})
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
    return high


def aggregate_high_intensity_targets(path: Path) -> pd.DataFrame:
    """Aggregate the official challenge targets from a raw rating file.

    The challenge predicts the mean and the spread of the ratings that each
    compound received at high intensity. Ratings are averaged per participant
    first, so a participant who rated a compound more often does not dominate
    the result, and then averaged across participants.
    """
    high = read_high_intensity_ratings(path)
    subject_means = high.groupby(["CID", "subject #"])[TARGETS].mean()
    means = subject_means.groupby(level="CID").mean()
    deviations = subject_means.groupby(level="CID").std(ddof=1)
    summary = pd.concat([means.add_prefix("mean_"), deviations.add_prefix("std_")], axis=1)
    ordered_columns = [f"mean_{target}" for target in TARGETS] + [
        f"std_{target}" for target in TARGETS
    ]
    return summary[ordered_columns].reset_index()


def read_official_leaderboard_targets(path: Path = LEADERBOARD_TARGETS_PATH) -> pd.DataFrame:
    """Read the organizers' own leaderboard targets (``LBs2.txt``): one row per molecule.

    The file lists the population mean (``value``) and spread (``sigma``) of every molecule
    and attribute. They are returned in the layout of ``aggregate_high_intensity_targets``,
    with the ``mean_*`` and ``std_*`` columns, so Subchallenge 2 is validated against the
    published answers and not against a recomputation of them.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Official leaderboard targets do not exist: {path}")
    table = pd.read_csv(path, sep="\t")
    missing = {"#oID", "descriptor", "value", "sigma"}.difference(table.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
    table["descriptor"] = table["descriptor"].map(normalize_target_name)
    means = table.pivot(index="#oID", columns="descriptor", values="value")
    deviations = table.pivot(index="#oID", columns="descriptor", values="sigma")
    absent = set(TARGETS).difference(means.columns)
    if absent:
        raise ValueError(f"{path.name} has no values for: {', '.join(sorted(absent))}")
    summary = pd.concat(
        [means[TARGETS].add_prefix("mean_"), deviations[TARGETS].add_prefix("std_")], axis=1
    )
    summary.index.name = "CID"
    return summary[TARGET_COLUMNS].reset_index().astype({"CID": "int64"})


def aggregate_individual_ratings(path: Path) -> tuple[list[int], np.ndarray]:
    """Return every participant's own high intensity rating of each compound.

    Subchallenge 1 predicts individual ratings. Ratings are averaged per participant
    first, like the population targets, and a rating that a participant never gave is
    NaN. The array is shaped (compounds, 21 attributes, 49 participants), with the
    compounds in ascending identifier order.
    """
    high = read_high_intensity_ratings(path)
    high["subject #"] = pd.to_numeric(high["subject #"], errors="coerce")
    high = high.loc[high["subject #"].isin(SUBJECT_IDS)].copy()
    high["subject #"] = high["subject #"].astype(int)

    subject_means = high.groupby(["CID", "subject #"])[TARGETS].mean()
    ratings = subject_means.unstack("subject #").reindex(
        columns=pd.MultiIndex.from_product([TARGETS, SUBJECT_IDS])
    )
    cids = [int(cid) for cid in ratings.index]
    values = ratings.to_numpy(dtype=np.float64).reshape(len(cids), len(TARGETS), len(SUBJECT_IDS))
    return cids, values
