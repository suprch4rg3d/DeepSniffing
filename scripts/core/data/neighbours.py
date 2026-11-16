"""Nearest-neighbour labels (Subchallenge 2): what the most similar training molecules smell like, as an extra
input of the model."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from scripts.core.data.ratings import STANDARD_LOG10_DILUTION
from scripts.core.metrics import TARGETS
from scripts.preprocessing.community_features import (
    MORGAN_SIMILARITY_PATH,
    NSPDK_CIDS_PATH,
    NSPDK_VECTORS_PATH,
)

# Subchallenge 2 gets an extra input token: the similarity weighted average of the labels of the
# most similar other molecules of the training set, 21 numbers, so the prediction can lean on what
# the most similar known molecules smell like. A molecule never sees its own label.
NEIGHBOUR_POWER = 4


# The labels of the nearest molecules are read twice, by two similarities. The NSPDK graph kernel of
# De Grave finds neighbours that predict the descriptors better than Morgan does (Z 19.8 against 16.4
# on the validation molecules), and Morgan is a little better for intensity.
NEIGHBOUR_SIMILARITIES = ("morgan", "nspdk")


NEIGHBOUR_COUNT = 20


def load_dream_similarity() -> pd.DataFrame:
    """Morgan similarity of the DREAM molecules to each other, from the challenge data."""
    table = pd.read_csv(MORGAN_SIMILARITY_PATH, index_col=0)
    table.index = table.index.astype("int64")
    table.columns = table.columns.astype("int64")
    return table.loc[:, table.columns.isin(table.index)]


@functools.lru_cache(maxsize=1)
def load_dream_nspdk_similarity() -> pd.DataFrame:
    """NSPDK similarity of the DREAM molecules to each other: the dot product of their vectors."""
    from sklearn.datasets import load_svmlight_file

    vectors, _ = load_svmlight_file(str(NSPDK_VECTORS_PATH))
    cids = pd.read_csv(NSPDK_CIDS_PATH, header=None)[0].astype("int64").to_numpy()
    position = {int(cid): index for index, cid in enumerate(cids)}
    dream = [int(cid) for cid in load_dream_similarity().index]
    rows = [position[cid] for cid in dream]
    gram = (vectors[rows] @ vectors[rows].T).toarray()
    return pd.DataFrame(gram, index=dream, columns=dream)


def dream_similarity_matrices(
    names: tuple[str, ...] = NEIGHBOUR_SIMILARITIES,
) -> list[pd.DataFrame]:
    """The similarity matrices that the neighbour token reads, in the order of ``names``."""
    loaders = {"morgan": load_dream_similarity, "nspdk": load_dream_nspdk_similarity}
    return [loaders[name]() for name in names]


def neighbour_labels(rows: pd.DataFrame, intensity_at_standard: bool) -> pd.DataFrame:
    """The 21 mean ratings of every molecule in ``rows``, one row for each molecule.

    They are the mean over the high concentration rows. The intensity of a training molecule is
    the one at 1/1000, which is the intensity that is scored. The validation rows already give
    that intensity, so they are used as they are.
    """
    high = rows.loc[rows["condition"].eq("high")]
    labels = high.groupby("CID")[[f"mean_{target}" for target in TARGETS]].mean()
    labels.columns = list(TARGETS)
    if intensity_at_standard:
        near = (rows["log10_concentration"] - STANDARD_LOG10_DILUTION).abs() < 0.5
        intensity = rows.loc[near].groupby("CID")[f"mean_{TARGETS[0]}"].mean()
        labels[TARGETS[0]] = intensity.reindex(labels.index)
    return labels


def neighbour_statistics(reference: pd.DataFrame) -> dict[str, np.ndarray]:
    """The average and spread of every label of the reference molecules."""
    values = reference.to_numpy(dtype=np.float64)
    spread = np.nanstd(values, axis=0)
    return {"mean": np.nanmean(values, axis=0), "scale": np.where(spread > 0, spread, 1.0)}


def neighbour_features(
    cids: np.ndarray,
    reference: pd.DataFrame,
    similarity: pd.DataFrame,
    statistics: dict[str, np.ndarray],
    power: int = NEIGHBOUR_POWER,
    count: int = NEIGHBOUR_COUNT,
) -> np.ndarray:
    """What the most similar reference molecules were rated, for every molecule in ``cids``.

    Each molecule gets the average of the labels of its ``count`` most similar reference
    molecules, weighted by the Morgan similarity to the power ``power``. A molecule is never its
    own neighbour, so a molecule of the reference does not see its own label. The result is
    standardised with the statistics of the reference. A label that no neighbour has is
    filled with the average.
    """
    unique = np.unique(np.asarray(cids, dtype=np.int64))
    weights = similarity.loc[unique, reference.index].to_numpy(dtype=np.float64) ** power
    weights = np.where(unique[:, None] == reference.index.to_numpy()[None, :], 0.0, weights)
    count = min(count, weights.shape[1])
    cutoff = -np.sort(-weights, axis=1)[:, count - 1 : count]
    weights = np.where(weights >= cutoff, weights, 0.0)
    values = reference.to_numpy(dtype=np.float64)
    known = np.isfinite(values)
    numerator = weights @ np.where(known, values, 0.0)
    denominator = weights @ known.astype(np.float64)
    mixed = np.where(
        denominator > 0, numerator / np.where(denominator > 0, denominator, 1.0), statistics["mean"]
    )
    standardised = (mixed - statistics["mean"]) / statistics["scale"]
    position = {int(cid): index for index, cid in enumerate(unique)}
    return standardised[[position[int(cid)] for cid in cids]].astype(np.float32)
