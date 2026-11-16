"""Assembling the data of a run: ``prepare_graph_data`` returns the loaders, the scalers and the official
targets of one subchallenge."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from scripts.core.data.features import GraphInputs, load_graph_inputs
from scripts.core.data.graphs import MoleculeRows, collate_molecules
from scripts.core.data.neighbours import (
    NEIGHBOUR_COUNT,
    NEIGHBOUR_POWER,
    NEIGHBOUR_SIMILARITIES,
    dream_similarity_matrices,
    neighbour_features,
    neighbour_labels,
    neighbour_statistics,
)
from scripts.core.data.scaling import (
    load_scalers,
    scale_masked_targets,
    transform_features,
    transform_names,
    transform_structure,
)
from scripts.core.data.targets import (
    fit_std_curve,
    intensity_output_columns,
    official_targets_q1,
    official_targets_q2,
    restrict_intensity_to_standard_dilution,
    training_frames,
)
from scripts.core.splits import (
    load_split,
)

GRAPH_BATCH_SIZE = 32


OFFICIAL_TEST_SIZE = 69


@dataclass
class GraphData:
    """Loaders, scalers, and evaluation targets for one challenge of the graph model.

    Mirrors ``DatasetBundle``. The ``validation_*`` scalers are fitted on the training
    split and are used to select the epoch. The plain ``feature_scaler`` and
    ``target_scaler`` are fitted on the molecules of the final model, which are the
    training and the validation molecules together, and serve the official test set.
    ``raw_targets`` hold the official targets, in ``validation_cids`` and ``test_cids``
    order, exactly as the shared bundle defines them: 42 values per molecule for
    ``q2``, and 21 attributes by 49 participants per molecule for ``q1``.
    """

    train_loader: DataLoader
    validation_loader: DataLoader
    full_loader: DataLoader
    test_loader: DataLoader
    validation_feature_scaler: dict[str, Pipeline]
    validation_target_scaler: StandardScaler
    feature_scaler: dict[str, Pipeline]
    target_scaler: StandardScaler
    raw_targets: dict[str, np.ndarray]
    dragon_columns: list[str]
    dragon_groups: dict[str, list[str]]
    morgan_columns: list[str]
    name_vocabulary: list[str]
    numeric_count: int
    name_count: int
    structure_count: int
    train_cids: list[int]
    validation_cids: list[int]
    full_cids: list[int]
    test_cids: list[int]
    target_columns: list[str]
    challenge: str
    blend_alpha: float | None
    std_curve: dict[str, np.ndarray] | None = None
    neighbours: dict | None = None

    @property
    def target_count(self) -> int:
        return len(self.target_columns)


def prepare_graph_data(
    challenge: str = "q2",
    batch_size: int = GRAPH_BATCH_SIZE,
    blend_alpha: float | None = None,
    inputs: GraphInputs | None = None,
) -> GraphData:
    """Build the training, validation, final model, and official test loaders.

    ``q2`` predicts the 21 means and standard deviations. ``q1`` predicts every
    participant's own rating and needs ``blend_alpha`` to smooth its training labels. Both
    train on both concentrations and are given the dilution as an input, because the
    ratings depend on it. Validation and test are scored at high concentration only.

    The split is the canonical one from ``scripts.core.splits``. The epoch is selected by training on the 338 official
    TrainSet molecules and validating on the 69 DREAM leaderboard molecules. The final
    model is then trained from scratch on all 407 of them for that many epochs. No official
    test molecule is ever used for training or for epoch selection. Validation and test are
    scored against the official per molecule targets, whatever the replicates in the
    training rows.
    """
    if challenge not in ("q1", "q2"):
        raise ValueError(f"Unknown challenge {challenge!r}; expected 'q1' or 'q2'.")
    is_q1 = challenge == "q1"
    if is_q1 and blend_alpha is None:
        raise ValueError("Subchallenge 1 needs a population blend alpha.")
    inputs = inputs or load_graph_inputs()
    rows, dragon_columns = inputs.rows, inputs.dragon_columns

    # ``load_split`` has checked that the split is the DREAM one, so the test molecules cannot be in training.
    split = load_split()
    frames = training_frames(challenge, inputs, split, blend_alpha)
    target_columns = frames.target_columns
    training, validation = frames.training, frames.validation
    full_training, official_test = frames.full_training, frames.official_test
    in_training = rows["CID"].isin(split.train)

    # The official per molecule targets, which validation and test are scored against.
    if is_q1:
        validation_cids, test_cids, raw_targets = official_targets_q1(validation, official_test)
    else:
        validation_cids, test_cids, raw_targets = official_targets_q2(
            validation, official_test, target_columns
        )
    if len(test_cids) != OFFICIAL_TEST_SIZE:
        raise ValueError(
            f"Expected the official {OFFICIAL_TEST_SIZE} molecule test set; found {len(test_cids)}."
        )

    std_curve = None if is_q1 else fit_std_curve(rows.loc[in_training])
    similarities = reference_train = reference_full = None
    statistics_train = statistics_full = None
    if not is_q1:
        similarities = dream_similarity_matrices()
        reference_train = neighbour_labels(rows.loc[in_training], True)
        reference_full = pd.concat(
            [reference_train, neighbour_labels(inputs.validation_rows, False)]
        )
        statistics_train = neighbour_statistics(reference_train)
        statistics_full = neighbour_statistics(reference_full)
    validation_feature_scaler, feature_scaler, validation_target_scaler, target_scaler = (
        load_scalers(challenge, frames, dragon_columns, blend_alpha)
    )

    def loader(frame, feature_scalers, scaler, shuffle: bool, final: bool = False) -> DataLoader:
        neighbours = None
        if reference_train is not None:
            neighbours = np.hstack(
                [
                    neighbour_features(
                        frame["CID"].to_numpy(),
                        reference_full if final else reference_train,
                        similarity,
                        statistics_full if final else statistics_train,
                        NEIGHBOUR_POWER,
                        NEIGHBOUR_COUNT,
                    )
                    for similarity in similarities
                ]
            )
        dragon, numeric = transform_features(
            frame,
            feature_scalers,
            dragon_columns,
            inputs.morgan_columns,
        )
        targets, mask = scale_masked_targets(frame, target_columns, scaler)
        mask = restrict_intensity_to_standard_dilution(
            mask, numeric, intensity_output_columns(challenge)
        )
        return DataLoader(
            MoleculeRows(
                frame,
                dragon,
                numeric,
                targets,
                mask,
                transform_names(frame),
                transform_structure(frame),
                neighbours,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_molecules,
        )

    train_loader = loader(training, validation_feature_scaler, validation_target_scaler, True)
    return GraphData(
        train_loader=train_loader,
        validation_loader=loader(
            validation, validation_feature_scaler, validation_target_scaler, False
        ),
        full_loader=loader(full_training, feature_scaler, target_scaler, True, True),
        test_loader=loader(official_test, feature_scaler, target_scaler, False, True),
        validation_feature_scaler=validation_feature_scaler,
        validation_target_scaler=validation_target_scaler,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        raw_targets=raw_targets,
        dragon_columns=dragon_columns,
        dragon_groups=inputs.dragon_groups,
        morgan_columns=inputs.morgan_columns,
        name_vocabulary=inputs.name_vocabulary,
        numeric_count=train_loader.dataset.numeric.shape[1],
        name_count=train_loader.dataset.name_features.shape[1],
        structure_count=train_loader.dataset.structure_features.shape[1],
        train_cids=sorted(training["CID"].unique().tolist()),
        validation_cids=validation_cids,
        full_cids=sorted(full_training["CID"].unique().tolist()),
        test_cids=test_cids,
        target_columns=target_columns,
        challenge=challenge,
        blend_alpha=blend_alpha,
        std_curve=std_curve,
        neighbours=(
            {
                "labels": reference_full,
                "mean": statistics_full["mean"],
                "scale": statistics_full["scale"],
                "power": NEIGHBOUR_POWER,
                "count": NEIGHBOUR_COUNT,
                "similarities": list(NEIGHBOUR_SIMILARITIES),
                "width": len(reference_full.columns) * len(NEIGHBOUR_SIMILARITIES),
            }
            if reference_full is not None
            else None
        ),
    )
