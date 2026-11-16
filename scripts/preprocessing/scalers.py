"""The last step of ``ds-build``: fit the scalers that Subchallenge 1 and 2 train with.

Every scaler is fitted on training molecules only, in two stages, because the model is trained twice:

    validation   the 338 training molecules: the model that chooses the epochs
    final        the 407 training and validation molecules: the model that is scored on the test set

Two files are written, and ``ds-train`` reads them back and refuses a file that was fitted on other
molecules, other columns or another blend weight than the run uses:

    scalers/feature_scalers.joblib   the Dragon scaler, and the scalers of the numeric block (RDKit descriptors,
                                     partial charges, EPI Suite): median fill, cube root for Dragon and EPI Suite,
                                     then standardise. Both subchallenges read the same molecules, so they share it.
    scalers/target_scalers.joblib    the target scaler of each subchallenge. Q1's is fitted on the ratings after
                                     they are blended with those of the participants who agree with each person.
"""

from __future__ import annotations

import joblib

from scripts.core.data import (
    FEATURE_SCALERS_PATH,
    TARGET_SCALERS_PATH,
    fit_feature_scaler,
    fit_masked_target_scaler,
    load_graph_inputs,
    stage_cids,
    training_frames,
)
from scripts.core.splits import load_split
from scripts.core.train import Q1_BLEND_ALPHA


def fit_scalers() -> tuple[dict, dict]:
    """The feature scalers and the target scalers of both subchallenges, fitted on the rows that training uses."""
    inputs = load_graph_inputs()
    split = load_split()
    features: dict = {}
    targets: dict = {}
    for challenge in ("q2", "q1"):
        blend_alpha = Q1_BLEND_ALPHA if challenge == "q1" else None
        frames = training_frames(challenge, inputs, split, blend_alpha)
        cids = stage_cids(frames)
        if challenge == "q2":
            features = {
                "validation": fit_feature_scaler(frames.training, inputs.dragon_columns),
                "final": fit_feature_scaler(frames.full_training, inputs.dragon_columns),
                "cids": cids,
            }
        elif cids != features["cids"]:
            raise ValueError(
                "Q1 and Q2 do not train on the same molecules, so they cannot share feature scalers."
            )
        targets[challenge] = {
            "validation": fit_masked_target_scaler(frames.training, frames.target_columns),
            "final": fit_masked_target_scaler(frames.full_training, frames.target_columns),
            "cids": cids,
            "blend_alpha": blend_alpha,
        }
    return features, targets


def main() -> None:
    features, targets = fit_scalers()
    FEATURE_SCALERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(features, FEATURE_SCALERS_PATH)
    joblib.dump(targets, TARGET_SCALERS_PATH)
    stages = {stage: len(cids) for stage, cids in features["cids"].items()}
    print(
        f"Saved the feature scalers ({', '.join(features['validation'])}) fitted on {stages} molecules "
        f"to {FEATURE_SCALERS_PATH}"
    )
    print(f"Saved the target scalers of {', '.join(targets)} to {TARGET_SCALERS_PATH}")


if __name__ == "__main__":
    main()
