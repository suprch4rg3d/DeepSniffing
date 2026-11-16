"""``ds-train``: train, select epochs, retrain and score Subchallenge 1 and 2.

The file reads from top to bottom: the pieces that both subchallenges use (the seed, the settings of a run,
the loss, one epoch of training, prediction), then ``train_q2``, then ``train_q1``. Each of those two is a
whole run written out in order, with no branches for the other subchallenge:

    1. load the data: 338 training molecules, 69 validation molecules, 69 test molecules;
    2. train on the training molecules and choose an epoch for each score group on the validation ones;
    3. (Q2 only) calibrate the predicted spread of the ratings on the validation predictions;
    4. train again from scratch on all 407 labeled molecules, keeping a copy at each chosen epoch;
    5. predict the 69 test molecules once, each score group from its own copy, and score them;
    6. save the model, the predictions and the scores, and the log of the run.

The split, the scalers and every input table are read from the files that ``ds-build``, ``ds-build`` and
``ds-build`` wrote. The architecture comes from ``scripts.training.custom_model``. The settings are the
constants of this module: a run has no options, so a result is always the one that the code documents.
The test molecules play no part in steps 2 to 4."""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

from scripts.core.data import load_graph_inputs
from scripts.core.train.loop import (
    MIN_SCORE_DELTA,
    TrainConfig,
    TrainingResult,
    calibration_loss,
    challenge_scores,
    dream_score_loss,
    graph_forward,
    inverse_target_transform,
    output_groups,
    predict_graph,
    resolve_device,
    seed_everything,
    train_graph_epoch,
)
from scripts.core.train.q1 import (
    Q1_BLEND_ALPHA,
    Q1_OUTPUT_DIR,
    Q1_TRAIN_CONFIG,
    train_q1,
)
from scripts.core.train.q2 import (
    Q2_CALIBRATION_WEIGHT,
    Q2_EPOCH_WINDOW,
    Q2_OUTPUT_DIR,
    Q2_TRAIN_CONFIG,
    SPREAD_WEIGHT_CANDIDATES,
    smoothed_best_epochs,
    train_q2,
)


class _Tee:
    """Write to the console and to the log file of the run at once."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def logged(path: Path):
    """Everything printed inside the block also goes to ``path``, so a model keeps the log that made it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as log:
        original = sys.stdout
        sys.stdout = _Tee(original, log)
        try:
            yield
        finally:
            sys.stdout = original


def main() -> None:
    """``ds-train``: train Subchallenge 2, then Subchallenge 1. Each leaves its log next to its model."""
    parser = argparse.ArgumentParser(description="Train the model on the DREAM data.")
    parser.add_argument(
        "--challenge",
        choices=("both", "q1", "q2"),
        default="both",
        help="Which subchallenge to train. Both are trained independently by default.",
    )
    challenges = parser.parse_args().challenge
    seed_everything()
    inputs = load_graph_inputs()
    if challenges in ("both", "q2"):
        with logged(Q2_OUTPUT_DIR / "train.log"):
            print("Training the Subchallenge 2 model")
            train_q2(inputs=inputs)
    if challenges in ("both", "q1"):
        with logged(Q1_OUTPUT_DIR / "train.log"):
            print("Training the Subchallenge 1 model")
            train_q1(inputs=inputs)


__all__ = [
    "MIN_SCORE_DELTA",
    "Q1_BLEND_ALPHA",
    "Q1_OUTPUT_DIR",
    "Q1_TRAIN_CONFIG",
    "Q2_CALIBRATION_WEIGHT",
    "Q2_EPOCH_WINDOW",
    "Q2_OUTPUT_DIR",
    "Q2_TRAIN_CONFIG",
    "SPREAD_WEIGHT_CANDIDATES",
    "TrainConfig",
    "TrainingResult",
    "calibration_loss",
    "challenge_scores",
    "dream_score_loss",
    "graph_forward",
    "inverse_target_transform",
    "output_groups",
    "predict_graph",
    "resolve_device",
    "seed_everything",
    "smoothed_best_epochs",
    "train_graph_epoch",
    "train_q1",
    "train_q2",
    "main",
]
