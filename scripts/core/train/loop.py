"""The building blocks that the training of Subchallenge 1 and 2 share: the seed, the settings of a run, the
loss, one epoch of training, and prediction in rating units."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.core.data import (
    STANDARD_LOG10_DILUTION,
    apply_std_curve,
)
from scripts.core.metrics import (
    OFFICIAL_NULL_SD,
    OFFICIAL_Q1_NULL_SD,
    SUBJECT_COUNT,
    TARGET_COUNT,
    dream_challenge_1_metrics,
    dream_challenge_metrics,
)
from scripts.core.splits import RANDOM_SEED
from scripts.training.custom_model import (
    GraphDescriptorModel,
)


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Seed every random number generator that can influence a run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters that are not part of the model definition."""

    epochs: int = 200
    patience: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    seed: int = RANDOM_SEED


def resolve_device(device: torch.device | str | None = None) -> torch.device:
    """Return the requested device, or CUDA when it is available.

    The default deliberately stays on CUDA or CPU. Apple's MPS backend changes
    floating point behaviour and is not bit reproducible, which would make the
    published scores impossible to compare, so it has to be requested on purpose.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def inverse_target_transform(predictions: np.ndarray, target_scaler) -> np.ndarray:
    """Undo target standardisation and clip ratings to the observed 0 to 100 range."""
    restored = target_scaler.inverse_transform(predictions)
    return np.clip(restored, 0.0, 100.0)


@dataclass
class TrainingResult:
    """What one run of ``train_q1`` or ``train_q2`` returns."""

    model: nn.Module
    history: dict[str, list] = field(default_factory=dict)
    best_epoch: int = 0
    test_predictions: np.ndarray | None = None
    test_metrics: dict | None = None
    group_epochs: dict[str, int] = field(default_factory=dict)
    group_states: dict[str, dict] = field(default_factory=dict)
    std_curve: dict | None = None
    validation_score: float = 0.0


# A group counts as improved only when its score rises by more than this.
MIN_SCORE_DELTA = 0.001


def calibration_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean squared error of the standardised predictions over the ratings that exist."""
    valid = torch.ones_like(target, dtype=torch.bool) if mask is None else mask.bool()
    valid = valid & torch.isfinite(target)
    if not valid.any():
        return prediction.sum() * 0.0
    return ((prediction - torch.where(valid, target, prediction.detach())) ** 2)[valid].mean()


def dream_score_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    unscale: tuple[torch.Tensor, torch.Tensor] | None = None,
    groups: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative DREAM score surrogate from differentiable per output correlations.

    Subchallenge 2 has 42 outputs and Subchallenge 1 has 21 attributes by 49 participants,
    and the output count decides which score groups are averaged. ``mask`` marks the
    ratings that exist. ``unscale`` is the (mean, scale) of the target scaler: the
    Subchallenge 1 objective is computed on ratings clamped to the 0 to 100 range.

    ``groups`` labels each row of the batch, here with its concentration. The correlation
    is then measured inside each group, after removing the mean of the group. Ratings at
    two concentrations differ by a constant, and intensity by a lot, so across the whole
    batch a model could score well from the concentration label alone. The official score
    compares molecules at one concentration, and so does this.
    """
    if unscale is not None:
        prediction = (prediction * unscale[1] + unscale[0]).clamp(0.0, 100.0)
    valid = torch.ones_like(target, dtype=torch.bool) if mask is None else mask.bool()
    valid = valid & torch.isfinite(target)
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    safe_prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
    counts = valid.sum(dim=0)
    if groups is None:
        groups = torch.zeros(prediction.shape[0], device=prediction.device)
    membership = torch.nn.functional.one_hot(torch.unique(groups, return_inverse=True)[1]).to(
        prediction.dtype
    )
    group_counts = (membership.T @ valid.to(prediction.dtype)).clamp_min(1)
    target_mean = membership @ ((membership.T @ safe_target) / group_counts)
    prediction_mean = membership @ ((membership.T @ safe_prediction) / group_counts)
    centered_target = torch.where(valid, target - target_mean, torch.zeros_like(target))
    centered_prediction = torch.where(
        valid, prediction - prediction_mean, torch.zeros_like(prediction)
    )
    covariance = (centered_target * centered_prediction).sum(dim=0)
    variance_product = centered_target.square().sum(dim=0) * centered_prediction.square().sum(dim=0)
    correlations = torch.where(
        (counts >= 2) & (variance_product > 1e-12),
        covariance / torch.sqrt(variance_product.clamp_min(1e-12)),
        torch.zeros_like(covariance),
    )

    if prediction.shape[1] == TARGET_COUNT * SUBJECT_COUNT:
        group_sizes = (SUBJECT_COUNT, SUBJECT_COUNT, (TARGET_COUNT - 2) * SUBJECT_COUNT)
        null_sds = OFFICIAL_Q1_NULL_SD.values()
    else:
        group_sizes = (1, 1, TARGET_COUNT - 2, 1, 1, TARGET_COUNT - 2)
        null_sds = OFFICIAL_NULL_SD.values()
    group_scores = []
    start = 0
    for size, null_sd in zip(group_sizes, null_sds, strict=True):
        group_scores.append(correlations[start : start + size].mean() / null_sd)
        start += size
    scores = torch.stack(group_scores)
    spreads = torch.tensor(list(null_sds), dtype=scores.dtype, device=scores.device)
    scores = scores * spreads / spreads.mean()
    return -scores.mean()


def challenge_scores(challenge: str, actual: np.ndarray, prediction: np.ndarray) -> dict:
    """Score a prediction with the official metric of its subchallenge."""
    if challenge == "q1":
        return dream_challenge_1_metrics(
            actual, prediction.reshape(-1, TARGET_COUNT, SUBJECT_COUNT)
        )
    return dream_challenge_metrics(actual, prediction)


def graph_forward(
    model: GraphDescriptorModel,
    batch: dict,
    device: torch.device,
    standard_dilution: bool = False,
) -> torch.Tensor:
    """Run the model on a batch, optionally as if every row had the standard dilution."""
    numeric = batch["numeric"].to(device)
    if standard_dilution:
        # Intensity is rated at 1/1000. For a molecule whose descriptors are the high rating, that
        # is the low rating of its pair, and for one whose descriptors are the low rating, it is
        # the high rating of its pair. So the flag is the opposite of the descriptors' flag.
        numeric = numeric.clone()
        numeric[:, -2] = STANDARD_LOG10_DILUTION
        numeric[:, -1] = 1.0 - numeric[:, -1]
    return model(
        batch["atoms"].to(device),
        batch["edges"].to(device),
        batch["edge_features"].to(device),
        batch["graph_ids"].to(device),
        batch["dragon"].to(device),
        numeric,
        batch["name_features"].to(device),
        batch["structure_features"].to(device),
        batch["neighbour_features"].to(device) if "neighbour_features" in batch else None,
    )


def train_graph_epoch(
    model: GraphDescriptorModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainConfig,
    target_scaler=None,
    calibration_weight: float = 0.0,
) -> float:
    """Run one epoch and return the mean negative DREAM score surrogate.

    Pass the ``target_scaler`` for Subchallenge 1, whose objective is computed on
    ratings in their original units. A ``calibration_weight`` above 0, used by Subchallenge 2,
    adds the squared error of the standardised predictions to the loss.
    """
    unscale = None
    if target_scaler is not None:
        unscale = (
            torch.as_tensor(target_scaler.mean_, dtype=torch.float32, device=device),
            torch.as_tensor(target_scaler.scale_, dtype=torch.float32, device=device),
        )
    model.train()
    total_loss, total_rows = 0.0, 0
    for batch in loader:
        optimizer.zero_grad()
        prediction = graph_forward(model, batch, device)
        targets, mask = batch["targets"].to(device), batch["mask"].to(device)
        score_loss = dream_score_loss(
            prediction,
            targets,
            mask,
            unscale,
            batch["numeric"][:, -1].to(device),
        )
        loss = score_loss
        if calibration_weight > 0.0:
            loss = loss + calibration_weight * calibration_loss(prediction, targets, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("The DREAM objective became non-finite.")
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Gradients became non-finite.")
        optimizer.step()
        total_loss += float(score_loss.detach().cpu()) * len(batch["targets"])
        total_rows += len(batch["targets"])
    return total_loss / total_rows


def predict_graph(
    model: GraphDescriptorModel,
    loader: DataLoader,
    target_scaler,
    device: torch.device,
    cids: list[int],
    intensity_columns: list[int] | None = None,
    std_curve: dict | None = None,
) -> np.ndarray:
    """Predict a loader in rating units, averaged per molecule in the order of ``cids``.

    With ``std_curve`` the spreads of Subchallenge 2 are mixed with the curve of the predicted
    means, after the averaging per molecule.

    With ``intensity_columns`` those outputs are predicted as if every row had the standard
    dilution, because intensity is learned at that dilution. Every other output uses the
    dilution of the row.

    A molecule can appear in several rows, because replicates are kept as separate
    samples. The official targets have one row per molecule, so predictions are
    averaged per molecule before they are scored.
    """
    model.eval()
    scaled = []
    with torch.no_grad():
        for batch in loader:
            output = graph_forward(model, batch, device)
            if intensity_columns:
                standard = graph_forward(model, batch, device, standard_dilution=True)
                output[:, intensity_columns] = standard[:, intensity_columns]
            scaled.append(output.cpu().numpy())
    rows = inverse_target_transform(np.concatenate(scaled), target_scaler)
    frame = pd.DataFrame(rows)
    frame["CID"] = loader.dataset.cids
    prediction = frame.groupby("CID").mean().loc[cids].to_numpy()
    if std_curve is not None:
        prediction = apply_std_curve(prediction, std_curve)
    return prediction


def output_groups(challenge: str) -> dict[str, list[int]]:
    """The output columns that each official score group is computed from.

    Subchallenge 1 has three groups, intensity, valence and the 19 descriptors, each over the
    49 participants. Subchallenge 2 has the same three for the means and for the spreads.
    The names are the keys of the z scores that ``challenge_scores`` returns.
    """
    sizes = (("intensity", 1), ("valence", 1), ("descriptors", TARGET_COUNT - 2))
    if challenge == "q1":
        groups, start = {}, 0
        for name, attributes in sizes:
            groups[name] = list(range(start * SUBJECT_COUNT, (start + attributes) * SUBJECT_COUNT))
            start += attributes
        return groups
    groups = {}
    for offset, suffix in ((0, "mean"), (TARGET_COUNT, "std")):
        start = offset
        for name, attributes in sizes:
            groups[f"{name}_{suffix}"] = list(range(start, start + attributes))
            start += attributes
    return groups
