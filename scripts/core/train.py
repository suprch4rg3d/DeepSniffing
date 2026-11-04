"""One training loop for every model.

A model only has to expose the usual ``forward(tokens, features, fingerprints)``
signature. Everything else, meaning batching, early stopping on the official
challenge score, checkpoint selection, and the final test evaluation, is handled
here so the architectures stay comparable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.core.data import DatasetBundle
from scripts.core.metrics import TARGET_COUNT, dream_challenge_metrics
from scripts.core.splits import RANDOM_SEED


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


@dataclass
class TrainingResult:
    """Outcome of one training run."""

    model: nn.Module
    history: dict[str, list] = field(default_factory=dict)
    best_epoch: int = 0
    test_predictions: np.ndarray | None = None
    test_metrics: dict | None = None


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


def predict_values(model: nn.Module, arrays, target_scaler, device: torch.device) -> np.ndarray:
    """Run the model over an encoded split and return ratings in original units."""
    model.eval()
    tokens, features, fingerprints = arrays[:3]
    with torch.no_grad():
        scaled = (
            model(tokens.to(device), features.to(device), fingerprints.to(device)).cpu().numpy()
        )
    return inverse_target_transform(scaled, target_scaler)


def train_and_evaluate(
    model: nn.Module,
    bundle: DatasetBundle,
    config: TrainConfig | None = None,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Train a model, keep the best validation checkpoint, and score the test set."""
    config = config or TrainConfig()
    device = resolve_device(device)
    model = model.to(device)

    train_loader = DataLoader(
        TensorDataset(*bundle.arrays["train"]),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.MSELoss()

    validation_tokens, validation_features, validation_fingerprints, validation_targets = (
        bundle.arrays["validation"]
    )
    validation_actual = bundle.raw_targets["validation"]

    history: dict[str, list] = {"train_loss": [], "validation_loss": [], "validation_metrics": []}
    best_score = float("-inf")
    best_weights: dict | None = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(config.epochs):
        model.train()
        train_loss_sum, train_count = 0.0, 0
        for tokens, features, fingerprints, targets in train_loader:
            tokens = tokens.to(device)
            features = features.to(device)
            fingerprints = fingerprints.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(tokens, features, fingerprints), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            train_loss_sum += loss.item() * len(targets)
            train_count += len(targets)

        model.eval()
        with torch.no_grad():
            validation_scaled = (
                model(
                    validation_tokens.to(device),
                    validation_features.to(device),
                    validation_fingerprints.to(device),
                )
                .cpu()
                .numpy()
            )
            validation_loss = loss_fn(torch.tensor(validation_scaled), validation_targets).item()

        validation_prediction = inverse_target_transform(validation_scaled, bundle.target_scaler)
        challenge_metrics = dream_challenge_metrics(validation_actual, validation_prediction)
        history["train_loss"].append(float(train_loss_sum / train_count))
        history["validation_loss"].append(float(validation_loss))
        history["validation_metrics"].append(challenge_metrics)

        score = challenge_metrics["dream_challenge_score"]
        if verbose:
            print(f"Epoch {epoch + 1}: validation DREAM score {score:.3f}")

        if score > best_score:
            best_score = score
            best_weights = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs == config.patience:
                if verbose:
                    print(f"Early stopping after {config.patience} epochs without improvement")
                break

    if best_weights is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_weights)

    result = TrainingResult(model=model, history=history, best_epoch=best_epoch)
    result.test_predictions = predict_values(
        model, bundle.arrays["test"], bundle.target_scaler, device
    )
    result.test_metrics = dream_challenge_metrics(
        bundle.raw_targets["test"], result.test_predictions
    )
    if result.test_metrics is not None and len(bundle.target_columns) != TARGET_COUNT * 2:
        raise ValueError("The bundle must provide 21 means and 21 standard deviations.")
    return result
