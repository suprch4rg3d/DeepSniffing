"""Hybrid odor model: SMILES convolution plus descriptors plus fingerprints.

The three input representations are encoded by separate branches and joined
before the regression head. This is the model that produced the reported DREAM
score, and it is the reference implementation that the standalone CNN and the
transformer baselines are compared against.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import torch
from torch import nn

from scripts.core.data import DatasetBundle, prepare_data
from scripts.core.metrics import (
    TARGETS,
    dream_challenge_metrics,
    prediction_matrix_from_submission,
    write_metrics,
)
from scripts.core.plots import (
    save_challenge_comparison_plot,
    save_test_metrics_plot,
    save_training_plots,
)
from scripts.core.train import TrainConfig, resolve_device, seed_everything, train_and_evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output/custom_model"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
PLOTS_DIR = OUTPUT_DIR / "plots"
METRICS_DIR = OUTPUT_DIR / "metrics"
REFERENCE_PREDICTIONS = (
    PROJECT_ROOT / "results/baseline/dream_challenge/challenge_2_testset_1446435104.txt"
)
REFERENCE_SCORE_PATH = (
    PROJECT_ROOT / "results/baseline/dream_challenge/official_testset_reference_score.json"
)

EMBEDDING_SIZE = 32
CONV_FILTERS = (48, 64)
KERNEL_SIZE = 3
DESCRIPTOR_HIDDEN_SIZE = 32
FINGERPRINT_HIDDEN_SIZE = 128
FINGERPRINT_LATENT_SIZE = 48
HEAD_HIDDEN_SIZE = 64


class CustomOdorModel(nn.Module):
    """Multi branch regressor producing 21 means and 21 standard deviations."""

    def __init__(
        self,
        vocab_size: int,
        feature_count: int,
        fingerprint_count: int,
        target_count: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE, padding_idx=0)
        self.smiles_cnn = nn.Sequential(
            nn.Conv1d(EMBEDDING_SIZE, CONV_FILTERS[0], kernel_size=KERNEL_SIZE, padding=1),
            nn.ReLU(),
            nn.Conv1d(CONV_FILTERS[0], CONV_FILTERS[1], kernel_size=KERNEL_SIZE, padding=1),
            nn.ReLU(),
        )
        self.descriptor_branch = nn.Sequential(
            nn.LayerNorm(feature_count),
            nn.Linear(feature_count, DESCRIPTOR_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.morgan_branch = nn.Sequential(
            nn.Linear(fingerprint_count, FINGERPRINT_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(FINGERPRINT_HIDDEN_SIZE, FINGERPRINT_LATENT_SIZE),
            nn.ReLU(),
        )
        joined_size = CONV_FILTERS[1] + DESCRIPTOR_HIDDEN_SIZE + FINGERPRINT_LATENT_SIZE
        self.output = nn.Sequential(
            nn.Linear(joined_size, HEAD_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HEAD_HIDDEN_SIZE, target_count),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        features: torch.Tensor,
        fingerprints: torch.Tensor,
    ) -> torch.Tensor:
        mask = tokens.ne(0).unsqueeze(1)
        embedded = self.embedding(tokens).transpose(1, 2)
        convolved = self.smiles_cnn(embedded)
        smiles_features = (convolved * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
        descriptor_features = self.descriptor_branch(features)
        morgan_features = self.morgan_branch(fingerprints)
        return self.output(
            torch.cat((smiles_features, descriptor_features, morgan_features), dim=1)
        )


def build_model(bundle: DatasetBundle) -> CustomOdorModel:
    """Instantiate the hybrid model with the shapes implied by a dataset bundle."""
    return CustomOdorModel(
        vocab_size=bundle.vocab_size,
        feature_count=bundle.feature_count,
        fingerprint_count=bundle.fingerprint_count,
        target_count=bundle.target_count,
    )


def prediction_frame(cids: list[int], predictions, targets: list[str]) -> pd.DataFrame:
    """Format predictions as an official DREAM submission table."""
    label_by_target = {
        "intensity_strength": "INTENSITY/STRENGTH",
        "valence_pleasantness": "VALENCE/PLEASANTNESS",
        "ammonia_urinous": "AMMONIA/URINOUS",
    }
    rows = []
    for index, cid in enumerate(cids):
        for target_index, target in enumerate(targets):
            rows.append(
                {
                    "#oID": cid,
                    "descriptor": label_by_target.get(target, target.upper()),
                    "value": float(predictions[index, target_index]),
                    "sigma": float(predictions[index, len(targets) + target_index]),
                }
            )
    return pd.DataFrame(rows)


def save_artifacts(bundle: DatasetBundle, result, output_dir: Path = OUTPUT_DIR) -> None:
    """Persist the checkpoint, scalers, vocabulary, metrics, and plots."""
    checkpoints = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    for path in (checkpoints, metrics_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    bundle.tokenizer.save(checkpoints / "smiles_vocabulary.json")
    joblib.dump(bundle.feature_scaler, checkpoints / "feature_scaler.joblib")
    joblib.dump(bundle.target_scaler, checkpoints / "target_scaler.joblib")

    test_metrics = dict(result.test_metrics)
    test_metrics["evaluation_set"] = "DREAM official TestSet.txt (69 molecules)"
    test_metrics["prediction_targets"] = "21 high-intensity means and 21 standard deviations"

    prediction_frame(bundle.test_cids, result.test_predictions, TARGETS).to_csv(
        metrics_dir / "custom_model_test_predictions.tsv", sep="\t", index=False
    )
    write_metrics(metrics_dir / "dream_challenge_test_metrics.json", test_metrics)
    save_test_metrics_plot(test_metrics, plots_dir)

    reference_metrics = None
    if REFERENCE_PREDICTIONS.is_file():
        reference_prediction = prediction_matrix_from_submission(
            REFERENCE_PREDICTIONS, bundle.test_cids
        )
        reference_metrics = dream_challenge_metrics(
            bundle.raw_targets["test"], reference_prediction
        )
        reference_metrics["source"] = str(REFERENCE_PREDICTIONS.relative_to(PROJECT_ROOT))
        reference_metrics["evaluation_set"] = "DREAM official TestSet.txt (69 molecules)"
        write_metrics(REFERENCE_SCORE_PATH, reference_metrics)

    save_challenge_comparison_plot(test_metrics, plots_dir, reference_metrics)
    save_training_plots(result.history, plots_dir)

    pd.DataFrame(
        [
            {
                "epoch": index + 1,
                "dream_challenge_score": values["dream_challenge_score"],
                **{
                    f"pearson_{key}": value for key, value in values["pearson_correlations"].items()
                },
                **{f"z_{key}": value for key, value in values["z_scores"].items()},
            }
            for index, values in enumerate(result.history["validation_metrics"])
        ]
    ).to_csv(metrics_dir / "validation_dream_challenge_metrics.csv", index=False)

    torch.save(
        {
            "model_name": "CustomOdorModel",
            "state_dict": {
                key: value.detach().cpu() for key, value in result.model.state_dict().items()
            },
            "descriptor_columns": bundle.features,
            "morgan_fingerprint_columns": bundle.fingerprint_columns,
            "morgan_radius": 2,
            "morgan_fingerprint_size": bundle.fingerprint_count,
            "morgan_include_chirality": True,
            "target_columns": bundle.target_columns,
            "target_names": list(TARGETS),
            "vocabulary_size": bundle.vocab_size,
            "max_sequence_length": 128,
            "prediction_outputs": bundle.target_count,
        },
        checkpoints / "custom_model.pt",
    )


def main() -> None:
    seed_everything()
    bundle = prepare_data()
    device = resolve_device()
    print(
        f"Using {device}; training CIDs {bundle.train_size}; "
        f"validation CIDs {bundle.validation_size}; "
        f"official test CIDs {bundle.test_size}"
    )

    seed_everything()
    model = build_model(bundle)
    result = train_and_evaluate(model, bundle, TrainConfig(), device)
    print(f"Using best validation checkpoint from epoch {result.best_epoch}")

    save_artifacts(bundle, result)
    print(f"Official DREAM Q2 test Z-score: {result.test_metrics['dream_challenge_score']:.3f}")
    print(f"Official test metrics: {METRICS_DIR / 'dream_challenge_test_metrics.json'}")
    print(f"Training plots: {PLOTS_DIR}")
    print(f"Checkpoints: {CHECKPOINTS_DIR}")


if __name__ == "__main__":
    main()
