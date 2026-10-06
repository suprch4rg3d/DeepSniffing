from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.preprocessing.enriched_feature_scaler import ENRICHED_FEATURE_COLUMNS
from scripts.preprocessing.smiles_tokenizer import SmilesTokenizer

DATA = ROOT / "data/processed/dream_pyrfume/molecules_enriched.csv"
MORGAN_FINGERPRINTS = ROOT / "data/processed/dream_pyrfume/morgan_fingerprints.csv"
TRAIN_RATINGS = ROOT / "data/raw/dream-olfaction-prediction/data/TrainSet.txt"
TEST_RATINGS = ROOT / "data/raw/dream-olfaction-prediction/data/TestSet.txt"
REFERENCE_PREDICTIONS = (
    ROOT / "results/baseline/dream_challenge/challenge_2_testset_1446435104.txt"
)
PUBLISHED_BENCHMARKS = (
    ROOT / "results/baseline/dream_challenge/published_challenge_scores.json"
)
OUTPUT = ROOT / "output/custom_model"
CHECKPOINTS = OUTPUT / "checkpoints"
PLOTS = OUTPUT / "plots"
METRICS = OUTPUT / "metrics"

SEED = 42
EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 32
MAX_LENGTH = 128
VALIDATION_FRACTION = 0.15

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
OFFICIAL_NULL_SD = {
    "intensity_mean": 0.119307474,
    "valence_mean": 0.126471379,
    "descriptors_mean": 0.026512975,
    "intensity_std": 0.119494682,
    "valence_std": 0.11490659,
    "descriptors_std": 0.02808499,
}


class CustomOdorModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feature_count: int,
        fingerprint_count: int,
        target_count: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.smiles_cnn = nn.Sequential(
            nn.Conv1d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.descriptor_branch = nn.Sequential(
            nn.LayerNorm(feature_count),
            nn.Linear(feature_count, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.morgan_branch = nn.Sequential(
            nn.Linear(fingerprint_count, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 48),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(144, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, target_count),
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


def normalize_target_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def aggregate_high_intensity_targets(path: Path) -> pd.DataFrame:
    ratings = pd.read_csv(path, sep="\t").rename(columns={"Compound Identifier": "CID"})
    required = {"CID", "Intensity", *TARGETS}
    normalized_columns = {
        column: normalize_target_name(column)
        for column in ratings.columns
        if column
        not in {"CID", "Odor", "Replicate", "Intensity", "Dilution", "subject #"}
    }
    ratings = ratings.rename(columns=normalized_columns)
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(
            f"DREAM rating file is missing columns: {', '.join(sorted(missing))}"
        )

    high = ratings.loc[
        ratings["Intensity"].astype(str).str.strip().str.lower() == "high"
    ].copy()
    if high.empty:
        raise ValueError(f"No high-intensity ratings found in {path}")
    high["CID"] = pd.to_numeric(high["CID"], errors="raise").astype("int64")
    for target in TARGETS:
        high[target] = pd.to_numeric(high[target], errors="coerce")

    if "subject #" not in high.columns:
        raise ValueError(f"DREAM rating file has no subject identifiers: {path}")
    subject_means = high.groupby(["CID", "subject #"])[TARGETS].mean()
    means = subject_means.groupby(level="CID").mean()
    standard_deviations = subject_means.groupby(level="CID").std(ddof=1)
    summary = pd.concat(
        [means.add_prefix("mean_"), standard_deviations.add_prefix("std_")], axis=1
    )
    ordered_columns = [f"mean_{target}" for target in TARGETS] + [
        f"std_{target}" for target in TARGETS
    ]
    return summary[ordered_columns].reset_index()


def pearson_across_molecules(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if len(actual) < 2 or np.std(actual) == 0 or np.std(predicted) == 0:
        return 0.0
    return float(np.corrcoef(actual, predicted)[0, 1])


def dream_challenge_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    target_count = len(TARGETS)
    if actual.shape != predicted.shape or actual.shape[1] != target_count * 2:
        raise ValueError(
            "DREAM Q2 scoring expects 21 means followed by 21 standard deviations."
        )

    correlations = {
        "intensity_mean": pearson_across_molecules(actual[:, 0], predicted[:, 0]),
        "valence_mean": pearson_across_molecules(actual[:, 1], predicted[:, 1]),
        "descriptors_mean": float(
            np.mean(
                [
                    pearson_across_molecules(actual[:, i], predicted[:, i])
                    for i in range(2, target_count)
                ]
            )
        ),
        "intensity_std": pearson_across_molecules(
            actual[:, target_count], predicted[:, target_count]
        ),
        "valence_std": pearson_across_molecules(
            actual[:, target_count + 1], predicted[:, target_count + 1]
        ),
        "descriptors_std": float(
            np.mean(
                [
                    pearson_across_molecules(
                        actual[:, target_count + i], predicted[:, target_count + i]
                    )
                    for i in range(2, target_count)
                ]
            )
        ),
    }
    z_scores = {
        name: correlations[name] / null_sd for name, null_sd in OFFICIAL_NULL_SD.items()
    }
    return {
        "pearson_correlations": correlations,
        "z_scores": z_scores,
        "dream_challenge_score": float(np.mean(list(z_scores.values()))),
    }


def prediction_matrix_from_submission(path: Path, cids: list[int]) -> np.ndarray:
    submission = pd.read_csv(path, sep="\t")
    submission = submission.rename(columns={"#oID": "CID"})
    if not {"CID", "descriptor", "value", "sigma"}.issubset(submission.columns):
        raise ValueError(f"Unexpected DREAM prediction file format: {path}")
    submission["CID"] = pd.to_numeric(submission["CID"], errors="raise").astype("int64")
    submission["target"] = (
        submission["descriptor"].astype(str).map(normalize_target_name)
    )
    value = submission.pivot(index="CID", columns="target", values="value")
    sigma = submission.pivot(index="CID", columns="target", values="sigma")
    expected_cids = set(cids)
    if set(value.index) != expected_cids or set(sigma.index) != expected_cids:
        raise ValueError(
            f"Reference predictions do not match the official test CIDs: {path}"
        )
    if set(value.columns) != set(TARGETS) or set(sigma.columns) != set(TARGETS):
        raise ValueError(
            f"Reference predictions do not contain exactly 21 DREAM targets: {path}"
        )
    value = value.loc[cids, TARGETS].to_numpy(dtype=np.float64)
    sigma = sigma.loc[cids, TARGETS].to_numpy(dtype=np.float64)
    return np.concatenate((value, sigma), axis=1)


def prepare_data():
    for path in (
        DATA,
        MORGAN_FINGERPRINTS,
        TRAIN_RATINGS,
        TEST_RATINGS,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    molecules = pd.read_csv(DATA)
    fingerprints = pd.read_csv(MORGAN_FINGERPRINTS)
    features = ENRICHED_FEATURE_COLUMNS.copy()
    if "CID" not in fingerprints.columns:
        raise ValueError(
            f"Morgan fingerprint file has no CID column: {MORGAN_FINGERPRINTS}"
        )
    fingerprint_columns = [
        column for column in fingerprints.columns if column.startswith("morgan_")
    ]
    expected_fingerprint_columns = [f"morgan_{index:04d}" for index in range(2048)]
    if fingerprint_columns != expected_fingerprint_columns:
        raise ValueError(
            "Morgan fingerprint columns must be morgan_0000 through morgan_2047 in "
            f"{MORGAN_FINGERPRINTS}"
        )
    if fingerprints["CID"].isna().any() or fingerprints["CID"].duplicated().any():
        raise ValueError("Morgan fingerprints must have one non-empty row per CID.")
    fingerprints["CID"] = pd.to_numeric(fingerprints["CID"], errors="raise").astype(
        "int64"
    )
    fingerprint_values = fingerprints[fingerprint_columns].to_numpy()
    if not np.isin(fingerprint_values, (0, 1)).all():
        raise ValueError("Morgan fingerprint columns must contain only binary values.")

    required_molecules = {"CID", "rdkit_canonical_smiles", *features}
    missing = required_molecules.difference(molecules.columns)
    if missing:
        raise ValueError(
            f"Enriched molecule data is missing: {', '.join(sorted(missing))}"
        )
    if molecules["CID"].duplicated().any():
        raise ValueError("Enriched molecule data must contain one row per CID.")
    molecules["CID"] = pd.to_numeric(molecules["CID"], errors="raise").astype("int64")
    if set(molecules["CID"]) != set(fingerprints["CID"]):
        raise ValueError(
            "Morgan fingerprint CIDs do not match the enriched molecule CIDs."
        )
    molecules = molecules.merge(
        fingerprints, on="CID", how="left", validate="one_to_one"
    )

    train_targets = aggregate_high_intensity_targets(TRAIN_RATINGS)
    test_targets = aggregate_high_intensity_targets(TEST_RATINGS)
    molecule_by_cid = molecules.set_index("CID")
    if not set(train_targets["CID"]).issubset(molecule_by_cid.index):
        raise ValueError(
            "Some official DREAM training CIDs have no molecular features."
        )
    if not set(test_targets["CID"]).issubset(molecule_by_cid.index):
        raise ValueError("Some official DREAM test CIDs have no molecular features.")

    train_targets = train_targets.merge(
        molecules, on="CID", how="left", validate="one_to_one"
    )
    test_targets = test_targets.merge(
        molecules, on="CID", how="left", validate="one_to_one"
    )
    target_columns = [f"mean_{target}" for target in TARGETS] + [
        f"std_{target}" for target in TARGETS
    ]
    train_targets = train_targets.dropna(subset=target_columns).copy()
    test_targets = test_targets.dropna(subset=target_columns).copy()
    if len(test_targets) != 69:
        raise ValueError(
            f"Expected the official 69-molecule test set; found {len(test_targets)}."
        )

    train_rows, validation_rows = train_test_split(
        np.arange(len(train_targets)),
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
        shuffle=True,
    )
    train = train_targets.iloc[train_rows].reset_index(drop=True)
    validation = train_targets.iloc[validation_rows].reset_index(drop=True)

    feature_scaler = StandardScaler().fit(train[features].to_numpy(dtype=np.float64))
    target_scaler = StandardScaler().fit(
        train[target_columns].to_numpy(dtype=np.float64)
    )
    tokenizer = SmilesTokenizer()
    train_smiles = train["rdkit_canonical_smiles"].fillna("").astype(str).tolist()
    if not train_smiles or any(not value.strip() for value in train_smiles):
        raise ValueError("Training split has missing SMILES or no rows.")
    tokenizer.fit(train_smiles)

    def encode_rows(rows: pd.DataFrame, with_targets: bool):
        smiles = rows["rdkit_canonical_smiles"].fillna("").astype(str).tolist()
        if any(not value.strip() for value in smiles):
            raise ValueError("A DREAM split contains missing canonical SMILES.")
        tokens = np.asarray(
            [tokenizer.encode(value, MAX_LENGTH) for value in smiles], dtype=np.int64
        )
        scaled_features = feature_scaler.transform(
            rows[features].to_numpy(dtype=np.float64)
        )
        fingerprint_values = rows[fingerprint_columns].to_numpy(dtype=np.float32)
        tensors = (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(scaled_features, dtype=torch.float32),
            torch.tensor(fingerprint_values, dtype=torch.float32),
        )
        if with_targets:
            scaled_targets = target_scaler.transform(
                rows[target_columns].to_numpy(dtype=np.float64)
            )
            tensors += (torch.tensor(scaled_targets, dtype=torch.float32),)
        return tensors

    arrays = {
        "train": encode_rows(train, True),
        "validation": encode_rows(validation, True),
        "test": encode_rows(test_targets, False),
    }
    raw_targets = {
        "validation": validation[target_columns].to_numpy(dtype=np.float64),
        "test": test_targets[target_columns].to_numpy(dtype=np.float64),
    }
    test_cids = test_targets["CID"].astype(int).tolist()
    return (
        arrays,
        raw_targets,
        test_cids,
        tokenizer,
        feature_scaler,
        target_scaler,
        features,
        fingerprint_columns,
        target_columns,
    )


def predict_values(model, arrays, target_scaler, device) -> np.ndarray:
    model.eval()
    tokens, features, fingerprints = arrays[:3]
    with torch.no_grad():
        scaled_prediction = (
            model(tokens.to(device), features.to(device), fingerprints.to(device))
            .cpu()
            .numpy()
        )
    predictions = target_scaler.inverse_transform(scaled_prediction)
    predictions[:, : len(TARGETS)] = np.clip(predictions[:, : len(TARGETS)], 0, 100)
    predictions[:, len(TARGETS) :] = np.clip(predictions[:, len(TARGETS) :], 0, 100)
    return predictions


def train_model(arrays, raw_targets, vocab_size, feature_count, target_scaler, device):
    train_loader = DataLoader(
        TensorDataset(*arrays["train"]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    fingerprint_count = arrays["train"][2].shape[1]
    model = CustomOdorModel(
        vocab_size, feature_count, fingerprint_count, len(TARGETS) * 2
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_score, best_weights, best_epoch, stale_epochs = float("-inf"), None, 0, 0
    history = {"train_loss": [], "validation_loss": [], "validation_metrics": []}
    (
        validation_tokens,
        validation_features,
        validation_fingerprints,
        validation_targets,
    ) = arrays["validation"]

    for epoch in range(EPOCHS):
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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(targets)
            train_count += len(targets)

        model.eval()
        with torch.no_grad():
            validation_prediction_scaled = (
                model(
                    validation_tokens.to(device),
                    validation_features.to(device),
                    validation_fingerprints.to(device),
                )
                .cpu()
                .numpy()
            )
            validation_loss = loss_fn(
                torch.tensor(validation_prediction_scaled), validation_targets
            ).item()
        validation_prediction = target_scaler.inverse_transform(
            validation_prediction_scaled
        )
        validation_prediction[:, : len(TARGETS)] = np.clip(
            validation_prediction[:, : len(TARGETS)], 0, 100
        )
        validation_prediction[:, len(TARGETS) :] = np.clip(
            validation_prediction[:, len(TARGETS) :], 0, 100
        )
        challenge_metrics = dream_challenge_metrics(
            raw_targets["validation"], validation_prediction
        )
        history["train_loss"].append(float(train_loss_sum / train_count))
        history["validation_loss"].append(float(validation_loss))
        history["validation_metrics"].append(challenge_metrics)
        score = challenge_metrics["dream_challenge_score"]
        print(f"Epoch {epoch + 1}: validation DREAM score {score:.3f}")

        if score > best_score:
            best_score = score
            best_weights = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs == PATIENCE:
                break

    if best_weights is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_weights)
    return model, history, best_epoch


def save_training_plots(history) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history["train_loss"], label="Training")
    axis.plot(history["validation_loss"], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Scaled MSE (42 outputs)")
    axis.set_title("Combined model training loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(PLOTS / "training_loss.png", dpi=160)
    plt.close(figure)

    metric_names = list(OFFICIAL_NULL_SD)
    figure, axis = plt.subplots(figsize=(9, 5))
    for name in metric_names:
        values = [entry["z_scores"][name] for entry in history["validation_metrics"]]
        axis.plot(range(1, len(values) + 1), values, label=name.replace("_", " "))
    scores = [entry["dream_challenge_score"] for entry in history["validation_metrics"]]
    axis.plot(
        range(1, len(scores) + 1),
        scores,
        color="black",
        linewidth=2.5,
        label="Mean Z-score",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Official DREAM Q2 Z-score components")
    axis.set_title("DREAM Subchallenge 2 validation metrics")
    axis.legend(fontsize=8, ncol=2)
    axis.text(
        0.01,
        -0.24,
        "Internal validation fold; the final 69-molecule test score is reported separately.",
        transform=axis.transAxes,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(PLOTS / "dream_challenge_metrics.png", dpi=160)
    plt.close(figure)


def save_test_metrics_plot(metrics: dict) -> None:
    names = list(OFFICIAL_NULL_SD)
    values = [metrics["z_scores"][name] for name in names]
    names.append("Mean Z-score")
    values.append(metrics["dream_challenge_score"])
    figure, axis = plt.subplots(figsize=(9, 4.5))
    bars = axis.bar(names, values, color=["#4c78a8"] * 6 + ["#f58518"])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Official DREAM Q2 Z-score")
    axis.set_title("Official test-set challenge metrics")
    axis.tick_params(axis="x", rotation=25)
    axis.bar_label(bars, fmt="%.2f", padding=3)
    figure.tight_layout()
    figure.savefig(PLOTS / "test_metrics.png", dpi=160)
    plt.close(figure)


def save_challenge_comparison_plot(
    custom_metrics: dict, reference_metrics: dict | None
) -> None:
    benchmarks = json.loads(PUBLISHED_BENCHMARKS.read_text(encoding="utf-8"))
    names = ["Custom model"]
    scores = [custom_metrics["dream_challenge_score"]]
    colors = ["#f58518"]
    if reference_metrics is not None:
        names.append("Stored reference prediction\n(unattributed)")
        scores.append(reference_metrics["dream_challenge_score"])
        colors.append("#4c78a8")
    display_names = {
        "Published challenge winner": "Published challenge\nwinner",
        "Published all-team model aggregate": "Published all-team\nmodel aggregate",
        "Published post-challenge ensemble": "Published post-challenge\nensemble",
    }
    for result in benchmarks["scores"]:
        names.append(display_names.get(result["name"], result["name"]))
        scores.append(float(result["score"]))
        colors.append("#72b7b2")

    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(names, scores, color=colors)
    axis.set_ylabel("DREAM Q2 mean Z-score (higher is better)")
    axis.set_title("Custom model compared with DREAM Subchallenge 2 scores")
    axis.bar_label(bars, fmt="%.2f", padding=3)
    axis.set_ylim(0, max(scores) * 1.18)
    axis.tick_params(axis="x", labelsize=9)
    figure.tight_layout()
    figure.savefig(PLOTS / "dream_challenge_comparison.png", dpi=160)
    plt.close(figure)


def prediction_frame(cids: list[int], predictions: np.ndarray) -> pd.DataFrame:
    rows = []
    for index, cid in enumerate(cids):
        for target_index, target in enumerate(TARGETS):
            descriptor = (
                target.upper().replace("_", "/")
                if target
                in {"intensity_strength", "valence_pleasantness", "ammonia_urinous"}
                else target.upper()
            )
            if target == "valence_pleasantness":
                descriptor = "VALENCE/PLEASANTNESS"
            elif target == "intensity_strength":
                descriptor = "INTENSITY/STRENGTH"
            elif target == "ammonia_urinous":
                descriptor = "AMMONIA/URINOUS"
            rows.append(
                {
                    "#oID": cid,
                    "descriptor": descriptor,
                    "value": float(predictions[index, target_index]),
                    "sigma": float(predictions[index, len(TARGETS) + target_index]),
                }
            )
    return pd.DataFrame(rows)


def write_metrics(path: Path, metrics: dict) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    seed_everything()
    (
        arrays,
        raw_targets,
        test_cids,
        tokenizer,
        feature_scaler,
        target_scaler,
        features,
        fingerprint_columns,
        target_columns,
    ) = prepare_data()
    for path in (CHECKPOINTS, PLOTS, METRICS):
        path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using {device}; training CIDs {len(arrays['train'][0])}; "
        f"validation CIDs {len(arrays['validation'][0])}; "
        f"official test CIDs {len(test_cids)}"
    )

    seed_everything()
    model, history, best_epoch = train_model(
        arrays,
        raw_targets,
        len(tokenizer.token_to_id),
        len(features),
        target_scaler,
        device,
    )
    print(f"Using best validation checkpoint from epoch {best_epoch}")
    tokenizer.save(CHECKPOINTS / "smiles_vocabulary.json")
    joblib.dump(feature_scaler, CHECKPOINTS / "feature_scaler.joblib")
    joblib.dump(target_scaler, CHECKPOINTS / "target_scaler.joblib")
    test_prediction = predict_values(model, arrays["test"], target_scaler, device)
    test_metrics = dream_challenge_metrics(raw_targets["test"], test_prediction)
    test_metrics["evaluation_set"] = "DREAM official TestSet.txt (69 molecules)"
    test_metrics["prediction_targets"] = (
        "21 high-intensity means and 21 standard deviations"
    )

    reference_metrics = None
    if REFERENCE_PREDICTIONS.is_file():
        reference_prediction = prediction_matrix_from_submission(
            REFERENCE_PREDICTIONS, test_cids
        )
        reference_metrics = dream_challenge_metrics(
            raw_targets["test"], reference_prediction
        )
        reference_metrics["source"] = str(REFERENCE_PREDICTIONS.relative_to(ROOT))
        reference_metrics["evaluation_set"] = (
            "DREAM official TestSet.txt (69 molecules)"
        )
        write_metrics(
            ROOT
            / "results/baseline/dream_challenge/official_testset_reference_score.json",
            reference_metrics,
        )

    prediction_frame(test_cids, test_prediction).to_csv(
        METRICS / "custom_model_test_predictions.tsv", sep="\t", index=False
    )
    write_metrics(METRICS / "dream_challenge_test_metrics.json", test_metrics)
    save_test_metrics_plot(test_metrics)
    save_challenge_comparison_plot(test_metrics, reference_metrics)
    pd.DataFrame(
        [
            {
                "epoch": index + 1,
                "dream_challenge_score": values["dream_challenge_score"],
                **{
                    f"pearson_{key}": val
                    for key, val in values["pearson_correlations"].items()
                },
                **{f"z_{key}": val for key, val in values["z_scores"].items()},
            }
            for index, values in enumerate(history["validation_metrics"])
        ]
    ).to_csv(METRICS / "validation_dream_challenge_metrics.csv", index=False)
    save_training_plots(history)
    torch.save(
        {
            "model_name": "CustomOdorModel",
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "descriptor_columns": features,
            "morgan_fingerprint_columns": fingerprint_columns,
            "morgan_radius": 2,
            "morgan_fingerprint_size": len(fingerprint_columns),
            "morgan_include_chirality": True,
            "target_columns": target_columns,
            "target_names": TARGETS,
            "vocabulary_size": len(tokenizer.token_to_id),
            "max_sequence_length": MAX_LENGTH,
            "prediction_outputs": 42,
        },
        CHECKPOINTS / "custom_model.pt",
    )

    print(
        f"Official DREAM Q2 test Z-score: {test_metrics['dream_challenge_score']:.3f}"
    )
    print(f"Official test metrics: {METRICS / 'dream_challenge_test_metrics.json'}")
    print(f"Training plots: {PLOTS}")
    print(f"Checkpoints: {CHECKPOINTS}")


if __name__ == "__main__":
    main()
