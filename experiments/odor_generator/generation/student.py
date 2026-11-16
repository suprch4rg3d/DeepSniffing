"""Judge 2, the student: a lightweight stand-in for Q2 that can read any molecule.

The full Q2 model needs Dragon descriptors, EPI Suite values, NSPDK similarities and odor-name windows, which
exist only for the 476 DREAM molecules. The student reads what any structure has (Morgan counts and RDKit
descriptors) and predicts the 21 DREAM population means (intensity, pleasantness and 19 descriptors, on the
0 to 100 scale) from the DREAM ratings of the training molecules. It does not use the weights of Q2. It
is scored on the DREAM validation and test molecules with the official metric, so that the loss of accuracy
against Q2 can be read off. The spreads are not predicted.

This is the only module that reads the DREAM data. It uses the project's own helpers (``scripts.core``).

Run:  poetry run python -m generation.student
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from scripts.core.data import aggregate_high_intensity_targets, read_official_leaderboard_targets
from scripts.core.metrics import OFFICIAL_NULL_SD, TARGETS, normalize_target_name
from scripts.core.splits import TRAIN_RATINGS_PATH

from .chemistry import finite, molecule_features, seed_everything
from .paths import DREAM_MOLECULES, DREAM_TEST_GOLD, RESULTS, STUDENT

MEAN_COLUMNS = [f"mean_{target}" for target in TARGETS]
LEARNING_RATE, WEIGHT_DECAY, BATCH, MAX_EPOCHS, PATIENCE, DROPOUT = 1e-3, 1e-2, 32, 300, 25, 0.3
FULL_Q2_TEST = {
    "intensity_mean": 6.69,
    "valence_mean": 5.45,
    "descriptors_mean": 21.25,
}  # the saved Q2, for reference


class Student(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(256, len(TARGETS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StudentPredictor:
    """The 21 predicted DREAM means (0 to 100) of any molecule, in the order of ``TARGETS``."""

    def __init__(self, path=STUDENT):
        self.saved = torch.load(path, weights_only=False)
        self.model = Student(self.saved["width"])
        self.model.load_state_dict(self.saved["state"])
        self.model.eval()

    def predict(self, smiles_list: list[str]) -> np.ndarray:
        counts, descriptors, _ = molecule_features(smiles_list)
        scaled = (
            np.clip(finite(descriptors), self.saved["low"], self.saved["high"]) - self.saved["mean"]
        ) / self.saved["spread"]
        features = torch.tensor(np.hstack([counts, scaled]).astype(np.float32))
        with torch.no_grad():
            standardized = self.model(features).numpy()
        return np.clip(
            standardized * self.saved["target_spread"] + self.saved["target_mean"], 0.0, 100.0
        )


def dream_targets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The 21 mean ratings of the training, validation and official test molecules, by CID."""
    train = aggregate_high_intensity_targets(TRAIN_RATINGS_PATH).set_index("CID")[MEAN_COLUMNS]
    # intensity is learned from the ratings at 1/1000, as in the challenge
    ratings = pd.read_csv(TRAIN_RATINGS_PATH, sep="\t").rename(
        columns={"Compound Identifier": "CID"}
    )
    ratings["Dilution"] = ratings["Dilution"].str.strip()
    column = next(c for c in ratings.columns if normalize_target_name(c) == "intensity_strength")
    at_standard = (
        ratings[ratings["Dilution"] == "1/1,000"]
        .groupby(["CID", "subject #"])[column]
        .mean()
        .groupby("CID")
        .mean()
    )
    train[f"mean_{TARGETS[0]}"] = at_standard.reindex(train.index)
    validation = read_official_leaderboard_targets().set_index("CID")[MEAN_COLUMNS]
    gold = pd.read_csv(DREAM_TEST_GOLD, sep="\t")
    gold["descriptor"] = gold["descriptor"].map(normalize_target_name)
    test = gold.pivot(index="#oID", columns="descriptor", values="value")[list(TARGETS)]
    test.columns = MEAN_COLUMNS
    return train, validation, test


def dream_scores(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """The Z-scores of the three mean groups, as the official metric computes them."""
    correlation = [np.corrcoef(predicted[:, a], truth[:, a])[0, 1] for a in range(len(TARGETS))]
    return {
        "intensity_mean": correlation[0] / OFFICIAL_NULL_SD["intensity_mean"],
        "valence_mean": correlation[1] / OFFICIAL_NULL_SD["valence_mean"],
        "descriptors_mean": float(np.mean(correlation[2:])) / OFFICIAL_NULL_SD["descriptors_mean"],
    }


def main() -> None:
    torch.set_num_threads(2)
    seed_everything()
    smiles_of = pd.read_csv(DREAM_MOLECULES, usecols=["CID", "rdkit_canonical_smiles"]).set_index(
        "CID"
    )["rdkit_canonical_smiles"]
    train, validation, test = dream_targets()
    parts = {}
    for name, table in (("train", train), ("validation", validation), ("test", test)):
        counts, descriptors, _ = molecule_features(smiles_of.loc[table.index].tolist())
        parts[name] = (counts, finite(descriptors), table.to_numpy(dtype=np.float64))

    low, high = (
        np.percentile(parts["train"][1], 1, axis=0),
        np.percentile(parts["train"][1], 99, axis=0),
    )
    clipped_train = np.clip(parts["train"][1], low, high)
    mean, spread = clipped_train.mean(0), clipped_train.std(0)
    spread = np.where(spread > 0, spread, 1.0)
    target_mean, target_spread = (
        np.nanmean(parts["train"][2], axis=0),
        np.nanstd(parts["train"][2], axis=0),
    )

    def features(name: str) -> torch.Tensor:
        counts, descriptors, _ = parts[name]
        return torch.tensor(
            np.hstack([counts, (np.clip(descriptors, low, high) - mean) / spread]).astype(
                np.float32
            )
        )

    x_train, raw = features("train"), parts["train"][2]
    known = torch.tensor(np.isfinite(raw))
    y_train = torch.tensor(
        np.nan_to_num((raw - target_mean) / target_spread, nan=0.0).astype(np.float32)
    )
    model = Student(x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def predict(name: str) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return np.clip(model(features(name)).numpy() * target_spread + target_mean, 0.0, 100.0)

    best, state, stale = -1e9, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = np.random.RandomState(epoch).permutation(len(x_train))
        for start in range(0, len(order), BATCH):
            batch = order[start : start + BATCH]
            optimizer.zero_grad()
            error = (model(x_train[batch]) - y_train[batch]) ** 2
            (error * known[batch]).sum().div(known[batch].sum()).backward()
            optimizer.step()
        score = sum(dream_scores(predict("validation"), parts["validation"][2]).values())
        if score > best + 1e-3:
            best, stale, state = score, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(state)
    report = {
        "validation": dream_scores(predict("validation"), parts["validation"][2]),
        "test": dream_scores(predict("test"), parts["test"][2]),
        "full_q2_test": FULL_Q2_TEST,
        "epochs": epoch + 1,
    }
    print(json.dumps(report, indent=2))
    torch.save(
        {
            "state": state,
            "width": x_train.shape[1],
            "low": low,
            "high": high,
            "mean": mean,
            "spread": spread,
            "target_mean": target_mean,
            "target_spread": target_spread,
        },
        STUDENT,
    )
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "student.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
