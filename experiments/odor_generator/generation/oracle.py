"""Judge 1, the oracle: the probability of every odor word for any molecule.

A small network on Morgan counts and RDKit descriptors, which RDKit computes for any molecule. It learns
from the odor words of Leffingwell and The Good Scents Company, on the labeled molecules of the training split,
stops on the validation split, and is scored on the test split. The DREAM molecules are never trained on, and
are only another test. It is a different kind of model from the decoder, so it can judge what the decoder writes.
It is used by ``judge`` and ``evaluate``.

Run:  poetry run python -m generation.oracle
"""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .chemistry import finite, molecule_features, seed_everything
from .dataset import load_dataset
from .paths import ORACLE, RESULTS

HIDDEN, DROPOUT = (512, 256), 0.4
LEARNING_RATE, WEIGHT_DECAY, BATCH, MAX_EPOCHS, PATIENCE = 1e-3, 1e-2, 128, 200, 15


class Oracle(torch.nn.Module):
    def __init__(self, width: int, outputs: int):
        super().__init__()
        sizes = (width, *HIDDEN)
        layers: list[torch.nn.Module] = []
        for before, after in zip(sizes[:-1], sizes[1:], strict=False):
            layers += [torch.nn.Linear(before, after), torch.nn.ReLU(), torch.nn.Dropout(DROPOUT)]
        self.net = torch.nn.Sequential(*layers, torch.nn.Linear(sizes[-1], outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OraclePredictor:
    """Scores SMILES strings: ``probabilities`` gives one probability for each odor word in ``words``."""

    def __init__(self, path=ORACLE):
        saved = torch.load(path, weights_only=False)
        self.words = saved["words"]
        self.low, self.high = saved["low"], saved["high"]
        self.mean, self.spread = saved["mean"], saved["spread"]
        self.model = Oracle(saved["width"], len(self.words))
        self.model.load_state_dict(saved["state"])
        self.model.eval()

    def probabilities(self, smiles_list: list[str]) -> np.ndarray:
        counts, descriptors, _ = molecule_features(smiles_list)
        scaled = (np.clip(finite(descriptors), self.low, self.high) - self.mean) / self.spread
        features = torch.tensor(np.hstack([counts, scaled]).astype(np.float32))
        with torch.no_grad():
            return torch.sigmoid(self.model(features)).numpy()


def main() -> None:
    seed_everything()
    frame, words = load_dataset()
    labeled = frame[frame["labeled"]]
    targets = np.array(
        [[float(w in row) for w in words] for row in labeled["word_list"]], dtype=np.float32
    )
    counts, descriptors, readable = molecule_features(labeled["smiles"].tolist())
    readable = np.array(readable)
    masks = {
        split: (labeled["split"] == split).to_numpy() & readable
        for split in ("train", "validation", "test", "dream")
    }

    clean = finite(descriptors)
    low, high = (
        np.percentile(clean[masks["train"]], 1, axis=0),
        np.percentile(clean[masks["train"]], 99, axis=0),
    )
    clipped = np.clip(clean, low, high)
    mean, spread = clipped[masks["train"]].mean(0), clipped[masks["train"]].std(0)
    spread = np.where(spread > 0, spread, 1.0)
    features = torch.tensor(np.hstack([counts, (clipped - mean) / spread]).astype(np.float32))
    labels = torch.tensor(targets)

    model = Oracle(features.shape[1], len(words))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_index = np.flatnonzero(masks["train"])

    def mean_auc(mask: np.ndarray) -> float:
        model.eval()
        with torch.no_grad():
            predicted = torch.sigmoid(model(features[mask])).numpy()
        truth = targets[mask]
        usable = [j for j in range(len(words)) if 0 < truth[:, j].sum() < len(truth)]
        return float(np.mean([roc_auc_score(truth[:, j], predicted[:, j]) for j in usable]))

    best, state, stale = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = np.random.RandomState(epoch).permutation(train_index)
        for start in range(0, len(order), BATCH):
            batch = order[start : start + BATCH]
            optimizer.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(
                model(features[batch]), labels[batch]
            ).backward()
            optimizer.step()
        score = mean_auc(masks["validation"])
        if score > best + 1e-4:
            best, stale, state = score, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(state)
    report = {
        "train_molecules": int(masks["train"].sum()),
        "validation_auc": best,
        "test_auc": mean_auc(masks["test"]),
        "dream_molecules_auc": mean_auc(masks["dream"]),
        "epochs": epoch + 1,
    }
    print(report)
    torch.save(
        {
            "state": state,
            "width": features.shape[1],
            "words": words,
            "low": low,
            "high": high,
            "mean": mean,
            "spread": spread,
        },
        ORACLE,
    )
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "oracle.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
