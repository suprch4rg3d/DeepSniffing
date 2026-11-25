"""A classic baseline for the classifier: ridge regression on Morgan fingerprints.

The transformer's classifier is scored by macro ROC-AUC over the odor tokens of a held-out set. This fits
the simplest model that has been the standard for molecular property prediction for a decade, on the same
molecules, the same fingerprints and the same split, so the report can say what the deep model adds.

    python -m scripts.analysis.baseline [--alpha 1.0]

Writes ``output/analysis/baseline.json`` with the scores of every split.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import Ridge

from scripts.core.metrics import odor_token_metrics
from scripts.core.training import OUTPUT_DIR
from scripts.core.vocabulary import Vocabulary
from scripts.preprocessing.odor_tables import DATASET_PATH

RDLogger.DisableLog("rdApp.*")

SPLITS = ("validation", "test", "dream")
RADIUS, BITS = 2, 2048


def fingerprints(smiles: list[str], radius: int = RADIUS, bits: int = BITS) -> np.ndarray:
    """The Morgan fingerprint of every molecule, as rows of a dense array."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    rows = np.zeros((len(smiles), bits), dtype=np.float32)
    for row, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(text)
        if molecule is not None:
            rows[row] = generator.GetFingerprintAsNumPy(molecule)
    return rows


def targets(table: pd.DataFrame, vocabulary: Vocabulary) -> np.ndarray:
    """One column per odor token, true where the molecule carries it."""
    index = {token: column for column, token in enumerate(vocabulary.odor_tokens)}
    truth = np.zeros((len(table), len(index)), dtype=np.int8)
    for row, tokens in enumerate(table["odor_tokens"].fillna("")):
        for token in str(tokens).split(";"):
            if token:
                truth[row, index[token]] = 1
    return truth


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ridge on Morgan fingerprints, scored like the classifier."
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="the ridge penalty")
    arguments = parser.parse_args(argv)

    vocabulary = Vocabulary.load()
    table = pd.read_csv(DATASET_PATH).fillna("")
    features = fingerprints(list(table["smiles"]))
    truth = targets(table, vocabulary)
    train_rows = (table["split"] == "train").to_numpy()

    model = Ridge(alpha=arguments.alpha)
    model.fit(features[train_rows], truth[train_rows])
    predicted = model.predict(features)

    report = {"alpha": arguments.alpha, "molecules": int(train_rows.sum()), "splits": {}}
    for split in SPLITS:
        rows = (table["split"] == split).to_numpy()
        found = odor_token_metrics(predicted[rows], truth[rows])
        report["splits"][split] = {
            "molecules": int(rows.sum()),
            "roc_auc_macro": found.get("roc_auc_macro"),
            "average_precision_macro": found.get("average_precision_macro"),
            "odor_tokens_scorable": found.get("odor_tokens_scorable"),
        }
        print(
            f"{split:10s} ROC-AUC {found['roc_auc_macro']:.3f}  AP {found['average_precision_macro']:.3f}"
        )

    folder = OUTPUT_DIR.parent / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "baseline.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
