"""The data the two stages learn from, as numbers rather than prose.

    python -m scripts.analysis.eda

Writes ``output/analysis/eda.json``: the size of each split, how many odor tokens exist and how often they
are used, which source names which molecules, and what the molecules weigh.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from scripts.core.training import OUTPUT_DIR
from scripts.core.vocabulary import Vocabulary
from scripts.preprocessing.odor_tables import DATASET_PATH

RDLogger.DisableLog("rdApp.*")


def summarize(table: pd.DataFrame, vocabulary: Vocabulary) -> dict:
    """The counts a reader needs before any result means anything."""
    tokens = [
        token for text in table["odor_tokens"].fillna("") for token in str(text).split(";") if token
    ]
    per_molecule = [
        len([t for t in str(text).split(";") if t]) for text in table["odor_tokens"].fillna("")
    ]
    usage = Counter(tokens)
    sources = Counter(
        source for text in table["sources"].fillna("") for source in str(text).split(";") if source
    )
    weights = []
    for smiles in table["smiles"]:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            weights.append(Descriptors.MolWt(molecule))
    return {
        "molecules": len(table),
        "splits": table["split"].value_counts().to_dict(),
        "odor_tokens": len(vocabulary.odor_tokens),
        "odor_tokens_used": len(usage),
        "odor_tokens_on_one_molecule": sum(1 for count in usage.values() if count == 1),
        "tokens_per_molecule": {
            "median": float(np.median(per_molecule)),
            "mean": float(np.mean(per_molecule)),
            "most": int(np.max(per_molecule)),
        },
        "labels_per_molecule": float(np.mean(per_molecule)),
        "sources": dict(sorted(sources.items(), key=lambda pair: -pair[1])),
        "weight": {
            "median": float(np.median(weights)),
            "share_over_350": float(np.mean([value > 350 for value in weights])),
            "heaviest": float(np.max(weights)),
        },
    }


def main() -> int:
    vocabulary = Vocabulary.load()
    table = pd.read_csv(DATASET_PATH).fillna("")
    report = summarize(table, vocabulary)
    folder = OUTPUT_DIR.parent / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "eda.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
