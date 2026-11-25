"""What the generator writes for one odor beside the molecules that carry it.

Issue #10. For one odor token, the molecules the model writes and the molecules a source actually named with that
token, compared by weight and by Morgan similarity.

    python -m scripts.analysis.generated_vs_real [--samples 40]

Writes ``output/analysis/generated_vs_real.json``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity

from scripts.core import evaluate
from scripts.core.training import OUTPUT_DIR, load_model, seed_everything
from scripts.core.vocabulary import Vocabulary
from scripts.preprocessing.odor_tables import DATASET_PATH

RDLogger.DisableLog("rdApp.*")

REQUESTS = ("fruity", "musk")
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprints(smiles: list[str]) -> list:
    prints = []
    for text in smiles:
        molecule = Chem.MolFromSmiles(text)
        if molecule is not None:
            prints.append(GENERATOR.GetFingerprint(molecule))
    return prints


def nearest(smiles: list[str], references: list) -> list[float]:
    values = []
    for text in smiles:
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            continue
        values.append(
            float(max(BulkTanimotoSimilarity(GENERATOR.GetFingerprint(molecule), references)))
        )
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generated against real molecules for one odor.")
    parser.add_argument("--samples", type=int, default=40)
    arguments = parser.parse_args(argv)

    seed_everything()
    vocabulary = Vocabulary.load()
    model = load_model(vocabulary, OUTPUT_DIR / "stage2/best.pt")
    table = pd.read_csv(DATASET_PATH).fillna("")

    report = {"samples_per_request": arguments.samples, "requests": {}}
    for request in REQUESTS:
        carried = table[
            table["odor_tokens"].apply(lambda text, token=request: token in str(text).split(";"))
            & (table["split"] == "train")
        ]
        real = sorted(carried["smiles"])
        written = [
            smiles
            for smiles in evaluate.write_molecules(model, vocabulary, (request,), arguments.samples)
            if smiles
        ]
        unique = sorted(set(written))
        real_prints = fingerprints(real)
        all_prints = fingerprints(sorted(table.loc[table["split"] == "train", "smiles"]))
        weights_real = [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in real]
        weights_written = [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in unique]
        to_real = nearest(unique, real_prints)
        report["requests"][request] = {
            "real_molecules": len(real),
            "generated": arguments.samples,
            "valid": len(written) / arguments.samples,
            "unique": len(unique) / max(len(written), 1),
            "similarity_to_real": {
                "median": float(np.median(to_real)),
                "above_0.5": float(np.mean([value >= 0.5 for value in to_real])),
            },
            "similarity_to_any_training": {"median": float(np.median(nearest(unique, all_prints)))},
            "similarity_to_real_values": [round(v, 4) for v in to_real],
            "similarity_to_any_values": [round(v, 4) for v in nearest(unique, all_prints)],
            "weights_generated": [round(v, 1) for v in weights_written],
            "weights_real": [round(v, 1) for v in weights_real],
            "weight": {
                "generated_median": float(np.median(weights_written)),
                "real_median": float(np.median(weights_real)),
            },
            "examples": unique[:8],
            "real_examples": real[:8],
        }
        print(
            f"{request}: {len(real)} real, {len(unique)} written, "
            f"median similarity {report['requests'][request]['similarity_to_real']['median']:.2f}"
        )

    folder = OUTPUT_DIR.parent / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "generated_vs_real.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
