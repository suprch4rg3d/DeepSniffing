"""Ask the decoder for an odor and rank what it writes.

    poetry run python -m generation.judge --words fruity green --count 500 --top 10
    poetry run python -m generation.judge --words fruity --pleasant 70 --strong 60

Every candidate is scored by the two judges, which never see the decoder:
- the oracle: the probability of each requested odor word,
- the student, a lightweight stand-in for Q2 trained on the DREAM ratings: its prediction of the DREAM attribute
  that matches a requested word (fruit for fruity, flower for floral, ...), and of pleasantness and intensity
  when you ask for them.
The score is the mean of the parts that were asked for, so a molecule ranks high only if it meets all of them.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from scripts.core.metrics import TARGETS

from .dataset import load_dataset
from .generator import load_generator, sample
from .oracle import OraclePredictor
from .student import StudentPredictor

# an odor word of the decoder and the DREAM attribute that measures the same thing
DREAM_ATTRIBUTE = {
    "fruity": "fruit",
    "floral": "flower",
    "sweet": "sweet",
    "alliaceous": "garlic",
    "fishy": "fish",
    "spicy": "spices",
    "woody": "wood",
    "grassy": "grass",
    "green": "grass",
    "burnt": "burnt",
    "acidic": "acid",
    "sour": "sour",
    "musk": "musky",
    "sweaty": "sweaty",
    "chemical": "chemical",
    "ammoniacal": "ammonia_urinous",
    "bready": "bakery",
    "warm": "warm",
    "cooling": "cold",
}
FUNCTIONAL_GROUPS = {
    "ester": "[CX3](=O)[OX2][#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "alcohol": "[OX2H][CX4]",
    "acid": "[CX3](=O)[OX2H1]",
    "thiol": "[SX2H]",
    "sulfide": "[#6][SX2][#6]",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "ring": "[R]",
}


def describe(smiles: str) -> dict:
    """Size, polarity, logP and the functional groups of a molecule."""
    molecule = Chem.MolFromSmiles(smiles)
    return {
        "heavy_atoms": molecule.GetNumAtoms(),
        "mol_weight": round(Descriptors.MolWt(molecule), 1),
        "logP": round(Descriptors.MolLogP(molecule), 2),
        "polar_surface_area": round(rdMolDescriptors.CalcTPSA(molecule), 1),
        "groups": [
            n
            for n, smarts in FUNCTIONAL_GROUPS.items()
            if molecule.HasSubstructMatch(Chem.MolFromSmarts(smarts))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write molecules for an odor and rank them.")
    parser.add_argument("--words", nargs="*", default=[], help="odor words to ask for")
    parser.add_argument(
        "--pleasant", type=float, default=None, help="wanted pleasantness, 0 to 100"
    )
    parser.add_argument("--strong", type=float, default=None, help="wanted intensity, 0 to 100")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--top", type=int, default=10)
    arguments = parser.parse_args()

    torch.manual_seed(7)
    model, vocabulary = load_generator()
    unknown = [w for w in arguments.words if w not in vocabulary.words]
    if unknown:
        raise SystemExit(
            f"Unknown odor words {unknown}. The words are: {', '.join(vocabulary.words)}"
        )
    oracle, student = OraclePredictor(), StudentPredictor()
    known = set(load_dataset()[0]["smiles"])

    written = sorted({s for s in sample(model, vocabulary, arguments.words, arguments.count) if s})
    print(f"{len(written)} different valid molecules written for {arguments.words or 'no request'}")

    columns = [oracle.words.index(w) for w in arguments.words]
    word_scores = (
        oracle.probabilities(written)[:, columns] if columns else np.zeros((len(written), 0))
    )
    profile = student.predict(written)
    parts = [word_scores[:, i] for i in range(word_scores.shape[1])]
    for word in arguments.words:
        if word in DREAM_ATTRIBUTE:
            parts.append(profile[:, TARGETS.index(DREAM_ATTRIBUTE[word])] / 100.0)
    if arguments.pleasant is not None:
        parts.append(
            1.0
            - np.abs(profile[:, TARGETS.index("valence_pleasantness")] - arguments.pleasant) / 100.0
        )
    if arguments.strong is not None:
        parts.append(
            1.0 - np.abs(profile[:, TARGETS.index("intensity_strength")] - arguments.strong) / 100.0
        )
    score = np.mean(parts, axis=0) if parts else np.zeros(len(written))

    for rank, index in enumerate(np.argsort(-score)[: arguments.top], start=1):
        smiles = written[index]
        print(
            f"\n{rank}. {smiles}   score {score[index]:.2f}   {'known molecule' if smiles in known else 'NEW'}"
        )
        print(
            "   words asked: "
            + ", ".join(f"{w} {word_scores[index, i]:.2f}" for i, w in enumerate(arguments.words))
        )
        print(
            f"   student: pleasantness {profile[index, TARGETS.index('valence_pleasantness')]:.0f}, "
            f"intensity {profile[index, TARGETS.index('intensity_strength')]:.0f}"
        )
        print(f"   features: {describe(smiles)}")


if __name__ == "__main__":
    main()
