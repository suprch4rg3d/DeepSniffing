"""How well the decoder follows an odor request.

1. Chemistry: how many of the written molecules are valid, different from each other, and new.
2. Does the request matter? For a list of requests, the oracle's score for the requested words and the share of
   molecules with an expected substructure, against molecules written with no request.
3. Molecules the decoder never saw (the test split and the DREAM molecules): ask for their own odor words and see
   whether something similar is written, against asking with another molecule's words.

Run:  poetry run python -m generation.evaluate
"""

from __future__ import annotations

import json
import random

import numpy as np
from rdkit import Chem

from .chemistry import morgan_bitvector, seed_everything, tanimoto
from .dataset import load_dataset
from .generator import load_generator, sample
from .oracle import OraclePredictor
from .paths import RESULTS

SAMPLES = 400
RETRO_SAMPLES = 100
RETRO_MOLECULES = 80
# a request, and a substructure that molecules written for it should often have (None when there is no simple one)
REQUESTS = {
    "fruity": "[CX3](=O)[OX2][#6]",  # an ester
    "alliaceous": "[#16]",
    "sulfurous": "[#16]",
    "garlic;onion": "[#16]",
    "fishy": "[#7]",
    "buttery": "[#6][CX3](=O)[#6]",
    "floral": None,
    "woody": None,
    "mint": None,
    "smoky;phenolic": None,
    "lemon;citrus": None,
    "cheesy": None,
    "fruity;green": None,
    "sweet;vanilla": None,
    "fatty;waxy": None,
    "green;fresh": None,
}


def novelty(samples: list[str | None], known: set[str]) -> dict:
    valid = [s for s in samples if s]
    unique = set(valid)
    return {
        "valid": len(valid) / len(samples),
        "unique": len(unique) / max(len(valid), 1),
        "new": len([s for s in unique if s not in known]) / max(len(unique), 1),
    }


def share_with(smiles_list: list[str], smarts: str | None) -> float | None:
    if smarts is None or not smiles_list:
        return None
    pattern = Chem.MolFromSmarts(smarts)
    return float(np.mean([Chem.MolFromSmiles(s).HasSubstructMatch(pattern) for s in smiles_list]))


def main() -> None:
    seed_everything()
    RESULTS.mkdir(exist_ok=True)
    frame, _ = load_dataset()
    model, vocabulary = load_generator()
    oracle = OraclePredictor()
    train_known = set(frame.loc[frame["split"] == "train", "smiles"])
    word_index = {w: i for i, w in enumerate(oracle.words)}
    report: dict = {}

    unconditional = [s for s in sample(model, vocabulary, [], SAMPLES * 2) if s]
    report["chemistry_without_request"] = novelty(
        sample(model, vocabulary, [], SAMPLES), train_known
    )
    baseline = oracle.probabilities(unconditional)
    print("chemistry without a request:", report["chemistry_without_request"])

    requests = {}
    for request, smarts in REQUESTS.items():
        asked = request.split(";")
        if any(w not in vocabulary.words for w in asked):
            print(f"skipped {request}: not in the vocabulary")
            continue
        written = sample(model, vocabulary, asked, SAMPLES)
        valid = [s for s in written if s]
        columns = [word_index[w] for w in asked]
        entry = {
            **novelty(written, train_known),
            "oracle_score_for_asked_words": float(oracle.probabilities(valid)[:, columns].mean()),
            "oracle_score_without_request": float(baseline[:, columns].mean()),
            "share_with_expected_substructure": share_with(valid, smarts),
            "share_with_it_without_request": share_with(unconditional, smarts),
            "examples": sorted(set(valid), key=len)[:6],
        }
        requests[request] = entry
        print(
            f"{request:16s} oracle {entry['oracle_score_for_asked_words']:.2f} against "
            f"{entry['oracle_score_without_request']:.2f} without a request; substructure "
            f"{entry['share_with_expected_substructure']} against {entry['share_with_it_without_request']}"
        )
    report["requests"] = requests

    held_out = {}
    for split in ("test", "dream"):
        rows = frame[(frame["split"] == split) & (frame["word_list"].map(len) >= 2)]
        rows = rows.sample(min(RETRO_MOLECULES, len(rows)), random_state=42)
        pool = list(rows["word_list"])
        own, other, own_oracle, other_oracle = [], [], [], []
        for position, (smiles, word_list) in enumerate(
            zip(rows["smiles"], rows["word_list"], strict=False)
        ):
            target = morgan_bitvector(smiles)
            asked = random.sample(word_list, min(5, len(word_list)))
            decoy = random.choice(pool[:position] + pool[position + 1 :])
            decoy = random.sample(decoy, min(5, len(decoy)))
            for request, best, scores in ((asked, own, own_oracle), (decoy, other, other_oracle)):
                written = sorted(
                    {s for s in sample(model, vocabulary, request, RETRO_SAMPLES) if s}
                )
                if not written:
                    best.append(0.0)
                    continue
                best.append(max(tanimoto(target, morgan_bitvector(s)) for s in written))
                scores.append(
                    float(
                        oracle.probabilities(written)[:, [word_index[w] for w in word_list]].mean()
                    )
                )
        held_out[split] = {
            "molecules": len(own),
            "best_similarity_asking_for_its_own_words": float(np.mean(own)),
            "best_similarity_asking_for_other_words": float(np.mean(other)),
            "share_above_0.5_own": float(np.mean(np.array(own) > 0.5)),
            "share_above_0.5_other": float(np.mean(np.array(other) > 0.5)),
            "oracle_score_of_its_words_own_request": float(np.mean(own_oracle)),
            "oracle_score_of_its_words_other_request": float(np.mean(other_oracle)),
        }
        print(split, held_out[split])
    report["held_out_molecules"] = held_out
    (RESULTS / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
