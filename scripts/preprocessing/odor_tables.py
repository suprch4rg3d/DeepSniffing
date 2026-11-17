"""A step of ``ds-build``: the tables that the two training stages read, and the vocabulary.

Stage 1 teaches the decoder the language of molecules and needs no odor words. Stage 2 teaches it the
words. Both read SELFIES, a text form of a molecule in which every sequence of symbols is a valid
molecule. A molecule is written with its stereochemistry, so a mirror image has other symbols.

    structures.csv       stage 1: cid, smiles, selfies, source, split (train or validation)
    odor_dataset.csv     stage 2: cid, smiles, selfies, words, sources, split (train, validation, test, dream)
    vocabulary.json      every token of the model: special tokens, SELFIES symbols and odor words

Rules, all about leakage:

- The 480 molecules of the DREAM challenge are held out of everything the model learns from. They are in
  ``odor_dataset.csv`` as split ``dream``, which is a test only, and not in ``structures.csv``.
- The odor molecules are split once, at random and by CID, with a fixed seed. A mirror image has its own
  CID, so the two forms of a molecule can fall on different sides, which is meant.
- The validation and test molecules of stage 2 are left out of ``structures.csv`` too, so that stage 1
  never sees the structure it will be tested on.
- Nothing is dropped for its size, its elements or a rare symbol.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem, RDLogger

from scripts.data.prepare_data import PYRFUME_DIR
from scripts.preprocessing.odor_words import MOLECULES_PATH as ODOR_MOLECULES_PATH
from scripts.preprocessing.odor_words import ODOR_DIR, known_structures

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRUCTURES_PATH = ODOR_DIR / "structures.csv"
DATASET_PATH = ODOR_DIR / "odor_dataset.csv"
VOCABULARY_PATH = ODOR_DIR / "vocabulary.json"
DREAM_MOLECULES_PATH = PYRFUME_DIR / "keller_2016/molecules.csv"

SEED = 42
VALIDATION_FRACTION = TEST_FRACTION = 0.05  # of the odor molecules that are not DREAM molecules
STRUCTURE_VALIDATION_FRACTION = 0.02  # of the stage 1 molecules
# <odor> starts a text that goes from odor words to a molecule, <mol> one that goes from a molecule to words.
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<sep>", "<odor>", "<mol>")


def canonical_smiles(smiles: object) -> str | None:
    """The canonical SMILES of a molecule with its stereochemistry, or None if RDKit cannot read it."""
    if not isinstance(smiles, str):
        return None
    molecule = Chem.MolFromSmiles(smiles)
    return None if molecule is None else Chem.MolToSmiles(molecule)


def to_selfies(smiles: str) -> str | None:
    """The SELFIES of a SMILES as one string of symbols (``[C][C][O]``), or None if it cannot be converted."""
    try:
        return sf.encoder(smiles)
    except Exception:
        return None


def symbols(selfies: str) -> list[str]:
    return list(sf.split_selfies(selfies))


def dream_cids() -> set[int]:
    return set(pd.read_csv(DREAM_MOLECULES_PATH)["CID"].astype(int))


def with_selfies(table: pd.DataFrame) -> pd.DataFrame:
    """Add the SELFIES column. A molecule that cannot be converted is reported and left out."""
    table = table.assign(selfies=table["smiles"].map(to_selfies))
    failed = table["selfies"].isna()
    if failed.any():
        print(f"{int(failed.sum())} molecules cannot be written in SELFIES and are left out.")
    return table[~failed].reset_index(drop=True)


def build_odor_dataset(odor: pd.DataFrame, held_out: set[int]) -> pd.DataFrame:
    """Stage 2: the molecules with odor words, split into train, validation, test and the DREAM molecules."""
    table = with_selfies(odor[["cid", "smiles", "words", "sources"]])
    split = np.full(len(table), "train", dtype=object)
    free = np.flatnonzero(~table["cid"].isin(held_out).to_numpy())
    free = np.random.RandomState(SEED).permutation(free)
    validation, test = int(VALIDATION_FRACTION * len(free)), int(TEST_FRACTION * len(free))
    split[free[:validation]] = "validation"
    split[free[validation : validation + test]] = "test"
    split[table["cid"].isin(held_out).to_numpy()] = "dream"
    return table.assign(split=split)


def build_structures(odor_dataset: pd.DataFrame, held_out: set[int]) -> pd.DataFrame:
    """Stage 1: every molecule we have, except the DREAM molecules and the stage 2 validation and test molecules."""
    odor_cids = set(odor_dataset["cid"])
    rows = []
    for cid, smiles in known_structures().items():
        smiles = canonical_smiles(smiles)
        if smiles is not None:
            rows.append(
                {"cid": cid, "smiles": smiles, "source": "odor" if cid in odor_cids else "other"}
            )
    # the odor molecules come with their own cleaned structure
    odor_rows = odor_dataset[["cid", "smiles"]].assign(source="odor")
    table = pd.concat([pd.DataFrame(rows, columns=["cid", "smiles", "source"]), odor_rows])
    table = table.drop_duplicates("cid", keep="last")
    unseen = set(odor_dataset.loc[odor_dataset["split"].isin(["validation", "test"]), "cid"])
    table = table[~table["cid"].isin(held_out | unseen)].sort_values("cid", ignore_index=True)
    table = with_selfies(table)
    split = np.full(len(table), "train", dtype=object)
    chosen = np.random.RandomState(SEED).permutation(len(table))
    split[chosen[: int(STRUCTURE_VALIDATION_FRACTION * len(table))]] = "validation"
    return table.assign(split=split)[["cid", "smiles", "selfies", "source", "split"]]


def build_vocabulary(structures: pd.DataFrame, odor_dataset: pd.DataFrame) -> dict[str, list[str]]:
    """Every token the model can read or write. Fixed before stage 1, so stage 2 never changes the model's shape."""
    found: set[str] = set()
    for selfies in [*structures["selfies"], *odor_dataset["selfies"]]:
        found.update(symbols(selfies))
    words: set[str] = set()
    for text in odor_dataset["words"]:
        words.update(text.split(";"))
    return {
        "special": list(SPECIAL_TOKENS),
        "symbols": sorted(found),
        "words": sorted(words),
    }


def main() -> None:
    if not ODOR_MOLECULES_PATH.is_file():
        raise FileNotFoundError(f"{ODOR_MOLECULES_PATH} not found. Run ds-build.")
    held_out = dream_cids()
    odor_dataset = build_odor_dataset(pd.read_csv(ODOR_MOLECULES_PATH).fillna(""), held_out)
    structures = build_structures(odor_dataset, held_out)
    vocabulary = build_vocabulary(structures, odor_dataset)

    structures.to_csv(STRUCTURES_PATH, index=False)
    odor_dataset.to_csv(DATASET_PATH, index=False)
    VOCABULARY_PATH.write_text(json.dumps(vocabulary, indent=1), encoding="utf-8")
    print(f"Saved {len(structures)} molecules for stage 1 to {STRUCTURES_PATH}")
    print(structures["split"].value_counts().to_string())
    print(f"Saved {len(odor_dataset)} molecules with odor words for stage 2 to {DATASET_PATH}")
    print(odor_dataset["split"].value_counts().to_string())
    print(
        f"Saved the vocabulary to {VOCABULARY_PATH}: {len(vocabulary['special'])} special tokens, "
        f"{len(vocabulary['symbols'])} SELFIES symbols, {len(vocabulary['words'])} odor words"
    )


if __name__ == "__main__":
    main()
