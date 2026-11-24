"""A step of ``ds-build``: the tables that the two training stages read, and the vocabulary.

Stage 1 teaches the decoder the language of molecules and needs no odor tokens. Stage 2 teaches it the
odor tokens. Both read SELFIES, a text form of a molecule in which every sequence of molecule tokens is a valid
molecule. A molecule is written with its stereochemistry, so a mirror image has other molecule tokens.

    structures.csv       stage 1: cid, smiles, selfies, source, split (train or validation)
    odor_dataset.csv     stage 2: cid, smiles, selfies, odor tokens, sources, split (train, validation, test, dream)
    vocabulary.json      every token of the model: special tokens, molecule tokens and odor tokens

Rules, all about leakage:

- The 480 molecules of the DREAM challenge are held out of everything the model learns from. They are in
  ``odor_dataset.csv`` as split ``dream``, which is a test only, and not in ``structures.csv``.
- The odor molecules are split once, at random and by CID, with a fixed seed. A mirror image has its own
  CID, so the two forms of a molecule can fall on different sides, which is meant.
- The validation and test molecules of stage 2 are left out of ``structures.csv`` too, so that stage 1
  never sees the structure it will be tested on.
- Nothing is dropped for its size, its elements or a rare molecule token.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem, RDLogger

from scripts.data.prepare_data import PYRFUME_DIR
from scripts.preprocessing.odor_tokens import MOLECULES_PATH as ODOR_MOLECULES_PATH
from scripts.preprocessing.odor_tokens import SCALE, TABLES_DIR, known_structures
from scripts.preprocessing.pubchem_structures import STRUCTURES_PATH as PUBCHEM_STRUCTURES_PATH
from scripts.preprocessing.pubchem_structures import for_validation

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRUCTURES_PATH = TABLES_DIR / "structures.csv"
DATASET_PATH = TABLES_DIR / "odor_dataset.csv"
VOCABULARY_PATH = TABLES_DIR / "vocabulary.json"
DREAM_MOLECULES_PATH = PYRFUME_DIR / "keller_2016/molecules.csv"

SEED = 42
VALIDATION_FRACTION = TEST_FRACTION = 0.05  # of the odor molecules that are not DREAM molecules
# <odor> starts a text that goes from odor tokens to a molecule, <mol> one that goes from a molecule to odor tokens.
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<sep>", "<odor>", "<mol>")


def canonical_smiles(smiles: object) -> str | None:
    """The canonical SMILES of a molecule with its stereochemistry, or None if RDKit cannot read it."""
    if not isinstance(smiles, str):
        return None
    molecule = Chem.MolFromSmiles(smiles)
    return None if molecule is None else Chem.MolToSmiles(molecule)


def to_selfies(smiles: str) -> str | None:
    """The SELFIES of a SMILES as one string of molecule tokens (``[C][C][O]``), or None if it cannot be converted."""
    try:
        return sf.encoder(smiles)
    except Exception:
        return None


def molecule_tokens(selfies: str) -> list[str]:
    return list(sf.split_selfies(selfies))


def dream_cids() -> set[int]:
    return set(pd.read_csv(DREAM_MOLECULES_PATH)["CID"].astype(int))


def read_pubchem_structures() -> pd.DataFrame:
    """The molecules of PubChem for the pretraining, or none when that step has not been run."""
    if not PUBCHEM_STRUCTURES_PATH.is_file():
        print(
            f"{PUBCHEM_STRUCTURES_PATH.name} not found; the pretraining gets the odor sources only."
        )
        return pd.DataFrame(columns=["cid", "smiles", "selfies", "source"])
    table = pd.read_csv(PUBCHEM_STRUCTURES_PATH)
    print(f"{len(table):,} molecules from PubChem for the pretraining")
    return table.assign(source="pubchem")


def with_selfies(table: pd.DataFrame) -> pd.DataFrame:
    """Add the SELFIES column. A molecule that cannot be converted is reported and left out."""
    table = table.assign(selfies=table["smiles"].map(to_selfies))
    failed = table["selfies"].isna()
    if failed.any():
        print(f"{int(failed.sum())} molecules cannot be written in SELFIES and are left out.")
    return table[~failed].reset_index(drop=True)


def build_odor_dataset(odor: pd.DataFrame, held_out: set[int]) -> pd.DataFrame:
    """Stage 2: the molecules with odor tokens, split into train, validation, test and the DREAM molecules."""
    table = with_selfies(odor[["cid", "smiles", "odor_tokens", "sources"]])
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
    table = table.drop_duplicates("cid", keep="last").sort_values("cid", ignore_index=True)
    table = with_selfies(table)
    # The molecules of PubChem come with their SELFIES already written, because there are too many of them
    # to write here one at a time; ours come first, so a molecule we have cleaned ourselves wins.
    table = pd.concat([table, read_pubchem_structures()], ignore_index=True)

    # Leakage: out go the DREAM molecules and the stage 2 validation and test molecules, by CID and by
    # structure. PubChem gives the same molecule more than one CID (a stereochemistry left unsaid, a salt),
    # so a CID is not enough to keep a held-out molecule out of the pretraining.
    unseen = set(odor_dataset.loc[odor_dataset["split"].isin(["validation", "test"]), "cid"])
    forbidden = odor_dataset.loc[odor_dataset["cid"].isin(held_out | unseen), "smiles"]
    forbidden = {smiles for smiles in forbidden.map(canonical_smiles) if smiles is not None}
    table = table[~table["cid"].isin(held_out | unseen) & ~table["smiles"].isin(forbidden)]
    # by CID, as everywhere else: two CIDs of one structure stay two molecules, and are not merged here
    table = table.drop_duplicates("cid").reset_index(drop=True)

    # The validation molecules are the same ones in every build, picked out of the whole pool by the
    # checksum of the CID (``pubchem_structures.for_validation``). Shuffling the table instead, as this
    # did before, gave different molecules whenever the data changed, and then a loss from one run could
    # not be set beside a loss from another.
    validation = np.array([for_validation(int(cid)) for cid in table["cid"]])
    split = np.where(validation, "validation", "train").astype(object)
    return table.assign(split=split)[["cid", "smiles", "selfies", "source", "split"]]


def build_vocabulary(structures: pd.DataFrame, odor_dataset: pd.DataFrame) -> dict[str, list[str]]:
    """Every token the model can read or write. Fixed before stage 1, so stage 2 never changes the model's shape."""
    found: set[str] = set()
    for selfies in [*structures["selfies"], *odor_dataset["selfies"]]:
        found.update(molecule_tokens(selfies))
    odor_tokens: set[str] = set()
    for text in odor_dataset["odor_tokens"]:
        odor_tokens.update(text.split(";"))
    return {
        "special": list(SPECIAL_TOKENS),
        "molecule_tokens": sorted(found),
        "odor_tokens": sorted(odor_tokens),
    }


def main() -> None:
    if SCALE != "full":
        raise RuntimeError(
            "This builds the full tables from every source; the dev ones are cut from them by "
            "scripts.preprocessing.odor_dev. Unset DS_SCALE to run this."
        )
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
    print(f"Saved {len(odor_dataset)} molecules with odor tokens for stage 2 to {DATASET_PATH}")
    print(odor_dataset["split"].value_counts().to_string())
    print(
        f"Saved the vocabulary to {VOCABULARY_PATH}: {len(vocabulary['special'])} special tokens, "
        f"{len(vocabulary['molecule_tokens'])} molecule tokens, {len(vocabulary['odor_tokens'])} odor tokens"
    )


if __name__ == "__main__":
    main()
