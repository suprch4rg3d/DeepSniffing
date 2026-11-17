"""A step of ``ds-build``: the same molecules written in other ways, in ``selfies_variants.csv``.

A molecule can be written as a string in many ways, depending on the atom the string starts from and the
order of its branches. All of them are the same molecule, with the same CID, the same stereochemistry and the
same smell. The tables hold the canonical writing only, so the model could learn that string by heart. This
step writes more strings of the same molecules, and the training takes a different one every epoch.

    selfies_variants.csv     cid, selfies: other writings of a molecule, at most ``VARIANTS`` per CID

Every variant is checked before it is kept, and the ones that fail are counted and left out:

1. the random SMILES is read again and its canonical SMILES, with the stereochemistry, is the canonical
   SMILES of the molecule, so a variant is never another molecule and never a mirror image;
2. its SELFIES is decoded again and gives that canonical SMILES once more.

The variants belong to the CID, so they are on the same side of the split as the molecule: the table has no
split of its own, and a reader takes the split from ``structures.csv`` or ``odor_dataset.csv``. The DREAM
molecules have no variants, because they are a test and are read in one way only.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem, RDLogger

from scripts.preprocessing.odor_words import ODOR_DIR

VARIANTS_PATH = ODOR_DIR / "selfies_variants.csv"
VARIANTS = 20  # the most variants of a molecule; a small molecule has fewer ways to be written
ATTEMPTS = 100  # random writings tried for a molecule before it is left with the ones found
SEED = 42

REJECTED = ("another molecule", "SELFIES gives another molecule")


def canonical(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    return None if molecule is None else Chem.MolToSmiles(molecule)


def variants_of(
    cid: int,
    smiles: str,
    selfies: str,
    seed: int = SEED,
) -> tuple[list[str], list[int]]:
    """Up to ``VARIANTS`` verified SELFIES of the molecule, other than ``selfies``, and the counts of the
    rejected writings (see ``REJECTED``)."""
    RDLogger.DisableLog("rdApp.*")
    rng = np.random.RandomState((seed + cid) % (2**31))
    molecule = Chem.MolFromSmiles(smiles)
    target = Chem.MolToSmiles(molecule)
    kept: dict[str, None] = {
        selfies: None
    }  # the canonical writing is not a variant, but counts as seen
    tried: set[str] = set()
    rejected = [0, 0]
    for _ in range(ATTEMPTS):
        if len(kept) > VARIANTS:
            break
        # the atoms are numbered in a random order, and the string is written in that order
        order = rng.permutation(molecule.GetNumAtoms()).tolist()
        shuffled = Chem.RenumberAtoms(molecule, order)
        written = Chem.MolToSmiles(shuffled, canonical=False)
        if written in tried:
            continue
        tried.add(written)
        if canonical(written) != target:
            rejected[0] += 1
            continue
        try:
            text = sf.encoder(written)
            decoded = canonical(sf.decoder(text))
        except Exception:
            decoded = None
        if decoded != target:
            rejected[1] += 1
            continue
        kept.setdefault(text)
    return [text for text in kept if text != selfies], rejected


def _work(row: tuple[int, str, str]) -> tuple[int, list[str], list[int]]:
    cid, smiles, selfies = row
    found, rejected = variants_of(cid, smiles, selfies)
    return cid, found, rejected


def build_variants(molecules: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """The variants of every molecule of ``molecules`` (columns cid, smiles, selfies)."""
    rows = list(zip(molecules["cid"], molecules["smiles"], molecules["selfies"], strict=True))
    rejected = [0, 0]
    out: list[tuple[int, str]] = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        for cid, found, counts in pool.map(_work, rows, chunksize=200):
            out.extend((cid, text) for text in found)
            rejected = [a + b for a, b in zip(rejected, counts, strict=True)]
    return pd.DataFrame(out, columns=["cid", "selfies"]), rejected


def save_checked(variants: pd.DataFrame, path) -> None:
    """Write the table, flush it to the disk, read it back and compare, and only then put it in place.

    A file that is cut short or filled with empty bytes (a crash or a power cut in the middle of the write
    leaves that) must never be taken for the real one, so it is checked before it replaces the old file.
    """
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        variants.to_csv(file, index=False)
        file.flush()
        os.fsync(file.fileno())
    if b"\x00" in temporary.read_bytes() or len(pd.read_csv(temporary)) != len(variants):
        raise OSError(f"{temporary} does not hold the {len(variants)} variants that were written")
    temporary.replace(path)


def add_to_vocabulary(variants: pd.DataFrame, path) -> list[str]:
    """Add the SELFIES symbols that only the variants use to the vocabulary, and return them.

    A variant is the same molecule written in another order, and some writings need symbols that the canonical
    one never does (a charged carbon, say). The model must be able to read every variant, so the symbols
    join the vocabulary, which is fixed before the training.
    """
    vocabulary = json.loads(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for text in variants["selfies"]:
        found.update(sf.split_selfies(text))
    new = sorted(found - set(vocabulary["symbols"]))
    if new:
        vocabulary["symbols"] = sorted({*vocabulary["symbols"], *found})
        path.write_text(json.dumps(vocabulary, indent=1), encoding="utf-8")
    return new


def main() -> None:
    from scripts.preprocessing.odor_tables import DATASET_PATH, STRUCTURES_PATH, VOCABULARY_PATH

    for path in (STRUCTURES_PATH, DATASET_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Run ds-build.")
    columns = ["cid", "smiles", "selfies"]
    structures = pd.read_csv(STRUCTURES_PATH)[columns]
    odor = pd.read_csv(DATASET_PATH)
    odor = odor.loc[odor["split"] != "dream", columns]
    molecules = pd.concat([structures, odor]).drop_duplicates("cid").reset_index(drop=True)

    variants, rejected = build_variants(molecules)
    save_checked(variants, VARIANTS_PATH)
    added = add_to_vocabulary(variants, VOCABULARY_PATH)
    print(
        f"{len(added)} SELFIES symbols of the variants were new and are now in {VOCABULARY_PATH}: {added}"
    )
    per_molecule = variants.groupby("cid").size().reindex(molecules["cid"], fill_value=0)
    print(f"Saved {len(variants)} variants of {len(molecules)} molecules to {VARIANTS_PATH}")
    print(
        f"per molecule: median {int(per_molecule.median())}, {int((per_molecule == 0).sum())} "
        f"without any variant, {int((per_molecule == VARIANTS).sum())} with the full {VARIANTS}"
    )
    for reason, count in zip(REJECTED, rejected, strict=True):
        print(f"rejected as {reason}: {count}")


if __name__ == "__main__":
    main()
