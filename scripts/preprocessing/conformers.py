"""Write PubChem's 3D structure of each molecule, in the atom order of our molecule table.

The graph model measures distances between atoms. A molecule that can twist has many 3D
shapes, so a shape built from a random start is one arbitrary choice. PubChem publishes one
computed 3D structure per compound, the same for everyone, and ``ds-prepare`` downloads it.

PubChem numbers the atoms in its own order. This step finds, for each molecule, which of its
atoms is which atom of the molecule that RDKit reads from our SMILES, because the per atom
charges and features of the model follow that order. It writes one row per heavy atom:

    CID, atom_index, symbol, x, y, z

A molecule whose structure is missing or does not map onto our SMILES is left out. Its graph
then has the bonds only, with every distance feature zero. No shape is built with RDKit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from scripts.core.splits import PROCESSED_DIR
from scripts.data.prepare_data import split_sdf_records
from scripts.preprocessing.dream_dataset import OUTPUT_PATH as MOLECULES_PATH

ROOT = Path(__file__).resolve().parents[2]
INPUT = MOLECULES_PATH
SDF_PATH = ROOT / "data/raw/pubchem/dream_conformers.sdf"
OUTPUT = PROCESSED_DIR / "pubchem_3d_coordinates.csv"
COLUMNS = ["CID", "atom_index", "symbol", "x", "y", "z"]


def pose_in_our_atom_order(record: str, smiles: str) -> np.ndarray:
    """The heavy atom coordinates of a PubChem record, ordered like the atoms of ``smiles``."""
    ours = Chem.MolFromSmiles(smiles)
    if ours is None:
        raise ValueError(f"RDKit could not read the SMILES {smiles!r}.")
    pubchem = Chem.MolFromMolBlock(record, removeHs=True)
    if pubchem is None or pubchem.GetNumConformers() == 0:
        raise ValueError("The PubChem record has no readable 3D structure.")
    if pubchem.GetNumAtoms() != ours.GetNumAtoms():
        raise ValueError(
            f"The atom counts differ: PubChem {pubchem.GetNumAtoms()}, ours {ours.GetNumAtoms()}."
        )

    # The match lists, for each PubChem atom, the atom of our molecule that it is.
    match = ours.GetSubstructMatch(pubchem)
    if len(match) != pubchem.GetNumAtoms() or len(set(match)) != len(match):
        raise ValueError("The PubChem atoms do not map onto the atoms of our SMILES.")
    for source, target in enumerate(match):
        if (
            pubchem.GetAtomWithIdx(source).GetAtomicNum()
            != ours.GetAtomWithIdx(target).GetAtomicNum()
        ):
            raise ValueError("A mapped atom is a different element.")

    coordinates = pubchem.GetConformer().GetPositions()
    if not np.isfinite(coordinates).all():
        raise ValueError("The PubChem coordinates are not finite.")
    # A lone atom has no distances, so its position at the origin is fine.
    if ours.GetNumAtoms() > 1 and np.abs(coordinates).max() < 1e-6:
        raise ValueError("The PubChem coordinates are all zero, which is a flat or missing shape.")

    ordered = np.zeros_like(coordinates)
    for source, target in enumerate(match):
        ordered[target] = coordinates[source]
    return ordered


def build_conformers() -> pd.DataFrame:
    for path in (INPUT, SDF_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Input not found: {path}. Run ds-prepare and ds-build.")
    molecules = pd.read_csv(INPUT)
    records = split_sdf_records(SDF_PATH.read_text(encoding="utf-8"))

    rows, skipped = [], {}
    for cid, smiles in molecules[["CID", "rdkit_canonical_smiles"]].itertuples(
        index=False, name=None
    ):
        cid = int(cid)
        if cid not in records:
            skipped[cid] = "PubChem has no 3D structure"
            continue
        try:
            coordinates = pose_in_our_atom_order(records[cid], str(smiles))
        except ValueError as error:
            skipped[cid] = str(error)
            continue
        symbols = [atom.GetSymbol() for atom in Chem.MolFromSmiles(str(smiles)).GetAtoms()]
        for index, (symbol, (x, y, z)) in enumerate(zip(symbols, coordinates, strict=True)):
            rows.append(
                (cid, index, symbol, round(float(x), 4), round(float(y), 4), round(float(z), 4))
            )

    result = pd.DataFrame(rows, columns=COLUMNS)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(
        f"Saved PubChem 3D structures of {result['CID'].nunique()} of {len(molecules)} molecules "
        f"to {OUTPUT}"
    )
    if skipped:
        print(f"No 3D structure, so bonds only, for {len(skipped)} molecules:")
        for cid, reason in sorted(skipped.items()):
            print(f"  CID {cid}: {reason}")
    return result


def main() -> None:
    build_conformers()


if __name__ == "__main__":
    main()
