from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/dream_pyrfume/molecule_labels.csv"
OUTPUT = ROOT / "data/processed/dream_pyrfume/morgan_fingerprints.csv"

RADIUS = 2
FINGERPRINT_SIZE = 2048
FINGERPRINT_COLUMNS = [f"morgan_{index:04d}" for index in range(FINGERPRINT_SIZE)]


def generate_fingerprints() -> pd.DataFrame:
    if not INPUT.is_file():
        raise FileNotFoundError(f"Molecule dataset not found: {INPUT}")

    molecules = pd.read_csv(INPUT)
    required = {"CID", "rdkit_canonical_smiles"}
    missing = required.difference(molecules.columns)
    if missing:
        raise ValueError(f"Molecule dataset is missing: {', '.join(sorted(missing))}")
    if molecules["CID"].isna().any() or molecules["CID"].duplicated().any():
        raise ValueError("Molecule dataset must have one non-empty CID per row.")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        countSimulation=False,
        includeChirality=True,
        useBondTypes=True,
        fpSize=FINGERPRINT_SIZE,
    )
    rows = []
    for cid, smiles in molecules[["CID", "rdkit_canonical_smiles"]].itertuples(
        index=False, name=None
    ):
        if pd.isna(smiles) or not str(smiles).strip():
            raise ValueError(f"CID {cid} has no RDKit-canonicalized SMILES.")
        molecule = Chem.MolFromSmiles(str(smiles).strip())
        if molecule is None:
            raise ValueError(f"CID {cid} has invalid canonical SMILES: {smiles!r}")
        fingerprint = generator.GetFingerprint(molecule)
        bits = np.zeros(FINGERPRINT_SIZE, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fingerprint, bits)
        rows.append({"CID": int(cid), **dict(zip(FINGERPRINT_COLUMNS, bits.tolist()))})

    result = pd.DataFrame(rows, columns=["CID", *FINGERPRINT_COLUMNS])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(
        f"Saved {len(result)} Morgan fingerprints ({RADIUS=}, "
        f"{FINGERPRINT_SIZE=} bits, chirality enabled) to {OUTPUT}"
    )
    print(f"RDKit version: {rdkit.__version__}")
    return result


def main() -> None:
    generate_fingerprints()


if __name__ == "__main__":
    main()
