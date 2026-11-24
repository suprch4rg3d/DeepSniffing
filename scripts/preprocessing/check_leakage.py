"""Check that no DREAM molecule reached the pretraining table.

    python -m scripts.preprocessing.check_leakage

Exits non-zero when one is found, so a build can be stopped before it is trained on.
"""

from __future__ import annotations

import sys

import pandas as pd

from scripts.preprocessing.odor_tables import (
    DREAM_MOLECULES_PATH,
    STRUCTURES_PATH,
    canonical_smiles,
)


def main() -> int:
    if not STRUCTURES_PATH.is_file():
        raise SystemExit(f"{STRUCTURES_PATH} not found. Run ds-build.")
    dream = pd.read_csv(DREAM_MOLECULES_PATH)
    cids = set(dream["CID"].astype(int))
    smiles = {
        found for found in dream["CanonicalSMILES"].map(canonical_smiles) if found is not None
    }
    table = pd.read_csv(STRUCTURES_PATH, usecols=["cid", "smiles"])
    by_cid = int(table["cid"].isin(cids).sum())
    by_structure = int(table["smiles"].isin(smiles).sum())
    print(
        f"DREAM molecules: {len(cids)} | in the pretraining by CID: {by_cid} "
        f"| by structure: {by_structure}"
    )
    if by_cid or by_structure:
        print("A DREAM molecule leaked into the pretraining.", file=sys.stderr)
        return 1
    print("No DREAM molecule is in the pretraining table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
