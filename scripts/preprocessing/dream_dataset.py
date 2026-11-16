"""Build the molecule table: one row per molecule, with the identity and the structure of each.

The 480 molecules of Keller and Vosshall (2016) are the 476 molecules of the DREAM challenge and four
more that took part in none of its sets. The table holds, for each of them, the CID, the odor name,
the SMILES from every source and the one canonical SMILES that all later steps use. It holds no rating
of any kind: the ratings are read from the DREAM rating files when the models train, so that no table
that feeds a model can carry the answers of the test molecules.

The SMILES comes from PubChem first, then the Pyrfume molecule registry, then Keller's own file. Whatever
the source, every other SMILES of the molecule has to describe the same connectivity, or the step stops.

    CID, OdorName, CAS, <the SMILES of each source>, rdkit_canonical_smiles, ...
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import rdkit
from rdkit import Chem

from scripts.core.splits import (
    PROCESSED_DIR,
    leaderboard_cids,
    official_test_cids,
    training_pool_cids,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KELLER_MOLECULES_PATH = PROJECT_ROOT / "data/raw/pyrfume-data/keller_2016/molecules.csv"
MOLECULE_REGISTRY_PATH = PROJECT_ROOT / "data/raw/pyrfume-data/molecules/molecules.csv"
PUBCHEM_PATH = PROJECT_ROOT / "data/raw/pubchem/dream_keller_compound_properties.csv"
OUTPUT_PATH = PROCESSED_DIR / "molecules.csv"

PUBCHEM_COLUMNS = {
    "SMILES": "pubchem_smiles",
    "MolecularFormula": "pubchem_molecular_formula",
    "MolecularWeight": "pubchem_molecular_weight",
    "ExactMass": "pubchem_exact_mass",
    "InChI": "pubchem_inchi",
    "InChIKey": "pubchem_inchi_key",
    "IUPACName": "pubchem_iupac_name",
}


def build_dataset() -> pd.DataFrame:
    input_files = (KELLER_MOLECULES_PATH, MOLECULE_REGISTRY_PATH, PUBCHEM_PATH)
    missing_files = [str(path) for path in input_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Required input file(s) not found: {', '.join(missing_files)}. Run ds-prepare."
        )

    molecules = pd.read_csv(KELLER_MOLECULES_PATH)
    registry = pd.read_csv(MOLECULE_REGISTRY_PATH)
    pubchem = pd.read_csv(PUBCHEM_PATH)

    missing = {"CID", "CanonicalSMILES"}.difference(molecules.columns)
    if missing:
        raise ValueError(f"Keller molecules file is missing columns: {', '.join(sorted(missing))}")
    missing = {"CID", "IsomericSMILES"}.difference(registry.columns)
    if missing:
        raise ValueError(
            f"Pyrfume molecule registry is missing columns: {', '.join(sorted(missing))}"
        )
    missing = {"CID", *PUBCHEM_COLUMNS}.difference(pubchem.columns)
    if missing:
        raise ValueError(f"PubChem file is missing columns: {', '.join(sorted(missing))}")

    molecules = molecules.drop(columns=["Unnamed: 0"], errors="ignore")
    molecules["CID"] = molecules["CID"].astype("int64")
    molecules = molecules.drop_duplicates("CID").rename(
        columns={"CanonicalSMILES": "keller_smiles"}
    )
    registry = registry[["CID", "IsomericSMILES"]].copy()
    registry["CID"] = registry["CID"].astype("int64")
    registry = registry.drop_duplicates("CID").rename(
        columns={"IsomericSMILES": "pyrfume_isomeric_smiles"}
    )
    molecules = molecules.merge(registry, on="CID", how="left", validate="one_to_one")

    pubchem = pubchem.rename(columns=PUBCHEM_COLUMNS)[["CID", *PUBCHEM_COLUMNS.values()]].copy()
    pubchem["CID"] = pd.to_numeric(pubchem["CID"], errors="raise").astype("int64")
    if pubchem["CID"].duplicated().any():
        raise ValueError("PubChem file contains duplicate CIDs.")
    molecules = molecules.merge(pubchem, on="CID", how="left", validate="one_to_one")

    keller_smiles = molecules["keller_smiles"].fillna("").astype(str).str.strip()
    registry_smiles = molecules["pyrfume_isomeric_smiles"].fillna("").astype(str).str.strip()
    pubchem_smiles = molecules["pubchem_smiles"].fillna("").astype(str).str.strip()
    molecules["source_smiles"] = pubchem_smiles
    molecules["smiles_source"] = "pubchem"
    use_registry_smiles = pubchem_smiles.eq("") & registry_smiles.ne("")
    molecules.loc[use_registry_smiles, "source_smiles"] = registry_smiles[use_registry_smiles]
    molecules.loc[use_registry_smiles, "smiles_source"] = "pyrfume_molecule_registry"
    use_keller_smiles = pubchem_smiles.eq("") & registry_smiles.eq("") & keller_smiles.ne("")
    molecules.loc[use_keller_smiles, "source_smiles"] = keller_smiles[use_keller_smiles]
    molecules.loc[use_keller_smiles, "smiles_source"] = "keller_2016"
    molecules["source_smiles"] = molecules["source_smiles"].replace("", pd.NA)
    molecules.loc[molecules["source_smiles"].isna(), "smiles_source"] = "missing"

    standardized_smiles = []
    for row in molecules[
        [
            "CID",
            "source_smiles",
            "keller_smiles",
            "pyrfume_isomeric_smiles",
        ]
    ].itertuples(index=False, name=None):
        cid, source_smiles, *source_variants = row
        molecule = Chem.MolFromSmiles(source_smiles) if pd.notna(source_smiles) else None
        if molecule is None:
            raise ValueError(f"CID {cid} has no valid SMILES in the available sources.")

        connectivity = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        for source_variant in source_variants:
            if pd.isna(source_variant) or not str(source_variant).strip():
                continue
            variant_molecule = Chem.MolFromSmiles(str(source_variant).strip())
            if variant_molecule is None:
                raise ValueError(f"CID {cid} has an invalid source SMILES: {source_variant}")
            variant_connectivity = Chem.MolToSmiles(
                variant_molecule, canonical=True, isomericSmiles=False
            )
            if variant_connectivity != connectivity:
                raise ValueError(
                    f"SMILES sources disagree on molecular connectivity for CID {cid}."
                )

        standardized_smiles.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
    molecules = molecules.drop(columns="source_smiles")
    molecules["rdkit_canonical_smiles"] = standardized_smiles
    molecules["rdkit_version"] = rdkit.__version__

    needed = set(training_pool_cids()) | set(leaderboard_cids()) | set(official_test_cids())
    absent = sorted(needed - set(molecules["CID"]))
    if absent:
        raise ValueError(f"The molecule table lacks molecules of the DREAM sets: {absent[:5]}")
    return molecules


def main() -> None:
    molecules = build_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    molecules.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(molecules)} molecules, with no ratings, to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
