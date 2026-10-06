from __future__ import annotations

from pathlib import Path

import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data/processed/dream_pyrfume/molecule_labels.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/dream_pyrfume/molecules_enriched.csv"


def describe_molecule(smiles: str, cid: int) -> dict[str, int | float | str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"CID {cid} has invalid RDKit-canonicalized SMILES: {smiles!r}")

    atoms = list(molecule.GetAtoms())
    return {
        "rdkit_molecular_formula": rdMolDescriptors.CalcMolFormula(molecule),
        "rdkit_atom_count_c": sum(atom.GetAtomicNum() == 6 for atom in atoms),
        "rdkit_atom_count_h": sum(
            atom.GetTotalNumHs(includeNeighbors=True) for atom in atoms
        ),
        "rdkit_atom_count_n": sum(atom.GetAtomicNum() == 7 for atom in atoms),
        "rdkit_atom_count_o": sum(atom.GetAtomicNum() == 8 for atom in atoms),
        "rdkit_atom_count_p": sum(atom.GetAtomicNum() == 15 for atom in atoms),
        "rdkit_atom_count_s": sum(atom.GetAtomicNum() == 16 for atom in atoms),
        "rdkit_atom_count_na": sum(atom.GetAtomicNum() == 11 for atom in atoms),
        "rdkit_atom_count_cl": sum(atom.GetAtomicNum() == 17 for atom in atoms),
        "rdkit_atom_count_i": sum(atom.GetAtomicNum() == 53 for atom in atoms),
        "rdkit_heavy_atom_count": Descriptors.HeavyAtomCount(molecule),
        "rdkit_aromatic_ring_count": rdMolDescriptors.CalcNumAromaticRings(molecule),
        "rdkit_aliphatic_ring_count": rdMolDescriptors.CalcNumAliphaticRings(molecule),
        "rdkit_rotatable_bond_count": rdMolDescriptors.CalcNumRotatableBonds(molecule),
        "rdkit_hydrogen_bond_donor_count": rdMolDescriptors.CalcNumHBD(molecule),
        "rdkit_hydrogen_bond_acceptor_count": rdMolDescriptors.CalcNumHBA(molecule),
        "rdkit_topological_polar_surface_area": Descriptors.TPSA(molecule),
        "rdkit_mol_logp": Descriptors.MolLogP(molecule),
    }


def enrich_dataset() -> pd.DataFrame:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Molecule dataset does not exist: {INPUT_PATH}")

    molecules = pd.read_csv(INPUT_PATH)
    required_columns = {"CID", "rdkit_canonical_smiles", "rdkit_version"}
    missing_columns = required_columns.difference(molecules.columns)
    if missing_columns:
        raise ValueError(
            "Input dataset is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
    if molecules["CID"].isna().any():
        raise ValueError("Input dataset contains missing CIDs.")
    if molecules["CID"].duplicated().any():
        raise ValueError("Expected one molecule row per CID.")
    recorded_versions = set(molecules["rdkit_version"].dropna().astype(str))
    if recorded_versions != {rdkit.__version__}:
        raise ValueError(
            "RDKit version differs from the version used to canonicalize SMILES; "
            "rebuild molecule_labels.csv with the current Poetry environment."
        )

    enriched_rows = []
    for cid, smiles in molecules[["CID", "rdkit_canonical_smiles"]].itertuples(
        index=False, name=None
    ):
        if pd.isna(smiles) or not str(smiles).strip():
            raise ValueError(f"CID {cid} has no RDKit-canonicalized SMILES.")
        enriched_rows.append(describe_molecule(str(smiles).strip(), int(cid)))

    descriptors = pd.DataFrame(enriched_rows, index=molecules.index)
    molecules = pd.concat([molecules, descriptors], axis=1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    molecules.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(molecules)} enriched molecules to {OUTPUT_PATH}")
    print(f"RDKit version: {rdkit.__version__}")
    return molecules


def main() -> None:
    enrich_dataset()


if __name__ == "__main__":
    main()
