from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import rdkit
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DREAM_TRAIN_PATH = PROJECT_ROOT / "data/raw/dream-olfaction-prediction/data/TrainSet.txt"
KELLER_MOLECULES_PATH = PROJECT_ROOT / "data/raw/pyrfume-data/keller_2016/molecules.csv"
MOLECULE_REGISTRY_PATH = PROJECT_ROOT / "data/raw/pyrfume-data/molecules/molecules.csv"
KELLER_TARGETS_PATH = PROJECT_ROOT / "data/raw/pyrfume-data/prediction_targets/keller_2016.csv"
PUBCHEM_PATH = PROJECT_ROOT / "data/raw/pubchem/dream_keller_compound_properties.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/dream_pyrfume"


def build_dataset() -> None:
    input_files = (
        DREAM_TRAIN_PATH,
        KELLER_MOLECULES_PATH,
        MOLECULE_REGISTRY_PATH,
        KELLER_TARGETS_PATH,
        PUBCHEM_PATH,
    )
    missing_files = [str(path) for path in input_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Required input file(s) not found: {', '.join(missing_files)}")

    molecules = pd.read_csv(KELLER_MOLECULES_PATH)
    registry = pd.read_csv(MOLECULE_REGISTRY_PATH)
    targets = pd.read_csv(KELLER_TARGETS_PATH)
    ratings = pd.read_csv(DREAM_TRAIN_PATH, sep="\t")
    pubchem = pd.read_csv(PUBCHEM_PATH)

    missing = {"CID", "CanonicalSMILES"}.difference(molecules.columns)
    if missing:
        raise ValueError(f"Keller molecules file is missing columns: {', '.join(sorted(missing))}")
    missing = {"CID", "IsomericSMILES"}.difference(registry.columns)
    if missing:
        raise ValueError(
            f"Pyrfume molecule registry is missing columns: {', '.join(sorted(missing))}"
        )
    missing = {"CID", "Descriptor", "Ave Rating"}.difference(targets.columns)
    if missing:
        raise ValueError(f"Keller targets file is missing columns: {', '.join(sorted(missing))}")
    missing = {
        "CID",
        "SMILES",
        "MolecularFormula",
        "MolecularWeight",
        "ExactMass",
        "InChI",
        "InChIKey",
        "IUPACName",
    }.difference(pubchem.columns)
    if missing:
        raise ValueError(f"PubChem file is missing columns: {', '.join(sorted(missing))}")
    ratings = ratings.rename(columns={"Compound Identifier": "CID"})
    if "CID" not in ratings.columns:
        raise ValueError("DREAM training file is missing the 'Compound Identifier' column")
    ratings_source = ratings.copy()

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

    pubchem = pubchem.rename(
        columns={
            "SMILES": "pubchem_smiles",
            "MolecularFormula": "pubchem_molecular_formula",
            "MolecularWeight": "pubchem_molecular_weight",
            "ExactMass": "pubchem_exact_mass",
            "InChI": "pubchem_inchi",
            "InChIKey": "pubchem_inchi_key",
            "IUPACName": "pubchem_iupac_name",
        }
    )
    pubchem = pubchem[
        [
            "CID",
            "pubchem_smiles",
            "pubchem_molecular_formula",
            "pubchem_molecular_weight",
            "pubchem_exact_mass",
            "pubchem_inchi",
            "pubchem_inchi_key",
            "pubchem_iupac_name",
        ]
    ].copy()
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

    targets["CID"] = targets["CID"].astype("int64")
    ratings["CID"] = pd.to_numeric(ratings["CID"]).astype("int64")

    labels = targets.pivot(index="CID", columns="Descriptor", values="Ave Rating")
    labels.columns = [
        f"label_{re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')}" for name in labels.columns
    ]
    labels = labels.reset_index()

    # Keep DREAM's participant ratings distinct from Keller's molecule-level averages.
    metadata_columns = {"CID", "Odor", "Replicate", "Intensity", "Dilution", "subject #"}
    dream_rating_columns = {
        column: re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
        for column in ratings.columns
        if column not in metadata_columns
    }
    ratings = ratings.rename(columns=dream_rating_columns)
    available_dream_columns = list(dream_rating_columns.values())
    for column in available_dream_columns:
        ratings[column] = pd.to_numeric(ratings[column], errors="coerce")
    dream_means = (
        ratings.groupby("CID")[available_dream_columns].mean().add_prefix("dream_train_mean_")
    )
    dream_stds = (
        ratings.groupby("CID")[available_dream_columns].std().add_prefix("dream_train_std_")
    )
    dream_summary = dream_means.join(dream_stds).reset_index()
    dream_counts = (
        ratings.groupby("CID")
        .agg(
            dream_train_rating_rows=("CID", "size"),
            dream_train_participant_count=("subject #", "nunique"),
            dream_train_dilution_count=("Dilution", "nunique"),
        )
        .reset_index()
    )
    dream_summary = dream_summary.merge(dream_counts, on="CID", how="left", validate="one_to_one")

    molecules_with_labels = molecules.merge(labels, on="CID", how="inner", validate="one_to_one")
    molecules_with_labels = molecules_with_labels.merge(
        dream_summary, on="CID", how="left", validate="one_to_one"
    )
    molecules_with_labels["has_dream_train_ratings"] = molecules_with_labels[
        "dream_train_rating_rows"
    ].notna()
    ratings_with_smiles = ratings_source.merge(
        molecules, on="CID", how="left", validate="many_to_one"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    molecules_with_labels.to_csv(OUTPUT_DIR / "molecule_labels.csv", index=False)
    ratings_with_smiles.to_csv(OUTPUT_DIR / "dream_ratings_with_smiles.csv", index=False)

    print(f"Saved molecule-level master dataset: {OUTPUT_DIR / 'molecule_labels.csv'}")
    print(f"Saved DREAM ratings: {OUTPUT_DIR / 'dream_ratings_with_smiles.csv'}")


def main() -> None:
    build_dataset()


if __name__ == "__main__":
    main()
