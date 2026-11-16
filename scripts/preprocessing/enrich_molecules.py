"""``ds-build``: build every file that the models read, in order.

1. ``molecules.csv``: the molecule table with its SMILES (``dream_dataset``).
2. ``morgan_fingerprints.csv``: Morgan substructure bits (``morgan_fingerprints``).
3. ``pubchem_3d_coordinates.csv``: PubChem's 3D coordinates in our atom order (``conformers``).
4. ``molecule_descriptors.csv``: the RDKit descriptors, the 4,869 DREAM Dragon descriptors and the per-atom
   Gasteiger charges with their summaries, added one after the other to the molecule table.
5. ``published_features.csv``: the EPI Suite, name, NSPDK and similarity features that the challenge
   published (``community_features``).
6. ``feature_selection.csv``: the columns that the models keep from each of the tables (``feature_selection``).
7. ``split.csv``: train, validation, test and unused molecules (``split_dataset``).
8. ``scalers/``: the feature and target scalers, fitted on training molecules only (``scalers``).

Every step is its own module with its own ``main()``, so one can be run alone with ``python -m``. This
command runs all of them, so that the files are always complete and in step with each other."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from scripts.core.splits import (
    PROCESSED_DIR,
    RATINGS_DIR,
    TEST_RATINGS_PATH,
    TRAIN_RATINGS_PATH,
    leaderboard_cids,
)
from scripts.preprocessing.dream_dataset import OUTPUT_PATH as INPUT_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROCESSED_DIR / "molecule_descriptors.csv"
DRAGON_SOURCE_PATH = RATINGS_DIR / "molecular_descriptors_data.txt"

EXPECTED_DRAGON_MOLECULES = 476
EXPECTED_DRAGON_FEATURES = 4869
# RDKit versions that canonicalized the stored SMILES and are accepted alongside the
# installed one. The descriptors below are computed from those stored SMILES, so an
# older environment's table can be enriched without rebuilding it.
OLDER_RDKIT_VERSIONS = frozenset({"2026.03.6"})
SMILES_COLUMN = "rdkit_canonical_smiles"
CHARGE_COLUMNS = (
    "rdkit_gasteiger_charge_min",
    "rdkit_gasteiger_charge_max",
    "rdkit_gasteiger_charge_mean",
    "rdkit_gasteiger_charge_std",
    "rdkit_gasteiger_positive_charge_sum",
    "rdkit_gasteiger_negative_charge_sum",
    "rdkit_gasteiger_absolute_charge_sum",
    "rdkit_gasteiger_implicit_hydrogen_charge_sum",
)


EXTERNAL_DRAGON_COLUMN = "dragon_complexity from pubmed"

# The DREAM table lists the Dragon descriptors in catalogue order, and each Dragon
# family is one contiguous block of columns. A family therefore starts at its first
# column, which is all that is needed to assign every column to a family and to the
# model group that family feeds. Each entry is (first column, family, model group).
DRAGON_FAMILY_STARTS = (
    ("dragon_MW", "Constitutional indices", "composition_properties"),
    ("dragon_nCIC", "Ring descriptors", "composition_properties"),
    ("dragon_ZM1", "Topological indices", "two_d_topology"),
    ("dragon_MWC01", "Walk and path counts", "two_d_topology"),
    ("dragon_X0", "Connectivity indices", "two_d_topology"),
    ("dragon_ISIZ", "Information indices", "two_d_topology"),
    ("dragon_J_A", "2D matrix-based descriptors", "two_d_topology"),
    ("dragon_ATS1m", "2D autocorrelations", "two_d_topology"),
    ("dragon_SpMax1_Bh(m)", "Burden eigenvalues", "two_d_topology"),
    ("dragon_P_VSA_LogP_1", "P_VSA-like descriptors", "two_d_topology"),
    ("dragon_Eta_alpha", "ETA indices", "two_d_topology"),
    ("dragon_SpMax_EA", "Edge adjacency indices", "two_d_topology"),
    ("dragon_G1", "Geometrical descriptors", "three_d_shape"),
    ("dragon_Wi_G", "3D matrix-based descriptors", "three_d_shape"),
    ("dragon_TDB01u", "3D autocorrelations", "three_d_shape"),
    ("dragon_RDF010u", "RDF descriptors", "three_d_shape"),
    ("dragon_Mor01u", "3D-MoRSE descriptors", "three_d_shape"),
    ("dragon_L1u", "WHIM descriptors", "three_d_shape"),
    ("dragon_ITH", "GETAWAY descriptors", "three_d_shape"),
    ("dragon_DP01", "Randic molecular profiles", "two_d_topology"),
    ("dragon_nCp", "Functional group counts", "two_d_local_patterns"),
    ("dragon_C-001", "Atom-centred fragments", "two_d_local_patterns"),
    ("dragon_SsCH3", "Atom-type E-state indices", "two_d_local_patterns"),
    ("dragon_CATS2D_00_DD", "CATS 2D", "two_d_local_patterns"),
    ("dragon_T(N..N)", "2D Atom Pairs", "two_d_local_patterns"),
    ("dragon_G(N..N)", "3D Atom Pairs", "three_d_shape"),
    ("dragon_Uc", "Molecular properties", "composition_properties"),
    ("dragon_cRo5", "Drug-like indices", "composition_properties"),
)


def dragon_feature_groups(dragon_columns: list[str]) -> dict[str, list[str]]:
    """Group the DREAM Dragon columns into the model's four descriptor groups.

    ``dragon_columns`` must be every ``dragon_`` column of the enriched table in
    source order. The one non-Dragon field, the PubMed complexity score, is left
    out. Groups are returned sorted by name, each keeping the source column order.
    """
    columns = [column for column in dragon_columns if column != EXTERNAL_DRAGON_COLUMN]
    positions = {column: index for index, column in enumerate(columns)}
    missing = [start for start, _, _ in DRAGON_FAMILY_STARTS if start not in positions]
    if missing:
        raise ValueError(f"Dragon family start columns are missing: {missing}")
    starts = [positions[start] for start, _, _ in DRAGON_FAMILY_STARTS]
    if starts[0] != 0 or starts != sorted(starts):
        raise ValueError("Dragon columns are not in the DREAM catalogue order.")

    groups: dict[str, list[str]] = {}
    family_index = 0
    for index, column in enumerate(columns):
        while family_index + 1 < len(starts) and index >= starts[family_index + 1]:
            family_index += 1
        groups.setdefault(DRAGON_FAMILY_STARTS[family_index][2], []).append(column)
    return dict(sorted(groups.items()))


def describe_molecule(smiles: str, cid: int) -> dict[str, int | float | str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"CID {cid} has invalid RDKit-canonicalized SMILES: {smiles!r}")

    atoms = list(molecule.GetAtoms())
    return {
        "rdkit_molecular_formula": rdMolDescriptors.CalcMolFormula(molecule),
        "rdkit_atom_count_c": sum(atom.GetAtomicNum() == 6 for atom in atoms),
        "rdkit_atom_count_h": sum(atom.GetTotalNumHs(includeNeighbors=True) for atom in atoms),
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


def add_rdkit_descriptors(molecules: pd.DataFrame) -> pd.DataFrame:
    """Add the RDKit counts and descriptors of each molecule's canonical SMILES."""
    required_columns = {"CID", "rdkit_canonical_smiles", "rdkit_version"}
    missing_columns = required_columns.difference(molecules.columns)
    if missing_columns:
        raise ValueError(
            f"Input dataset is missing required columns: {', '.join(sorted(missing_columns))}"
        )
    if molecules["CID"].isna().any():
        raise ValueError("Input dataset contains missing CIDs.")
    if molecules["CID"].duplicated().any():
        raise ValueError("Expected one molecule row per CID.")
    recorded_versions = set(molecules["rdkit_version"].dropna().astype(str))
    if not recorded_versions or not recorded_versions <= {rdkit.__version__, *OLDER_RDKIT_VERSIONS}:
        raise ValueError(
            "RDKit version differs from the version used to canonicalize SMILES; "
            "rebuild molecules.csv with the current Poetry environment."
        )
    if recorded_versions != {rdkit.__version__}:
        print(
            f"molecules.csv was canonicalized with RDKit {sorted(recorded_versions)}; "
            f"continuing with the older environment's SMILES under RDKit {rdkit.__version__}."
        )

    enriched_rows = []
    for cid, smiles in molecules[["CID", "rdkit_canonical_smiles"]].itertuples(
        index=False, name=None
    ):
        if pd.isna(smiles) or not str(smiles).strip():
            raise ValueError(f"CID {cid} has no RDKit-canonicalized SMILES.")
        enriched_rows.append(describe_molecule(str(smiles).strip(), int(cid)))

    descriptors = pd.DataFrame(enriched_rows, index=molecules.index)
    return pd.concat([molecules, descriptors], axis=1)


def require_dragon_source() -> None:
    """The Dragon table is commercial software's output, published by the organizers: ``ds-prepare`` downloads it."""
    if not DRAGON_SOURCE_PATH.is_file():
        raise FileNotFoundError(
            f"The DREAM Dragon descriptors are missing: {DRAGON_SOURCE_PATH}. Run ds-prepare."
        )


def read_challenge_cids(path: Path) -> set[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Required DREAM ratings file not found: {path}")
    ratings = pd.read_csv(path, sep="\t", usecols=["Compound Identifier"])
    return set(pd.to_numeric(ratings["Compound Identifier"], errors="raise").astype("int64"))


def read_dragon_descriptors() -> pd.DataFrame:
    with DRAGON_SOURCE_PATH.open(encoding="utf-8-sig", newline="") as file:
        headers = next(csv.reader(file, delimiter="\t"), None)
    if not headers or headers[0] != "CID":
        raise ValueError("DREAM Dragon descriptor file must start with the CID column.")

    descriptor_columns = []
    for index, name in enumerate(headers[1:], start=1):
        # The source has a numeric Dragon descriptor also named CID.
        descriptor_name = f"CID_descriptor_{index}" if name == "CID" else name
        descriptor_columns.append(f"dragon_{descriptor_name}")

    descriptors = pd.read_csv(DRAGON_SOURCE_PATH, sep="\t", encoding="utf-8-sig")
    if len(descriptors.columns) != len(headers):
        raise ValueError("DREAM Dragon descriptor table has an unexpected number of columns.")
    if len(headers) - 1 != EXPECTED_DRAGON_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_DRAGON_FEATURES} Dragon features; found {len(headers) - 1}."
        )
    descriptors.columns = ["CID", *descriptor_columns]
    if descriptors.empty or descriptors["CID"].isna().any():
        raise ValueError("DREAM Dragon descriptor table has no valid CID rows.")
    if len(descriptors) != EXPECTED_DRAGON_MOLECULES:
        raise ValueError(
            f"Expected Dragon descriptors for {EXPECTED_DRAGON_MOLECULES} molecules; "
            f"found {len(descriptors)}."
        )

    descriptors["CID"] = pd.to_numeric(descriptors["CID"], errors="raise").astype("int64")
    if descriptors["CID"].duplicated().any():
        raise ValueError("DREAM Dragon descriptor table contains duplicate CIDs.")
    for column in descriptor_columns:
        descriptors[column] = pd.to_numeric(descriptors[column], errors="raise")
    values = descriptors[descriptor_columns].to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("DREAM Dragon descriptor table has infinite values.")
    return descriptors


def add_dragon_descriptors(molecules: pd.DataFrame) -> pd.DataFrame:
    """Join the DREAM Dragon descriptors onto the molecules."""
    require_dragon_source()
    molecules = molecules.copy()
    if "CID" not in molecules.columns or molecules["CID"].isna().any():
        raise ValueError("Enriched molecule dataset must contain non-empty CIDs.")
    molecules["CID"] = pd.to_numeric(molecules["CID"], errors="raise").astype("int64")
    if molecules["CID"].duplicated().any():
        raise ValueError("Enriched molecule dataset must contain one row per CID.")

    descriptors = read_dragon_descriptors()
    descriptor_cids = set(descriptors["CID"])
    if not descriptor_cids.issubset(set(molecules["CID"])):
        raise ValueError("DREAM Dragon table contains CIDs missing from the enriched dataset.")

    challenge_cids = (
        read_challenge_cids(TRAIN_RATINGS_PATH)
        | read_challenge_cids(TEST_RATINGS_PATH)
        | set(leaderboard_cids())
    )
    missing_challenge_cids = challenge_cids - descriptor_cids
    if missing_challenge_cids:
        raise ValueError(
            "DREAM Dragon descriptors are missing official train/leaderboard/test CIDs: "
            f"{sorted(missing_challenge_cids)}"
        )

    result = molecules.merge(descriptors, on="CID", how="left", validate="one_to_one")
    missing_cids = sorted(set(result["CID"]) - descriptor_cids)
    missing_values = int(result[[c for c in result if c.startswith("dragon_")]].isna().sum().sum())
    print(f"Joined {len(descriptors.columns) - 1} Dragon features onto {len(result)} molecules.")
    if missing_values:
        print(
            f"Dragon source has {missing_values} undefined descriptor values; "
            "the trainer will impute them from training-set medians."
        )
    if missing_cids:
        print(f"No Dragon data for non-challenge CIDs: {missing_cids}")
    return result


def charge_features(smiles: str, cid: int) -> dict[str, object]:
    """Per-atom Gasteiger charges and molecule-level charge summaries."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"CID {cid} has invalid canonical SMILES: {smiles!r}")

    AllChem.ComputeGasteigerCharges(molecule, nIter=12, throwOnParamFailure=False)
    atom_charges = np.asarray(
        [float(atom.GetProp("_GasteigerCharge")) for atom in molecule.GetAtoms()],
        dtype=np.float64,
    )
    hydrogen_charges = np.asarray(
        [
            float(atom.GetProp("_GasteigerHCharge")) if atom.HasProp("_GasteigerHCharge") else 0.0
            for atom in molecule.GetAtoms()
        ],
        dtype=np.float64,
    )
    if not np.isfinite(atom_charges).all() or not np.isfinite(hydrogen_charges).all():
        raise ValueError(f"CID {cid} produced non-finite Gasteiger charges.")

    values = {
        "rdkit_gasteiger_atom_charges": json.dumps(atom_charges.tolist()),
        "rdkit_gasteiger_implicit_hydrogen_charges": json.dumps(hydrogen_charges.tolist()),
        "rdkit_gasteiger_charge_min": float(atom_charges.min()),
        "rdkit_gasteiger_charge_max": float(atom_charges.max()),
        "rdkit_gasteiger_charge_mean": float(atom_charges.mean()),
        "rdkit_gasteiger_charge_std": float(atom_charges.std()),
        "rdkit_gasteiger_positive_charge_sum": float(atom_charges[atom_charges > 0].sum()),
        "rdkit_gasteiger_negative_charge_sum": float(atom_charges[atom_charges < 0].sum()),
        "rdkit_gasteiger_absolute_charge_sum": float(np.abs(atom_charges).sum()),
        "rdkit_gasteiger_implicit_hydrogen_charge_sum": float(hydrogen_charges.sum()),
    }
    if not np.isfinite([values[name] for name in CHARGE_COLUMNS]).all():
        raise ValueError(f"CID {cid} produced non-finite charge summary features.")
    return values


def add_charge_features(molecules: pd.DataFrame) -> pd.DataFrame:
    """Add the Gasteiger charge features of each molecule."""
    required = {"CID", SMILES_COLUMN}
    if missing := required.difference(molecules.columns):
        raise ValueError(f"Source table is missing columns: {sorted(missing)}")
    if molecules["CID"].duplicated().any():
        raise ValueError("Source table must contain one row per CID.")

    rows = []
    for cid, smiles in molecules[["CID", SMILES_COLUMN]].itertuples(index=False, name=None):
        if pd.isna(smiles) or not str(smiles).strip():
            raise ValueError(f"CID {cid} has no RDKit-canonical SMILES.")
        rows.append(charge_features(str(smiles), int(cid)))

    extra_features = pd.DataFrame(rows, index=molecules.index)
    output = pd.concat(
        [molecules.drop(columns=list(extra_features.columns), errors="ignore"), extra_features],
        axis=1,
    )
    return output


def main() -> None:
    """Build every file that the models read, in the order of the module docstring."""
    # Imported here: the later steps import ``scripts.core.data``, which imports this module.
    from scripts.preprocessing import (
        community_features,
        conformers,
        dream_dataset,
        feature_selection,
        morgan_fingerprints,
        scalers,
        split_dataset,
    )

    dream_dataset.main()
    morgan_fingerprints.main()
    conformers.main()
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Molecule table does not exist: {INPUT_PATH}.")
    descriptors = add_charge_features(
        add_dragon_descriptors(add_rdkit_descriptors(pd.read_csv(INPUT_PATH)))
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptors.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Saved {len(descriptors)} molecules with RDKit, Dragon and charge descriptors to {OUTPUT_PATH}"
    )
    print(f"RDKit version: {rdkit.__version__}")
    community_features.main()
    feature_selection.main()
    split_dataset.main()
    scalers.main()


if __name__ == "__main__":
    main()
