"""The tables of ``ds-build`` joined into one row per molecule, and the rating rows that carry them."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.core.data.ratings import STANDARD_LOG10_DILUTION, SUBJECT_IDS, read_rating_table
from scripts.core.metrics import TARGETS, normalize_target_name
from scripts.core.splits import (
    LEADERBOARD_RATINGS_PATH,
    TEST_RATINGS_PATH,
    TRAIN_RATINGS_PATH,
    load_split,
)
from scripts.preprocessing.community_features import (
    COMMUNITY_BLOCKS,
)
from scripts.preprocessing.community_features import OUTPUT as COMMUNITY_PATH
from scripts.preprocessing.enrich_molecules import OUTPUT_PATH as GRAPH_MOLECULES_PATH
from scripts.preprocessing.enrich_molecules import dragon_feature_groups
from scripts.preprocessing.feature_selection import keep_selected
from scripts.preprocessing.morgan_fingerprints import OUTPUT as FINGERPRINTS_PATH

RDKIT_COLUMNS = (
    "rdkit_atom_count_c",
    "rdkit_atom_count_h",
    "rdkit_atom_count_n",
    "rdkit_atom_count_o",
    "rdkit_atom_count_p",
    "rdkit_atom_count_s",
    "rdkit_atom_count_na",
    "rdkit_atom_count_cl",
    "rdkit_atom_count_i",
    "rdkit_heavy_atom_count",
    "rdkit_aromatic_ring_count",
    "rdkit_aliphatic_ring_count",
    "rdkit_rotatable_bond_count",
    "rdkit_hydrogen_bond_donor_count",
    "rdkit_hydrogen_bond_acceptor_count",
    "rdkit_topological_polar_surface_area",
    "rdkit_mol_logp",
)


INTERACTION_COLUMNS = (
    "rdkit_gasteiger_charge_min",
    "rdkit_gasteiger_charge_max",
    "rdkit_gasteiger_charge_mean",
    "rdkit_gasteiger_charge_std",
    "rdkit_gasteiger_positive_charge_sum",
    "rdkit_gasteiger_negative_charge_sum",
    "rdkit_gasteiger_absolute_charge_sum",
    "rdkit_gasteiger_implicit_hydrogen_charge_sum",
)


NAME_COLUMN_PREFIX = "dream_odor_name_"


EXPECTED_DRAGON_COLUMNS = 4868


EXPECTED_MORGAN_BITS = 2048


EXPECTED_DREAM_MOLECULES = 476


def load_dragon_feature_map(
    all_dragon_columns: list[str],
) -> tuple[list[str], dict[str, list[str]]]:
    """Return the Dragon columns and their four model groups.

    The groups are derived from the order of the DREAM Dragon columns, see
    ``dragon_feature_groups``, so no separate mapping file is needed.
    """
    dragon_groups = dragon_feature_groups(all_dragon_columns)
    dragon_columns = [column for columns in dragon_groups.values() for column in columns]
    if len(dragon_columns) != EXPECTED_DRAGON_COLUMNS or len(set(dragon_columns)) != len(
        dragon_columns
    ):
        raise ValueError("Dragon feature map must identify 4,868 unique Dragon columns.")
    # Keep the source column order, which is how the model indexes the Dragon matrix.
    grouped = set(dragon_columns)
    return [column for column in all_dragon_columns if column in grouped], dragon_groups


def dragon_group_indices(
    dragon_columns: list[str], dragon_groups: dict[str, list[str]]
) -> dict[str, list[int]]:
    """Translate Dragon group column names into positions in the Dragon matrix."""
    positions = {column: index for index, column in enumerate(dragon_columns)}
    return {
        group: [positions[column] for column in columns] for group, columns in dragon_groups.items()
    }


def name_windows(name: str) -> set[str]:
    """Return overlapping 4-character windows from a DREAM molecule label."""
    normalized_name = re.sub(r"\s*\(replicate\)\s*$", "", name.strip(), flags=re.I)
    normalized = normalized_name.upper().replace(" ", "_")
    return {normalized[index : index + 4] for index in range(max(0, len(normalized) - 3))}


def build_dream_odor_name_features(
    molecules: pd.DataFrame, dream_cids: set[int]
) -> tuple[pd.DataFrame, list[str]]:
    """Build binary 4-character windows from DREAM's OdorName per CID."""
    if "OdorName" not in molecules:
        raise ValueError("Molecule table must include DREAM's OdorName field.")
    source = molecules.loc[molecules["CID"].isin(dream_cids), ["CID", "OdorName"]]
    if source["CID"].duplicated().any() or source["OdorName"].isna().any():
        raise ValueError("DREAM molecule names must have one non-empty name per CID.")
    names_by_cid = {
        int(cid): str(name).strip() for cid, name in source.itertuples(index=False, name=None)
    }
    if set(names_by_cid) != dream_cids or len(names_by_cid) != EXPECTED_DREAM_MOLECULES:
        raise ValueError(
            f"Expected names for all {EXPECTED_DREAM_MOLECULES} DREAM molecules; "
            f"found {len(names_by_cid)}."
        )

    windows_by_cid = {cid: name_windows(name) for cid, name in names_by_cid.items()}
    vocabulary = sorted(set().union(*windows_by_cid.values()))
    columns = [f"{NAME_COLUMN_PREFIX}{index:05d}" for index in range(len(vocabulary))]
    feature_index = {window: index for index, window in enumerate(vocabulary)}
    matrix = np.zeros((len(names_by_cid), len(vocabulary)), dtype=np.float32)
    cids = sorted(names_by_cid)
    for row_index, cid in enumerate(cids):
        for window in windows_by_cid[cid]:
            matrix[row_index, feature_index[window]] = 1.0
    result = pd.DataFrame(matrix, columns=columns)
    result.insert(0, "CID", cids)
    return result, vocabulary


@dataclass
class MoleculeFeatures:
    """Every feature of the 476 DREAM molecules, one row per molecule, and the columns of each block."""

    molecules: pd.DataFrame
    dragon_columns: list[str]
    dragon_groups: dict[str, list[str]]
    morgan_columns: list[str]
    name_columns: list[str]
    name_vocabulary: list[str]
    dragon_cids: set[int]


def load_molecule_features() -> MoleculeFeatures:
    """Join the Dragon, RDKit, charge, Morgan, odor-name and published features of every molecule."""
    required_paths = [GRAPH_MOLECULES_PATH, FINGERPRINTS_PATH, COMMUNITY_PATH]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required training input is missing: {path}. "
                "Run ds-build to create the published features."
            )

    molecules = pd.read_csv(GRAPH_MOLECULES_PATH)
    fingerprints = pd.read_csv(FINGERPRINTS_PATH)
    all_dragon_columns = [column for column in molecules if column.startswith("dragon_")]
    morgan_columns = [column for column in fingerprints if column.startswith("morgan_")]
    required = {
        "CID",
        "rdkit_canonical_smiles",
        "rdkit_gasteiger_atom_charges",
        "rdkit_gasteiger_implicit_hydrogen_charges",
        *RDKIT_COLUMNS,
        *INTERACTION_COLUMNS,
    }
    if missing := required.difference(molecules.columns):
        raise ValueError(f"Molecule table is missing columns: {sorted(missing)}")
    if (
        len(all_dragon_columns) != EXPECTED_DRAGON_COLUMNS + 1
        or len(morgan_columns) != EXPECTED_MORGAN_BITS
    ):
        raise ValueError(
            "Expected 4,869 DREAM descriptor-source columns (4,868 Dragon plus "
            "one PubMed field) and 2,048 Morgan bits."
        )

    molecules["CID"] = pd.to_numeric(molecules["CID"], errors="raise").astype("int64")
    if molecules["CID"].duplicated().any():
        raise ValueError("Molecule input must contain one row per CID before feature joins.")
    dragon_columns, dragon_groups = load_dragon_feature_map(all_dragon_columns)
    # The feature selection (ds-build) names the columns that are kept from each dataset.
    dragon_columns = keep_selected("dragon", dragon_columns)
    dragon_groups = {
        group: keep_selected("dragon", columns) for group, columns in dragon_groups.items()
    }
    if not all(dragon_groups.values()):
        raise ValueError("The feature selection leaves a Dragon group empty. Run ds-build.")
    morgan_columns = keep_selected("morgan", morgan_columns)
    molecules[dragon_columns] = (
        molecules[dragon_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    dragon_cids = set(
        molecules.loc[molecules[dragon_columns].notna().any(axis=1), "CID"].astype(int)
    )
    name_reference, name_vocabulary = build_dream_odor_name_features(molecules, dragon_cids)
    name_columns = [
        column for column in name_reference.columns if column.startswith(NAME_COLUMN_PREFIX)
    ]
    kept_names = keep_selected("name_own", name_columns)
    if len(kept_names) != len(name_columns):
        # Column number i holds window i of the vocabulary, so both lose the same entries.
        position = {column: index for index, column in enumerate(name_columns)}
        name_vocabulary = [name_vocabulary[position[column]] for column in kept_names]
        name_reference = name_reference[["CID", *kept_names]]
        name_columns = kept_names
    molecules = molecules.merge(name_reference, on="CID", how="left", validate="one_to_one")
    fingerprints["CID"] = pd.to_numeric(fingerprints["CID"], errors="raise").astype("int64")
    molecules = molecules.merge(fingerprints, on="CID", how="inner", validate="one_to_one")
    community = pd.read_csv(COMMUNITY_PATH)
    community["CID"] = pd.to_numeric(community["CID"], errors="raise").astype("int64")
    chosen = ["CID"]
    for block, prefix in COMMUNITY_BLOCKS.items():
        columns = [column for column in community.columns if column.startswith(prefix)]
        chosen += keep_selected(block, columns)
    molecules = molecules.merge(community[chosen], on="CID", how="left", validate="one_to_one")
    return MoleculeFeatures(
        molecules,
        dragon_columns,
        dragon_groups,
        morgan_columns,
        name_columns,
        name_vocabulary,
        dragon_cids,
    )


def load_ratings(ratings_path: Path, features: MoleculeFeatures) -> pd.DataFrame:
    """One row per (CID, replicate, condition) of a DREAM rating file.

    A row holds the population mean and standard deviation of each attribute, every participant's own
    rating, the concentration, and every feature of the molecule.
    """
    if not ratings_path.is_file():
        raise FileNotFoundError(f"Required training input is missing: {ratings_path}.")
    ratings = read_rating_table(ratings_path).rename(columns={"Compound Identifier": "CID"})
    ratings["CID"] = pd.to_numeric(ratings["CID"], errors="raise").astype("int64")
    missing_rating_cids = set(ratings["CID"]) - features.dragon_cids
    if missing_rating_cids:
        raise ValueError(
            f"DREAM Dragon table is missing rating CIDs: {sorted(missing_rating_cids)}"
        )
    rating_rename = {
        column: normalize_target_name(column)
        for column in ratings.columns
        if column not in {"CID", "Odor", "Replicate", "Intensity", "Dilution", "subject #"}
    }
    ratings = ratings.rename(columns=rating_rename)
    ratings["condition"] = ratings["Intensity"].astype(str).str.strip().str.lower()
    ratings["replicate_key"] = ratings["Replicate"].fillna("").astype(str).str.strip()
    ratings["subject #"] = pd.to_numeric(ratings["subject #"], errors="coerce")
    ratings = ratings.loc[ratings["subject #"].isin(SUBJECT_IDS)].copy()
    ratings["subject #"] = ratings["subject #"].astype(int)
    if not ratings["condition"].isin({"high", "low"}).all():
        raise ValueError(f"{ratings_path.name} contains an unexpected intensity condition.")

    dilution = (
        ratings["Dilution"]
        .astype(str)
        .str.strip()
        .str.strip(" \"'")
        .str.replace(",", "", regex=False)
    )
    denominator = pd.to_numeric(dilution.str.split("/", n=1).str[-1], errors="raise")
    ratings["log10_concentration"] = -np.log10(denominator)
    ratings["is_high_concentration"] = ratings["condition"].eq("high").astype(float)
    if ratings_path in (LEADERBOARD_RATINGS_PATH, TEST_RATINGS_PATH):
        # The leaderboard and test files list one dilution for each molecule, and every molecule was
        # also rated at 1/1000, which is where its intensity comes from. A molecule listed at 1/10
        # has the pair 1/10 and 1/1000, so its descriptors are the high rating. A molecule listed at
        # 1/100,000 has the pair 1/1000 and 1/100,000, so its descriptors are the low rating. The
        # training rows that these molecules resemble are labelled the same way: a flag of "high" only
        # for dilutions stronger than 1/1000.
        ratings["is_high_concentration"] = (
            ratings["log10_concentration"] > STANDARD_LOG10_DILUTION
        ).astype(float)
    for target in TARGETS:
        if target not in ratings:
            raise ValueError(f"{ratings_path.name} is missing target {target!r}.")
        ratings[target] = pd.to_numeric(ratings[target], errors="coerce")

    # A replicate is a separate DREAM sample. Keep it distinct while averaging
    # duplicate rows within a subject, then average subjects for the population target.
    sample_keys = ["CID", "replicate_key", "condition"]
    by_subject = ratings.groupby([*sample_keys, "subject #"])[list(TARGETS)].mean()
    means = by_subject.groupby(level=sample_keys).mean()
    deviations = by_subject.groupby(level=sample_keys).std(ddof=1)
    individual = by_subject.unstack("subject #").reindex(
        columns=pd.MultiIndex.from_product([TARGETS, SUBJECT_IDS])
    )
    individual.columns = [
        f"q1_{target}_subject_{subject}" for target, subject in individual.columns
    ]
    target_frame = pd.concat(
        [means.add_prefix("mean_"), deviations.add_prefix("std_")], axis=1
    ).reset_index()
    target_frame = target_frame.merge(
        individual.reset_index(), on=sample_keys, validate="one_to_one"
    )
    condition_frame = (
        ratings.groupby(sample_keys)[["log10_concentration", "is_high_concentration"]]
        .first()
        .reset_index()
    )
    targets = target_frame.merge(condition_frame, on=sample_keys, validate="one_to_one")
    rows = targets.merge(features.molecules, on="CID", how="inner", validate="many_to_one")
    if rows.empty:
        raise ValueError(f"Could not build target rows from {ratings_path}.")
    if rows[features.name_columns].isna().any().any():
        raise ValueError("DREAM OdorName features are missing for one or more rating CIDs.")
    return rows.reset_index(drop=True)


@dataclass
class GraphInputs:
    """Rating rows and molecule features, loaded once and shared by both challenges."""

    rows: pd.DataFrame
    validation_rows: pd.DataFrame
    test_rows: pd.DataFrame
    dragon_columns: list[str]
    dragon_groups: dict[str, list[str]]
    morgan_columns: list[str]
    name_vocabulary: list[str]


def load_graph_inputs() -> GraphInputs:
    """Load the TrainSet, leaderboard, and TestSet rows with every molecule feature joined on."""
    features = load_molecule_features()
    rows = load_ratings(TRAIN_RATINGS_PATH, features)
    validation_rows = load_ratings(LEADERBOARD_RATINGS_PATH, features)
    test_rows = load_ratings(TEST_RATINGS_PATH, features)
    split = load_split()
    if set(validation_rows["CID"]) != set(split.validation):
        raise ValueError("The validation rows must be exactly the DREAM leaderboard set.")
    held_out_cids = set(split.test) | set(split.validation)
    leaked = set(rows["CID"]) & held_out_cids
    if leaked:
        raise ValueError(f"Held-out molecules are in the training rows: {sorted(leaked)}")
    print(
        f"Using {len(features.dragon_columns)} Dragon descriptors in {len(features.dragon_groups)} groups "
        f"and {len(features.name_vocabulary)} odor name features."
    )
    return GraphInputs(
        rows=rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        dragon_columns=features.dragon_columns,
        dragon_groups=features.dragon_groups,
        morgan_columns=features.morgan_columns,
        name_vocabulary=features.name_vocabulary,
    )
