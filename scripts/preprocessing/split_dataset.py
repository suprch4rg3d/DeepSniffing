from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data/processed/dream_pyrfume"
BASE_INPUT_PATH = DATASET_DIR / "molecule_labels.csv"
ENRICHED_INPUT_PATH = DATASET_DIR / "molecules_enriched.csv"
BASE_OUTPUT_PATH = DATASET_DIR / "molecule_labels_with_split.csv"
ENRICHED_OUTPUT_PATH = DATASET_DIR / "molecules_enriched_with_split.csv"
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_SEED = 42


def split_molecules(molecules: pd.DataFrame) -> pd.DataFrame:
    fractions = (TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION)
    if any(fraction <= 0 or fraction >= 1 for fraction in fractions):
        raise ValueError("Each split fraction must be greater than 0 and less than 1.")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Train, validation, and test fractions must sum to 1.")
    required_columns = {"CID", "rdkit_canonical_smiles"}
    missing_columns = required_columns.difference(molecules.columns)
    if missing_columns:
        raise ValueError(
            "Input dataset is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
    if molecules["CID"].isna().any():
        raise ValueError("Input dataset contains missing CIDs; resolve them before splitting.")
    if molecules["CID"].duplicated().any():
        raise ValueError("Expected one row per CID. Aggregate duplicate compounds first.")
    canonical_smiles = molecules["rdkit_canonical_smiles"].fillna("").astype(str).str.strip()
    if canonical_smiles.eq("").any():
        raise ValueError("Input dataset contains missing RDKit-canonicalized SMILES.")

    cids = molecules["CID"].to_numpy()
    train_validation_fraction = TRAIN_FRACTION + VALIDATION_FRACTION
    train_validation_cids, test_cids = train_test_split(
        cids, test_size=TEST_FRACTION, random_state=RANDOM_SEED, shuffle=True
    )
    relative_validation_fraction = VALIDATION_FRACTION / train_validation_fraction
    train_cids, validation_cids = train_test_split(
        train_validation_cids,
        test_size=relative_validation_fraction,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    split_by_cid = {}
    for cid in train_cids:
        split_by_cid[cid] = "train"
    for cid in validation_cids:
        split_by_cid[cid] = "validation"
    for cid in test_cids:
        split_by_cid[cid] = "test"
    molecules["split"] = molecules["CID"].map(split_by_cid)
    if molecules["split"].isna().any():
        raise RuntimeError("Failed to assign a split to every molecule.")
    return molecules


def create_splits() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not BASE_INPUT_PATH.is_file():
        raise FileNotFoundError(f"Input dataset does not exist: {BASE_INPUT_PATH}")

    base = split_molecules(pd.read_csv(BASE_INPUT_PATH))
    BASE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(BASE_OUTPUT_PATH, index=False)

    enriched = None
    if ENRICHED_INPUT_PATH.is_file():
        enriched = pd.read_csv(ENRICHED_INPUT_PATH)
        if set(enriched["CID"]) != set(base["CID"]):
            raise ValueError("Base and enriched datasets must contain the same CIDs.")
        split_by_cid = base.set_index("CID")["split"]
        enriched["split"] = enriched["CID"].map(split_by_cid)
        if enriched["split"].isna().any():
            raise RuntimeError("Failed to assign an enriched molecule split.")
        enriched.to_csv(ENRICHED_OUTPUT_PATH, index=False)
    return base, enriched


def main() -> None:
    base, enriched = create_splits()
    dataset = enriched if enriched is not None else base
    output_path = ENRICHED_OUTPUT_PATH if enriched is not None else BASE_OUTPUT_PATH
    counts = dataset["split"].value_counts().reindex(
        ["train", "validation", "test"], fill_value=0
    )
    print(f"Saved {len(base)} base molecules to {BASE_OUTPUT_PATH}")
    print(f"Saved {len(dataset)} molecules to {output_path}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
