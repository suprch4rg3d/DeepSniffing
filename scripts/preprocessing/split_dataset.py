from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.core.splits import (
    SPLIT_NAMES,
    TEST,
    VALIDATION,
    assign_split_column,
    official_test_cids,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data/processed/dream_pyrfume"
BASE_INPUT_PATH = DATASET_DIR / "molecule_labels.csv"
ENRICHED_INPUT_PATH = DATASET_DIR / "molecules_enriched.csv"
BASE_OUTPUT_PATH = DATASET_DIR / "molecule_labels_with_split.csv"
ENRICHED_OUTPUT_PATH = DATASET_DIR / "molecules_enriched_with_split.csv"


def validate_split(molecules: pd.DataFrame) -> None:
    """Guard against the leakage bug that this module used to have."""
    if not {"CID", "split"}.issubset(molecules.columns):
        raise ValueError("Split dataset requires CID and split columns.")
    if molecules["split"].isna().any():
        raise ValueError("Split dataset contains molecules without a split.")

    unknown = set(molecules["split"]) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"Split dataset contains unknown split labels: {sorted(unknown)}")

    test_cids = set(official_test_cids())
    labelled_test = set(molecules.loc[molecules["split"] == TEST, "CID"])
    if labelled_test != test_cids:
        missing = sorted(test_cids - labelled_test)[:5]
        extra = sorted(labelled_test - test_cids)[:5]
        raise ValueError(
            "The test split must be exactly the official DREAM test set. "
            f"Missing: {missing}. Unexpected: {extra}."
        )

    leaked = labelled_test.intersection(
        set(molecules.loc[molecules["split"].isin(["train", VALIDATION]), "CID"])
    )
    if leaked:
        raise ValueError(
            f"Official test compounds were assigned to train or validation: {sorted(leaked)[:5]}"
        )


def split_molecules(molecules: pd.DataFrame) -> pd.DataFrame:
    """Attach the canonical split assignment to a molecule table."""
    required_columns = {"CID", "rdkit_canonical_smiles"}
    missing_columns = required_columns.difference(molecules.columns)
    if missing_columns:
        raise ValueError(
            f"Input dataset is missing required columns: {', '.join(sorted(missing_columns))}"
        )

    canonical_smiles = molecules["rdkit_canonical_smiles"].fillna("").astype(str).str.strip()
    if canonical_smiles.eq("").any():
        raise ValueError("Input dataset contains missing RDKit-canonicalized SMILES.")

    result = assign_split_column(molecules)
    validate_split(result)
    return result


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
        validate_split(enriched)
        enriched.to_csv(ENRICHED_OUTPUT_PATH, index=False)
    return base, enriched


def main() -> None:
    base, enriched = create_splits()
    dataset = enriched if enriched is not None else base
    output_path = ENRICHED_OUTPUT_PATH if enriched is not None else BASE_OUTPUT_PATH
    counts = dataset["split"].value_counts().reindex(SPLIT_NAMES, fill_value=0)
    print(f"Saved {len(base)} base molecules to {BASE_OUTPUT_PATH}")
    print(f"Saved {len(dataset)} molecules to {output_path}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
