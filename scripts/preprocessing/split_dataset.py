"""A step of ``ds-build``: write the canonical split that Subchallenge 1 and 2 train, validate and test on.

The split follows the DREAM challenge: the 338 TrainSet molecules train, the 69 leaderboard
molecules validate, and the 69 official test molecules are held out. The four molecules of the table
that took part in none of these sets are marked unused. ``ds-train`` reads this file, and checks it
against the DREAM lists again before it trains:

    CID, split, train_order      train_order is the fixed order of the training molecules
"""

from __future__ import annotations

import pandas as pd

from scripts.core.splits import SPLIT_NAMES, SPLIT_PATH, build_split, load_split
from scripts.preprocessing.dream_dataset import OUTPUT_PATH as MOLECULES_PATH


def create_split() -> pd.DataFrame:
    if not MOLECULES_PATH.is_file():
        raise FileNotFoundError(f"Molecule table not found: {MOLECULES_PATH}. Run ds-build.")
    cids = pd.to_numeric(pd.read_csv(MOLECULES_PATH, usecols=["CID"])["CID"], errors="raise")
    table = build_split(cids.astype("int64").to_numpy())
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(SPLIT_PATH, index=False)
    load_split()  # the file that was just written must pass the checks that training makes
    return table


def main() -> None:
    table = create_split()
    counts = table["split"].value_counts().reindex(SPLIT_NAMES, fill_value=0)
    print(f"Saved the split of {len(table)} molecules to {SPLIT_PATH}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
