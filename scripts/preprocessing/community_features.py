"""Join the features that the DREAM organizers and community phase published, keyed by CID.

The challenge published several feature families for its 476 molecules. They cannot be
recomputed from a SMILES string here, so they are read from the downloaded files and
written to one table, like the Morgan fingerprints:

    orgname_*   the organizers' binary 4-letter windows of the molecule names, which are
                a larger vocabulary than the windows built from the DREAM odor names
    epi_*       EPI Suite physical property estimates, such as vapor pressure and log Kow
    nspdk_*     NSPDK graph-kernel substructure features
    nsim_*      NSPDK similarity to a fixed set of reference odorants, the dot product of
                the NSPDK vectors that Kurt De Grave published
    msim_*      Morgan similarity to the same reference odorants

The similarities compare every DREAM molecule with about 2,400 odorants, and the DREAM
molecules are among those odorants. A molecule's similarity to itself is 1, so each
molecule would own one column that no other molecule can match. Those 476 columns are
dropped, which leaves similarities to reference odorants that are not DREAM molecules.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.core.splits import PROCESSED_DIR, RATINGS_DIR

DREAM_DIR = RATINGS_DIR.parent
NAME_PATH = DREAM_DIR / "data/name_features.txt"
EPI_PATH = DREAM_DIR / "data/DREAM_episuite_descriptors.txt"
MORGAN_SIMILARITY_PATH = DREAM_DIR / "data/morgan_sim.csv"
NSPDK_PATH = DREAM_DIR / "hulab/nspdk2.csv"
NSPDK_VECTORS_PATH = DREAM_DIR / "data/derived/nspdk_r3_d4_unaug.svm"
NSPDK_CIDS_PATH = DREAM_DIR / "data/derived/nspdk_cid.csv"
OUTPUT = PROCESSED_DIR / "published_features.csv"

NAME_PREFIX = "orgname_"
EPI_PREFIX = "epi_"
NSPDK_PREFIX = "nspdk_"
NSPDK_SIMILARITY_PREFIX = "nsim_"
MORGAN_SIMILARITY_PREFIX = "msim_"
# The model reads these together as one block of structure features.
STRUCTURE_PREFIXES = (NSPDK_PREFIX, NSPDK_SIMILARITY_PREFIX, MORGAN_SIMILARITY_PREFIX)

EXPECTED_MOLECULES = 476
# A name window or a substructure that only one molecule has cannot be learned from: the
# model could only memorise that molecule. Columns that two or more molecules share stay.
MIN_MOLECULES = 2

# The feature block that each column prefix belongs to, as ``feature_selection`` names them.
COMMUNITY_BLOCKS = {
    "name_organizer": NAME_PREFIX,
    "epi": EPI_PREFIX,
    "nspdk": NSPDK_PREFIX,
    "nsim": NSPDK_SIMILARITY_PREFIX,
    "msim": MORGAN_SIMILARITY_PREFIX,
}


def read_name_windows() -> pd.DataFrame:
    """The organizers' name windows: the first column is the CID, the rest are 0 or 1."""
    table = pd.read_csv(NAME_PATH, sep="\t", header=None)
    table.index = pd.Index(table.iloc[:, 0].astype("int64"), name="CID")
    windows = table.iloc[:, 1:]
    if not np.isin(windows.to_numpy(), (0, 1)).all():
        raise ValueError(f"Name windows must be binary: {NAME_PATH}")
    windows = windows.loc[:, windows.sum(axis=0) >= MIN_MOLECULES].astype(np.uint8)
    windows.columns = [f"{NAME_PREFIX}{index:05d}" for index in range(windows.shape[1])]
    return windows


def read_epi_suite() -> pd.DataFrame:
    """EPI Suite estimates as numbers. The SMILES column is dropped and YES or NO becomes 1 or 0."""
    table = pd.read_csv(EPI_PATH, sep="\t")
    table.index = pd.Index(table["CID"].astype("int64"), name="CID")
    table = table.drop(columns=["CID", "SMILES"])
    for column in table.columns:
        if not pd.api.types.is_numeric_dtype(table[column]):
            text = table[column].astype(str).str.strip().str.upper()
            table[column] = pd.to_numeric(
                text.map({"YES": 1, "NO": 0}).fillna(text), errors="coerce"
            )
    table = table.astype(np.float64).replace([np.inf, -np.inf], np.nan)
    varying = table.nunique(dropna=True) > 1
    table = table.loc[:, varying]
    names = [re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_") for name in table.columns]
    table.columns = [f"{EPI_PREFIX}{index:02d}_{name}" for index, name in enumerate(names)]
    return table


def read_morgan_similarity() -> pd.DataFrame:
    """Morgan similarity to the reference odorants, without each molecule's own column."""
    morgan = pd.read_csv(MORGAN_SIMILARITY_PATH, index_col=0)
    morgan.index = pd.Index(morgan.index.astype("int64"), name="CID")
    morgan.columns = morgan.columns.astype("int64")
    dream_cids = set(morgan.index)
    reference = [position for position, cid in enumerate(morgan.columns) if cid not in dream_cids]
    own = np.array([list(morgan.columns).index(cid) for cid in morgan.index])
    if not np.allclose(morgan.to_numpy()[np.arange(len(morgan)), own], 1.0, atol=0.05):
        raise ValueError("A Morgan similarity column does not belong to its own molecule.")
    kept = morgan.iloc[:, reference].astype(np.float64)
    kept.columns = [f"{MORGAN_SIMILARITY_PREFIX}{position:04d}" for position in reference]
    return kept


def read_nspdk_similarity(dream_cids: list[int]) -> pd.DataFrame:
    """NSPDK similarity to the reference odorants, without each molecule's own column.

    The organizers define the similarity as the dot product of the NSPDK feature vectors, the
    Gramian matrix in De Grave's script. Only the vectors are published, so the dot products
    are computed here.
    """
    cids = pd.read_csv(NSPDK_CIDS_PATH, header=None, dtype="int64")[0].to_numpy()
    rows, columns, values = [], [], []
    with NSPDK_VECTORS_PATH.open(encoding="utf-8") as file:
        for row, line in enumerate(file):
            for item in line.split()[1:]:
                feature, value = item.split(":")
                rows.append(row)
                columns.append(int(feature))
                values.append(float(value))
    vectors = sparse.csr_matrix((values, (rows, columns)), shape=(row + 1, max(columns) + 1))
    if vectors.shape[0] != len(cids):
        raise ValueError("The NSPDK vectors and the list of their molecules differ in length.")

    position = {int(cid): index for index, cid in enumerate(cids)}
    missing = sorted(set(dream_cids) - set(position))
    if missing:
        raise ValueError(f"The NSPDK vectors have no molecule for CIDs {missing[:5]}.")
    own = np.array([position[cid] for cid in dream_cids])
    gram = (vectors[own] @ vectors.T).toarray()
    if not np.allclose(gram[np.arange(len(own)), own], 1.0, atol=0.05):
        raise ValueError(
            "The NSPDK vectors are not normalised, so their dot product is no similarity."
        )
    reference = [index for index, cid in enumerate(cids) if int(cid) not in set(dream_cids)]
    kept = pd.DataFrame(
        gram[:, reference],
        index=pd.Index(dream_cids, name="CID"),
        columns=[f"{NSPDK_SIMILARITY_PREFIX}{index:04d}" for index in reference],
    )
    return kept


def read_similarities() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The NSPDK and the Morgan similarity of every DREAM molecule to the reference odorants."""
    morgan = read_morgan_similarity()
    return read_nspdk_similarity([int(cid) for cid in morgan.index]), morgan


def read_nspdk() -> pd.DataFrame:
    """NSPDK substructure features that at least ``MIN_MOLECULES`` molecules share."""
    table = pd.read_csv(NSPDK_PATH, index_col=0)
    table.index = pd.Index(table.index.astype("int64"), name="CID")
    table = table.loc[:, (table.to_numpy() != 0).sum(axis=0) >= MIN_MOLECULES].astype(np.float64)
    table.columns = [f"{NSPDK_PREFIX}{index:05d}" for index in range(table.shape[1])]
    return table


def build_community_features() -> pd.DataFrame:
    for path in (
        NAME_PATH,
        EPI_PATH,
        MORGAN_SIMILARITY_PATH,
        NSPDK_PATH,
        NSPDK_VECTORS_PATH,
        NSPDK_CIDS_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required source file not found: {path}. Run ds-prepare.")

    nspdk_similarity, morgan_similarity = read_similarities()
    families = {
        "orgname": read_name_windows(),
        "epi": read_epi_suite(),
        "nspdk": read_nspdk(),
        "nsim": nspdk_similarity,
        "msim": morgan_similarity,
    }
    cids = families["epi"].index
    for name, table in families.items():
        if table.index.duplicated().any() or set(table.index) != set(cids):
            raise ValueError(f"The {name} features must cover the same unique molecules.")
    if len(cids) != EXPECTED_MOLECULES:
        raise ValueError(f"Expected {EXPECTED_MOLECULES} DREAM molecules; found {len(cids)}.")

    result = pd.concat([table.loc[cids] for table in families.values()], axis=1).copy()
    result = result.sort_index().reset_index()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False, float_format="%.6g")
    counts = ", ".join(f"{name} {table.shape[1]}" for name, table in families.items())
    print(
        f"Saved {len(result)} molecules with {result.shape[1] - 1} features ({counts}) to {OUTPUT}"
    )
    return result


def main() -> None:
    build_community_features()


if __name__ == "__main__":
    main()
