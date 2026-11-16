"""Choose the exact feature columns that the model takes from each dataset.

Several datasets describe the same molecules, so their columns overlap: a descriptor that
RDKit computes is usually one that Dragon already holds, and many Morgan bits, name
windows, and substructure counts are constant or nearly identical to another column.
This step decides, once, which columns are kept and writes the decision to
``feature_selection.csv``, one row per candidate column with the reason it was dropped.

The rules use only the features, never the ratings:

    constant     the column has one value for every DREAM molecule
    rare         a binary or substructure column that only one molecule has, which the
                 model could only memorise
    duplicate    the column has a rank correlation of at least 0.995 with a column that was
                 already kept, which is recorded in ``duplicate_of``

Blocks are visited in a fixed priority order, and columns in their source order, so the
first column of a group of duplicates is the one that is kept. Earlier blocks win: Dragon is
the most established source, then RDKit and charge features, EPI Suite, Morgan bits, the
name windows, the NSPDK substructures, and last the two similarity sets.

The selection sees the features of all 476 DREAM molecules, test molecules included, just
as the name windows already do. No rating is involved, so no target can leak.

    poetry run ds-build
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts.core.splits import PROCESSED_DIR
from scripts.preprocessing.community_features import COMMUNITY_BLOCKS, MIN_MOLECULES
from scripts.preprocessing.community_features import OUTPUT as COMMUNITY_PATH
from scripts.preprocessing.enrich_molecules import OUTPUT_PATH as MOLECULES_PATH
from scripts.preprocessing.morgan_fingerprints import OUTPUT as FINGERPRINTS_PATH

MANIFEST_PATH = PROCESSED_DIR / "feature_selection.csv"

# 0.99 drops P_VSA_i_1, a top valence feature, as a copy of P_VSA_MR_8 (rank correlation 0.9916).
# With 0.995 it is kept (test score 35.85 against 36.34 with 0.99, within the run-to-run noise).
CORRELATION_THRESHOLD = 0.995
CHUNK = 1000

# Priority order. Each block name is how the model asks for its columns.
SPARSE_BLOCKS = {"morgan", "name_own", "name_organizer", "nspdk"}


def rank_standardised(values: np.ndarray) -> np.ndarray:
    """Rank each column, so the correlation is a Spearman correlation, then standardise."""
    filled = np.where(np.isnan(values), np.nanmedian(values, axis=0), values)
    ranks = np.apply_along_axis(rankdata, 0, filled)
    ranks -= ranks.mean(axis=0)
    scale = ranks.std(axis=0)
    return (ranks / np.where(scale == 0, 1.0, scale)).astype(np.float32)


def select_features(
    blocks: dict[str, pd.DataFrame], threshold: float = CORRELATION_THRESHOLD
) -> pd.DataFrame:
    """Return the manifest for blocks of columns that describe the same molecules."""
    n_molecules = len(next(iter(blocks.values())))
    kept = np.zeros((n_molecules, 0), dtype=np.float32)
    kept_names: list[str] = []
    records = []
    for block, frame in blocks.items():
        columns = list(frame.columns)
        values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        values = np.where(np.isinf(values), np.nan, values)
        with np.errstate(all="ignore"):
            spread = np.nanstd(values, axis=0)
        constant = ~np.isfinite(spread) | (spread == 0)
        support = (np.nan_to_num(values) != 0).sum(axis=0)
        rare = (support < MIN_MOLECULES) & ~constant if block in SPARSE_BLOCKS else constant & False
        reason = np.where(constant, "constant", np.where(rare, "rare", "")).astype(object)
        duplicate_of = np.full(len(columns), "", dtype=object)

        usable = np.flatnonzero(reason == "")
        standardised = rank_standardised(values[:, usable])
        for start in range(0, len(usable), CHUNK):
            chunk = standardised[:, start : start + CHUNK]
            positions = usable[start : start + CHUNK]
            copy_of: list[str | None] = [None] * chunk.shape[1]
            if kept.shape[1]:
                correlation = np.abs(chunk.T @ kept) / n_molecules
                best = correlation.argmax(axis=1)
                hit = correlation[np.arange(len(best)), best] >= threshold
                for index in np.flatnonzero(hit):
                    copy_of[index] = kept_names[best[index]]
            alive = [index for index, name in enumerate(copy_of) if name is None]
            gram = np.abs(chunk[:, alive].T @ chunk[:, alive]) / n_molecules
            chosen: list[int] = []
            for place, index in enumerate(alive):
                if chosen:
                    nearest = chosen[int(np.argmax(gram[place, chosen]))]
                    if gram[place, nearest] >= threshold:
                        copy_of[index] = columns[positions[alive[nearest]]]
                        continue
                chosen.append(place)
            survivors = [alive[place] for place in chosen]
            kept = np.hstack([kept, chunk[:, survivors]])
            kept_names += [columns[positions[index]] for index in survivors]
            for index, name in enumerate(copy_of):
                if name is not None:
                    reason[positions[index]] = "duplicate"
                    duplicate_of[positions[index]] = name

        for index, column in enumerate(columns):
            records.append(
                {
                    "block": block,
                    "column": column,
                    "kept": reason[index] == "",
                    "reason": reason[index] or "kept",
                    "duplicate_of": duplicate_of[index],
                }
            )
    return pd.DataFrame(records)


def candidate_blocks() -> dict[str, pd.DataFrame]:
    """Every candidate column, by block, for the 476 DREAM molecules."""
    # Imported here because the data module reads the manifest this module writes.
    from scripts.core.data import (
        INTERACTION_COLUMNS,
        RDKIT_COLUMNS,
        build_dream_odor_name_features,
        load_dragon_feature_map,
    )

    for path in (MOLECULES_PATH, FINGERPRINTS_PATH, COMMUNITY_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required table not found: {path}. Build the data first.")
    molecules = pd.read_csv(MOLECULES_PATH)
    molecules["CID"] = molecules["CID"].astype("int64")
    community = pd.read_csv(COMMUNITY_PATH).set_index("CID")
    fingerprints = pd.read_csv(FINGERPRINTS_PATH).set_index("CID")
    cids = sorted(int(cid) for cid in community.index)
    indexed = molecules.set_index("CID")

    dragon_columns, _ = load_dragon_feature_map([c for c in molecules if c.startswith("dragon_")])
    has_dragon = indexed.index[indexed[dragon_columns].notna().any(axis=1)]
    names, _ = build_dream_odor_name_features(molecules, {int(cid) for cid in has_dragon})
    names = names.set_index("CID")

    def prefixed(prefix: str) -> list[str]:
        return [column for column in community.columns if column.startswith(prefix)]

    return {
        "dragon": indexed.loc[cids, dragon_columns],
        "rdkit": indexed.loc[cids, list(RDKIT_COLUMNS)],
        "charge": indexed.loc[cids, list(INTERACTION_COLUMNS)],
        "epi": community.loc[cids, prefixed("epi_")],
        "morgan": fingerprints.loc[cids],
        "name_own": names.loc[cids],
        "name_organizer": community.loc[cids, prefixed("orgname_")],
        "nspdk": community.loc[cids, prefixed("nspdk_")],
        "msim": community.loc[cids, prefixed(COMMUNITY_BLOCKS["msim"])],
        "nsim": community.loc[cids, prefixed(COMMUNITY_BLOCKS["nsim"])],
    }


def build_manifest() -> pd.DataFrame:
    manifest = select_features(candidate_blocks())
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    summary = manifest.groupby("block", sort=False).agg(
        candidates=("column", "size"),
        kept=("kept", "sum"),
        constant=("reason", lambda r: int((r == "constant").sum())),
        rare=("reason", lambda r: int((r == "rare").sum())),
        duplicate=("reason", lambda r: int((r == "duplicate").sum())),
    )
    print(summary.to_string())
    print(
        f"\nKept {int(manifest['kept'].sum())} of {len(manifest)} columns "
        f"(duplicate threshold {CORRELATION_THRESHOLD}, rare means one molecule only)."
    )
    print(f"Saved the manifest to {MANIFEST_PATH}")
    return manifest


@functools.lru_cache(maxsize=4)
def _read_manifest(path: str, modified: int) -> dict[str, frozenset[str]]:
    manifest = pd.read_csv(path)
    kept = manifest.loc[manifest["kept"].astype(bool)]
    return {block: frozenset(group["column"]) for block, group in kept.groupby("block")}


def selected_columns(block: str, path: Path | None = None) -> frozenset[str] | None:
    """The columns of ``block`` that the manifest keeps, or None when there is no manifest."""
    path = path or MANIFEST_PATH
    if not path.is_file():
        return None
    selection = _read_manifest(str(path), path.stat().st_mtime_ns)
    if block not in selection:
        raise ValueError(f"The feature manifest has no block {block!r}. Run ds-build again.")
    return selection[block]


def keep_selected(block: str, columns: list[str]) -> list[str]:
    """The given columns that the manifest keeps, in their order, or all without a manifest."""
    selected = selected_columns(block)
    return list(columns) if selected is None else [c for c in columns if c in selected]


def main() -> None:
    build_manifest()


if __name__ == "__main__":
    main()
