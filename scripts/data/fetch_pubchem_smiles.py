#!/usr/bin/env python3
"""Fetch SMILES from PubChem for every compound in the DREAM dataset.

Reads the PubChem CIDs found in data/raw/dream-olfaction/{TrainSet,TestSet}.txt,
queries the PubChem PUG-REST API in batches of 100, and writes the mapping to
data/raw/dream-olfaction/cid_smiles.tsv.

Note: PubChem currently serves the canonical-SMILES property under the
"ConnectivitySMILES" JSON key; we accept both that and "CanonicalSMILES".

Usage:  python scripts/data/fetch_pubchem_smiles.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RAW_DIR = Path("data/raw/dream-olfaction")
OUT_FILE = RAW_DIR / "cid_smiles.tsv"
PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"
BATCH = 100
RETRIES = 3
SMILES_KEYS = ("ConnectivitySMILES", "CanonicalSMILES")


def read_cids(path: Path) -> set[int]:
    """Extract unique CIDs from the first column of a DREAM data file."""
    cids: set[int] = set()
    with path.open() as fh:
        next(fh)  # skip header
        for line in fh:
            cid = line.split("\t", 1)[0].strip()
            if cid.isdigit():
                cids.add(int(cid))
    return cids


def fetch_batch(cids: list[int]) -> dict[int, str]:
    """Fetch SMILES for a batch of CIDs; returns {cid: smiles}."""
    url = f"{PUG_REST}/{','.join(map(str, cids))}/property/CanonicalSMILES/JSON"
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                payload = json.load(resp)
            rows = {}
            for entry in payload["PropertyTable"]["Properties"]:
                cid = int(entry["CID"])
                smiles = next((entry[k] for k in SMILES_KEYS if entry.get(k)), "")
                if smiles:
                    rows[cid] = smiles
            return rows
        except (urllib.error.HTTPError, KeyError, ValueError) as err:
            if attempt == RETRIES - 1:
                print(f"  ! failed for batch {cids[0]}..{cids[-1]}: {err}", file=sys.stderr)
                return {}
            time.sleep(2 * (attempt + 1))
    return {}


def main() -> int:
    if not RAW_DIR.exists():
        print("Run scripts/data/download_dream_data.sh first.", file=sys.stderr)
        return 1

    cids = read_cids(RAW_DIR / "TrainSet.txt") | read_cids(RAW_DIR / "TestSet.txt")

    # Load anything previously fetched successfully.
    cache: dict[int, str] = {}
    if OUT_FILE.exists():
        with OUT_FILE.open() as fh:
            next(fh, None)
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2 and parts[1]:
                    cache[int(parts[0])] = parts[1]

    todo = sorted(cids - set(cache))
    print(f"{len(cids)} unique compound CIDs; {len(cache)} already cached; fetching {len(todo)}")

    failed: list[int] = []
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        cache.update(fetch_batch(batch))
        unresolved = [c for c in batch if c not in cache]
        if unresolved:
            failed += unresolved
            print(f"  ! unresolved in batch: {unresolved}", file=sys.stderr)
        print(f"  fetched {min(i + BATCH, len(todo))}/{len(todo)}")

    # Rewrite the file from scratch (single header, sorted, no stale rows).
    with OUT_FILE.open("w") as fh:
        fh.write("cid\tsmiles\n")
        for cid in sorted(cids):
            if cid in cache:
                fh.write(f"{cid}\t{cache[cid]}\n")

    missing = sorted(cids - set(cache))
    print(f"Done. Wrote {len(cids) - len(missing)}/{len(cids)} molecules to {OUT_FILE}")
    if missing:
        print(f"  ! missing SMILES for: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())