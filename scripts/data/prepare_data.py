from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data/raw"
DREAM_DATA_DIR = RAW_DIR / "dream-olfaction-prediction/data"
PYRFUME_DIR = RAW_DIR / "pyrfume-data"
OUTPUT_PATH = RAW_DIR / "pubchem/dream_keller_compound_properties.csv"

# Pin source revisions so a rerun uses the same source data.
DREAM_COMMIT = "fb47cb343cdfd5cd8b06b33161dfa8b82c0319c6"
PYRFUME_COMMIT = "8054ea98ed675005ec10e67359902f500e4911b0"
DREAM_FILES = (
    "TrainSet.txt",
    "TestSet.txt",
    "CID_leaderboard.txt",
    "CID_testset.txt",
    "dilution_leaderboard.txt",
    "dilution_testset.txt",
)
PYRFUME_FILES = (
    "keller_2016/molecules.csv",
    "molecules/molecules.csv",
    "prediction_targets/keller_2016.csv",
)

PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"
PROPERTIES = (
    "SMILES",
    "MolecularFormula",
    "MolecularWeight",
    "ExactMass",
    "InChI",
    "InChIKey",
    "IUPACName",
)
BATCH_SIZE = 100
RETRIES = 3


def download_file(url: str, destination: Path, force: bool) -> None:
    if destination.is_file() and not force:
        print(f"Already present: {destination.relative_to(PROJECT_ROOT)}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DeepSniffing data setup"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            temporary_path.write_bytes(response.read())
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"Downloaded: {destination.relative_to(PROJECT_ROOT)}")


def download_sources(force: bool) -> None:
    for filename in DREAM_FILES:
        url = (
            "https://raw.githubusercontent.com/dream-olfaction/olfaction-prediction/"
            f"{DREAM_COMMIT}/data/{filename}"
        )
        download_file(url, DREAM_DATA_DIR / filename, force)

    for filename in PYRFUME_FILES:
        url = (
            "https://raw.githubusercontent.com/pyrfume/pyrfume-data/"
            f"{PYRFUME_COMMIT}/{filename}"
        )
        download_file(url, PYRFUME_DIR / filename, force)


def read_dream_cids(path: Path) -> set[int]:
    cids: set[int] = set()
    with path.open(encoding="utf-8") as file:
        next(file, None)
        for line in file:
            cid = line.split("\t", 1)[0].strip()
            if cid.isdigit():
                cids.add(int(cid))
    return cids


def fetch_pubchem_batch(cids: list[int]) -> dict[int, dict[str, str]]:
    property_names = ",".join(PROPERTIES)
    url = f"{PUG_REST}/{','.join(map(str, cids))}/property/{property_names}/JSON"

    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.load(response)
            return {
                int(entry["CID"]): {
                    name: str(entry.get(name, "")) for name in PROPERTIES
                }
                for entry in payload["PropertyTable"]["Properties"]
            }
        except (urllib.error.URLError, KeyError, ValueError) as error:
            if attempt == RETRIES - 1:
                print(
                    f"PubChem request failed for CIDs {cids[0]}–{cids[-1]}: {error}",
                    file=sys.stderr,
                )
                return {}
            time.sleep(2 * (attempt + 1))
    return {}


def fetch_pubchem_properties() -> int:
    train_path = DREAM_DATA_DIR / "TrainSet.txt"
    test_path = DREAM_DATA_DIR / "TestSet.txt"
    targets_path = PYRFUME_DIR / "prediction_targets/keller_2016.csv"
    for path in (train_path, test_path, targets_path):
        if not path.is_file():
            print(f"Required source file not found: {path}", file=sys.stderr)
            return 1

    cids = read_dream_cids(train_path) | read_dream_cids(test_path)
    with targets_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            cid = row.get("CID", "").strip()
            if cid.isdigit():
                cids.add(int(cid))

    cached: dict[int, dict[str, str]] = {}
    if OUTPUT_PATH.is_file():
        with OUTPUT_PATH.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("CID", "").isdigit():
                    cached[int(row["CID"])] = {
                        name: row.get(name, "") for name in PROPERTIES
                    }

    pending = sorted(cids - set(cached))
    print(
        f"{len(cids)} unique CIDs; {len(cached)} cached; "
        f"fetching {len(pending)} from PubChem."
    )
    missing: list[int] = []
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start : start + BATCH_SIZE]
        fetched = fetch_pubchem_batch(batch)
        cached.update(fetched)
        missing.extend(cid for cid in batch if cid not in fetched)
        print(f"Fetched {min(start + BATCH_SIZE, len(pending))}/{len(pending)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["CID", *PROPERTIES])
        writer.writeheader()
        for cid in sorted(cids):
            if cid in cached:
                writer.writerow({"CID": cid, **cached[cid]})

    print(
        f"Saved {len(cids) - len(missing)}/{len(cids)} records to "
        f"{OUTPUT_PATH.relative_to(PROJECT_ROOT)}"
    )
    if missing:
        print(f"Missing PubChem CIDs: {sorted(missing)}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download DREAM/Pyrfume inputs and fetch PubChem properties."
    )
    parser.add_argument(
        "--download-only", action="store_true", help="download raw sources only"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="redownload source files even when already present",
    )
    args = parser.parse_args()

    download_sources(force=args.force_download)
    if args.download_only:
        return 0
    return fetch_pubchem_properties()


if __name__ == "__main__":
    raise SystemExit(main())
