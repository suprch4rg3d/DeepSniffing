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
DREAM_COMMUNITY_DIR = RAW_DIR / "dream-olfaction-prediction/hulab"
PYRFUME_DIR = RAW_DIR / "pyrfume-data"
OUTPUT_PATH = RAW_DIR / "pubchem/dream_keller_compound_properties.csv"

# Pin source revisions so a rerun uses the same source data.
DREAM_COMMIT = "fb47cb343cdfd5cd8b06b33161dfa8b82c0319c6"
PYRFUME_COMMIT = "8054ea98ed675005ec10e67359902f500e4911b0"
DREAM_FILES = (
    "TrainSet.txt",
    "TestSet.txt",
    "LBs1.txt",
    "LBs2.txt",
    "CID_leaderboard.txt",
    "CID_testset.txt",
    "dilution_leaderboard.txt",
    # Features that the organizers and the community phase published for the DREAM molecules.
    # The Dragon descriptors come from commercial software, so only these published values exist.
    "molecular_descriptors_data.txt",
    "DREAM_episuite_descriptors.txt",
    "morgan_sim.csv",
    "name_features.txt",
    # De Grave's NSPDK vectors and the molecules they describe. The similarity is computed
    # from them, because the similarity matrix itself is not published.
    "derived/nspdk_r3_d4_unaug.svm",
    "derived/nspdk_cid.csv",
)
# NSPDK graph-kernel features, in the repository folder of the community phase.
DREAM_COMMUNITY_FILES = ("nspdk2.csv",)
DREAM_COMMUNITY_PATH = "opc_python/hulab/collaboration/features"
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

# PubChem's computed 3D structures. One request asks for a whole batch of compounds, and a
# compound without a 3D record is left out of the answer, so the whole set takes a handful
# of requests. PubChem allows five requests per second.
CONFORMERS_PATH = RAW_DIR / "pubchem/dream_conformers.sdf"
CONFORMERS_UNAVAILABLE_PATH = RAW_DIR / "pubchem/dream_conformers_unavailable.txt"
CONFORMER_BATCH_SIZE = 100
CONFORMER_PAUSE_SECONDS = 0.25
CONFORMER_TIMEOUT_SECONDS = 120


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

    for filename in DREAM_COMMUNITY_FILES:
        url = (
            "https://raw.githubusercontent.com/dream-olfaction/olfaction-prediction/"
            f"{DREAM_COMMIT}/{DREAM_COMMUNITY_PATH}/{filename}"
        )
        download_file(url, DREAM_COMMUNITY_DIR / filename, force)

    for filename in PYRFUME_FILES:
        url = f"https://raw.githubusercontent.com/pyrfume/pyrfume-data/{PYRFUME_COMMIT}/{filename}"
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
                int(entry["CID"]): {name: str(entry.get(name, "")) for name in PROPERTIES}
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


def read_wanted_cids() -> set[int] | None:
    """The DREAM and Keller compounds, or None when a source file is missing."""
    train_path = DREAM_DATA_DIR / "TrainSet.txt"
    test_path = DREAM_DATA_DIR / "TestSet.txt"
    targets_path = PYRFUME_DIR / "prediction_targets/keller_2016.csv"
    for path in (train_path, test_path, targets_path):
        if not path.is_file():
            print(f"Required source file not found: {path}", file=sys.stderr)
            return None

    cids = read_dream_cids(train_path) | read_dream_cids(test_path)
    # The leaderboard and test molecules are also listed on their own, which guards against a
    # Keller file that lacks one of them.
    for name in ("CID_leaderboard.txt", "CID_testset.txt"):
        listed = DREAM_DATA_DIR / name
        if listed.is_file():
            cids |= {int(token) for token in listed.read_text(encoding="utf-8").split()}
    with targets_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            cid = row.get("CID", "").strip()
            if cid.isdigit():
                cids.add(int(cid))
    return cids


def fetch_pubchem_properties() -> int:
    cids = read_wanted_cids()
    if cids is None:
        return 1

    cached: dict[int, dict[str, str]] = {}
    if OUTPUT_PATH.is_file():
        with OUTPUT_PATH.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("CID", "").isdigit():
                    cached[int(row["CID"])] = {name: row.get(name, "") for name in PROPERTIES}

    pending = sorted(cids - set(cached))
    print(f"{len(cids)} unique CIDs; {len(cached)} cached; fetching {len(pending)} from PubChem.")
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


def split_sdf_records(text: str) -> dict[int, str]:
    """Split a PubChem SDF into records, keyed by the compound identifier on each first line.

    A record that is cut short or does not start with an identifier is an error, so a broken
    download can never be mistaken for a compound without a 3D structure.
    """
    if not text.strip():
        return {}
    if not text.rstrip().endswith("$$$$"):
        raise ValueError("The SDF ends in the middle of a record.")
    records: dict[int, str] = {}
    for chunk in text.split("$$$$"):
        record = chunk.strip("\r\n")
        if not record.strip():
            continue
        first_line = record.split("\n", 1)[0].strip()
        if not first_line.isdigit() or "M  END" not in record:
            raise ValueError(f"Malformed SDF record starting {first_line!r}.")
        records[int(first_line)] = record + "\n$$$$\n"
    return records


def write_atomically(path: Path, text: str) -> None:
    """Write a file so that an interrupted run never leaves half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".part")
    temporary_path.write_text(text, encoding="utf-8", newline="\n")
    temporary_path.replace(path)


def fetch_conformer_batch(cids: list[int]) -> dict[int, str]:
    """Request the 3D structures of a batch of compounds, retrying a broken answer.

    PubChem answers 404 when none of the compounds has a 3D structure, and otherwise leaves
    out the ones that have none. Anything else is retried and then reported.
    """
    url = f"{PUG_REST}/{','.join(map(str, cids))}/record/SDF?record_type=3d"
    request = urllib.request.Request(url, headers={"User-Agent": "DeepSniffing data setup"})
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=CONFORMER_TIMEOUT_SECONDS) as response:
                records = split_sdf_records(response.read().decode("utf-8"))
            unexpected = set(records) - set(cids)
            if unexpected:
                raise ValueError(
                    f"PubChem returned compounds that were not requested: {unexpected}"
                )
            return records
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return {}
            failure: Exception = error
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            failure = error
        if attempt == RETRIES - 1:
            raise failure
        time.sleep(2 * (attempt + 1))
    return {}


def fetch_pubchem_conformers() -> int:
    """Download PubChem's 3D structure of every DREAM and Keller compound that has one.

    Structures already on disk are kept and only the missing compounds are requested, so a
    rerun is cheap and an interrupted run continues where it stopped. A compound that PubChem
    has no 3D structure for is remembered, so it is not requested again, and is built with
    RDKit later.
    """
    cids = read_wanted_cids()
    if cids is None:
        return 1

    cached: dict[int, str] = {}
    if CONFORMERS_PATH.is_file():
        cached = split_sdf_records(CONFORMERS_PATH.read_text(encoding="utf-8"))
    unavailable: set[int] = set()
    if CONFORMERS_UNAVAILABLE_PATH.is_file():
        unavailable = {int(token) for token in CONFORMERS_UNAVAILABLE_PATH.read_text().split()}

    pending = sorted(cids - set(cached) - unavailable)
    print(
        f"{len(cids)} compounds; {len(cached)} 3D structures cached, {len(unavailable)} known "
        f"to have none; requesting {len(pending)} in batches of {CONFORMER_BATCH_SIZE}."
    )
    failed = False
    for start in range(0, len(pending), CONFORMER_BATCH_SIZE):
        batch = pending[start : start + CONFORMER_BATCH_SIZE]
        try:
            records = fetch_conformer_batch(batch)
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            print(
                f"PubChem 3D request failed for CIDs {batch[0]}-{batch[-1]}: {error}",
                file=sys.stderr,
            )
            failed = True
            break
        cached.update(records)
        unavailable.update(set(batch) - set(records))
        print(f"Fetched {min(start + CONFORMER_BATCH_SIZE, len(pending))}/{len(pending)}")
        time.sleep(CONFORMER_PAUSE_SECONDS)

    write_atomically(CONFORMERS_PATH, "".join(cached[cid] for cid in sorted(cached)))
    write_atomically(
        CONFORMERS_UNAVAILABLE_PATH,
        "".join(f"{cid}\n" for cid in sorted(unavailable - set(cached))),
    )
    print(
        f"Saved {len(cached)} 3D structures to {CONFORMERS_PATH}; "
        f"{len(unavailable - set(cached))} compounds have none: {sorted(unavailable - set(cached))}"
    )
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download DREAM/Pyrfume inputs and fetch PubChem properties."
    )
    parser.add_argument("--download-only", action="store_true", help="download raw sources only")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="redownload source files even when already present",
    )
    args = parser.parse_args()

    download_sources(force=args.force_download)
    if args.download_only:
        return 0
    properties_status = fetch_pubchem_properties()
    return max(properties_status, fetch_pubchem_conformers())


if __name__ == "__main__":
    raise SystemExit(main())
