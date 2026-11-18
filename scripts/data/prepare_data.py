from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data/raw"
PYRFUME_DIR = RAW_DIR / "pyrfume-data"
# Every compound of PubChem as CID and SMILES, for the pretraining. 1.5 GB, so it is not downloaded by
# ``ds-prepare`` with the rest: ``scripts.preprocessing.pubchem_structures`` asks for it when it needs it.
PUBCHEM_SMILES_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"
PUBCHEM_SMILES_PATH = RAW_DIR / "pubchem/CID-SMILES.gz"
# The weight of every compound, which PubChem has worked out already: cid, formula, mass, mass. Reading it
# is far cheaper than asking RDKit for the weight of 125 million molecules.
PUBCHEM_MASS_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-Mass.gz"
PUBCHEM_MASS_PATH = RAW_DIR / "pubchem/CID-Mass.gz"

PYRFUME_COMMIT = "8054ea98ed675005ec10e67359902f500e4911b0"
PYRFUME_FILES = (
    "keller_2016/molecules.csv",
    "molecules/molecules.csv",
)
# Odor tokens of thousands of molecules, one folder for each source. Every source lists the odor tokens
# a perfumery or flavor reference gives to a molecule (yes or no), except Dravnieks, whose
# ratings are graded. Each folder also holds the structures of its molecules.
ODOR_SOURCES = {
    "leffingwell": ("behavior.csv", "stimuli.csv"),
    "goodscents": ("behavior.csv", "molecules.csv", "stimuli.csv"),
    "arctander_1960": ("behavior_1.csv", "behavior_2.csv", "molecules.csv", "stimuli.csv"),
    "flavordb": ("behavior.csv", "molecules.csv", "stimuli.csv"),
    "sharma_2021a": ("behavior.csv", "molecules.csv", "stimuli.csv"),
    "sharma_2021b": ("behavior_1.csv", "molecules.csv", "stimuli.csv", "odors.csv"),
    "ifra_2019": ("behavior.csv", "molecules.csv", "stimuli.csv"),
    "aromadb": ("behavior.csv", "molecules.csv", "stimuli.csv"),
    "sigma_2014": ("behavior.csv", "molecules.csv", "stimuli.csv"),
    "flavornet": ("behavior.csv", "molecules.csv", "stimuli.csv"),
    "dravnieks_1985": ("behavior_1.csv", "behavior_2.csv", "molecules.csv", "stimuli.csv"),
}

PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"
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


def download_streamed(url: str, destination: Path, force: bool) -> Path:
    """Download a file too big to hold in memory, in pieces, and only put it in place once it is whole."""
    if destination.is_file() and not force:
        print(f"Already present: {destination.relative_to(PROJECT_ROOT)}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DeepSniffing data setup"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            expected = int(response.headers.get("Content-Length") or 0)
            with temporary_path.open("wb") as file:
                shutil.copyfileobj(response, file, length=1024 * 1024)
        written = temporary_path.stat().st_size
        if expected and written != expected:
            raise OSError(f"{url} gave {written} bytes of {expected}")
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"Downloaded: {destination.relative_to(PROJECT_ROOT)} ({written / 1e9:.2f} GB)")
    return destination


def unzipped(archive: Path, force: bool = False) -> Path:
    """Unpack a ``.gz`` beside itself, once, and return the path to the plain file.

    A gzip stream can only be read from its start, straight through, by one thread: that is how DEFLATE
    works and no number of cores changes it. A plain file can be read from anywhere, so unpacking once
    turns every later pass over the data into something that can be split and shared out.
    """
    destination = archive.with_suffix("")
    if destination.is_file() and not force:
        print(f"Already unpacked: {destination.relative_to(PROJECT_ROOT)}")
        return destination
    temporary_path = destination.with_name(destination.name + ".part")
    print(f"Unpacking {archive.name} ...", flush=True)
    try:
        with gzip.open(archive, "rb") as packed, temporary_path.open("wb") as plain:
            shutil.copyfileobj(packed, plain, length=8 * 1024 * 1024)
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    print(
        f"Unpacked: {destination.relative_to(PROJECT_ROOT)} ({destination.stat().st_size / 1e9:.2f} GB)"
    )
    return destination


def download_sources(force: bool) -> None:
    for filename in PYRFUME_FILES:
        url = f"https://raw.githubusercontent.com/pyrfume/pyrfume-data/{PYRFUME_COMMIT}/{filename}"
        download_file(url, PYRFUME_DIR / filename, force)

    for source, filenames in ODOR_SOURCES.items():
        for filename in filenames:
            path = f"{source}/{filename}"
            url = f"https://raw.githubusercontent.com/pyrfume/pyrfume-data/{PYRFUME_COMMIT}/{path}"
            download_file(url, PYRFUME_DIR / path, force)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the odor sources of Pyrfume.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="redownload source files even when already present",
    )
    download_sources(force=parser.parse_args().force_download)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
