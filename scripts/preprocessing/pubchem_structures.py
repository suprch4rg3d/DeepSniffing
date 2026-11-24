"""A step of ``ds-build``: many more molecules for the pretraining, from PubChem.

The pretraining learns the language of molecules and needs no odor tokens, so any molecule will do. PubChem
lists every compound it knows as a CID and a SMILES in one file, and the weight of every compound in
another, so the two together give far more molecules than the odor sources hold.

    pubchem_structures.csv     cid, smiles, selfies

The work is done in four steps, each of which is skipped if what it makes is already there:

1. **download** ``CID-SMILES.gz`` and ``CID-Mass.gz`` (1.5 GB and 1.3 GB);
2. **unpack** both. A gzip stream can only be read from its start by one thread, so while the files stay
   packed nothing can be shared out; unpacked, they can be read from anywhere;
3. **the weight mask**: one bit per CID, true where the compound weighs ``MAX_WEIGHT`` or less. The
   weight never has to be worked out, because PubChem has worked it out already. The file is cut into
   pieces at line ends and every core takes a piece.
4. **convert**: the same again over the structures. A core takes a piece, keeps the lines whose CID is
   marked in the mask, turns those into canonical SMILES and SELFIES, and writes them to a part file of
   its own; the part files are joined at the end. Nothing is held in memory but the piece being read.

A molecule is left out, and counted, when RDKit cannot read its SMILES, when SELFIES cannot write it, when
it is longer than the model can read at all (``MAX_MOLECULE_TOKENS``; the decoder has a position for 1024 molecule
molecule tokens), or when it is too heavy to be a molecule that smells (``MAX_WEIGHT``). To be smelled at all a
molecule has to evaporate and reach the nose, and heavy molecules do not; 300 or so is a rule of thumb
rather than a wall, so the limit here is taken from the molecules we have rather than from the rule: of the
7,516 with odor tokens, 98.5% weigh 350 or less and the median weighs 174.

Nothing is left out for the elements it is made of: the odor molecules themselves hold sodium, iron,
calcium, zinc and boron. Nothing here knows about the splits either -- the molecules held out of the
pretraining are taken out by ``odor_tables.build_structures``, where that decision already lives.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from scripts.data.prepare_data import (
    PUBCHEM_MASS_PATH,
    PUBCHEM_MASS_URL,
    PUBCHEM_SMILES_PATH,
    PUBCHEM_SMILES_URL,
    download_streamed,
    unzipped,
)
from scripts.preprocessing.odor_tokens import ODOR_DIR

RDLogger.DisableLog("rdApp.*")

STRUCTURES_PATH = ODOR_DIR / "pubchem_structures.csv"
MASK_PATH = (
    ODOR_DIR / "pubchem_light.npy"
)  # the weight mask, on disk so every core can share one copy
LIGHT_PATH = (
    ODOR_DIR / "pubchem_light.csv"
)  # cid, smiles: the light molecules, before any chemistry
SHARDS_DIR = ODOR_DIR / "pubchem_parts"  # what each core writes, joined when they are all done
MAX_CID = 300_000_000  # above every CID PubChem has handed out (the highest is about 180 million)
MAX_WEIGHT = 350.0  # see the module docstring
# How many of the light molecules are converted by default. It is what the models were trained on: with
# 33 tokens a molecule it comes to about 100 million tokens, some twenty a parameter of a model of this
# size, and it is what the tables and the packing after it can read in memory. The pool itself, all of
# PubChem at or under MAX_WEIGHT, stays on disk, so a different number is one command.
TARGET = 3_400_000
MAX_MOLECULE_TOKENS = 1024  # the decoder's positions for a molecule (``model.MAX_POSITIONS``)
MAX_SMILES = (
    200  # nothing under MAX_WEIGHT needs more, so this throws the big ones out before RDKit
)
PIECE = 64 * 1024 * 1024  # bytes of a file one core reads at a time
ALL = 1 << 32  # a cutoff no checksum reaches, so every molecule is taken
# The pretraining's validation molecules, fixed for good: the same ones whatever else is converted, so
# that a loss from one run can be set beside a loss from another. They are picked out of the whole pool
# by the checksum of the CID, not by shuffling a table, because a shuffle depends on how long the table
# is and so gives different molecules every time the data changes.
VALIDATION_MOLECULES = 5_000
POOL = 58_854_892  # compounds at or under MAX_WEIGHT, from the mask
VALIDATION_CUTOFF = round(ALL * VALIDATION_MOLECULES / POOL)
WORKERS = max(1, (os.cpu_count() or 2) - 1)
REASONS = (
    "too long",
    "RDKit cannot read it",
    "SELFIES cannot write it",
    "too heavy to smell",
)


def convert(row: tuple[int, str]) -> tuple[int, str, str] | int:
    """A molecule as ``(cid, canonical smiles, selfies)``, or the index of the reason it was left out."""
    cid, smiles = row
    if len(smiles) > MAX_SMILES:
        return 0
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return 1
    if Descriptors.MolWt(molecule) > MAX_WEIGHT:
        return 3
    canonical = Chem.MolToSmiles(molecule)
    try:
        selfies = sf.encoder(canonical)
        length = len(list(sf.split_selfies(selfies)))
    except Exception:
        return 2
    if selfies is None or length > MAX_MOLECULE_TOKENS:
        return 0
    return cid, canonical, selfies


def pieces(path: Path, each: int = PIECE) -> list[tuple[int, int]]:
    """The file cut into pieces of about ``each`` bytes, every piece starting where a line starts.

    A cut lands in the middle of a line, so it is moved on to the end of that line. The line it was in
    belongs to the piece before, and so every line is in one piece and no line is in two.
    """
    size = path.stat().st_size
    edges = [0]
    with path.open("rb") as file:
        for cut in range(each, size, each):
            file.seek(cut)
            file.readline()
            where = file.tell()
            if edges[-1] < where < size:
                edges.append(where)
    edges.append(size)
    # pairwise, so the second list is one shorter on purpose
    return [(start, end) for start, end in zip(edges, edges[1:], strict=False) if start < end]


def lines_of(path: Path, start: int, end: int) -> list[str]:
    """The lines of one piece of a file, read and decoded in one go."""
    with path.open("rb") as file:
        file.seek(start)
        block = file.read(end - start)
    return block.decode("utf-8", "replace").splitlines()


def _light_cids(job: tuple[Path, int, int, float]) -> np.ndarray:
    """The CIDs of one piece of the mass file whose compounds are light enough."""
    path, start, end, limit = job
    found: list[int] = []
    for line in lines_of(path, start, end):
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            if float(parts[2]) <= limit:
                found.append(int(parts[0]))
        except ValueError:
            continue
    return np.array(found, dtype=np.int64)


def light_mask(mass_path: Path, limit: float = MAX_WEIGHT, workers: int = WORKERS) -> np.ndarray:
    """One bit per CID, true where the compound weighs ``limit`` or less.

    A CID is a number, so the answer is an array and looking a compound up is reading one place in it.
    Nothing is sorted and nothing is searched for.
    """
    mask = np.zeros(MAX_CID + 1, dtype=bool)
    jobs = [(mass_path, start, end, limit) for start, end in pieces(mass_path)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for cids in pool.map(_light_cids, jobs):
            if cids.size:
                if cids.max() > MAX_CID:
                    raise ValueError(f"CID {cids.max()} is past MAX_CID ({MAX_CID}); raise it")
                mask[cids] = True
    return mask


_SHARED_MASK: np.ndarray | None = None


def _mask_of(path: Path) -> np.ndarray:
    """The mask, mapped from the file once per core. The pages are shared, so there is one copy in all."""
    global _SHARED_MASK
    if _SHARED_MASK is None:
        _SHARED_MASK = np.load(path, mmap_mode="r")
    return _SHARED_MASK


def _select_piece(job: tuple[Path, int, int, Path, Path]) -> int:
    """Write the ``cid,smiles`` of one piece's light molecules to a part file; return how many."""
    path, start, end, mask_path, shard_path = job
    mask = _mask_of(mask_path)
    kept = 0
    with shard_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(
            file, lineterminator="\n"
        )  # never \r\n: it has broken these tables before
        for line in lines_of(path, start, end):
            cid, _, smiles = line.partition("\t")
            smiles = smiles.strip()
            if not smiles:
                continue
            try:
                number = int(cid)
            except ValueError:
                continue
            if number <= MAX_CID and mask[number]:
                writer.writerow((number, smiles))
                kept += 1
    return kept


def chosen(cid: int, cutoff: int) -> bool:
    """Whether this molecule is in the sample.

    The CID is mixed first and the mix compared with ``cutoff``, rather than taking every n-th CID:
    PubChem hands CIDs out in the order compounds were registered, so a batch of ten thousand analogues
    deposited together sits in one run of consecutive numbers, and every n-th of them is not an
    independent draw. A checksum of the number has nothing to do with that order, and gives the same
    sample every time it is run.
    """
    return mixed(cid) < cutoff


def mixed(cid: int) -> int:
    """A number from 0 to 2**32 that has nothing to do with the order the CIDs were handed out in.

    Signed, because a molecule PubChem could not identify is given a negative id by ``odor_tokens``. A
    sample of the molecules with the smallest numbers is the same sample whatever its size, so a small one
    is always inside a larger one (``odor_dev`` relies on this).
    """
    return zlib.crc32(cid.to_bytes(8, "big", signed=True))


def for_validation(cid: int) -> bool:
    """Whether this molecule is one of the pretraining's fixed validation molecules."""
    return zlib.crc32(cid.to_bytes(8, "big", signed=True) + b"validation") < VALIDATION_CUTOFF


def _convert_piece(job: tuple[Path, int, int, Path, int]) -> tuple[int, list[int], int]:
    """Convert one piece of the light table into a part file; return kept, rejected and read."""
    path, start, end, shard_path, cutoff = job
    kept, rejected, read = 0, [0] * len(REASONS), 0
    with shard_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        for line in lines_of(path, start, end):
            cid, _, smiles = line.partition(",")
            smiles = smiles.strip()
            if not smiles or not cid.isdigit():
                continue  # the header, or a line that is not a molecule
            number = int(cid)
            # the validation molecules are always converted, however small the sample is
            if not (chosen(number, cutoff) or for_validation(number)):
                continue
            read += 1
            outcome = convert((number, smiles))
            if isinstance(outcome, int):
                rejected[outcome] += 1
            else:
                writer.writerow(outcome)
                kept += 1
    return kept, rejected, read


def join_shards(shards: list[Path], destination: Path, header: bytes) -> None:
    """Put the part files together into one table, written beside the old one and renamed when whole."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    # Binary throughout: a text layer and its ``buffer`` do not share one order of writes, and the header
    # would not stay at the top.
    with temporary.open("wb") as table:
        table.write(header)
        for shard in shards:
            if shard.is_file():
                with shard.open("rb") as part:
                    shutil.copyfileobj(part, table, length=8 * 1024 * 1024)
        table.flush()
        os.fsync(table.fileno())
    temporary.replace(destination)


def shard_paths(count: int) -> list[Path]:
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    return [SHARDS_DIR / f"{number:05d}.csv" for number in range(count)]


def select(smiles_path: Path, mask_path: Path, destination: Path, workers: int = WORKERS) -> int:
    """Write ``cid,smiles`` for every light molecule: the cheap step, with no chemistry in it at all."""
    cuts = pieces(smiles_path)
    shards = shard_paths(len(cuts))
    jobs = [
        (smiles_path, start, end, mask_path, shard)
        for (start, end), shard in zip(cuts, shards, strict=True)
    ]
    kept = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, more in enumerate(pool.map(_select_piece, jobs), start=1):
            kept += more
            print(f"  piece {done} of {len(jobs)}: {kept:,} light molecules", flush=True)
    join_shards(shards, destination, header=b"cid,smiles\n")
    shutil.rmtree(SHARDS_DIR, ignore_errors=True)
    return kept


def build(
    light_path: Path, destination: Path, cutoff: int = ALL, workers: int = WORKERS
) -> tuple[int, list[int], int]:
    """Turn the light molecules into canonical SMILES and SELFIES: the step the time goes into."""
    cuts = pieces(light_path)
    shards = shard_paths(len(cuts))
    jobs = [
        (light_path, start, end, shard, cutoff)
        for (start, end), shard in zip(cuts, shards, strict=True)
    ]
    kept, rejected, read = 0, [0] * len(REASONS), 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, (more, counts, saw) in enumerate(pool.map(_convert_piece, jobs), start=1):
            kept, read = kept + more, read + saw
            rejected = [a + b for a, b in zip(rejected, counts, strict=True)]
            print(f"  piece {done} of {len(jobs)}: {kept:,} of {read:,} converted", flush=True)
    join_shards(shards, destination, header=b"cid,smiles,selfies\n")
    shutil.rmtree(SHARDS_DIR, ignore_errors=True)
    return kept, rejected, read


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--target",
        type=int,
        default=TARGET,
        help="about how many molecules to convert, drawn evenly from the pool (0: all of them, "
        "which is 58.85 million and more than the later steps can read)",
    )
    parser.add_argument("--workers", type=int, default=WORKERS, help="cores to read and convert on")
    parser.add_argument(
        "--step",
        choices=("select", "convert", "all"),
        default="all",
        help="stop after picking the light molecules out, or do only the chemistry on what was picked",
    )
    arguments = parser.parse_args(argv)

    # Downloaded once, then unpacked once: everything after this reads plain files, in pieces, on
    # every core at once.
    smiles_path = unzipped(download_streamed(PUBCHEM_SMILES_URL, PUBCHEM_SMILES_PATH, force=False))
    mass_path = unzipped(download_streamed(PUBCHEM_MASS_URL, PUBCHEM_MASS_PATH, force=False))

    # 3. the weight mask
    if MASK_PATH.is_file():
        print(f"Already built: {MASK_PATH.name}")
        mask = np.load(MASK_PATH, mmap_mode="r")
    else:
        print(f"Reading the weight of every compound on {arguments.workers} cores", flush=True)
        mask = light_mask(mass_path, MAX_WEIGHT, arguments.workers)
        np.save(MASK_PATH, mask)
    print(f"{int(np.count_nonzero(mask)):,} compounds weigh {MAX_WEIGHT:.0f} or less")

    # 4. the light molecules as they are written, with no chemistry done to them yet
    if arguments.step in ("select", "all"):
        if LIGHT_PATH.is_file():
            print(f"Already built: {LIGHT_PATH.name}")
        else:
            print(f"Picking the light molecules out of {smiles_path.name}", flush=True)
            found = select(smiles_path, MASK_PATH, LIGHT_PATH, arguments.workers)
            print(f"Saved {found:,} molecules to {LIGHT_PATH}")
    if arguments.step == "select":
        return

    # 5. the slow step, over all of the pool or over a sample of it
    if not LIGHT_PATH.is_file():
        raise FileNotFoundError(f"{LIGHT_PATH} not found. Run this with --step select first.")
    pool = int(np.count_nonzero(mask))
    cutoff, where = ALL, "every one"
    if 0 < arguments.target < pool:
        cutoff = max(1, round(ALL * arguments.target / pool))
        where = f"about {arguments.target:,} of the {pool:,}"
    print(f"Converting {where} on {arguments.workers} cores", flush=True)
    kept, rejected, read = build(LIGHT_PATH, STRUCTURES_PATH, cutoff, arguments.workers)
    print(f"Saved {kept:,} of the {read:,} molecules read to {STRUCTURES_PATH}")
    for reason, count in zip(REASONS, rejected, strict=True):
        if count:
            print(f"  left out, {reason}: {count:,}")


if __name__ == "__main__":
    main()
