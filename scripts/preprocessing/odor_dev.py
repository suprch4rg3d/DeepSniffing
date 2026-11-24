"""The dev scale: a small, fixed copy of the training data, for trying ideas in minutes.

A full run reads 3.4 million molecules and takes about three hours for the pretraining alone, which is too
slow to learn anything from. So an idea is first tried here, on a few hundred thousand, against a baseline
that was trained on the same few hundred thousand: same molecules, same validation molecules, same
vocabulary, same settings, and only the one thing under test different. What helps here is then checked
once at full scale.

    python -m scripts.preprocessing.odor_dev                    # build it (about a minute)
    $env:DS_SCALE = "dev"                                       # then everything reads and writes dev
    python -m scripts.core --max-epochs 8                  # a whole run, a few minutes

It is cut out of the full tables, so nothing is downloaded or converted again, and the full files are only
read: ``data/processed/odor`` is never written to. The dev files go to ``data/processed/odor_dev``:

    structures.csv        the pretraining molecules: the validation ones, every molecule of the odor
                          sources, and a share of the PubChem sample
    odor_dataset.csv      the fine-tuning molecules -- all of them, unchanged: they are only 7,516 and
                          they are the part that has to be right
    vocabulary.json       the full vocabulary, copied. A smaller set would have fewer molecule tokens, and
                          a different vocabulary is a different model, so a loss at dev scale could not be
                          set beside one at full scale. With the same tokens it can, and a checkpoint
                          trained at one scale loads at the other.
    stage1_*.npy / .json  the packed tokens of structures.csv

What is kept of the PubChem molecules is the ones with the smallest ``pubchem_structures.mixed`` number, so
the choice is deterministic, unrelated to the order of the CIDs, and a smaller dev set is always inside a
larger one.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from scripts.core.vocabulary import Vocabulary
from scripts.preprocessing.molecule_tokens import write as write_tokens
from scripts.preprocessing.odor_tokens import ODOR_DIR, PROJECT_ROOT
from scripts.preprocessing.pubchem_structures import mixed

DEV_DIR = PROJECT_ROOT / "data/processed/odor_dev"
MOLECULES = (
    500_000  # about a seventh of the full pretraining set; a pass over it takes about a minute
)
COLUMNS = ["cid", "smiles", "selfies", "source", "split"]


def select(structures: pd.DataFrame, target: int) -> pd.DataFrame:
    """The rows of the full pretraining table that make up the dev one, about ``target`` of them.

    The validation molecules and the molecules of the odor sources are always kept: the first are the
    yardstick and must be the same at every scale, the second are the ones the fine-tuning later reads,
    and there are only some thirty thousand of them. The rest of the room is given to the PubChem
    molecules with the smallest ``mixed`` numbers.
    """
    always = (structures["split"] == "validation") | (structures["source"] != "pubchem")
    room = max(0, target - int(always.sum()))
    candidates = structures[~always]
    order = candidates["cid"].map(mixed).to_numpy().argsort(kind="stable")
    chosen = candidates.iloc[order[:room]]
    return pd.concat([structures[always], chosen]).sort_values("cid", ignore_index=True)


def build(full_dir: Path = ODOR_DIR, dev_dir: Path = DEV_DIR, target: int = MOLECULES) -> dict:
    """Cut the dev set out of the full tables in ``full_dir`` and write it, packed, into ``dev_dir``."""
    for name in ("structures.csv", "odor_dataset.csv", "vocabulary.json"):
        if not (full_dir / name).is_file():
            raise FileNotFoundError(f"{full_dir / name} not found. Build the full tables first.")
    dev_dir.mkdir(parents=True, exist_ok=True)

    structures = pd.read_csv(full_dir / "structures.csv")
    dev = select(structures, target)
    dev[COLUMNS].to_csv(dev_dir / "structures.csv", index=False, lineterminator="\n")
    # the fine-tuning data and the vocabulary are copied as they are, which is what keeps the scales
    # comparable
    for name in ("odor_dataset.csv", "vocabulary.json"):
        shutil.copyfile(full_dir / name, dev_dir / name)

    vocabulary = Vocabulary.load(dev_dir / "vocabulary.json")
    tokens, splits = write_tokens(vocabulary, dev_dir / "structures.csv", dev_dir)
    return {
        "molecules": len(dev),
        "validation": int((dev["split"] == "validation").sum()),
        "from_odor_sources": int((dev["source"] != "pubchem").sum()),
        "from_pubchem": int((dev["source"] == "pubchem").sum()),
        "tokens": len(tokens),
        "packed_molecules": len(splits),
        "vocabulary": len(vocabulary),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--molecules", type=int, default=MOLECULES, help="about how many pretraining molecules"
    )
    arguments = parser.parse_args(argv)
    report = build(target=arguments.molecules)
    print(f"Built the dev set in {DEV_DIR}")
    for name, value in report.items():
        print(f"  {name:20s} {value:,}")
    print("\nTo use it: $env:DS_SCALE = 'dev'   (then ds-train, python -m scripts.core.evaluate)")


if __name__ == "__main__":
    main()
