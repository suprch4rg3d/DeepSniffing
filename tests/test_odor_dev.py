"""Tests for the dev scale: a small fixed copy of the data to try ideas on."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.preprocessing import odor_dev
from scripts.preprocessing.odor_tables import SPECIAL_TOKENS

MOLECULE_TOKENS = ["[C]", "[O]", "[N]", "[=O]"]
ODOR_TOKENS = ["fruity", "green"]
ROOT = Path(__file__).resolve().parents[1]


def full_tables(
    directory: Path, pubchem: int = 400, odor: int = 20, other: int = 30, valid: int = 10
):
    """A stand-in for the full tables: some of every kind of row the dev set has to treat differently."""
    rows = []
    cid = 1
    for source, count, split in (
        ("pubchem", pubchem, "train"),
        ("odor", odor, "train"),
        ("other", other, "train"),
        ("pubchem", valid, "validation"),
    ):
        for _ in range(count):
            rows.append((cid, "CCO", "[C][C][O]", source, split))
            cid += 1
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=odor_dev.COLUMNS).to_csv(directory / "structures.csv", index=False)
    pd.DataFrame({"cid": [1], "smiles": ["CCO"], "selfies": ["[C][C][O]"]}).to_csv(
        directory / "odor_dataset.csv", index=False
    )
    (directory / "vocabulary.json").write_text(
        json.dumps(
            {
                "special": list(SPECIAL_TOKENS),
                "molecule_tokens": MOLECULE_TOKENS,
                "odor_tokens": ODOR_TOKENS,
            }
        ),
        encoding="utf-8",
    )
    return directory


def fingerprint(directory: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())}


def test_the_dev_set_has_about_the_number_of_molecules_asked_for(tmp_path):
    full = full_tables(tmp_path / "full")
    report = odor_dev.build(full, tmp_path / "dev", target=200)
    assert report["molecules"] == 200
    assert report["packed_molecules"] == 200


def test_the_validation_molecules_and_the_odor_sources_are_always_kept(tmp_path):
    full = full_tables(tmp_path / "full")
    odor_dev.build(full, tmp_path / "dev", target=100)
    dev = pd.read_csv(tmp_path / "dev/structures.csv")
    every = pd.read_csv(full / "structures.csv")
    yardstick = set(every.loc[every["split"] == "validation", "cid"])
    assert yardstick <= set(dev["cid"])  # the same validation molecules at every scale
    assert set(every.loc[every["source"] != "pubchem", "cid"]) <= set(dev["cid"])


def test_a_smaller_dev_set_is_inside_a_larger_one(tmp_path):
    full = full_tables(tmp_path / "full")
    odor_dev.build(full, tmp_path / "small", target=150)
    odor_dev.build(full, tmp_path / "large", target=300)
    small = set(pd.read_csv(tmp_path / "small/structures.csv")["cid"])
    large = set(pd.read_csv(tmp_path / "large/structures.csv")["cid"])
    assert small < large


def test_the_same_build_gives_the_same_molecules_every_time(tmp_path):
    full = full_tables(tmp_path / "full")
    odor_dev.build(full, tmp_path / "one", target=200)
    odor_dev.build(full, tmp_path / "two", target=200)
    assert fingerprint(tmp_path / "one") == fingerprint(tmp_path / "two")


def test_the_vocabulary_and_the_fine_tuning_data_are_copied_unchanged(tmp_path):
    """A different vocabulary is a different model: the scales are only comparable if it is the same."""
    full = full_tables(tmp_path / "full")
    odor_dev.build(full, tmp_path / "dev", target=100)
    for name in ("vocabulary.json", "odor_dataset.csv"):
        assert (tmp_path / "dev" / name).read_bytes() == (full / name).read_bytes()


def test_the_dev_set_is_packed_with_that_vocabulary(tmp_path):
    full = full_tables(tmp_path / "full")
    odor_dev.build(full, tmp_path / "dev", target=100)
    stamp = json.loads((tmp_path / "dev/stage1_tokens.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((full / "vocabulary.json").read_text(encoding="utf-8"))
    assert stamp["tokens"] == [
        *vocabulary["special"],
        *vocabulary["molecule_tokens"],
        *vocabulary["odor_tokens"],
    ]
    tokens = np.load(tmp_path / "dev/stage1_tokens.npy")
    offsets = np.load(tmp_path / "dev/stage1_offsets.npy")
    assert (
        int(offsets[-1]) == len(tokens) == 100 * 3
    )  # a hundred molecules of three molecule tokens each


def test_building_the_dev_set_never_writes_to_the_full_tables(tmp_path):
    full = full_tables(tmp_path / "full")
    before = fingerprint(full)
    odor_dev.build(full, tmp_path / "dev", target=100)
    assert fingerprint(full) == before
    assert set(p.name for p in full.iterdir()) == set(before)  # and no file was added


def test_a_target_below_what_must_be_kept_still_keeps_it(tmp_path):
    full = full_tables(tmp_path / "full")
    report = odor_dev.build(full, tmp_path / "dev", target=1)
    assert report["from_pubchem"] == report["validation"]  # only the validation ones are PubChem's
    assert report["from_odor_sources"] == 50


def test_the_full_tables_must_exist(tmp_path):
    try:
        odor_dev.build(tmp_path / "nothing", tmp_path / "dev")
    except FileNotFoundError as error:
        assert "Build the full tables first" in str(error)
    else:
        raise AssertionError("it should have refused")


def run_python(code: str, scale: str | None) -> subprocess.CompletedProcess:
    environment = {**os.environ}
    environment.pop("DS_SCALE", None)
    if scale is not None:
        environment["DS_SCALE"] = scale
    return subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=environment, capture_output=True, text=True
    )


def test_ds_scale_moves_only_the_files_the_training_reads():
    code = (
        "from scripts.preprocessing import odor_tables, molecule_tokens, odor_tokens;"
        "print(odor_tables.STRUCTURES_PATH.parent.name, molecule_tokens.TOKENS_PATH.parent.name,"
        " odor_tokens.MOLECULES_PATH.parent.name)"
    )
    assert run_python(code, None).stdout.split() == ["odor", "odor", "odor"]
    assert run_python(code, "dev").stdout.split() == [
        "odor_dev",
        "odor_dev",
        "odor",
    ]  # odor tokens stay


def test_an_unknown_scale_is_refused_rather_than_read_as_the_full_one():
    result = run_python("import scripts.preprocessing.odor_tokens", "huge")
    assert result.returncode != 0 and "DS_SCALE" in result.stderr


def test_the_dev_scale_cannot_run_the_full_table_builder():
    result = run_python("from scripts.preprocessing import odor_tables; odor_tables.main()", "dev")
    assert result.returncode != 0 and "odor_dev" in result.stderr
