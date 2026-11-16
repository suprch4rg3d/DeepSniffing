"""Tests for the feature selection: which columns are kept from each dataset, and why."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from scripts.preprocessing import feature_selection as module

MOLECULES = 20


@pytest.fixture
def blocks():
    generator = np.random.default_rng(0)
    base = generator.normal(size=MOLECULES)
    first = pd.DataFrame(
        {
            "base": base,
            "constant": np.full(MOLECULES, 3.0),
            "copy_of_base": base * 2.5 + 1.0,  # the same quantity in other units
            "monotone_copy": np.exp(base),  # same ordering of the molecules
            "independent": generator.normal(size=MOLECULES),
        }
    )
    bits = np.zeros((MOLECULES, 4))
    bits[:5, 0] = 1  # five molecules
    bits[:2, 1] = 1  # two molecules
    bits[7, 2] = 1  # one molecule only
    bits[:5, 3] = 1  # the same five molecules as the first column
    second = pd.DataFrame(bits, columns=["bit_five", "bit_two", "bit_one", "bit_five_again"])
    third = pd.DataFrame({"repeat_of_base": base + 0.001 * generator.normal(size=MOLECULES)})
    return {"first": first, "second": second, "third": third}


@pytest.fixture
def sparse(monkeypatch):
    monkeypatch.setattr(module, "SPARSE_BLOCKS", {"second"})


def reasons(manifest: pd.DataFrame) -> dict[str, str]:
    return dict(zip(manifest["column"], manifest["reason"], strict=True))


def test_every_candidate_column_gets_one_row_with_a_reason(blocks, sparse):
    manifest = module.select_features(blocks)

    assert len(manifest) == sum(frame.shape[1] for frame in blocks.values())
    assert manifest["column"].is_unique
    assert set(manifest["reason"]) <= {"kept", "constant", "rare", "duplicate"}
    assert (manifest["kept"] == (manifest["reason"] == "kept")).all()


def test_constant_columns_are_dropped(blocks, sparse):
    assert reasons(module.select_features(blocks))["constant"] == "constant"


def test_a_column_held_by_one_molecule_is_rare_but_two_molecules_are_enough(blocks, sparse):
    found = reasons(module.select_features(blocks))

    assert found["bit_one"] == "rare"
    assert found["bit_two"] == "kept"
    assert found["bit_five"] == "kept"


def test_copies_are_dropped_and_say_which_column_they_copy(blocks, sparse):
    manifest = module.select_features(blocks).set_index("column")

    for column in ("copy_of_base", "monotone_copy"):
        assert manifest.loc[column, "reason"] == "duplicate"
        assert manifest.loc[column, "duplicate_of"] == "base"
    assert manifest.loc["independent", "kept"]


def test_the_earlier_block_keeps_its_column_when_two_blocks_repeat_each_other(blocks, sparse):
    manifest = module.select_features(blocks).set_index("column")

    assert manifest.loc["base", "kept"]
    assert manifest.loc["repeat_of_base", "reason"] == "duplicate"
    assert manifest.loc["repeat_of_base", "duplicate_of"] == "base"
    assert manifest.loc["bit_five_again", "duplicate_of"] == "bit_five"


def test_the_threshold_decides_what_counts_as_a_copy(blocks, sparse):
    strict = module.select_features(blocks, threshold=1.01)

    assert (strict["reason"] != "duplicate").all()


def test_missing_values_do_not_break_the_selection(blocks, sparse):
    blocks["first"].loc[0, "independent"] = np.nan
    blocks["first"].loc[1, "base"] = np.inf

    manifest = module.select_features(blocks)

    assert len(manifest) == sum(frame.shape[1] for frame in blocks.values())


def test_keep_selected_keeps_every_column_when_there_is_no_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "MANIFEST_PATH", tmp_path / "missing.csv")

    assert module.selected_columns("dragon") is None
    assert module.keep_selected("dragon", ["a", "b"]) == ["a", "b"]


def test_keep_selected_follows_the_manifest_and_keeps_the_column_order(monkeypatch, tmp_path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "block": ["dragon"] * 4 + ["epi"],
            "column": ["d1", "d2", "d3", "d4", "e1"],
            "kept": [True, False, True, True, True],
            "reason": ["kept", "constant", "kept", "kept", "kept"],
            "duplicate_of": [None] * 5,
        }
    ).to_csv(path, index=False)
    monkeypatch.setattr(module, "MANIFEST_PATH", path)

    assert module.keep_selected("dragon", ["d4", "d2", "d1", "unknown"]) == ["d4", "d1"]
    assert module.selected_columns("epi") == frozenset({"e1"})
    with pytest.raises(ValueError, match="ds-build"):
        module.selected_columns("morgan")


def test_a_rewritten_manifest_is_read_again(monkeypatch, tmp_path):
    path = tmp_path / "manifest.csv"
    monkeypatch.setattr(module, "MANIFEST_PATH", path)
    table = pd.DataFrame(
        {
            "block": ["dragon"],
            "column": ["d1"],
            "kept": [True],
            "reason": ["kept"],
            "duplicate_of": [None],
        }
    )
    table.to_csv(path, index=False)
    assert module.selected_columns("dragon") == frozenset({"d1"})

    table.assign(column="d2").to_csv(path, index=False)
    modified = path.stat().st_mtime_ns
    os.utime(path, ns=(modified, modified + 10**9))  # timestamps can be too coarse to differ

    assert module.selected_columns("dragon") == frozenset({"d2"})
