"""Tests for the canonical split, including the leakage regression."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.core.data import (
    SPLIT_TABLE_PATH,
    TEST_RATINGS_PATH,
    TRAIN_RATINGS_PATH,
    aggregate_high_intensity_targets,
)
from scripts.core.splits import (
    TEST,
    TRAIN,
    UNUSED,
    VALIDATION,
    assign_split_column,
    assign_splits,
    official_test_cids,
    ordered_split_cids,
    training_pool_cids,
)


def test_official_test_set_has_the_expected_size():
    assert len(official_test_cids()) == 69


def test_training_pool_matches_the_dream_training_file():
    assert len(training_pool_cids()) == 338


def test_official_test_set_is_disjoint_from_the_training_pool():
    assert set(official_test_cids()).isdisjoint(training_pool_cids())


def test_assignment_holds_the_test_set_out():
    assignment = assign_splits(training_pool_cids() + official_test_cids())

    counts = pd.Series(assignment).value_counts()
    assert counts[TEST] == 69
    assert counts[TRAIN] == 287
    assert counts[VALIDATION] == 51

    trained_on = {cid for cid, split in assignment.items() if split in (TRAIN, VALIDATION)}
    assert trained_on.isdisjoint(official_test_cids())


def test_assignment_marks_unknown_molecules_as_unused():
    assignment = assign_splits([1, 2, 3])

    assert set(assignment.values()) == {UNUSED}


def test_assignment_is_deterministic():
    pool = training_pool_cids()

    assert assign_splits(pool) == assign_splits(pool)


def test_assignment_rejects_an_empty_input():
    with pytest.raises(ValueError):
        assign_splits([])


def test_assignment_rejects_an_impossible_validation_fraction():
    with pytest.raises(ValueError):
        assign_splits(training_pool_cids(), validation_fraction=1.5)


def test_ordered_split_covers_the_pool_once():
    train_cids, validation_cids = ordered_split_cids(training_pool_cids())

    assert len(train_cids) + len(validation_cids) == 338
    assert set(train_cids).isdisjoint(validation_cids)


def test_split_table_on_disk_matches_the_code():
    table = pd.read_csv(SPLIT_TABLE_PATH)
    recomputed = assign_split_column(table)

    mismatched = (table["split"] != recomputed["split"]).sum()
    assert mismatched == 0, (
        "molecule_labels_with_split.csv is stale. Rebuild it with "
        "python -m scripts.preprocessing.split_dataset"
    )


def test_test_rows_in_the_split_table_are_the_official_test_set():
    table = pd.read_csv(SPLIT_TABLE_PATH)

    labelled = set(table.loc[table["split"] == TEST, "CID"])
    assert labelled == set(official_test_cids())


def test_target_aggregation_matches_the_official_definition():
    train_targets = aggregate_high_intensity_targets(TRAIN_RATINGS_PATH)
    test_targets = aggregate_high_intensity_targets(TEST_RATINGS_PATH)

    assert len(train_targets) == 338
    assert len(test_targets) == 69
    assert set(test_targets["CID"]).issubset(official_test_cids())

    for frame in (train_targets, test_targets):
        means = [column for column in frame.columns if column.startswith("mean_")]
        deviations = [column for column in frame.columns if column.startswith("std_")]
        assert len(means) == 21
        assert len(deviations) == 21
        assert frame[means + deviations].notna().all().all()


def test_target_aggregation_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        aggregate_high_intensity_targets(tmp_path / "does_not_exist.txt")
