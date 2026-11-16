"""Tests for the canonical split, including the leakage regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.core.data import (
    aggregate_high_intensity_targets,
    read_official_leaderboard_targets,
)
from scripts.core.metrics import TARGET_COLUMNS, normalize_target_name
from scripts.core.splits import (
    LEADERBOARD_RATINGS_PATH,
    TEST,
    TEST_RATINGS_PATH,
    TRAIN,
    TRAIN_RATINGS_PATH,
    UNUSED,
    VALIDATION,
    assign_splits,
    build_split,
    leaderboard_cids,
    load_split,
    official_test_cids,
    training_pool_cids,
)


def test_official_test_set_has_the_expected_size():
    assert len(official_test_cids()) == 69


def test_training_pool_matches_the_dream_training_file():
    assert len(training_pool_cids()) == 338


def test_official_test_set_is_disjoint_from_the_training_pool():
    assert set(official_test_cids()).isdisjoint(training_pool_cids())


def test_leaderboard_set_has_the_expected_size():
    assert len(leaderboard_cids()) == 69


def test_leaderboard_set_is_disjoint_from_training_and_test():
    assert set(leaderboard_cids()).isdisjoint(training_pool_cids())
    assert set(leaderboard_cids()).isdisjoint(official_test_cids())


def test_assignment_trains_on_the_trainset_and_validates_on_the_leaderboard():
    assignment = assign_splits(training_pool_cids() + leaderboard_cids() + official_test_cids())

    counts = pd.Series(assignment).value_counts()
    assert counts[TRAIN] == 338
    assert counts[VALIDATION] == 69
    assert counts[TEST] == 69
    assert {cid for cid, split in assignment.items() if split == TRAIN} == set(training_pool_cids())
    assert {cid for cid, split in assignment.items() if split == VALIDATION} == set(
        leaderboard_cids()
    )

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


def test_the_split_table_trains_on_the_whole_pool_and_validates_on_the_leaderboard():
    table = build_split(training_pool_cids() + leaderboard_cids() + official_test_cids() + [1])

    train = table[table["split"] == TRAIN]
    assert set(train["CID"]) == set(training_pool_cids())
    assert len(train) == 338
    assert set(table.loc[table["split"] == VALIDATION, "CID"]) == set(leaderboard_cids())
    assert set(table.loc[table["split"] == TEST, "CID"]) == set(official_test_cids())
    assert table.loc[table["CID"] == 1, "split"].item() == UNUSED


def test_the_training_order_is_one_fixed_shuffle_of_the_training_molecules():
    first = build_split(training_pool_cids() + official_test_cids())
    second = build_split(training_pool_cids() + official_test_cids())

    assert first.equals(second)
    order = first.dropna(subset=["train_order"]).sort_values("train_order")["CID"].tolist()
    assert sorted(order) == sorted(training_pool_cids())
    assert order != sorted(order)
    assert first.loc[first["split"] != TRAIN, "train_order"].isna().all()


def test_the_saved_split_loads_and_is_the_dream_split():
    split = load_split()

    assert len(split.train) == 338 and len(split.validation) == 69 and len(split.test) == 69
    assert set(split.train) == set(training_pool_cids())
    assert set(split.validation) == set(leaderboard_cids())
    assert set(split.test) == set(official_test_cids())
    assert set(split.train).isdisjoint(split.test) and set(split.validation).isdisjoint(split.test)


def test_a_split_file_with_a_held_out_molecule_in_training_is_refused(tmp_path):
    table = build_split(training_pool_cids() + leaderboard_cids() + official_test_cids())
    leaked = official_test_cids()[0]
    table.loc[table["CID"] == leaked, "split"] = TRAIN
    path = tmp_path / "split.csv"
    table.to_csv(path, index=False)

    with pytest.raises(ValueError, match="not the DREAM set"):
        load_split(path)


def test_a_split_file_that_is_missing_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="ds-build"):
        load_split(tmp_path / "split.csv")


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


def test_leaderboard_ratings_match_the_official_leaderboard_targets():
    """LBs1.txt, pivoted and aggregated, must reproduce the organizers' LBs2.txt."""
    targets = aggregate_high_intensity_targets(LEADERBOARD_RATINGS_PATH).set_index("CID")
    official = pd.read_csv(LEADERBOARD_RATINGS_PATH.with_name("LBs2.txt"), sep="	")
    official["descriptor"] = official["descriptor"].map(normalize_target_name)

    assert set(targets.index) == set(leaderboard_cids())
    for column, value in (("mean_", "value"), ("std_", "sigma")):
        expected = official.pivot(index="#oID", columns="descriptor", values=value)
        actual = targets[[f"{column}{name}" for name in expected.columns]]
        actual.columns = expected.columns
        np.testing.assert_allclose(
            actual.loc[expected.index].to_numpy(), expected.to_numpy(), atol=1e-3
        )


def test_target_aggregation_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        aggregate_high_intensity_targets(tmp_path / "does_not_exist.txt")


def test_the_official_leaderboard_targets_are_read_in_the_target_layout():
    official = read_official_leaderboard_targets()

    assert list(official.columns) == ["CID", *TARGET_COLUMNS]
    assert set(official["CID"]) == set(leaderboard_cids())
    assert official[TARGET_COLUMNS].notna().all().all()


def test_the_official_leaderboard_targets_equal_the_aggregated_ratings():
    official = read_official_leaderboard_targets().set_index("CID")
    aggregated = aggregate_high_intensity_targets(LEADERBOARD_RATINGS_PATH).set_index("CID")

    np.testing.assert_allclose(
        aggregated.loc[official.index, TARGET_COLUMNS].to_numpy(),
        official[TARGET_COLUMNS].to_numpy(),
        atol=1e-3,
    )
