"""Tests for comparing stage 2 runs molecule by molecule."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.core import compare


def run(own, none):
    return {"per_molecule": {"own": list(own), "none": list(none), "other": [None] * len(own)}}


def test_a_paired_difference_is_the_mean_of_the_differences_and_its_interval():
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    result = compare.paired(reference, reference - 0.5)
    assert result["mean"] == pytest.approx(-0.5)
    assert result["interval_95"] == pytest.approx(
        0.0
    )  # the same shift everywhere: nothing uncertain
    assert result["lower_for"] == 1.0


def test_pairing_sees_what_two_averages_cannot():
    """The molecules differ a great deal from each other, and the second run is better on every one."""
    rng = np.random.default_rng(0)
    reference = rng.normal(1.0, 1.0, 300)  # a wide spread between molecules
    better = reference - 0.02  # a small improvement, the same everywhere
    # side by side, the improvement is lost in a spread of 1.0 ...
    assert abs(reference.mean() - better.mean()) < reference.std() / np.sqrt(300) * 1.96
    # ... but taken molecule by molecule it is exact
    result = compare.paired(reference, better)
    assert abs(result["mean"]) > result["interval_95"]


def test_a_difference_inside_the_noise_is_not_called_real():
    rng = np.random.default_rng(1)
    reference = rng.normal(1.0, 0.3, 200)
    other = reference + rng.normal(0.0, 0.3, 200)  # no real change, only scatter
    result = compare.paired(reference, other)
    assert abs(result["mean"]) < 2 * result["interval_95"]


def test_the_rows_hold_the_reference_first_and_each_run_against_it():
    reference = run([1.0, 1.2, 0.8], [1.1, 1.3, 0.9])
    better = run([0.9, 1.1, 0.7], [1.1, 1.3, 0.9])
    rows = compare.compare(reference, {"better": better})
    assert [r["run"] for r in rows] == ["reference", "better"]
    assert "against_reference" not in rows[0]
    assert rows[1]["against_reference"]["mean"] == pytest.approx(-0.1)
    assert rows[1]["gain"] == pytest.approx(0.2)  # none minus own, on average


def test_runs_scored_on_different_molecules_are_refused():
    with pytest.raises(ValueError, match="molecules"):
        compare.compare(run([1.0, 1.0], [1.0, 1.0]), {"x": run([1.0], [1.0])})


def test_a_run_not_evaluated_yet_or_evaluated_before_the_losses_were_kept_is_said_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run evaluate"):
        compare.read_run("nothing", folder=tmp_path)
    (tmp_path / "evaluation_validation_old.json").write_text(
        json.dumps({"conditioning": {"loss_with_own_odor_tokens": 1.0}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Run evaluate again"):
        compare.read_run("old", folder=tmp_path)


def test_a_run_is_read_by_its_tag_and_the_untagged_one_is_stage2(tmp_path):
    for suffix, value in (("", 1.0), ("_film", 2.0)):
        report = {"conditioning": run([value], [value + 1])}
        (tmp_path / f"evaluation_validation{suffix}.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
    assert compare.read_run(None, folder=tmp_path)["per_molecule"]["own"] == [1.0]
    assert compare.read_run("stage2", folder=tmp_path)["per_molecule"]["own"] == [
        1.0
    ]  # by its folder's name
    assert compare.read_run("film", folder=tmp_path)["per_molecule"]["own"] == [2.0]
