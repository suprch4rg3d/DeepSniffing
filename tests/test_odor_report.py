"""Tests for the accuracy, the history file, the figures and the table of results."""

from __future__ import annotations

import math

import pandas as pd
import torch

from scripts.core import history, plots, report
from scripts.core.metrics import Tally, perplexity


def test_the_accuracy_counts_the_molecule_tokens_the_model_ranked_first():
    tally = Tally()
    assert math.isnan(tally.accuracy)  # nothing counted yet
    tally.add(torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]), torch.tensor([0, 1, 1]))
    assert tally.accuracy == 2 / 3
    tally.add(torch.tensor([[0.0, 1.0]]), torch.tensor([1]))
    assert tally.accuracy == 3 / 4


def test_the_perplexity_is_the_exponential_of_the_loss():
    assert perplexity(0.0) == 1.0
    assert math.isclose(perplexity(math.log(2)), 2.0)


def test_the_history_is_written_and_read_back(tmp_path):
    rows = [{"epoch": 1, "train": 1.0}, {"epoch": 2, "train": 0.5}]
    history.save(tmp_path, rows)
    assert history.load(tmp_path)["train"].tolist() == [1.0, 0.5]


def test_the_curves_are_drawn_for_the_columns_a_history_has(tmp_path):
    rows = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train": [1.0, 0.8, 0.7],
            "validation": [0.9, 0.8, 0.85],
            "validation_accuracy": [0.5, 0.6, 0.6],
        }
    )
    names = {path.name for path in plots.training_curves(rows, tmp_path)}
    assert names == {
        "loss.png",
        "accuracy.png",
    }  # no request gain or naming column, so no such figure


def test_the_report_has_a_row_for_each_evaluated_split_and_its_figures(monkeypatch, tmp_path):
    evaluation = {
        "conditioning": {
            "molecules": 3,
            "loss_with_own_odor_tokens": 0.7,
            "perplexity_with_own_odor_tokens": 2.01,
            "gain_over_no_odor_tokens": {"mean": 0.2, "interval_95": 0.05, "helped": 0.9},
            "per_molecule": {"own": [0.7, 0.6, 0.8], "none": [0.9, 0.8, 0.9]},
        },
        "generation": {
            "fruity": {
                "valid": 1.0,
                "unique": 0.9,
                "novel": 0.8,
                "weight_median": 150.0,
                "weight_over_350": 0.0,
            }
        },
    }
    import json

    (tmp_path / "evaluation_test.json").write_text(json.dumps(evaluation), encoding="utf-8")
    assert list(report.read_evaluations(tmp_path, None)) == ["test"]
    table = report.results_table({"test": evaluation})
    assert "| test | 3 | 0.7000 | 2.01 | +0.200 ± 0.050 | 90% |" in table
    monkeypatch.setattr(report, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["report"])
    assert report.main() == 0
    folder = tmp_path / "report"
    assert {p.name for p in folder.iterdir()} == {
        "results.md",
        "gain_test.png",
        "generation_test.png",
    }


def test_the_classifier_table_reports_the_scores_it_has():
    evaluation = {
        "molecules": 352,
        "roc_auc_macro": 0.812,
        "average_precision_macro": 0.272,
        "precision_at_5": 0.5,
        "precision_at_own_count": 0.62,
    }
    table = report.classifier_table({"test": evaluation})
    assert "| test | 352 | 0.812 | 0.272 | 0.500 | 0.620 |" in table


def test_the_report_includes_the_classifier_when_its_evaluations_exist(monkeypatch, tmp_path):
    import json

    generator = {
        "conditioning": {
            "molecules": 1,
            "loss_with_own_odor_tokens": 0.7,
            "perplexity_with_own_odor_tokens": 2.0,
            "gain_over_no_odor_tokens": {"mean": 0.2, "interval_95": 0.05, "helped": 1.0},
            "per_molecule": {"own": [0.7], "none": [0.9]},
        }
    }
    classifier = {
        "molecules": 352,
        "roc_auc_macro": 0.812,
        "average_precision_macro": 0.272,
        "precision_at_5": 0.5,
        "precision_at_own_count": 0.62,
    }
    (tmp_path / "evaluation_test.json").write_text(json.dumps(generator), encoding="utf-8")
    (tmp_path / "stage3").mkdir()
    (tmp_path / "stage3" / "evaluation_test.json").write_text(
        json.dumps(classifier), encoding="utf-8"
    )
    monkeypatch.setattr(report, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["report"])
    assert report.main() == 0
    text = (tmp_path / "report" / "results.md").read_text(encoding="utf-8")
    assert "## The classifier" in text
    assert "| test | 352 | 0.812 | 0.272 | 0.500 | 0.620 |" in text
