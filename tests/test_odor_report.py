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
        "generation.png",
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


def test_the_generation_table_shows_how_close_the_nearest_training_molecule_is():
    generation = {
        "fruity": {
            "valid": 1.0,
            "unique": 0.9,
            "novel": 0.7,
            "weight_median": 150.0,
            "weight_over_350": 0.0,
            "similarity": {"median": 0.42, "above_0.7": 0.25, "molecules": 180},
        }
    }
    assert "| fruity | 1.00 | 0.90 | 0.70 | 150 | 0.0% | 0.42 | 25% |" in report.generation_table(
        generation
    )


def test_the_per_token_table_names_the_tokens_it_ranks_worst():
    rows = [{"token": "almond bitter almond", "auc": 0.41, "molecules": 2}]
    assert "| almond bitter almond | 0.410 | 2 |" in report.per_token_table(rows)


def test_the_baseline_table_puts_the_two_models_side_by_side():
    baseline = {"splits": {"test": {"roc_auc_macro": 0.736}}}
    classifier = {"test": {"roc_auc_macro": 0.811}}
    assert "| test | 0.736 | 0.811 | +0.075 |" in report.baseline_table(baseline, classifier)


def test_the_data_table_counts_what_the_run_learned_from():
    eda = {
        "molecules": 7516,
        "splits": {"train": 6344, "validation": 352, "test": 352, "dream": 468},
        "odor_tokens": 890,
        "odor_tokens_on_one_molecule": 211,
        "tokens_per_molecule": {"median": 4.0, "mean": 5.8, "most": 34},
        "sources": {"goodscents": 4247},
        "weight": {"median": 174.2, "share_over_350": 0.015, "heaviest": 2633.0},
    }
    table = report.eda_table(eda)
    assert "| molecules | 7,516: 6,344 train, 352 validation, 352 test, 468 DREAM |" in table
    assert "| odor tokens | 890, 211 on a single molecule |" in table
