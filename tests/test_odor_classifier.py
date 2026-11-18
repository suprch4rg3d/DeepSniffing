"""Tests for step 3, the classifier: a head on the frozen pretrained decoder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.core import classifier, training
from scripts.core import model as model_module
from scripts.core.vocabulary import Vocabulary
from scripts.preprocessing.odor_tables import SPECIAL_TOKENS

MOLECULE_TOKENS = ["[C]", "[O]", "[=O]", "[N]"]
ODOR_TOKENS = ["fruity", "green", "sweet"]
VOCABULARY = Vocabulary(
    (*SPECIAL_TOKENS, *MOLECULE_TOKENS, *ODOR_TOKENS),
    len(SPECIAL_TOKENS),
    len(SPECIAL_TOKENS) + len(MOLECULE_TOKENS),
)
# two kinds of molecule, told apart by their molecule tokens, each with its own odor token
ITEMS = [("[C][C][C]", ["fruity"]), ("[O][O][N]", ["green"]), ("[C][N]", [])] * 4


def tiny(monkeypatch):
    for name, value in (("D_MODEL", 16), ("HEADS", 2), ("LAYERS", 2), ("FEEDFORWARD", 32)):
        monkeypatch.setattr(model_module, name, value)
    torch.manual_seed(0)
    return model_module.Decoder(VOCABULARY)


def test_the_frozen_decoder_is_read_for_the_molecules_that_have_odor_tokens(monkeypatch):
    model = tiny(monkeypatch)
    features, truth = classifier.read(model, VOCABULARY, ITEMS)
    assert features.shape == (
        8,
        2 * 16,
    )  # the last state and the mean state, of the 8 with odor tokens
    assert truth.shape == (8, len(ODOR_TOKENS))
    assert truth[0].tolist() == [1, 0, 0] and truth[1].tolist() == [0, 1, 0]
    assert not np.allclose(features[0], features[1])  # different molecules, different features


def test_the_classifier_learns_the_odor_tokens_and_leaves_the_decoder_alone(monkeypatch, tmp_path):
    model = tiny(monkeypatch)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    monkeypatch.setattr(training, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(training, "load_model", lambda vocabulary, path: model)
    monkeypatch.setattr(classifier, "load_odor", lambda split: ITEMS)
    (tmp_path / "stage1").mkdir()
    (tmp_path / "stage1" / "best.pt").write_bytes(b"")
    classifier.train(VOCABULARY, max_epochs=40)
    assert all(
        torch.equal(before[k], v.cpu()) for k, v in model.state_dict().items()
    )  # not one weight moved
    rows = pd.read_csv(tmp_path / "stage3" / "history.csv")
    assert {"epoch", "train", "validation", "roc_auc", "precision_at_5"} <= set(rows)
    assert rows["validation"].iloc[-1] < rows["validation"].iloc[0]
    report = classifier.evaluate(VOCABULARY, "validation")
    assert report["roc_auc_macro"] == pytest.approx(1.0)  # the two kinds are told apart
    assert report["molecules"] == 8


def test_the_validation_molecules_can_be_trained_on_for_the_epochs_the_held_out_run_chose(
    monkeypatch, tmp_path
):
    model = tiny(monkeypatch)
    monkeypatch.setattr(training, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(training, "load_model", lambda vocabulary, path: model)
    monkeypatch.setattr(
        classifier, "load_odor", lambda split: ITEMS if split == "train" else ITEMS[:3]
    )
    (tmp_path / "stage1").mkdir()
    (tmp_path / "stage1" / "best.pt").write_bytes(b"")
    classifier.train(VOCABULARY, max_epochs=30)
    chosen = int(pd.read_csv(tmp_path / "stage3" / "history.csv")["validation"].idxmin()) + 1
    classifier.train(VOCABULARY, tag="full", include_validation=True)
    rows = pd.read_csv(tmp_path / "stage3_full" / "history.csv")
    assert len(rows) == chosen  # as many epochs as the held-out run chose, and no early stop
    assert (tmp_path / "stage3_full" / "best.pt").is_file()
    assert classifier.evaluate(VOCABULARY, "validation", "full")["molecules"] == 2
