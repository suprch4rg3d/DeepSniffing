"""The Subchallenge 1 model is fixed: its best result is documented in ``Q1_SETUP.md``.

Work on Subchallenge 2 must not change it. These tests build the Subchallenge 1 model the way
the training does and compare it with the saved Subchallenge 1 checkpoint, layer by layer.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.core.data import dragon_group_indices
from scripts.training import custom_model
from scripts.training.custom_model import (
    GRAPH_DROPOUT,
    GRAPH_HIDDEN_SIZE,
    MESSAGE_PASSING_LAYERS,
    TOKEN_DROPOUT,
    build_graph_model,
)

CHECKPOINT = Path(__file__).resolve().parents[1] / "output/q1/checkpoints/custom_model.pt"


def test_the_numbers_that_define_the_subchallenge_1_architecture():
    assert GRAPH_HIDDEN_SIZE == 64
    assert MESSAGE_PASSING_LAYERS == 3
    assert GRAPH_DROPOUT == 0.25
    assert TOKEN_DROPOUT == 0.15


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="the saved Subchallenge 1 model is missing")
def test_the_model_built_for_subchallenge_1_has_the_layers_of_the_saved_model():
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    data = SimpleNamespace(
        dragon_columns=saved["dragon_columns"],
        dragon_groups=saved["dragon_feature_groups"],
        numeric_count=saved["numeric_count"],
        name_count=saved["name_count"],
        target_count=len(saved["target_columns"]),
        structure_count=saved["structure_count"],
        challenge="q1",
        neighbours=None,
    )

    model = build_graph_model(data)

    built = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    stored = {name: tuple(value.shape) for name, value in saved["state_dict"].items()}
    assert built == stored  # the same layers, with the same shapes, and nothing added
    model.load_state_dict(saved["state_dict"])  # and the saved weights fit
    assert model.token_dropout == TOKEN_DROPOUT
    assert dragon_group_indices(saved["dragon_columns"], saved["dragon_feature_groups"])


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="the saved Subchallenge 1 model is missing")
def test_the_saved_subchallenge_1_model_still_gives_the_same_answers():
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    data = SimpleNamespace(
        dragon_columns=saved["dragon_columns"],
        dragon_groups=saved["dragon_feature_groups"],
        numeric_count=saved["numeric_count"],
        name_count=saved["name_count"],
        target_count=len(saved["target_columns"]),
        structure_count=saved["structure_count"],
        challenge="q1",
        neighbours=None,
    )
    model = build_graph_model(data)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    generator = torch.Generator().manual_seed(0)
    atoms = torch.randn(7, custom_model.ATOM_FEATURE_COUNT, generator=generator)
    edges = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])
    edge_features = torch.randn(6, custom_model.EDGE_FEATURE_COUNT, generator=generator)
    graph_ids = torch.zeros(7, dtype=torch.long)
    arguments = (
        atoms,
        edges,
        edge_features,
        graph_ids,
        torch.randn(1, len(saved["dragon_columns"]), generator=generator),
        torch.randn(1, saved["numeric_count"], generator=generator),
        torch.randn(1, saved["name_count"], generator=generator),
        torch.randn(1, saved["structure_count"], generator=generator),
    )

    with torch.no_grad():
        first = model(*arguments)
        second = model(*arguments)

    assert first.shape == (1, len(saved["target_columns"]))
    assert torch.equal(first, second)  # deterministic in evaluation
    # Numbers taken from the saved model when this test was written: the first five of the 1,029
    # outputs for this input, and their sum. They change if any layer or its use changes.
    assert [round(float(value), 4) for value in first[0, :5]] == pytest.approx(
        [-0.6028, -0.4698, -0.7158, -0.4340, -0.5118], abs=2e-4
    )
    assert float(first.sum()) == pytest.approx(-284.525, abs=0.01)


def test_the_training_settings_of_subchallenge_1_are_the_documented_ones():
    from scripts.core.train import Q1_BLEND_ALPHA, Q1_TRAIN_CONFIG, Q2_TRAIN_CONFIG

    assert Q1_TRAIN_CONFIG.learning_rate == 0.0005
    assert Q1_TRAIN_CONFIG.weight_decay == 0.001
    assert Q1_TRAIN_CONFIG.batch_size == 64
    assert Q1_TRAIN_CONFIG.gradient_clip == 5.0
    assert Q1_BLEND_ALPHA == 0.8
    # Subchallenge 2 has its own learning rate, and it leaves the rest as it was.
    assert Q2_TRAIN_CONFIG.learning_rate == 0.001
    assert Q2_TRAIN_CONFIG.batch_size == 32
    assert Q2_TRAIN_CONFIG.weight_decay == Q1_TRAIN_CONFIG.weight_decay
