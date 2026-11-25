import numpy as np
import torch

from scripts.analysis.request_layers import adapter_norms
from scripts.core.training import OUTPUT_DIR, load_model
from scripts.core.vocabulary import Vocabulary


def test_zeroed_adapters_give_zero_norms():
    vocabulary = Vocabulary.load()
    model = load_model(vocabulary, OUTPUT_DIR / "stage2/best.pt")
    for adapter in model.request:
        torch.nn.init.zeros_(adapter.weight)
    scale, shift = adapter_norms(model, vocabulary)
    assert np.allclose(scale, 0.0)
    assert np.allclose(shift, 0.0)


def test_norms_have_one_column_per_layer():
    vocabulary = Vocabulary.load()
    model = load_model(vocabulary, OUTPUT_DIR / "stage2/best.pt")
    scale, shift = adapter_norms(model, vocabulary)
    assert scale.shape == (len(vocabulary.odor_tokens), len(model.request))
    assert shift.shape == scale.shape
