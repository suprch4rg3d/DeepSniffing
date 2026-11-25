import numpy as np

from scripts.analysis.hidden_states import separation


def test_separation_is_high_when_the_clusters_are_obvious():
    rng = np.random.default_rng(0)
    states = np.vstack([rng.normal(location, 0.1, (20, 4)) for location in (0.0, 8.0, 16.0)])
    labels = np.repeat([0, 1, 2], 20)
    assert separation(states, labels) > 0.9


def test_separation_is_low_when_the_labels_are_mixed():
    rng = np.random.default_rng(0)
    states = rng.normal(0.0, 1.0, (60, 4))
    labels = rng.integers(0, 3, 60)
    assert separation(states, labels) < 0.3
