import numpy as np
import torch

from scripts.analysis.odor_map import correlations, jaccard, spearman


def test_jaccard_of_two_sets():
    assert jaccard({1, 2}, {2, 3}) == 1 / 3
    assert jaccard({1, 2}, {1, 2}) == 1.0
    assert jaccard(set(), set()) == 0.0


def test_spearman_is_one_for_the_same_order_and_minus_one_for_the_reverse():
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])) == 1.0
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([30.0, 20.0, 10.0])) == -1.0


def test_only_tokens_with_enough_molecules_are_compared():
    sets = [{1, 2, 3}, {2, 3, 4}, {1, 4}, {5}]
    before = torch.eye(4)
    found = correlations(before, before, sets, minimum=2)
    assert found["tokens"] == 3
    assert found["pairs"] == 3
