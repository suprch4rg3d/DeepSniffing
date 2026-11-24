"""Tests for the measurements: the chemistry of what is written, and the odor token metrics."""

from __future__ import annotations

import random

import numpy as np
import torch

from scripts.core import evaluate, metrics
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


def test_a_written_molecule_becomes_canonical_smiles():
    assert evaluate.molecule_of("[C][C][O]") == "CCO"
    assert evaluate.molecule_of("not selfies at all") is None


def test_the_chemistry_shares_are_what_they_say():
    written = ["CCO", "CCO", "CCN", None, "CCCC"]  # 4 valid, 3 different, 2 of them new
    known = {"CCO"}
    out = evaluate.chemistry(written, known)
    assert out["written"] == 5
    assert out["valid"] == 4 / 5
    assert out["unique"] == 3 / 4  # of the valid ones
    assert out["novel"] == 2 / 3  # of the different ones: CCN and CCCC
    assert out["weight_median"] > 0


def test_nothing_valid_does_not_divide_by_zero():
    out = evaluate.chemistry([None, None], set())
    assert out["valid"] == 0 and out["unique"] == 0 and out["novel"] == 0
    assert out["weight_median"] is None


def test_the_weight_share_notices_heavy_molecules():
    out = evaluate.chemistry(["C" * 40, "CCO"], set())
    assert out["weight_over_350"] == 0.5


def test_roc_auc_is_one_when_the_scores_order_the_odor_tokens_perfectly():
    truth = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    scores = np.array([[9.0, -9.0], [8.0, -8.0], [-9.0, 9.0], [-8.0, 8.0]])
    out = metrics.odor_token_metrics(scores, truth)
    assert out["roc_auc_macro"] == 1.0
    assert out["odor_tokens_scorable"] == 2


def test_roc_auc_is_a_half_when_the_scores_say_nothing():
    rng = np.random.default_rng(0)
    truth = np.zeros((400, 2), dtype=np.int8)
    truth[:200, 0] = 1
    truth[200:, 1] = 1
    out = metrics.odor_token_metrics(rng.normal(size=(400, 2)), truth)
    assert 0.4 < out["roc_auc_macro"] < 0.6  # chance, as it must be


def test_a_odor_token_no_molecule_has_cannot_be_scored_and_is_left_out():
    truth = np.array([[1, 0, 0], [0, 1, 0]])  # the third odor token is on nothing
    scores = np.array([[2.0, 1.0, 5.0], [1.0, 2.0, 5.0]])
    out = metrics.odor_token_metrics(scores, truth)
    assert out["odor_tokens_scorable"] == 2 and out["odor_tokens_in_all"] == 3


def test_no_scorable_odor_token_is_reported_rather_than_crashing():
    truth = np.zeros((3, 2), dtype=np.int8)
    assert metrics.odor_token_metrics(np.zeros((3, 2)), truth) == {"odor_tokens_scorable": 0}


def tiny(monkeypatch):
    for name, value in (("D_MODEL", 16), ("HEADS", 2), ("LAYERS", 2), ("FEEDFORWARD", 32)):
        monkeypatch.setattr(model_module, name, value)
    torch.manual_seed(0)
    return model_module.Decoder(VOCABULARY)


def test_a_large_request_is_written_in_groups_and_all_of_it_comes_back(monkeypatch):
    model = tiny(monkeypatch)
    monkeypatch.setattr(evaluate, "AT_A_TIME", 4)
    sizes = []
    original = evaluate.write_group
    monkeypatch.setattr(
        evaluate,
        "write_group",
        lambda m, v, w, count, *a: (sizes.append(count), original(m, v, w, count, *a))[1],
    )
    written = evaluate.write_molecules(model, VOCABULARY, ("fruity",), count=10)
    assert len(written) == 10 and sizes == [4, 4, 2]  # nothing lost, and no group above the limit


def test_the_model_writes_only_molecule_tokens_never_an_odor_token(monkeypatch):
    """A request asks for odor tokens; what comes back must be a molecule, so no odor token may be written."""
    model = tiny(monkeypatch)
    written = evaluate.write_molecules(model, VOCABULARY, ("fruity",), count=8)
    assert len(written) == 8
    # an untrained model writes nonsense, but nonsense made of molecule tokens: every string that did decode
    # came from the molecule token range, and none of the requests leaked an odor token into the molecule
    assert all(smiles is None or isinstance(smiles, str) for smiles in written)


PAIRS = [("[C][C]", ["fruity"]), ("[O][O]", ["green"]), ("[N][N]", ["sweet"])] * 3


def test_a_model_that_needs_its_request_shows_a_positive_gain(monkeypatch):
    """Each molecule can only be told apart by its odor token, so a trained model has to read the request."""
    from scripts.core import data, training

    model = tiny(monkeypatch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = evaluate.conditioning(model, VOCABULARY, PAIRS)
    for epoch in range(60):
        source = data.batches(
            VOCABULARY, PAIRS, [data.odor_to_molecule], random.Random(epoch), True
        )
        training.run_epoch(model, source, optimizer)
    after = evaluate.conditioning(model, VOCABULARY, PAIRS)

    assert after["gain_over_no_odor_tokens"]["mean"] > 0.3  # the odor tokens are worth a lot
    assert after["gain_over_other_odor_tokens"]["mean"] > 0.3  # and the right ones, not just any
    assert after["gain_over_no_odor_tokens"]["helped"] > 0.9  # for nearly every molecule
    assert (
        abs(before["gain_over_no_odor_tokens"]["mean"]) < 0.3
    )  # an untrained model has no such gain


def test_conditioning_is_the_same_every_time_and_counts_the_molecules_with_odor(monkeypatch):
    model = tiny(monkeypatch)
    items = [
        *PAIRS,
        ("[C][O]", []),
    ]  # one without odor tokens is left out, as there is nothing to ask for
    first = evaluate.conditioning(model, VOCABULARY, items)
    assert first == evaluate.conditioning(model, VOCABULARY, items)
    assert first["molecules"] == len(PAIRS)


def test_the_losses_of_every_molecule_are_kept_and_agree_with_their_averages(monkeypatch):
    model = tiny(monkeypatch)
    result = evaluate.conditioning(model, VOCABULARY, PAIRS)
    each = result["per_molecule"]
    assert len(each["own"]) == len(each["none"]) == len(each["other"]) == len(PAIRS)
    assert np.isclose(np.mean(each["own"]), result["loss_with_own_odor_tokens"])
    assert np.isclose(np.mean(each["none"]), result["loss_with_no_odor_tokens"])
    assert np.isclose(
        result["perplexity_with_own_odor_tokens"], np.exp(result["loss_with_own_odor_tokens"])
    )


def test_the_same_molecule_gets_the_same_request_whatever_the_model(monkeypatch):
    """What lets two runs be compared molecule by molecule: the requests do not depend on the model."""
    first = tiny(monkeypatch)
    torch.manual_seed(5)
    second = model_module.Decoder(VOCABULARY)  # a different model
    a = evaluate.conditioning(first, VOCABULARY, PAIRS)["per_molecule"]
    b = evaluate.conditioning(second, VOCABULARY, PAIRS)["per_molecule"]
    assert [x is None for x in a["other"]] == [
        x is None for x in b["other"]
    ]  # the same molecules have strangers


def test_the_request_text_asks_for_the_odor_tokens_and_scores_only_the_molecule():
    from scripts.core import data

    ids, start = data.request_text(VOCABULARY, "[C][O]", ["fruity", "green"])
    assert [VOCABULARY.tokens[i] for i in ids[:2]] == ["<bos>", "<odor>"]
    assert VOCABULARY.tokens[ids[start - 1]] == "<sep>"
    empty, begin = data.request_text(VOCABULARY, "[C][O]", [])
    assert begin == start - 2  # two odor tokens fewer in front of the molecule


def test_the_two_measured_requests_always_or_never_name_odor_tokens():
    from scripts.core import data

    rng = random.Random(0)
    for _ in range(50):
        asked, start = data.asked_molecule(VOCABULARY, "[C]", ["fruity", "green", "sweet"], rng)
        none, begin = data.unasked_molecule(VOCABULARY, "[C]", ["fruity", "green", "sweet"], rng)
        assert start > begin  # odor tokens are always in the first, never in the second
        assert VOCABULARY.tokens[none[begin - 1]] == "<sep>" and begin == 3


def test_full_rate_embeddings_move_only_the_token_table_up_to_the_full_rate(monkeypatch):
    from scripts.core import training

    model = tiny(monkeypatch)
    slow = {
        id(p): g["lr"] for g in training.parameter_groups(model, 1e-3, 0.5) for p in g["params"]
    }
    fast = {
        id(p): g["lr"]
        for g in training.parameter_groups(model, 1e-3, 0.5, full_rate_embeddings=True)
        for p in g["params"]
    }
    assert slow[id(model.tokens.weight)] < 1e-3  # as before: the table is the slowest
    assert fast[id(model.tokens.weight)] == 1e-3  # now at the full rate
    assert (
        fast[id(model.positions.weight)] == slow[id(model.positions.weight)]
    )  # nothing else moved
    assert (
        fast[id(model.body.layers[0].linear1.weight)]
        == slow[id(model.body.layers[0].linear1.weight)]
    )


def test_the_log_probability_of_a_molecule_agrees_with_the_loss_we_already_trust(monkeypatch):
    """The sum over a molecule's tokens is its mean loss times how many tokens are scored."""
    from scripts.core import data

    model = tiny(monkeypatch)
    molecules = ["[C][C][O]", "[N][N]", "[O][C][C][N]"]
    logp = evaluate.molecule_logprobs(model, VOCABULARY, molecules, ["fruity", "green"])
    for selfies, value in zip(molecules, logp, strict=True):
        ids, start = data.request_text(VOCABULARY, selfies, ["fruity", "green"])
        mean_loss = evaluate.request_loss(model, VOCABULARY, selfies, ["fruity", "green"])
        assert np.isclose(value, -mean_loss * (len(ids) - start), atol=1e-3)
    assert (logp < 0).all()  # a probability is below one


def test_the_log_probabilities_do_not_depend_on_how_the_molecules_are_batched(monkeypatch):
    model = tiny(monkeypatch)
    molecules = ["[C][C][O]", "[N]", "[O][C][C][N][N]", "[C][O]", "[N][N][N]"]
    whole = evaluate.molecule_logprobs(model, VOCABULARY, molecules, ["sweet"], chunk=128)
    piecemeal = evaluate.molecule_logprobs(model, VOCABULARY, molecules, ["sweet"], chunk=2)
    assert np.allclose(whole, piecemeal, atol=1e-4)  # the padding of a batch changes nothing


def test_the_generator_ranks_the_molecules_with_the_odor_token_first_once_it_has_learned_them(
    monkeypatch,
):
    from scripts.core import data, training

    model = tiny(monkeypatch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = evaluate.generator_ranking(model, VOCABULARY, PAIRS)
    for epoch in range(60):
        source = data.batches(
            VOCABULARY, PAIRS, [data.odor_to_molecule], random.Random(epoch), True
        )
        training.run_epoch(model, source, optimizer)
    after = evaluate.generator_ranking(model, VOCABULARY, PAIRS)
    assert after["odor_tokens_scored"] == 3
    assert after["roc_auc_macro"] > 0.95  # it can tell which molecules each odor token belongs to
    assert after["roc_auc_macro"] > before["roc_auc_macro"]


COMBINED = [
    ("[C][C]", ["fruity", "green"]),
    ("[O][O]", ["green", "sweet"]),
    ("[N][N]", ["fruity", "sweet"]),
    ("[C][O]", ["fruity", "green", "sweet"]),
] * 3


def test_a_request_of_several_odor_tokens_finds_the_molecules_that_carry_all_of_them(monkeypatch):
    from scripts.core import data, training

    model = tiny(monkeypatch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for epoch in range(120):
        source = data.batches(
            VOCABULARY, COMBINED, [data.odor_to_molecule], random.Random(epoch), True
        )
        training.run_epoch(model, source, optimizer)
    report = evaluate.combination_ranking(
        model, VOCABULARY, COMBINED, COMBINED, sizes=(1, 2), per_size=20
    )
    assert report["2"]["all"]["requests"] > 0
    assert report["2"]["all"]["roc_auc"] > 0.8  # well above the 0.5 of chance
    assert report["2"]["all"]["at_random"] < 0.7  # and that is against a real base rate


def test_a_combination_no_training_molecule_has_is_reported_as_unseen(monkeypatch):
    model = tiny(monkeypatch)
    held_out = [
        ("[C][C]", ["fruity", "green"]),
        ("[O][O]", ["sweet"]),
        ("[N][N]", ["green", "sweet"]),
    ] * 2
    training = [
        ("[C][C]", ["fruity"]),
        ("[O][O]", ["green"]),
        ("[N][N]", ["sweet"]),
    ]  # no pair at all
    report = evaluate.combination_ranking(
        model, VOCABULARY, held_out, training, sizes=(2,), per_size=10
    )
    assert report["2"]["unseen_combinations"]["requests"] == report["2"]["all"]["requests"] > 0
    together = evaluate.combination_ranking(
        model, VOCABULARY, held_out, held_out, sizes=(2,), per_size=10
    )
    assert (
        together["2"]["unseen_combinations"]["requests"] == 0
    )  # every pair is in the training set


def test_a_odor_token_with_too_few_molecules_is_left_out_of_the_ranking(monkeypatch):
    model = tiny(monkeypatch)
    items = [("[C][C]", ["fruity"]), ("[O][O]", ["green"]), ("[N][N]", ["green"])]
    assert evaluate.generator_ranking(model, VOCABULARY, items, least=3) == {
        "odor_tokens_scored": 0
    }


def test_the_guidance_curve_has_the_plain_model_at_one_and_no_request_at_zero(monkeypatch):
    from scripts.core import data, training

    model = tiny(monkeypatch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for epoch in range(40):
        source = data.batches(
            VOCABULARY, PAIRS, [data.odor_to_molecule], random.Random(epoch), True
        )
        training.run_epoch(model, source, optimizer)
    curve = evaluate.guidance_curve(model, VOCABULARY, PAIRS, strengths=(0.0, 1.0, 2.0))
    assert set(curve) == {"0", "1", "2"}
    plain = evaluate.conditioning(model, VOCABULARY, PAIRS)
    # At 1 it is the model's own probability, taken over the molecule tokens it may write instead of over
    # everything. A trained model puts next to nothing on an odor token or a special token inside a molecule, so
    # this is the loss of the model as it is, and never above it.
    assert curve["1"]["loss"] <= plain["loss_with_own_odor_tokens"] + 1e-4
    assert plain["loss_with_own_odor_tokens"] - curve["1"]["loss"] < 0.1
    assert (
        curve["0"]["loss"] <= plain["loss_with_no_odor_tokens"] + 1e-4
    )  # and 0 is the empty request
    assert np.isclose(curve["1"]["perplexity"], np.exp(curve["1"]["loss"]))


def test_pushing_on_a_request_the_model_follows_lowers_the_loss_of_the_real_molecules(monkeypatch):
    """A model that follows its request but is unsure of it should do better when it is pushed on."""
    from scripts.core import data, training

    model = tiny(monkeypatch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for epoch in range(8):  # a few steps only: it has learned the request, and is not yet sure
        source = data.batches(
            VOCABULARY, PAIRS, [data.odor_to_molecule], random.Random(epoch), True
        )
        training.run_epoch(model, source, optimizer)
    curve = evaluate.guidance_curve(model, VOCABULARY, PAIRS, strengths=(0.0, 1.0, 2.0))
    assert curve["1"]["loss"] < curve["0"]["loss"]  # the request helps
    assert curve["2"]["loss"] < curve["1"]["loss"]  # and more of it helps more, while it is unsure


def test_a_row_that_never_ended_is_not_a_molecule_and_a_finished_one_is():
    eos, pad = VOCABULARY.special("<eos>"), VOCABULARY.special("<pad>")
    c, o = VOCABULARY.index["[C]"], VOCABULARY.index["[O]"]
    rows = [
        [c, c, o, eos, pad, pad],  # finished, with padding after: CCO
        [c, c, o, c, c, c],  # cut off by the limit: never finished
        [eos, pad, pad, pad, pad, pad],  # ended at once: nothing was written
    ]
    assert evaluate.decode(rows, VOCABULARY) == ["CCO", None, None]


def test_the_writing_stops_at_the_limit_and_does_not_run_to_the_models_end(monkeypatch):
    """The limit is what keeps the memory bounded: no row can be longer than it."""
    model = tiny(monkeypatch)
    monkeypatch.setattr(evaluate, "MAX_NEW_MOLECULE_TOKENS", 4)
    seen = []
    original = evaluate.decode
    monkeypatch.setattr(
        evaluate,
        "decode",
        lambda rows, vocabulary: (seen.extend(rows), original(rows, vocabulary))[1],
    )
    evaluate.write_molecules(model, VOCABULARY, ("fruity",), count=6)
    assert seen and max(len(row) for row in seen) <= 4


def test_guidance_one_is_the_model_zero_is_no_request_and_more_goes_further():
    asked = torch.tensor([[2.0, 0.0, -1.0]])
    bare = torch.tensor([[0.0, 0.0, 0.0]])
    at_one = evaluate.guided(asked, bare, 1.0)
    assert torch.allclose(torch.softmax(at_one, -1), torch.softmax(asked, -1), atol=1e-6)
    at_zero = evaluate.guided(asked, bare, 0.0)
    assert torch.allclose(torch.softmax(at_zero, -1), torch.softmax(bare, -1), atol=1e-6)
    more = torch.softmax(evaluate.guided(asked, bare, 3.0), -1)
    assert more[0, 0] > torch.softmax(asked, -1)[0, 0]  # what the request favours is favoured more
    assert more[0, 2] < torch.softmax(asked, -1)[0, 2]  # and what it disfavours, less


def test_guided_writing_runs_and_changes_nothing_when_nothing_is_asked(monkeypatch):
    model = tiny(monkeypatch)
    torch.manual_seed(0)
    plain = evaluate.write_molecules(model, VOCABULARY, (), count=6)
    torch.manual_seed(0)
    same = evaluate.write_molecules(model, VOCABULARY, (), count=6, guidance=4.0)
    assert plain == same  # no request: there is nothing to push away from
    assert len(evaluate.write_molecules(model, VOCABULARY, ("fruity",), count=5, guidance=2.5)) == 5
