"""Tests for the odor decoder: the training texts, the positions, the loss mask and a few training steps."""

from __future__ import annotations

import random

import pytest
import torch

from scripts.core.odor import data, train
from scripts.core.odor import model as model_module
from scripts.core.odor.vocabulary import Vocabulary
from scripts.preprocessing.odor_tables import SPECIAL_TOKENS

SYMBOLS = ["[C]", "[O]", "[=O]"]
WORDS = ["fruity", "green", "sweet"]
VOCABULARY = Vocabulary(
    (*SPECIAL_TOKENS, *SYMBOLS, *WORDS), len(SPECIAL_TOKENS), len(SPECIAL_TOKENS) + len(SYMBOLS)
)
SELFIES = "[C][C][=O]"


def tokens(ids: list[int]) -> list[str]:
    return [VOCABULARY.tokens[i] for i in ids]


def test_vocabulary_groups():
    assert len(VOCABULARY) == 12
    assert VOCABULARY.symbol_ids(SELFIES) == [6, 6, 8]
    assert VOCABULARY.word_ids(["green"]) == [10]


def test_a_mixture_is_encoded_with_its_dot():
    vocabulary = Vocabulary(
        (*SPECIAL_TOKENS, "[C]", "[Cl]", "."), len(SPECIAL_TOKENS), len(SPECIAL_TOKENS) + 3
    )
    assert vocabulary.symbol_ids("[C].[Cl]") == [6, 8, 7]


def test_words_to_molecule_writes_the_request_then_the_molecule():
    ids, start = data.words_to_molecule(VOCABULARY, SELFIES, ["fruity"], random.Random(1))
    assert tokens(ids)[:2] == ["<bos>", "<odor>"]
    assert tokens(ids)[start - 1] == "<sep>"  # the loss starts right after the separator
    assert tokens(ids[start:]) == ["[C]", "[C]", "[=O]", "<eos>"]


def test_words_to_molecule_sometimes_asks_for_nothing():
    rng = random.Random(0)
    counts = [
        len(data.words_to_molecule(VOCABULARY, SELFIES, WORDS, rng)[0]) - 4 - 4 for _ in range(300)
    ]  # the words in the request: the length without <bos>, <odor>, <sep>, <eos> and the 3 symbols
    assert counts.count(0) > 10  # about one in ten
    assert max(counts) <= len(WORDS)


def test_molecule_to_words_names_every_word_of_the_molecule():
    ids, start = data.molecule_to_words(VOCABULARY, SELFIES, WORDS, random.Random(2))
    assert tokens(ids)[:2] == ["<bos>", "<mol>"]
    assert tokens(ids)[start - 1] == "<sep>"
    assert sorted(tokens(ids[start:-1])) == sorted(WORDS)
    assert tokens(ids)[-1] == "<eos>"


def test_stage_1_text_is_the_molecule_alone():
    ids, start = data.molecule_text(VOCABULARY, SELFIES, [], random.Random(0))
    assert tokens(ids) == ["<bos>", "[C]", "[C]", "[=O]", "<eos>"]
    assert start == 1


def test_batches_pad_and_count_the_loss_only_on_the_prediction():
    items = [(SELFIES, WORDS), ("[O]", ["green"])]
    seen = list(
        data.batches(
            VOCABULARY,
            items,
            [data.molecule_to_words],
            random.Random(0),
            shuffle=False,
            batch_size=2,
        )
    )
    ids, mask = seen[0]
    assert ids.shape[0] == 2 and ids.shape == mask.shape
    pad = VOCABULARY.special("<pad>")
    assert not (mask & (ids == pad)).any()  # padding never counts
    for row in range(2):
        sep = (ids[row] == VOCABULARY.special("<sep>")).nonzero().item()
        assert not mask[row, : sep + 1].any()  # the molecule is the instruction
        assert mask[row, sep + 1 :][ids[row, sep + 1 :] != pad].all()


def test_long_texts_get_small_batches():
    short, long = "[C]" * 5, "[C]" * 200
    items = [(short, [])] * 10 + [(long, [])] * 10
    seen = list(
        data.batches(
            VOCABULARY,
            items,
            [data.molecule_text],
            random.Random(0),
            shuffle=False,
            batch_size=64,
            budget=3 * 200 * 200,
        )
    )
    sizes = sorted((ids.shape[0], ids.shape[1]) for ids, _ in seen)
    assert (10, 7) in sizes  # the short texts share one batch
    assert (
        max(rows for rows, length in sizes if length > 100) <= 3
    )  # the long ones are cut to the budget
    assert sum(rows for rows, _ in sizes) == 20  # and every text is in a batch


def test_positions_count_only_the_molecule_and_give_the_words_none():
    ids = torch.tensor(
        [[1, 5, 6, 6, 8, 3, 11, 9, 2]]
    )  # <bos> <mol> C C =O <sep> sweet fruity <eos>
    positions = model_module.positions_of(ids, VOCABULARY)[0].tolist()
    assert positions == [0, 0, 1, 2, 3, 4, 0, 0, 5]


def tiny(monkeypatch):
    for name, value in (("D_MODEL", 16), ("HEADS", 2), ("LAYERS", 2), ("FEEDFORWARD", 32)):
        monkeypatch.setattr(model_module, name, value)
    torch.manual_seed(0)
    return model_module.Decoder(VOCABULARY)


def test_the_order_of_the_words_changes_nothing(monkeypatch):
    model = tiny(monkeypatch).eval()
    one = torch.tensor([[1, 4, 9, 10, 3, 6, 7, 2]])  # <bos> <odor> fruity green <sep> C O <eos>
    other = torch.tensor([[1, 4, 10, 9, 3, 6, 7, 2]])  # the same words, the other way round
    with torch.no_grad():
        first, second = model(one), model(other)
    assert torch.allclose(first[0, 4:], second[0, 4:], atol=1e-5)  # from <sep> on, nothing differs


def test_the_attention_is_causal(monkeypatch):
    model = tiny(monkeypatch).eval()
    ids = torch.tensor([[1, 4, 9, 3, 6, 7, 2]])
    changed = ids.clone()
    changed[0, -1] = 6
    with torch.no_grad():
        assert torch.allclose(model(ids)[0, :-1], model(changed)[0, :-1], atol=1e-5)


def test_training_lowers_the_loss(monkeypatch):
    model = tiny(monkeypatch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    items = [(SELFIES, ["fruity"]), ("[O][C]", ["green"])] * 4
    kinds = [data.words_to_molecule, data.molecule_to_words]
    before = train.run_epoch(model, VOCABULARY, items, kinds, random.Random(0))
    for epoch in range(30):
        train.run_epoch(model, VOCABULARY, items, kinds, random.Random(epoch), optimizer)
    after = train.run_epoch(model, VOCABULARY, items, kinds, random.Random(0))
    assert after < before * 0.7


def test_a_saved_model_with_other_tokens_is_refused(monkeypatch, tmp_path):
    model = tiny(monkeypatch)
    path = tmp_path / "best.pt"
    train.save_atomically({"state": model.state_dict(), "tokens": ("other",)}, path)
    with pytest.raises(ValueError, match="another vocabulary"):
        train.load_model(VOCABULARY, path)


def test_the_output_layer_shares_the_embedding(monkeypatch):
    model = tiny(monkeypatch)
    assert model.head.weight is model.tokens.weight


def test_weight_decay_only_on_matrices_and_a_smaller_rate_for_lower_layers(monkeypatch):
    model = tiny(monkeypatch)
    plain = train.parameter_groups(model, 1e-3)
    assert {group["lr"] for group in plain} == {1e-3}
    decayed = {id(p) for group in plain if group["weight_decay"] > 0 for p in group["params"]}
    assert id(model.tokens.weight) not in decayed  # embeddings are not decayed
    assert id(model.norm.weight) not in decayed  # nor are norms
    assert id(model.body.layers[0].linear1.weight) in decayed  # matrices are
    layered = train.parameter_groups(model, 1e-3, layer_decay=0.5)
    rates = {id(p): group["lr"] for group in layered for p in group["params"]}
    assert rates[id(model.body.layers[1].linear1.weight)] == 1e-3  # the top layer: the full rate
    assert rates[id(model.body.layers[0].linear1.weight)] == 0.5e-3  # one below: half
    assert rates[id(model.tokens.weight)] < rates[id(model.body.layers[0].linear1.weight)]


def test_the_learning_rate_warms_up_and_decays_to_a_tenth(monkeypatch):
    model = tiny(monkeypatch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = train.warmup_cosine(optimizer, total_steps=100)
    rates = []
    for _ in range(100):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[0] < rates[4] <= 1.0  # warming up
    assert max(rates) == pytest.approx(1.0, abs=0.01)
    assert rates[-1] == pytest.approx(train.FINAL_LEARNING_RATE, abs=0.02)  # and down


def test_dropout_can_be_changed_after_the_model_is_built(monkeypatch):
    model = tiny(monkeypatch)
    model_module.set_dropout(model, 0.3)
    probabilities = {m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)}
    assert probabilities == {0.3}
    assert all(
        m.dropout == 0.3 for m in model.modules() if isinstance(m, torch.nn.MultiheadAttention)
    )


def test_the_word_precision_is_a_share_and_rises_with_training(monkeypatch):
    model = tiny(monkeypatch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    items = [(SELFIES, ["fruity", "green"]), ("[O][C]", ["sweet"])] * 4
    before = train.word_precision(model, VOCABULARY, items, top=1)
    for epoch in range(40):
        train.run_epoch(
            model, VOCABULARY, items, [data.molecule_to_words], random.Random(epoch), optimizer
        )
    after = train.word_precision(model, VOCABULARY, items, top=1)
    assert 0 <= before <= 1 and after == 1.0


def test_the_sizes_build_models_that_differ(monkeypatch):
    for name in ("D_MODEL", "HEADS", "LAYERS", "FEEDFORWARD", "SIZE"):
        monkeypatch.setattr(model_module, name, getattr(model_module, name))  # restored afterwards
    model_module.use_size("small")
    small = sum(p.numel() for p in model_module.Decoder(VOCABULARY).parameters())
    model_module.use_size("base")
    base = sum(p.numel() for p in model_module.Decoder(VOCABULARY).parameters())
    assert small < base / 3


def test_a_save_keeps_the_one_before_and_a_broken_file_falls_back(tmp_path):
    path = tmp_path / "state.pt"
    train.save_atomically({"epoch": 1}, path)
    train.save_atomically({"epoch": 2}, path)
    assert train.load_saved(path)["epoch"] == 2
    assert path.with_suffix(".prev").is_file()
    path.write_bytes(b"\x00" * 100)  # cut short by a crash
    assert train.load_saved(path)["epoch"] == 1  # the one before is used
    path.unlink()
    path.with_suffix(".prev").unlink()
    assert train.load_saved(path) is None
