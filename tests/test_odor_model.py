"""Tests for the odor decoder: the training texts, the positions, the loss mask and a few training steps."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.core import data, generator, train, training
from scripts.core import model as model_module
from scripts.core.vocabulary import Vocabulary
from scripts.preprocessing.odor_tables import SPECIAL_TOKENS

MOLECULE_TOKENS = ["[C]", "[O]", "[=O]"]
ODOR_TOKENS = ["fruity", "green", "sweet"]
VOCABULARY = Vocabulary(
    (*SPECIAL_TOKENS, *MOLECULE_TOKENS, *ODOR_TOKENS),
    len(SPECIAL_TOKENS),
    len(SPECIAL_TOKENS) + len(MOLECULE_TOKENS),
)
SELFIES = "[C][C][=O]"


def tokens(ids: list[int]) -> list[str]:
    return [VOCABULARY.tokens[i] for i in ids]


def test_vocabulary_groups():
    assert len(VOCABULARY) == 12
    assert VOCABULARY.molecule_ids(SELFIES) == [6, 6, 8]
    assert VOCABULARY.odor_ids(["green"]) == [10]


def test_a_mixture_is_encoded_with_its_dot():
    vocabulary = Vocabulary(
        (*SPECIAL_TOKENS, "[C]", "[Cl]", "."), len(SPECIAL_TOKENS), len(SPECIAL_TOKENS) + 3
    )
    assert vocabulary.molecule_ids("[C].[Cl]") == [6, 8, 7]


def test_odor_to_molecule_writes_the_request_then_the_molecule():
    ids, start = data.odor_to_molecule(VOCABULARY, SELFIES, ["fruity"], random.Random(1))
    assert tokens(ids)[:2] == ["<bos>", "<odor>"]
    assert tokens(ids)[start - 1] == "<sep>"  # the loss starts right after the separator
    assert tokens(ids[start:]) == ["[C]", "[C]", "[=O]", "<eos>"]


def test_odor_to_molecule_sometimes_asks_for_nothing():
    rng = random.Random(0)
    counts = [
        len(data.odor_to_molecule(VOCABULARY, SELFIES, ODOR_TOKENS, rng)[0]) - 4 - 4
        for _ in range(300)
    ]  # the odor tokens in the request: the length without <bos>, <odor>, <sep>, <eos> and the 3 molecule tokens
    assert counts.count(0) > 10  # about one in ten
    assert max(counts) <= len(ODOR_TOKENS)


def test_stage_1_text_is_the_molecule_alone():
    ids, start = data.molecule_text(VOCABULARY, SELFIES, [], random.Random(0))
    assert tokens(ids) == ["<bos>", "[C]", "[C]", "[=O]", "<eos>"]
    assert start == 1


def test_batches_pad_and_count_the_loss_only_on_the_prediction():
    items = [(SELFIES, ODOR_TOKENS), ("[O]", ["green"])]
    seen = list(
        data.batches(
            VOCABULARY,
            items,
            [data.odor_to_molecule],
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
        assert not mask[row, : sep + 1].any()  # the request is the instruction
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


def test_positions_count_only_the_molecule_and_give_the_odor_tokens_none():
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


def test_the_order_of_the_odor_tokens_changes_nothing(monkeypatch):
    model = tiny(monkeypatch).eval()
    one = torch.tensor([[1, 4, 9, 10, 3, 6, 7, 2]])  # <bos> <odor> fruity green <sep> C O <eos>
    other = torch.tensor([[1, 4, 10, 9, 3, 6, 7, 2]])  # the same odor tokens, the other way round
    with torch.no_grad():
        first, second = model(one), model(other)
    assert torch.allclose(first[0, 4:], second[0, 4:], atol=1e-5)  # from <sep> on, nothing differs


def with_request_weights(model):
    """Give the request adapters weights, as they are after training, so the tests exercise them."""
    torch.manual_seed(1)
    for adapter in model.request:
        torch.nn.init.normal_(adapter.weight, std=0.3)
    return model.eval()


def test_the_adapters_start_as_nothing_so_the_model_is_what_it_was(monkeypatch):
    """The layer loop is the encoder's own loop, and the zero adapters add nothing to it."""
    model = tiny(monkeypatch).eval()
    ids = torch.tensor([[1, 4, 9, 10, 3, 6, 7, 2]])  # <bos> <odor> fruity green <sep> C O <eos>
    with torch.no_grad():
        positions = model_module.positions_of(ids, VOCABULARY).clamp(max=model_module.MAX_POSITIONS)
        state = model.tokens(ids) + model.positions(positions)
        reference = model.norm(model.body(state, mask=model.blocked(ids)))
        assert torch.allclose(model.hidden(ids), reference, atol=1e-6)


def test_the_order_of_the_odor_tokens_still_changes_nothing_once_the_adapters_are_trained(
    monkeypatch,
):
    model = with_request_weights(tiny(monkeypatch))
    one = torch.tensor([[1, 4, 9, 10, 3, 6, 7, 2]])
    other = torch.tensor([[1, 4, 10, 9, 3, 6, 7, 2]])
    with torch.no_grad():
        assert torch.allclose(model(one)[0, 4:], model(other)[0, 4:], atol=1e-5)


def test_the_adapters_change_the_molecule_and_never_what_comes_before_it(monkeypatch):
    plain = tiny(monkeypatch).eval()
    trained = with_request_weights(tiny(monkeypatch))
    trained.load_state_dict(
        {
            **plain.state_dict(),
            **{k: v for k, v in trained.state_dict().items() if k.startswith("request.")},
        }
    )
    ids = torch.tensor([[1, 4, 9, 10, 3, 6, 7, 2]])  # the separator is at index 4
    with torch.no_grad():
        a, b = plain(ids), trained(ids)
    assert torch.allclose(a[0, :4], b[0, :4], atol=1e-5)  # the start and the odor tokens: untouched
    assert not torch.allclose(a[0, 4:], b[0, 4:], atol=1e-3)  # the molecule: changed by the request


def test_without_a_request_the_adapters_add_nothing(monkeypatch):
    """The pretraining and the naming have no odor tokens, so the adapters must not touch them."""
    plain = tiny(monkeypatch).eval()
    trained = with_request_weights(tiny(monkeypatch))
    trained.load_state_dict(
        {
            **plain.state_dict(),
            **{k: v for k, v in trained.state_dict().items() if k.startswith("request.")},
        }
    )
    ids = torch.tensor(
        [[1, 6, 7, 6, 2], [1, 5, 6, 7, 3]]
    )  # a molecule alone, and a molecule to name
    with torch.no_grad():
        assert torch.allclose(plain(ids), trained(ids), atol=1e-6)


def test_the_attention_is_still_causal_with_the_adapters_trained(monkeypatch):
    model = with_request_weights(tiny(monkeypatch))
    ids = torch.tensor([[1, 4, 9, 3, 6, 7, 2]])
    changed = ids.clone()
    changed[0, -1] = 6
    with torch.no_grad():
        assert torch.allclose(model(ids)[0, :-1], model(changed)[0, :-1], atol=1e-5)


def test_a_model_saved_with_the_old_naming_head_still_loads(monkeypatch, tmp_path):
    """The pretrained decoder was saved when the decoder still had a naming head, under the key ``words``."""
    model = tiny(monkeypatch).eval()
    old = {**model.state_dict(), "words.weight": torch.zeros(3, 16)}
    path = tmp_path / "old.pt"
    training.save_atomically(
        {"state": old, "tokens": VOCABULARY.tokens, "size": model_module.SIZE}, path
    )
    loaded = training.load_model(VOCABULARY, path)
    ids = torch.tensor([[1, 4, 9, 3, 6, 7, 2]])
    with torch.no_grad():
        assert torch.allclose(model(ids), loaded(ids), atol=1e-6)


def test_a_model_saved_before_the_adapters_existed_still_loads_and_is_unchanged(
    monkeypatch, tmp_path
):
    model = tiny(monkeypatch).eval()
    old = {k: v for k, v in model.state_dict().items() if not k.startswith("request.")}
    path = tmp_path / "old.pt"
    training.save_atomically(
        {"state": old, "tokens": VOCABULARY.tokens, "size": model_module.SIZE}, path
    )
    loaded = training.load_model(VOCABULARY, path)
    ids = torch.tensor([[1, 4, 9, 3, 6, 7, 2]])
    with torch.no_grad():
        assert torch.allclose(model(ids), loaded(ids), atol=1e-6)


def test_a_saved_model_with_something_else_wrong_is_still_refused(monkeypatch, tmp_path):
    model = tiny(monkeypatch)
    broken = {k: v for k, v in model.state_dict().items() if not k.startswith("body.layers.0.")}
    path = tmp_path / "broken.pt"
    training.save_atomically(
        {"state": broken, "tokens": VOCABULARY.tokens, "size": model_module.SIZE}, path
    )
    with pytest.raises(ValueError, match="does not fit"):
        training.load_model(VOCABULARY, path)


def packed_molecules():
    """Three molecules as the packed tokens hold them: [C][C], [O], [N][C][O]."""
    c, o, n = (VOCABULARY.index[s] for s in ("[C]", "[O]", "[=O]"))
    tokens = np.array([c, c, o, n, c, o], dtype=np.int16)
    offsets = np.array([0, 2, 3, 6])
    return tokens, offsets, np.array([0, 1, 2])


def test_a_molecule_can_be_written_after_an_empty_request_from_the_packed_tokens():
    batch = next(
        data.packed_batches(VOCABULARY, packed_molecules(), random.Random(0), False, request=True)
    )
    ids, mask = batch
    assert (ids[:, 0] == VOCABULARY.special("<bos>")).all()
    assert (ids[:, 1] == VOCABULARY.special("<odor>")).all()
    assert (ids[:, 2] == VOCABULARY.special("<sep>")).all()
    assert not mask[:, :3].any()  # the request is not predicted
    pad = VOCABULARY.special("<pad>")
    for row in range(3):
        assert mask[row][ids[row] != pad][3:].all()  # every molecule token and the end is
        assert VOCABULARY.tokens[int(ids[row][ids[row] != pad][-1])] == "<eos>"


def test_the_plain_packed_batches_are_what_they_were(monkeypatch):
    ids, mask = next(data.packed_batches(VOCABULARY, packed_molecules(), random.Random(0), False))
    assert (ids[:, 0] == VOCABULARY.special("<bos>")).all()
    assert not mask[:, 0].any() and mask[:, 1].all()  # the loss starts at the first molecule token


def test_replay_mixes_pretraining_molecules_into_the_batches_of_the_fine_tuning():
    texts = list(
        data.batches(
            VOCABULARY,
            [("[C][C]", ["fruity"])] * 4,
            [data.odor_to_molecule],
            random.Random(0),
            False,
        )
    )
    mixed = list(data.with_replay(iter(texts), VOCABULARY, packed_molecules(), 2, random.Random(0)))
    assert len(mixed) == len(texts) + 1  # the texts, and a batch of the two molecules replayed
    replayed = [
        b for b in mixed if (b[0][:, 2] == VOCABULARY.special("<sep>")).all() and b[0].shape[1] <= 8
    ]
    assert replayed, "the replayed molecules are in there"
    again = list(data.with_replay(iter(texts), VOCABULARY, packed_molecules(), 2, random.Random(0)))
    assert all(
        torch.equal(a[0], b[0]) for a, b in zip(mixed, again, strict=True)
    )  # the same every time


def test_replay_cannot_ask_for_more_molecules_than_there_are():
    mixed = list(data.with_replay(iter([]), VOCABULARY, packed_molecules(), 1000, random.Random(0)))
    assert sum(len(b[0]) for b in mixed) == 3


def test_the_average_of_the_weights_lags_behind_the_weights_and_starts_equal(monkeypatch):
    model = tiny(monkeypatch)
    average = training.WeightAverage(model, decay=0.9)
    assert all(torch.equal(average.shadow[k], v) for k, v in model.state_dict().items())
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.tokens.weight.add_(1.0)  # the weights move by one
    average.update(model)
    moved = (average.shadow["tokens.weight"] - before["tokens.weight"]).mean().item()
    assert 0 < moved < 1.0  # the average has moved, but not all the way
    for _ in range(300):
        average.update(model)
    assert torch.allclose(
        average.shadow["tokens.weight"], model.tokens.weight, atol=1e-3
    )  # it catches up


def test_the_average_is_swapped_in_and_the_weights_come_back_even_after_an_error(monkeypatch):
    model = tiny(monkeypatch)
    average = training.WeightAverage(model, decay=0.9)
    with torch.no_grad():
        model.tokens.weight.add_(1.0)
    own = model.tokens.weight.detach().clone()
    with average.applied(model):
        assert not torch.equal(
            model.tokens.weight, own
        )  # inside it, the average is what the model has
    assert torch.equal(model.tokens.weight, own)
    with pytest.raises(RuntimeError), average.applied(model):
        raise RuntimeError("something went wrong while scoring")
    assert torch.equal(model.tokens.weight, own)  # and the model is not left holding the average


def test_a_run_with_an_average_saves_the_average_but_goes_on_training_its_own_weights(
    monkeypatch, tmp_path
):
    items = [(SELFIES, ["fruity"]), ("[O][C]", ["green"])] * 4

    def run(folder, ema):
        folder.mkdir()
        training.train_stage(
            folder,
            tiny(monkeypatch),
            VOCABULARY,
            lambda rng: data.batches(VOCABULARY, items, [data.molecule_text], rng, True),
            {
                "molecule": lambda: data.batches(
                    VOCABULARY, items, [data.molecule_text], random.Random(0), False
                )
            },
            score="molecule",
            learning_rate=1e-3,
            max_epochs=3,
            patience=3,
            ema=ema,
        )
        return (
            torch.load(folder / "best.pt", weights_only=False, map_location="cpu"),
            torch.load(folder / "state.pt", weights_only=False, map_location="cpu"),
        )

    best, state = run(tmp_path / "averaged", 0.9)
    assert state["ema"] is not None and state["ema_steps"] > 0
    assert not all(torch.equal(best["state"][k], state["model"][k]) for k in best["state"])
    plain_best, plain_state = run(tmp_path / "plain", 0.0)
    assert plain_state["ema"] is None  # off: nothing is kept, and nothing changes


def test_a_plain_stage_2_command_has_the_best_settings_and_each_option_changes_one_thing():
    """The defaults are what the results were made with, so a command with no options reproduces them."""
    plain = train.build_parser().parse_args(["--stage", "2"])
    assert plain.full_rate_embeddings is True
    assert plain.replay is None  # as many as there are molecules with odor tokens
    assert (plain.dropout, plain.layer_decay) == (generator.DROPOUT, generator.LAYER_DECAY)
    assert (plain.lr, plain.ema, plain.tag) == (None, 0.0, None)
    assert not hasattr(plain, "odor_token_weight")  # the generator trains on no other loss
    changed = train.build_parser().parse_args(
        ["--stage", "2", "--no-full-rate-embeddings", "--replay", "0", "--ema", "0.99"]
    )
    assert changed.full_rate_embeddings is False and changed.replay == 0
    assert changed.ema == 0.99
    assert changed.dropout == plain.dropout  # and nothing else moved


def test_the_three_steps_can_each_be_asked_for():
    parser = train.build_parser()
    assert parser.parse_args([]).stage == "all"
    assert [parser.parse_args(["--stage", s]).stage for s in "123"] == ["1", "2", "3"]


def test_replay_defaults_to_as_many_molecules_as_there_are_with_odor(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(generator, "load_odor", lambda split: [("[C][C]", ["fruity"])] * 5)
    monkeypatch.setattr(
        generator, "load_packed", lambda split: seen.setdefault("loaded", packed_molecules())
    )
    monkeypatch.setattr(training, "load_model", lambda vocabulary, path: tiny(monkeypatch))
    monkeypatch.setattr(training, "OUTPUT_DIR", tmp_path)
    (tmp_path / "stage1").mkdir()
    (tmp_path / "stage1" / "best.pt").write_bytes(b"")
    replayed = []
    monkeypatch.setattr(
        generator,
        "with_replay",
        lambda texts, vocabulary, packed, count, rng: (replayed.append(count), iter(list(texts)))[
            1
        ],
    )
    generator.train(VOCABULARY, max_epochs=1)
    assert replayed and set(replayed) == {5}  # five molecules with odor tokens, so five replayed
    replayed.clear()
    generator.train(VOCABULARY, max_epochs=1, tag="none", replay=0)
    assert replayed == []  # and 0 turns it off


def test_the_validation_molecules_are_added_to_the_training_when_asked(monkeypatch, tmp_path):
    monkeypatch.setattr(
        generator,
        "load_odor",
        lambda split: [("[C][C]", ["fruity"])] * (5 if split == "train" else 2),
    )
    monkeypatch.setattr(generator, "load_packed", lambda split: packed_molecules())
    monkeypatch.setattr(training, "load_model", lambda vocabulary, path: tiny(monkeypatch))
    monkeypatch.setattr(training, "OUTPUT_DIR", tmp_path)
    (tmp_path / "stage1").mkdir()
    (tmp_path / "stage1" / "best.pt").write_bytes(b"")
    replayed = []
    monkeypatch.setattr(
        generator,
        "with_replay",
        lambda texts, vocabulary, packed, count, rng: (replayed.append(count), iter(list(texts)))[
            1
        ],
    )
    generator.train(VOCABULARY, max_epochs=1, tag="held")
    generator.train(VOCABULARY, max_epochs=1, tag="all", include_validation=True)
    assert set(replayed) == {5, 7}  # as many replayed as there are to learn from: 5, then 5 and 2


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
    kinds = [data.odor_to_molecule]

    def source(seed, shuffle=False):
        return data.batches(VOCABULARY, items, kinds, random.Random(seed), shuffle)

    before = training.run_epoch(model, source(0))
    for epoch in range(30):
        training.run_epoch(model, source(epoch, True), optimizer)
    after = training.run_epoch(model, source(0))
    assert after < before * 0.7


def test_a_saved_model_with_other_tokens_is_refused(monkeypatch, tmp_path):
    model = tiny(monkeypatch)
    path = tmp_path / "best.pt"
    training.save_atomically({"state": model.state_dict(), "tokens": ("other",)}, path)
    with pytest.raises(ValueError, match="another vocabulary"):
        training.load_model(VOCABULARY, path)


def test_the_output_layer_shares_the_embedding(monkeypatch):
    model = tiny(monkeypatch)
    assert model.head.weight is model.tokens.weight


def test_weight_decay_only_on_matrices_and_a_smaller_rate_for_lower_layers(monkeypatch):
    model = tiny(monkeypatch)
    plain = training.parameter_groups(model, 1e-3)
    assert {group["lr"] for group in plain} == {1e-3}
    decayed = {id(p) for group in plain if group["weight_decay"] > 0 for p in group["params"]}
    assert id(model.tokens.weight) not in decayed  # embeddings are not decayed
    assert id(model.norm.weight) not in decayed  # nor are norms
    assert id(model.body.layers[0].linear1.weight) in decayed  # matrices are
    layered = training.parameter_groups(model, 1e-3, layer_decay=0.5)
    rates = {id(p): group["lr"] for group in layered for p in group["params"]}
    assert rates[id(model.body.layers[1].linear1.weight)] == 1e-3  # the top layer: the full rate
    assert rates[id(model.body.layers[0].linear1.weight)] == 0.5e-3  # one below: half
    assert rates[id(model.tokens.weight)] < rates[id(model.body.layers[0].linear1.weight)]


def test_the_learning_rate_warms_up_and_decays_to_a_tenth(monkeypatch):
    model = tiny(monkeypatch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = training.warmup_cosine(optimizer, total_steps=100)
    rates = []
    for _ in range(100):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[0] < rates[4] <= 1.0  # warming up
    assert max(rates) == pytest.approx(1.0, abs=0.01)
    assert rates[-1] == pytest.approx(training.FINAL_LEARNING_RATE, abs=0.02)  # and down


def test_dropout_can_be_changed_after_the_model_is_built(monkeypatch):
    model = tiny(monkeypatch)
    model_module.set_dropout(model, 0.3)
    probabilities = {m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)}
    assert probabilities == {0.3}
    assert all(
        m.dropout == 0.3 for m in model.modules() if isinstance(m, torch.nn.MultiheadAttention)
    )


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
    training.save_atomically({"epoch": 1}, path)
    training.save_atomically({"epoch": 2}, path)
    assert training.load_saved(path)["epoch"] == 2
    assert path.with_suffix(".prev").is_file()
    path.write_bytes(b"\x00" * 100)  # cut short by a crash
    assert training.load_saved(path)["epoch"] == 1  # the one before is used
    path.unlink()
    path.with_suffix(".prev").unlink()
    assert training.load_saved(path) is None


def test_a_finished_run_keeps_its_accuracy_and_leaves_no_copies_of_the_one_before(
    monkeypatch, tmp_path
):
    items = [(SELFIES, ["fruity"]), ("[O][C]", ["green"])] * 4
    training.train_stage(
        tmp_path,
        tiny(monkeypatch),
        VOCABULARY,
        lambda rng: data.batches(VOCABULARY, items, [data.molecule_text], rng, True),
        {
            "molecule": lambda: data.batches(
                VOCABULARY, items, [data.molecule_text], random.Random(0), False
            )
        },
        score="molecule",
        learning_rate=1e-3,
        max_epochs=3,
        patience=3,
    )
    history = pd.read_csv(tmp_path / "history.csv")
    for column in ("train_accuracy", "molecule_accuracy", "validation_accuracy"):
        assert history[column].between(0, 1).all()
    assert not list(
        tmp_path.glob("*.prev")
    )  # three epochs saved over each other, and none is left over
    assert (tmp_path / "best.pt").is_file() and (tmp_path / "state.pt").is_file()


def test_the_batch_limits_can_be_changed_without_editing_the_code():
    from scripts.core import data

    budget, size = data.ATTENTION_BUDGET, data.BATCH_SIZE
    try:
        data.use_budget(1000)
        data.use_batch_size(7)
        assert data.ATTENTION_BUDGET == 1000
        assert data.BATCH_SIZE == 7
    finally:
        data.use_budget(budget)
        data.use_batch_size(size)
