"""Tests for the SMILES tokenizer."""

from __future__ import annotations

import pytest

from scripts.preprocessing.smiles_tokenizer import SmilesTokenizer


@pytest.fixture
def tokenizer() -> SmilesTokenizer:
    fitted = SmilesTokenizer()
    fitted.fit(["CCO", "CCN", "c1ccccc1"])
    return fitted


def test_special_tokens_keep_their_reserved_ids(tokenizer):
    assert tokenizer.token_to_id[SmilesTokenizer.PAD] == 0
    assert tokenizer.token_to_id[SmilesTokenizer.UNK] == 1


def test_fit_is_order_independent():
    first = SmilesTokenizer()
    first.fit(["CCO", "CCN"])
    second = SmilesTokenizer()
    second.fit(["CCN", "CCO"])

    assert first.token_to_id == second.token_to_id


def test_tokenize_splits_two_letter_elements():
    assert SmilesTokenizer().tokenize("CCBr") == ["C", "C", "Br"]


def test_tokenize_splits_bracket_atoms():
    assert SmilesTokenizer().tokenize("[NH4+]") == ["[NH4+]"]


def test_tokenize_rejects_unparseable_text():
    with pytest.raises(ValueError):
        SmilesTokenizer().tokenize("C C")


def test_encode_pads_to_the_requested_length(tokenizer):
    encoded = tokenizer.encode("CCO", max_length=8)

    assert len(encoded) == 8
    assert encoded[3:] == [tokenizer.token_to_id[SmilesTokenizer.PAD]] * 5


def test_encode_truncates_to_the_requested_length(tokenizer):
    encoded = tokenizer.encode("c1ccccc1", max_length=3)

    assert len(encoded) == 3


def test_encode_maps_unknown_tokens_to_the_unknown_id(tokenizer):
    encoded = tokenizer.encode("CCX", max_length=3)

    assert encoded[2] == tokenizer.token_to_id[SmilesTokenizer.UNK]


def test_encode_rejects_non_positive_length(tokenizer):
    with pytest.raises(ValueError):
        tokenizer.encode("CCO", max_length=0)


def test_save_and_load_round_trip(tokenizer, tmp_path):
    path = tmp_path / "vocabulary.json"
    tokenizer.save(path)

    restored = SmilesTokenizer()
    restored.load(path)

    assert restored.token_to_id == tokenizer.token_to_id
    assert restored.encode("CCO", 6) == tokenizer.encode("CCO", 6)


def test_load_rejects_a_vocabulary_with_broken_special_tokens(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"<PAD>": 5, "<UNK>": 1}', encoding="utf-8")

    with pytest.raises(ValueError):
        SmilesTokenizer().load(path)
