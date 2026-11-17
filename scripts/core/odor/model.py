"""The decoder: a small GPT over the tokens of ``vocabulary.json``.

The attention is causal. A token has a position only when it is part of the molecule. The odor words are a
set, not a sequence, so they have no position (position 0, which also stands for the start tokens). That alone
would not make them a set, because a word that looks at the words before it carries their order into the
layers above. So the words of a request (the ones before the separator) look only at the start tokens and at
themselves, and every later token sees all of them. Nothing a request leads to depends on the order in which
its words were listed. The words that the model writes (after the separator) are an ordinary causal text, so
that it does not name a word twice. A molecule symbol, the separator and the end token count their place
in the molecule from 1.
"""

from __future__ import annotations

import torch
from torch import nn

from scripts.core.odor.vocabulary import Vocabulary

D_MODEL, HEADS, LAYERS, FEEDFORWARD, DROPOUT = 256, 8, 6, 1024, 0.1
# The sizes of the model: d_model, heads, layers, feed-forward width. A small molecule table has little to teach
# a big model, so a smaller one is offered next to the base one.
SIZE = "base"  # the size ``Decoder`` builds now (see ``use_size``)
SIZES = {"base": (256, 8, 6, 1024), "small": (128, 4, 4, 512)}
MAX_POSITIONS = 1024  # the molecule region: the longest molecule has 837 symbols


def use_size(name: str) -> None:
    """Make ``Decoder`` build the model of this size (a key of ``SIZES``)."""
    global D_MODEL, HEADS, LAYERS, FEEDFORWARD, SIZE
    D_MODEL, HEADS, LAYERS, FEEDFORWARD = SIZES[name]
    SIZE = name


def set_dropout(model: nn.Module, probability: float) -> None:
    """Change the dropout of every layer of a model that is already built, for example before fine-tuning."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = probability
        elif isinstance(module, nn.MultiheadAttention):
            module.dropout = probability


def positions_of(ids: torch.Tensor, vocabulary: Vocabulary) -> torch.Tensor:
    """0 for the start tokens, the padding and the odor words, and 1, 2, 3 ... for the rest, in order."""
    free = ids >= vocabulary.first_word
    for name in ("<pad>", "<bos>", "<odor>", "<mol>"):
        free |= ids == vocabulary.special(name)
    counted = ~free
    return torch.cumsum(counted, dim=1) * counted


class Decoder(nn.Module):
    def __init__(self, vocabulary: Vocabulary):
        super().__init__()
        self.vocabulary = vocabulary
        self.tokens = nn.Embedding(len(vocabulary), D_MODEL)
        self.positions = nn.Embedding(MAX_POSITIONS + 1, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, HEADS, FEEDFORWARD, DROPOUT, batch_first=True, norm_first=True
        )
        self.body = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, len(vocabulary), bias=False)
        self.head.weight = self.tokens.weight  # the output layer and the embedding are one table

    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        """The hidden state of every token: the representation that the analysis reads."""
        positions = positions_of(ids, self.vocabulary).clamp(max=MAX_POSITIONS)
        state = self.tokens(ids) + self.positions(positions)
        return self.norm(self.body(state, mask=self.blocked(ids)))

    def blocked(self, ids: torch.Tensor) -> torch.Tensor:
        """The attention mask (True: not allowed), one for every head of every text of the batch."""
        batch, length = ids.shape
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=ids.device), 1)
        mask = causal[None].repeat(batch, 1, 1)
        is_word = ids >= self.vocabulary.first_word
        in_request = torch.cumsum(ids == self.vocabulary.special("<sep>"), dim=1) == 0
        others = is_word[:, None, :] & ~torch.eye(length, dtype=torch.bool, device=ids.device)
        mask |= (is_word & in_request)[:, :, None] & others
        return mask.repeat_interleave(HEADS, dim=0)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(ids))
