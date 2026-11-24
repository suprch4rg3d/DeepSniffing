"""The decoder: a small GPT over the tokens of ``vocabulary.json``.

The attention is causal. A token has a position only when it is part of the molecule. The odor tokens are a
set, not a sequence, so they have no position (position 0, which also stands for the start tokens). That alone
would not make them a set, because an odor token that looks at the odor tokens before it carries their order into the
layers above. So the odor tokens of a request (the ones before the separator) look only at the start tokens and at
themselves, and every later token sees all of them. Nothing a request leads to depends on the order in which
its odor tokens were listed. The odor tokens that the model writes (after the separator) are an ordinary causal text, so
that it does not name an odor token twice. A molecule token, the separator and the end token count their place
in the molecule from 1.
"""

from __future__ import annotations

import torch
from torch import nn

from scripts.core.vocabulary import Vocabulary

D_MODEL, HEADS, LAYERS, FEEDFORWARD, DROPOUT = 256, 8, 6, 1024, 0.1
# The sizes of the model: d_model, heads, layers, feed-forward width. A small molecule table has little to teach
# a big model, so a smaller one is offered next to the base one.
SIZE = "base"  # the size ``Decoder`` builds now (see ``use_size``)
SIZES = {"base": (256, 8, 6, 1024), "small": (128, 4, 4, 512)}
MAX_POSITIONS = 1024  # the molecule region: the longest molecule has 837 molecule tokens


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
    """0 for the start tokens, the padding and the odor tokens, and 1, 2, 3 ... for the rest, in order."""
    free = ids >= vocabulary.first_odor
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
        # The request reaches the molecule in a second way. Through attention alone, a molecule token has to look
        # back to a few tokens at the start and weigh them against everything it learned from millions of
        # molecules that had no request, and measured on molecules the model was trained on for 80 epochs
        # it hardly did. So the odor tokens are also pooled into one vector, which scales and shifts the state
        # at every layer for every token from the separator on (the way a class or a text is fed into a
        # conditional image model). The pool is an average, so the order of the odor tokens still cannot matter.
        # The weights start at zero and have no bias, so the pretrained model is exactly what it was when
        # the fine-tuning begins, and without a request (the pretraining) they add nothing.
        self.request = nn.ModuleList(
            nn.Linear(D_MODEL, 2 * D_MODEL, bias=False) for _ in range(LAYERS)
        )
        for adapter in self.request:
            nn.init.zeros_(adapter.weight)

    def request_of(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Which tokens are the odor tokens asked for, and which tokens answer (the separator and after), as 0/1."""
        before = torch.cumsum(ids == self.vocabulary.special("<sep>"), dim=1) == 0
        asked = (ids >= self.vocabulary.first_odor) & before
        return asked.float(), (~before).float()

    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        """The hidden state of every token: the representation that the analysis reads."""
        positions = positions_of(ids, self.vocabulary).clamp(max=MAX_POSITIONS)
        embedded = self.tokens(ids)
        state = embedded + self.positions(positions)
        asked, answering = self.request_of(ids)
        # the mean of the asked odor tokens' embeddings; nothing asked gives zero, and so no change at all
        pooled = (embedded * asked[..., None]).sum(dim=1) / asked.sum(dim=1, keepdim=True).clamp(
            min=1
        )
        answering = answering[..., None].to(state.dtype)
        mask = self.blocked(ids)
        for layer, adapter in zip(self.body.layers, self.request, strict=True):
            scale, shift = adapter(pooled).chunk(2, dim=-1)
            state = state * (1 + answering * scale[:, None]) + answering * shift[:, None]
            state = layer(state, src_mask=mask)
        return self.norm(state)

    def blocked(self, ids: torch.Tensor) -> torch.Tensor:
        """The attention mask (True: not allowed), one for every head of every text of the batch."""
        batch, length = ids.shape
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=ids.device), 1)
        mask = causal[None].repeat(batch, 1, 1)
        is_odor_token = ids >= self.vocabulary.first_odor
        in_request = torch.cumsum(ids == self.vocabulary.special("<sep>"), dim=1) == 0
        others = is_odor_token[:, None, :] & ~torch.eye(length, dtype=torch.bool, device=ids.device)
        mask |= (is_odor_token & in_request)[:, :, None] & others
        return mask.repeat_interleave(HEADS, dim=0)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(ids))
