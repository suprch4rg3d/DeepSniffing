"""The numbers a run is judged by besides its loss: the next token, the perplexity, and the odor token scores."""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


class Tally:
    """Counts how often the most probable molecule token is the true one, over the molecule tokens the loss counts."""

    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def add(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """``logits`` is (molecule tokens, vocabulary) and ``targets`` the true molecule tokens, (molecule tokens,)."""
        self.correct += int((logits.argmax(dim=-1) == targets).sum())
        self.total += targets.numel()

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else float("nan")


def perplexity(loss: float) -> float:
    """The loss in nats per molecule token as the number of equally likely molecule tokens the model is torn between."""
    return math.exp(loss)


def odor_token_metrics(scores: np.ndarray, truth: np.ndarray) -> dict:
    """Macro ROC-AUC and average precision over the odor tokens that can be scored at all.

    ``scores`` and ``truth`` are (molecules, odor tokens); an odor token needs a molecule with it and one without it.
    """
    scorable = (truth.sum(axis=0) > 0) & (truth.sum(axis=0) < len(truth))
    if not scorable.any():
        return {"odor_tokens_scorable": 0}
    auc = roc_auc_score(truth[:, scorable], scores[:, scorable], average=None)
    ap = average_precision_score(truth[:, scorable], scores[:, scorable], average=None)
    common = truth[:, scorable].sum(axis=0) >= 10
    return {
        "odor_tokens_scorable": int(scorable.sum()),
        "odor_tokens_in_all": int(truth.shape[1]),
        "roc_auc_macro": float(np.mean(auc)),
        "roc_auc_macro_odor_tokens_seen_10_times": float(np.mean(auc[common]))
        if common.any()
        else None,
        "average_precision_macro": float(np.mean(ap)),
    }


def top_precision(scores: np.ndarray, truth: np.ndarray, top: int | None) -> float:
    """The share of the ``top`` best-scored odor tokens of a molecule that are true, averaged over the molecules.

    With ``top`` None each molecule is asked for as many odor tokens as it has, which can reach 1. A fixed number
    cannot: half of the molecules have fewer than five odor tokens, so precision@5 cannot pass 0.744.
    """
    shares = []
    for row, true in zip(scores, truth, strict=True):
        asked = min(top or int(true.sum()), len(row))
        shares.append(true[np.argsort(-row)[:asked]].sum() / asked)
    return float(np.mean(shares))


def per_token_scores(scores: np.ndarray, truth: np.ndarray, tokens: tuple[str, ...]) -> list[dict]:
    """ROC-AUC and support for every odor token that can be scored, worst first.

    The macro average hides which tokens fail. A token needs at least one molecule with it and one without,
    so the ones that appear everywhere or nowhere are not scored at all.
    """
    rows = []
    for column, token in enumerate(tokens):
        positives = int(truth[:, column].sum())
        if 0 < positives < len(truth):
            rows.append(
                {
                    "token": token,
                    "auc": float(roc_auc_score(truth[:, column], scores[:, column])),
                    "molecules": positives,
                }
            )
    return sorted(rows, key=lambda row: row["auc"])
