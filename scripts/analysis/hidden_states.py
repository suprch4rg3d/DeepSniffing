"""The state of a molecule, conditioned and not, in three settings.

Issue #11. For validation molecules that carry exactly one of eight frequent odor tokens, take the mean of the
last-layer state over the molecule tokens in three settings: the generator asked for the molecule's own tokens,
the generator asked for nothing, and the frozen pretrained decoder. Project each to two dimensions and report a
silhouette score per setting, because the picture is not the evidence.

    python -m scripts.analysis.hidden_states [--per-token 25]

Writes ``output/analysis/hidden_states.json`` and ``output/odor/report/hidden_states.png``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score

from scripts.core import data, plots
from scripts.core.training import OUTPUT_DIR, load_model
from scripts.core.vocabulary import Vocabulary

TOKENS, PER_TOKEN = 8, 25


def frequent_tokens(vocabulary: Vocabulary, count: int = TOKENS) -> list[str]:
    """The most frequent odor tokens of the training molecules that the vocabulary knows."""
    table = data.load_odor("train")
    counts = Counter(token for _selfies, tokens in table for token in tokens)
    known = set(vocabulary.odor_tokens)
    return [token for token, _ in counts.most_common() if token in known][:count]


def one_token_items(
    items: list, frequent: list[str], per_token: int = PER_TOKEN
) -> tuple[list, np.ndarray]:
    """Validation molecules that carry exactly one of the frequent tokens, up to a few per token."""
    used = Counter()
    kept, labels = [], []
    for selfies, tokens in items:
        present = [token for token in tokens if token in frequent]
        if len(present) != 1:
            continue
        token = present[0]
        if used[token] >= per_token:
            continue
        used[token] += 1
        kept.append((selfies, tokens))
        labels.append(frequent.index(token))
    return kept, np.array(labels)


def mean_state(model, vocabulary: Vocabulary, selfies: str, tokens: list[str], setting: str):
    """The mean of the last-layer state over the molecule tokens."""
    if setting == "own":
        ids, start = data.request_text(
            vocabulary, selfies, tokens[: data.MAX_ODOR_TOKENS_IN_REQUEST]
        )
    elif setting == "empty":
        ids, start = data.request_text(vocabulary, selfies, [])
    else:
        ids, start = data.molecule_text(vocabulary, selfies, [], random.Random(0))
    with torch.no_grad():
        hidden = model.hidden(torch.tensor([ids]))
    return hidden[0, start:].mean(dim=0).numpy()


def separation(states: np.ndarray, labels: np.ndarray) -> float:
    """How well the odor token can be read from the state, as a silhouette score."""
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(silhouette_score(states, labels))


def knn_accuracy(states: np.ndarray, labels: np.ndarray, k: int = 5) -> tuple[float, np.ndarray]:
    """The share of molecules whose k nearest neighbours carry the same token, leaving each one out."""
    distance = pairwise_distances(states)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argsort(distance, axis=1)[:, :k]
    votes = labels[nearest]
    predicted = np.array([np.bincount(row).argmax() for row in votes])
    return float((predicted == labels).mean()), predicted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Molecule states in three settings.")
    parser.add_argument("--per-token", type=int, default=PER_TOKEN)
    arguments = parser.parse_args(argv)

    vocabulary = Vocabulary.load()
    stage1 = load_model(vocabulary, OUTPUT_DIR / "stage1/best.pt").eval()
    stage2 = load_model(vocabulary, OUTPUT_DIR / "stage2/best.pt").eval()
    frequent = frequent_tokens(vocabulary)
    items, labels = one_token_items(
        [item for item in data.load_odor("validation") if item[1]], frequent, arguments.per_token
    )
    states = {
        "own": np.array([mean_state(stage2, vocabulary, s, t, "own") for s, t in items]),
        "empty": np.array([mean_state(stage2, vocabulary, s, t, "empty") for s, t in items]),
        "frozen": np.array([mean_state(stage1, vocabulary, s, t, "frozen") for s, t in items]),
    }
    projection = PCA(n_components=2, svd_solver="full").fit(states["frozen"])
    accuracy, votes = knn_accuracy(states["own"], labels)
    confusion = np.zeros((len(frequent), len(frequent)), dtype=int)
    for true, predicted in zip(labels, votes, strict=True):
        confusion[true, predicted] += 1
    report = {
        "tokens": frequent,
        "molecules": len(items),
        "per_token": {
            frequent[index]: count for index, count in Counter(labels.tolist()).most_common()
        },
        "silhouette": {name: separation(values, labels) for name, values in states.items()},
        "knn_accuracy": {name: knn_accuracy(values, labels)[0] for name, values in states.items()},
        "chance": float(max(Counter(labels.tolist()).values()) / len(labels)),
        "confusion": confusion.tolist(),
        "displacement": float(np.median(np.linalg.norm(states["own"] - states["empty"], axis=1))),
        "coordinates": {
            name: projection.transform(values).round(4).tolist() for name, values in states.items()
        },
        "labels": labels.tolist(),
    }
    plots.hidden_states_figure(report, frequent, OUTPUT_DIR / "report/hidden_states.png")
    folder = OUTPUT_DIR.parent / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "hidden_states.json"
    target.write_text(json.dumps(report, indent=1))
    for name, found in report["silhouette"].items():
        print(
            f"{name:6} silhouette {found:+.3f}, kNN {report['knn_accuracy'][name]:.3f}"
            f" over {len(items)} molecules"
        )
    print(f"chance {report['chance']:.3f}, median displacement {report['displacement']:.3f}")
    print(f"wrote {target} and the figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
