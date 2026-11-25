"""Where the odor tokens sit, and whether closeness follows the data.

Issue #8: the 890 odor-token embeddings before and after step 2, the Spearman correlation between the cosine of
two embeddings and the Jaccard overlap of the molecules they label, and the two maps.

    python -m scripts.analysis.odor_map

Writes ``output/analysis/odor_map.json`` and the figures ``odor_map.png`` and ``odor_heatmap.png`` under
``output/odor/report/``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scripts.core import plots
from scripts.core.training import OUTPUT_DIR, load_model
from scripts.core.vocabulary import Vocabulary
from scripts.preprocessing.odor_tables import DATASET_PATH

MINIMUM_MOLECULES = 10  # a token with fewer has too few neighbours to compare


def label_sets(vocabulary: Vocabulary, split: str = "train") -> list[set[int]]:
    """The training molecules each odor token labels, as row numbers of the fine-tuning table."""
    table = pd.read_csv(DATASET_PATH).fillna("")
    table = table[table["split"] == split]
    index = {token: position for position, token in enumerate(vocabulary.odor_tokens)}
    sets: list[set[int]] = [set() for _ in index]
    for row, text in enumerate(table["odor_tokens"]):
        for token in str(text).split(";"):
            if token in index:
                sets[index[token]].add(row)
    return sets


def jaccard(first: set[int], second: set[int]) -> float:
    union = len(first | second)
    return len(first & second) / union if union else 0.0


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Rank correlation, with ties averaged, which is what Spearman means."""
    a = pd.Series(first).rank().to_numpy()
    b = pd.Series(second).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


def seriate(similarity: np.ndarray) -> list[int]:
    """An order that puts similar tokens next to each other, greedily from the most central one."""
    left = list(range(len(similarity)))
    order = [left.pop(int(np.argmax(similarity.sum(axis=1))))]
    while left:
        last = order[-1]
        following = max(left, key=lambda index: similarity[last, index])
        left.remove(following)
        order.append(following)
    return order


def correlations(
    before: torch.Tensor,
    after: torch.Tensor,
    sets: list[set[int]],
    minimum: int = MINIMUM_MOLECULES,
) -> dict:
    """Cosine against label overlap, over the token pairs where both tokens have enough molecules."""
    kept = [position for position, found in enumerate(sets) if len(found) >= minimum]
    overlap = np.array(
        [jaccard(sets[i], sets[j]) for a, i in enumerate(kept) for j in kept[a + 1 :]]
    )
    answer = {"minimum_molecules": minimum, "tokens": len(kept), "pairs": len(overlap)}
    for name, matrix in (("before", before), ("after", after)):
        rows = torch.nn.functional.normalize(matrix[kept], dim=1)
        cosine = (rows @ rows.T).numpy()
        pairs = np.array([cosine[i, j] for i in range(len(kept)) for j in range(i + 1, len(kept))])
        answer[name] = spearman(pairs, overlap)
        if name == "after":
            counts = np.array([min(len(sets[i]), len(sets[j])) for i, j in _pairs(kept)])
            answer["pair_data"] = {
                "cosine": [round(float(value), 4) for value in pairs],
                "jaccard": [round(float(value), 4) for value in overlap],
                "min_count": counts.tolist(),
            }
            answer["zero_overlap_pairs"] = int((overlap == 0).sum())
            shared = overlap > 0
            answer["shared_overlap_pairs"] = int(shared.sum())
            answer["spearman_shared"] = spearman(pairs[shared], overlap[shared])
            edges = (0.0, 1e-9, 0.02, 0.05, 0.1, 1.0)
            answer["overlap_bins"] = [
                {
                    "label": "none" if low == 0.0 else f"{low:.2f} to {high:.2f}",
                    "pairs": int(((overlap >= low) & (overlap < high)).sum()),
                    "mean_cosine": float(pairs[(overlap >= low) & (overlap < high)].mean()),
                }
                for low, high in zip(edges[:-1], edges[1:], strict=True)
            ]
    return answer


def _pairs(kept: list[int]):
    for first, left in enumerate(kept):
        for right in kept[first + 1 :]:
            yield left, right


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="The odor-token map and its correlation with the data."
    )
    parser.add_argument("--minimum", type=int, default=MINIMUM_MOLECULES)
    arguments = parser.parse_args(argv)

    vocabulary = Vocabulary.load()
    stage1 = load_model(vocabulary, OUTPUT_DIR / "stage1/best.pt")
    stage2 = load_model(vocabulary, OUTPUT_DIR / "stage2/best.pt")
    before = stage1.tokens.weight.detach()[vocabulary.first_odor :]
    after = stage2.tokens.weight.detach()[vocabulary.first_odor :]
    sets = label_sets(vocabulary)
    found = correlations(before, after, sets, arguments.minimum)
    frequency = np.array([len(s) for s in sets])
    order = np.argsort(-frequency, kind="stable")[:40]
    names = [vocabulary.odor_tokens[i] for i in order]

    coordinates = {}
    for name, matrix in (("before", before), ("after", after)):
        coordinates[name] = PCA(n_components=2, svd_solver="full").fit_transform(matrix.numpy())
    spread = {
        name: TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(
            matrix.numpy()
        )
        for name, matrix in (("before", before), ("after", after))
    }
    plots.odor_map_figure(
        coordinates["before"],
        coordinates["after"],
        names,
        order.tolist(),
        OUTPUT_DIR / "report/odor_map.png",
    )
    rows = torch.nn.functional.normalize(after[order], dim=1)
    similarity = (rows @ rows.T).numpy()
    near = seriate(similarity)
    plots.odor_heatmap(
        similarity[np.ix_(near, near)],
        [names[i] for i in near],
        OUTPUT_DIR / "report/odor_heatmap.png",
    )

    plots.odor_tsne_figure(
        spread["before"],
        spread["after"],
        names[:15],
        order[:15].tolist(),
        OUTPUT_DIR / "report/odor_tsne.png",
    )
    plots.odor_overlap_figure(found["overlap_bins"], OUTPUT_DIR / "report/odor_overlap.png")

    found["top_tokens"] = names
    found["top_positions"] = order.tolist()
    found["tsne_before"] = spread["before"].round(4).tolist()
    found["tsne_after"] = spread["after"].round(4).tolist()
    found["heatmap_tokens"] = [names[i] for i in near]
    found["heatmap"] = [
        [round(float(value), 4) for value in row] for row in similarity[np.ix_(near, near)]
    ]
    folder = OUTPUT_DIR.parent / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "odor_map.json"
    target.write_text(json.dumps(found, indent=1))
    print(
        f"tokens with at least {arguments.minimum} molecules: {found['tokens']}, pairs: {found['pairs']}"
    )
    print(
        f"Spearman, cosine against label overlap: before {found['before']:.3f}, after {found['after']:.3f}"
    )
    print(f"wrote {target} and the two figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
