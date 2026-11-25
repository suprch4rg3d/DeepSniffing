"""How the odor token embeddings move between pretraining and fine-tuning.

Issue #8: the embeddings a token has before step 2 and after it. The pretrained model has never read an odor
token; after fine-tuning each one has been read as a request thousands of times.

    python -m scripts.analysis.embeddings [--neighbours 10]

Writes ``output/analysis/embeddings.json``: per token, the cosine between its two embeddings, the share of its
nearest neighbours that survived fine-tuning, its frequency, and a two-dimensional PCA of each stage so the
notebook can draw both maps.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from scripts.core.training import OUTPUT_DIR, load_model
from scripts.core.vocabulary import Vocabulary
from scripts.preprocessing.odor_tables import DATASET_PATH


def frequencies(vocabulary: Vocabulary) -> np.ndarray:
    table = pd.read_csv(DATASET_PATH).fillna("")
    used = Counter(
        token for text in table["odor_tokens"] for token in str(text).split(";") if token
    )
    return np.array([used.get(token, 0) for token in vocabulary.odor_tokens])


def neighbours(matrix: torch.Tensor, k: int) -> torch.Tensor:
    """The k nearest tokens of each token by cosine, itself excluded."""
    normal = torch.nn.functional.normalize(matrix, dim=1)
    similarity = normal @ normal.T
    similarity.fill_diagonal_(-2.0)
    return similarity.topk(k, dim=1).indices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="The odor token embeddings before and after step 2."
    )
    parser.add_argument("--neighbours", type=int, default=10)
    arguments = parser.parse_args(argv)

    vocabulary = Vocabulary.load()
    before = load_model(vocabulary, OUTPUT_DIR / "stage1/best.pt").tokens.weight.detach()
    after = load_model(vocabulary, OUTPUT_DIR / "stage2/best.pt").tokens.weight.detach()
    before, after = before[vocabulary.first_odor :], after[vocabulary.first_odor :]

    cosine = torch.nn.functional.cosine_similarity(before, after, dim=1).numpy()
    relative = ((after - before).norm(dim=1) / before.norm(dim=1)).numpy()
    first = neighbours(before, arguments.neighbours)
    second = neighbours(after, arguments.neighbours)
    overlap = np.array(
        [
            len(set(a.tolist()) & set(b.tolist())) / arguments.neighbours
            for a, b in zip(first, second, strict=True)
        ]
    )
    frequency = frequencies(vocabulary)
    coords_before = PCA(n_components=2).fit_transform(before.numpy()).round(4)
    coords_after = PCA(n_components=2).fit_transform(after.numpy()).round(4)

    stage1 = load_model(vocabulary, OUTPUT_DIR / "stage1/best.pt")
    stage2 = load_model(vocabulary, OUTPUT_DIR / "stage2/best.pt")
    adapters = {
        "stage1": [float(p.norm()) for p in stage1.request.parameters()],
        "stage2": [float(p.norm()) for p in stage2.request.parameters()],
    }

    report = {
        "tokens": list(vocabulary.odor_tokens),
        "frequency": frequency.tolist(),
        "cosine": cosine.round(4).tolist(),
        "relative_change": relative.round(4).tolist(),
        "neighbour_overlap": overlap.round(4).tolist(),
        "adapter_norm": adapters,
        "neighbours": arguments.neighbours,
        "pca_before": coords_before.tolist(),
        "pca_after": coords_after.tolist(),
    }
    folder = OUTPUT_DIR.parent / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "embeddings.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"{len(cosine)} odor tokens")
    print(f"cosine before and after: median {np.median(cosine):.3f}, mean {np.mean(cosine):.3f}")
    print(f"relative move: median {np.median(relative):.1%}, mean {np.mean(relative):.1%}")
    print(f"neighbours kept: mean {np.mean(overlap):.0%} of {arguments.neighbours}")
    print(f"tokens whose embedding barely moved (cosine > 0.99): {np.mean(cosine > 0.99):.0%}")
    print("adapter norms stage1:", [round(value, 1) for value in adapters["stage1"]])
    print("adapter norms stage2:", [round(value, 1) for value in adapters["stage2"]])
    print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
