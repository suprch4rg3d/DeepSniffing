"""Where in the decoder the request acts, one layer at a time.

Issue #9. The six request adapters start at exactly zero and are trained by step 2. This zeros one of them at a
time, on a saved generator, and measures the loss of writing a held-out molecule when it is told its own odor
tokens. The rise in loss is how much that layer's adapter was carrying.

    python -m scripts.analysis.ablations [--molecules 120]

Writes ``output/analysis/ablations.json``.
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch

from scripts.core import data, evaluate
from scripts.core.training import OUTPUT_DIR, load_model, seed_everything
from scripts.core.vocabulary import Vocabulary


def mean_loss(model, vocabulary, items, seed: int = 1) -> float:
    """The loss of writing each molecule after a request for its own tokens, averaged."""
    rng = random.Random(seed)
    losses = []
    for selfies, odor_tokens in items:
        asked = rng.sample(odor_tokens, min(data.MAX_ODOR_TOKENS_IN_REQUEST, len(odor_tokens)))
        losses.append(evaluate.request_loss(model, vocabulary, selfies, asked))
    return float(np.mean(losses))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zero one request adapter at a time.")
    parser.add_argument(
        "--molecules", type=int, default=120, help="held-out molecules to average over"
    )
    parser.add_argument("--split", default="test")
    arguments = parser.parse_args(argv)

    seed_everything()
    vocabulary = Vocabulary.load()
    model = load_model(vocabulary, OUTPUT_DIR / "stage2/best.pt")
    items = [item for item in data.load_odor(arguments.split) if item[1]]
    random.Random(0).shuffle(items)
    items = items[: arguments.molecules]
    layers = len(model.request)

    saved = [adapter.weight.detach().clone() for adapter in model.request]
    intact = mean_loss(model, vocabulary, items)
    results = {"split": arguments.split, "molecules": len(items), "intact": intact, "zeroed": {}}
    for layer in range(layers):
        model.request[layer].weight.data.zero_()
        results["zeroed"][str(layer + 1)] = mean_loss(model, vocabulary, items)
        model.request[layer].weight.data.copy_(saved[layer])
    for adapter, weight in zip(model.request, saved, strict=True):
        adapter.weight.data.copy_(weight)

    with torch.no_grad():
        model.request.eval()
        for adapter in model.request:
            adapter.weight.data.zero_()
        results["all_zero"] = mean_loss(model, vocabulary, items)
        for adapter, weight in zip(model.request, saved, strict=True):
            adapter.weight.data.copy_(weight)

    folder = OUTPUT_DIR.parent / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "ablations.json"
    path.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"{len(items)} molecules from {arguments.split}, intact loss {intact:.4f}")
    for layer, value in results["zeroed"].items():
        print(f"  layer {layer} zeroed: {value:.4f}  (+{value - intact:.4f})")
    print(f"  all six zeroed:  {results['all_zero']:.4f}  (+{results['all_zero'] - intact:.4f})")
    print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
