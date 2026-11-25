"""Where the request acts: how strongly each adapter changes the state, layer by layer.

Issue #9. Each of the six adapters turns the mean of the requested odor tokens into a scale and a shift, and the
state from the separator on becomes ``state * (1 + scale) + shift``. This measures both, per layer and per token,
and puts them beside the norm of a typical hidden state on validation molecules.

    python -m scripts.analysis.request_layers [--molecules 100]

Writes ``output/analysis/request_layers.json`` and two figures under ``output/odor/report/``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from scripts.core import data, plots
from scripts.core.training import OUTPUT_DIR, load_model
from scripts.core.vocabulary import Vocabulary


def adapter_norms(model, vocabulary: Vocabulary) -> tuple[np.ndarray, np.ndarray]:
    """The norm of scale and of shift at each layer for every odor token, as two arrays of tokens by layers."""
    embeddings = model.tokens.weight.detach()[vocabulary.first_odor :]
    scale = np.zeros((len(embeddings), len(model.request)))
    shift = np.zeros_like(scale)
    with torch.no_grad():
        for layer, adapter in enumerate(model.request):
            turned = adapter(embeddings)
            first, second = turned.chunk(2, dim=1)
            scale[:, layer] = first.norm(dim=1).numpy()
            shift[:, layer] = second.norm(dim=1).numpy()
    return scale, shift


def hidden_norms(model, vocabulary: Vocabulary, items: list, count: int) -> list[float]:
    """The median norm of the state of the molecule tokens after each layer, over a few validation molecules."""
    captured: list[torch.Tensor] = []
    hooks = [
        layer.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
        for layer in model.body.layers
    ]
    lengths = []
    try:
        model.eval()
        with torch.no_grad():
            for selfies, odor_tokens in items[:count]:
                ids, start = data.request_text(
                    vocabulary, selfies, odor_tokens[: data.MAX_ODOR_TOKENS_IN_REQUEST]
                )
                captured.clear()
                model.hidden(torch.tensor([ids]))
                lengths.append([float(state[0, start:].norm(dim=1).median()) for state in captured])
    finally:
        for hook in hooks:
            hook.remove()
    return np.median(np.array(lengths), axis=0).tolist()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The request's scale and shift at each layer.")
    parser.add_argument("--molecules", type=int, default=100)
    arguments = parser.parse_args(argv)

    vocabulary = Vocabulary.load()
    model = load_model(vocabulary, OUTPUT_DIR / "stage2/best.pt")
    scale, shift = adapter_norms(model, vocabulary)
    items = [item for item in data.load_odor("validation") if item[1]]
    hidden = hidden_norms(model, vocabulary, items, arguments.molecules)
    means = {
        "scale": scale.mean(axis=0).tolist(),
        "shift": shift.mean(axis=0).tolist(),
        "hidden": hidden,
    }
    means["shift_over_hidden"] = (np.array(means["shift"]) / np.array(hidden)).tolist()
    means["scale_over_hidden"] = (np.array(means["scale"]) / np.array(hidden)).tolist()

    table = data.load_odor("train")
    counts = {token: 0 for token in vocabulary.odor_tokens}
    for _selfies, odor_tokens in table:
        for token in odor_tokens:
            counts[token] += 1
    frequency = np.array([counts[token] for token in vocabulary.odor_tokens])
    order = np.argsort(-frequency, kind="stable")[:30]
    means["top_tokens"] = [vocabulary.odor_tokens[i] for i in order]
    means["top_shift"] = [[round(float(value), 3) for value in row] for row in shift[order]]

    plots.request_norms(means, OUTPUT_DIR / "report/request_norms.png")
    plots.request_heatmap(
        shift[order], means["top_tokens"], OUTPUT_DIR / "report/request_heatmap.png"
    )
    folder = OUTPUT_DIR.parent / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "request_layers.json"
    target.write_text(json.dumps(means, indent=1))
    print("mean scale", [round(value, 3) for value in means["scale"]])
    print("mean shift", [round(value, 3) for value in means["shift"]])
    print("shift over hidden", [round(value, 3) for value in means["shift_over_hidden"]])
    print(f"wrote {target} and the two figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
