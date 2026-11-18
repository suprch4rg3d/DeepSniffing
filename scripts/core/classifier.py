"""Step 3, the classifier: it names the odor tokens of a molecule from the frozen pretrained decoder.

The decoder of step 1 is not changed by anything here. It is run once over the molecules, ``<bos> molecule tokens
<eos>`` exactly as in its pretraining, and what it computes is kept: the state at the last molecule token, which has
read the whole molecule, and the mean of the states of every molecule token. A small head is trained on top of that to
score each of the 890 odor tokens, a yes or no per odor token. Only the head learns, so nothing here can spoil the
generator (step 2) and there is no loss of another job to balance against; and the body is run only once, so a
head trains in seconds.

    the loss    the binary cross entropy per molecule, summed over the odor tokens. A molecule's odor tokens are a
                set, so every odor token is judged on its own, and nothing is divided away: a molecule with ten
                odor tokens weighs ten odor tokens' worth and one with a single odor token weighs one.
    the score   the macro ROC-AUC over the odor tokens that can be scored (chance is 0.5), on the validation
                molecules; the epoch is chosen by the loss on them, not by that number.

    python -m scripts.core --stage 3                     train the head
    python -m scripts.core.classifier --split test       measure it on a held-out split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.core import history, training
from scripts.core import model as model_module
from scripts.core.data import ATTENTION_BUDGET, load_odor
from scripts.core.metrics import odor_token_metrics, top_precision
from scripts.core.model import Decoder
from scripts.core.vocabulary import Vocabulary

HEAD = "mlp"  # "linear": one layer, the odor tokens are directions in the representation; "mlp": a hidden layer first
LEARNING_RATE = 1e-2
MAX_EPOCHS = 200
PATIENCE = 20
HIDDEN = 512
DROPOUT = 0.3
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 256


def folder(tag: str | None = None) -> Path:
    """Where a run is kept: ``stage3``, or ``stage3_<tag>`` for a run that tries something else."""
    return training.OUTPUT_DIR / (f"stage3_{tag}" if tag else "stage3")


@torch.no_grad()
def read(model: Decoder, vocabulary: Vocabulary, items: list) -> tuple[np.ndarray, np.ndarray]:
    """What the frozen decoder computes for each molecule that has odor tokens, and which odor tokens are true.

    The features of a molecule are the state at its last molecule token and the mean of the states of its molecule tokens,
    side by side. The decoder only looks backwards, so the last state is the one that has seen it all.
    """
    model.eval()
    device = training.device_of(model)
    odor_tokens = len(vocabulary) - vocabulary.first_odor
    rows = []
    for selfies, has in items:
        if has:
            ids = [
                vocabulary.special("<bos>"),
                *vocabulary.molecule_ids(selfies),
                vocabulary.special("<eos>"),
            ]
            truth = np.zeros(odor_tokens, dtype=np.float32)
            for odor_token in has:
                truth[vocabulary.index[odor_token] - vocabulary.first_odor] = 1.0
            rows.append((ids, truth))
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]))
    features = np.zeros((len(rows), 2 * model.tokens.embedding_dim), dtype=np.float32)
    pad = vocabulary.special("<pad>")
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order)
            and end - start < 64
            and (end - start + 1) * len(rows[order[end]][0]) ** 2 <= ATTENTION_BUDGET
        ):
            end += 1
        chunk = order[start:end]
        longest = max(len(rows[i][0]) for i in chunk)
        ids = torch.full((len(chunk), longest), pad, dtype=torch.long)
        for row, i in enumerate(chunk):
            ids[row, : len(rows[i][0])] = torch.tensor(rows[i][0])
        states = model.hidden(ids.to(device)).float().cpu().numpy()
        for row, i in enumerate(chunk):
            size = len(rows[i][0])  # <bos>, the molecule tokens, <eos>
            features[i] = np.concatenate(
                [states[row, size - 2], states[row, 1 : size - 1].mean(axis=0)]
            )
        start = end
    truth = (
        np.stack([truth for _, truth in rows]) if rows else np.zeros((0, odor_tokens), np.float32)
    )
    return features, truth


def make_head(kind: str, inputs: int, outputs: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(inputs, outputs)
    return nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(inputs, HIDDEN),
        nn.GELU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN, outputs),
    )


def odor_token_loss(scores: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """The binary cross entropy over the odor tokens, summed for a molecule and averaged over the molecules."""
    return nn.functional.binary_cross_entropy_with_logits(scores, truth, reduction="sum") / len(
        truth
    )


def train(
    vocabulary: Vocabulary,
    max_epochs: int | None = None,
    kind: str = HEAD,
    tag: str | None = None,
    include_validation: bool = False,
) -> None:
    """Train the head on the frozen decoder.

    With ``include_validation`` the validation molecules are trained on too, for the last model once the
    settings are fixed. With none held out there is no epoch to choose by, so it trains for as many epochs
    as the held-out run (``stage3``) chose, and keeps the weights it ends with.
    """
    print(
        "step 3, the classifier: the odor tokens of a molecule, from the frozen pretrained decoder"
    )
    start = training.OUTPUT_DIR / "stage1/best.pt"
    if not start.is_file():
        raise FileNotFoundError(f"{start} not found. Run ds-train --stage 1 first.")
    model = training.load_model(vocabulary, start).to(training.DEVICE)
    for parameter in model.parameters():
        parameter.requires_grad_(False)  # the body is read, never trained
    train_items, validation_items = load_odor("train"), load_odor("validation")
    x, y = read(model, vocabulary, train_items)
    xv, yv = read(model, vocabulary, validation_items)
    epochs = max_epochs or MAX_EPOCHS
    if include_validation:
        rows_before = history.load(
            folder()
        )  # the held-out run chose its epoch by the validation loss
        epochs = int(rows_before["validation"].idxmin()) + 1
        x, y = np.concatenate([x, xv]), np.concatenate([y, yv])
        print(
            f"the validation molecules are trained on, for {epochs} epochs as the held-out run chose"
        )
    print(
        f"{len(x)} training and {len(xv)} validation molecules with odor tokens, {y.shape[1]} odor tokens"
    )
    del model
    mean, spread = x.mean(axis=0), x.std(axis=0) + 1e-6
    device = training.DEVICE
    x, xv = (torch.tensor((a - mean) / spread, device=device) for a in (x, xv))
    y_gpu, yv_gpu = torch.tensor(y, device=device), torch.tensor(yv, device=device)

    torch.manual_seed(training.SEED)
    head = make_head(kind, x.shape[1], y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rows: list[dict[str, float]] = []
    best, kept, stale = float("inf"), None, 0
    out = folder(tag)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        head.train()
        total = 0.0
        for batch in torch.randperm(len(x), device=device).split(BATCH_SIZE):
            loss = odor_token_loss(head(x[batch]), y_gpu[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        head.eval()
        with torch.no_grad():
            scores = head(xv)
            held = float(odor_token_loss(scores, yv_gpu))
        found = odor_token_metrics(scores.cpu().numpy(), yv.astype(np.int8))
        row = {
            "epoch": epoch,
            "train": total / len(x),
            "validation": held,
            "roc_auc": found.get("roc_auc_macro", float("nan")),
            "average_precision": found.get("average_precision_macro", float("nan")),
            "precision_at_5": top_precision(scores.cpu().numpy(), yv, 5),
            "precision_at_own_count": top_precision(scores.cpu().numpy(), yv, None),
        }
        rows.append(row)
        history.save(out, rows)
        print(
            f"epoch {epoch:03d}: train {row['train']:.3f}  validation {held:.3f}"
            f"  ROC-AUC {row['roc_auc']:.4f}  average precision {row['average_precision']:.4f}",
            flush=True,
        )
        if include_validation:
            kept = {
                key: value.clone() for key, value in head.state_dict().items()
            }  # the last epoch
        elif held < best - 1e-4:
            best, stale = held, 0
            kept = {key: value.clone() for key, value in head.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    torch.save(
        {
            "head": kept,
            "kind": kind,
            "mean": mean,
            "spread": spread,
            "tokens": vocabulary.tokens,
            "size": model_module.SIZE,
        },
        out / "best.pt",
    )
    print(
        f"trained for {epochs} epochs" if include_validation else f"best validation loss {best:.3f}"
    )


def load(
    vocabulary: Vocabulary, tag: str | None = None
) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    """The trained head, with the mean and spread its features were standardised by."""
    saved = torch.load(folder(tag) / "best.pt", weights_only=False, map_location="cpu")
    if saved["tokens"] != vocabulary.tokens:
        raise ValueError("the classifier was trained with another vocabulary. Train it again.")
    head = make_head(saved["kind"], len(saved["mean"]), len(vocabulary) - vocabulary.first_odor)
    head.load_state_dict(saved["head"])
    return head.eval(), saved["mean"], saved["spread"]


def evaluate(vocabulary: Vocabulary, split: str, tag: str | None = None) -> dict:
    """The scores of the trained classifier on a split, from the frozen decoder and its head."""
    head, mean, spread = load(vocabulary, tag)
    model = training.load_model(vocabulary, training.OUTPUT_DIR / "stage1/best.pt")
    model.to(training.DEVICE)
    features, truth = read(model, vocabulary, load_odor(split))
    with torch.no_grad():
        scores = head(torch.tensor((features - mean) / spread)).numpy()
    report = odor_token_metrics(scores, truth.astype(np.int8))
    report["precision_at_5"] = top_precision(scores, truth, 5)
    report["precision_at_own_count"] = top_precision(scores, truth, None)
    report["molecules"] = len(truth)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the classifier on a held-out split.")
    parser.add_argument("--split", choices=("validation", "test", "dream"), default="validation")
    parser.add_argument("--tag", default=None, help="a run that was trained with --tag")
    arguments = parser.parse_args(argv)
    training.seed_everything()
    vocabulary = Vocabulary.load()
    report = {"split": arguments.split, **evaluate(vocabulary, arguments.split, arguments.tag)}
    path = folder(arguments.tag) / f"evaluation_{arguments.split}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
