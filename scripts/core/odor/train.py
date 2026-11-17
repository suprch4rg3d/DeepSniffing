"""``ds-train``: train the odor decoder in two stages.

    stage 1   every molecule of ``structures.csv``, no odor words: the decoder learns to write molecules.
    stage 2   the molecules of ``odor_dataset.csv``, from the stage 1 decoder: it learns to write a molecule for
              odor words, and to name the words of a molecule. Both directions train the same weights.

The training follows the usual practice for transformers:

- AdamW with betas (0.9, 0.95); weight decay on the weight matrices only, not on biases, norms or embeddings;
- a learning rate that warms up and then decays along a cosine to a tenth of its peak;
- gradient clipping, and half precision on a GPU;
- in stage 2 a higher dropout, and a smaller learning rate for the layers nearer the input, so that the odor
  words do not wipe out what stage 1 taught the lower layers.

A stage stops when its score has not improved for a few epochs (``SETTINGS``). The score of stage 1 is the
validation loss. The score of stage 2 is how well the model names the words of the validation molecules (the
share of its five best words that are true, ``word_precision``), because that is what the loss only stands for.
A stage saves its state after every epoch and resumes from it when it is started again, so an interrupted run
loses at most one epoch. The best model of a stage is kept in ``<output>/<stage>/best.pt``, the numbers of
every epoch in ``history.csv``.

    ds-train                    stage 1, then stage 2
    ds-train --stage 1          one stage
    ds-train --scratch          stage 2 from random weights, to measure what stage 1 gives (``stage2_scratch``)
    ds-train --size small       a smaller model, in ``output/odor_small``
    ds-train --no-variants      the canonical writing of every molecule only
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from scripts.core.odor import model as model_module
from scripts.core.odor.data import (
    Item,
    batches,
    load_odor,
    load_structures,
    load_variants,
    molecule_text,
    molecule_to_words,
    words_to_molecule,
)
from scripts.core.odor.model import Decoder
from scripts.core.odor.vocabulary import Vocabulary

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "output/odor"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # a GPU when there is one
WEIGHT_DECAY, CLIP = 0.01, 1.0
BETAS = (0.9, 0.95)
WARMUP_STEPS, WARMUP_FRACTION = (
    200,
    0.05,
)  # warm up for this many steps, but at most this share of them
FINAL_LEARNING_RATE = 0.1  # the cosine ends at this fraction of the peak
STAGE_2_DROPOUT = 0.2
LAYER_DECAY = (
    0.8  # each layer nearer the input learns at this fraction of the one above, in stage 2
)
TOP_WORDS = 5
MIN_IMPROVEMENT = 1e-3
# stage: (learning rate, most epochs, patience)
SETTINGS = {1: (5e-4, 40, 4), 2: (3e-4, 80, 6)}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def parameter_groups(
    model: Decoder, learning_rate: float, layer_decay: float | None = None
) -> list[dict]:
    """The parameters in groups: weight decay on matrices only, and a learning rate per layer if asked for."""
    layers = len(model.body.layers)
    groups: dict[tuple[bool, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        decays = parameter.ndim >= 2 and not name.startswith(("tokens", "positions"))
        scale = 1.0
        if layer_decay is not None:
            if name.startswith("body.layers."):
                scale = layer_decay ** (layers - 1 - int(name.split(".")[2]))
            elif name.startswith(("tokens", "positions")):
                scale = layer_decay**layers
        groups.setdefault((decays, scale), []).append(parameter)
    return [
        {
            "params": found,
            "weight_decay": WEIGHT_DECAY if decays else 0.0,
            "lr": learning_rate * scale,
        }
        for (decays, scale), found in groups.items()
    ]


def warmup_cosine(optimizer: torch.optim.Optimizer, total_steps: int):
    """A linear warmup, then a cosine down to ``FINAL_LEARNING_RATE`` of the peak, per optimizer step."""
    warmup = min(WARMUP_STEPS, max(1, int(WARMUP_FRACTION * total_steps)))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return FINAL_LEARNING_RATE + (1 - FINAL_LEARNING_RATE) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def half_precision() -> torch.dtype | None:
    """bfloat16 where the GPU has it, else float16 (which needs a loss scaler); None on a CPU."""
    if DEVICE.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def run_epoch(
    model: Decoder,
    vocabulary: Vocabulary,
    items: list[Item],
    kinds: list,
    rng: random.Random,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """The mean loss per predicted token over the items. With an optimizer the model is trained."""
    model.train(optimizer is not None)
    dtype = half_precision()
    total, count = 0.0, 0
    for ids, mask in batches(vocabulary, items, kinds, rng, shuffle=optimizer is not None):
        ids, mask = ids.to(DEVICE), mask.to(DEVICE)
        with torch.set_grad_enabled(optimizer is not None):
            with torch.autocast(DEVICE.type, dtype=dtype, enabled=dtype is not None):
                logits = model(ids[:, :-1])
            target_mask = mask[:, 1:]
            loss = nn.functional.cross_entropy(logits[target_mask].float(), ids[:, 1:][target_mask])
        if optimizer is not None:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
        total += float(loss.detach()) * int(target_mask.sum())
        count += int(target_mask.sum())
    return total / count


@torch.no_grad()
def word_precision(
    model: Decoder, vocabulary: Vocabulary, items: list[Item], top: int = TOP_WORDS
) -> float:
    """How many of the ``top`` words that the model rates highest for a molecule are among its true words.

    The model reads the molecule and the separator, and the next token is its first word. The words it rates
    highest there are its best guesses. Averaged over the molecules that have words.
    """
    model.eval()
    prompts = []
    for selfies, words in items:
        if words:
            ids = [
                vocabulary.special("<bos>"),
                vocabulary.special("<mol>"),
                *vocabulary.symbol_ids(selfies),
                vocabulary.special("<sep>"),
            ]
            prompts.append((ids, {vocabulary.index[word] for word in words}))
    prompts.sort(key=lambda prompt: len(prompt[0]))
    pad, first_word = vocabulary.special("<pad>"), vocabulary.first_word
    shares: list[float] = []
    start = 0
    while start < len(prompts):
        end = start + 1
        while (
            end < len(prompts)
            and end - start < 64
            and (end - start + 1) * len(prompts[end][0]) ** 2 <= 64 * 160 * 160
        ):
            end += 1
        chunk = prompts[start:end]
        ids = torch.full((len(chunk), len(chunk[-1][0])), pad, dtype=torch.long)
        for row, (tokens, _) in enumerate(chunk):
            ids[row, : len(tokens)] = torch.tensor(tokens)
        ids = ids.to(DEVICE)
        last = torch.tensor([len(tokens) - 1 for tokens, _ in chunk], device=DEVICE)
        state = model.hidden(ids)[torch.arange(len(chunk), device=DEVICE), last]
        best = model.head(state)[:, first_word:].topk(top, dim=1).indices + first_word
        for row, (_, truth) in enumerate(chunk):
            shares.append(sum(int(token) in truth for token in best[row].tolist()) / top)
        start = end
    return sum(shares) / max(1, len(shares))


def save_atomically(content: dict, path: Path) -> None:
    """Write ``content`` so that a crash or a power cut cannot leave a broken file in place of the old one.

    The new file is written beside the old one and flushed to the disk before it replaces it, and the old
    one is kept as ``<name>.prev`` for one more round. (A file that is only written, and not flushed, can be
    left empty or cut short by a power cut, even though its name and size are there.)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        torch.save(content, file)
        file.flush()
        os.fsync(file.fileno())
    if path.is_file():
        path.replace(path.with_suffix(".prev"))
    temporary.replace(path)


def load_saved(path: Path) -> dict | None:
    """The saved state at ``path``, or at its ``.prev`` if the newer file cannot be read; None if neither can."""
    for candidate in (path, path.with_suffix(".prev")):
        if candidate.is_file():
            try:
                return torch.load(candidate, weights_only=False, map_location=DEVICE)
            except Exception as error:  # a file cut short by a crash
                print(
                    f"{candidate.name} cannot be read ({type(error).__name__}); trying the one before"
                )
    return None


def train_stage(
    folder: Path,
    model: Decoder,
    vocabulary: Vocabulary,
    train_items: list[Item],
    validation_items: list[Item],
    kinds: dict[str, list],
    stage: int,
    max_epochs: int | None = None,
    variants: dict[str, list[str]] | None = None,
    layer_decay: float | None = None,
) -> float:
    """Train until the score stops improving; return the best validation loss.

    ``kinds`` names the texts that every molecule is shown as. The model trains on all of them and
    is validated on each. In stage 1 the mean of these losses is the score; in stage 2 it is the word
    precision (see ``word_precision``). With ``variants`` (the other writings of a molecule, by its
    canonical SELFIES) every epoch writes each training molecule in one of them at random; the validation
    always reads the canonical writing, so the losses stay comparable.
    """
    model.to(DEVICE)
    learning_rate, most_epochs, patience = SETTINGS[stage]
    most_epochs = max_epochs or most_epochs
    every_kind = [kind for group in kinds.values() for kind in group]
    optimizer = torch.optim.AdamW(
        parameter_groups(model, learning_rate, layer_decay), lr=learning_rate, betas=BETAS
    )
    steps_per_epoch = sum(
        1 for _ in batches(vocabulary, train_items, every_kind, random.Random(0), True)
    )
    scheduler = warmup_cosine(optimizer, steps_per_epoch * most_epochs)
    scaler = torch.amp.GradScaler() if half_precision() == torch.float16 else None
    by_words = "molecule_to_words" in kinds
    state_path, best_path = folder / "state.pt", folder / "best.pt"
    history: list[dict[str, float]] = []
    done, best, stale = (
        0,
        float("inf"),
        0,
    )  # best: the lowest "cost" of the score (loss, or 1 - precision)
    best_loss = float("inf")
    saved = load_saved(state_path)
    if saved is not None:
        if saved["tokens"] == vocabulary.tokens and saved.get("size") == model_module.SIZE:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            done, best, stale = saved["epoch"], saved["best"], saved["stale"]
            best_loss, history = saved["best_loss"], saved["history"]
            print(f"resuming after epoch {done}")
        else:
            print("the saved state belongs to another vocabulary or size; training from the start")

    for epoch in range(done, most_epochs):
        if stale >= patience:
            break
        rng = random.Random(SEED + epoch)
        epoch_items = train_items
        if variants:
            epoch_items = [
                (rng.choice([selfies, *variants.get(selfies, [])]), words)
                for selfies, words in train_items
            ]
        train_loss = run_epoch(
            model, vocabulary, epoch_items, every_kind, rng, optimizer, scheduler, scaler
        )
        losses = {
            name: run_epoch(model, vocabulary, validation_items, group, random.Random(0))
            for name, group in kinds.items()
        }
        loss = sum(losses.values()) / len(losses)
        row = {"epoch": epoch + 1, "train": train_loss, "validation": loss, **losses}
        detail = "  ".join(f"{name} {value:.4f}" for name, value in losses.items())
        cost = loss
        if by_words:
            precision = word_precision(model, vocabulary, validation_items)
            row["word_precision"] = precision
            detail += f"  word precision@{TOP_WORDS} {precision:.3f}"
            cost = 1 - precision
        history.append(row)
        print(
            f"epoch {epoch + 1:02d}: train {train_loss:.4f}  validation {loss:.4f}  ({detail})",
            flush=True,
        )
        if cost < best - MIN_IMPROVEMENT:
            best, best_loss, stale = cost, loss, 0
            save_atomically(
                {
                    "state": model.state_dict(),
                    "tokens": vocabulary.tokens,
                    "size": model_module.SIZE,
                },
                best_path,
            )
        else:
            stale += 1
        save_atomically(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "best": best,
                "best_loss": best_loss,
                "stale": stale,
                "history": history,
                "tokens": vocabulary.tokens,
                "size": model_module.SIZE,
            },
            state_path,
        )
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
    if by_words:
        print(f"best word precision@{TOP_WORDS} {1 - best:.3f} (validation loss {best_loss:.4f})")
    else:
        print(f"best validation loss {best_loss:.4f}")
    return best_loss


def load_model(vocabulary: Vocabulary, path: Path) -> Decoder:
    """The decoder saved at ``path``, refused if it was trained with other tokens or of another size."""
    saved = torch.load(path, weights_only=False, map_location="cpu")
    if saved["tokens"] != vocabulary.tokens:
        raise ValueError(f"{path} was trained with another vocabulary. Train again.")
    if saved.get("size", model_module.SIZE) != model_module.SIZE:
        raise ValueError(f"{path} is a {saved['size']} model, not a {model_module.SIZE} one.")
    model = Decoder(vocabulary)
    model.load_state_dict(saved["state"])
    return model.eval()


def train_stage_1(
    vocabulary: Vocabulary, max_epochs: int | None = None, augment: bool = True
) -> None:
    print("stage 1: the language of molecules")
    model = Decoder(vocabulary)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, {len(vocabulary)} tokens")
    train, validation = load_structures("train"), load_structures("validation")
    print(f"{len(train)} training and {len(validation)} validation molecules")
    train_stage(
        OUTPUT_DIR / "stage1",
        model,
        vocabulary,
        train,
        validation,
        {"molecule": [molecule_text]},
        stage=1,
        max_epochs=max_epochs,
        variants=load_variants() if augment else None,
    )


def train_stage_2(
    vocabulary: Vocabulary, scratch: bool, max_epochs: int | None = None, augment: bool = True
) -> None:
    print("stage 2: odor words, from random weights" if scratch else "stage 2: odor words")
    if scratch:
        model = Decoder(vocabulary)
    else:
        start = OUTPUT_DIR / "stage1/best.pt"
        if not start.is_file():
            raise FileNotFoundError(f"{start} not found. Run ds-train --stage 1 first.")
        model = load_model(vocabulary, start)
    model_module.set_dropout(model, STAGE_2_DROPOUT)
    train, validation = load_odor("train"), load_odor("validation")
    print(f"{len(train)} training and {len(validation)} validation molecules")
    train_stage(
        OUTPUT_DIR / ("stage2_scratch" if scratch else "stage2"),
        model,
        vocabulary,
        train,
        validation,
        {"words_to_molecule": [words_to_molecule], "molecule_to_words": [molecule_to_words]},
        stage=2,
        max_epochs=max_epochs,
        variants=load_variants() if augment else None,
        layer_decay=None if scratch else LAYER_DECAY,
    )


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def logged(path: Path):
    """Everything printed inside the block also goes to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        original = sys.stdout
        sys.stdout = _Tee(original, log)
        try:
            yield
        finally:
            sys.stdout = original


def main(argv: list[str] | None = None) -> int:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Train the odor decoder.")
    parser.add_argument("--stage", choices=("1", "2", "both"), default="both")
    parser.add_argument("--scratch", action="store_true", help="stage 2 from random weights")
    parser.add_argument(
        "--max-epochs", type=int, default=None, help="stop each stage after this many epochs"
    )
    parser.add_argument(
        "--no-variants", action="store_true", help="train on the canonical writing only"
    )
    parser.add_argument(
        "--size", choices=tuple(model_module.SIZES), default="base", help="the size of the model"
    )
    args = parser.parse_args(argv)
    model_module.use_size(args.size)
    if args.size != "base":
        OUTPUT_DIR = PROJECT_ROOT / f"output/odor_{args.size}"
    seed_everything()
    vocabulary = Vocabulary.load()
    with logged(OUTPUT_DIR / "train.log"):
        print(f"device: {DEVICE}, {args.size} model")
        if args.stage in ("1", "both") and not args.scratch:
            train_stage_1(vocabulary, args.max_epochs, not args.no_variants)
        if args.stage in ("2", "both"):
            train_stage_2(vocabulary, args.scratch, args.max_epochs, not args.no_variants)
    return 0
