"""The engine the steps that train the decoder share: one epoch, one loop, saving, loading.

It knows nothing of any step. It trains a ``Decoder`` on the loss of writing molecule tokens, saves its state
after every epoch, and resumes from it, so an interrupted run loses at most one epoch. What a step is
(its data, its settings, its score) is in ``pretrain`` and ``generator``; the classifier does not use it,
since it trains a head on a frozen model and nothing of the decoder.

The training follows the usual practice for transformers:

- AdamW with betas (0.9, 0.95); weight decay on the weight matrices only, not on biases, norms or embeddings;
- a learning rate that warms up and then decays along a cosine to a tenth of its peak;
- gradient clipping, and half precision on a GPU;
- optionally a smaller learning rate for the layers nearer the input (``layer_decay``).
"""

from __future__ import annotations

import contextlib
import math
import os
import random
from pathlib import Path

import torch
from torch import nn

from scripts.core import model as model_module
from scripts.core.history import save
from scripts.core.metrics import Tally
from scripts.core.model import Decoder
from scripts.core.vocabulary import Vocabulary
from scripts.preprocessing.odor_tokens import SCALE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# ``output/odor`` at the full scale, ``output/odor_dev`` at the dev one (``DS_SCALE``, see odor_tokens)
OUTPUT_DIR = PROJECT_ROOT / ("output/odor" if SCALE == "full" else f"output/odor_{SCALE}")
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # a GPU when there is one
WEIGHT_DECAY, CLIP = 0.01, 1.0
BETAS = (0.9, 0.95)
WARMUP_STEPS, WARMUP_FRACTION = (
    200,
    0.05,
)  # warm up for this many steps, but at most this share of them
FINAL_LEARNING_RATE = 0.1  # the cosine ends at this fraction of the peak
MIN_IMPROVEMENT = 1e-3


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def parameter_groups(
    model: Decoder,
    learning_rate: float,
    layer_decay: float | None = None,
    full_rate_embeddings: bool = False,
) -> list[dict]:
    """The parameters in groups: weight decay on matrices only, and a learning rate per layer if asked for.

    With ``full_rate_embeddings`` the token table learns at the full rate whatever the layer decay is.
    The decay treats the table as the most pretrained part of the model, which is true of the molecule
    molecule tokens but not of the odor tokens: they were never read as input before the fine-tuning, so they
    are the one part of it that is new, and the one that has to learn what each odor token asks for.
    """
    layers = len(model.body.layers)
    groups: dict[tuple[bool, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        decays = parameter.ndim >= 2 and not name.startswith(("tokens", "positions"))
        scale = 1.0
        if layer_decay is not None:
            if name.startswith("body.layers."):
                scale = layer_decay ** (layers - 1 - int(name.split(".")[2]))
            elif name.startswith("tokens") and full_rate_embeddings:
                scale = 1.0
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


def device_of(model: nn.Module) -> torch.device:
    """Where the model is: everything it is given has to be there too, and that is not always ``DEVICE``."""
    return next(model.parameters()).device


def half_precision(device: torch.device = DEVICE) -> torch.dtype | None:
    """bfloat16 where the GPU has it, else float16 (which needs a loss scaler); None on a CPU."""
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def run_epoch(
    model: Decoder,
    batch_source,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler: torch.amp.GradScaler | None = None,
    after_step=None,
    tally: Tally | None = None,
) -> float:
    """The mean loss of writing a molecule token over the batches of ``batch_source``.

    With an optimizer the model is trained. A ``tally`` that is given is filled with how often the most
    probable molecule token was the true one. ``batch_source`` is anything that yields ``(ids, mask)``: the token
    numbers of a batch and the mask of the tokens the loss counts (``data.batches``, or ``data.packed_batches``
    for the pretraining read from its token array).
    """
    model.train(optimizer is not None)
    device = device_of(model)
    dtype = half_precision(device)
    total, count = 0.0, 0
    for ids, mask in batch_source:
        ids, mask = ids.to(device), mask.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = model.head(model.hidden(ids[:, :-1]))
            target_mask = mask[:, 1:]
            if not target_mask.any():
                continue
            picked, true = logits[target_mask].float(), ids[:, 1:][target_mask]
            loss = nn.functional.cross_entropy(picked, true)
            if tally is not None:
                tally.add(picked.detach(), true)
            weight = int(target_mask.sum())
            total += float(loss.detach()) * weight
            count += weight
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
            if after_step is not None:
                after_step()
    return total / max(1, count)


class WeightAverage:
    """An exponential moving average of the weights, which is what the model is judged and saved as.

    The weights of a run wander from one epoch to the next, and on a validation set this small the score
    of the single best epoch is partly luck (the best epoch of a run scored 0.695 on the measure it was
    picked by and 0.712 when scored again on fresh requests). Averaging the weights over the last few
    hundred steps removes most of that wandering and is routine in fine-tuning for the same reason. The
    training goes on with its own weights; the average is only swapped in to be scored and saved.

    The decay starts low and rises to ``decay``, so the average is not weighed down by the first steps.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay, self.steps = decay, 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.steps += 1
        rate = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for key, value in model.state_dict().items():
            self.shadow[key].mul_(rate).add_(value.detach(), alpha=1 - rate)

    @contextlib.contextmanager
    def applied(self, model: nn.Module):
        """Inside the block the model has the averaged weights; afterwards it has its own again."""
        own = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        try:
            yield
        finally:
            model.load_state_dict(own)


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
    train_batches,
    validation_batches: dict[str, object],
    score: str,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    layer_decay: float | None = None,
    full_rate_embeddings: bool = False,
    ema: float = 0.0,
    derived=None,
) -> float:
    """Train until the score stops improving; return the best validation loss.

    ``train_batches(rng)`` and each of ``validation_batches`` give a fresh run of batches when called,
    so an epoch can be read again without holding it. The model is validated on each kind on its own, and
    the loss of the one called ``score`` decides which epoch is the best. ``derived`` may turn the losses
    of one epoch into more numbers for the history (it is given the dict of losses).

    With ``ema`` above zero the model is scored and saved as the moving average of its weights (see
    ``WeightAverage``) and not as the weights of the moment.
    """
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, learning_rate, layer_decay, full_rate_embeddings),
        lr=learning_rate,
        betas=BETAS,
    )
    steps_per_epoch = sum(1 for _ in train_batches(random.Random(0)))
    scheduler = warmup_cosine(optimizer, steps_per_epoch * max_epochs)
    scaler = torch.amp.GradScaler() if half_precision() == torch.float16 else None
    state_path, best_path = folder / "state.pt", folder / "best.pt"
    history: list[dict[str, float]] = []
    done, best, stale = 0, float("inf"), 0
    average = WeightAverage(model, ema) if ema else None
    saved = load_saved(state_path)
    if saved is not None:
        if saved["tokens"] == vocabulary.tokens and saved.get("size") == model_module.SIZE:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            done, best, stale, history = (
                saved["epoch"],
                saved["best"],
                saved["stale"],
                saved["history"],
            )
            if average is not None and saved.get("ema") is not None:
                average.shadow, average.steps = saved["ema"], saved["ema_steps"]
            print(f"resuming after epoch {done}")
        else:
            print("the saved state belongs to another vocabulary or size; training from the start")

    for epoch in range(done, max_epochs):
        if stale >= patience:
            break
        rng = random.Random(SEED + epoch)
        train_tally = Tally()
        train_loss = run_epoch(
            model,
            train_batches(rng),
            optimizer,
            scheduler,
            scaler,
            after_step=(lambda: average.update(model)) if average else None,
            tally=train_tally,
        )
        with average.applied(model) if average else contextlib.nullcontext():
            losses, tallies = {}, {}
            for name, make in validation_batches.items():
                tallies[name] = Tally()
                losses[name] = run_epoch(model, make(), tally=tallies[name])
            loss = losses[score]
            row = {"epoch": epoch + 1, "train": train_loss, "validation": loss, **losses}
            row["train_accuracy"] = train_tally.accuracy
            for name, tally in tallies.items():
                row[f"{name}_accuracy"] = tally.accuracy
            row["validation_accuracy"] = row[f"{score}_accuracy"]
            detail = "  ".join(f"{name} {value:.4f}" for name, value in losses.items())
            for name, value in (derived(losses) if derived else {}).items():
                row[name] = value
                detail += f"  {name.replace('_', ' ')} {value:+.4f}"
            history.append(row)
            print(
                f"epoch {epoch + 1:02d}: train {train_loss:.4f}  validation {loss:.4f}  ({detail})",
                flush=True,
            )
            if loss < best - MIN_IMPROVEMENT:
                best, stale = loss, 0
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
                "stale": stale,
                "history": history,
                "tokens": vocabulary.tokens,
                "size": model_module.SIZE,
                "ema": average.shadow if average else None,
                "ema_steps": average.steps if average else 0,
            },
            state_path,
        )
        save(folder, history)
    # the copies kept against a crash while training are of no use once it has finished
    for leftover in (state_path.with_suffix(".prev"), best_path.with_suffix(".prev")):
        leftover.unlink(missing_ok=True)
    print(f"best validation loss {best:.4f}")
    return best


def load_model(vocabulary: Vocabulary, path: Path) -> Decoder:
    """The decoder saved at ``path``, refused if it was trained with other tokens or of another size."""
    saved = torch.load(path, weights_only=False, map_location="cpu")
    if saved["tokens"] != vocabulary.tokens:
        raise ValueError(f"{path} was trained with another vocabulary. Train again.")
    if saved.get("size", model_module.SIZE) != model_module.SIZE:
        raise ValueError(f"{path} is a {saved['size']} model, not a {model_module.SIZE} one.")
    model = Decoder(vocabulary)
    # A model saved before the request adapters existed has no weights for them, and they start at zero,
    # which leaves such a model exactly what it was. A model saved before the classifier was a step of its
    # own has a naming head the decoder no longer has, and that is dropped. Nothing else may differ.
    result = model.load_state_dict(saved["state"], strict=False)
    wrong = [k for k in result.missing_keys if not k.startswith("request.")] + [
        k for k in result.unexpected_keys if not k.startswith("words.")
    ]
    if wrong:
        raise ValueError(f"{path} does not fit the model: {wrong[:4]}")
    return model.eval()
