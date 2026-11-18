"""Step 2, the generator: the decoder learns to write a molecule for a set of odor tokens.

It starts from the pretrained decoder and is fine-tuned on the molecules of ``odor_dataset.csv`` as requests,
``<odor> odor token odor token <sep>`` followed by the molecule. The request is a set: its odor tokens have no position and
cannot see each other, so their order cannot matter. **The loss is on the molecule only**; nothing else is
trained here, and the classifier (step 3) is not part of it.

What makes the best model, all of them the defaults (each was one change from the run before it, and the
other direction of each setting was worse by more than the noise of 0.004):

- **request adapters** (in the model): the odor tokens are averaged, and the average scales and shifts the state at
  every layer for every molecule token of the molecule. Their weights start at zero, so the pretrained model is
  unchanged at the start. Without them the model hardly used its request.
- **full-rate embeddings**: the odor tokens are the one part that is new, and layer-wise decay would train
  them slowest.
- **replay**: as many pretraining molecules as there are molecules with odor tokens are mixed into each epoch after
  an empty request, as in the fine-tuning of language models, so that what the pretraining knew stays in use.
- dropout 0.2, layer decay 0.8, peak rate 3e-4, 160 epochs (an epoch is the molecules once, so 80 gave
  a loss of 0.7290 and 160 gave 0.7057), with no early stopping: the score swings by more
  between epochs than it improves (352 validation molecules), and the cosine only arrives where it is going
  if it is let finish.

The score is the loss of writing the validation molecules after a request for their own odor tokens. Beside it the
history has the loss with an empty request and the difference, ``request_gain``: zero means the model is not
reading its request, whatever the loss itself says.

    python -m scripts.core --stage 2 [--tag NAME]       a tagged run is kept in stage2_NAME
"""

from __future__ import annotations

import random
from pathlib import Path

from scripts.core import model as model_module
from scripts.core import training
from scripts.core.data import (
    asked_molecule,
    batches,
    load_odor,
    load_packed,
    odor_to_molecule,
    unasked_molecule,
    with_replay,
)
from scripts.core.vocabulary import Vocabulary

LEARNING_RATE, MAX_EPOCHS, PATIENCE = 3e-4, 160, 160
DROPOUT = 0.2
LAYER_DECAY = 0.8  # each layer nearer the input learns at this fraction of the one above


def folder(tag: str | None = None) -> Path:
    """Where a run is kept: ``stage2``, or ``stage2_<tag>`` for a run that tries something else."""
    return training.OUTPUT_DIR / (f"stage2_{tag}" if tag else "stage2")


def request_gain(losses: dict[str, float]) -> dict[str, float]:
    """How much better a molecule is written when the request names its odor tokens than when it names none."""
    return {"request_gain": losses["without_odor"] - losses["with_odor"]}


def train(
    vocabulary: Vocabulary,
    max_epochs: int | None = None,
    tag: str | None = None,
    full_rate_embeddings: bool = True,
    dropout: float = DROPOUT,
    layer_decay: float | None = LAYER_DECAY,
    learning_rate: float | None = None,
    replay: int | None = None,
    ema: float = 0.0,
    include_validation: bool = False,
) -> None:
    """Fine-tune the pretrained decoder. Each argument changes one thing from the defaults.

    ``replay`` mixes that many molecules of the pretraining into every epoch, written after an empty
    request; by default as many as there are molecules with odor tokens, 0 turns it off.

    With ``include_validation`` the validation molecules are trained on as well, for the last model once
    the settings are fixed. They are still measured every epoch, but they are no longer held out, so
    that loss is a training loss and only the test and DREAM molecules say how good the model is.
    """
    print("step 2, the generator: odor tokens to molecule" + (f" ({tag})" if tag else ""))
    start = training.OUTPUT_DIR / "stage1/best.pt"
    if not start.is_file():
        raise FileNotFoundError(f"{start} not found. Run ds-train --stage 1 first.")
    model = training.load_model(vocabulary, start)
    model_module.set_dropout(model, dropout)
    train_set, validation = load_odor("train"), load_odor("validation")
    if include_validation:
        train_set = train_set + validation
    print(
        f"{len(train_set)} training and {len(validation)} validation molecules"
        + (" (the validation molecules are trained on)" if include_validation else "")
    )
    replay = len(train_set) if replay is None else replay
    packed = (
        load_packed("train") if replay else None
    )  # mapped, not read: it costs no memory to hold

    def training_batches(rng: random.Random):
        texts = batches(vocabulary, train_set, [odor_to_molecule], rng, shuffle=True)
        return with_replay(texts, vocabulary, packed, replay, rng) if replay else texts

    # the loss of the first is what is trained and what decides the best epoch; the other two are only
    # measured, to see what the request is worth
    measured = {
        "odor_to_molecule": [odor_to_molecule],
        "with_odor": [asked_molecule],
        "without_odor": [unasked_molecule],
    }
    training.train_stage(
        folder(tag),
        model,
        vocabulary,
        training_batches,
        {
            name: (
                lambda group=group: batches(vocabulary, validation, group, random.Random(0), False)
            )
            for name, group in measured.items()
        },
        score="odor_to_molecule",
        learning_rate=learning_rate or LEARNING_RATE,
        max_epochs=max_epochs or MAX_EPOCHS,
        patience=PATIENCE,
        layer_decay=layer_decay,
        full_rate_embeddings=full_rate_embeddings,
        ema=ema,
        derived=request_gain,
    )
