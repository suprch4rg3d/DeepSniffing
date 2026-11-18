"""Step 1, the pretraining: the decoder learns the language of molecules.

Every molecule of ``structures.csv`` as ``<bos> molecule tokens <eos>``, no odor tokens. The loss is on every molecule token.
24 epochs of a cosine schedule; the best epoch by the loss on the fixed validation molecules is kept in
``output/odor/stage1/best.pt`` and the numbers of every epoch in ``history.csv``.

    python -m scripts.core --stage 1
"""

from __future__ import annotations

import random
from pathlib import Path

from scripts.core import training
from scripts.core.data import load_packed, packed_batches
from scripts.core.model import Decoder
from scripts.core.vocabulary import Vocabulary

LEARNING_RATE, MAX_EPOCHS, PATIENCE = 5e-4, 24, 24


def folder() -> Path:
    return training.OUTPUT_DIR / "stage1"


def train(vocabulary: Vocabulary, max_epochs: int | None = None) -> None:
    print("step 1, the pretraining: the language of molecules")
    model = Decoder(vocabulary)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, {len(vocabulary)} tokens")
    # read from the packed tokens: millions of molecules cost no memory and are not built again
    train_set, validation = load_packed("train"), load_packed("validation")
    tokens = int(train_set[1][-1])  # the last offset is how many tokens there are in all
    print(
        f"{len(train_set[2]):,} training and {len(validation[2]):,} validation molecules,"
        f" {tokens:,} tokens"
    )
    training.train_stage(
        folder(),
        model,
        vocabulary,
        lambda rng: packed_batches(vocabulary, train_set, rng, shuffle=True),
        {"molecule": lambda: packed_batches(vocabulary, validation, random.Random(0), False)},
        score="molecule",
        learning_rate=LEARNING_RATE,
        max_epochs=max_epochs or MAX_EPOCHS,
        patience=PATIENCE,
    )
