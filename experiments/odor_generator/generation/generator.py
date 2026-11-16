"""The decoder: a small GPT that writes a molecule for a list of odor words.

A sequence is ``<bos> <word> <word> ... <sep> SELFIES symbols ... <eos>``. The words are the odor that was
asked for, in random order, and ``<none>`` stands for "no words". During training each labeled molecule is
shown with a random subset of its own words (1 to 6), and in a tenth of the cases with none, and a molecule
without words is always shown with ``<none>``. So the decoder can be asked for any combination of words, and
also for nothing. The loss is on the molecule and the end token only: the words are an instruction to follow,
not something to predict. The attention is causal and the positions are learned.

Training saves its state after every epoch and resumes from it when it is started again, so an interrupted run
loses at most one epoch. ``data/generator.pt`` always holds the best model so far.

Run:  poetry run python -m generation.generator
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch import nn

from .chemistry import SEED, seed_everything, selfies_to_smiles, selfies_tokens
from .dataset import load_dataset
from .paths import GENERATOR, GENERATOR_TRAINING, VOCABULARY
from .vocabulary import (
    BOS,
    EOS,
    MAX_SYMBOLS,
    MAX_WORDS_IN_PROMPT,
    NONE,
    PAD,
    SEP,
    Vocabulary,
    build_vocabulary,
)

D_MODEL, HEADS, LAYERS, FEEDFORWARD, DROPOUT = 256, 8, 6, 1024, 0.1
BATCH, LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE = 128, 5e-4, 0.01, 80, 6
MAX_LENGTH = (
    1 + MAX_WORDS_IN_PROMPT + 1 + MAX_SYMBOLS + 1
)  # <bos>, the words, <sep>, the molecule, <eos>
NO_WORDS_PROBABILITY = 0.1
MIN_IMPROVEMENT = 1e-3


class Generator(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        self.tokens = nn.Embedding(vocabulary_size, D_MODEL)
        self.positions = nn.Embedding(MAX_LENGTH, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, HEADS, FEEDFORWARD, DROPOUT, batch_first=True, norm_first=True
        )
        self.body = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocabulary_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        length = ids.shape[1]
        positions = torch.arange(length, device=ids.device)
        hidden = self.tokens(ids) + self.positions(positions)
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=ids.device), diagonal=1
        )
        return self.head(self.norm(self.body(hidden, mask=causal)))


def prompt_ids(vocabulary: Vocabulary, words: list[str]) -> list[int]:
    ids = [vocabulary.index[BOS]]
    ids += [vocabulary.word_id(w) for w in words] if words else [vocabulary.index[NONE]]
    return ids + [vocabulary.index[SEP]]


def encode_example(
    vocabulary: Vocabulary, symbols: list[str], words: list[str], rng: random.Random
):
    """The prompt and the body (molecule and end token) of one training example."""
    chosen: list[str] = []
    if words and rng.random() >= NO_WORDS_PROBABILITY:
        chosen = rng.sample(words, rng.randint(1, min(MAX_WORDS_IN_PROMPT, len(words))))
    prompt = prompt_ids(vocabulary, chosen)
    body = [vocabulary.index[s] for s in symbols] + [vocabulary.index[EOS]]
    return prompt, body


def batches(vocabulary: Vocabulary, molecules: list, rng: random.Random, shuffle: bool = True):
    """Padded batches of token ids, and the mask of the positions where the loss is taken."""
    order = list(range(len(molecules)))
    if shuffle:
        rng.shuffle(order)
    pad = vocabulary.index[PAD]
    for start in range(0, len(order), BATCH):
        rows = [
            encode_example(vocabulary, *molecules[i], rng) for i in order[start : start + BATCH]
        ]
        longest = max(len(p) + len(b) for p, b in rows)
        ids = torch.full((len(rows), longest), pad, dtype=torch.long)
        loss_mask = torch.zeros((len(rows), longest), dtype=torch.bool)
        for r, (prompt, body) in enumerate(rows):
            sequence = prompt + body
            ids[r, : len(sequence)] = torch.tensor(sequence)
            loss_mask[r, len(prompt) : len(sequence)] = True
        yield ids, loss_mask


def epoch_loss(
    model: Generator, vocabulary: Vocabulary, molecules: list, rng: random.Random, optimizer=None
) -> float:
    """The mean loss per molecule token over one pass. With an optimizer the model is trained."""
    model.train(optimizer is not None)
    total, count = 0.0, 0
    for ids, loss_mask in batches(vocabulary, molecules, rng, shuffle=optimizer is not None):
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(ids[:, :-1])
            loss = nn.functional.cross_entropy(
                logits[loss_mask[:, 1:]], ids[:, 1:][loss_mask[:, 1:]]
            )
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.detach()) * int(loss_mask.sum())
        count += int(loss_mask.sum())
    return total / count


@torch.no_grad()
def sample(
    model: Generator, vocabulary: Vocabulary, words: list[str], count: int, temperature: float = 1.0
) -> list[str | None]:
    """``count`` molecules written for ``words``, as canonical SMILES; None where the writing is not a molecule."""
    model.eval()
    ids = torch.tensor([prompt_ids(vocabulary, words)] * count)
    finished = torch.zeros(count, dtype=torch.bool)
    eos = vocabulary.index[EOS]
    for _ in range(MAX_LENGTH - ids.shape[1]):
        logits = model(ids)[:, -1] / temperature
        end_logit = logits[:, eos].clone()
        logits[:, : vocabulary.first_symbol] = -float("inf")  # only a symbol or the end
        logits[:, eos] = end_logit
        next_ids = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)
        next_ids[finished] = vocabulary.index[PAD]
        ids = torch.cat([ids, next_ids[:, None]], dim=1)
        finished |= next_ids == eos
        if bool(finished.all()):
            break
    start = len(prompt_ids(vocabulary, words))
    results = []
    for row in ids[:, start:].tolist():
        symbols = []
        for token in row:
            if token in (eos, vocabulary.index[PAD]):
                break
            symbols.append(vocabulary.tokens[token])
        results.append(selfies_to_smiles(symbols) if symbols else None)
    return results


def load_generator() -> tuple[Generator, Vocabulary]:
    """The trained decoder and its vocabulary."""
    vocabulary = Vocabulary.load(VOCABULARY)
    saved = torch.load(GENERATOR, weights_only=False)
    if saved["vocabulary_size"] != len(vocabulary):
        raise ValueError(
            "data/generator.pt was trained with another vocabulary. Train the generator again."
        )
    model = Generator(len(vocabulary))
    model.load_state_dict(saved["state"])
    return model.eval(), vocabulary


def save_atomically(content: dict, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(content, temporary)
    temporary.replace(path)


def main() -> None:
    seed_everything()
    frame, words = load_dataset()
    vocabulary = build_vocabulary(frame, words)
    vocabulary.save(VOCABULARY)

    def molecules(split: str) -> list:
        rows = frame[frame["split"] == split]
        pairs = (
            (selfies_tokens(s), w) for s, w in zip(rows["smiles"], rows["word_list"], strict=False)
        )
        return [(symbols, w) for symbols, w in pairs if vocabulary.can_write(symbols)]

    train, validation = molecules("train"), molecules("validation")
    print(
        f"vocabulary {len(vocabulary)} tokens ({len(words)} odor words); "
        f"{len(train)} training and {len(validation)} validation molecules"
    )
    model = Generator(len(vocabulary))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")

    done, best, stale = 0, float("inf"), 0
    if GENERATOR_TRAINING.exists():
        saved = torch.load(GENERATOR_TRAINING, weights_only=False)
        if saved["tokens"] == vocabulary.tokens:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            done, best, stale = saved["epoch"], saved["best"], saved["stale"]
            print(f"resuming after epoch {done} (best validation loss {best:.4f})")
        else:
            print("the saved training state belongs to another vocabulary; training from the start")

    for epoch in range(done, MAX_EPOCHS):
        if stale >= PATIENCE:
            break
        train_loss = epoch_loss(model, vocabulary, train, random.Random(SEED + epoch), optimizer)
        loss = epoch_loss(model, vocabulary, validation, random.Random(0))
        print(f"epoch {epoch + 1:02d}: train {train_loss:.4f} validation {loss:.4f}", flush=True)
        if loss < best - MIN_IMPROVEMENT:
            best, stale = loss, 0
            save_atomically(
                {"state": model.state_dict(), "vocabulary_size": len(vocabulary)}, GENERATOR
            )
        else:
            stale += 1
        save_atomically(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best": best,
                "stale": stale,
                "tokens": vocabulary.tokens,
            },
            GENERATOR_TRAINING,
        )
    print(f"best validation loss {best:.4f}")


if __name__ == "__main__":
    main()
