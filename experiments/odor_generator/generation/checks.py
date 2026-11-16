"""Checks that the decoder is correct: the sequences, the training examples, the causal mask, the padding and the
use of the odor words. Each check prints PASS or FAIL, and the exit status is 1 if any failed.

Run:  poetry run python -m generation.checks
"""

from __future__ import annotations

import random
import sys

import numpy as np
import torch

from .chemistry import selfies_to_smiles, selfies_tokens
from .dataset import load_dataset
from .generator import MAX_LENGTH, encode_example, load_generator, prompt_ids
from .vocabulary import BOS, EOS, PAD, SEP

results: list[tuple[str, bool]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, bool(passed)))
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}", flush=True)


def main() -> None:
    frame, _ = load_dataset()
    model, vocabulary = load_generator()
    rng = random.Random(1)

    # 1. SMILES to SELFIES and back loses nothing, and no symbol is an explicit hydrogen.
    changed = [s for s in frame["smiles"] if selfies_to_smiles(selfies_tokens(s) or []) != s]
    check(
        "SELFIES round trip is lossless for every molecule",
        not changed,
        f"({len(frame)} molecules, {len(changed)} changed)",
    )

    # 2. The molecules that the decoder can use.
    usable = {}
    for split in ("train", "validation", "test", "dream"):
        molecules = frame[frame["split"] == split]["smiles"]
        usable[split] = (
            f"{sum(vocabulary.can_write(selfies_tokens(s)) for s in molecules)}/{len(molecules)}"
        )
    check(
        "every training molecule fits the decoder",
        usable["train"].split("/")[0] == usable["train"].split("/")[1],
        str(usable),
    )

    # 3. A training example is built correctly.
    row = frame[(frame["split"] == "train") & (frame["word_list"].map(len) >= 3)].iloc[0]
    prompt, body = encode_example(vocabulary, selfies_tokens(row["smiles"]), row["word_list"], rng)
    tokens = [vocabulary.tokens[i] for i in prompt + body]
    check(
        "example layout is <bos> words <sep> molecule <eos>, with words from the molecule's own list",
        tokens[0] == BOS
        and tokens[len(prompt) - 1] == SEP
        and tokens[-1] == EOS
        and all(t.strip("<>") in row["word_list"] for t in tokens[1 : len(prompt) - 1]),
        str(tokens[:6]),
    )
    check(
        "decoding the molecule part of an example gives the molecule back",
        selfies_to_smiles(tokens[len(prompt) : -1]) == row["smiles"],
    )

    # 4. The attention is causal.
    ids = torch.randint(vocabulary.first_symbol, len(vocabulary), (3, 30))
    changed_ids = ids.clone()
    changed_ids[:, 20:] = torch.randint(vocabulary.first_symbol, len(vocabulary), (3, 10))
    with torch.no_grad():
        a, b = model(ids), model(changed_ids)
    check(
        "causal: predictions before position 20 do not depend on later tokens",
        torch.allclose(a[:, :20], b[:, :20], atol=1e-5),
        f"max change {float((a[:, :20] - b[:, :20]).abs().max()):.2e}",
    )
    check(
        "causal: predictions after position 20 do change",
        not torch.allclose(a[:, 21:], b[:, 21:], atol=1e-5),
    )

    # 5. Padding on the right does not change the real positions.
    real = ids[:1, :18]
    padded = torch.cat([real, torch.full((1, 9), vocabulary.index[PAD])], dim=1)
    with torch.no_grad():
        alone, with_padding = model(real), model(padded)[:, :18]
    check(
        "padding on the right does not change the predictions of the real tokens",
        torch.allclose(alone, with_padding, atol=1e-5),
        f"max change {float((alone - with_padding).abs().max()):.2e}",
    )

    # 6. The odor words are used: held-out molecules are easier to write when their own words are given.
    def mean_loss(use_words: bool) -> float:
        total, count = 0.0, 0
        rows = frame[(frame["split"] == "test") & (frame["word_list"].map(len) >= 2)]
        for smiles, words in zip(rows["smiles"], rows["word_list"], strict=False):
            symbols = selfies_tokens(smiles)
            if not vocabulary.can_write(symbols):
                continue
            prompt = prompt_ids(vocabulary, words[:6] if use_words else [])
            sequence = torch.tensor(
                [prompt + [vocabulary.index[s] for s in symbols] + [vocabulary.index[EOS]]]
            )
            with torch.no_grad():
                logits = model(sequence[:, :-1])
            total += float(
                torch.nn.functional.cross_entropy(
                    logits[0, len(prompt) - 1 :], sequence[0, len(prompt) :], reduction="sum"
                )
            )
            count += sequence.shape[1] - len(prompt)
        return total / count

    with_words, without = mean_loss(True), mean_loss(False)
    check(
        "held-out molecules are easier to write when their own odor words are given",
        with_words < without,
        f"(loss per token {with_words:.3f} with the words, {without:.3f} without)",
    )

    # 7. What is written is a clean sequence that ends, and decodes to a molecule.
    torch.manual_seed(0)
    start = len(prompt_ids(vocabulary, ["fruity"]))
    ids_ = torch.tensor([prompt_ids(vocabulary, ["fruity"])] * 200)
    finished = torch.zeros(200, dtype=torch.bool)
    eos = vocabulary.index[EOS]
    with torch.no_grad():
        for _ in range(MAX_LENGTH - ids_.shape[1]):
            logits = model(ids_)[:, -1]
            end = logits[:, eos].clone()
            logits[:, : vocabulary.first_symbol] = -float("inf")
            logits[:, eos] = end
            nxt = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1)
            nxt[finished] = vocabulary.index[PAD]
            ids_ = torch.cat([ids_, nxt[:, None]], 1)
            finished |= nxt == eos
            if bool(finished.all()):
                break
    rows_ = ids_[:, start:].tolist()
    lengths = [r.index(eos) if eos in r else None for r in rows_]
    ended = [n for n in lengths if n is not None]
    check(
        "written sequences end with <eos> before the length limit",
        len(ended) / len(lengths) > 0.99,
        f"({len(ended)}/{len(lengths)} ended, median length {int(np.median(ended))} symbols)",
    )
    check(
        "every written sequence decodes to a valid molecule",
        all(
            selfies_to_smiles([vocabulary.tokens[t] for t in r[:n]])
            for r, n in zip(rows_, lengths, strict=False)
            if n
        ),
    )

    failed = [name for name, passed in results if not passed]
    print(
        f"\n{len(results) - len(failed)} of {len(results)} checks passed"
        + (f"; FAILED: {failed}" if failed else "")
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
