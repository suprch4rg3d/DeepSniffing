"""What the generator (step 2) is worth.

    python -m scripts.core.evaluate                  the validation molecules
    python -m scripts.core.evaluate --split test     the test split, once and at the end
    python -m scripts.core.evaluate --split dream    the 468 DREAM molecules, the external benchmark
    python -m scripts.core.evaluate --tag NAME       a generator run that was trained with --tag NAME

**The score** is judged only by the data we have: no rule of ours and no model of ours decides what is
right. It is the probability that the model gives the real molecules of a held-out set when it is told
their own odor tokens (``conditioning``):

    loss_with_own_odor_tokens     the loss, in nats a molecule token, of writing a held-out molecule after a request
                            for its own odor tokens. Lower is better; ``perplexity`` is its exponential.
    gain_over_no_odor_tokens      the loss without the odor tokens minus the loss with them: how much the odor tokens add.
    gain_over_other_odor_tokens   the same against the odor tokens of another molecule that shares none of them.

It is a proper scoring rule and it never calls a molecule wrong. One odor token can describe countless
molecules, and a molecule that lacks a label is not a mistake, so nothing is counted as an error for
being absent from the list of an odor token. Two runs are compared molecule by molecule with ``compare``, which
is far more sensitive than their averages.

**Writing molecules for odor tokens** is what the model is for, so what it writes is also looked at. A
loss says how probable the model thinks a molecule is, and a sample shows what it does with that. What
comes out is counted by what the data and RDKit define:

    valid        a written molecule RDKit can read. SELFIES is built so that every string is a valid
                 molecule, so this should be about 1.0 -- it is here to catch a mistake, not as a result.
    unique       different molecules among the valid ones. A model can score a fine loss and write the
                 same memorised molecule every time; this is what catches that.
    novel        valid molecules in neither stage's training set -- the pretraining's millions count,
                 not only the 6,344 with odor tokens. This is the whole point of a generator:
                 a molecule nobody has made. A novel molecule that smells of nothing is worth nothing,
                 so novelty is read beside the two below, never alone.
    weight       what the written molecules weigh. The pretraining was filtered to 350 or less because
                 that is where molecules that smell live; this is the test of whether that carried
                 through to what the model writes.

**One diagnostic, off unless asked for**, because it is not a score:

    --ranking    ranks the held-out molecules by how much a request raises their probability and reports
                 the ROC-AUC of that, per odor token and for requests of several odor tokens. It counts every
                 molecule without the odor token as wrong, which is not so (see above), so it can only be read
                 as a rough guide and is not something to push up.

The loss is also reported as **perplexity**, the loss read as a number of choices: a perplexity of 3 means
the model is about as unsure as someone choosing between three equally good next molecule tokens.

The classifier (step 3) is measured by ``python -m scripts.core.classifier``, not here.
"""

from __future__ import annotations

import argparse
import json
import math
import random

import numpy as np
import pandas as pd
import selfies as sf
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.metrics import roc_auc_score

from scripts.core import data, generator, training
from scripts.core import model as model_module
from scripts.core.data import (
    batches,
    load_odor,
    odor_to_molecule,
    request_text,
)
from scripts.core.model import Decoder
from scripts.core.training import device_of, load_model, run_epoch, seed_everything
from scripts.core.vocabulary import Vocabulary
from scripts.preprocessing.odor_tables import DATASET_PATH, STRUCTURES_PATH

RDLogger.DisableLog("rdApp.*")

SAMPLES_PER_REQUEST = 200  # molecules written for each odor token asked about
TEMPERATURE = 1.0
# The most molecule tokens a molecule is given to be written in. Of the pretraining molecules, half have 29 and
# 99.99% have 143 or fewer, so 200 loses nothing real. It is a limit on memory, not on chemistry: the
# batch keeps going until every molecule in it has ended, and one that never does would take it to the
# model's 1,024 positions, where the attention mask of 200 molecules alone is 6.7 GB (more than the GPU
# has, and the rest spills into the memory of the machine). One that has not ended by the limit was
# not finished, so it is counted as not written and is not cut off and read as a molecule.
MAX_NEW_MOLECULE_TOKENS = 200
REQUESTS = (  # a few odor tokens on their own, and the two carvones' odor tokens
    ("fruity",),
    ("woody",),
    ("musk",),
    ("sulfurous",),
    ("minty", "spearmint"),
    ("minty", "caraway"),
    (),  # nothing asked for, so what the model writes when left alone
)


def canonical(smiles: str) -> str | None:
    """The canonical SMILES, or None. An empty molecule is not a molecule: RDKit reads "" quite
    happily and gives "" back, and SELFIES decodes anything it cannot make sense of to "", so without
    this a model that writes nothing at all would be counted as writing valid molecules."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(molecule)


def molecule_of(selfies: str) -> str | None:
    """The canonical SMILES of a written molecule, or None if it is not one after all."""
    try:
        return canonical(sf.decoder(selfies))
    except Exception:
        return None


def guided(asked: torch.Tensor, bare: torch.Tensor, strength: float) -> torch.Tensor:
    """Classifier-free guidance: the log-probabilities with the request, pushed away from those without.

    ``strength`` 1 is the model as it is, 0 is what it writes when nothing is asked, and above 1 goes
    further in the direction the request moves it. The model has been trained with the request left out
    one time in ten, so it has both predictions to give. The mix is taken in log-probabilities, which is
    the usual form, and the caller renormalises it over the molecule tokens it allows.
    """
    asked, bare = torch.log_softmax(asked.float(), dim=-1), torch.log_softmax(bare.float(), dim=-1)
    return bare + strength * (asked - bare)


AT_A_TIME = 100  # molecules written together: more than this and a guided request fills the GPU


def write_molecules(
    model: Decoder,
    vocabulary: Vocabulary,
    odor_tokens: tuple[str, ...],
    count: int = SAMPLES_PER_REQUEST,
    temperature: float = TEMPERATURE,
    guidance: float = 1.0,
) -> list[str | None]:
    """``count`` molecules written for ``odor tokens``, in groups of ``AT_A_TIME`` so the memory stays bounded.

    The molecules are independent, so writing them in groups gives the same kind of sample as writing
    them all together. See ``write_group`` for what a group does.
    """
    written: list[str | None] = []
    for begin in range(0, count, AT_A_TIME):
        written += write_group(
            model, vocabulary, odor_tokens, min(AT_A_TIME, count - begin), temperature, guidance
        )
    return written


@torch.no_grad()
def write_group(
    model: Decoder,
    vocabulary: Vocabulary,
    odor_tokens: tuple[str, ...],
    count: int,
    temperature: float = TEMPERATURE,
    guidance: float = 1.0,
) -> list[str | None]:
    """``count`` molecules written for ``odor tokens``, as canonical SMILES; None where it is not a molecule.

    With a ``guidance`` other than 1 the request is made to count for more (or less) than the model gives
    it on its own, at the price of a second pass per molecule token (see ``guided``). It does nothing when no
    odor tokens are asked for.
    """
    model.eval()
    device = device_of(model)
    start = [vocabulary.special("<bos>"), vocabulary.special("<odor>")]
    prompt = [*start, *vocabulary.odor_ids(list(odor_tokens)), vocabulary.special("<sep>")]
    ids = torch.tensor([prompt] * count, device=device)
    steered = guidance != 1.0 and bool(odor_tokens)
    if steered:
        bare = torch.tensor([[*start, vocabulary.special("<sep>")]] * count, device=device)
    eos, pad = vocabulary.special("<eos>"), vocabulary.special("<pad>")
    done = torch.zeros(count, dtype=torch.bool, device=device)
    for _ in range(min(MAX_NEW_MOLECULE_TOKENS, model_module.MAX_POSITIONS - len(prompt))):
        # only the last position is asked for: the output layer over every position of every molecule
        # is a table of 2,077 numbers each, and it is thrown away but for the last row
        logits = model.head(model.hidden(ids)[:, -1]) / temperature
        if steered:
            logits = guided(logits, model.head(model.hidden(bare)[:, -1]) / temperature, guidance)
        ending = logits[:, eos].clone()
        logits[:, : vocabulary.first_molecule] = -float(
            "inf"
        )  # a molecule token, or the end: nothing else
        logits[:, vocabulary.first_odor :] = -float("inf")  # never an odor token here
        logits[:, eos] = ending
        nxt = torch.multinomial(torch.softmax(logits.float(), dim=-1), 1).squeeze(1)
        nxt[done] = pad
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        if steered:
            bare = torch.cat([bare, nxt[:, None]], dim=1)
        done |= nxt == eos
        if bool(done.all()):
            break
    return decode(ids[:, len(prompt) :].tolist(), vocabulary)


def decode(rows: list[list[int]], vocabulary: Vocabulary) -> list[str | None]:
    """The molecules written as token rows: canonical SMILES, or None for one that is not a molecule.

    A row that never reached the end token was not finished and is None, whatever its molecule tokens would
    decode to: SELFIES reads any string as some molecule, so a text cut off in the middle would otherwise
    be counted as a valid one.
    """
    eos = vocabulary.special("<eos>")
    written = []
    for row in rows:
        if eos not in row:
            written.append(None)
            continue
        molecule_tokens = [vocabulary.tokens[token] for token in row[: row.index(eos)]]
        written.append(molecule_of("".join(molecule_tokens)) if molecule_tokens else None)
    return written


def chemistry(written: list[str | None], known: set[str]) -> dict:
    """The shares that are valid, different from each other and new, and what they weigh."""
    valid = [smiles for smiles in written if smiles]
    unique = sorted(set(valid))
    weights = [Descriptors.MolWt(Chem.MolFromSmiles(smiles)) for smiles in unique]
    return {
        "written": len(written),
        "valid": len(valid) / max(len(written), 1),
        "unique": len(unique) / max(len(valid), 1),
        "novel": sum(smiles not in known for smiles in unique) / max(len(unique), 1),
        "weight_median": float(np.median(weights)) if weights else None,
        "weight_over_350": float(np.mean([w > 350 for w in weights])) if weights else None,
        "examples": unique[:5],
    }


def request_loss(model: Decoder, vocabulary: Vocabulary, selfies: str, asked: list[str]) -> float:
    """The loss per molecule token of writing this molecule when the request names the odor tokens in ``asked``."""
    ids, start = request_text(vocabulary, selfies, asked)
    mask = torch.zeros(1, len(ids), dtype=torch.bool)
    mask[0, start:] = True
    return run_epoch(model, [(torch.tensor([ids]), mask)])


def conditioning(model: Decoder, vocabulary: Vocabulary, items: list, seed: int = 1) -> dict:
    """Whether the request changes how likely the model finds a molecule, which a sample cannot settle.

    Every molecule is scored three times with exact probabilities and nothing sampled: with a request for
    its own odor tokens, with an empty request, and with the odor tokens of another molecule that shares none of
    them. The gain is the loss without minus the loss with, so a positive gain means the request helps. If
    the model reads the request, the own odor tokens gain on both the empty request and the wrong odor tokens; if it
    does not, both gains are zero within their interval. The loss of writing a molecule says nothing of
    this by itself: a model that ignores every request can write molecules as well as one that does not.
    """
    rng = random.Random(seed)
    items = [(selfies, odor_tokens) for selfies, odor_tokens in items if odor_tokens]
    own, none = [], []
    wrong_losses: list[
        float | None
    ] = []  # aligned with the molecules; None where nobody shares no odor token
    for index, (selfies, odor_tokens) in enumerate(items):
        strangers = [
            w for j, (_, w) in enumerate(items) if j != index and not set(w) & set(odor_tokens)
        ]
        asked = rng.sample(odor_tokens, min(data.MAX_ODOR_TOKENS_IN_REQUEST, len(odor_tokens)))
        own.append(request_loss(model, vocabulary, selfies, asked))
        none.append(request_loss(model, vocabulary, selfies, []))
        if strangers:
            wrong = rng.choice(strangers)
            wrong = rng.sample(wrong, min(data.MAX_ODOR_TOKENS_IN_REQUEST, len(wrong)))
            wrong_losses.append(request_loss(model, vocabulary, selfies, wrong))
        else:
            wrong_losses.append(None)
    other = [(o, w) for o, w in zip(own, wrong_losses, strict=True) if w is not None]

    def gain(without: np.ndarray, with_: np.ndarray) -> dict:
        difference = without - with_
        return {
            "mean": float(difference.mean()),
            "interval_95": float(1.96 * difference.std() / np.sqrt(len(difference))),
            "helped": float((difference > 0).mean()),
        }

    per_molecule = {"own": list(own), "none": list(none), "other": wrong_losses}
    own, none = np.array(own), np.array(none)
    result = {
        "molecules": len(own),
        "loss_with_own_odor_tokens": float(own.mean()),
        "perplexity_with_own_odor_tokens": float(np.exp(own.mean())),
        "loss_with_no_odor_tokens": float(none.mean()),
        "gain_over_no_odor_tokens": gain(none, own),
        # kept so that two runs can be compared molecule by molecule (see ``compare``): the requests are
        # drawn from the same seed, so the same molecule gets the same request in every run
        "per_molecule": per_molecule,
    }
    if other:
        pairs = np.array(other)
        result["loss_with_other_odor_tokens"] = float(pairs[:, 1].mean())
        result["gain_over_other_odor_tokens"] = gain(pairs[:, 1], pairs[:, 0])
    return result


@torch.no_grad()
def guidance_curve(
    model: Decoder,
    vocabulary: Vocabulary,
    items: list,
    strengths: tuple[float, ...] = (0.0, 1.0, 1.25, 1.5, 2.0, 3.0),
    seed: int = 1,
) -> dict:
    """How likely the real held-out molecules are under guided writing, at each guidance strength.

    Guidance (see ``guided``) turns the next-molecule token probabilities into new ones, and those define a
    distribution the model can be asked about like any other: with the true molecule written out molecule token by
    molecule token, how much probability does it give each next molecule token? That is the loss of writing the molecule,
    exact and with nothing sampled, for the data's own pairs of molecule and odor tokens. It is the same kind of
    score as ``conditioning``, so a strength can be chosen by it with no judge of ours: the best strength
    is the one that makes the real molecules most probable. Strength 1 is the model as it is and 0 is
    what it writes when nothing is asked, so the curve says whether pushing on the request helps.

    The probabilities are taken over what the writer may write (a molecule token or the end), as in sampling.
    """
    model.eval()
    device = device_of(model)
    rng = random.Random(seed)
    allowed = torch.zeros(len(vocabulary), dtype=torch.bool, device=device)
    allowed[vocabulary.first_molecule : vocabulary.first_odor] = True
    allowed[vocabulary.special("<eos>")] = True
    items = [(selfies, odor_tokens) for selfies, odor_tokens in items if odor_tokens]
    losses: dict[float, list[float]] = {strength: [] for strength in strengths}
    for selfies, odor_tokens in items:
        asked = rng.sample(odor_tokens, min(data.MAX_ODOR_TOKENS_IN_REQUEST, len(odor_tokens)))
        predictions = []
        for request in (asked, []):
            ids, start = request_text(vocabulary, selfies, request)
            x = torch.tensor([ids], device=device)
            logp = torch.log_softmax(model.head(model.hidden(x[:, :-1])).float(), dim=-1)[0]
            predictions.append((logp[start - 1 :], x[0, start:]))
        (with_request, target), (without, _) = predictions
        for strength in strengths:
            mixed = (without + strength * (with_request - without)).masked_fill(
                ~allowed, -float("inf")
            )
            taken = torch.log_softmax(mixed, dim=-1).gather(-1, target[:, None]).squeeze(-1)
            losses[strength].append(float(-taken.mean()))
    return {
        f"{strength:g}": {"loss": float(np.mean(v)), "perplexity": float(np.exp(np.mean(v)))}
        for strength, v in losses.items()
    }


@torch.no_grad()
def molecule_logprobs(
    model: Decoder, vocabulary: Vocabulary, molecules: list[str], asked: list[str], chunk: int = 128
) -> np.ndarray:
    """The log-probability, in nats, that the model writes each molecule when the request names ``asked``.

    The sum over the molecule's molecule tokens and its end, not the mean: it is the probability of the whole
    molecule. Nothing is sampled.
    """
    model.eval()
    device = device_of(model)
    texts = [request_text(vocabulary, selfies, asked) for selfies in molecules]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i][0]))
    pad = vocabulary.special("<pad>")
    out = np.zeros(len(texts))
    for begin in range(0, len(order), chunk):
        rows = order[begin : begin + chunk]
        longest = max(len(texts[i][0]) for i in rows)
        ids = torch.full((len(rows), longest), pad, dtype=torch.long)
        mask = torch.zeros((len(rows), longest), dtype=torch.bool)
        for row, i in enumerate(rows):
            tokens, start = texts[i]
            ids[row, : len(tokens)] = torch.tensor(tokens)
            mask[row, start : len(tokens)] = True
        ids, mask = ids.to(device), mask.to(device)
        logp = torch.log_softmax(model.head(model.hidden(ids[:, :-1])).float(), dim=-1)
        taken = logp.gather(-1, ids[:, 1:, None]).squeeze(-1) * mask[:, 1:]
        for row, i in enumerate(rows):
            out[i] = float(taken[row].sum())
    return out


def generator_ranking(model: Decoder, vocabulary: Vocabulary, items: list, least: int = 3) -> dict:
    """How well the generator, asked for one odor token, ranks the molecules that have that odor token first.

    For an odor token, every held-out molecule is scored by how much more likely the request makes it than no
    request: log p(molecule | odor token) - log p(molecule | nothing). By Bayes' rule that is also how much more
    likely the odor token is given the molecule, so ranking by it is a classifier built from the generator, and
    the molecules that really carry the odor token should rise to the top. The ROC-AUC of that ranking is taken
    odor token by odor token and averaged. Chance is 0.5, 1 is perfect, and it needs only the labels of the held-out
    molecules, no model that judges smell. It is the generating direction's counterpart of the macro
    ROC-AUC of naming the odor tokens, and the two can be set side by side.

    An odor token is scored when at least ``least`` molecules have it and ``least`` do not.
    """
    items = [(selfies, odor_tokens) for selfies, odor_tokens in items if odor_tokens]
    molecules = [selfies for selfies, _ in items]
    base = molecule_logprobs(model, vocabulary, molecules, [])
    candidates = sorted({odor_token for _, odor_tokens in items for odor_token in odor_tokens})
    aucs, counts = {}, {}
    for odor_token in candidates:
        truth = np.array([odor_token in odor_tokens for _, odor_tokens in items])
        if truth.sum() < least or (~truth).sum() < least:
            continue
        pmi = molecule_logprobs(model, vocabulary, molecules, [odor_token]) - base
        aucs[odor_token] = float(roc_auc_score(truth, pmi))
        counts[odor_token] = int(truth.sum())
    if not aucs:
        return {"odor_tokens_scored": 0}
    values = np.array(list(aucs.values()))
    frequent = np.array([counts[w] >= 10 for w in aucs])
    best = sorted(aucs, key=aucs.get, reverse=True)
    return {
        "odor_tokens_scored": len(aucs),
        "roc_auc_macro": float(values.mean()),
        "roc_auc_macro_odor_tokens_in_10_molecules": float(values[frequent].mean())
        if frequent.any()
        else None,
        "odor_tokens_above_0_8": float((values >= 0.8).mean()),
        "best": {w: round(aucs[w], 3) for w in best[:5]},
        "worst": {w: round(aucs[w], 3) for w in best[-5:]},
    }


def combination_ranking(
    model: Decoder,
    vocabulary: Vocabulary,
    items: list,
    train_items: list,
    sizes: tuple[int, ...] = (1, 2, 3, 4, 5),
    per_size: int = 100,
    seed: int = 0,
) -> dict:
    """The same ranking as ``generator_ranking``, for requests of several odor tokens, which is how it is used.

    A request of k odor tokens is drawn from the odor tokens of one held-out molecule, so that a correct answer
    exists. A held-out molecule is correct for it when it carries all k of them. The molecules are ranked by
    how much more likely the request makes them than no request, and what is reported, per size, is
    the ROC-AUC of that ranking, how often the best-ranked molecule is correct, and the share of the
    best five that are, averaged over the requests; with, for comparison, the share that would be
    correct if the molecules were taken at random.

    Requests are also told apart by whether the combination is *unseen*: no training molecule carries all
    its odor tokens. For these the model cannot have learned the combination as a whole and has to put the odor tokens
    together, which is what an arbitrary request asks of it. Their numbers are reported on their own.
    """
    items = [(selfies, odor_tokens) for selfies, odor_tokens in items if odor_tokens]
    molecules = [selfies for selfies, _ in items]
    base = molecule_logprobs(model, vocabulary, molecules, [])
    seen_sets = [set(odor_tokens) for _, odor_tokens in train_items if odor_tokens]
    rng = random.Random(seed)
    report = {}
    for size in sizes:
        pool = [i for i, (_, odor_tokens) in enumerate(items) if len(odor_tokens) >= size]
        rows, done = [], set()
        for i in rng.sample(pool, min(per_size, len(pool))):
            request = tuple(sorted(rng.sample(items[i][1], size)))
            if request in done:
                continue
            done.add(request)
            truth = np.array([set(request) <= set(odor_tokens) for _, odor_tokens in items])
            if truth.all() or not truth.any():
                continue  # nothing to tell apart
            score = molecule_logprobs(model, vocabulary, molecules, list(request)) - base
            best = np.argsort(-score)
            rows.append(
                {
                    "auc": float(roc_auc_score(truth, score)),
                    "top1": float(truth[best[0]]),
                    "top5": float(truth[best[:5]].mean()),
                    "chance": float(truth.mean()),
                    "unseen": not any(set(request) <= seen for seen in seen_sets),
                }
            )

        def summary(part: list[dict]) -> dict:
            if not part:
                return {"requests": 0}
            return {
                "requests": len(part),
                "roc_auc": float(np.mean([r["auc"] for r in part])),
                "best_is_correct": float(np.mean([r["top1"] for r in part])),
                "best_five_correct": float(np.mean([r["top5"] for r in part])),
                "at_random": float(np.mean([r["chance"] for r in part])),
            }

        report[str(size)] = {
            "all": summary(rows),
            "unseen_combinations": summary([r for r in rows if r["unseen"]]),
        }
    return report


def known_molecules(with_pretraining: bool = True) -> set[str]:
    """Every molecule the model could have learned by heart, which is what ``novel`` is measured against.

    Both stages count. The fine-tuning molecules are the obvious ones, but the pretraining saw millions
    more, and a molecule it was shown there is not new just because it has no odor tokens -- without
    them, ``novel`` would be flattered by every pretraining molecule the model gives back.
    """
    odor = pd.read_csv(DATASET_PATH)
    known = {
        smiles
        for smiles in odor.loc[odor["split"] == "train", "smiles"].map(canonical)
        if smiles is not None
    }
    if with_pretraining:
        for block in pd.read_csv(STRUCTURES_PATH, usecols=["smiles", "split"], chunksize=500_000):
            known.update(block.loc[block["split"] == "train", "smiles"])  # already canonical
    return known


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the odor decoder.")
    parser.add_argument("--split", choices=("validation", "test", "dream"), default="validation")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_REQUEST)
    parser.add_argument("--size", choices=tuple(model_module.SIZES), default="base")
    parser.add_argument(
        "--tag", default=None, help="a generator run kept apart with --tag when trained"
    )
    parser.add_argument(
        "--no-sampling", action="store_true", help="skip writing molecules: the exact numbers only"
    )
    parser.add_argument(
        "--ranking",
        action="store_true",
        help="also the ranking ROC-AUCs, a diagnostic only: they call every molecule without the odor token wrong",
    )
    parser.add_argument(
        "--guidance",
        action="store_true",
        help="also the likelihood of the real molecules at several guidance strengths",
    )
    arguments = parser.parse_args(argv)
    model_module.use_size(arguments.size)
    seed_everything()

    vocabulary = Vocabulary.load()
    model = load_model(vocabulary, generator.folder(arguments.tag) / "best.pt")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    items = load_odor(arguments.split)
    report: dict = {"split": arguments.split, "molecules": len(items)}

    # the loss of writing the molecules after a request for some of their odor tokens, and its perplexity
    loss = run_epoch(model, batches(vocabulary, items, [odor_to_molecule], random.Random(0), False))
    report["odor_to_molecule"] = {"loss": loss, "perplexity": math.exp(loss)}

    # The score: the probability the model gives the real molecules when it is told their own odor tokens, and
    # how much the odor tokens add to it. Nothing here calls any molecule wrong: one odor token can describe
    # countless molecules, and a molecule that lacks a label is not a mistake.
    report["conditioning"] = conditioning(model, vocabulary, items)
    if arguments.guidance:
        report["guidance_curve"] = guidance_curve(model, vocabulary, items)

    # writing molecules: what the model is for, judged only by what the data itself defines
    if not arguments.no_sampling:
        if not STRUCTURES_PATH.is_file():
            raise SystemExit(
                f"{STRUCTURES_PATH} is missing: novel is measured against the pretraining molecules. "
                "Run ds-build, or pass --no-sampling for the exact numbers only."
            )
        known = known_molecules()
        report["generation"] = {}
        for odor_tokens in REQUESTS:
            written = write_molecules(model, vocabulary, odor_tokens, arguments.samples)
            report["generation"][" + ".join(odor_tokens) or "(nothing asked for)"] = chemistry(
                written, known
            )

    if arguments.ranking:
        report["generator_ranking"] = generator_ranking(model, vocabulary, items)
        train_items = [row for row in load_odor("train") if row[1]]
        report["combinations"] = combination_ranking(model, vocabulary, items, train_items)

    suffix = f"_{arguments.tag}" if arguments.tag else ""
    path = training.OUTPUT_DIR / f"evaluation_{arguments.split}{suffix}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
