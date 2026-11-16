# Odor generator: from odor words to molecules

You give it odor words, for example `fruity green`, and it writes new molecules that should smell that way. It is a
side experiment. The DREAM models do not use it, and nothing in the project imports it. Only `generation/student.py`
reads the project, to get the DREAM ratings.

## The approach

There are three small models, and the approach is simple.

```
 odor words ──► GENERATOR ──► candidate molecules ──┬──► ORACLE   probability of each odor word
 (fruity green)  a GPT-style                        │                (learned from word lists)
                 decoder, writes                    └──► STUDENT  the 21 DREAM mean ratings
                 the molecule one symbol                              (learned from DREAM ratings)
                 at a time                          the judges never see the generator;
                                                    a molecule ranks high only if both are satisfied
```

- **The generator** is a decoder-only transformer, the same kind of model as GPT. It is trained to predict the next
  symbol of a text that starts with the odor words and continues with the molecule. That is what makes it
  generative: to write a molecule it reads the words, then picks one symbol after another until it writes the end
  symbol.
- **The oracle** is a small fully connected network that reads any molecule and gives the probability of each odor
  word.
- **The student** is a small fully connected network that reads any molecule and predicts the 21 DREAM mean
  ratings. It is a light stand-in for Q2: the full Q2 needs Dragon, EPI Suite and NSPDK tables that exist only for
  the 476 DREAM molecules, so it cannot score a molecule that the generator has just invented. The student reads
  only what RDKit computes for any structure. It shares no weights with Q2.

## The text the generator learns

```
 <bos> <fruity> <green> <sep> [C][C][O][C][Branch1][C][C][=O] <eos>
        └ the odor words ┘      └ the molecule in SELFIES ┘
```

The molecule is written in SELFIES, a text form in which every sequence of symbols decodes to a valid molecule,
and it is turned into a SMILES string (`CCOC(C)=O`) afterwards. During training each labeled molecule is shown with
a random subset of its own words (1 to 6), and a tenth of the time with none, so a request can name any
combination of words or none. The loss is on the molecule and the end symbol only: the words are an instruction to
follow, not something to predict.

## The architecture

```
 ids of the text (words, <sep>, symbols so far)
   │
   │  token embedding (135 tokens -> 256)  +  learned position embedding (-> 256)
   v
 6 identical layers, each:
     LayerNorm -> causal self-attention, 8 heads (a position sees only earlier positions) -> add
     LayerNorm -> Linear(256 -> 1024) -> ReLU -> Linear(1024 -> 256)                    -> add
     dropout 0.1
   v
 LayerNorm(256) -> Linear(256 -> 135)  ─► a score for every possible next symbol
```

4,831,111 parameters. The oracle is `Linear -> 512 -> 256 -> 101 words` with ReLU and dropout 0.4 on Morgan counts
and RDKit descriptors. The student is `Linear -> 512 -> 256 -> 21 ratings` with ReLU and dropout 0.3 on the same
features.

**Training the generator.** AdamW, learning rate 5e-4, weight decay 0.01, batch 128, gradient clipping at 1. It
stops when the validation loss has not improved for 6 epochs, and ran 30 epochs; the best validation loss is 0.826
per symbol. It saves its state after every epoch, so an interrupted run loses at most one epoch and resumes by
itself.

**Writing a molecule.** The words are placed in front, then symbols are drawn one at a time from the model's
probabilities (temperature 1) until the end symbol or the length limit.

## Data

All from the Pyrfume repository (pinned commit `8054ea98`): the molecule registry (about 10,000 odor and flavor
molecules, no odor words), Leffingwell (3,522 molecules, 113 odor words) and The Good Scents Company (4,626
molecules with descriptor text, mapped to the same words). Molecules are written without stereochemistry. A
molecule is dropped if it is a mixture or a salt, an isotope-labelled copy, has an element outside H C N O F P S Cl
Br I, has fewer than 3 or more than 40 heavy atoms, or uses a SELFIES symbol that fewer than 3 molecules have.
Words that fewer than 30 training molecules have are dropped, which leaves 101.

That gives 8,021 molecules: 7,120 for training, 215 for validation, 215 for testing, and **471 DREAM molecules that
are held out completely**. Neither the generator nor the oracle learns from the DREAM molecules, and they are the
hardest test of whether the generator follows a request. The molecules without odor words are all training
molecules and teach chemistry.

## Results

**The judges.**

| judge | learns from | test result |
|---|---|---|
| oracle | 3,877 labeled molecules | mean AUC 0.867 on the test split, 0.876 on the DREAM molecules |
| student | the 338 DREAM training molecules | Z-score 5.92 intensity, 5.19 pleasantness, 20.21 descriptors (means only); the full Q2 scores 6.69, 5.45 and 21.25 |

**The generator.** For each request, 400 molecules were written and judged.

- Without a request every molecule is valid, 97% are different from each other, and 78% are new, meaning not in
  the training data.
- It follows requests whose odor has a clear chemical signature. Share of the written molecules that contain the
  expected group, against the share when nothing is asked:

| request | with the request | without | oracle probability, with / without |
|---|---:|---:|---|
| garlic, onion | 98% sulfur | 5% | 0.52 / 0.01 |
| alliaceous | 94% sulfur | 5% | 0.31 / 0.01 |
| sulfurous | 84% sulfur | 5% | 0.62 / 0.03 |
| fishy | 57% nitrogen | 26% | 0.21 / 0.03 |
| fruity | 52% ester | 23% | 0.51 / 0.22 |
| buttery | 21% ketone | 9% | 0.06 / 0.01 |

- On molecules it never saw, it comes closer to the real molecule when it is given that molecule's own words. For
  80 DREAM molecules, with 100 molecules written for each, the best one had a similarity of 0.56 to the real molecule
  with its own words and 0.35 with other words, and 51% of the DREAM molecules got a close match (similarity above 0.5)
  against 14% with other words.
- It writes words with few training molecules (mint, citrus, buttery) poorly: the oracle probability is 0.06 to 0.11.

All 10 correctness checks pass (`results/checks.log`): the SELFIES round trip is lossless for all 8,021 molecules,
the attention is causal, padding does not change the predictions, and every written sequence ends and decodes to a
valid molecule.

## Files

```
generation/            every module has a main() and runs as  python -m generation.<module>
  paths.py             every file location, in one place
  chemistry.py         RDKit and SELFIES helpers, and the rule that cleans a molecule
  dataset.py           the molecules and their odor words, and the splits
  vocabulary.py        the tokens of the generator
  generator.py         the generator: the model, its training and its sampling
  oracle.py            judge 1
  student.py           judge 2
  judge.py             asks the generator for molecules for an odor and ranks them with both judges
  evaluate.py          does the generator follow a request? on molecules it never saw
  checks.py            correctness checks of the sequences and of the generator
data/                  raw Pyrfume files (data/raw), the dataset, the vocabulary and the trained models
results/               the numbers and logs of the last run
```

## Run it (from this folder)

```bash
poetry run python -m generation.dataset      # builds data/dataset.csv from data/raw
poetry run python -m generation.student      # judge 2
poetry run python -m generation.oracle       # judge 1
poetry run python -m generation.generator    # the generator; resumes if interrupted
poetry run python -m generation.checks       # correctness checks
poetry run python -m generation.evaluate     # how well it follows a request
poetry run python -m generation.judge --words fruity green --pleasant 70 --top 10
```

`judge` prints, for each candidate, its score (the mean of the parts that were asked for), the oracle probability of
each requested word, the student's prediction of pleasantness and intensity, and simple features such as heavy
atoms, molecular weight, logP and the functional groups. For example, `--words fruity green` returns esters such as
`CCCC=CCC(=O)OC` (fruity 0.98, green 0.86, new).

## Plugging it into the project

1. Copy `generation/` to `scripts/generation/`, and move the data (`data/dataset.csv`, `words.json`,
   `vocabulary.json`, `generator.pt`, `oracle.pt`, `student.pt`) to the folder that `paths.ROOT` points to.
2. Change `ROOT` and `PROJECT_ROOT` in `paths.py`. Nothing else holds a path.
3. Imports inside the package are relative, so they keep working. `student.py` already imports from `scripts.core`.
4. Add a console script `ds-generate = "scripts.generation.judge:main"` to `pyproject.toml`, and move the checks into
   `tests/`.
5. Add `selfies` to the dependencies (`poetry add selfies`). It was installed with `pip` here and is not in the lock
   file.

## Limits

- The judges are predictions. Nobody can say whether a molecule smells as predicted without people to smell it, and
  ease of synthesis and safety are not checked. The oracle and the generator learn from the same word lists, so
  their agreement shows consistency with those lists, not a human check.
- With about 7,000 training molecules the generator writes odorant-like molecules, and many of them are very small
  (the median is 17 symbols), including trivial ones such as `CC` or `CS` for some requests.
- The random split puts close relatives of a test molecule in training. The DREAM molecules are the fairer test.
