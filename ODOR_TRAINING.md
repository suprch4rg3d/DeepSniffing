# The odor decoder: data, training, evaluation and what was found

The decoder (`scripts/core/`) is a small GPT over molecule tokens and odor tokens. **Its job is to write a
molecule for a set of odor tokens**: given `fruity` and `green`, or any other odor tokens, it writes a molecule one
molecule token at a time. It is built in three steps that each stand alone: the **pretraining** on molecules alone,
the **generator** (odor tokens → molecule), which is the goal, and the **classifier** (molecule → odor tokens).

## The data

| step | command | what it makes |
|---|---|---|
| odor tokens | `odor_tokens` | the odor tokens of 7,516 molecules from the odor sources: `molecules.csv`, `odor_tokens.csv`, `dropped.csv` |
| PubChem | `pubchem_structures` | `pubchem_structures.csv`: 3.4 million molecules for the pretraining, see below |
| tables | `odor_tables` | `structures.csv` (pretraining), `odor_dataset.csv` (fine-tuning), `vocabulary.json` |
| tokens | `molecule_tokens` | `stage1_tokens.npy` and its companions: the pretraining molecules as token numbers |

`ds-build` runs all four in order. Every step is skipped if what it makes is already there, so an interrupted
build goes on from where it stopped.

**Where the pretraining molecules come from.** `pubchem_structures` downloads PubChem's `CID-SMILES.gz` (1.5 GB,
124,863,650 compounds) and `CID-Mass.gz` (1.3 GB), unpacks both once, and then works on them in pieces of 64 MB,
one piece to a core:

1. the weight mask: one bit per CID, true where the compound weighs 350 or less (58,854,892 of them). PubChem
   publishes the weights, so no chemistry is needed to find them;
2. `pubchem_light.csv`: the `cid,smiles` of those compounds, 2.5 GB, with no chemistry done to them yet;
3. the conversion to canonical SMILES and SELFIES of 3.4 million of them (`--target`, which is the only slow
   part: some 0.6 ms a molecule). They are chosen by a checksum of the CID and not by taking every n-th one,
   because CIDs are handed out in the order compounds were registered and a batch of analogues sits on consecutive
   numbers. The same molecules are taken every time.

*Why 350.* A molecule has to evaporate to be smelled. Of the 7,516 molecules with odor tokens, 98.5% weigh 350 or
less and the median is 174; PubChem's median is 388. Nothing is left out for the elements it is made of: the odor
molecules themselves hold sodium, zinc and boron.

**Leakage.** The 352 validation, 352 test and 468 DREAM molecules are kept out of the pretraining by CID *and* by
structure (PubChem gives one molecule more than one CID). The 5,017 validation molecules of the pretraining are
chosen by a checksum of the CID, so they are the same in every build and a loss from one run can be set beside a
loss from another.

**What is in git.** Code, and the small files that steps 2 and 3 start from or that a rebuild could not give again:
`data/processed/odor/vocabulary.json` (the token numbers a saved model is tied to, so a model cannot be loaded without
exactly its token list), `data/processed/odor/odor_dataset.csv` (the fine-tuning table and its split),
`data/raw/pubchem/odor_pubchem.csv` (the cache of PubChem answers, so a rerun needs no network) and the pretrained
decoder `output/odor/stage1/best.pt` (23 MB). With those, the classifier and the generator train and are measured
without any download. The generator's replay reads the packed pretraining tokens, and `evaluate` reads
`structures.csv` to count what is novel: both are over GitHub's 100 MB limit and are made again by `ds-build`
(`--replay 0` and `--no-sampling` do without them). Everything else is made by `ds-prepare` (the odor sources, from a
pinned commit) and `ds-build` (the PubChem download and every table, the packed tokens and the intermediates) and is
in `.gitignore`, as are the other trained models in `output/`. The earlier odor profile model and DREAM code are on the release of
`archive/structure-to-odor-profile`.

## Training: three steps that each stand alone

| step | module | command | learns | folder | about |
|---|---|---|---|---|---|
| 1. pretraining | `pretrain.py` | `--stage 1` | to write molecules (no odor tokens) | `output/odor/stage1` | 3 hours |
| 2. generator | `generator.py` | `--stage 2` | odor tokens → molecule | `output/odor/stage2` | minutes |
| 3. classifier | `classifier.py` | `--stage 3` | molecule → odor tokens | `output/odor/stage3` | a minute |

```
poetry install
python -m scripts.core --stage 1       # the pretraining: 24 epochs on an RTX 3060
python -m scripts.core --stage 2       # the generator, with no options
python -m scripts.core --stage 3       # the classifier
python -m scripts.core                 # all three, one after the other
```

Steps 2 and 3 both start from the pretrained decoder of step 1 and **do not depend on each other**: the generator
trains on the loss of writing the molecule and nothing else, and the classifier trains only a head on the frozen
decoder and cannot change one weight of it. Each step has its own folder, its own `history.csv` and its own scores,
so no loss of one is in another. The shared engine (one epoch, the loop, saving and loading) is `training.py`;
`train.py` is only the command line. The numbers of a run are drawn by `plots.py` and tabled by `report.py`
(see *How a model is judged*).

A step saves its state after every epoch and starts again from it when the same command is run, so an interrupted
run loses at most one epoch; the copies kept against a crash (`*.prev`) are deleted when a step finishes. Logs go to
`output/odor/train.log`. **A plain `--stage 2` trains the best generator found**: every default is the best setting.

### Step 1, the pretraining

3,422,276 molecules (100.5 million tokens) as `<bos> molecule tokens <eos>`, the loss on every molecule token, 24
epochs of a cosine schedule. The vocabulary is 2,077 tokens (6 special, 1,181 molecule tokens, 890 odor tokens; the
odor tokens are only used from step 2 on). Best validation loss **0.7858** (perplexity 2.19) at epoch 22; the last
epochs are flat, so this model size has reached what the data can give. No augmentation is used: with millions of
real molecules there is nothing to learn by heart. **Nothing but a `--stage 1` run writes to `stage1`**; steps 2 and
3 only read `stage1/best.pt`.

### Step 2, the generator

Fine-tunes the pretrained decoder on the 6,344 molecules with odor tokens, which it sees as requests, `<odor> odor
token odor token <sep>` followed by the molecule. The request is a set: its odor tokens have no position and cannot
see each other, so their order cannot matter (a test checks it). **The loss is on the molecule only.** What makes
the best model, all of them defaults:

| setting | what it does |
|---|---|
| **request adapters** (in the model) | The odor tokens are averaged and the average scales and shifts the state at every layer, for every molecule token of the molecule. The weights start at zero, so the pretrained model is unchanged at the start. Without them the model hardly used its request. |
| **`--full-rate-embeddings`** | The odor tokens are the one part that is new; the layer-wise decay would train them slowest. |
| **replay** (`--replay`) | As many pretraining molecules as there are molecules with odor tokens are mixed into each epoch after an empty request, as in LLM fine-tuning. |
| dropout 0.2, layer decay 0.8, peak rate 3e-4, **160 epochs** | Each is at its best: the other direction of each was worse. An epoch is the molecules once, so 80 epochs gave 0.7290 and 160 gave 0.7057 (see *What was found*). |

Options change one thing from that: `--tag NAME` keeps a run in `output/odor/stage2_NAME`, `--seed`, `--ema`,
`--dropout`, `--layer-decay`, `--lr`, `--replay 0`, `--no-full-rate-embeddings`.

### Step 3, the classifier

The pretrained decoder is run once over the molecules, `<bos> molecule tokens <eos>` exactly as in its
pretraining, and two things are kept for each: the state at the last molecule token (which has read the whole
molecule) and the mean of the states of all of them. A small head (a hidden layer of 512, dropout 0.3) is trained on
those to score each of the 890 odor tokens as yes or no, with the binary cross entropy per molecule summed over the
odor tokens. The epoch is chosen by the validation loss and the head is scored by the macro ROC-AUC. Only the head
learns. It is built from the frozen decoder because that is cheap, cannot hurt the generator, and needs no balance
between two losses.

## How a model is judged

Only by the data we have. No rule of ours and no model of ours says what is right, and a molecule that lacks a
label is not wrong: one odor token can describe countless molecules.

**The generator** is scored by the probability it gives the *real* molecules of a held-out set when it is told
their own odor tokens:

```
python -m scripts.core.evaluate                 # validation
python -m scripts.core.evaluate --split test    # once, at the end
python -m scripts.core.evaluate --split dream   # the 468 DREAM molecules
python -m scripts.core.compare REFERENCE OTHER  # two runs, molecule by molecule
```

- **`loss_with_own_odor_tokens`**: nats per molecule token of writing a held-out molecule after a request for its
  own odor tokens (lower is better; its exponential is the perplexity).
- **`gain_over_no_odor_tokens`** and **`gain_over_other_odor_tokens`**: how much the odor tokens add, against an
  empty request and against another molecule's odor tokens. A model that ignores its request has a gain of zero; the
  first version of the fine-tuning had +0.002, which is how it was found out.
- What it writes is counted by what the data and RDKit define: **valid**, **unique**, **novel** (in neither step's
  training set) and **weight**. Writing stops at 200 molecule tokens (99.99% of molecules have 143 or fewer), and a
  molecule that has not ended by then is counted as not written.
- `compare` takes the difference between two runs for each molecule on the same requests and gives its 95%
  interval. **It covers only the noise of the 352 molecules.** The noise of the training itself is about **0.004**
  (the same settings retrained with other seeds moved by that much).
- Off unless asked for: `--ranking` (ROC-AUC of ranking the molecules that carry an odor token, which counts every
  molecule without it as wrong and so is a rough guide only) and `--guidance`.

**The classifier** is scored by the macro ROC-AUC over the odor tokens that can be scored (chance is 0.5), the
average precision, and the share of its best-scored odor tokens that are true:

```
python -m scripts.core.classifier --split test
```

**Every step's numbers over its epochs** are in `history.csv`: the loss, and the share of molecule tokens the model
ranked first (`*_accuracy`) in steps 1 and 2, the request gain in step 2, and the ROC-AUC in step 3.

```
python -m scripts.core.plots --stage 1|2|3      # the curves, in <folder>/plots
python -m scripts.core.report                   # results.md and the figures of the evaluations, in output/odor/report
```

Each file does one job: `metrics.py` the numbers (accuracy, perplexity, ROC-AUC), `history.py` the record of a
run, `plots.py` every figure, `report.py` the table of results.

## What was found

**The generator**, loss of writing a held-out validation molecule after a request for its own odor tokens (from
runs that trained the generator together with the classifier on one decoder, which is why they are listed as they
were; the weight of the classifier's loss in them was 0.05):

| run | loss | change |
|---|---|---|
| the first fine-tuning (no adapters) | 0.8443 | |
| + request adapters | 0.7304 | −0.114 |
| + odor tokens at the full learning rate | 0.7218 | −0.009 (real) |
| + replay | 0.7116 | −0.010 (real) |

Tried and no better (each one change from the best run of its time, and each worse by more than the noise unless
said): dropout 0.1 and 0.3, peak rate 1e-3 and 2e-4, no layer decay and layer decay 0.6, replay of 20,000, a moving
average of the weights (within the noise: it steadies training and scores the same). **Guidance** at a strength
above 1 makes the real molecules less probable (1: 0.7118, 1.25: 0.7184, 1.5: 0.7393, 2: 0.8171), so the model is
already right about its request and guidance is left off.

**Splitting the classifier out** (this version) first scored 0.7290 on validation, worse than the 0.7126 it came from,
apparently because an epoch no longer held the classifier's batches and so had about half the steps. With 160 epochs it scores
**0.7057**, lower than both by more than the noise (paired against the 80-epoch run: −0.0233 ± 0.0080).

**The final models**, settings fixed on validation. The numbers are from the Colab run recorded in `RUN.txt`:
commit `e33cda6`, an L4, torch 2.14.1, rdkit 2026.9.1. An earlier run, before the fix that removed the twelve
DREAM molecules with no odor tokens from the pretraining, gave 0.7062, 0.7096 and 0.6759; that is what the
comparisons above refer to, and only the validation, test and classifier numbers are comparable with it.

| held-out set | molecules | loss | perplexity | gain over no odor tokens | odor tokens help |
|---|---|---|---|---|---|
| validation | 352 | 0.7112 | 2.04 | +0.218 ± 0.019 | 90% |
| **test** | 352 | **0.7124** | 2.04 | **+0.212 ± 0.020** | 90% |
| **DREAM** | 468 | **0.6972** | 2.01 | **+0.298 ± 0.016** | 97% |

What the generator writes, 200 molecules a request, for `fruity`, `woody`, `musk`, `sulfurous` and two with
`minty`: valid 1.00, unique 0.89 to 1.00, novel 0.70 to 0.86, median weight 148 to 228 and 0 to 2% above 350 (the
full table is in `output/odor/report/results.md`).

| classifier | molecules | ROC-AUC | average precision |
|---|---|---|---|
| validation | 352 | 0.832 | 0.224 |
| test | 352 | 0.811 | 0.268 |
| DREAM | 468 | 0.776 | 0.224 |

**Training on the validation molecules too** (`--include-validation --tag full`, which keeps the run in `stage2_full`
and `stage3_full`) adds the 352 validation molecules to the training once the settings are fixed. Scored on the molecules
still held out, it did not help: the generator on test is 0.7060 against 0.7096 (−0.0036 ± 0.0070, within the noise) and
on DREAM 0.6886 against 0.6759 (+0.0127 ± 0.0052, a little worse), and the classifier's ROC-AUC is 0.814 against 0.812
on test and 0.776 against 0.779 on DREAM. The held-out models stay the final ones. No other public source adds
molecules: Pyrfume's remaining odor-description datasets hold 6 molecules we do not have.

## Limits to keep in mind

- **The score says how probable the real molecules are, not what a written molecule smells like.** Nobody has
  smelled the molecules the generator writes. The checks above show they are valid, varied, new and the right size.
- Many settings were compared on the validation molecules, so its numbers are a little optimistic. The test and
  DREAM numbers are not, except that the 160 epochs were chosen after an 80-epoch run had already been scored on the
  test split.
- With no request the generator writes a generic molecule of the pretraining (median weight 264), not an odorant:
  that is what replay teaches the empty request.
- The odor token list has defects that were not fixed, because a new odor token list is a new vocabulary and so a new
  step 1: odor tokens glued together by one source (`almond bitter almond`), and `almost odorless`.
- `<mol>` is still in the vocabulary although nothing uses it any more (removing it would renumber every token).
- A bigger model is the open lever: step 1 stopped improving at this size.

## The dev scale: trying an idea in minutes

A full pretraining epoch takes seven minutes, so ideas are tried first on a small fixed copy of the data. (Stage 2
itself takes seven minutes at full scale and needs no dev scale.)

```
python -m scripts.preprocessing.odor_dev           # builds data/processed/odor_dev, about a minute
$env:DS_SCALE = "dev"                               # from here on everything reads and writes the dev files
python -m scripts.core --stage 1 --max-epochs 8
Remove-Item Env:DS_SCALE                            # back to the full scale
```

The dev set has 500,000 pretraining molecules (15 million tokens); the validation molecules, the fine-tuning data
and the vocabulary are *copied*, not rebuilt, so a loss at dev scale is measured on the same molecules with the same
tokens. Compare runs only with runs at the same scale and the same `--max-epochs`. The dev results go to
`output/odor_dev`. `DS_SCALE` moves only the tables, the vocabulary, the packed tokens and the output folder, and
the full table builder refuses to run under `dev`. A fresh stage 1 trained there leaves the request adapters at
exactly zero.

The earlier odor profile model, its DREAM Subchallenge 1 and 2 code, results and documents are removed from the tree; they are kept in git at the tag `archive/structure-to-odor-profile`.
