# DeepSniffing

Computers already see, hear, and talk — we're teaching them to _smell_.

Predicting odors from molecular structures, and generating molecules from odors.
Semester project for MAI642 — Deep Learning.

## Project overview

DeepSniffing explores the link between a molecule's structure and how it smells, in both directions. It is a team
project for MAI642 — Deep Learning.

A small decoder-only transformer (a GPT) reads and writes molecules as text. A molecule is written as a SELFIES
string, which is cut into **molecule tokens**, and an odor such as `fruity` or `musk` is one **odor token**. The
model is built in three steps, and each step stands alone:

1. **Pretraining.** The decoder learns the language of molecules from 3.4 million PubChem molecules, with no odor
   information.
2. **Generator** (the goal): odor tokens → molecule. Given a set of odor tokens, in any order, the decoder writes a
   molecule one molecule token at a time.
3. **Classifier**: molecule → odor tokens. A small head on the frozen pretrained decoder predicts which of the 890
   odor tokens a molecule has.

The odor tokens come from eleven perfumery and flavor sources in Pyrfume, which together give 7,516 molecules with
odor tokens. The DREAM Olfaction Prediction Challenge molecules are held out of all training and used as an external
test. RDKit is used to parse and canonicalize structures and to check what the model writes.

A model is judged only by the data we have: the probability it gives held-out molecules when told their own odor
tokens (generator), and the ROC-AUC of naming their odor tokens (classifier). Nothing the model writes is judged by a
rule of ours, because one odor can describe countless molecules and a molecule that lacks a label is not wrong. The
details, the results and the limits are in [ODOR_TRAINING.md](ODOR_TRAINING.md).

The repository contains scripts for preparing the data, training the three steps, measuring them, and reviewing
the results. Model results should be considered alongside the split and the evaluation method used: the
validation molecules were used to choose settings, the test molecules and the DREAM molecules were not.

## Project state

The repository contains the data pipeline (odor sources and the PubChem download), the three training steps, the
evaluation, plots and report scripts, and tests. The pretrained decoder, the generator and the classifier are
trained; their results are in [ODOR_TRAINING.md](ODOR_TRAINING.md). The earlier work, which predicted odor profiles
from structure, is no longer in the tree: it is at the git tag
`archive/structure-to-odor-profile`, and the parts no commit holds are attached to that release.

## Repository layout

```
DeepSniffing/
├── data/
│   ├── raw/                                    Downloaded, not committed
│   │   ├── pyrfume-data/                       Odor sources (leffingwell, goodscents, ...) and the DREAM molecule list
│   │   └── pubchem/                            CID-SMILES and CID-Mass archives; odor_pubchem.csv (committed cache)
│   └── processed/odor/
│       ├── vocabulary.json                     The 2,077 tokens (committed: a saved model is tied to it)
│       ├── molecules.csv, odor_tokens.csv      Odor tokens of the 7,516 molecules, and their counts
│       ├── odor_dataset.csv                    Fine-tuning set with the train/validation/test/dream split
│       ├── structures.csv                      The 3.4 million pretraining molecules
│       └── stage1_tokens.npy (+ companions)    The pretraining molecules packed as token numbers
├── scripts/
│   ├── data/
│   │   ├── prepare_data.py                     Download the odor sources and the PubChem archives
│   │   └── odor_pubchem.py                     Ask PubChem for what an odor source leaves out
│   ├── preprocessing/
│   │   ├── odor_tokens.py                      Read the odor sources and merge their odor tokens
│   │   ├── pubchem_structures.py               Filter PubChem to 350 Da or less and convert to SELFIES
│   │   ├── odor_tables.py                      Build the tables, the split and the vocabulary
│   │   ├── molecule_tokens.py                  Pack the pretraining molecules as token numbers
│   │   ├── build_odor.py                       Run the four steps above (ds-build)
│   │   └── odor_dev.py                         A small fixed copy of the data for quick tests
│   └── core/
│       ├── model.py, vocabulary.py, data.py    The decoder, its tokens and its training texts
│       ├── training.py                         The shared engine: one epoch, the loop, saving, loading
│       ├── pretrain.py                         Step 1: the pretraining
│       ├── generator.py                        Step 2: odor tokens -> molecule
│       ├── classifier.py                       Step 3: molecule -> odor tokens
│       ├── train.py                            The command line (ds-train)
│       ├── evaluate.py, compare.py             Measure the generator, compare two runs
│       ├── metrics.py, history.py              The numbers of a run, and the record of it
│       └── plots.py, report.py                 Every figure, and the table of results
├── tests/                                      Tests of the pipeline, the model and the steps
├── output/odor/                                Trained models, histories, plots, reports (only stage1/best.pt is committed)
│   ├── stage1/  stage2/  stage3/               The pretraining, the generator, the classifier
│   └── report/                                 results.md and the figures of the evaluations
├── docs/
│   └── mai642_project_description.md           Course project description
├── ODOR_TRAINING.md                            Data, training, scoring, results and limits
├── pyproject.toml                              Poetry project and dependency configuration
├── poetry.lock                                 Locked dependency versions
├── requirements.txt                            Pip dependency list
└── README.md                                   Setup, data preparation, and training guide
```

## Requirements

- Python 3.13 (>=3.13, < 3.14 as declared in `pyproject.toml`)
- [Poetry](https://python-poetry.org) (2.x) — the project uses poetry for dependency management and installs
  itself in editable mode, so the `ds-*` commands below run the working tree.
- A GPU is not required but makes pretraining practical: it takes about three hours on an RTX 3060.
- About 25 GB of disk for the PubChem download and the files built from it.

## Installation

**Clone the repository:**

```bash
git clone https://github.com/suprch4rg3d/DeepSniffing.git
cd DeepSniffing
```

**With Poetry (recommended):**

```bash
poetry shell      # create and activate the project environment
poetry install    # install the dependencies, including PyTorch, RDKit and SELFIES
```

**With plain pip (no Poetry):**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Development

The repository uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting, configured in
`pyproject.toml` (line length 100, import sorting enabled). It is installed with poetry as a dev dependency. Run
both, and the tests, before pushing:

```bash
poetry run ruff check .
poetry run ruff format .
poetry run pytest
```

## Data

### Build the datasets from raw data

From the repository root, run these commands in order. The first downloads any missing odor source files. The
second downloads PubChem's compound list (1.5 GB and 1.3 GB), unpacks it, and builds every table. Each step is skipped
if what it makes is already there, so an interrupted build goes on from where it stopped.

The commands are written for macOS/Linux; on Windows the same commands work unchanged in PowerShell or cmd.

If the processed tables are already on your machine, you can use them directly and skip rebuilding.

```bash
poetry run ds-prepare      # download the odor sources from Pyrfume (a pinned revision)
poetry run ds-build        # download PubChem, then build every table and the packed tokens
```

`ds-prepare` skips files that are already present; `poetry run ds-prepare --force-download` redownloads them.
`ds-build` runs four modules in order, each of which can also be run alone with `python -m`:

1. `scripts/preprocessing/odor_tokens.py` merges the odor tokens of the eleven sources, gives every molecule its
   PubChem CID and structure, and asks PubChem (through `scripts/data/odor_pubchem.py`) only for what a source leaves out.
2. `scripts/preprocessing/pubchem_structures.py` downloads PubChem's `CID-SMILES.gz` and `CID-Mass.gz`, keeps the
   58,854,892 compounds of 350 Da or less, and converts a sample of 3.4 million of them to canonical SMILES and SELFIES.
3. `scripts/preprocessing/odor_tables.py` builds the pretraining table, the fine-tuning table, the split and the
   vocabulary.
4. `scripts/preprocessing/molecule_tokens.py` packs the pretraining molecules into token numbers.

The scripts create (in `data/processed/odor/` unless a path says otherwise):

- `molecules.csv` — one row per molecule with odor tokens: `cid`, `smiles`, the `sources` that name it, and its
  `odor_tokens` joined by `;` (7,516 molecules).
- `odor_tokens.csv` — one row per odor token: how many molecules and sources carry it, and the spellings of the
  sources that were merged into it (`fruity` has 2,713 molecules and the spellings `fruity;fruit`).
- `dropped.csv` — every source row that was left out, with the reason (for example, PubChem has no CID for it).
- `odor_dataset.csv` — the fine-tuning set: `cid`, `smiles`, `odor_tokens`, `sources`, `selfies` and `split`
  (6,344 train, 352 validation, 352 test and 468 dream).
- `structures.csv` — the pretraining set: `cid`, `smiles`, `selfies`, `source` and `split` (3,422,276 train and 5,017
  validation molecules; the source is `pubchem`, `odor` for the 6,344 training molecules of the fine-tuning set, or
  `other`).
- `vocabulary.json` — the 2,077 tokens: 6 special, 1,181 molecule tokens and 890 odor tokens.
- `stage1_tokens.npy`, `stage1_offsets.npy`, `stage1_split.npy` and `stage1_tokens.json` — the pretraining molecules
  as token numbers, where each one starts, its split, and the vocabulary they were packed with.
- `pubchem_light.csv`, `pubchem_light.npy` and `pubchem_structures.csv` — the intermediate PubChem files: every
  compound of 350 Da or less, the mask that selects them, and the 3.4 million sampled and converted.
- `data/raw/pubchem/odor_pubchem.csv` — the cache of PubChem answers for the odor molecules, so a rerun needs no
  network.

In the tables, `selfies` is the molecule as the text the model reads and writes, made from the structure in `smiles`. A SELFIES string is cut into molecule tokens; an odor token is a term such as
`fruity`. The odor tokens of a molecule are a set, so their order in `odor_tokens` means nothing.

Why 350 Da: a molecule has to evaporate to be smelled. Of the 7,516 molecules with odor tokens, 98.5% weigh 350 or
less, and PubChem's median is 388 against 174 for the odor molecules. Nothing is left out for the elements it is made
of: the odor molecules themselves hold sodium, zinc and boron.

The 352 validation, 352 test and 468 DREAM molecules are kept out of the pretraining by CID and by structure (PubChem
gives one molecule more than one CID), so the model never sees them in either step. The splits are reproducible (a
fixed seed, and a checksum of the CID for the pretraining validation molecules).

Committed are the files that the generator and the classifier start from: `vocabulary.json`, `odor_dataset.csv`, the
small cache `odor_pubchem.csv` and the pretrained decoder `output/odor/stage1/best.pt` (23 MB). The pretraining inputs
(`structures.csv`, 674 MB, and the packed tokens, 201 MB) are over GitHub's 100 MB limit, so they, and everything else,
are rebuilt by the two commands above and are in `.gitignore`.

### Optional: a small copy for quick tests

A full pretraining epoch takes about seven minutes, so ideas are tried first on a small fixed copy of the data: 500,000
pretraining molecules, with the same validation molecules, fine-tuning data and vocabulary.

```bash
poetry run python -m scripts.preprocessing.odor_dev      # builds data/processed/odor_dev, about a minute
$env:DS_SCALE = "dev"                                    # PowerShell; use `export DS_SCALE=dev` elsewhere
poetry run python -m scripts.core --stage 1 --max-epochs 8
```

The dev results go to `output/odor_dev`, so they never mix with the full-scale ones. Compare dev runs only with dev
runs at the same number of epochs. Details are in [ODOR_TRAINING.md](ODOR_TRAINING.md).

### Model development

The training code is a small decoder-only transformer that reads and writes SELFIES molecule tokens and odor
tokens. The three steps are `pretrain.py`, `generator.py` and `classifier.py` in `scripts/core/`, on a shared
engine (`training.py`); the models and their numbers go to `output/odor/`. The generator is scored by the
probability it gives held-out molecules when told their own odor tokens, and by what it writes (valid, unique,
novel, weight). The classifier is scored by ROC-AUC. Neither score calls an unlabelled molecule wrong.

```bash
poetry run python -m scripts.core --stage 1     # step 1, the pretraining
poetry run python -m scripts.core --stage 2     # step 2, the generator: odor tokens -> molecule
poetry run python -m scripts.core --stage 3     # step 3, the classifier: molecule -> odor tokens
poetry run python -m scripts.core               # all three, one after the other
```

Steps 2 and 3 both start from the pretrained decoder and do not depend on each other. The generator trains on the
loss of writing the molecule and nothing else. The classifier trains only a head on the frozen decoder and cannot
change a weight of it. Each step has its own folder in `output/odor/`, its own `history.csv` and its own scores. A
step saves its state after every epoch and resumes when the same command is run again.

```bash
poetry run python -m scripts.core.evaluate --split test       # measure the generator (validation, test, dream)
poetry run python -m scripts.core.classifier --split test     # measure the classifier
poetry run python -m scripts.core.plots --stage 2             # the curves of a step (1, 2 or 3)
poetry run python -m scripts.core.report                      # results.md and the figures of the evaluations
```

## Glossary

### Molecule structure

| Term | Meaning in this project |
|---|---|
| **CID** | PubChem Compound Identifier: the number used to match a molecule across the source files. |
| **SMILES** | Text notation for a molecule's atoms and bonds. |
| **Canonical SMILES** | A consistent ordering of a molecule's SMILES, generated by RDKit, so that strings can be compared. It does not verify that a source assigned the right structure to a CID. |
| **SELFIES** | A text notation for molecules in which every string is a valid molecule. The decoder reads and writes SELFIES, so what it writes is valid by construction. |
| **RDKit** | The open-source chemistry toolkit used here to parse SMILES, canonicalize them, and check and weigh what the model writes. [RDKit documentation](https://www.rdkit.org/docs/GettingStartedInPython.html) |
| **Stereochemistry** | The three-dimensional arrangement of atoms around parts of a molecule. Isomeric SMILES can encode it when the source structure includes it. |
| **Atom** | A unit of an element in a molecule, such as carbon or oxygen. |
| **Pyrfume** | A public collection of olfaction datasets. The odor sources of this project (Leffingwell, The Good Scents Company, Arctander, FlavorDB, Sharma, IFRA, AromaDB, Sigma, Flavornet, Dravnieks, Keller) are read from it at a pinned revision. |
| **PubChem** | The public database of chemical molecules. It gives the structures for the pretraining and the CID that matches a molecule across sources. |
| **Molecular weight (Da)** | The mass of a molecule. The pretraining keeps molecules of 350 Da or less, because larger molecules rarely evaporate enough to smell. |

### Tokens and the model

| Term | Meaning in this project |
|---|---|
| **Molecule token** | One piece of a SELFIES string, such as `[C]` or `[=O]`. There are 1,181 of them. A molecule is written as a sequence of molecule tokens. |
| **Odor token** | One odor term, such as `fruity` or `musk`. There are 890 of them. A request is a set of odor tokens. |
| **Special token** | A marker such as `<bos>`, `<eos>`, `<odor>` or `<sep>`. There are 6. |
| **Vocabulary** | The 2,077 tokens (6 special, 1,181 molecule, 890 odor) and their numbers, in `vocabulary.json`. A saved model is tied to it. |
| **Request** | The text `<odor> odor-token odor-token <sep>` that comes before a molecule in the generator. Its odor tokens are a set: they have no position, so their order cannot matter. |
| **Decoder-only transformer** | A model that reads tokens left to right and predicts the next one. Here it has 6 layers, 8 attention heads and width 256, about 6.3 million parameters. |
| **Embedding** | A table of 256 numbers for each token. The odor-token embeddings are the model's meaning of each odor token, and are learned in step 2. |
| **Request adapter** | A small layer in each decoder layer. It averages the asked odor tokens into one vector and uses it to scale and shift the state for every molecule token after `<sep>`. Its weights start at zero, so it changes nothing until it is trained. |
| **Pretraining** | Step 1: learning to write molecules from millions of molecules with no odor tokens. |
| **Generator** | Step 2: the decoder fine-tuned to write a molecule for a set of odor tokens. The loss is on the molecule only. |
| **Classifier** | Step 3: a head trained on the frozen pretrained decoder to score each odor token for a molecule. |
| **Odor descriptor** | A term people use to describe an odor, such as `fruity` or `woody`. Each is one odor token. A source lists the terms that a perfumery or flavor reference gives to a molecule. |
| **Attention** | The operation that lets a token look at earlier tokens. It is causal here: a token never sees the ones after it. |
| **Dropout** | Randomly switching off part of the model during training so that it does not memorize. 0.1 in the pretraining, 0.2 in the generator. |
| **Layer decay** | In the generator, each layer nearer the input learns at 0.8 of the rate of the layer above it, so the odor tokens do not wipe out what the pretraining taught the lower layers. |
| **Epoch** | One pass over the training molecules. |
| **Replay** | Mixing pretraining molecules, written after an empty request, into the generator's epochs, so that what the pretraining knew stays in use. |

### Evaluation

| Term | Meaning in this project |
|---|---|
| **Training split** | Molecules used to update model weights. |
| **Validation split** | 352 held-out molecules used to choose settings and the best epoch. |
| **Test split** | 352 held-out molecules, looked at once at the end. |
| **DREAM molecules** | 468 molecules of the DREAM Olfaction Prediction Challenge, held out of everything the model learns from, used as an external test. |
| **Loss** | How surprised the model is by the real molecule, in nats per molecule token. Lower is better. |
| **Nat** | The unit of the loss: the natural logarithm of how unlikely something is. |
| **Perplexity** | The loss as a number of equally likely choices: 2 means the model is as unsure as choosing between two tokens. |
| **Gain** | The loss without the odor tokens minus the loss with them. Zero means the model ignores its request. |
| **ROC-AUC** | How well the classifier ranks the molecules that have an odor token above those that do not, averaged over the odor tokens. 0.5 is chance and 1 is perfect. |
| **Valid / unique / novel** | Of the molecules the generator writes: the share RDKit can read, the share that differ from each other, and the share found in neither step's training set. |

## Project description

See [docs/mai642_project_description.md](docs/mai642_project_description.md).
