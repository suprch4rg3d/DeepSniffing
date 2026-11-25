# DeepSniffing

Computers already see, hear, and talk. We're teaching them to _smell_.

Predicting odors from molecular structures, and generating molecules from odors.
Semester project for MAI642 Deep Learning.

## Project overview

DeepSniffing explores the link between a molecule's structure and how it smells, in both directions. It is a team
project for MAI642 Deep Learning.

A small decoder-only transformer (a GPT) reads and writes molecules as text. A molecule is written as a SELFIES
string, which is cut into **molecule tokens**, and an odor such as `fruity` or `musk` is one **odor token**. The
model is built in three steps, and each step stands alone:

1. **Pretraining.** The decoder learns the language of molecules from 3.4 million PubChem molecules, with no odor
   information.
2. **Generator** (the goal): odor tokens → molecule. Given a set of odor tokens, in any order, the decoder writes a
   molecule one molecule token at a time.
3. **Classifier**: molecule → odor tokens. A small head on the frozen pretrained decoder predicts which of the 890
   odor tokens a molecule has.

The odor tokens come from nine perfumery and flavor sources in Pyrfume, which together give 7,516 molecules with
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

The repository holds the whole decoder: the data pipeline, the three training steps, the evaluation, the plots and
the report, eight analysis scripts and ten notebooks. The pretrained decoder, the generator and the classifier are
trained and committed, with their histories, their evaluations and [the report](output/odor/report/results.md). The
numbers and the limits are in [ODOR_TRAINING.md](ODOR_TRAINING.md).

The earlier work, which predicted odor profiles from structure, is not in the tree. It is at the git tag
`archive/structure-to-odor-profile`, and the material no commit holds is on that release.

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
│   ├── data/                                   Download the odor sources and the PubChem answers
│   ├── preprocessing/                          Merge the sources, filter PubChem, build the tables and the vocabulary
│   ├── core/                                   The decoder, the three steps, the evaluation, the report
│   └── analysis/                               Eight scripts, one question each, one JSON each
├── tests/                                      Tests of the pipeline, the model, the steps and the analyses
├── output/
│   ├── odor/                                   Checkpoints, histories, evaluations, results.md and figures
│   └── analysis/                               One JSON per analysis script
├── notebooks/                                  Ten notebooks; none stores outputs
├── docs/
│   ├── mai642_project_description.md           Course project description
│   └── glossary.md                             The terms, from molecule tokens to macro ROC-AUC
├── RUN.txt                                     Commit, date, GPU and versions of the committed run
├── ODOR_TRAINING.md                            Data, training, scoring, results and limits
├── pyproject.toml, poetry.lock                  Poetry project and locked versions
├── requirements.txt                            Pip dependency list
├── LICENSE                                     MIT, for the code
└── README.md                                   Setup, data preparation, training and analysis
```

## Requirements

- Python 3.13 (>=3.13, < 3.14 as declared in `pyproject.toml`)
- [Poetry](https://python-poetry.org) (2.x). The project uses Poetry for dependency management and installs
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

1. `scripts/preprocessing/odor_tokens.py` merges the odor tokens of the nine sources, gives every molecule its
   PubChem CID and structure, and asks PubChem (through `scripts/data/odor_pubchem.py`) only for what a source leaves out.
2. `scripts/preprocessing/pubchem_structures.py` downloads PubChem's `CID-SMILES.gz` and `CID-Mass.gz`, keeps the
   58,854,892 compounds of 350 Da or less, and converts a sample of 3.4 million of them to canonical SMILES and SELFIES.
3. `scripts/preprocessing/odor_tables.py` builds the pretraining table, the fine-tuning table, the split and the
   vocabulary.
4. `scripts/preprocessing/molecule_tokens.py` packs the pretraining molecules into token numbers.

The scripts create (in `data/processed/odor/` unless a path says otherwise):

- `molecules.csv`: one row per molecule with odor tokens: `cid`, `smiles`, the `sources` that name it, and its
  `odor_tokens` joined by `;` (7,516 molecules).
- `odor_tokens.csv`: one row per odor token: how many molecules and sources carry it, and the spellings of the
  sources that were merged into it (`fruity` has 2,713 molecules and the spellings `fruity;fruit`).
- `dropped.csv`: every source row that was left out, with the reason (for example, PubChem has no CID for it).
- `odor_dataset.csv`: the fine-tuning set: `cid`, `smiles`, `odor_tokens`, `sources`, `selfies` and `split`
  (6,344 train, 352 validation, 352 test and 468 dream).
- `structures.csv`: the pretraining set: `cid`, `smiles`, `selfies`, `source` and `split` (3,422,276 train and 5,017
  validation molecules; the source is `pubchem`, `odor` for the 6,344 training molecules of the fine-tuning set, or
  `other`).
- `vocabulary.json`: the 2,077 tokens: 6 special, 1,181 molecule tokens and 890 odor tokens.
- `stage1_tokens.npy`, `stage1_offsets.npy`, `stage1_split.npy` and `stage1_tokens.json`: the pretraining molecules
  as token numbers, where each one starts, its split, and the vocabulary they were packed with.
- `pubchem_light.csv`, `pubchem_light.npy` and `pubchem_structures.csv`: the intermediate PubChem files: every
  compound of 350 Da or less, the mask that selects them, and the 3.4 million sampled and converted.
- `data/raw/pubchem/odor_pubchem.csv`: the cache of PubChem answers for the odor molecules, so a rerun needs no
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

## Analysis

`scripts/analysis/` holds the measurements behind the notebooks. Each script reads the committed checkpoints or
tables, writes one JSON under `output/analysis/`, and several draw a figure under `output/odor/report/`.

| Script | Question | Headline |
|---|---|---|
| `eda.py` | What is in the data? | 7,516 molecules, 890 odor tokens, 211 on a single molecule |
| `baseline.py` | What does a classic model get? | ridge on Morgan bits: ROC-AUC 0.725, 0.736 and 0.670 |
| `embeddings.py` | Did step 2 move the odor tokens? | median move 3.1 percent, 68 percent of neighbours kept |
| `ablations.py` | Which adapter carries the request? | +0.008 to +0.065 nats per adapter, +0.227 for all six |
| `generated_vs_real.py` | Does it write molecules near the real ones? | fruity 0.67 against 0.68, musk 0.48 against 0.61 |
| `odor_map.py` | Do related odors sit together? | Spearman 0.018 before step 2, -0.103 after, +0.162 among shared pairs |
| `request_layers.py` | Where does the request act? | shift norm 6.9 to 69.6, its share of the state 17 to 10 percent |
| `hidden_states.py` | Is the odor readable from a molecule's state? | silhouette -0.005, five-nearest-neighbour vote 0.57 against 0.22 chance |

The notebooks read the committed files and draw the figures. `00_data_overview` is the data, and
`analysis_classifier`, `analysis_generator`, `analysis_representations`, `analysis_ablation`,
`analysis_generated` and `analysis_hidden` take one claim each. `final_report` is the graded document.
`deepsniffing-on-colab` runs the whole pipeline on an L4, and `batch-budget-experiment-on-colab` compares two
attention budgets. None of them stores outputs.

## Glossary

The terms used here, from molecule tokens to macro ROC-AUC, are in [docs/glossary.md](docs/glossary.md).

## License

The code is MIT, Copyright 2026 Emmanouil Oikonomou, Konstantin Krasovitskiy and Anastasia Kyriakou. The full text
is in [LICENSE](LICENSE).

The tables under `data/processed/` come from Pyrfume's sources and PubChem and keep the terms of those sources. The
trained weights under `output/odor/` are provided as they are, with no warranty.

## Project description

See [docs/mai642_project_description.md](docs/mai642_project_description.md).
