# DeepSniffing

Computers already see, hear, and talk — we're teaching them to _smell_.

Predicting human odor profiles from molecular structures with machine learning.
Semester project for MAI642 — Deep Learning.

## Project overview

DeepSniffing predicts how people perceive the smell of a molecule from its chemical structure. It is a team
project for MAI642 — Deep Learning (Emmanouil Oikonomou, Konstantin Krasovitskiy, Anastasia Kyriakou).

The data is the DREAM Olfaction Prediction Challenge: 476 molecules, each rated by 49 people on 21 attributes
(intensity, pleasantness and 19 odor descriptors such as garlic, fruit and fish). Molecule information comes from
Pyrfume and PubChem, and RDKit parses the structures and calculates chemical features. There are two tasks:

- **Subchallenge 1** predicts what each of the 49 participants would rate.
- **Subchallenge 2** predicts the panel's mean rating and how much people disagree (the spread).

One neural network for each task, trained on the challenge data only, is scored with the official DREAM Z-score, so
the results compare directly with the published ones.

| | test score | published winner |
|---|---:|---:|
| Subchallenge 1 | 34.52 | 34.18 |
| Subchallenge 2 | 10.05 | 8.87 |

## Where to read more

- `PIPELINE.md`: the commands in order, what each reads and writes, and how a model is trained.
- `Q1_SETUP.md` and `Q2_SETUP.md`: the setup of each model and how it was reached.
- `RESULTS.md`: the scores, what helped, what did not, and how reliable the numbers are.
- `PROJECT_SCOPE.md`: what the project covers.

## Repository layout

```
DeepSniffing/
├── data/
│   ├── raw/                                        Downloaded by ds-prepare
│   │   ├── dream-olfaction-prediction/data/
│   │   │   ├── TrainSet.txt, TestSet.txt           DREAM ratings for training and test
│   │   │   ├── LBs1.txt, LBs2.txt                  DREAM leaderboard ratings (validation)
│   │   │   ├── CID_* and dilution_*                DREAM challenge molecule lists
│   │   │   ├── molecular_descriptors_data.txt      The published Dragon descriptors
│   │   │   ├── DREAM_episuite_descriptors.txt      EPI Suite property estimates
│   │   │   ├── morgan_sim.csv                      Morgan similarity to reference odorants
│   │   │   ├── name_features.txt                   Organizers' 4-letter name windows
│   │   │   └── derived/nspdk_*                     De Grave's NSPDK vectors and their CIDs
│   │   ├── dream-olfaction-prediction/hulab/nspdk2.csv   NSPDK substructure features
│   │   ├── pyrfume-data/                           Keller molecule metadata and ratings
│   │   └── pubchem/                                PubChem properties and computed 3D structures
│   ├── processed/dream_pyrfume/                    One table per purpose, each written by one command
│   │   ├── molecules.csv                           One row per molecule: CID, SMILES, odor name (no ratings)
│   │   ├── split.csv                               train / validation / test / unused, and the training order
│   │   ├── morgan_fingerprints.csv                 2,048 Morgan substructure bits
│   │   ├── pubchem_3d_coordinates.csv              PubChem's 3D position of each heavy atom
│   │   ├── molecule_descriptors.csv                RDKit and Dragon descriptors and per-atom partial charges
│   │   ├── published_features.csv                  EPI Suite, name, NSPDK and similarity features
│   │   ├── feature_selection.csv                   Every candidate column, kept or dropped, and why
│   │   ├── scalers/                                Feature and target scalers (ds-build)
│   │   └── inspection/                             Checks' plots (ds-inspect; not committed)
│   └── reference/dream_challenge/                  Published scores, stored predictions, the official scorer
├── scripts/
│   ├── data/prepare_data.py                        ds-prepare: download the DREAM, Pyrfume and PubChem data
│   ├── preprocessing/
│   │   ├── dream_dataset.py                        The molecule table
│   │   ├── morgan_fingerprints.py                  Morgan fingerprints
│   │   ├── conformers.py                           PubChem 3D coordinates in our atom order
│   │   ├── enrich_molecules.py                     RDKit, Dragon and charge descriptors; the ds-build command
│   │   ├── community_features.py                   The published EPI Suite, name, NSPDK and similarity features
│   │   ├── feature_selection.py                    Which columns each model keeps
│   │   ├── split_dataset.py                        A step of ds-build: writes split.csv
│   │   ├── scalers.py                              A step of ds-build: writes the scalers
│   │   └── dataset_inspector.py                    ds-inspect
│   ├── core/
│   │   ├── splits.py                               The split, and where the raw ratings are
│   │   ├── data/                                   Loads what the commands wrote, one file per job:
│   │   │   ├── ratings.py                          the DREAM rating files and the published reference scores
│   │   │   ├── features.py                         the molecule tables joined into rating rows
│   │   │   ├── scaling.py                          the scalers that ds-build wrote, read back and applied
│   │   │   ├── targets.py                          the targets of the two subchallenges
│   │   │   ├── neighbours.py                       nearest-neighbour labels (Q2)
│   │   │   ├── graphs.py                           molecule graphs and batches
│   │   │   └── assemble.py                         prepare_graph_data: the loaders of one run
│   │   ├── metrics.py                              The official DREAM scores
│   │   ├── train/                                  ds-train
│   │   │   ├── loop.py                             the loss, one epoch, prediction, the settings of a run
│   │   │   ├── q2.py                               train_q2: the whole Q2 run, including saving its model
│   │   │   └── q1.py                               train_q1: the same for Q1
│   │   └── plots.py                                Draws the figures from the saved outputs
│   └── training/
│       ├── custom_model.py                         The 3D molecule graph network with per-attribute attention
│       ├── predict.py                              ds-predict: score a molecule
│       └── run_experiments.py                      ds-experiments: reserved, not active
├── output/
│   ├── q2/                                         Subchallenge 2: checkpoints/, metrics/, plots/, train.log
│   └── q1/                                         Subchallenge 1: the same
├── tests/                                          Pytest suite for the data contract, scoring and training
├── docs/mai642_project_proposal.md                 Course project proposal
├── PIPELINE.md                                     The pipeline on one page
├── RESULTS.md, Q1_SETUP.md, Q2_SETUP.md            Results and the setup of each model
├── pyproject.toml, poetry.lock, requirements.txt   Dependencies
└── README.md                                       This guide
```

## Requirements

- Python 3.13 (>=3.13, < 3.14 as declared in `pyproject.toml`)
- [Poetry](https://python-poetry.org) (2.x), which manages the dependencies. `poetry install` also
  installs the project, which registers the `ds-*` commands.

## Installation

**Clone the repository:**

```bash
git clone https://github.com/suprch4rg3d/DeepSniffing.git
cd DeepSniffing
```

**With Poetry (recommended):**

```bash
poetry shell      # create and activate the project environment
poetry install    # install the dependencies, including pandas and RDKit
```

**With plain pip (no Poetry):**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Development

The repository uses [ruff](https://docs.astral.sh/ruff/) for linting and
formatting, configured in `pyproject.toml` (line length 100, import sorting
enabled). Ruff and pytest are installed with poetry as dev dependencies. Run
all three before pushing:

```bash
poetry run ruff check .
poetry run ruff format .
poetry run pytest
```

### Available commands

`poetry install` registers these console scripts. They run from any directory,
and each one is a thin wrapper around a module under `scripts/`. With plain pip
instead of Poetry, call the modules directly, for example
`python -m scripts.training.predict`.

| Command | Purpose |
|---|---|
| `ds-prepare` | Download the DREAM, Pyrfume and PubChem sources |
| `ds-build` | Build every file the models read, in order: the tables (molecules, Morgan bits, 3D coordinates, RDKit, Dragon and charge descriptors, the published features, the feature selection), then the split, then the scalers |
| `ds-inspect` | Check every input of the models, and draw summary plots |
| `ds-train` | Train and score Subchallenge 1 and 2 (`--challenge q1` or `q2` for one) |
| `ds-plot` | Draw the figures of the trained models from the files they left |
| `ds-predict` | Score a molecule |
| `ds-experiments` | Reserved for the ablations; it only says where they are recorded |

## Data

### Build the tables from raw data

From the repository root, run these commands in order. `ds-prepare` downloads
any missing DREAM and Pyrfume files, then fetches PubChem's properties and 3D
structures. It skips files that are already present and reuses PubChem records
that it fetched before, so an interrupted run continues where it stopped.

The commands are Poetry console scripts, installed with `poetry install`. The
module form works as well, for example `poetry run python -m scripts.data.prepare_data`.
They are written for macOS and Linux, and work unchanged in PowerShell or cmd.

If the processed tables are already in the repository, you can skip to `ds-train`.

```bash
poetry run ds-prepare
poetry run ds-build
poetry run ds-inspect
```

`PIPELINE.md` shows what each command reads and writes. In short:

- `ds-prepare --download-only` downloads the raw sources without querying PubChem, and
  `--force-download` downloads the DREAM and Pyrfume files again. PubChem's 3D structures go to
  `data/raw/pubchem/dream_conformers.sdf` in batches of 100, with a wait between requests; a
  cut-off answer is rejected, and molecules without a 3D structure are listed in
  `dream_conformers_unavailable.txt` and not requested again.
- `ds-build` first writes the six input tables of `data/processed/dream_pyrfume/`, each named for what it
  holds, and stops at the first problem.
- Then it writes `split.csv`. Train is all 338 molecules of `TrainSet.txt`, validation is the 69 DREAM
  leaderboard molecules (rated in `LBs1.txt`), and test is the official DREAM test set of 69 molecules, which is
  never sampled and never used for training or epoch selection. The other four molecules have no DREAM
  ratings and are labelled `unused`. The training order is one fixed shuffle of the training molecules.
- Last it writes `scalers/feature_scalers.joblib` and `scalers/target_scalers.joblib`: each scaler is fitted
  on training molecules only, once for the model that selects epochs (338 molecules) and once for the final model
  (407). `ds-train` reads them and refuses a file fitted on other molecules, other columns or another Q1 blend
  weight than the run uses.
- `ds-inspect` prints PASS or FAIL for each check (the split is the DREAM split, no table of molecules carries
  a rating, every DREAM molecule has Dragon descriptors and the published features, the feature selection kept
  columns in every block, the training molecules have all 21 means and spreads, the scalers match the run), and
  draws the plots to `inspection/`. It exits with status 1 if a check fails. It notes the two DREAM molecules
  that PubChem has no 3D structure for; their graph has the bonds only.

In `molecules.csv`, `pubchem_smiles` preserves PubChem's original SMILES string. RDKit parses that string and
writes its canonical isomeric form to `rdkit_canonical_smiles`, the column to use as model input. The pipeline
does not fetch or include PubChem `ConnectivitySMILES`. No table of molecules carries a rating: the ratings are
read from the raw DREAM files when a model trains.

### Organizer and community features

`ds-prepare` also downloads the feature files that the DREAM organizers and the community phase published for
the 476 challenge molecules, from the same pinned commit as the ratings. They cannot be recomputed from a SMILES
string here, so `ds-build` joins them into `published_features.csv`, keyed by CID:

- `orgname_*`: the organizers' binary 4-letter windows of the molecule names, a larger vocabulary than the
  windows built from the DREAM odor names.
- `epi_*`: EPI Suite physical property estimates, such as vapor pressure and log Kow.
- `nspdk_*`: NSPDK graph-kernel substructure features.
- `nsim_*` and `msim_*`: NSPDK and Morgan similarity to about 2,000 reference odorants. The NSPDK similarity is
  the dot product of De Grave's NSPDK vectors, which is how its author defines it, so it is computed from the
  published vectors. The DREAM molecules are among the reference odorants, and each would own a column of
  similarity 1 to itself, so those 476 columns are dropped.

Name windows and substructures that only one molecule has are dropped. The graph model reads the EPI Suite
values with the numeric block, the organizers' name windows with the odor-name features, and the NSPDK and
similarity features as a structure block of their own.

The organizers' name windows are numbers without their letters, so what each one means is unknown. They raise the
official test score from 34.35 to 36.34 in the same setup, mostly on valence and the descriptors. The validation
score did not move, so the gain shows on the test set only. They are part of both models.

### Which columns the model takes

Several datasets describe the same molecules, so their columns overlap: RDKit's carbon count is Dragon's `nC`,
and many Morgan bits, name windows, and substructures are constant or copies of each other. The last step of
`ds-build` decides, once, which columns are kept from each dataset and writes the decision to
`feature_selection.csv`, one row per candidate column with the reason it was dropped. It uses only the features,
never the ratings, and sees the features of all 476 DREAM molecules, test molecules included, as the name windows
already do. A column is dropped when it is:

- **constant:** it has one value for every DREAM molecule;
- **rare:** a binary or substructure column that only one molecule has, which the model could only memorise;
- **a duplicate:** its rank correlation with a column already kept is at least 0.995. The 0.99 threshold dropped
  `P_VSA_i_1`, a top valence feature, as a copy of `P_VSA_MR_8` (rank correlation 0.9916).

Datasets are visited in a fixed priority order, Dragon, RDKit, charge, EPI Suite, Morgan bits, name windows,
NSPDK, and the similarity sets, and the first column of a group of copies is the one that is kept. The data
loading reads the selection, and a fitted scaler remembers its columns, so a saved model never needs it again.

### RDKit and Dragon descriptors

`ds-build` adds counts for C, H, N, O, P, S, Na, Cl, and I; heavy atoms, aromatic and aliphatic rings, rotatable
bonds, hydrogen-bond donors and acceptors, topological polar surface area, and estimated logP, all prefixed
`rdkit_`; the 4,868 Dragon descriptors that the challenge published; and the per-atom Gasteiger charges with their
summaries. The descriptor values are calculated from `rdkit_canonical_smiles`.

### The models

Models predict the 21 DREAM attributes of a molecule, and results can be compared with the official challenge
scores using the DREAM scoring method. Benchmark reference files and the official scoring tools are under
`data/reference/dream_challenge/`.

The model is a graph neural network fused with descriptor features. Each molecule becomes a 3D graph: its atoms sit
at the positions of PubChem's computed 3D structure, the same for everyone, and every pair of atoms within 6 Å is
connected. A few molecules that PubChem has no structure for keep their bonds with all distance features set to
zero. Atoms carry element, degree, charge, hybridisation, aromaticity, hydrogen count, chirality, and Gasteiger
charge features. Edges carry bond type, conjugation, ring, and stereo flags plus a 16-value radial encoding of the
distance.

The model reads all the columns of every table, about 19,000 in all. A small network turns each group of columns
into one vector of 64 numbers, called a token, and attention combines the tokens. Subchallenge 1 has 8 tokens and
Subchallenge 2 has a ninth, the neighbour token. The diagrams show every layer. `Linear(a -> b)` is a fully
connected layer from a numbers to b numbers.

#### The tokens (both subchallenges)

```
 DRAGON  2,008 columns, in 4 families
   family 1 (88 columns)   Linear(88 -> 16)   + ReLU + Dropout --+
   family 2 (768 columns)  Linear(768 -> 16)  + ReLU + Dropout --+--> four vectors of 16 numbers
   family 3 (289 columns)  Linear(289 -> 16)  + ReLU + Dropout --+       |
   family 4 (863 columns)  Linear(863 -> 16)  + ReLU + Dropout --+       |
                                                  each one:  Linear(16 -> 64) --> 4 Dragon tokens (64)
                                                                          |
                          the four vectors joined (64) -> Linear(64 -> 32) + ReLU + Dropout
                                                                          v
                                                          Dragon context (32 numbers)
                                                                          |
 MOLECULE GRAPH  atoms with 45 features, bonds with 28 features           | added to every step below
   every atom:  Linear(45 -> 64) + ReLU                                   |
   3 rounds of message passing                                            v
     message on every bond:  Linear(64 + 28 + 32 -> 64) + ReLU    (sending atom, bond and distance features,
                                                                    Dragon context)
     all messages that arrive at an atom are added up
     update of every atom:   Linear(64 + 64 + 32 -> 64) + ReLU    (own numbers, the sum of messages,
                                                                    Dragon context)
   mean pooling: the atoms of a molecule are averaged ------------------> graph token (64)

 NUMERIC BLOCK  701 columns (RDKit, charges, EPI Suite, Morgan bits, dilution)
   Linear(701 -> 64) + ReLU + Dropout -> Linear(64 -> 64) + ReLU ---------> numeric token (64)

 ODOR NAMES  10,608 columns
   Linear(10,608 -> 32) + ReLU + Dropout -> Linear(32 -> 64) -------------> names token (64)

 STRUCTURE  4,990 columns (NSPDK substructures and similarities, Morgan similarities)
   Linear(4,990 -> 32) + ReLU + Dropout -> Linear(32 -> 64) --------------> structure token (64)
```

#### Subchallenge 1: one query for every attribute and participant

```
 8 tokens, each 64 numbers (graph, 4 Dragon, numeric, names, structure)
   |
   | token dropout 0.15: while training each token is hidden with probability 0.15 (one always stays)
   v
 keys = Linear(64 -> 64)(tokens)          values = Linear(64 -> 64)(tokens)
   |
   | for each of the 21 attributes and each of the 49 participants there is a query, 64 learned numbers
   v
 ATTENTION   score of token i  = query . key_i / 8
             weights           = softmax over the 8 tokens        (they sum to 1)
             context           = sum of weight_i x value_i  +  the query          (64 numbers)
   |
   v
 HEAD of the attribute (the same head for all 49 participants)
   LayerNorm(64) -> Linear(64 -> 32) -> ReLU -> Dropout -> Linear(32 -> 1)
   |
   |  + a learned offset for this participant and attribute
   v
 one rating.   21 attributes x 49 participants = 1,029 outputs
```

#### Subchallenge 2: four experts and a gate, a mean and a spread for each attribute

```
 9 tokens: the 8 above, and the neighbour token
   NEIGHBOURS  42 numbers (what the 20 most similar training molecules were rated,
               by Morgan and by NSPDK similarity)
   Linear(42 -> 32) + ReLU + Dropout -> Linear(32 -> 64) ------------------> neighbour token (64)
   |
   | token dropout 0.3: while training each token is hidden with probability 0.3 (one always stays)
   v
 4 EXPERTS, each reads a view of the tokens plus the numeric token (which holds the dilution)
   expert 1  the 4 Dragon tokens
   expert 2  the graph token and the names token
   expert 3  the structure token
   expert 4  the neighbour token
   each expert has its own keys = Linear(64 -> 64), values = Linear(64 -> 64), and for every attribute its own
   query (64 learned numbers) and its own head:
     ATTENTION   weights = softmax(query . key_i / 8) over the tokens of its view
                 context = sum of weight_i x value_i + the query
     HEAD        LayerNorm(64) -> Linear(64 -> 32) -> ReLU -> Dropout -> Linear(32 -> 2)    (a mean and a spread)
   |
   | expert dropout 0.2: while training an expert is switched off for a molecule with probability 0.2
   v
 GATE   the average of the visible tokens (64) -> Linear(64 -> 21 x 4) + a learned bias
        softmax over the 4 experts for each attribute            (the weights sum to 1)
   |
   v
 for each attribute:  mean   = sum over experts of  gate weight x expert mean
                      spread = sum over experts of  gate weight x expert spread
 21 means, then 21 spreads = 42 outputs
   |
   v
 after the network:  the spread is mixed with a curve of the predicted mean
   spread = weight x curve(mean) + (1 - weight) x network spread
```

Both subchallenges train on the high and the low concentration ratings, and the model is given the dilution of
each rating as an input, because the ratings depend on it. Validation and test are scored at high concentration
only.

### Train, plot and predict

Train the models. Each writes a checkpoint, its metrics and predictions, and a `train.log`, and reports the
official test score:

```bash
poetry run ds-train                   # Subchallenge 2, then Subchallenge 1
poetry run ds-train --challenge q1    # one subchallenge only (q1 or q2)
poetry run ds-plot                    # draw the figures from the saved outputs
```

Each subchallenge is one function, `train_q2` in `scripts/core/train/q2.py` and `train_q1` in
`scripts/core/train/q1.py`, and trains in stages:

1. The epoch of each score group is selected by training on the 338 TrainSet molecules and scoring the 69
   leaderboard molecules after every epoch, with early stopping.
2. Subchallenge 2 only: the spread of a rating is calibrated on the validation predictions.
3. The final model is trained from scratch on the TrainSet and leaderboard molecules together (407) for the
   selected number of epochs.
4. The final model is scored once on the official test set.

A run uses one fixed seed (`RANDOM_SEED = 42` in `scripts/core/splits.py`, which seeds the split order, the model
initialisation, and the batch order), one model, and no folds. Subchallenge 1 smooths every participant's
training rating with a fixed weight, `Q1_BLEND_ALPHA = 0.8` in `scripts/core/train/q1.py`.

Subchallenge 2 writes to `output/q2/` and Subchallenge 1 to `output/q1/`, each with `checkpoints/`, `metrics/`,
`plots/` and `train.log`.

Score a molecule the model has never seen. The command prints the predicted mean and spread for each of the 21
DREAM attributes:

```bash
poetry run ds-predict --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
```

Add `--top 5` to show only the strongest attributes, `--json` for machine readable output, or
`--smiles-file molecules.txt` to score a list.

## Glossary

### Molecule structure and features

| Term | Meaning in this project |
|---|---|
| **Atom** | A unit of an element in a molecule, such as carbon or oxygen. |
| **Element counts** | Counts of carbon (C), hydrogen (H), nitrogen (N), oxygen (O), phosphorus (P), sulfur (S), sodium (Na), chlorine (Cl), and iodine (I). These are separate numeric model features. |
| **SMILES** | Text notation for a molecule’s atoms and bonds. The model reads the RDKit-canonicalized SMILES string. |
| **Canonical SMILES** | A consistent ordering of a molecule’s SMILES representation, generated by RDKit. Canonicalization makes strings easier to compare; it does not verify that the source assigned the right structure to a CID. |
| **Stereochemistry** | The three-dimensional arrangement of atoms around parts of a molecule. Isomeric SMILES can encode this when the source structure includes it. |
| **RDKit** | The open-source chemistry toolkit used here to parse SMILES, calculate molecular features, and generate fingerprints. [RDKit documentation](https://www.rdkit.org/docs/GettingStartedInPython.html) |
| **Molecular descriptor** | A calculated number that summarizes one property of a molecule, such as atom count or estimated lipophilicity. |
| **Heavy atom** | Any atom other than hydrogen. |
| **Aromatic ring** | A ring with a stable shared electron system, such as the ring in benzene. |
| **Aliphatic ring** | A ring that is not aromatic. |
| **Rotatable bond** | A bond that can rotate and change the molecule’s shape. RDKit follows its own rules for which bonds count. |
| **Hydrogen-bond donor / acceptor** | A donor can provide a hydrogen for a hydrogen bond; an acceptor can receive one. RDKit estimates their counts from the structure. |
| **TPSA** | Topological polar surface area: an estimate of the molecule’s polar surface, which relates to how it interacts with water and other polar substances. |
| **MolLogP** | RDKit’s estimated log partition value between oil-like and water-like environments. Larger values generally indicate greater preference for oil-like environments. |
| **Morgan fingerprint** | A fixed-length vector summarizing small atom-and-bond neighborhoods in a molecule. Our RDKit version uses radius 2 and 2048 binary bits, with stereochemistry enabled. A bit set to 1 means a hashed neighborhood is present. Different neighborhoods can sometimes map to the same bit. [RDKit Morgan fingerprint documentation](https://www.rdkit.org/docs/GettingStartedInPython.html) |
| **Morgan radius 2 / ECFP4-style** | The fingerprint considers atom neighborhoods grown out to two bonds from each center atom. Radius 2 is commonly called ECFP4-style because the largest neighborhood spans up to four bonds across. |
| **Fingerprint bit** | One position in the 2048-value Morgan vector. It records whether a hashed local structure was found; it does not correspond to a specific element. |

### DREAM ratings and model terms

| Term | Meaning in this project |
|---|---|
| **CID** | PubChem Compound Identifier: the number used to match a molecule across the source files. |
| **DREAM attribute** | One of 21 properties people rated: intensity, pleasantness, and 19 odor descriptors such as sweet, fish, or woody. |
| **Bakery** | An odor reminiscent of bread, cake, or other baked food. |
| **Sweet** | An odor associated with sweet foods, candy, or sugar. |
| **Fruit** | An odor reminiscent of fruit. |
| **Fish** | An odor reminiscent of fish or seafood. |
| **Garlic** | An odor reminiscent of garlic. |
| **Spices** | An odor reminiscent of spices or seasoning. |
| **Cold** | An odor associated with a cool sensation, often minty or camphor-like. |
| **Sour** | An odor associated with sour foods or vinegar. |
| **Burnt** | An odor reminiscent of something scorched or burned. |
| **Acid** | A sharp, acidic-smelling odor; this is a sensory label, not a measurement of pH. |
| **Warm** | An odor associated with a warm sensation or impression. |
| **Musky** | An odor reminiscent of musk. |
| **Sweaty** | An odor reminiscent of sweat or body odor. |
| **Ammonia / urinous** | An odor reminiscent of ammonia or urine. |
| **Decayed** | An odor reminiscent of rotting or decomposing material. |
| **Wood** | An odor reminiscent of wood. |
| **Grass** | An odor reminiscent of grass or freshly cut greenery. |
| **Flower** | An odor reminiscent of flowers. |
| **Chemical** | An odor people perceive as chemical-like, such as some solvents or cleaners. |
| **Intensity / strength** | How strong the odor was rated. The source column named `Intensity` with values `low` and `high` identifies the stimulus level; the `INTENSITY/STRENGTH` column contains the rating. |
| **Valence / pleasantness** | How pleasant the odor was rated. Here, “valence” means emotional pleasantness, not an atom’s chemical valence. |
| **Odor descriptor** | A word people used to describe an odor, such as bakery, fruit, garlic, or chemical. Its rating measures how strongly that quality was perceived. |
| **Mean target** | The average high-intensity rating for one molecule and attribute. Repeated ratings from each person are averaged first, then averaged across people. |
| **Standard-deviation target** | How much people’s average ratings differed for one molecule and attribute. The custom model predicts this alongside each mean. |
| **Feature scaler** | A saved transformation that puts numeric inputs on comparable ranges. `ds-build` fits it on training molecules only, and the model's checkpoint keeps a copy. Binary Morgan bits are not scaled. |
| **Target scaler** | A transformation used to put rating outputs on a training-friendly scale. `ds-build` fits it on training molecules only. Predictions are converted back to rating units before scoring. |
| **Training split** | The 338 TrainSet molecules, used to update model weights. |
| **Validation split** | The 69 DREAM leaderboard molecules, used to track performance and select the best epoch. The epoch-selection model does not train on them. Once the epoch is chosen, the final model is retrained from scratch on the training and validation molecules together (407 molecules), as the challenge teams could, and is scored once on the test set. |
| **Test split** | The official 69-molecule DREAM test set, used for the final score. |
| **DREAM Q2 mean Z-score** | The official Subchallenge 2 score. It averages six correlation-based Z-score components for means and standard deviations. Higher is better; it is not a percentage. |
| **Message passing** | The graph network step in which every atom updates its own vector from its neighbours' vectors and the bonds and distances to them. Three rounds let information travel across the molecule. |
| **Attention** | A learned weighting. For each DREAM attribute, the model decides how much to rely on each of the input tokens (eight in Subchallenge 1, nine in Subchallenge 2) when predicting it. |
| **Model branch** | One path that processes one input type. This model has separate branches for the molecule graph, the Dragon descriptor families, the RDKit, EPI Suite, Morgan, and charge features, the odor-name features, and the structure features; their learned representations are the tokens that each attribute attends over. |

## Project proposal

See [docs/mai642_project_proposal.md](docs/mai642_project_proposal.md).
