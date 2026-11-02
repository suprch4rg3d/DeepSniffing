# DeepSniffing

Computers already see, hear, and talk — we're teaching them to _smell_.

Predicting human odor profiles from molecular structures with machine learning.
Semester project for MAI642 — Deep Learning.

## Project overview

DeepSniffing explores whether a molecule's chemical structure can help predict
how people perceive its odor. It is a team project for MAI642 — Deep Learning.

The project uses molecule structures and human odor ratings from the DREAM
Olfaction Prediction Challenge, together with molecule information from
Pyrfume and PubChem. RDKit is used to parse structures and calculate chemical
features.

The prediction task covers 21 DREAM attributes: odor intensity, pleasantness,
and 19 odor descriptors. The project compares ways to represent molecules,
including canonical SMILES, calculated RDKit descriptors, and Morgan
fingerprints. Models are evaluated with the DREAM challenge scoring method so
their results can be compared with released benchmark results.

The repository contains scripts for preparing and inspecting data, generating
molecular features, training models, and reviewing results. Model results
should be considered alongside the split and evaluation method used in each
experiment.

## Project state

The repository contains the data preparation pipeline, molecular feature
generation scripts, model training code, and DREAM benchmark references.

## Repository layout

```
DeepSniffing/
├── data/
│   ├── raw/
│   │   ├── dream-olfaction-prediction/data/
│   │   │   ├── TrainSet.txt, TestSet.txt       DREAM ratings for training and test
│   │   │   ├── CID_* and dilution_*            DREAM challenge molecule lists
│   │   ├── pyrfume-data/
│   │   │   ├── keller_2016/molecules.csv       Keller molecule metadata
│   │   │   ├── molecules/molecules.csv         Pyrfume molecule metadata
│   │   │   └── prediction_targets/keller_2016.csv  Keller odor ratings
│   │   └── pubchem/
│   │       └── dream_keller_compound_properties.csv  PubChem properties and source SMILES
│   └── processed/dream_pyrfume/
│       ├── molecule_labels.csv                 Base molecule-level dataset
│       ├── dream_ratings_with_smiles.csv       DREAM ratings joined with molecule data
│       ├── molecule_labels_with_split.csv      Base dataset with train/validation/test split
│       ├── morgan_fingerprints.csv              2048-bit Morgan fingerprints keyed by CID
│       ├── molecules_enriched.csv              Base molecules plus RDKit descriptors
│       ├── molecules_enriched_with_split.csv   Enriched dataset with split
│       ├── scalers/                             Saved preprocessing scalers
│       └── inspection/                          Generated PNGs (created by dataset_inspector.py; not committed)
├── scripts/
│   ├── core/
│   │   ├── data.py                             Canonical data loading and the target definitions
│   │   ├── metrics.py                          Official DREAM Q2 scoring
│   │   ├── splits.py                           Canonical train/validation/test assignment
│   │   ├── train.py                            Shared training loop with early stopping
│   │   └── plots.py                            Figures produced by a training run
│   ├── data/
│   │   └── prepare_data.py                     Download raw DREAM/Pyrfume data and PubChem properties
│   ├── preprocessing/
│   │   ├── dream_dataset.py                    Build molecule and DREAM-rating datasets
│   │   ├── morgan_fingerprints.py              Generate Morgan fingerprints from canonical SMILES
│   │   ├── enrich_molecules.py                 Generate RDKit molecular features
│   │   ├── split_dataset.py                    Assign molecule-level data splits
│   │   ├── dataset_inspector.py                Create dataset summary plots
│   │   ├── feature_scaler.py                   Fit the standard feature scaler
│   │   ├── enriched_feature_scaler.py          Fit the enriched feature scaler
│   │   ├── target_scaler.py                    Fit the DREAM target scaler
│   │   └── smiles_tokenizer.py                 Tokenize canonical SMILES
│   └── training/
│       ├── custom_model.py                     SMILES CNN, descriptor, and Morgan hybrid model
│       ├── predict.py                          Score novel SMILES strings
│       ├── run_experiments.py                  Ablation matrix across models and input branches
│       ├── cnn.py                              SMILES-only CNN baseline (not implemented yet)
│       └── transformer.py                      Transformer baseline (not implemented yet)
├── output/custom_model/
│   ├── checkpoints/                            Trained model, vocabulary, and fitted scalers
│   ├── plots/                                  Training and metric comparison PNGs
│   └── metrics/                                Test predictions and metric reports
├── results/baseline/dream_challenge/
│   ├── challenge_2_testset_*.txt               Stored DREAM test-set predictions
│   ├── official_testset_gold_standard.tsv      DREAM ratings in scorer input format
│   ├── official_testset_reference_score.json   Calculated reference metrics
│   ├── published_challenge_scores.json         Published DREAM benchmark scores
│   └── official_scoring/                       Organizer Q2 scorer and submission validator
├── tests/                                      Pytest suite for the data contract and scoring
├── docs/
│   └── mai642_project_proposal.md              Course project proposal
├── pyproject.toml                              Poetry project and dependency configuration
├── poetry.lock                                 Locked dependency versions
├── requirements.txt                            Pip dependency list
└── README.md                                   Setup, data preparation, and training guide
```

## Requirements

- Python 3.13 (>=3.13, < 3.14 as declared in `pyproject.toml`)
- [Poetry](https://python-poetry.org) (2.x) — the project uses poetry for
  dependency management; package mode is disabled, so `poetry install` only
  sets up the dependency environment, not the project itself.

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

## Data

### Build the base datasets from raw data

From the repository root, run these commands in order. The first downloads any
missing DREAM and Pyrfume files, then fetches PubChem compound properties. It
skips source files that are already present and reuses previously fetched
PubChem records.

Scripts are executed as modules from the repository root, which is why every
command below uses `python -m`. Running a file path directly, such as
`python scripts/data/prepare_data.py`, does not resolve the `scripts` package.

The commands are written for macOS/Linux; on Windows the same commands work
unchanged in PowerShell or cmd.

If the processed datasets are already included in the repository, you can use
them directly and skip rebuilding.

```bash
poetry run python -m scripts.data.prepare_data
poetry run python -m scripts.preprocessing.dream_dataset
poetry run python -m scripts.preprocessing.morgan_fingerprints
poetry run python -m scripts.preprocessing.split_dataset
poetry run python -m scripts.preprocessing.dataset_inspector
poetry run python -m scripts.preprocessing.target_scaler
```

The scripts create:

- `data/raw/pubchem/dream_keller_compound_properties.csv` — PubChem properties
  for DREAM and Keller compounds.
- `data/processed/dream_pyrfume/molecule_labels.csv` — one row per molecule,
  with Keller target ratings and DREAM rating summaries.
- `data/processed/dream_pyrfume/morgan_fingerprints.csv` — 2048-bit,
  radius-2 Morgan fingerprints generated from `rdkit_canonical_smiles` and keyed by CID.
- `data/processed/dream_pyrfume/dream_ratings_with_smiles.csv` — DREAM rating
  rows joined to molecule information.
- `data/processed/dream_pyrfume/molecule_labels_with_split.csv` — the base
  molecule dataset with reproducible 70/15/15 train, validation, and test
  splits (random seed 42).
- `data/processed/dream_pyrfume/inspection/base/` — PNG plots for the base
  dataset.

In the processed datasets, `pubchem_smiles` preserves PubChem's original
SMILES string. RDKit parses that string and writes its canonical isomeric form
to `rdkit_canonical_smiles`, the column to use as model input. The pipeline
does not fetch or include PubChem `ConnectivitySMILES`.

### Optional RDKit descriptor enrichment

The base dataset does not require the extra numeric descriptors. To create a
separate enriched dataset, run these additional commands after the base
`dream_dataset.py` step:

```bash
poetry run python -m scripts.preprocessing.enrich_molecules
poetry run python -m scripts.preprocessing.split_dataset
poetry run python -m scripts.preprocessing.dataset_inspector
poetry run python -m scripts.preprocessing.feature_scaler
poetry run python -m scripts.preprocessing.enriched_feature_scaler
```

This creates `molecules_enriched.csv`, then
`molecules_enriched_with_split.csv`, and saves its plots under
`data/processed/dream_pyrfume/inspection/enriched/`. The base dataset and its
outputs remain separate. The splitter uses `molecules_enriched.csv` whenever
it exists; otherwise it splits `molecule_labels.csv`. The inspector selects
the matching split dataset automatically. To return to the base dataset,
remove `molecules_enriched.csv` and `molecules_enriched_with_split.csv` before
running those two scripts.

Enrichment adds counts for C, H, N, O, P, S, Na, Cl, and I; heavy atoms,
aromatic and aliphatic rings, rotatable bonds, hydrogen-bond donors and
acceptors, topological polar surface area, and estimated logP. Numeric columns
are prefixed with `rdkit_`; the formula and RDKit version are retained for
traceability. The target scaler fits the 21 DREAM per-molecule mean targets:
intensity, pleasantness, and 19 semantic odor descriptors. It uses only
training molecules with DREAM ratings. Repeated DREAM ratings are aggregated
by molecule in preprocessing, so the model predicts population means rather
than individual participant ratings. The feature scalers are separate: one
uses the established descriptor subset and the other uses the expanded
descriptor set. Scalers fit on training rows only and save artifacts under
`data/processed/dream_pyrfume/scalers/`. DREAM standard-deviation summaries and
other `dream_train_*` columns are metadata, not model inputs.

`feature_scaler.py` fits the 13-feature standard set from the enriched split.
`enriched_feature_scaler.py` fits the 17-feature expanded set, including
separate P, Na, Cl, and I counts. Both feature scalers use
columns produced by `enrich_molecules.py`; the descriptor values are calculated
from `rdkit_canonical_smiles`. `target_scaler.py` fits the DREAM mean targets
from the base split, so it is run once.

The element-count columns cover elements present in the current dataset. Some
are very rare, but they remain separate input features for the enriched model.

### Model development

The training code supports experiments with SMILES-based neural networks and
combined molecular inputs. Depending on the experiment, inputs can include
RDKit descriptors and Morgan fingerprints alongside canonical SMILES. Models
predict molecule-level DREAM ratings, and results can be compared with official
challenge benchmarks using the DREAM scoring method.

Training scripts and model outputs are organized under `scripts/training/`
and `output/`. Benchmark reference files and scoring tools are under
`results/baseline/dream_challenge/`. See those directories for the experiment
scripts, saved results, and supporting data.

Train the hybrid model, which writes a checkpoint and reports the official
test score:

```bash
poetry run python -m scripts.training.custom_model
```

Score a molecule the model has never seen. The command prints the predicted
mean and spread for each of the 21 DREAM attributes:

```bash
poetry run python -m scripts.training.predict --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
```

Add `--top 5` to show only the strongest attributes, `--json` for machine
readable output, or `--smiles-file molecules.txt` to score a list.

To download raw sources without querying PubChem, run:

```bash
poetry run python -m scripts.data.prepare_data --download-only
```

Use `--force-download` with `prepare_data.py` to redownload the DREAM and
Pyrfume source files.

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
| **Feature scaler** | A saved transformation that puts numeric inputs on comparable ranges. The custom trainer fits its own feature scaler using only its training fold and stores it with the model. Binary Morgan bits are not scaled. |
| **Target scaler** | A transformation used to put rating outputs on a training-friendly scale. Predictions are converted back to rating units before scoring. |
| **Training split** | Molecules used to update model weights. |
| **Validation split** | Held-out training molecules used to track performance and select the best epoch. They are not used to update model weights. |
| **Test split** | The official 69-molecule DREAM test set, used for the final score. |
| **DREAM Q2 mean Z-score** | The official Subchallenge 2 score. It averages six correlation-based Z-score components for means and standard deviations. Higher is better; it is not a percentage. |
| **1D CNN** | The SMILES branch of the custom model. It scans nearby tokens in the SMILES sequence to learn local patterns. |
| **Model branch** | One path that processes one input type. This model has separate branches for SMILES, RDKit descriptors, and Morgan fingerprints; their learned representations are joined before predicting ratings. |

## Project proposal

See [docs/mai642_project_proposal.md](docs/mai642_project_proposal.md).
