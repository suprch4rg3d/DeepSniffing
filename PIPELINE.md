# The pipeline

From the raw DREAM data to the scores. The setup of each model is in `Q1_SETUP.md` and `Q2_SETUP.md`, and the
results are in `RESULTS.md`.

## The task

A molecule goes in, and the model predicts how people rate its smell on the 21 DREAM attributes (intensity,
pleasantness and 19 descriptors such as garlic, fruit and fish). The data is the DREAM Olfaction Prediction
Challenge: 476 molecules, each rated by 49 people. The score is the official DREAM Z-score (higher is better).

- **Subchallenge 1 (Q1)** predicts what each of the 49 participants would rate.
- **Subchallenge 2 (Q2)** predicts the panel's average rating, and how much people disagree (the spread).

## The whole flow

Seven commands. Every processed file is named for what it holds.

```
 ds-prepare   download the DREAM, Pyrfume and PubChem files            data/raw/...
      |
 ds-build     build every file the models read, in this order:
      |         molecules.csv                one row per molecule: CID, SMILES, odor name (no ratings)
      |         morgan_fingerprints.csv      2,048 Morgan substructure bits
      |         pubchem_3d_coordinates.csv   PubChem's 3D position of each heavy atom
      |         molecule_descriptors.csv     RDKit + Dragon descriptors + per-atom partial charges
      |         published_features.csv       EPI Suite, name, NSPDK and similarity features
      |         feature_selection.csv        which columns each model keeps, and why the others go
      |         split.csv                    train 338 / validation 69 / test 69 / unused 4
      |         scalers/                     feature and target scalers, fitted on training molecules only
      |
 ds-inspect   checks every input and draws summary plots (run it before a long training)
      |
 ds-train     trains and scores Q1 and Q2     output/q2/   output/q1/    (model, metrics, predictions, train.log)
 ds-plot      draws the figures of the trained models from output/q1 and output/q2
 ds-predict   scores one DREAM molecule        a printed table
 ds-experiments  reserved for the ablations    (not active; they are recorded in Q1_SETUP.md and Q2_SETUP.md)
```

## The steps, in order

Run these from the repository root. If the processed files are already in the repository, start at `ds-train`.

| # | command | reads | writes | what it does |
|---|---|---|---|---|
| 1 | `ds-prepare` | the internet (DREAM and Pyrfume repositories, PubChem) | `data/raw/...` | downloads the source files that are missing, including the Dragon table, and PubChem's properties and 3D structure of every molecule |
| 2 | `ds-build` | `data/raw/...` | the six tables, `split.csv` and `scalers/` | runs every step in order and stops at the first problem: the tables, then the split (each molecule is train, validation, test or unused, and the order of the training molecules is fixed), then the scalers (fitted on training molecules only, once for the model that picks epochs, 338 molecules, and once for the final model, 407) |
| 3 | `ds-inspect` | everything above | `inspection/` | prints PASS or FAIL for each check and draws the plots; exits with status 1 on a failure |
| 4 | `ds-train` | the tables, the split, the scalers, the raw ratings | `output/q2/`, `output/q1/` | trains, selects epochs, retrains, scores the test set once, saves the model |
| 5 | `ds-plot` | `output/q2/`, `output/q1/` | `output/*/plots/` | draws the figures from the saved metrics, with no retraining |

```bash
poetry run ds-prepare
poetry run ds-build
poetry run ds-inspect
poetry run ds-train                     # both subchallenges; add  --challenge q1  or  q2  for one
poetry run ds-plot
poetry run ds-predict --smiles 702      # a DREAM molecule, by CID or SMILES
```

`ds-train` reads the split and the scalers from their files. A file made for other molecules, other columns or
another Q1 blend weight than the run uses stops the run and says to run `ds-build` again, so a stale file never
changes what a model trains on.

## What the model reads (one molecule)

| token | what it is |
|---|---|
| graph | the atoms and bonds, with the 3D distance of every atom pair within 6 angstrom (PubChem structure) |
| Dragon, 4 tokens | the Dragon descriptors, in four families: composition, 2D topology, 2D local patterns, 3D shape |
| numeric block | RDKit descriptors, partial-charge summaries, EPI Suite values, Morgan bits, and the dilution |
| odor names | windows of the chemical's name (ours and the organizers') |
| structure | NSPDK substructures and the similarity tables |
| neighbours (Q2 only) | what the 20 most similar training molecules were rated |

Every attribute has its own query that attends over these tokens, and its own small head. Tokens are hidden at
random while training, so no single block carries the prediction. The network is in
`scripts/training/custom_model.py`, and each subchallenge builds its own variant of it.

## How a run works (the same for Q1 and Q2)

1. **Split.** 338 training molecules, 69 leaderboard molecules for validation, 69 official test molecules.
   The test molecules are never used for training or for any choice.
2. **Pick epochs.** Train on the 338 and score the 69 validation molecules after every epoch. Each score group
   (Q1: 3, Q2: 6) keeps its own best epoch. In Q2 the epoch is where the score averaged over 7 epochs peaks.
3. **Calibrate (Q2 only).** On the validation predictions, fit how the spread of a rating follows its mean.
4. **Retrain.** Train from scratch on all 407 labeled molecules up to the latest chosen epoch, and keep a copy of
   the weights at each group's epoch.
5. **Score once.** Predict the 69 test molecules with those copies and compute the official DREAM score.

Each subchallenge is one function that follows these steps from top to bottom: `train_q2` in
`scripts/core/train/q2.py` and `train_q1` in `scripts/core/train/q1.py`. What they share (the loss, one epoch,
prediction) is in `scripts/core/train/loop.py`.

## What differs between the two

| | Q1 | Q2 |
|---|---|---|
| predicts | every participant's rating (21 x 49 outputs) | 21 means and 21 spreads |
| training labels | each rating blended with the participants who agree with that person (weight 0.8) | the panel mean and spread |
| extras | none | four experts with a gate, the neighbour token, a curve that gives the spread from the mean, a calibration loss |
| settings | learning rate 0.0005, batch 64, token dropout 0.15 | learning rate 0.001, batch 32, token dropout 0.3 |
| locked by a test | yes (`tests/test_q1_architecture.py`) | no |
| test score of the saved model | 34.52 (published winner 34.18) | 10.05 (published winner 8.87) |

The training settings are constants in `scripts/core/train/` and `scripts/training/custom_model.py`. The only
run-time option is `ds-train --challenge`.

## Where things are

| | |
|---|---|
| code | `scripts/data` (download), `scripts/preprocessing` (creates the tables, the split and the scalers, and inspects them), `scripts/core` (loads them and trains: data, metrics, splits, training, plots), `scripts/training` (model, prediction) |
| the Q2 model and its numbers | `output/q2/`: `checkpoints/`, `metrics/`, `plots/`, `train.log` |
| the Q1 model and its numbers | `output/q1/`: the same |
| published scores and the scoring tools | `data/reference/dream_challenge/` |
| tests | `tests/` (run with `poetry run pytest`) |

A run uses a fixed seed (42), but two runs do not give the same numbers: two runs of Q2 differ by about 0.3.

## What each file does

One rule decides where code lives: **`preprocessing/` creates the files, and `core/` loads and uses them.** So the
command that writes the split is in `preprocessing/` and the code that reads and checks it is in `core/`, and the
same for the scalers.

| file | does |
|---|---|
| `scripts/data/prepare_data.py` | `ds-prepare`: downloads |
| `scripts/preprocessing/dream_dataset.py` | writes `molecules.csv` |
| `scripts/preprocessing/morgan_fingerprints.py` | writes the Morgan bits |
| `scripts/preprocessing/conformers.py` | writes PubChem's 3D coordinates in our atom order |
| `scripts/preprocessing/enrich_molecules.py` | writes the RDKit, Dragon and charge descriptors, and is the `ds-build` command that runs every step |
| `scripts/preprocessing/community_features.py` | writes the published EPI Suite, name, NSPDK and similarity features |
| `scripts/preprocessing/feature_selection.py` | writes which columns are kept |
| `scripts/preprocessing/split_dataset.py` | a step of `ds-build`: writes `split.csv` |
| `scripts/preprocessing/scalers.py` | a step of `ds-build`: writes the scalers |
| `scripts/preprocessing/dataset_inspector.py` | `ds-inspect`: checks every input |
| `scripts/core/splits.py` | the official DREAM lists, and reading and checking `split.csv` |
| `scripts/core/data/` | loading what the commands wrote, one file per job (below) |
| `scripts/core/metrics.py` | the official DREAM scores |
| `scripts/core/train/` | `ds-train`: `train_q2` and `train_q1`, and what they share (below) |
| `scripts/core/plots.py` | `ds-plot`: the figures, drawn from the saved outputs |
| `scripts/training/custom_model.py` | the network |
| `scripts/training/predict.py` | `ds-predict` |
| `scripts/training/run_experiments.py` | `ds-experiments`: reserved, prints where the ablations are recorded |

`scripts/core/data/` has one file for each job: `ratings.py` (the DREAM rating files and the published reference
scores), `features.py` (the molecule tables joined into rating rows), `scaling.py` (the scalers), `targets.py` (the
targets of the two subchallenges and the rows each stage learns from), `neighbours.py` (nearest-neighbour labels
for Q2), `graphs.py` (molecule graphs and batches) and `assemble.py` (`prepare_graph_data`, the loaders of one run).

`scripts/core/train/` has `loop.py` (the loss, one epoch, prediction and the settings of a run), `q2.py` (`train_q2`,
the whole Q2 run including saving its model), `q1.py` (`train_q1`, the same for Q1) and `__init__.py` (the `ds-train`
command and its log).

Every function, class and constant in `scripts/` is used by one of the seven commands.
