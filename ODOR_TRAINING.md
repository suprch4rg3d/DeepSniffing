# Training the odor decoder on another machine

The decoder (`scripts/core/odor/`) writes molecules for odor words and names the odor words of a molecule. It
trains in two stages: stage 1 on the molecules alone, stage 2 on the molecules with odor words. Both stages use a
GPU when there is one and the CPU otherwise.

## What is in the repository

| path | what |
|---|---|
| `data/processed/odor/structures.csv` | stage 1: every molecule (CID, SMILES, SELFIES, split) |
| `data/processed/odor/odor_dataset.csv` | stage 2: the molecules with odor words, and the DREAM molecules as split `dream` |
| `data/processed/odor/vocabulary.json` | every token: special tokens, SELFIES symbols, odor words |
| `data/processed/odor/molecules.csv`, `words.csv`, `dropped.csv` | the cleaned word tables and what was left out |
| `data/raw/pubchem/odor_pubchem.csv` | the PubChem lookups, so that `ds-build` does not ask again |

`data/processed/odor/selfies_variants.csv` (155 MB, the same molecules written in other atom orders) is **not** in
the repository, because it is too big for GitHub. It is rebuilt from the tables above, and the result is always the
same.

## Steps

```
git clone git@github.com:suprch4rg3d/DeepSniffing.git
cd DeepSniffing
git checkout krasodev
poetry install

# 1. the variants (about 5 minutes on 8 or more cores; every variant is checked, and the file is written
#    safely and read back before it replaces the old one)
python -m scripts.preprocessing.odor_variants

# 2. train: stage 1, then stage 2, then stage 2 from random weights to measure what stage 1 gives
python -m scripts.core.odor --stage 1
python -m scripts.core.odor --stage 2
python -m scripts.core.odor --stage 2 --scratch
```

The training saves its state after every epoch and starts again from it when the same command is run again, so
an interrupted run loses at most one epoch. Results are in `output/odor/<stage>/` (`best.pt`, `history.csv`) and
in `output/odor/train.log`. `--size small` trains a smaller model into `output/odor_small/`, `--no-variants` trains
on the canonical writing only, `--max-epochs N` stops each stage early.

## What was found so far

- Stage 1 on the laptop (CPU) reached validation loss 0.887 at epoch 13 of 40 and was still improving.
- The first run, without the variants and with a constant learning rate, got stage 2 to a validation loss of
  about 2.41 for the pretrained and the from-scratch model alike: the pretraining gave nothing there, and stage 2
  overfitted. The variants, a learning-rate schedule, a higher stage 2 dropout, layer-wise learning rates and
  early stopping on word precision were added to address that; they have not been tested on stage 2 yet.
- The old DREAM code (Q1/Q2) is untouched but its downloads are commented out in `scripts/data/prepare_data.py`,
  and the old commands (`ds-inspect`, `ds-plot`, `ds-predict`, `ds-experiments`) are commented out in
  `pyproject.toml`. `ds-prepare`, `ds-build` and `ds-train` are the commands of the odor decoder.
