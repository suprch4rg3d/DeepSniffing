# Subchallenge 2: the setup

Subchallenge 2 predicts, for each of the 21 attributes, the mean and the spread (standard deviation) of the 49
participants' ratings: 42 outputs. There are six score groups: the mean and the spread of intensity, of
pleasantness and of the descriptors. The numbers below are those of the saved model
(`output/q2/checkpoints/custom_model.pt`) and the code.

## Inputs

The model reads nine tokens per molecule. Each token is one vector that a small network makes from a group of
columns, and the counts below are the columns of each group that the feature selection keeps
(`data/processed/dream_pyrfume/feature_selection.csv`). The first eight are the same as in Subchallenge 1.

| token | what it is | columns in the model |
|---|---|---:|
| graph | atoms and bonds from RDKit, with PubChem 3D distances on the edges (3 message passing layers, 64 units) | 45 atom features, 28 edge features |
| Dragon, 4 families | composition and properties, 3D shape, 2D local patterns, 2D topology | **2,008** |
| numeric block | RDKit descriptors 7, partial charge summaries 8, EPI Suite 48, Morgan bits 636, concentration 2 | **701** |
| odor names | our own 4-letter windows of the DREAM odor names, and the organizers' name windows | **10,608** |
| structure | NSPDK substructures, NSPDK similarities, Morgan similarities | **4,990** |
| neighbours | what the 20 most similar training molecules were rated, once by Morgan and once by NSPDK similarity | **42** numbers |

- The model has about 943,000 parameters.
- The neighbour token is the similarity-weighted average (similarity to the power 4) of the labels of the 20 most
  similar other training molecules, 21 numbers for each of the two similarities. A molecule never sees its own
  label.
- The dilution of the row is an input. Validation and test molecules are labelled by their pair, as in
  `Q1_SETUP.md`: a molecule listed at 1/10 has its descriptors at the high rating, one listed at 1/100,000 has
  them at the low rating, and intensity is the rating at 1/1000 for all of them.

## Architecture

The branches that make the tokens are those of Subchallenge 1 (`Q1_SETUP.md`, Architecture): the four Dragon
families each go through a layer to 16 numbers and are merged into a Dragon context that is added to every step of
the 3 rounds of message passing on the molecule graph, the numeric block, the names and the structure features each
go through their own layers, and every token is a vector of 64 numbers. Subchallenge 2 adds a branch for the
neighbour token, so there are nine tokens, and it combines them differently:

- **Two outputs for each attribute.** Each attribute has one learned query and one small head that gives its mean
  and its spread. There are no participant-specific queries.
- **Four experts.** Each expert is an attention network of its own over a view of the molecule: the Dragon tokens,
  the graph with the names, the structure features, or the neighbour token. Every expert also reads the numeric
  token, which holds the concentration. For each attribute, each expert has its own query and head. A gate
  mixes the four answers for each attribute: it looks at the average of the visible tokens and gives weights that
  sum to 1. The experts are part of one network and are trained together in one run.
- **Dropout of tokens and experts.** While training, each token is hidden with a probability of 0.3, and each expert
  is switched off for a molecule with a probability of 0.2. At least one token and one expert always stay.
- **The spread follows the mean.** The spread of the ratings is large for middle means and small near the ends of
  the 0 to 100 scale. After the network answers, a curve of the predicted mean (fitted on the training molecules) is
  mixed into the predicted spread: `weight x curve(mean) + (1 - weight) x the network's own spread`. The weights
  are fixed (`STD_FROM_MEAN_WEIGHTS` in `scripts/core/data/targets.py`): 1.0 for intensity, 0.03 for pleasantness
  and 0.35 to 1.0 for the descriptors.

## Training rules

- **Data:** the 338 TrainSet molecules for training, the 69 leaderboard molecules for validation, the 69 official
  test molecules. The final model is trained again on all 407 molecules.
- **Rows:** the high and the low concentration ratings of every molecule, as separate rows. The targets of a row
  are the mean and the standard deviation of the 49 participants' ratings.
- **Intensity:** learned only from the ratings given at 1/1000, and predicted at that dilution. The curve of the
  intensity spread is fitted on the same rows. All other attributes use all rows.
- **Loss:** a differentiable version of the official score, with each score group counting equally per point of
  correlation and correlations measured inside each concentration group. A calibration loss is added
  (`Q2_CALIBRATION_WEIGHT = 2` in `scripts/core/train/q2.py`): the score loss compares molecules only, so the level
  of the predictions drifted, and intensity at 1/1000 was predicted below zero for 16 of 69 test molecules. A
  squared error on the standardised predictions pins the level and the scale.
- **Regularization:** dropout 0.25, token dropout 0.3, expert dropout 0.2.
- **Optimizer** (`Q2_TRAIN_CONFIG`): AdamW, learning rate 0.001, weight decay 0.001, gradient clipping at 5, batch
  32, up to 300 epochs, early stopping after 100 epochs without any group improving, seed 42.
- **Epochs:** chosen separately for each of the six groups. The epoch of a group is where its validation score,
  averaged over 7 epochs, is highest (`Q2_EPOCH_WINDOW`), not where one epoch peaked. The final model trains once up
  to the latest of them, and a copy is kept at each group's epoch.
- **Fitted on the validation molecules** after the epochs are chosen, with the predictions of each group's chosen
  epoch: how far the predicted means are pulled towards their centre before the curve is read (0.83 for intensity,
  1 for the others), and how much of the spread comes from the curve (0.75 for intensity, 0.5 for the others).

## Result

The saved model scored **10.05** on the official test set and 9.64 on validation: intensity 6.69 / 4.20,
pleasantness 5.45 / 3.65 and descriptors 21.25 / 19.06, as mean / spread. Two more runs of the same setup scored
9.90 and 9.68.

| | test score |
|---|---:|
| the saved model | 10.05 |
| published challenge winner | 8.87 |
| published all-team aggregate | 9.17 |
| published post-challenge ensemble | 9.92 |
| stored reference predictions | 9.63 |

How the test score grew, one run each: the Q1 setup 9.0, with intensity at 1/1000 and the curve 9.2, with the
calibration loss and the pair labelling 9.5, with the spread weights chosen on validation 9.58, with the neighbour
token and the experts 9.66, with the learning rate of 0.001 and the 7-epoch rule 9.7 to 10.05.

A single run is good to about 0.3: the same setting run twice gave 9.98 and 9.69, and seven seeds of the setup
before the experts had a standard deviation of 0.18. The saved model is a good run, because its intensity spread
(4.2, against 2.4 to 3.5 in other runs) is high. Expect 9.7 to 10.

`ds-predict` reproduces the stored test predictions for the DREAM molecules. It reads Subchallenge 2 models only.

## What did not work

Each was run once or a few times and gave nothing beyond the noise of about 0.3, so none is in the code.

- Intensity-conditioned descriptor heads, and a learned spread curve inside the network.
- Test-time dropout, averaging many passes of one model (+0.06).
- Model width 32 or 128 instead of 64; six experts instead of four; a fifth expert on transport features.
- Neighbours by 3D shape, and pretrained Uni-Mol embeddings (as inputs or as neighbours): clearly weaker than
  Morgan and NSPDK neighbours.
- Extra features of the same molecules (MACCS keys, Coulomb matrix, charges).
- Every participant's own rating as an extra loss, label-aware mixup, a running average of the weights, and a token
  per attribute built from its own best Dragon columns (it overfit the molecules it was ranked on).
- Other learning rates (0.0003, 0.002), weight decay 0.003, 10 or 40 neighbours, neighbour power 8, expert dropout
  0 or 0.4, token dropout 0.45 or 0.6, calibration weight 0.5 or 6: none clearly better than the setting kept.
- Leaving one input block out (structure, name windows, Morgan bits) cost nothing measurable alone, and leaving
  all three out cost about 0.5 on validation and 0.75 on test, so all inputs are kept.
- Outside odor data. Leffingwell and The Good Scents Company list odor words for thousands of molecules that are
  not in DREAM. With every DREAM molecule removed, the predicted words as an extra token (validation 9.45, test
  9.71) and a graph encoder pre-trained on them (9.40, 9.64) did not beat the saved model (9.64, 10.05). The network
  already has that information from the fingerprints, the names and the neighbours. The data and the code were
  removed, and the saved model uses the challenge data only.
