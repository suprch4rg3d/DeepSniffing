# Subchallenge 1: the setup

Subchallenge 1 predicts every participant's own rating of the 21 attributes: 21 x 49 outputs. The numbers below are
those of the saved model (`output/q1/checkpoints/custom_model.pt`), the feature selection and the code.

## Inputs

The model reads eight tokens per molecule. Each token is one vector that a small network makes from a group of
columns, and the counts below are the columns of each group that the feature selection keeps
(`data/processed/dream_pyrfume/feature_selection.csv`, duplicate threshold 0.995).

| token | what it is | columns in the model | candidates before selection |
|---|---|---:|---:|
| graph | atoms and bonds from RDKit, with PubChem 3D distances on the edges (3 message passing layers, 64 units) | 45 atom features, 28 edge features | |
| Dragon, 4 families | composition and properties 88, 3D shape 768, 2D local patterns 289, 2D topology 863 | **2,008** | 4,868 |
| numeric block | RDKit descriptors 7, partial charge summaries 8, EPI Suite 48, Morgan bits 636, concentration 2 | **701** | |
| odor names | our own 4-letter windows of the DREAM odor names 498, and the organizers' name windows 10,110 | **10,608** | 1,606 + 11,785 |
| structure | NSPDK substructures 1,963, NSPDK similarities 1,434, Morgan similarities 1,593 | **4,990** | 6,163 + 1,961 + 1,961 |

- The model has 771,218 parameters in one copy. The checkpoint keeps one copy for each of the three score groups.
- The feature selection drops constant columns, columns that only one molecule has, and a column whose rank
  correlation with a kept column is 0.995 or more.
- The dilution of the row is an input (log10 concentration and a high concentration flag). The validation and test
  files list one dilution per molecule, and every molecule was also rated at 1/1000, where its intensity comes
  from. A molecule listed at 1/10 has the pair 1/10 and 1/1000, so its descriptors are the high rating, flag 1. A
  molecule listed at 1/100,000 has the pair 1/1000 and 1/100,000, so its descriptors are the low rating, flag 0.
  Intensity is asked at 1/1000 with the other flag of the pair. The data supports this: for the molecules listed
  at 1/100,000, pleasantness is about 47, like the training's low rows, and intensity about 55, like the 1/1000
  high rows.

## Architecture

Each group of columns goes through its own small network and comes out as one token, a vector of 64 numbers. The
tokens are then combined by attention, and a small head for each attribute turns the result into the predictions.

```
 Dragon family 1..4  (88, 768, 289, 863 columns)  each: layer to 16 numbers, ReLU, dropout
      |                                            |-> the four 16-number vectors are each projected to 64: 4 tokens
      |                                            |-> and merged into one 32-number "Dragon context"
      |                                                        |
 atoms (45 features) and bonds (28 features)                   | the Dragon context is added to every step
      -> layer to 64 per atom                                  v
      -> 3 rounds of message passing -------------------- (64 units each)
      -> average over the atoms of the molecule --------> graph token
 numeric block (701)   layer to 64, ReLU, dropout, layer to 64, ReLU ------> numeric token
 odor names (10,608)   layer to 32, ReLU, dropout, then to 64 --------------> names token
 structure (4,990)     layer to 32, ReLU, dropout, then to 64 --------------> structure token

 the 8 tokens --> attention --> one small head for each attribute --> the predictions
```

- **The graph.** Each atom starts as its 45 features, turned into 64 numbers. In each of the 3 rounds an atom
  receives a message from every atom it is connected to. A message is made from the sender's 64 numbers, the 28
  features of the bond or the distance between the two atoms, and the Dragon context. The atom adds up its
  messages and updates itself from its own numbers, the sum and the Dragon context. After 3 rounds the atoms of a
  molecule are averaged into the graph token. The Dragon context is how the Dragon descriptors influence the graph
  as well as being tokens of their own.
- **Attention.** Each of the 21 attributes has a learned query. It is compared with all the tokens, the
  comparison scores are turned into weights that sum to 1, and the attribute reads the weighted mix of the tokens.
  So each attribute decides for itself how much to rely on the graph, on each Dragon family, on the names and on
  the structure features.
- **Subchallenge 1 asks for each participant separately.** Every (attribute, participant) pair has its own query,
  so there are 21 x 49 of them. The head of the attribute gives one number for each participant, and a learned
  offset for each participant and attribute is added. The output is the 21 x 49 ratings.
- **Token dropout.** While training, each token is hidden from the attributes with a probability of 0.15, and at
  least one token always stays. No block can carry the prediction alone.

## Training rules

- **Data:** the 338 TrainSet molecules for training, the 69 leaderboard molecules for validation, the 69 official
  test molecules. The final model is trained again on all 407 molecules.
- **Rows:** the high and the low concentration ratings of every molecule, as separate rows.
- **Intensity:** learned only from the ratings given at 1/1000 (280 of the 338 molecules), and predicted as if
  every molecule were at 1/1000. All other attributes use all rows.
- **Labels:** each participant's rating is blended with a mean of the others' ratings, with weight 0.8 on the
  individual rating (`Q1_BLEND_ALPHA`). The mean weights each other participant by how well they agree with this
  participant over the training ratings of that attribute (their correlation, and nothing for a negative one). A
  participant never counts for themselves. Where nobody has a weight, the plain population mean is used.
- **Loss:** a differentiable version of the official score. Each score group's correlation counts equally per
  point, and correlations are measured inside each concentration group.
- **Regularization:** dropout 0.25, and token dropout 0.15, which hides each input token at random.
- **Optimizer:** AdamW, learning rate 0.0005, weight decay 0.001, gradient clipping at 5, batch 64, up to 300
  epochs, early stopping after 100 epochs without any group improving, seed 42.
- **Epochs:** chosen separately for intensity, valence and the descriptors on the validation molecules. The final
  model trains once up to the latest of them, and a copy is kept at each group's epoch. Each group's predictions
  come from its own copy.

## Result

The saved model scored **34.52** on the official test set (intensity 30.4, valence 22.1, descriptors 51.1) and
29.81 on validation (the mean of the groups' best scores, so slightly optimistic). The published winner's total is
34.18. Reruns of Subchallenge 1 differ by 1 to 2 points.

## What helped and what did not

Helped, in the order they were added: choosing the epoch separately for each score group (test 33.4), token
dropout 0.15 (34.5), the balanced loss, learning intensity only from the ratings at 1/1000, the organizers' name
windows (36.3), the duplicate threshold of 0.995, labelling the held-out rows by their pair, and smoothing each
rating towards the participants who agree with that person (the best validation score).

Did not help, so not in the code: compressing the odor names to 32 numbers or the structure block to 64, a
decaying learning rate, keeping intensity from the names and structure, adding averaged high and low rows, and a
blend weight of 0.2 instead of 0.8 (lower on validation and on test).

Single runs differ by 1 to 2 points, so only the larger moves are reliable: the epoch per group, token dropout,
intensity at 1/1000 and the organizers' name windows.

Two checks explain the limits. Simple models (ridge, random forest) on the Dragon features score about 18 to 19
for intensity on the validation molecules, the same as the network, so intensity is limited by the labels and the
features, not by the size of the network. And our top Dragon features for intensity (ATS2s, ATS1s) and valence
(SssO) are the ones that the DREAM paper names.

## The Subchallenge 1 model is locked

This result is the best of Subchallenge 1, so its architecture does not change. `tests/test_q1_architecture.py`
checks it against the saved model in `output/q1/`: the numbers that define the architecture (64 hidden units, 3
message passing layers, dropout 0.25, token dropout 0.15), every layer of the model that the code builds for
Subchallenge 1 with its shape (no layer may be added or resized), that the saved weights load into it, and its
outputs for one fixed input. Work on Subchallenge 2 uses layers and settings that exist only when the challenge is
`q2` (the neighbour token, the experts, the calibration loss, the spread curve), and a Subchallenge 1 model never
contains them.
