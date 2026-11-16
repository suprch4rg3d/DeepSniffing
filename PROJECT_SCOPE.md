# DeepSniffing: project scope

MAI642 Deep Learning semester project. Team: Emmanouil Oikonomou, Konstantin Krasovitskiy,
Anastasia Kyriakou.

This note records what the project covers.

## Objective

Predict the odor profile that people perceive for a chemical compound from its molecular
structure, as continuous ratings of the 21 attributes scored by the DREAM Olfaction
Prediction Challenge (intensity, pleasantness and 19 odor descriptors).

The project covers both DREAM sub-challenges:

- Sub-challenge 1: each participant's own rating of each attribute (21 x 49 outputs).
- Sub-challenge 2: the mean and standard deviation across participants (42 outputs).

## Data

- DREAM Olfaction Prediction Challenge data, built on the Keller & Vosshall (2016) ratings, from
  49 participants at two concentrations.
- 476 molecules: 338 for training, 69 for the leaderboard (used as validation) and 69 for the
  official test set. Four more molecules appear only in the original Keller data and are unused.
- Molecular data from PubChem (SMILES, computed 3D structures) and Pyrfume, checked against each
  other. Descriptor tables published with the challenge: Dragon (four families), EPI Suite, NSPDK
  substructures and similarities, plus RDKit descriptors, Morgan fingerprints, text windows
  built from the DREAM odor names, and the organizers' 4-letter name windows.

## Model (current working version)

A custom model that combines a molecular graph with descriptor tables. The exact inputs, training
rules and the history of what was tried are in `Q1_SETUP.md`.

- A three-layer message-passing network over the atom graph (64 hidden units). Edges carry the
  bond type and the 3D distance between atoms, taken from PubChem. A summary of the Dragon
  descriptors is added to every message and update step.
- Small branches for the four Dragon families (2,008 columns), the numeric block (RDKit, partial
  charges, EPI Suite, Morgan bits and the concentration inputs), the odor-name windows (our own
  and the organizers', 10,608 columns) and the NSPDK and similarity features (4,990 columns).
- The graph vector and the descriptor encodings form 8 tokens (a ninth, the neighbour token, in Sub-challenge 2). Each of the 21 attributes has its
  own attention query and output head. In Sub-challenge 1 every participant also has a query and
  a rating offset for each attribute.
- About 0.77 million parameters for Sub-challenge 1 in the current setup.

Training: AdamW, learning rate 0.0005, weight decay 0.001, dropout 0.25 and token dropout 0.15,
gradient clipping at 5, seed 42. The loss is a differentiable version of the official DREAM
score, with every score group counting equally per point of correlation. The epoch is chosen
separately for intensity, valence and the descriptors on the validation molecules. Intensity is
learned from the ratings at 1/1000 only. The final model is retrained from scratch on all 407
training and validation molecules, and the official test set is scored once.

## Scope and limitations

- Only the 21 DREAM attributes are modelled. The original Keller data contains more descriptors.
- The data set is small (476 molecules), so overfitting is the main risk and the model is small.
- The 3D information is PubChem's computed structures, used as distances on the graph edges.
  Four molecules in the data have no such structure.
- The code uses a GPU when one is available and the CPU otherwise.

## Deliverables

1. Dataset pipeline: scripts that download the source data and PubChem structures, build the
   graphs, descriptors and fingerprints, select features, and produce the fixed 338/69/69 split.
2. Trained models for Sub-challenge 1 and 2.
3. Evaluation with the official DREAM scoring: each score is a Pearson correlation across
   molecules divided by the null model's standard deviation. Sub-challenge 1 averages 3 scores
   (intensity, pleasantness, descriptors) and Sub-challenge 2 averages 6 (the same groups, for the
   mean and for the standard deviation). Results are compared with the published DREAM winners.
4. Final academic report.

## External odor data is not used

Other odor-label datasets exist (Leffingwell and The Good Scents Company, from the Pyrfume repository). They were
tried in two ways, with every DREAM molecule removed first, and neither gave a gain beyond the noise of a single
run (`Q2_SETUP.md`). They were removed, and the project uses the DREAM data only.
