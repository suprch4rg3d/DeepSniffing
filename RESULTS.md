# DeepSniffing: results

The task is to predict how people rate the smell of a molecule from its structure, on the DREAM Olfaction
Prediction Challenge (Keller et al., *Science* 2017, with the ratings of Keller and Vosshall, 2016). The score is
the official one: the mean of the Z-scores of the correlations over the score groups. **Higher is better.** Every
number is on the official test set of 69 molecules unless it says validation. The details are in `Q1_SETUP.md` and
`Q2_SETUP.md`.

## Summary

| | our score (test) | published reference |
|---|---:|---|
| **Subchallenge 1** (each participant's own rating) | **34.52** (intensity 30.4, pleasantness 22.1, descriptors 51.1) | winner 34.18 |
| **Subchallenge 2** (population mean and spread) | **10.05** for the saved model, 9.9 and 9.7 for two more runs (saved model: intensity 6.69 / 4.20, pleasantness 5.45 / 3.65, descriptors 21.25 / 19.06, as mean / spread) | winner 8.87, all-team aggregate 9.17, post-challenge ensemble 9.92 |

- **Subchallenge 2** is above the challenge winner and the aggregate, and level with the best later result. Over
  three runs of the final setup the test score was 10.05, 9.90 and 9.68. The gap to the winner (8.87) and the
  aggregate (9.17) is larger than the noise, and the post-challenge ensemble (9.92) is inside it.
- **Subchallenge 1** is level with the winner. 34.52 against 34.18 is a gap of 0.3, and reruns differ by 1 to 2
  points.
- Each result is one network trained on the challenge data only. There is no ensemble of separately trained models
  and there are no outside molecules.

## Data and split

476 molecules with DREAM ratings, split as the challenge did: 338 for training, 69 leaderboard molecules for
validation (choosing epochs and the spread settings) and 69 official test molecules. The final model is trained
again on the 407 training and validation molecules. No test molecule is used for training or for choosing an epoch.

Each molecule has the graph of its atoms and bonds with 3D distances (PubChem structures), 2,008 Dragon
descriptors in four families, a numeric block (RDKit, EPI Suite, Morgan bits), windows of its odor name (our own
and the organizers'), and structure features (NSPDK substructures and similarities, Morgan similarities). The
Dragon values are the ones that the organizers published, so the model scores the 476 DREAM molecules only.

## The model

One neural network: 771,000 parameters for Subchallenge 1 and 943,000 for Subchallenge 2. A message passing network
reads the molecule graph. The graph and each descriptor group become one token, and each of the 21 attributes has
its own query that attends over the tokens and feeds its own small head. Tokens are hidden at random during
training, so no block carries the prediction alone.

- **Subchallenge 1** predicts every participant's rating (21 x 49 outputs). Its architecture is locked by
  `tests/test_q1_architecture.py` against the saved model.
- **Subchallenge 2** predicts 21 means and 21 spreads. Four experts, each a small attention network over its own
  view (the Dragon descriptors, the graph with the names, the similarity features, the labels of the 20 most
  similar training molecules), are mixed by a gate for every attribute inside the same network. Every expert also
  reads the concentration.

## What made the difference

**Subchallenge 1:** choosing the epoch separately for intensity, pleasantness and the descriptors (test 33.4),
token dropout (34.5), learning intensity only from the ratings at 1/1000, the organizers' name windows (36.3), and
smoothing each participant's rating towards the people who agree with them (the best validation score).

**Subchallenge 2** takes the same model and adds intensity at 1/1000, a curve that gives the spread of a rating from
its mean (large for middle ratings, small near the ends), a neighbour token, a calibration loss that pins the level
and scale of the predictions, spread weights chosen on validation, the experts, and a rule for choosing epochs:
each score group takes the epoch where its validation score, averaged over 7 epochs, peaks, and not the epoch of a
single lucky peak. Together they took the test score from 9.0 to 9.7 to 10.05.

## What did not help

Each idea was run once or a few times and gave nothing beyond the noise of about 0.3, so none is in the code:
model width, a fifth or sixth expert, test-time dropout, intensity seen by the other attributes' heads, a learned
or shifted spread curve, every participant's own rating as an extra loss, label-aware mixup, extra features of the
same molecules (MACCS, Coulomb matrix, charges), pretrained Uni-Mol embeddings, neighbours by 3D shape, and other
learning rates, weight decays and neighbour counts. Weighting molecules by the number of raters is not possible,
because every molecule has exactly 49.

Two runs used odor words from outside data (Leffingwell and The Good Scents Company, 4,655 molecules, every DREAM
molecule removed): the predicted words as an extra token (test 9.71) and a graph encoder pre-trained on them (9.64).
Neither beat the saved model (10.05). They were removed, and the saved model uses the challenge data only.

The network already gets the structure information that these ideas offered from several sources. The limit is
about 338 noisy rated molecules, not what the network knows about chemistry. The details are in `Q1_SETUP.md` and
`Q2_SETUP.md`.

## How reliable the numbers are

- **One run is good to about 0.3** in Subchallenge 2. Seven seeds of an earlier setup gave a standard deviation of
  0.18 (range 9.32 to 9.82), and the same setting run three times scored 10.05, 9.90 and 9.68. Training with a
  fixed seed does not give the same result twice.
- **The saved model is a good run.** Its 10.05 owes a lot to the intensity spread (4.2, against 2.4 to 3.5 in other
  runs). For one run, expect about 9.7 to 10.
- **Subchallenge 1** reruns differ by 1 to 2 points.
- **Differences below about 0.4** in Subchallenge 2 cannot be told from luck with single runs.
- **Whole runs were compared by their test score too**, so the test numbers are slightly optimistic.

## Reproduce

```bash
poetry install
poetry run ds-prepare                       # download the raw data
poetry run ds-build                         # the tables, the split and the scalers
poetry run ds-inspect                       # check every input
poetry run ds-train --challenge q2          # Subchallenge 2, writes output/q2/
poetry run ds-train --challenge q1          # Subchallenge 1, writes output/q1/
poetry run ds-plot                          # draw the figures from the saved outputs
poetry run ds-predict --smiles 702          # scores one of the 476 DREAM molecules
poetry run pytest                           # the tests, including the lock on the Subchallenge 1 model
```

The training settings are fixed in the code. `ds-predict` gives the same answers as the stored test predictions for
the DREAM molecules, and cannot score a molecule outside the 476.

## Limits

- The model scores only the 476 DREAM molecules, because Dragon is commercial software and its values exist only
  for those.
- The results are single runs on one split. We did not cross-validate.
- Pleasantness and the two spreads remain the weakest groups. The spread of intensity depends on the predicted
  mean being right near the middle of the scale, where small errors matter most.

## Sources

- Keller et al., *Predicting human olfactory perception from chemical features of odor molecules*, Science 2017
  (https://pmc.ncbi.nlm.nih.gov/articles/PMC5455768/); Keller and Vosshall, 2016.
- Mayhew et al., PNAS 2022 (volatility and hydrophobicity); Khan et al., 2007 (pleasantness axis); Lee et al.,
  Science 2023 (Principal Odor Map).
- Post-challenge ensemble: Information Fusion 2025, https://doi.org/10.1016/j.inffus.2025.103038.
