# [MAI642-DL] Semester Project Description

Team: Emmanouil Oikonomou, Konstantin Krasovitskiy, Anastasia Kyriakou.

## Title

DeepSniffing: Writing Molecules for Odor Words with a Small Decoder-Only Transformer

## 1. Overview

Computer vision, speech and language have all been rebuilt around deep learning. Smell has not.
Interest is not the problem. A molecule either smells of something or it does not, and no single
measurement says which.

This project approaches the problem as a language problem in both directions. A molecule is written as
a SELFIES string, a text notation in which every string is a valid molecule, and an odor is one token
such as `fruity` or `musk`. A small decoder-only transformer, six layers, eight heads and a width of
256, about 6.3 million parameters, learns to read and write both. The question it answers is: given a
set of odor words, can a model write a molecule that smells like them, and can it name the odor words
of a molecule it has never seen?

The project is built in three steps, and each step stands alone.

1. **Pretraining.** The decoder learns the language of molecules from 3.4 million PubChem structures,
   with no odor information at all.
2. **Generator.** The decoder is fine-tuned to write a molecule for a set of odor tokens. This is the
   goal of the project.
3. **Classifier.** A small head on the frozen pretrained decoder scores each of 890 odor tokens for a
   molecule.

## 2. Data

The odor tokens come from eleven perfumery and flavor sources collected in Pyrfume: Leffingwell, The
Good Scents Company, Arctander, FlavorDB, Sharma, IFRA, AromaDB, Sigma, Flavornet, Dravnieks and
Keller. Together they describe 7,516 molecules with 890 distinct odor tokens.

The pretraining set is drawn from PubChem, restricted to molecules of 350 Da or less, because a
molecule has to evaporate to be smelled. 3.4 million compounds are sampled by a checksum of their
identifier so that the same molecules are taken on every build, and converted to canonical SMILES and
SELFIES.

The fine-tuning data is split into 6,344 training molecules, 352 validation and 352 test molecules.
The 468 molecules of the DREAM Olfaction Prediction Challenge are held out of every stage, including
the pretraining, and are used as an external test. The split is by molecule and by structure, not by
row, so a compound that PubChem registers under two identifiers cannot appear on both sides.

## 3. How the model is judged

Only by data we already have. No rule of ours decides whether a written molecule is good, because one
odor describes countless molecules and a molecule without a label is not wrong.

The generator is scored by the probability it gives the real molecules of a held-out set when it is
told their own odor tokens, in nats per molecule token, together with the gain over an empty request
and over another molecule's odor tokens. What it writes is counted by what RDKit and the data define:
the share that parses, the share that differ from each other, the share that were in neither training
set, and the molecular weight, since the point is a molecule that could be smelled.

The classifier is scored by macro ROC-AUC over the odor tokens that can be scored, where chance is
0.5.

Results on the held-out sets, taken from the repository's own evaluation:

| set | molecules | generator loss | perplexity | gain over no odor tokens | classifier ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| validation | 352 | 0.7062 | 2.03 | +0.224 | 0.832 |
| test | 352 | 0.7096 | 2.03 | +0.215 | 0.812 |
| DREAM | 468 | 0.6759 | 1.97 | +0.323 | 0.779 |

The gain matters as much as the loss. A model that ignores its request has a gain of zero, and the
first version of the fine-tuning did, at +0.002. Odor tokens change what is written, and on 89 percent
of the test molecules and 97 percent of the DREAM molecules they make the true molecule more likely.

## 4. How the project changed, and why

The project did not start here, and the record should say so.

The first plan predicted the human odor profile of a molecule: 21 perceptual attributes scored from 0
to 100, as in the DREAM challenge, from molecular structure alone. The inputs were RDKit descriptors
and Morgan fingerprints, the model a hybrid network with one output per attribute, and the score the
official DREAM Q2 metric. It worked. The model reproduced a Q2 score of 6.896 on the official 69
molecule test set, against 8.87 for the winning entry of the challenge and 9.17 for the aggregate of
all teams, and it did so with a clean split, a test suite and a continuous integration pipeline.

Feedback on the project was that this read as a machine learning project rather than a deep learning
one. That judgement can be argued: the DREAM challenge itself was won by gradient boosting on
descriptors, and a task that the best classical models solve is not made deep by wrapping it in a
network. But the criticism pointed at something real. The deep model was doing the easy half. The
interesting learning problem, the one that needs a network with millions of parameters and a corpus
behind it, is the language of molecules: what makes a string of atoms a molecule at all, and how a
perceptual word attaches to it.

So we changed the question. Instead of predicting a rating vector from a descriptor table, the model
now learns to generate molecular text and to condition that generation on odor words. This is a deep learning problem: a transformer trained on 3.4 million molecules, fine-tuned
with adapters that inject the request into every layer, and a classifier head trained on frozen
representations. None of it can be replaced by a random forest.

The earlier work is not thrown away. It is preserved in the repository at the git tag
`archive/structure-to-odor-profile`, together with its data pipeline, its official-metric scoring and its
results.
It remains the project's evidence that molecular structure carries enough information about odor to
be worth modelling at all.

## 5. What is being explored next

Three directions are open, in the order they will be tried.

- **A latent space for molecules.** A conditional variational autoencoder on top of the pretrained
  decoder, so that a molecule has a continuous vector that can be edited, interpolated and searched,
  instead of only being generated. The decoder already injects the odor request as a scale and a
  shift at every layer, so a latent code can use the same mechanism. The known risk is posterior
  collapse: a strong autoregressive decoder tends to ignore the latent code unless it is trained
  against it. This is an exploration and a negative result is a result.
- **Reading the conditioning.** Where in the decoder the request acts, and how the odor token
  embeddings move between the pretrained and the fine-tuned model.
- **A larger model.** Pretraining stopped improving at 6.3 million parameters, so scale is the
  obvious lever left for the generator.

## 6. Limits

The score says how probable the real molecules are, not what a written molecule smells like. Nobody
has smelled what the generator writes. The checks show that its output is valid, varied, new and the
right size for a molecule that could evaporate, and that is as far as the available data goes.

Many settings were chosen on the validation split, so its numbers are slightly optimistic. The test
and DREAM numbers are not, except that the number of training epochs was chosen after an eighty epoch
run had already been scored on the test split.

The odor vocabulary has defects that were left alone, because a new vocabulary means a new
pretraining: some sources glue words together (`almond bitter almond`), and `almost odorless` is a
token like any other.

## 7. Team

Emmanouil Oikonomou built the original DREAM pipeline and its evaluation harness, and works on the
data pipeline, the evaluation and the latent space exploration. Konstantin Krasovitskiy built the
decoder, the generator and the classifier. Anastasia Kyriakou works on the analysis, the classic
baselines and the final report.
