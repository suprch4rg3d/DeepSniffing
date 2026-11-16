# DREAM comparison data

## Subchallenge 2

Metric: organizer Q2 mean Z-score. Higher is better.

| Reference | Score | Source |
|---|---:|---|
| Challenge winner | 8.87 | [DREAM paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5455768/) |
| Participant model aggregate | 9.17 | [DREAM paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5455768/) |
| Post-challenge ensemble | 9.92 | [Follow-up paper](https://doi.org/10.1016/j.inffus.2025.103038) |
| Stored test-set predictions | 9.6338 | `official_testset_reference_score.json` |

The final custom model is trained on 407 molecules (the 338 TrainSet and the 69 leaderboard molecules), the same data the challenge teams could use. The custom model metrics are saved under `output/q2/metrics/` and `output/q1/metrics/`. The comparison chart is `output/q2/plots/dream_challenge_comparison.png`, drawn by `python -m scripts.core.plots`.

The stored test-set predictions are `challenge_2_testset_1446435104.txt`. The scorer gold standard is `official_testset_gold_standard.tsv`. The organizer's Q2 scorer and validation script are in `official_scoring/`. The chart reads `published_challenge_scores.json`.

## Subchallenge 1

Metric: organizer Q1 mean Z-score (every participant's own rating). Higher is better. The chart reads `published_challenge_scores_q1.json`.

| Reference | Score | Source |
|---|---:|---|
| Challenge winner (GuanLab) | 34.18 | [DREAM paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5455768/) |
| Participant model aggregate | 34.02 | [DREAM paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5455768/) |
| Post-challenge ensemble | 36.45 | [DREAM paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5455768/) |

There are no stored Q1 reference predictions, so the Q1 chart shows the custom model and the three published scores.

### Targets per score group

The DREAM paper gives one individual-level Pearson correlation per score group. It does not
name the model behind them. Dividing them by the Subchallenge 1 null model standard deviations
gives a mean Z-score of 34.2, which matches the winner's 34.18, so they most likely describe
GuanLab. That link is an inference. The Z-scores below are derived here, not published, and are
also stored in `published_challenge_scores_q1.json`.

| Score group | Correlation | Z-score |
|---|---:|---:|
| Intensity | 0.56 | 29.9 |
| Pleasantness (valence) | 0.41 | 23.3 |
| 19 descriptors | 0.21 | 49.5 |

The paper names the easiest attributes (intensity, garlic, pleasantness, sweet, fruit, spices,
burnt) and the hardest (acid, cold, warm, wood, urinous, chemical, musky). The per attribute
values are only in its figures and in its Data File S1.
