# DREAM Subchallenge 2 comparison data

Metric: organizer Q2 mean Z-score. Higher is better.

| Reference | Score | Source |
|---|---:|---|
| Challenge winner | 8.87 | [DREAM paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5455768/) |
| Participant model aggregate | 9.17 | [DREAM paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5455768/) |
| Post-challenge ensemble | 9.92 | [Follow-up paper](https://doi.org/10.1016/j.inffus.2025.103038) |
| Stored test-set predictions | 9.6338 | `official_testset_reference_score.json` |

The custom model metrics are saved under `output/custom_model/metrics/`. The comparison chart is `output/custom_model/plots/dream_challenge_comparison.png`.

The stored test-set predictions are `challenge_2_testset_1446435104.txt`. The scorer gold standard is `official_testset_gold_standard.tsv`. The organizer's Q2 scorer and validation script are in `official_scoring/`.
