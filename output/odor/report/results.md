# Results

| split | molecules | loss | perplexity | gain over no odor tokens | odor tokens help |
|---|---|---|---|---|---|
| validation | 352 | 0.7112 | 2.04 | +0.218 ± 0.019 | 90% |
| test | 352 | 0.7124 | 2.04 | +0.212 ± 0.020 | 90% |
| dream | 468 | 0.6972 | 2.01 | +0.298 ± 0.016 | 97% |

## The data

| measure | value |
|---|---|
| molecules | 7,516: 6,344 train, 352 validation, 352 test, 468 DREAM |
| odor tokens | 890, 211 on a single molecule |
| odor tokens per molecule | median 4, mean 5.8, most 34 |
| sources | 9 |
| weight | median 174 Da, 1.5% over 350 |

## What it wrote

| request | valid | unique | novel | median weight | over 350 | nearest training | over 0.7 |
|---|---|---|---|---|---|---|---|
| fruity | 1.00 | 0.99 | 0.70 | 184 | 0.0% | n/a | n/a |
| woody | 1.00 | 1.00 | 0.86 | 204 | 2.0% | n/a | n/a |
| musk | 1.00 | 0.89 | 0.85 | 228 | 0.6% | n/a | n/a |
| sulfurous | 1.00 | 0.95 | 0.74 | 148 | 0.0% | n/a | n/a |
| minty + spearmint | 1.00 | 0.98 | 0.80 | 168 | 0.5% | n/a | n/a |
| minty + caraway | 1.00 | 0.99 | 0.77 | 160 | 0.0% | n/a | n/a |
| (nothing asked for) | 1.00 | 1.00 | 0.99 | 271 | 6.0% | n/a | n/a |

## The classifier

| split | molecules | ROC-AUC | average precision | precision at 5 | precision at its own count |
|---|---|---|---|---|---|
| validation | 352 | 0.832 | 0.224 | 0.359 | 0.393 |
| test | 352 | 0.811 | 0.268 | 0.381 | 0.399 |
| dream | 468 | 0.776 | 0.224 | 0.580 | 0.425 |

## The baseline

| split | ridge ROC-AUC | decoder ROC-AUC | ridge average precision | decoder average precision |
|---|---|---|---|---|
| validation | 0.725 | 0.832 | 0.223 | 0.224 |
| test | 0.736 | 0.811 | 0.261 | 0.268 |
| dream | 0.670 | 0.776 | 0.178 | 0.224 |

### The ten odor tokens it ranks worst on the test molecules

| odor token | ROC-AUC | molecules |
|---|---|---|
| linden | 0.043 | 1 |
| linden flower | 0.057 | 1 |
| powerful | 0.068 | 1 |
| fat | 0.134 | 1 |
| concord grape | 0.154 | 1 |
| sea | 0.193 | 2 |
| opoponax | 0.197 | 2 |
| bean green bean | 0.199 | 1 |
| tart | 0.225 | 1 |
| peppery | 0.254 | 1 |
