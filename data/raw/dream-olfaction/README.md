# DREAM Olfaction Prediction Challenge — raw data

Raw data files from the [DREAM Olfaction Prediction
Challenge](https://github.com/dream-olfaction/olfaction-prediction), published in
_Keller et al., "Predicting human olfactory perception from chemical features of
odor molecules", Science (2017)_. DOI: [10.1126/science.aal2014](https://dx.doi.org/10.1126/science.aal2014).

| File                                                | Contents                                                                                                                                               |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `TrainSet.txt`                                      | Per-subject intensity ratings (0–100) of 21 odor descriptors for every compound at two dilutions, plus `INTENSITY/STRENGTH` and `VALENCE/PLEASANTNESS` |
| `TestSet.txt`                                       | Same schema for the held-out test molecules                                                                                                            |
| `CID_leaderboard.txt` / `CID_testset.txt`           | PubChem CIDs of the leaderboard and test molecules                                                                                                     |
| `dilution_leaderboard.txt` / `dilution_testset.txt` | Dilutions used for each compound                                                                                                                       |
| `cid_smiles.tsv`                                    | `cid → canonical SMILES` mapping fetched from PubChem (see `scripts/data/fetch_pubchem_smiles.py`)                                                     |

**Note on the raw schema:** `TrainSet.txt`/`TestSet.txt` are subject-level, so every
(cid, replicate) pair appears many times.

Reproduce the download: `bash scripts/data/download_dream_data.sh`
