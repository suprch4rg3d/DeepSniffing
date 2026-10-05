#!/usr/bin/env bash
# Fetches the official DREAM Olfaction Prediction Challenge data files.
#
# Source: https://github.com/dream-olfaction/olfaction-prediction (Science 2017,
# DOI: 10.1126/science.aal2014). Files are stored under data/raw/dream-olfaction/.
#
# Usage:  bash scripts/data/download_dream_data.sh
set -euo pipefail

REPO="dream-olfaction/olfaction-prediction"
BRANCH="master"
DEST="data/raw/dream-olfaction"

FILES=(
  TrainSet.txt
  TestSet.txt
  CID_leaderboard.txt
  CID_testset.txt
  dilution_leaderboard.txt
  dilution_testset.txt
)

mkdir -p "$DEST"
for f in "${FILES[@]}"; do
  url="https://raw.githubusercontent.com/${REPO}/${BRANCH}/data/${f}"
  echo "Downloading ${f} ..."
  curl -sSL --fail --retry 3 -o "${DEST}/${f}" "${url}"
done

echo "Downloaded $(( ${#FILES[@]} )) files into ${DEST}/"
echo "Next: fetch SMILES for every compound CID, e.g.  python scripts/data/fetch_pubchem_smiles.py"