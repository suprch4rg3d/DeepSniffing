# DeepSniffing

Computers already see, hear, and talk — we're teaching them to _smell_.

Predicting human odor profiles (sweet, musky, fruity, …) from SMILES strings
using sequence-based deep learning. Semester project for MAI642 — Deep Learning.

## Project state

This is the scaffolding phase: repository layout, dependencies, and raw data
are in place; models and training code come next.

## Repository layout

```
data/raw/dream-olfaction/   DREAM Olfaction Prediction Challenge dataset
                            (train/test ratings + CID → SMILES mapping)
scripts/data/               scripts to download and prepare the data
docs/                       project proposal
```

## Requirements

- Python 3.13 (>=3.13, <3.14 as declared in `pyproject.toml`)
- [Poetry](https://python-poetry.org) (2.x) — the project uses poetry for
  dependency management; package mode is disabled, so `poetry install` only
  sets up the dependency environment, not the project itself.

## Installation

**Clone the repository (all platforms):**

```bash
git clone https://github.com/suprch4rg3d/DeepSniffing.git
cd DeepSniffing
```

This project targets **Python 3.13** (see `requires-python` in `pyproject.toml`).
If you have several Python versions installed, make sure `python` (or `py` on
Windows) resolves to 3.13 before creating the environment.

### Method A — Poetry (recommended)

Identical on macOS, Linux, and Windows:

```bash
poetry shell      # create and activate the project environment
poetry install    # install the dependencies (numpy, pandas, torch, rdkit, ...)
```

### Method B — Plain pip

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell or cmd):**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> If `python` is not recognised on Windows, use the launcher instead:
> `py -3.13 -m venv .venv`. To leave the environment later, run `deactivate`.

### Verify the installation

```bash
python -c "import numpy, pandas, sklearn, torch, rdkit; print('dependencies OK')"
```

## Data

The dataset (train/test ratings, CID lists, dilutions, and the CID → SMILES
mapping) is already committed under `data/raw/dream-olfaction/`. See the
`README.md` there for provenance and schema.

To re-fetch everything from scratch:

```bash
bash scripts/data/download_dream_data.sh    # downloads the DREAM files
python scripts/data/fetch_pubchem_smiles.py # CID → SMILES from PubChem
```

## Project proposal

See [docs/mai642_project_proposal.md](docs/mai642_project_proposal.md).
