"""Every file location of the odor generator.

To move this package into the project (``scripts/generation/``), change ``ROOT`` to the folder that
should hold the data and the results, for example ``PROJECT_ROOT / "data/generation"``, and keep
the rest. All other modules take their paths from here.
"""

from pathlib import Path

# The folder that holds ``data/`` and ``results/``. Here: experiments/odor_generator.
ROOT = Path(__file__).resolve().parents[1]
# The DeepSniffing project, which holds the DREAM data that the student learns from.
PROJECT_ROOT = ROOT.parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"  # the Pyrfume files: leffingwell, goodscents and the molecule registry
RESULTS = ROOT / "results"

DATASET = DATA / "dataset.csv"
WORDS = DATA / "words.json"
VOCABULARY = DATA / "vocabulary.json"
GENERATOR = DATA / "generator.pt"  # the best model
GENERATOR_TRAINING = (
    DATA / "generator_training.pt"
)  # the state to resume an interrupted training from
ORACLE = DATA / "oracle.pt"
STUDENT = DATA / "student.pt"

DREAM_MOLECULES = PROJECT_ROOT / "data/processed/dream_pyrfume/molecules.csv"
DREAM_TEST_GOLD = PROJECT_ROOT / "data/reference/dream_challenge/official_testset_gold_standard.tsv"
