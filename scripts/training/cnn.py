from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/dream_pyrfume/molecule_labels_with_split.csv"
TARGET_SCALER = ROOT / "data/processed/dream_pyrfume/scalers/target_scaler.joblib"
OUTPUT = ROOT / "output/cnn"
CHECKPOINTS = OUTPUT / "checkpoints"
PLOTS = OUTPUT / "plots"
METRICS = OUTPUT / "metrics"

SEED = 42
EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 32
MAX_LENGTH = 128

EMBEDDING_SIZE = 32
CONV_FILTERS = (48, 64)
KERNEL_SIZE = 3
REGRESSION_HIDDEN_SIZE = 48
DROPOUT = 0.2


def main() -> None:
    pass


if __name__ == "__main__":
    main()
